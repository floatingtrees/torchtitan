# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from attn_gym.sparse.selected_attention import AuxRequest, selected_attention
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import BaseAttention, FlexAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .compressor import Compressor, Indexer


def _assert_spmd_attention_type(tensor, *, tp):
    if get_spmd_backend() == "spmd_types":
        spmd.assert_type(
            tensor,
            {"dp": spmd.S(0), "cp": spmd.S(1), "tp": tp},
        )


class DSV4FlexAttention(FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexAttention.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.window_size = config.window_size
        self.softmax_scale = config.softmax_scale
        self.block_size = config.block_size

    def get_window_topk_idxs(
        self,
        *,
        bsz: int,
        seqlen: int,
        device,
    ) -> torch.Tensor:
        """Return sliding-window KV indices in the concatenated KV space.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length and uncompressed KV length.
            device: Device used for the generated index tensor.

        Returns:
            Tensor of shape ``[B, L, W]``. Valid entries are uncompressed KV
            positions in ``[0, L)`` and padded entries are ``-1``.
        """
        window = min(seqlen, self.window_size)
        q_idx = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (q_idx - window + 1).clamp_min(0) + torch.arange(window, device=device)
        idxs = torch.where(idxs <= q_idx, idxs, -1)
        return idxs.unsqueeze(0).expand(bsz, -1, -1)

    def _build_block_mask(
        self,
        bsz: int,
        seqlen: int,
        kv_len: int,
        selected_indices: torch.Tensor,
        device,
    ) -> BlockMask:
        """Build a FlexAttention block mask from selected KV indices.

        Args:
            bsz: Batch size.
            seqlen: Query sequence length.
            kv_len: Length of the concatenated KV sequence.
            selected_indices: Tensor of shape ``[B, L, K]`` containing final KV
                positions in ``[0, kv_len)``; ``-1`` entries are ignored.
            device: Device used for mask tensors.

        Returns:
            ``BlockMask`` whose block list and token-level predicate encode
            exactly the selected KV positions.
        """
        bs = self.block_size
        bq, bk = bs if isinstance(bs, tuple) else (bs, bs)
        assert (
            seqlen % bq == 0
        ), f"seqlen ({seqlen}) must be divisible by Q block size ({bq})"
        n_kv_blocks = (kv_len + bk - 1) // bk
        n_q_blocks = seqlen // bq

        valid = selected_indices >= 0
        safe_indices = selected_indices.clamp(0, kv_len - 1)

        selected_blocks = (safe_indices // bk).reshape(
            bsz, n_q_blocks, bq * selected_indices.size(-1)
        )
        block_values = valid.reshape(selected_blocks.shape).to(torch.int32)
        bm = torch.zeros(
            bsz, 1, n_q_blocks, n_kv_blocks, dtype=torch.int32, device=device
        )
        bm[:, 0].scatter_add_(-1, selected_blocks, block_values)
        bm = (bm > 0).to(torch.int32)
        kv_num_blocks = bm.sum(dim=-1).to(torch.int32)
        kv_indices = torch.argsort(bm, dim=-1, descending=True, stable=True).to(
            torch.int32
        )

        selected_count = torch.zeros(
            bsz, seqlen, kv_len, dtype=torch.int32, device=device
        )
        selected_count.scatter_add_(2, safe_indices, valid.to(torch.int32))
        selected_mask = selected_count > 0

        def dsa_mask_mod(b, h, q_idx, kv_idx):
            return selected_mask[b, q_idx, kv_idx]

        return BlockMask.from_kv_blocks(
            kv_num_blocks,
            kv_indices,
            BLOCK_SIZE=(bq, bk),
            mask_mod=dsa_mask_mod,
            seq_lengths=(seqlen, kv_len),
        )

    def _forward_impl(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "DSV4FlexAttention does not accept attention_masks; "
                "the DSA block mask is built internally."
            )
        if attn_sink is None:
            raise ValueError("DSV4FlexAttention requires attn_sink")

        seqlen, _, head_dim = q.size()
        sink_idx = seqlen

        kv = swa_k.unsqueeze(1)
        sink_kv = kv.new_zeros((1, 1, head_dim))
        kv = torch.cat([kv, sink_kv], dim=0)
        kv = kv.expand(-1, q.size(1), -1)

        with spmd.no_typecheck():
            selected_indices = [
                self.get_window_topk_idxs(bsz=1, seqlen=seqlen, device=q.device)
            ]
            sink_indices = torch.full(
                (1, seqlen, 1), sink_idx, dtype=torch.int64, device=q.device
            )
            selected_indices.append(sink_indices)
            selected_indices = torch.cat(selected_indices, dim=-1)

            block_mask = self._build_block_mask(
                1, seqlen, kv.size(0), selected_indices, q.device
            )

            def v4_sink_score_mod(score, b, h, q_idx, kv_idx):
                return torch.where(kv_idx == sink_idx, attn_sink[h], score)

            return super().forward(
                q,
                kv,
                kv,
                attention_masks=block_mask,
                score_mod=v4_sink_score_mod,
                scale=self.softmax_scale,
            )


class SlidingWindowAttention(DSV4FlexAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4FlexAttention.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        return self._forward_impl(
            q,
            swa_k,
            attn_sink,
            attention_masks=attention_masks,
        )


class _InjectAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(*args, **kwargs):
        ctx, carrier, aux_loss = args
        ctx.save_for_backward(aux_loss)
        return carrier

    @staticmethod
    def backward(*args, **kwargs):
        ctx, grad_output = args
        (aux_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss)


def _indexer_loss(
    main_query_BNTD,
    selected_compressed_kv_BTKD,
    attention_lse_BNT,
    selected_indexer_logits_BTK,
    selected_is_valid_BTK,
):
    compressed_attention_logits_BNTK = (
        torch.einsum(
            "bntd,btkd->bntk",
            main_query_BNTD.detach().float(),
            selected_compressed_kv_BTKD.detach().float(),
        )
        * main_query_BNTD.shape[-1] ** -0.5
    )
    compressed_teacher_mass_BTK = torch.exp(
        compressed_attention_logits_BNTK - attention_lse_BNT.detach().float()[..., None]
    ).sum(dim=1)
    valid_BTK = selected_is_valid_BTK.float()
    compressed_teacher_mass_BTK = compressed_teacher_mass_BTK * valid_BTK
    eps = torch.finfo(torch.float32).tiny
    compressed_teacher_probs_BTK = (
        compressed_teacher_mass_BTK
        / compressed_teacher_mass_BTK.sum(dim=-1, keepdim=True).clamp_min(eps)
    ).detach()
    masked_indexer_logits_BTK = selected_indexer_logits_BTK.float().masked_fill(
        ~selected_is_valid_BTK, float("-inf")
    )
    row_has_valid_BT = selected_is_valid_BTK.any(dim=-1)
    masked_indexer_logits_BTK = torch.where(
        row_has_valid_BT[..., None],
        masked_indexer_logits_BTK,
        torch.zeros_like(masked_indexer_logits_BTK),
    )
    indexer_probs_BTK = F.softmax(masked_indexer_logits_BTK, dim=-1)
    kl_BT = (
        compressed_teacher_probs_BTK
        * (
            compressed_teacher_probs_BTK.clamp_min(eps).log()
            - indexer_probs_BTK.clamp_min(eps).log()
        )
    ).sum(dim=-1)
    valid_kl_BT = torch.where(row_has_valid_BT, kl_BT, torch.zeros_like(kl_BT))
    return valid_kl_BT.sum() / row_has_valid_BT.sum().clamp_min(1)


class DSV4SelectedAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        index_topk: int
        backend: str

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.index_topk = config.index_topk
        self.backend = config.backend

    def _forward_selected(
        self,
        q,
        swa_k,
        cmp_k,
        kv_indices,
        attn_sink,
        *,
        return_lse,
    ):
        q_BNTD = q.transpose(0, 1).unsqueeze(0)
        swa_k_BNTD = swa_k.unsqueeze(0).unsqueeze(0)
        cmp_k_BNCD = cmp_k.unsqueeze(0).unsqueeze(0)
        kv_indices_BTK = kv_indices.unsqueeze(0)
        sink_N = None if self.backend == "cute" else attn_sink
        if return_lse:
            out_BNTD, aux = selected_attention(
                q_BNTD,
                swa_k_BNTD,
                cmp_k_BNCD,
                kv_indices_BTK,
                sink_N,
                None,
                self.window_size,
                backend=self.backend,
                return_aux=AuxRequest(lse=True),
            )
            return out_BNTD, aux.lse
        out_BNTD = selected_attention(
            q_BNTD,
            swa_k_BNTD,
            cmp_k_BNCD,
            kv_indices_BTK,
            sink_N,
            None,
            self.window_size,
            backend=self.backend,
        )
        return out_BNTD, None

    @staticmethod
    def _finish_output(out_BNTD, q):
        out_TND = out_BNTD.squeeze(0).transpose(0, 1)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            spmd.assert_type_like(out_TND, q)
        return out_TND


class HeavilyCompressedAttention(DSV4SelectedAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4SelectedAttention.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "HeavilyCompressedAttention does not accept attention_masks"
            )
        with spmd.no_typecheck():
            seqlen = q.shape[0]
            num_compressed = cmp_k.shape[0]
            kv_indices_TK = torch.arange(
                num_compressed, device=q.device, dtype=torch.int64
            ).expand(seqlen, -1)
            causal_limit_T1 = (
                torch.arange(1, seqlen + 1, device=q.device).unsqueeze(1)
                // self.compress_ratio
            )
            kv_indices_TK = torch.where(
                kv_indices_TK < causal_limit_T1, kv_indices_TK, -1
            )
            out_BNTD, _ = self._forward_selected(
                q,
                swa_k,
                cmp_k,
                kv_indices_TK,
                attn_sink,
                return_lse=False,
            )
        return self._finish_output(out_BNTD, q)


class CompressedSparseAttention(DSV4SelectedAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(DSV4SelectedAttention.Config):
        pass

    def forward(
        self,
        q,
        swa_k,
        cmp_k,
        idx_q,
        idx_k,
        idx_w,
        attn_sink,
        *,
        attention_masks=None,
    ) -> torch.Tensor:
        if attention_masks is not None:
            raise ValueError(
                "CompressedSparseAttention does not accept attention_masks"
            )
        with spmd.no_typecheck():
            seqlen = q.shape[0]
            kv_indices_TK, selected_indexer_logits_TK = Indexer.select(
                idx_q,
                idx_k,
                idx_w,
                seqlen=seqlen,
                ratio=self.compress_ratio,
                topk=self.index_topk,
            )
            causal_limit_T1 = (
                torch.arange(1, seqlen + 1, device=q.device).unsqueeze(1)
                // self.compress_ratio
            )
            selected_is_valid_TK = kv_indices_TK < causal_limit_T1
            selected_compressed_kv_TKD = cmp_k[kv_indices_TK]
            causal_kv_indices_TK = torch.where(selected_is_valid_TK, kv_indices_TK, -1)
            out_BNTD, attention_lse_BNT = self._forward_selected(
                q,
                swa_k,
                cmp_k,
                causal_kv_indices_TK,
                attn_sink,
                return_lse=True,
            )
            aux_loss = _indexer_loss(
                q.transpose(0, 1).unsqueeze(0),
                selected_compressed_kv_TKD.unsqueeze(0),
                attention_lse_BNT,
                selected_indexer_logits_TK.unsqueeze(0),
                selected_is_valid_TK.unsqueeze(0),
            )
            out_BNTD = _InjectAuxLoss.apply(out_BNTD, aux_loss)
        return self._finish_output(out_BNTD, q)


class Attention(BaseAttention):
    """DeepSeek V4 attention wrapper around sparse inner attention.

    The module projects Q/KV, applies pre- and post-phase RoPE, prepares
    optional compressed/indexer tensors, and delegates sparse attention to
    ``DSV4FlexAttention``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        dim: int
        n_heads: int
        inner_attention: Module.Config
        rope: RoPE.Config
        head_dim: int = 512
        rope_head_dim: int = 64
        q_lora_rank: int = 1024
        o_lora_rank: int = 1024
        n_groups: int = 8
        compress_ratio: int = 1
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Sub-module configs — declared as fields so the sharding system can
        # set sharding_config on them before build().
        wq_a: Linear.Config
        q_norm: RMSNorm.Config
        wq_b: Linear.Config
        wkv: Linear.Config
        kv_norm: RMSNorm.Config
        wo_a: Linear.Config
        wo_b: Linear.Config
        attn_sink: Linear.Config

        # Compressor/indexer are conditional, so keep them here too.
        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rope_head_dim = cfg.rope_head_dim
        self.q_lora_rank = cfg.q_lora_rank
        self.o_lora_rank = cfg.o_lora_rank
        self.n_groups = cfg.n_groups
        self.compress_ratio = cfg.compress_ratio
        self.norm_eps = cfg.norm_eps
        self.softmax_scale = cfg.head_dim**-0.5
        self.layer_id = cfg.layer_id
        self.n_layers = cfg.n_layers
        self.rope = cfg.rope.build()

        # Build all sub-modules from their configs.
        self.wq_a = cfg.wq_a.build()
        self.q_norm = cfg.q_norm.build()
        self.wq_b = cfg.wq_b.build()
        self.wkv = cfg.wkv.build()
        self.kv_norm = cfg.kv_norm.build()
        self.wo_a = cfg.wo_a.build()
        self.wo_b = cfg.wo_b.build()
        self.attn_sink = cfg.attn_sink.build()

        if cfg.compressor is not None:
            self.compressor = cfg.compressor.build()
        if cfg.indexer is not None:
            self.indexer = cfg.indexer.build()
        if cfg.compressor_128 is not None:
            self.compressor_128 = cfg.compressor_128.build()

        self.inner_attention = cfg.inner_attention.build()

    def forward(self, x, attention_masks=None, positions=None):
        """Apply one DeepSeek V4 attention layer over folded tokens."""
        num_tokens = x.size(0)
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        with spmd.local():
            q = q.view(num_tokens, -1, self.head_dim)
            _assert_spmd_attention_type(q, tp=spmd.S(1))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.kv_norm(self.wkv(x))
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(1), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(1)], dim=-1)

        cmp_k = idx_q = idx_k = idx_w = None
        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            idx_q, idx_k, idx_w = self.indexer(
                x.detach(), qr.detach(), positions=positions
            )
        if self.compress_ratio == 4:
            cmp_k = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            cmp_k = self.compressor_128(x, positions=positions)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if self.compress_ratio == 4:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                idx_q,
                idx_k,
                idx_w,
                attn_sink_param,
                attention_masks=attention_masks,
            )
        elif self.compress_ratio > 1:
            o = self.inner_attention(
                q,
                kv,
                cmp_k,
                attn_sink_param,
                attention_masks=attention_masks,
            )
        else:
            o = self.inner_attention(
                q,
                kv,
                attn_sink_param,
                attention_masks=attention_masks,
            )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, positions=positions, inverse=True)
        o = torch.cat([o_nope, o_rope], dim=-1)

        with spmd.local():
            n_local_heads = o.shape[1]
            n_local_groups = self.n_groups // (self.n_heads // n_local_heads)
            o = o.view(num_tokens, n_local_groups, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
            wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
                spmd.assert_type(
                    wo_a,
                    {"dp": spmd.R, "cp": spmd.R, "tp": spmd.S(0)},
                )
        o = torch.einsum("tgd,grd->tgr", o, wo_a)
        with spmd.local():
            o = o.reshape(num_tokens, -1)
            _assert_spmd_attention_type(o, tp=spmd.S(1))
        return self.wo_b(o)
