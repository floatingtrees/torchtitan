# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import BaseAttention
from torchtitan.models.common.nn_modules import Linear, RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.protocols.module import Module

from .compressor import Compressor, Indexer


# Tensor dimension legend:
#   B: batch, L: sequence, N: query heads, D: head dimension
#   K: padded key/value length, W: sparse key count, X: flattened L * W
def _gather_sparse_attention_core(
    query_BLND: torch.Tensor,
    kv_padded_BKD: torch.Tensor,
    topk_indices_BLW: torch.Tensor,
    invalid_mask_BLW: torch.Tensor,
    attn_sink_N: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Apply gather-based sparse attention to precomputed sparse indices."""
    batch_size, sequence_length, num_heads, head_dim = query_BLND.shape
    num_sparse_keys = topk_indices_BLW.shape[-1]

    indices_BXD = (
        topk_indices_BLW.reshape(batch_size, -1)
        .unsqueeze(-1)
        .expand(-1, -1, head_dim)
    )
    kv_gathered_BLWD = torch.gather(kv_padded_BKD, 1, indices_BXD).reshape(
        batch_size,
        sequence_length,
        num_sparse_keys,
        head_dim,
    )

    query_BNLD = query_BLND.transpose(1, 2)
    scores_BNLW = (
        torch.einsum("bnld,blwd->bnlw", query_BNLD, kv_gathered_BLWD)
        * softmax_scale
    )
    scores_BNLW = scores_BNLW.masked_fill(
        invalid_mask_BLW.unsqueeze(1),
        torch.finfo(scores_BNLW.dtype).min,
    )

    sink_BNL1 = attn_sink_N.reshape(1, -1, 1, 1).expand(
        batch_size,
        num_heads,
        sequence_length,
        1,
    )
    scores_with_sink_BNLW = torch.cat([scores_BNLW, sink_BNL1], dim=-1)
    probabilities_BNLW = F.softmax(
        scores_with_sink_BNLW.float(), dim=-1
    ).to(scores_with_sink_BNLW.dtype)[..., :-1]
    output_BNLD = torch.einsum(
        "bnlw,blwd->bnld", probabilities_BNLW, kv_gathered_BLWD
    )
    return output_BNLD.transpose(1, 2).contiguous()


_compiled_gather_sparse_attention_core = torch.compile(
    _gather_sparse_attention_core
)


class DSAFlexAttention(Module):
    """Gather-based sparse attention for DeepSeek V4.

    Replaces FlexAttention with explicit gather/scatter ops and standard
    softmax over the sparse window + compressed tokens.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        block_size: int = 128  # unused, kept for config compat

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale

    def forward(
        self,
        query_states,
        kv_states,
        attn_sink,
        kv_compress,
        compress_topk_idxs,
    ):
        bsz, seqlen, _, _ = query_states.size()

        # Build window indices: each query attends to its preceding window_size tokens
        # topk_idxs: (seqlen, window_size)
        base = torch.arange(seqlen, device=query_states.device).unsqueeze(1)
        window_offsets = torch.arange(
            min(seqlen, self.window_size), device=query_states.device
        )
        window_topk = (base - self.window_size + 1).clamp(0) + window_offsets
        # Mark positions beyond the query as invalid (-1)
        window_topk = torch.where(window_topk > base, -1, window_topk)
        # Expand to batch: (bsz, seqlen, window_size)
        topk_idxs = window_topk.unsqueeze(0).expand(bsz, -1, -1)

        # Append compressed token indices if compress_ratio > 1
        if self.compress_ratio > 1 and compress_topk_idxs.size(-1) > 0:
            topk_idxs = torch.cat(
                [topk_idxs, compress_topk_idxs.to(topk_idxs.device)], dim=-1
            )

        # Concatenate kv with compressed kv
        if self.compress_ratio > 1 and kv_compress.size(1) > 0:
            kv_states = torch.cat([kv_states, kv_compress], dim=1)

        kv_len = kv_states.size(1)

        # Pad KV with one zero token at end for invalid index mapping
        kv_padded = F.pad(kv_states, (0, 0, 0, 1))  # (B, kv_len+1, D)

        # Map invalid positions (-1) to the pad token index
        topk_idxs = topk_idxs.where(
            topk_idxs >= 0, torch.tensor(kv_len, device=topk_idxs.device)
        )
        topk_idxs = topk_idxs.long()
        invalid_mask = topk_idxs == kv_len
        if torch.compiler.is_compiling():
            return _gather_sparse_attention_core(
                query_states,
                kv_padded,
                topk_idxs,
                invalid_mask,
                attn_sink,
                self.softmax_scale,
            )
        return _compiled_gather_sparse_attention_core(
            query_states,
            kv_padded,
            topk_idxs,
            invalid_mask,
            attn_sink,
            self.softmax_scale,
        )


class GetAttnScores(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, query, key, attention_masks, num_attn_heads, attn_scale):
        if num_attn_heads > 1:
            key = key.repeat_interleave(num_attn_heads, dim=1)
        attn = (query @ key.transpose(-1, -2)) * attn_scale
        if attention_masks is not None:
            attn.masked_fill_(attention_masks, float("-inf"))
        attn = F.softmax(attn.float(), dim=-1)
        attn = attn.sum(dim=1)
        return attn


class Attention(BaseAttention):
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
        window_size: int = 128
        norm_eps: float = 1e-6
        index_n_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 512
        n_layers: int = 4
        layer_id: int = 0
        mask_type: str = "causal"

        # Sub-module configs — declared as fields so the sharding system can
        # set sharding_config on them before build().
        wq_a: Linear.Config | None = None
        q_norm: RMSNorm.Config | None = None
        wq_b: Linear.Config | None = None
        wkv: Linear.Config | None = None
        kv_norm: RMSNorm.Config | None = None
        wo_a: Linear.Config | None = None
        wo_b: Linear.Config | None = None
        attn_sink: Linear.Config | None = None

        # Compressor/indexer are conditional, so keep them here too.
        compressor: Compressor.Config | None = None
        compressor_128: Compressor.Config | None = None
        indexer: Indexer.Config | None = None
        sparse_attn: DSAFlexAttention.Config | None = None

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
        self.window_size = cfg.window_size
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

        self.sparse_attn = cfg.sparse_attn.build()
        self._dsa_loss_tracker = None

    def set_dsa_loss_tracker(self, tracker):
        self._dsa_loss_tracker = tracker

    def forward(self, x, attention_masks=None, positions=None):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.norm_eps)
        q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)

        q_rope, kv_rope = self.rope(q_rope, kv_rope.unsqueeze(2), positions)
        q = torch.cat([q_nope, q_rope], dim=-1)
        kv = torch.cat([kv_nope, kv_rope.squeeze(2)], dim=-1)

        kv_compress = compress_topk_idxs = None

        if self.compress_ratio > 1 and hasattr(self, "indexer"):
            base = torch.arange(seqlen, device=x.device).unsqueeze(1)
            compress_causal_limit = (base + 1) // self.compress_ratio
            compress_causal_mask = (
                torch.arange(
                    seqlen // self.compress_ratio, device=x.device
                ).unsqueeze(0)
                >= compress_causal_limit
            )
            compress_topk_idxs, _ = self.indexer(
                x.detach(), qr.detach(),
                compress_causal_mask, compress_causal_limit,
                positions=positions,
                offset=kv.size(1),
            )

        if self.compress_ratio == 4:
            kv_compress = self.compressor(x, positions=positions)
        elif self.compress_ratio > 1:
            kv_compress = self.compressor_128(x, positions=positions)

        attn_sink_param = self.attn_sink.weight.squeeze(-1)
        if kv_compress is None:
            kv_compress = kv.new_empty((bsz, 0, self.head_dim))
        if compress_topk_idxs is None:
            compress_topk_idxs = torch.empty(
                (bsz, seqlen, 0), dtype=torch.int64, device=x.device
            )
        o = self.sparse_attn(
            q, kv, attn_sink_param, kv_compress, compress_topk_idxs,
        )

        o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, o_rope, positions)[0]
        o = torch.cat([o_nope, o_rope], dim=-1)

        n_local_groups = self.n_groups // (self.n_heads // o.shape[2])
        o = o.view(bsz, seqlen, n_local_groups, -1)
        # wo_a is a Linear module; access its weight directly for the grouped
        # einsum (not a standard Linear forward).
        wo_a = self.wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.reshape(bsz, seqlen, -1))
