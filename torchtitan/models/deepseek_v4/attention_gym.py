# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym selected-attention override for DeepSeek V4.

Install the upstream attention-gym, then activate this module with::

    pip install "attn_gym[sparse] @ git+https://github.com/meta-pytorch/attention-gym.git"

    --override.imports torchtitan.models.deepseek_v4.attention_gym

The override dispatches by the layer's compression ratio:

- ratio 1 -> selected_attention with topk=0 (pure sliding window)
- ratio 4 -> selected_attention with indexer-selected topk blocks (CSA)
- other ratios -> selected_attention with all causally-valid blocks (HCA)

All three layer types route through the same upstream ``selected_attention``
API, which accepts pre-computed query, local_kv, sparse_kv, and kv_indices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import spmd_types as spmd
import torch

from torchtitan.config import derive, override
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

from .attention import Attention

# Shape suffix legend for this file:
# B: batch, L: sequence, M: model, A: query LoRA, H: attention heads,
# D: attention head, I: index heads, J: index head dim, R: compression ratio.

try:
    from attn_gym.sparse import selected_attention as _selected_attention

    _ATTENTION_GYM_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    _selected_attention = None
    _ATTENTION_GYM_IMPORT_ERROR = error


DP = MeshAxisName.DP
CP = MeshAxisName.CP
TP = MeshAxisName.TP


def _activation_layout(
    *,
    sequence_axis: int,
    tp: spmd.PerMeshAxisSpmdType,
) -> SpmdLayout:
    return SpmdLayout(
        {
            DP: spmd.S(0),
            CP: spmd.S(sequence_axis),
            TP: tp,
        }
    )


def _selected_attention_sharding_config() -> ShardingConfig:
    """Build a sharding config for the SelectedAttentionKernel.

    The kernel always takes: query, local_kv, sparse_kv, kv_indices,
    attention_sink -- for SWA layers, sparse_kv and kv_indices are
    empty tensors with a zero-length dimension.
    """
    q_BHLD = _activation_layout(sequence_axis=2, tp=spmd.S(1))
    shared_B1LD = _activation_layout(sequence_axis=2, tp=spmd.R)
    shared_grad_B1LD = _activation_layout(sequence_axis=2, tp=spmd.P)
    indices_BLK = _activation_layout(sequence_axis=1, tp=spmd.R)
    sink_H = dense_param_placement(tp=spmd.S(0))

    input_shardings = {
        "query_BHLD": q_BHLD,
        "local_kv_B1LD": shared_B1LD,
        "sparse_kv_B1ND": shared_B1LD,
        "kv_indices_BLK": indices_BLK,
        "attention_sink_H": sink_H,
    }
    grad_placements = (
        q_BHLD,
        shared_grad_B1LD,
        shared_grad_B1LD,
        indices_BLK,
        sink_H,
    )
    return ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=q_BHLD,
        local_map=LocalMapConfig(in_grad_placements=grad_placements),
    )


def _rename_attention_input_sharding(
    sharding_config: ShardingConfig | None,
) -> ShardingConfig | None:
    if sharding_config is None:
        return None

    remapped = {}
    for field_name in ("in_src_shardings", "in_dst_shardings"):
        shardings = getattr(sharding_config, field_name)
        if shardings is None:
            remapped[field_name] = None
            continue
        shardings = dict(shardings)
        assert "x" in shardings
        shardings["x_BLM"] = shardings.pop("x")
        remapped[field_name] = shardings
    return replace(sharding_config, **remapped)


class SelectedAttentionKernel(Module):
    """Local-tensor boundary around upstream selected_attention.

    Wraps the attn_gym.sparse.selected_attention API to handle all three
    DeepSeek V4 layer types (SWA, HCA, CSA) through a single kernel call.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        sliding_window_size: int
        backend: str = "cute"

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.sliding_window_size = config.sliding_window_size
        self.backend = config.backend

    def forward(
        self,
        query_BHLD: torch.Tensor,
        local_kv_B1LD: torch.Tensor,
        sparse_kv_B1ND: torch.Tensor,
        kv_indices_BLK: torch.Tensor,
        attention_sink_H: torch.Tensor,
    ) -> torch.Tensor:
        assert _selected_attention is not None

        # The CuTe backend delegates to FA4 which does not support attention
        # sinks. Pass None so the backend skips the sink logit entirely.
        sink = None if self.backend == "cute" else attention_sink_H

        return _selected_attention(
            query=query_BHLD,
            local_kv=local_kv_B1LD,
            sparse_kv=sparse_kv_B1ND,
            kv_indices=kv_indices_BLK,
            attention_sink=sink,
            doc_ids=None,
            sliding_window_size=self.sliding_window_size,
            backend=self.backend,
        )


class AttentionGymSelectedAttention(Attention):
    """DeepSeek V4 attention using upstream selected_attention for all layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        selected_attention_kernel: SelectedAttentionKernel.Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.selected_attention_kernel = config.selected_attention_kernel.build()

    def _query_inputs(
        self,
        x_BLM: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qr_BLA = self.q_norm(self.wq_a(x_BLM))
        q_BLHD = self.wq_b(qr_BLA).unflatten(-1, (self.n_heads, self.head_dim))
        q_BLHD = q_BLHD * torch.rsqrt(
            q_BLHD.square().mean(-1, keepdim=True) + self.norm_eps
        )
        return qr_BLA, q_BLHD.transpose(1, 2).contiguous()

    def _project_output(self, out_BHLD: torch.Tensor) -> torch.Tensor:
        bsz, _, seqlen, _ = out_BHLD.shape
        out_BLHD = out_BHLD.transpose(1, 2).contiguous()
        num_local_groups = self.n_groups // (self.n_heads // out_BLHD.shape[2])
        out_BLGD = out_BLHD.view(bsz, seqlen, num_local_groups, -1)
        wo_a_GAD = self.wo_a.weight.view(num_local_groups, self.o_lora_rank, -1)
        out_BLGA = torch.einsum("blgd,gad->blga", out_BLGD, wo_a_GAD)
        return self.wo_b(out_BLGA.reshape(bsz, seqlen, -1))

    def _prepare_local_kv(self, x_BLM: torch.Tensor, positions=None) -> torch.Tensor:
        """Prepare local KV: project, norm, RoPE. Returns (B, 1, L, D)."""
        rd = self.rope_head_dim
        kv = self.wkv(x_BLM)
        kv = self.kv_norm(kv)
        kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
        kv_rope = self.rope(
            kv_rope.unsqueeze(2), kv_rope.unsqueeze(2), positions
        )[0].squeeze(2)
        kv_BLD = torch.cat([kv_nope, kv_rope], dim=-1)
        return kv_BLD.unsqueeze(1)

    def _prepare_query(
        self, x_BLM: torch.Tensor, positions=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare query: project, norm, RoPE. Returns (qr_BLA, q_BHLD)."""
        rd = self.rope_head_dim
        qr_BLA, q_BHLD = self._query_inputs(x_BLM)
        q_nope, q_rope = torch.split(
            q_BHLD, [self.head_dim - rd, rd], dim=-1
        )
        q_rope, _ = self.rope(q_rope, q_rope, positions)
        q_BHLD = torch.cat([q_nope, q_rope], dim=-1)
        return qr_BLA, q_BHLD

    def _prepare_compressed_kv(
        self, x_BLM: torch.Tensor, positions=None
    ) -> torch.Tensor:
        """Prepare compressed KV via the compressor. Returns (B, 1, N, D)."""
        if self.compress_ratio == 4:
            compressed_BND = self.compressor(x_BLM, positions=positions)
        else:
            compressed_BND = self.compressor_128(x_BLM, positions=positions)
        return compressed_BND.unsqueeze(1)

    def _compute_kv_indices_hca(
        self, seq_len: int, num_blocks: int, device: torch.device, batch: int
    ) -> torch.Tensor:
        """Build kv_indices for HCA: each query attends to all causally-valid blocks.

        Returns (B, L, num_blocks) with -1 for positions beyond the causal boundary.
        """
        query_positions = torch.arange(seq_len, device=device)
        completed_blocks = (query_positions + 1) // self.compress_ratio
        block_indices = torch.arange(num_blocks, device=device)
        # Each query can attend to blocks 0..completed_blocks[q]-1.
        # Mask out blocks beyond the causal boundary with -1.
        valid = block_indices.unsqueeze(0) < completed_blocks.unsqueeze(1)
        indices_LN = block_indices.unsqueeze(0).expand(seq_len, -1).clone()
        indices_LN[~valid] = -1
        return indices_LN.unsqueeze(0).expand(batch, -1, -1)

    def _compute_kv_indices_csa(
        self,
        x_BLM: torch.Tensor,
        qr_BLA: torch.Tensor,
        seq_len: int,
        num_blocks: int,
        positions=None,
    ) -> torch.Tensor:
        """Build kv_indices for CSA via the indexer.

        Returns (B, L, topk) with -1 for invalid positions.
        """
        device = x_BLM.device
        base = torch.arange(seq_len, device=device).unsqueeze(1)
        compress_causal_limit = (base + 1) // self.compress_ratio
        compress_causal_mask = (
            torch.arange(num_blocks, device=device).unsqueeze(0)
            >= compress_causal_limit
        )
        compress_topk_idxs, _ = self.indexer(
            x_BLM.detach(),
            qr_BLA.detach(),
            compress_causal_mask,
            compress_causal_limit,
            positions=positions,
            offset=0,
        )
        # The indexer uses offset to shift indices into a concatenated KV
        # tensor (local + compressed). Here sparse_kv is separate, so offset=0
        # gives us direct indices into the compressed KV. The indexer returns -1
        # for invalid slots.
        return compress_topk_idxs

    def _apply_output_rope(
        self, out_BHLD: torch.Tensor, positions=None
    ) -> torch.Tensor:
        """Apply RoPE to the rope portion of the attention output.

        In MLA, the shared KV projection embeds RoPE in the value side. The
        base Attention class applies a second RoPE rotation to the output so
        that the output projection weights can absorb a consistent rotation.
        """
        rd = self.rope_head_dim
        o_nope, o_rope = torch.split(out_BHLD, [self.head_dim - rd, rd], dim=-1)
        o_rope = self.rope(o_rope, o_rope, positions)[0]
        return torch.cat([o_nope, o_rope], dim=-1)

    def forward(self, x_BLM, attention_masks=None, positions=None):
        bsz, seq_len, _ = x_BLM.shape

        qr_BLA, q_BHLD = self._prepare_query(x_BLM, positions)
        local_kv_B1LD = self._prepare_local_kv(x_BLM, positions)
        attention_sink_H = self.attn_sink.weight.squeeze(-1)

        if self.compress_ratio == 1:
            # SWA: no sparse branch -- pass empty sparse_kv and indices.
            head_dim = q_BHLD.shape[-1]
            sparse_kv_B1ND = q_BHLD.new_empty(bsz, 1, 0, head_dim)
            kv_indices_BLK = torch.empty(
                bsz, seq_len, 0, dtype=torch.int64, device=x_BLM.device
            )
        elif self.compress_ratio == 4:
            # CSA: indexer-selected topk compressed blocks
            sparse_kv_B1ND = self._prepare_compressed_kv(x_BLM, positions)
            num_blocks = sparse_kv_B1ND.shape[2]
            kv_indices_BLK = self._compute_kv_indices_csa(
                x_BLM, qr_BLA, seq_len, num_blocks, positions
            )
        else:
            # HCA: attend to all causally-valid compressed blocks
            sparse_kv_B1ND = self._prepare_compressed_kv(x_BLM, positions)
            num_blocks = sparse_kv_B1ND.shape[2]
            kv_indices_BLK = self._compute_kv_indices_hca(
                seq_len, num_blocks, x_BLM.device, bsz
            )

        out_BHLD = self.selected_attention_kernel(
            q_BHLD,
            local_kv_B1LD,
            sparse_kv_B1ND,
            kv_indices_BLK,
            attention_sink_H,
        )
        out_BHLD = self._apply_output_rope(out_BHLD, positions)
        return self._project_output(out_BHLD)


@override(
    "deepseek_v4_attention_gym",
    target=Attention.Config,
    exact=True,
    description="Upstream selected_attention for all DeepSeek V4 layer types.",
)
def attention_gym(cfg: Attention.Config) -> AttentionGymSelectedAttention.Config:
    if _ATTENTION_GYM_IMPORT_ERROR is not None:
        raise ImportError(
            "The DeepSeek V4 Attention Gym override requires the upstream "
            "meta-pytorch/attention-gym with the selected_attention API."
        ) from _ATTENTION_GYM_IMPORT_ERROR

    selected_attention_kernel = SelectedAttentionKernel.Config(
        sliding_window_size=cfg.window_size,
        sharding_config=_selected_attention_sharding_config(),
    )
    return derive(
        cfg,
        AttentionGymSelectedAttention.Config,
        selected_attention_kernel=selected_attention_kernel,
        sharding_config=_rename_attention_input_sharding(cfg.sharding_config),
    )


__all__ = [
    "AttentionGymSelectedAttention",
    "SelectedAttentionKernel",
]
