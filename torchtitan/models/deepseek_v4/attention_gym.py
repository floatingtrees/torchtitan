# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym hybrid attention override for DeepSeek V4.

Install the dependency versions documented in ``attention_gym_csa``, then
activate this module with::

    --override.imports torchtitan.models.deepseek_v4.attention_gym

The override dispatches by the layer's compression ratio:

- ratio 1 -> sliding-window attention (SWA)
- ratio 4 -> compressed sparse attention (CSA)
- other ratios -> heavily compressed attention (HCA)

The CuTe kernels target SM100, CUDA 13.3, BF16, shared KV, and an attention
head dimension of 512.
"""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch

from torchtitan.config import derive, override
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig

from .attention import Attention
from .attention_gym_csa import (
    _activation_layout,
    _csa_sharding_config,
    _prepare_attention_gym_for_cuda_graphs,
    _rename_attention_input_sharding,
    AttentionGymCSAAttention,
    AttentionGymCSAKernel,
)

# Shape suffix legend for this file:
# B: batch, L: sequence, M: model, H: attention heads, D: attention head,
# R: compression ratio.

try:
    from attn_gym.sparse import heavily_compressed_attention, sliding_window_attention

    _ATTENTION_GYM_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    heavily_compressed_attention = None
    sliding_window_attention = None
    _ATTENTION_GYM_IMPORT_ERROR = error


def _hca_sharding_config() -> ShardingConfig:
    q_BHLD = _activation_layout(sequence_axis=2, tp=spmd.S(1))
    shared_B1LD = _activation_layout(sequence_axis=2, tp=spmd.R)
    shared_grad_B1LD = _activation_layout(sequence_axis=2, tp=spmd.P)
    replicated_param = dense_param_placement(tp=spmd.R)
    partial_param = dense_param_placement(tp=spmd.P)
    sink_H = dense_param_placement(tp=spmd.S(0))

    input_shardings = {
        "q_BHLD": q_BHLD,
        "kv_B1LD": shared_B1LD,
        "c_B1LD": shared_B1LD,
        "z_B1LD": shared_B1LD,
        "b_RD": replicated_param,
        "kv_norm_weight_D": replicated_param,
        "compressed_kv_norm_weight_D": replicated_param,
        "attention_sink_H": sink_H,
    }
    return ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=q_BHLD,
        local_map=LocalMapConfig(
            in_grad_placements=(
                q_BHLD,
                shared_grad_B1LD,
                shared_grad_B1LD,
                shared_grad_B1LD,
                partial_param,
                partial_param,
                partial_param,
                sink_H,
            )
        ),
    )


def _swa_sharding_config() -> ShardingConfig:
    q_BHLD = _activation_layout(sequence_axis=2, tp=spmd.S(1))
    shared_B1LD = _activation_layout(sequence_axis=2, tp=spmd.R)
    shared_grad_B1LD = _activation_layout(sequence_axis=2, tp=spmd.P)
    replicated_param = dense_param_placement(tp=spmd.R)
    partial_param = dense_param_placement(tp=spmd.P)
    sink_H = dense_param_placement(tp=spmd.S(0))

    input_shardings = {
        "q_BHLD": q_BHLD,
        "kv_B1LD": shared_B1LD,
        "kv_norm_weight_D": replicated_param,
        "attention_sink_H": sink_H,
    }
    return ShardingConfig(
        in_src_shardings=input_shardings,
        in_dst_shardings=dict(input_shardings),
        out_src_shardings=q_BHLD,
        local_map=LocalMapConfig(
            in_grad_placements=(
                q_BHLD,
                shared_grad_B1LD,
                partial_param,
                sink_H,
            )
        ),
    )


class AttentionGymHCAKernel(Module):
    """Local-tensor boundary around Attention Gym's SM100 CuTe HCA kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        compression_ratio: int
        sliding_window_size: int
        rope_dims: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.compression_ratio = config.compression_ratio
        self.sliding_window_size = config.sliding_window_size
        self.rope_dims = config.rope_dims

    def forward(
        self,
        q_BHLD: torch.Tensor,
        kv_B1LD: torch.Tensor,
        c_B1LD: torch.Tensor,
        z_B1LD: torch.Tensor,
        b_RD: torch.Tensor,
        kv_norm_weight_D: torch.Tensor,
        compressed_kv_norm_weight_D: torch.Tensor,
        attention_sink_H: torch.Tensor,
    ) -> torch.Tensor:
        assert heavily_compressed_attention is not None
        return heavily_compressed_attention(
            q_BHLD,
            kv_B1LD,
            c_B1LD,
            z_B1LD,
            b_RD,
            kv_norm_weight_D,
            compressed_kv_norm_weight_D,
            attention_sink_H,
            self.compression_ratio,
            self.sliding_window_size,
            self.rope_dims,
            True,
            backend="cute",
        )


class AttentionGymSWAKernel(Module):
    """Local-tensor boundary around Attention Gym's SM100 CuTe SWA kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        sliding_window_size: int
        rope_dims: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.sliding_window_size = config.sliding_window_size
        self.rope_dims = config.rope_dims

    def forward(
        self,
        q_BHLD: torch.Tensor,
        kv_B1LD: torch.Tensor,
        kv_norm_weight_D: torch.Tensor,
        attention_sink_H: torch.Tensor,
    ) -> torch.Tensor:
        assert sliding_window_attention is not None
        return sliding_window_attention(
            q_BHLD,
            kv_B1LD,
            kv_norm_weight_D,
            attention_sink_H,
            self.sliding_window_size,
            self.rope_dims,
            True,
            backend="cute",
        )


class AttentionGymAttention(AttentionGymCSAAttention):
    """DeepSeek V4 hybrid attention using Attention Gym CuTe kernels."""

    @dataclass(kw_only=True, slots=True)
    class Config(AttentionGymCSAAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        assert config.hca_kernel is not None
        assert config.swa_kernel is not None
        self.hca_kernel = config.hca_kernel.build()
        self.swa_kernel = config.swa_kernel.build()

    def forward(self, x_BLM, attention_masks=None, positions=None):
        if self.compress_ratio == 4:
            return super().forward(x_BLM, attention_masks, positions)

        _, q_BHLD = self._query_inputs(x_BLM)
        kv_B1LD = self.wkv(x_BLM).unsqueeze(1).contiguous()
        attention_sink_H = self.attn_sink.weight.squeeze(-1)

        if self.compress_ratio == 1:
            out_BHLD = self.swa_kernel(
                q_BHLD,
                kv_B1LD,
                self.kv_norm.weight,
                attention_sink_H,
            )
        else:
            compressor = self.compressor_128
            c_B1LD = compressor.wkv(x_BLM).unsqueeze(1).contiguous()
            z_B1LD = compressor.wgate(x_BLM).unsqueeze(1).contiguous()
            out_BHLD = self.hca_kernel(
                q_BHLD,
                kv_B1LD,
                c_B1LD,
                z_B1LD,
                compressor.ape.weight,
                self.kv_norm.weight,
                compressor.norm.weight,
                attention_sink_H,
            )
        return self._project_output(out_BHLD)


@override(
    "deepseek_v4_attention_gym",
    target=Attention.Config,
    exact=True,
    description="Attention Gym SM100 CuTe SWA, CSA, and HCA for DeepSeek V4.",
)
def attention_gym(cfg: Attention.Config) -> AttentionGymAttention.Config:
    if _ATTENTION_GYM_IMPORT_ERROR is not None:
        raise ImportError(
            "The DeepSeek V4 Attention Gym override requires the "
            "floatingtrees/attention-gym fork with sparse dependencies."
        ) from _ATTENTION_GYM_IMPORT_ERROR

    _prepare_attention_gym_for_cuda_graphs()

    csa_kernel = AttentionGymCSAKernel.Config(
        compression_ratio=cfg.compress_ratio,
        num_topk_blocks=cfg.index_topk,
        sliding_window_size=cfg.window_size,
        rope_dims=cfg.rope_head_dim,
        sharding_config=_csa_sharding_config(),
    )
    hca_kernel = AttentionGymHCAKernel.Config(
        compression_ratio=cfg.compress_ratio,
        sliding_window_size=cfg.window_size,
        rope_dims=cfg.rope_head_dim,
        sharding_config=_hca_sharding_config(),
    )
    swa_kernel = AttentionGymSWAKernel.Config(
        sliding_window_size=cfg.window_size,
        rope_dims=cfg.rope_head_dim,
        sharding_config=_swa_sharding_config(),
    )
    return derive(
        cfg,
        AttentionGymAttention.Config,
        csa_kernel=csa_kernel,
        hca_kernel=hca_kernel,
        swa_kernel=swa_kernel,
        sharding_config=_rename_attention_input_sharding(cfg.sharding_config),
    )


__all__ = [
    "AttentionGymAttention",
    "AttentionGymHCAKernel",
    "AttentionGymSWAKernel",
]
