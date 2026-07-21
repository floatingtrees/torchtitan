# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention Gym Triton compressed sparse attention for DeepSeek V4.

Install the validated ``floatingtrees/attention-gym`` revision, then activate
this module with::

    pip install git+https://github.com/floatingtrees/attention-gym.git@437fe90

    --override.imports torchtitan.overrides.deepseek_v4_attention_gym_csa

Only DeepSeek V4 layers with compression ratio 4 use the CSA kernel. Other
layers retain the stock attention implementation. The Triton backend supports
attention head dimensions up to and including 256.

The kernel does not apply the indexer's Hadamard rotation or FP4 quantization.
This override intentionally omits both until the kernel supports them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import spmd_types as spmd
import torch

from torchtitan.config import derive, override
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.deepseek_v4.attention import Attention
from torchtitan.models.deepseek_v4.compressor import Compressor
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import LocalMapConfig, ShardingConfig, SpmdLayout

# Shape suffix legend for this file:
# B: batch, L: sequence, M: model, A: query LoRA, H: attention heads,
# D: attention head, I: index heads, J: index head, C: compressor channel,
# R: compression ratio.

try:
    from attn_gym.sparse.compressed_sparse_attention.api import (
        compressed_sparse_attention,
    )

    _ATTENTION_GYM_IMPORT_ERROR: ImportError | None = None
except ImportError as error:
    compressed_sparse_attention = None
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


def _csa_sharding_config() -> ShardingConfig:
    q_BHLD = _activation_layout(sequence_axis=2, tp=spmd.S(1))
    shared_B1LD = _activation_layout(sequence_axis=2, tp=spmd.R)
    shared_grad_B1LD = _activation_layout(sequence_axis=2, tp=spmd.P)
    index_weight_BLI = _activation_layout(sequence_axis=1, tp=spmd.R)
    replicated_param = dense_param_placement(tp=spmd.R)
    partial_param = dense_param_placement(tp=spmd.P)
    sink_H = dense_param_placement(tp=spmd.S(0))

    input_shardings = {
        "q_BHLD": q_BHLD,
        "q_index_BILJ": shared_B1LD,
        "kv_B1LD": shared_B1LD,
        "c_a_B1LD": shared_B1LD,
        "c_b_B1LD": shared_B1LD,
        "z_a_B1LD": shared_B1LD,
        "z_b_B1LD": shared_B1LD,
        "b_a_RD": replicated_param,
        "b_b_RD": replicated_param,
        "index_weight_BLI": index_weight_BLI,
        "k_index_a_B1LJ": shared_B1LD,
        "k_index_b_B1LJ": shared_B1LD,
        "z_index_a_B1LJ": shared_B1LD,
        "z_index_b_B1LJ": shared_B1LD,
        "b_index_a_RJ": replicated_param,
        "b_index_b_RJ": replicated_param,
        "kv_norm_weight_D": replicated_param,
        "index_norm_weight_J": replicated_param,
        "compressed_kv_norm_weight_D": replicated_param,
        "attention_sink_H": sink_H,
    }
    grad_placements = (
        q_BHLD,
        shared_B1LD,
        shared_grad_B1LD,
        shared_grad_B1LD,
        shared_grad_B1LD,
        shared_grad_B1LD,
        shared_grad_B1LD,
        partial_param,
        partial_param,
        index_weight_BLI,
        shared_B1LD,
        shared_B1LD,
        shared_B1LD,
        shared_B1LD,
        replicated_param,
        replicated_param,
        partial_param,
        replicated_param,
        partial_param,
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


class AttentionGymTritonCSAKernel(Module):
    """Local-tensor boundary around Attention Gym's Triton CSA kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        compression_ratio: int
        num_topk_blocks: int
        sliding_window_size: int
        rope_dims: int

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.compression_ratio = config.compression_ratio
        self.num_topk_blocks = config.num_topk_blocks
        self.sliding_window_size = config.sliding_window_size
        self.rope_dims = config.rope_dims

    def forward(
        self,
        q_BHLD: torch.Tensor,
        q_index_BILJ: torch.Tensor,
        kv_B1LD: torch.Tensor,
        c_a_B1LD: torch.Tensor,
        c_b_B1LD: torch.Tensor,
        z_a_B1LD: torch.Tensor,
        z_b_B1LD: torch.Tensor,
        b_a_RD: torch.Tensor,
        b_b_RD: torch.Tensor,
        index_weight_BLI: torch.Tensor,
        k_index_a_B1LJ: torch.Tensor,
        k_index_b_B1LJ: torch.Tensor,
        z_index_a_B1LJ: torch.Tensor,
        z_index_b_B1LJ: torch.Tensor,
        b_index_a_RJ: torch.Tensor,
        b_index_b_RJ: torch.Tensor,
        kv_norm_weight_D: torch.Tensor,
        index_norm_weight_J: torch.Tensor,
        compressed_kv_norm_weight_D: torch.Tensor,
        attention_sink_H: torch.Tensor,
    ) -> torch.Tensor:
        assert compressed_sparse_attention is not None
        return compressed_sparse_attention(
            q_BHLD,
            q_index_BILJ,
            kv_B1LD,
            c_a_B1LD,
            c_b_B1LD,
            z_a_B1LD,
            z_b_B1LD,
            b_a_RD,
            b_b_RD,
            index_weight_BLI,
            k_index_a_B1LJ,
            k_index_b_B1LJ,
            z_index_a_B1LJ,
            z_index_b_B1LJ,
            b_index_a_RJ,
            b_index_b_RJ,
            kv_norm_weight_D,
            index_norm_weight_J,
            compressed_kv_norm_weight_D,
            attention_sink_H,
            self.compression_ratio,
            self.num_topk_blocks,
            self.sliding_window_size,
            self.rope_dims,
            True,
            backend="triton",
        )


class AttentionGymTritonCSAAttention(Attention):
    """DeepSeek V4 attention using Attention Gym CSA on ratio-4 layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Attention.Config):
        csa_kernel: AttentionGymTritonCSAKernel.Config

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.csa_kernel = config.csa_kernel.build()

    @staticmethod
    def _compressor_inputs(
        x_BLM: torch.Tensor,
        compressor: Compressor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        head_dim = compressor.head_dim
        kv_BLCD = compressor.wkv(x_BLM).unflatten(-1, (2, head_dim))
        gate_BLCD = compressor.wgate(x_BLM).unflatten(-1, (2, head_dim))
        bias_RCD = compressor.ape.weight.unflatten(-1, (2, head_dim))

        # Channel 1 covers the current block. Channel 0 is shifted back one
        # block by Attention Gym to reproduce DeepSeek V4's overlap transform.
        c_a_B1LD = kv_BLCD[:, :, 1, :].unsqueeze(1).contiguous()
        c_b_B1LD = kv_BLCD[:, :, 0, :].unsqueeze(1).contiguous()
        z_a_B1LD = gate_BLCD[:, :, 1, :].unsqueeze(1).contiguous()
        z_b_B1LD = gate_BLCD[:, :, 0, :].unsqueeze(1).contiguous()
        b_a_RD = bias_RCD[:, 1, :].contiguous()
        b_b_RD = bias_RCD[:, 0, :].contiguous()
        return c_a_B1LD, c_b_B1LD, z_a_B1LD, z_b_B1LD, b_a_RD, b_b_RD

    def forward(self, x_BLM, attention_masks=None, positions=None):
        if self.compress_ratio != 4:
            return super().forward(x_BLM, attention_masks, positions)

        bsz, seqlen, _ = x_BLM.shape
        qr_BLA = self.q_norm(self.wq_a(x_BLM))

        q_BLHD = self.wq_b(qr_BLA).unflatten(-1, (self.n_heads, self.head_dim))
        q_BLHD = q_BLHD * torch.rsqrt(
            q_BLHD.square().mean(-1, keepdim=True) + self.norm_eps
        )
        q_BHLD = q_BLHD.transpose(1, 2).contiguous()

        q_index_BLIJ = self.indexer.wq_b(qr_BLA).unflatten(
            -1, (self.indexer.num_index_heads, self.indexer.head_dim)
        )
        q_index_BILJ = q_index_BLIJ.transpose(1, 2).contiguous().detach()

        kv_B1LD = self.wkv(x_BLM).unsqueeze(1)
        (
            c_a_B1LD,
            c_b_B1LD,
            z_a_B1LD,
            z_b_B1LD,
            b_a_RD,
            b_b_RD,
        ) = self._compressor_inputs(x_BLM, self.compressor)
        (
            k_index_a_B1LJ,
            k_index_b_B1LJ,
            z_index_a_B1LJ,
            z_index_b_B1LJ,
            b_index_a_RJ,
            b_index_b_RJ,
        ) = tuple(
            tensor.detach()
            for tensor in self._compressor_inputs(
                x_BLM.detach(), self.indexer.compressor
            )
        )

        index_weight_BLI = self.indexer.weights_proj(x_BLM.detach()).detach()
        attention_sink_H = self.attn_sink.weight.squeeze(-1)
        out_BHLD = self.csa_kernel(
            q_BHLD,
            q_index_BILJ,
            kv_B1LD,
            c_a_B1LD,
            c_b_B1LD,
            z_a_B1LD,
            z_b_B1LD,
            b_a_RD,
            b_b_RD,
            index_weight_BLI,
            k_index_a_B1LJ,
            k_index_b_B1LJ,
            z_index_a_B1LJ,
            z_index_b_B1LJ,
            b_index_a_RJ,
            b_index_b_RJ,
            self.kv_norm.weight,
            self.indexer.compressor.norm.weight.detach(),
            self.compressor.norm.weight,
            attention_sink_H,
        )

        out_BLHD = out_BHLD.transpose(1, 2).contiguous()
        num_local_groups = self.n_groups // (self.n_heads // out_BLHD.shape[2])
        out_BLGD = out_BLHD.view(bsz, seqlen, num_local_groups, -1)
        wo_a_GAD = self.wo_a.weight.view(num_local_groups, self.o_lora_rank, -1)
        out_BLGA = torch.einsum("blgd,gad->blga", out_BLGD, wo_a_GAD)
        return self.wo_b(out_BLGA.reshape(bsz, seqlen, -1))


@override(
    "deepseek_v4_attention_gym_triton_csa",
    target=Attention.Config,
    exact=True,
    description="Attention Gym Triton CSA for DeepSeek V4 ratio-4 layers.",
)
def attention_gym_csa(cfg: Attention.Config) -> AttentionGymTritonCSAAttention.Config:
    if _ATTENTION_GYM_IMPORT_ERROR is not None:
        raise ImportError(
            "The DeepSeek V4 CSA override requires the floatingtrees/attention-gym "
            "fork to be installed."
        ) from _ATTENTION_GYM_IMPORT_ERROR
    if cfg.head_dim > 256:
        raise ValueError(
            "The Attention Gym Triton CSA backend requires head_dim <= 256, "
            f"but DeepSeek V4 configured head_dim={cfg.head_dim}."
        )

    csa_kernel = AttentionGymTritonCSAKernel.Config(
        compression_ratio=cfg.compress_ratio,
        num_topk_blocks=cfg.index_topk,
        sliding_window_size=cfg.window_size,
        rope_dims=cfg.rope_head_dim,
        sharding_config=_csa_sharding_config(),
    )
    return derive(
        cfg,
        AttentionGymTritonCSAAttention.Config,
        csa_kernel=csa_kernel,
        sharding_config=_rename_attention_input_sharding(cfg.sharding_config),
    )


__all__ = ["AttentionGymTritonCSAAttention", "AttentionGymTritonCSAKernel"]
