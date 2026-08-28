# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from torchtitan.components.checkpoint import CheckpointManager

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common import ComplexRoPE
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import _build_v4_layers, model_registry


def deepseek_v4_debugmodel() -> Trainer.Config:
    return Trainer.Config(
        loss=CrossEntropyLoss.Config(),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("debugmodel"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=8,
            seq_len=2048,
            steps=10,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        activation_checkpoint=None,
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_large_debug_config() -> Trainer.Config:
    config = deepseek_v4_debugmodel()
    assert config.model_spec is not None

    num_layers = 8
    compression_ratios = (1, 4, 128, 4, 1, 4, 128, 4)
    rope = ComplexRoPE.Config(
        dim=64,
        max_seq_len=4096 * 4,
        theta=10000.0,
        scaling="none",
    )
    compressed_rope = ComplexRoPE.Config(
        dim=64,
        max_seq_len=4096 * 4,
        theta=40000.0,
        scaling="yarn",
        rope_factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=65536,
    )
    layers = _build_v4_layers(
        n_layers=num_layers,
        dim=256,
        n_heads=128,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        n_groups=2,
        compress_ratios=compression_ratios,
        window_size=512,
        norm_eps=1e-6,
        index_n_heads=64,
        index_head_dim=64,
        index_topk=64,
        moe_inter_dim=256,
        num_experts=4,
        num_shared_experts=1,
        top_k=1,
        vocab_size=2048,
        n_hash_layers=2,
        route_norm=False,
        route_scale=1.5,
        load_balance_coeff=1e-3,
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
        rope=rope,
        rope_compress=compressed_rope,
        hc_mult=4,
        sinkhorn_iters=20,
        hc_eps=1e-6,
        dense_layers=set(),
    )

    config.model_spec.model = replace(
        config.model_spec.model,
        layers=layers,
        compress_ratios=compression_ratios,
        n_layers=num_layers,
    )
    config.model_spec.flavor = "large_debug"
    config.training = replace(
        config.training,
        local_batch_size=8,
        seq_len=4096,
        steps=126,
        dtype="bfloat16",
        disable_cuda_graphs=False,
    )
    config.activation_checkpoint = FullAC.Config()
    config.compile = replace(
        config.compile,
        enable=True,
        components=["model"],
        backend="inductor",
    )
    config.override.imports.append("torchtitan.models.deepseek_v4.attention_gym_csa")
    return config


def deepseek_v4_flash_config() -> Trainer.Config:
    config = deepseek_v4_debugmodel()
    config.model_spec = model_registry("flash")
    config.model_spec.flavor = "flash"
    config.training = replace(
        config.training,
        local_batch_size=1,
        seq_len=4096,
        steps=100,
        dtype="bfloat16",
        disable_cuda_graphs=False,
    )
    config.activation_checkpoint = FullAC.Config()
    config.compile = replace(
        config.compile,
        enable=True,
        components=["model"],
        backend="inductor",
    )
    config.override.imports.append("torchtitan.models.deepseek_v4.attention_gym_csa")
    return config


def deepseek_v4_pro_config() -> Trainer.Config:
    config = deepseek_v4_debugmodel()
    config.model_spec = model_registry("pro")
    config.model_spec.flavor = "pro"
    config.training = replace(
        config.training,
        local_batch_size=1,
        seq_len=4096,
        steps=100,
        dtype="bfloat16",
        disable_cuda_graphs=False,
    )
    config.activation_checkpoint = FullAC.Config()
    config.compile = replace(
        config.compile,
        enable=True,
        components=["model"],
        backend="inductor",
    )
    config.override.imports.append("torchtitan.models.deepseek_v4.attention_gym_csa")
    return config
