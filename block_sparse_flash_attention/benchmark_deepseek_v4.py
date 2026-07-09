"""Benchmark: DeepSeek V4 debug model training step.

Runs the debug model for N training steps with:
  1. flex_attention (current default)
  2. block-sparse CuTe DSL kernels (our custom implementation)

Reports time per step and TFLOPs for each.

Usage:
    python block_sparse_flash_attention/benchmark_deepseek_v4.py
"""
import sys
sys.path.insert(0, '/nvdl/torchtitan')

import os
import time
import math

import torch
import torch.distributed as dist

from torchtitan.models.deepseek_v4 import model_registry, DeepSeekV4Model
from torchtitan.models.deepseek_v4.config_registry import deepseek_v4_debugmodel


def setup_single_gpu():
    """Initialize minimal distributed env for single-GPU run."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def build_model(use_block_sparse=False):
    """Build the DeepSeek V4 debug model.

    If use_block_sparse=True, monkey-patches DSAFlexAttention with our
    block-sparse CuTe DSL implementation.
    """
    if use_block_sparse:
        # Patch the DSAFlexAttention config to use our implementation
        from block_sparse_flash_attention.dsa_block_sparse_attention import (
            DSABlockSparseAttention,
        )
        import torchtitan.models.deepseek_v4.attention as dsa_attn_module

        # Save original
        OriginalDSAFlexAttention = dsa_attn_module.DSAFlexAttention

        # Replace DSAFlexAttention with our block-sparse version
        # We need to make DSABlockSparseAttention accept the same Config
        class PatchedDSABlockSparse(DSABlockSparseAttention):
            """Patched version that accepts DSAFlexAttention.Config."""
            @classmethod
            def from_flex_config(cls, config):
                return cls(DSABlockSparseAttention.Config(
                    window_size=config.window_size,
                    compress_ratio=config.compress_ratio,
                    softmax_scale=config.softmax_scale,
                    block_size=(256, 128),
                ))

        # Monkey-patch the build method
        original_build = OriginalDSAFlexAttention.Config.build

        def patched_build(self):
            return PatchedDSABlockSparse(DSABlockSparseAttention.Config(
                window_size=self.window_size,
                compress_ratio=self.compress_ratio,
                softmax_scale=self.softmax_scale,
                block_size=(256, 128),
            ))

        OriginalDSAFlexAttention.Config.build = patched_build

    # Build model from registry
    model_spec = model_registry("debugmodel")
    model_config = model_spec.model
    model = model_config.build()

    if use_block_sparse:
        # Restore original
        OriginalDSAFlexAttention.Config.build = original_build

    return model, model_config


def run_training_steps(model, seq_len, batch_size, vocab_size, num_steps, device, label=""):
    """Run training steps and measure time per step."""
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)

    # Warmup: compile + first few iterations
    print(f"  [{label}] Warming up (3 steps)...")
    for i in range(3):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        optimizer.zero_grad()
        output = model(input_ids, positions=positions)
        # Simple loss: cross-entropy on next token prediction
        logits = output[:, :-1, :].contiguous().view(-1, vocab_size)
        targets = input_ids[:, 1:].contiguous().view(-1)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    print(f"  [{label}] Warmup done. Running {num_steps} timed steps...")

    # Timed steps
    times = []
    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad()
        output = model(input_ids, positions=positions)
        logits = output[:, :-1, :].contiguous().view(-1, vocab_size)
        targets = input_ids[:, 1:].contiguous().view(-1)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_time_ms = (t1 - t0) * 1000
        times.append(step_time_ms)
        print(f"    step {step}: {step_time_ms:.2f} ms  loss={loss.item():.4f}")

    return times


def compute_model_tflops(model_config, batch_size, seq_len, time_ms):
    """Estimate TFLOPs for the model training step.

    Uses standard transformer FLOP estimation:
    FLOPs per step ~ 6 * num_params * batch_size * seq_len (for fwd + bwd)
    Plus attention FLOPs: 4 * n_layers * n_heads * seq_len^2 * head_dim * batch_size * 3
    (factor of 3 for fwd + bwd)
    """
    # Count parameters
    num_params = sum(p.numel() for p in model_config.build().parameters())

    # Simplified: 6 * N * B * S for linear layers (fwd+bwd+weight_update)
    linear_flops = 6 * num_params * batch_size * seq_len

    # Attention FLOPs (approximate for sparse attention -- use full for comparison)
    n_layers = model_config.n_layers
    n_heads = 16  # debug model
    head_dim = 256
    # 4 * (QK + PV) * 3 (fwd + 2.5x bwd) per layer
    attn_flops = 4 * n_layers * n_heads * seq_len * seq_len * head_dim * batch_size * 3

    total_flops = linear_flops + attn_flops
    tflops = total_flops / (time_ms * 1e-3) / 1e12
    return tflops, total_flops


def main():
    device = "cuda"
    num_steps = 10

    setup_single_gpu()

    # Get config params
    trainer_config = deepseek_v4_debugmodel()
    seq_len = trainer_config.training.seq_len
    batch_size = trainer_config.training.local_batch_size
    vocab_size = 2048  # debug model vocab

    print("=" * 80)
    print("  DEEPSEEK V4 DEBUG MODEL: FLEX ATTENTION vs BLOCK-SPARSE CuTe DSL")
    print(f"  seq_len={seq_len}, batch_size={batch_size}, steps={num_steps}")
    print(f"  head_dim=256, n_heads=16, n_layers=4, window_size=16")
    print("=" * 80)

    # --- Run with flex attention (default) ---
    print("\n" + "=" * 80)
    print("  [1] FLEX ATTENTION (Triton backend)")
    print("=" * 80)
    model_flex, model_config = build_model(use_block_sparse=False)
    times_flex = run_training_steps(
        model_flex, seq_len, batch_size, vocab_size, num_steps, device, label="FLEX"
    )
    del model_flex
    torch.cuda.empty_cache()

    # --- Run with block-sparse CuTe DSL kernels ---
    print("\n" + "=" * 80)
    print("  [2] BLOCK-SPARSE CuTe DSL KERNELS")
    print("=" * 80)
    model_sparse, _ = build_model(use_block_sparse=True)
    times_sparse = run_training_steps(
        model_sparse, seq_len, batch_size, vocab_size, num_steps, device, label="SPARSE"
    )
    del model_sparse
    torch.cuda.empty_cache()

    # --- Results ---
    print("\n\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)

    median_flex = sorted(times_flex)[len(times_flex) // 2]
    median_sparse = sorted(times_sparse)[len(times_sparse) // 2]
    mean_flex = sum(times_flex) / len(times_flex)
    mean_sparse = sum(times_sparse) / len(times_sparse)

    # Rough TFLOPS estimate
    num_params = sum(
        p.numel() for p in build_model(use_block_sparse=False)[0].parameters()
    )
    # 6 * N * B * S for the full model (linear layers dominate)
    step_flops = 6 * num_params * batch_size * seq_len
    tflops_flex = step_flops / (median_flex * 1e-3) / 1e12
    tflops_sparse = step_flops / (median_sparse * 1e-3) / 1e12

    speedup = median_flex / median_sparse

    print(f"\n  Model params: {num_params:,}")
    print(f"  Step FLOPs (6*N*B*S): {step_flops:.2e}")
    print()
    print(f"  {'Metric':<25} {'Flex Attention':>18} {'Block-Sparse CuTe':>18} {'Speedup':>10}")
    print(f"  {'-'*25} {'-'*18} {'-'*18} {'-'*10}")
    print(f"  {'Median time/step (ms)':<25} {median_flex:>18.2f} {median_sparse:>18.2f} {speedup:>9.2f}x")
    print(f"  {'Mean time/step (ms)':<25} {mean_flex:>18.2f} {mean_sparse:>18.2f} {mean_flex/mean_sparse:>9.2f}x")
    print(f"  {'TFLOPS (6*N*B*S/time)':<25} {tflops_flex:>18.2f} {tflops_sparse:>18.2f}")
    print()

    # Per-step breakdown
    print(f"  Per-step times (ms):")
    print(f"  {'Step':<6} {'Flex':>10} {'Sparse':>10} {'Ratio':>8}")
    for i, (tf, ts) in enumerate(zip(times_flex, times_sparse)):
        print(f"  {i:<6} {tf:>10.2f} {ts:>10.2f} {tf/ts:>7.2f}x")

    print()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
