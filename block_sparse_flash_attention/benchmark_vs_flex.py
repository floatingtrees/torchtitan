"""Benchmark: Block-Sparse CuTe DSL Kernels vs Flex Attention (Triton backend).

Runs a simulated training step (forward + backward) using both backends and
reports wall-clock time per step and TFLOPs for each configuration.

Requires a Blackwell GPU (SM100) for the CuTe DSL kernels.

Usage:
    python block_sparse_flash_attention/benchmark_vs_flex.py
"""
import sys
sys.path.insert(0, '/nvdl/torchtitan')

import math
import time

import torch
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_config,
    to_cute_block_sparse_tensors,
)
from flash_attn.cute.utils import AuxData
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
import cutlass.cute as cute

from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
)

from block_sparse_flash_attention.kernel_fwd import BlackwellFusedMultiHeadAttentionForward
from block_sparse_flash_attention.kernel_bwd import BlackwellFusedMultiHeadAttentionBackward


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ceildiv(a, b):
    return (a + b - 1) // b


def create_causal_block_mask_sparse(B, H, S_q, S_k, bsq, bsk, device):
    """Block mask for standard causal attention."""
    num_m = ceildiv(S_q, bsq)
    num_n = ceildiv(S_k, bsk)
    mask_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    mask_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)
    full_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    full_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)
    for m in range(num_m):
        qs, qe = m * bsq, min((m + 1) * bsq, S_q)
        fc, mc = 0, 0
        for n in range(num_n):
            ks, ke = n * bsk, min((n + 1) * bsk, S_k)
            if qs >= ke - 1:
                full_idx[:, :, m, fc] = n
                fc += 1
            elif qe - 1 >= ks:
                mask_idx[:, :, m, mc] = n
                mc += 1
        mask_cnt[:, :, m] = mc
        full_cnt[:, :, m] = fc
    return mask_cnt, mask_idx, full_cnt, full_idx


def create_sliding_window_block_mask_sparse(B, H, S_q, S_k, bsq, bsk, window_size, device):
    """Block mask for causal + sliding window attention."""
    num_m = ceildiv(S_q, bsq)
    num_n = ceildiv(S_k, bsk)
    mask_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    mask_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)
    full_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    full_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)
    for m in range(num_m):
        qs, qe = m * bsq, min((m + 1) * bsq, S_q)
        fc, mc = 0, 0
        for n in range(num_n):
            ks, ke = n * bsk, min((n + 1) * bsk, S_k)
            last_q_window_start = (qe - 1) - window_size + 1
            block_fully_in_window = ks >= last_q_window_start
            if not block_fully_in_window:
                continue
            causal_ok = (qe - 1) >= ks
            if not causal_ok:
                continue
            causal_full = qs >= (ke - 1)
            first_q_window_start = qs - window_size + 1
            window_full = ks >= first_q_window_start
            if causal_full and window_full:
                full_idx[:, :, m, fc] = n
                fc += 1
            else:
                mask_idx[:, :, m, mc] = n
                mc += 1
        mask_cnt[:, :, m] = mc
        full_cnt[:, :, m] = fc
    return mask_cnt, mask_idx, full_cnt, full_idx


def create_sliding_window_plus_random_block_mask_sparse(
    B, H, S_q, S_k, bsq, bsk, window_size, num_random_tokens, device, seed=42
):
    """Block mask for causal + sliding window + random token access."""
    rng = torch.Generator(device="cpu").manual_seed(seed)
    random_positions = torch.randperm(S_k, generator=rng)[:num_random_tokens].sort().values
    random_block_set = set()
    for pos in random_positions.tolist():
        random_block_set.add(pos // bsk)

    num_m = ceildiv(S_q, bsq)
    num_n = ceildiv(S_k, bsk)
    mask_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    mask_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)
    full_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    full_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)

    for m in range(num_m):
        qs, qe = m * bsq, min((m + 1) * bsq, S_q)
        fc, mc = 0, 0
        for n in range(num_n):
            ks, ke = n * bsk, min((n + 1) * bsk, S_k)
            causal_ok = (qe - 1) >= ks
            if not causal_ok:
                continue
            last_q_window_start = (qe - 1) - window_size + 1
            block_in_window = ks >= last_q_window_start
            has_random = n in random_block_set
            if not block_in_window and not has_random:
                continue
            causal_full = qs >= (ke - 1)
            first_q_window_start = qs - window_size + 1
            window_full = ks >= first_q_window_start
            if causal_full and window_full:
                full_idx[:, :, m, fc] = n
                fc += 1
            else:
                mask_idx[:, :, m, mc] = n
                mc += 1
        mask_cnt[:, :, m] = mc
        full_cnt[:, :, m] = fc
    return mask_cnt, mask_idx, full_cnt, full_idx, random_positions


def transpose_block_sparse(mask_cnt, mask_idx, full_cnt, full_idx, num_m, num_n, B, H, device):
    """Transpose block-sparse indices: forward (m->n) to backward (n->m)."""
    bwd_mask_cnt = torch.zeros(B, H, num_n, dtype=torch.int32, device=device)
    bwd_mask_idx = torch.zeros(B, H, num_n, num_m, dtype=torch.int32, device=device)
    bwd_full_cnt = torch.zeros(B, H, num_n, dtype=torch.int32, device=device)
    bwd_full_idx = torch.zeros(B, H, num_n, num_m, dtype=torch.int32, device=device)

    mc = mask_cnt.cpu().numpy()
    mi = mask_idx.cpu().numpy()
    fc = full_cnt.cpu().numpy()
    fi = full_idx.cpu().numpy()
    bmc = bwd_mask_cnt.cpu().numpy()
    bmi = bwd_mask_idx.cpu().numpy()
    bfc = bwd_full_cnt.cpu().numpy()
    bfi = bwd_full_idx.cpu().numpy()

    for b in range(B):
        for h in range(H):
            for m in range(num_m):
                for i in range(mc[b, h, m]):
                    n = mi[b, h, m, i]
                    bmi[b, h, n, bmc[b, h, n]] = m
                    bmc[b, h, n] += 1
                for i in range(fc[b, h, m]):
                    n = fi[b, h, m, i]
                    bfi[b, h, n, bfc[b, h, n]] = m
                    bfc[b, h, n] += 1

    bwd_mask_cnt = torch.from_numpy(bmc).to(device=device, dtype=torch.int32)
    bwd_mask_idx = torch.from_numpy(bmi).to(device=device, dtype=torch.int32)
    bwd_full_cnt = torch.from_numpy(bfc).to(device=device, dtype=torch.int32)
    bwd_full_idx = torch.from_numpy(bfi).to(device=device, dtype=torch.int32)
    return bwd_mask_cnt, bwd_mask_idx, bwd_full_cnt, bwd_full_idx


def count_active_blocks(mask_cnt, full_cnt):
    """Count total active blocks (mask + full) across all batch/head/m-tiles."""
    return (mask_cnt.sum() + full_cnt.sum()).item()


def compute_attention_flops(B, H, S_q, S_k, D, num_active_blocks, bsq, bsk):
    """Compute attention FLOPs for the sparse pattern.

    Each active block contributes bsq * bsk element-pairs for QK and PV matmuls.
    Total FLOPs = 2 * (QK matmul) + 2 * (PV matmul)
                = 2 * (num_active * bsq * bsk * D) + 2 * (num_active * bsq * bsk * D)
                = 4 * num_active * bsq * bsk * D

    For dense causal: num_active_blocks ~ (num_m * num_n) / 2
    """
    flops_per_block = 4 * bsq * bsk * D
    total_flops = num_active_blocks * flops_per_block
    return total_flops


def compute_dense_causal_flops(B, H, S_q, S_k, D):
    """Approximate dense causal attention FLOPs (for TFLOPS comparison).

    Standard formula: 4 * B * H * S_q * S_k * D * (causal factor ~0.5)
    """
    return 2 * B * H * S_q * S_k * D


# ---------------------------------------------------------------------------
# Block-sparse CuTe DSL kernel compilation and execution
# ---------------------------------------------------------------------------

_fwd_compile_cache = {}
_bwd_compile_cache = {}


def compile_sparse_fwd(q, k, v, bst, scale, causal=True):
    """Compile the block-sparse forward kernel (cached)."""
    B, S_q, H, D = q.shape
    _, S_k, H_kv, _ = k.shape
    device = q.device

    out = torch.zeros_like(q)
    lse = torch.zeros(B, H, S_q, device=device, dtype=torch.float32)

    tile_m, tile_n, q_stage = 128, 128, 2
    normalized, _, _ = normalize_block_sparse_config(
        bst, batch_size=B, num_head=H, seqlen_q=S_q, seqlen_k=S_k,
        block_size=(tile_m, tile_n), q_stage=q_stage,
    )

    cache_key = (B, S_q, S_k, H, D, causal, q.dtype, "fwd")
    if cache_key not in _fwd_compile_cache:
        kernel_obj = BlackwellFusedMultiHeadAttentionForward(
            head_dim=D, qhead_per_kvhead=H // H_kv, is_causal=causal,
        )
        q_cute = to_cute_tensor(q)
        k_cute = to_cute_tensor(k)
        v_cute = to_cute_tensor(v)
        out_cute = to_cute_tensor(out)
        lse_cute = to_cute_tensor(lse, assumed_align=4)
        sparse_cute = to_cute_block_sparse_tensors(normalized)
        print(f"    Compiling forward kernel (S={S_q})...", end=" ", flush=True)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            kernel_obj, q_cute, k_cute, v_cute, out_cute, lse_cute,
            float(scale), None, None, None, None, None, None, None, None, None,
            sparse_cute, AuxData(), stream, options="--enable-tvm-ffi",
        )
        _fwd_compile_cache[cache_key] = (compiled, normalized)
        print("done")

    return _fwd_compile_cache[cache_key]


def run_sparse_fwd(q, k, v, compiled_fn, normalized, scale):
    """Execute the compiled forward kernel."""
    B, S_q, H, D = q.shape
    device = q.device
    out = torch.zeros_like(q)
    lse = torch.zeros(B, H, S_q, device=device, dtype=torch.float32)

    compiled_fn(
        q, k, v, out, lse, scale,
        None, None, None, None, None, None, None, None, None,
        (normalized.mask_block_cnt, normalized.mask_block_idx,
         normalized.full_block_cnt, normalized.full_block_idx,
         normalized.cu_total_m_blocks, normalized.cu_block_idx_offsets,
         normalized.dq_write_order, normalized.dq_write_order_full),
        AuxData(),
    )
    return out, lse


def compile_sparse_bwd(q, k, v, dO, out, lse, bst, bwd_tensors, scale, causal=True):
    """Compile the block-sparse backward kernel (cached)."""
    B, S_q, H, D = q.shape
    _, S_k, H_kv, _ = k.shape
    device = q.device

    tile_m, tile_n, q_stage = 128, 128, 2
    normalized_fwd, _, _ = normalize_block_sparse_config(
        bst, batch_size=B, num_head=H, seqlen_q=S_q, seqlen_k=S_k,
        block_size=(tile_m, tile_n), q_stage=q_stage,
    )
    bwd_mask_cnt, bwd_mask_idx, bwd_full_cnt, bwd_full_idx = bwd_tensors

    cache_key = (B, S_q, S_k, H, D, causal, q.dtype, "bwd")
    if cache_key not in _bwd_compile_cache:
        sparse_cute_fwd = to_cute_block_sparse_tensors(normalized_fwd)
        sparse_cute_bwd = to_cute_block_sparse_tensors(
            BlockSparseTensorsTorch(
                mask_block_cnt=bwd_mask_cnt, mask_block_idx=bwd_mask_idx,
                full_block_cnt=bwd_full_cnt, full_block_idx=bwd_full_idx,
                block_size=(tile_n, tile_m),
            )._replace(
                cu_total_m_blocks=None, cu_block_idx_offsets=None,
                dq_write_order=None, dq_write_order_full=None,
            )
        )
        kernel_obj = BlackwellFusedMultiHeadAttentionBackward(
            head_dim=D, is_causal=causal, use_2cta_instrs=True, cluster_size=2,
            blocksparse_tensors_fwd=sparse_cute_fwd,
            blocksparse_tensors_bwd=sparse_cute_bwd,
        )

        log2_e = math.log2(math.e)
        lse_log2 = lse * log2_e
        dpsum = (dO.float() * out.float()).sum(dim=-1).transpose(1, 2).contiguous()
        dQ = torch.zeros(B, S_q, H, D, device=device, dtype=q.dtype)
        dK = torch.zeros(B, S_k, H_kv, D, device=device, dtype=k.dtype)
        dV = torch.zeros(B, S_k, H_kv, D, device=device, dtype=v.dtype)

        print(f"    Compiling backward kernel (S={S_q})...", end=" ", flush=True)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            kernel_obj,
            to_cute_tensor(q), to_cute_tensor(k), to_cute_tensor(v),
            to_cute_tensor(dO),
            to_cute_tensor(lse_log2, assumed_align=4),
            to_cute_tensor(dpsum, assumed_align=4),
            to_cute_tensor(dQ), to_cute_tensor(dK), to_cute_tensor(dV),
            float(scale),
            None, None, None, None, None, None, None, None, None,
            AuxData(), sparse_cute_fwd, sparse_cute_bwd,
            stream, options="--enable-tvm-ffi",
        )
        _bwd_compile_cache[cache_key] = (compiled, normalized_fwd)
        print("done")

    return _bwd_compile_cache[cache_key]


def run_sparse_bwd(q, k, v, dO, out, lse, compiled_fn, normalized_fwd, bwd_tensors, scale):
    """Execute the compiled backward kernel."""
    B, S_q, H, D = q.shape
    _, S_k, H_kv, _ = k.shape
    device = q.device

    log2_e = math.log2(math.e)
    lse_log2 = lse * log2_e
    dpsum = (dO.float() * out.float()).sum(dim=-1).transpose(1, 2).contiguous()
    dQ = torch.zeros(B, S_q, H, D, device=device, dtype=q.dtype)
    dK = torch.zeros(B, S_k, H_kv, D, device=device, dtype=k.dtype)
    dV = torch.zeros(B, S_k, H_kv, D, device=device, dtype=v.dtype)

    bwd_mask_cnt, bwd_mask_idx, bwd_full_cnt, bwd_full_idx = bwd_tensors
    sparse_fwd_runtime = (
        normalized_fwd.mask_block_cnt, normalized_fwd.mask_block_idx,
        normalized_fwd.full_block_cnt, normalized_fwd.full_block_idx,
        normalized_fwd.cu_total_m_blocks, normalized_fwd.cu_block_idx_offsets,
        normalized_fwd.dq_write_order, normalized_fwd.dq_write_order_full,
    )
    sparse_bwd_runtime = (
        bwd_mask_cnt, bwd_mask_idx,
        bwd_full_cnt, bwd_full_idx,
        None, None, None, None,
    )

    compiled_fn(
        q, k, v, dO,
        lse_log2, dpsum,
        dQ, dK, dV,
        scale,
        None, None, None, None, None, None, None, None, None,
        AuxData(), sparse_fwd_runtime, sparse_bwd_runtime,
    )
    return dQ, dK, dV


# ---------------------------------------------------------------------------
# Flex attention (Triton backend)
# ---------------------------------------------------------------------------

_compiled_flex_attention = torch.compile(flex_attention)


def create_flex_causal_block_mask(B, H, S_q, S_k, device):
    def causal_mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return create_block_mask(causal_mask_mod, B, H, S_q, S_k, device=device, _compile=True)


def create_flex_sliding_window_block_mask(B, H, S_q, S_k, window_size, device):
    def sw_mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) < window_size
        return causal & in_window
    return create_block_mask(sw_mask_mod, B, H, S_q, S_k, device=device, _compile=True)


def create_flex_sliding_window_random_block_mask(
    B, H, S_q, S_k, window_size, random_positions, device
):
    is_random_lookup = torch.zeros(S_k, dtype=torch.bool, device=device)
    is_random_lookup[random_positions.to(device)] = True

    def swr_mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        in_window = (q_idx - kv_idx) < window_size
        is_random = is_random_lookup[kv_idx]
        return causal & (in_window | is_random)
    return create_block_mask(swr_mask_mod, B, H, S_q, S_k, device=device, _compile=True)


def run_flex_fwd_bwd(q_BSHD, k_BSHD, v_BSHD, dO_BSHD, block_mask, scale):
    """Run flex_attention forward + backward (Triton backend).

    Returns (out, dQ, dK, dV) in (B, S, H, D) layout.
    """
    q = q_BSHD.transpose(1, 2).detach().requires_grad_(True)
    k = k_BSHD.transpose(1, 2).detach().requires_grad_(True)
    v = v_BSHD.transpose(1, 2).detach().requires_grad_(True)
    out = _compiled_flex_attention(
        q, k, v,
        block_mask=block_mask,
        scale=scale,
        kernel_options={"BACKEND": "TRITON"},
    )
    out.backward(dO_BSHD.transpose(1, 2))
    return (
        out.transpose(1, 2).detach(),
        q.grad.transpose(1, 2),
        k.grad.transpose(1, 2),
        v.grad.transpose(1, 2),
    )


# ---------------------------------------------------------------------------
# Benchmarking infrastructure
# ---------------------------------------------------------------------------

def benchmark_fn(fn, warmup=5, rep=20):
    """Benchmark a function, returning median time in milliseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rep):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def format_results_table(results):
    """Format benchmark results as an aligned table."""
    headers = [
        "Pattern", "SeqLen",
        "Sparse ms", "Flex ms", "Speedup",
        "Sparse TFLOPS", "Flex TFLOPS",
        "Dense-eq TFLOPS (sparse)", "Dense-eq TFLOPS (flex)",
    ]
    rows = []
    for r in results:
        rows.append([
            r["pattern"],
            str(r["seqlen"]),
            f"{r['t_sparse']:.3f}",
            f"{r['t_flex']:.3f}",
            f"{r['speedup']:.2f}x",
            f"{r['tflops_sparse']:.1f}",
            f"{r['tflops_flex']:.1f}",
            f"{r['dense_eq_tflops_sparse']:.1f}",
            f"{r['dense_eq_tflops_flex']:.1f}",
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def make_row(cells, sep="|"):
        parts = [f" {cells[i]:<{col_widths[i]}} " for i in range(len(cells))]
        return sep + sep.join(parts) + sep

    border_top = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    border_mid = "+" + "+".join("=" * (w + 2) for w in col_widths) + "+"

    lines = [border_top, make_row(headers), border_mid]
    for row in rows:
        lines.append(make_row(row))
    lines.append(border_top)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark_config(
    B, H, D, S, device, dtype, pattern, window_size=None, num_random_tokens=None
):
    """Benchmark one configuration: sparse CuTe DSL vs flex attention."""
    bsq, bsk = 256, 128
    scale = 1.0 / math.sqrt(D)
    H_kv = H  # GQA ratio = 1 for simplicity

    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H_kv, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H_kv, D, device=device, dtype=dtype)
    dO = torch.randn(B, S, H, D, device=device, dtype=dtype)

    # Build block-sparse mask
    random_positions = None
    if pattern == "causal":
        mask_cnt, mask_idx, full_cnt, full_idx = create_causal_block_mask_sparse(
            B, H, S, S, bsq, bsk, device
        )
    elif pattern == "sliding_window":
        mask_cnt, mask_idx, full_cnt, full_idx = create_sliding_window_block_mask_sparse(
            B, H, S, S, bsq, bsk, window_size, device
        )
    elif pattern == "sliding_window_random":
        mask_cnt, mask_idx, full_cnt, full_idx, random_positions = (
            create_sliding_window_plus_random_block_mask_sparse(
                B, H, S, S, bsq, bsk, window_size, num_random_tokens, device
            )
        )
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    bst = BlockSparseTensorsTorch(
        mask_block_cnt=mask_cnt, mask_block_idx=mask_idx,
        full_block_cnt=full_cnt, full_block_idx=full_idx,
        block_size=(bsq, bsk),
    )

    # Count active blocks for FLOPS calculation
    num_active_blocks = count_active_blocks(mask_cnt, full_cnt)
    num_m = ceildiv(S, bsq)
    num_n = ceildiv(S, bsk)
    total_possible_blocks = num_m * num_n
    sparsity_ratio = num_active_blocks / (total_possible_blocks * B * H)

    # Sparse FLOPs (actual compute performed)
    sparse_flops = compute_attention_flops(B, H, S, S, D, num_active_blocks, bsq, bsk)
    # Dense equivalent FLOPs (what dense causal would do)
    dense_causal_flops = compute_dense_causal_flops(B, H, S, S, D)

    # --- Compile and prepare sparse kernels ---
    fwd_compiled, normalized = compile_sparse_fwd(q, k, v, bst, scale, causal=True)

    # Run forward once to get out/lse for backward
    out_sparse, lse_sparse = run_sparse_fwd(q, k, v, fwd_compiled, normalized, scale)
    torch.cuda.synchronize()

    # Transpose for backward
    bwd_tensors = transpose_block_sparse(
        mask_cnt, mask_idx, full_cnt, full_idx, num_m, num_n, B, H, device
    )
    bwd_compiled, normalized_fwd = compile_sparse_bwd(
        q, k, v, dO, out_sparse, lse_sparse, bst, bwd_tensors, scale, causal=True
    )

    # --- Prepare flex attention ---
    if pattern == "causal":
        flex_block_mask = create_flex_causal_block_mask(B, H, S, S, device)
    elif pattern == "sliding_window":
        flex_block_mask = create_flex_sliding_window_block_mask(B, H, S, S, window_size, device)
    elif pattern == "sliding_window_random":
        flex_block_mask = create_flex_sliding_window_random_block_mask(
            B, H, S, S, window_size, random_positions, device
        )

    # Warmup flex
    for _ in range(3):
        run_flex_fwd_bwd(q, k, v, dO, flex_block_mask, scale)
    torch.cuda.synchronize()

    # --- Benchmark sparse (fwd + bwd) ---
    def sparse_step():
        out, lse = run_sparse_fwd(q, k, v, fwd_compiled, normalized, scale)
        run_sparse_bwd(q, k, v, dO, out, lse, bwd_compiled, normalized_fwd, bwd_tensors, scale)

    t_sparse = benchmark_fn(sparse_step)

    # --- Benchmark flex (fwd + bwd) ---
    def flex_step():
        run_flex_fwd_bwd(q, k, v, dO, flex_block_mask, scale)

    t_flex = benchmark_fn(flex_step)

    # --- Compute TFLOPS ---
    # For a training step, both fwd and bwd do attention computation.
    # Backward does roughly 2.5x the FLOPs of forward (dQ, dK, dV all require matmuls).
    # Simplified: total step FLOPs ~ 3.5 * fwd_flops (1x fwd + 2.5x bwd)
    step_flops_sparse = 3.5 * sparse_flops
    step_flops_dense_eq = 3.5 * dense_causal_flops

    tflops_sparse = step_flops_sparse / (t_sparse * 1e-3) / 1e12
    tflops_flex = step_flops_sparse / (t_flex * 1e-3) / 1e12
    dense_eq_tflops_sparse = step_flops_dense_eq / (t_sparse * 1e-3) / 1e12
    dense_eq_tflops_flex = step_flops_dense_eq / (t_flex * 1e-3) / 1e12

    speedup = t_flex / t_sparse

    result = {
        "pattern": pattern,
        "seqlen": S,
        "t_sparse": t_sparse,
        "t_flex": t_flex,
        "speedup": speedup,
        "tflops_sparse": tflops_sparse,
        "tflops_flex": tflops_flex,
        "dense_eq_tflops_sparse": dense_eq_tflops_sparse,
        "dense_eq_tflops_flex": dense_eq_tflops_flex,
        "sparsity_ratio": sparsity_ratio,
        "num_active_blocks": num_active_blocks,
    }

    print(f"    {pattern:25s} S={S:5d}  "
          f"sparse={t_sparse:.2f}ms  flex={t_flex:.2f}ms  "
          f"speedup={speedup:.2f}x  "
          f"sparse_tflops={tflops_sparse:.1f}  flex_tflops={tflops_flex:.1f}  "
          f"sparsity={sparsity_ratio:.2%}")

    del q, k, v, dO, out_sparse, lse_sparse
    torch.cuda.empty_cache()

    return result


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, H, D = 4, 16, 256
    window_size = 512
    num_random_tokens = 128
    seq_lengths = [2048, 4096, 8192, 16384]

    print("=" * 80)
    print("  BLOCK-SPARSE CuTe DSL vs FLEX ATTENTION: TRAINING STEP BENCHMARK")
    print(f"  B={B}, H={H}, D={D}, dtype=bf16")
    print(f"  Measuring: forward + backward per step")
    print(f"  Metrics: wall-clock time (ms), effective TFLOPS, dense-equivalent TFLOPS")
    print("=" * 80)

    all_results = []

    # --- Causal ---
    print("\n  [1] CAUSAL ATTENTION")
    print("  " + "-" * 60)
    for S in seq_lengths:
        result = benchmark_config(B, H, D, S, device, dtype, "causal")
        all_results.append(result)

    # --- Sliding Window ---
    print(f"\n  [2] SLIDING WINDOW (window_size={window_size})")
    print("  " + "-" * 60)
    for S in seq_lengths:
        if window_size >= S:
            continue
        result = benchmark_config(
            B, H, D, S, device, dtype, "sliding_window", window_size=window_size
        )
        all_results.append(result)

    # --- Sliding Window + Random ---
    print(f"\n  [3] SLIDING WINDOW + {num_random_tokens} RANDOM TOKENS")
    print("  " + "-" * 60)
    for S in seq_lengths:
        if window_size >= S:
            continue
        result = benchmark_config(
            B, H, D, S, device, dtype, "sliding_window_random",
            window_size=window_size, num_random_tokens=num_random_tokens,
        )
        all_results.append(result)

    # --- Summary Table ---
    print("\n\n")
    print("=" * 80)
    print("  SUMMARY: TIME PER STEP AND TFLOPS")
    print("=" * 80)
    print()
    print("  Sparse TFLOPS = effective FLOPs (only active blocks) / time")
    print("  Dense-eq TFLOPS = equivalent dense causal FLOPs / time (for MFU comparison)")
    print()
    print(format_results_table(all_results))
    print()

    # --- Per-pattern summary ---
    patterns = ["causal", "sliding_window", "sliding_window_random"]
    for pat in patterns:
        pat_results = [r for r in all_results if r["pattern"] == pat]
        if not pat_results:
            continue
        print(f"\n  {pat.upper()} -- Average speedup: "
              f"{sum(r['speedup'] for r in pat_results) / len(pat_results):.2f}x")
        for r in pat_results:
            print(f"    S={r['seqlen']:5d}: sparse={r['t_sparse']:.2f}ms "
                  f"flex={r['t_flex']:.2f}ms "
                  f"({r['speedup']:.2f}x) "
                  f"sparsity={r['sparsity_ratio']:.1%}")

    print()


if __name__ == "__main__":
    main()
