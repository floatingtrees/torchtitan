"""Triton fused sparse attention kernel for DeepSeek V4.

Fuses gather + matmul + softmax + weighted-sum into a single kernel.
Uses vectorized loads and reduction within each thread block.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attn_fwd_kernel(
    Q_ptr, KV_ptr, IDX_ptr, SINK_ptr, OUT_ptr, LSE_ptr,
    stride_qb, stride_qs, stride_qh,
    stride_kvb, stride_kvs,
    stride_ib, stride_is,
    stride_ob, stride_os, stride_oh,
    stride_lb, stride_lh,
    SEQLEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    W: tl.constexpr,
    KV_LEN: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Each program handles one (batch, seq_pos, head) tuple.

    Loads Q once, then iterates over W gathered KV tokens computing
    online softmax. All arithmetic in float32 for numerical stability.
    """
    pid = tl.program_id(0)
    num_per_batch = SEQLEN * N_HEADS
    batch_id = pid // num_per_batch
    remainder = pid % num_per_batch
    seq_id = remainder // N_HEADS
    head_id = remainder % N_HEADS

    # Pointers
    q_base = Q_ptr + batch_id * stride_qb + seq_id * stride_qs + head_id * stride_qh
    idx_base = IDX_ptr + batch_id * stride_ib + seq_id * stride_is
    kv_base = KV_ptr + batch_id * stride_kvb

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < HEAD_DIM

    # Load Q: (BLOCK_D,) in float32
    q = tl.load(q_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    sink_val = tl.load(SINK_ptr + head_id).to(tl.float32)

    # Initialize online softmax with sink (avoids -inf - (-inf) = NaN)
    m_prev = sink_val
    l_prev = 1.0  # exp(sink - sink) = 1
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Process W KV tokens with online softmax
    for w in tl.static_range(W):
        idx = tl.load(idx_base + w)
        is_valid = idx < KV_LEN

        kv = tl.load(kv_base + idx * stride_kvs + d_range, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * kv, axis=0) * SM_SCALE
        score = tl.where(is_valid, score, float("-inf"))

        m_new = tl.maximum(m_prev, score)
        alpha = tl.exp(m_prev - m_new)
        l_prev = l_prev * alpha
        acc = acc * alpha
        p = tl.exp(score - m_new)
        l_prev = l_prev + p
        acc = acc + p * kv
        m_prev = m_new

    # Normalize and store
    out = acc / l_prev
    out_base = OUT_ptr + batch_id * stride_ob + seq_id * stride_os + head_id * stride_oh
    tl.store(out_base + d_range, out.to(Q_ptr.dtype.element_ty), mask=d_mask)

    # Store LSE
    lse = m_prev + tl.log(l_prev)
    tl.store(LSE_ptr + batch_id * stride_lb + head_id * stride_lh + seq_id, lse)


@triton.jit
def _sparse_attn_bwd_kernel(
    Q_ptr, KV_ptr, IDX_ptr, SINK_ptr, DO_ptr, LSE_ptr,
    DQ_ptr, DKV_ptr, DSINK_ptr,
    stride_qb, stride_qs, stride_qh,
    stride_kvb, stride_kvs,
    stride_ib, stride_is,
    stride_lb, stride_lh,
    SEQLEN: tl.constexpr,
    N_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    W: tl.constexpr,
    KV_LEN: tl.constexpr,
    SM_SCALE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_per_batch = SEQLEN * N_HEADS
    batch_id = pid // num_per_batch
    remainder = pid % num_per_batch
    seq_id = remainder // N_HEADS
    head_id = remainder % N_HEADS

    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < HEAD_DIM

    q_base = Q_ptr + batch_id * stride_qb + seq_id * stride_qs + head_id * stride_qh
    do_base = DO_ptr + batch_id * stride_qb + seq_id * stride_qs + head_id * stride_qh
    idx_base = IDX_ptr + batch_id * stride_ib + seq_id * stride_is
    kv_base = KV_ptr + batch_id * stride_kvb

    q = tl.load(q_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    do = tl.load(do_base + d_range, mask=d_mask, other=0.0).to(tl.float32)
    lse = tl.load(LSE_ptr + batch_id * stride_lb + head_id * stride_lh + seq_id)
    sink_val = tl.load(SINK_ptr + head_id).to(tl.float32)

    # Pass 1: compute D = sum_i p_i * (dO . v_i)
    D_val = 0.0
    for w in tl.static_range(W):
        idx = tl.load(idx_base + w)
        is_valid = idx < KV_LEN
        kv = tl.load(kv_base + idx * stride_kvs + d_range, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * kv, axis=0) * SM_SCALE
        score = tl.where(is_valid, score, float("-inf"))
        p = tl.exp(score - lse)
        D_val += p * tl.sum(do * kv, axis=0)

    # Pass 2: compute dQ, scatter dKV
    dq_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for w in tl.static_range(W):
        idx = tl.load(idx_base + w)
        is_valid = idx < KV_LEN
        kv = tl.load(kv_base + idx * stride_kvs + d_range, mask=d_mask, other=0.0).to(tl.float32)
        score = tl.sum(q * kv, axis=0) * SM_SCALE
        score = tl.where(is_valid, score, float("-inf"))
        p = tl.exp(score - lse)
        dov = tl.sum(do * kv, axis=0)
        ds = p * (dov - D_val)
        dq_acc += ds * SM_SCALE * kv
        dkv_val = (ds * SM_SCALE * q + p * do)
        dkv_ptr = DKV_ptr + batch_id * stride_kvb + idx * stride_kvs + d_range
        tl.atomic_add(dkv_ptr, dkv_val.to(DKV_ptr.dtype.element_ty), mask=d_mask & is_valid)

    # Store dQ
    dq_base = DQ_ptr + batch_id * stride_qb + seq_id * stride_qs + head_id * stride_qh
    tl.store(dq_base + d_range, dq_acc.to(DQ_ptr.dtype.element_ty), mask=d_mask)

    # dSink
    p_sink = tl.exp(sink_val - lse)
    dsink_val = -p_sink * D_val
    tl.atomic_add(DSINK_ptr + head_id, dsink_val)


class SparseAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv_padded, indices, attn_sink, sm_scale, kv_len, W):
        B, S, H, D = q.shape
        out = torch.empty_like(q)
        lse = torch.empty((B, H, S), device=q.device, dtype=torch.float32)
        BLOCK_D = triton.next_power_of_2(D)

        grid = (B * S * H,)
        _sparse_attn_fwd_kernel[grid](
            q, kv_padded, indices, attn_sink, out, lse,
            q.stride(0), q.stride(1), q.stride(2),
            kv_padded.stride(0), kv_padded.stride(1),
            indices.stride(0), indices.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            SEQLEN=S, N_HEADS=H, HEAD_DIM=D, W=W,
            KV_LEN=kv_len, SM_SCALE=sm_scale,
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(q, kv_padded, indices, attn_sink, lse)
        ctx.sm_scale = sm_scale
        ctx.kv_len = kv_len
        ctx.W = W
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv_padded, indices, attn_sink, lse = ctx.saved_tensors
        B, S, H, D = q.shape
        BLOCK_D = triton.next_power_of_2(D)

        dq = torch.zeros_like(q)
        dkv = torch.zeros_like(kv_padded)
        dsink = torch.zeros(H, device=q.device, dtype=torch.float32)

        grid = (B * S * H,)
        _sparse_attn_bwd_kernel[grid](
            q, kv_padded, indices, attn_sink, do, lse,
            dq, dkv, dsink,
            q.stride(0), q.stride(1), q.stride(2),
            kv_padded.stride(0), kv_padded.stride(1),
            indices.stride(0), indices.stride(1),
            lse.stride(0), lse.stride(1),
            SEQLEN=S, N_HEADS=H, HEAD_DIM=D, W=ctx.W,
            KV_LEN=ctx.kv_len, SM_SCALE=ctx.sm_scale,
            BLOCK_D=BLOCK_D,
        )

        return dq, dkv, None, dsink.to(attn_sink.dtype), None, None, None


def triton_sparse_attention(q, kv_padded, indices, attn_sink, sm_scale, kv_len, W):
    """Fused sparse attention: gather + matmul + softmax + weighted sum."""
    return SparseAttnFunc.apply(q, kv_padded, indices, attn_sink, sm_scale, kv_len, W)
