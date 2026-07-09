"""Block-sparse CuTe DSL attention replacement for DSAFlexAttention.

Wraps the forward and backward kernels from block_sparse_flash_attention/ in a
torch.autograd.Function so they can be used as a drop-in replacement for
DSAFlexAttention in the DeepSeek V4 model.

This module is used for benchmarking the CuTe DSL kernels against flex attention.
"""
import sys
sys.path.insert(0, '/nvdl/torchtitan')

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    normalize_block_sparse_config,
    to_cute_block_sparse_tensors,
)
from flash_attn.cute.utils import AuxData
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
import cutlass.cute as cute

from block_sparse_flash_attention.kernel_fwd import BlackwellFusedMultiHeadAttentionForward
from block_sparse_flash_attention.kernel_bwd import BlackwellFusedMultiHeadAttentionBackward

from torchtitan.models.common.attention import FlexAttention
from torchtitan.protocols.module import Module


# ---------------------------------------------------------------------------
# Block-sparse mask construction for window attention
# ---------------------------------------------------------------------------

def ceildiv(a, b):
    return (a + b - 1) // b


def build_window_block_sparse_mask(B, H, S_q, S_k, bsq, bsk, window_size, device):
    """Build block-sparse mask for causal + sliding window attention.

    Returns (mask_cnt, mask_idx, full_cnt, full_idx) tensors.
    """
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


def build_window_compress_block_sparse_mask(
    B, H, S_q, kv_len, bsq, bsk, window_size, compress_ratio, device
):
    """Build block-sparse mask for window + compressed causal tokens.

    The KV sequence is [original_tokens (S_q) | compressed_tokens (S_q // compress_ratio)].
    - Window part: causal + sliding window over original tokens
    - Compress part: causal access to compressed tokens

    Returns (mask_cnt, mask_idx, full_cnt, full_idx) tensors.
    """
    compress_len = S_q // compress_ratio
    num_m = ceildiv(S_q, bsq)
    num_n = ceildiv(kv_len, bsk)
    mask_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    mask_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)
    full_cnt = torch.zeros(B, H, num_m, dtype=torch.int32, device=device)
    full_idx = torch.zeros(B, H, num_m, num_n, dtype=torch.int32, device=device)

    for m in range(num_m):
        qs, qe = m * bsq, min((m + 1) * bsq, S_q)
        fc, mc = 0, 0
        for n in range(num_n):
            ks, ke = n * bsk, min((n + 1) * bsk, kv_len)

            # Determine if this KV block is in the original or compressed region
            if ke <= S_q:
                # Original token region: causal + window
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
            elif ks >= S_q:
                # Compressed token region: causal access
                # compress_idx range for this block: [ks - S_q, ke - S_q)
                compress_start = ks - S_q
                # A query at position q can see compressed tokens with
                # index < (q + 1) // compress_ratio
                # For the last query in this block (qe - 1):
                max_compress_visible = (qe - 1 + 1) // compress_ratio
                if compress_start >= max_compress_visible:
                    continue
                # Check if full block: all queries can see all compressed keys
                min_compress_visible = (qs + 1) // compress_ratio
                compress_end = min(ke - S_q, compress_len)
                if min_compress_visible >= compress_end:
                    full_idx[:, :, m, fc] = n
                    fc += 1
                else:
                    mask_idx[:, :, m, mc] = n
                    mc += 1
            else:
                # Block straddles boundary -- treat as mask block
                mask_idx[:, :, m, mc] = n
                mc += 1

        mask_cnt[:, :, m] = mc
        full_cnt[:, :, m] = fc
    return mask_cnt, mask_idx, full_cnt, full_idx


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
            for m_idx in range(num_m):
                for i in range(mc[b, h, m_idx]):
                    n = mi[b, h, m_idx, i]
                    bmi[b, h, n, bmc[b, h, n]] = m_idx
                    bmc[b, h, n] += 1
                for i in range(fc[b, h, m_idx]):
                    n = fi[b, h, m_idx, i]
                    bfi[b, h, n, bfc[b, h, n]] = m_idx
                    bfc[b, h, n] += 1

    return (
        torch.from_numpy(bmc).to(device=device, dtype=torch.int32),
        torch.from_numpy(bmi).to(device=device, dtype=torch.int32),
        torch.from_numpy(bfc).to(device=device, dtype=torch.int32),
        torch.from_numpy(bfi).to(device=device, dtype=torch.int32),
    )


# ---------------------------------------------------------------------------
# Kernel compilation cache
# ---------------------------------------------------------------------------

_fwd_cache = {}
_bwd_cache = {}


def _get_compiled_fwd(B, S_q, S_k, H, H_kv, D, dtype, causal, bst):
    """Get or compile the forward kernel."""
    cache_key = (B, S_q, S_k, H, H_kv, D, dtype, causal)
    if cache_key in _fwd_cache:
        return _fwd_cache[cache_key]

    device = bst.mask_block_cnt.device
    tile_m, tile_n, q_stage = 128, 128, 2
    normalized, _, _ = normalize_block_sparse_config(
        bst, batch_size=B, num_head=H, seqlen_q=S_q, seqlen_k=S_k,
        block_size=(tile_m, tile_n), q_stage=q_stage,
    )

    q_dummy = torch.zeros(B, S_q, H, D, device=device, dtype=dtype)
    k_dummy = torch.zeros(B, S_k, H_kv, D, device=device, dtype=dtype)
    v_dummy = torch.zeros(B, S_k, H_kv, D, device=device, dtype=dtype)
    out_dummy = torch.zeros_like(q_dummy)
    lse_dummy = torch.zeros(B, H, S_q, device=device, dtype=torch.float32)

    kernel_obj = BlackwellFusedMultiHeadAttentionForward(
        head_dim=D, qhead_per_kvhead=H // H_kv, is_causal=causal,
    )
    sparse_cute = to_cute_block_sparse_tensors(normalized)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        to_cute_tensor(q_dummy), to_cute_tensor(k_dummy), to_cute_tensor(v_dummy),
        to_cute_tensor(out_dummy), to_cute_tensor(lse_dummy, assumed_align=4),
        1.0,  # scale placeholder
        None, None, None, None, None, None, None, None, None,
        sparse_cute, AuxData(), stream, options="--enable-tvm-ffi",
    )
    _fwd_cache[cache_key] = (compiled, normalized)
    return compiled, normalized


def _get_compiled_bwd(B, S_q, S_k, H, H_kv, D, dtype, causal, bst, bwd_tensors):
    """Get or compile the backward kernel."""
    cache_key = (B, S_q, S_k, H, H_kv, D, dtype, causal)
    if cache_key in _bwd_cache:
        return _bwd_cache[cache_key]

    device = bst.mask_block_cnt.device
    tile_m, tile_n, q_stage = 128, 128, 2
    normalized_fwd, _, _ = normalize_block_sparse_config(
        bst, batch_size=B, num_head=H, seqlen_q=S_q, seqlen_k=S_k,
        block_size=(tile_m, tile_n), q_stage=q_stage,
    )
    bwd_mask_cnt, bwd_mask_idx, bwd_full_cnt, bwd_full_idx = bwd_tensors

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

    q_dummy = torch.zeros(B, S_q, H, D, device=device, dtype=dtype)
    k_dummy = torch.zeros(B, S_k, H_kv, D, device=device, dtype=dtype)
    v_dummy = torch.zeros(B, S_k, H_kv, D, device=device, dtype=dtype)
    dO_dummy = torch.zeros_like(q_dummy)
    lse_log2_dummy = torch.zeros(B, H, S_q, device=device, dtype=torch.float32)
    dpsum_dummy = torch.zeros(B, H, S_q, device=device, dtype=torch.float32)
    dQ_dummy = torch.zeros_like(q_dummy)
    dK_dummy = torch.zeros_like(k_dummy)
    dV_dummy = torch.zeros_like(v_dummy)

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        to_cute_tensor(q_dummy), to_cute_tensor(k_dummy), to_cute_tensor(v_dummy),
        to_cute_tensor(dO_dummy),
        to_cute_tensor(lse_log2_dummy, assumed_align=4),
        to_cute_tensor(dpsum_dummy, assumed_align=4),
        to_cute_tensor(dQ_dummy), to_cute_tensor(dK_dummy), to_cute_tensor(dV_dummy),
        1.0,  # scale placeholder
        None, None, None, None, None, None, None, None, None,
        AuxData(), sparse_cute_fwd, sparse_cute_bwd,
        stream, options="--enable-tvm-ffi",
    )
    _bwd_cache[cache_key] = (compiled, normalized_fwd)
    return compiled, normalized_fwd


# ---------------------------------------------------------------------------
# Autograd function wrapping fwd + bwd kernels
# ---------------------------------------------------------------------------

class BlockSparseAttentionFn(torch.autograd.Function):
    """Autograd function for block-sparse CuTe DSL attention."""

    @staticmethod
    def forward(ctx, q, k, v, scale, bst, bwd_tensors, causal):
        B, S_q, H, D = q.shape
        _, S_k, H_kv, _ = k.shape
        device = q.device

        compiled_fwd, normalized = _get_compiled_fwd(
            B, S_q, S_k, H, H_kv, D, q.dtype, causal, bst
        )

        out = torch.zeros_like(q)
        lse = torch.zeros(B, H, S_q, device=device, dtype=torch.float32)

        compiled_fwd(
            q, k, v, out, lse, scale,
            None, None, None, None, None, None, None, None, None,
            (normalized.mask_block_cnt, normalized.mask_block_idx,
             normalized.full_block_cnt, normalized.full_block_idx,
             normalized.cu_total_m_blocks, normalized.cu_block_idx_offsets,
             normalized.dq_write_order, normalized.dq_write_order_full),
            AuxData(),
        )

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.bst = bst
        ctx.bwd_tensors = bwd_tensors
        ctx.causal = causal
        return out, lse

    @staticmethod
    def backward(ctx, grad_out, grad_lse):
        q, k, v, out, lse = ctx.saved_tensors
        scale = ctx.scale
        bst = ctx.bst
        bwd_tensors = ctx.bwd_tensors
        causal = ctx.causal

        B, S_q, H, D = q.shape
        _, S_k, H_kv, _ = k.shape
        device = q.device

        compiled_bwd, normalized_fwd = _get_compiled_bwd(
            B, S_q, S_k, H, H_kv, D, q.dtype, causal, bst, bwd_tensors
        )

        log2_e = math.log2(math.e)
        lse_log2 = lse * log2_e
        dpsum = (grad_out.float() * out.float()).sum(dim=-1).transpose(1, 2).contiguous()
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

        compiled_bwd(
            q, k, v, grad_out,
            lse_log2, dpsum,
            dQ, dK, dV,
            scale,
            None, None, None, None, None, None, None, None, None,
            AuxData(), sparse_fwd_runtime, sparse_bwd_runtime,
        )
        return dQ, dK, dV, None, None, None, None


# ---------------------------------------------------------------------------
# DSABlockSparseAttention: drop-in replacement for DSAFlexAttention
# ---------------------------------------------------------------------------

class DSABlockSparseAttention(Module):
    """Block-sparse CuTe DSL attention replacement for DSAFlexAttention.

    Uses the custom Blackwell kernels for forward and backward instead of
    flex_attention's Triton backend.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        window_size: int
        compress_ratio: int
        softmax_scale: float
        block_size: int | tuple[int, int] = (256, 128)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.softmax_scale
        if isinstance(config.block_size, tuple):
            self.bsq, self.bsk = config.block_size
        else:
            self.bsq = self.bsk = config.block_size
        # Cache for block-sparse masks (keyed by shape)
        self._mask_cache = {}

    def _get_block_sparse_mask(self, B, H, S_q, kv_len, device):
        """Get or build the block-sparse mask for the given shape."""
        cache_key = (B, H, S_q, kv_len)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        if self.compress_ratio <= 1:
            mask_cnt, mask_idx, full_cnt, full_idx = build_window_block_sparse_mask(
                B, H, S_q, kv_len, self.bsq, self.bsk, self.window_size, device
            )
        else:
            mask_cnt, mask_idx, full_cnt, full_idx = build_window_compress_block_sparse_mask(
                B, H, S_q, kv_len, self.bsq, self.bsk,
                self.window_size, self.compress_ratio, device
            )

        bst = BlockSparseTensorsTorch(
            mask_block_cnt=mask_cnt, mask_block_idx=mask_idx,
            full_block_cnt=full_cnt, full_block_idx=full_idx,
            block_size=(self.bsq, self.bsk),
        )

        num_m = ceildiv(S_q, self.bsq)
        num_n = ceildiv(kv_len, self.bsk)
        bwd_tensors = transpose_block_sparse(
            mask_cnt, mask_idx, full_cnt, full_idx, num_m, num_n, B, H, device
        )

        self._mask_cache[cache_key] = (bst, bwd_tensors)
        return bst, bwd_tensors

    def forward(
        self,
        query_states,
        kv_states,
        attn_sink,
        kv_compress,
        compress_topk_idxs,
    ):
        bsz, seqlen, n_heads, head_dim = query_states.size()

        if self.compress_ratio > 1 and kv_compress is not None:
            kv_states = torch.cat([kv_states, kv_compress], dim=1)
        kv_len = kv_states.size(1)

        # Expand kv from (B, S_k, 1, D) or (B, S_k, D) to (B, S_k, H_kv, D)
        if kv_states.dim() == 3:
            kv_BSHD = kv_states.unsqueeze(2)
        else:
            kv_BSHD = kv_states
        H_kv = kv_BSHD.size(2)

        # Get block-sparse mask
        bst, bwd_tensors = self._get_block_sparse_mask(
            bsz, n_heads, seqlen, kv_len, query_states.device
        )

        # Run block-sparse attention
        out, lse = BlockSparseAttentionFn.apply(
            query_states, kv_BSHD, kv_BSHD,
            self.softmax_scale, bst, bwd_tensors, True,
        )

        # Apply attention sink correction (same as DSAFlexAttention)
        sink_logit = attn_sink  # (n_heads,)
        # lse shape: (B, H, S_q) -- convert to (B, S, H) for broadcasting
        lse_BLN = lse.transpose(1, 2)  # (B, S_q, H)
        correction = torch.sigmoid(
            lse_BLN - sink_logit.float()[None, None, :]
        )
        out = out * correction.to(out.dtype).unsqueeze(-1)

        return out
