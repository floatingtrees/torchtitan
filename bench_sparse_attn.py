"""Benchmark three sparse attention variants from attention.py across branches.

Variants:
1. fixed_score_mod  -- FlexAttention with block mask + sink out_transform
2. gather_attention (5d7aaee8) -- gather KV + einsum + softmax with appended sink
3. c2d8a7ab        -- full matmul + scatter mask + appended sink

Benchmarked with torch.compile, forward only, at compress_ratio=1, 4, and 128.
"""

import time

import torch
import torch.nn.functional as F

# ============================================================================
# Config
# ============================================================================
BATCH = 8
SEQ_LEN = 2048
HEAD_DIM = 256
N_HEADS = 16
WINDOW_SIZE = 128
INDEX_TOPK = 128
SCALE = 0.0625
DTYPE = torch.float32
WARMUP = 5
ITERS = 20
DEVICE = "cuda"

# ============================================================================
# Variant 1: fixed_score_mod (FlexAttention with block mask)
# ============================================================================
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)


def sparse_attn_flex(query_states, kv_states, attn_sink, kv_compress,
                     compress_topk_idxs, compress_ratio):
    """FlexAttention variant from fixed_score_mod branch."""
    bsz, seqlen, n_heads, head_dim = query_states.size()

    if compress_ratio > 1:
        kv_states = torch.cat([kv_states, kv_compress], dim=1)
    kv_len = kv_states.size(1)

    key_value_states = kv_states.unsqueeze(2)

    if compress_ratio == 4:
        # Indexer block causal mask
        compress_len = seqlen // compress_ratio
        topk = compress_topk_idxs.size(-1)

        def v4_sparse_mask_mod(b, h, q_idx, kv_idx):
            in_window = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < WINDOW_SIZE)
            )
            selected_compress = compress_topk_idxs[b, q_idx, 0] == kv_idx
            for idx in range(1, topk):
                selected_compress = selected_compress | (
                    compress_topk_idxs[b, q_idx, idx] == kv_idx
                )
            in_compressed_topk = (
                (kv_idx >= seqlen)
                & (kv_idx < seqlen + compress_len)
                & selected_compress
            )
            return in_window | in_compressed_topk

        block_mask = create_block_mask(
            v4_sparse_mask_mod,
            B=bsz, H=None, Q_LEN=seqlen, KV_LEN=kv_len,
            device=query_states.device, BLOCK_SIZE=128,
        )
    elif compress_ratio > 1:
        # Compress causal mask
        compress_len = seqlen // compress_ratio

        def compress_causal_mask_mod(b, h, q_idx, kv_idx):
            in_window = (
                (kv_idx < seqlen)
                & (kv_idx <= q_idx)
                & (q_idx - kv_idx < WINDOW_SIZE)
            )
            compress_idx = kv_idx - seqlen
            in_compressed_causal = (
                (kv_idx >= seqlen)
                & (kv_idx < seqlen + compress_len)
                & (compress_idx < (q_idx + 1) // compress_ratio)
            )
            return in_window | in_compressed_causal

        block_mask = create_block_mask(
            compress_causal_mask_mod,
            B=bsz, H=None, Q_LEN=seqlen, KV_LEN=kv_len,
            device=query_states.device, BLOCK_SIZE=128,
        )
    else:
        def window_mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx - kv_idx < WINDOW_SIZE)

        block_mask = create_block_mask(
            window_mask_mod,
            B=bsz, H=None, Q_LEN=seqlen, KV_LEN=kv_len,
            device=query_states.device, BLOCK_SIZE=128,
        )

    sink_logit = attn_sink

    q = query_states.transpose(1, 2)
    k = key_value_states.transpose(1, 2)
    v = key_value_states.transpose(1, 2)

    out, lse = flex_attention(
        q, k, v,
        block_mask=block_mask,
        scale=SCALE,
        enable_gqa=True,
        return_lse=True,
    )
    lse_BLN = lse.permute(0, 2, 1)
    out_BLNH = out.permute(0, 2, 1, 3)
    correction = torch.sigmoid(lse_BLN - sink_logit.float()[None, None, :])
    out_BLNH = out_BLNH * correction.to(out_BLNH.dtype).unsqueeze(-1)
    return out_BLNH




def _build_flex_equivalent_topk_indices(
    bsz,
    seqlen,
    compress_ratio,
    compress_topk_idxs,
    device,
):
    """
    Build explicit sparse indices with the same semantics as sparse_attn_flex.

    Flex semantics:
      cr == 1:
        local causal window only

      cr == 4:
        local causal window + selected compressed indices from compress_topk_idxs

      cr > 1 and cr != 4:
        local causal window + all causal compressed indices
    """
    base = torch.arange(seqlen, device=device).unsqueeze(1)

    # ----------------------------------------------------------------------
    # 1. Local causal sliding window
    # ----------------------------------------------------------------------
    window_offsets = torch.arange(
        min(seqlen, WINDOW_SIZE),
        device=device,
    )

    window_topk = (base - WINDOW_SIZE + 1).clamp(min=0) + window_offsets
    window_topk = torch.where(window_topk <= base, window_topk, -1)

    topk_idxs = window_topk.unsqueeze(0).expand(bsz, -1, -1)

    # ----------------------------------------------------------------------
    # 2. Compressed side of the mask
    # ----------------------------------------------------------------------
    if compress_ratio == 4:
        # Match v4_sparse_mask_mod:
        #
        #   selected_compress = any(compress_topk_idxs[b, q, :] == kv_idx)
        #
        # Flex treats this as a set membership test. Duplicates do not matter.
        compress_len = seqlen // compress_ratio

        comp = compress_topk_idxs.to(device)

        # Flex also requires kv_idx to actually be inside the compressed range:
        #   seqlen <= kv_idx < seqlen + compress_len
        valid_comp = (comp >= seqlen) & (comp < seqlen + compress_len)
        comp = torch.where(valid_comp, comp, -1)

        topk_idxs = torch.cat([topk_idxs, comp], dim=-1)

    elif compress_ratio > 1:
        # Match compress_causal_mask_mod:
        #
        #   compress_idx = kv_idx - seqlen
        #   compress_idx < (q_idx + 1) // compress_ratio
        #
        # This branch does NOT use compress_topk_idxs in FlexAttention.
        compress_len = seqlen // compress_ratio

        comp_idx = torch.arange(compress_len, device=device).unsqueeze(0)
        causal_limit = (base + 1) // compress_ratio

        comp = seqlen + comp_idx
        comp = torch.where(comp_idx < causal_limit, comp, -1)

        comp = comp.unsqueeze(0).expand(bsz, -1, -1)
        topk_idxs = torch.cat([topk_idxs, comp], dim=-1)

    return topk_idxs


def _apply_flex_sink_correction(attn_output_bhsd, masked_logits, attn_sink):
    """
    Match sparse_attn_flex sink behavior.

    FlexAttention first computes attention over the real, mask-allowed KV tokens.
    Then it applies:

        correction = sigmoid(lse - sink_logit)

    where lse = logsumexp(real_allowed_logits).

    This is equivalent to adding a sink logit with zero value, but this version
    mirrors the FlexAttention code path more directly.
    """
    lse = torch.logsumexp(masked_logits.float(), dim=-1)  # [B, H, S]

    correction = torch.sigmoid(
        lse - attn_sink.float()[None, :, None]
    )

    return attn_output_bhsd * correction.to(attn_output_bhsd.dtype).unsqueeze(-1)

# ============================================================================
# Variant 2: gather_attention (5d7aaee8) -- gather + einsum
# ============================================================================

def fmt_bytes(n):
    return f"{n / 1024 ** 2:8.1f} MiB"


def sparse_attn_gather(
    query_states,
    kv_states,
    attn_sink,
    kv_compress,
    compress_topk_idxs,
    compress_ratio,
):
    """
    Gather-based sparse attention with the same mask semantics as sparse_attn_flex.

    Still computes sparse attention by:
      1. gathering selected KV rows
      2. torch.einsum("bhsd,bswd->bhsw", ...)
      3. softmax
      4. torch.einsum("bhsw,bswd->bhsd", ...)
    """
    bsz, seqlen, n_heads, head_dim = query_states.size()
    device = query_states.device

    # ----------------------------------------------------------------------
    # 1. Concatenate normal KV with compressed KV, same as FlexAttention.
    # ----------------------------------------------------------------------
    if compress_ratio > 1 and kv_compress.size(1) > 0:
        kv_states = torch.cat([kv_states, kv_compress], dim=1)

    kv_len = kv_states.size(1)

    # ----------------------------------------------------------------------
    # 2. Build explicit sparse indices matching the FlexAttention block mask.
    # ----------------------------------------------------------------------
    topk_idxs = _build_flex_equivalent_topk_indices(
        bsz=bsz,
        seqlen=seqlen,
        compress_ratio=compress_ratio,
        compress_topk_idxs=compress_topk_idxs,
        device=device,
    )

    # ----------------------------------------------------------------------
    # 3. Convert invalid -1 indices to a padded sentinel row.
    # ----------------------------------------------------------------------
    sentinel = kv_len

    topk_idxs = topk_idxs.to(device)
    topk_idxs = topk_idxs.masked_fill(topk_idxs < 0, sentinel).long()

    # ----------------------------------------------------------------------
    # 4. De-duplicate selected indices.
    #
    # FlexAttention's mask checks membership:
    #
    #     selected = any(compress_topk_idxs[b, q, :] == kv_idx)
    #
    # so duplicates must not get multiple softmax entries.
    #
    # Sorting makes duplicates adjacent. Order does not matter for attention
    # over a set, because softmax + weighted sum are permutation-invariant.
    # ----------------------------------------------------------------------
    topk_idxs, _ = torch.sort(topk_idxs, dim=-1)

    duplicate_mask = torch.zeros_like(topk_idxs, dtype=torch.bool)
    duplicate_mask[..., 1:] = topk_idxs[..., 1:] == topk_idxs[..., :-1]

    invalid_mask = (topk_idxs == sentinel) | duplicate_mask

    W = topk_idxs.size(-1)

    # ----------------------------------------------------------------------
    # 5. Pad KV with one dummy row used for invalid/sentinel indices.
    # ----------------------------------------------------------------------
    kv_padded = F.pad(kv_states, (0, 0, 0, 1))  # [B, L + 1, D]

    # ----------------------------------------------------------------------
    # 6. Gather selected KV rows.
    # ----------------------------------------------------------------------
    idx_flat = topk_idxs.reshape(bsz, -1)       # [B, S * W]

    idx_expanded = idx_flat.unsqueeze(-1).expand(
        -1,
        -1,
        head_dim,
    )                                          # [B, S * W, D]

    kv_gathered = torch.gather(
        kv_padded,
        dim=1,
        index=idx_expanded,
    )                                          # [B, S * W, D]

    kv_gathered = kv_gathered.reshape(
        bsz,
        seqlen,
        W,
        head_dim,
    )                                          # [B, S, W, D]

    # ----------------------------------------------------------------------
    # 7. Sparse logits via einsum path preserved.
    # ----------------------------------------------------------------------
    q = query_states.transpose(1, 2)            # [B, H, S, D]

    sparse_logits = torch.einsum(
        "bhsd,bswd->bhsw",
        q,
        kv_gathered,
    ) * SCALE                                  # [B, H, S, W]

    mask_value = torch.finfo(sparse_logits.dtype).min

    masked_logits = sparse_logits.masked_fill(
        invalid_mask.unsqueeze(1),
        mask_value,
    )

    # ----------------------------------------------------------------------
    # 8. Softmax over real gathered KV positions only.
    # ----------------------------------------------------------------------
    attn_probs = F.softmax(
        masked_logits.float(),
        dim=-1,
    ).to(masked_logits.dtype)

    # ----------------------------------------------------------------------
    # 9. Sparse value aggregation via einsum path preserved.
    # ----------------------------------------------------------------------
    attn_output = torch.einsum(
        "bhsw,bswd->bhsd",
        attn_probs,
        kv_gathered,
    )                                          # [B, H, S, D]

    # ----------------------------------------------------------------------
    # 10. Apply sink exactly like sparse_attn_flex.
    # ----------------------------------------------------------------------
    attn_output = _apply_flex_sink_correction(
        attn_output_bhsd=attn_output,
        masked_logits=masked_logits,
        attn_sink=attn_sink,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


# ============================================================================
# Variant 3: c2d8a7ab -- full matmul + scatter mask
# ============================================================================


def sparse_attn_scatter(
    query_states,
    kv_states,
    attn_sink,
    kv_compress,
    compress_topk_idxs,
    compress_ratio,
):
    """
    Scatter-mask variant with the same mask semantics as sparse_attn_flex.

    Still computes attention logits with full matmul:
        q @ kv.T

    Then applies a dense additive mask created with scatter_.
    """
    bsz, seqlen, n_heads, head_dim = query_states.size()
    device = query_states.device

    # ----------------------------------------------------------------------
    # 1. Concatenate normal KV with compressed KV, same as FlexAttention.
    # ----------------------------------------------------------------------
    if compress_ratio > 1 and kv_compress.size(1) > 0:
        kv_states = torch.cat([kv_states, kv_compress], dim=1)

    kv_len = kv_states.size(1)

    # ----------------------------------------------------------------------
    # 2. Build explicit sparse indices matching the FlexAttention block mask.
    # ----------------------------------------------------------------------
    topk_idxs = _build_flex_equivalent_topk_indices(
        bsz=bsz,
        seqlen=seqlen,
        compress_ratio=compress_ratio,
        compress_topk_idxs=compress_topk_idxs,
        device=device,
    )

    # ----------------------------------------------------------------------
    # 3. Dense attention logits: matmul path preserved.
    # ----------------------------------------------------------------------
    q = query_states.transpose(1, 2)          # [B, H, S, D]
    kv_expanded = kv_states.unsqueeze(1)      # [B, 1, L, D]

    attn_logits = torch.matmul(
        q,
        kv_expanded.transpose(2, 3),
    ) * SCALE                                # [B, H, S, L]

    # ----------------------------------------------------------------------
    # 4. Build dense additive mask using scatter_.
    # ----------------------------------------------------------------------
    sentinel = kv_len

    topk_idxs = topk_idxs.to(device)
    topk_idxs_masked = topk_idxs.masked_fill(topk_idxs < 0, sentinel)

    mask_value = torch.finfo(attn_logits.dtype).min

    index_mask = torch.full(
        (bsz, 1, seqlen, kv_len + 1),
        fill_value=mask_value,
        dtype=attn_logits.dtype,
        device=device,
    )

    index_mask.scatter_(
        dim=-1,
        index=topk_idxs_masked.unsqueeze(1).long(),
        value=0,
    )

    # Drop dummy sentinel column.
    index_mask = index_mask[..., :-1]         # [B, 1, S, L]

    masked_logits = attn_logits + index_mask  # [B, H, S, L]

    # ----------------------------------------------------------------------
    # 5. Softmax over real KV positions only.
    # ----------------------------------------------------------------------
    attn_probs = F.softmax(
        masked_logits.float(),
        dim=-1,
    ).to(masked_logits.dtype)

    # ----------------------------------------------------------------------
    # 6. Value matmul path preserved.
    # ----------------------------------------------------------------------
    attn_output = torch.matmul(
        attn_probs,
        kv_expanded,
    )                                        # [B, H, S, D]

    # ----------------------------------------------------------------------
    # 7. Apply sink exactly like sparse_attn_flex.
    # ----------------------------------------------------------------------
    attn_output = _apply_flex_sink_correction(
        attn_output_bhsd=attn_output,
        masked_logits=masked_logits,
        attn_sink=attn_sink,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output

# ============================================================================
# Benchmark harness
# ============================================================================


def make_inputs(compress_ratio):
    q = torch.randn(BATCH, SEQ_LEN, N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    kv = torch.randn(BATCH, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    sink = torch.randn(N_HEADS, dtype=DTYPE, device=DEVICE)

    if compress_ratio > 1:
        compress_len = SEQ_LEN // compress_ratio
        kv_compress = torch.randn(BATCH, compress_len, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        # Build compress_topk_idxs: indices into the compressed KV (offset by SEQ_LEN)
        # Simulate the indexer output: for each query position, pick INDEX_TOPK
        # compressed positions (capped by causal limit)
        offset = SEQ_LEN
        compress_topk_idxs = torch.randint(
            0, compress_len, (BATCH, SEQ_LEN, INDEX_TOPK),
            device=DEVICE, dtype=torch.int64,
        ) + offset
        # Apply causal masking: compressed pos must be < (q_idx+1) // compress_ratio
        base = torch.arange(SEQ_LEN, device=DEVICE).unsqueeze(1)
        causal_limit = (base + 1) // compress_ratio  # (S, 1)
        raw_indices = compress_topk_idxs - offset  # (B, S, INDEX_TOPK)
        invalid = raw_indices >= causal_limit.unsqueeze(0)
        compress_topk_idxs = compress_topk_idxs.where(~invalid, torch.tensor(-1, device=DEVICE))
    else:
        kv_compress = torch.empty(BATCH, 0, HEAD_DIM, dtype=DTYPE, device=DEVICE)
        compress_topk_idxs = torch.empty(BATCH, SEQ_LEN, 0, dtype=torch.int64, device=DEVICE)

    return q, kv, sink, kv_compress, compress_topk_idxs


def fmt_bytes(n):
    return f"{n / 1024 ** 2:8.1f} MiB"


def bench(fn, name, q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio):
    # Use grad-enabled leaf tensors for this benchmark.
    # This avoids needing to modify make_inputs().
    q = q.detach().requires_grad_(True)
    kv = kv.detach().requires_grad_(True)
    sink = sink.detach().requires_grad_(True)

    if kv_compress.numel() > 0:
        kv_compress = kv_compress.detach().requires_grad_(True)
    else:
        kv_compress = kv_compress.detach()

    compress_topk_idxs = compress_topk_idxs.detach()

    grad_tensors = [q, kv, sink, kv_compress]

    def zero_grads():
        for t in grad_tensors:
            if t.requires_grad:
                t.grad = None

    # ----------------------------------------------------------------------
    # Warmup.
    #
    # For torch.compile, this also warms up / compiles the backward graph.
    # ----------------------------------------------------------------------
    for _ in range(WARMUP):
        zero_grads()

        out = fn(q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
        loss = out.float().sum()
        loss.backward()

        del out, loss

    torch.cuda.synchronize()
    zero_grads()

    # ----------------------------------------------------------------------
    # Memory baseline.
    #
    # This baseline includes inputs, CUDA context, compiled graph state, etc.
    # The "+..." number below is the extra peak over this baseline.
    # ----------------------------------------------------------------------
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()

    torch.cuda.reset_peak_memory_stats()

    # ----------------------------------------------------------------------
    # Timed forward + backward.
    #
    # CUDA events let us report separate forward and backward GPU times
    # without synchronizing every iteration.
    # ----------------------------------------------------------------------
    fwd_events = []
    bwd_events = []

    for _ in range(ITERS):
        zero_grads()

        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        out = fn(q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
        fwd_end.record()

        # This reduction is just to create a scalar loss for backward.
        # It is intentionally not included in either the forward or backward
        # event timing.
        loss = out.float().sum()

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        fwd_events.append((fwd_start, fwd_end))
        bwd_events.append((bwd_start, bwd_end))

        del out, loss

    torch.cuda.synchronize()

    fwd_ms = sum(start.elapsed_time(end) for start, end in fwd_events) / ITERS
    bwd_ms = sum(start.elapsed_time(end) for start, end in bwd_events) / ITERS

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    extra_allocated = peak_allocated - baseline_allocated
    extra_reserved = peak_reserved - baseline_reserved

    print(
        f"  {name:30s}: "
        f"{fwd_ms:8.3f} ms/iter fwd | "
        f"{bwd_ms:8.3f} ms/iter bwd | "
        f"peak alloc {fmt_bytes(peak_allocated)} "
        f"(+{fmt_bytes(extra_allocated).strip()}) | "
        f"peak reserved {fmt_bytes(peak_reserved)} "
        f"(+{fmt_bytes(extra_reserved).strip()})"
    )

    zero_grads()

    return fwd_ms, bwd_ms, peak_allocated, peak_reserved


def run_for_compress_ratio(compress_ratio):
    print(f"\n{'=' * 60}")
    print(f"  compress_ratio = {compress_ratio}")
    print(f"{'=' * 60}")

    q, kv, sink, kv_compress, compress_topk_idxs = make_inputs(compress_ratio)

    print(q.shape, kv_compress.shape, compress_topk_idxs.shape)
    # Need separate compiled functions per compress_ratio since shapes differ
    fn_flex = torch.compile(sparse_attn_flex)
    fn_gather = torch.compile(sparse_attn_gather)
    fn_scatter = torch.compile(sparse_attn_scatter)

    print("\n  --- Forward Pass (compiled) ---")
    
    bench(fn_gather, "gather_attention (5d7aaee8)", q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    bench(fn_scatter, "scatter_mask (c2d8a7ab)", q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    bench(fn_flex, "flex_attention (score_mod)", q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    '''
    print("\n  --- Forward Pass (eager) ---")
    bench(sparse_attn_flex, "flex_attention (score_mod)", q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    bench(sparse_attn_gather, "gather_attention (5d7aaee8)", q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    bench(sparse_attn_scatter, "scatter_mask (c2d8a7ab)", q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    '''
    # Correctness check: all three should produce the same output
    print("\n  --- Correctness (atol=1e-3, rtol=1e-3) ---")
    out_flex = sparse_attn_flex(q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    out_gather = sparse_attn_gather(q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)
    out_scatter = sparse_attn_scatter(q, kv, sink, kv_compress, compress_topk_idxs, compress_ratio)

    def check(name_a, out_a, name_b, out_b):
        close = torch.allclose(out_a, out_b, atol=1e-3, rtol=1e-3)
        if close:
            print(f"  {name_a} vs {name_b}: PASS")
        else:
            diff = (out_a - out_b).abs()
            print(f"  {name_a} vs {name_b}: FAIL (max_diff={diff.max().item():.6f}, "
                  f"mean_diff={diff.mean().item():.6f})")

    check("flex", out_flex, "gather", out_gather)
    check("flex", out_flex, "scatter", out_scatter)
    check("gather", out_gather, "scatter", out_scatter)

def main():
    print("=" * 60)
    print("Sparse Attention Benchmark")
    print("=" * 60)
    print(f"  batch={BATCH}  seq_len={SEQ_LEN}  head_dim={HEAD_DIM}  n_heads={N_HEADS}")
    print(f"  window_size={WINDOW_SIZE}  index_topk={INDEX_TOPK}  scale={SCALE:f}  dtype={DTYPE}")
    print(f"  warmup={WARMUP}  iters={ITERS}")

    for cr in [1, 4, 128]:
        run_for_compress_ratio(cr)

    print()


if __name__ == "__main__":
    main()
