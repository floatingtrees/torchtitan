"""Fused Triton SWA, CSA, and HCA with shared key/value states.

Tensor dimension legend:
    B: batch, L: sequence, N: query heads, D: head dimension
    Q: queries in a tile, H: heads in a tile, R: flattened Q * H rows
    K: keys in a tile
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _build_csa_bitmap_kernel(
    indices,
    bitmap,
    max_selected,
    sequence_length: tl.constexpr,
    compress_length: tl.constexpr,
    num_topk: tl.constexpr,
    block_topk: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_topk)
    values = tl.load(
        indices + row * num_topk + offsets,
        mask=offsets < num_topk,
        other=-1,
    )
    compressed = values - sequence_length
    valid = (offsets < num_topk) & (compressed >= 0) & (compressed < compress_length)
    safe_compressed = tl.where(valid, compressed, 0)
    word = safe_compressed // 32
    bit = safe_compressed % 32
    num_words: tl.constexpr = triton.cdiv(compress_length, 32)
    tl.atomic_or(
        bitmap + row * num_words + word,
        1 << bit,
        mask=valid,
    )
    maximum = tl.max(tl.where(valid, compressed + 1, 0), axis=0)
    tl.store(max_selected + row, maximum)


@triton.jit
def _build_csa_min_query_kernel(
    bitmap,
    min_query,
    sequence_length: tl.constexpr,
    compress_length: tl.constexpr,
    block_queries: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch = tl.program_id(1)
    queries = tl.arange(0, block_queries)
    num_words: tl.constexpr = triton.cdiv(compress_length, 32)
    first_word = key_block * 2
    minimum = sequence_length
    for query_start in tl.range(0, sequence_length, block_queries):
        query_pos = query_start + queries
        query_valid = query_pos < sequence_length
        bitmap_offset = (batch * sequence_length + query_pos) * num_words + first_word
        first_bits = tl.load(
            bitmap + bitmap_offset,
            mask=query_valid & (first_word < num_words),
            other=0,
        )
        second_bits = tl.load(
            bitmap + bitmap_offset + 1,
            mask=query_valid & (first_word + 1 < num_words),
            other=0,
        )
        selected = (first_bits | second_bits) != 0
        block_minimum = tl.min(tl.where(selected, query_pos, sequence_length), axis=0)
        minimum = tl.minimum(minimum, block_minimum)
    num_key_blocks: tl.constexpr = triton.cdiv(compress_length, 64)
    tl.store(min_query + batch * num_key_blocks + key_block, minimum)


@triton.jit
def _reduce_gradient_partials_kernel(
    partials,
    output,
    num_elements: tl.constexpr,
    num_partials: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < num_elements
    accumulator = tl.zeros([block_size], tl.float32)
    for partial in tl.static_range(0, num_partials):
        accumulator += tl.load(
            partials + partial * num_elements + offsets,
            mask=valid,
            other=0.0,
        )
    tl.store(output + offsets, accumulator, mask=valid)


@triton.jit
def _reduce_sink_partials_kernel(
    partials,
    output,
    num_partials: tl.constexpr,
    block_size: tl.constexpr,
):
    head = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    accumulator = tl.zeros([block_size], tl.float32)
    for start in tl.static_range(0, num_partials, block_size):
        partial = start + offsets
        accumulator += tl.load(
            partials + head * num_partials + partial,
            mask=partial < num_partials,
            other=0.0,
        )
    tl.store(output + head, tl.sum(accumulator, axis=0))


@triton.jit
def _online_attention_update(
    query_RD,
    kv_KD,
    allowed_RK,
    maximum_R,
    denominator_R,
    accumulator_RD,
    scale: tl.constexpr,
):
    logits_RK = tl.dot(query_RD, tl.trans(kv_KD), input_precision="tf32") * scale
    logits_RK = tl.where(allowed_RK, logits_RK, float("-inf"))
    block_maximum_R = tl.max(logits_RK, axis=1)
    new_maximum_R = tl.maximum(maximum_R, block_maximum_R)
    alpha_R = tl.exp(maximum_R - new_maximum_R)
    probabilities_RK = tl.exp(logits_RK - new_maximum_R[:, None])
    accumulator_RD *= alpha_R[:, None]
    accumulator_RD += tl.dot(
        probabilities_RK.to(query_RD.dtype),
        kv_KD,
        input_precision="tf32",
    )
    denominator_R = denominator_R * alpha_R + tl.sum(probabilities_RK, axis=1)
    return new_maximum_R, denominator_R, accumulator_RD


@triton.jit
def _blocked_forward_kernel(
    query,
    kv,
    kv_compress,
    csa_bitmap,
    csa_max_selected,
    sink,
    output,
    lse,
    stride_qb: tl.constexpr,
    stride_ql: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvl: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cl: tl.constexpr,
    sequence_length: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    window_size: tl.constexpr,
    compress_length: tl.constexpr,
    compress_ratio: tl.constexpr,
    scale: tl.constexpr,
    mode: tl.constexpr,
    specialize_window: tl.constexpr,
    block_queries: tl.constexpr,
    block_heads: tl.constexpr,
    block_rows: tl.constexpr,
    block_dim: tl.constexpr,
    block_keys: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)
    rows = tl.arange(0, block_rows)
    dims = tl.arange(0, block_dim)
    query_pos = query_block * block_queries + rows // block_heads
    head = head_block * block_heads + rows % block_heads
    row_valid = (query_pos < sequence_length) & (head < num_heads)
    query_RD = tl.load(
        query
        + batch * stride_qb
        + query_pos[:, None] * stride_ql
        + head[:, None] * stride_qn
        + dims[None, :],
        mask=row_valid[:, None] & (dims[None, :] < head_dim),
        other=0.0,
    )
    sink_R = tl.load(sink + head, mask=row_valid, other=0.0).to(tl.float32)
    maximum_R = sink_R
    denominator_R = tl.full([block_rows], 1.0, tl.float32)
    accumulator_RD = tl.zeros([block_rows, block_dim], tl.float32)

    normal_start = tl.maximum(query_block * block_queries - window_size + 1, 0)
    num_normal_keys: tl.constexpr = window_size + block_queries - 1
    if specialize_window and window_size == 128 and block_queries <= 32:
        keys = tl.arange(0, 128)
        key_pos = normal_start + keys
        key_valid = key_pos < sequence_length
        kv_KD = tl.load(
            kv + batch * stride_kvb + key_pos[:, None] * stride_kvl + dims[None, :],
            mask=key_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        allowed_RK = (
            row_valid[:, None]
            & key_valid[None, :]
            & (key_pos[None, :] <= query_pos[:, None])
            & (query_pos[:, None] - key_pos[None, :] < window_size)
        )
        maximum_R, denominator_R, accumulator_RD = _online_attention_update(
            query_RD,
            kv_KD,
            allowed_RK,
            maximum_R,
            denominator_R,
            accumulator_RD,
            scale,
        )
        tail_keys = tl.arange(0, 16)
        key_pos = normal_start + 128 + tail_keys
        key_valid = key_pos < sequence_length
        kv_KD = tl.load(
            kv + batch * stride_kvb + key_pos[:, None] * stride_kvl + dims[None, :],
            mask=key_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        allowed_RK = (
            row_valid[:, None]
            & key_valid[None, :]
            & (key_pos[None, :] <= query_pos[:, None])
            & (query_pos[:, None] - key_pos[None, :] < window_size)
        )
        maximum_R, denominator_R, accumulator_RD = _online_attention_update(
            query_RD,
            kv_KD,
            allowed_RK,
            maximum_R,
            denominator_R,
            accumulator_RD,
            scale,
        )
    else:
        for key_start in tl.static_range(0, num_normal_keys, block_keys):
            keys = tl.arange(0, block_keys)
            key_pos = normal_start + key_start + keys
            key_valid = key_pos < sequence_length
            kv_KD = tl.load(
                kv + batch * stride_kvb + key_pos[:, None] * stride_kvl + dims[None, :],
                mask=key_valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            )
            allowed_RK = (
                row_valid[:, None]
                & key_valid[None, :]
                & (key_pos[None, :] <= query_pos[:, None])
                & (query_pos[:, None] - key_pos[None, :] < window_size)
            )
            maximum_R, denominator_R, accumulator_RD = _online_attention_update(
                query_RD,
                kv_KD,
                allowed_RK,
                maximum_R,
                denominator_R,
                accumulator_RD,
                scale,
            )

    if mode != 0:
        num_words: tl.constexpr = triton.cdiv(compress_length, 32)
        compressed_limit = compress_length
        if mode == 1:
            compressed_limit = tl.max(
                tl.load(
                    csa_max_selected + batch * sequence_length + query_pos,
                    mask=row_valid,
                    other=0,
                ),
                axis=0,
            )
        for key_start in tl.range(
            0,
            compressed_limit,
            block_keys,
            num_stages=1,
            loop_unroll_factor=1,
        ):
            compressed_keys = tl.arange(0, block_keys)
            compressed_pos = key_start + compressed_keys
            compressed_key_valid = compressed_pos < compress_length
            kv_compressed_KD = tl.load(
                kv_compress
                + batch * stride_cb
                + compressed_pos[:, None] * stride_cl
                + dims[None, :],
                mask=compressed_key_valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            )
            if mode == 1:
                safe_query_pos = tl.minimum(query_pos, sequence_length - 1)
                words_RK = tl.load(
                    csa_bitmap
                    + (batch * sequence_length + safe_query_pos[:, None]) * num_words
                    + compressed_pos[None, :] // 32,
                    mask=row_valid[:, None] & compressed_key_valid[None, :],
                    other=0,
                )
                selected_RK = (words_RK & (1 << (compressed_pos[None, :] % 32))) != 0
                compressed_allowed_RK = (
                    row_valid[:, None] & compressed_key_valid[None, :] & selected_RK
                )
            else:
                compressed_allowed_RK = (
                    row_valid[:, None]
                    & compressed_key_valid[None, :]
                    & (
                        compressed_pos[None, :]
                        < (query_pos[:, None] + 1) // compress_ratio
                    )
                )
            maximum_R, denominator_R, accumulator_RD = _online_attention_update(
                query_RD,
                kv_compressed_KD,
                compressed_allowed_RK,
                maximum_R,
                denominator_R,
                accumulator_RD,
                scale,
            )

    output_RD = accumulator_RD / denominator_R[:, None]
    output_offsets = (
        batch * stride_qb
        + query_pos[:, None] * stride_ql
        + head[:, None] * stride_qn
        + dims[None, :]
    )
    tl.store(
        output + output_offsets,
        output_RD,
        mask=row_valid[:, None] & (dims[None, :] < head_dim),
    )
    tl.store(
        lse + (batch * sequence_length + query_pos) * num_heads + head,
        maximum_R + tl.log(denominator_R),
        mask=row_valid,
    )


@triton.jit
def _backward_attention_update(
    query_RD,
    grad_output_RD,
    kv_KD,
    allowed_RK,
    lse_R,
    delta_R,
    grad_query_RD,
    scale: tl.constexpr,
):
    logits_RK = tl.dot(query_RD, tl.trans(kv_KD), input_precision="tf32") * scale
    probabilities_RK = tl.exp(logits_RK - lse_R[:, None])
    probabilities_RK = tl.where(allowed_RK, probabilities_RK, 0.0)
    grad_value_RK = tl.dot(grad_output_RD, tl.trans(kv_KD), input_precision="tf32")
    grad_logits_RK = probabilities_RK * (grad_value_RK - delta_R[:, None])
    grad_query_RD += scale * tl.dot(
        grad_logits_RK.to(query_RD.dtype),
        kv_KD,
        input_precision="tf32",
    )
    return grad_query_RD, probabilities_RK, grad_logits_RK


@triton.jit
def _blocked_backward_kernel(
    query,
    kv,
    kv_compress,
    csa_bitmap,
    csa_max_selected,
    sink,
    output,
    lse,
    grad_output,
    grad_query,
    grad_kv,
    grad_kv_compress,
    grad_sink_partial,
    probabilities,
    grad_logits,
    stride_qb: tl.constexpr,
    stride_ql: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvl: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cl: tl.constexpr,
    sequence_length: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    window_size: tl.constexpr,
    compress_length: tl.constexpr,
    compress_ratio: tl.constexpr,
    scale: tl.constexpr,
    mode: tl.constexpr,
    block_queries: tl.constexpr,
    block_heads: tl.constexpr,
    block_rows: tl.constexpr,
    block_dim: tl.constexpr,
    block_keys: tl.constexpr,
    workspace_width: tl.constexpr,
    num_sink_partials: tl.constexpr,
    grad_output_broadcast: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)
    rows = tl.arange(0, block_rows)
    dims = tl.arange(0, block_dim)
    normal_block_keys: tl.constexpr = 32 if mode == 1 else 64
    normal_keys = tl.arange(0, normal_block_keys)
    query_pos = query_block * block_queries + rows // block_heads
    head = head_block * block_heads + rows % block_heads
    row_valid = (query_pos < sequence_length) & (head < num_heads)
    offsets_RD = (
        batch * stride_qb
        + query_pos[:, None] * stride_ql
        + head[:, None] * stride_qn
        + dims[None, :]
    )
    mask_RD = row_valid[:, None] & (dims[None, :] < head_dim)
    query_RD = tl.load(query + offsets_RD, mask=mask_RD, other=0.0)
    output_RD = tl.load(output + offsets_RD, mask=mask_RD, other=0.0)
    if grad_output_broadcast:
        grad_output_RD = (
            tl.zeros([block_rows, block_dim], query_RD.dtype)
            + tl.load(grad_output)
        )
    else:
        grad_output_RD = tl.load(
            grad_output + offsets_RD, mask=mask_RD, other=0.0
        )
    lse_R = tl.load(
        lse + (batch * sequence_length + query_pos) * num_heads + head,
        mask=row_valid,
        other=0.0,
    )
    delta_R = tl.sum(output_RD * grad_output_RD, axis=1)
    grad_query_RD = tl.zeros([block_rows, block_dim], tl.float32)

    normal_start = tl.maximum(query_block * block_queries - window_size + 1, 0)
    num_normal_keys: tl.constexpr = window_size + block_queries - 1
    num_full_normal_keys: tl.constexpr = (
        num_normal_keys // normal_block_keys
    ) * normal_block_keys
    for key_start in tl.range(
        0,
        num_full_normal_keys,
        normal_block_keys,
        num_stages=1 if mode == 1 else 2,
        loop_unroll_factor=1,
        warp_specialize=True,
    ):
        key_pos = normal_start + key_start + normal_keys
        key_valid = key_pos < sequence_length
        safe_key_pos = tl.minimum(key_pos, sequence_length - 1)
        kv_KD = tl.load(
            kv
            + batch * stride_kvb
            + safe_key_pos[:, None] * stride_kvl
            + dims[None, :],
            mask=key_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        allowed_RK = (
            row_valid[:, None]
            & key_valid[None, :]
            & (key_pos[None, :] <= query_pos[:, None])
            & (query_pos[:, None] - key_pos[None, :] < window_size)
        )
        grad_query_RD, probabilities_RK, grad_logits_RK = _backward_attention_update(
            query_RD,
            grad_output_RD,
            kv_KD,
            allowed_RK,
            lse_R,
            delta_R,
            grad_query_RD,
            scale,
        )
        slots_RK = query_pos[:, None] - key_pos[None, :]
        tl.store(
            probabilities
            + (
                (batch * sequence_length + query_pos[:, None]) * num_heads
                + head[:, None]
            )
            * workspace_width
            + slots_RK,
            probabilities_RK,
            mask=allowed_RK,
        )
        tl.store(
            grad_logits
            + (
                (batch * sequence_length + query_pos[:, None]) * num_heads
                + head[:, None]
            )
            * workspace_width
            + slots_RK,
            grad_logits_RK,
            mask=allowed_RK,
        )

    num_tail_normal_keys: tl.constexpr = num_normal_keys - num_full_normal_keys
    merge_tail_with_compressed: tl.constexpr = (
        mode == 2
        and compress_length > 0
        and num_tail_normal_keys + compress_length <= 32
    )
    if num_tail_normal_keys > 0 and not merge_tail_with_compressed:
        tail_block_keys: tl.constexpr = max(
            16, triton.next_power_of_2(num_tail_normal_keys)
        )
        tail_keys = tl.arange(0, tail_block_keys)
        tail_key_pos = normal_start + num_full_normal_keys + tail_keys
        tail_key_valid = (tail_keys < num_tail_normal_keys) & (
            tail_key_pos < sequence_length
        )
        tail_safe_key_pos = tl.minimum(tail_key_pos, sequence_length - 1)
        tail_kv_KD = tl.load(
            kv
            + batch * stride_kvb
            + tail_safe_key_pos[:, None] * stride_kvl
            + dims[None, :],
            mask=tail_key_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        tail_allowed_RK = (
            row_valid[:, None]
            & tail_key_valid[None, :]
            & (tail_key_pos[None, :] <= query_pos[:, None])
            & (query_pos[:, None] - tail_key_pos[None, :] < window_size)
        )
        grad_query_RD, tail_probabilities_RK, tail_grad_logits_RK = (
            _backward_attention_update(
                query_RD,
                grad_output_RD,
                tail_kv_KD,
                tail_allowed_RK,
                lse_R,
                delta_R,
                grad_query_RD,
                scale,
            )
        )
        tail_slots_RK = query_pos[:, None] - tail_key_pos[None, :]
        tail_workspace_offsets_RK = (
            (batch * sequence_length + query_pos[:, None]) * num_heads + head[:, None]
        ) * workspace_width + tail_slots_RK
        tl.store(
            probabilities + tail_workspace_offsets_RK,
            tail_probabilities_RK,
            mask=tail_allowed_RK,
        )
        tl.store(
            grad_logits + tail_workspace_offsets_RK,
            tail_grad_logits_RK,
            mask=tail_allowed_RK,
        )

    if mode != 0:
        keys = tl.arange(0, block_keys)
        num_words: tl.constexpr = triton.cdiv(compress_length, 32)
        if merge_tail_with_compressed:
            normal_member = keys < num_tail_normal_keys
            merged_normal_pos = normal_start + num_full_normal_keys + keys
            normal_valid = normal_member & (merged_normal_pos < sequence_length)
            normal_kv_KD = tl.load(
                kv
                + batch * stride_kvb
                + merged_normal_pos[:, None] * stride_kvl
                + dims[None, :],
                mask=normal_valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            )
            merged_compressed_pos = keys - num_tail_normal_keys
            compressed_valid = ~normal_member & (
                merged_compressed_pos < compress_length
            )
            safe_compressed_pos = tl.maximum(merged_compressed_pos, 0)
            compressed_kv_KD = tl.load(
                kv_compress
                + batch * stride_cb
                + safe_compressed_pos[:, None] * stride_cl
                + dims[None, :],
                mask=compressed_valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            )
            merged_kv_KD = tl.where(
                normal_member[:, None], normal_kv_KD, compressed_kv_KD
            )
            normal_allowed_RK = (
                row_valid[:, None]
                & normal_valid[None, :]
                & (merged_normal_pos[None, :] <= query_pos[:, None])
                & (query_pos[:, None] - merged_normal_pos[None, :] < window_size)
            )
            compressed_allowed_RK = (
                row_valid[:, None]
                & compressed_valid[None, :]
                & (
                    merged_compressed_pos[None, :]
                    < (query_pos[:, None] + 1) // compress_ratio
                )
            )
            merged_allowed_RK = normal_allowed_RK | compressed_allowed_RK
            grad_query_RD, merged_probabilities_RK, merged_grad_logits_RK = (
                _backward_attention_update(
                    query_RD,
                    grad_output_RD,
                    merged_kv_KD,
                    merged_allowed_RK,
                    lse_R,
                    delta_R,
                    grad_query_RD,
                    scale,
                )
            )
            merged_slots_RK = tl.where(
                normal_member[None, :],
                query_pos[:, None] - merged_normal_pos[None, :],
                window_size + merged_compressed_pos[None, :],
            )
            merged_workspace_offsets_RK = (
                (batch * sequence_length + query_pos[:, None]) * num_heads
                + head[:, None]
            ) * workspace_width + merged_slots_RK
            tl.store(
                probabilities + merged_workspace_offsets_RK,
                merged_probabilities_RK,
                mask=merged_allowed_RK,
            )
            tl.store(
                grad_logits + merged_workspace_offsets_RK,
                merged_grad_logits_RK,
                mask=merged_allowed_RK,
            )
        else:
            compressed_limit = compress_length
            if mode == 1:
                compressed_limit = tl.max(
                    tl.load(
                        csa_max_selected + batch * sequence_length + query_pos,
                        mask=row_valid,
                        other=0,
                    ),
                    axis=0,
                )
            for key_start in tl.range(
                0,
                compressed_limit,
                block_keys,
                num_stages=1,
                loop_unroll_factor=1,
                warp_specialize=True,
            ):
                compressed_pos = key_start + keys
                key_valid = compressed_pos < compress_length
                safe_compressed_pos = tl.minimum(compressed_pos, compress_length - 1)
                kv_KD = tl.load(
                    kv_compress
                    + batch * stride_cb
                    + safe_compressed_pos[:, None] * stride_cl
                    + dims[None, :],
                    mask=key_valid[:, None] & (dims[None, :] < head_dim),
                    other=0.0,
                )
                if mode == 1:
                    safe_query_pos = tl.minimum(query_pos, sequence_length - 1)
                    words_RK = tl.load(
                        csa_bitmap
                        + (batch * sequence_length + safe_query_pos[:, None])
                        * num_words
                        + compressed_pos[None, :] // 32,
                        mask=row_valid[:, None] & key_valid[None, :],
                        other=0,
                    )
                    selected_RK = (
                        words_RK & (1 << (compressed_pos[None, :] % 32))
                    ) != 0
                    allowed_RK = row_valid[:, None] & key_valid[None, :] & selected_RK
                else:
                    allowed_RK = (
                        row_valid[:, None]
                        & key_valid[None, :]
                        & (
                            compressed_pos[None, :]
                            < (query_pos[:, None] + 1) // compress_ratio
                        )
                    )
                grad_query_RD, probabilities_RK, grad_logits_RK = (
                    _backward_attention_update(
                        query_RD,
                        grad_output_RD,
                        kv_KD,
                        allowed_RK,
                        lse_R,
                        delta_R,
                        grad_query_RD,
                        scale,
                    )
                )
                slots_RK = window_size + compressed_pos[None, :]
                workspace_offsets_RK = (
                    (batch * sequence_length + query_pos[:, None]) * num_heads
                    + head[:, None]
                ) * workspace_width + slots_RK
                tl.store(
                    probabilities + workspace_offsets_RK,
                    probabilities_RK,
                    mask=allowed_RK,
                )
                tl.store(
                    grad_logits + workspace_offsets_RK,
                    grad_logits_RK,
                    mask=allowed_RK,
                )

    tl.store(grad_query + offsets_RD, grad_query_RD, mask=mask_RD)
    sink_R = tl.load(sink + head, mask=row_valid, other=0.0)
    sink_contribution_R = tl.where(
        row_valid,
        -tl.exp(sink_R - lse_R) * delta_R,
        0.0,
    )
    sink_contribution_QH = tl.reshape(sink_contribution_R, [block_queries, block_heads])
    head_offsets = head_block * block_heads + tl.arange(0, block_heads)
    num_query_blocks: tl.constexpr = triton.cdiv(sequence_length, block_queries)
    sink_partial = batch * num_query_blocks + query_block
    tl.store(
        grad_sink_partial + head_offsets * num_sink_partials + sink_partial,
        tl.sum(sink_contribution_QH, axis=0),
        mask=head_offsets < num_heads,
    )


@triton.jit
def _key_major_backward_kernel(
    query,
    kv,
    kv_compress,
    csa_bitmap,
    csa_min_selected,
    output,
    lse,
    grad_output,
    probabilities,
    grad_logits,
    grad_kv,
    stride_qb: tl.constexpr,
    stride_ql: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_kvb: tl.constexpr,
    stride_kvl: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cl: tl.constexpr,
    sequence_length: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    window_size: tl.constexpr,
    compress_length: tl.constexpr,
    compress_ratio: tl.constexpr,
    scale: tl.constexpr,
    mode: tl.constexpr,
    compressed: tl.constexpr,
    block_queries: tl.constexpr,
    block_heads: tl.constexpr,
    block_rows: tl.constexpr,
    block_dim: tl.constexpr,
    block_keys: tl.constexpr,
    num_key_blocks: tl.constexpr,
    num_query_splits: tl.constexpr,
    num_head_blocks: tl.constexpr,
    store_partials: tl.constexpr,
    workspace_width: tl.constexpr,
    grad_output_broadcast: tl.constexpr,
):
    linear_key_block = tl.program_id(0)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)
    key_block = linear_key_block % num_key_blocks
    query_split = linear_key_block // num_key_blocks
    rows = tl.arange(0, block_rows)
    dims = tl.arange(0, block_dim)
    keys = tl.arange(0, block_keys)
    head = head_block * block_heads + rows % block_heads
    key_pos = key_block * block_keys + keys
    if compressed:
        key_valid = key_pos < compress_length
        kv_KD = tl.load(
            kv_compress
            + batch * stride_cb
            + key_pos[:, None] * stride_cl
            + dims[None, :],
            mask=key_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        queries_per_split: tl.constexpr = triton.cdiv(sequence_length, num_query_splits)
        split_start = query_split * queries_per_split
        split_end = tl.minimum(split_start + queries_per_split, sequence_length)
        if mode == 1:
            first_query = tl.load(csa_min_selected + batch * num_key_blocks + key_block)
        else:
            first_query = (key_block * block_keys + 1) * compress_ratio - 1
        query_start = tl.maximum(split_start, first_query)
        num_query_positions: tl.constexpr = queries_per_split
    else:
        key_valid = key_pos < sequence_length
        kv_KD = tl.load(
            kv + batch * stride_kvb + key_pos[:, None] * stride_kvl + dims[None, :],
            mask=key_valid[:, None] & (dims[None, :] < head_dim),
            other=0.0,
        )
        total_query_positions: tl.constexpr = window_size + block_keys - 1
        queries_per_split: tl.constexpr = triton.cdiv(
            total_query_positions, num_query_splits
        )
        query_start = key_block * block_keys + query_split * queries_per_split
        split_end = sequence_length
        num_query_positions: tl.constexpr = queries_per_split

    if compressed and query_start >= split_end:
        return
    accumulator_KD = tl.zeros([block_keys, block_dim], tl.float32)
    for query_offset in tl.range(
        0,
        num_query_positions,
        block_queries,
        num_stages=2,
        loop_unroll_factor=1,
        warp_specialize=True,
    ):
        query_pos = query_start + query_offset + rows // block_heads
        query_in_split = query_offset + rows // block_heads < num_query_positions
        row_valid = (
            query_in_split
            & (query_pos < sequence_length)
            & ((not compressed) | (query_pos < split_end))
            & (head < num_heads)
        )
        offsets_RD = (
            batch * stride_qb
            + query_pos[:, None] * stride_ql
            + head[:, None] * stride_qn
            + dims[None, :]
        )
        mask_RD = row_valid[:, None] & (dims[None, :] < head_dim)
        query_RD = tl.load(query + offsets_RD, mask=mask_RD, other=0.0)
        if grad_output_broadcast:
            grad_output_RD = (
                tl.zeros([block_rows, block_dim], query_RD.dtype)
                + tl.load(grad_output)
            )
        else:
            grad_output_RD = tl.load(
                grad_output + offsets_RD, mask=mask_RD, other=0.0
            )
        if compressed:
            if mode == 1:
                num_words: tl.constexpr = triton.cdiv(compress_length, 32)
                safe_query_pos = tl.minimum(query_pos, sequence_length - 1)
                words_RK = tl.load(
                    csa_bitmap
                    + (batch * sequence_length + safe_query_pos[:, None]) * num_words
                    + key_pos[None, :] // 32,
                    mask=row_valid[:, None] & key_valid[None, :],
                    other=0,
                )
                selected_RK = (words_RK & (1 << (key_pos[None, :] % 32))) != 0
                allowed_RK = row_valid[:, None] & key_valid[None, :] & selected_RK
            else:
                allowed_RK = (
                    row_valid[:, None]
                    & key_valid[None, :]
                    & (key_pos[None, :] < (query_pos[:, None] + 1) // compress_ratio)
                )
        else:
            allowed_RK = (
                row_valid[:, None]
                & key_valid[None, :]
                & (key_pos[None, :] <= query_pos[:, None])
                & (query_pos[:, None] - key_pos[None, :] < window_size)
            )
        if compressed:
            slots_RK = window_size + key_pos[None, :]
        else:
            slots_RK = query_pos[:, None] - key_pos[None, :]
        workspace_offsets_RK = (
            (batch * sequence_length + query_pos[:, None]) * num_heads + head[:, None]
        ) * workspace_width + slots_RK
        probabilities_RK = tl.load(
            probabilities + workspace_offsets_RK,
            mask=allowed_RK,
            other=0.0,
        )
        grad_logits_RK = tl.load(
            grad_logits + workspace_offsets_RK,
            mask=allowed_RK,
            other=0.0,
        )
        accumulator_KD += tl.dot(
            tl.trans(probabilities_RK.to(query_RD.dtype)),
            grad_output_RD,
            input_precision="tf32",
        )
        accumulator_KD += scale * tl.dot(
            tl.trans(grad_logits_RK.to(query_RD.dtype)),
            query_RD,
            input_precision="tf32",
        )

    if compressed:
        grad_ptr = (
            grad_kv + batch * stride_cb + key_pos[:, None] * stride_cl + dims[None, :]
        )
    else:
        grad_ptr = (
            grad_kv + batch * stride_kvb + key_pos[:, None] * stride_kvl + dims[None, :]
        )
    if store_partials:
        partial_index = query_split * num_head_blocks + head_block
        if compressed:
            partial_stride = stride_cb * tl.num_programs(1)
        else:
            partial_stride = stride_kvb * tl.num_programs(1)
        grad_ptr += partial_index * partial_stride
    tl.store(
        grad_ptr,
        accumulator_KD,
        mask=key_valid[:, None] & (dims[None, :] < head_dim),
    )


class SparseAttention(torch.autograd.Function):
    @staticmethod
    @torch.library.triton_op(
        "torchtitan_deepseek_v4_sparse_attention::forward", mutates_args=()
    )
    def _forward_op(
        q: torch.Tensor,
        kv: torch.Tensor,
        sink: torch.Tensor,
        kv_comp: torch.Tensor,
        indices: torch.Tensor,
        ratio: int,
        window: int,
        scale: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        b, length, n, d = q.shape
        mode = 0 if ratio == 1 or kv_comp.shape[1] == 0 else 1 if ratio == 4 else 2
        num_words = triton.cdiv(kv_comp.shape[1], 32) if kv_comp.shape[1] else 0
        csa_bitmap = torch.empty(0, dtype=torch.int32, device=q.device)
        csa_max_selected = torch.empty(0, dtype=torch.int32, device=q.device)
        csa_min_selected = torch.empty(0, dtype=torch.int32, device=q.device)
        if mode == 1:
            csa_bitmap = torch.zeros(
                (b, length, num_words), dtype=torch.int32, device=q.device
            )
            csa_max_selected = torch.zeros(
                (b, length), dtype=torch.int32, device=q.device
            )
            csa_min_selected = torch.full(
                (b, triton.cdiv(kv_comp.shape[1], 64)),
                length,
                dtype=torch.int32,
                device=q.device,
            )
            if indices.shape[-1] > 0:
                block_topk = triton.next_power_of_2(indices.shape[-1])
                torch.library.wrap_triton(_build_csa_bitmap_kernel)[
                    (b * length,)
                ](
                    indices,
                    csa_bitmap,
                    csa_max_selected,
                    sequence_length=length,
                    compress_length=kv_comp.shape[1],
                    num_topk=indices.shape[-1],
                    block_topk=block_topk,
                    num_warps=4,
                )
                torch.library.wrap_triton(_build_csa_min_query_kernel)[
                    (triton.cdiv(kv_comp.shape[1], 64), b)
                ](
                    csa_bitmap,
                    csa_min_selected,
                    sequence_length=length,
                    compress_length=kv_comp.shape[1],
                    block_queries=256,
                    num_warps=4,
                )
        output = torch.empty_like(q)
        lse = torch.empty((b, length, n), device=q.device, dtype=torch.float32)
        if q.dtype == torch.float32:
            block_queries = 4
            block_heads = 8
        else:
            block_queries = 8
            block_heads = 16
        block_rows = block_queries * block_heads
        block_dim = max(16, triton.next_power_of_2(d))
        block_keys = 64 if q.dtype == torch.float32 else (256 if mode == 1 else 64)
        torch.library.wrap_triton(_blocked_forward_kernel)[
            (triton.cdiv(length, block_queries), b, triton.cdiv(n, block_heads))
        ](
            q,
            kv,
            kv_comp,
            csa_bitmap,
            csa_max_selected,
            sink,
            output,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv_comp.stride(0),
            kv_comp.stride(1),
            sequence_length=length,
            num_heads=n,
            head_dim=d,
            window_size=min(length, window),
            compress_length=kv_comp.shape[1],
            compress_ratio=ratio,
            scale=scale,
            mode=mode,
            specialize_window=q.dtype != torch.float32,
            block_queries=block_queries,
            block_heads=block_heads,
            block_rows=block_rows,
            block_dim=block_dim,
            block_keys=block_keys,
            num_warps=8,
            num_stages=3 if mode == 1 else 1,
        )
        return output, lse, csa_bitmap, csa_max_selected, csa_min_selected

    @staticmethod
    def forward(ctx, q, kv, sink, kv_comp, indices, ratio, window, scale):
        (
            output,
            lse,
            csa_bitmap,
            csa_max_selected,
            csa_min_selected,
        ) = torch.ops.torchtitan_deepseek_v4_sparse_attention.forward.default(
            q,
            kv,
            sink,
            kv_comp,
            indices,
            ratio,
            window,
            scale,
        )
        ctx.save_for_backward(
            q,
            kv,
            kv_comp,
            sink,
            output,
            lse,
            csa_bitmap,
            csa_max_selected,
            csa_min_selected,
        )
        ctx.args = ratio, window, scale
        return output

    @staticmethod
    @torch.library.triton_op(
        "torchtitan_deepseek_v4_sparse_attention::backward", mutates_args=()
    )
    def _backward_op(
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_comp: torch.Tensor,
        sink: torch.Tensor,
        output: torch.Tensor,
        lse: torch.Tensor,
        csa_bitmap: torch.Tensor,
        csa_max_selected: torch.Tensor,
        csa_min_selected: torch.Tensor,
        do: torch.Tensor,
        ratio: int,
        window: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = 0 if ratio == 1 or kv_comp.shape[1] == 0 else 1 if ratio == 4 else 2
        b, length, n, d = q.shape
        dq = torch.empty_like(q)
        dkv = torch.empty_like(kv)
        dcomp = torch.empty_like(kv_comp)
        dsink = torch.empty_like(sink)
        grad_output_broadcast = all(stride == 0 for stride in do.stride())
        grad_output = do if grad_output_broadcast else do.contiguous()
        workspace_width = min(length, window)
        if mode != 0:
            workspace_width += kv_comp.shape[1]
        probabilities = torch.empty(
            (b, length, n, workspace_width), dtype=q.dtype, device=q.device
        )
        grad_logits = torch.empty_like(probabilities)
        block_queries = 4
        block_heads = 16
        block_rows = block_queries * block_heads
        block_dim = max(16, triton.next_power_of_2(d))
        block_keys = 64 if mode == 1 else 32
        num_query_blocks = triton.cdiv(length, block_queries)
        num_sink_partials = b * num_query_blocks
        sink_partials = torch.empty(
            (n, num_sink_partials), dtype=torch.float32, device=q.device
        )
        torch.library.wrap_triton(_blocked_backward_kernel)[
            (num_query_blocks, b, triton.cdiv(n, block_heads))
        ](
            q,
            kv,
            kv_comp,
            csa_bitmap,
            csa_max_selected,
            sink,
            output,
            lse,
            grad_output,
            dq,
            dkv,
            dcomp,
            sink_partials,
            probabilities,
            grad_logits,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv_comp.stride(0),
            kv_comp.stride(1),
            sequence_length=length,
            num_heads=n,
            head_dim=d,
            window_size=min(length, window),
            compress_length=kv_comp.shape[1],
            compress_ratio=ratio,
            scale=scale,
            mode=mode,
            block_queries=block_queries,
            block_heads=block_heads,
            block_rows=block_rows,
            block_dim=block_dim,
            block_keys=block_keys,
            workspace_width=workspace_width,
            num_sink_partials=num_sink_partials,
            grad_output_broadcast=grad_output_broadcast,
            num_warps=8,
            num_stages=1,
        )
        torch.library.wrap_triton(_reduce_sink_partials_kernel)[(n,)](
            sink_partials,
            dsink,
            num_partials=num_sink_partials,
            block_size=256,
            num_warps=4,
        )
        key_block_queries = 4 if q.dtype == torch.float32 else 8
        key_block_heads = 16
        key_block_rows = key_block_queries * key_block_heads
        key_block_keys = 64 if q.dtype == torch.float32 else 128
        num_normal_key_blocks = triton.cdiv(length, key_block_keys)
        num_normal_query_splits = 1
        num_head_blocks = triton.cdiv(n, key_block_heads)
        store_normal_partials = num_head_blocks > 1
        if store_normal_partials:
            dkv_accumulator = torch.empty(
                (num_head_blocks, b, length, d),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            dkv_accumulator = dkv
        key_grid = (
            num_normal_key_blocks * num_normal_query_splits,
            b,
            num_head_blocks,
        )
        torch.library.wrap_triton(_key_major_backward_kernel)[key_grid](
            q,
            kv,
            kv_comp,
            csa_bitmap,
            csa_min_selected,
            output,
            lse,
            grad_output,
            probabilities,
            grad_logits,
            dkv_accumulator,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            kv_comp.stride(0),
            kv_comp.stride(1),
            sequence_length=length,
            num_heads=n,
            head_dim=d,
            window_size=min(length, window),
            compress_length=kv_comp.shape[1],
            compress_ratio=ratio,
            scale=scale,
            mode=mode,
            compressed=False,
            block_queries=key_block_queries,
            block_heads=key_block_heads,
            block_rows=key_block_rows,
            block_dim=block_dim,
            block_keys=key_block_keys,
            num_key_blocks=num_normal_key_blocks,
            num_query_splits=num_normal_query_splits,
            num_head_blocks=num_head_blocks,
            store_partials=store_normal_partials,
            workspace_width=workspace_width,
            grad_output_broadcast=grad_output_broadcast,
            num_warps=8,
            num_stages=1,
        )
        reduction_block_size = 256
        if store_normal_partials:
            torch.library.wrap_triton(_reduce_gradient_partials_kernel)[
                (triton.cdiv(kv.numel(), reduction_block_size),)
            ](
                dkv_accumulator,
                dkv,
                num_elements=kv.numel(),
                num_partials=num_head_blocks,
                block_size=reduction_block_size,
                num_warps=4,
            )
        if mode != 0 and kv_comp.shape[1] > 0:
            compressed_key_block_keys = 64 if mode == 1 else 16
            compressed_block_queries = key_block_queries if mode == 1 else 4
            compressed_block_rows = compressed_block_queries * key_block_heads
            num_compressed_key_blocks = triton.cdiv(
                kv_comp.shape[1], compressed_key_block_keys
            )
            num_query_splits = 8 if mode == 1 else 32
            dcomp_partials = torch.zeros(
                (
                    num_query_splits * num_head_blocks,
                    b,
                    kv_comp.shape[1],
                    d,
                ),
                dtype=torch.float32,
                device=q.device,
            )
            compressed_grid = (
                num_compressed_key_blocks * num_query_splits,
                b,
                num_head_blocks,
            )
            torch.library.wrap_triton(_key_major_backward_kernel)[compressed_grid](
                q,
                kv,
                kv_comp,
                csa_bitmap,
                csa_min_selected,
                output,
                lse,
                grad_output,
                probabilities,
                grad_logits,
                dcomp_partials,
                q.stride(0),
                q.stride(1),
                q.stride(2),
                kv.stride(0),
                kv.stride(1),
                kv_comp.stride(0),
                kv_comp.stride(1),
                sequence_length=length,
                num_heads=n,
                head_dim=d,
                window_size=min(length, window),
                compress_length=kv_comp.shape[1],
                compress_ratio=ratio,
                scale=scale,
                mode=mode,
                compressed=True,
                block_queries=compressed_block_queries,
                block_heads=key_block_heads,
                block_rows=compressed_block_rows,
                block_dim=block_dim,
                block_keys=compressed_key_block_keys,
                num_key_blocks=num_compressed_key_blocks,
                num_query_splits=num_query_splits,
                num_head_blocks=num_head_blocks,
                store_partials=True,
                workspace_width=workspace_width,
                grad_output_broadcast=grad_output_broadcast,
                num_warps=8,
                num_stages=1,
            )
            torch.library.wrap_triton(_reduce_gradient_partials_kernel)[
                (triton.cdiv(kv_comp.numel(), reduction_block_size),)
            ](
                dcomp_partials,
                dcomp,
                num_elements=kv_comp.numel(),
                num_partials=num_query_splits * num_head_blocks,
                block_size=reduction_block_size,
                num_warps=4,
            )
        return dq, dkv, dsink, dcomp

    @staticmethod
    def backward(ctx, do):
        (
            q,
            kv,
            kv_comp,
            sink,
            output,
            lse,
            csa_bitmap,
            csa_max_selected,
            csa_min_selected,
        ) = ctx.saved_tensors
        ratio, window, scale = ctx.args
        dq, dkv, dsink, dcomp = (
            torch.ops.torchtitan_deepseek_v4_sparse_attention.backward.default(
                q,
                kv,
                kv_comp,
                sink,
                output,
                lse,
                csa_bitmap,
                csa_max_selected,
                csa_min_selected,
                do,
                ratio,
                window,
                scale,
            )
        )
        return dq, dkv, dsink, dcomp, None, None, None, None


def _validate_inputs(
    query_states,
    kv_states,
    attn_sink,
    kv_compress,
    compress_topk_indices,
    compress_ratio,
    window_size,
):
    tensors = (
        query_states,
        kv_states,
        attn_sink,
        kv_compress,
        compress_topk_indices,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("all sparse attention inputs must be CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all sparse attention inputs must be on the same device")
    if query_states.ndim != 4:
        raise ValueError("query_states must have shape [B, L, N, D]")
    if kv_states.ndim != 3 or kv_compress.ndim != 3:
        raise ValueError("KV tensors must have shape [B, L, D]")
    if attn_sink.ndim != 1 or compress_topk_indices.ndim != 3:
        raise ValueError("attn_sink and compressed indices have invalid ranks")

    batch_size, sequence_length, num_heads, head_dim = query_states.shape
    if batch_size < 1 or sequence_length < 1 or num_heads < 1:
        raise ValueError("batch, sequence, and head counts must be positive")
    if kv_states.shape != (batch_size, sequence_length, head_dim):
        raise ValueError("kv_states shape does not match query_states")
    if kv_compress.shape[0] != batch_size or kv_compress.shape[2] != head_dim:
        raise ValueError("kv_compress shape does not match query_states")
    if attn_sink.shape[0] != num_heads:
        raise ValueError("attn_sink must contain one value per query head")
    if compress_topk_indices.shape[:2] != (batch_size, sequence_length):
        raise ValueError("compressed indices must have shape [B, L, K]")
    if not 1 <= compress_ratio:
        raise ValueError("compress_ratio must be positive")
    expected_compress_length = (
        0 if compress_ratio == 1 else sequence_length // compress_ratio
    )
    if kv_compress.shape[1] != expected_compress_length:
        raise ValueError(
            "kv_compress length must be zero for SWA and L // ratio otherwise"
        )
    if not 1 <= head_dim <= 256:
        raise ValueError("head_dim must be between 1 and 256")
    if not 1 <= window_size:
        raise ValueError("window_size must be positive")

    supported_dtypes = (torch.bfloat16, torch.float16, torch.float32)
    if query_states.dtype not in supported_dtypes:
        raise ValueError("query_states must use bfloat16, float16, or float32")
    if not all(
        tensor.dtype == query_states.dtype
        for tensor in (kv_states, attn_sink, kv_compress)
    ):
        raise ValueError("query, KV, and sink tensors must use the same dtype")
    if compress_topk_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("compressed indices must use int32 or int64")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all sparse attention inputs must be contiguous")


def sparse_attention(
    query_states,
    kv_states,
    attn_sink,
    kv_compress,
    compress_topk_indices,
    compress_ratio,
    *,
    window_size=128,
    scale=0.0625,
):
    """Apply SWA, CSA, or HCA and provide gradients for all float inputs.

    ``compress_ratio == 1`` selects SWA, ``compress_ratio == 4`` selects CSA,
    and every other ratio greater than one selects HCA. CSA indices use set
    semantics: duplicates collapse and invalid or out-of-range values are
    ignored.
    """
    _validate_inputs(
        query_states,
        kv_states,
        attn_sink,
        kv_compress,
        compress_topk_indices,
        compress_ratio,
        window_size,
    )
    return SparseAttention.apply(
        query_states,
        kv_states,
        attn_sink,
        kv_compress,
        compress_topk_indices,
        compress_ratio,
        window_size,
        scale,
    )
