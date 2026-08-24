"""Single-token Triton attention over Dart packed pages.

This is an intermediate kernel, not the final model integration.  It fuses
page unpack, QK, online softmax state update, and SV for one page at a time.
The Python wrapper still walks page descriptors, which keeps the page-table
semantics visible while removing full-page dequantized temporaries.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch

from .attention import streamed_dart_attention
from .cache import DartKVCache
from .mixed import MixedQuantizedKey
from .page_table import DartPageRun, DartPageTable
from .quantization import QuantizedTensor
from .triton_ops import TRITON_AVAILABLE, triton


if TRITON_AVAILABLE:
    import triton.language as tl

    @triton.jit
    def _fused_page_kernel(
        query_ptr,
        query_stride_b,
        query_stride_h,
        query_stride_d,
        dense_key_ptr,
        dense_key_stride_b,
        dense_key_stride_h,
        dense_key_stride_t,
        dense_key_stride_d,
        key_values_ptr,
        key_values_stride_b,
        key_values_stride_h,
        key_values_stride_d,
        key_values_stride_p,
        key_low_ptr,
        key_low_stride_b,
        key_low_stride_h,
        key_low_stride_d,
        key_low_stride_g,
        key_low_stride_p,
        key_high_ptr,
        key_high_stride_b,
        key_high_stride_h,
        key_high_stride_k,
        key_high_stride_g,
        key_high_stride_p,
        key_index_ptr,
        key_index_stride_b,
        key_index_stride_h,
        key_index_stride_k,
        key_scale_ptr,
        key_scale_stride_b,
        key_scale_stride_h,
        key_scale_stride_d,
        key_scale_stride_g,
        key_zero_ptr,
        dense_value_ptr,
        dense_value_stride_b,
        dense_value_stride_h,
        dense_value_stride_t,
        dense_value_stride_d,
        value_values_ptr,
        value_values_stride_b,
        value_values_stride_h,
        value_values_stride_t,
        value_values_stride_p,
        value_scale_ptr,
        value_scale_stride_b,
        value_scale_stride_h,
        value_scale_stride_t,
        value_scale_stride_g,
        value_zero_ptr,
        max_ptr,
        max_stride_b,
        max_stride_h,
        sum_ptr,
        sum_stride_b,
        sum_stride_h,
        output_ptr,
        output_stride_b,
        output_stride_h,
        output_stride_d,
        page_tokens,
        head_dim,
        query_heads,
        kv_heads,
        key_group_size,
        value_group_size,
        promoted_channels,
        scale,
        KEY_MODE: tl.constexpr,
        VALUE_MODE: tl.constexpr,
        KEY_BITS: tl.constexpr,
        KEY_VALUES_PER_BYTE: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        VALUE_VALUES_PER_BYTE: tl.constexpr,
        K_PAD: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        batch_index = pid // query_heads
        query_head = pid % query_heads
        group = query_heads // kv_heads
        kv_head = query_head // group
        token_offsets = tl.arange(0, BLOCK_T)
        channel_offsets = tl.arange(0, BLOCK_D)
        token_mask = token_offsets < page_tokens
        channel_mask = channel_offsets < head_dim
        matrix_mask = token_mask[:, None] & channel_mask[None, :]

        q = tl.load(
            query_ptr
            + batch_index * query_stride_b
            + query_head * query_stride_h
            + channel_offsets * query_stride_d,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)

        if KEY_MODE == 0:
            key = tl.load(
                dense_key_ptr
                + batch_index * dense_key_stride_b
                + kv_head * dense_key_stride_h
                + token_offsets[:, None] * dense_key_stride_t
                + channel_offsets[None, :] * dense_key_stride_d,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
        elif KEY_MODE == 1:
            key_group = token_offsets // key_group_size
            key_packed = token_offsets // KEY_VALUES_PER_BYTE
            key_shift = (token_offsets % KEY_VALUES_PER_BYTE) * KEY_BITS
            key_offset = (
                batch_index * key_values_stride_b
                + kv_head * key_values_stride_h
                + channel_offsets[None, :] * key_values_stride_d
                + key_packed[:, None] * key_values_stride_p
            )
            key_q = tl.load(key_values_ptr + key_offset, mask=matrix_mask, other=0).to(tl.int32)
            key_q = (key_q >> key_shift[:, None]) & ((1 << KEY_BITS) - 1)
            key_meta_offset = (
                batch_index * key_scale_stride_b
                + kv_head * key_scale_stride_h
                + channel_offsets[None, :] * key_scale_stride_d
                + key_group[:, None] * key_scale_stride_g
            )
            key_scale = tl.load(key_scale_ptr + key_meta_offset, mask=matrix_mask, other=1.0).to(tl.float32)
            key_zero = tl.load(key_zero_ptr + key_meta_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            key = key_q.to(tl.float32) * key_scale + key_zero
        else:
            key_group = token_offsets // key_group_size
            token_in_group = token_offsets % key_group_size
            key_packed = token_in_group // 4
            key_shift = (token_in_group % 4) * 2
            low_offset = (
                batch_index * key_low_stride_b
                + kv_head * key_low_stride_h
                + channel_offsets[None, :] * key_low_stride_d
                + key_group[:, None] * key_low_stride_g
                + key_packed[:, None] * key_low_stride_p
            )
            low = tl.load(key_low_ptr + low_offset, mask=matrix_mask, other=0).to(tl.int32)
            low = (low >> key_shift[:, None]) & 0x3
            candidate = tl.arange(0, K_PAD)[None, None, :]
            candidate_mask = candidate < promoted_channels
            candidate_index = tl.load(
                key_index_ptr
                + batch_index * key_index_stride_b
                + kv_head * key_index_stride_h
                + candidate * key_index_stride_k,
                mask=candidate_mask,
                other=-1,
            )
            selected = candidate_index == channel_offsets[None, :, None]
            selected_any = tl.sum(selected.to(tl.int32), axis=2) > 0
            selected_slot = tl.sum(tl.where(selected, candidate, 0), axis=2)
            high_offset = (
                batch_index * key_high_stride_b
                + kv_head * key_high_stride_h
                + selected_slot * key_high_stride_k
                + key_group[:, None] * key_high_stride_g
                + key_packed[:, None] * key_high_stride_p
            )
            high = tl.load(
                key_high_ptr + high_offset,
                mask=matrix_mask & selected_any,
                other=0,
            ).to(tl.int32)
            high = (high >> key_shift[:, None]) & 0x3
            key_q = low | (high << 2)
            key_meta_offset = (
                batch_index * key_scale_stride_b
                + kv_head * key_scale_stride_h
                + channel_offsets[None, :] * key_scale_stride_d
                + key_group[:, None] * key_scale_stride_g
            )
            key_scale = tl.load(key_scale_ptr + key_meta_offset, mask=matrix_mask, other=1.0).to(tl.float32)
            key_zero = tl.load(key_zero_ptr + key_meta_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            key = key_q.to(tl.float32) * key_scale + key_zero

        if VALUE_MODE == 0:
            value = tl.load(
                dense_value_ptr
                + batch_index * dense_value_stride_b
                + kv_head * dense_value_stride_h
                + token_offsets[:, None] * dense_value_stride_t
                + channel_offsets[None, :] * dense_value_stride_d,
                mask=matrix_mask,
                other=0.0,
            ).to(tl.float32)
        else:
            value_packed = channel_offsets // VALUE_VALUES_PER_BYTE
            value_shift = (channel_offsets % VALUE_VALUES_PER_BYTE) * VALUE_BITS
            value_offset = (
                batch_index * value_values_stride_b
                + kv_head * value_values_stride_h
                + token_offsets[:, None] * value_values_stride_t
                + value_packed[None, :] * value_values_stride_p
            )
            value_q = tl.load(value_values_ptr + value_offset, mask=matrix_mask, other=0).to(tl.int32)
            value_q = (value_q >> value_shift[None, :]) & ((1 << VALUE_BITS) - 1)
            value_group = channel_offsets // value_group_size
            value_meta_offset = (
                batch_index * value_scale_stride_b
                + kv_head * value_scale_stride_h
                + token_offsets[:, None] * value_scale_stride_t
                + value_group[None, :] * value_scale_stride_g
            )
            value_scale = tl.load(value_scale_ptr + value_meta_offset, mask=matrix_mask, other=1.0).to(tl.float32)
            value_zero = tl.load(value_zero_ptr + value_meta_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            value = value_q.to(tl.float32) * value_scale + value_zero

        logits = tl.sum(q[None, :] * key, axis=1) * scale
        logits = tl.where(token_mask, logits, float("-inf"))
        page_max = tl.max(logits, axis=0)
        old_max = tl.load(max_ptr + batch_index * max_stride_b + query_head * max_stride_h)
        old_sum = tl.load(sum_ptr + batch_index * sum_stride_b + query_head * sum_stride_h)
        new_max = tl.maximum(old_max, page_max)
        old_weight = tl.exp(old_max - new_max)
        weights = tl.exp(logits - new_max)
        weights = tl.where(token_mask, weights, 0.0)
        old_output = tl.load(
            output_ptr
            + batch_index * output_stride_b
            + query_head * output_stride_h
            + channel_offsets * output_stride_d,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        page_output = tl.sum(weights[:, None] * value, axis=0)
        new_output = old_output * old_weight + page_output
        new_sum = old_sum * old_weight + tl.sum(weights, axis=0)
        tl.store(max_ptr + batch_index * max_stride_b + query_head * max_stride_h, new_max)
        tl.store(sum_ptr + batch_index * sum_stride_b + query_head * sum_stride_h, new_sum)
        tl.store(
            output_ptr
            + batch_index * output_stride_b
            + query_head * output_stride_h
            + channel_offsets * output_stride_d,
            new_output,
            mask=channel_mask,
        )

    @triton.jit
    def _fused_uniform_page_run_kernel(
        query_ptr,
        query_stride_b,
        query_stride_h,
        query_stride_d,
        key_values_ptr,
        key_values_stride_n,
        key_values_stride_b,
        key_values_stride_h,
        key_values_stride_d,
        key_values_stride_p,
        key_scale_ptr,
        key_scale_stride_n,
        key_scale_stride_b,
        key_scale_stride_h,
        key_scale_stride_d,
        key_scale_stride_g,
        key_zero_ptr,
        value_values_ptr,
        value_values_stride_n,
        value_values_stride_b,
        value_values_stride_h,
        value_values_stride_t,
        value_values_stride_p,
        value_scale_ptr,
        value_scale_stride_n,
        value_scale_stride_b,
        value_scale_stride_h,
        value_scale_stride_t,
        value_scale_stride_g,
        value_zero_ptr,
        max_ptr,
        max_stride_b,
        max_stride_h,
        sum_ptr,
        sum_stride_b,
        sum_stride_h,
        output_ptr,
        output_stride_b,
        output_stride_h,
        output_stride_d,
        page_count,
        page_tokens,
        head_dim,
        query_heads,
        kv_heads,
        key_group_size,
        value_group_size,
        scale,
        KEY_BITS: tl.constexpr,
        KEY_VALUES_PER_BYTE: tl.constexpr,
        VALUE_BITS: tl.constexpr,
        VALUE_VALUES_PER_BYTE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Walk one page-major uniform run per ``(batch, query_head)`` program."""

        pid = tl.program_id(0)
        batch_index = pid // query_heads
        query_head = pid % query_heads
        group = query_heads // kv_heads
        kv_head = query_head // group
        token_offsets = tl.arange(0, BLOCK_T)
        channel_offsets = tl.arange(0, BLOCK_D)
        token_mask = token_offsets < page_tokens
        channel_mask = channel_offsets < head_dim
        matrix_mask = token_mask[:, None] & channel_mask[None, :]
        q = tl.load(
            query_ptr
            + batch_index * query_stride_b
            + query_head * query_stride_h
            + channel_offsets * query_stride_d,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        old_max = tl.load(max_ptr + batch_index * max_stride_b + query_head * max_stride_h)
        old_sum = tl.load(sum_ptr + batch_index * sum_stride_b + query_head * sum_stride_h)
        old_output = tl.load(
            output_ptr
            + batch_index * output_stride_b
            + query_head * output_stride_h
            + channel_offsets * output_stride_d,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)

        for page_index in tl.range(0, page_count):
            key_group = token_offsets // key_group_size
            key_packed = token_offsets // KEY_VALUES_PER_BYTE
            key_shift = (token_offsets % KEY_VALUES_PER_BYTE) * KEY_BITS
            key_offset = (
                page_index * key_values_stride_n
                + batch_index * key_values_stride_b
                + kv_head * key_values_stride_h
                + channel_offsets[None, :] * key_values_stride_d
                + key_packed[:, None] * key_values_stride_p
            )
            key_q = tl.load(key_values_ptr + key_offset, mask=matrix_mask, other=0).to(tl.int32)
            key_q = (key_q >> key_shift[:, None]) & ((1 << KEY_BITS) - 1)
            key_meta_offset = (
                page_index * key_scale_stride_n
                + batch_index * key_scale_stride_b
                + kv_head * key_scale_stride_h
                + channel_offsets[None, :] * key_scale_stride_d
                + key_group[:, None] * key_scale_stride_g
            )
            key_scale_value = tl.load(key_scale_ptr + key_meta_offset, mask=matrix_mask, other=1.0).to(tl.float32)
            key_zero_value = tl.load(key_zero_ptr + key_meta_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            key = key_q.to(tl.float32) * key_scale_value + key_zero_value

            value_packed = channel_offsets // VALUE_VALUES_PER_BYTE
            value_shift = (channel_offsets % VALUE_VALUES_PER_BYTE) * VALUE_BITS
            value_offset = (
                page_index * value_values_stride_n
                + batch_index * value_values_stride_b
                + kv_head * value_values_stride_h
                + token_offsets[:, None] * value_values_stride_t
                + value_packed[None, :] * value_values_stride_p
            )
            value_q = tl.load(value_values_ptr + value_offset, mask=matrix_mask, other=0).to(tl.int32)
            value_q = (value_q >> value_shift[None, :]) & ((1 << VALUE_BITS) - 1)
            value_group = channel_offsets // value_group_size
            value_meta_offset = (
                page_index * value_scale_stride_n
                + batch_index * value_scale_stride_b
                + kv_head * value_scale_stride_h
                + token_offsets[:, None] * value_scale_stride_t
                + value_group[None, :] * value_scale_stride_g
            )
            value_scale_value = tl.load(value_scale_ptr + value_meta_offset, mask=matrix_mask, other=1.0).to(tl.float32)
            value_zero_value = tl.load(value_zero_ptr + value_meta_offset, mask=matrix_mask, other=0.0).to(tl.float32)
            value = value_q.to(tl.float32) * value_scale_value + value_zero_value

            logits = tl.sum(q[None, :] * key, axis=1) * scale
            logits = tl.where(token_mask, logits, float("-inf"))
            page_max = tl.max(logits, axis=0)
            new_max = tl.maximum(old_max, page_max)
            old_weight = tl.exp(old_max - new_max)
            weights = tl.exp(logits - new_max)
            weights = tl.where(token_mask, weights, 0.0)
            page_output = tl.sum(weights[:, None] * value, axis=0)
            old_output = old_output * old_weight + page_output
            old_sum = old_sum * old_weight + tl.sum(weights, axis=0)
            old_max = new_max

        tl.store(max_ptr + batch_index * max_stride_b + query_head * max_stride_h, old_max)
        tl.store(sum_ptr + batch_index * sum_stride_b + query_head * sum_stride_h, old_sum)
        tl.store(
            output_ptr
            + batch_index * output_stride_b
            + query_head * output_stride_h
            + channel_offsets * output_stride_d,
            old_output,
            mask=channel_mask,
        )


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _strides(tensor: torch.Tensor, length: int) -> tuple[int, ...]:
    strides = tuple(tensor.stride())
    return strides + (0,) * (length - len(strides))


def _dummy_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(1, dtype=torch.uint8, device=device),
        torch.empty(1, dtype=torch.float32, device=device),
        torch.empty(1, dtype=torch.int32, device=device),
    )


def _launch_fused_page(
    query: torch.Tensor,
    key_segment: torch.Tensor | QuantizedTensor | MixedQuantizedKey,
    value_segment: torch.Tensor | QuantizedTensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_output: torch.Tensor,
    *,
    scale: float,
) -> None:
    batch, query_heads, _, head_dim = query.shape
    kv_heads = key_segment.shape[1] if isinstance(key_segment, torch.Tensor) else key_segment.original_shape[1]
    page_tokens = key_segment.shape[-2] if isinstance(key_segment, torch.Tensor) else key_segment.original_shape[-2]
    block_t = _next_power_of_two(page_tokens)
    block_d = _next_power_of_two(head_dim)
    dummy_u8, dummy_float, dummy_i32 = _dummy_tensors(query.device)

    key_mode = 0
    key_values = dummy_u8
    key_low = dummy_u8
    key_high = dummy_u8
    key_index = dummy_i32
    key_scale = dummy_float
    key_zero = dummy_float
    key_dense = dummy_float
    key_values_strides = (0, 0, 0, 0)
    key_low_strides = (0, 0, 0, 0, 0)
    key_high_strides = (0, 0, 0, 0, 0)
    key_index_strides = (0, 0, 0)
    key_scale_strides = (0, 0, 0, 0)
    key_dense_strides = (0, 0, 0, 0)
    promoted_channels = 0
    key_bits = 2
    key_group_size = 1
    key_k_pad = 1
    if isinstance(key_segment, torch.Tensor):
        key_dense = key_segment
        key_dense_strides = _strides(key_segment, 4)
    elif isinstance(key_segment, MixedQuantizedKey):
        key_mode = 2
        key_low = key_segment.low_values
        key_high = key_segment.high_values
        key_index = key_segment.channel_indices
        key_scale = key_segment.scale
        key_zero = key_segment.zero_point
        key_low_strides = _strides(key_low, 5)
        key_high_strides = _strides(key_high, 5)
        key_index_strides = _strides(key_index, 3)
        key_scale_strides = _strides(key_scale, 4)
        key_group_size = key_segment.group_size
        promoted_channels = key_segment.promoted_channels
        key_k_pad = _next_power_of_two(promoted_channels)
    else:
        key_mode = 1
        key_values = key_segment.values
        key_scale = key_segment.scale
        key_zero = key_segment.zero_point
        key_values_strides = _strides(key_values, 4)
        key_scale_strides = _strides(key_scale, 4)
        key_group_size = key_segment.group_size
        key_bits = key_segment.bits

    value_mode = 0
    value_dense = dummy_float
    value_values = dummy_u8
    value_scale = dummy_float
    value_zero = dummy_float
    value_dense_strides = (0, 0, 0, 0)
    value_values_strides = (0, 0, 0, 0)
    value_scale_strides = (0, 0, 0, 0)
    value_bits = 2
    value_group_size = 1
    if isinstance(value_segment, torch.Tensor):
        value_dense = value_segment
        value_dense_strides = _strides(value_segment, 4)
    else:
        value_mode = 1
        value_values = value_segment.values
        value_scale = value_segment.scale
        value_zero = value_segment.zero_point
        value_values_strides = _strides(value_values, 4)
        value_scale_strides = _strides(value_scale, 4)
        value_group_size = value_segment.group_size
        value_bits = value_segment.bits

    grid = (batch * query_heads,)
    _fused_page_kernel[grid](
        query,
        query.stride(0),
        query.stride(1),
        query.stride(3),
        key_dense,
        *key_dense_strides,
        key_values,
        *key_values_strides,
        key_low,
        *key_low_strides,
        key_high,
        *key_high_strides,
        key_index,
        *key_index_strides,
        key_scale,
        *key_scale_strides,
        key_zero,
        value_dense,
        *value_dense_strides,
        value_values,
        *value_values_strides,
        value_scale,
        *value_scale_strides,
        value_zero,
        running_max,
        running_max.stride(0),
        running_max.stride(1),
        running_sum,
        running_sum.stride(0),
        running_sum.stride(1),
        running_output,
        running_output.stride(0),
        running_output.stride(1),
        running_output.stride(2),
        page_tokens,
        head_dim,
        query_heads,
        kv_heads,
        key_group_size,
        value_group_size,
        promoted_channels,
        scale,
        KEY_MODE=key_mode,
        VALUE_MODE=value_mode,
        KEY_BITS=key_bits,
        KEY_VALUES_PER_BYTE=8 // key_bits,
        VALUE_BITS=value_bits,
        VALUE_VALUES_PER_BYTE=8 // value_bits,
        K_PAD=key_k_pad,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )


def _launch_fused_uniform_page_run(
    query: torch.Tensor,
    run: DartPageRun,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_output: torch.Tensor,
    *,
    scale: float,
) -> None:
    """Launch one Triton program per query head for a stacked page run."""

    if run.device != query.device:
        run = run.to(query.device)
    run.validate()
    batch, query_heads, _, head_dim = query.shape
    page_tokens = int(run.token_counts[0].item())
    if bool((run.token_counts != page_tokens).any().item()):
        raise ValueError("uniform page run requires equal token counts")
    block_t = _next_power_of_two(page_tokens)
    block_d = _next_power_of_two(head_dim)
    grid = (batch * query_heads,)
    _fused_uniform_page_run_kernel[grid](
        query,
        query.stride(0),
        query.stride(1),
        query.stride(3),
        run.key_values,
        *_strides(run.key_values, 5),
        run.key_scale,
        *_strides(run.key_scale, 5),
        run.key_zero_point,
        run.value_values,
        *_strides(run.value_values, 5),
        run.value_scale,
        *_strides(run.value_scale, 5),
        run.value_zero_point,
        running_max,
        running_max.stride(0),
        running_max.stride(1),
        running_sum,
        running_sum.stride(0),
        running_sum.stride(1),
        running_output,
        running_output.stride(0),
        running_output.stride(1),
        running_output.stride(2),
        run.page_count,
        page_tokens,
        head_dim,
        query_heads,
        run.key_values.shape[2],
        run.key_group_size,
        run.value_group_size,
        scale,
        KEY_BITS=run.key_bits,
        KEY_VALUES_PER_BYTE=8 // run.key_bits,
        VALUE_BITS=run.value_bits,
        VALUE_VALUES_PER_BYTE=8 // run.value_bits,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )


def fused_dart_attention(
    query: torch.Tensor,
    cache: DartKVCache,
    *,
    scale: Optional[float] = None,
    fallback: bool = True,
    page_table: Optional[DartPageTable] = None,
    page_runs: Optional[Sequence[DartPageRun]] = None,
) -> torch.Tensor:
    """Run single-token page attention with fused Triton page updates.

    The wrapper supports one query token.  It keeps one online-softmax state
    tensor for each query head and updates it page by page, so it never calls
    ``cache.get()`` or creates a dense full-cache temporary.  CPU, non-CUDA,
    non-single-token, or missing-Triton calls optionally fall back to the
    PyTorch streaming reference.
    """

    if query.ndim != 4:
        raise ValueError("query must have shape [B, Hq, 1, D]")
    if query.shape[-2] != 1:
        if fallback:
            return streamed_dart_attention(query, cache, scale=scale)
        raise ValueError("fused_dart_attention currently supports one query token")
    if not TRITON_AVAILABLE or query.device.type != "cuda":
        if fallback:
            return streamed_dart_attention(query, cache, scale=scale)
        raise RuntimeError("fused_dart_attention requires CUDA and Triton")
    cache_shape = cache.shape
    if cache_shape is None or cache_shape[2] == 0:
        raise ValueError("cache is empty")
    batch, kv_heads, _, head_dim = cache_shape
    if query.shape[0] != batch or query.shape[-1] != head_dim:
        raise ValueError("query batch and head_dim must match the cache")
    if query.shape[1] % kv_heads:
        raise ValueError("query heads must be divisible by cache KV heads")
    factor = scale if scale is not None else 1.0 / math.sqrt(head_dim)
    segments = cache.iter_segments()
    supplied_page_table = page_table is not None
    table = page_table if supplied_page_table else cache.page_table(device=query.device)
    if table.device != query.device:
        table = table.to(query.device)
    if table.page_count != len(segments) or table.seen_tokens != cache.seen_tokens:
        raise ValueError("page_table does not describe the current cache segments")
    if page_runs is None:
        runs = (
            tuple(table.uniform_quantized_runs(cache))
            if supplied_page_table
            else cache.page_runs(device=query.device)
        )
    else:
        runs = tuple(page_runs)
    run_by_start: dict[int, DartPageRun] = {}
    for run in runs:
        run.validate()
        if run.first_page_index in run_by_start:
            raise ValueError("page_runs contain duplicate first_page_index")
        stop = run.first_page_index + run.page_count
        if run.first_page_index < 0 or stop > len(segments):
            raise ValueError("page_run is outside the current cache page range")
        expected_ids = table.page_ids[0, run.first_page_index:stop].to(run.page_indices.device)
        if not torch.equal(expected_ids, run.page_indices):
            raise ValueError("page_run logical IDs do not match page_table")
        if bool((table.key_modes[run.first_page_index:stop] != 1).any().item()) or bool(
            (table.value_modes[run.first_page_index:stop] != 1).any().item()
        ):
            raise ValueError("page_runs may only contain uniform quantized pages")
        run_by_start[run.first_page_index] = run
    running_max = torch.full((batch, query.shape[1]), -torch.inf, dtype=torch.float32, device=query.device)
    running_sum = torch.zeros_like(running_max)
    running_output = torch.zeros((batch, query.shape[1], head_dim), dtype=torch.float32, device=query.device)

    # The Qwen decode adapter appends one token before every call, which
    # invalidates descriptor caches by design.  Consume the current packed
    # segments directly in that hot path instead of rebuilding page metadata
    # and uniform runs for every token.  Explicit page_table/page_runs callers
    # retain the page-major batching path below.
    if page_table is None and page_runs is None:
        for key_segment, value_segment in segments:
            _launch_fused_page(
                query,
                key_segment,
                value_segment,
                running_max,
                running_sum,
                running_output,
                scale=factor,
            )
        return (running_output / running_sum.clamp_min(1e-20).unsqueeze(-1)).unsqueeze(2).to(query.dtype)

    page_index = 0
    while page_index < len(segments):
        run = run_by_start.get(page_index)
        if run is not None:
            _launch_fused_uniform_page_run(
                query,
                run,
                running_max,
                running_sum,
                running_output,
                scale=factor,
            )
            page_index += run.page_count
            continue
        key_segment, value_segment = segments[page_index]
        if int(table.token_counts[page_index].item()) != _segment_tokens(key_segment):
            raise ValueError(f"page_table token count mismatch at page {page_index}")
        _launch_fused_page(
            query,
            key_segment,
            value_segment,
            running_max,
            running_sum,
            running_output,
            scale=factor,
        )
        page_index += 1
    return (running_output / running_sum.clamp_min(1e-20).unsqueeze(-1)).unsqueeze(2).to(query.dtype)


def _segment_tokens(segment: torch.Tensor | QuantizedTensor | MixedQuantizedKey) -> int:
    return segment.shape[-2] if isinstance(segment, torch.Tensor) else segment.original_shape[-2]


__all__ = ["fused_dart_attention"]
