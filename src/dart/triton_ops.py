"""Optional Triton page dequantization kernels.

The kernels mirror DartKV's PyTorch storage objects and are deliberately
smaller than a fused attention implementation.  They are useful for freezing
the packed layout: every CUDA result is compared with the corresponding
``QuantizedTensor.dequantize`` or ``MixedQuantizedKey.dequantize`` result.
CPU tensors and installations without Triton use the PyTorch reference path.
"""

from __future__ import annotations

from typing import Optional

import torch

from .mixed import MixedQuantizedKey
from .quantization import QuantizedTensor

try:  # Triton is an optional GPU acceleration dependency.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - depends on the host installation
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None


if TRITON_AVAILABLE:

    @triton.jit
    def _dequantize_axis_last_kernel(
        values_ptr,
        scale_ptr,
        zero_ptr,
        output_ptr,
        n_elements,
        original_dim,
        group_size,
        values_row_stride,
        metadata_row_stride,
        BITS: tl.constexpr,
        VALUES_PER_BYTE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        row = offsets // original_dim
        channel = offsets % original_dim
        packed_offset = row * values_row_stride + channel // VALUES_PER_BYTE
        packed = tl.load(values_ptr + packed_offset, mask=mask, other=0).to(tl.int32)
        shift = (channel % VALUES_PER_BYTE) * BITS
        quantized = (packed >> shift) & ((1 << BITS) - 1)
        group = channel // group_size
        scale = tl.load(scale_ptr + row * metadata_row_stride + group, mask=mask, other=1.0).to(tl.float32)
        zero = tl.load(zero_ptr + row * metadata_row_stride + group, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets, quantized.to(tl.float32) * scale + zero, mask=mask)


    @triton.jit
    def _dequantize_axis_token_kernel(
        values_ptr,
        values_stride_b,
        values_stride_h,
        values_stride_d,
        values_stride_p,
        scale_ptr,
        scale_stride_b,
        scale_stride_h,
        scale_stride_d,
        scale_stride_g,
        zero_ptr,
        output_ptr,
        n_elements,
        heads,
        tokens,
        head_dim,
        group_size,
        BITS: tl.constexpr,
        VALUES_PER_BYTE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        channel = offsets % head_dim
        token = (offsets // head_dim) % tokens
        head = (offsets // (tokens * head_dim)) % heads
        batch = offsets // (heads * tokens * head_dim)
        packed_offset = (
            batch * values_stride_b
            + head * values_stride_h
            + channel * values_stride_d
            + (token // VALUES_PER_BYTE) * values_stride_p
        )
        packed = tl.load(values_ptr + packed_offset, mask=mask, other=0).to(tl.int32)
        shift = (token % VALUES_PER_BYTE) * BITS
        quantized = (packed >> shift) & ((1 << BITS) - 1)
        group = token // group_size
        metadata_offset = (
            batch * scale_stride_b
            + head * scale_stride_h
            + channel * scale_stride_d
            + group * scale_stride_g
        )
        scale = tl.load(scale_ptr + metadata_offset, mask=mask, other=1.0).to(tl.float32)
        zero = tl.load(zero_ptr + metadata_offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets, quantized.to(tl.float32) * scale + zero, mask=mask)


    @triton.jit
    def _dequantize_mixed_key_kernel(
        low_ptr,
        low_stride_b,
        low_stride_h,
        low_stride_d,
        low_stride_g,
        low_stride_p,
        high_ptr,
        high_stride_b,
        high_stride_h,
        high_stride_k,
        high_stride_g,
        high_stride_p,
        channel_indices_ptr,
        channel_indices_stride_b,
        channel_indices_stride_h,
        channel_indices_stride_k,
        scale_ptr,
        scale_stride_b,
        scale_stride_h,
        scale_stride_d,
        scale_stride_g,
        zero_ptr,
        output_ptr,
        n_elements,
        heads,
        tokens,
        head_dim,
        group_size,
        promoted_channels,
        K_PAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        channel = offsets % head_dim
        token = (offsets // head_dim) % tokens
        head = (offsets // (tokens * head_dim)) % heads
        batch = offsets // (heads * tokens * head_dim)
        group = token // group_size
        token_in_group = token % group_size
        packed_index = token_in_group // 4
        shift = (token_in_group % 4) * 2

        low_offset = (
            batch * low_stride_b
            + head * low_stride_h
            + channel * low_stride_d
            + group * low_stride_g
            + packed_index * low_stride_p
        )
        low = tl.load(low_ptr + low_offset, mask=mask, other=0).to(tl.int32)
        low = (low >> shift) & 0x3

        candidate = tl.arange(0, K_PAD)[None, :]
        candidate_mask = candidate < promoted_channels
        candidate_index = tl.load(
            channel_indices_ptr
            + batch[:, None] * channel_indices_stride_b
            + head[:, None] * channel_indices_stride_h
            + candidate * channel_indices_stride_k,
            mask=candidate_mask & mask[:, None],
            other=-1,
        )
        selected = candidate_index == channel[:, None]
        selected_any = tl.sum(selected.to(tl.int32), axis=1) > 0
        selected_slot = tl.sum(tl.where(selected, candidate, 0), axis=1)
        high_offset = (
            batch * high_stride_b
            + head * high_stride_h
            + selected_slot * high_stride_k
            + group * high_stride_g
            + packed_index * high_stride_p
        )
        high = tl.load(high_ptr + high_offset, mask=mask & selected_any, other=0).to(tl.int32)
        high = (high >> shift) & 0x3
        quantized = low | (high << 2)

        metadata_offset = (
            batch * scale_stride_b
            + head * scale_stride_h
            + channel * scale_stride_d
            + group * scale_stride_g
        )
        scale = tl.load(scale_ptr + metadata_offset, mask=mask, other=1.0).to(tl.float32)
        zero = tl.load(zero_ptr + metadata_offset, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets, quantized.to(tl.float32) * scale + zero, mask=mask)


def triton_available() -> bool:
    """Return whether the Triton Python package is importable."""

    return TRITON_AVAILABLE


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def _launch_axis_last(data: QuantizedTensor) -> torch.Tensor:
    output = torch.empty(data.original_shape, dtype=data.output_dtype, device=data.values.device)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _dequantize_axis_last_kernel[grid](
        data.values,
        data.scale,
        data.zero_point,
        output,
        n_elements,
        data.original_dim,
        data.group_size,
        data.values.shape[-1],
        data.scale.shape[-1],
        BITS=data.bits,
        VALUES_PER_BYTE=8 // data.bits,
        BLOCK=256,
    )
    return output


def _launch_axis_token(data: QuantizedTensor) -> torch.Tensor:
    if len(data.original_shape) != 4:
        raise ValueError("Triton token-axis dequantization expects [B, H, T, D]")
    output = torch.empty(data.original_shape, dtype=data.output_dtype, device=data.values.device)
    batch, heads, tokens, head_dim = data.original_shape
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _dequantize_axis_token_kernel[grid](
        data.values,
        data.values.stride(0),
        data.values.stride(1),
        data.values.stride(2),
        data.values.stride(3),
        data.scale,
        data.scale.stride(0),
        data.scale.stride(1),
        data.scale.stride(2),
        data.scale.stride(3),
        data.zero_point,
        output,
        n_elements,
        heads,
        tokens,
        head_dim,
        data.group_size,
        BITS=data.bits,
        VALUES_PER_BYTE=8 // data.bits,
        BLOCK=256,
    )
    return output


def _launch_mixed_key(data: MixedQuantizedKey) -> torch.Tensor:
    output = torch.empty(data.original_shape, dtype=data.output_dtype, device=data.low_values.device)
    batch, heads, tokens, head_dim = data.original_shape
    promoted_channels = data.promoted_channels
    k_pad = _next_power_of_two(promoted_channels)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]),)
    _dequantize_mixed_key_kernel[grid](
        data.low_values,
        data.low_values.stride(0),
        data.low_values.stride(1),
        data.low_values.stride(2),
        data.low_values.stride(3),
        data.low_values.stride(4),
        data.high_values,
        data.high_values.stride(0),
        data.high_values.stride(1),
        data.high_values.stride(2),
        data.high_values.stride(3),
        data.high_values.stride(4),
        data.channel_indices,
        data.channel_indices.stride(0),
        data.channel_indices.stride(1),
        data.channel_indices.stride(2),
        data.scale,
        data.scale.stride(0),
        data.scale.stride(1),
        data.scale.stride(2),
        data.scale.stride(3),
        data.zero_point,
        output,
        n_elements,
        heads,
        tokens,
        head_dim,
        data.group_size,
        promoted_channels,
        K_PAD=k_pad,
        BLOCK=256,
    )
    return output


def triton_dequantize(data: QuantizedTensor | MixedQuantizedKey, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Dequantize with Triton on CUDA, otherwise use the PyTorch reference."""

    device = data.values.device if isinstance(data, QuantizedTensor) else data.low_values.device
    if not TRITON_AVAILABLE or device.type != "cuda":
        return data.dequantize(dtype=dtype)
    if isinstance(data, MixedQuantizedKey):
        restored = _launch_mixed_key(data)
    elif data.axis == len(data.original_shape) - 1:
        restored = _launch_axis_last(data)
    elif data.axis == len(data.original_shape) - 2:
        restored = _launch_axis_token(data)
    else:
        raise ValueError("Triton dequantization supports only the last or token axis")
    return restored if dtype is None else restored.to(dtype)


__all__ = ["TRITON_AVAILABLE", "triton_available", "triton_dequantize"]
