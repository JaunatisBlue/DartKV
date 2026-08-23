"""Reference group-wise affine quantization used by DartKV.

The implementation deliberately keeps the representation simple and device
agnostic. Values are bit-packed for 2/4/8-bit storage, while scales and
offsets are stored per group. It is intended as a correctness baseline before
introducing Triton or CUDA kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


_SUPPORTED_BITS = (2, 4, 8)


def _validate_bits(bits: int) -> None:
    if bits not in _SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {_SUPPORTED_BITS}, got {bits}")


def _pack(values: torch.Tensor, bits: int, packed_dim: int) -> torch.Tensor:
    """Pack a uint8 tensor along its final dimension."""

    values_per_byte = 8 // bits
    padding = packed_dim - values.shape[-1]
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    values = values.reshape(*values.shape[:-1], -1, values_per_byte)
    shifts = torch.arange(values_per_byte, device=values.device, dtype=torch.uint8)
    shifts = shifts * bits
    return torch.sum(values << shifts, dim=-1, dtype=torch.uint8)


def _unpack(packed: torch.Tensor, bits: int, packed_dim: int) -> torch.Tensor:
    """Unpack a bit-packed uint8 tensor along its final dimension."""

    values_per_byte = 8 // bits
    shifts = torch.arange(values_per_byte, device=packed.device, dtype=torch.uint8)
    shifts = shifts * bits
    values = (packed.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return values.reshape(*packed.shape[:-1], packed.shape[-1] * values_per_byte)[..., :packed_dim]


def _metadata_dtype(data: torch.Tensor, requested: Optional[torch.dtype]) -> torch.dtype:
    if requested is not None:
        if requested not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError(f"metadata dtype must be floating point, got {requested}")
        return requested
    return data.dtype if data.dtype in (torch.float16, torch.bfloat16) else torch.float32


@dataclass
class QuantizedTensor:
    """A bit-packed affine-quantized tensor.

    ``scale`` and ``zero_point`` are named after the conventional affine
    representation; ``zero_point`` stores the floating-point group minimum,
    not an integer zero-point. ``output_dtype`` records the dtype to use when
    reconstructing the original tensor.
    """

    values: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    original_shape: Tuple[int, ...]
    original_dim: int
    padded_dim: int
    packed_dim: int
    group_size: int
    bits: int
    output_dtype: torch.dtype

    @property
    def groups(self) -> int:
        return (self.padded_dim + self.group_size - 1) // self.group_size

    @property
    def nbytes(self) -> int:
        """Tensor storage used by this object, excluding Python overhead."""

        return self.values.numel() * self.values.element_size() + self.scale.numel() * self.scale.element_size() + self.zero_point.numel() * self.zero_point.element_size()

    @property
    def dense_nbytes(self) -> int:
        return _numel(self.original_shape) * torch.tensor([], dtype=self.output_dtype).element_size()

    @property
    def compression_ratio(self) -> float:
        return self.dense_nbytes / self.nbytes if self.nbytes else 1.0

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Reconstruct the tensor on the same device as the packed values."""

        unpacked = _unpack(self.values, self.bits, self.packed_dim)[..., : self.padded_dim]
        outer_shape = self.original_shape[:-1]
        grouped = unpacked.reshape(*outer_shape, self.groups, self.group_size).to(self.scale.dtype)
        restored = grouped * self.scale.unsqueeze(-1) + self.zero_point.unsqueeze(-1)
        restored = restored.reshape(*outer_shape, self.padded_dim)[..., : self.original_dim]
        return restored.to(dtype or self.output_dtype)

    def to(self, device: torch.device | str) -> "QuantizedTensor":
        return QuantizedTensor(
            values=self.values.to(device),
            scale=self.scale.to(device),
            zero_point=self.zero_point.to(device),
            original_shape=self.original_shape,
            original_dim=self.original_dim,
            padded_dim=self.padded_dim,
            packed_dim=self.packed_dim,
            group_size=self.group_size,
            bits=self.bits,
            output_dtype=self.output_dtype,
        )


def _numel(shape: Tuple[int, ...]) -> int:
    result = 1
    for size in shape:
        result *= size
    return result


def quantize(
    data: torch.Tensor,
    *,
    bits: int = 2,
    group_size: int = 64,
    metadata_dtype: Optional[torch.dtype] = None,
) -> QuantizedTensor:
    """Quantize ``data`` along its last dimension.

    Groups are asymmetric min/max affine ranges. The final group and the
    final packed byte are padded internally, so dimensions that are not
    divisible by ``group_size`` remain supported without changing the output
    shape after dequantization.
    """

    _validate_bits(bits)
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(f"group_size must be a positive integer, got {group_size}")
    if data.ndim == 0:
        raise ValueError("quantize expects a tensor with at least one dimension")
    if not torch.is_floating_point(data):
        raise TypeError(f"quantize expects a floating-point tensor, got {data.dtype}")
    if data.shape[-1] == 0:
        raise ValueError("quantize does not support an empty final dimension")
    if not torch.isfinite(data).all():
        raise ValueError("quantize expects all values to be finite")

    original_shape = tuple(data.shape)
    original_dim = data.shape[-1]
    groups = (original_dim + group_size - 1) // group_size
    padded_dim = groups * group_size
    flat = data.reshape(-1, original_dim).float()
    if padded_dim > original_dim:
        # Replicating the final element avoids making padding alter the range.
        pad = flat[..., -1:].expand(-1, padded_dim - original_dim)
        flat = torch.cat((flat, pad), dim=-1)

    grouped = flat.reshape(-1, groups, group_size)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    qmax = float((1 << bits) - 1)
    scale = (maximum - minimum) / qmax
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    quantized = ((grouped - minimum.unsqueeze(-1)) / scale.unsqueeze(-1)).round()
    quantized = quantized.clamp_(0, qmax).to(torch.uint8).reshape(*data.shape[:-1], padded_dim)

    values_per_byte = 8 // bits
    packed_dim = ((padded_dim + values_per_byte - 1) // values_per_byte) * values_per_byte
    values = _pack(quantized, bits, packed_dim)
    meta_dtype = _metadata_dtype(data, metadata_dtype)
    return QuantizedTensor(
        values=values,
        scale=scale.reshape(*data.shape[:-1], groups).to(meta_dtype),
        zero_point=minimum.reshape(*data.shape[:-1], groups).to(meta_dtype),
        original_shape=original_shape,
        original_dim=original_dim,
        padded_dim=padded_dim,
        packed_dim=packed_dim,
        group_size=group_size,
        bits=bits,
        output_dtype=data.dtype,
    )


def dequantize(data: QuantizedTensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Functional alias for :meth:`QuantizedTensor.dequantize`."""

    return data.dequantize(dtype=dtype)


__all__ = ["QuantizedTensor", "dequantize", "quantize"]
