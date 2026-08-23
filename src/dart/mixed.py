"""PyTorch reference implementation of Dart's mixed-precision key cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .quantization import _pack, _unpack


@dataclass
class MixedQuantizedKey:
    """Packed key tensor with 2-bit values and optional promoted 4-bit channels.

    The input layout is ``[batch, kv_heads, sequence, head_dim]``. Quantization
    groups tokens, so every channel has its own scale/minimum for each token
    group. All channels store their low two bits; promoted channels additionally
    store the upper two bits and an index table. This mirrors the storage idea
    of Dart/Kitty while remaining a straightforward PyTorch reference path.
    """

    low_values: torch.Tensor
    high_values: torch.Tensor
    channel_indices: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    original_shape: Tuple[int, ...]
    padded_tokens: int
    group_size: int
    low_bits: int
    promote_bits: int
    output_dtype: torch.dtype

    @property
    def groups(self) -> int:
        return self.padded_tokens // self.group_size

    @property
    def promoted_channels(self) -> int:
        return self.channel_indices.shape[-1]

    @property
    def head_dim(self) -> int:
        return self.original_shape[-1]

    @property
    def nbytes(self) -> int:
        tensors = (self.low_values, self.high_values, self.channel_indices, self.scale, self.zero_point)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def dense_nbytes(self) -> int:
        return self._numel(self.original_shape) * torch.tensor([], dtype=self.output_dtype).element_size()

    @property
    def compression_ratio(self) -> float:
        return self.dense_nbytes / self.nbytes if self.nbytes else 1.0

    @property
    def promote_mask(self) -> torch.Tensor:
        mask = torch.zeros(
            (*self.original_shape[:2], self.head_dim),
            dtype=torch.bool,
            device=self.channel_indices.device,
        )
        if self.promoted_channels:
            mask.scatter_(-1, self.channel_indices.long(), True)
        return mask

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        packed_tokens = self.low_values.shape[-1]
        low = _unpack(self.low_values, self.low_bits, packed_tokens * (8 // self.low_bits))
        low = low[..., : self.group_size]

        high_bits = self.promote_bits - self.low_bits
        high = _unpack(self.high_values, high_bits, self.high_values.shape[-1] * (8 // high_bits))
        high = high[..., : self.group_size]
        high_full = torch.zeros(
            (*self.original_shape[:2], self.head_dim, self.groups, self.group_size),
            dtype=low.dtype,
            device=low.device,
        )
        if self.promoted_channels:
            expanded_indices = self.channel_indices.long()[..., None, None].expand(
                *self.channel_indices.shape, self.groups, self.group_size
            )
            high_full.scatter_(2, expanded_indices, high)
        quantized = low | (high_full << self.low_bits)
        restored = quantized.to(self.scale.dtype) * self.scale.unsqueeze(-1) + self.zero_point.unsqueeze(-1)
        restored = restored.reshape(*self.original_shape[:2], self.head_dim, self.padded_tokens)
        restored = restored[..., : self.original_shape[-2]].transpose(-1, -2)
        return restored.to(dtype or self.output_dtype)

    def to(self, device: torch.device | str) -> "MixedQuantizedKey":
        return MixedQuantizedKey(
            low_values=self.low_values.to(device),
            high_values=self.high_values.to(device),
            channel_indices=self.channel_indices.to(device),
            scale=self.scale.to(device),
            zero_point=self.zero_point.to(device),
            original_shape=self.original_shape,
            padded_tokens=self.padded_tokens,
            group_size=self.group_size,
            low_bits=self.low_bits,
            promote_bits=self.promote_bits,
            output_dtype=self.output_dtype,
        )

    @staticmethod
    def _numel(shape: Tuple[int, ...]) -> int:
        result = 1
        for size in shape:
            result *= size
        return result


def select_key_channels(
    key_states: torch.Tensor,
    promote_ratio: float,
    *,
    strategy: str = "magnitude",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select the most sensitive key channels for promotion.

    Returns a boolean mask ``[B, H, D]`` and sorted channel indices
    ``[B, H, K]``. Selection is deterministic for fixed input tensors.
    """

    if key_states.ndim != 4:
        raise ValueError("key_states must have shape [B, H, T, D]")
    if not 0.0 <= promote_ratio <= 1.0:
        raise ValueError("promote_ratio must be in [0, 1]")
    if strategy not in {"magnitude", "variance"}:
        raise ValueError("strategy must be 'magnitude' or 'variance'")
    batch, heads, _, head_dim = key_states.shape
    count = min(head_dim, int(head_dim * promote_ratio + 1e-6))
    mask = torch.zeros(batch, heads, head_dim, dtype=torch.bool, device=key_states.device)
    if count == 0:
        return mask, torch.empty(batch, heads, 0, dtype=torch.long, device=key_states.device)
    if count == head_dim:
        indices = torch.arange(head_dim, device=key_states.device).view(1, 1, -1).expand(batch, heads, -1)
        return torch.ones_like(mask), indices
    if strategy == "magnitude":
        score = key_states.abs().mean(dim=-2)
    else:
        centered = key_states - key_states.mean(dim=-2, keepdim=True)
        score = centered.square().mean(dim=-2)
    indices = score.topk(count, dim=-1, sorted=True).indices
    mask.scatter_(-1, indices, True)
    return mask, indices


def quantize_key_mixed(
    key_states: torch.Tensor,
    *,
    group_size: int = 128,
    promote_ratio: float = 0.25,
    promote_bits: int = 4,
    strategy: str = "magnitude",
    metadata_dtype: Optional[torch.dtype] = None,
) -> MixedQuantizedKey:
    """Quantize a key chunk with low-bit storage and channel promotion."""

    if key_states.ndim != 4:
        raise ValueError("key_states must have shape [B, H, T, D]")
    if not torch.is_floating_point(key_states):
        raise TypeError("key_states must be floating point")
    if key_states.shape[-2] == 0 or key_states.shape[-1] == 0:
        raise ValueError("key_states sequence and head dimensions must be non-empty")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if promote_bits != 4:
        raise ValueError("the reference mixed representation currently supports promote_bits=4")
    if not torch.isfinite(key_states).all():
        raise ValueError("key_states must contain finite values")

    mask, indices = select_key_channels(key_states, promote_ratio, strategy=strategy)
    original_shape = tuple(key_states.shape)
    batch, heads, tokens, head_dim = key_states.shape
    groups = (tokens + group_size - 1) // group_size
    padded_tokens = groups * group_size
    data = key_states.transpose(-1, -2).contiguous()
    if padded_tokens > tokens:
        data = torch.cat((data, data[..., -1:].expand(-1, -1, -1, padded_tokens - tokens)), dim=-1)
    grouped = data.reshape(batch, heads, head_dim, groups, group_size).float()
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    promoted = mask.unsqueeze(-1)
    qmax = torch.where(promoted, torch.full_like(minimum, (1 << promote_bits) - 1), torch.full_like(minimum, 3.0))
    scale = (maximum - minimum) / qmax
    scale = torch.where(scale > 1e-12, scale, torch.ones_like(scale))
    quantized = ((grouped - minimum.unsqueeze(-1)) / scale.unsqueeze(-1)).round()
    quantized = torch.minimum(torch.maximum(quantized, torch.zeros_like(quantized)), qmax.unsqueeze(-1)).to(torch.uint8)
    packed_tokens = (group_size + 3) // 4
    low_values = _pack(quantized & 0x3, 2, packed_tokens * 4)
    high_bits = promote_bits - 2
    if indices.shape[-1]:
        selected = quantized.gather(
            2,
            indices[..., None, None].expand(batch, heads, indices.shape[-1], groups, group_size),
        ) >> 2
        high_values = _pack(selected, high_bits, packed_tokens * (8 // high_bits))
    else:
        high_values = torch.empty(
            batch,
            heads,
            0,
            groups,
            packed_tokens * (8 // high_bits),
            dtype=torch.uint8,
            device=key_states.device,
        )
    meta_dtype = metadata_dtype or (key_states.dtype if key_states.dtype in (torch.float16, torch.bfloat16) else torch.float32)
    if meta_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("metadata_dtype must be floating point")
    return MixedQuantizedKey(
        low_values=low_values,
        high_values=high_values,
        channel_indices=indices.to(torch.int32),
        scale=scale.to(meta_dtype),
        zero_point=minimum.to(meta_dtype),
        original_shape=original_shape,
        padded_tokens=padded_tokens,
        group_size=group_size,
        low_bits=2,
        promote_bits=promote_bits,
        output_dtype=key_states.dtype,
    )


__all__ = ["MixedQuantizedKey", "quantize_key_mixed", "select_key_channels"]
