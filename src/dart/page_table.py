"""Device-resident page descriptors for DartKV cache segments.

Kitty uses a per-batch table of integer page IDs so its Triton kernels can walk
allocated pages without carrying Python objects into the kernel.  DartKV keeps
the storage objects readable, but exposes the same lifecycle as a compact
device tensor table.  The table is the next boundary for a page-run kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence, Tuple

import torch

from .cache import Segment
from .mixed import MixedQuantizedKey
from .quantization import QuantizedTensor

if TYPE_CHECKING:
    from .cache import DartKVCache


DENSE_PAGE = 0
QUANTIZED_PAGE = 1
MIXED_PAGE = 2
_VALID_PAGE_MODES = (DENSE_PAGE, QUANTIZED_PAGE, MIXED_PAGE)


@dataclass(frozen=True)
class DartPageTable:
    """Batch-indexed page descriptors stored on one device.

    ``page_ids`` follows Kitty's convention and has shape ``[B, N]``.  Each
    row maps a batch item to the page segment index in ``cache.iter_segments``;
    the actual packed tensors remain owned by ``DartKVCache`` for now.  The
    other arrays are page-major because all current updates keep batch rows at
    the same sequence length.
    """

    page_ids: torch.Tensor
    sequence_starts: torch.Tensor
    token_counts: torch.Tensor
    key_modes: torch.Tensor
    value_modes: torch.Tensor
    valid: torch.Tensor
    seen_tokens: int
    page_size: int

    @classmethod
    def from_cache(cls, cache: "DartKVCache", *, device: torch.device | str | None = None) -> "DartPageTable":
        shape = cache.shape
        batch = shape[0] if shape is not None else 0
        segments = cache.iter_segments()
        starts = []
        counts = []
        key_modes = []
        value_modes = []
        sequence_start = 0
        for key_segment, value_segment in segments:
            key_tokens = _segment_tokens(key_segment)
            value_tokens = _segment_tokens(value_segment)
            if key_tokens != value_tokens:
                raise RuntimeError("DartKVCache key/value page lengths diverged")
            starts.append(sequence_start)
            counts.append(key_tokens)
            key_modes.append(_segment_mode(key_segment))
            value_modes.append(_segment_mode(value_segment))
            sequence_start += key_tokens
        if sequence_start != cache.seen_tokens:
            raise RuntimeError(
                f"page table covers {sequence_start} tokens, expected {cache.seen_tokens}"
            )
        target = torch.device(device) if device is not None else _cache_device(cache)
        page_count = len(segments)
        page_ids = torch.arange(page_count, dtype=torch.int64, device=target)
        page_ids = page_ids.unsqueeze(0).expand(batch, -1).contiguous()
        return cls(
            page_ids=page_ids,
            sequence_starts=torch.tensor(starts, dtype=torch.int32, device=target),
            token_counts=torch.tensor(counts, dtype=torch.int32, device=target),
            key_modes=torch.tensor(key_modes, dtype=torch.int8, device=target),
            value_modes=torch.tensor(value_modes, dtype=torch.int8, device=target),
            valid=torch.ones((batch, page_count), dtype=torch.bool, device=target),
            seen_tokens=cache.seen_tokens,
            page_size=cache.config.page_size,
        )

    @property
    def batch_size(self) -> int:
        return self.page_ids.shape[0]

    @property
    def page_count(self) -> int:
        return self.page_ids.shape[1]

    @property
    def device(self) -> torch.device:
        return self.page_ids.device

    def __len__(self) -> int:
        return self.page_count

    def validate(self) -> "DartPageTable":
        """Check shape, mode, ID, order, and total-length invariants."""

        page_count = self.page_count
        if self.page_ids.ndim != 2 or self.valid.shape != self.page_ids.shape:
            raise ValueError("page_ids and valid must have shape [batch, page_count]")
        for name, tensor in (
            ("sequence_starts", self.sequence_starts),
            ("token_counts", self.token_counts),
            ("key_modes", self.key_modes),
            ("value_modes", self.value_modes),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != page_count:
                raise ValueError(f"{name} must have shape [page_count]")
        if self.page_size <= 0 or self.seen_tokens < 0:
            raise ValueError("page_size and seen_tokens must be non-negative/positive")
        if page_count:
            expected_starts = torch.cumsum(
                torch.cat((torch.zeros(1, dtype=torch.int32, device=self.device), self.token_counts[:-1])),
                dim=0,
            )
            if not torch.equal(self.sequence_starts, expected_starts):
                raise ValueError("sequence_starts are not contiguous")
            if bool((self.token_counts <= 0).any().item()):
                raise ValueError("token_counts must be positive")
            if int(self.token_counts.sum().item()) != self.seen_tokens:
                raise ValueError("token_counts do not cover seen_tokens")
            key_supported = (self.key_modes[..., None] == torch.tensor(_VALID_PAGE_MODES, device=self.device)).any(dim=-1)
            if bool((~key_supported).any().item()):
                raise ValueError("key_modes contain an unsupported mode")
            value_supported = (self.value_modes[..., None] == torch.tensor(_VALID_PAGE_MODES, device=self.device)).any(dim=-1)
            if bool((~value_supported).any().item()):
                raise ValueError("value_modes contain an unsupported mode")
            if bool((self.page_ids[self.valid] < 0).any().item()) or bool((self.page_ids[self.valid] >= page_count).any().item()):
                raise ValueError("valid page IDs are outside the page range")
        elif self.seen_tokens != 0:
            raise ValueError("an empty page table cannot cover non-empty cache")
        return self

    def sequence_ranges(self) -> torch.Tensor:
        """Return inclusive-exclusive ``[start, stop]`` ranges on the table device."""

        return torch.stack((self.sequence_starts, self.sequence_starts + self.token_counts), dim=-1)

    def reorder(self, batch_indices: torch.Tensor | Sequence[int]) -> "DartPageTable":
        """Reorder batch rows like a beam update without touching packed storage."""

        indices = torch.as_tensor(batch_indices, dtype=torch.long, device=self.device)
        if indices.ndim != 1:
            raise ValueError("batch_indices must be one-dimensional")
        if indices.numel() and bool(((indices < 0) | (indices >= self.batch_size)).any().item()):
            raise IndexError("batch_indices contain an invalid batch row")
        return DartPageTable(
            page_ids=self.page_ids.index_select(0, indices),
            sequence_starts=self.sequence_starts,
            token_counts=self.token_counts,
            key_modes=self.key_modes,
            value_modes=self.value_modes,
            valid=self.valid.index_select(0, indices),
            seen_tokens=self.seen_tokens,
            page_size=self.page_size,
        )

    def to(self, device: torch.device | str) -> "DartPageTable":
        """Move descriptor tensors while preserving the logical page IDs."""

        return DartPageTable(
            page_ids=self.page_ids.to(device),
            sequence_starts=self.sequence_starts.to(device),
            token_counts=self.token_counts.to(device),
            key_modes=self.key_modes.to(device),
            value_modes=self.value_modes.to(device),
            valid=self.valid.to(device),
            seen_tokens=self.seen_tokens,
            page_size=self.page_size,
        )


def _cache_device(cache: "DartKVCache") -> torch.device:
    shape = cache.shape
    if shape is None:
        return torch.device("cpu")
    segments = cache.iter_segments()
    if not segments:
        return torch.device("cpu")
    key_segment, _ = segments[0]
    if isinstance(key_segment, torch.Tensor):
        return key_segment.device
    if isinstance(key_segment, QuantizedTensor):
        return key_segment.values.device
    return key_segment.low_values.device


def _segment_tokens(segment: Segment) -> int:
    return segment.shape[-2] if isinstance(segment, torch.Tensor) else segment.original_shape[-2]


def _segment_mode(segment: Segment) -> int:
    if isinstance(segment, torch.Tensor):
        return DENSE_PAGE
    if isinstance(segment, MixedQuantizedKey):
        return MIXED_PAGE
    return QUANTIZED_PAGE


__all__ = [
    "DENSE_PAGE",
    "MIXED_PAGE",
    "QUANTIZED_PAGE",
    "DartPageTable",
]
