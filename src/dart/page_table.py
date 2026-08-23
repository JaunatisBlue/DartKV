"""Device-resident page descriptors for DartKV cache segments.

Kitty uses a per-batch table of integer page IDs so its Triton kernels can walk
allocated pages without carrying Python objects into the kernel.  DartKV keeps
the storage objects readable, but exposes the same lifecycle as a compact
device tensor table.  The table is the next boundary for a page-run kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

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

    def uniform_quantized_runs(self, cache: "DartKVCache") -> tuple["DartPageRun", ...]:
        """Stack consecutive full uniform pages into page-major run tensors.

        The page table remains the source of logical IDs; the returned run owns
        views with a leading page dimension so a Triton program can walk several
        pages while carrying one online-softmax state.  Dense sink pages,
        mixed keys, and the pending tail intentionally terminate a run until
        their dedicated kernels are added.
        """

        segments = cache.iter_segments()
        if len(segments) != self.page_count or cache.seen_tokens != self.seen_tokens:
            raise ValueError("cache and page_table describe different page lifecycles")
        runs: list[DartPageRun] = []
        start = 0
        while start < len(segments):
            key_segment, value_segment = segments[start]
            if not _is_uniform_full_page(key_segment, value_segment, self.page_size):
                start += 1
                continue
            end = start + 1
            while end < len(segments):
                next_key, next_value = segments[end]
                if not _is_uniform_full_page(next_key, next_value, self.page_size):
                    break
                if not _same_uniform_layout(key_segment, value_segment, next_key, next_value):
                    break
                end += 1
            runs.append(_stack_uniform_run(self, segments[start:end], start))
            start = end
        return tuple(runs)


@dataclass(frozen=True)
class DartPageRun:
    """Page-major packed fields for one consecutive uniform quantized run."""

    first_page_index: int
    page_indices: torch.Tensor
    sequence_starts: torch.Tensor
    token_counts: torch.Tensor
    key_values: torch.Tensor
    key_scale: torch.Tensor
    key_zero_point: torch.Tensor
    value_values: torch.Tensor
    value_scale: torch.Tensor
    value_zero_point: torch.Tensor
    key_bits: int
    value_bits: int
    key_group_size: int
    value_group_size: int
    page_size: int

    @property
    def page_count(self) -> int:
        return self.page_indices.shape[0]

    @property
    def device(self) -> torch.device:
        return self.key_values.device

    def validate(self) -> "DartPageRun":
        if self.page_indices.ndim != 1:
            raise ValueError("page_indices must have shape [run_pages]")
        page_count = self.page_count
        if page_count == 0:
            raise ValueError("a page run cannot be empty")
        if self.first_page_index < 0:
            raise ValueError("first_page_index must be non-negative")
        for name, tensor in (
            ("sequence_starts", self.sequence_starts),
            ("token_counts", self.token_counts),
        ):
            if tensor.ndim != 1 or tensor.shape[0] != page_count:
                raise ValueError(f"{name} must have shape [run_pages]")
        for name, tensor in (
            ("key_values", self.key_values),
            ("key_scale", self.key_scale),
            ("key_zero_point", self.key_zero_point),
            ("value_values", self.value_values),
            ("value_scale", self.value_scale),
            ("value_zero_point", self.value_zero_point),
        ):
            if tensor.ndim < 1 or tensor.shape[0] != page_count:
                raise ValueError(f"{name} must be page-major with first dimension {page_count}")
        if bool((self.token_counts <= 0).any().item()):
            raise ValueError("run token_counts must be positive")
        if page_count > 1:
            if not torch.equal(self.page_indices[1:], self.page_indices[:-1] + 1):
                raise ValueError("page IDs in a run must be consecutive")
            expected = self.sequence_starts[:-1] + self.token_counts[:-1]
            if not torch.equal(self.sequence_starts[1:], expected):
                raise ValueError("run sequence_starts are not contiguous")
        if self.key_bits not in (2, 4, 8) or self.value_bits not in (2, 4, 8):
            raise ValueError("run bit widths must be 2, 4, or 8")
        if self.key_group_size <= 0 or self.value_group_size <= 0 or self.page_size <= 0:
            raise ValueError("run group sizes and page_size must be positive")
        return self

    def to(self, device: torch.device | str) -> "DartPageRun":
        """Move all page-major fields to ``device``."""

        return DartPageRun(
            first_page_index=self.first_page_index,
            page_indices=self.page_indices.to(device),
            sequence_starts=self.sequence_starts.to(device),
            token_counts=self.token_counts.to(device),
            key_values=self.key_values.to(device),
            key_scale=self.key_scale.to(device),
            key_zero_point=self.key_zero_point.to(device),
            value_values=self.value_values.to(device),
            value_scale=self.value_scale.to(device),
            value_zero_point=self.value_zero_point.to(device),
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            key_group_size=self.key_group_size,
            value_group_size=self.value_group_size,
            page_size=self.page_size,
        )


def _is_uniform_full_page(key_segment: Segment, value_segment: Segment, page_size: int) -> bool:
    return (
        isinstance(key_segment, QuantizedTensor)
        and isinstance(value_segment, QuantizedTensor)
        and key_segment.original_shape[-2] == page_size
        and value_segment.original_shape[-2] == page_size
    )


def _same_uniform_layout(
    key_segment: QuantizedTensor,
    value_segment: QuantizedTensor,
    next_key: Segment,
    next_value: Segment,
) -> bool:
    if not isinstance(next_key, QuantizedTensor) or not isinstance(next_value, QuantizedTensor):
        return False
    return (
        key_segment.original_shape == next_key.original_shape
        and value_segment.original_shape == next_value.original_shape
        and key_segment.values.shape == next_key.values.shape
        and value_segment.values.shape == next_value.values.shape
        and key_segment.scale.shape == next_key.scale.shape
        and value_segment.scale.shape == next_value.scale.shape
        and key_segment.bits == next_key.bits
        and value_segment.bits == next_value.bits
        and key_segment.group_size == next_key.group_size
        and value_segment.group_size == next_value.group_size
    )


def _stack_uniform_run(
    table: DartPageTable,
    segments: Sequence[tuple[Segment, Segment]],
    first_page_index: int,
) -> DartPageRun:
    keys = [pair[0] for pair in segments]
    values = [pair[1] for pair in segments]
    if not all(isinstance(segment, QuantizedTensor) for segment in (*keys, *values)):
        raise TypeError("uniform page runs require QuantizedTensor key/value segments")
    key_segments = [segment for segment in keys if isinstance(segment, QuantizedTensor)]
    value_segments = [segment for segment in values if isinstance(segment, QuantizedTensor)]
    first_key = key_segments[0]
    first_value = value_segments[0]
    storage_device = first_key.values.device
    page_indices = table.page_ids[0, first_page_index:first_page_index + len(segments)].to(storage_device)
    run = DartPageRun(
        first_page_index=first_page_index,
        page_indices=page_indices,
        sequence_starts=table.sequence_starts[first_page_index:first_page_index + len(segments)].to(storage_device),
        token_counts=table.token_counts[first_page_index:first_page_index + len(segments)].to(storage_device),
        key_values=torch.stack([segment.values for segment in key_segments], dim=0),
        key_scale=torch.stack([segment.scale for segment in key_segments], dim=0),
        key_zero_point=torch.stack([segment.zero_point for segment in key_segments], dim=0),
        value_values=torch.stack([segment.values for segment in value_segments], dim=0),
        value_scale=torch.stack([segment.scale for segment in value_segments], dim=0),
        value_zero_point=torch.stack([segment.zero_point for segment in value_segments], dim=0),
        key_bits=first_key.bits,
        value_bits=first_value.bits,
        key_group_size=first_key.group_size,
        value_group_size=first_value.group_size,
        page_size=table.page_size,
    )
    return run.validate()


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
    "DartPageRun",
]
