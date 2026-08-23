"""DartKV: a small, readable PyTorch KV-cache reference implementation."""

from .cache import DartKVCache, DartKVCacheConfig, DartPageMetadata, DartSegmentLayout
from .attention import dense_attention, streamed_dart_attention
from .fused_attention import fused_dart_attention
from .page_table import DENSE_PAGE, MIXED_PAGE, QUANTIZED_PAGE, DartPageTable
from .mixed import MixedQuantizedKey, quantize_key_mixed, select_key_channels
from .quantization import QuantizedTensor, dequantize, quantize, quantize_axis
from .triton_ops import TRITON_AVAILABLE, triton_available, triton_dequantize

__all__ = [
    "DartKVCache",
    "DartKVCacheConfig",
    "DartPageMetadata",
    "DartSegmentLayout",
    "dense_attention",
    "fused_dart_attention",
    "DENSE_PAGE",
    "MIXED_PAGE",
    "QUANTIZED_PAGE",
    "DartPageTable",
    "QuantizedTensor",
    "MixedQuantizedKey",
    "dequantize",
    "quantize",
    "quantize_axis",
    "quantize_key_mixed",
    "select_key_channels",
    "streamed_dart_attention",
    "TRITON_AVAILABLE",
    "triton_available",
    "triton_dequantize",
]

__version__ = "0.1.0"
