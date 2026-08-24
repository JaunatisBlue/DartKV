"""Optional integrations with external model libraries."""

from .huggingface import DartHFCache
from .qwen3 import attach_dart_fused_attention

__all__ = ["DartHFCache", "attach_dart_fused_attention"]
