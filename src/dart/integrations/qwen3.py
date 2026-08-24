"""Qwen3 attention adapter that consumes Dart's packed cache directly.

The regular ``DartHFCache`` path remains compatible with every Transformers
attention backend and materializes dense K/V tensors.  This adapter keeps that
prefill behavior, then routes one-token decode calls through
``fused_dart_attention`` so the packed pages are the source of truth.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ..fused_attention import fused_dart_attention
from .huggingface import DartHFCache


def _qwen3_modules():
    from transformers.models.qwen3 import modeling_qwen3 as modules

    return modules


class DartQwen3Attention:  # populated dynamically to keep imports optional
    """Placeholder replaced by :func:`build_dart_qwen3_attention_class`."""


def build_dart_qwen3_attention_class():
    """Build the Qwen3 attention subclass against the installed Transformers."""

    modules = _qwen3_modules()
    BaseAttention = modules.Qwen3Attention

    class _DartQwen3Attention(BaseAttention):
        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value=None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ):
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            query_states, key_states = modules.apply_rotary_pos_emb(query_states, key_states, cos, sin)

            use_dart_decode = isinstance(past_key_value, DartHFCache)
            if past_key_value is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                if use_dart_decode:
                    cache_kwargs["dart_fused_attention"] = True
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

            output_attentions = bool(kwargs.get("output_attentions", False))
            if (
                use_dart_decode
                and not past_key_value.last_update_was_prefill(self.layer_idx)
                and query_states.shape[-2] == 1
                and not output_attentions
            ):
                attn_output = fused_dart_attention(
                    query_states,
                    past_key_value.layer_for_attention(self.layer_idx),
                    scale=self.scaling,
                    fallback=True,
                ).transpose(1, 2).contiguous()
                attn_weights = None
            else:
                attention_interface = modules.eager_attention_forward
                if self.config._attn_implementation != "eager":
                    attention_interface = modules.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    sliding_window=self.sliding_window,
                    **kwargs,
                )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

    _DartQwen3Attention.__name__ = "DartQwen3Attention"
    return _DartQwen3Attention


def attach_dart_fused_attention(model: nn.Module) -> nn.Module:
    """Replace every Qwen3 self-attention module with the Dart adapter."""

    attention_class = build_dart_qwen3_attention_class()
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("attach_dart_fused_attention expects a Qwen3 causal-LM model")
    for layer_idx, layer in enumerate(layers):
        original = layer.self_attn
        replacement = attention_class(model.config, layer_idx)
        replacement.load_state_dict(original.state_dict())
        reference_parameter = next(original.parameters())
        replacement.to(device=reference_parameter.device, dtype=reference_parameter.dtype)
        replacement.train(original.training)
        layer.self_attn = replacement
    return model


__all__ = ["attach_dart_fused_attention", "build_dart_qwen3_attention_class"]
