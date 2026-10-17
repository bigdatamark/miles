# [MODULE DOC]
# Qwen3 dense-model numerics for exact true-on-policy matching between the FSDP trainer and the
# rollout engine: a final-norm variant with the engine's rounding order, a rule choosing which
# weights are synchronized in float32, and a patch that swaps the final norm's class in place.
"""Qwen3 dense model math required by the formal true-on-policy contract."""

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


class Qwen3FinalRMSNorm(Qwen3RMSNorm):
    # [CLASS DOC]
    # RMS normalization for the model's last norm layer, subclassing the library norm (same
    # weight and epsilon attributes) but computing the result with a specific precision order.
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC]
        # Casts input to float32, divides by the root of the mean of squares over the last axis plus
        # the epsilon (computed in float32), then casts the normalized values to bfloat16 and
        # multiplies by the weight also cast to bfloat16. Returns a bfloat16 tensor of the same
        # shape. (The stock norm multiplies the weight before the final cast back.)
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(torch.bfloat16) * hidden_states.to(torch.bfloat16)


_FP32_SYNC_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
)


def resolve_qwen3_dense_sync_dtype(name: str, checkpoint_dtype: torch.dtype) -> torch.dtype:
    # [FUNCTION DOC]
    # Given a parameter name and the checkpoint dtype, returns float32 for the token embedding
    # weight and for the per-layer input norm, post-attention norm and attention query/key norm
    # weights (matched by name suffix), and the checkpoint dtype for everything else.
    if name == "model.embed_tokens.weight" or name.endswith(_FP32_SYNC_SUFFIXES):
        return torch.float32
    return checkpoint_dtype


def apply_qwen3_dense_true_on_policy_patch(model) -> bool:
    # [FUNCTION DOC]
    # Locates the base model (the model's inner "model" attribute if present, else the model itself)
    # and its final norm. Returns False if the norm is already the patched class. Raises a type
    # error if the final norm is not a stock Qwen3 RMS norm. Otherwise reassigns the norm object's
    # class to the patched class in place (keeping its weights) and returns True.
    base_model = getattr(model, "model", model)
    final_norm = getattr(base_model, "norm", None)
    if isinstance(final_norm, Qwen3FinalRMSNorm):
        return False
    if not isinstance(final_norm, Qwen3RMSNorm):
        raise TypeError("Expected a Qwen3 model with a final Qwen3RMSNorm")
    final_norm.__class__ = Qwen3FinalRMSNorm
    return True
