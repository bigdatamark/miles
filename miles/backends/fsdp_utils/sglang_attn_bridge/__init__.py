# [PACKAGE DOC]
# Bridge that lets the FSDP trainer compute attention with the same Triton kernel the SGLang
# rollout engine uses, so training forward passes are bitwise aligned with inference (true
# on-policy). Re-exports the function that registers and activates the custom attention backend;
# the implementation lives in the triton patch module.
"""SGLang attention bridge for FSDP true on-policy path."""

from .hf_sglang_triton_patch import apply_sglang_triton_attention_patch

__all__ = ["apply_sglang_triton_attention_patch"]
