# [MODULE DOC]
# Helpers for exporting megatron models through the megatron bridge: a temporary model patch and
# mapping of the dsa attention backend argument onto the provider. The megatron import is optional
# (the unwrap function is none when megatron is missing, so the patch then fails when used).
from contextlib import contextmanager

try:
    from megatron.core.utils import unwrap_model
except ImportError:
    unwrap_model = None


@contextmanager
def patch_megatron_model(model):
    # [FUNCTION DOC] Context manager preparing a list of model chunks for bridge export. Unwraps the
    # first chunk and ensures its config has the shared embeddings and output weights attribute
    # (copied from the model if missing; remembered so it is deleted again on exit). Also calls the
    # float32 expert bias restore method on every submodule of every chunk that defines it (mixed
    # precision wrappers cast the bias to bf16). The restoration is permanent (not undone). Yields
    # nothing. Raises a type error if megatron is not installed.
    unwrapped_model = unwrap_model(model)[0]
    model_config = unwrapped_model.config
    attribute_was_added = False
    if not hasattr(model_config, "share_embeddings_and_output_weights"):
        model_config.share_embeddings_and_output_weights = unwrapped_model.share_embeddings_and_output_weights
        attribute_was_added = True

    # Float16Module casts buffers to bf16, but expert_bias must stay fp32.
    # Restore before bridge export reads the values.
    for m in model:
        for module in m.modules():
            if hasattr(module, "_maintain_float32_expert_bias"):
                module._maintain_float32_expert_bias()

    try:
        yield
    finally:
        if attribute_was_added:
            delattr(model_config, "share_embeddings_and_output_weights")


def apply_dsa_backend_args(provider, args) -> None:
    # [FUNCTION DOC] Reads the dsa attention backend from args (default megatron). If the provider
    # has an attention backend attribute (bridge style), sets it to that value. Otherwise, if it has
    # a kernel backend attribute (main style), sets it to the explicit kernel backend argument if
    # truthy, else tilelang for tilelang and none for megatron (a backend outside these two raises a
    # key error). Does nothing if the provider has neither. Mutates the provider.
    """Map --dsa-attention-backend onto the provider's dsa_attention_backend (bridge) or dsa_kernel_backend (main)."""
    backend = getattr(args, "dsa_attention_backend", "megatron")
    if hasattr(provider, "dsa_attention_backend"):
        provider.dsa_attention_backend = backend
    elif hasattr(provider, "dsa_kernel_backend"):
        explicit = getattr(args, "dsa_kernel_backend", None)
        provider.dsa_kernel_backend = explicit or {"tilelang": "tilelang", "megatron": "none"}[backend]
