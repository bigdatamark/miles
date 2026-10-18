# [PACKAGE DOC] Kimi K3 model plugin for Megatron: a hybrid of multi-latent attention layers and Kimi
# delta (KDA) linear-attention layers, with "attention residual" block aggregation across layers
# (learned softmax mixing over earlier block outputs), pipeline-boundary packing of that state, native
# LoRA, and custom ops. Modules: layers (attention layers and the transformer layer), lora, model
# (spec and provider), ops (norm and residual aggregation), pipeline (stage-boundary packing). The
# package lazily exposes the spec builder so importing it stays cheap.
def get_kimi_k3_spec(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Lazily imports the model module's spec builder and calls it with the given
    # arguments, returning its result.
    from .model import get_kimi_k3_spec as get_spec

    return get_spec(*args, **kwargs)


__all__ = ["get_kimi_k3_spec"]
