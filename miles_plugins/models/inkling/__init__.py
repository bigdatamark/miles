# [PACKAGE DOC] Inkling multimodal model plugin for Megatron. Exposes three lazy entry points that
# forward to the model module only when called (keeping package import cheap): the layer-spec
# builder, the text model provider, and the multimodal model provider. Other modules in the package:
# audio and image preprocessing, a multimodal processor, tower modules, layers, LoRA support, custom
# ops, and a vision encoder.
def get_inkling_spec(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Lazily imports the model module's spec builder and calls it with the
    # given arguments, returning its result.
    from .model import get_inkling_spec as get_spec

    return get_spec(*args, **kwargs)


def inkling_model_provider(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Lazily imports and calls the text model provider; returns its result.
    from .model import inkling_model_provider as provider

    return provider(*args, **kwargs)


def inkling_mm_model_provider(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Lazily imports and calls the multimodal model provider; returns its result.
    from .model import inkling_mm_model_provider as provider

    return provider(*args, **kwargs)


__all__ = ["get_inkling_spec", "inkling_mm_model_provider", "inkling_model_provider"]
