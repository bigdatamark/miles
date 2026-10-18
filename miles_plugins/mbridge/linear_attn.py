# [MODULE DOC]
# Mixin for bridges of models with linear attention layers: hides the internal weight layout version
# marker parameters from the mapping used for loading Hugging Face weights.
class LinearAttentionBridgeMixin:
    # [CLASS DOC] Mixin; must precede the real bridge class in the base class list so its override
    # wraps the bridge's own mapping method.
    """Keeps the linear-attention weight_layout_version marker out of HF weight loading."""

    def _weight_name_mapping_mcore_local_to_global(self, model):
        # [FUNCTION DOC] Take the parent's mapping (from local Megatron parameter names of the model to
        # global names) and return a new mapping without the entries whose global name ends with the
        # layout version marker. The original mapping is not modified.
        mapping = super()._weight_name_mapping_mcore_local_to_global(model)
        return {k: v for k, v in mapping.items() if not v.endswith("weight_layout_version")}
