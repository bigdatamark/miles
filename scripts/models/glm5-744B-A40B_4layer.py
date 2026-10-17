# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer pruned GLM-5 (3 dense plus
# 1 MoE layer), derived from the sibling full definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling glm5-744B-A40B flag string with 1 MoE layer.
    # Override for 4-layer pruned model (first 4 layers: 3 dense + 1 MoE)
    return load_sibling_model_args(__file__, "glm5-744B-A40B", n_moe_layers=1)
