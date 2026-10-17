# [MODULE DOC] HELPER: Megatron model-argument definition for the 5-layer pruned GLM-5.2 (3 dense
# plus 2 MoE layers), derived from the sibling full definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling glm5.2-744B-A40B flag string with 2 MoE layers. The
    # layout keeps computing layers 0 to 2 and skip layers 3 and 4 so the cross-layer DSA index
    # sharing path is exercised.
    # Override for 5-layer pruned model (first 5 layers: 3 dense + 2 MoE).
    # Keeps at least one computing + one skip layer so the DSA cross-layer index
    # sharing path is exercised (computing layers 0,1,2; skip layers 3,4).
    return load_sibling_model_args(__file__, "glm5.2-744B-A40B", n_moe_layers=2)
