# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer truncation of GLM-5.3
# Flash (1 dense plus 3 MoE layers), derived from the sibling full definition. Import has no side
# effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling glm5.3-flash flag string with 4 layers and a
    # leading dense layer count of 1.
    return load_sibling_model_args(__file__, "glm5.3-flash", nlayers=4, first_k_dense_replace=1)
