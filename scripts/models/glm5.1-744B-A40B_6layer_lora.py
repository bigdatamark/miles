# [MODULE DOC] HELPER: Megatron model-argument definition for the 6-layer pruned GLM-5.1 toy model
# used for LoRA runs (3 dense plus 3 MoE layers), derived from the GLM-5.1 LoRA definition. Import
# has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling glm5.1-744B-A40B_lora flag string with 3 MoE layers.
    # Override for the 6-layer pruned GLM-5.1 toy (jybsuper/GLM-5.1-6layer):
    # first 6 layers = 3 dense + 3 MoE.
    return load_sibling_model_args(__file__, "glm5.1-744B-A40B_lora", n_moe_layers=3)
