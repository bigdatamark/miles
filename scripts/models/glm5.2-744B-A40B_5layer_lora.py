# [MODULE DOC] HELPER: Megatron model-argument definition for the 5-layer pruned GLM-5.2 used for
# LoRA runs (3 dense plus 2 MoE layers), derived from the GLM-5.2 LoRA definition. Import has no
# side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling glm5.2-744B-A40B_lora flag string with 2 MoE layers.
    # Override for the 5-layer pruned model (first 5 layers: 3 dense + 2 MoE). Keeps at least
    # one computing + one skip layer so the DSA cross-layer index sharing path is exercised.
    return load_sibling_model_args(__file__, "glm5.2-744B-A40B_lora", n_moe_layers=2)
