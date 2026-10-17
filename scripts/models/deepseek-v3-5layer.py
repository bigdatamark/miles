# [MODULE DOC] HELPER: Megatron model-argument definition for a 5-layer truncation of DeepSeek-V3
# (smoke-test size), derived from the sibling full model definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the flag string of the sibling deepseek-v3 definition with the
    # layer count overridden to 5. With the sibling's first-dense-layer count of 3 this gives three
    # dense layers and two MoE layers. Takes no arguments.
    return load_sibling_model_args(__file__, "deepseek-v3", nlayers=5)
