# [MODULE DOC] HELPER: Megatron model-argument definition for a 5-layer truncation of DeepSeek-V3.2
# (smoke-test size), derived from the sibling full definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the flag string of the sibling deepseek-v32 definition with the
    # layer count overridden to 5 (its dense-layer count stays at the sibling's default). No arguments.
    return load_sibling_model_args(__file__, "deepseek-v32", nlayers=5)
