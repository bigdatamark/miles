# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer truncation of
# DeepSeek-V4.1 (smoke-test size), derived from the sibling full definition. Import has no side
# effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the flag string of the sibling deepseek-v4.1 definition with 4
    # layers and a per-layer compression ratio list of 0, 2, 1, 1. No arguments.
    return load_sibling_model_args(__file__, "deepseek-v4.1", nlayers=4, compress_ratios="[0,2,1,1]")
