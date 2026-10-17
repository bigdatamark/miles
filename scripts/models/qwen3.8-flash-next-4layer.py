# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer truncation of Qwen3.8
# Flash-Next (smoke-test size), derived from the full definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling qwen3.8-flash-next flag string with 4 layers.
    return load_sibling_model_args(__file__, "qwen3.8-flash-next", nlayers=4)
