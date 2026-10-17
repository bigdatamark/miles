# [MODULE DOC] HELPER: Megatron model-argument definition for a 5-layer truncation of Qwen3
# 30B-A3B (smoke-test size), derived from the full definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling qwen3-30B-A3B flag string with 5 layers.
    return load_sibling_model_args(__file__, "qwen3-30B-A3B", nlayers=5)
