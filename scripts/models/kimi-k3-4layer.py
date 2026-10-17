# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer Kimi-K3 truncation (full
# 896 experts), derived from the kimi-k3 definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling kimi-k3 flag string with 4 layers.
    return load_sibling_model_args(__file__, "kimi-k3", nlayers=4)
