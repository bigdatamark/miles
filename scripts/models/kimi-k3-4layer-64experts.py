# [MODULE DOC] HELPER: Megatron model-argument definition for a 4-layer Kimi-K3 truncation with
# only 64 experts (smaller than the full 896), derived from the kimi-k3 definition. Import has no
# side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling kimi-k3 flag string with 4 layers and 64 experts.
    return load_sibling_model_args(__file__, "kimi-k3", nlayers=4, num_experts=64)
