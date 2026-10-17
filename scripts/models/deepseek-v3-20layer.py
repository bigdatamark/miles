# [MODULE DOC] HELPER: Megatron model-argument definition for a 20-layer truncation of DeepSeek-V3
# (debug / smoke-test size). Contains no flags of its own; it is derived from the sibling full
# model definition. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the flag string of the sibling deepseek-v3 definition loaded by
    # path with the layer count overridden to 20 (the sibling recomputes the MoE layer frequency
    # for that count). Takes no arguments.
    return load_sibling_model_args(__file__, "deepseek-v3", nlayers=20)
