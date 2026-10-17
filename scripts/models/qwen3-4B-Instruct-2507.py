# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3 4B Instruct 2507, identical to
# Qwen3 4B except for a rotary base of 5,000,000; derived from the qwen3-4B definition. Import has
# no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling qwen3-4B flag string with the rotary base set to
    # 5000000 (passed as an integer; the sibling embeds it into the flag text).
    return load_sibling_model_args(__file__, "qwen3-4B", rotary_base=5000000)
