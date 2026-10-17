# [MODULE DOC] HELPER: Megatron model-argument definition for Qwen3.8 27B, which ships the same
# config as Qwen3.5 27B and so reuses that definition unchanged. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # [FUNCTION DOC] HELPER: return the sibling qwen3.5-27B flag string unchanged.
    """Qwen3.8-27B ships the same config.json as Qwen3.5-27B, down to every shape and gate."""
    return load_sibling_model_args(__file__, "qwen3.5-27B")
