# [MODULE DOC] HELPER: Megatron model-argument definition for Kimi-K2.5, derived from the
# Kimi-K2-Thinking definition with a different YaRN fast-beta. Import has no side effects.
from model_args_utils import load_sibling_model_args


def model_args(nlayers: int = 61, first_k_dense_replace: int = 1) -> str:
    # [FUNCTION DOC] HELPER: return the sibling kimi-k2-thinking flag string for the given total
    # layers (default 61) and leading dense layers (default 1), with the YaRN fast-beta overridden
    # to 32 (the sibling default is 1).
    return load_sibling_model_args(
        __file__,
        "kimi-k2-thinking",
        nlayers=nlayers,
        first_k_dense_replace=first_k_dense_replace,
        beta_fast=32,
    )
