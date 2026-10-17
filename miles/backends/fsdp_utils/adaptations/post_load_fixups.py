# [MODULE DOC] Registry of fixups that re-assert checkpoint weights on a model after it has
# been loaded (for cases where the HF loading path overwrote some parameters), and the dispatcher
# that runs the ones matching a config. The registry is a module list filled by register calls.
"""Generic post-load fixup registry and dispatcher for FSDP model adaptations."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class PostLoadFixup:
    # [CLASS DOC] Mutable record of one fixup: a name, a predicate on the HF config, and an apply
    # action taking the model and the checkpoint path and returning the integer count of
    # parameters it re-asserted (zero meaning nothing was changed).
    name: str
    applies_to: Callable  # (hf_config) -> bool
    apply: Callable  # (model, ckpt_path) -> int (count of params re-asserted)


_FIXUPS: list[PostLoadFixup] = []


def register_post_load_fixup(fixup: PostLoadFixup) -> None:
    # [FUNCTION DOC] Append a fixup to the module registry (no de-duplication).
    _FIXUPS.append(fixup)


def apply_post_load_fixups(model, hf_config, ckpt_path) -> list[str]:
    # [FUNCTION DOC] For each registered fixup in order whose predicate is true for the config,
    # call its action with the model and checkpoint path (mutating the model); collect the names of
    # fixups whose action returned a truthy count. Returns the list of names (empty if none).
    """Run every registered fixup whose arch-predicate matches; return the names that fired."""
    fired = []
    for f in _FIXUPS:
        if f.applies_to(hf_config) and f.apply(model, ckpt_path):
            fired.append(f.name)
    return fired
