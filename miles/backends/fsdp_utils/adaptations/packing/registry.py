# [MODULE DOC] Registry of per-architecture packing patches, plus the dispatcher the actor
# calls. A patch has a name, a predicate on the HF config, a lifetime (config: applied before the
# model is built and takes no argument; post_load: applied to the built model) and an apply
# action. The registry is a module list filled by register calls and queried in registration order.
"""Registry that unifies packed-sequence layout handling across FSDP-backend architectures.

Stateful archs (GatedDeltaNet, Mamba2 hybrid, ...) each reset per-document state under THD packing but
hook different methods/kernels. Each registers a ``PackingPatch`` and the actor dispatches once per
lifetime; archs that need nothing (e.g. ``glm4_moe_lite``) just don't register. All patches share the
boundary derivation in ``boundaries.py``. Two lifetimes:
  * ``"config"``    — patch the transformers classes before model construction; ``apply()`` takes no model.
  * ``"post_load"`` — patch the instantiated model after ``from_pretrained``; ``apply(model)`` takes it.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class PackingPatch:
    # [CLASS DOC] Mutable record of one packing patch. Fields: name (reported when applied);
    # applies_to (predicate taking the HF config); lifetime (the text config or post_load);
    # apply (callable taking nothing for config lifetime or the model for post_load, expected to
    # return a truthy value when it applied something).
    name: str
    applies_to: Callable  # (hf_config) -> bool
    lifetime: str  # "config" | "post_load"
    apply: Callable  # config: apply() ; post_load: apply(model) ; both return truthy when applied


_PACKING_PATCHES: list[PackingPatch] = []


def register_packing_patch(patch: PackingPatch) -> None:
    # [FUNCTION DOC] Append a patch to the module registry (no de-duplication or validation of
    # the lifetime text).
    _PACKING_PATCHES.append(patch)


def get_packing_patches(hf_config, lifetime: str) -> list[PackingPatch]:
    # [FUNCTION DOC] Return, in registration order, the registered patches whose lifetime text
    # equals the given one and whose predicate is true for the config. Does not apply anything.
    return [p for p in _PACKING_PATCHES if p.lifetime == lifetime and p.applies_to(hf_config)]


def apply_packing(target, hf_config, lifetime: str) -> list[str]:
    # [FUNCTION DOC] Apply every matching patch and return the list of names that fired. For each
    # patch from the lookup: when the lifetime is exactly post_load, call its action with the
    # target model; for any other lifetime value call it with no argument (the target is ignored).
    # A patch counts as fired if its action returned a truthy value OR returned none; only an
    # explicit false-like non-none result (such as false or zero) means it did nothing.
    # KNOWN ISSUE: the field comment says patches return truthy when applied, yet returning none is
    # also counted as applied, so a patch that forgets to return anything is reported as fired.
    """Apply every registered packing patch matching this config + lifetime (idempotent). ``target`` is the
    model for ``post_load``, ignored for ``config``. Returns the names that fired (empty when no arch matches)."""
    fired = []
    for p in get_packing_patches(hf_config, lifetime):
        applied = p.apply(target) if lifetime == "post_load" else p.apply()
        if applied or applied is None:
            fired.append(p.name)
    return fired
