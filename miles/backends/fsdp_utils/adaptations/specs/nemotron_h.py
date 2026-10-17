# [MODULE DOC] Registers, at import, the three hooks for the NemotronH (Mamba2 hybrid)
# architecture: a config-time patch repairing a layer-pattern parser corrupted by sglang, a
# post-load packing patch (resets per-document state, matched to sglang behavior) and a post-load
# fixup reloading mixer parameters that transformers re-initialized after loading. The model
# specific implementations live in the sibling models package and are imported lazily inside the
# wrappers defined here.
"""NemotronH (Mamba2 hybrid) adaptations: a config-time repair of sglang's stale pattern-parsing
monkeypatch, a post-load packed-doc reset, and a clobber-reload that re-asserts the checkpoint over
mixer params transformers' _init_weights re-inits after loading."""

from ..class_patches import ModelPatchHook, register_model_patch
from ..packing.registry import PackingPatch, register_packing_patch
from ..post_load_fixups import PostLoadFixup, register_post_load_fixup


def _is_nemotron_h(hf_config) -> bool:
    # [FUNCTION DOC] HELPER: True when the config's model type (empty text if missing) equals
    # exactly nemotron_h.
    return str(getattr(hf_config, "model_type", "") or "") == "nemotron_h"


def _repair_pattern_to_list(hf_config, args) -> None:
    # [FUNCTION DOC] HELPER: Config-time hook (both arguments unused) that undoes a process-wide
    # monkeypatch of the NemotronH config class's static layer-pattern parser. It parses a probe
    # pattern of a Mamba character followed by a dash: if the parser raises a key error (older
    # transformers without dash support) nothing is done; if the result is exactly one Mamba entry
    # (the dash was silently dropped, so the bug is present) the parser is replaced on the class by
    # a correct static one mapping M to mamba, E to moe, star to attention and dash to mlp (unknown
    # characters raise a key error); any other probe result means the parser is healthy and the
    # class is left alone. Side effect: mutates the transformers config class.
    """``import sglang`` monkeypatches ``NemotronHConfig._pattern_to_list`` to drop unmapped chars,
    silently deleting every ``-`` (MLP) layer, so any transformers-side NemotronH constructed afterwards
    is mis-shaped (wrong depth, shifted attention layers). Probe for that and re-assert the full native
    mapping; a healthy implementation (keeps the MLP entry) is left untouched."""
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    try:
        probe = NemotronHConfig._pattern_to_list("M-")
    except KeyError:
        return  # pre-'-' transformers: no silent corruption to repair; construction will fail loudly
    if probe != ["mamba"]:
        return

    @staticmethod
    def _pattern_to_list(pattern: str) -> list:
        # [FUNCTION DOC] HELPER: Replacement parser: convert each character of the pattern text to
        # its layer kind name via the fixed four-entry mapping and return the list in order.
        pattern_mapping = {"M": "mamba", "E": "moe", "*": "attention", "-": "mlp"}
        return [pattern_mapping[char] for char in pattern]

    NemotronHConfig._pattern_to_list = _pattern_to_list


def _packing_applies(hf_config) -> bool:
    # [FUNCTION DOC] HELPER: Predicate for the packing patch: true when the lower-cased model type
    # text (empty if missing) CONTAINS nemotron_h (looser than the exact match used by the other
    # two hooks of this module).
    return "nemotron_h" in str(getattr(hf_config, "model_type", "") or "").lower()


def _packing_apply(model):
    # [FUNCTION DOC] HELPER: Post-load packing action: lazily import and call the model package's
    # function that patches a built NemotronH model so packed documents reset state per document in
    # the same way sglang does; returns that function's result (used as the applied flag).
    from ...models.nemotron_h import apply_nemotron_h_sglang_match_patch

    return apply_nemotron_h_sglang_match_patch(model)


def _post_load_apply(model, ckpt_path):
    # [FUNCTION DOC] HELPER: Post-load fixup action: lazily import and call the model package's
    # function that re-reads from the checkpoint folder the mixer parameters transformers
    # re-initialized after loading; returns its result (the count of parameters reloaded).
    from ...models.nemotron_h import reload_nemotron_h_clobbered_weights

    return reload_nemotron_h_clobbered_weights(model, ckpt_path)


register_model_patch(ModelPatchHook("nemotron_h_pattern_repair", _is_nemotron_h, _repair_pattern_to_list))
register_packing_patch(PackingPatch("nemotron_h_packing", _packing_applies, "post_load", _packing_apply))
register_post_load_fixup(PostLoadFixup("nemotron_h_clobber_reload", _is_nemotron_h, _post_load_apply))
