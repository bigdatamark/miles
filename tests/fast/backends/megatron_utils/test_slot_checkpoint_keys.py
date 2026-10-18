# [MODULE DOC]
# Test for the LoRA slot checkpoint helper that rewrites per-slot storage keys (a numbered
# adapters entry) into a slot-agnostic canonical key, so a checkpoint saved from one adapter slot
# can be loaded into any other slot.
"""A slot checkpoint's storage keys are slot-agnostic, so any free slot can reload it."""

from types import SimpleNamespace

from miles.backends.megatron_utils.lora.checkpoint import _canonicalize_slot_keys


def test_saved_and_loading_slots_meet_at_the_canonical_key():
    # [FUNCTION DOC] Two fake weight entries have keys for the same layer's input linear weight
    # under adapter slot 2 (saved) and slot 0 (loading). Each is canonicalized in place with its
    # own slot number inside a state mapping under the weights entry. Asserts both keys are then
    # equal to the key with the "adapters.<n>" part replaced by the singular "adapter".
    saved = SimpleNamespace(key="decoder.layers.3.linear_qkv.adapters.2.linear_in.weight")
    loading = SimpleNamespace(key="decoder.layers.3.linear_qkv.adapters.0.linear_in.weight")
    _canonicalize_slot_keys({"weights": {"w": saved}}, 2)
    _canonicalize_slot_keys({"weights": {"w": loading}}, 0)
    assert saved.key == loading.key == "decoder.layers.3.linear_qkv.adapter.linear_in.weight"
