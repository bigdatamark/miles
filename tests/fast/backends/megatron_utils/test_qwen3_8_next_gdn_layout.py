# [MODULE DOC]
# Tests that Qwen3.8-Next treats its replicated gated delta net (GDN) linear-attention weights
# as already being in the HuggingFace layout: the raw exporter must pass them through without the
# group-major permutation that Qwen3.5 applies, and the mbridge bridge for Qwen3.8-Next must
# declare no group-major parameter names while Qwen3.5 declares the two GDN names below.
"""Qwen3.8-Next runs the replicated GDN in HF's layout, so its weight conversion must not take the
head-sharded layer's group-major permutation the way Qwen3.5's does."""

import sys

import pytest

GDN_NAMES = ("self_attention.linear_attn.in_proj_qkv.weight", "self_attention.linear_attn.conv1d.weight")


@pytest.mark.parametrize("name", GDN_NAMES)
def test_the_export_passes_the_replicated_gdn_weights_through(name, monkeypatch):
    # [FUNCTION DOC] Parametrized over the two GDN parameter names (input qkv projection weight
    # and convolution weight). Skips if torch is missing. Removes the Qwen3.5 exporter module from
    # the module table (restored by monkeypatch) and re-imports it, then replaces its group-major
    # to flat conversion function with one that fails the test if called. Converts a 4-by-2 zero
    # tensor named as layer 3 of the decoder with the Qwen3.8-Next exporter (args none). Asserts
    # the result is exactly one pair: the HuggingFace language-model layer-3 name (the name with
    # the "self_attention." prefix dropped) and the very same tensor.
    torch = pytest.importorskip("torch")
    monkeypatch.delitem(sys.modules, "miles.backends.megatron_utils.megatron_to_hf.qwen3_5", raising=False)
    from miles.backends.megatron_utils.megatron_to_hf import qwen3_5 as qwen3_5_export
    from miles.backends.megatron_utils.megatron_to_hf.qwen3_8_next import convert_qwen3_8_next_to_hf

    def _fail(*_args, **_kwargs):
        # [FUNCTION DOC] HELPER: Accepts any arguments and always raises an assertion error saying
        # Qwen3.8-Next must not take Qwen3.5's group-major permutation.
        raise AssertionError("Qwen3.8-Next must not take Qwen3.5's group-major permutation")

    monkeypatch.setattr(qwen3_5_export, "qkv_group_major_to_flat", _fail)
    param = torch.zeros(4, 2)
    converted = convert_qwen3_8_next_to_hf(None, f"module.module.decoder.layers.3.{name}", param)

    hf_rest = name[len("self_attention.") :]
    assert converted == [(f"model.language_model.layers.3.{hf_rest}", param)]


def test_the_bridge_opts_out_of_the_group_major_names():
    # [FUNCTION DOC] Skips if mbridge is missing. Asserts the Qwen3.5 bridge class's group-major
    # GDN name tuple equals the two names above, and the Qwen3.8-Next bridge's is empty.
    pytest.importorskip("mbridge")
    from miles_plugins.mbridge.qwen3_5 import Qwen3_5Bridge
    from miles_plugins.mbridge.qwen3_8_next import Qwen38NextBridge

    assert Qwen3_5Bridge._GDN_GROUP_MAJOR == GDN_NAMES
    assert Qwen38NextBridge._GDN_GROUP_MAJOR == ()
