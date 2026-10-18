# [MODULE DOC]
# Test (skipped when the mbridge package is missing) of the Kimi K3 mbridge weight conversion: the
# kda recurrent state parameters must stay float32 even when the bridge dtype is bfloat16, while
# ordinary projection weights follow the bridge dtype.
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("mbridge")
from miles_plugins.mbridge.kimi_k3 import KimiK3Bridge  # noqa: E402


@pytest.mark.parametrize("name", ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight", "A_log", "dt_bias"))
def test_kda_state_stays_fp32_against_the_bridge_dtype(name):
    # [FUNCTION DOC] Parametrized over the q, k, v convolution weights, the A log and the dt bias. Builds
    # a bridge object without running its constructor, with bfloat16 dtype and a config of one kda head,
    # and converts a one-element float32 weight to mcore format under the linear attention name. Asserts
    # the result is float32 and exactly equal to the input. Then sets no vocab-size divisor and converts
    # a q projection weight under the linear attention name, asserting it comes out bfloat16.
    """Losing the fp32 override silently downcasts the KDA state during conversion."""
    bridge = object.__new__(KimiK3Bridge)
    bridge.dtype = torch.bfloat16
    bridge.config = SimpleNamespace(kimi_linear_num_heads=1)
    weight = torch.tensor([0.1234567], dtype=torch.float32)

    converted = bridge._weight_to_mcore_format(f"decoder.layers.0.self_attention.linear_attn.{name}", [weight])

    assert converted.dtype == torch.float32
    torch.testing.assert_close(converted, weight, rtol=0, atol=0)

    bridge.make_vocab_size_divisible_by = None
    assert (
        bridge._weight_to_mcore_format("decoder.layers.0.self_attention.linear_attn.q_proj.weight", [weight]).dtype
        == torch.bfloat16
    )
