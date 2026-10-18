# [MODULE DOC]
# Test for the Qwen2 Megatron-to-HuggingFace weight converter: it must accept the explicit layer
# norm parameter names used by the true-on-policy model variant.
from argparse import Namespace

import torch


def test_qwen2_converter_accepts_explicit_true_on_policy_layernorm_names():
    # [FUNCTION DOC] Uses a small model description (hidden size 4, kv channels 2, 2 attention
    # heads, 1 query group) and a ones vector of length 4. Asserts that converting the layer-0
    # input layer norm weight gives the single pair (HuggingFace input layer norm name, same
    # tensor), and converting the layer-0 pre-MLP layer norm weight gives the single pair
    # (HuggingFace post-attention layer norm name, same tensor).
    from miles.backends.megatron_utils.megatron_to_hf.qwen2 import convert_qwen2_to_hf

    args = Namespace(
        hidden_size=4,
        kv_channels=2,
        num_attention_heads=2,
        num_query_groups=1,
    )
    param = torch.ones(4)

    assert convert_qwen2_to_hf(
        args,
        "module.module.decoder.layers.0.input_layernorm.weight",
        param,
    ) == [("model.layers.0.input_layernorm.weight", param)]
    assert convert_qwen2_to_hf(
        args,
        "module.module.decoder.layers.0.pre_mlp_layernorm.weight",
        param,
    ) == [("model.layers.0.post_attention_layernorm.weight", param)]
