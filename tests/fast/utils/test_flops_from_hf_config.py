# [MODULE DOC]
# Tests that the adapter turning a HuggingFace-style model config into FLOPs-calculation arguments
# yields the same forward FLOPs as hand-written Megatron-style arguments. Both sides are compared
# via the forward-FLOPs calculator over two fixed sequence lengths (1024 and 3072). Scenarios:
# dense grouped-query models, fallbacks for head dimension and key/value heads, various mixture-of-
# experts layouts (every layer, Mixtral/gpt-oss style, dense prefix, sparse step, shared experts),
# multi-head latent attention, multimodal configs (text tower used), and error handling.
from __future__ import annotations

from types import SimpleNamespace

import pytest

from miles.utils.flops_utils import calculate_fwd_flops, flops_args_from_hf_config

SEQLENS = [1024, 3072]


def megatron_args(**overrides):
    # [FUNCTION DOC] HELPER: Build a simple namespace of Megatron-style model arguments for a dense
    # model (hidden 2048, 16 heads, 4 query groups, vocab 152064, 8 layers, FFN 8192, kv channels
    # 128, no experts, router top-k one, no latent attention), with any keyword overrides taking
    # precedence over those defaults.
    base = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=4,
        vocab_size=152064,
        num_layers=8,
        ffn_hidden_size=8192,
        kv_channels=128,
        num_experts=None,
        moe_ffn_hidden_size=None,
        moe_router_topk=1,
        moe_shared_expert_intermediate_size=None,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_head_dim=0,
        qk_pos_emb_head_dim=0,
        v_head_dim=0,
    )
    return SimpleNamespace(**(base | overrides))


def hf_config(**overrides):
    # [FUNCTION DOC] HELPER: Build a simple namespace mimicking a HuggingFace dense config that
    # corresponds to the default Megatron arguments (hidden 2048, 16 heads, 4 key/value heads,
    # vocab 152064, 8 layers, intermediate 8192, head dim 128), with keyword overrides applied on
    # top (extra attributes are added as given).
    base = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=152064,
        num_hidden_layers=8,
        intermediate_size=8192,
        head_dim=128,
    )
    return SimpleNamespace(**(base | overrides))


def assert_same(hf, megatron):
    # [FUNCTION DOC] HELPER: Adapt the HuggingFace-style config to FLOPs arguments and assert the
    # forward FLOPs for the shared sequence lengths equal those computed from the given Megatron
    # arguments.
    adapted = flops_args_from_hf_config(hf)
    assert calculate_fwd_flops(SEQLENS, adapted) == calculate_fwd_flops(SEQLENS, megatron)


def test_dense_gqa_model_matches():
    # [FUNCTION DOC] Default dense config versus default Megatron arguments give equal FLOPs.
    assert_same(hf_config(), megatron_args())


def test_head_dim_falls_back_to_hidden_over_heads():
    # [FUNCTION DOC] Remove the head dimension from the config; the adapter must derive it as
    # hidden size divided by head count (128), matching Megatron arguments with that kv channels.
    hf = hf_config()
    del hf.head_dim
    assert_same(hf, megatron_args(kv_channels=2048 // 16))


def test_missing_kv_heads_means_full_multi_head():
    # [FUNCTION DOC] Remove the key/value head count; the adapter must treat it as full
    # multi-head attention (16 query groups).
    hf = hf_config()
    del hf.num_key_value_heads
    assert_same(hf, megatron_args(num_query_groups=16))


def test_every_layer_moe_matches():
    # [FUNCTION DOC] A config with 64 experts, expert width 1024 and 8 experts per token must equal
    # Megatron arguments with every one of the 8 layers being mixture-of-experts.
    assert_same(
        hf_config(num_experts=64, moe_intermediate_size=1024, num_experts_per_tok=8),
        megatron_args(
            num_experts=64,
            moe_ffn_hidden_size=1024,
            moe_router_topk=8,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_mixtral_style_experts_sized_by_plain_intermediate_size():
    # [FUNCTION DOC] A config using the local-experts attribute (8 experts, 2 per token) and no
    # dedicated expert width must use the plain intermediate size (8192) as expert width.
    assert_same(
        hf_config(num_local_experts=8, num_experts_per_tok=2),
        megatron_args(
            num_experts=8,
            moe_ffn_hidden_size=8192,
            moe_router_topk=2,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_gpt_oss_style_experts_sized_by_plain_intermediate_size():
    # [FUNCTION DOC] Same as the Mixtral-style case with 32 experts, 4 per token and intermediate
    # size 2880 used for both the dense FFN and the experts.
    assert_same(
        hf_config(num_local_experts=32, num_experts_per_tok=4, intermediate_size=2880),
        megatron_args(
            num_experts=32,
            ffn_hidden_size=2880,
            moe_ffn_hidden_size=2880,
            moe_router_topk=4,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_expert_width_is_not_borrowed_when_the_config_declares_one():
    # [FUNCTION DOC] When the config declares its own expert width (1024), the adapted expert FFN
    # size must be that value rather than the plain intermediate size.
    adapted = flops_args_from_hf_config(
        hf_config(num_local_experts=8, num_experts_per_tok=2, moe_intermediate_size=1024)
    )
    assert adapted.moe_ffn_hidden_size == 1024


def test_dense_config_keeps_no_expert_width():
    # [FUNCTION DOC] A dense config must adapt to an expert FFN size of none.
    assert flops_args_from_hf_config(hf_config()).moe_ffn_hidden_size is None


def test_all_moe_config_without_a_dense_ffn_width():
    # [FUNCTION DOC] A 256-expert config with an expert width and no dense intermediate size
    # attribute must still adapt, giving a dense FFN size of none, and match the Megatron side.
    hf = hf_config(num_experts=256, moe_intermediate_size=512, num_experts_per_tok=8)
    del hf.intermediate_size
    assert_same(
        hf,
        megatron_args(
            ffn_hidden_size=None,
            num_experts=256,
            moe_ffn_hidden_size=512,
            moe_router_topk=8,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_shared_experts_sized_from_the_borrowed_expert_width():
    # [FUNCTION DOC] A config with routed experts, no declared expert width, and two shared
    # experts: the expert width is borrowed from the intermediate size (2048) and the shared
    # expert size is the shared-expert count times that width (4096).
    hf = hf_config(n_routed_experts=256, num_experts_per_tok=6, n_shared_experts=2, intermediate_size=2048)
    assert_same(
        hf,
        megatron_args(
            ffn_hidden_size=2048,
            num_experts=256,
            moe_ffn_hidden_size=2048,
            moe_router_topk=6,
            moe_shared_expert_intermediate_size=2 * 2048,
            moe_layer_freq=[1] * 8,
        ),
    )


def test_a_config_that_hides_its_ffn_width_is_rejected_up_front():
    # [FUNCTION DOC] A dense config lacking the intermediate size must make the adapter raise a
    # value error whose message says the config does not expose the FFN widths.
    hf = hf_config()
    del hf.intermediate_size
    with pytest.raises(ValueError, match="does not expose the FFN widths"):
        flops_args_from_hf_config(hf)


def test_deepseek_style_dense_prefix_matches():
    # [FUNCTION DOC] Routed-experts config with the first three layers dense and two shared
    # experts: expected layer pattern is three dense then five mixture layers, shared expert size
    # twice the expert width.
    assert_same(
        hf_config(
            n_routed_experts=64,
            moe_intermediate_size=1024,
            num_experts_per_tok=8,
            first_k_dense_replace=3,
            n_shared_experts=2,
        ),
        megatron_args(
            num_experts=64,
            moe_ffn_hidden_size=1024,
            moe_router_topk=8,
            moe_shared_expert_intermediate_size=2 * 1024,
            moe_layer_freq=[0, 0, 0, 1, 1, 1, 1, 1],
        ),
    )


def test_qwen_style_sparse_step_and_dense_layers_match():
    # [FUNCTION DOC] Config with sparse step two, an explicit dense-only layer index five, and a
    # shared expert size of 2048: expected layer pattern marks mixture layers at positions 1, 3
    # and 7 (every second layer, excluding the dense-only layer) of 8.
    assert_same(
        hf_config(
            num_experts=64,
            moe_intermediate_size=1024,
            num_experts_per_tok=8,
            decoder_sparse_step=2,
            mlp_only_layers=[5],
            shared_expert_intermediate_size=2048,
        ),
        megatron_args(
            num_experts=64,
            moe_ffn_hidden_size=1024,
            moe_router_topk=8,
            moe_shared_expert_intermediate_size=2048,
            moe_layer_freq=[0, 1, 0, 1, 0, 0, 0, 1],
        ),
    )


def test_mla_matches():
    # [FUNCTION DOC] Multi-head latent attention config (query rank 1536, key/value rank 512, nope
    # dim 128, rope dim 64, value dim 128) must map onto the corresponding Megatron latent
    # attention arguments.
    assert_same(
        hf_config(
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        ),
        megatron_args(
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_head_dim=128,
            qk_pos_emb_head_dim=64,
            v_head_dim=128,
        ),
    )


def test_moe_layers_actually_cost_more_than_dense():
    # [FUNCTION DOC] Sanity check that a 64-expert, top-4 config with wide experts yields strictly
    # more forward FLOPs than the dense config.
    dense = calculate_fwd_flops(SEQLENS, flops_args_from_hf_config(hf_config()))
    moe = calculate_fwd_flops(
        SEQLENS,
        flops_args_from_hf_config(hf_config(num_experts=64, moe_intermediate_size=8192, num_experts_per_tok=4)),
    )
    assert moe > dense


@pytest.mark.parametrize("first_dense", [0, 8, 99])
def test_dense_prefix_clamps_to_the_layer_count(first_dense):
    # [FUNCTION DOC] Parametrized over dense-prefix sizes 0, 8 and 99 for an 8-layer model;
    # assert the adapted layer pattern always has exactly 8 entries (prefix clamped to layer
    # count).
    args = flops_args_from_hf_config(
        hf_config(
            n_routed_experts=64, moe_intermediate_size=1024, num_experts_per_tok=8, first_k_dense_replace=first_dense
        )
    )
    assert len(args.moe_layer_freq) == 8


def test_multimodal_config_sizes_the_text_tower_not_the_vision_tower():
    # [FUNCTION DOC] A wrapper config whose top-level attributes describe a tiny tower but that
    # carries a nested text config with the standard dense dimensions; the adapter must use the
    # nested text config, matching default Megatron arguments.
    vlm = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        text_config=SimpleNamespace(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=4,
            vocab_size=152064,
            num_hidden_layers=8,
            intermediate_size=8192,
            head_dim=128,
        ),
    )
    assert_same(vlm, megatron_args())


def test_get_text_config_wins_over_a_text_config_attribute():
    # [FUNCTION DOC] A config exposing a text-config getter callable that returns an 8-layer text
    # config, while its own top-level layer count is 1; assert the adapted layer count is 8 (the
    # getter is used). Despite the name, no competing text config attribute is set in the test.
    text = SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        vocab_size=152064,
        num_hidden_layers=8,
        intermediate_size=8192,
        head_dim=128,
    )
    cfg = SimpleNamespace(num_hidden_layers=1, hidden_size=64, num_attention_heads=4, get_text_config=lambda: text)
    assert flops_args_from_hf_config(cfg).num_layers == 8
