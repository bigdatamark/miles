# [MODULE DOC]
# Tests for the Megatron model provider in true-on-policy mode with the local transformer
# implementation: the layer-spec builder must receive the true-on-policy flags and the model must
# be built with the expected sharing of embeddings, and the critic provider must build an untied
# scalar output head. Config conversion, layer spec and GPT model are replaced by fakes.
# The whole module is currently skipped via a module-level mark (pending the Megatron bump).
from argparse import Namespace
from types import SimpleNamespace

import pytest

# TODO: add back after megatron bump
pytestmark = pytest.mark.skip(
    reason="true-on-policy megatron backend is not in miles-main-20260819 yet; re-enable in the true-on-policy follow-up PR."
)


def _make_args():
    # [FUNCTION DOC] HELPER: Returns an arguments namespace for the provider: no custom provider
    # path, raw conversion mode, local transformer implementation, no spec, no experts, no grouped
    # GEMM, query/key layer norm on, no latent attention, RMS normalization, no FP8 parameter
    # gather, padded vocabulary 151936, max positions 4096, rotary position embedding with
    # percent 1.0 and base 1000000, no rope scaling, tied embeddings, and no MTP layers.
    return Namespace(
        custom_model_provider_path=None,
        megatron_to_hf_mode="raw",
        transformer_impl="local",
        spec=None,
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=False,
        normalization="RMSNorm",
        fp8_param_gather=False,
        padded_vocab_size=151936,
        max_position_embeddings=4096,
        fp16_lm_cross_entropy=False,
        untie_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=1000000,
        use_rope_scaling=False,
        mtp_num_layers=0,
    )


def _patch_local_model_provider(monkeypatch):
    # [FUNCTION DOC] HELPER: Imports the provider module and patches (via monkeypatch) its config
    # conversion, local layer-spec getter and GPT model class with fakes. Returns the provider
    # module and a dictionary that collects the keyword arguments the fake layer-spec getter
    # received.
    from miles.backends.megatron_utils import model_provider as provider_module

    captured = {}

    def fake_core_transformer_config_from_args(_args):
        # [FUNCTION DOC] HELPER: Ignores its input and returns a simple config with hidden size
        # 2560, no sequence parallel, no kitchen, the true-on-policy contract name for the dense
        # Qwen3 version 1, no kitchen attention, and the scaled-dot-product attention backend.
        return SimpleNamespace(
            hidden_size=2560,
            sequence_parallel=False,
            use_kitchen=False,
            true_on_policy_contract="qwen3_dense_true_on_policy_v1",
            use_kitchen_attention=False,
            kitchen_attention_backend="sdpa",
        )

    def fake_get_gpt_layer_local_spec(**kwargs):
        # [FUNCTION DOC] HELPER: Records all keyword arguments into the capture dictionary and
        # returns the string "layer-spec".
        captured.update(kwargs)
        return "layer-spec"

    class FakeGPTModel:
        # [CLASS DOC] HELPER: Fake GPT model that keeps its constructor keyword arguments, its
        # config, and the share-embeddings flag taken from them.
        def __init__(self, **kwargs):
            # [FUNCTION DOC] HELPER: Stores the keyword arguments, and copies the config and the
            # share-embeddings-and-output-weights values out of them (missing keys raise a key
            # error).
            self.kwargs = kwargs
            self.config = kwargs["config"]
            self.share_embeddings_and_output_weights = kwargs["share_embeddings_and_output_weights"]

    monkeypatch.setattr(
        provider_module,
        "core_transformer_config_from_args",
        fake_core_transformer_config_from_args,
    )
    monkeypatch.setattr(
        provider_module,
        "get_gpt_layer_local_spec",
        fake_get_gpt_layer_local_spec,
    )
    monkeypatch.setattr(provider_module, "GPTModel", FakeGPTModel)

    return provider_module, captured


def test_local_model_provider_passes_true_on_policy_spec_flags(monkeypatch):
    # [FUNCTION DOC] Builds the default-role provider and calls it. Asserts the model received the
    # fake layer spec, shares embeddings and output weights, and that the layer-spec getter was
    # called with RMS normalization, the true-on-policy backend on, kitchen off, kitchen attention
    # off and the scaled-dot-product kitchen attention backend.
    provider_module, captured = _patch_local_model_provider(monkeypatch)

    provider = provider_module.get_model_provider_func(_make_args())
    model = provider()

    assert model.kwargs["transformer_layer_spec"] == "layer-spec"
    assert model.share_embeddings_and_output_weights is True
    assert captured["normalization"] == "RMSNorm"
    assert captured["use_true_on_policy_backend"] is True
    assert captured["use_kitchen"] is False
    assert captured["use_kitchen_attention"] is False
    assert captured["kitchen_attention_backend"] == "sdpa"


def test_critic_provider_builds_untied_scalar_output_head(monkeypatch):
    # [FUNCTION DOC] Builds the critic-role provider and calls it for a last pipeline stage only
    # (no pre-process, post-process on). Asserts the model does not share embeddings with the
    # output weights and its output layer weight has shape (1, 2560): one output over hidden size.
    provider_module, _ = _patch_local_model_provider(monkeypatch)

    provider = provider_module.get_model_provider_func(_make_args(), role="critic")
    model = provider(pre_process=False, post_process=True)

    assert model.share_embeddings_and_output_weights is False
    assert tuple(model.output_layer.weight.shape) == (1, 2560)
