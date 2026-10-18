# [MODULE DOC]
# Cpu tests of the Kimi K3 lora plugin: native export of lora adapters to hf named, chunked weights
# (one chunk per adapter), target resolution validation, grouped linear token boundaries, and the
# embedding hook that keeps lora in the autograd graph under full recompute. Helpers build small fake
# adapters (attention, dense mlp, kda attention, experts, shared experts) with arange-valued parameters
# and a fake model holding them plus a decoder with two layers.
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles.utils.lora.utils import get_adapter_target_modules, matches_lora_target
from miles_plugins.models.kimi_k3.lora import (
    KimiK3LoRAAdapter,
    _enable_full_recompute_input_grads,
    _grouped_linear,
    export_kimi_k3_lora_hf_chunks,
    resolve_kimi_k3_adapter_targets,
)


def _parameter(*shape):
    # [FUNCTION DOC] HELPER: Returns a trainable parameter of the given shape holding the float values
    # 0, 1, 2, ... in row-major order.
    return nn.Parameter(torch.arange(torch.tensor(shape).prod()).reshape(shape).float())


def _mla_attention_adapter():
    # [FUNCTION DOC] HELPER: Builds an mla-attention adapter for layer 3's self attention prefix with
    # lora A and B parameters for q-a (2x8, 4x2), kv-a (2x8, 6x2) and output (2x3, 8x2).
    adapter = KimiK3LoRAAdapter("mla_attention", "language_model.model.layers.3.self_attn.")
    adapter.register_parameter("q_a_lora_A", _parameter(2, 8))
    adapter.register_parameter("q_a_lora_B", _parameter(4, 2))
    adapter.register_parameter("kv_a_lora_A", _parameter(2, 8))
    adapter.register_parameter("kv_a_lora_B", _parameter(6, 2))
    adapter.register_parameter("o_lora_A", _parameter(2, 3))
    adapter.register_parameter("o_lora_B", _parameter(8, 2))
    return adapter


def _expert_adapter():
    # [FUNCTION DOC] HELPER: Builds an experts adapter for layer 4's sparse moe experts prefix with 3
    # experts: shared-outer w1 and w3 A (2x8), per-expert w1 and w3 B (3x5x2), per-expert w2 A (3x2x5)
    # and shared-outer w2 B (8x2).
    adapter = KimiK3LoRAAdapter(
        "experts",
        "language_model.model.layers.4.block_sparse_moe.experts.",
    )
    adapter.register_parameter("w1_lora_A", _parameter(2, 8))
    adapter.register_parameter("w3_lora_A", _parameter(2, 8))
    adapter.register_parameter("w1_lora_B", _parameter(3, 5, 2))
    adapter.register_parameter("w3_lora_B", _parameter(3, 5, 2))
    adapter.register_parameter("w2_lora_A", _parameter(3, 2, 5))
    adapter.register_parameter("w2_lora_B", _parameter(8, 2))
    return adapter


def _shared_expert_adapter():
    # [FUNCTION DOC] HELPER: Builds a shared-experts adapter for layer 4 with fc1 A (2x8), fc1 B (10x2),
    # fc2 A (2x5) and fc2 B (8x2). The fc1 B output of 10 splits into gate and up halves of 5.
    adapter = KimiK3LoRAAdapter(
        "shared_experts",
        "language_model.model.layers.4.block_sparse_moe.shared_experts.",
    )
    adapter.register_parameter("fc1_lora_A", _parameter(2, 8))
    adapter.register_parameter("fc1_lora_B", _parameter(10, 2))
    adapter.register_parameter("fc2_lora_A", _parameter(2, 5))
    adapter.register_parameter("fc2_lora_B", _parameter(8, 2))
    return adapter


def _dense_adapter():
    # [FUNCTION DOC] HELPER: Builds a dense-mlp adapter for layer 3's mlp prefix with the same four
    # parameter shapes as the shared experts adapter.
    adapter = KimiK3LoRAAdapter(
        "dense_mlp",
        "language_model.model.layers.3.mlp.",
    )
    adapter.register_parameter("fc1_lora_A", _parameter(2, 8))
    adapter.register_parameter("fc1_lora_B", _parameter(10, 2))
    adapter.register_parameter("fc2_lora_A", _parameter(2, 5))
    adapter.register_parameter("fc2_lora_B", _parameter(8, 2))
    return adapter


def _kda_attention_adapter():
    # [FUNCTION DOC] HELPER: Builds a kda-attention adapter for layer 4's self attention prefix with
    # only output projection lora A (2x3) and B (8x2).
    adapter = KimiK3LoRAAdapter(
        "kda_attention",
        "language_model.model.layers.4.self_attn.",
    )
    adapter.register_parameter("o_lora_A", _parameter(2, 3))
    adapter.register_parameter("o_lora_B", _parameter(8, 2))
    return adapter


def _model_with_adapters(*, include_shared_experts=True):
    # [FUNCTION DOC] HELPER: Builds a module holding a module list of adapters in the order mla
    # attention, dense mlp, kda attention, experts, and (when requested, default true) shared experts,
    # plus a fake decoder with two layers: layer number 4 (non-kda attention, empty mlp) and layer number
    # 5 (kda attention, mlp with experts and shared experts).
    mla_attention = _mla_attention_adapter()
    dense = _dense_adapter()
    kda_attention = _kda_attention_adapter()
    experts = _expert_adapter()
    shared_experts = _shared_expert_adapter()
    adapters = [mla_attention, dense, kda_attention, experts]
    if include_shared_experts:
        adapters.append(shared_experts)

    model = nn.Module()
    model.adapters = nn.ModuleList(adapters)
    model.decoder = SimpleNamespace(
        layers=[
            SimpleNamespace(
                layer_number=4,
                self_attention=SimpleNamespace(is_kda=False),
                mlp=SimpleNamespace(),
            ),
            SimpleNamespace(
                layer_number=5,
                self_attention=SimpleNamespace(is_kda=True),
                mlp=SimpleNamespace(
                    experts=SimpleNamespace(),
                    shared_experts=SimpleNamespace(),
                ),
            ),
        ]
    )
    return model


@pytest.mark.parametrize("include_fc2", [True, False])
def test_native_export_is_chunked_by_adapter(monkeypatch, include_fc2):
    # [FUNCTION DOC] Parametrized over whether the experts adapter includes the second projection (fc2).
    # Forces tensor and expert parallel world sizes to 1, builds the model and default hf targets; when
    # fc2 is excluded, drops its flag, w2 targets and w2 parameters. Resolves adapter targets (not
    # canonical, shared outer experts loras) and exports the chunks. Asserts: every exported module name
    # matches some target and every target matches some exported name; exactly 5 chunks, one per adapter.
    # Chunk 0 (attention) has q-a lora A shape 2x8, kv-a-with-mqa lora B shape 6x2, output lora A 2x3.
    # Chunk 3 (experts): w1 lora A is exported with a leading expert dim of 1 (1x2x8), w1 lora B stays
    # 3x5x2, and with fc2 w2 lora A is 3x2x5 and w2 lora B is 1x8x2, otherwise no w2 names exist. Chunk 4
    # (shared experts): gate and up lora A 2x8, gate and up lora B 5x2 (fc1 B split in halves), down lora
    # A 2x5 and down lora B 8x2.
    """A wrong expert dim is a shape mismatch in SGLang's LoRA pool; a wrong HF name is silently dropped."""
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)

    model = _model_with_adapters()
    targets = _default_hf_targets()
    model.adapters[3].include_fc2 = include_fc2
    if not include_fc2:
        targets = [target for target in targets if not target.endswith(".experts.*.w2")]
        del model.adapters[3].w2_lora_A
        del model.adapters[3].w2_lora_B
    adapter_targets = resolve_kimi_k3_adapter_targets(targets, canonical=False, experts_shared_outer_loras=True)
    chunks = list(export_kimi_k3_lora_hf_chunks([model]))
    exported = get_adapter_target_modules(name for chunk in chunks for name, _ in chunk)
    assert all(any(matches_lora_target(name, target) for target in adapter_targets) for name in exported)
    assert all(any(matches_lora_target(name, target) for name in exported) for target in adapter_targets)

    assert len(chunks) == 5
    attention = dict(chunks[0])
    assert attention["language_model.model.layers.3.self_attn.q_a_proj.lora_A.weight"].shape == (2, 8)
    assert attention["language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.lora_B.weight"].shape == (
        6,
        2,
    )
    assert attention["language_model.model.layers.3.self_attn.o_proj.lora_A.weight"].shape == (2, 3)

    experts = dict(chunks[3])
    prefix = "language_model.model.layers.4.block_sparse_moe.experts."
    assert experts[f"{prefix}w1.lora_A.weight"].shape == (1, 2, 8)
    assert experts[f"{prefix}w1.lora_B.weight"].shape == (3, 5, 2)
    if include_fc2:
        assert experts[f"{prefix}w2.lora_A.weight"].shape == (3, 2, 5)
        assert experts[f"{prefix}w2.lora_B.weight"].shape == (1, 8, 2)
    else:
        assert not any(".w2." in name for name in experts)

    shared_experts = dict(chunks[4])
    prefix = "language_model.model.layers.4.block_sparse_moe.shared_experts."
    assert shared_experts[f"{prefix}gate_proj.lora_A.weight"].shape == (2, 8)
    assert shared_experts[f"{prefix}gate_proj.lora_B.weight"].shape == (5, 2)
    assert shared_experts[f"{prefix}up_proj.lora_A.weight"].shape == (2, 8)
    assert shared_experts[f"{prefix}up_proj.lora_B.weight"].shape == (5, 2)
    assert shared_experts[f"{prefix}down_proj.lora_A.weight"].shape == (2, 5)
    assert shared_experts[f"{prefix}down_proj.lora_B.weight"].shape == (8, 2)


def _default_hf_targets():
    # [FUNCTION DOC] HELPER: Resolves hf lora targets for a small kimi k3 config: 5 hidden layers, first
    # 4 dense, moe frequency 1, 3 experts, 1 shared expert.
    return resolve_hf_lora_targets(
        dict(
            model_type="kimi_k3",
            text_config=dict(
                num_hidden_layers=5,
                first_k_dense_replace=4,
                moe_layer_freq=1,
                num_experts=3,
                num_shared_experts=1,
            ),
        )
    )


@pytest.mark.parametrize("case", ["missing", "extra", "canonical", "per-expert"])
def test_native_target_resolution_rejects_unsupported_layout(case):
    # [FUNCTION DOC] Parametrized over four invalid inputs to the adapter target resolver. Missing:
    # the q-a projection target removed, expects a not-implemented error mentioning missing. Extra: an
    # unsupported q-b projection target appended, expects not-implemented mentioning unsupported.
    # Canonical: canonical lora mode requested, expects an assertion error mentioning canonical lora.
    # Per-expert: shared-outer experts loras disabled, expects not-implemented mentioning shared-outer.
    # The error type for the canonical case is assertion error; all others are not-implemented.
    targets = _default_hf_targets()
    if case == "missing":
        targets = [target for target in targets if not target.endswith(".q_a_proj")]
    elif case == "extra":
        targets.append("language_model.model.layers.*.self_attn.q_b_proj")
    error = AssertionError if case == "canonical" else NotImplementedError
    message = {
        "missing": "missing=",
        "extra": "unsupported=",
        "canonical": "canonical_lora",
        "per-expert": "shared-outer",
    }[case]
    with pytest.raises(error, match=message):
        resolve_kimi_k3_adapter_targets(
            targets,
            canonical=case == "canonical",
            experts_shared_outer_loras=case != "per-expert",
        )


def test_native_export_rejects_missing_shared_expert_adapter(monkeypatch):
    # [FUNCTION DOC] Forces tensor and expert parallel world sizes to 1, builds a model without the
    # shared experts adapter, and asserts that consuming the export raises a runtime error saying the
    # adapter layout is incomplete.
    """A layer without its adapter must fail the export, not ship a partial adapter SGLang accepts."""
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 1)

    model = _model_with_adapters(include_shared_experts=False)

    with pytest.raises(RuntimeError, match="adapter layout is incomplete"):
        list(export_kimi_k3_lora_hf_chunks([model]))


def test_grouped_linear_uses_expert_token_boundaries():
    # [FUNCTION DOC] Inputs are three tokens of width 2; two experts have 1x2 weights selecting the
    # first and second feature. With token counts [1, 2] the grouped linear applies expert 0 to the first
    # token and expert 1 to the other two; asserts the output is [[1],[4],[6]].
    """Wrong boundaries apply expert i's adapter to expert j's tokens without any error."""
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])

    output = _grouped_linear(inputs, weights, [1, 2])

    torch.testing.assert_close(output, torch.tensor([[1.0], [4.0], [6.0]]))


def test_full_recompute_keeps_native_lora_in_autograd_graph():
    # [FUNCTION DOC] Builds a module with a frozen all-ones embedding (8x4), a 4x4 ones adapter
    # parameter, full recompute granularity and pre-process on, and installs the input-grads hook. In
    # train mode, embeds two tokens, runs a reentrant checkpointed matrix product with the adapter, and
    # back-propagates the sum. Asserts the embedding output requires grad, the frozen embedding weight
    # gets no gradient, and the adapter gradient is a 4x4 tensor of 2.0 (two tokens). In eval mode asserts
    # the embedding output does not require grad.
    """Under full recompute with a frozen base the segment input carries no grad, so every LoRA
    gradient is zero; the embedding hook must fix that in training and stay out of eval."""
    model = nn.Module()
    model.embedding = nn.Embedding.from_pretrained(torch.ones(8, 4), freeze=True)
    model.adapter = nn.Parameter(torch.ones(4, 4))
    model.config = SimpleNamespace(recompute_granularity="full")
    model.pre_process = True
    model.embedding.requires_grad_(False)
    _enable_full_recompute_input_grads(model)

    model.train()
    hidden_states = model.embedding(torch.tensor([[1, 2]]))
    output = checkpoint(lambda inputs: inputs @ model.adapter, hidden_states, use_reentrant=True)
    output.sum().backward()

    assert hidden_states.requires_grad
    assert model.embedding.weight.grad is None
    torch.testing.assert_close(model.adapter.grad, torch.full_like(model.adapter, 2.0))

    model.eval()
    assert not model.embedding(torch.tensor([[1, 2]])).requires_grad
