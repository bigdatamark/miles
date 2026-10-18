# [MODULE DOC] CPU tests of which tensors the NVFP4 checkpoint converter and the runtime NVFP4
# quantizer treat as quantization candidates. Real quantization kernels are replaced by fakes
# (zero 4-bit packed data, unit scales) so only scope is tested: the selector is limited to routed
# expert weights of the language decoder below the configured layer count, excluding multi-modal,
# MTP, shared-expert, dense and non-expert weights; full conversions of a MiMo-like multi-modal
# checkpoint (cross-shard gate/up pairing, preserved excluded tensors, ignore list) and of a
# nested text-config checkpoint (layer count from the text config, bf16 edge layers); and the
# runtime quantizer dispatching only for the main language decoder prefix. Module constants:
# the supported decoder layer root prefixes and a fragment naming expert zero.
"""Exercise checkpoint scope on CPU; numerical NVFP4 kernels are covered by GPU tests."""

import importlib.util
import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
from tools import convert_hf_to_nvfp4 as converter

LAYER_ROOTS = ("model.layers", "language_model.model.layers", "model.language_model.layers", "language_model.layers")
EXPERT = ".mlp.experts.0."


@pytest.fixture
def fake_quantizers(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture replacing the converter's single-weight and paired
    # gate/up quantizers with fakes (reverted by monkeypatch) that append "single" or "pair" to a
    # call list. Returns that list, which fills during conversions.
    calls = []

    def output(weight):
        # [FUNCTION DOC] HELPER: Fake quantization result for a rows-by-cols weight: a zero 8-bit
        # tensor of rows by cols/2 (packed data), a ones tensor of rows by cols/16 in 8-bit float
        # (block scales) and a scalar 1.0 (global scale).
        rows, cols = weight.shape
        return (
            torch.zeros((rows, cols // 2), dtype=torch.uint8),
            torch.ones((rows, cols // 16)).to(torch.float8_e4m3fn),
            torch.tensor(1.0),
        )

    def single(weight):
        # [FUNCTION DOC] HELPER: Fake single-weight quantizer; records "single" and returns the
        # fake result.
        calls.append("single")
        return output(weight)

    def pair(gate, up):
        # [FUNCTION DOC] HELPER: Fake paired quantizer; records "pair" and returns the fake
        # result for each of the two weights.
        calls.append("pair")
        return output(gate), output(up)

    monkeypatch.setattr(converter, "quantize_nvfp4", single)
    monkeypatch.setattr(converter, "nvfp4_quantize_1d_pair", pair)
    return calls


def _convert(tmp_path, config, shards, **kwargs):
    # [FUNCTION DOC] HELPER: Writes a source checkpoint under the temporary path (a config file
    # plus one safetensors file per entry in the shards mapping from filename to tensor mapping),
    # runs the NVFP4 converter on CPU into a sibling output directory with the extra keyword
    # arguments, and validates the output: the index's weight map points each tensor at the shard
    # holding it, the union of loaded shards matches the index names, the metadata total size
    # equals the tensors' byte size, and the config's ignore list equals the quantization config's
    # excluded modules. Returns (all output tensors by name, the ignore list, the output model
    # config).
    source, target = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(config))
    for filename, tensors in shards.items():
        safetensors.torch.save_file(tensors, source / filename)
    converter.convert_nvfp4(str(source), str(target), "cpu", **kwargs)
    index = json.loads((target / "model.safetensors.index.json").read_text())
    actual = {}
    for filename in set(index["weight_map"].values()):
        tensors = safetensors.torch.load_file(target / filename)
        assert all(index["weight_map"][name] == filename for name in tensors)
        actual.update(tensors)
    assert set(index["weight_map"]) == set(actual)
    assert index["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in actual.values())
    cfg = json.loads((target / "config.json").read_text())
    quant_cfg = json.loads((target / "hf_quant_config.json").read_text())
    ignore = cfg["quantization_config"]["ignore"]
    assert ignore == quant_cfg["quantization"]["exclude_modules"]
    return actual, ignore, cfg


def _assert_unchanged(actual, name, original, ignore):
    # [FUNCTION DOC] HELPER: Asserts the named output tensor has the original dtype and equals the
    # original; that no weight scale, second-level weight scale or input scale tensor exists for
    # its module; and that the module is covered by some ignore rule (equal to it or below it).
    assert actual[name].dtype == original.dtype
    assert torch.equal(actual[name], original)
    module = name.removesuffix(".weight")
    assert not any(f"{module}.{suffix}" in actual for suffix in ("weight_scale", "weight_scale_2", "input_scale"))
    assert any(module == rule or module.startswith(rule.rstrip(".") + ".") for rule in ignore)


@pytest.mark.parametrize("root", LAYER_ROOTS)
def test_selector_is_bounded_to_language_decoder(root):
    # [FUNCTION DOC] Parametrized over the four layer roots. With a 70-layer limit, a routed
    # expert down projection in layer 69 is quantized but in layer 70 is not; shared experts, a
    # nested shared-expert experts path, a dense MLP, and an expert weight whose name is in the
    # given skip tuple are not.
    weight = torch.ones((2, 16), dtype=torch.bfloat16)
    assert converter.should_quantize(f"{root}.69{EXPERT}down_proj.weight", weight, num_hidden_layers=70)
    assert not converter.should_quantize(f"{root}.70{EXPERT}down_proj.weight", weight, num_hidden_layers=70)
    assert not converter.should_quantize(f"{root}.0.mlp.shared_experts.down_proj.weight", weight)
    assert not converter.should_quantize(f"{root}.0.mlp.shared_expert.experts.0.down_proj.weight", weight)
    assert not converter.should_quantize(f"{root}.0.mlp.down_proj.weight", weight)
    assert not converter.should_quantize(f"{root}.0{EXPERT}down_proj.weight", weight, ("down_proj",))


@pytest.mark.parametrize(
    "root",
    [
        "model.mtp.layers",
        "model.mtp_layers",
        "mtp.layers",
        "visual.layers",
        "audio_encoder.layers",
        "speech_embeddings.layers",
        "vision.model.layers",
        "model.visual.layers",
    ],
)
def test_non_language_experts_are_not_candidates(root):
    # [FUNCTION DOC] Parametrized over MTP, visual, audio, speech and vision layer roots: an
    # expert gate projection under those roots (with an unsupported shape) is not selected.
    # An unsupported shape in an excluded tower must never reach quantizer validation.
    assert not converter.should_quantize(f"{root}.0{EXPERT}gate_proj.weight", torch.ones((2, 7)))


def test_mimo_multimodal_checkpoint_scope_and_cross_shard_pair(tmp_path, fake_quantizers):
    # [FUNCTION DOC] Converts a MiMo-like checkpoint with 70 layers, one next-token-prediction
    # layer and vision and audio configs. The last main layer has gate and down projections in
    # one shard and the up projection in another (cross-shard pair). Many excluded tensors (MTP,
    # layer 70, visual, audio, speech, other MoE naming schemes, a fused expert bank, shared
    # experts, attention, lm head) have distinct filler values. Asserts the fake quantizers were
    # called once each as pair and single, the config keeps its model type and layer count, the
    # three main expert weights become 8-bit packed tensors of shape 2 by 8 with all three scale
    # tensors and are not ignored, and every excluded tensor is unchanged, unscaled and covered by
    # the ignore list (expert tensors by their experts-level prefix).
    # MiMo-V2.6-Pro-RL uses a flat 70-layer text config and a separate model.mtp namespace.
    config = {
        "model_type": "mimo_v2",
        "num_hidden_layers": 70,
        "num_nextn_predict_layers": 1,
        "vision_config": {},
        "audio_config": {},
    }
    main = f"model.layers.69{EXPERT}"
    gate, up, down = (main + suffix + ".weight" for suffix in ("gate_proj", "up_proj", "down_proj"))
    excluded = [
        f"model.mtp.layers.0{EXPERT}gate_proj.weight",
        f"model.layers.70{EXPERT}gate_proj.weight",
        f"visual.layers.0{EXPERT}gate_proj.weight",
        f"audio_encoder.layers.0{EXPERT}gate_proj.weight",
        f"speech_embeddings.layers.0{EXPERT}gate_proj.weight",
        "model.mtp.layers.0.block_sparse_moe.experts.0.w1.weight",
        "audio_encoder.layers.0.moe.experts.0.w2.weight",
        "model.mtp.layers.0.mlp.experts.gate_up_proj.weight",
        "model.layers.69.mlp.shared_experts.gate_proj.weight",
        "model.layers.69.self_attn.q_proj.weight",
        "lm_head.weight",
    ]
    preserved = {name: torch.full((2, 16), idx + 1, dtype=torch.bfloat16) for idx, name in enumerate(excluded)}
    bank_name = "model.mtp.layers.0.mlp.experts.gate_up_proj.weight"
    preserved[bank_name] = torch.ones((2, 2, 16), dtype=torch.bfloat16)
    weight = torch.ones((2, 16), dtype=torch.bfloat16)
    actual, ignore, cfg = _convert(
        tmp_path,
        config,
        {"a.safetensors": {gate: weight, down: weight.clone(), **preserved}, "b.safetensors": {up: weight}},
    )
    assert sorted(fake_quantizers) == ["pair", "single"]
    assert cfg["model_type"] == "mimo_v2" and cfg["num_hidden_layers"] == 70
    for name in (gate, up, down):
        module = name.removesuffix(".weight")
        assert actual[name].dtype == torch.uint8 and actual[name].shape == (2, 8)
        assert all(f"{module}.{suffix}" in actual for suffix in ("weight_scale", "weight_scale_2", "input_scale"))
        assert not any(module == rule or module.startswith(rule.rstrip(".") + ".") for rule in ignore)
    for name, original in preserved.items():
        _assert_unchanged(actual, name, original, ignore)
        if ".experts." in name:
            assert name.split(".experts.", 1)[0] + ".experts" in ignore


@pytest.mark.parametrize("root", LAYER_ROOTS[1:])
def test_nested_text_config_controls_layer_bounds_and_bf16_edges(tmp_path, fake_quantizers, root):
    # [FUNCTION DOC] Parametrized over the non-first layer roots. Uses a config whose outer layer
    # count (99) disagrees with the nested text config (3), four layers of expert down
    # projections, and one layer kept in bf16 at each end. Asserts only the single quantizer was
    # used once, the nested quantization config gets the ignore list, algorithm NVFP4 and method
    # modelopt, only layer 1 is quantized to 8-bit packed form while layers 0, 2, 3 are unchanged
    # and ignored (rules for layers 0 and 2 present), and no ignore rule uses the default model
    # layer root.
    # The outer count deliberately disagrees: language depth must come from text_config.
    config = {
        "num_hidden_layers": 99,
        "text_config": {"num_hidden_layers": 3, "quantization_config": {"quant_method": "fp8"}},
    }
    tensors = {
        f"{root}.{idx}{EXPERT}down_proj.weight": torch.full((2, 16), idx + 1, dtype=torch.bfloat16) for idx in range(4)
    }
    actual, ignore, cfg = _convert(
        tmp_path, config, {"model.safetensors": tensors}, num_layers_at_start_in_bf16=1, num_layers_at_end_in_bf16=1
    )
    assert fake_quantizers == ["single"]
    assert cfg["text_config"]["quantization_config"]["ignore"] == ignore
    assert cfg["text_config"]["quantization_config"]["quant_algo"] == "NVFP4"
    assert cfg["text_config"]["quantization_config"]["quant_method"] == "modelopt"
    for name, original in tensors.items():
        if name == f"{root}.1{EXPERT}down_proj.weight":
            assert actual[name].dtype == torch.uint8
        else:
            _assert_unchanged(actual, name, original, ignore)
    assert {f"{root}.0.", f"{root}.2."}.issubset(ignore)
    assert not any(rule.startswith("model.layers.") for rule in ignore)


@pytest.mark.parametrize(
    "prefix, eligible",
    [
        ("decoder", True),
        ("module.module.decoder", True),
        ("language_model.decoder", True),
        ("module.language_model.decoder", True),
        ("mtp", False),
        ("module.module.mtp", False),
        ("mtp.decoder", False),
        ("mtp.layers.0.transformer_layer.decoder", False),
        ("vision_model.decoder", False),
        ("audio_encoder.decoder", False),
        ("module.vision_model.decoder", False),
    ],
)
def test_runtime_quantizer_only_dispatches_main_language_decoder(monkeypatch, prefix, eligible):
    # [FUNCTION DOC] Parametrized over module prefixes with an eligibility flag. Loads the
    # runtime NVFP4 quantizer module directly from its file path under a private module name
    # (executing it), replaces its expert-parameter quantization helper with one returning a
    # fixed converted list (reverted by monkeypatch), and calls the params quantization function
    # on a layer-zero expert weight name under the prefix. Asserts the converted list is returned
    # for main language decoder prefixes and the original list object otherwise (MTP, vision and
    # audio prefixes).
    # Load this leaf directly so CPU tests need neither Megatron nor other processor backends.
    path = (
        Path(__file__).resolve().parents[2]
        / "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_nvfp4.py"
    )
    spec = importlib.util.spec_from_file_location("nvfp4_scope_quantizer", path)
    quantizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quantizer)
    original, converted = [], [("quantized", torch.zeros(1))]
    monkeypatch.setattr(quantizer, "_quantize_moe_params", lambda *_: converted)
    actual = quantizer.quantize_params_nvfp4(
        None, f"{prefix}.layers.0.mlp.experts.linear_fc2.weight0", original, {"quant_method": "nvfp4"}
    )
    assert actual is (converted if eligible else original)
