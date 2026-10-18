# [MODULE DOC]
# Tests for the multi-token-prediction (MTP) support of the Qwen3.5 mbridge plugin and of the raw
# Qwen3.5 Megatron-to-HuggingFace exporter: expert versus dense MLP name mapping for MTP layers,
# MTP block spec built from the current transformer layer spec (including the virtual pipeline
# stage), preserved column order of the embedding-and-hidden projection, gated attention in the
# built config, and the exporter's name for that projection. The megatron and mbridge packages are
# replaced by in-memory stubs (installed per test and removed afterwards by an autouse fixture)
# and plugin files are loaded directly from their paths.
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def _reset_stubs_after_test():
    # [FUNCTION DOC] HELPER: Autouse fixture. Snapshots the global module table before each test;
    # after the test removes every module added meanwhile and restores the snapshot's entries, so
    # stubbed packages do not leak into later tests.
    # the stub standing in for megatron is a plain module rather than a package, so a stub left behind
    # makes every later import of a megatron submodule fail in whichever test happens to run next
    saved = dict(sys.modules)
    yield
    for name in set(sys.modules) - set(saved):
        del sys.modules[name]
    sys.modules.update(saved)


def install_bridge_stubs():
    # [FUNCTION DOC] HELPER: Builds plain (non-package) stub modules for megatron, its core, models
    # and GPT layer-specs submodules (the latter with an MTP block-spec getter returning a pair of
    # the marker "mtp-spec" and the passed layer spec), and for mbridge with a model-registration
    # decorator that returns the class unchanged plus a stub Qwen2 MoE bridge class. The stub
    # bridge has a keyword-to-HuggingFace-name table for MLP weights (shared experts, pre-MLP
    # layer norm, router, shared expert gate, routed experts fc1/fc2), an MLP name mapping method
    # (layer number from the third dotted segment; expert id taken from the text after the word
    # "weight"; raises not-implemented when nothing matches), an attention mapping method that
    # always raises, a layer-spec getter returning a marker string that includes the virtual
    # stage when given, a GPT-args getter returning a small mapping, a model-provider factory that
    # combines both, a weight-to-mcore method returning the single input weight, a
    # weight-to-HF method returning the name and weight as lists, and a config builder echoing its
    # keyword arguments. Registers all stubs in the global module table. Then loads the real
    # linear-attention mixin file directly (four directories above this file, bypassing the
    # package init) and registers it, with a bare parent package stub, in the module table.
    # Side effects: mutates the global module table; the autouse fixture undoes it.
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    gpt_layer_specs_mod.get_gpt_mtp_block_spec = lambda _config, transformer_layer_spec, **_kwargs: (
        "mtp-spec",
        transformer_layer_spec,
    )

    mbridge_mod = types.ModuleType("mbridge")
    mbridge_core_mod = types.ModuleType("mbridge.core")
    mbridge_models_mod = types.ModuleType("mbridge.models")

    def register_model(_names):
        # [FUNCTION DOC] HELPER: Stub registration decorator factory; ignores the names and
        # returns a decorator that returns the class unchanged.
        def decorator(cls):
            # [FUNCTION DOC] HELPER: Returns the given class unchanged.
            return cls

        return decorator

    class Qwen2MoEBridge:
        # [CLASS DOC] HELPER: Stub of the base Qwen2 MoE bridge providing the parent behavior that
        # the Qwen3.5 plugin subclass extends, with just enough logic for the tests.
        _MLP_MAPPING = {
            "shared_experts.linear_fc1.weight": [
                "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
                "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
            ],
            "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
            "shared_experts.linear_fc2.weight": ["model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"],
            "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
            "shared_experts.gate_weight": ["model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
            "mlp.experts.linear_fc1": [
                "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
            ],
            "mlp.experts.linear_fc2": ["model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"],
        }

        def _weight_name_mapping_mlp(self, name: str) -> list[str]:
            # [FUNCTION DOC] HELPER: Maps a Megatron MLP parameter name to HuggingFace names. The
            # layer number is the third dot-separated segment. Walks the keyword table in order and
            # uses the first keyword contained in the name; for templates containing an expert-id
            # placeholder the expert id is the text after the last occurrence of "weight" in the
            # name; templates are formatted with the layer number (and expert id). Raises a
            # not-implemented error naming the parameter if no keyword matches.
            layer_number = name.split(".")[2]
            convert_names = []
            for keyword, mapping_names in self._MLP_MAPPING.items():
                if keyword in name:
                    if "{expert_id}" in mapping_names[0]:
                        expert_id = name.split("weight")[-1]
                        convert_names.extend(
                            [x.format(layer_number=layer_number, expert_id=expert_id) for x in mapping_names]
                        )
                    else:
                        convert_names.extend([x.format(layer_number=layer_number) for x in mapping_names])
                    break
            if len(convert_names) == 0:
                raise NotImplementedError(f"Unsupported parameter name: {name}")
            return convert_names

        def _weight_name_mapping_attention(self, name: str) -> list[str]:
            # [FUNCTION DOC] HELPER: Always raises a not-implemented error naming the parameter
            # (the tests must not reach attention mapping).
            raise NotImplementedError(f"Unexpected attention mapping lookup: {name}")

        def _get_transformer_layer_spec(self, vp_stage=None):
            # [FUNCTION DOC] HELPER: Returns the marker string REAL_LAYER_SPEC, with a VP suffix
            # and the virtual pipeline stage number appended when a stage is given.
            return "REAL_LAYER_SPEC" if vp_stage is None else f"REAL_LAYER_SPEC_VP{vp_stage}"

        def _get_gptmodel_args(self) -> dict:
            # [FUNCTION DOC] HELPER: Returns a mapping with a single entry base set to "ok".
            return {"base": "ok"}

        def _model_provider(self, callbacks):
            # [FUNCTION DOC] HELPER: Returns a provider function; the callbacks are ignored.
            def provider(pre_process, post_process, vp_stage=None):
                # [FUNCTION DOC] HELPER: Fetches the layer spec for the given virtual stage and
                # the GPT args, and returns a mapping with the layer spec under the transformer
                # layer spec key plus all GPT args. The pre/post-process flags are unused.
                transformer_layer_spec = self._get_transformer_layer_spec(vp_stage)
                gptmodel_args = self._get_gptmodel_args()
                return {"transformer_layer_spec": transformer_layer_spec, **gptmodel_args}

            return provider

        def _weight_to_mcore_format(self, _mcore_weights_name, hf_weights):
            # [FUNCTION DOC] HELPER: Asserts exactly one HuggingFace weight was given and returns
            # it unchanged; the name is ignored.
            assert len(hf_weights) == 1
            return hf_weights[0]

        def _weight_to_hf_format(self, mcore_weights_name, mcore_weights):
            # [FUNCTION DOC] HELPER: Returns a pair of lists: the single name and the single
            # weight, unchanged.
            return [mcore_weights_name], [mcore_weights]

        def _build_base_config(self, **kwargs):
            # [FUNCTION DOC] HELPER: Returns the keyword arguments as a mapping.
            return kwargs

    mbridge_core_mod.register_model = register_model
    mbridge_models_mod.Qwen2MoEBridge = Qwen2MoEBridge

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["mbridge"] = mbridge_mod
    sys.modules["mbridge.core"] = mbridge_core_mod
    sys.modules["mbridge.models"] = mbridge_models_mod

    # the real mixin is dependency-free; loading its file directly skips the package
    # __init__, which imports every bridge and their real megatron/mbridge deps
    mixin_path = Path(__file__).resolve().parents[4] / "miles_plugins" / "mbridge" / "linear_attn.py"
    spec = importlib.util.spec_from_file_location("miles_plugins.mbridge.linear_attn", mixin_path)
    mixin_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mixin_mod)
    sys.modules["miles_plugins.mbridge"] = types.ModuleType("miles_plugins.mbridge")
    sys.modules["miles_plugins.mbridge.linear_attn"] = mixin_mod


def load_bridge_module():
    # [FUNCTION DOC] HELPER: Installs the bridge stubs, removes any previously loaded copy of the
    # test module name from the module table, loads the Qwen3.5 mbridge plugin file from the
    # repository path under that private name, executes it, and returns the module. Raises an
    # assertion error if the spec has no loader.
    install_bridge_stubs()
    module_path = Path(__file__).resolve().parents[4] / "miles_plugins" / "mbridge" / "qwen3_5.py"
    module_name = "test_qwen3_5_bridge_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_raw_export_module():
    # [FUNCTION DOC] HELPER: Loads the raw Qwen3.5 Megatron-to-HuggingFace exporter file from the
    # repository's Megatron utilities path under a private module name (replacing any previous
    # copy in the module table), executes it without installing stubs, and returns the module.
    # Raises an assertion error if the spec has no loader.
    module_path = (
        Path(__file__).resolve().parents[4] / "miles" / "backends" / "megatron_utils" / "megatron_to_hf" / "qwen3_5.py"
    )
    module_name = "test_qwen3_5_raw_export_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mtp_moe_expert_mapping_uses_individual_hf_weights():
    # [FUNCTION DOC] Creates a bridge without running its constructor and converts MTP routed
    # expert fc1 and fc2 parameter names for expert 42. Asserts fc1 gives separate gate and up
    # projection names and fc2 gives a down projection name, all under the per-expert MTP layer
    # path with expert id 42.
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    fc1_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight42")
    fc2_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.experts.linear_fc2.weight42")

    assert fc1_names == [
        "mtp.layers.0.mlp.experts.42.gate_proj.weight",
        "mtp.layers.0.mlp.experts.42.up_proj.weight",
    ]
    assert fc2_names == ["mtp.layers.0.mlp.experts.42.down_proj.weight"]


def test_mtp_dense_mlp_mapping_still_uses_dense_hf_weights():
    # [FUNCTION DOC] Converts MTP dense (non-expert) MLP fc1 and fc2 names. Asserts fc1 gives the
    # dense gate and up projection names and fc2 the dense down projection name for MTP layer 0.
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    fc1_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.linear_fc1.weight")
    fc2_names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.linear_fc2.weight")

    assert fc1_names == ["mtp.layers.0.mlp.gate_proj.weight", "mtp.layers.0.mlp.up_proj.weight"]
    assert fc2_names == ["mtp.layers.0.mlp.down_proj.weight"]


def test_mtp_block_spec_uses_current_transformer_layer_spec():
    # [FUNCTION DOC] A bridge with a placeholder config and a text config of one MTP hidden layer
    # builds its model provider and calls it with virtual pipeline stage 3. Asserts the result's
    # transformer layer spec is the stub's stage-3 spec and its MTP block spec is the pair of the
    # MTP marker and that same stage-3 spec (so MTP is built from the current spec).
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.config = "CONFIG_OBJECT"
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))

    provider = bridge._model_provider([])
    result = provider(True, True, vp_stage=3)

    assert result["transformer_layer_spec"] == "REAL_LAYER_SPEC_VP3"
    assert result["mtp_block_spec"] == ("mtp-spec", "REAL_LAYER_SPEC_VP3")


def test_eh_proj_keeps_column_order_when_loading_to_mcore():
    # [FUNCTION DOC] Converts a 3-by-8 float tensor named as the MTP layer-0 embedding-and-hidden
    # projection weight to Megatron format and asserts the result equals the input (the two
    # concatenated column halves are not swapped).
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)

    weight = torch.arange(24, dtype=torch.float32).view(3, 8)
    converted = bridge._weight_to_mcore_format("mtp.layers.0.eh_proj.weight", [weight])

    assert torch.equal(converted, weight)


def test_build_config_enables_gated_attention_when_transformer_config_supports_it():
    # [FUNCTION DOC] A bridge with one MTP hidden layer whose transformer config class declares
    # the MTP-layer-count and attention-output-gate fields builds its config. Asserts the config
    # has MTP layers 1 and attention output gate true.
    module = load_bridge_module()
    bridge = module.Qwen3_5Bridge.__new__(module.Qwen3_5Bridge)
    bridge.hf_config = types.SimpleNamespace(text_config=types.SimpleNamespace(mtp_num_hidden_layers=1))
    bridge.TransformerConfigClass = types.SimpleNamespace(
        __dataclass_fields__={
            "mtp_num_layers": None,
            "attention_output_gate": None,
        }
    )

    config = bridge._build_config()

    assert config["mtp_num_layers"] == 1
    assert config["attention_output_gate"] is True


def test_raw_qwen3_5_mtp_export_keeps_eh_proj_column_order():
    # [FUNCTION DOC] Converts the 3-by-8 MTP layer-0 projection weight with the raw exporter
    # (empty args object). Asserts exactly one pair: the HuggingFace name for the MTP fusion
    # projection weight and the same unchanged tensor.
    module = load_raw_export_module()

    weight = torch.arange(24, dtype=torch.float32).view(3, 8)
    converted = module.convert_qwen3_5_to_hf(
        types.SimpleNamespace(), "module.module.mtp.layers.0.eh_proj.weight", weight
    )

    assert converted == [("mtp.fc.weight", weight)]


def test_the_stubs_do_not_outlive_the_test_that_installed_them():
    # [FUNCTION DOC] Guard for the cleanup fixture: if a megatron module is present in the module
    # table it must be a package (have a search path); a leftover plain stub fails the test.
    # Depends on test order (it only detects leakage if run after a stubbing test without the
    # fixture undoing it).
    """A megatron left stubbed is not a package, so the next test to import one of its submodules fails."""
    stub = sys.modules.get("megatron")

    assert stub is None or hasattr(stub, "__path__")
