# [MODULE DOC]
# Test for the Nemotron-H Megatron-bridge plugin's weight-mapping registry: after the plugin
# builds its bridge subclass, the mapping registry must contain a Mamba convolution mapping for
# both the dotted and underscored Megatron names of the convolution weight and bias, keeping the
# base registry's own mapping objects, order and names untouched. The Megatron bridge packages
# are replaced with stub modules and the plugin file is loaded directly from its path.
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class Mapping:
    # [CLASS DOC] HELPER: Stub for the bridge's generic parameter mapping; holds a Megatron
    # parameter name pattern and a HuggingFace parameter name pattern.
    def __init__(self, megatron_param: str, hf_param: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the two names.
        self.megatron_param = megatron_param
        self.hf_param = hf_param


class MambaMapping(Mapping):
    # [CLASS DOC] HELPER: Stub for the bridge's Mamba convolution mapping; same fields as the
    # generic stub and used to check the mapping kind.
    pass


def load_registry(monkeypatch: pytest.MonkeyPatch, base: list[Mapping], attribute: str) -> list[Mapping]:
    # [FUNCTION DOC] HELPER: Installs stub modules (via monkeypatch, undone at test end) for the
    # bridge mapping registry, model bridge (whose registration decorator is the identity), the
    # parameter mappings (the two stubs above), the Nemotron-H base bridge (whose mapping registry
    # method returns a registry of the given base mappings) and the Mamba model module. Loads the
    # plugin file under a private module name from the repository path four directories above
    # this file, executes it, builds the bridge subclass, instantiates it and asks for its
    # mapping registry. Returns the registry's list stored under the given attribute name.
    # Raises an assertion error if the module spec cannot be created.
    class Registry:
        # [CLASS DOC] HELPER: Stub registry that stores the given mappings as a list under the
        # attribute name chosen by the test (the real class's attribute name varies by version).
        def __init__(self, *mappings: Mapping) -> None:
            # [FUNCTION DOC] HELPER: Sets the named attribute to a list of the mappings.
            setattr(self, attribute, list(mappings))

    modules = {
        "megatron.bridge.models.conversion.mapping_registry": {"MegatronMappingRegistry": Registry},
        "megatron.bridge.models.conversion.model_bridge": {
            "MegatronModelBridge": SimpleNamespace(register_bridge=lambda **kwargs: lambda cls: cls)
        },
        "megatron.bridge.models.conversion.param_mapping": {
            "AutoMapping": Mapping,
            "MambaConv1dMapping": MambaMapping,
        },
        "megatron.bridge.models.nemotronh.nemotron_h_bridge": {
            "NemotronHBridge": type("NemotronHBridge", (), {"mapping_registry": lambda self: Registry(*base)})
        },
        "megatron.core.models.mamba": {"MambaModel": object},
    }
    for name, attributes in modules.items():
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)

    path = Path(__file__).resolve().parents[4] / "miles_plugins/megatron_bridge/nemotron_h.py"
    spec = importlib.util.spec_from_file_location("nemotron_h_mapping_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = module._build_bridge_subclass()().mapping_registry()
    return getattr(registry, attribute)


@pytest.mark.parametrize("attribute", ["mappings", "_mappings"])
@pytest.mark.parametrize("existing_suffixes", [(), ("weight",), ("weight", "bias")])
def test_convolution_mapping_coverage(
    monkeypatch: pytest.MonkeyPatch, attribute: str, existing_suffixes: tuple[str, ...]
) -> None:
    # [FUNCTION DOC] Parametrized over the registry attribute name (public or underscored) and
    # over which underscored convolution mappings the base registry already has (none, weight
    # only, weight and bias). The base always has the legacy dotted weight and bias mappings.
    # Loads the registry and asserts: names are unique; for both separators (dot and underscore)
    # and both suffixes there is a Mamba convolution mapping to the HuggingFace convolution
    # weight/bias; every base mapping object is present unchanged (same object); the base list's
    # names are unchanged; and the router weight mapping is present.
    legacy = [
        MambaMapping(f"decoder.layers.*.mixer.conv1d.{suffix}", f"backbone.layers.*.mixer.conv1d.{suffix}")
        for suffix in ("weight", "bias")
    ]
    existing = [
        MambaMapping(f"decoder.layers.*.mixer.conv1d_{suffix}", f"backbone.layers.*.mixer.conv1d.{suffix}")
        for suffix in existing_suffixes
    ]
    base = legacy + existing
    original_names = [mapping.megatron_param for mapping in base]

    mappings = load_registry(monkeypatch, base, attribute)
    by_name = {mapping.megatron_param: mapping for mapping in mappings}
    assert len(by_name) == len(mappings)
    for separator in (".", "_"):
        for suffix in ("weight", "bias"):
            mapping = by_name[f"decoder.layers.*.mixer.conv1d{separator}{suffix}"]
            assert isinstance(mapping, MambaMapping)
            assert mapping.hf_param == f"backbone.layers.*.mixer.conv1d.{suffix}"
    for mapping in base:
        assert by_name[mapping.megatron_param] is mapping
    assert [mapping.megatron_param for mapping in base] == original_names
    assert "decoder.layers.*.mlp.router.weight" in by_name
