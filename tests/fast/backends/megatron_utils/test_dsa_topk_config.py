# [MODULE DOC] CPU tests of the GLM-5 native DSA (sparse attention) spec factory's resolution of
# the flashinfer top-k policy from environment-style settings: when the indexer top-k backend is
# flashinfer, the factory reads the tie-break setting (none maps to 0, "small" to 1, "large" in any
# case to 2) and the deterministic flag from the sglang environment registry and stores them on the
# config; an invalid tie-break value raises a runtime error before the native layer-spec factory is
# called; and with the torch backend the settings are not consulted at all. The spec module is
# loaded from its file path with a mocked native spec factory, and the environment registry is
# faked through the module table.
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def native_dsa_spec(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Creates a mock factory returning the string "native-spec" and,
    # via the monkeypatch fixture, registers a fake Megatron experimental-attention spec module whose
    # transformer-block spec function is that mock. Loads the GLM-5 Megatron spec source file (four
    # levels above this test, under the plugins package) by path under a private module name, executes
    # it, and returns (module, mock factory).
    factory = Mock(return_value="native-spec")
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.models.gpt.experimental_attention_variant_module_specs",
        SimpleNamespace(get_transformer_block_with_experimental_attention_variant_spec=factory),
    )
    path = Path(__file__).resolve().parents[4] / "miles_plugins/models/glm5/megatron_spec.py"
    spec = importlib.util.spec_from_file_location("native_dsa_spec_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, factory


def _set_topk_environment(monkeypatch, *, tie_break, deterministic):
    # [FUNCTION DOC] HELPER: Registers (via monkeypatch) a fake sglang environment module whose
    # registry offers two settings, the flashinfer tie-break and the deterministic flag, each with a
    # getter returning the given constant.
    envs = SimpleNamespace(
        SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=SimpleNamespace(get=lambda: tie_break),
        SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=SimpleNamespace(get=lambda: deterministic),
    )
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", SimpleNamespace(envs=envs))


@pytest.mark.parametrize(
    ("tie_break", "deterministic", "expected"),
    [(None, False, 0), ("small", False, 1), ("large", True, 2), ("LARGE", False, 2)],
)
def test_native_spec_resolves_the_shared_flashinfer_policy_in_the_worker(
    monkeypatch, native_dsa_spec, tie_break, expected, deterministic
):
    # [FUNCTION DOC] Parametrized over (none, false) -> 0; ("small", false) -> 1; ("large", true) -> 2;
    # ("LARGE", false) -> 2. Sets the fake environment after the spec module import (as the worker
    # does), builds a config with the flashinfer backend, calls the spec getter with virtual pipeline
    # stage 2, and asserts the config's tie-break code equals the expectation and its deterministic
    # flag is the same object value as supplied.
    module, _ = native_dsa_spec
    config = SimpleNamespace(dsa_indexer_topk_backend="flashinfer")
    # The worker environment is supplied after importing the spec factory.
    _set_topk_environment(monkeypatch, tie_break=tie_break, deterministic=deterministic)

    module.get_dsa_spec(None, config, vp_stage=2)
    assert config.dsa_indexer_topk_tie_break == expected
    assert config.dsa_indexer_topk_deterministic is deterministic


def test_invalid_flashinfer_policy_fails_before_constructing_native_layers(monkeypatch, native_dsa_spec):
    # [FUNCTION DOC] With tie-break "random" (invalid) and flashinfer backend, asserts the spec
    # getter raises a runtime error naming the tie-break setting and that the native spec factory was
    # never called.
    module, factory = native_dsa_spec
    _set_topk_environment(monkeypatch, tie_break="random", deterministic=True)
    with pytest.raises(RuntimeError, match="SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK"):
        module.get_dsa_spec(None, SimpleNamespace(dsa_indexer_topk_backend="flashinfer"), vp_stage=None)
    factory.assert_not_called()


def test_torch_backend_does_not_resolve_flashinfer_policy(monkeypatch, native_dsa_spec):
    # [FUNCTION DOC] With the torch backend and an invalid tie-break setting in the fake environment,
    # asserts the spec getter completes without error (the policy is not read for this backend).
    module, _ = native_dsa_spec
    config = SimpleNamespace(dsa_indexer_topk_backend="torch")
    _set_topk_environment(monkeypatch, tie_break="invalid", deterministic=True)

    module.get_dsa_spec(None, config, vp_stage=None)
