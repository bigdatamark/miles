# [MODULE DOC] Unit tests for helper functions of the debug-utils run_megatron worker entry module
# (argument parsing, source patching, forward/backward driver, dumper finalization). Placed in the
# GPU test directory because that module imports heavy Megatron and sglang packages at top level.
# To import it, the test first imports the real miles package tree, then installs stub modules
# (mock objects) in the module table for the missing Megatron training/core modules and for several
# miles Megatron-utils submodules, and finally imports the functions under test. All tests mock the
# worker module's collaborators; no real distributed or model execution happens. Each fast-GPU test
# file runs in its own subprocess so the module stubbing cannot leak. Registered for 2-GPU CUDA and ROCm
# nightly suites (megatron label).
"""Unit tests for worker/main.py helper functions.

Lives under tests/fast-gpu/ because worker/main.py has heavy top-level
megatron / sglang imports that need a real GPU container to resolve cleanly.
Each fast-gpu file runs in its own subprocess (per-file isolation), so the
sys.modules stubbing below cannot leak across files.
"""

from __future__ import annotations

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["megatron"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=20,
    suite="nightly-stage-c-2-gpu-mi350",
    labels=["megatron"],
)

import argparse
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import torch


def _ensure_module(dotted: str) -> ModuleType:
    # [FUNCTION DOC] HELPER: For a dotted module path, walks each ancestor prefix and registers an
    # empty stub module in the global module table for any prefix not already present (existing
    # modules are untouched). Returns the module for the full path. Side effect: mutates the module
    # table.
    """Ensure *dotted* exists in sys.modules, creating stubs for any missing segments."""
    parts = dotted.split(".")
    for i in range(len(parts)):
        partial = ".".join(parts[: i + 1])
        if partial not in sys.modules:
            sys.modules[partial] = ModuleType(partial)
    return sys.modules[dotted]


# Import the real `miles` package tree before any stubbing below. `_ensure_module`
# walks dotted ancestors and stubs every missing segment, so stubbing a
# `miles.backends.megatron_utils.*` leaf while `miles` is not yet imported would
# replace the real `miles` package with an empty (non-package) stub -- and the
# later `from miles.utils...main import` would then fail with "miles is not a
# package". Importing the real parent package here registers the genuine
# `miles`, `miles.backends`, and `miles.backends.megatron_utils` packages in
# sys.modules so only the leaf modules get stubbed.
import miles.backends.megatron_utils  # noqa: E402,F401

# Stub modules whose top-level imports in main.py would fail.
_STUBS: dict[str, dict[str, Any]] = {
    "megatron.training.arguments": {
        "parse_args": MagicMock(),
        "validate_args": MagicMock(),
    },
    "megatron.training.training": {
        "get_model": MagicMock(),
    },
    "megatron.core.enums": {
        "ModelType": MagicMock(),
    },
    "megatron.core.pipeline_parallel": {
        "get_forward_backward_func": MagicMock(),
    },
    "megatron.core.mpu": MagicMock(),
}

for _mod_path, _attrs in _STUBS.items():
    mod = _ensure_module(_mod_path)
    if isinstance(_attrs, dict):
        for attr_name, attr_val in _attrs.items():
            if not hasattr(mod, attr_name):
                setattr(mod, attr_name, attr_val)
    else:
        # Replace the whole module with a MagicMock
        sys.modules[_mod_path] = _attrs

# Also ensure miles.backends.megatron_utils sub-modules have their needed symbols
for _sub in [
    "miles.backends.megatron_utils.arguments",
    "miles.backends.megatron_utils.checkpoint",
    "miles.backends.megatron_utils.initialize",
    "miles.backends.megatron_utils.model_provider",
]:
    mod = _ensure_module(_sub)
    # Provide any names imported by main.py from these modules
    for name in [
        "set_default_megatron_args",
        "load_checkpoint",
        "init",
        "get_model_provider_func",
    ]:
        if not hasattr(mod, name):
            setattr(mod, name, MagicMock())

from miles.utils.debug_utils.run_megatron.worker.main import (  # noqa: E402
    _apply_source_patches,
    _finalize_dumper,
    _parse_args,
    _run_forward_backward,
)

_MODULE = "miles.utils.debug_utils.run_megatron.worker.main"


class TestParseArgs:
    # [CLASS DOC] Tests for the worker's argument parser wrapper, with the Megatron parser and the
    # script-args bridge patched out.
    @patch(f"{_MODULE}.WORKER_SCRIPT_ARGS_BRIDGE")
    @patch(f"{_MODULE}.parse_args")
    def test_ref_load_overrides_args_load(
        self,
        mock_parse_args: MagicMock,
        mock_bridge: MagicMock,
    ) -> None:
        # [FUNCTION DOC] The patched Megatron parser returns a namespace with no load path; the
        # patched bridge returns script args whose reference-load path is set to a path object. Asserts
        # the returned args' load field becomes the path as a string and the returned script args are
        # exactly the bridge's object.
        """When script_args.ref_load is set, args.load is overridden."""
        args = argparse.Namespace(load=None)
        mock_parse_args.return_value = args

        script_args = MagicMock()
        script_args.ref_load = Path("/some/path")
        mock_bridge.from_namespace.return_value = script_args

        returned_args, returned_script = _parse_args()

        assert returned_args.load == "/some/path"
        assert returned_script is script_args

    @patch(f"{_MODULE}.WORKER_SCRIPT_ARGS_BRIDGE")
    @patch(f"{_MODULE}.parse_args")
    def test_ref_load_none_preserves_original_load(
        self,
        mock_parse_args: MagicMock,
        mock_bridge: MagicMock,
    ) -> None:
        # [FUNCTION DOC] Same setup but the namespace has load "/orig" and the reference-load is none;
        # asserts the load field is unchanged.
        """When script_args.ref_load is None, args.load stays as-is."""
        args = argparse.Namespace(load="/orig")
        mock_parse_args.return_value = args

        script_args = MagicMock()
        script_args.ref_load = None
        mock_bridge.from_namespace.return_value = script_args

        returned_args, _ = _parse_args()

        assert returned_args.load == "/orig"


class TestApplySourcePatches:
    # [CLASS DOC] Tests for the helper that loads a YAML patch config file and applies source patches.
    @patch(f"{_MODULE}.apply_patches_from_config")
    def test_reads_yaml_and_calls_patcher(
        self,
        mock_apply: MagicMock,
        tmp_path: Path,
    ) -> None:
        # [FUNCTION DOC] Writes a small YAML config file in a temp directory, calls the helper with
        # its path, and asserts the patched patch-applier was called exactly once with the file's text
        # as first positional argument and with either an extra-imports keyword or a second positional
        # argument.
        # KNOWN ISSUE: the last assertion is weak (an either-or condition), so it does not pin down
        # how the extra imports are passed.
        config_file = tmp_path / "patches.yaml"
        config_file.write_text("patches:\n  - target: foo")

        _apply_source_patches(config_file)

        mock_apply.assert_called_once()
        call_args = mock_apply.call_args
        assert call_args[0][0] == "patches:\n  - target: foo"
        assert "extra_imports" in call_args[1] or len(call_args[0]) > 1


class TestRunForwardBackward:
    # [CLASS DOC] Tests for the worker's forward/backward driver with the distributed module and the
    # pipeline schedule getter patched out (rank reported as 1, i.e. not the first rank).
    @patch(f"{_MODULE}.dist")
    @patch(f"{_MODULE}.get_forward_backward_func")
    def test_forward_only_when_run_backward_false(
        self,
        mock_get_fb: MagicMock,
        mock_dist: MagicMock,
    ) -> None:
        # [FUNCTION DOC] The schedule function mock returns an empty list. With script args saying
        # no backward, a one-sample batch (4 tokens, labels with ignore index) and a single mock model
        # chunk, asserts the schedule function was called with forward-only true.
        """run_backward=False → forward_only=True passed to the func."""
        mock_fb_func = MagicMock(return_value=[])
        mock_get_fb.return_value = mock_fb_func
        mock_dist.get_rank.return_value = 1

        args = argparse.Namespace(seq_length=4, micro_batch_size=1)
        script = MagicMock()
        script.run_backward = False

        model = [MagicMock()]
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "position_ids": torch.arange(4).unsqueeze(0),
            "labels": torch.tensor([[2, 3, 4, -100]]),
        }

        _run_forward_backward(args=args, script=script, model=model, batch=batch)

        call_kwargs = mock_fb_func.call_args[1]
        assert call_kwargs["forward_only"] is True

    @patch(f"{_MODULE}.dist")
    @patch(f"{_MODULE}.get_forward_backward_func")
    def test_no_logits_captured_returns_none(
        self,
        mock_get_fb: MagicMock,
        mock_dist: MagicMock,
    ) -> None:
        # [FUNCTION DOC] Same mocks; with a 2-token batch and no backward, asserts the driver
        # returns none when the schedule function captured no logits (as on non-last pipeline stages).
        """If no logits captured (non-last PP stage), returns None."""
        mock_fb_func = MagicMock(return_value=[])
        mock_get_fb.return_value = mock_fb_func
        mock_dist.get_rank.return_value = 1

        args = argparse.Namespace(seq_length=4, micro_batch_size=1)
        script = MagicMock()
        script.run_backward = False

        result = _run_forward_backward(
            args=args,
            script=script,
            model=[MagicMock()],
            batch={
                "input_ids": torch.tensor([[1, 2]]),
                "position_ids": torch.arange(2).unsqueeze(0),
                "labels": torch.tensor([[2, -100]]),
            },
        )

        assert result is None


class TestFinalizeDumper:
    # [CLASS DOC] Tests for the helper that finishes the tensor dumper when an environment variable
    # enables it; the dumper object is patched with a mock.
    @patch(f"{_MODULE}.dumper")
    def test_dumper_enable_env_triggers_step_and_disable(
        self,
        mock_dumper: MagicMock,
    ) -> None:
        # [FUNCTION DOC] With the dumper-enable environment variable set to 1, calls the helper and
        # asserts the dumper's step was called once and its configure was called once with enable false.
        with patch.dict(os.environ, {"DUMPER_ENABLE": "1"}):
            _finalize_dumper()

        mock_dumper.step.assert_called_once()
        mock_dumper.configure.assert_called_once_with(enable=False)

    @patch(f"{_MODULE}.dumper")
    def test_no_dumper_enable_env_does_nothing(
        self,
        mock_dumper: MagicMock,
    ) -> None:
        # [FUNCTION DOC] With the environment cleared (and the enable variable explicitly removed and
        # restored afterwards), calls the helper and asserts the dumper's step and configure were never
        # called.
        # KNOWN ISSUE: the variable removal and restore are redundant since the environment is already
        # cleared by the patch context.
        with patch.dict(os.environ, {}, clear=True):
            env_backup = os.environ.pop("DUMPER_ENABLE", None)
            try:
                _finalize_dumper()
            finally:
                if env_backup is not None:
                    os.environ["DUMPER_ENABLE"] = env_backup

        mock_dumper.step.assert_not_called()
        mock_dumper.configure.assert_not_called()


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
