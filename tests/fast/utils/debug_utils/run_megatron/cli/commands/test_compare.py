# [MODULE DOC] Tests for the compare command implementation (compare_impl in
# miles.utils.debug_utils.run_megatron.cli.commands.compare). The command first runs the activation
# comparator as a shell command (sglang's debug comparator module) with baseline and target paths
# and optional override dims, patch config and diff threshold; then, only if both logprob
# directories are provided, compares logprobs with a threshold (default 1e-3). Any failure of
# either step exits the process with code 1, but the logprob step still runs after an activation
# failure. The shell runner and the logprob comparison function are mocked.
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.commands.args import CompareArgs
from miles.utils.debug_utils.run_megatron.cli.commands.compare import compare_impl


def _make_compare_args(**overrides: object) -> CompareArgs:
    # [FUNCTION DOC] HELPER: Build compare arguments with baseline dir "/baseline" and target dir
    # "/target", updated by keyword overrides. Returns the arguments object.
    defaults = dict(
        baseline_dir=Path("/baseline"),
        target_dir=Path("/target"),
    )
    defaults.update(overrides)
    return CompareArgs(**defaults)  # type: ignore[arg-type]


class TestActivationComparison:
    # [CLASS DOC] Tests of the activation comparator shell command built by compare_impl.
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_calls_comparator(self, mock_exec: MagicMock) -> None:
        # [FUNCTION DOC] The shell runner is called exactly once and its command text contains the
        # sglang debug comparator module name.
        compare_impl(_make_compare_args())
        mock_exec.assert_called_once()
        cmd = mock_exec.call_args[0][0]
        assert "sglang.srt.debug_utils.comparator" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_required_args_in_command(self, mock_exec: MagicMock) -> None:
        # [FUNCTION DOC] The command contains the baseline-path and target-path flags and the two
        # directory values.
        compare_impl(_make_compare_args())
        cmd = mock_exec.call_args[0][0]
        assert "--baseline-path" in cmd
        assert "--target-path" in cmd
        assert "/baseline" in cmd
        assert "/target" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_optional_args_included(self, mock_exec: MagicMock) -> None:
        # [FUNCTION DOC] With override dims for baseline and target, a patch config path and a diff
        # threshold set, the command contains the four corresponding flags.
        compare_impl(
            _make_compare_args(
                override_baseline_dims="b s h",
                override_target_dims="b s v",
                patch_config=Path("/patch.yaml"),
                diff_threshold=0.01,
            )
        )
        cmd = mock_exec.call_args[0][0]
        assert "--override-baseline-dims" in cmd
        assert "--override-target-dims" in cmd
        assert "--patch-config" in cmd
        assert "--diff-threshold" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_optional_args_excluded(self, mock_exec: MagicMock) -> None:
        # [FUNCTION DOC] With defaults, none of the four optional flags appear in the command.
        compare_impl(_make_compare_args())
        cmd = mock_exec.call_args[0][0]
        assert "--override-baseline-dims" not in cmd
        assert "--override-target-dims" not in cmd
        assert "--patch-config" not in cmd
        assert "--diff-threshold" not in cmd


class TestActivationFailure:
    # [CLASS DOC] Tests of the activation comparator failing.
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_activation_failure_exits(self, mock_exec: MagicMock) -> None:
        # [FUNCTION DOC] The shell runner raises a called-process error (return code 1); asserts the
        # command exits the process with code 1.
        mock_exec.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test")
        with pytest.raises(SystemExit) as exc_info:
            compare_impl(_make_compare_args())
        assert exc_info.value.code == 1


class TestLogprobBranch:
    # [CLASS DOC] Tests of the optional logprob comparison step (both the shell runner and the
    # logprob comparison function are mocked; the mocked shell runner succeeds by default).
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_logprob_called_when_dirs_provided(self, mock_exec: MagicMock, mock_logprob: MagicMock) -> None:
        # [FUNCTION DOC] With both logprob directories and a threshold of 0.005, the logprob
        # comparison is called once with keyword arguments baseline dir, target dir and threshold.
        mock_logprob.return_value = True
        compare_impl(
            _make_compare_args(
                baseline_logprob_dir=Path("/bl_logprob"),
                target_logprob_dir=Path("/tg_logprob"),
                logprob_threshold=0.005,
            )
        )
        mock_logprob.assert_called_once_with(
            baseline_dir=Path("/bl_logprob"),
            target_dir=Path("/tg_logprob"),
            threshold=0.005,
        )

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_logprob_not_called_when_dirs_missing(self, mock_exec: MagicMock, mock_logprob: MagicMock) -> None:
        # [FUNCTION DOC] Without logprob directories the logprob comparison is never called.
        compare_impl(_make_compare_args())
        mock_logprob.assert_not_called()

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_logprob_default_threshold(self, mock_exec: MagicMock, mock_logprob: MagicMock) -> None:
        # [FUNCTION DOC] With a null logprob threshold, the logprob comparison receives a threshold
        # of 1e-3.
        mock_logprob.return_value = True
        compare_impl(
            _make_compare_args(
                baseline_logprob_dir=Path("/bl"),
                target_logprob_dir=Path("/tg"),
                logprob_threshold=None,
            )
        )
        call_kwargs = mock_logprob.call_args[1]
        assert call_kwargs["threshold"] == 1e-3

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_logprob_failure_exits(self, mock_exec: MagicMock, mock_logprob: MagicMock) -> None:
        # [FUNCTION DOC] The logprob comparison returns false; asserts the command exits with code 1.
        mock_logprob.return_value = False
        with pytest.raises(SystemExit) as exc_info:
            compare_impl(
                _make_compare_args(
                    baseline_logprob_dir=Path("/bl"),
                    target_logprob_dir=Path("/tg"),
                )
            )
        assert exc_info.value.code == 1

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_activation_pass_logprob_fail(self, mock_exec: MagicMock, mock_logprob: MagicMock) -> None:
        # [FUNCTION DOC] Activation step succeeds (mock) but the logprob comparison returns false;
        # asserts the command exits the process. KNOWN ISSUE: duplicates test_logprob_failure_exits
        # except it does not check the exit code.
        mock_logprob.return_value = False
        with pytest.raises(SystemExit):
            compare_impl(
                _make_compare_args(
                    baseline_logprob_dir=Path("/bl"),
                    target_logprob_dir=Path("/tg"),
                )
            )

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.compare_logprobs")
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.compare.run_shell_command")
    def test_activation_fail_logprob_pass(self, mock_exec: MagicMock, mock_logprob: MagicMock) -> None:
        # [FUNCTION DOC] The activation shell command fails while the logprob comparison would pass;
        # asserts the command still exits the process and that the logprob comparison was called
        # exactly once (the logprob step runs even after an activation failure).
        mock_exec.side_effect = subprocess.CalledProcessError(returncode=1, cmd="test")
        mock_logprob.return_value = True
        with pytest.raises(SystemExit):
            compare_impl(
                _make_compare_args(
                    baseline_logprob_dir=Path("/bl"),
                    target_logprob_dir=Path("/tg"),
                )
            )
        mock_logprob.assert_called_once()
