# [MODULE DOC] Tests for helpers of the run-and-compare command
# (miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare): running a baseline and a
# target configuration through run_impl (mocked) with per-run output directories, optional routing
# replay wiring (baseline dumps the replay, target loads it, baseline must be single-rank), logprob
# output directories, and extra per-side arguments; and appending extra arguments to a dictionary
# of common run fields.
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.commands.args import CommonRunArgs
from miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare import (
    _append_extra_args,
    _run_baseline_and_target,
)
from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig


def _make_common_fields(**overrides: object) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Build the dictionary form of common run arguments (model type
    # "deepseek_v3", checkpoint "/fake/hf", other fields default) via the dataclass-to-dictionary
    # conversion, then apply the keyword overrides. Returns a new dictionary on every call.
    base = CommonRunArgs(
        model_type="deepseek_v3",
        hf_checkpoint=Path("/fake/hf"),
    )
    result = dataclasses.asdict(base)
    result.update(overrides)
    return result


class TestRunAndCompare:
    # [CLASS DOC] Tests of the baseline/target runner with the run implementation mocked; the
    # recorded calls' first positional argument is the run arguments object.
    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl")
    def test_calls_run_twice(self, mock_run: MagicMock) -> None:
        # [FUNCTION DOC] Running baseline (tp 1) and target (tp 2) with no replay results in exactly
        # two calls of the run implementation.
        _run_baseline_and_target(
            baseline_config=ParallelConfig(tp=1),
            target_config=ParallelConfig(tp=2),
            baseline_output=Path("/tmp/baseline"),
            target_output=Path("/tmp/target"),
            replay_dir=None,
            common_fields=_make_common_fields(),
            baseline_extra_args="",
            target_extra_args="",
            baseline_logprob_dir=None,
            target_logprob_dir=None,
        )
        assert mock_run.call_count == 2

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl")
    def test_output_dir_uses_dir_name(self, mock_run: MagicMock) -> None:
        # [FUNCTION DOC] Output paths are built from a base directory plus each parallel config's
        # directory name; asserts the first call's output directory contains the baseline's name and
        # the second call's contains the target's name (baseline run first).
        baseline_config = ParallelConfig(tp=1)
        target_config = ParallelConfig(tp=2)

        _run_baseline_and_target(
            baseline_config=baseline_config,
            target_config=target_config,
            baseline_output=Path("/tmp/out") / baseline_config.dir_name(),
            target_output=Path("/tmp/out") / target_config.dir_name(),
            replay_dir=None,
            common_fields=_make_common_fields(),
            baseline_extra_args="",
            target_extra_args="",
            baseline_logprob_dir=None,
            target_logprob_dir=None,
        )

        baseline_args = mock_run.call_args_list[0][0][0]
        target_args = mock_run.call_args_list[1][0][0]

        assert baseline_config.dir_name() in str(baseline_args.output_dir)
        assert target_config.dir_name() in str(target_args.output_dir)

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl")
    def test_replay_baseline_nproc1_required(self, mock_run: MagicMock) -> None:
        # [FUNCTION DOC] With a replay directory and a baseline of tensor parallel 2, a value error
        # mentioning "single-rank baseline" is raised.
        with pytest.raises(ValueError, match="single-rank baseline"):
            _run_baseline_and_target(
                baseline_config=ParallelConfig(tp=2),
                target_config=ParallelConfig(tp=2),
                baseline_output=Path("/tmp/baseline"),
                target_output=Path("/tmp/target"),
                replay_dir=Path("/tmp/replay"),
                common_fields=_make_common_fields(),
                baseline_extra_args="",
                target_extra_args="",
                baseline_logprob_dir=None,
                target_logprob_dir=None,
            )

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl")
    def test_replay_paths_passed_correctly(self, mock_run: MagicMock) -> None:
        # [FUNCTION DOC] With a replay directory and a single-rank baseline, the baseline run gets a
        # routing replay dump path and no load path, and the target run gets a load path and no dump path.
        _run_baseline_and_target(
            baseline_config=ParallelConfig(tp=1),
            target_config=ParallelConfig(tp=2),
            baseline_output=Path("/tmp/baseline"),
            target_output=Path("/tmp/target"),
            replay_dir=Path("/tmp/replay"),
            common_fields=_make_common_fields(),
            baseline_extra_args="",
            target_extra_args="",
            baseline_logprob_dir=None,
            target_logprob_dir=None,
        )

        baseline_args = mock_run.call_args_list[0][0][0]
        target_args = mock_run.call_args_list[1][0][0]

        assert baseline_args.routing_replay_dump_path is not None
        assert baseline_args.routing_replay_load_path is None
        assert target_args.routing_replay_dump_path is None
        assert target_args.routing_replay_load_path is not None

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl")
    def test_logprob_dirs_wired_when_provided(self, mock_run: MagicMock) -> None:
        # [FUNCTION DOC] Logprob directories passed for baseline and target end up as the logprob
        # output of the first and second run respectively.
        _run_baseline_and_target(
            baseline_config=ParallelConfig(tp=1),
            target_config=ParallelConfig(tp=2),
            baseline_output=Path("/tmp/baseline"),
            target_output=Path("/tmp/target"),
            replay_dir=None,
            common_fields=_make_common_fields(),
            baseline_extra_args="",
            target_extra_args="",
            baseline_logprob_dir=Path("/tmp/baseline/logprobs"),
            target_logprob_dir=Path("/tmp/target/logprobs"),
        )

        baseline_args = mock_run.call_args_list[0][0][0]
        target_args = mock_run.call_args_list[1][0][0]

        assert baseline_args.logprob_output == Path("/tmp/baseline/logprobs")
        assert target_args.logprob_output == Path("/tmp/target/logprobs")

    @patch("miles.utils.debug_utils.run_megatron.cli.commands.run_and_compare.run_impl")
    def test_extra_args_appended_to_baseline_and_target(self, mock_run: MagicMock) -> None:
        # [FUNCTION DOC] Per-side extra arguments "--baseline-flag" and "--target-flag" appear in the
        # extra arguments of the baseline and target runs respectively.
        _run_baseline_and_target(
            baseline_config=ParallelConfig(tp=1),
            target_config=ParallelConfig(tp=2),
            baseline_output=Path("/tmp/baseline"),
            target_output=Path("/tmp/target"),
            replay_dir=None,
            common_fields=_make_common_fields(),
            baseline_extra_args="--baseline-flag",
            target_extra_args="--target-flag",
            baseline_logprob_dir=None,
            target_logprob_dir=None,
        )

        baseline_args = mock_run.call_args_list[0][0][0]
        target_args = mock_run.call_args_list[1][0][0]

        assert "--baseline-flag" in baseline_args.extra_args
        assert "--target-flag" in target_args.extra_args


class TestAppendExtraArgs:
    # [CLASS DOC] Tests of the helper that appends extra arguments text to the common fields.
    def test_appends_extra_to_existing_extra_args(self) -> None:
        # [FUNCTION DOC] Existing extra arguments "--foo" plus "--bar" give "--foo --bar".
        result = _append_extra_args(
            _make_common_fields(extra_args="--foo"),
            "--bar",
        )
        assert result["extra_args"] == "--foo --bar"

    def test_empty_extra_preserves_original(self) -> None:
        # [FUNCTION DOC] Appending an empty string to fields with extra arguments "--foo" leaves the
        # result equal to freshly built fields with the same value.
        # KNOWN ISSUE: the identity comparison against a freshly built dictionary is always false
        # (a new dictionary is created each call), so only the equality half of the assertion matters.
        result = _append_extra_args(
            _make_common_fields(extra_args="--foo"),
            "",
        )
        assert result is _make_common_fields(extra_args="--foo") or result == _make_common_fields(extra_args="--foo")

    def test_empty_both_returns_original(self) -> None:
        # [FUNCTION DOC] With empty existing extra arguments and an empty addition, the very same
        # dictionary object is returned (no copy).
        common = _make_common_fields(extra_args="")
        result = _append_extra_args(common, "")
        assert result is common
