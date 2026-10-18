# [MODULE DOC] Tests for the run command implementation (run_impl in
# miles.utils.debug_utils.run_megatron.cli.commands.run), which builds a torchrun shell command that
# launches the Megatron worker with dumper environment exports and script arguments. Validation:
# routing replay requires a single rank. Command tests mock only the shell runner, token id
# generation, token file writing and model argument loading; command assembly itself runs for real.
from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from miles.utils.debug_utils.run_megatron.cli.commands.args import RunArgs
from miles.utils.debug_utils.run_megatron.cli.commands.run import run_impl


def _make_run_args(**overrides: object) -> RunArgs:
    # [FUNCTION DOC] HELPER: Build run arguments with model type "deepseek_v3", Hugging Face
    # checkpoint "/fake/hf" and output directory "/tmp/dump", updated by keyword overrides.
    defaults = dict(
        model_type="deepseek_v3",
        hf_checkpoint=Path("/fake/hf"),
        output_dir=Path("/tmp/dump"),
    )
    defaults.update(overrides)
    return RunArgs(**defaults)  # type: ignore[arg-type]


class TestRunImplValidation:
    # [CLASS DOC] Tests of argument validation in run_impl (no mocks; validation fails before any
    # command runs).
    def test_routing_replay_requires_nproc1(self) -> None:
        # [FUNCTION DOC] Routing replay with tensor parallel 2 raises a value error mentioning
        # "single-rank".
        with pytest.raises(ValueError, match="single-rank"):
            run_impl(
                _make_run_args(
                    tp=2,
                    routing_replay_dump_path=Path("/dump"),
                )
            )

    def test_routing_replay_pp_gt1_also_fails(self) -> None:
        # [FUNCTION DOC] Routing replay with pipeline parallel 2 raises a value error mentioning
        # "single-rank".
        with pytest.raises(ValueError, match="single-rank"):
            run_impl(
                _make_run_args(
                    pp=2,
                    routing_replay_dump_path=Path("/dump"),
                )
            )


class TestRunImplExecCommand:
    # [CLASS DOC] Tests of the shell command text produced by run_impl.
    """Only mock run_shell_command, generate_token_ids, write_token_ids_to_tmpfile,
    and load_model_args — let the rest (build_worker_args, build_dumper_env,
    build_torchrun_cmd, ParallelConfig, WorkerScriptArgs) run for real."""

    @pytest.fixture(autouse=True)
    def _patch_externals(self) -> Generator[None, None, None]:
        # [FUNCTION DOC] HELPER: Autouse fixture applying four patches for the duration of each
        # test: the run module's shell runner (kept as an attribute of the test instance for
        # inspection), token id generation (returns the integers 0 to 199), token file writing
        # (returns "/tmp/tokens.json") and the worker executor's model argument loader (returns
        # "--num-layers 61"). Yields while the patches are active.
        with (
            patch("miles.utils.debug_utils.run_megatron.cli.commands.run.run_shell_command") as mock_exec,
            patch(
                "miles.utils.debug_utils.run_megatron.cli.commands.run.generate_token_ids",
                return_value=list(range(200)),
            ),
            patch(
                "miles.utils.debug_utils.run_megatron.cli.commands.run.write_token_ids_to_tmpfile",
                return_value=Path("/tmp/tokens.json"),
            ),
            patch(
                "miles.utils.debug_utils.run_megatron.cli.worker_executor.load_model_args",
                return_value="--num-layers 61",
            ),
        ):
            self.mock_exec = mock_exec
            yield

    def test_cmd_contains_torchrun(self) -> None:
        # [FUNCTION DOC] Default arguments: the shell runner is called once and the command contains "torchrun".
        run_impl(_make_run_args())
        self.mock_exec.assert_called_once()
        cmd = self.mock_exec.call_args[0][0]
        assert "torchrun" in cmd

    def test_cmd_contains_parallel_config(self) -> None:
        # [FUNCTION DOC] Tensor parallel 4 and pipeline parallel 2 give flags for both sizes and a
        # processes-per-node count of 8.
        run_impl(_make_run_args(tp=4, pp=2))
        cmd = self.mock_exec.call_args[0][0]
        assert "--tensor-model-parallel-size 4" in cmd
        assert "--pipeline-model-parallel-size 2" in cmd
        assert "--nproc-per-node 8" in cmd

    def test_cmd_contains_dumper_env(self) -> None:
        # [FUNCTION DOC] With output directory "/out/test" and dumper filter "logits", the command
        # contains single-quoted exports enabling the dumper (value 1), setting its directory and its filter.
        run_impl(_make_run_args(output_dir=Path("/out/test"), dumper_filter="logits"))
        cmd = self.mock_exec.call_args[0][0]
        assert "export DUMPER_ENABLE='1'" in cmd
        assert "export DUMPER_DIR='/out/test'" in cmd
        assert "export DUMPER_FILTER='logits'" in cmd

    def test_cmd_contains_script_args(self) -> None:
        # [FUNCTION DOC] With role "critic" and top-k 5, the command contains "--script-role critic"
        # and "--script-top-k 5".
        run_impl(_make_run_args(role="critic", top_k=5))
        cmd = self.mock_exec.call_args[0][0]
        assert "--script-role critic" in cmd
        assert "--script-top-k 5" in cmd

    def test_routing_replay_nproc1_ok(self) -> None:
        # [FUNCTION DOC] Routing replay with tensor parallel 1 is accepted and the command contains
        # the use-routing-replay flag.
        run_impl(
            _make_run_args(
                tp=1,
                routing_replay_dump_path=Path("/dump"),
            )
        )
        cmd = self.mock_exec.call_args[0][0]
        assert "--use-routing-replay" in cmd

    def test_backward_enables_grad_env(self) -> None:
        # [FUNCTION DOC] With the run-backward option, the command exports the dumper model-grad
        # enable variable with value 1.
        run_impl(_make_run_args(run_backward=True))
        cmd = self.mock_exec.call_args[0][0]
        assert "export DUMPER_ENABLE_MODEL_GRAD='1'" in cmd
