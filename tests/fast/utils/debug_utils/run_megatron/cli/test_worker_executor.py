# [MODULE DOC] Tests for the worker executor helpers of the run_megatron tool
# (miles.utils.debug_utils.run_megatron.cli.worker_executor): environment dictionary for the
# dumper, Megatron flag string (parallel sizes, sequence length, batch sizes, bf16, routing replay,
# sequence parallel), worker argument string (Megatron flags plus script arguments plus extra
# arguments) and the torchrun shell command (model arguments loaded by a mocked loader, process
# count, worker arguments, Megatron on the Python path).
from pathlib import Path
from unittest.mock import patch

from tests.fast.utils.debug_utils.run_megatron.conftest import make_script_args

from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig
from miles.utils.debug_utils.run_megatron.cli.worker_executor import (
    _build_megatron_flags,
    build_dumper_env,
    build_torchrun_cmd,
    build_worker_args,
)


class TestBuildDumperEnv:
    # [CLASS DOC] Tests of the dumper environment variable dictionary.
    def test_basic_env_vars(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Always sets the dumper enable variable to "1", the directory to the output
        # directory string and the experiment name to "standalone".
        env = build_dumper_env(output_dir=tmp_path, run_backward=False, dumper_filter="")
        assert env["DUMPER_ENABLE"] == "1"
        assert env["DUMPER_DIR"] == str(tmp_path)
        assert env["DUMPER_EXP_NAME"] == "standalone"

    def test_filter_when_non_empty(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A non-empty filter string is exported as the dumper filter variable.
        env = build_dumper_env(output_dir=tmp_path, run_backward=False, dumper_filter="hidden_states")
        assert env["DUMPER_FILTER"] == "hidden_states"

    def test_filter_omitted_when_empty(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] An empty filter leaves the dumper filter variable out.
        env = build_dumper_env(output_dir=tmp_path, run_backward=False, dumper_filter="")
        assert "DUMPER_FILTER" not in env

    def test_backward_enables_grad(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With run-backward true, the model-grad enable variable is "1".
        env = build_dumper_env(output_dir=tmp_path, run_backward=True, dumper_filter="")
        assert env["DUMPER_ENABLE_MODEL_GRAD"] == "1"

    def test_no_grad_forward_only(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With run-backward false, the model-grad variable is absent.
        env = build_dumper_env(output_dir=tmp_path, run_backward=False, dumper_filter="")
        assert "DUMPER_ENABLE_MODEL_GRAD" not in env


class TestBuildMegatronFlags:
    # [CLASS DOC] Tests of the private Megatron flag string builder.
    def test_parallel_sizes(self) -> None:
        # [FUNCTION DOC] A config with tp 2, pp 4, cp 2, ep 8, etp 2 produces the tensor, pipeline,
        # context, expert and expert-tensor parallel size flags with those values.
        flags = _build_megatron_flags(
            parallel=ParallelConfig(tp=2, pp=4, cp=2, ep=8, etp=2),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--tensor-model-parallel-size 2" in flags
        assert "--pipeline-model-parallel-size 4" in flags
        assert "--context-parallel-size 2" in flags
        assert "--expert-model-parallel-size 8" in flags
        assert "--expert-tensor-parallel-size 2" in flags

    def test_seq_length_batch_size(self) -> None:
        # [FUNCTION DOC] Sequence length 256 and batch size 4 give flags for sequence length 256 and
        # both micro and global batch size 4.
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=256,
            batch_size=4,
            use_routing_replay=False,
        )
        assert "--seq-length 256" in flags
        assert "--micro-batch-size 4" in flags
        assert "--global-batch-size 4" in flags

    def test_bf16_always(self) -> None:
        # [FUNCTION DOC] The bf16 flag is always present.
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--bf16" in flags

    def test_routing_replay_on(self) -> None:
        # [FUNCTION DOC] Routing replay enabled adds the use-routing-replay flag.
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=True,
        )
        assert "--use-routing-replay" in flags

    def test_routing_replay_off(self) -> None:
        # [FUNCTION DOC] Routing replay disabled omits the use-routing-replay flag.
        flags = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--use-routing-replay" not in flags

    def test_sp_on_off(self) -> None:
        # [FUNCTION DOC] The sequence-parallel flag is present when enabled and absent when disabled.
        flags_on = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=True,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--sequence-parallel" in flags_on

        flags_off = _build_megatron_flags(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            use_routing_replay=False,
        )
        assert "--sequence-parallel" not in flags_off


class TestBuildWorkerArgs:
    # [CLASS DOC] Tests of the worker argument string builder (Megatron flags, script arguments, extra text).
    def test_includes_parallel_flags(self) -> None:
        # [FUNCTION DOC] The result contains the tensor parallel size flag for tp 2.
        result = build_worker_args(
            parallel=ParallelConfig(tp=2),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=make_script_args(),
            extra_args="",
        )
        assert "--tensor-model-parallel-size 2" in result

    def test_includes_script_args(self) -> None:
        # [FUNCTION DOC] Script args with checkpoint "/my/hf" produce "--script-hf-checkpoint /my/hf".
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=make_script_args(hf_checkpoint=Path("/my/hf")),
            extra_args="",
        )
        assert "--script-hf-checkpoint /my/hf" in result

    def test_extra_args_appended(self) -> None:
        # [FUNCTION DOC] Extra arguments text is appended at the very end of the result.
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=make_script_args(),
            extra_args="--custom-flag value",
        )
        assert result.endswith("--custom-flag value")

    def test_empty_extra_args(self) -> None:
        # [FUNCTION DOC] With empty extra arguments the result has no trailing space.
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=make_script_args(),
            extra_args="",
        )
        assert not result.endswith(" ")

    def test_routing_replay_flag(self) -> None:
        # [FUNCTION DOC] Script args with a routing replay dump path make the result contain the
        # use-routing-replay flag.
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=make_script_args(routing_replay_dump_path=Path("/dump")),
            extra_args="",
        )
        assert "--use-routing-replay" in result

    def test_no_routing_replay(self) -> None:
        # [FUNCTION DOC] Script args without any replay path give a result without the flag.
        result = build_worker_args(
            parallel=ParallelConfig(),
            sp=False,
            seq_length=128,
            batch_size=1,
            script_args=make_script_args(),
            extra_args="",
        )
        assert "--use-routing-replay" not in result


class TestBuildTorchrunCmd:
    # [CLASS DOC] Tests of the torchrun command builder; the model argument loader is patched in
    # each test to return "--num-layers 61".
    @patch("miles.utils.debug_utils.run_megatron.cli.worker_executor.load_model_args")
    def test_basic_structure(self, mock_load: object) -> None:
        # [FUNCTION DOC] For model type deepseek_v3, Megatron at "/megatron", 4 processes and worker
        # arguments "--foo bar", the command contains "torchrun", the loaded model arguments and the
        # Python path variable.
        mock_load.return_value = "--num-layers 61"  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="deepseek_v3",
            megatron_path=Path("/megatron"),
            nproc=4,
            worker_args="--foo bar",
        )
        assert "torchrun" in cmd
        assert "--num-layers 61" in cmd
        assert "PYTHONPATH" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.worker_executor.load_model_args")
    def test_nproc(self, mock_load: object) -> None:
        # [FUNCTION DOC] Process count 8 yields "--nproc-per-node 8" in the command.
        mock_load.return_value = "--num-layers 61"  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="test",
            megatron_path=Path("/megatron"),
            nproc=8,
            worker_args="",
        )
        assert "--nproc-per-node 8" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.worker_executor.load_model_args")
    def test_worker_args_in_cmd(self, mock_load: object) -> None:
        # [FUNCTION DOC] The given worker arguments text appears in the command.
        mock_load.return_value = "--num-layers 61"  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="test",
            megatron_path=Path("/megatron"),
            nproc=1,
            worker_args="--my-flag 42",
        )
        assert "--my-flag 42" in cmd

    @patch("miles.utils.debug_utils.run_megatron.cli.worker_executor.load_model_args")
    def test_megatron_in_pythonpath(self, mock_load: object) -> None:
        # [FUNCTION DOC] With Megatron at "/my/megatron" the command contains "PYTHONPATH=/my/megatron".
        mock_load.return_value = "--num-layers 61"  # type: ignore[union-attr]
        cmd = build_torchrun_cmd(
            model_type="test",
            megatron_path=Path("/my/megatron"),
            nproc=1,
            worker_args="",
        )
        assert "PYTHONPATH=/my/megatron" in cmd
