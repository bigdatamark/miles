# [MODULE DOC] Tests for the legacy (v1) command utility module that wraps the current backends:
# the legacy train config (positional fields cuda core dump, node count, extra env vars and
# output directory) is converted to the current config before launch and forwarded with the
# pre-submit hook and job lifetime; re-exports (hardware resolution works through the legacy
# module and the module advertises all its public names in its export list); the legacy free
# functions (cpu, gpu and multi node command helpers, checkpoint conversion, rsync, remote Ray
# worker start, dataset download, FP8 cast, NVLink check) reach the shell and Ray node boundaries
# which are replaced by recorders; and the Ray backend factory rejects non-Ray configs.
import dataclasses
from pathlib import Path
from typing import Literal

import pytest

import miles.utils.external_utils.command_utils.base_backend as base_backend
import miles.utils.external_utils.command_utils.legacy as legacy
import miles.utils.external_utils.command_utils.ray_backend.backend as ray_backend
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig as CurrentExecuteTrainConfig
from miles.utils.workers.types import ClusterBackend


class _RecordingBackend:
    # [CLASS DOC] HELPER: Fake backend that records the keyword arguments of each train call.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of recorded calls.
        self.calls: list[dict[str, object]] = []

    def execute_train(self, **kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: Appends the keyword arguments mapping to the recorded calls.
        self.calls.append(kwargs)


class TestExecuteTrainConfig:
    # [CLASS DOC] Test of the legacy train config conversion.
    @pytest.mark.parametrize("external_mooncake", [False, True])
    def test_positional_v1_config_is_converted_before_launch(
        self, monkeypatch: pytest.MonkeyPatch, external_mooncake: bool
    ) -> None:
        # [FUNCTION DOC] Parametrized over the external Mooncake flag. A legacy config built
        # positionally (core dump on, 4 nodes, an env assignment string, an output directory) plus
        # the keyword external mooncake flag is launched through the legacy train function with a
        # recording backend factory. Asserts: the legacy config's positional (non keyword only)
        # fields are exactly core dump, node count, extra env vars, output dir in that order; the
        # converted current config has the same values and the mooncake flag; and the backend got
        # that same config object, the same pre-submit hook function and the launcher job lifetime.
        """The v1 positional field order and values must reach the current backend unchanged."""
        config = legacy.ExecuteTrainConfig(True, 4, "MY_VAR=value", "/output", external_mooncake=external_mooncake)
        backend = _RecordingBackend()
        current_configs: list[CurrentExecuteTrainConfig] = []

        def before_ray_job_submit() -> None:
            # [FUNCTION DOC] HELPER: Placeholder hook that does nothing; only its identity is checked.
            pass

        def _create_backend(current_config: CurrentExecuteTrainConfig) -> _RecordingBackend:
            # [FUNCTION DOC] HELPER: Fake backend factory; records the converted config and returns
            # the shared recording backend.
            current_configs.append(current_config)
            return backend

        monkeypatch.setattr(legacy, "_create_ray_backend", _create_backend)

        legacy.execute_train(
            train_args="--train-backend fsdp",
            num_gpus_per_node=8,
            megatron_model_type=None,
            config=config,
            before_ray_job_submit=before_ray_job_submit,
            job_lifetime="launcher",
        )

        assert [field.name for field in dataclasses.fields(legacy.ExecuteTrainConfig) if not field.kw_only] == [
            "cuda_core_dump",
            "num_nodes",
            "extra_env_vars",
            "output_dir",
        ]
        current_config = current_configs[0]
        assert current_config.cuda_core_dump is True
        assert current_config.external_mooncake is external_mooncake
        assert current_config.num_nodes == 4
        assert current_config.extra_env_vars == "MY_VAR=value"
        assert current_config.output_dir == "/output"
        assert backend.calls[0]["config"] is current_config
        assert backend.calls[0]["before_ray_job_submit"] is before_ray_job_submit
        assert backend.calls[0]["job_lifetime"] == "launcher"


@dataclasses.dataclass
class _LauncherConfig:
    # [CLASS DOC] HELPER: Minimal launcher config with a hardware field allowing auto or h100
    # (default h100), the only thing hardware resolution reads.
    """The shape of a v1 launcher's own config: it carries the --hardware option resolve_hardware reads."""

    hardware: Literal["auto", "h100"] = "h100"


class TestWhatTheV1ModuleReExports:
    # [CLASS DOC] Tests of names the legacy module re-exports.
    def test_a_v1_launcher_can_resolve_its_hardware_through_this_module(self):
        # [FUNCTION DOC] The legacy module's hardware resolution function is the very same object
        # as the base backend's.
        """Every launcher that leaves --hardware on auto calls U.resolve_hardware(self) while building its config."""
        assert legacy.resolve_hardware is base_backend.resolve_hardware

    def test_it_answers_for_a_launcher_config_that_names_its_hardware(self):
        # [FUNCTION DOC] Resolving a launcher config naming h100 returns h100.
        """The v1 api carried this function, so importing it is not enough; it has to work through here."""
        assert legacy.resolve_hardware(_LauncherConfig()) == "h100"

    def test_it_refuses_a_hardware_the_launcher_has_no_profile_for(self):
        # [FUNCTION DOC] Resolving a launcher config naming gb200 (outside its allowed values)
        # raises an assertion error saying there is no verified profile.
        """A launcher that reaches this through the v1 module gets the same check as one that does not."""
        with pytest.raises(AssertionError, match="no verified profile"):
            legacy.resolve_hardware(_LauncherConfig(hardware="gb200"))

    def test_every_name_the_module_advertises_is_a_name_it_has(self):
        # [FUNCTION DOC] Every name in the legacy module's export list exists as an attribute of
        # the module (the list of missing names is empty).
        """__all__ is what a v1 launcher's star import reads, and a name missing from the module is an error."""
        assert [name for name in legacy.__all__ if not hasattr(legacy, name)] == []

    def test_the_re_export_is_advertised_rather_than_only_imported(self):
        # [FUNCTION DOC] The hardware resolution name is listed in the legacy module's export list.
        """A star import is how the launch script guide tells older ray launchers to reach these."""
        assert "resolve_hardware" in legacy.__all__


class TestLegacyFreeFunctions:
    # [CLASS DOC] Tests of the legacy module's free functions, with the shell and Ray node command
    # boundaries replaced by recorders (provided by the command boundaries fixture).
    @pytest.fixture
    def command_boundaries(self, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
        # [FUNCTION DOC] HELPER: Fixture; patches the shell command runner in both the base and
        # Ray backend modules with a recorder, and the Ray backend's all nodes execution with a
        # second recorder. Returns the pair (recorded shell commands, recorded multi node command
        # descriptions). Patches are undone at teardown.
        shell_commands: list[str] = []
        multi_node_commands: list[str] = []

        def run_shell_command(cmd: str, capture_output: bool = False) -> str:
            # [FUNCTION DOC] HELPER: Fake shell runner; records the command and returns the text 7
            # followed by a newline when output is captured, else the word completed.
            shell_commands.append(cmd)
            return "7\n" if capture_output else "completed"

        def exec_command_all_ray_nodes(
            cmd: str, capture_output: bool = False, num_nodes: int | None = None
        ) -> list[str]:
            # [FUNCTION DOC] HELPER: Fake all-nodes execution; records a description with the node
            # count, capture flag and command, and returns two strings stating the node count and
            # the capture flag.
            multi_node_commands.append(f"nodes={num_nodes} capture={capture_output} command={cmd}")
            return [f"nodes={num_nodes}", f"capture={capture_output}"]

        monkeypatch.setattr(base_backend, "run_shell_command", run_shell_command)
        monkeypatch.setattr(ray_backend, "run_shell_command", run_shell_command)
        monkeypatch.setattr(ray_backend, "exec_command_all_ray_nodes", exec_command_all_ray_nodes)
        return shell_commands, multi_node_commands

    def test_exec_command_cpu_returns_captured_shell_output(
        self, command_boundaries: tuple[list[str], list[str]]
    ) -> None:
        # [FUNCTION DOC] The legacy cpu command helper with capture returns the faked shell output
        # and ran exactly the given command once.
        """The v1 CPU helper must expose output from the host command boundary."""
        shell_commands, _ = command_boundaries

        assert legacy.exec_command_cpu("cpu command", capture_output=True) == "7\n"
        assert shell_commands == ["cpu command"]

    def test_exec_command_gpu_returns_captured_shell_output(
        self, command_boundaries: tuple[list[str], list[str]]
    ) -> None:
        # [FUNCTION DOC] The legacy gpu command helper with capture returns the faked shell output
        # and ran exactly the given command once.
        """The v1 GPU helper must expose output from the host command boundary."""
        shell_commands, _ = command_boundaries

        assert legacy.exec_command_gpu("gpu command", capture_output=True) == "7\n"
        assert shell_commands == ["gpu command"]

    def test_exec_command_multi_node_returns_ray_node_outputs(
        self, command_boundaries: tuple[list[str], list[str]]
    ) -> None:
        # [FUNCTION DOC] The legacy multi node helper with capture and 2 nodes returns the faked
        # Ray results and the recorder saw the node count, capture flag and command.
        """The v1 multi-node helper must expose results from the Ray node boundary."""
        _, multi_node_commands = command_boundaries

        assert legacy.exec_command_multi_node("cluster command", capture_output=True, num_nodes=2) == [
            "nodes=2",
            "capture=True",
        ]
        assert multi_node_commands == ["nodes=2 capture=True command=cluster command"]

    def test_convert_checkpoint_reaches_the_ray_node_boundary(
        self, command_boundaries: tuple[list[str], list[str]], tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Converting a checkpoint (multi node, 2 nodes, 8 GPUs, extra args, a
        # temporary destination, an HF checkpoint path and a Megatron path) issues exactly one
        # multi node command for 2 nodes without output capture. The command text includes 8
        # processes per node, the node count placeholder, the HF checkpoint option and the extra
        # args.
        """The v1 checkpoint helper must launch the conversion on the requested Ray nodes."""
        _, multi_node_commands = command_boundaries

        legacy.convert_checkpoint(
            "model",
            "qwen2.5-7B",
            8,
            multinode=True,
            num_nodes=2,
            extra_args="--trust-remote-code",
            dir_dst=str(tmp_path),
            hf_checkpoint="/hf/model",
            megatron_path="/megatron",
        )

        assert len(multi_node_commands) == 1
        assert multi_node_commands[0].startswith("nodes=2 capture=False command=")
        assert "--nproc-per-node 8" in multi_node_commands[0]
        assert "--nnodes={{nnodes}}" in multi_node_commands[0]
        assert "--hf-checkpoint /hf/model" in multi_node_commands[0]
        assert "--trust-remote-code" in multi_node_commands[0]

    def test_rsync_simple_reaches_every_requested_ray_node(
        self, command_boundaries: tuple[list[str], list[str]]
    ) -> None:
        # [FUNCTION DOC] The rsync helper over 4 nodes issues one multi node command that creates
        # the destination directory and rsyncs the source directory contents with progress output.
        """The v1 rsync helper must copy source contents to the destination on Ray nodes."""
        _, multi_node_commands = command_boundaries

        legacy.rsync_simple("/source", "/destination", num_nodes=4)

        assert multi_node_commands == [
            "nodes=4 capture=False command=mkdir -p /destination && " "rsync -a --info=progress2 /source/ /destination"
        ]

    def test_ssh_start_ray_workers_reaches_the_host_shell(
        self, command_boundaries: tuple[list[str], list[str]]
    ) -> None:
        # [FUNCTION DOC] Starting remote Ray workers for a head address and 8 GPUs issues exactly
        # one shell command containing the ray start address (on the standard port 6379) and the
        # GPU count.
        """The v1 worker helper must start remote Ray workers through the host shell."""
        shell_commands, _ = command_boundaries

        legacy.ssh_start_ray_workers("10.0.0.1", 8)

        assert len(shell_commands) == 1
        assert "ray start --address=10.0.0.1:6379 --num-gpus 8" in shell_commands[0]

    def test_hf_download_dataset_reaches_the_host_shell(self, command_boundaries: tuple[list[str], list[str]]) -> None:
        # [FUNCTION DOC] Downloading an org slash dataset repository issues exactly one shell
        # command that downloads the dataset repo type into the datasets directory under a folder
        # named after the dataset part only.
        """The v1 dataset helper must download into a directory named after the dataset."""
        shell_commands, _ = command_boundaries

        legacy.hf_download_dataset("org/dataset")

        assert shell_commands == ["hf download --repo-type dataset org/dataset --local-dir /root/datasets/dataset"]

    def test_fp8_cast_bf16_reaches_the_host_shell(
        self, command_boundaries: tuple[list[str], list[str]], tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Casting FP8 to BF16 into an existing empty temporary directory (no model
        # index file) issues exactly one shell command with the input FP8 path and the output BF16
        # path options.
        """The v1 FP8 helper must launch a cast when the destination has no index."""
        shell_commands, _ = command_boundaries
        destination = str(tmp_path)

        legacy.fp8_cast_bf16("/fp8", destination)

        assert len(shell_commands) == 1
        assert "--input-fp8-hf-path /fp8" in shell_commands[0]
        assert f"--output-bf16-hf-path {destination}" in shell_commands[0]

    def test_check_has_nvlink_reports_the_system_probe_result(
        self, command_boundaries: tuple[list[str], list[str]]
    ) -> None:
        # [FUNCTION DOC] With the faked shell output 7 (a nonzero link count), the NVLink check
        # returns true, and the one shell command is the nvidia-smi topology query that counts
        # NVLink entries.
        """The v1 NVLink helper must interpret the captured nvidia-smi link count."""
        shell_commands, _ = command_boundaries

        assert legacy.check_has_nvlink() is True
        assert shell_commands == ["nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l"]


class TestCreateRayBackend:
    # [CLASS DOC] Test of the legacy private Ray backend factory.
    def test_the_v1_api_rejects_a_non_ray_backend(self) -> None:
        # [FUNCTION DOC] Creating a Ray backend from a Kubernetes config raises an assertion error
        # mentioning the v1 command utils API.
        """The v1 API must reject Kubernetes before attempting to create its backend."""
        config = CurrentExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES)

        with pytest.raises(AssertionError, match="v1 command_utils API"):
            legacy._create_ray_backend(config)
