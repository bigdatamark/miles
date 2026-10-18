# [MODULE DOC] Unit tests for the external command utilities used by launch scripts: the command
# backend (ray backend by default; helm/kubernetes backend is checked only for being safely
# recorded) with execute_train, convert_checkpoint, hf_download_dataset, fp8_cast_bf16, Mooncake
# master startup and prepare commands; rsync command building; the train environment variable
# merge order; extra env var parsing; NVLink detection; wandb args; run id creation; boolean env
# parsing; pseudo file encoding; hardware tables and detection; and building the execute-train
# config from the environment. Commands are never executed: a recorder fixture captures the shell
# commands the backend would run. No GPU or cluster is needed.
import json
import os
import platform
import shlex
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from tests.fast.launch_scripts.py_harness import launcher_hardware_literals
from tests.fast.utils.command_recorder import patch_helper, record_commands


import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils.command_utils import base_backend, common
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainRequest
from miles.utils.external_utils.command_utils.ray_backend import command as ray_command
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.file_arg_utils import resolve_file_arg
from miles.utils.typer_utils import SCRIPT_ENV_VAR_PREFIX


def _backend():
    # [FUNCTION DOC] HELPER: Creates and returns a command backend from the default launcher config
    # (ray backend unless the environment says otherwise).
    return command_utils.default_config().create_backend()


@pytest.fixture(autouse=True)
def bare_environment(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture: removes every environment variable whose name starts
    # with the launcher script prefix, so a developer's pod environment cannot change the selected
    # backend. Side effect: monkeypatched environment (restored after each test).
    """Every test here launches onto ray, and a workbench pod exports variables that choose another backend."""
    for name in [key for key in os.environ if key.startswith(SCRIPT_ENV_VAR_PREFIX)]:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def commands(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture returning the list that records every shell command the
    # backend would run (the recorder replaces real execution). Also makes NVLink detection report
    # false for the ray backend, unsets the Ray address, NCCL NVLS and wandb key variables, enables
    # Ray job submission, and sets the master address to localhost.
    recorded = record_commands(monkeypatch)
    patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)
    for name in ("RAY_ADDRESS", "NCCL_NVLS_ENABLE", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    return recorded


def _runtime_env(submit_command):
    # [FUNCTION DOC] HELPER: From a recorded job submit command line, finds the argument that
    # carries the runtime-env JSON, parses it and returns its mapping of environment variables.
    # Raises a stop-iteration error if the argument is absent.
    arg = next(arg for arg in shlex.split(submit_command) if arg.startswith("--runtime-env-json="))
    return json.loads(arg.split("=", 1)[1])["env_vars"]


def _submit(commands):
    # [FUNCTION DOC] HELPER: Returns the last recorded command containing a Ray job submit; fails
    # the test with an assertion message listing all recorded commands if there is none.
    submits = [command for command in commands if "ray job submit" in command]
    assert submits, f"no ray job submit was recorded, only {commands}"
    return submits[-1]


class TestTheRecorderCoversEveryBackendTheEnvironmentCanChoose:
    # [CLASS DOC] Verifies that with the kubernetes environment variables set, the command recorder
    # still intercepts every command, so no Job is installed on a real cluster, and that new
    # backends cannot override the guarded GPU execution hook.
    @pytest.fixture
    def submitted_jobs(self, monkeypatch) -> list[str]:
        # [FUNCTION DOC] HELPER: Fixture that replaces the helm backend's run-on-nodes function with a
        # fake appending each command to a list and returning a one-element list holding none; yields
        # that list of submitted commands.
        """Stand in for the cluster: record what would have been installed instead of installing it."""
        from miles.utils.external_utils.command_utils.helm_backend import command_job

        submitted: list[str] = []

        def fake_run_on_nodes(context, cmd: str, **kwargs) -> list[str | None]:
            # [FUNCTION DOC] HELPER: Records the command and returns a result list with none.
            submitted.append(cmd)
            return [None]

        monkeypatch.setattr(command_job, "run_on_nodes", fake_run_on_nodes)
        return submitted

    @pytest.fixture
    def kubernetes_environment(self, monkeypatch) -> None:
        # [FUNCTION DOC] HELPER: Fixture setting the cluster backend variable to "kubernetes" and the
        # namespace variable to a fake namespace.
        """A workbench pod exports exactly these, and dataclass_from_env reads them like command line options."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", "kubernetes")
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-someones-namespace")

    @pytest.fixture
    def recorded(self, monkeypatch, kubernetes_environment, submitted_jobs) -> list[str]:
        # [FUNCTION DOC] HELPER: Fixture that installs the command recorder after the kubernetes
        # environment and the fake job submitter are in place, and returns the recorded list.
        return record_commands(monkeypatch)

    def test_a_kubernetes_environment_records_a_gpu_command_instead_of_installing_a_job(
        self, recorded, submitted_jobs
    ):
        # [FUNCTION DOC] Under the kubernetes environment, executing a single-node GPU command
        # ("nvidia-smi", 8 GPUs per node) is recorded verbatim and no job is submitted.
        """A stub naming only the ray backend leaves the kubernetes one live, and a fast test then takes real gpus."""
        _backend().exec_command_gpu("nvidia-smi", num_gpus_per_node=8)

        assert submitted_jobs == []
        assert recorded == ["nvidia-smi"]

    def test_a_kubernetes_environment_records_a_multi_node_command_instead_of_installing_a_job(
        self, recorded, submitted_jobs
    ):
        # [FUNCTION DOC] Under the kubernetes environment, a multi-node command (2 nodes, 8 GPUs each,
        # with a node-count placeholder) is recorded with a "[multi_node num_nodes=2]" prefix and no
        # job is submitted.
        """The multi-node form is the one convert_checkpoint reaches, i.e. the one that really installed a Job."""
        _backend().exec_command_multi_node("torchrun --nnodes={{nnodes}}", num_nodes=2, num_gpus_per_node=8)

        assert submitted_jobs == []
        assert recorded == ["[multi_node num_nodes=2] torchrun --nnodes={{nnodes}}"]

    def test_a_kubernetes_environment_converts_a_checkpoint_without_reaching_the_cluster(
        self, recorded, submitted_jobs, tmp_path
    ):
        # [FUNCTION DOC] Under the kubernetes environment, a multi-node checkpoint conversion (Qwen3-4B,
        # 2 nodes) into a temporary directory records exactly one command and submits no job.
        """This is the observed incident: the recorded tmp_path showed up in a Job installed on a real namespace."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=8,
            multinode=True,
            num_nodes=2,
            dir_dst=str(tmp_path),
        )

        assert submitted_jobs == []
        assert len(recorded) == 1

    def test_a_backend_added_later_is_refused_the_override_that_caused_this(self):
        # [FUNCTION DOC] Declaring a subclass of the base command backend that overrides the public
        # GPU execution method raises an assertion error (at class definition) telling the author to
        # implement the inner hook instead.
        """Enumerating backends in the stub is what rotted, so the ban has to be enforced where they are declared."""
        with pytest.raises(AssertionError, match="implement _exec_command_gpu_inner instead"):

            class SlurmCommandBackend(base_backend.BaseCommandBackend):
                # [CLASS DOC] HELPER: Deliberately invalid backend subclass used only to trigger the
                # definition-time check.
                def exec_command_gpu(self, cmd, capture_output=False, num_gpus_per_node=None): ...


class TestExecuteTrainConfig:
    # [CLASS DOC] Tests of defaults of the execute-train config object.
    def test_num_nodes_reads_the_slurm_allocation_when_the_config_is_built(self, monkeypatch):
        # [FUNCTION DOC] With the Slurm node-count variable set to 8, a freshly built config has 8
        # nodes; after removing the variable, a new config has 1 node (default evaluated at build
        # time, not import time).
        """A plain class-level default would bake in the allocation at import and ignore later changes."""
        monkeypatch.setenv("SLURM_JOB_NUM_NODES", "8")
        assert command_utils.ExecuteTrainConfig().num_nodes == 8

        monkeypatch.delenv("SLURM_JOB_NUM_NODES")
        assert command_utils.ExecuteTrainConfig().num_nodes == 1


class TestConvertCheckpoint:
    # [CLASS DOC] Tests of the backend's convert_checkpoint: source path defaulting, skipping already
    # converted release checkpoints (tracker file plus layout version stamp), stamping new
    # conversions and single versus multi-node command templates.
    def test_preserves_source_paths_on_the_pythonpath(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] With PYTHONPATH set to two entries and a custom Megatron path, the GPU
        # command generated for a conversion begins with a PYTHONPATH assignment (shell quoted) made
        # of the repo base dir, the Megatron path, then the pre-existing entries, joined with the path
        # separator. The GPU executor is replaced by a collector (accepting any keyword arguments).
        """The converter runs out-of-process, so miles and megatron must be on its PYTHONPATH."""
        commands = []
        monkeypatch.setenv("PYTHONPATH", "/sglang:/existing")
        patch_helper(
            monkeypatch,
            "exec_command_gpu",
            lambda self, cmd, capture_output=False, **kwargs: commands.append(cmd),
        )

        _backend().convert_checkpoint(
            model_name="model",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=1,
            dir_dst=str(tmp_path),
            megatron_path="/megatron",
        )

        expected = os.pathsep.join([str(command_utils.repo_base_dir), "/megatron", "/sglang", "/existing"])
        assert f"PYTHONPATH={shlex.quote(expected)} " in commands[0]

    def test_defaults_the_hf_checkpoint_to_the_model_name(self, commands, tmp_path):
        # [FUNCTION DOC] Without an explicit HF checkpoint the command reads from the models root
        # directory joined with the model name and saves to the destination dir joined with "<model
        # name>_torch_dist".
        """Callers that only pass a model name get /root/models/<model_name> as the source."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert "--hf-checkpoint /root/models/Qwen3-4B " in commands[0]
        assert f"--save {tmp_path}/Qwen3-4B_torch_dist " in commands[0]

    def test_an_explicit_hf_checkpoint_wins_over_the_default(self, commands, tmp_path):
        # [FUNCTION DOC] An explicit HF checkpoint path is used in the command and the default models
        # root does not appear anywhere in it.
        """Callers converting a checkpoint that does not live under /root/models must be honoured."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=8,
            dir_dst=str(tmp_path),
            hf_checkpoint="/elsewhere/Qwen3-4B",
        )

        assert "--hf-checkpoint /elsewhere/Qwen3-4B " in commands[0]
        assert "/root/models" not in commands[0]

    def test_skips_an_already_released_destination(self, commands, tmp_path):
        # [FUNCTION DOC] Creates the destination directory with a latest-iteration tracker file
        # containing "release" and the current conversion version stamp file; asserts that no
        # command is recorded (conversion skipped).
        """A tracker reading 'release' plus the current layout stamp means the conversion already finished."""
        dst = tmp_path / "Qwen3-4B_torch_dist"
        dst.mkdir()
        (dst / "latest_checkpointed_iteration.txt").write_text("release\n")
        (dst / base_backend.CONVERSION_VERSION_FILE).write_text(base_backend.CONVERSION_VERSION)

        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert commands == []

    def test_a_fresh_conversion_stamps_the_layout(self, commands, tmp_path):
        # [FUNCTION DOC] After a conversion into an empty destination, the version stamp file inside
        # the converted directory exists and holds the current conversion version string.
        """The stamp the skip path reads is written once the conversion has run."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        stamp = tmp_path / "Qwen3-4B_torch_dist" / base_backend.CONVERSION_VERSION_FILE
        assert stamp.read_text() == base_backend.CONVERSION_VERSION

    @pytest.mark.parametrize("stamp_text", [None, "0"], ids=["unstamped", "stale"])
    def test_reconverts_a_release_written_before_the_current_layout(self, commands, tmp_path, stamp_text):
        # [FUNCTION DOC] Parametrized by stamp text (no stamp file, or the stale stamp "0"). With a
        # "release" tracker but a missing or stale stamp, the conversion runs again: exactly one
        # command is recorded.
        """A cached conversion from before a layout change would load scrambled, so it is redone."""
        dst = tmp_path / "Qwen3-4B_torch_dist"
        dst.mkdir()
        (dst / "latest_checkpointed_iteration.txt").write_text("release\n")
        if stamp_text is not None:
            (dst / base_backend.CONVERSION_VERSION_FILE).write_text(stamp_text)

        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert len(commands) == 1

    def test_reruns_when_the_tracker_holds_an_iteration(self, commands, tmp_path):
        # [FUNCTION DOC] A tracker file containing the iteration number "42" (not "release") does not
        # count as done: exactly one conversion command is recorded.
        """Only the literal 'release' marker counts as done; an iteration number does not."""
        dst = tmp_path / "Qwen3-4B_torch_dist"
        dst.mkdir()
        (dst / "latest_checkpointed_iteration.txt").write_text("42")

        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert len(commands) == 1

    def test_multinode_uses_torchrun_rendezvous_placeholders(self, commands, tmp_path):
        # [FUNCTION DOC] A 2-node conversion with extra args "--extra 1" produces a command containing
        # the master address, node count and node rank template placeholders (double-brace tokens)
        # and ending with the extra args.
        """Multi-node conversion must template the placeholders exec_command_multi_node substitutes."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B",
            megatron_model_type="qwen3-4B",
            num_gpus_per_node=8,
            multinode=True,
            num_nodes=2,
            dir_dst=str(tmp_path),
            extra_args="--extra 1",
        )

        assert "--master-addr {{master_addr}}" in commands[0]
        assert "--nnodes={{nnodes}}" in commands[0]
        assert "--node-rank {{node_rank}}" in commands[0]
        assert commands[0].endswith("--extra 1")

    def test_single_node_omits_the_rendezvous_placeholders(self, commands, tmp_path):
        # [FUNCTION DOC] A single-node conversion command contains no master address flag.
        """A single-node conversion has nothing to rendezvous with."""
        _backend().convert_checkpoint(
            model_name="Qwen3-4B", megatron_model_type="qwen3-4B", num_gpus_per_node=8, dir_dst=str(tmp_path)
        )

        assert "--master-addr" not in commands[0]


class TestRsyncCmd:
    # [CLASS DOC] Tests of the rsync command string builder (pure string output).
    def test_creates_the_destination_before_copying(self):
        # [FUNCTION DOC] The command for source "/src" and destination "/dst" is exactly: make the
        # destination directory (with parents), then archive-mode rsync with progress info copying
        # "/src/" into "/dst".
        """rsync fails on a missing destination, so the mkdir has to precede it."""
        assert command_utils.rsync_cmd("/src", "/dst") == "mkdir -p /dst && rsync -a --info=progress2 /src/ /dst"

    def test_copies_the_contents_rather_than_nesting_the_source(self):
        # [FUNCTION DOC] The command ends with the source given a trailing slash followed by the
        # destination, so the contents are copied instead of the directory being nested.
        """Without the trailing slash rsync would land the tree at /dst/src and every reader would miss it."""
        assert command_utils.rsync_cmd("/src", "/dst").endswith(" /src/ /dst")


class TestHfDownloadDataset:
    # [CLASS DOC] Tests of the backend's Hugging Face dataset download command.
    def test_strips_the_namespace_from_the_local_dir(self, commands):
        # [FUNCTION DOC] Downloading "zhuzilin/dapo-math-17k" into "/data" records exactly one
        # command: the hf download for the dataset repo type with local dir "/data/dapo-math-17k".
        """The local directory is named after the dataset, not after owner/dataset."""
        _backend().hf_download_dataset("zhuzilin/dapo-math-17k", data_dir="/data")

        assert commands == ["hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /data/dapo-math-17k"]


class TestFp8CastBf16:
    # [CLASS DOC] Tests of the backend's FP8-to-BF16 cast helper.
    def test_skips_when_the_output_index_already_exists(self, commands, tmp_path):
        # [FUNCTION DOC] If the output directory already contains a safetensors index file, no
        # command is recorded.
        """A safetensors index in the destination means the cast already ran."""
        (tmp_path / "model.safetensors.index.json").write_text("{}")

        _backend().fp8_cast_bf16("/src", str(tmp_path))

        assert commands == []

    def test_runs_when_the_output_is_absent(self, commands, tmp_path):
        # [FUNCTION DOC] Without the index file the first recorded command carries the input FP8
        # path flag and the output BF16 path flag with the given paths.
        """Without the index file the cast must actually be invoked."""
        _backend().fp8_cast_bf16("/src", str(tmp_path))

        assert "--input-fp8-hf-path /src " in commands[0]
        assert f"--output-bf16-hf-path {tmp_path} " in commands[0]


class TestStartMooncakeMaster:
    # [CLASS DOC] Tests of Mooncake master startup in the ray backend: routing via the command
    # recorder, external/remote and local masters, port parsing, reuse of a ready server, metrics
    # port choice, restart and wait, and log reporting on failure.
    @pytest.fixture
    def unready_master(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Fixture making the ray command module report that no TCP server is
        # ready, and replacing its shell runner and server-ready waiter with no-ops.
        monkeypatch.setattr(ray_command, "_is_tcp_server_ready", lambda host, port: False)
        monkeypatch.setattr(ray_command, "run_shell_command", lambda *args, **kwargs: None)
        monkeypatch.setattr(ray_command, "wait_for_server_ready", lambda *args, **kwargs: None)

    def test_the_backend_routes_startup_through_its_command_boundary(self, commands, monkeypatch):
        # [FUNCTION DOC] Runs execute_train with the Mooncake object store backend while the module's
        # direct shell runner is replaced by a collector. Asserts the collector got nothing and the
        # recorded backend commands include a Mooncake master start command with an RPC port flag.
        """Backend recorders must see Mooncake startup instead of letting a fast test execute it on the host."""
        direct_commands: list[str] = []
        monkeypatch.setattr(ray_command, "_is_tcp_server_ready", lambda host, port: False)
        monkeypatch.setattr(ray_command, "run_shell_command", direct_commands.append)
        monkeypatch.setattr(ray_command, "wait_for_server_ready", lambda *args, **kwargs: None)

        _backend().execute_train(
            train_args="--object-store-backend mooncake",
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
        )

        assert direct_commands == []
        assert any("mooncake_master --rpc_port" in command for command in commands)

    def test_a_remote_master_the_caller_configured_is_left_alone(self, commands, unready_master):
        # [FUNCTION DOC] With the external Mooncake flag in the config and a remote master address in
        # the store init kwargs, no Mooncake master start command is recorded.
        """The endpoint is the caller's to run, and probing localhost would replace an unrelated process."""
        _backend().execute_train(
            train_args=(
                "--object-store-backend mooncake "
                '--mooncake-store-init-kwargs \'{"master_server_address": "mooncake-host:60051"}\''
            ),
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
            config=command_utils.ExecuteTrainConfig(external_mooncake=True),
        )

        assert not any("mooncake_master --rpc_port" in command for command in commands)

    def test_a_local_master_is_started_on_the_port_the_caller_named(self, commands, unready_master):
        # [FUNCTION DOC] A master address "127.0.0.1:60051" in the init kwargs (given with an equals
        # sign form of the flag) leads to a master start command using RPC port 60051.
        """Starting the default port instead would leave the run pointing at a master nobody runs."""
        _backend().execute_train(
            train_args=(
                "--object-store-backend mooncake "
                '--mooncake-store-init-kwargs=\'{"master_server_address": "127.0.0.1:60051"}\''
            ),
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
        )

        assert any("mooncake_master --rpc_port 60051" in command for command in commands)

    def test_an_address_whose_port_is_not_a_number_fails_the_launch(self, commands, unready_master):
        # [FUNCTION DOC] An address "mooncake-host:notaport" raises an assertion error matching "must
        # be host:port" and no command at all is recorded.
        """A typo in the caller's address names no endpoint at all, so the launch must fail before it starts."""
        with pytest.raises(AssertionError, match="must be host:port"):
            _backend().execute_train(
                train_args=(
                    "--object-store-backend mooncake "
                    '--mooncake-store-init-kwargs \'{"master_server_address": "mooncake-host:notaport"}\''
                ),
                num_gpus_per_node=1,
                megatron_model_type="qwen3-4B",
            )

        assert commands == []

    def test_a_local_master_named_by_an_ipv6_literal_is_started(self, commands, unready_master):
        # [FUNCTION DOC] An address in bracketed IPv6 loopback form with port 60051 is recognized as
        # local and a master start command using RPC port 60051 is recorded.
        """`[::1]:port` is the only way to write a loopback IPv6 endpoint, and it names this host too."""
        _backend().execute_train(
            train_args=(
                "--object-store-backend mooncake "
                '--mooncake-store-init-kwargs=\'{"master_server_address": "[::1]:60051"}\''
            ),
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
        )

        assert any("mooncake_master --rpc_port 60051" in command for command in commands)

    def test_reuses_a_ready_server(self, monkeypatch):
        # [FUNCTION DOC] With the readiness probe patched to true, start_mooncake_master runs no shell
        # command and does not wait for readiness.
        """An already listening master must not be restarted out from under its clients."""
        commands = []
        waits = []
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command._is_tcp_server_ready",
            lambda host, port: True,
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.run_shell_command", commands.append
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.wait_for_server_ready",
            lambda *args, **kwargs: waits.append((args, kwargs)),
        )

        ray_command.start_mooncake_master()

        assert commands == []
        assert waits == []

    def test_lets_the_os_choose_the_metrics_port(self, monkeypatch):
        # [FUNCTION DOC] With the probe patched to false and collectors for the shell runner and
        # waiter, a default start runs one command with RPC port 50051 and metrics port 0, and waits
        # once for ("127.0.0.1", 50051) with a 30 s timeout.
        """Binding port zero atomically avoids collisions with other listeners."""
        commands = []
        waits = []
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command._is_tcp_server_ready",
            lambda host, port: False,
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.run_shell_command", commands.append
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.wait_for_server_ready",
            lambda *args, **kwargs: waits.append((args, kwargs)),
        )

        ray_command.start_mooncake_master()

        assert len(commands) == 1
        assert "mooncake_master --rpc_port 50051 --metrics_port 0" in commands[0]
        assert waits == [(("127.0.0.1", 50051), {"timeout": 30})]

    def test_restarts_and_waits_until_ready(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] With explicit ports 50151/50152, timeout 12 and a log path containing a
        # space, one command is run that kills any old master for that RPC port (anchored pattern),
        # starts the master with both ports, and redirects output to the shell-quoted log path in the
        # background; the waiter is called once for ("127.0.0.1", 50151) with timeout 12.
        """A dead master is replaced and the caller blocks until the new one answers."""
        commands = []
        waits = []
        log_path = tmp_path / "mooncake master.log"
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command._is_tcp_server_ready",
            lambda host, port: False,
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.run_shell_command", commands.append
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.wait_for_server_ready",
            lambda *args, **kwargs: waits.append((args, kwargs)),
        )

        ray_command.start_mooncake_master(rpc_port=50151, metrics_port=50152, timeout=12, log_path=log_path)

        assert len(commands) == 1
        assert "pkill -f '^mooncake_master --rpc_port 50151 '" in commands[0]
        assert "mooncake_master --rpc_port 50151 --metrics_port 50152" in commands[0]
        assert f"> {shlex.quote(str(log_path))} 2>&1 &" in commands[0]
        assert waits == [(("127.0.0.1", 50151), {"timeout": 12})]

    def test_reports_the_log_when_startup_fails(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] The log file holds two lines and the waiter raises a runtime error "not
        # ready". Asserts the raised runtime error message includes the log text ("fatal startup
        # error"), that two commands were run (start and a second kill/cleanup) and that both carry
        # the anchored kill pattern for RPC port 50051.
        """The log is the only clue about why the master refused to come up."""
        log_path = tmp_path / "mooncake_master.log"
        log_path.write_text("bind failed\nfatal startup error\n")
        commands = []
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command._is_tcp_server_ready",
            lambda host, port: False,
        )
        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.run_shell_command", commands.append
        )

        def fail_wait(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement waiter that always raises a runtime error "not
            # ready".
            raise RuntimeError("not ready")

        monkeypatch.setattr(
            "miles.utils.external_utils.command_utils.ray_backend.command.wait_for_server_ready", fail_wait
        )

        with pytest.raises(RuntimeError, match="fatal startup error"):
            ray_command.start_mooncake_master(log_path=log_path)

        assert len(commands) == 2
        assert all("pkill -f '^mooncake_master --rpc_port 50051 '" in command for command in commands)


class TestPrepareCmd:
    # [CLASS DOC] Tests of the per-role prepare commands of execute_train (staging commands run on
    # every node).
    def test_runs_the_command_on_every_node_before_submitting_the_job(self, commands):
        # [FUNCTION DOC] With a prepare command for the trainer role, finds the recorded index of
        # that command (ends with it) and of the job submit; asserts the prepare command is a
        # multi-node command and comes before the submit.
        """The trainers read what the command stages, so a job submitted first would read nothing."""
        _backend().execute_train(
            train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B", prepare_cmd={"trainer": "cp a b"}
        )

        prepared = next(index for index, command in enumerate(commands) if command.endswith("cp a b"))
        submitted = next(index for index, command in enumerate(commands) if "ray job submit" in command)
        assert commands[prepared].startswith("[multi_node ")
        assert prepared < submitted

    def test_runs_the_command_after_ray_is_up(self, commands):
        # [FUNCTION DOC] The recorded index of the head start command is smaller than that of the
        # prepare command (ray must be up before multi-node execution).
        """exec_command_multi_node reaches the nodes through ray, which a preceding command cannot use."""
        _backend().execute_train(
            train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B", prepare_cmd={"trainer": "cp a b"}
        )

        started = next(index for index, command in enumerate(commands) if "ray start --head" in command)
        prepared = next(index for index, command in enumerate(commands) if command.endswith("cp a b"))
        assert started < prepared

    def test_leaves_the_run_alone_when_nothing_is_asked_for(self, commands):
        # [FUNCTION DOC] Without any prepare command, no recorded command is a multi-node command.
        """A run that stages nothing must not gain an empty command on every node."""
        _backend().execute_train(train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B")

        assert not any(command.startswith("[multi_node ") for command in commands)

    def test_rejects_a_role_no_backend_knows_how_to_prepare(self, commands):
        # [FUNCTION DOC] A prepare command keyed by the unknown role "rollout" raises an assertion
        # error whose message mentions "rollout".
        """A misspelled role would be silently dropped and the run would read unstaged data."""
        with pytest.raises(AssertionError, match="rollout"):
            _backend().execute_train(
                train_args="",
                num_gpus_per_node=1,
                megatron_model_type="qwen3-4B",
                prepare_cmd={"rollout": "cp a b"},
            )


class TestExtraManifests:
    # [CLASS DOC] Tests that the ray backend rejects extra Kubernetes manifests.
    def test_a_ray_launch_refuses_the_manifests_it_could_only_ignore(self, commands):
        # [FUNCTION DOC] Passing a non-empty extra manifests list to the ray backend raises an
        # assertion error mentioning "extra_manifests".
        """Nothing installs them without a helm release, and a silently missing engine looks like a hang."""
        with pytest.raises(AssertionError, match="extra_manifests"):
            _backend().execute_train(
                train_args="",
                num_gpus_per_node=1,
                megatron_model_type="qwen3-4B",
                extra_manifests=["apiVersion: v1\nkind: Service\n"],
            )


class TestExecuteTrain:
    # [CLASS DOC] Tests of the ray backend's execute_train: ray cluster lifecycle, environment
    # variables in the runtime env (Python unbuffered, paths, CUDA connections, NCCL, proxy, core
    # dumps, caller versus config precedence), model argument expansion and quoting, FSDP and
    # torchtitan handling, the job submit command shape and the before-submit callback.
    @pytest.mark.parametrize("external_ray", [False, True])
    def test_launcher_failure_stops_only_a_cluster_it_owns(
        self,
        commands: list[str],
        monkeypatch: pytest.MonkeyPatch,
        unavailable_ray_job_client: None,
        external_ray: bool,
    ) -> None:
        # [FUNCTION DOC] Uses a fixture (defined outside this file, presumably in a conftest) that
        # makes the Ray job client unavailable. Sets the external-ray variable from the parameter
        # and runs execute_train with launcher job lifetime; asserts a runtime error "Ray job
        # service unavailable", that a forced ray stop was recorded only when ray is not external,
        # and that no job submit was recorded.
        """A failed launcher-owned job cleans up local Ray without stopping a borrowed cluster."""
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", str(int(external_ray)))

        with pytest.raises(RuntimeError, match="Ray job service unavailable"):
            _backend().execute_train(
                train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B", job_lifetime="launcher"
            )

        assert ("ray stop --force" in commands) is not external_ray
        assert not any("ray job submit" in command for command in commands)

    def test_removed_router_env_is_checked_in_the_launch_override(self, commands: list[str]) -> None:
        # [FUNCTION DOC] A config whose extra env vars text sets the removed router external host
        # variable makes execute_train raise a value error naming the session-server external host
        # flag, and nothing is recorded.
        """A per-launch config cannot bypass rejection of an obsolete router address."""
        config = command_utils.ExecuteTrainConfig(extra_env_vars="MILES_ROUTER_EXTERNAL_HOST=trainer.tailnet")

        with pytest.raises(ValueError, match="--session-server-external-host"):
            _backend().execute_train(train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B", config=config)

        assert commands == []

    def test_a_leftover_router_external_host_fails_before_launching(self, commands, monkeypatch):
        # [FUNCTION DOC] With the removed router external host variable exported in the process
        # environment, execute_train raises a value error naming the session-server external host
        # flag and records no command.
        """Nothing reads the removed variable, so an old export would silently send agents to the placed address."""
        monkeypatch.setenv("MILES_ROUTER_EXTERNAL_HOST", "trainer.tailnet")

        with pytest.raises(ValueError, match="--session-server-external-host"):
            _backend().execute_train(train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B")

        assert commands == []

    def test_exports_unbuffered_python_to_ray(self, commands):
        # [FUNCTION DOC] Of the recorded commands, exactly two (ray start and job submit) export the
        # unbuffered-Python variable, none contains the misspelled variant, and each export sets it
        # to 1 followed by a command chain.
        """Ray start and job submit must export the correctly spelled PYTHONUNBUFFERED."""

        _backend().execute_train(
            train_args="",
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
        )

        exports = [command for command in commands if "export PYTHONUNBUFFERED" in command]
        assert len(exports) == 2
        assert not any("PYTHONBUFFERED" in command for command in commands)
        assert all("export PYTHONUNBUFFERED=1 &&" in command for command in exports)

    def test_unbuffers_the_ray_workers_too(self, commands, monkeypatch):
        # [FUNCTION DOC] With external ray, the runtime-env JSON argument of the submit command has
        # the unbuffered-Python variable set to "1".
        """An export only reaches the submitting client; the ray workers read the runtime environment."""
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")

        _backend().execute_train(train_args="", num_gpus_per_node=1, megatron_model_type="qwen3-4B")

        runtime_env_arg = next(arg for arg in shlex.split(_submit(commands)) if arg.startswith("--runtime-env-json="))
        assert json.loads(runtime_env_arg.split("=", 1)[1])["env_vars"]["PYTHONUNBUFFERED"] == "1"

    def test_preserves_source_paths_in_the_ray_runtime(self, commands, monkeypatch):
        # [FUNCTION DOC] With process PYTHONPATH of two entries, a custom Megatron path and caller
        # extra env vars (a PYTHONPATH of two entries and a value containing an apostrophe), asserts
        # the runtime env PYTHONPATH is the repo base dir, the Megatron path, then the caller's
        # entries, then remaining process entries (order and de-duplication by path joining), and the
        # apostrophe value survives intact through shell quoting.
        """A caller-supplied PYTHONPATH must be prepended to, not replace, the checkouts miles needs."""
        monkeypatch.setenv("PYTHONPATH", "/sglang:/existing")
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")

        _backend().execute_train(
            train_args="",
            num_gpus_per_node=1,
            megatron_model_type="qwen3-4B",
            megatron_path="/megatron",
            extra_env_vars={"PYTHONPATH": "/custom:/sglang", "QUOTED_VALUE": "it's preserved"},
        )

        submit_command = _submit(commands)
        runtime_env_arg = next(arg for arg in shlex.split(submit_command) if arg.startswith("--runtime-env-json="))
        runtime_env = json.loads(runtime_env_arg.split("=", 1)[1])
        expected = os.pathsep.join([str(command_utils.repo_base_dir), "/megatron", "/custom", "/sglang", "/existing"])
        assert runtime_env["env_vars"]["PYTHONPATH"] == expected
        assert runtime_env["env_vars"]["QUOTED_VALUE"] == "it's preserved"

    def test_rejects_fsdp_with_a_megatron_model_type(self, commands):
        # [FUNCTION DOC] A train backend FSDP flag together with a non-empty Megatron model type
        # raises an assertion error.
        """FSDP runs have no megatron model config, so a model type means the launcher is confused."""
        with pytest.raises(AssertionError):
            _backend().execute_train(
                train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type="qwen"
            )

    def test_rejects_megatron_without_a_model_type(self, commands):
        # [FUNCTION DOC] The default (Megatron) backend with a model type of none raises an assertion
        # error.
        """Without a model type the submitted job would carry no architecture flags at all."""
        with pytest.raises(AssertionError):
            _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type=None)

    def test_starts_a_local_ray_cluster_by_default(self, commands):
        # [FUNCTION DOC] Without external ray, the first recorded command contains a forced ray stop
        # and the second starts a head node on 127.0.0.1 with 8 GPUs and usage stats disabled.
        """Without MILES_SCRIPT_EXTERNAL_RAY the launcher owns the ray cluster lifecycle."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert "ray stop --force; " in commands[0]
        assert "ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats" in commands[1]

    def test_leaves_an_external_ray_cluster_alone(self, commands, monkeypatch):
        # [FUNCTION DOC] With the external-ray variable set to 1, no recorded command stops or starts
        # ray, and none kills ray processes.
        """With an external cluster we must neither stop nor start ray."""
        monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert not any("ray stop" in command or "ray start" in command for command in commands)
        assert not any("pkill -9 ray" in command for command in commands)

    def test_runs_the_callback_before_submitting(self, commands):
        # [FUNCTION DOC] The before-submit callback appends "CALLBACK" to the recorded list; asserts
        # the command just before it is the ray head start and that it precedes the job submit.
        """before_ray_job_submit exists to prepare state the job will read."""
        _backend().execute_train(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            before_ray_job_submit=lambda: commands.append("CALLBACK"),
        )

        assert "ray start --head" in commands[commands.index("CALLBACK") - 1]
        assert commands.index("CALLBACK") < commands.index(_submit(commands))

    def test_can_skip_the_ray_job_submit(self, commands, monkeypatch):
        # [FUNCTION DOC] With the submit-enable variable set to 0, no job submit is recorded and the
        # last recorded command is the ray head start.
        """Preparation-only runs leave the prepared Ray cluster intact for the later submission stage."""
        monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "0")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert not any("ray job submit" in command for command in commands)
        assert "ray start --head" in commands[-1]

    def test_a_finished_run_does_not_sweep_every_process_on_the_host(self, commands):
        # [FUNCTION DOC] The last recorded command is the job submit, and exactly one recorded command
        # force-kills sglang processes (the pre-launch cleanup, not a post-run sweep).
        """Run-owned workers clean themselves up, so the launcher must not kill unrelated concurrent processes."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert "ray job submit" in commands[-1]
        assert sum("pkill -9 sglang" in command for command in commands) == 1

    def test_expands_the_model_config_into_the_submitted_command(self, commands):
        # [FUNCTION DOC] For model type qwen3-4B with train args "--x 1", the submit command contains
        # the layer-count flag with 36 layers, no shell source statement, and ends with the train
        # args followed by the deploy-component flag set to all.
        """The megatron model type is expanded into the argv its model script declares."""
        _backend().execute_train(train_args="--x 1", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        submit = _submit(commands)
        assert "--num-layers 36 " in submit
        assert "source" not in submit
        assert submit.endswith("--x 1 --deploy-component all")

    def test_quotes_the_model_args_the_shell_would_otherwise_reinterpret(self, commands):
        # [FUNCTION DOC] For the 5-layer DeepSeek-V3 model type, the MoE layer frequency value (a
        # bracketed list that is a shell glob) appears single-quoted in the submit command, and
        # splitting the command with shell rules gives the unquoted value right after the flag.
        """--moe-layer-freq [1,1,1] is a glob; an unquoted token expands against the launch directory."""
        _backend().execute_train(train_args="--x 1", num_gpus_per_node=8, megatron_model_type="deepseek-v3-5layer")

        submit = _submit(commands)
        assert "--moe-layer-freq '[0,0,0,1,1]'" in submit
        assert shlex.split(submit)[shlex.split(submit).index("--moe-layer-freq") + 1] == "[0,0,0,1,1]"

    def test_model_args_survive_a_shell_round_trip_unchanged(self, commands):
        # [FUNCTION DOC] Splits the submit command with shell rules; asserts the last two tokens are
        # the deploy-component flag and "all", and that the tokens before them end with exactly the
        # declared model arguments (loaded from the model definition and split on whitespace).
        """Quoting is only correct if the training process still receives exactly the declared argv."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="deepseek-v3-5layer")

        declared = load_model_args("deepseek-v3-5layer").split()
        submitted = shlex.split(_submit(commands))
        model_argv = submitted[: len(submitted) - 2]

        assert submitted[len(submitted) - 2 :] == ["--deploy-component", "all"]
        assert model_argv[len(model_argv) - len(declared) :] == declared

    def test_omits_the_model_args_for_fsdp(self, commands):
        # [FUNCTION DOC] With the FSDP train backend and no model type, the submit command has no
        # layer-count flag.
        """FSDP has no megatron model config to expand."""
        _backend().execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)

        assert "--num-layers" not in _submit(commands)

    def test_drops_cuda_device_max_connections_for_fsdp(self, commands):
        # [FUNCTION DOC] With the FSDP backend the runtime env has no CUDA max-connections entry.
        """Pinning it to 1 breaks computation/communication overlap on FSDP."""
        _backend().execute_train(train_args="--train-backend fsdp", num_gpus_per_node=8, megatron_model_type=None)

        assert "CUDA_DEVICE_MAX_CONNECTIONS" not in _runtime_env(_submit(commands))

    def test_pins_cuda_device_max_connections_for_megatron(self, commands):
        # [FUNCTION DOC] With the Megatron backend the runtime env's CUDA max-connections is "1".
        """Megatron requires the serialized copy engine ordering."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(_submit(commands))["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"

    def test_derives_nvls_from_nvlink_detection(self, commands, monkeypatch):
        # [FUNCTION DOC] With NVLink detection patched to true and no preset variable, the runtime
        # env's NCCL NVLS enable is "1".
        """NCCL_NVLS_ENABLE follows the detected topology when it is not preset."""
        patch_helper(monkeypatch, "_check_has_nvlink", lambda self: True, backend_class=RayCommandBackend)

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(_submit(commands))["NCCL_NVLS_ENABLE"] == "1"

    def test_lets_the_environment_override_nvls(self, commands, monkeypatch):
        # [FUNCTION DOC] With detection true but the NVLS variable preset to "0" in the environment,
        # the runtime env's value is "0".
        """An explicit NCCL_NVLS_ENABLE wins over topology detection."""
        patch_helper(monkeypatch, "_check_has_nvlink", lambda self: True, backend_class=RayCommandBackend)
        monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert _runtime_env(_submit(commands))["NCCL_NVLS_ENABLE"] == "0"

    def test_forwards_selected_nccl_variables_only_when_present(self, commands, monkeypatch):
        # [FUNCTION DOC] With the NCCL socket interface set to "eth0" and NCCL debug unset, the
        # runtime env has the former and lacks the latter.
        """Optional debug knobs are passed through, and absent ones must not appear as empty strings."""
        monkeypatch.setenv("NCCL_SOCKET_IFNAME", "eth0")
        monkeypatch.delenv("NCCL_DEBUG", raising=False)

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["NCCL_SOCKET_IFNAME"] == "eth0"
        assert "NCCL_DEBUG" not in runtime_env

    def test_bypasses_the_proxy_for_the_master_node(self, commands, monkeypatch):
        # [FUNCTION DOC] With master address 10.0.0.1, the runtime env's no-proxy value is
        # "127.0.0.1,10.0.0.1" and its master address entry is "10.0.0.1".
        """Routing intra-cluster traffic through a proxy hangs the job."""
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["no_proxy"] == "127.0.0.1,10.0.0.1"
        assert runtime_env["MASTER_ADDR"] == "10.0.0.1"

    def test_enables_cuda_core_dumps_on_request(self, commands):
        # [FUNCTION DOC] With a config requesting CUDA core dumps and output dir "/out", the runtime
        # env enables coredump-on-exception ("1") and sets the core dump file to a path under /out
        # using host, process and time placeholders.
        """The core dump knobs only appear when the config asks for them."""
        config = command_utils.ExecuteTrainConfig(cuda_core_dump=True, output_dir="/out")

        config.create_backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1"
        assert runtime_env["CUDA_COREDUMP_FILE"] == "/out/cuda_coredump_%h.%p.%t"

    def test_omits_cuda_core_dumps_by_default(self, commands):
        # [FUNCTION DOC] By default the runtime env lacks the coredump-on-exception entry.
        """Core dumps are expensive, so they must stay off unless asked for."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert "CUDA_ENABLE_COREDUMP_ON_EXCEPTION" not in _runtime_env(_submit(commands))

    def test_lets_config_extra_env_vars_win_over_the_argument(self, commands):
        # [FUNCTION DOC] A config extra env var MY_VAR=from_config and an argument MY_VAR=from_argument
        # plus OTHER=kept: the runtime env has MY_VAR from the config and OTHER from the argument.
        """The CLI-supplied overrides are applied last so an operator can always override a script."""
        config = command_utils.ExecuteTrainConfig(extra_env_vars="MY_VAR=from_config")

        config.create_backend().execute_train(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            extra_env_vars={"MY_VAR": "from_argument", "OTHER": "kept"},
        )

        runtime_env = _runtime_env(_submit(commands))
        assert runtime_env["MY_VAR"] == "from_config"
        assert runtime_env["OTHER"] == "kept"

    def test_addresses_the_local_dashboard_unless_ray_address_is_set(self, commands, monkeypatch):
        # [FUNCTION DOC] Without the Ray address variable the submit command has an explicit address
        # flag for the local dashboard (127.0.0.1:8265); after setting the variable, a new launch's
        # submit command (the last one) has no address flag.
        """RAY_ADDRESS already tells the ray CLI where to go; passing --address too would conflict."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")
        assert '--address="http://127.0.0.1:8265"' in _submit(commands)

        monkeypatch.setenv("RAY_ADDRESS", "http://10.0.0.1:8265")
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")
        assert "--address=" not in _submit(commands)

    def test_resolves_a_relative_train_script_against_the_repo(self, commands):
        # [FUNCTION DOC] The default relative train script "train.py" is turned into an absolute path
        # under the repo base dir in the submit command's python invocation.
        """Launchers pass train.py, which only makes sense relative to the checkout."""
        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")

        assert f"-- python3 {command_utils.repo_base_dir}/train.py " in _submit(commands)

    def test_keeps_an_absolute_train_script(self, commands):
        # [FUNCTION DOC] An absolute train script path "/opt/train.py" is used unchanged in the
        # submit command's python invocation.
        """An absolute path is already unambiguous and must not be rewritten."""
        _backend().execute_train(
            train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B", train_script="/opt/train.py"
        )

        assert "-- python3 /opt/train.py " in _submit(commands)


class TestBuildTrainEnvVars:
    # [CLASS DOC] Tests of the shared train environment variable builder: key order, FSDP handling,
    # and the precedence backend < caller < config, plus optional core dump variables.
    @staticmethod
    def _request(**overrides):
        # [FUNCTION DOC] HELPER: Builds an execute-train request with fixed defaults (empty train
        # args, 8 GPUs per node, qwen3-4B, a train script path, non-FSDP, empty extra env vars,
        # Megatron path, no callback, empty prepare commands and manifests); keyword overrides
        # replace defaults. Returns the request object.
        defaults = dict(
            train_args="",
            num_gpus_per_node=8,
            megatron_model_type="qwen3-4B",
            train_script="/opt/train.py",
            train_backend_fsdp=False,
            extra_env_vars={},
            megatron_path="/root/Megatron-LM",
            before_ray_job_submit=None,
            prepare_cmd={},
            extra_manifests=[],
        )
        return ExecuteTrainRequest(**{**defaults, **overrides})

    def test_keeps_the_backend_vars_where_the_serialized_order_expects_them(self):
        # [FUNCTION DOC] With backend vars NCCL NVLS enable and master address, the resulting key
        # order is exactly: unbuffered Python, CUDA max connections, NCCL NVLS enable, master address.
        """The ray backend serializes this dict verbatim, so the launch snapshots pin its key order."""
        env = common.train_env_vars(
            self._request(),
            {"NCCL_NVLS_ENABLE": "0", "MASTER_ADDR": "10.0.0.1"},
            config=command_utils.ExecuteTrainConfig(),
        )

        assert list(env) == ["PYTHONUNBUFFERED", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "MASTER_ADDR"]

    def test_omits_the_connection_limit_for_fsdp(self):
        # [FUNCTION DOC] For an FSDP request the result lacks the CUDA max-connections entry.
        """Capping the connections breaks FSDP's computation and communication overlap."""
        env = common.train_env_vars(
            self._request(train_backend_fsdp=True), {}, config=command_utils.ExecuteTrainConfig()
        )

        assert "CUDA_DEVICE_MAX_CONNECTIONS" not in env

    def test_lets_the_caller_override_a_backend_var(self):
        # [FUNCTION DOC] A request extra var master address "caller" beats the backend's "backend".
        """extra_env_vars is merged last so a script can win over anything the backend chose."""
        env = common.train_env_vars(
            self._request(extra_env_vars={"MASTER_ADDR": "caller"}),
            {"MASTER_ADDR": "backend"},
            config=command_utils.ExecuteTrainConfig(),
        )

        assert env["MASTER_ADDR"] == "caller"

    def test_lets_the_config_override_the_caller(self):
        # [FUNCTION DOC] A config extra env var A=from-operator beats the request's A=from-script.
        """The operator's --extra-env-vars is the last word, above what the script hardcoded."""
        env = common.train_env_vars(
            self._request(extra_env_vars={"A": "from-script"}),
            {},
            config=command_utils.ExecuteTrainConfig(extra_env_vars="A=from-operator"),
        )

        assert env["A"] == "from-operator"

    def test_adds_the_coredump_vars_only_when_asked(self):
        # [FUNCTION DOC] With a core-dump config and output dir "/runs" the result enables
        # coredump-on-exception and sets the dump file under /runs; with a default config the
        # coredump entry is absent.
        """Core dumps are large, so they stay off until a run opts in."""
        config = command_utils.ExecuteTrainConfig(cuda_core_dump=True, output_dir="/runs")

        env = common.train_env_vars(self._request(), {}, config=config)

        assert env["CUDA_ENABLE_COREDUMP_ON_EXCEPTION"] == "1"
        assert env["CUDA_COREDUMP_FILE"] == "/runs/cuda_coredump_%h.%p.%t"
        assert "CUDA_ENABLE_COREDUMP_ON_EXCEPTION" not in common.train_env_vars(
            self._request(), {}, config=command_utils.ExecuteTrainConfig()
        )


class TestParseExtraEnvVars:
    # [CLASS DOC] Tests of the private parser for the operator's extra env vars text.
    @pytest.mark.parametrize(
        "text, expected",
        [
            ('{"A": "1", "B": "2"}', {"A": "1", "B": "2"}),
            ("A=1 B=2", {"A": "1", "B": "2"}),
            ("", {}),
            ("   ", {}),
        ],
    )
    def test_accepts_json_and_shell_style(self, text, expected):
        # [FUNCTION DOC] Parametrized: a JSON object, space-separated KEY=VALUE pairs, an empty string
        # and a whitespace-only string parse to the expected dict (empty inputs give an empty dict).
        """Operators pass either a JSON object or plain KEY=VALUE pairs."""
        assert common._parse_extra_env_vars(text) == expected


class TestCheckHasNvlink:
    # [CLASS DOC] Tests of the backend's private NVLink detection which runs a GPU command with
    # captured output and interprets the printed count.
    @pytest.fixture
    def nvlink_probe(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Fixture returning an install function. Calling install with an
        # output text patches the backend's GPU command execution with a fake returning that text
        # and recording each capture-output flag; install returns the list of recorded flags.
        def install(output: str) -> list[bool]:
            # [FUNCTION DOC] HELPER: Patches the GPU executor to return the given output; returns
            # the list that collects capture-output flags.
            captured = []

            def fake_exec_command(self, cmd: str, capture_output: bool = False, **kwargs) -> str:
                # [FUNCTION DOC] HELPER: Records the capture-output flag and returns the canned output.
                captured.append(capture_output)
                return output

            patch_helper(monkeypatch, "exec_command_gpu", fake_exec_command)
            return captured

        return install

    def test_reports_true_when_links_are_counted(self, nvlink_probe):
        # [FUNCTION DOC] Output "4" makes the check return true, and the command was run once with
        # output capture on.
        """A non-zero NVLink count from nvidia-smi means NVLink is present."""
        captured = nvlink_probe("4\n")

        assert _backend()._check_has_nvlink() is True
        assert captured == [True]

    def test_reports_false_without_links(self, nvlink_probe):
        # [FUNCTION DOC] Output "0" makes the check return false, with output capture on.
        """Without capture_output the real helper returns None and int(None) would abort the launch."""
        captured = nvlink_probe("0\n")

        assert _backend()._check_has_nvlink() is False
        assert captured == [True]


class TestGetDefaultWandbArgs:
    # [CLASS DOC] Tests of the default wandb argument string builder.
    def test_is_empty_without_an_api_key(self, monkeypatch):
        # [FUNCTION DOC] With the wandb API key unset, the result is an empty string.
        """Unconfigured wandb must not inject half-populated flags."""
        monkeypatch.delenv("WANDB_API_KEY", raising=False)

        assert command_utils.get_default_wandb_args("tests/fast/utils/test_thing.py") == ""

    def test_names_the_project_after_the_test_file(self, monkeypatch):
        # [FUNCTION DOC] With the key set and run id "RUNID" for test_qwen3_4b.py: args enable wandb,
        # name the project "miles-test_qwen3_4b", the group "RUNID" and pass the quoted key.
        """The project name is how runs are found later, so it tracks the test file."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        monkeypatch.delenv("GITHUB_COMMIT_NAME", raising=False)

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/test_qwen3_4b.py", run_id="RUNID")

        assert "--use-wandb " in args
        assert "--wandb-project miles-test_qwen3_4b " in args
        assert "--wandb-group RUNID " in args
        assert "--wandb-key 'secret' " in args

    def test_qualifies_a_short_test_name_with_its_directory(self, monkeypatch):
        # [FUNCTION DOC] For a file named run.py in the megatron directory, the project name becomes
        # "miles-megatron_run" (directory name prefixed because the stem is short).
        """Short stems like 'run.py' are ambiguous on their own."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/run.py", run_id="RUNID")

        assert "--wandb-project miles-megatron_run " in args

    def test_decorates_the_group_with_commit_and_prefix(self, monkeypatch):
        # [FUNCTION DOC] With commit name "abc123" in the environment, prefix "myprefix" and run id
        # "RUNID", the group is "myprefix_RUNID_abc123".
        """CI runs need the commit in the group name, and callers may add their own prefix."""
        monkeypatch.setenv("WANDB_API_KEY", "secret")
        monkeypatch.setenv("GITHUB_COMMIT_NAME", "abc123")

        args = command_utils.get_default_wandb_args("tests/e2e/megatron/test_qwen3_4b.py", "myprefix", run_id="RUNID")

        assert "--wandb-group myprefix_RUNID_abc123 " in args


class TestCreateRunId:
    # [CLASS DOC] Tests of run id creation (date-time-random format).
    def test_is_a_timestamp_with_a_random_suffix(self):
        # [FUNCTION DOC] The id splits on dashes into a 6-digit date, a 6-digit time and a 3-digit
        # random number (all numeric).
        """Concurrent runs on one machine must not collide on the run id."""
        date_part, time_part, random_part = command_utils.create_run_id().split("-")

        assert len(date_part) == 6 and date_part.isdigit()
        assert len(time_part) == 6 and time_part.isdigit()
        assert len(random_part) == 3 and random_part.isdigit()

    def test_varies_within_the_same_second(self):
        # [FUNCTION DOC] Fifty ids created in a loop have more than one distinct random suffix.
        # Note: probabilistic, with a tiny chance of failure if all suffixes collide.
        """Runs launched together in one second must not share a wandb group or dump directory."""
        suffixes = {command_utils.create_run_id().split("-")[2] for _ in range(50)}

        assert len(suffixes) > 1


class TestGetBoolEnvVar:
    # [CLASS DOC] Tests of boolean environment variable parsing.
    @pytest.mark.parametrize(
        "value, expected",
        [("true", True), ("TRUE", True), ("1", True), ("false", False), ("0", False), ("maybe", False)],
    )
    def test_understands_the_usual_spellings(self, monkeypatch, value, expected):
        # [FUNCTION DOC] Parametrized: "true", "TRUE" and "1" parse as true; "false", "0" and
        # "maybe" parse as false (no error for unknown text).
        """Anything not recognizably truthy is treated as false rather than raising."""
        monkeypatch.setenv("SOME_FLAG", value)

        assert command_utils.get_bool_env_var("SOME_FLAG") is expected

    def test_falls_back_to_the_supplied_default(self, monkeypatch):
        # [FUNCTION DOC] For an unset variable the result is false by default, and true when the
        # default text "1" is supplied.
        """An unset variable takes the default, which is itself parsed as a string."""
        monkeypatch.delenv("SOME_FLAG", raising=False)

        assert command_utils.get_bool_env_var("SOME_FLAG") is False
        assert command_utils.get_bool_env_var("SOME_FLAG", "1") is True


class TestGetEnvEnableInfiniteRun:
    # [CLASS DOC] Tests of the infinite-run environment switch.
    def test_defaults_to_off(self, monkeypatch):
        # [FUNCTION DOC] False when the test-infinite-run variable is unset; true when set to
        # "true".
        """Infinite runs must be opt-in; a stuck CI job is expensive."""
        monkeypatch.delenv("MILES_TEST_ENABLE_INFINITE_RUN", raising=False)
        assert command_utils.get_env_enable_infinite_run() is False

        monkeypatch.setenv("MILES_TEST_ENABLE_INFINITE_RUN", "true")
        assert command_utils.get_env_enable_infinite_run() is True


class TestEncodePseudoFile:
    # [CLASS DOC] Tests of encoding text as a pseudo file argument that file-argument resolution
    # decodes back.
    def test_round_trips_through_resolve_file_arg(self):
        # [FUNCTION DOC] Encoding "hello: world" and resolving the result with the file-argument
        # resolver gives back the original text.
        """The encoded argument is what the training process will be asked to resolve."""
        encoded = command_utils.encode_pseudo_file("hello: world")

        assert resolve_file_arg(encoded) == "hello: world"

    def test_is_deterministic(self):
        # [FUNCTION DOC] Encoding the same text twice gives equal results.
        """A hot restart must recompute the identical launch command."""
        assert command_utils.encode_pseudo_file("hello: world") == command_utils.encode_pseudo_file("hello: world")

    def test_survives_a_command_line_round_trip(self):
        # [FUNCTION DOC] Encoding multi-line text with quotes and splitting a command line holding it
        # (after a flag) with shell rules gives a single token equal to the encoded value.
        """The value is interpolated into a shell command, so it must not need quoting."""
        encoded = command_utils.encode_pseudo_file("a: 1\nb: 'two words'\n")

        assert shlex.split(f"--custom-config-path {encoded}")[1] == encoded


def _hardware_launchers_accept() -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of hardware names accepted by any launcher script's
    # hardware option (union of the literal values per launcher), excluding the "auto" sentinel.
    """Every value any launcher's `--hardware` takes, minus the sentinel that stands for "ask the node"."""
    return set().union(*launcher_hardware_literals().values()) - {"auto"}


class TestHardwareTables:
    # [CLASS DOC] Consistency tests of the hardware tables plus train-backend classification checks
    # (the latter are not about the tables; they sit here without their own class).
    def test_every_hardware_a_launcher_accepts_declares_its_gpus_per_node(self):
        # [FUNCTION DOC] Every hardware name some launcher accepts is a key of the GPUs-per-node
        # table.
        """A launcher whose hardware is missing here raises KeyError before doing anything."""
        assert _hardware_launchers_accept() <= command_utils.NUM_GPUS_OF_HARDWARE.keys()

    def test_every_hardware_with_a_generation_also_has_a_gpu_count(self):
        # [FUNCTION DOC] The keys of the generation table are a subset of the GPU count table keys.
        """Every launcher reads the GPU count, while only some read the generation."""
        assert command_utils.GENERATION_HARDWARE.keys() <= command_utils.NUM_GPUS_OF_HARDWARE.keys()

    def test_rejects_torchtitan_with_a_megatron_model_type(self, commands):
        # [FUNCTION DOC] The torchtitan train backend flag combined with a Megatron model type raises
        # an assertion error from execute_train.
        with pytest.raises(AssertionError):
            _backend().execute_train(
                train_args="--train-backend torchtitan", num_gpus_per_node=8, megatron_model_type="qwen"
            )

    def test_an_unknown_backend_is_not_classified_as_megatron(self, commands):
        # [FUNCTION DOC] An unknown train backend name with a Megatron model type raises an assertion
        # error, showing unknown backends are not treated as Megatron.
        """The classification is an allowlist, so a backend added later is not read as
        Megatron and handed Megatron's checkpoint conversion and TP-overlap env var."""
        with pytest.raises(AssertionError):
            _backend().execute_train(
                train_args="--train-backend somethingnew", num_gpus_per_node=8, megatron_model_type="qwen"
            )

    def test_tp_overlap_env_var_is_megatron_only(self, commands):
        # [FUNCTION DOC] A Megatron launch has CUDA max-connections "1" in its last runtime env; FSDP
        # and torchtitan launches (no model type) do not have it.
        """CUDA_DEVICE_MAX_CONNECTIONS=1 serializes copy engines, which costs the
        torch-native backends the compute/communication overlap they rely on."""

        def last_runtime_env():
            # [FUNCTION DOC] HELPER: Finds the most recent recorded command carrying the runtime-env
            # argument and returns its parsed environment mapping.
            submit = next(c for c in reversed(commands) if "--runtime-env-json=" in c)
            return _runtime_env(submit)

        _backend().execute_train(train_args="", num_gpus_per_node=8, megatron_model_type="qwen3-4B")
        assert last_runtime_env()["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"

        for backend in ("fsdp", "torchtitan"):
            _backend().execute_train(
                train_args=f"--train-backend {backend}", num_gpus_per_node=8, megatron_model_type=None
            )
            assert "CUDA_DEVICE_MAX_CONNECTIONS" not in last_runtime_env()


def _fake_torch(monkeypatch, *, capability, machine, total_memory=141 * 1024**3, name="fake", hip=None):
    # [FUNCTION DOC] HELPER: Installs a fake torch module in the module table and fakes the
    # platform machine string. The fake has a HIP version (hip argument) and CUDA functions:
    # availability (true), device name, device capability tuple and device properties with total
    # memory (default 141 GiB). Side effects: monkeypatched, undone after the test.
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            version=SimpleNamespace(hip=hip),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda: name,
                get_device_capability=lambda: capability,
                get_device_properties=lambda device: SimpleNamespace(total_memory=total_memory),
            ),
        ),
    )


class TestDetectHardware:
    # [CLASS DOC] Tests of GPU hardware detection from a fake torch module: compute capability,
    # CPU architecture, memory and ROCm device name.
    @pytest.mark.parametrize(
        ("capability", "machine", "expected"),
        [
            ((10, 0), "x86_64", "B200"),
            ((10, 0), "aarch64", "GB200"),
            ((10, 3), "x86_64", "B300"),
            ((10, 3), "aarch64", "GB300"),
        ],
    )
    def test_grace_is_what_separates_the_superchip_from_the_pcie_part(
        self, monkeypatch, capability, machine, expected
    ):
        # [FUNCTION DOC] Parametrized over capability (10,0) or (10,3) and machine x86_64 or aarch64:
        # x86_64 gives B200 / B300 and aarch64 (Grace CPU) gives GB200 / GB300.
        """GB200/GB300 carry the same die as B200/B300, so the host CPU is the only signal."""
        _fake_torch(monkeypatch, capability=capability, machine=machine)

        assert command_utils.detect_hardware() == expected

    @pytest.mark.parametrize(("total_memory", "expected"), [(80 * 1024**3, "H100"), (141 * 1024**3, "H200")])
    def test_hopper_is_split_by_memory(self, monkeypatch, total_memory, expected):
        # [FUNCTION DOC] Capability (9,0) on x86_64 gives H100 for 80 GiB and H200 for 141 GiB.
        """H100 and H200 share sm90; only the HBM capacity tells them apart."""
        _fake_torch(monkeypatch, capability=(9, 0), machine="x86_64", total_memory=total_memory)

        assert command_utils.detect_hardware() == expected

    def test_sm107_is_rubin(self, monkeypatch):
        # [FUNCTION DOC] Capability (10,7) on aarch64 is detected as "Rubin"; the GPU count table
        # gives 4 GPUs per node and the generation table maps it to "Rubin".
        _fake_torch(monkeypatch, capability=(10, 7), machine="aarch64")

        assert command_utils.detect_hardware() == "Rubin"
        assert command_utils.NUM_GPUS_OF_HARDWARE["Rubin"] == 4
        assert command_utils.GENERATION_HARDWARE["Rubin"] == "Rubin"

    @pytest.mark.parametrize(
        ("capability", "expected"),
        [((9, 4), "MI300X"), ((9, 4), "MI325X"), ((9, 5), "MI350X"), ((9, 5), "MI355X")],
    )
    def test_rocm_reads_the_device_name(self, monkeypatch, capability, expected):
        # [FUNCTION DOC] With a HIP version set, the hardware is read from the device name ("AMD
        # Instinct <name>"): MI300X, MI325X (both capability 9.4) and MI350X, MI355X (9.5); the
        # name, not the capability, decides.
        _fake_torch(monkeypatch, capability=capability, machine="x86_64", name=f"AMD Instinct {expected}", hip="6.4")

        assert command_utils.detect_hardware() == expected

    def test_an_unrecognized_device_asks_for_an_explicit_hardware(self, monkeypatch):
        # [FUNCTION DOC] An unrecognized device (capability (12,0), an RTX consumer name) raises an
        # assertion error telling the user to pass the hardware option.
        """Silently guessing the wrong profile costs a whole run; a launcher flag is one word."""
        _fake_torch(monkeypatch, capability=(12, 0), machine="x86_64", name="NVIDIA GeForce RTX 5090")

        with pytest.raises(AssertionError, match="pass --hardware"):
            command_utils.detect_hardware()

    def test_every_detectable_hardware_is_a_table_entry(self, monkeypatch):
        # [FUNCTION DOC] For capabilities (9,0), (10,0), (10,3), (10,7) and both machine types, the
        # detected hardware name is a key of the GPUs-per-node table.
        """A detected name the tables do not carry is a KeyError at the first lookup."""
        for capability in ((9, 0), (10, 0), (10, 3), (10, 7)):
            for machine in ("x86_64", "aarch64"):
                _fake_torch(monkeypatch, capability=capability, machine=machine)
                assert command_utils.detect_hardware() in command_utils.NUM_GPUS_OF_HARDWARE


class TestFromEnv:
    # [CLASS DOC] Tests of building an execute-train config from environment variables with
    # keyword overrides.
    def test_the_environment_fills_the_launcher_fields(self, monkeypatch):
        # [FUNCTION DOC] With cluster backend "kubernetes" and a namespace in the environment, the
        # built config has the kubernetes cluster backend enum, that namespace and the keyword
        # output dir "/tmp/out".
        """A workbench exports MILES_SCRIPT_* and a config built from the environment reads them."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", "kubernetes")
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-someones-namespace")

        config = command_utils.ExecuteTrainConfig.from_env(output_dir="/tmp/out")

        assert config.cluster_backend is base_backend.ClusterBackend.KUBERNETES
        assert config.namespace == "miles-someones-namespace"
        assert config.output_dir == "/tmp/out"

    def test_keyword_arguments_override_the_environment(self, monkeypatch):
        # [FUNCTION DOC] A namespace keyword "other" wins over the namespace in the environment.
        """A recipe that names a launcher field wins over the environment."""
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-someones-namespace")

        assert command_utils.ExecuteTrainConfig.from_env(namespace="other").namespace == "other"

    def test_keyword_arguments_reach_the_derivations_of_post_init(self):
        # [FUNCTION DOC] Defines a config subclass deriving the HF checkpoint path from the model name
        # in its post-init hook when none is given; building it from env with model name
        # "Inkling-Small-4layer" yields checkpoint "/root/models/Inkling-Small-4layer".
        """A recipe derives checkpoint paths from the model name it names, not from the field's default."""

        @dataclass
        class _Config(command_utils.ExecuteTrainConfig):
            # [CLASS DOC] HELPER: Config subclass with a model name (default "Inkling") and optional
            # HF checkpoint filled in post-init from the model name.
            model_name: str = "Inkling"
            hf_checkpoint: str | None = None

            def __post_init__(self):
                # [FUNCTION DOC] HELPER: If no HF checkpoint was given, sets it to the models root
                # joined with the model name.
                if self.hf_checkpoint is None:
                    self.hf_checkpoint = f"/root/models/{self.model_name}"

        assert _Config.from_env(model_name="Inkling-Small-4layer").hf_checkpoint == "/root/models/Inkling-Small-4layer"
