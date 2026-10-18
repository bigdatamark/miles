# [MODULE DOC] Tests for the Kubernetes (helm) launcher entrypoint's execute_train flow, driven
# end to end with all external processes faked. Covers: the launch record written to the run
# directory, credential redaction in that record, the run environment used while computing worker
# specs, rejection of a user supplied env report flag, a golden snapshot of the helm commands plus
# generated values file, pass-through of extra manifests, and a check on the snapshot directory.
# Fixtures: a router command spec, an inference engine command spec, and a trainer serve spec, a
# fake infra values file, and frozen run id, launch token and run uuid constants so output is
# deterministic. The helm and kubectl process runner is replaced by a recorder that returns an
# empty manifest for dry runs.
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot, sanitize

from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import POOL_CATEGORY_TRAINER_ENGINE
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "helm_backend"

FROZEN_RUN_ID = "260101-000000-000"
FROZEN_LAUNCH_TOKEN = "260101-000000-000001"
FROZEN_RUN_UUID = "f52ecf8e9d7d4889"
NAMESPACE = "rl"
PYTHON_PLACEHOLDER = "<PYTHON>"


def _router() -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a command worker spec for the inference router: zero GPUs, one
    # primary port fixed at 30000, no extra env vars, and a launch command that starts the sglang
    # router bound to its own primary address host on port 30000.
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=30000)],
        env_var=lambda ctx: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda ctx: (
            f"python -m sglang_router.launch_router --host {ctx.self_addrs['primary'].host} --port 30000"
        ),
    )


def _engine() -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a command worker spec for an inference engine pool: category is
    # the inference engine pool category, ports are a primary port (8000) and a master mode
    # distributed init port (9000), one env var disabling NVSHMEM over NCCL, and scheduling of 2
    # cells with 2 workers each, 0.2 GPU per worker, 8 GPU slots per worker, 8 GPUs per node. The
    # launch command starts the sglang server with the worker's index in its cell as node rank and
    # the distributed init address taken from the dist_init self address.
    return CommandWorkerSpec(
        name="inference-engine-0-0",
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[
            PortInfo(name="primary", static_port=8000),
            PortInfo(name="dist_init", static_port=9000, mode="master"),
        ],
        env_var=lambda ctx: {"NVSHMEM_DISABLE_NCCL": "1"},
        scheduling=SchedulingSpec(
            num_cells=2,
            num_workers_per_cell=2,
            num_gpus_per_worker=0.2,
            num_gpu_slots_per_worker=8,
            num_gpus_per_node=8,
        ),
        launch_command=lambda ctx: (
            f"python -m sglang.launch_server --node-rank {ctx.worker_in_cell_index} "
            f"--dist-init-addr {ctx.self_addrs['dist_init'].host}:{ctx.self_addrs['dist_init'].port}"
        ),
    )


def _trainer() -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve worker spec for the trainer pool: trainer engine
    # category, a master mode port 9000, an env var disabling the NCCL cumem feature, scheduling of
    # 2 cells with 8 workers each, 0.4 GPU per worker, 1 GPU slot per worker, 8 GPUs per node, the
    # Megatron train actor as worker class, and empty constructor kwargs.
    return ServeWorkerSpec(
        name="trainer-engine-actor",
        category=POOL_CATEGORY_TRAINER_ENGINE,
        port_infos=[PortInfo(name="master", static_port=9000, mode="master")],
        env_var=lambda ctx: {"NCCL_CUMEM_ENABLE": "0"},
        scheduling=SchedulingSpec(
            num_cells=2,
            num_workers_per_cell=8,
            num_gpus_per_worker=0.4,
            num_gpu_slots_per_worker=1,
            num_gpus_per_node=8,
        ),
        worker_class="miles.backends.megatron_utils.actor.MegatronTrainRayActor",
        ctor_kwargs=lambda ctx: {},
    )


def _request(
    *,
    train_args: str = "--rollout-num-gpus 8",
    extra_env_vars: dict[str, str] | None = None,
    extra_manifests: list[str] | None = None,
):
    # [FUNCTION DOC] HELPER: Builds a train request with fixed values: 8 GPUs per node, no model
    # type, train script at a fake repo path, FSDP backend off, a fixed Megatron path, no pre
    # submit hook, empty prepare commands. Keyword only inputs: train args string (default sets 8
    # rollout GPUs), extra env vars mapping and extra manifests list (both default to empty when
    # not given). Returns the request.
    return ExecuteTrainRequest(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars=extra_env_vars or {},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=extra_manifests or [],
    )


def helm_values_file(sandbox: Path) -> Path:
    # [FUNCTION DOC] HELPER: Writes a user side infra values file named infra in the sandbox
    # directory and returns its path. Content: image repository and tag, one host path volume
    # called cluster storage (host path and mount both inside the sandbox), and a runs root path
    # under that storage. Side effect: file write.
    values_file = sandbox / "infra.yaml"
    values_file.write_text(
        yaml.safe_dump(
            {
                "infra": {
                    "image": {"repository": "myregistry.example/miles", "tag": "v1"},
                    "volumes": [
                        {
                            "name": "cluster-storage",
                            "hostPath": {"path": f"{sandbox}/cluster-storage"},
                            "mounts": [{"mountPath": f"{sandbox}/cluster-storage"}],
                        }
                    ],
                    "paths": {"runsRoot": f"{sandbox}/cluster-storage/miles_data"},
                }
            }
        )
    )
    return values_file


def run_dir(sandbox: Path) -> Path:
    # [FUNCTION DOC] HELPER: Returns the run directory the launcher is expected to use for the
    # frozen run id: cluster storage, then the data directory, then the runs directory, then the
    # run id, all under the sandbox. Pure path computation.
    return sandbox / "cluster-storage" / "miles_data" / "miles-runs" / FROZEN_RUN_ID


def values_file(sandbox: Path) -> Path:
    # [FUNCTION DOC] HELPER: Returns the path of the generated helm values file for the frozen
    # launch token inside the run directory (a values subdirectory, named after the token with a
    # yaml suffix). Pure path computation.
    return run_dir(sandbox) / "values" / f"values-{FROZEN_LAUNCH_TOKEN}.yaml"


def record_launch(monkeypatch, sandbox: Path, on_compute_specs=None, **request_overrides) -> list[str]:
    # [FUNCTION DOC] HELPER: Runs the launcher's execute_train once with everything external
    # faked, and returns the list of recorded commands, each joined into one space separated
    # string. Patches: the process runner (records each command; for a dry run command it returns
    # a json document with an empty manifest, otherwise empty output, always exit code zero), the
    # helm get manifest lookup (reports no installed release), the repo base directory, the launch
    # token generator and the run uuid generator (frozen constants), plus the launch input stubs.
    # The optional on_compute_specs callback is invoked whenever specs are computed; request
    # overrides are forwarded to the request builder. The config uses the test namespace, frozen
    # run id and the generated infra values file. Side effects: monkeypatching, files written
    # under the sandbox by the launcher.
    recorded: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Fake process runner: appends the command (parts converted to text
        # and joined by spaces) to the recorded list and returns a completed process with exit
        # code zero, empty stderr, and as stdout a json object with an empty manifest field when
        # the command contains the dry run flag, else empty text. Extra keyword args are ignored.
        recorded.append(" ".join(str(part) for part in command))
        rendered = json.dumps({"manifest": ""}) if "--dry-run" in command else ""
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=rendered, stderr="")

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    monkeypatch.setattr(Helm, "get_manifest", staticmethod(lambda release, namespace: None))
    monkeypatch.setattr(entrypoint, "repo_base_dir", str(REPO_ROOT))
    monkeypatch.setattr(naming, "_new_launch_token", lambda: FROZEN_LAUNCH_TOKEN)
    monkeypatch.setattr(entrypoint, "generate_run_uuid", lambda: FROZEN_RUN_UUID)

    _stub_launch_inputs(monkeypatch, specs=[_router(), _engine(), _trainer()], on_compute_specs=on_compute_specs)

    entrypoint.execute_train(
        request=_request(**request_overrides),
        config=ExecuteTrainConfig(
            namespace=NAMESPACE, run_id=FROZEN_RUN_ID, helm_values=(str(helm_values_file(sandbox)),)
        ),
    )
    return recorded


def freeze(text: str, sandbox: Path) -> str:
    # [FUNCTION DOC] HELPER: Normalizes text for snapshot comparison: replaces the running Python
    # interpreter path with a placeholder, then applies the shared sanitizer with the sandbox path.
    # Returns the normalized text.
    return sanitize(text.replace(sys.executable, PYTHON_PLACEHOLDER), sandbox=sandbox)


def format_launch(commands: list[str], values_text: str, sandbox: Path) -> str:
    # [FUNCTION DOC] HELPER: Renders the golden snapshot text. For each recorded command, a header
    # line with a hash prefix and the zero based index, the frozen command, and a blank line.
    # Then a header line for pseudo file 1, the frozen values file text, and a trailing blank
    # line; all joined by newlines. Returns the string.
    lines: list[str] = []
    for index, command in enumerate(commands):
        lines.append(f"### {index}")
        lines.append(freeze(command, sandbox=sandbox))
        lines.append("")
    lines.append("### pseudo file 1")
    lines.append(freeze(values_text, sandbox=sandbox))
    lines.append("")
    return "\n".join(lines)


def _stub_launch_inputs(monkeypatch, *, specs, colocate: bool = False, on_compute_specs=None) -> None:
    # [FUNCTION DOC] HELPER: Patches the entrypoint's inputs: the spec computation returns the
    # given specs (after calling the optional callback), argument parsing returns a simple
    # namespace (colocate flag as given, deploy component all, no instance id, empty argv, empty
    # train env vars, wandb off, no wandb run id), the Mooncake plan from args returns nothing,
    # and the follow until finished step becomes a no-op. Returns nothing.
    def compute_specs(args):
        # [FUNCTION DOC] HELPER: Fake spec computation: calls the on_compute_specs callback when
        # one was given, then returns the stubbed specs regardless of the args.
        if on_compute_specs is not None:
            on_compute_specs()
        return specs

    monkeypatch.setattr(entrypoint, "compute_specs", compute_specs)
    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: SimpleNamespace(
            colocate=colocate,
            deploy_component="all",
            deploy_instance_id=None,
            argv=[],
            train_env_vars={},
            use_wandb=False,
            wandb_run_id=None,
        ),
    )
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)


class TestTheLauncherRecordsWhatItLaunched:
    # [CLASS DOC] Tests of the per launch audit record the launcher writes into the run directory
    # and hands to the pods by path.
    def test_the_record_lands_beside_the_values_it_describes(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches once, then asserts exactly one json record exists under the run
        # directory's launches folder and that it holds the run id, namespace, the train script as
        # second element of the orchestrator command, and the path of the generated values file.
        """The pods' argv is computed here, so this is the only place it can be recorded from."""
        record_launch(monkeypatch, tmp_path)

        records = sorted((run_dir(tmp_path) / "launches").glob("*.json"))
        assert len(records) == 1
        recorded = json.loads(records[0].read_text())

        assert recorded["run_id"] == FROZEN_RUN_ID
        assert recorded["namespace"] == NAMESPACE
        assert recorded["orchestrator_command"][1] == "/repo/train.py"
        assert recorded["values_file"] == str(values_file(tmp_path))

    def test_the_pods_are_pointed_at_that_record(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches once and asserts the generated values file's run section has a
        # launch record field equal to the path of the single record file written.
        """A record on a shared disk nobody reads is not what reaches the wandb config."""
        record_launch(monkeypatch, tmp_path)
        rendered = yaml.safe_load(values_file(tmp_path).read_text())

        records = sorted((run_dir(tmp_path) / "launches").glob("*.json"))
        assert rendered["run"]["launchRecord"] == str(records[0])

    def test_the_pods_are_handed_a_path_rather_than_the_record_itself(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches once and asserts the launch record field in the values is an
        # absolute path (starts with a slash) and does not contain record content such as the
        # orchestrator command key.
        """A whole environment dump in an env var is passed to every process of every pod."""
        record_launch(monkeypatch, tmp_path)
        rendered = yaml.safe_load(values_file(tmp_path).read_text())

        assert rendered["run"]["launchRecord"].startswith("/")
        assert "orchestrator_command" not in rendered["run"]["launchRecord"]

    def test_the_record_carries_no_secret_the_launcher_was_given(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches with a secret wandb key in the train args and a token plus a
        # harmless variable in the extra env vars, then reads the record file text and asserts
        # neither secret value appears and a redaction marker (a sha256 digest prefix) does.
        """The record lands on a shared disk and in the wandb config, so a key in argv would leak twice."""
        record_launch(
            monkeypatch,
            tmp_path,
            train_args="--rollout-num-gpus 8 --wandb-key s3cret",
            extra_env_vars={"HF_TOKEN": "t0ken", "PYTHONUNBUFFERED": "1"},
        )
        on_disk = sorted((run_dir(tmp_path) / "launches").glob("*.json"))[0].read_text()

        assert "s3cret" not in on_disk
        assert "t0ken" not in on_disk
        assert "redacted-sha256:" in on_disk

    def test_the_pods_still_receive_the_credentials_they_run_with(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches with a token in the extra env vars and asserts the generated
        # values file still carries the unredacted token in the run env section.
        """Only the audit copy is redacted; redacting the pods' own environment would break the run."""
        record_launch(monkeypatch, tmp_path, extra_env_vars={"HF_TOKEN": "t0ken"})
        rendered = yaml.safe_load(values_file(tmp_path).read_text())

        assert rendered["run"]["env"]["HF_TOKEN"] == "t0ken"

    def test_the_whole_record_is_on_disk_before_helm_installs_anything(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Replaces the helm upgrade call with a probe that, at call time, asserts
        # exactly one record file exists and captures its parsed content. After launching, asserts
        # the upgrade was called once and the captured record already named the values file.
        """helm upgrade is irreversible; recording afterwards can leave running pods with no record."""
        helm_run_snapshot: list[dict] = []

        def note_upgrade(**kwargs) -> None:
            # [FUNCTION DOC] HELPER: Fake helm upgrade: asserts exactly one launch record file
            # exists already, and appends its parsed json content to the captured list.
            written = sorted((run_dir(tmp_path) / "launches").glob("*.json"))
            assert len(written) == 1
            helm_run_snapshot.append(json.loads(written[0].read_text()))

        monkeypatch.setattr(Helm, "upgrade", staticmethod(note_upgrade))
        record_launch(monkeypatch, tmp_path)

        assert len(helm_run_snapshot) == 1
        assert helm_run_snapshot[0]["values_file"] == str(values_file(tmp_path))


class TestTheRunEnvironment:
    # [CLASS DOC] Tests that the launcher applies the run's extra env vars to its own process
    # environment while specs are computed.
    def test_computes_the_worker_specs_under_the_environment_the_run_will_have(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches with an extra env var for an sglang memory imbalance check set to
        # true, and a spec computation callback that records the variable's value from the
        # process environment. Asserts it was seen exactly once, with the value true.
        """A spec reads env vars of its own, and the launcher's environment is not the run's."""
        seen: list[str | None] = []

        record_launch(
            monkeypatch,
            tmp_path,
            on_compute_specs=lambda: seen.append(os.environ.get("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK")),
            extra_env_vars={"SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "true"},
        )

        assert seen == ["true"]


class TestTheLauncherOwnsTheReport:
    # [CLASS DOC] Tests that user train args may not set the env report option, which the
    # launcher owns.
    def test_a_run_may_not_name_its_own_env_report(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launching with the env report flag and a separate value in the train args
        # must raise an assertion error whose message mentions the flag.
        """--env-report outranks the variable the chart sets, so the pods would report a launch that never ran."""
        with pytest.raises(AssertionError, match="--env-report"):
            record_launch(monkeypatch, tmp_path, train_args="--rollout-num-gpus 8 --env-report {}")

    def test_an_inline_env_report_is_refused_too(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Same as the previous test but with the flag joined to its value by an
        # equals sign; must also raise an assertion error mentioning the flag.
        with pytest.raises(AssertionError, match="--env-report"):
            record_launch(monkeypatch, tmp_path, train_args="--rollout-num-gpus 8 --env-report={}")


class TestKubernetesLaunchSnapshot:
    # [CLASS DOC] Golden snapshot tests of a whole Kubernetes launch.
    def test_the_helm_argv_and_the_generated_values_match_the_recording(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches once, formats the recorded commands plus the generated values
        # file text, and compares against the kubernetes launch golden text file in the snapshot
        # directory via the shared snapshot assertion helper.
        """The values file is the whole training recipe, so a snapshot of only the argv would prove little."""
        commands = record_launch(monkeypatch, tmp_path)

        recorded = format_launch(commands, values_file(tmp_path).read_text(), tmp_path)

        assert_matches_snapshot(SNAPSHOT_DIR / "kubernetes_launch.txt", recorded, "kubernetes launcher recording")

    def test_the_generated_values_carry_no_infra_section(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches once and asserts the top level keys of the generated values file
        # are exactly the run section, i.e. the user's infra section is not written by the launcher.
        """infra is the user's half of the contract; the launcher writing it would silently override a cluster."""
        record_launch(monkeypatch, tmp_path)

        assert set(yaml.safe_load(values_file(tmp_path).read_text())) == {"run"}


class TestTheLauncherPassesExtraManifestsToTheChart:
    # [CLASS DOC] Tests of how the request's extra manifests are forwarded into the values file.
    def test_the_manifests_the_caller_named_reach_the_values_file_verbatim(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches with two yaml manifest texts (a Service and a StatefulSet) as
        # extra manifests and asserts the values file's extra manifests entry is the identical list
        # of strings, unchanged.
        """The chart installs this text as it stands, so the launcher may not reformat it on the way."""
        manifests = [
            "apiVersion: v1\nkind: Service\nmetadata:\n  name: external-sglang\n",
            "apiVersion: apps/v1\nkind: StatefulSet\nmetadata:\n  name: external-sglang\n",
        ]
        record_launch(monkeypatch, tmp_path, extra_manifests=manifests)

        assert yaml.safe_load(values_file(tmp_path).read_text())["extraManifests"] == manifests

    def test_a_launch_that_named_none_writes_no_such_section(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Launches without extra manifests and asserts the values file has no extra
        # manifests key at all (not even an empty one).
        """An empty section in the values would still override whatever a user's own values file set."""
        record_launch(monkeypatch, tmp_path)

        assert "extraManifests" not in yaml.safe_load(values_file(tmp_path).read_text())


class TestSnapshotFiles:
    # [CLASS DOC] Hygiene check on the golden snapshot directory.
    def test_the_recorded_files_are_exactly_the_declared_ones(self):
        # [FUNCTION DOC] Asserts the set of text file names in the snapshot directory equals
        # exactly the single kubernetes launch file, so orphan goldens fail the test.
        """A renamed or deleted case must not leave an orphan golden behind."""
        recorded = {path.name for path in SNAPSHOT_DIR.glob("*.txt")}

        assert recorded == {"kubernetes_launch.txt"}
