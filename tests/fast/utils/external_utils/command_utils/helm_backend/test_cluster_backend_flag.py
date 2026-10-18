# [MODULE DOC] Tests for what the Kubernetes backend's execute_train adds to the train argv the
# pods receive, with the helm, config writing and spec computation steps faked. Covers: the
# cluster backend flag (added as kubernetes, not repeated when equal in either spelling, refused
# when it conflicts or is a trailing flag with no value, and not mistaken inside another flag's
# value), the orchestrator command versus worker argv, naming validation of the run id, one shared
# run uuid per launch (new per launch, kept if supplied), the pod-side dispatch on the flag, the
# deploy component flag (always added, appended over an equal user flag, conflicts refused),
# release and object naming per deployed part, the api server host (whole run only), and run uuid
# resolution (split launches must be given one, first installs mint one, upgrades inherit the
# installed one). An autouse fixture points the shared root at the test temporary directory.
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.ray.test_wiring import stub_kubernetes_capability

from miles.ray import wiring
from miles.utils.external_utils.command_utils.base_backend import (
    _DEPLOY_COMPONENT_FLAG,
    CLUSTER_BACKEND_FLAG,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan, MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.run_uuid import RUN_UUID_LENGTH
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec


def declared_cluster_backends(argv: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the list of values given for the cluster backend flag in the
    # argv, in order, via the argv manipulator.
    return ArgvManipulator.get(argv, CLUSTER_BACKEND_FLAG)


def declared_deploy_components(argv: list[str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the list of values given for the deploy component flag in
    # the argv, in order, via the argv manipulator.
    return ArgvManipulator.get(argv, _DEPLOY_COMPONENT_FLAG)


NAMESPACE = "rl"
RUN_ID = "260101-000000-000"
RUN_ID_SHORT_ENOUGH_FOR_FULL_OBJECT_NAMES = "260101-00"


SPLIT_RUN_UUID = "0123456789abcdef"


def _release(deploy_component: DeployComponent = DeployComponent.ALL) -> str:
    # [FUNCTION DOC] HELPER: Serializes the release name for the module's run id, the given deploy
    # component (default all) and no instance id.
    return ReleaseName(run_id=RUN_ID, deploy_component=deploy_component, deploy_instance_id=None).serialize()


def _config(run_id: str = RUN_ID, deploy_component: DeployComponent = DeployComponent.ALL) -> ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Builds a Kubernetes backend config for the test namespace with the
    # given run id and deploy component; the run uuid is the fixed split uuid when the component is
    # a split one, otherwise unset.
    return ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace=NAMESPACE,
        run_id=run_id,
        run_uuid=SPLIT_RUN_UUID if deploy_component.is_split() else None,
        deploy_component=deploy_component,
    )


def _request(train_args: str) -> ExecuteTrainRequest:
    # [FUNCTION DOC] HELPER: Builds a train request with the given train args and fixed other
    # fields (8 GPUs per node, fake script path, FSDP off, no extras).
    # KNOWN ISSUE: this helper is never called in this module (dead code).
    return ExecuteTrainRequest(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=[],
    )


def _router() -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Returns a minimal router command spec: primary port 30000, no env
    # vars, one zero-GPU worker, and a fixed command running the router module.
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=30000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec.single(num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


@pytest.fixture(autouse=True)
def run_files_under_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture; replaces the infra info shared root lookup with a
    # function returning the test's temporary directory (the real root points at cluster storage
    # that CPU test lanes do not have).
    # the shared root comes from the chart and points at the cluster's storage mount, which the
    # cpu lane neither has nor may create
    monkeypatch.setattr(
        entrypoint.InfraInfo,
        "shared_root",
        staticmethod(lambda _infra, *, namespace: str(tmp_path)),
    )


def launch_argv(
    monkeypatch: pytest.MonkeyPatch,
    *,
    train_args: str,
    run_id: str = RUN_ID,
    deploy_component: DeployComponent = DeployComponent.ALL,
    recorded_releases: list[str] | None = None,
) -> list[str]:
    # [FUNCTION DOC] HELPER: Runs the Kubernetes backend's execute_train with the entrypoint's
    # collaborators faked, and returns the argv the spec computation received. Inputs: monkeypatch,
    # keyword only train args text, run id (default the module's), deploy component (default all),
    # and an optional list that receives the release name of each helm upgrade. Train args are
    # prefixed with the fsdp train backend flag and 8 GPUs per node is used. Fakes: spec
    # computation (records a copy of the args' argv, returns one router), argument parsing (reads
    # the process argv, picks the last deploy component flag value or all, wandb off), the
    # Mooncake plan (none), the values file write, trainer controller addresses (empty), helm
    # manifest lookup (nothing installed), the helm upgrade render (empty manifest), the previous
    # generation defusing, the helm dependency build, helm upgrade (records the release) and the
    # follow-until-finished step. Asserts the spec computation ran exactly once.
    recorded: list[list[str]] = []

    def fake_compute_specs(args: Any) -> list[CommandWorkerSpec]:
        # [FUNCTION DOC] HELPER: Records a copy of the args' argv and returns a list with one router.
        recorded.append(list(args.argv))
        return [_router()]

    def fake_parse_args() -> SimpleNamespace:
        # [FUNCTION DOC] HELPER: Builds the parsed args from the process command line (all but the
        # program name): colocate off, deploy component the last declared flag value or all, no
        # instance id, the argv itself, empty train env vars, wandb off, no wandb run id.
        argv = list(sys.argv[1:])
        declared = declared_deploy_components(argv)
        return SimpleNamespace(
            colocate=False,
            deploy_component=declared[-1] if declared else DeployComponent.ALL.value,
            deploy_instance_id=None,
            argv=argv,
            train_env_vars={},
            use_wandb=False,
            wandb_run_id=None,
        )

    def fake_upgrade(**kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Fake helm upgrade; appends the release keyword to the recorded
        # releases list when one was supplied, otherwise does nothing.
        if recorded_releases is not None:
            recorded_releases.append(kwargs["release"])

    def fake_render_upgrade(*, release: str, namespace: str, chart: Any, values_files: list[Any]) -> Manifest:
        # [FUNCTION DOC] HELPER: Fake helm render of an upgrade; returns an empty manifest in the
        # given namespace regardless of the other inputs.
        return Manifest.parse("", namespace=namespace)

    monkeypatch.setattr(entrypoint, "compute_specs", fake_compute_specs)
    monkeypatch.setattr(entrypoint, "parse_args", fake_parse_args)
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_write_helm_values", lambda path, values: None)
    monkeypatch.setattr(entrypoint, "_compute_trainer_controller_addrs", lambda args, *, release, namespace: {})
    monkeypatch.setattr(Helm, "get_manifest", staticmethod(lambda release, namespace: None))
    monkeypatch.setattr(Helm, "render_upgrade", staticmethod(fake_render_upgrade))
    monkeypatch.setattr(
        entrypoint,
        "_defuse_previous_generation",
        lambda release, *, namespace, superseded_state_file, state_file: None,
    )
    monkeypatch.setattr(Helm, "build_dependencies", lambda chart: None)
    monkeypatch.setattr(Helm, "upgrade", staticmethod(fake_upgrade))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)

    KubernetesCommandBackend(_config(run_id, deploy_component)).execute_train(
        train_args=f"--train-backend fsdp {train_args}", num_gpus_per_node=8, megatron_model_type=None
    )
    assert len(recorded) == 1
    return recorded[0]


def values_of(train_argv: list[str]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds the helm values for one router spec using a launch plan with
    # the module's run id, a fixed state file path, a fixed all-in-one release name, namespace rl,
    # the orchestrator command made of python, the train script and the given argv, and the given
    # argv as worker argv. Returns the values mapping.
    return build_values(
        [_router()],
        LaunchPlan(
            run_id=RUN_ID,
            state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
            release="miles-run-260101-000000-000",
            namespace="rl",
            orchestrator_command=["python", "/repo/train.py", *train_argv],
            worker_argv=train_argv,
        ),
    ).as_values()


class TestExecuteTrainTellsThePodsItsBackend:
    # [CLASS DOC] Tests of the cluster backend flag added to the argv.
    def test_the_train_argv_the_launcher_receives_names_kubernetes(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The recorded argv declares the cluster backend exactly once, as
        # kubernetes.
        """This argv is the only thing the pods are told about the run, so the backend has to be in it."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        assert declared_cluster_backends(argv) == ["kubernetes"]

    def test_the_orchestrator_command_and_the_worker_argv_both_carry_it(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Values built from the recorded argv: the orchestrator command declares the
        # backend as kubernetes, while the router's static worker command declares none (the
        # router is a plain command, not a served worker).
        # KNOWN ISSUE: the test name says the worker argv carries the flag too, but the assertion
        # only checks that the router's own command has none; the worker argv is not inspected.
        """The orchestrator and its workers have to agree, and each reads its own copy of the argv."""
        run = values_of(launch_argv(monkeypatch, train_args="--rollout-num-gpus 8"))["run"]

        assert declared_cluster_backends(run["orchestrator"]["command"]) == ["kubernetes"]
        assert declared_cluster_backends(run["staticWorkers"][0]["command"]) == []

    def test_a_user_supplied_agreeing_flag_is_not_repeated(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Train args already naming kubernetes: the final argv contains the flag
        # exactly once.
        """A run relaunched from a recorded command line already carries the flag the launcher would add."""
        argv = launch_argv(monkeypatch, train_args="--cluster-backend kubernetes --rollout-num-gpus 8")

        assert argv.count("--cluster-backend") == 1

    def test_a_user_supplied_conflicting_flag_stops_the_launch(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Train args naming the ray backend make the launch raise an assertion
        # error mentioning ray.
        """Launching a ray-flagged run onto kubernetes would install pods nothing ever drives."""
        with pytest.raises(AssertionError, match="ray"):
            launch_argv(monkeypatch, train_args="--cluster-backend ray")

    def test_a_user_supplied_agreeing_equals_form_is_not_repeated_either(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The equals sign spelling naming kubernetes: the declared backends are
        # exactly kubernetes once, and the joined token itself is present in the argv.
        """`--flag=value` is one token, so a search for the space form would miss it and add a second flag."""
        argv = launch_argv(monkeypatch, train_args="--cluster-backend=kubernetes --rollout-num-gpus 8")

        assert declared_cluster_backends(argv) == ["kubernetes"]
        assert "--cluster-backend=kubernetes" in argv

    def test_the_equals_form_of_a_conflicting_flag_stops_the_launch_too(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The equals sign spelling naming ray raises an assertion error mentioning ray.
        """The conflict check has to see the same tokens argparse will."""
        with pytest.raises(AssertionError, match="ray"):
            launch_argv(monkeypatch, train_args="--cluster-backend=ray")

    def test_does_not_mistake_another_flags_value_for_a_declaration(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A data path flag whose quoted value contains the text of the backend flag
        # and ray: the launch succeeds, the declared backends are exactly kubernetes, and the data
        # path token is kept whole in the argv.
        """A substring search would find the flag inside `--data-path=--cluster-backend ray` and refuse the launch."""
        argv = launch_argv(monkeypatch, train_args="--data-path='--cluster-backend ray'")

        assert declared_cluster_backends(argv) == ["kubernetes"]
        assert "--data-path=--cluster-backend ray" in argv

    def test_a_trailing_flag_that_names_nothing_stops_the_launch(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Train args ending with the bare backend flag raise an assertion error
        # mentioning the last argument.
        """argparse would fail on this argv inside the pod, where the failure is much harder to read."""
        with pytest.raises(AssertionError, match="last argument"):
            launch_argv(monkeypatch, train_args="--rollout-num-gpus 8 --cluster-backend")


class TestARunIsNamedBeforeAnythingIsInstalled:
    # [CLASS DOC] Tests of run id validation before any install step.
    def test_a_run_id_that_cannot_name_a_kubernetes_object_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A run id with uppercase and underscore characters makes the launch raise
        # an assertion error saying the id names every object.
        """helm fails on it long after the launcher could have said which value was wrong."""
        with pytest.raises(AssertionError, match="names every object"):
            launch_argv(monkeypatch, train_args="--rollout-num-gpus 8", run_id="Qwen3_4B")


class TestEveryPodOfARunSharesOneRunUuid:
    # [CLASS DOC] Tests of the run uuid flag added to the argv.
    def test_the_launcher_stamps_one_uuid_on_every_pod(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The recorded argv has exactly one run uuid flag value, whose length is
        # the run uuid length constant.
        """Left unset, every pod mints its own uuid at parse time and nothing about the run can be joined up."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        [stamped] = ArgvManipulator.get(argv, "--run-uuid")
        assert len(stamped) == RUN_UUID_LENGTH

    def test_two_launches_of_one_run_id_are_two_trainings(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Two launches with the same run id and args stamp different uuids.
        """A run id is reused to resume, so deriving the uuid from it would let a stale deployment pass the handshake."""
        first = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")
        second = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        assert ArgvManipulator.get(first, "--run-uuid") != ArgvManipulator.get(second, "--run-uuid")

    def test_a_uuid_the_train_args_already_carry_is_left_alone(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Train args that already carry a run uuid: the argv still has exactly that
        # one value.
        """Relaunching from a recorded command line must keep the uuid its artefacts were written under."""
        argv = launch_argv(monkeypatch, train_args="--run-uuid 0123456789abcdef --rollout-num-gpus 8")

        assert ArgvManipulator.get(argv, "--run-uuid") == ["0123456789abcdef"]


class TestThePodDispatchesOnThatFlag:
    # [CLASS DOC] Test of the in-pod dispatch on the backend flag.
    def test_that_argv_selects_the_kubernetes_branch_in_the_pod(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Takes the declared backend from the launcher's argv, stubs the Kubernetes
        # capability, then builds an args namespace with that backend and 8 GPUs per node. The
        # wiring's backend capability lookup returns the stub capability, the stub computed specs
        # from exactly those args, and the declared value converts to the Kubernetes backend enum.
        """The flag is only worth adding if the in-pod dispatch reads it and skips the ray worker manager."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")
        declared = declared_cluster_backends(argv)

        stub = stub_kubernetes_capability(monkeypatch)

        args = SimpleNamespace(cluster_backend=declared[0], num_gpus_per_node=8)
        assert wiring.get_backend_capability(args) is stub.capability
        assert stub.specs_computed_from == [args]
        assert ClusterBackend(declared[0]) is ClusterBackend.KUBERNETES


class TestExecuteTrainTellsThePodsWhichPartOfTheRunTheyAre:
    # [CLASS DOC] Tests of the deploy component flag and the naming that depends on it.
    def test_a_whole_run_is_told_so_too(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] An all-in-one launch declares the deploy component as all.
        """Every launch says which part it is, so a pod never has to fall back on a default nobody wrote down."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8")

        assert declared_deploy_components(argv) == ["all"]

    def test_the_train_argv_names_the_part_the_launch_deploys(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A trainer launch declares the deploy component as trainer.
        """The pods filter the run's spec table by this flag, so it is what makes them a subset of the run."""
        argv = launch_argv(monkeypatch, train_args="--rollout-num-gpus 8", deploy_component=DeployComponent.TRAINER)

        assert declared_deploy_components(argv) == ["trainer"]

    def test_the_release_is_named_after_the_part_it_installs(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A trainer launch performs exactly one helm upgrade, for the release named
        # with the miles run prefix, the run id and a trainer suffix.
        """Two parts of one run id share a release name unless the part is in it, and would uninstall each other."""
        releases: list[str] = []

        launch_argv(
            monkeypatch,
            train_args="--rollout-num-gpus 8",
            deploy_component=DeployComponent.TRAINER,
            recorded_releases=releases,
        )

        assert releases == [f"miles-run-{RUN_ID}-trainer"]

    def test_every_object_of_a_split_release_is_named_after_that_release(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Using a short run id so full object names fit, a trainer launch's values
        # (built with the recorded release) name the orchestrator object and the router static
        # worker object with the trainer release as prefix.
        """The chart computes no names, so an object named after the whole run would collide with the whole run."""
        run_id = RUN_ID_SHORT_ENOUGH_FOR_FULL_OBJECT_NAMES
        releases: list[str] = []
        argv = launch_argv(
            monkeypatch,
            train_args="--rollout-num-gpus 8",
            run_id=run_id,
            deploy_component=DeployComponent.TRAINER,
            recorded_releases=releases,
        )

        run = _values_of_release(argv, run_id=run_id, release=releases[0])["run"]

        assert run["objectNames"]["orchestrator"] == f"miles-run-{run_id}-trainer-orchestrator"
        assert run["staticWorkers"][0]["objectName"] == f"miles-run-{run_id}-trainer-inference-router-0"

    def test_a_user_supplied_agreeing_flag_is_appended_over_rather_than_detected(self, monkeypatch):
        # [FUNCTION DOC] Train args already containing the trainer deploy component: the launcher
        # still appends its own, so the declared components are trainer twice (the last one wins
        # when parsed).
        """A relaunch from a recorded command line repeats the flag, and the last one argparse reads still wins."""
        argv = launch_argv(
            monkeypatch,
            train_args="--deploy-component trainer --rollout-num-gpus 8",
            deploy_component=DeployComponent.TRAINER,
        )

        assert declared_deploy_components(argv) == ["trainer", "trainer"]

    def test_a_user_supplied_conflicting_flag_stops_the_launch(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Train args naming the trainer component for an all-in-one launch make it
        # raise an assertion error mentioning the deploy component flag.
        """Everything this launch installs is named after its own part, so pods told another part are orphans."""
        with pytest.raises(AssertionError, match="deploy-component"):
            launch_argv(monkeypatch, train_args="--deploy-component trainer --rollout-num-gpus 8")


class TestApiServerHost:
    # [CLASS DOC] Tests of the backend's api server host for whole and split runs.
    def test_a_whole_run_answers_on_its_own_orchestrator(self):
        # [FUNCTION DOC] For an all-in-one config, the host is the release name, an orchestrator
        # suffix, then the namespace and the svc domain, dot separated.
        """The api server runs beside the orchestration script, which is a pod of the run's only release."""
        config = _config()
        host = KubernetesCommandBackend(config).api_server_host(config)

        assert host == f"{_release()}-orchestrator.{NAMESPACE}.svc"

    @pytest.mark.parametrize("component", [DeployComponent.PRIMARY, DeployComponent.TRAINER])
    def test_no_deployment_of_a_split_run_has_an_api_server_to_name(self, component):
        # [FUNCTION DOC] Parametrized over the primary and trainer components: asking for the host
        # raises an assertion error telling the user to set the api server port to 0.
        """A split run is refused an api server, so any host answered here would only ever time out."""
        config = _config(deploy_component=component)
        backend = KubernetesCommandBackend(config)

        with pytest.raises(AssertionError, match="--api-server-port 0"):
            backend.api_server_host(config)


def _values_of_release(train_argv: list[str], *, run_id: str, release: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds helm values for one router spec with a plan using the given
    # run id and release, namespace rl, an empty state file, an empty orchestrator command, and
    # the argv as worker argv. Returns the values mapping.
    return build_values(
        [_router()],
        LaunchPlan(
            run_id=run_id,
            state_file="",
            release=release,
            namespace=NAMESPACE,
            orchestrator_command=[],
            worker_argv=train_argv,
        ),
    ).as_values()


_INSTALLED_MANIFEST = """
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: miles-run-260101-000000-000-all-orchestrator
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: orchestrator
          command: ["python", "train.py", "--run-uuid", "aaaabbbbccccdddd"]
"""


class TestTheRunUuidALaunchStamps:
    # [CLASS DOC] Tests of the entrypoint's private run uuid resolution, which takes the config,
    # the installed manifest (or none) and the release.
    def test_a_split_launch_has_to_be_given_one(self):
        # [FUNCTION DOC] A trainer launch whose config has no run uuid raises an assertion error
        # mentioning the run uuid flag.
        """The parts of a run are joined by nothing else, so a launcher inventing one would split the run in two."""
        config = _config(deploy_component=DeployComponent.TRAINER)
        config.run_uuid = None

        with pytest.raises(AssertionError, match="--run-uuid"):
            entrypoint._resolve_run_uuid(config, installed_manifest=None, release=_release(DeployComponent.TRAINER))

    def test_a_split_launch_keeps_the_one_it_was_given(self):
        # [FUNCTION DOC] A trainer launch with a configured run uuid and nothing installed
        # resolves to that same uuid.
        """It is the value the layer deploying every part chose, and changing it would fail the handshake."""
        config = _config(deploy_component=DeployComponent.TRAINER)

        assert (
            entrypoint._resolve_run_uuid(config, installed_manifest=None, release=_release(DeployComponent.TRAINER))
            == SPLIT_RUN_UUID
        )

    def test_a_first_install_mints_one(self):
        # [FUNCTION DOC] An all-in-one config with nothing installed resolves to a freshly
        # generated uuid of the run uuid length.
        """A single deployment is the whole run, so nothing outside it has to agree on the value."""
        stamped = entrypoint._resolve_run_uuid(_config(), installed_manifest=None, release=_release())

        assert len(stamped) == RUN_UUID_LENGTH

    def test_an_upgrade_in_place_keeps_the_uuid_the_pods_already_carry(self):
        # [FUNCTION DOC] With an installed manifest whose orchestrator command carries a run uuid
        # flag, the all-in-one resolution returns that installed uuid.
        """Resizing is the same training, and a fresh uuid would change every pod argv and trip the upgrade guard."""
        installed = Manifest.parse(_INSTALLED_MANIFEST, namespace=NAMESPACE)

        assert (
            entrypoint._resolve_run_uuid(_config(), installed_manifest=installed, release=_release())
            == "aaaabbbbccccdddd"
        )
