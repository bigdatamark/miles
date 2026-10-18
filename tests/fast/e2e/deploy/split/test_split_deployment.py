# [MODULE DOC]
# Unit tests for the split-deployment launcher: running one training as several separately
# installed helm releases (trainer, inference engines, and one primary script-driving release).
# Everything external is replaced by in-memory fakes: the per-deployment launch callable is a
# recorder, the training entry point is patched with a recorder, and the helm client plus the
# release-removal helper are patched with a fake that tracks uninstalled releases. Module constants
# fix the namespace, run id, run uuid, a sentinel file name, and a two-cell test mode.
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.e2e.deploy.conftest_deploy.split import split_deployment
from tests.e2e.deploy.conftest_deploy.split.split_deployment import (
    RunDeployment,
    create_launch_of_mode,
    run_split_training,
    run_split_training_into,
)
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.workers.types import ClusterBackend, DeployComponent

NAMESPACE: str = "rl"
RUN_ID: str = "demo"
RUN_UUID: str = "0123456789abcdef"
SENTINEL_NAME: str = "written-by-a-deployment.txt"

MODE: FTTestMode = FTTestMode(
    model_name="Qwen3-0.6B",
    model_hf_repo="Qwen/Qwen3-0.6B",
    megatron_model_type="qwen3-0.6B",
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    parallel_args="--context-parallel-size 2",
)


@pytest.fixture
def config() -> ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Fixture returning a Kubernetes train config using the module namespace,
    # run id and run uuid, with no deploy component set.
    return ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID, run_uuid=RUN_UUID
    )


def _release(deploy_component: DeployComponent, deploy_instance_id: str | None = None) -> str:
    # [FUNCTION DOC] HELPER: Builds the serialized helm release name for the module run id, the given
    # deploy component and optional instance id. Returns the string.
    return ReleaseName(
        run_id=RUN_ID, deploy_component=deploy_component, deploy_instance_id=deploy_instance_id
    ).serialize()


class TestRunSplitTraining:
    # [CLASS DOC] Tests for the split-training driver: launch order, shared run identity,
    # argument passing, release teardown (including on failure) and validation of the deployment list.
    def test_it_installs_the_workers_before_the_script_that_drives_them(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Installs the standard four deployments and asserts the launch order of
        # components is trainer, inference, inference, primary.
        """The driving launch blocks until the run finishes, so anything installed after it never arrives."""
        _install(config=config, launches=deployment_launches)

        assert [one.deploy_component for one in deployment_launches.records] == [
            DeployComponent.TRAINER,
            DeployComponent.INFERENCE,
            DeployComponent.INFERENCE,
            DeployComponent.PRIMARY,
        ]

    def test_every_deployment_of_one_run_carries_the_same_run_uuid(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Asserts the set of run uuids seen by all launches is exactly the configured one.
        """Nothing but the run uuid joins releases that separate launches installed."""
        _install(config=config, launches=deployment_launches)

        assert {one.run_uuid for one in deployment_launches.records} == {RUN_UUID}

    def test_a_run_whose_launches_would_each_invent_an_identity_is_refused(self, deployment_launches, fake_helm):
        # [FUNCTION DOC] With a config lacking a run uuid, asserts installation raises an assertion
        # error whose message mentions the release being joined only by the run uuid.
        """Deployments agreeing on nothing but their release names are as many runs as there are launches."""
        config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, namespace=NAMESPACE, run_id=RUN_ID)

        with pytest.raises(AssertionError, match="joined by nothing but the run uuid"):
            _install(config=config, launches=deployment_launches)

    def test_a_run_whose_launches_name_no_namespace_is_refused(self, deployment_launches, fake_helm):
        # [FUNCTION DOC] With an empty namespace in the config, asserts installation raises an
        # assertion error saying a namespace is half of every release name.
        """Releases installed into whatever namespace is default reach each other under names nothing holds."""
        config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, namespace="", run_id=RUN_ID, run_uuid=RUN_UUID
        )

        with pytest.raises(AssertionError, match="a namespace is half of every such name"):
            _install(config=config, launches=deployment_launches)

    def test_every_deployment_keeps_the_arguments_it_was_built_with(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Asserts each launch received its own train args and instance id, in order:
        # trainer (no id), inference-a (a), inference-b (b), primary (no id).
        """A deployment's arguments describe only what it carries, so they must not be swapped between launches."""
        _install(config=config, launches=deployment_launches)

        assert [(one.train_args, one.deploy_instance_id) for one in deployment_launches.records] == [
            ("trainer", None),
            ("inference-a", "a"),
            ("inference-b", "b"),
            ("primary", None),
        ]

    def test_it_removes_every_release_it_installed_when_the_run_ends(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Asserts the fake helm saw uninstalls of all four releases in reverse install
        # order (primary, inference b, inference a, trainer).
        """A workers-only release finishes no training, and the driving one only tears itself down minutes later."""
        _install(config=config, launches=deployment_launches)

        assert fake_helm.uninstalled == [
            _release(DeployComponent.PRIMARY),
            _release(DeployComponent.INFERENCE, "b"),
            _release(DeployComponent.INFERENCE, "a"),
            _release(DeployComponent.TRAINER),
        ]

    def test_it_uninstalls_them_even_when_the_run_fails(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Makes the primary launch raise; asserts a runtime error propagates and all
        # four releases were still uninstalled.
        """A failed run that leaked its engine releases would hold the cluster's gpus until someone noticed."""
        deployment_launches.fail_on = DeployComponent.PRIMARY

        with pytest.raises(RuntimeError):
            _install(config=config, launches=deployment_launches)

        assert len(fake_helm.uninstalled) == 4

    def test_it_uninstalls_a_deployment_whose_own_launch_failed(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Makes inference launches raise; asserts a runtime error propagates and the
        # release of the failing inference instance a is among the uninstalled ones.
        """helm may have installed the release before the launch raised, so it is torn down too."""
        deployment_launches.fail_on = DeployComponent.INFERENCE

        with pytest.raises(RuntimeError):
            _install(config=config, launches=deployment_launches)

        assert _release(DeployComponent.INFERENCE, "a") in fake_helm.uninstalled

    def test_a_release_that_refuses_to_be_uninstalled_does_not_strand_the_others(
        self, config, deployment_launches, fake_helm
    ):
        # [FUNCTION DOC] Makes uninstalling inference b fail; asserts the run still completes without
        # error and the other three releases were uninstalled.
        """One release nobody can take down would otherwise leave every earlier one holding gpus for good."""
        fake_helm.fail_to_uninstall = _release(DeployComponent.INFERENCE, "b")

        _install(config=config, launches=deployment_launches)

        assert set(fake_helm.uninstalled) == {
            _release(DeployComponent.PRIMARY),
            _release(DeployComponent.INFERENCE, "a"),
            _release(DeployComponent.TRAINER),
        }

    def test_a_failed_uninstall_does_not_replace_the_failure_that_ended_the_run(
        self, config, deployment_launches, fake_helm
    ):
        # [FUNCTION DOC] Makes both the primary launch and an inference uninstall fail; asserts the
        # raised error is the primary launch's runtime error, not the teardown error.
        """The run's own failure is what has to be read; a teardown error on top of it hides the cause."""
        deployment_launches.fail_on = DeployComponent.PRIMARY
        fake_helm.fail_to_uninstall = _release(DeployComponent.INFERENCE, "b")

        with pytest.raises(RuntimeError, match="the primary deployment failed"):
            _install(config=config, launches=deployment_launches)

    def test_it_refuses_deployments_that_drive_the_run_before_the_workers(
        self, config, deployment_launches, fake_helm
    ):
        # [FUNCTION DOC] Passes the deployment list reversed (primary first); asserts an assertion
        # error about blocking until the run finishes and that nothing was launched.
        """Catching the order here is cheaper than watching a run wait for workers nobody installed."""
        with pytest.raises(AssertionError, match="blocks until the run finishes"):
            _install(config=config, launches=deployment_launches, deployments=list(reversed(_deployments())))

        assert not deployment_launches.records

    def test_it_refuses_deployments_that_carry_no_orchestration_script(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Passes only worker deployments (no primary); asserts an assertion error
        # requiring exactly one primary deployment.
        """Workers alone train nothing, and the pipeline would report a pass for a run that never started."""
        workers_only = [one for one in _deployments() if one.deploy_component is not DeployComponent.PRIMARY]

        with pytest.raises(AssertionError, match="exactly one deployment"):
            _install(config=config, launches=deployment_launches, deployments=workers_only)

    def test_it_refuses_a_deployment_that_carries_the_whole_run_by_itself(
        self, config, deployment_launches, fake_helm
    ):
        # [FUNCTION DOC] Passes a single deployment of the all component; asserts an assertion error
        # saying it carries a whole run in one release, and that nothing was launched.
        """An `all` deployment drives the run and would pass every check below while splitting nothing."""
        deployments = [RunDeployment(deploy_component=DeployComponent.ALL, train_args="everything")]

        with pytest.raises(AssertionError, match="carries a whole run in one release"):
            _install(config=config, launches=deployment_launches, deployments=deployments)

        assert not deployment_launches.records

    def test_it_refuses_two_deployments_that_would_install_one_release(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Passes two unnamed inference deployments plus a primary; asserts an assertion
        # error saying one would install over the other.
        """Unnamed engine deployments overwrite each other, leaving a run with half the engines it counted on."""
        deployments = [
            RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-a"),
            RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-b"),
            RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args="primary"),
        ]

        with pytest.raises(AssertionError, match="install one release over the other"):
            _install(config=config, launches=deployment_launches, deployments=deployments)

    def test_it_refuses_to_name_a_deployment_a_run_holds_only_one_of(self, config, deployment_launches, fake_helm):
        # [FUNCTION DOC] Gives the sole primary deployment an instance id; asserts an assertion error
        # saying the id tells it apart from nothing.
        """An instance id reads as one of several, and a run drives exactly one orchestration script."""
        deployments = [
            RunDeployment(deploy_component=DeployComponent.TRAINER, train_args="trainer"),
            RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args="primary", deploy_instance_id="a"),
        ]

        with pytest.raises(AssertionError, match="tells it apart from nothing"):
            _install(config=config, launches=deployment_launches, deployments=deployments)

    def test_it_fails_when_helm_does_not_know_a_release_it_just_installed(
        self, config, deployment_launches, fake_helm
    ):
        # [FUNCTION DOC] Makes the fake helm report no manifest for any release; asserts an assertion
        # error saying helm does not know the release.
        """A silently absent release turns into a run that waits forever, which is far harder to read."""
        fake_helm.known = False

        with pytest.raises(AssertionError, match="helm does not know it"):
            _install(config=config, launches=deployment_launches)


class TestRunSplitTrainingInto:
    # [CLASS DOC] Tests for the variant that trains into a dump directory under a given test mode:
    # mode propagation and one-time clearing of the dump directory.
    def test_it_trains_every_deployment_under_the_mode_the_side_runs(self, config, tmp_path, launches, fake_helm):
        # [FUNCTION DOC] Runs into a temp dump dir; asserts all four recorded launches used the module mode.
        """A deployment launched under another mode would install a topology the comparison never asked for."""
        _run(config=config, dump_dir=str(tmp_path / "target"))

        assert [one.mode for one in launches.records] == [MODE] * 4

    def test_it_clears_what_an_earlier_run_left_before_the_first_deployment_starts(
        self, config, tmp_path, launches, fake_helm
    ):
        # [FUNCTION DOC] Seeds the dump dir with a sentinel file, lets the recorder observe it, runs;
        # asserts no deployment saw the sentinel.
        """A deployment writing into a stale dump directory would be compared against another run's numbers."""
        dump_dir = _seeded_dump_dir(tmp_path)
        launches.observe_dump_dir = dump_dir

        _run(config=config, dump_dir=dump_dir)

        assert launches.sentinel_seen == [False, False, False, False]

    def test_it_clears_the_dump_directory_once_and_leaves_it_to_the_run_after_that(
        self, config, tmp_path, launches, fake_helm
    ):
        # [FUNCTION DOC] Seeds a sentinel, makes each deployment observe then write it; asserts only
        # the first deployment sees it absent and the later three see it present (cleared once).
        """Clearing it again between deployments would delete what the deployments before it already wrote."""
        dump_dir = _seeded_dump_dir(tmp_path)
        launches.observe_dump_dir = dump_dir
        launches.write_sentinel = True

        _run(config=config, dump_dir=dump_dir)

        assert launches.sentinel_seen == [False, True, True, True]

    def test_a_run_whose_dump_directory_does_not_exist_yet_still_starts(self, config, tmp_path, launches, fake_helm):
        # [FUNCTION DOC] Runs with a nonexistent dump dir path; asserts all four launches happen.
        """The first run under a test name has nothing to clear, which is no reason to refuse to deploy."""
        _run(config=config, dump_dir=str(tmp_path / "never-written"))

        assert len(launches.records) == 4


def _install(
    *, config: ExecuteTrainConfig, launches: "_DeploymentRecorder", deployments: list[RunDeployment] | None = None
) -> None:
    # [FUNCTION DOC] HELPER: Runs split training with the standard deployments (or the given list)
    # using the recorder's launch callable and the given config. Returns nothing.
    run_split_training(
        deployments=_deployments() if deployments is None else deployments, launch=launches.launch, config=config
    )


def _run(*, config: ExecuteTrainConfig, dump_dir: str) -> None:
    # [FUNCTION DOC] HELPER: Runs split training into the dump dir with the standard deployments and
    # a launch bound to the module test mode.
    run_split_training_into(
        deployments=_deployments(), launch=create_launch_of_mode(MODE), config=config, dump_dir=dump_dir
    )


def _seeded_dump_dir(tmp_path: Path) -> str:
    # [FUNCTION DOC] HELPER: Creates a target directory under the temp path containing the sentinel
    # file with stale content; returns its path as a string.
    dump_dir = tmp_path / "target"
    dump_dir.mkdir()
    (dump_dir / SENTINEL_NAME).write_text("an earlier run wrote this")
    return str(dump_dir)


def _deployments() -> list[RunDeployment]:
    # [FUNCTION DOC] HELPER: Returns the standard ordered deployment list: trainer, inference a,
    # inference b, primary, with train args naming each.
    return [
        RunDeployment(deploy_component=DeployComponent.TRAINER, train_args="trainer"),
        RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-a", deploy_instance_id="a"),
        RunDeployment(deploy_component=DeployComponent.INFERENCE, train_args="inference-b", deploy_instance_id="b"),
        RunDeployment(deploy_component=DeployComponent.PRIMARY, train_args="primary"),
    ]


@dataclass
class _DeploymentLaunch:
    # [CLASS DOC] HELPER: Record of one launch: train args, deploy component, instance id, run uuid.
    train_args: str
    deploy_component: DeployComponent
    deploy_instance_id: str | None
    run_uuid: str | None


@dataclass
class _DeploymentRecorder:
    # [CLASS DOC] HELPER: Fake launch callable that records each launch and can be told to raise for
    # a chosen deploy component (fail_on).
    records: list[_DeploymentLaunch] = field(default_factory=list)
    fail_on: DeployComponent | None = None

    def launch(self, train_args: str, config: ExecuteTrainConfig) -> None:
        # [FUNCTION DOC] HELPER: Appends a record of the args and the config's component, instance id
        # and run uuid. Raises a runtime error naming the component if it matches the fail_on component.
        self.records.append(
            _DeploymentLaunch(
                train_args=train_args,
                deploy_component=config.deploy_component,
                deploy_instance_id=config.deploy_instance_id,
                run_uuid=config.run_uuid,
            )
        )
        if config.deploy_component is self.fail_on:
            raise RuntimeError(f"the {config.deploy_component.value} deployment failed")


@pytest.fixture
def deployment_launches() -> _DeploymentRecorder:
    # [FUNCTION DOC] HELPER: Fixture returning a fresh deployment recorder.
    return _DeploymentRecorder()


@dataclass
class _Launch:
    # [CLASS DOC] HELPER: Record of one training launch: train args, test mode, deploy component.
    train_args: str
    mode: FTTestMode
    deploy_component: DeployComponent


@dataclass
class _LaunchRecorder:
    # [CLASS DOC] HELPER: Replacement for the training entry point. Records launches and, when an
    # observed dump directory is set, records whether the sentinel file existed at each launch and
    # optionally writes the sentinel itself.
    records: list[_Launch] = field(default_factory=list)
    observe_dump_dir: str | None = None
    write_sentinel: bool = False
    sentinel_seen: list[bool] = field(default_factory=list)

    def run_training(self, *, train_args: str, mode: FTTestMode, config: ExecuteTrainConfig) -> None:
        # [FUNCTION DOC] HELPER: Records the launch. If no observed dump dir is set, stops. Otherwise
        # appends whether the sentinel file exists, and if write_sentinel is set creates parent dirs
        # and writes the sentinel file (side effect on disk).
        self.records.append(_Launch(train_args=train_args, mode=mode, deploy_component=config.deploy_component))
        if self.observe_dump_dir is None:
            return

        sentinel = Path(self.observe_dump_dir) / SENTINEL_NAME
        self.sentinel_seen.append(sentinel.exists())
        if self.write_sentinel:
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_text("this deployment wrote it")


@pytest.fixture
def launches(monkeypatch) -> _LaunchRecorder:
    # [FUNCTION DOC] HELPER: Fixture that patches the split-deployment module's training entry point
    # with a fresh recorder's method and returns the recorder.
    recorder = _LaunchRecorder()
    monkeypatch.setattr(split_deployment, "run_training", recorder.run_training)
    return recorder


@dataclass
class _FakeHelm:
    # [CLASS DOC] HELPER: Fake helm client. Reports a release manifest as known or unknown, and
    # records removals, raising for one configured release name.
    known: bool = True
    fail_to_uninstall: str | None = None
    uninstalled: list[str] = field(default_factory=list)

    def get_manifest(self, release: str, namespace: str) -> object | None:
        # [FUNCTION DOC] HELPER: Returns an opaque object when releases are known, otherwise none.
        return object() if self.known else None

    def remove(self, *, release: str, namespace: str) -> None:
        # [FUNCTION DOC] HELPER: Raises a runtime error if the release is the configured failing one;
        # otherwise appends the release name to the uninstalled list.
        if release == self.fail_to_uninstall:
            raise RuntimeError(f"helm refused to uninstall {release}")
        self.uninstalled.append(release)


@pytest.fixture
def fake_helm(monkeypatch) -> _FakeHelm:
    # [FUNCTION DOC] HELPER: Fixture that patches the module's helm client and its
    # release-removal-and-wait helper with a fresh fake, and returns the fake.
    helm = _FakeHelm()
    monkeypatch.setattr(split_deployment, "Helm", helm)
    monkeypatch.setattr(split_deployment, "remove_release_and_wait", helm.remove)
    return helm
