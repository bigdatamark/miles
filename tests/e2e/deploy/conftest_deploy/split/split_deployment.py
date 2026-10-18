# [MODULE DOC]
# HELPER: Installs one training run as several separate helm releases ("split deployment") for the
# deploy tests, and validates and cleans up after them. A deployment record names a deploy
# component, its training arguments and an optional instance id, and derives its release name from
# the run id. The run-side factory used by the comparison framework runs the baseline side as one
# plain release and the target side as split deployments (with a fresh run uuid). Installation
# order: in the given order, with the deployment carrying the orchestration script last (its launch
# blocks until the run finishes); every non-final release is checked to be known to helm; in all
# cases (including errors) the releases installed so far are removed in reverse order, ignoring
# (but logging) removal failures.
import dataclasses
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass

from tests.e2e.ft.conftest_ft.app import (
    TARGET_SIDE,
    RunSideFn,
    RunSideRequest,
    remove_release_and_wait,
    run_one_release,
)
from tests.e2e.ft.conftest_ft.execution import run_training
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName
from miles.utils.run_uuid import generate_run_uuid
from miles.utils.workers.types import DeployComponent

logger = logging.getLogger(__name__)


# ============================== deployment types ==============================


@dataclass(frozen=True)
class RunDeployment:
    # [CLASS DOC] HELPER: Immutable description of one deployment (helm release) of a split run: the
    # deploy component it carries, the training argument string to launch it with, and an optional
    # instance id telling apart several deployments of one component.
    deploy_component: DeployComponent
    train_args: str
    deploy_instance_id: str | None = None

    def release(self, run_id: str) -> str:
        # [FUNCTION DOC] Return the serialized helm release name built from the run id, this
        # deployment's component and its instance id.
        return ReleaseName(
            run_id=run_id, deploy_component=self.deploy_component, deploy_instance_id=self.deploy_instance_id
        ).serialize()


LaunchDeploymentFn = Callable[[str, command_utils.ExecuteTrainConfig], None]
BuildDeploymentsFn = Callable[[RunSideRequest], list[RunDeployment]]
BuildBaselineArgsFn = Callable[[FTTestMode, str, bool, command_utils.ExecuteTrainConfig], str]


# ============================== run side wiring ===============================


def create_split_run_side(
    *, build_baseline_args: BuildBaselineArgsFn, build_deployments: BuildDeploymentsFn
) -> RunSideFn:
    # [FUNCTION DOC] Create the function that runs one comparison side. Returns the function; it is
    # built from a callable producing the baseline training arguments and one producing the list of
    # deployments for the target side.
    def run_side(request: RunSideRequest) -> None:
        # [FUNCTION DOC] HELPER: Run one side. A side other than the target: build the baseline
        # arguments from the request's mode, dump directory, dumper switch and configuration, and run
        # one plain release with them. Target side: copy the request with a freshly generated run
        # uuid in its configuration, build the deployments from that copy, and install them as
        # split training into the copy's dump directory (removed first), launching each through the
        # training runner for the request's mode.
        if request.side != TARGET_SIDE:
            baseline = build_baseline_args(request.mode, request.dump_dir, request.enable_dumper, request.config)
            run_one_release(dataclasses.replace(request, train_args=baseline))
            return

        split = dataclasses.replace(request, config=dataclasses.replace(request.config, run_uuid=generate_run_uuid()))
        run_split_training_into(
            deployments=build_deployments(split),
            launch=create_launch_of_mode(split.mode),
            config=split.config,
            dump_dir=split.dump_dir,
        )

    return run_side


def create_launch_of_mode(mode: FTTestMode) -> LaunchDeploymentFn:
    # [FUNCTION DOC] Return a launch function bound to the test mode.
    def launch(train_args: str, config: command_utils.ExecuteTrainConfig) -> None:
        # [FUNCTION DOC] HELPER: Run the fault-tolerance training runner with the given arguments
        # and configuration and the bound mode (blocks until that launch ends).
        run_training(train_args=train_args, mode=mode, config=config)

    return launch


# ========================= installing the deployments =========================


def run_split_training_into(
    *,
    deployments: list[RunDeployment],
    launch: LaunchDeploymentFn,
    config: command_utils.ExecuteTrainConfig,
    dump_dir: str,
) -> None:
    # [FUNCTION DOC] Delete the dump directory tree if it exists (so no stale output is read), then
    # install and run the deployments with the split training function. Side effects: file
    # deletion, cluster releases.
    if os.path.exists(dump_dir):
        shutil.rmtree(dump_dir)

    run_split_training(deployments=deployments, launch=launch, config=config)


def run_split_training(
    *, deployments: list[RunDeployment], launch: LaunchDeploymentFn, config: command_utils.ExecuteTrainConfig
) -> None:
    # [FUNCTION DOC] Install a split run. Asserts the deployments form one run (see the validation
    # helper), that the configuration has a namespace (release names are resolved in-cluster) and a
    # run uuid (the only thing joining the separately installed deployments). The driver release is
    # the last deployment's. For each deployment in order: record its release name as installed,
    # log it, and call the launch function with its training arguments and a copy of the
    # configuration carrying its deploy component and instance id; after a launch returns, for
    # releases other than the driver release, assert helm knows the release in the namespace. The
    # driver launch blocks until the run finishes. In all cases, finally remove every release
    # recorded as installed, in reverse order, waiting for each removal; removal failures are
    # logged and do not stop the remaining removals. Exceptions from launches propagate.
    _assert_deployments_form_one_run(deployments)
    assert config.namespace, (
        "the deployments of one run reach each other by in-cluster release names, and a namespace is half of "
        "every such name"
    )
    assert config.run_uuid is not None, (
        f"the deployments of run {config.run_id} are installed by launches of their own and joined by nothing but "
        f"the run uuid, so whatever installs them all has to name it"
    )

    driver_release = deployments[-1].release(config.run_id)
    installed: list[str] = []
    try:
        for deployment in deployments:
            release = deployment.release(config.run_id)
            installed.append(release)
            logger.info(
                f"Installing {release}, the {deployment.deploy_component.value} deployment of run "
                f"{config.run_id} ({config.run_uuid})"
            )
            launch(
                deployment.train_args,
                dataclasses.replace(
                    config,
                    deploy_component=deployment.deploy_component,
                    deploy_instance_id=deployment.deploy_instance_id,
                ),
            )
            if release != driver_release:
                _assert_release_installed(release=release, namespace=config.namespace)
    finally:
        for release in reversed(installed):
            _remove_whatever_can_be_removed(release=release, namespace=config.namespace)


# ============================= cleanup and checks =============================


def _remove_whatever_can_be_removed(*, release: str, namespace: str) -> None:
    # [FUNCTION DOC] HELPER: Remove the release from the namespace and wait until it is gone; any
    # exception is logged with its traceback (saying the release may still hold GPUs) and
    # swallowed.
    try:
        remove_release_and_wait(release=release, namespace=namespace)
    except Exception:
        logger.exception(
            f"Removing {release} from namespace {namespace} failed, so it may still hold the gpus of this run; "
            f"the releases installed before it are removed next"
        )


def _assert_deployments_form_one_run(deployments: list[RunDeployment]) -> None:
    # [FUNCTION DOC] HELPER: Validate a deployment list. Asserts: (1) none of the components carries
    # a whole run in one release (every component must be a split one); (2) exactly one deployment
    # carries the orchestration script, and it is the last in the list; (3) no two deployments
    # share the same (component, instance id) pair; (4) a deployment has an instance id only if its
    # component accepts one. Messages explain each violation.
    unsplit = [one.deploy_component.value for one in deployments if not one.deploy_component.is_split()]
    assert not unsplit, (
        f"these deployments install {unsplit}, which carries a whole run in one release: a run installed that way "
        f"is the unsplit shape, and nothing here would be deployed apart from anything else"
    )

    drivers = [one for one in deployments if one.deploy_component.deploys_orchestration_script()]
    assert len(drivers) == 1, (
        f"exactly one deployment of a run carries the orchestration script, and these carry it {len(drivers)} "
        f"time(s): {[one.deploy_component.value for one in deployments]}"
    )
    assert deployments[-1] is drivers[0], (
        f"installing the orchestration script blocks until the run finishes, so it goes last; here it is "
        f"{deployments.index(drivers[0])} of {len(deployments)}"
    )

    named = [(one.deploy_component, one.deploy_instance_id) for one in deployments]
    assert len(set(named)) == len(named), (
        f"two of these deployments are named the same and would install one release over the other: "
        f"{[(one.value, instance) for one, instance in named]}"
    )

    for deployment in deployments:
        assert deployment.deploy_instance_id is None or deployment.deploy_component.takes_instance_id(), (
            f"a run holds one {deployment.deploy_component.value} deployment, so naming this one "
            f"{deployment.deploy_instance_id!r} tells it apart from nothing"
        )


def _assert_release_installed(*, release: str, namespace: str) -> None:
    # [FUNCTION DOC] HELPER: Assert helm returns a manifest for the release in the namespace (not
    # nothing), i.e. an install that returned without error really created the release.
    assert Helm.get_manifest(release, namespace) is not None, (
        f"installing {release} returned without an error, but helm does not know it in namespace {namespace}: the "
        f"run would wait forever for workers nothing carries"
    )
