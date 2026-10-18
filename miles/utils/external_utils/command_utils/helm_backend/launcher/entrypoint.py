# [MODULE DOC] Kubernetes training launcher. execute_train installs (or upgrades) a helm release
# for a training run and follows it: it validates the run id, finds the installed manifest,
# decides the run uuid, parses the training arguments exactly as the pods will, computes worker
# specs, renders chart values and a launch record, guards upgrades with a manifest diff, defuses a
# superseded orchestrator generation, runs helm upgrade, and (for releases that carry the
# orchestration script) follows pod logs until the run exits, collecting a diagnosis on failure.
# Also: leftover CI release cleanup, trainer-controller address reporting for split deployments,
# and a YAML dumper that single-quotes string values and leaves keys plain. Splitting a run into
# several releases is by "deploy component"; only releases carrying the orchestration script are
# followed.
from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.ray.specs.train import (
    TRAINER_CONTROLLER_ADDRS_FLAG,
    compute_trainer_controller_pool_id,
    compute_trainer_ids,
    specs_trainer_controller,
)
from miles.utils.arguments import parse_args
from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR
from miles.utils.external_utils.command_utils.base_backend import (
    CLUSTER_BACKEND_FLAG,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import (
    MOONCAKE_INIT_KWARGS_FLAG,
    ArgvManipulator,
    chart_dir,
    repo_base_dir,
    train_env_vars,
)
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher import manifest_diff
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import CI_LABEL, Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.hot_restart import (
    compute_orchestrator_object_key,
    plan_hot_restart,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.launch_record import (
    LaunchRecord,
    installed_launch_record_file,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest, ManifestObjectKey
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import farewell, with_observability
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.diagnosis import collect_diagnosis
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability.pod_facts import observed_pod
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    InfraInfo,
    LaunchPlan,
    MooncakeInfo,
    MooncakePlan,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName, RunFiles, RunNames
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.observer import wait_for_run
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.file_utils import atomic_write_text
from miles.utils.object_store import ObjectStoreBackend
from miles.utils.run_uuid import generate_run_uuid, validate_run_uuid
from miles.utils.workers.naming import DNS_LABEL_PATTERN
from miles.utils.workers.serving.utils import override_argv, override_env
from miles.utils.workers.types import ClusterBackend, DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_cell_addrs
from miles.utils.workers.worker_spec import RPC_PORT_NAME

logger = logging.getLogger(__name__)

_RUN_UUID_FLAG = "--run-uuid"
_ENV_REPORT_FLAG = "--env-report"
_WANDB_RUN_ID_FLAG = "--wandb-run-id"
_RUN_ID_PATTERN = re.compile(DNS_LABEL_PATTERN)


class RunExitedError(SystemExit):
    # [CLASS DOC] Process-exit signal carrying the failed run's exit code; raised so the launcher
    # process exits with the same status. Attribute exit_code.
    def __init__(self, exit_code: int) -> None:
        # [FUNCTION DOC] Initializes the system exit with the code and also stores it as an
        # attribute.
        super().__init__(exit_code)
        self.exit_code = exit_code


def execute_train(*, request: ExecuteTrainRequest, config: ExecuteTrainConfig) -> None:
    # [FUNCTION DOC] Launches (or relaunches) a run. Steps: (1) assert the run id is a valid DNS
    # label. (2) Build the release name from run id, deploy component and instance id; fetch the
    # installed manifest (none if absent). (3) Resolve the run uuid. (4) Build the training
    # environment and parse the training arguments the way pods will see them; assert the parsed
    # deploy component and instance id match the config. (5) Compute worker specs with the
    # environment applied. (6) Locate the chart and the shared root from infrastructure values, and
    # the run directory. (7) In CI runs, uninstall leftover CI releases of other runs; build chart
    # dependencies. (8) Decide the orchestrator command (only if this release carries the
    # orchestration script), the hot restart plan, and, for trainer-only releases, the trainer
    # controller addresses. (9) Prepare a values file and a launch record file. (10) If a release is
    # installed, render a proposal carrying the installed state file and diff it against the
    # installed manifest to learn whether the orchestrator would be rebuilt; a rebuilt orchestrator
    # that carries the orchestration script gets a fresh state file, otherwise the carried one is
    # kept. (11) Render the final proposal; for an installed release, assert the upgrade is
    # allowed (diff reuse when the state file is unchanged). (12) If the orchestrator is rebuilt,
    # defuse the previous generation. (13) Write the launch record, run helm upgrade. (14) A release
    # without the orchestration script ends here, logging how to uninstall and reach it. A split
    # deployment logs the object store arguments to give the other deployments. (15) Otherwise
    # follow the run until it finishes; a non-zero run exit raises the run-exited error.
    # Side effects: files under the shared run directory, helm and kubectl commands, cluster
    # changes.
    run_id = config.run_id
    assert _RUN_ID_PATTERN.fullmatch(
        run_id
    ), f"run_id {run_id!r} names every object this run installs, so it has to match {_RUN_ID_PATTERN.pattern}"

    namespace = config.namespace
    release = ReleaseName(
        run_id=run_id,
        deploy_component=config.deploy_component,
        deploy_instance_id=config.deploy_instance_id,
    ).serialize()
    installed_manifest = Helm.get_manifest(release, namespace)
    run_uuid = _resolve_run_uuid(config, installed_manifest=installed_manifest, release=release)
    env = train_env_vars(request, {}, config=config)
    pod_argv, args = _compute_train_argv(request, run_uuid=run_uuid, release=release, namespace=namespace, env=env)
    deploy_component = DeployComponent(args.deploy_component)
    assert (deploy_component, args.deploy_instance_id) == (config.deploy_component, config.deploy_instance_id), (
        f"the run's pods are told {deploy_component.value}/{args.deploy_instance_id!r}, the release is named "
        f"{config.deploy_component.value}/{config.deploy_instance_id!r}"
    )
    deploys_orchestration_script = deploy_component.deploys_orchestration_script()

    with override_env(env):
        specs = compute_specs(args)
    chart = chart_dir(repo_base_dir=repo_base_dir)
    shared_root = InfraInfo.shared_root(InfraInfo.load(chart, list(config.helm_values)), namespace=namespace)
    run_directory = RunFiles.run_dir(shared_root=shared_root, run_id=run_id)

    if config.ci_run:
        _uninstall_leftover_ci_releases(namespace, keep_run_id=run_id)
    Helm.build_dependencies(chart)

    orchestrator_command = ["python", request.train_script, *pod_argv] if deploys_orchestration_script else []
    hot_restart_plan = plan_hot_restart(
        components=config.parsed_hot_restart,
        deploy_component=deploy_component,
        release=release,
        installed_manifest=installed_manifest,
    )
    reachable_at = (
        _compute_trainer_controller_addrs(args, release=release, namespace=namespace)
        if deploy_component is DeployComponent.TRAINER
        else {}
    )
    values_path = RunFiles.new_values_file(run_directory=run_directory)
    values_files: list[str | Path] = [*config.helm_values, values_path]
    record_path = RunFiles.new_record_file(run_directory=run_directory)

    def render_and_propose(state_file: Path | None) -> tuple[LaunchRecord, Manifest]:
        # [FUNCTION DOC] HELPER: Builds the launch plan for the given orchestrator state file (none
        # becomes an empty string): run id, release, namespace, orchestrator command, worker argv,
        # environment, colocate flag, mooncake plan, preparation commands, extra manifests, restart
        # stamp and stamped components. Computes the launch record from the plan, the values file
        # and the controller addresses. The plan given to the values builder additionally names the
        # launch record file the pods should read (the installed one if a release exists, else the
        # new record path). Writes the chart values file, then runs a dry-run upgrade to get the
        # proposed manifest. Returns (record, proposed manifest). Overwrites the values file on each
        # call.
        plan = LaunchPlan(
            run_id=run_id,
            release=release,
            namespace=namespace,
            state_file=str(state_file) if state_file is not None else "",
            orchestrator_command=orchestrator_command,
            worker_argv=pod_argv,
            env=env,
            colocate=bool(args.colocate),
            mooncake_plan=_compute_mooncake_plan(args),
            prepare_cmd=request.prepare_cmd,
            extra_manifests=request.extra_manifests,
            restart_at=hot_restart_plan.restart_at,
            stamped_components=hot_restart_plan.stamped_components,
        )
        computed = LaunchRecord.compute(plan=plan, values_file=values_path, reachable_at=reachable_at)
        rendered = plan.model_copy(
            update={
                "launch_record": _compute_pod_record_file(
                    installed_manifest=installed_manifest, record_path=record_path
                ),
            }
        )
        _write_helm_values(values_path, build_values(specs, rendered).as_values())
        return computed, Helm.render_upgrade(
            release=release, namespace=namespace, chart=chart, values_files=values_files
        )

    carried_state_file = (
        _carried_state_file(installed_manifest=installed_manifest, release=release)
        if deploys_orchestration_script
        else None
    )

    rebuilds_orchestrator = True
    if installed_manifest is not None:
        _, carried_manifest = render_and_propose(carried_state_file)
        carried_diff = manifest_diff.diff_manifests(
            before=installed_manifest,
            after=carried_manifest,
            allow_diff_object_keys=hot_restart_plan.allow_diff_object_keys,
        )
        rebuilds_orchestrator = carried_diff.rebuilds(key=compute_orchestrator_object_key(release))
    state_file = (
        RunFiles.new_state_file(run_directory=run_directory)
        if rebuilds_orchestrator and deploys_orchestration_script
        else carried_state_file
    )
    record, proposed_manifest = render_and_propose(state_file)

    if installed_manifest is not None:
        _assert_upgrade_is_allowed(
            diff=(
                carried_diff
                if state_file == carried_state_file
                else manifest_diff.diff_manifests(
                    before=installed_manifest,
                    after=proposed_manifest,
                    allow_diff_object_keys=hot_restart_plan.allow_diff_object_keys,
                )
            ),
            release=release,
            skip_upgrade_check=config.skip_upgrade_check,
            allow_diff_object_keys=hot_restart_plan.allow_diff_object_keys,
        )
    if rebuilds_orchestrator:
        _defuse_previous_generation(
            release, namespace=namespace, superseded_state_file=carried_state_file, state_file=state_file
        )

    record.write(path=record_path)
    logger.info(f"What this launch launched is recorded under {record_path}")

    Helm.upgrade(
        release=release,
        namespace=namespace,
        chart=chart,
        values_files=values_files,
        ci_run=config.ci_run,
    )

    if not deploys_orchestration_script:
        logger.info(
            f"Installed {release}, which carries no orchestration script: it has no training to finish, so it stays "
            f"up until you uninstall it with `helm uninstall {release} --namespace {namespace}`"
        )
        if reachable_at:
            logger.info(f"Reach it with {_describe_trainer_controller_addrs(reachable_at)}")
        return

    if deploy_component.is_split():
        logger.info(
            f"The other deployments of this run share the object store of {release}, so give each of them "
            f"{_describe_shared_object_store(_compute_mooncake_plan(args), release=release, namespace=namespace)}"
        )

    _follow_until_finished(release=release, namespace=namespace, state_file=state_file)


def _compute_trainer_controller_addrs(args: Any, *, release: str, namespace: str) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: For each trainer id from the arguments, finds the trainer controller
    # pool spec, takes the address of cell 0's RPC port, qualifies the host to a fully qualified
    # service name in the namespace, and returns a mapping from trainer id to "host:port".
    specs_by_pool_id = {spec.name: spec for spec in specs_trainer_controller(args)}
    addrs = {}
    for trainer_id in compute_trainer_ids(args):
        spec = specs_by_pool_id[compute_trainer_controller_pool_id(trainer_id)]
        addr = static_cell_addrs(spec=spec, release=release, cell_index=0)[RPC_PORT_NAME]
        host = RunNames.service_fqdn(name=addr.host, namespace=namespace)
        addrs[trainer_id] = f"{host}:{addr.port}"
    return addrs


def _describe_trainer_controller_addrs(reachable_at: dict[str, str]) -> str:
    # [FUNCTION DOC] HELPER: The command-line flag for trainer controller addresses followed by
    # space-separated "trainer=address" entries.
    entries = " ".join(f"{trainer_id}={addr}" for trainer_id, addr in reachable_at.items())
    return f"{TRAINER_CONTROLLER_ADDRS_FLAG} {entries}"


def _describe_shared_object_store(plan: MooncakePlan | None, *, release: str, namespace: str) -> str:
    # [FUNCTION DOC] HELPER: Asserts a mooncake plan exists (a split run's orchestration release
    # must run the object store master). Returns the command-line text selecting the mooncake
    # backend with init kwargs (json, shell-quoted) pointing at the release's master service host.
    assert plan is not None, (
        f"{release} carries the orchestration script of a split run, so it runs the object store master the other "
        f"deployments redeem their references at, and a run without one shares nothing"
    )
    init_kwargs = MooncakeInfo.cluster_init_kwargs(plan, host=MooncakeInfo.master_service_host(release, namespace))
    return (
        f"--object-store-backend {ObjectStoreBackend.MOONCAKE.value} "
        f"{MOONCAKE_INIT_KWARGS_FLAG} {shlex.quote(json.dumps(init_kwargs))}"
    )


def _follow_until_finished(*, release: str, namespace: str, state_file: Path) -> None:
    # [FUNCTION DOC] HELPER: Streams observability for all pods of the release while waiting for
    # the run to end, reading the orchestrator pod and the currently active state file from the
    # cluster. If the run's exit code is non-zero, collects a diagnosis and raises the run-exited
    # error with that code. In every case logs a farewell message pointing at the orchestrator
    # workload. Interrupting the launcher stops watching, not the run.
    logger.info(f"Following every pod of {release}; ctrl+c stops watching, not the run")
    orchestrator_workload = naming.component_name(release, naming.ORCHESTRATOR_COMPONENT)

    try:
        with with_observability(namespace=namespace, selector=Kubectl.release_selector(release)):
            outcome = wait_for_run(
                state_file=state_file,
                read_pod=lambda: observed_pod(namespace, orchestrator_workload),
                read_active_state_file=lambda: _active_state_file(release=release, namespace=namespace),
            )

        if outcome.exit_code != 0:
            _collect_diagnosis(release=release, namespace=namespace, state_file=state_file)
            raise RunExitedError(outcome.exit_code)
    finally:
        logger.info(farewell(namespace=namespace, release=release, workload=orchestrator_workload))


def _resolve_run_uuid(config: ExecuteTrainConfig, *, installed_manifest: Manifest | None, release: str) -> str:
    # [FUNCTION DOC] HELPER: Priority: (1) the config's run uuid, validated. (2) Asserts the
    # deployment is not a split component (those must be given a uuid). (3) The uuid found in the
    # installed orchestrator container's arguments, if a release is installed and has one. (4) A
    # newly generated uuid.
    if (given := config.run_uuid) is not None:
        return validate_run_uuid(given)

    assert not config.deploy_component.is_split(), (
        f"--deploy-component {config.deploy_component.value} installs one part of a run whose other parts are "
        f"installed by other launches, and they are joined by nothing but the run uuid, so the layer that deploys "
        f"them all has to name it with --run-uuid"
    )

    if installed_manifest is not None:
        installed = installed_manifest.flag_value(
            _RUN_UUID_FLAG,
            stateful_set=RunNames.orchestrator_object(release=release),
            container=naming.ORCHESTRATOR_COMPONENT,
        )
        if installed is not None:
            return installed

    return generate_run_uuid()


def _compute_train_argv(
    request: ExecuteTrainRequest, *, run_uuid: str, release: str, namespace: str, env: dict[str, str]
) -> tuple[list[str], Any]:
    # [FUNCTION DOC] HELPER: Builds the pod argument list: the model arguments (shell-safe, split)
    # followed by the request's train arguments (split). Asserts the arguments do not define the
    # env-report flag (the launcher supplies it). Adds the cluster backend flag (Kubernetes) and
    # the run uuid flag if the user did not give them. Parses the list with the training argument
    # parser under temporarily overridden argv and environment; asserts the launcher report
    # variable is not in the parsed training env vars. If wandb is on and no wandb run id was given,
    # generates one, stores it on the parsed args and adds it to the argument list. Finally applies
    # the cluster mooncake master to the argument list. Returns (pod argv, parsed args).
    # TODO in code: the wandb id handling should move to the args refactor.
    argv = [*shlex.split(shell_safe_model_args(request.megatron_model_type)), *shlex.split(request.train_args)]
    assert not ArgvManipulator.is_defined(argv, _ENV_REPORT_FLAG), (
        f"{_ENV_REPORT_FLAG} is what this launcher tells the pods about the launch that installed them, and an "
        f"argument of that name outranks it, so the pods would report a launch that never happened; drop it"
    )
    if not ArgvManipulator.is_defined(argv, CLUSTER_BACKEND_FLAG):
        argv = ArgvManipulator.set(argv, CLUSTER_BACKEND_FLAG, ClusterBackend.KUBERNETES.value)
    if not ArgvManipulator.is_defined(argv, _RUN_UUID_FLAG):
        argv = ArgvManipulator.set(argv, _RUN_UUID_FLAG, run_uuid)

    with override_argv(argv), override_env(env):
        args = parse_args()
    assert LAUNCHER_REPORT_ENV_VAR not in args.train_env_vars

    # TODO: remove after args refactor handles wandb ids
    if args.use_wandb and args.wandb_run_id is None:
        args.wandb_run_id = _generate_wandb_run_id()
        argv = ArgvManipulator.set(argv, _WANDB_RUN_ID_FLAG, args.wandb_run_id)

    pod_argv = MooncakeInfo.with_cluster_master(
        argv, plan=_compute_mooncake_plan(args), host=MooncakeInfo.master_service_host(release, namespace)
    )
    return pod_argv, args


def _generate_wandb_run_id() -> str:
    # [FUNCTION DOC] HELPER: A fresh run id from wandb's id generator (imported lazily).
    from wandb.sdk.lib.runid import generate_id

    return generate_id()


def _compute_mooncake_plan(args) -> MooncakePlan | None:
    # [FUNCTION DOC] HELPER: None unless the args' deploy component deploys the orchestration
    # script; otherwise the mooncake plan derived from the args.
    if not DeployComponent(args.deploy_component).deploys_orchestration_script():
        return None
    return MooncakeInfo.plan_of_args(args)


def _compute_pod_record_file(*, installed_manifest: Manifest | None, record_path: Path) -> str | None:
    # [FUNCTION DOC] HELPER: The launch record file path the pods should read: the new record path
    # as text for a first install; otherwise the record file already attached to the installed
    # manifest (may be none), so an upgrade keeps pods pointing at the original record.
    if installed_manifest is None:
        return str(record_path)
    return installed_launch_record_file(manifest=installed_manifest)


def _carried_state_file(*, installed_manifest: Manifest | None, release: str) -> Path | None:
    # [FUNCTION DOC] HELPER: None if nothing is installed. Otherwise the state file named by the
    # orchestrator container of the installed stateful set; asserts one is named (message tells to
    # uninstall or use a new run id).
    if installed_manifest is None:
        return None

    attached_state_file = installed_manifest.state_file(
        stateful_set=RunNames.orchestrator_object(release=release), container=naming.ORCHESTRATOR_COMPONENT
    )
    assert attached_state_file is not None, (
        f"Run {release} is installed but its orchestrator names no state file, so this launch cannot tell what it "
        f"is watching; uninstall it, or launch under a new run id"
    )
    return attached_state_file


def _active_state_file(*, release: str, namespace: str) -> Path | None:
    # [FUNCTION DOC] HELPER: Reads the orchestrator stateful set from the cluster as yaml. If the
    # read fails with a not-found message, returns none; any other failure raises a runtime error.
    # Otherwise parses it as a manifest and returns the state file it names (asserting one exists).
    # Used while following the run to notice a newer orchestrator generation.
    object_name = RunNames.orchestrator_object(release=release)
    result = Kubectl.run_raw("get", "statefulset", object_name, "--namespace", namespace, "--output", "yaml")
    if result.returncode != 0:
        error = (result.stderr + result.stdout).lower()
        if "not found" in error or "notfound" in error:
            return None
        raise RuntimeError(
            f"Cannot read the active orchestrator generation of {release}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    manifest = Manifest.parse(result.stdout, namespace=namespace)
    return _carried_state_file(installed_manifest=manifest, release=release)


def _assert_upgrade_is_allowed(
    *,
    diff: manifest_diff.ManifestDiffs,
    release: str,
    skip_upgrade_check: bool,
    allow_diff_object_keys: frozenset[ManifestObjectKey],
) -> None:
    # [FUNCTION DOC] HELPER: Keyword-only. If the diff between installed and proposed manifests is
    # allowed, logs a summary of the allowed changes and returns. Otherwise builds an error message
    # describing the diff (mentioning the objects this launch may rebuild, if any) and advising a
    # new run id or the skip option. Raises a system exit with that message unless upgrade checking
    # is skipped, in which case it logs the message as a warning and continues.
    if diff.is_allowed:
        logger.info(
            f"Run {release} already exists; upgrading it with these allowed changes:\n{diff.summarize_allowed_changes()}"
        )
        return

    allowed = ", ".join(sorted(f"{key.kind}/{key.name}" for key in allow_diff_object_keys))
    rebuildable = f", and more than the objects this launch is allowed to rebuild ({allowed})" if allowed else ""
    message = (
        f"Run {release} already exists and the relaunch would change more than its size{rebuildable}:\n"
        f"{diff.describe()}\n"
        f"launch under a new run id, or pass --skip-upgrade-check to apply this anyway and accept the restarts"
    )
    if not skip_upgrade_check:
        raise SystemExit(message)
    logger.warning(f"upgrade check skipped: {message}")


def _belongs_to_run(release: str, *, run_id: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the release name parses under the naming rules and carries the
    # run id. A release that does not parse (or any parse error, logged with its trace) belongs to
    # no run.
    try:
        return (parsed := ReleaseName.parse(release)) is not None and parsed.run_id == run_id
    except Exception:
        logger.info(
            f"Release {release} does not follow the Miles naming rules, so it belongs to no run", exc_info=True
        )
        return False


def _uninstall_leftover_ci_releases(namespace: str, *, keep_run_id: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Lists releases in the namespace carrying the CI label, drops those that
    # belong to the kept run, logs and uninstalls the rest one by one (a failure raises and stops
    # the cleanup). Returns the uninstalled release names.
    listed = Helm.list_releases(namespace=namespace, selector=f"{CI_LABEL}=true")
    releases = [release for release in listed if not _belongs_to_run(release, run_id=keep_run_id)]
    for release in releases:
        logger.info(f"Uninstalling the leftover ci release {release} before this run installs its own")
        Helm.uninstall(release=release, namespace=namespace)
    return releases


def _defuse_previous_generation(
    release: str, *, namespace: str, superseded_state_file: Path | None, state_file: Path | None
) -> None:
    # [FUNCTION DOC] HELPER: Keyword-only after the release. Prepares the cluster for a new
    # orchestrator generation. If there is a superseded state file: writes a marker file next to
    # it (atomically, creating directories) whose content is the new state file path, telling the
    # old orchestrator to leave the release uninstall to the new generation. Then deletes the
    # release's pending uninstall job (failing if deletion fails) so it cannot remove the release
    # being installed.
    if superseded_state_file is not None:
        marker = RunFiles.superseded_marker(state_file=superseded_state_file)
        logger.info(
            f"Writing {marker}, so the orchestrator of {superseded_state_file} leaves the uninstall of {release} "
            f"to the generation this launch installs"
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(marker, f"{state_file}\n")

    job = RunNames.uninstall_job(release=release)
    logger.info(f"Deleting {job} if it is pending, so it cannot uninstall the release this launch installs")
    Kubectl.delete_job(job, namespace=namespace, check=True)


def _collect_diagnosis(*, release: str, namespace: str, state_file: Path) -> None:
    # [FUNCTION DOC] HELPER: Finds all releases of the run (logging when there is more than one),
    # and collects a diagnosis of their pods into the state file's directory. Any failure to
    # collect is logged as a warning and swallowed. On success logs the diagnosis directory and,
    # if incomplete, warns with the names of what could not be collected.
    releases = _releases_of_run(release=release, namespace=namespace)
    if len(releases) > 1:
        logger.info(
            f"This run is deployed as {len(releases)} releases, and a deployment other than {release} can be what "
            f"failed, so the diagnosis covers all of them: {', '.join(releases)}"
        )

    try:
        diagnosis = collect_diagnosis(
            namespace=namespace,
            output_dir=state_file.parent,
            selector=Kubectl.releases_selector(releases),
            state_file=state_file,
        )
    except Exception:
        logger.warning("Could not collect a diagnosis of the failed run", exc_info=True)
        return

    logger.info(f"The pods of this failed run are described under {diagnosis.directory}")
    if not diagnosis.is_complete:
        logger.warning(f"The diagnosis is incomplete, these could not be collected: {', '.join(diagnosis.missing)}")


def _releases_of_run(*, release: str, namespace: str) -> list[str]:
    # [FUNCTION DOC] HELPER: If the release name does not parse, returns just it. Otherwise lists
    # releases whose names start with the run's prefix (regular-expression filter on the escaped
    # prefix), keeps those belonging to the run, adds the given release, and returns the sorted
    # unique names. A listing failure logs a warning and returns only the given release.
    parsed = ReleaseName.parse(release)
    if parsed is None:
        return [release]

    try:
        listed = Helm.list_releases(
            namespace=namespace, name_filter=f"^{re.escape(ReleaseName.run_prefix(run_id=parsed.run_id))}"
        )
    except Exception:
        logger.warning(f"Could not list what else run {parsed.run_id} installed; diagnosing {release} alone")
        logger.debug("Listing the releases of the run failed", exc_info=True)
        return [release]

    return sorted({one for one in listed if _belongs_to_run(one, run_id=parsed.run_id)} | {release})


def _write_helm_values(path: Path, values: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Creates parent directories and writes the values as block-style yaml
    # with sorted keys using the custom dumper (single-quoted strings, plain keys). Overwrites.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(values, Dumper=_HelmValuesDumper, default_flow_style=False, sort_keys=True))


class _HelmValuesDumper(yaml.SafeDumper):
    # [CLASS DOC] HELPER: Safe yaml dumper for helm values. Mapping keys are emitted without
    # quoting style, while string values (registered at module end) are single-quoted so values
    # such as numbers-as-text or booleans-as-text stay strings for helm.
    def represent_mapping(self, tag: str, mapping: Any, flow_style: bool | None = None) -> yaml.MappingNode:
        # [FUNCTION DOC] HELPER: Builds the mapping node with the base behavior, then clears the
        # style of every key node (so keys are plain even though strings are quoted). Returns it.
        node = super().represent_mapping(tag, mapping, flow_style=flow_style)
        for key, _ in node.value:
            key.style = None
        return node


def _represent_helm_value_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    # [FUNCTION DOC] HELPER: Yaml representer: a string scalar with single-quote style.
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="'")


_HelmValuesDumper.add_representer(str, _represent_helm_value_str)
