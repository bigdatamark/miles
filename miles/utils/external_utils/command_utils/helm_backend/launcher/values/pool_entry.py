# [MODULE DOC]
# Converts one worker spec into a pool entry of the helm values. Because an entry describes a whole
# pool rather than one pod, the launch command and environment are rendered once with a fixed cell
# index and sentinel values for per-pod data, and then sentinels become runtime placeholders.
# Command specs use their own launch command; serve specs run the serve module (wrapped by the
# process supervisor when a pod hosts several workers).
from __future__ import annotations

import shlex

from miles.utils.external_utils.colocate_pairing.config import PairingLayout
from miles.utils.external_utils.command_utils.base_backend import TRAINER_ROLE
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    PoolEntry,
    PortEntry,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    SECTION_OF_CATEGORY,
    TRAINER_ENGINES_SECTION,
    LaunchPlan,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.placeholders import (
    LEADER_ADDRESS_PLACEHOLDER,
    RENDERED_CELL_INDEX,
    WORKER_INDEX_SENTINEL,
    real_or_sentinel_gpu_ids,
    sentinel_to_placeholder,
    sentinels_to_placeholders,
)
from miles.utils.workers.argv_utils import python_argv_prefix
from miles.utils.workers.naming import compute_port_name
from miles.utils.workers.types import PlatformAccess
from miles.utils.workers.worker_provider.kubernetes.helm import env
from miles.utils.workers.worker_spec import (
    BaseWorkerSpec,
    CommandWorkerSpec,
    HostAndPort,
    LaunchCommandContext,
    NamedHostAndPorts,
    ServeWorkerSpec,
)

_BIND_HOST = "0.0.0.0"

_SERVE_MODULE = "miles.utils.workers.serving.serve"
_SUPERVISOR_MODULE = "miles.utils.workers.process_supervisor"
_SPECS_FN = "miles.ray.specs.entrypoint.compute_specs_from_argv"


def build_entry(
    spec: BaseWorkerSpec,
    plan: LaunchPlan,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    pairing_layout: PairingLayout | None = None,
) -> PoolEntry:
    # [FUNCTION DOC] Builds the pool entry for a spec. Asserts the spec has at least one cell. Name
    # and pool id are the spec name; object name is the component name from release and spec name.
    # Command: the spec command (with optional trainer prepare wrapping). Ports: one per spec port
    # with the computed port name and static port. Env: command specs' environment (none when
    # empty; serve specs have none). Meta: gpu id list label. Service account from platform access.
    # Replicas: cell count. Size: pods per cell only when above one. Resources: gpu limit only if
    # the pod needs gpus. Restart stamp: rendered for the spec name. The pairing layout only
    # determines whether pods are sub-node (use runtime base gpu id).
    assert spec.scheduling.num_cells > 0, (
        f"Spec '{spec.name}' asks for {spec.scheduling.num_cells} cells; a spec a run has turned off is dropped "
        f"before conversion, because a values entry always renders at least one pod"
    )
    is_sub_node = _is_sub_node(pairing_layout)
    context = _launch_context(
        spec,
        addresses=addresses,
        cell_index=RENDERED_CELL_INDEX,
        worker_in_cell_index=WORKER_INDEX_SENTINEL,
        is_sub_node=is_sub_node,
    )
    pods_per_cell = spec.scheduling.pods_per_cell()
    gpus_per_pod = spec.scheduling.gpus_per_pod()

    return PoolEntry(
        name=spec.name,
        object_name=naming.component_name(plan.release, spec.name),
        pool_id=spec.name,
        command=_with_prepare_cmd(_command_of_spec(spec, context, plan=plan), spec, plan=plan),
        ports=[PortEntry(name=compute_port_name(port.name), port=port.static_port) for port in spec.port_infos],
        env=_command_env_of_spec(spec, context, addresses=addresses, is_sub_node=is_sub_node) or None,
        meta=_meta_of_spec(spec) or None,
        service_account_name=_service_account_name(spec, plan=plan),
        replicas=spec.scheduling.num_cells,
        size=pods_per_cell if pods_per_cell > 1 else None,
        resources={"limits": {"nvidia.com/gpu": gpus_per_pod}} if gpus_per_pod else None,
        restart_at=plan.rendered_restart_at(spec.name),
    )


def _service_account_name(spec: BaseWorkerSpec, *, plan: LaunchPlan) -> str | None:
    # [FUNCTION DOC] HELPER: None when the spec needs no platform access, else the platform account
    # name for the release and access level.
    if (access := spec.platform_access) is PlatformAccess.NONE:
        return None
    return naming.platform_account_name(release=plan.release, access=access)


def _command_env_of_spec(
    spec: BaseWorkerSpec,
    context: LaunchCommandContext,
    *,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    is_sub_node: bool,
) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Environment of a command spec's pods. Serve specs return an empty
    # mapping. Otherwise the spec's environment is computed for the rendering context and for a
    # second context (cell 1, worker 1); asserts both are equal, proving the environment does not
    # depend on cell or worker (it is rendered once for the whole pool). Returns it with sentinel
    # values replaced by placeholders.
    if isinstance(spec, ServeWorkerSpec):
        return {}

    first = dict(spec.env_var(context))
    second = dict(
        spec.env_var(
            _launch_context(
                spec,
                addresses=addresses,
                cell_index=1,
                worker_in_cell_index=1,
                is_sub_node=is_sub_node,
            )
        )
    )
    assert first == second, (
        f"Spec '{spec.name}' builds its environment out of the cell and worker it is given, but a values entry "
        f"describes a whole pool and is rendered before any of them exists; serve the spec so its pod can "
        f"compute the environment itself, or drop the dependency"
    )
    return {name: sentinel_to_placeholder(value, spec) for name, value in first.items()}


def _with_prepare_cmd(command: list[str], spec: BaseWorkerSpec, plan: LaunchPlan) -> list[str]:
    # [FUNCTION DOC] HELPER: For trainer-section specs with a non-empty trainer prepare command in
    # the plan, returns a shell invocation that runs the prepare command, then replaces the shell
    # with the quoted original command. Asserts pods take whole nodes (gpus per pod at least gpus
    # per node) so that two pods never run the prepare command on one node concurrently. Otherwise
    # returns the command unchanged.
    if SECTION_OF_CATEGORY[spec.category] != TRAINER_ENGINES_SECTION:
        return command
    prepare = plan.prepare_cmd.get(TRAINER_ROLE)
    if not prepare:
        return command

    assert spec.scheduling.gpus_per_pod() >= spec.scheduling.num_gpus_per_node, (
        f"A prepare command runs once per pod of '{spec.name}', but that pool takes "
        f"{spec.scheduling.gpus_per_pod()} of a node's {spec.scheduling.num_gpus_per_node} gpus, so two of its "
        f"pods can land on one node and run the command against the same node-local path at the same time; "
        f"give the pool whole nodes, or serialize the command yourself with flock"
    )
    return ["bash", "-c", f"{prepare} && exec {shlex.join(command)}"]


def _meta_of_spec(spec: BaseWorkerSpec) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Empty mapping for gpu-less specs; otherwise a single entry under the
    # gpu ids meta label key holding the comma-joined ids zero to gpus per pod minus one.
    # KNOWN ISSUE: for sub-node colocated pods the real gpu ids are offset at runtime (base gpu id),
    # yet this label always lists ids starting at zero.
    gpus_per_pod = spec.scheduling.gpus_per_pod()
    if not gpus_per_pod:
        return {}
    return {env.DEFAULT_LABEL_KEYS.gpu_ids_meta: ",".join(str(gpu_id) for gpu_id in range(gpus_per_pod))}


def _launch_context(
    spec: BaseWorkerSpec,
    addresses: dict[str, dict[str, NamedHostAndPorts]],
    *,
    cell_index: int,
    worker_in_cell_index: int,
    is_sub_node: bool = False,
) -> LaunchCommandContext:
    # [FUNCTION DOC] HELPER: Builds the launch context used to render commands. Own addresses: per
    # port, host is the leader address placeholder for master-mode ports and the any-address bind
    # host otherwise, with the static port. Gpu ids (global and local) come from the real or
    # sentinel gpu id helper. Pool addresses: for each pool the list of its cells' addresses.
    self_addrs = {
        port.name: HostAndPort(
            host=LEADER_ADDRESS_PLACEHOLDER if port.mode == "master" else _BIND_HOST,
            port=port.static_port,
        )
        for port in spec.port_infos
    }
    pod_gpu_ids = real_or_sentinel_gpu_ids(spec, is_sub_node=is_sub_node)
    return LaunchCommandContext(
        cell_index=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        gpu_ids=pod_gpu_ids,
        local_gpu_ids=pod_gpu_ids,
        self_addrs=self_addrs,
        pool_addrs={pool_id: list(cells.values()) for pool_id, cells in addresses.items()},
    )


def _command_of_spec(spec: BaseWorkerSpec, context: LaunchCommandContext, plan: LaunchPlan) -> list[str]:
    # [FUNCTION DOC] HELPER: Command specs: shell-split the spec's launch command and substitute
    # sentinels with placeholders. Serve specs: the serve command. Anything else raises an
    # assertion error.
    match spec:
        case CommandWorkerSpec():
            return sentinels_to_placeholders(shlex.split(spec.launch_command(context)), spec)
        case ServeWorkerSpec():
            return _serve_command(spec, plan)
        case _:
            raise AssertionError(f"{spec.name} is neither launched by a command nor served over rpc: {spec}")


def _serve_command(spec: ServeWorkerSpec, plan: LaunchPlan) -> list[str]:
    # [FUNCTION DOC] HELPER: Argument vector running the serve module with the specs function, the
    # pool id (spec name), a separator and then the plan's worker arguments. If a pod hosts more
    # than one worker, wraps it in the process supervisor module with the number of subprocesses.
    interpreter_prefix = python_argv_prefix()
    workers_per_pod = spec.scheduling.workers_per_pod()
    serve = [
        *interpreter_prefix,
        "-m",
        _SERVE_MODULE,
        "--specs",
        _SPECS_FN,
        "--pool-id",
        spec.name,
        "--",
    ] + plan.worker_argv
    if workers_per_pod == 1:
        return serve
    return [
        *interpreter_prefix,
        "-m",
        _SUPERVISOR_MODULE,
        "--num-subprocesses",
        str(workers_per_pod),
        "--",
    ] + serve


def _is_sub_node(pairing_layout: PairingLayout | None) -> bool:
    # [FUNCTION DOC] HELPER: True when a pairing layout exists and its inference pods hold fewer
    # gpus than a node.
    if pairing_layout is None:
        return False
    return pairing_layout.num_gpus_per_inference_pod < pairing_layout.num_gpus_per_node
