# [MODULE DOC]
# Declarative description of worker pools. A spec declares a pool: its name (the pool id), network
# ports, environment variables, scheduling shape (cells, workers per cell, gpus, placement group),
# optional metadata, deployment component and platform access. Two kinds exist: command specs
# (run a shell command) and serve specs (instantiate a worker class and serve it over rpc). Also
# defines the contexts passed to the spec callbacks and the endpoint types.
from collections.abc import Callable
from typing import Any, Literal

from pydantic import ConfigDict, model_validator

from miles.utils.math_utils import exact_div
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.types import DeployComponent, PlatformAccess

RPC_PORT_NAME = "rpc"
MASTER_PORT_NAME = "master"
DEFAULT_RPC_PORT = 8000


def _port_info_name(port_info: "PortInfo | dict") -> str:
    # [FUNCTION DOC] HELPER: Name of a port declaration given either as a model or as a raw mapping.
    return port_info["name"] if isinstance(port_info, dict) else port_info.name


class PortInfo(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Declaration of one named port of a worker. Fields: name; base static port; mode (per worker:
    # each worker gets its own port, or master: only the first worker of a cell has it, shared by the
    # cell); whether the port may be allocated dynamically instead of using the static port; how many
    # consecutive ports to reserve (default one); whether the static port is shifted by the cell
    # index. Offsetting by cell together with dynamic allocation is rejected.
    name: str
    static_port: int
    mode: Literal["per_worker", "master"] = "per_worker"
    allow_dynamic: bool = False
    num_consecutive: int = 1
    offset_by_cell: bool = False

    def effective_static_port(self, *, worker_in_pod_index: int) -> int:
        # [FUNCTION DOC] The static port a worker listens on: in per-worker mode the static port
        # plus the worker's index in its pod times the number of consecutive ports; in master mode
        # just the static port. The cell offset is not applied here.
        if self.mode == "per_worker":
            return self.static_port + worker_in_pod_index * self.num_consecutive
        return self.static_port

    @model_validator(mode="after")
    def _reject_offsetting_a_dynamically_allocated_port(self) -> "PortInfo":
        # [FUNCTION DOC] HELPER: Validator run after construction; asserts the port is not both
        # offset by cell and dynamically allocated. Returns the model.
        assert not (
            self.offset_by_cell and self.allow_dynamic
        ), f"Port {self.name!r} cannot be offset by cell index: it is allocated dynamically"
        return self


class SchedulingSpec(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Shape and placement of a pool. Fields: number of cells; workers per cell; gpus per worker
    # (fractional allowed); cpus per worker (default 0.2); gpu slots per worker in the placement group
    # (default zero); gpus per node (default zero, meaning unknown); placement group name (none
    # means no placement group); the slot offset of this pool inside that group; and whether to pin
    # to the head node. Helper methods derive pod geometry for Kubernetes.
    num_cells: int
    num_workers_per_cell: int
    num_gpus_per_worker: float
    num_cpus_per_worker: float = 0.2
    num_gpu_slots_per_worker: int = 0
    num_gpus_per_node: int = 0
    pg_name: str | None = None
    pg_slot_offset: int = 0
    pin_to_head: bool = False

    def gpus_per_cell(self) -> int:
        # [FUNCTION DOC] Gpu slots used by one cell: workers per cell times gpu slots per worker.
        return self.num_workers_per_cell * self.num_gpu_slots_per_worker

    def pods_per_cell(self) -> int:
        # [FUNCTION DOC] Number of pods one cell spans: one if the cell's gpu slots fit on one node,
        # otherwise the exact quotient of cell gpu slots by gpus per node (asserts exact divisibility
        # via the exact division helper).
        # KNOWN ISSUE: with gpus per node left at its default of zero and a cell that needs any gpu
        # slots, this divides by zero (or fails the exact division check) instead of giving a clear
        # message.
        gpus_per_cell = self.gpus_per_cell()
        if gpus_per_cell <= self.num_gpus_per_node:
            return 1
        return exact_div(gpus_per_cell, self.num_gpus_per_node)

    def gpus_per_pod(self) -> int:
        # [FUNCTION DOC] Gpu slots per pod: the cell's gpu slots divided exactly by pods per cell.
        return exact_div(self.gpus_per_cell(), self.pods_per_cell())

    def workers_per_pod(self) -> int:
        # [FUNCTION DOC] Workers per pod: workers per cell divided exactly by pods per cell.
        return exact_div(self.num_workers_per_cell, self.pods_per_cell())

    @classmethod
    def single(cls, num_gpus_per_worker: float, pin_to_head: bool = False) -> "SchedulingSpec":
        # [FUNCTION DOC] Class-level constructor for the simplest shape: one cell with one worker,
        # the given gpus per worker and head pinning flag (default false); everything else default.
        return SchedulingSpec(
            num_cells=1,
            num_workers_per_cell=1,
            num_gpus_per_worker=num_gpus_per_worker,
            pin_to_head=pin_to_head,
        )


# TODO: improve meta computation logic later
class WorkerMetaContext(FrozenStrictBaseModel):
    # [CLASS DOC] Input to a spec's metadata function: the index of the cell.
    cell_index: int


class WorkerLaunchContext(FrozenStrictBaseModel):
    # [CLASS DOC] Where a worker is launched: cell index, worker index within the cell and the
    # worker's gpu ids. Input to a spec's environment variable function.
    cell_index: int
    worker_in_cell_index: int
    gpu_ids: list[int]


class WorkerCtorContext(WorkerLaunchContext):
    # [CLASS DOC] Launch context plus the backend capability, passed to a serve spec's constructor
    # keyword argument function. Frozen, unknown fields forbidden, arbitrary types allowed (for the
    # capability).
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    capability: BackendCapability


SpecMetaFn = Callable[[WorkerMetaContext], dict[str, Any]]


class BaseWorkerSpec(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Common part of a pool declaration. Fields: pool name; optional category; port declarations;
    # a function from launch context to environment variables; the scheduling shape; an optional
    # function from a meta context to a metadata mapping; the deployment component carrying the
    # pool (default primary; may not be the all selector); and the platform access level (default none).
    name: str
    category: str | None = None
    port_infos: list[PortInfo]
    env_var: Callable[[WorkerLaunchContext], dict[str, str]]
    scheduling: SchedulingSpec
    meta: SpecMetaFn | None = None
    deploy_component: DeployComponent = DeployComponent.PRIMARY
    platform_access: PlatformAccess = PlatformAccess.NONE

    @model_validator(mode="after")
    def _reject_selector_component(self) -> "BaseWorkerSpec":
        # [FUNCTION DOC] HELPER: Validator run after construction; asserts the deploy component is a
        # concrete component, not the all selector. Returns the model.
        assert (
            self.deploy_component is not DeployComponent.ALL
        ), f"pool {self.name} must name the one component it is deployed with, not the selector for all of them"
        return self


class HostAndPort(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable network endpoint: host, port and an optional external host that peers
    # outside the cluster must use (none when the host works for them too).
    host: str
    port: int
    # The host that peers outside the cluster reach this worker on; None when the placed host works for them too.
    external_host: str | None = None

    @property
    def netloc(self) -> str:
        # [FUNCTION DOC] Property: host and port joined by a colon (host used as is, so an ipv6 host
        # must already be bracketed).
        return f"{self.host}:{self.port}"

    @property
    def external_netloc(self) -> str:
        # [FUNCTION DOC] Property: external host (falling back to the host if absent or empty) and
        # port joined by a colon.
        """``host:port`` for a peer outside the cluster, falling back to the placed host."""
        return f"{self.external_host or self.host}:{self.port}"

    @property
    def addr(self):
        # [FUNCTION DOC] Property: the http base url of the endpoint (plain http scheme plus netloc).
        return f"http://{self.netloc}"


# dict key: name
NamedHostAndPorts = dict[str, HostAndPort]


class LaunchCommandContext(WorkerLaunchContext):
    # [CLASS DOC] Launch context for command specs, adding: this worker's endpoints by name
    # (including the cell's master endpoints), the endpoints of all workers of every pool (a list per
    # pool name), and the actor-local gpu ids.
    self_addrs: NamedHostAndPorts
    pool_addrs: dict[str, list[NamedHostAndPorts]]
    local_gpu_ids: list[int]


class CommandWorkerSpec(BaseWorkerSpec):
    # [CLASS DOC] Spec whose workers run a shell command produced by the launch command function
    # from the launch command context. The manager prefixes it with exec, so it must be a single
    # simple command without shell control flow.
    # A single simple command: the manager execs it, so it must not rely on shell control flow.
    launch_command: Callable[[LaunchCommandContext], str]


class ServeWorkerSpec(BaseWorkerSpec):
    # [CLASS DOC]
    # Spec whose workers are instances of a Python class given by its registered path, constructed
    # with keyword arguments from the constructor function and served over rpc or ray calls. Optional
    # concurrency groups map group names to thread counts for ray actors. An rpc port (dynamic,
    # per worker, starting at 8000) is added automatically to the port declarations if none is named
    # as the rpc port.
    worker_class: str
    ctor_kwargs: Callable[[WorkerCtorContext], dict[str, Any]]
    concurrency_groups: dict[str, int] | None = None

    @model_validator(mode="before")
    @classmethod
    def _inject_rpc_port(cls, values: dict) -> dict:
        # [FUNCTION DOC] HELPER: Validator run before construction on the raw input mapping. If port
        # declarations are present and none is named as the rpc port, return a copy with a dynamic
        # per-worker rpc port at the default static port appended; otherwise return the input as is.
        # Does nothing when no port declarations are given.
        if "port_infos" not in values:
            return values

        port_infos = list(values["port_infos"])
        if all(_port_info_name(port_info) != RPC_PORT_NAME for port_info in port_infos):
            port_infos.append(
                PortInfo(name=RPC_PORT_NAME, static_port=DEFAULT_RPC_PORT, mode="per_worker", allow_dynamic=True)
            )
        return {**values, "port_infos": port_infos}
