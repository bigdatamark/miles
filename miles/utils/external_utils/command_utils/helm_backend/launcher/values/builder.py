# [MODULE DOC]
# Turns a list of worker specs plus a launch plan into the typed run values of the miles run helm
# chart. Specs are bucketed by category into three sections (static workers, inference engines,
# trainer engines); each becomes a pool entry. Also computes static cell addresses, the object names
# of chart-owned objects, and the optional colocate pairing configuration.
from __future__ import annotations

from miles.utils.external_utils.colocate_pairing.config import PairingConfig
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.colocate import pairing_config
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    AutoUninstallSection,
    MilesRunChartValues,
    ObjectNames,
    OrchestratorSection,
    PoolEntry,
    RunValues,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import (
    INFERENCE_ENGINES_SECTION,
    SECTION_OF_CATEGORY,
    STATIC_WORKERS_SECTION,
    TRAINER_ENGINES_SECTION,
    LaunchPlan,
    MooncakeInfo,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.pool_entry import build_entry
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.types import PlatformAccess
from miles.utils.workers.worker_provider.kubernetes.helm.naming import static_cell_addrs
from miles.utils.workers.worker_spec import RPC_PORT_NAME, BaseWorkerSpec, NamedHostAndPorts, ServeWorkerSpec

_COLOCATE_PAIRING_COMPONENT = "colocate-pairing"


def build_values(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> MilesRunChartValues:
    # [FUNCTION DOC] Public entry point. Builds the run section from the specs and plan and wraps it
    # in the chart values, attaching the plan's extra manifests (or none when that list is empty).
    # Raises assertion errors for inconsistent input (see the run-values builder).
    return MilesRunChartValues(run=_build_run_values(specs, plan), extra_manifests=plan.extra_manifests or None)


def _build_run_values(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> RunValues:
    # [FUNCTION DOC] HELPER: Builds the run values. Steps: (1) assert the orchestrator command and
    # state file are either both set or both unset; (2) drop specs with zero cells; (3) for serve
    # specs assert the worker ports fit; (4) compute static cell addresses; (5) compute the colocate
    # pairing config (only when colocating with inference pools present) and a mapping from pool
    # id to pairing layout; (6) for each spec build a pool entry placed in the section of its
    # category, asserting that restart stamps and service account names appear only in the static
    # workers section (other templates would ignore them); (7) assemble run values: run id, state
    # file and launch record (state file none when empty), object names, orchestrator section with
    # its rendered restart stamp, auto-uninstall disabled only for releases without an orchestrator
    # command, the three entry lists, environment copy (none when empty), mooncake section when a
    # mooncake plan exists, and colocate config. Raises assertion errors on violated invariants.
    assert bool(plan.orchestrator_command) == bool(plan.state_file), (
        "The orchestration script and the exit file it writes come together: a release carries both, or it carries "
        "neither and is a deployment of workers that outlives any one run"
    )
    specs = _deployed_specs(specs)
    for spec in specs:
        if isinstance(spec, ServeWorkerSpec):
            _assert_worker_ports_fit(spec)
    addresses = _compute_addresses(specs, plan.release)

    colocate = _pairing_config(specs, plan)
    layout_of_pool = {pool.pool_id: pool.layout for pool in colocate.inference_pools} if colocate else {}

    entries: dict[str, list[PoolEntry]] = {
        STATIC_WORKERS_SECTION: [],
        INFERENCE_ENGINES_SECTION: [],
        TRAINER_ENGINES_SECTION: [],
    }
    for spec in specs:
        section = SECTION_OF_CATEGORY[spec.category]
        entry = build_entry(spec, plan=plan, addresses=addresses, pairing_layout=layout_of_pool.get(spec.name))
        assert section == STATIC_WORKERS_SECTION or entry.restart_at is None, (
            f"only the {STATIC_WORKERS_SECTION} template renders a restart stamp, so stamping {spec.name} in "
            f"{section} would roll nothing while this launch believes it rolled a pod"
        )
        assert section == STATIC_WORKERS_SECTION or entry.service_account_name is None, (
            f"only the {STATIC_WORKERS_SECTION} template renders a service account, so naming one for "
            f"{spec.name} in {section} would leave its pods on the namespace default; they would run until "
            f"the first api call and fail with a 403 nothing connects back to this spec"
        )
        entries[section].append(entry)

    return RunValues(
        id=plan.run_id,
        state_file=plan.state_file or None,
        launch_record=plan.launch_record,
        object_names=_object_names(plan.release),
        orchestrator=OrchestratorSection(
            command=plan.orchestrator_command,
            restart_at=plan.rendered_restart_at(naming.ORCHESTRATOR_COMPONENT),
        ),
        auto_uninstall=None if plan.orchestrator_command else AutoUninstallSection(enabled=False),
        static_workers=entries[STATIC_WORKERS_SECTION],
        inference_engines=entries[INFERENCE_ENGINES_SECTION],
        trainer_engines=entries[TRAINER_ENGINES_SECTION],
        env=dict(plan.env) or None,
        mooncake=MooncakeInfo.section(plan.mooncake_plan) if plan.mooncake_plan is not None else None,
        colocate=colocate,
    )


def _pairing_config(specs: list[BaseWorkerSpec], plan: LaunchPlan) -> PairingConfig | None:
    # [FUNCTION DOC] HELPER: Returns the colocate pairing config, or none when the plan is not
    # colocating or no spec belongs to the inference engines section; else delegates to the colocate
    # module.
    if not plan.colocate or not any(SECTION_OF_CATEGORY[spec.category] == INFERENCE_ENGINES_SECTION for spec in specs):
        return None
    return pairing_config(specs, plan)


def _object_names(release: str) -> ObjectNames:
    # [FUNCTION DOC] HELPER: Computes the names of chart-owned objects for the release: orchestrator,
    # read and read-delete platform accounts, mooncake master, colocate pairing controller, and the
    # uninstall job and its manifest.
    return ObjectNames(
        orchestrator=naming.component_name(release, naming.ORCHESTRATOR_COMPONENT),
        platform_read=naming.platform_account_name(release=release, access=PlatformAccess.READ),
        platform_read_delete=naming.platform_account_name(release=release, access=PlatformAccess.READ_DELETE),
        mooncake_master=MooncakeInfo.master_object_name(release),
        colocate_pairing=naming.component_name(release, _COLOCATE_PAIRING_COMPONENT),
        uninstall=RunNames.uninstall_job(release=release),
        uninstall_manifest=RunNames.uninstall_manifest(release=release),
    )


def _deployed_specs(specs: list[BaseWorkerSpec]) -> list[BaseWorkerSpec]:
    # [FUNCTION DOC] HELPER: Keeps only specs that have at least one cell, preserving order.
    return [spec for spec in specs if spec.scheduling.num_cells > 0]


def _compute_addresses(specs: list[BaseWorkerSpec], release: str) -> dict[str, dict[str, NamedHostAndPorts]]:
    # [FUNCTION DOC] HELPER: For static-workers-section specs only, returns a nested mapping from
    # pool name to a mapping from cell id (derived from pool id and cell index) to that cell's
    # static host and named ports, for every cell index below the spec's cell count.
    return {
        spec.name: {
            compute_cell_id(pool_id=spec.name, cell_index=cell_index): static_cell_addrs(
                spec=spec, release=release, cell_index=cell_index
            )
            for cell_index in range(spec.scheduling.num_cells)
        }
        for spec in specs
        if SECTION_OF_CATEGORY[spec.category] == STATIC_WORKERS_SECTION
    }


def _assert_worker_ports_fit(spec: ServeWorkerSpec) -> None:
    # [FUNCTION DOC] HELPER: Asserts that the consecutive rpc ports used by the workers of one pod
    # (starting at the rpc static port, one per worker) do not overlap any other named port range
    # (static port through static port plus its consecutive count). Raises an assertion error naming
    # the spec and the colliding port otherwise. Raises a stop-iteration error if the spec has no
    # rpc-named port.
    workers_per_pod = spec.scheduling.workers_per_pod()
    rpc_port = next(port.static_port for port in spec.port_infos if port.name == RPC_PORT_NAME)
    for port in spec.port_infos:
        if port.name == RPC_PORT_NAME:
            continue
        assert rpc_port + workers_per_pod <= port.static_port or port.static_port + port.num_consecutive <= rpc_port, (
            f"Spec '{spec.name}' serves {workers_per_pod} workers per pod from {RPC_PORT_NAME} port {rpc_port} "
            f"upwards, which reaches into the {port.num_consecutive} port(s) '{port.name}' claims from "
            f"{port.static_port}"
        )
