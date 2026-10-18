# [MODULE DOC]
# Tests that the per-cell metadata a spec declares survives the whole path to the consumer: spec,
# rendered helm values, pod annotations (fake pods), the kubernetes worker provider's observation
# of those pods, and the cell info metadata. Consumers (the inference controller and the trainer
# group) index metadata keys by literal names; the tests find those names by static analysis of
# the consumer source files and require that every one is present in the observed metadata.
# A fake pod listing api replaces the cluster, and the trainer environment is made gpu-less.
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod

from miles.ray import wiring
from miles.ray.specs import train as specs_train
from miles.ray.specs.inference import POOL_CATEGORY_INFERENCE_ENGINE
from miles.ray.specs.train import compute_trainer_pool_id, specs_trainer
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.reconcile.k8s_api import PodListPage
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.helm import env
from miles.utils.workers.worker_provider.kubernetes.helm.builder import compute_helm_backend_capability
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec

NAMESPACE = "rl"
_RELEASE = "miles-run-260101-000000-000"
ENGINE_POOL_ID = "inference-engine-0-0"
TRAINER_POOL_ID = compute_trainer_pool_id("actor")
GPUS_PER_NODE = 8

INFERENCE_CONTROLLER = Path(wiring.__file__).resolve().parent / "rollout" / "inference_controller.py"
TRAIN_GROUP = Path(wiring.__file__).resolve().parent / "train" / "group.py"

LAYOUT = LaunchPlan(
    run_id="260101-000000-000",
    state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
    release=_RELEASE,
    namespace="rl",
    orchestrator_command=["python", "train.py"],
    worker_argv=["--cluster-backend", "kubernetes"],
)


class FakePodApi:
    # [CLASS DOC] HELPER: Fake kubernetes pod api serving a fixed list of pods. The watch stream
    # never delivers an event.
    def __init__(self, pods: list[Any]) -> None:
        # [FUNCTION DOC] HELPER: Stores the pods to serve.
        self.pods = pods

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        # [FUNCTION DOC] HELPER: Returns a page holding a copy of the stored pods with resource
        # version "1". The namespace and selector arguments are ignored (no filtering).
        return PodListPage(pods=list(self.pods), resource_version="1")

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        # [FUNCTION DOC] HELPER: Fake watch stream; sleeps for an hour and then yields once with no
        # event, so effectively no event is ever delivered. All arguments are ignored.
        await asyncio.sleep(3600)
        yield None


@pytest.fixture(autouse=True)
def _fake_pod_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Resets the shared fake pod api holder and makes the
    # kubernetes provider obtain its pod api from that holder, so tests never talk to a cluster.
    fake_pod_api.reset()
    monkeypatch.setattr(core_provider, "_kubernetes_pod_api", fake_pod_api.installed)


def engine_spec(*, num_cells: int = 2, gpu_offset: int = 0) -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a hand-written inference engine command spec: the given number
    # of cells (default 2) of one worker each, taking a whole 8-gpu node as slots, in the rollout
    # placement group at the given slot offset. The per-cell metadata (model id qwen3-4b, decode
    # worker type, 8 gpus per engine, no api key, no offload, weight updates on) includes a gpu
    # offset equal to the spec offset plus the cell index times the gpus of one cell.
    scheduling = SchedulingSpec(
        num_cells=num_cells,
        num_workers_per_cell=1,
        num_gpus_per_worker=0.2,
        num_gpu_slots_per_worker=GPUS_PER_NODE,
        num_gpus_per_node=GPUS_PER_NODE,
        pg_name="rollout",
        pg_slot_offset=gpu_offset,
    )
    return CommandWorkerSpec(
        name=ENGINE_POOL_ID,
        category=POOL_CATEGORY_INFERENCE_ENGINE,
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda context: {},
        scheduling=scheduling,
        launch_command=lambda context: "python -m sglang.launch_server",
        meta=lambda context: dict(
            model_id="qwen3-4b",
            worker_type="decode",
            num_gpus_per_engine=GPUS_PER_NODE,
            gpu_offset=gpu_offset + context.cell_index * scheduling.num_workers_per_cell * GPUS_PER_NODE,
            sglang_api_key=None,
            needs_offload=False,
            update_weights=True,
        ),
    )


def trainer_args(*, num_cells: int) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a minimal trainer argument object: actor only, one node of 8
    # gpus per cell (node count equals the requested cell count), independent dp when more than
    # one cell, tensor parallel size 8 and no other parallelism, megatron backend, no offload, no
    # fault tolerance.
    return SimpleNamespace(
        use_critic=False,
        actor_num_nodes=num_cells,
        actor_num_gpus_per_node=GPUS_PER_NODE,
        indep_dp=num_cells > 1,
        tensor_model_parallel_size=GPUS_PER_NODE,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=None,
        train_backend="megatron",
        train_env_vars={},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        offload_train_target="cpu",
        megatron_config=None,
        use_fault_tolerance=False,
    )


def specs_of(*, engine_cells: int, trainer_cells: int) -> list[Any]:
    # [FUNCTION DOC] HELPER: Returns the hand-written engine spec followed by the real trainer
    # specs for the requested numbers of cells.
    return [engine_spec(num_cells=engine_cells), *specs_trainer(trainer_args(num_cells=trainer_cells))]


def observed_meta(
    *, pool_id: str, cell_id_suffix: int, engine_cells: int = 2, trainer_cells: int = 1
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Runs the full metadata path for one cell and returns the metadata the
    # provider reports. Steps: build the specs, render them to helm values, find the values entry
    # for the pool (matched by pool id or name) in the inference or trainer sections, create fake
    # pods for the cell (one per entry size, default 1) whose annotations carry the entry's
    # metadata under the meta annotation prefix, install a fake pod api serving them, create the
    # helm backend capability and a dynamic worker provider over the pool, start watching cells
    # (ignoring callbacks), read the cell info for the requested cell index, stop the watch and
    # return a copy of the info metadata. Raises an assertion error if no cell info exists, and a
    # stop iteration error if the pool has no values entry. Side effect: installs the fake pod api.
    specs = specs_of(engine_cells=engine_cells, trainer_cells=trainer_cells)
    values = build_values(specs, LAYOUT).as_values()
    entry = next(
        candidate
        for section in ("inferenceEngines", "trainerEngines")
        for candidate in values["run"][section]
        if (candidate.get("poolId") or candidate["name"]) == pool_id
    )

    pods = [
        make_pod(
            name=f"{pool_id}-{cell_id_suffix}-{worker_index}",
            pool_id=pool_id,
            cell_id_suffix=str(cell_id_suffix),
            pod_in_cell_index=str(worker_index),
            annotations={f"miles.radixark.io/meta-{key}": value for key, value in entry.get("meta", {}).items()},
        )
        for worker_index in range(entry.get("size", 1))
    ]
    fake_pod_api.install(FakePodApi(pods))
    factory = compute_helm_backend_capability(specs=specs)
    provider = factory.dynamic_worker_provider(pool_ids=[pool_id])

    async def scenario() -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Starts the cell watch, reads the cell info for the requested cell,
        # asserts it exists, returns a copy of its metadata, and always stops the watch afterwards.
        stop = await provider.watch_cells(_ignore_cell)
        try:
            info = provider.cell_info(compute_cell_id(pool_id=pool_id, cell_index=cell_id_suffix))
            assert info is not None
            return dict(info.meta)
        finally:
            await stop()

    return asyncio.run(scenario())


def indexed_meta_keys(source: Path, *, attribute: str) -> set[str]:
    # [FUNCTION DOC] HELPER: Parses the given python source file and returns the set of string
    # literal keys used in subscript expressions on an attribute with the given name (for example
    # keys read from a meta attribute). Only subscripts whose key is a literal string count.
    tree = ast.parse(source.read_text())
    return {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == attribute
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }


async def _ignore_cell(cell_id: str, info: Any) -> None:
    # [FUNCTION DOC] HELPER: Cell reconcile callback that does nothing.
    return None


@pytest.fixture(autouse=True)
def trainer_env_without_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Replaces the fp8 scaling default probe with one
    # returning "0" (no gpu needed) and sets the namespace and release environment variables the
    # helm provider reads. Undone at test end.
    monkeypatch.setattr(specs_train, "default_fp8_block_scaling_fp32_scales", lambda: "0")
    monkeypatch.setenv(env.NAMESPACE_ENV_VAR, NAMESPACE)
    monkeypatch.setenv(env.RELEASE_ENV_VAR, _RELEASE)


class TestWhatAnEngineCellReports:
    # [CLASS DOC] Tests the metadata reported for engine cells after the full round trip.
    def test_carries_every_fact_the_inference_controller_indexes(self):
        # [FUNCTION DOC] Every literal meta key that the inference controller source indexes is
        # present in the observed metadata of engine cell 0.
        """The first reconcile builds ServerCellMetadata out of these keys, so a missing one is a KeyError."""
        meta = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=0)

        assert indexed_meta_keys(INFERENCE_CONTROLLER, attribute="meta") <= set(meta)

    def test_gives_each_cell_its_own_gpu_offset(self):
        # [FUNCTION DOC] Engine cells 0 and 1 report gpu offsets 0 and 8 (the gpus of one node).
        """Two cells of one pool_id share a values entry, and a shared offset would update one engine twice."""
        first = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=0)
        second = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=1)

        assert (first["gpu_offset"], second["gpu_offset"]) == (0, GPUS_PER_NODE)

    def test_keeps_the_types_the_consumers_expect(self):
        # [FUNCTION DOC] For engine cell 1, the gpu offset is an integer, the offload flag is a
        # boolean and the model id is qwen3-4b, i.e. the annotation round trip restores types.
        """ServerCellMetadata declares ints and bools, and a stringly typed offset would index the wrong gpus."""
        meta = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=1)

        assert isinstance(meta["gpu_offset"], int)
        assert isinstance(meta["needs_offload"], bool)
        assert meta["model_id"] == "qwen3-4b"

    def test_still_reports_the_gpus_the_values_file_annotated(self):
        # [FUNCTION DOC] The gpu ids metadata of engine cell 0 is the comma joined list of gpu ids
        # 0 to 7, as annotated in the rendered values.
        """gpu_ids is what the pod itself was given, so it travels on the pod rather than being recomputed."""
        meta = observed_meta(pool_id=ENGINE_POOL_ID, cell_id_suffix=0)

        assert meta[DEFAULT_LABEL_KEYS.gpu_ids_meta] == ",".join(str(gpu_id) for gpu_id in range(GPUS_PER_NODE))


class TestWhatATrainerCellReports:
    # [CLASS DOC] Tests the metadata reported for trainer cells after the full round trip.
    def test_carries_every_fact_the_trainer_group_indexes(self):
        # [FUNCTION DOC] Every literal meta key that the trainer group source indexes is present in
        # the observed metadata of trainer cell 0.
        """The group turns an observation into a TrainerCell and reads its index straight out of the meta."""
        meta = observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=0)

        assert indexed_meta_keys(TRAIN_GROUP, attribute="meta") <= set(meta)

    def test_carries_the_same_facts_for_every_cell_of_the_pool(self):
        # [FUNCTION DOC] With two trainer cells, cells 0 and 1 report cell indices 0 and 1, and all
        # other metadata entries are identical between the two cells.
        """Cells of one indep-dp pool_id share every fact but the index that tells them apart."""
        first = observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=0, trainer_cells=2)
        second = observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=1, trainer_cells=2)

        assert (first["cell_index"], second["cell_index"]) == (0, 1)
        assert {k: v for k, v in first.items() if k != "cell_index"} == {
            k: v for k, v in second.items() if k != "cell_index"
        }

    def test_reports_the_role_the_spec_was_built_for(self):
        # [FUNCTION DOC] The actor pool's cell 0 metadata has role actor.
        """A critic pool_id observes the same way, and the role is how a consumer tells the two apart."""
        assert observed_meta(pool_id=TRAINER_POOL_ID, cell_id_suffix=0)["role"] == "actor"
