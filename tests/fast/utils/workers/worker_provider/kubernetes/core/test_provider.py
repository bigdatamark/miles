# [MODULE DOC] Tests for the kubernetes worker provider against a fake pod api (list returns the
# configured pods; the watch stream yields configured events then blocks). Covered: reading before
# the watch starts fails with a lifecycle assertion; address lookup from observed pods; the watch
# reports pre-existing cells (None for cells not alive), hides other pools' cells, finishes the
# initial listing before returning, ignores non-worker pods, closes the api client when stopped,
# pushes the same info a direct query returns, and reports a cell leaving the wanted set as gone;
# the pod label selector scopes to the wanted pools; cell info (rank ordering, metadata, per-rank
# worker names for multi-worker pods, pod names, absence); spec-computed metadata (per cell, python
# types kept, pod annotations win, disagreement refused); worker infos (rank order, pod-ip or
# headless-service host, rpc handles with the worker class methods, gpus, restart generation,
# missing pods, per-rank port offset and gpu shares, several cells at once). Most tests run their
# own event loop with asyncio run; the autouse fixture resets the fake api and patches the
# provider's api factory.
import asyncio
from dataclasses import dataclass, field

import pytest
from tests.fast.utils.workers.worker_provider.kubernetes import fake_pod_api
from tests.fast.utils.workers.worker_provider.kubernetes.core.test_pod_view import make_pod, make_unlabelled_pod
from tests.fast.utils.workers.worker_provider.kubernetes.run_specs import make_pool_spec

from miles.utils.workers.reconcile.k8s_api import PodListPage, PodWatchEvent
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider.kubernetes.core import provider as core_provider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo, KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS
from miles.utils.workers.worker_provider.utils import build_rpc_handle_of_worker_info
from miles.utils.workers.worker_spec import HostAndPort

NAMESPACE = "rl"
SELECTOR = "app.kubernetes.io/instance=r"


@dataclass
class FakePodApi:
    # [CLASS DOC] HELPER: Fake kubernetes pod api. Fields: pods returned by listing, events yielded
    # by each stream, the resource version of listings, a delay before events are yielded (seconds)
    # and a record of every label selector used.
    pods: list = field(default_factory=list)
    events: list = field(default_factory=list)
    resource_version: str = "1"
    event_delay: float = 0.0

    selectors: list = field(default_factory=list)

    async def list_pods(self, *, namespace, label_selector):
        # [FUNCTION DOC] HELPER: Records the selector and returns a page holding a copy of the
        # configured pods and the configured resource version; the namespace is ignored.
        self.selectors.append(label_selector)
        return PodListPage(pods=list(self.pods), resource_version=self.resource_version)

    async def stream_pods(self, *, namespace, label_selector, resource_version, timeout_seconds):
        # [FUNCTION DOC] HELPER: Async generator. Records the selector, sleeps the configured delay,
        # yields each configured event, then sleeps an hour (the stream never ends on its own).
        # Namespace, resource version and timeout are ignored.
        self.selectors.append(label_selector)
        await asyncio.sleep(self.event_delay)
        for event in self.events:
            yield event
        await asyncio.sleep(3600)


@pytest.fixture(autouse=True)
def _fake_pod_api(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture; resets the shared fake api registry and replaces the
    # provider module's pod api factory with the registry's installed context manager.
    fake_pod_api.reset()
    monkeypatch.setattr(core_provider, "_kubernetes_pod_api", fake_pod_api.installed)


def _run(api, *, ports, worker_classes=None, spec_metas=None, workers_per_pod=None) -> KubernetesRunInfo:
    # [FUNCTION DOC] HELPER: Installs the api in the fake registry and builds run info for the rl
    # namespace and test selector with default label keys. The ports argument maps pool id to a
    # port mapping and defines the pools; optional mappings give each pool's worker class, metadata
    # callable and workers per pod (default 1).
    fake_pod_api.install(api)
    worker_classes = worker_classes or {}
    spec_metas = spec_metas or {}
    workers_per_pod = workers_per_pod or {}
    return KubernetesRunInfo(
        namespace=NAMESPACE,
        label_selector=SELECTOR,
        label_keys=DEFAULT_LABEL_KEYS,
        specs={
            pool_id: make_pool_spec(
                pool_id,
                ports=spec_ports,
                worker_class=worker_classes.get(pool_id),
                meta=spec_metas.get(pool_id),
                workers_per_pod=workers_per_pod.get(pool_id, 1),
            )
            for pool_id, spec_ports in ports.items()
        },
    )


def _provider(api, worker_ports=None, pool_ids=("engine",), **kwargs):
    # [FUNCTION DOC] HELPER: Builds a kubernetes provider over a run (ports default to a single
    # engine pool with primary 8000; extra keyword arguments go to the run builder) watching the
    # given pool ids (default engine) with periodic resync disabled.
    return KubernetesWorkerProvider(
        run=_run(api, ports=worker_ports or {"engine": {"primary": 8000}}, **kwargs),
        pool_ids=list(pool_ids),
        resync_period=None,
    )


def _cell_info(provider, cell_id="engine-00000"):
    # [FUNCTION DOC] HELPER: Runs its own event loop: starts the watch, reads the cell info of the
    # given cell id (default engine-00000), stops the watch and returns the info (or none).
    async def scenario():
        # [FUNCTION DOC] HELPER: Starts the watch, returns the cell info, always stops the watch.
        stop = await _watch(provider, [])
        try:
            return provider.cell_info(cell_id)
        finally:
            await stop()

    return asyncio.run(scenario())


async def _watch(provider, reported):
    # [FUNCTION DOC] HELPER: Starts the provider's cell watch with a callback that appends each
    # (cell id, info) report to the given list; returns the stop function.
    async def reconcile(cell_id, info):
        # [FUNCTION DOC] HELPER: Appends the report to the list.
        reported.append((cell_id, info))

    return await provider.watch_cells(reconcile)


def _relabelled(pool_id):
    # [FUNCTION DOC] HELPER: Builds a watch event of type modified for pod engine-0-0 relabelled
    # with the given pool id, resource version 2, not rejecting the cursor.
    pod = make_pod(name="engine-0-0", labels={DEFAULT_LABEL_KEYS.pool_id: pool_id})
    return PodWatchEvent(type="MODIFIED", pod=pod, resource_version="2", rejects_cursor=False)


async def _run_watch(provider, reported):
    # [FUNCTION DOC] HELPER: Starts the watch recording reports into the list, lets it run for 0.1
    # seconds and stops it.
    async def reconcile(cell_id, info):
        # [FUNCTION DOC] HELPER: Appends the report to the list.
        reported.append((cell_id, info))

    stop = await provider.watch_cells(reconcile)
    await asyncio.sleep(0.1)
    await stop()


class TestObservationLifecycle:
    # [CLASS DOC] Tests of the requirement to start the watch before reading.
    @pytest.mark.parametrize(
        "reader",
        ["cell_ids", "cell_info", "pod_names_of_cell", "get_worker_infos", "get_addrs"],
    )
    async def test_reading_cells_before_watch_cells_starts_explains_the_required_lifecycle(self, reader: str) -> None:
        # [FUNCTION DOC] Parametrized over five readers (cell ids, cell info, pod names of a cell,
        # worker infos, addresses); each, called on a provider whose watch never started, raises
        # an assertion error saying the watch must be running before the cells can be read.
        """Observation-backed readers explain that watch_cells must be running before they can be used."""
        provider = _provider(FakePodApi())

        with pytest.raises(AssertionError, match="watch_cells must be running before the cells can be read"):
            if reader == "cell_ids":
                provider.cell_ids()
            elif reader == "cell_info":
                provider.cell_info("engine-00000")
            elif reader == "pod_names_of_cell":
                provider.pod_names_of_cell("engine-00000")
            elif reader == "get_worker_infos":
                provider.get_worker_infos(cell_ids=["engine-00000"])
            else:
                await provider.get_addrs("engine-00000-00000")


class TestGetAddrs:
    # [CLASS DOC] Tests of address lookup from observed pods.
    def test_answers_a_cell_member_from_its_pod(self):
        # [FUNCTION DOC] One pod with ip 10.1.2.3; after the watch starts, the primary address of
        # worker engine-00000-00000 is host 10.1.2.3, port 8000.
        """A cell's members only exist once scheduled, so their addresses come from observation."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])
        provider = _provider(api)

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the primary address of the worker,
            # always stops the watch.
            stop = await _watch(provider, [])
            try:
                return (await provider.get_addrs("engine-00000-00000"))["primary"]
            finally:
                await stop()

        assert asyncio.run(scenario()) == HostAndPort(host="10.1.2.3", port=8000)

    def test_refuses_a_worker_it_has_never_seen(self):
        # [FUNCTION DOC] With no pods, asking for worker engine-00009-00009 raises an assertion
        # error saying no observed pod serves it.
        """Returning a guess would send traffic to whatever happens to answer at that address."""
        provider = _provider(FakePodApi())

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, asks for the unknown worker's addresses,
            # always stops the watch.
            stop = await _watch(provider, [])
            try:
                await provider.get_addrs("engine-00009-00009")
            finally:
                await stop()

        with pytest.raises(AssertionError, match="no observed pod serves"):
            asyncio.run(scenario())


class TestWatchCells:
    # [CLASS DOC] Tests of the cell watch: initial reporting, filtering, lifecycle and client
    # cleanup.
    def test_reports_the_cells_that_already_existed(self):
        # [FUNCTION DOC] Two pods in cells 0 and 1; after 50 ms the sorted reported cell ids are
        # engine-00000 and engine-00001.
        """A controller starting against a running release must not think the cluster is empty."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-1-0", cell_id_suffix="1")])
        provider = _provider(api)
        reported = []

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, waits 50 ms, stops the watch.
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert sorted(cell_id for cell_id, _ in reported) == ["engine-00000", "engine-00001"]

    def test_finishes_the_initial_listing_before_returning(self):
        # [FUNCTION DOC] Immediately after the watch call returns (no waiting), the provider's cell
        # ids are exactly engine-00000.
        """Otherwise a caller reading the cells right after cannot tell "not listed yet" from "not there"."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api)

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the cell ids at once, always stops
            # the watch.
            stop = await _watch(provider, [])
            try:
                return provider.cell_ids()
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-00000"]

    def test_hides_a_cell_of_another_controller(self):
        # [FUNCTION DOC] Pods of pools engine and trainer, provider over engine only; the reports
        # with info present name only engine-00000.
        """Several controllers share a namespace, and each must see only the specs it was given."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="trainer-0-0", pool_id="trainer")])
        provider = _provider(api)
        reported = []

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, waits 50 ms, stops the watch.
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert [cell_id for cell_id, info in reported if info is not None] == ["engine-00000"]

    def test_reports_a_cell_that_is_not_alive_yet_as_gone(self):
        """A cell that is still starting has nothing a consumer can drive, which is what None says."""
        # [FUNCTION DOC] A cell with one ready and one unready pod: the only report is the cell id
        # with no info (reported gone).
        api = FakePodApi(
            pods=[make_pod(name="engine-0-0"), make_pod(name="engine-0-1", pod_in_cell_index="1", ready=False)]
        )
        provider = _provider(api)
        reported = []

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, waits 50 ms, stops the watch.
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert reported == [("engine-00000", None)]

    def test_says_nothing_about_a_cell_of_another_pool(self):
        # [FUNCTION DOC] Only a trainer pod exists and the provider watches engine; nothing is
        # reported at all.
        """A pod outside this view's pools is keyed to no cell, so there is nothing to report about it."""
        api = FakePodApi(pods=[make_pod(name="trainer-0-0", pool_id="trainer")])
        provider = _provider(api)
        reported = []

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, waits 50 ms, stops the watch.
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            await stop()

        asyncio.run(scenario())

        assert reported == []

    def test_the_api_client_is_closed_when_the_watch_stops(self):
        # [FUNCTION DOC] While the watch runs no close has been recorded; after stopping, exactly
        # this api is in the registry's close list.
        """Nothing else owns the session, so a watch that stops without closing it leaks one per call."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api)

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, waits 50 ms, asserts no close happened yet,
            # stops the watch.
            stop = await _watch(provider, [])
            await asyncio.sleep(0.05)
            assert fake_pod_api.CLOSE_CALLS == []
            await stop()

        asyncio.run(scenario())

        assert fake_pod_api.CLOSE_CALLS == [api]

    def test_ignores_a_pod_that_is_not_a_worker(self):
        # [FUNCTION DOC] A worker pod plus an unlabelled pod; the cell ids are only engine-00000.
        """A namespace holds other pods, and one of them must not become a phantom cell."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_unlabelled_pod("prometheus-0")])
        provider = _provider(api)

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the cell ids, always stops the watch.
            stop = await _watch(provider, [])
            try:
                return provider.cell_ids()
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-00000"]

    def test_pushes_exactly_what_a_direct_query_would_answer(self):
        # [FUNCTION DOC] Two pods of one cell sharing a model id annotation; the single pushed
        # report's info equals the info obtained by a direct cell info query.
        """One watch and one set of pure functions: a pushed cell and a pulled one cannot be allowed to disagree."""
        api = FakePodApi(
            pods=[
                make_pod(
                    name="engine-0-0", pod_in_cell_index="0", annotations={"miles.radixark.io/meta-model_id": "glm"}
                ),
                make_pod(
                    name="engine-0-1", pod_in_cell_index="1", annotations={"miles.radixark.io/meta-model_id": "glm"}
                ),
            ]
        )
        provider = _provider(api)
        reported = []

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, waits 50 ms, returns the directly queried
            # cell info, always stops the watch.
            stop = await _watch(provider, reported)
            await asyncio.sleep(0.05)
            try:
                return provider.cell_info("engine-00000")
            finally:
                await stop()

        queried = asyncio.run(scenario())

        assert [info for _, info in reported] == [queried]


class TestWatchedPods:
    # [CLASS DOC] Tests of the selector the provider sends to the api.
    def test_asks_the_apiserver_only_for_the_pools_it_was_built_for(self):
        # [FUNCTION DOC] Run with engine and router pools, provider over engine; every recorded
        # selector is the base plus "<pool label> in (engine)".
        """A run holds every pool_id's pods, and streaming all of them into every component would not scale."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, worker_ports={"engine": {"primary": 8000}, "router": {"primary": 9000}})

        asyncio.run(_run_watch(provider, []))

        assert all(
            selector == f"{SELECTOR},{DEFAULT_LABEL_KEYS.pool_id} in (engine)" for selector in api.selectors
        ), api.selectors

    def test_names_every_pool_it_watches_in_one_selector(self):
        # [FUNCTION DOC] Provider over router and engine; the first selector lists both, sorted, in
        # one "in" expression.
        """One component may watch several specs, and each extra watch is another stream to the apiserver."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(
            api,
            worker_ports={"engine": {"primary": 8000}, "router": {"primary": 9000}},
            pool_ids=("router", "engine"),
        )

        asyncio.run(_run_watch(provider, []))

        assert api.selectors[0] == f"{SELECTOR},{DEFAULT_LABEL_KEYS.pool_id} in (engine,router)"

    def test_a_provider_over_no_pool_asks_for_no_pod_at_all(self):
        # [FUNCTION DOC] A provider over no pool never uses the bare base selector.
        """A train-only run still builds the controller, and it must not end up watching the whole release."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, pool_ids=())

        asyncio.run(_run_watch(provider, []))

        assert all(selector != SELECTOR for selector in api.selectors), api.selectors


class TestWatchCellsStateCommit:
    # [CLASS DOC] Tests of reports triggered by watch events.
    def test_a_reported_cell_leaving_the_wanted_set_is_reported_as_gone(self):
        # [FUNCTION DOC] A pod listed as engine, then an event relabelling it to another pool after
        # 20 ms; reports are two for engine-00000, first with info, then without (gone).
        """From this view the cell is gone, and only None says so; dropping it silently strands the consumer."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")], events=[_relabelled("someone-else")], event_delay=0.02)
        provider = _provider(api)
        reported = []

        asyncio.run(_run_watch(provider, reported))

        assert [info is None for _, info in reported] == [False, True]
        assert [cell_id for cell_id, _ in reported] == ["engine-00000", "engine-00000"]


class TestCellInfo:
    # [CLASS DOC] Tests of the cell info the provider answers for observed cells.
    def test_orders_the_workers_by_rank(self):
        # [FUNCTION DOC] Pods listed index 1 then 0; the info's worker names are rank 0 then rank 1.
        """Consumers index this list by rank, so an arbitrary order would scramble the mapping."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-1", pod_in_cell_index="1"),
                make_pod(name="engine-0-0", pod_in_cell_index="0"),
            ]
        )
        provider = _provider(api)

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the cell info, always stops the watch.
            stop = await _watch(provider, [])
            try:
                return provider.cell_info("engine-00000")
            finally:
                await stop()

        assert asyncio.run(scenario()).worker_names == ["engine-00000-00000", "engine-00000-00001"]

    def test_carries_the_meta_a_platform_annotated(self):
        # [FUNCTION DOC] A pod annotated with a meta-prefixed model id glm; the info's meta is
        # exactly the model id mapping.
        """An engine's model id is a domain fact its consumers need and the pod is where it travels."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "glm"})])
        provider = _provider(api)

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the cell info, always stops the watch.
            stop = await _watch(provider, [])
            try:
                return provider.cell_info("engine-00000")
            finally:
                await stop()

        assert asyncio.run(scenario()).meta == {"model_id": "glm"}

    def test_names_every_rank_of_a_pod_that_serves_more_than_one(self):
        # [FUNCTION DOC] Two pods of two workers each; the worker names are the four consecutive
        # ranks 0 to 3 of the cell.
        """Consumers index this list by rank, so a list of pods would hide every rank but the first of each."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-0-1", pod_in_cell_index="1")])
        provider = _provider(api, workers_per_pod={"engine": 2})

        assert _cell_info(provider).worker_names == [
            "engine-00000-00000",
            "engine-00000-00001",
            "engine-00000-00002",
            "engine-00000-00003",
        ]

    def test_still_reports_the_pods_themselves_for_the_operations_that_delete_them(self):
        # [FUNCTION DOC] Two pods listed out of order with two workers per pod; the pod names of
        # the cell are engine-0-0 then engine-0-1.
        """Healing recreates pods, and a rank name is not something kubernetes could delete."""
        api = FakePodApi(pods=[make_pod(name="engine-0-1", pod_in_cell_index="1"), make_pod(name="engine-0-0")])
        provider = _provider(api, workers_per_pod={"engine": 2})

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the pod names of the cell, always
            # stops the watch.
            stop = await _watch(provider, [])
            try:
                return provider.pod_names_of_cell("engine-00000")
            finally:
                await stop()

        assert asyncio.run(scenario()) == ["engine-0-0", "engine-0-1"]

    def test_is_absent_for_a_cell_with_no_pods(self):
        # [FUNCTION DOC] With no pods, the cell info of engine-00007 is none.
        """A cell that was deleted must read as gone rather than as an empty cell."""
        provider = _provider(FakePodApi())

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the cell info, always stops the watch.
            stop = await _watch(provider, [])
            try:
                return provider.cell_info("engine-00007")
            finally:
                await stop()

        assert asyncio.run(scenario()) is None


def _spec_meta(context) -> dict:
    # [FUNCTION DOC] HELPER: Spec metadata callable returning role actor, the context's cell index,
    # needs_offload false (a boolean) and model_id glm.
    return {"role": "actor", "cell_index": context.cell_index, "needs_offload": False, "model_id": "glm"}


class TestSpecMeta:
    # [CLASS DOC] Tests of cell metadata computed from the spec and merged with pod annotations.
    def test_evaluates_the_meta_of_the_spec_for_every_cell_that_was_observed(self):
        # [FUNCTION DOC] Two cells; each cell's info carries the spec metadata computed for its own
        # cell index (0 and 1) and role actor.
        """Two cells of one pool are two different cells, so one evaluation would collapse them into one."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0"), make_pod(name="engine-1-0", cell_id_suffix="1")])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        first = _cell_info(provider, cell_id="engine-00000")
        second = _cell_info(provider, cell_id="engine-00001")

        assert (first.meta["cell_index"], second.meta["cell_index"]) == (0, 1)
        assert (first.meta["role"], second.meta["role"]) == ("actor", "actor")

    def test_keeps_the_python_types_the_spec_computed(self):
        # [FUNCTION DOC] The boolean needs_offload value stays a boolean false in the cell meta.
        """A chart can only carry strings, which is why this meta is computed here rather than rendered."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        meta = _cell_info(provider).meta

        assert meta["needs_offload"] is False

    def test_reports_nothing_of_its_own_for_a_spec_that_declares_no_meta(self):
        # [FUNCTION DOC] Without a spec metadata callable and annotations, the meta is empty.
        """Most specs have no facts to add, and an invented key would look like a fact to a consumer."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])

        assert _cell_info(_provider(api)).meta == {}

    def test_lets_a_pod_annotation_override_a_key_the_spec_also_computed(self):
        # [FUNCTION DOC] The spec computes model_id glm and the pod annotates qwen; the cell meta
        # model id is qwen.
        """The pod is what a platform actually created, so its own account of itself wins."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "qwen"})])
        provider = _provider(api, spec_metas={"engine": _spec_meta})

        assert _cell_info(provider).meta["model_id"] == "qwen"

    def test_refuses_a_cell_whose_pods_annotate_one_key_differently(self):
        # [FUNCTION DOC] Two pods annotate model_id glm and qwen; the cell info computation raises an
        # assertion error showing the key with the tuple of both values.
        """A cell reports one value per key, and picking one silently would depend on the store's order."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-model_id": "glm"}),
                make_pod(
                    name="engine-0-1", pod_in_cell_index="1", annotations={"miles.radixark.io/meta-model_id": "qwen"}
                ),
            ]
        )

        with pytest.raises(AssertionError, match="'model_id': \\('glm', 'qwen'\\)"):
            _cell_info(_provider(api))

    def test_accepts_pods_that_agree_about_a_key(self):
        # [FUNCTION DOC] Both pods annotate model_id glm; the cell meta is that single entry.
        """Every pod of a pool_id carries the same values entry, so agreement is the normal case."""
        annotations = {"miles.radixark.io/meta-model_id": "glm"}
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", annotations=annotations),
                make_pod(name="engine-0-1", pod_in_cell_index="1", annotations=annotations),
            ]
        )

        assert _cell_info(_provider(api)).meta == {"model_id": "glm"}


class _FakeTrainWorker:
    # [CLASS DOC] HELPER: Placeholder served worker class with two methods, used to check that rpc
    # handles know the worker class's methods.
    def init(self, rank: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the rank unchanged.
        return rank

    def kill_self(self) -> None:
        # [FUNCTION DOC] HELPER: Does nothing and returns none.
        return None


def _trainer_provider(api, pool_ids=("engine",), **kwargs):
    # [FUNCTION DOC] HELPER: Builds a provider over an engine pool with master port 9000 and rpc
    # port 8000 served by the fake train worker class (extra keyword arguments go to the run
    # builder), watching the given pool ids (default engine), no periodic resync.
    return KubernetesWorkerProvider(
        run=_run(
            api,
            ports={"engine": {"master": 9000, "rpc": 8000}},
            worker_classes={"engine": f"{__name__}._FakeTrainWorker"},
            **kwargs,
        ),
        pool_ids=list(pool_ids),
        resync_period=None,
    )


def _worker_infos(provider, cell_id="engine-00000"):
    # [FUNCTION DOC] HELPER: Runs its own event loop: starts the watch, returns the worker infos of
    # the given cell (default engine-00000) and stops the watch. Assertion errors from the
    # provider propagate.
    async def scenario():
        # [FUNCTION DOC] HELPER: Starts the watch, returns the first group of worker infos, always
        # stops the watch.
        stop = await _watch(provider, [])
        try:
            return provider.get_worker_infos(cell_ids=[cell_id])[0]
        finally:
            await stop()

    return asyncio.run(scenario())


def _worker_handle(provider, cell_id="engine-00000"):
    # [FUNCTION DOC] HELPER: Runs its own event loop: starts the watch, gets the worker infos of
    # the cell (default engine-00000), builds the handles and returns the first worker's handle.
    async def scenario():
        # [FUNCTION DOC] HELPER: Starts the watch, returns the first worker's handle, always stops
        # the watch.
        stop = await _watch(provider, [])
        try:
            (infos,) = provider.get_worker_infos(cell_ids=[cell_id])
            return provider.get_handles_of_worker_infos(infos)[infos[0].name]
        finally:
            await stop()

    return asyncio.run(scenario())


class TestGetWorkerInfos:
    # [CLASS DOC] Tests of the worker infos the provider answers for observed cells.
    def test_orders_the_workers_by_the_rank_label(self):
        # [FUNCTION DOC] Three pods listed in scrambled order; the worker names run rank 0, 1, 2.
        """A trainer cell reads rank 0 as its master, so an arbitrary pod order would misconfigure it."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-2", pod_in_cell_index="2", pod_ip="10.0.0.3"),
                make_pod(name="engine-0-0", pod_in_cell_index="0", pod_ip="10.0.0.1"),
                make_pod(name="engine-0-1", pod_in_cell_index="1", pod_ip="10.0.0.2"),
            ]
        )

        infos = _worker_infos(_trainer_provider(api))

        assert [info.name for info in infos] == ["engine-00000-00000", "engine-00000-00001", "engine-00000-00002"]

    def test_addresses_a_worker_at_its_pod_ip_on_the_spec_ports(self):
        # [FUNCTION DOC] Pod ip 10.1.2.3; the worker's addresses are master 9000 and rpc 8000 on
        # that ip.
        """Every rank has its own network namespace, so each publishes the spec's ports at its own ip."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])

        infos = _worker_infos(_trainer_provider(api))

        assert infos[0].self_addrs == {
            "master": HostAndPort(host="10.1.2.3", port=9000),
            "rpc": HostAndPort(host="10.1.2.3", port=8000),
        }

    def test_falls_back_to_the_headless_service_name_of_a_pod_without_an_ip(self):
        # [FUNCTION DOC] A pod with no ip and subdomain engine; the rpc host is
        # "engine-0-0.engine.rl.svc".
        """A pod's ip appears late, but its dns name is stable from the moment the workload names it."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip=None, subdomain="engine")])

        infos = _worker_infos(_trainer_provider(api))

        assert infos[0].self_addrs["rpc"].host == f"engine-0-0.engine.{NAMESPACE}.svc"

    def test_hands_out_an_rpc_handle_pointed_at_the_worker(self):
        # [FUNCTION DOC] The handle is an rpc handle with transport server url
        # "http://10.1.2.3:8000".
        """A trainer cell drives its ranks through these handles, so they must talk to the right pod."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.1.2.3")])

        handle = _worker_handle(_trainer_provider(api))

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._transport._server_url == "http://10.1.2.3:8000"

    def test_the_handle_knows_the_methods_of_the_worker_class(self):
        # [FUNCTION DOC] The handle's init attribute is callable, while looking up an unknown method
        # name raises an attribute error.
        """A typo would otherwise become a 404 at call time, deep inside a training step."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])

        handle = _worker_handle(_trainer_provider(api))

        assert callable(handle.init)
        with pytest.raises(AttributeError):
            handle.__getattr__("nonexistent_method")

    def test_reports_the_gpus_a_platform_annotated(self):
        # [FUNCTION DOC] A gpu id annotation "2,3" gives worker gpu ids [2, 3].
        """Colocation is verified against these ids, so they travel with the worker rather than beside it."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "2,3"})])

        assert _worker_infos(_trainer_provider(api))[0].gpu_ids == [2, 3]

    def test_counts_a_pod_restart_as_a_new_worker_generation(self):
        # [FUNCTION DOC] A pod with restart count 2 gives worker generation 2.
        """A restarted pod kept its name and lost its memory, and a consumer must be able to tell."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", restarts=2)])

        assert _worker_infos(_trainer_provider(api))[0].generation == 2

    def test_refuses_a_cell_that_is_missing_a_pod(self):
        # [FUNCTION DOC] Only pod index 1 exists; asking for worker infos raises an assertion error
        # matching "missing pods".
        """Driving half a cell would let the missing ranks' collective hang the whole run."""
        api = FakePodApi(pods=[make_pod(name="engine-0-1", pod_in_cell_index="1")])

        with pytest.raises(AssertionError, match="missing pods"):
            _worker_infos(_trainer_provider(api))

    def test_refuses_a_cell_it_has_never_observed(self):
        # [FUNCTION DOC] Worker infos for engine-00009 with no pods raise an assertion error
        # matching "no observed worker pods".
        """Returning no workers would read as a cell with nothing to do rather than as an error."""
        with pytest.raises(AssertionError, match="no observed worker pods"):
            _worker_infos(_trainer_provider(FakePodApi()), cell_id="engine-00009")

    def test_a_worker_that_is_not_served_names_no_class_and_refuses_a_handle(self):
        # [FUNCTION DOC] A command pool (no worker class): the single info has no worker class, and
        # building an rpc handle from it raises an assertion error matching "is not served".
        """A command worker has no rpc surface at all, so the refusal belongs where one is asked for."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0")])
        provider = _provider(api, worker_ports={"engine": {"rpc": 8000}})

        (info,) = _worker_infos(provider)

        assert info.worker_class is None
        with pytest.raises(AssertionError, match="is not served"):
            build_rpc_handle_of_worker_info(info)

    def test_fans_a_pod_out_into_one_worker_per_rank_it_serves(self):
        # [FUNCTION DOC] One pod with 3 workers per pod yields workers rank 0, 1, 2.
        """A supervised pod runs one worker process per rank, and each of them has to be driven separately."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 3}))

        assert [info.name for info in infos] == ["engine-00000-00000", "engine-00000-00001", "engine-00000-00002"]

    def test_offsets_the_rpc_port_of_each_rank_the_way_the_process_binds_it(self):
        # [FUNCTION DOC] With 2 workers per pod the rpc ports are 8000 and 8001 while the master
        # port is 9000 for both (master-mode ports are not offset).
        """serve_inner listens on port + worker_in_pod_index, so any other guess reaches the wrong rank or nothing."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

        assert [info.self_addrs["rpc"].port for info in infos] == [8000, 8001]
        assert [info.self_addrs["master"].port for info in infos] == [9000, 9000]

    def test_numbers_the_ranks_of_the_second_pod_after_those_of_the_first(self):
        # [FUNCTION DOC] Two pods of two workers each, listed out of order; names are ranks 0 to 3
        # and the rpc hosts are the first pod's ip twice, then the second's twice.
        """A rank's name is its index in the cell, which spans the pods rather than restarting in each."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-1", pod_in_cell_index="1", pod_ip="10.0.0.2"),
                make_pod(name="engine-0-0", pod_in_cell_index="0", pod_ip="10.0.0.1"),
            ]
        )

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

        assert [info.name for info in infos] == [
            "engine-00000-00000",
            "engine-00000-00001",
            "engine-00000-00002",
            "engine-00000-00003",
        ]
        assert [info.self_addrs["rpc"].host for info in infos] == ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2"]

    def test_gives_each_rank_its_own_share_of_the_gpus_of_the_pod(self):
        # [FUNCTION DOC] Pod gpus "0,1,2,3" with 2 workers: gpu ids [0,1] and [2,3].
        """A rank takes the gpu slots at its own offset, exactly as the pod's own process numbering does."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "0,1,2,3"})])

        infos = _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

        assert [info.gpu_ids for info in infos] == [[0, 1], [2, 3]]

    def test_refuses_a_pod_whose_gpus_do_not_divide_among_its_ranks(self):
        # [FUNCTION DOC] Pod gpus "0,1,2" with 2 workers raises an assertion error matching "equal
        # share".
        """Handing a rank a partial share would silently point two ranks at one gpu."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", annotations={"miles.radixark.io/meta-gpu_ids": "0,1,2"})])

        with pytest.raises(AssertionError, match="equal share"):
            _worker_infos(_trainer_provider(api, workers_per_pod={"engine": 2}))

    def test_resolves_the_address_of_a_rank_that_no_pod_is_named_after(self):
        # [FUNCTION DOC] One pod with 2 workers; the address of worker rank 1 has rpc host 10.0.0.1
        # and port 8001.
        """Consumers hold the rank names this provider handed out, so it has to answer for them."""
        api = FakePodApi(pods=[make_pod(name="engine-0-0", pod_ip="10.0.0.1")])
        provider = _trainer_provider(api, workers_per_pod={"engine": 2})

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the worker's addresses, always stops
            # the watch.
            stop = await _watch(provider, [])
            try:
                return await provider.get_addrs("engine-00000-00001")
            finally:
                await stop()

        assert asyncio.run(scenario())["rpc"] == HostAndPort(host="10.0.0.1", port=8001)

    def test_answers_for_several_cells_at_once(self):
        # [FUNCTION DOC] Two cells with one pod each; asking for both cell ids returns two groups
        # whose worker names are engine-00000-00000 and engine-00001-00000.
        """A controller resynchronises every cell it owns, and one round trip per cell is the shape."""
        api = FakePodApi(
            pods=[
                make_pod(name="engine-0-0", cell_id_suffix="0"),
                make_pod(name="engine-1-0", cell_id_suffix="1"),
            ]
        )
        provider = _trainer_provider(api)

        async def scenario():
            # [FUNCTION DOC] HELPER: Starts the watch, returns the worker infos of both cells,
            # always stops the watch.
            stop = await _watch(provider, [])
            try:
                return provider.get_worker_infos(cell_ids=["engine-00000", "engine-00001"])
            finally:
                await stop()

        assert [[info.name for info in infos] for infos in asyncio.run(scenario())] == [
            ["engine-00000-00000"],
            ["engine-00001-00000"],
        ]
