# [MODULE DOC]
# Unit tests for the Kubernetes implementation of cell operations (listing cells, suspending a
# cell by deleting its pods, resuming, and injecting faults into a worker). A fake provider and
# fake worker handles replace the real cluster access; pod deletion is replaced by a recording
# function or by a fake kubernetes client module installed into the module table.
import asyncio
import sys
from types import ModuleType
from typing import Any

import pytest

from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.cell_operations import kubernetes as cell_operations_kubernetes
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo


class FakeHandle:
    # [CLASS DOC] HELPER: Fake worker handle that records fault-injection calls and submissions
    # into lists shared with the provider. Its configured effect decides what a fault injection
    # does: return normally, raise a worker-unreachable error, hang for an hour, or raise a
    # given exception.
    def __init__(
        self,
        name: str,
        *,
        calls: list[tuple[str, str]],
        submissions: list[tuple[str, str, str]],
        effect: str | Exception = "return",
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the worker name, the shared call and submission lists
        # (appended to, not copied) and the effect (the string "return", "unreachable",
        # "never_answers", or an exception instance to raise).
        self._name = name
        self._calls = calls
        self._submissions = submissions
        self._effect = effect

    async def inject_fault(self, *, mode: str) -> None:
        # [FUNCTION DOC] HELPER: Records the worker name and mode in the shared call list, then
        # applies the configured effect: raises a worker-unreachable error for "unreachable",
        # sleeps for an hour for "never_answers", raises the exception if the effect is one,
        # otherwise returns nothing.
        self._calls.append((self._name, mode))
        if self._effect == "unreachable":
            raise WorkerUnreachableError(f"{self._name} is gone")
        if self._effect == "never_answers":
            await asyncio.sleep(3600)
        if isinstance(self._effect, Exception):
            raise self._effect

    async def submit_without_result(self, method_name: str, /, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Mimics the acknowledgement-only submission of a worker method.
        # Records worker name, method name and the mode keyword in the submissions list, then
        # delegates to the fault injection with that mode (so the configured effect applies).
        self._submissions.append((self._name, method_name, kwargs["mode"]))
        await self.inject_fault(mode=kwargs["mode"])


class FakeProvider:
    # [CLASS DOC] HELPER: Fake worker provider backed by a fixed mapping from cell id to cell
    # info. Counts how many times the cell watch was started and exposes the recorded fault
    # injections and submissions of the handles it hands out.
    def __init__(
        self,
        infos: dict[str, CellInfo],
        *,
        start_delay: float = 0.0,
        handle_effect: str | Exception = "return",
        unserved_workers: tuple[str, ...] = (),
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the cell infos, the simulated delay before the watch
        # becomes ready (seconds), the effect given to every handle, and the names of workers
        # that are not served over rpc (they get no handle). Initializes the watch counter to
        # zero and the injection and submission records to empty lists.
        self._infos = infos
        self._start_delay = start_delay
        self._handle_effect = handle_effect
        self._unserved_workers = unserved_workers
        self.watches = 0
        self.injections: list[tuple[str, str]] = []
        self.submissions: list[tuple[str, str, str]] = []

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # [FUNCTION DOC] HELPER: Returns, for each requested cell id in order, the list of fake
        # worker infos of that cell (empty for an unknown cell).
        return [self._worker_infos_of_cell(cell_id) for cell_id in cell_ids]

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, FakeHandle]:
        # [FUNCTION DOC] HELPER: Returns a mapping from worker name to a fake handle wired to this
        # provider's shared records and configured effect, skipping workers marked as unserved.
        return {
            info.name: FakeHandle(
                info.name,
                calls=self.injections,
                submissions=self.submissions,
                effect=self._handle_effect,
            )
            for info in infos
            if info.name not in self._unserved_workers
        }

    async def watch_cells(self, reconcile):
        # [FUNCTION DOC] HELPER: Fake start of the cell watch. Increments the watch counter,
        # sleeps for the configured start delay, and returns an async stop function that does
        # nothing. The reconcile callback is ignored.
        self.watches += 1
        await asyncio.sleep(self._start_delay)
        return _stop_watching

    def cell_ids(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns all known cell ids in sorted order.
        return sorted(self._infos)

    def cell_info(self, cell_id: str) -> CellInfo | None:
        # [FUNCTION DOC] HELPER: Returns the cell info for the id, or nothing if unknown.
        return self._infos.get(cell_id)

    def pod_names_of_cell(self, cell_id: str) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns a copy of the worker names of the cell, which double as
        # pod names in this fake; an empty list for an unknown cell.
        info = self._infos.get(cell_id)
        return list(info.worker_names) if info is not None else []

    def _worker_infos_of_cell(self, cell_id: str) -> list[WorkerInfo]:
        # [FUNCTION DOC] HELPER: Builds one worker info per worker name of the cell, with
        # generation zero, no addresses, no gpu ids and a fake worker class; empty for an
        # unknown cell.
        info = self._infos.get(cell_id)
        return [
            WorkerInfo(name=name, generation=0, self_addrs={}, gpu_ids=[], worker_class="fake.Worker")
            for name in (info.worker_names if info is not None else [])
        ]


def _info(cell_id="trainer-engine-actor-0", pool_id="trainer-engine-actor", workers=("trainer-engine-actor-0-0",)):
    # [FUNCTION DOC] HELPER: Builds an alive cell info with the given cell id, pool id and worker
    # names (defaults describe the first trainer cell with one worker), a fixed workers hash and
    # empty metadata.
    return CellInfo(
        cell_id=cell_id,
        pool_id=pool_id,
        alive=True,
        worker_names=list(workers),
        workers_hash="h",
        meta={},
    )


@pytest.fixture
def deleted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, list[str]]]:
    # [FUNCTION DOC] HELPER: Fixture that replaces the module's pod-deletion function with a fake
    # recording each call as a pair of namespace and pod-name list. Yields (returns) that
    # record list, so tests can assert which pods would have been deleted. No cluster is touched.
    recorded: list[tuple[str, list[str]]] = []

    async def fake_delete_pods(*, namespace: str, pod_names: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Appends the namespace and a copy of the pod names to the record.
        recorded.append((namespace, list(pod_names)))

    monkeypatch.setattr(cell_operations_kubernetes, "_delete_pods", fake_delete_pods)
    return recorded


def _operations(infos, *, start_delay: float = 0.0, handle_effect: str | Exception = "return", unserved_workers=()):
    # [FUNCTION DOC] HELPER: Builds Kubernetes cell operations in namespace "rl" on top of a fake
    # provider created from the cell infos, start delay, handle effect and unserved workers.
    provider = FakeProvider(
        infos, start_delay=start_delay, handle_effect=handle_effect, unserved_workers=unserved_workers
    )
    return KubernetesCellOperations(provider=provider, namespace="rl")


class TestCellInfos:
    # [CLASS DOC] Tests for listing cell infos, including filtering by pool id.
    def test_reports_the_cells_of_the_specs_it_was_asked_about(self):
        # [FUNCTION DOC] Builds a trainer cell and an engine cell, asks for the trainer pool only,
        # and asserts that just the trainer cell id is listed.
        """A trainer handler must not list the inference cells that share the namespace."""
        infos = {"trainer-engine-actor-0": _info(), "engine-0": _info(cell_id="engine-0", pool_id="engine")}
        operations = _operations(infos)

        listed = asyncio.run(operations.cell_infos(pool_ids=["trainer-engine-actor"]))

        assert list(listed) == ["trainer-engine-actor-0"]

    def test_reports_nothing_when_no_cell_exists_yet(self):
        # [FUNCTION DOC] With no cells at all, asserts the listing is an empty mapping.
        """A run whose pods are still being scheduled has no cells, which is not an error."""
        assert asyncio.run(_operations({}).cell_infos(pool_ids=["trainer-engine-actor"])) == {}


class TestWatching:
    # [CLASS DOC] Tests for the lazy start, sharing, failure retry and cancellation of the cell
    # watch that the operations object starts on first use.
    def test_the_first_read_starts_the_watch_it_needs(self):
        # [FUNCTION DOC] One cell listing on a fresh operations object; asserts the provider's
        # watch was started exactly once.
        """Nothing else starts it, and reading the store before the reflector filled it reports an empty run."""
        operations = _operations({"trainer-engine-actor-0": _info()})

        asyncio.run(operations.cell_infos(pool_ids=["trainer-engine-actor"]))

        assert operations._provider.watches == 1

    def test_later_reads_reuse_the_watch_already_running(self, deleted):
        # [FUNCTION DOC] Runs two listings and then a suspend in one event loop (pod deletion
        # faked by the fixture); asserts the watch was started only once overall.
        """A second reflector would double the apiserver load and leak the first one's session."""
        operations = _operations({"trainer-engine-actor-0": _info()})

        async def scenario():
            # [FUNCTION DOC] HELPER: Lists cells of the trainer-engine-actor pool twice, then suspends
            # the cell trainer-engine-actor-0, sequentially.
            await operations.cell_infos(pool_ids=["trainer-engine-actor"])
            await operations.cell_infos(pool_ids=["trainer-engine-actor"])
            await operations.suspend(cell_id="trainer-engine-actor-0")

        asyncio.run(scenario())

        assert operations._provider.watches == 1

    def test_concurrent_first_reads_start_one_watch_between_them(self, deleted):
        # [FUNCTION DOC] With a slow-starting watch (50 ms), gathers three concurrent first
        # listings; asserts the watch was started exactly once. The deleted fixture is unused.
        """The api server gathers its handlers, which share one instance, so the very first request races itself."""
        operations = _operations({"trainer-engine-actor-0": _info()}, start_delay=0.05)

        async def scenario():
            # [FUNCTION DOC] HELPER: Issues three concurrent listings of the trainer-engine-actor pool
            # and waits for all of them.
            await asyncio.gather(*[operations.cell_infos(pool_ids=["trainer-engine-actor"]) for _ in range(3)])

        asyncio.run(scenario())

        assert operations._provider.watches == 1

    async def test_a_failed_watch_is_retried_by_the_next_operation(self) -> None:
        # [FUNCTION DOC] Uses a provider whose first watch start raises a runtime error. Asserts
        # the first listing propagates that error, the second listing succeeds with the trainer
        # cell, and the watch was started twice in total.
        """A failed watch propagates once and the next operation starts a fresh watch."""

        class FailingOnceProvider(FakeProvider):
            # [CLASS DOC] HELPER: Provider whose first watch start fails.
            async def watch_cells(self, reconcile: Any) -> Any:
                # [FUNCTION DOC] HELPER: Counts the start; raises a runtime error on the first
                # one, and returns the no-op stop function afterwards.
                self.watches += 1
                if self.watches == 1:
                    raise RuntimeError("watch failed")
                return _stop_watching

        provider = FailingOnceProvider({"trainer-engine-actor-0": _info()})
        operations = KubernetesCellOperations(provider=provider, namespace="rl")

        with pytest.raises(RuntimeError, match="watch failed"):
            await operations.cell_infos(pool_ids=["trainer-engine-actor"])

        listed = await operations.cell_infos(pool_ids=["trainer-engine-actor"])

        assert list(listed) == ["trainer-engine-actor-0"]
        assert provider.watches == 2

    async def test_cancelling_the_waiter_cancels_and_discards_its_watch(self) -> None:
        # [FUNCTION DOC] Starts a listing task against a provider whose watch never finishes,
        # waits until the watch started, cancels the task, and asserts the task ends cancelled,
        # the watch coroutine itself was cancelled, and the operations object's pending-watch
        # slot was cleared (private attribute checked).
        """A cancelled request must not leave its unfinished cell watch running in the background."""
        watch_started = asyncio.Event()
        watch_cancelled = asyncio.Event()

        class CancellationAwareProvider(FakeProvider):
            # [CLASS DOC] HELPER: Provider whose watch blocks forever and signals when it starts
            # and when it is cancelled.
            async def watch_cells(self, reconcile: Any) -> Any:
                # [FUNCTION DOC] HELPER: Counts the start, sets the started event, waits forever,
                # and sets the cancelled event when it is torn down.
                self.watches += 1
                watch_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    watch_cancelled.set()

        provider = CancellationAwareProvider({"trainer-engine-actor-0": _info()})
        operations = KubernetesCellOperations(provider=provider, namespace="rl")
        waiter = asyncio.create_task(operations.cell_infos(pool_ids=["trainer-engine-actor"]))
        await watch_started.wait()

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert watch_cancelled.is_set()
        assert operations._watching is None


class TestSuspend:
    # [CLASS DOC] Tests that suspending a cell deletes exactly its pods in the run's namespace.
    def test_deletes_the_pods_of_the_cell_in_the_runs_namespace(self, deleted):
        # [FUNCTION DOC] Suspends a two-pod cell; asserts one deletion in namespace "rl" naming
        # both pods.
        """Deleting them is the whole operation: the workload brings the group back by itself."""
        operations = _operations({"trainer-engine-actor-0": _info(workers=("p0", "p1"))})

        asyncio.run(operations.suspend(cell_id="trainer-engine-actor-0"))

        assert deleted == [("rl", ["p0", "p1"])]

    def test_touches_no_other_cell(self, deleted):
        # [FUNCTION DOC] With two cells, suspends the first; asserts only the first cell's pod is
        # deleted.
        """Healing one dp group must leave the others training."""
        infos = {
            "trainer-engine-actor-0": _info(workers=("a",)),
            "trainer-engine-actor-1": _info(cell_id="trainer-engine-actor-1", workers=("b",)),
        }

        asyncio.run(_operations(infos).suspend(cell_id="trainer-engine-actor-0"))

        assert deleted == [("rl", ["a"])]

    def test_refuses_a_cell_with_no_pods(self, deleted):
        # [FUNCTION DOC] Suspends an unknown cell; asserts an assertion error whose message
        # mentions "no pods". The deleted fixture is only there to block real deletion.
        """There is nothing to delete, and silently succeeding would report a heal that never happened."""
        with pytest.raises(AssertionError, match="no pods"):
            asyncio.run(_operations({}).suspend(cell_id="trainer-engine-actor-0"))


class TestDeletePods:
    # [CLASS DOC] Tests the real pod-deletion function against a fake kubernetes client package.
    async def test_delete_pods_uses_the_in_cluster_client_and_requested_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Installs fake kubernetes client, config and package modules into the
        # module table (restored by monkeypatch), calls the deletion function for two pods in
        # namespace "training", and asserts the in-cluster config was loaded exactly once and
        # each pod was deleted in that namespace, in the given order.
        """Pod deletion loads in-cluster config and targets every requested pod in its namespace."""
        config_loads: list[None] = []
        deletions: list[tuple[str, str]] = []

        class FakeApiClient:
            # [CLASS DOC] HELPER: Fake async-context-manager api client.
            async def __aenter__(self) -> Any:
                # [FUNCTION DOC] HELPER: Returns the client itself.
                return self

            async def __aexit__(self, *args: Any) -> None:
                # [FUNCTION DOC] HELPER: Does nothing and does not suppress exceptions.
                return None

        class FakeCoreV1Api:
            # [CLASS DOC] HELPER: Fake core api wrapper that records pod deletions.
            def __init__(self, api_client: FakeApiClient) -> None:
                # [FUNCTION DOC] HELPER: Keeps the given client.
                self.api_client = api_client

            async def delete_namespaced_pod(self, *, name: str, namespace: str) -> None:
                # [FUNCTION DOC] HELPER: Appends a pair of pod name and namespace to the
                # deletions list.
                deletions.append((name, namespace))

        client_module = ModuleType("kubernetes_asyncio.client")
        client_module.ApiClient = FakeApiClient
        client_module.CoreV1Api = FakeCoreV1Api
        config_module = ModuleType("kubernetes_asyncio.config")
        config_module.load_incluster_config = lambda: config_loads.append(None)
        package = ModuleType("kubernetes_asyncio")
        package.client = client_module
        package.config = config_module
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio.client", client_module)
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio.config", config_module)

        await cell_operations_kubernetes._delete_pods(namespace="training", pod_names=["pod-0", "pod-1"])

        assert config_loads == [None]
        assert deletions == [("pod-0", "training"), ("pod-1", "training")]


class TestResume:
    # [CLASS DOC] Tests the resume operation, which is unsupported on Kubernetes.
    def test_says_it_cannot_promise_the_moment_a_cell_comes_back(self):
        # [FUNCTION DOC] Resumes a cell; asserts a not-implemented error whose message contains
        # "no moment to return at".
        """The workload recreates a deleted cell on its own schedule, so a caller that waited here would be lied to."""
        with pytest.raises(NotImplementedError, match="no moment to return at"):
            asyncio.run(_operations({"trainer-engine-actor-0": _info()}).resume(cell_id="trainer-engine-actor-0"))


class TestInjectFault:
    # [CLASS DOC] Tests fault injection into a chosen worker of a cell: submission style, worker
    # selection, mode forwarding, tolerated and propagated failures, and argument validation.
    def test_submits_the_crash_without_waiting_for_a_result(self) -> None:
        # [FUNCTION DOC] Injects a kill fault into worker 0 of a one-worker cell; asserts the
        # worker received an acknowledgement-only submission of the method "inject_fault" with
        # mode "sigkill".
        """A self-crashing RPC is sent through the acknowledgement-only worker-handle operation."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

        assert operations._provider.submissions == [("engine-0-0", "inject_fault", "sigkill")]

    def test_calls_the_worker_the_sub_index_picks(self):
        # [FUNCTION DOC] With a two-worker cell and sub index 1, asserts only the second worker
        # was called, with mode "sigkill".
        """A multi-pod cell is crashed by crashing one named rank, not whichever rank came first."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0", "engine-0-1"))})

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=1))

        assert operations._provider.injections == [("engine-0-1", "sigkill")]

    def test_passes_the_requested_mode_to_the_worker(self):
        # [FUNCTION DOC] Injects a segfault-mode fault; asserts the worker got mode "segfault".
        """The caller chose the failure mode, so the worker must not be crashed some other way."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SEGFAULT, sub_index=0))

        assert operations._provider.injections == [("engine-0-0", "segfault")]

    def test_a_worker_that_dies_before_answering_is_a_success(self):
        # [FUNCTION DOC] Handle raises a worker-unreachable error; asserts the injection returns
        # normally and the call was still made.
        """The call kills its own callee, so an unreachable worker is the outcome that was asked for."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, handle_effect="unreachable"
        )

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

        assert operations._provider.injections == [("engine-0-0", "sigkill")]

    def test_a_worker_that_never_answers_does_not_hang_the_caller(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Shrinks the module's injection timeout to 50 ms, uses a handle that never
        # answers; asserts the injection returns (timeout swallowed) and the call was made.
        """A killed process leaves the rpc poll retrying for an hour, which would block the api server request."""
        monkeypatch.setattr(cell_operations_kubernetes, "INJECT_FAULT_TIMEOUT_SECONDS", 0.05)
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, handle_effect="never_answers"
        )

        asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

        assert operations._provider.injections == [("engine-0-0", "sigkill")]

    def test_an_unexpected_rpc_failure_is_propagated(self):
        # [FUNCTION DOC] Handle raises a runtime error; asserts the same error escapes the
        # injection.
        """An unrelated RPC failure must not be mistaken for confirmation that the worker crashed."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))},
            handle_effect=RuntimeError("rpc protocol failed"),
        )

        with pytest.raises(RuntimeError, match="rpc protocol failed"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))

    def test_a_sub_index_beyond_the_cell_is_rejected(self):
        # [FUNCTION DOC] Sub index 1 on a one-worker cell; asserts an assertion error mentioning
        # "out of range".
        """Injecting into a neighbouring cell by accident would corrupt the test's premise."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))})

        with pytest.raises(AssertionError, match="out of range"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=1))

    def test_a_negative_sub_index_is_rejected(self):
        # [FUNCTION DOC] Sub index -1 on a two-worker cell; asserts an assertion error mentioning
        # "out of range".
        """Negative indexing would silently select the last worker instead of failing."""
        operations = _operations({"engine-0": _info(cell_id="engine-0", workers=("engine-0-0", "engine-0-1"))})

        with pytest.raises(AssertionError, match="out of range"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=-1))

    def test_a_worker_that_is_not_served_over_rpc_is_rejected(self):
        # [FUNCTION DOC] The provider returns no handle for the only worker; asserts an assertion
        # error mentioning "not served over rpc".
        """There is no call to make, and succeeding here would report a crash that never happened."""
        operations = _operations(
            {"engine-0": _info(cell_id="engine-0", workers=("engine-0-0",))}, unserved_workers=("engine-0-0",)
        )

        with pytest.raises(AssertionError, match="not served over rpc"):
            asyncio.run(operations.inject_fault(cell_id="engine-0", mode=FailureMode.SIGKILL, sub_index=0))


async def _stop_watching() -> None:
    # [FUNCTION DOC] HELPER: No-op async stop function returned by the fake watch starters.
    return None
