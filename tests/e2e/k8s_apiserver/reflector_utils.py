# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Test helpers for exercising the Kubernetes pod reflector and reconcile loop
# against a real API server: a key function mapping a pod to its cell label value, a pod API
# wrapper that counts lists and records watch cursors, a recorder serving as the reconcile
# callback, and an async context manager that runs a reconcile loop over a reflector (watch
# retry delay 0.5 seconds) and always stops it.

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from tests.e2e.k8s_apiserver.utils import CELL_LABEL

from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.k8s_api import KubernetesPodApi, PodListPage, PodWatchEvent
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.loop import ReconcileLoop


def pod_cell(pod: Pod) -> str:
    # [FUNCTION DOC] HELPER: Returns the value of the cell label on the pod (the reconcile key);
    # raises a key error if the pod lacks the label.
    return pod.metadata.labels[CELL_LABEL]


class CountingPodApi:
    # [CLASS DOC] HELPER: Decorator over a pod API that delegates list and stream calls to the
    # wrapped API while recording how many lists happened (public counter) and the resource
    # version cursor of every stream opened, in order (public list). Used to assert whether the
    # reflector relisted or merely reconnected.
    def __init__(self, *, inner: KubernetesPodApi) -> None:
        # [FUNCTION DOC] HELPER: Wraps the given pod API with a zero list count and no recorded
        # cursors.
        self.list_count = 0
        self.stream_cursors: list[str] = []
        self._inner = inner

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        # [FUNCTION DOC] HELPER: Increments the list count, then returns the wrapped API's list page.
        self.list_count += 1
        return await self._inner.list_pods(namespace=namespace, label_selector=label_selector)

    def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        # [FUNCTION DOC] HELPER: Records the resource version cursor, then returns the wrapped API's
        # event stream for the same arguments (the stream is not consumed here).
        self.stream_cursors.append(resource_version)
        return self._inner.stream_pods(
            namespace=namespace,
            label_selector=label_selector,
            resource_version=resource_version,
            timeout_seconds=timeout_seconds,
        )


class ReconcileRecorder:
    # [CLASS DOC] HELPER: Reconcile callback for tests. Each call records the key (public ordered
    # list) and a snapshot, per key, of the names of the pods the bound loop then holds under that
    # key (public mapping, last snapshot wins). The loop must be assigned (public attribute)
    # before the recorder is driven.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with no keys, no snapshots and no bound loop.
        self.keys: list[str] = []
        self.snapshots: dict[str, list[str]] = {}
        self.loop: ReconcileLoop | None = None

    async def __call__(self, key: str) -> None:
        # [FUNCTION DOC] HELPER: Reconcile callback. Asserts a loop is bound, appends the key, and
        # stores the names of the pods currently in the loop's store whose parent is the key.
        assert self.loop is not None, "the recorder must be bound to a loop before it is driven"
        self.keys.append(key)
        self.snapshots[key] = [pod.metadata.name for pod in self.loop.get_by_parent(key)]

    def count(self, key: str) -> int:
        # [FUNCTION DOC] HELPER: How many times the key was reconciled so far.
        return self.keys.count(key)


@dataclass(frozen=True)
class RunningLoop:
    # [CLASS DOC] HELPER: Immutable pair of a started reconcile loop and the recorder that is its
    # reconcile callback.
    loop: ReconcileLoop
    reconciles: ReconcileRecorder


@asynccontextmanager
async def running_reconcile_loop(
    *, kube_client: KubernetesPodApi, namespace: str, watch_timeout_seconds: int = 60
) -> AsyncIterator[RunningLoop]:
    # [FUNCTION DOC] HELPER: Async context manager. Builds a reflector over the given pod API,
    # namespace and the cell label as selector, with the given watch timeout (default 60 seconds)
    # and a 0.5 second retry delay; builds a reconcile loop fed by the reflector, using a fresh
    # recorder as the reconcile callback and the pod's cell label as the key; binds the recorder
    # to the loop; starts the loop; yields the loop and recorder; always stops the loop on exit.
    reflector = KubernetesReflector(
        kube_client=kube_client,
        namespace=namespace,
        label_selector=CELL_LABEL,
        watch_timeout_seconds=watch_timeout_seconds,
        retry_delay=0.5,
    )
    recorder = ReconcileRecorder()
    loop = ReconcileLoop(source=reflector.watch, reconcile=recorder, key_map=pod_cell)
    recorder.loop = loop

    await loop.start()
    try:
        yield RunningLoop(loop=loop, reconciles=recorder)
    finally:
        await loop.stop()


def pod_names_of(loop: ReconcileLoop, cell: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Names of the pods the loop's store holds under the given cell key, in
    # the store's order.
    return [pod.metadata.name for pod in loop.get_by_parent(cell)]
