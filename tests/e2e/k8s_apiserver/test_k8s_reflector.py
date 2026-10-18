# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] Tests of the Kubernetes pod reflector, its asyncio pod API adapter, and the
# reconcile loop against a real API server (CPU stage, estimated 600 seconds, no labels). Groups:
# the adapter round trip and watch teardown, watch timeouts reconnecting from the cursor without a
# relist, label selector scoping, reconcile loop membership tracking and wake-ups, label mutation
# (reparenting, leaving and re-entering the selector), recovery after an API server restart,
# recovery from a real compacted-cursor expiry by relisting, and synthesis of missed deletions by a
# relist. Short watch timeout used: 5 seconds. Fixtures come from the package conftest.

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from kubernetes_asyncio import client as kubernetes_client
from tests.ci.ci_register import register_cpu_ci
from tests.e2e.k8s_apiserver.apiserver import ApiserverEnvironment, compact_etcd_to_head, restart_apiserver
from tests.e2e.k8s_apiserver.reflector_utils import CountingPodApi, pod_names_of, running_reconcile_loop
from tests.e2e.k8s_apiserver.utils import CELL_LABEL, pod_body, wait_until, wait_until_serving

from miles.utils.workers.reconcile.k8s_api import (
    KubernetesAsyncioPodApi,
    KubernetesPodApi,
    PodWatchEvent,
    exception_rejects_cursor,
)

register_cpu_ci(est_time=600, suite="stage-b-cpu", labels=[])

_SHORT_WATCH_TIMEOUT = 5


class TestKubernetesAsyncioPodApi:
    # [CLASS DOC] Tests of the asyncio pod API adapter, and of watch timeout and selector scoping.
    async def test_the_adapter_round_trips_pods_through_a_real_apiserver(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Creates pod-a; lists through the adapter and asserts only pod-a is
        # returned; opens a watch from the list's resource version; creates pod-b and deletes
        # pod-a; waits for a deleted event; asserts the added/deleted events, in order, are exactly
        # added pod-b then deleted pod-a (other event types such as modifications are ignored).
        """LIST returns what was created, and WATCH reports a later creation and deletion."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )

        listed = await api.list_pods(namespace=apiserver_namespace, label_selector=CELL_LABEL)
        assert [pod.metadata.name for pod in listed.pods] == ["pod-a"]

        collector = _StreamCollector(
            api.stream_pods(
                namespace=apiserver_namespace,
                label_selector=CELL_LABEL,
                resource_version=listed.resource_version,
                timeout_seconds=120,
            )
        )
        try:
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-b", cell="cell-a")
            )
            await apiserver_core_v1.delete_namespaced_pod(namespace=apiserver_namespace, name="pod-a")
            await wait_until(
                lambda: any(event.type == "DELETED" for event in collector.events),
                description="the deletion to reach the watch",
            )
        finally:
            await collector.close()

        lifecycle = [
            (event.type, event.pod.metadata.name) for event in collector.events if event.type in ("ADDED", "DELETED")
        ]
        assert lifecycle == [("ADDED", "pod-b"), ("DELETED", "pod-a")]

    async def test_an_abandoned_watch_closes_cleanly_and_a_later_one_still_opens(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Opens a watch, creates pod-a, waits for its event, remembers that event's
        # resource version and closes the watch mid-stream (must not raise). Opens a new watch from
        # that version, creates pod-b, and asserts the new watch delivers exactly pod-b.
        """Tearing down a live watch mid-stream neither raises nor wedges the next watch."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        listed = await api.list_pods(namespace=apiserver_namespace, label_selector=CELL_LABEL)

        abandoned = _StreamCollector(
            api.stream_pods(
                namespace=apiserver_namespace,
                label_selector=CELL_LABEL,
                resource_version=listed.resource_version,
                timeout_seconds=120,
            )
        )
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )
        await abandoned.wait_for(1)
        cursor = abandoned.events[0].pod.metadata.resource_version
        await abandoned.close()

        reopened = _StreamCollector(
            api.stream_pods(
                namespace=apiserver_namespace,
                label_selector=CELL_LABEL,
                resource_version=cursor,
                timeout_seconds=120,
            )
        )
        try:
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-b", cell="cell-a")
            )
            await reopened.wait_for(1)
        finally:
            await reopened.close()

        assert [event.pod.metadata.name for event in reopened.events] == ["pod-b"]

    async def test_the_loop_keeps_tracking_across_a_real_watch_timeout(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Runs a reconcile loop over a counting API with a 5 second watch timeout;
        # waits (up to 60 seconds) until at least two watches were opened; creates a pod and waits
        # until the loop sees it through the reopened watch; afterwards asserts exactly one list
        # call happened in total (a timeout must reconnect from the cursor, not relist).
        """A server-side watch timeout reconnects from the cursor instead of relisting."""
        api = CountingPodApi(inner=KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1))

        async with running_reconcile_loop(
            kube_client=api, namespace=apiserver_namespace, watch_timeout_seconds=_SHORT_WATCH_TIMEOUT
        ) as running:
            await wait_until(
                lambda: len(api.stream_cursors) >= 2,
                description="the first watch to time out and a second one to open",
                timeout=_SHORT_WATCH_TIMEOUT * 12,
            )
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
            )
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-a"],
                description="an event delivered through the reopened watch",
            )

        assert api.list_count == 1, "a watch timeout must reconnect from the cursor, not relist"

    async def test_the_label_selector_scopes_both_the_list_and_the_watch(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Creates a pod without the cell label; asserts the selector-filtered list is
        # empty; opens a watch from the list version; creates a second unlabelled pod and then a
        # labelled one; waits for one event; asserts the stream delivered exactly the labelled pod.
        """Objects outside the selector reach neither the list nor the stream."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        unlabelled = pod_body(name="pod-unlabelled", cell="cell-a")
        unlabelled.metadata.labels = {}
        await apiserver_core_v1.create_namespaced_pod(namespace=apiserver_namespace, body=unlabelled)

        listed = await api.list_pods(namespace=apiserver_namespace, label_selector=CELL_LABEL)
        assert listed.pods == []

        collector = _StreamCollector(
            api.stream_pods(
                namespace=apiserver_namespace,
                label_selector=CELL_LABEL,
                resource_version=listed.resource_version,
                timeout_seconds=120,
            )
        )
        try:
            second_unlabelled = pod_body(name="pod-unlabelled-2", cell="cell-a")
            second_unlabelled.metadata.labels = {}
            await apiserver_core_v1.create_namespaced_pod(namespace=apiserver_namespace, body=second_unlabelled)
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-labelled", cell="cell-a")
            )
            await collector.wait_for(1)
        finally:
            await collector.close()

        assert [event.pod.metadata.name for event in collector.events] == ["pod-labelled"]


class TestReconcileLoopAgainstAnApiserver:
    # [CLASS DOC] Tests of the reconcile loop's store and wake-ups driven by real server events.
    async def test_the_loop_tracks_cell_membership_as_pods_come_and_go(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] With pod-a pre-created in cell-a, starts the loop and asserts the initial
        # list populated cell-a. Creates pod-b (cell-a) and pod-c (cell-b), waits until cell-a has
        # two pods and cell-b one, then deletes pod-a and waits until cell-a holds only pod-b.
        """Membership converges on creation and on deletion, keyed by the cell label."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )

        async with running_reconcile_loop(kube_client=api, namespace=apiserver_namespace) as running:
            assert [pod.metadata.name for pod in running.loop.get_by_parent("cell-a")] == ["pod-a"]

            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-b", cell="cell-a")
            )
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-c", cell="cell-b")
            )
            await wait_until(
                lambda: len(running.loop.get_by_parent("cell-a")) == 2
                and len(running.loop.get_by_parent("cell-b")) == 1,
                description="both cells to be observed",
            )

            await apiserver_core_v1.delete_namespaced_pod(namespace=apiserver_namespace, name="pod-a")
            await wait_until(
                lambda: [pod.metadata.name for pod in running.loop.get_by_parent("cell-a")] == ["pod-b"],
                description="the deleted pod to leave its cell",
            )

    async def test_every_membership_change_wakes_the_cell(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Starts the loop, creates pod-a in cell-a, waits for at least one reconcile
        # of cell-a, records the count, deletes the pod, waits for the count to increase, and
        # asserts cell-b was never reconciled.
        """Real events reach reconcile, not only the store: each change raises that cell's reconcile count."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)

        async with running_reconcile_loop(kube_client=api, namespace=apiserver_namespace) as running:
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
            )
            await wait_until(
                lambda: running.reconciles.count("cell-a") >= 1,
                description="the creation to drive a reconcile",
            )
            after_creation = running.reconciles.count("cell-a")

            await apiserver_core_v1.delete_namespaced_pod(namespace=apiserver_namespace, name="pod-a")
            await wait_until(
                lambda: running.reconciles.count("cell-a") > after_creation,
                description="the deletion to drive a further reconcile",
            )

            assert running.reconciles.keys.count("cell-b") == 0, "a cell nothing happened to must never be woken"

    async def test_reconcile_reads_a_store_that_already_has_the_event(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Starts the loop, creates pod-a in cell-a, and waits until the recorder's
        # snapshot taken during a reconcile of cell-a already lists pod-a, i.e. the store was
        # updated before the reconcile was queued.
        """The store-before-enqueue invariant holds against a real stream, not just against fakes."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)

        async with running_reconcile_loop(kube_client=api, namespace=apiserver_namespace) as running:
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
            )
            await wait_until(
                lambda: running.reconciles.snapshots.get("cell-a") == ["pod-a"],
                description="a reconcile that already sees the pod its event announced",
            )


class TestLabelMutationAgainstAnApiserver:
    # [CLASS DOC] Tests of how patching the cell label of an existing pod is seen by the stream
    # and by the loop.
    async def test_a_patched_cell_label_moves_the_pod_to_its_new_cell(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Pod-a starts in cell-a; with the loop running, patches its cell label to
        # cell-b and waits until the loop shows it under cell-b and cell-a is empty.
        """A real MODIFIED event carries the new label, and the loop reparents the pod on it."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )

        async with running_reconcile_loop(kube_client=api, namespace=apiserver_namespace) as running:
            assert pod_names_of(running.loop, "cell-a") == ["pod-a"]

            await apiserver_core_v1.patch_namespaced_pod(
                name="pod-a",
                namespace=apiserver_namespace,
                body={"metadata": {"labels": {CELL_LABEL: "cell-b"}}},
            )
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-b") == ["pod-a"]
                and running.loop.get_by_parent("cell-a") == [],
                description="the pod to move to the cell its new label names",
            )

    async def test_leaving_and_reentering_the_selector_arrives_as_deleted_then_added(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] Opens a selector-filtered watch from the list version; removes the cell label
        # from pod-a and waits for one event; restores the label and waits for a second; asserts
        # the events are exactly a deleted then an added event, both for pod-a.
        """A filtered watch reports selector departure as DELETED and reentry as ADDED."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )
        listed = await api.list_pods(namespace=apiserver_namespace, label_selector=CELL_LABEL)

        collector = _StreamCollector(
            api.stream_pods(
                namespace=apiserver_namespace,
                label_selector=CELL_LABEL,
                resource_version=listed.resource_version,
                timeout_seconds=120,
            )
        )
        try:
            await apiserver_core_v1.patch_namespaced_pod(
                name="pod-a", namespace=apiserver_namespace, body={"metadata": {"labels": {CELL_LABEL: None}}}
            )
            await collector.wait_for(1)
            await apiserver_core_v1.patch_namespaced_pod(
                name="pod-a", namespace=apiserver_namespace, body={"metadata": {"labels": {CELL_LABEL: "cell-a"}}}
            )
            await collector.wait_for(2)
        finally:
            await collector.close()

        lifecycle = [(event.type, event.pod.metadata.name) for event in collector.events]
        assert lifecycle == [("DELETED", "pod-a"), ("ADDED", "pod-a")]

    async def test_the_loop_drops_a_pod_whose_cell_label_is_removed(
        self, apiserver_core_v1: kubernetes_client.CoreV1Api, apiserver_namespace: str
    ) -> None:
        # [FUNCTION DOC] With the loop running and pod-a in cell-a, removes the pod's cell label and
        # waits until cell-a is empty (the pod itself still exists).
        """Losing the selector label removes the pod from its cell without any pod deletion."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )

        async with running_reconcile_loop(kube_client=api, namespace=apiserver_namespace) as running:
            assert pod_names_of(running.loop, "cell-a") == ["pod-a"]

            await apiserver_core_v1.patch_namespaced_pod(
                name="pod-a", namespace=apiserver_namespace, body={"metadata": {"labels": {CELL_LABEL: None}}}
            )
            await wait_until(
                lambda: running.loop.get_by_parent("cell-a") == [],
                description="the unlabelled pod to leave its cell",
            )


class TestApiserverRestart:
    # [CLASS DOC] Test that the loop survives an API server container restart.
    async def test_the_loop_reconverges_after_the_apiserver_restarts(
        self,
        apiserver_core_v1: kubernetes_client.CoreV1Api,
        apiserver_namespace: str,
        apiserver_environment: ApiserverEnvironment,
    ) -> None:
        # [FUNCTION DOC] Runs the loop with a 5 second watch timeout and pod-a pre-created; waits
        # until pod-a is observed; restarts the server and waits until it serves; creates pod-b and
        # waits until cell-a holds exactly pod-a and pod-b in that order.
        """Losing the apiserver mid-watch is a transient failure the loop recovers from on its own."""
        api = KubernetesAsyncioPodApi(core_v1_api=apiserver_core_v1)
        await apiserver_core_v1.create_namespaced_pod(
            namespace=apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )

        async with running_reconcile_loop(
            kube_client=api, namespace=apiserver_namespace, watch_timeout_seconds=_SHORT_WATCH_TIMEOUT
        ) as running:
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-a"], description="the first pod to be observed"
            )

            restart_apiserver(apiserver_environment)
            await wait_until_serving(apiserver_core_v1)
            await apiserver_core_v1.create_namespaced_pod(
                namespace=apiserver_namespace, body=pod_body(name="pod-b", cell="cell-a")
            )

            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-a", "pod-b"],
                description="the cell to converge again after the apiserver came back",
            )


class TestCursorExpiryAgainstAnApiserver:
    # [CLASS DOC] Tests using the watch-cache-less server, where compacting etcd invalidates cursors.
    async def test_a_compacted_cursor_is_reported_in_a_shape_the_reflector_reads_as_expired(
        self,
        expiring_apiserver_core_v1: kubernetes_client.CoreV1Api,
        expiring_apiserver_namespace: str,
        expiring_apiserver_environment: ApiserverEnvironment,
    ) -> None:
        # [FUNCTION DOC] Lists to get a cursor; writes three pods outside the selector and compacts
        # etcd to head; opens a watch from the stale cursor through the adapter and asserts its
        # first outcome (event or exception) is recognized as a cursor rejection by the
        # reflector's own expiry detection.
        """The 410 the fakes encode is the 410 a real apiserver sends for a compacted cursor."""
        api = KubernetesAsyncioPodApi(core_v1_api=expiring_apiserver_core_v1)
        listed = await api.list_pods(namespace=expiring_apiserver_namespace, label_selector=CELL_LABEL)
        stale_cursor = listed.resource_version

        await _write_pods_outside_selector(expiring_apiserver_core_v1, expiring_apiserver_namespace)
        compact_etcd_to_head(expiring_apiserver_environment)

        outcome = await _first_watch_outcome(
            api.stream_pods(
                namespace=expiring_apiserver_namespace,
                label_selector=CELL_LABEL,
                resource_version=stale_cursor,
                timeout_seconds=30,
            )
        )
        assert _reads_as_expired(outcome), f"a compacted cursor was not reported as an expiry {outcome=}"

    async def test_the_loop_relists_after_a_real_cursor_expiry(
        self,
        expiring_apiserver_core_v1: kubernetes_client.CoreV1Api,
        expiring_apiserver_namespace: str,
        expiring_apiserver_environment: ApiserverEnvironment,
    ) -> None:
        # [FUNCTION DOC] Runs the loop (5 second watch timeout) with pod-a pre-created; waits until
        # observed and asserts exactly one list so far. Then writes pods outside the selector,
        # compacts etcd, restarts the server and waits until it serves; creates pod-b; waits until
        # cell-a converges to pod-a and pod-b; asserts more than one list occurred (the expired
        # cursor was recovered by relisting).
        """A cursor destroyed by compaction costs one relist, not the fleet."""
        api = CountingPodApi(inner=KubernetesAsyncioPodApi(core_v1_api=expiring_apiserver_core_v1))
        await expiring_apiserver_core_v1.create_namespaced_pod(
            namespace=expiring_apiserver_namespace, body=pod_body(name="pod-a", cell="cell-a")
        )

        async with running_reconcile_loop(
            kube_client=api, namespace=expiring_apiserver_namespace, watch_timeout_seconds=_SHORT_WATCH_TIMEOUT
        ) as running:
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-a"], description="the first pod to be observed"
            )
            assert api.list_count == 1

            await _write_pods_outside_selector(expiring_apiserver_core_v1, expiring_apiserver_namespace)
            compact_etcd_to_head(expiring_apiserver_environment)
            restart_apiserver(expiring_apiserver_environment)
            await wait_until_serving(expiring_apiserver_core_v1)

            await expiring_apiserver_core_v1.create_namespaced_pod(
                namespace=expiring_apiserver_namespace, body=pod_body(name="pod-b", cell="cell-a")
            )
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-a", "pod-b"],
                description="the cell to converge after its cursor expired",
            )

        assert api.list_count > 1, "an expired cursor must be recovered by a relist"


class TestRelistDeletionSynthesisAgainstAnApiserver:
    # [CLASS DOC] Test that deletions missed while the watch was disconnected are found by relisting.
    async def test_a_deletion_missed_while_disconnected_is_synthesized_by_the_relist(
        self,
        expiring_apiserver_core_v1: kubernetes_client.CoreV1Api,
        expiring_apiserver_namespace: str,
        expiring_apiserver_environment: ApiserverEnvironment,
    ) -> None:
        # [FUNCTION DOC] Wraps the adapter in a gate (to hold the watch down) and a counter; creates
        # pod-a and pod-b; runs the loop (5 second watch timeout) until both are observed. Closes
        # the gate and waits (up to 60 seconds) until a watch attempt is blocked; deletes pod-b,
        # writes pods outside the selector, compacts etcd, and opens the gate. Waits until cell-a
        # holds only pod-a, and asserts more than one list occurred, so the deletion came from a
        # relist.
        """A pod deleted while the watch was down leaves the store through the relist, not a replayed event."""
        gated = _GatedPodApi(inner=KubernetesAsyncioPodApi(core_v1_api=expiring_apiserver_core_v1))
        api = CountingPodApi(inner=gated)
        for name in ("pod-a", "pod-b"):
            await expiring_apiserver_core_v1.create_namespaced_pod(
                namespace=expiring_apiserver_namespace, body=pod_body(name=name, cell="cell-a")
            )

        async with running_reconcile_loop(
            kube_client=api, namespace=expiring_apiserver_namespace, watch_timeout_seconds=_SHORT_WATCH_TIMEOUT
        ) as running:
            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-a", "pod-b"],
                description="both pods to be observed",
            )

            gated.close_gate()
            await wait_until(
                lambda: gated.blocked_attempts >= 1,
                description="the reflector to be held disconnected",
                timeout=_SHORT_WATCH_TIMEOUT * 12,
            )
            await expiring_apiserver_core_v1.delete_namespaced_pod(
                namespace=expiring_apiserver_namespace, name="pod-b"
            )
            await _write_pods_outside_selector(expiring_apiserver_core_v1, expiring_apiserver_namespace)
            compact_etcd_to_head(expiring_apiserver_environment)
            gated.open_gate()

            await wait_until(
                lambda: pod_names_of(running.loop, "cell-a") == ["pod-a"],
                description="the missed deletion to be synthesized by the relist",
            )

        assert api.list_count > 1, "the deletion can only have arrived through a relist"


class _GatedPodApi:
    # [CLASS DOC] HELPER: Pod API decorator whose watch streams can be held back. While the gate is
    # closed, each newly started stream blocks before connecting and counts as a blocked attempt
    # (public counter); lists are never gated. The gate starts open.
    def __init__(self, *, inner: KubernetesPodApi) -> None:
        # [FUNCTION DOC] HELPER: Wraps the pod API; zero blocked attempts; gate open.
        self.blocked_attempts = 0
        self._inner = inner
        self._gate = asyncio.Event()
        self._gate.set()

    def close_gate(self) -> None:
        # [FUNCTION DOC] HELPER: Closes the gate so later-started streams block.
        self._gate.clear()

    def open_gate(self) -> None:
        # [FUNCTION DOC] HELPER: Opens the gate, releasing blocked streams.
        self._gate.set()

    async def list_pods(self, *, namespace: str, label_selector: str) -> Any:
        # [FUNCTION DOC] HELPER: Passes straight through to the wrapped API (never gated).
        return await self._inner.list_pods(namespace=namespace, label_selector=label_selector)

    def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        # [FUNCTION DOC] HELPER: Returns a gated stream generator for the arguments (nothing runs
        # until it is iterated).
        return self._gated_stream(
            namespace=namespace,
            label_selector=label_selector,
            resource_version=resource_version,
            timeout_seconds=timeout_seconds,
        )

    async def _gated_stream(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        # [FUNCTION DOC] HELPER: Async generator. If the gate is closed when started, counts a
        # blocked attempt and waits for the gate to open. Then opens the wrapped stream, re-yields
        # every event, and always closes the wrapped stream at the end.
        if not self._gate.is_set():
            self.blocked_attempts += 1
            await self._gate.wait()
        stream = self._inner.stream_pods(
            namespace=namespace,
            label_selector=label_selector,
            resource_version=resource_version,
            timeout_seconds=timeout_seconds,
        )
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()


async def _write_pods_outside_selector(core_v1_api: kubernetes_client.CoreV1Api, namespace: str) -> None:
    # [FUNCTION DOC] HELPER: Creates three pods with no labels in the namespace, advancing the
    # server's revision without producing events visible to selector-filtered watches.
    for index in range(3):
        body = pod_body(name=f"pod-unselected-{index}", cell="cell-ignored")
        body.metadata.labels = {}
        await core_v1_api.create_namespaced_pod(namespace=namespace, body=body)


async def _first_watch_outcome(stream: AsyncGenerator[PodWatchEvent, None]) -> Any:
    # [FUNCTION DOC] HELPER: Returns the first event of the stream, none if it ends empty, or the
    # exception it raised (not re-raised). Always closes the stream.
    try:
        async for event in stream:
            return event
        return None
    except Exception as exception:
        return exception
    finally:
        await stream.aclose()


def _reads_as_expired(outcome: Any) -> bool:
    # [FUNCTION DOC] HELPER: For an exception outcome, defers to the reflector's cursor-rejection
    # check on exceptions; for a watch event outcome, uses the event's cursor-rejection flag;
    # anything else is false.
    if isinstance(outcome, BaseException):
        return exception_rejects_cursor(outcome)
    return isinstance(outcome, PodWatchEvent) and outcome.rejects_cursor


class _StreamCollector:
    # [CLASS DOC] HELPER: Consumes a pod watch stream in a background task, appending each event to
    # a public list; supports waiting for a number of events and closing.
    def __init__(self, stream: AsyncGenerator[PodWatchEvent, None]) -> None:
        # [FUNCTION DOC] HELPER: Starts the background consumer task immediately (needs a running
        # event loop).
        self.events: list[PodWatchEvent] = []
        self._stream = stream
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Background task body: appends every stream event to the list.
        async for event in self._stream:
            self.events.append(event)

    async def wait_for(self, count: int, *, timeout: float = 120.0) -> None:
        # [FUNCTION DOC] HELPER: Waits (default 120 seconds) until at least count events arrived or
        # the consumer task finished; then asserts enough events arrived, failing with the events
        # and the task's exception if the stream ended early.
        await wait_until(
            lambda: len(self.events) >= count or self._task.done(),
            description=f"{count} watch event(s)",
            timeout=timeout,
        )
        assert len(self.events) >= count, f"the stream ended early {self.events=} {self._task.exception()=}"

    async def close(self) -> None:
        # [FUNCTION DOC] HELPER: Cancels the consumer task, waits for it (swallowing its errors), and
        # closes the stream.
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        await self._stream.aclose()
