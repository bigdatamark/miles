# [MODULE DOC]
# Tests for the hot-restart cluster observer: the trainer health url computation, parsing of pod
# facts (uid and summed container restart counts, sorted by name) and workload facts (generation,
# restart stamp from the pod template annotation, template fingerprint independent of key order,
# stamped-twice refusal) for stateful sets and leader-worker sets, the observer's recording rules
# (only whole-release, settled, complete observations are recorded; attempts and failures are
# counted only once the release was up; raised reads are tolerated), and the background
# observation context manager (closing snapshot after the body, reporting an observer that does
# not stop in time, not leaking the thread when the body raises). The reader function is replaced
# by scripted fakes; no cluster is contacted.
import threading
import time

import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart import cluster_observer as cluster_module
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    LEADER_WORKER_SET_KIND,
    POD_KIND,
    STATEFUL_SET_KIND,
    ClusterObserver,
    ClusterSnapshot,
    PodFact,
    compute_trainer_rpc_url,
    parse_pod_facts,
    parse_workload_facts,
)
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ENGINE_POOL,
    NAMESPACE,
    ORCHESTRATOR,
    RELEASE,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import RESTART_AT_ANNOTATION


class TestComputeTrainerRpcUrl:
    # [CLASS DOC] Test of the trainer health url computation.
    def test_the_trainer_is_asked_for_its_boot_uuid_at_its_own_pod(self):
        # [FUNCTION DOC] The url for the demo release, namespace and trainer id starts with
        # "http://", contains the namespace and ends with the health path "/v1/health".
        """A trainer that did restart answers the same url with a different boot uuid."""
        url = compute_trainer_rpc_url(release=RELEASE, namespace=NAMESPACE, trainer_id="actor")

        assert url.startswith("http://")
        assert NAMESPACE in url
        assert url.endswith("/v1/health")


class TestParsePodFacts:
    # [CLASS DOC] Test of pod fact parsing.
    def test_a_pod_is_identified_by_the_uid_it_was_created_with(self):
        # [FUNCTION DOC] A pod list payload with pods "b" (two containers restarted 1 and 2 times)
        # and "a" (empty status) parses to pod facts sorted by name: a with restart count 0 and
        # b with the summed count 3.
        """A replaced pod keeps its name, so only the uid tells it apart from the one that survived."""
        payload = {
            "items": [
                {
                    "metadata": {"name": "b", "uid": "uid-b"},
                    "status": {"containerStatuses": [{"restartCount": 1}, {"restartCount": 2}]},
                },
                {"metadata": {"name": "a", "uid": "uid-a"}, "status": {}},
            ]
        }

        assert parse_pod_facts(payload) == (
            PodFact(name="a", uid="uid-a", restart_count=0),
            PodFact(name="b", uid="uid-b", restart_count=3),
        )


class TestParseWorkloadFacts:
    # [CLASS DOC] Tests of workload fact parsing from list payloads.
    def test_the_generation_and_the_stamp_of_each_statefulset_are_read(self):
        # [FUNCTION DOC] For two stateful sets (orchestrator generation 2 with a restart
        # annotation "t1" on its pod template; trainer generation 1 without), asserts kind, name,
        # generation and stamp (none for the trainer), that fingerprints are 64-character digests
        # and differ between the two.
        """A StatefulSet records both its generation and the canonical content of its pod template."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ORCHESTRATOR, "generation": 2},
                    "spec": {"template": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}}},
                },
                {"metadata": {"name": TRAINER, "generation": 1}, "spec": {"template": {"metadata": {}}}},
            ]
        }

        orchestrator, trainer = parse_workload_facts(payload, kind=STATEFUL_SET_KIND)

        assert (
            orchestrator.kind,
            orchestrator.name,
            orchestrator.generation,
            orchestrator.restart_at,
        ) == (STATEFUL_SET_KIND, ORCHESTRATOR, 2, "t1")
        assert (trainer.kind, trainer.name, trainer.generation, trainer.restart_at) == (
            STATEFUL_SET_KIND,
            TRAINER,
            1,
            None,
        )
        assert len(orchestrator.pod_template_fingerprint) == 64
        assert orchestrator.pod_template_fingerprint != trainer.pod_template_fingerprint

    def test_a_leaderworkerset_carries_its_stamp_on_the_template_of_its_group(self):
        # [FUNCTION DOC] For a leader-worker set whose stamp "t1" is on the worker template,
        # asserts the parsed kind, name, generation 3, stamp "t1" and a 64-character fingerprint.
        """The trainer cells and the engines of a run are leaderworkersets, not statefulsets."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ENGINE_POOL, "generation": 3},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "workerTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}}
                        }
                    },
                }
            ]
        }

        [engine_pool] = parse_workload_facts(payload, kind=LEADER_WORKER_SET_KIND)

        assert (engine_pool.kind, engine_pool.name, engine_pool.generation, engine_pool.restart_at) == (
            LEADER_WORKER_SET_KIND,
            ENGINE_POOL,
            3,
            "t1",
        )
        assert len(engine_pool.pod_template_fingerprint) == 64

    def test_equivalent_template_key_orders_have_the_same_fingerprint(self):
        # [FUNCTION DOC] Two payloads with the same template but different object key orders (and
        # different generations) give different generations yet the same fingerprint.
        """Kubernetes may return object keys in another order without changing the pod template."""
        first = {
            "items": [
                {
                    "metadata": {"name": TRAINER, "generation": 1},
                    "spec": {"template": {"metadata": {"labels": {"a": "1", "b": "2"}}, "spec": {}}},
                }
            ]
        }
        second = {
            "items": [
                {
                    "metadata": {"generation": 2, "name": TRAINER},
                    "spec": {"template": {"spec": {}, "metadata": {"labels": {"b": "2", "a": "1"}}}},
                }
            ]
        }

        [before] = parse_workload_facts(first, kind=STATEFUL_SET_KIND)
        [after] = parse_workload_facts(second, kind=STATEFUL_SET_KIND)

        assert before.generation != after.generation
        assert before.pod_template_fingerprint == after.pod_template_fingerprint

    def test_an_object_stamped_twice_over_is_refused(self):
        # [FUNCTION DOC] A leader-worker set with different stamps on both the leader and worker
        # templates raises an assertion error matching "restart stamps".
        """One stamp per replaced object is what makes counting the stamps count the take-overs."""
        payload = {
            "items": [
                {
                    "metadata": {"name": ENGINE_POOL, "generation": 3},
                    "spec": {
                        "leaderWorkerTemplate": {
                            "leaderTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t1"}}},
                            "workerTemplate": {"metadata": {"annotations": {RESTART_AT_ANNOTATION: "t2"}}},
                        }
                    },
                }
            ]
        }

        with pytest.raises(AssertionError, match="restart stamps"):
            parse_workload_facts(payload, kind=LEADER_WORKER_SET_KIND)


def _observer() -> ClusterObserver:
    # [FUNCTION DOC] HELPER: Builds an observer for the demo release, namespace and trainer id
    # "actor".
    return ClusterObserver(release=RELEASE, namespace=NAMESPACE, trainer_id="actor")


class TestClusterObserver:
    # [CLASS DOC] Tests of the observer's recording and counting rules with a scripted snapshot
    # reader.
    def _install_reader(self, monkeypatch, snapshots: list[ClusterSnapshot]) -> None:
        # [FUNCTION DOC] HELPER: Replaces the module's snapshot reader with a fake that returns
        # the given snapshots one per call, in order (popping from a copy); a call beyond the list
        # raises an index error.
        remaining = list(snapshots)
        monkeypatch.setattr(cluster_module, "read_cluster_snapshot", lambda **_kwargs: remaining.pop(0))

    def test_a_read_that_could_not_see_the_whole_release_is_counted_not_recorded(self, monkeypatch):
        # [FUNCTION DOC] Two settled snapshots then one with pod reads missing: only one snapshot
        # is recorded (the second settled one is a duplicate state) and counts are 2 attempts and
        # 1 failure.
        """A verdict read off two lucky observations of a run nobody could reach proves nothing."""
        observer = _observer()
        self._install_reader(
            monkeypatch,
            [
                _settled_snapshot(),
                _settled_snapshot(),
                cluster_snapshot(pods=[], workloads=[workload_fact(TRAINER)], reads_missing=(POD_KIND,)),
            ],
        )

        observer.observe_once()
        observer.observe_once()
        observer.observe_once()

        assert len(observer.snapshots) == 1
        assert (observer.attempts, observer.failures) == (2, 1)

    def test_the_polls_taken_before_the_release_was_installed_are_no_attempt_at_all(self, monkeypatch):
        # [FUNCTION DOC] Five polls: an empty snapshot, a snapshot with a pod but no workloads, a
        # snapshot with a workload and reads missing, then two settled snapshots. One snapshot is
        # recorded; attempts 1 and failures 1, since polls before the release was installed (no
        # workloads) do not count.
        # KNOWN ISSUE: the third poll (workload present, pod reads missing) is counted as the one
        # failure; attempts counts only polls after the first workload was seen, so the exact
        # boundary is implied rather than stated.
        """Counting them would divide a whole-release reading by however slowly helm happened to install."""
        observer = _observer()
        self._install_reader(
            monkeypatch,
            [
                cluster_snapshot(pods=[], workloads=[]),
                cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[]),
                cluster_snapshot(pods=[], workloads=[workload_fact(TRAINER)], reads_missing=(POD_KIND,)),
                _settled_snapshot(),
                _settled_snapshot(),
            ],
        )

        for _ in range(5):
            observer.observe_once()

        assert len(observer.snapshots) == 1
        assert (observer.attempts, observer.failures) == (1, 1)

    def test_a_pods_read_that_failed_is_a_failed_read_and_not_a_vanished_release(self, monkeypatch):
        # [FUNCTION DOC] With the low-level object reader returning nothing for pods (a failed
        # read) and an empty list otherwise, and the boot uuid reader fixed, the cluster snapshot
        # reader marks the pod kind as missing and the snapshot does not describe the whole
        # release.
        """These used to be the same thing, which left the "reads missing" branch unreachable."""
        monkeypatch.setattr(
            cluster_module,
            "_read_objects",
            lambda **kwargs: None if kwargs["kind"] == POD_KIND else {"items": []},
        )
        monkeypatch.setattr(cluster_module, "read_boot_uuid", lambda _url: "boot-a")

        snapshot = cluster_module.read_cluster_snapshot(
            release=RELEASE, namespace=NAMESPACE, trainer_rpc_url="http://x"
        )

        assert snapshot.reads_missing == (POD_KIND,) and not snapshot.describes_whole_release

    def test_a_read_that_raised_leaves_the_run_being_watched_rather_than_ending_it(self, monkeypatch):
        # [FUNCTION DOC] When the snapshot reader raises, the warn-wrapping observe call returns
        # normally, records no snapshot and counts one failure.
        """kubectl answers late now and then, and the observer is the one thread this run cannot lose."""
        observer = _observer()
        monkeypatch.setattr(cluster_module, "read_cluster_snapshot", _raise_boom)

        observer.observe_once_or_warn()

        assert observer.snapshots == []
        assert observer.failures == 1

    def test_a_read_that_raised_while_the_release_was_up_is_an_attempt_all_the_same(self, monkeypatch):
        # [FUNCTION DOC] Two settled observations, then a raising reader: one snapshot recorded,
        # 2 attempts and 1 failure (a raise after the release is up counts as an attempt).
        """Regression: a cluster that only ever raised scored a perfect ratio, since nothing counted against it."""
        observer = _observer()
        self._install_reader(monkeypatch, [_settled_snapshot(), _settled_snapshot()])
        observer.observe_once_or_warn()
        observer.observe_once_or_warn()
        monkeypatch.setattr(cluster_module, "read_cluster_snapshot", _raise_boom)

        observer.observe_once_or_warn()

        assert len(observer.snapshots) == 1
        assert (observer.attempts, observer.failures) == (2, 1)

    def test_a_read_that_raised_before_the_release_was_up_is_no_attempt_at_all(self, monkeypatch):
        # [FUNCTION DOC] A raising reader from the start: 0 attempts and 1 failure.
        """The same window the empty polls are outside of: helm had not installed anything to read yet."""
        observer = _observer()
        monkeypatch.setattr(cluster_module, "read_cluster_snapshot", _raise_boom)

        observer.observe_once_or_warn()

        assert (observer.attempts, observer.failures) == (0, 1)

    def test_a_release_being_uninstalled_is_not_recorded_as_a_run_losing_its_pods(self, monkeypatch):
        # [FUNCTION DOC] An observation with no pods and no workloads records nothing.
        """The last observation happens while the run is torn down, and every pod is gone by then."""
        observer = _observer()
        self._install_reader(monkeypatch, [cluster_snapshot(pods=[], workloads=[])])

        observer.observe_once()

        assert observer.snapshots == []

    def test_a_release_still_being_installed_is_not_recorded_as_a_settled_run(self, monkeypatch):
        # [FUNCTION DOC] First a snapshot with only the trainer workload and pod (release still
        # installing), then two settled snapshots (orchestrator and trainer). Only the settled
        # snapshot is recorded, and the observer's settled workload set is {orchestrator, trainer}.
        """A first workload and pod are non-empty, and reading only those loses every pod not created yet."""
        observer = _observer()
        self._install_reader(
            monkeypatch,
            [
                cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)]),
                _settled_snapshot(),
                _settled_snapshot(),
            ],
        )

        for _ in range(3):
            observer.observe_once()

        assert observer.snapshots == [_settled_snapshot()]
        assert observer._settled_workloads == frozenset({ORCHESTRATOR, TRAINER})

    def test_a_partial_listing_of_a_settled_release_is_not_recorded(self, monkeypatch):
        # [FUNCTION DOC] Two settled snapshots followed by one listing only the trainer (a missing
        # workload): only the settled snapshot is recorded.
        """A missing workload leaves an empty pod set, which reads as a healthy pod having been replaced."""
        observer = _observer()
        self._install_reader(
            monkeypatch,
            [
                _settled_snapshot(),
                _settled_snapshot(),
                cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)]),
            ],
        )

        for _ in range(3):
            observer.observe_once()

        assert observer.snapshots == [_settled_snapshot()]

    def test_a_settled_release_whose_pods_were_replaced_is_recorded(self, monkeypatch):
        # [FUNCTION DOC] Two identical settled snapshots then one with a new orchestrator uid: the
        # recorded list is [settled, replaced] (an identical repeat is not re-recorded, a changed
        # state is).
        """A take-over replacing pods is the one event these snapshots exist to catch."""
        observer = _observer()
        replaced = _settled_snapshot(orchestrator_uid="uid-o-2")
        self._install_reader(monkeypatch, [_settled_snapshot(), _settled_snapshot(), replaced])

        for _ in range(3):
            observer.observe_once()

        assert observer.snapshots == [_settled_snapshot(), replaced]


def _settled_snapshot(*, orchestrator_uid: str = "uid-o-1") -> ClusterSnapshot:
    # [FUNCTION DOC] HELPER: Snapshot with orchestrator (uid default "uid-o-1") and trainer
    # ("uid-t") pods and the two matching stateful set workloads.
    return cluster_snapshot(
        pods=[pod_fact(f"{ORCHESTRATOR}-0", uid=orchestrator_uid), pod_fact(f"{TRAINER}-0", uid="uid-t")],
        workloads=[workload_fact(ORCHESTRATOR), workload_fact(TRAINER)],
    )


def _raise_boom(**_kwargs) -> None:
    # [FUNCTION DOC] HELPER: Fake reader; ignores its arguments and raises a runtime error
    # ("kubectl said no").
    raise RuntimeError("kubectl said no")


def _wait_until_threads_left(count: int) -> None:
    # [FUNCTION DOC] HELPER: Polls every 10 ms for up to 5 seconds until the process's active
    # thread count is at most the given count; returns silently on timeout.
    deadline = time.monotonic() + 5.0
    while threading.active_count() > count and time.monotonic() < deadline:
        time.sleep(0.01)


class TestObservingTheClusterInTheBackground:
    # [CLASS DOC] Tests of the background observation context manager.
    def test_the_closing_snapshot_is_taken_after_the_body_returns(self, monkeypatch):
        # [FUNCTION DOC] Replaces the periodic and single observation methods with recorders and
        # enters the context with an empty body; the last recorded call is the single (closing)
        # observation.
        """The run ends inside the body, and the frame that shows its last pods comes after."""
        observer = _observer()
        taken: list[str] = []
        monkeypatch.setattr(
            cluster_module.ClusterObserver, "observe_once_or_warn", lambda _self: taken.append("polled")
        )
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once", lambda _self: taken.append("closing"))

        with cluster_module.observing_cluster(observer, poll_interval_seconds=0.0):
            pass

        assert taken[-1] == "closing"

    def test_an_observer_still_mid_read_when_asked_to_stop_is_reported(self, monkeypatch):
        # [FUNCTION DOC] The periodic observation blocks on an event (up to 30 s) and the join
        # timeout is shortened to 50 ms; exiting the context raises an assertion error matching
        # "still reading the run". The finally block releases the event and waits for the thread
        # count to return to its previous value.
        """Reading what it collected while it is still writing would race it."""
        release = threading.Event()
        monkeypatch.setattr(
            cluster_module.ClusterObserver, "observe_once_or_warn", lambda _self: release.wait(timeout=30.0)
        )
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once", lambda _self: None)
        monkeypatch.setattr(cluster_module, "JOIN_TIMEOUT_SECONDS", 0.05)
        before = threading.active_count()

        try:
            with pytest.raises(AssertionError, match="still reading the run"):
                with cluster_module.observing_cluster(_observer(), poll_interval_seconds=0.0):
                    pass
        finally:
            release.set()
            _wait_until_threads_left(before)

    def test_a_body_that_raised_does_not_leave_the_observer_running(self, monkeypatch):
        # [FUNCTION DOC] With no-op observation methods, a body raising a custom exception
        # propagates it, and after waiting the active thread count equals its prior value (the
        # poller thread was stopped).
        """A leaked poller keeps reading a release the next test is about to install over."""
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once_or_warn", lambda _self: None)
        monkeypatch.setattr(cluster_module.ClusterObserver, "observe_once", lambda _self: None)
        before = threading.active_count()

        with pytest.raises(_BodyFailed):
            with cluster_module.observing_cluster(_observer(), poll_interval_seconds=0.0):
                raise _BodyFailed

        _wait_until_threads_left(before)
        assert threading.active_count() == before


class _BodyFailed(Exception):
    # [CLASS DOC] HELPER: Marker exception raised from a context body in tests.
    pass
