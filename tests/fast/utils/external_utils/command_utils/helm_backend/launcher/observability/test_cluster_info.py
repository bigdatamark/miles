# [MODULE DOC] Unit tests for the cluster info reporting of the Helm launcher observability code.
# The module under test summarizes the pods selected by a label selector in a namespace (counts of
# running, starting, pending, gated, failed and restarted pods), turns Kubernetes warning events
# into report lines (busiest first, truncated with a count of omitted ones), gives a hint pointing
# at the cluster's own dashboards for very large runs, keys events by pod identity, and runs a
# background watcher (started and stopped by a context manager) that logs only changes: new pods,
# changed summaries and warning events seen for the first time. Pod and event lookups are replaced
# with fakes through monkeypatching, the polling interval is shortened, and log output is
# captured; no cluster is contacted.
import logging

import pytest

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_event,
    make_pod,
    wait_for,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import cluster_info, polling
from miles.utils.workers.k8s_types import Event, Pod


class TestPodSummary:
    # [CLASS DOC] Tests for the one-line pod summary: "No pods yet" for an empty list, otherwise
    # "N pods:" followed by comma-separated non-empty categories among running, starting, pending,
    # gated, failed and restarted.
    def test_says_nothing_is_up_before_the_firstmake_pod(self):
        # [FUNCTION DOC] An empty pod list is summarized as exactly "No pods yet".
        # KNOWN ISSUE: the test name contains a typo ("firstmake_pod", presumably "first pod").
        """helm returns before the scheduler has acted, and an empty summary would read as a hang."""
        assert cluster_info._pod_summary([]) == "No pods yet"

    def test_separates_a_serving_pod_from_one_still_starting(self):
        # [FUNCTION DOC] A ready running pod and a running but not ready pod give exactly
        # "2 pods: 1 running, 1 starting".
        """A Running pod that is not ready is loading a model, which is the phase users wait through."""
        assert cluster_info._pod_summary([make_pod(), make_pod(ready=False)]) == "2 pods: 1 running, 1 starting"

    def test_counts_a_gated_pod_apart_from_a_pending_one(self):
        # [FUNCTION DOC] Two pending pods, one of them scheduling-gated, give exactly "2 pods: 1
        # pending, 1 gated" (a gated pod is not also counted as pending).
        """A colocate pod waits for its trainer's node on purpose; calling that pending looks like a stall."""
        pods = [make_pod(phase="Pending"), make_pod(phase="Pending", scheduling_gated=True)]

        assert cluster_info._pod_summary(pods) == "2 pods: 1 pending, 1 gated"

    def test_surfaces_failures_and_restarts(self):
        # [FUNCTION DOC] A failed pod and a pod with 3 container restarts: the summary contains
        # both "1 failed" and "1 restarted".
        """A crash looping pod is the one thing a user must not have to go looking for."""
        summary = cluster_info._pod_summary([make_pod(phase="Failed", ready=False), make_pod(restarts=3)])

        assert "1 failed" in summary and "1 restarted" in summary

    def test_omits_the_categories_that_are_empty(self):
        # [FUNCTION DOC] A single healthy pod gives exactly "1 pods: 1 running" (the count word
        # is not singularized, and zero categories are omitted).
        """A healthy run should read as one short line, not a table of zeroes."""
        assert cluster_info._pod_summary([make_pod()]) == "1 pods: 1 running"


class TestWarningLines:
    # [CLASS DOC] Tests for converting event objects to report lines: only warning-type events
    # are reported, sorted by occurrence count descending, capped (eight events give five lines
    # plus a trailer), each line in the form pod name, reason, "x" count, message.
    def test_names_the_reason_a_pod_cannot_be_scheduled(self):
        # [FUNCTION DOC] A failed-scheduling warning produces a first line containing the reason
        # and the message text about insufficient GPUs.
        """An unschedulable pod stays Pending forever, and its phase alone never says why."""
        events = [
            make_event(reason="FailedScheduling", message="0/8 nodes are available: insufficient nvidia.com/gpu")
        ]

        lines = cluster_info._warning_lines(events)

        assert "FailedScheduling" in lines[0]
        assert "insufficient nvidia.com/gpu" in lines[0]

    def test_names_the_image_a_pod_could_not_pull(self):
        # [FUNCTION DOC] A warning with reason "Failed", message "pull access denied" and
        # count 4 yields a first line containing "Failed x4" and the message.
        """An image pull failure is the other startup hang users hit, and it is invisible in the pod phase."""
        lines = cluster_info._warning_lines([make_event(reason="Failed", message="pull access denied", count=4)])

        assert "Failed x4" in lines[0]
        assert "pull access denied" in lines[0]

    def test_keeps_quiet_about_the_ordinary_events_of_a_healthy_run(self):
        # [FUNCTION DOC] A normal-type event (reason "Pulled") produces no lines.
        """Pulled, Created and Started arrive for every pod and would bury the one line that matters."""
        assert cluster_info._warning_lines([make_event(reason="Pulled", event_type="Normal")]) == []

    def test_reports_the_busiest_events_first_and_says_how_many_it_left_out(self):
        # [FUNCTION DOC] Eight warnings from pods p1 to p8 with counts 1 to 8: the first line
        # begins "p8: FailedScheduling x8" (highest count first) and the last line is exactly
        # "... and 3 more warning events", i.e. five events are listed.
        """A large run repeats one failure per pod, and printing all of them hides the summary line."""
        events = [make_event(pod_name=f"p{index}", count=index) for index in range(1, 9)]

        lines = cluster_info._warning_lines(events)

        assert lines[0].startswith("p8: FailedScheduling x8")
        assert lines[-1] == "... and 3 more warning events"


class TestScaleHint:
    # [CLASS DOC] Tests for the hint shown for large runs (above 100 pods, per the tests here)
    # recommending the cluster's own observability stack.
    def test_stays_quiet_for_a_run_a_user_can_read(self):
        # [FUNCTION DOC] Ten pods give no hint (none).
        """Advice nobody needs trains people to skip the output that matters."""
        assert cluster_info._scale_hint([make_pod() for _ in range(10)]) is None

    def test_points_a_large_run_at_the_cluster_dashboards(self):
        # [FUNCTION DOC] Two hundred pods give a hint text that mentions "200 pods".
        """A summary of hundreds of pods is worse than the tool built for it."""
        hint = cluster_info._scale_hint([make_pod() for _ in range(200)])

        assert "200 pods" in hint


class TestEventKey:
    # [CLASS DOC] Tests for the key identifying an event so repeated events are reported once.
    def test_tells_two_pods_of_one_name_apart(self):
        # [FUNCTION DOC] Two events for pods with the same name but different uids have
        # different keys (the pod uid is part of the key).
        """A pod rebuilt under its ordinal name is a different pod, and its warning has not been reported."""
        old = make_event(pod_name="trainer-0", uid="uid-old")
        new = make_event(pod_name="trainer-0", uid="uid-new")

        assert cluster_info._event_key(old) != cluster_info._event_key(new)


class TestClusterInfoWatcher:
    # [CLASS DOC] Tests for the watcher object (namespace and selector) whose report-changes
    # method polls pods and events once and logs only what is new since the previous call.
    def test_does_not_repeat_an_unchanged_pod_summary(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] With the pod lookup patched to return one running pod and the event
        # lookup patched to return no events, two consecutive change reports log the summary
        # "1 pods: 1 running" exactly once.
        """Polling an unchanged pod snapshot twice emits its summary only once."""
        pods = [make_pod(name="trainer-0")]

        def selected_pods(namespace: str, selector: str) -> list[Pod]:
            # [FUNCTION DOC] HELPER: Fake pod lookup returning the fixed pod list.
            return pods

        def pod_events(*, namespace: str, pods: list[Pod]) -> list[Event]:
            # [FUNCTION DOC] HELPER: Fake event lookup returning no events.
            return []

        monkeypatch.setattr(cluster_info, "selected_pods", selected_pods)
        monkeypatch.setattr(cluster_info, "pod_events", pod_events)
        watcher = cluster_info._ClusterInfoWatcher(namespace="rl", selector="app=x")

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            watcher.report_changes()
            watcher.report_changes()

        assert [record.message for record in caplog.records].count("1 pods: 1 running") == 1

    def test_reports_the_large_run_hint_only_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] With 101 pods and no events, two change reports log the scale hint
        # (recognized by the phrase "cluster's own observability stack") exactly once in total.
        """Polling the same large run twice emits the scale hint only once."""
        pods = [make_pod(name=f"trainer-{index}", uid=str(index)) for index in range(101)]

        def selected_pods(namespace: str, selector: str) -> list[Pod]:
            # [FUNCTION DOC] HELPER: Fake pod lookup returning the fixed 101 pods.
            return pods

        def pod_events(*, namespace: str, pods: list[Pod]) -> list[Event]:
            # [FUNCTION DOC] HELPER: Fake event lookup returning no events.
            return []

        monkeypatch.setattr(cluster_info, "selected_pods", selected_pods)
        monkeypatch.setattr(cluster_info, "pod_events", pod_events)
        watcher = cluster_info._ClusterInfoWatcher(namespace="rl", selector="app=x")

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            watcher.report_changes()
            watcher.report_changes()

        assert sum("cluster's own observability stack" in record.message for record in caplog.records) == 1


class TestWithClusterInfo:
    # [CLASS DOC] Tests for the context manager that runs the watcher in a background thread for
    # the duration of the with block (polling interval patched to 0.01 seconds), with fake pod and
    # event lookups.
    def test_reports_what_it_finds_and_stops_when_the_caller_leaves(self, monkeypatch, caplog):
        # [FUNCTION DOC] With one fake pod "trainer-0", waits inside the block until its name
        # appears in the log, then leaves. Asserts the log contains "pod trainer-0 appeared:
        # Running" and "1 pods: 1 running". The stopping is not asserted directly.
        """A watcher that outlived its run would keep talking over whatever the user does next."""
        monkeypatch.setattr(cluster_info, "selected_pods", lambda namespace, selector: [make_pod(name="trainer-0")])
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [])
        monkeypatch.setattr(polling, "POLL_INTERVAL_SECONDS", 0.01)

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            with cluster_info.with_cluster_info(namespace="rl", selector="app=x"):
                wait_for(lambda: "trainer-0" in caplog.text)

        assert "pod trainer-0 appeared: Running" in caplog.text
        assert "1 pods: 1 running" in caplog.text

    def test_reports_a_warning_event_only_once(self, monkeypatch, caplog):
        # [FUNCTION DOC] The fake event lookup returns the same failed-scheduling warning on
        # every poll; after it first shows in the log and the block is left, the text
        # "FailedScheduling" occurs exactly once in the captured log.
        """A pending pod carries the same event every poll, and repeating it would scroll the rest away."""
        monkeypatch.setattr(cluster_info, "selected_pods", lambda namespace, selector: [make_pod(name="trainer-0")])
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [make_event(pod_name="trainer-0")])
        monkeypatch.setattr(polling, "POLL_INTERVAL_SECONDS", 0.01)

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            with cluster_info.with_cluster_info(namespace="rl", selector="app=x"):
                wait_for(lambda: "FailedScheduling" in caplog.text)

        assert caplog.text.count("FailedScheduling") == 1

    def test_keeps_watching_when_a_poll_fails(self, monkeypatch, caplog):
        # [FUNCTION DOC] The fake pod lookup raises a runtime error on its first call and
        # returns a pod afterwards. The test waits until the pod appears in the log and asserts
        # the message "pod trainer-0 appeared", showing the watcher survived the first failure.
        """An api server hiccup must not silence the run's status for as long as the run lasts."""
        calls = []

        def flaky(namespace, selector):
            # [FUNCTION DOC] HELPER: Fake pod lookup; records the call, raises a runtime error
            # on the first call, and returns one pod "trainer-0" on later calls.
            calls.append(namespace)
            if len(calls) == 1:
                raise RuntimeError("api server said no")
            return [make_pod(name="trainer-0")]

        monkeypatch.setattr(cluster_info, "selected_pods", flaky)
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [])
        monkeypatch.setattr(polling, "POLL_INTERVAL_SECONDS", 0.01)

        with caplog.at_level(logging.INFO, logger=cluster_info.__name__):
            with cluster_info.with_cluster_info(namespace="rl", selector="app=x"):
                wait_for(lambda: "trainer-0" in caplog.text)

        assert "pod trainer-0 appeared" in caplog.text
