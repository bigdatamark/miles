# [MODULE DOC] Unit tests for the top-level observability helpers of the Helm launcher: the
# context manager that, for a namespace and label selector, runs both the cluster status watcher
# and the log follower for the lifetime of the with block, and the farewell message printed when
# the launcher leaves (how to read logs again, how to tear down, the observability boundary
# statement). Kubectl JSON queries and subprocess start are replaced by fakes via monkeypatching;
# no cluster or subprocess is used.
import logging
import threading
from collections.abc import Iterator

import pytest

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_container,
    make_pod,
    wait_for,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher import observability
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import log_follower
from miles.utils.workers.k8s_types import EventList, Pod, PodList
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL


class TestWithObservability:
    # [CLASS DOC] Tests of the combined context manager.
    def test_reports_cluster_status_and_follows_logs_for_the_context_lifetime(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Replaces the kubectl JSON getter with a fake that records every call
        # and returns one running pod "trainer-0" (container id "docker://a") for pod queries
        # (each pod query waits on a two-party barrier, so both the status watcher and the log
        # follower must query concurrently before either proceeds) and an empty event list for
        # event queries. Replaces the process start function with one returning a fake process
        # that yields one timestamped line "training" and then blocks until terminated or killed.
        # Inside the block, waits until the log has both "1 pods: 1 running" and
        # "[trainer-0/app] training". Asserts: exactly two pod queries with the pod list return
        # type, namespace "rl" and selector "app=x" (one per component); an event query was made
        # with the event list return type, namespace "rl" and the field selector for Pod
        # kind and Warning type; and at least one process was started and all of them were killed
        # after leaving the block.
        """Both views of a run stay active only while the launcher watches it."""
        calls: list[tuple[str, dict[str, object]]] = []
        pod_queries = threading.Barrier(2)
        pod: Pod = make_pod(
            name="trainer-0",
            containers=(make_container(container_id="docker://a", running=True),),
        )

        def get_json(kind: str, **kwargs: object) -> PodList | EventList:
            # [FUNCTION DOC] HELPER: Fake JSON getter: records the kind and keyword arguments;
            # for pods waits at the barrier and returns the one-pod list; for events returns an
            # empty event list; any other kind fails an assertion.
            calls.append((kind, kwargs))
            if kind == "pods":
                pod_queries.wait()
                return PodList(items=[pod])
            assert kind == "events"
            return EventList(items=[])

        class FakeProcess:
            # [CLASS DOC] HELPER: Fake log subprocess whose output yields one line and then
            # blocks until terminate or kill is called.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes killed flag (false), return code 0, an
                # event signalling termination and the output generator.
                self.killed = False
                self.returncode = 0
                self.terminated = threading.Event()
                self.stdout = self._lines()

            def poll(self) -> int | None:
                # [FUNCTION DOC] HELPER: Returns 0 when killed, none otherwise.
                return 0 if self.killed else None

            def wait(self, timeout: float | None = None) -> int:
                # [FUNCTION DOC] HELPER: Returns 0 immediately.
                return 0

            def terminate(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the killed flag and the termination event.
                self.killed = True
                self.terminated.set()

            def kill(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the killed flag and the termination event.
                self.killed = True
                self.terminated.set()

            def _lines(self) -> Iterator[str]:
                # [FUNCTION DOC] HELPER: Yields one timestamped line "training", then waits for
                # the termination event.
                yield "2026-08-10T00:00:00.1Z training\n"
                self.terminated.wait()

        processes: list[FakeProcess] = []

        def popen(_command: list[str], **_kwargs: object) -> FakeProcess:
            # [FUNCTION DOC] HELPER: Fake process start: creates a fake process, records it and
            # returns it; ignores the command and options.
            process = FakeProcess()
            processes.append(process)
            return process

        monkeypatch.setattr(Kubectl, "get_json", get_json)
        monkeypatch.setattr(log_follower.subprocess, "Popen", popen)

        with caplog.at_level(logging.INFO):
            with observability.with_observability(namespace="rl", selector="app=x"):
                wait_for(lambda: "1 pods: 1 running" in caplog.text and "[trainer-0/app] training" in caplog.text)

        pod_calls = [kwargs for kind, kwargs in calls if kind == "pods"]
        assert pod_calls == [
            {"return_type": PodList, "namespace": "rl", "selector": "app=x"},
            {"return_type": PodList, "namespace": "rl", "selector": "app=x"},
        ]
        assert (
            "events",
            {
                "return_type": EventList,
                "namespace": "rl",
                "field_selector": "involvedObject.kind=Pod,type=Warning",
            },
        ) in calls
        assert processes and all(process.killed for process in processes)


class TestFarewell:
    # [CLASS DOC] Tests of the farewell text builder, taking a namespace, release name and
    # workload (statefulset) name and returning a message string.
    def test_tells_the_user_how_to_look_again_and_how_to_stop(self):
        # [FUNCTION DOC] The message contains a "kubectl logs" command and the exact text "tear
        # down earlier: helm uninstall -n rl miles-run-x".
        """The release outlives the launcher, so both commands are the only way back to the run."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "kubectl logs" in message
        assert "tear down earlier: helm uninstall -n rl miles-run-x" in message

    def test_the_farewell_offers_the_whole_release_as_well_as_the_orchestrator(self):
        # [FUNCTION DOC] The message names the orchestrator statefulset and also a selector on
        # the instance label equal to the release name ("--selector" followed by the label key,
        # an equals sign and the release).
        """Sending a user to one statefulset makes every other pod of their run invisible."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "statefulset/r-miles-run-orchestrator" in message
        assert f"--selector {INSTANCE_LABEL}=miles-run-x" in message

    def test_the_farewell_always_says_where_this_summary_stops(self):
        # [FUNCTION DOC] The message contains the module's observability boundary constant text
        # and the phrase "observability stack".
        """A user who mistakes a vanilla launcher for monitoring will not notice what it never reported."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert observability._OBSERVABILITY_BOUNDARY in message
        assert "observability stack" in message

    def test_says_the_release_removes_itself_when_it_will(self):
        # [FUNCTION DOC] The message says the release "uninstalls itself about two minutes"
        # later and still gives the manual tear-down command.
        # KNOWN ISSUE: the farewell builder is called with the same arguments as in the earlier
        # tests, and no argument selects a self-removing release, so the test name ("when it
        # will") suggests a condition that is not set up here; and it asserts the same
        # tear-down line as the first farewell test.
        """A user told to uninstall by hand would keep doing it, and wonder why nothing was there."""
        message = observability.farewell(namespace="rl", release="miles-run-x", workload="r-miles-run-orchestrator")

        assert "uninstalls itself about two minutes" in message
        assert "tear down earlier: helm uninstall -n rl miles-run-x" in message
