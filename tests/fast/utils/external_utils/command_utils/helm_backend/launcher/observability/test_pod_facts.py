# [MODULE DOC] Unit tests for the pod fact helpers of the Helm launcher observability code:
# computing human-readable status change lines between two pod snapshots, deriving the container
# runs to follow from pods (live container, and previous life of a restarted container, keyed by
# pod uid, container name and previous flag), listing the pods selected by a label selector,
# listing Warning events of pods owned by the release (matched by pod uid), and observing
# one named workload pod (phase and startup failure reason). Kubernetes queries go through the
# command wrapper's process runner, replaced by a fake that answers with canned JSON keyed by the
# resource kind argument of the command.
import json
import subprocess

import pytest
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_container,
    make_pod,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import pod_facts


class TestStatusChanges:
    # [CLASS DOC] Tests for the function comparing the previous and current pod lists and
    # returning a list of message lines ("pod NAME appeared: PHASE", "... is now PHASE", "... is
    # gone"), with extra notes for gated scheduling, not-ready state and restart counts.
    def test_announces_a_pod_the_scheduler_has_just_created(self):
        # [FUNCTION DOC] A new pending, not-ready pod named "p" yields exactly "pod p appeared:
        # Pending".
        """Pods appear one by one over minutes, and a silent launcher reads as a hung one."""
        assert pod_facts.status_changes([], [make_pod(phase="Pending", ready=False)]) == ["pod p appeared: Pending"]

    def test_announces_the_step_from_scheduled_to_serving(self):
        # [FUNCTION DOC] A pod going from not ready to ready yields "pod p is now Running".
        """Loading a model takes minutes, so the move out of that state is the progress users wait for."""
        assert pod_facts.status_changes([make_pod(ready=False)], [make_pod(ready=True)]) == ["pod p is now Running"]

    def test_says_nothing_when_a_poll_finds_the_same_pods(self):
        # [FUNCTION DOC] Identical snapshots yield an empty list.
        """A line every few seconds for an unchanged run would bury the pod logs it sits between."""
        assert pod_facts.status_changes([make_pod()], [make_pod()]) == []

    def test_reports_a_pod_that_disappeared(self):
        # [FUNCTION DOC] A pod present before and absent now yields "pod p is gone".
        """A deleted or evicted pod stops producing logs, and its silence otherwise looks like a hang."""
        assert pod_facts.status_changes([make_pod()], []) == ["pod p is gone"]

    def test_separates_a_gated_pod_from_a_merely_pending_one(self):
        # [FUNCTION DOC] A new pending, scheduling-gated pod yields "pod p appeared: Pending
        # (scheduling gated)".
        """A colocate pod waits for its trainer's node on purpose; calling that pending looks like a stall."""
        changes = pod_facts.status_changes([], [make_pod(phase="Pending", ready=False, scheduling_gated=True)])

        assert changes == ["pod p appeared: Pending (scheduling gated)"]

    def test_reports_a_restart_a_phase_alone_would_hide(self):
        # [FUNCTION DOC] A running ready pod whose restart count goes from 0 to 2 yields "pod p
        # is now Running, restarted 2 times".
        """A crash looping pod stays Running between crashes, so only the count says anything is wrong."""
        assert pod_facts.status_changes([make_pod()], [make_pod(restarts=2)]) == [
            "pod p is now Running, restarted 2 times"
        ]

    def test_reports_the_restarts_of_a_pod_that_never_became_ready(self):
        # [FUNCTION DOC] A running, not-ready pod whose restart count goes from 0 to 3 yields
        # "pod p is now Running (not ready yet), restarted 3 times".
        """A crash-looping pod stays unready, so its increasing restart count must remain visible."""
        changes = pod_facts.status_changes([make_pod(ready=False)], [make_pod(ready=False, restarts=3)])

        assert changes == ["pod p is now Running (not ready yet), restarted 3 times"]


class TestContainerRuns:
    # [CLASS DOC] Tests for the function turning pods into a mapping of run keys to container run
    # records (with pod, container, previous flag, container id and running flag). Containers with
    # no container id and no previous life produce no runs; a recorded previous container id adds
    # a second run flagged previous and not running.
    def test_finds_the_container_a_running_pod_is_serving_from(self):
        # [FUNCTION DOC] A pod with one running container of known id gives exactly one run for
        # pod "p", container "app", running true.
        """A pod with no container id has not started yet, and following it would attach to nothing."""
        pods = [make_pod(containers=(make_container(container_id="docker://a", running=True),))]

        runs = pod_facts.container_runs(pods)

        assert [(run.key.pod, run.key.container, run.running) for run in runs.values()] == [("p", "app", True)]

    def test_treats_a_replaced_container_as_a_new_one_to_follow(self):
        # [FUNCTION DOC] Two snapshots where the container id changed from "a" to "b" give the
        # same set of run keys but different container ids in the run values, so a consumer can
        # detect replacement by comparing ids under an unchanged key.
        """A container that crashed and came back writes a fresh log the old stream would never show."""
        before = pod_facts.container_runs([make_pod(containers=(make_container(container_id="docker://a"),))])
        after = pod_facts.container_runs([make_pod(containers=(make_container(container_id="docker://b"),))])

        assert set(before) == set(after)
        assert [run.container_id for run in before.values()] != [run.container_id for run in after.values()]

    def test_does_not_confuse_a_rebuilt_pod_with_the_one_it_replaced(self):
        # [FUNCTION DOC] Two snapshots with different pod uids (u1 and u2) but the same runtime
        # container id give run records whose container id values differ, because a run's
        # container id is the pod uid joined with the runtime container id; the test uses that to
        # distinguish a rebuilt pod from the one it replaced (the run keys are equal).
        """A pod recreated under the same name is a different pod, and its logs start over."""
        before = pod_facts.container_runs(
            [make_pod(uid="u1", containers=(make_container(container_id="docker://a"),))]
        )
        after = pod_facts.container_runs([make_pod(uid="u2", containers=(make_container(container_id="docker://a"),))])

        assert [run.container_id for run in before.values()] != [run.container_id for run in after.values()]

    def test_keeps_the_life_a_container_already_ended(self):
        # [FUNCTION DOC] A container with id "b" and previous container id "a" gives two runs,
        # one previous and one current (previous flags sorted equal to false and true), and every
        # previous run is not running.
        """A container that crashed on startup is the case this whole layer exists to make readable."""
        container = make_container(container_id="docker://b", previous_container_id="docker://a")

        runs = pod_facts.container_runs([make_pod(containers=(container,))])

        assert sorted(run.key.previous for run in runs.values()) == [False, True]
        assert all(not run.running for run in runs.values() if run.key.previous)

    def test_ignores_a_container_that_never_ran(self):
        # [FUNCTION DOC] A container with no container id and no previous life gives an empty
        # mapping.
        """A pod waiting on an image pull has no logs at all, and asking for them only prints an error."""
        assert pod_facts.container_runs([make_pod(containers=(make_container(),))]) == {}


class FakeKubectlProcess:
    # [CLASS DOC] HELPER: Fake of the command wrapper's process runner. Holds a mapping from
    # resource kind word (the third element of the command, such as "pods", "pod" or "events") to
    # a JSON-serializable answer, and a list of all commands received. Calling it returns a
    # successful completed process whose output is the JSON text of the answer for that kind, or
    # empty text when no answer is configured (mimicking an ignore-not-found result).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty answers mapping and empty commands list.
        self.answers: dict[str, dict] = {}
        self.commands: list[list[str]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] HELPER: Records the command, looks up the answer by the kind (third
        # element), and returns a completed process with exit code 0, the JSON text (or empty
        # text) as output and empty error output. Extra keyword arguments are ignored.
        self.commands.append(argv)
        answer = self.answers.get(argv[2])
        stdout = json.dumps(answer) if answer is not None else ""
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=stdout, stderr="")


@pytest.fixture
def kubectl_process(monkeypatch: pytest.MonkeyPatch) -> FakeKubectlProcess:
    # [FUNCTION DOC] HELPER: Fixture installing a fresh fake process runner in the command
    # wrapper module for the test and returning it so tests can set answers and inspect
    # commands.
    fake = FakeKubectlProcess()
    monkeypatch.setattr(command_wrapper, "run_process", fake)
    return fake


def _raw_pod(name: str, *, phase="Running", ready=True, containers=()) -> dict:
    # [FUNCTION DOC] HELPER: Builds the raw (JSON-shaped) dictionary of a pod: metadata with the
    # name and uid "uid-" plus name, and a status with the phase, a single Ready condition ("True"
    # or "False" by the ready flag) and the given raw container status dictionaries.
    return {
        "metadata": {"name": name, "uid": f"uid-{name}"},
        "status": {
            "phase": phase,
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "containerStatuses": list(containers),
        },
    }


def _raw_event(
    pod_name: str, reason="FailedScheduling", message="no node", count=1, event_type="Warning", uid=None
) -> dict:
    # [FUNCTION DOC] HELPER: Builds the raw (JSON-shaped) dictionary of an event about a pod: the
    # involved object (name, kind "Pod", uid defaulting to "uid-" plus the name), reason, message,
    # count and type with the given defaults (failed scheduling warning, "no node", count 1).
    return {
        "involvedObject": {"name": pod_name, "kind": "Pod", "uid": uid or f"uid-{pod_name}"},
        "reason": reason,
        "message": message,
        "count": count,
        "type": event_type,
    }


class TestSelectedPods:
    # [CLASS DOC] Tests for listing pods by selector and converting them into typed pod objects.
    def test_reads_the_fields_the_watchers_report_on(self, kubectl_process):
        # [FUNCTION DOC] A raw pod with one container (id "docker://b", 2 restarts, running,
        # previously terminated container "docker://a") is parsed so that name, uid and phase
        # match, the ready check is true, the restart total is 2, and the container id, previous
        # container id and running state are exposed.
        """Every status line and every attach decision is built out of exactly these fields."""
        container = {
            "name": "app",
            "containerID": "docker://b",
            "restartCount": 2,
            "state": {"running": {}},
            "lastState": {"terminated": {"containerID": "docker://a"}},
        }
        kubectl_process.answers["pods"] = {"items": [_raw_pod("trainer-0", containers=[container])]}

        pod = pod_facts.selected_pods("rl", "app=x")[0]

        assert (pod.metadata.name, pod.metadata.uid, pod.status.phase) == ("trainer-0", "uid-trainer-0", "Running")
        assert (pod_facts.is_pod_ready(pod), pod_facts.restarts_of_pod(pod)) == (True, 2)
        assert pod.status.container_statuses[0].container_id == "docker://b"
        assert pod.status.container_statuses[0].last_state.terminated.container_id == "docker://a"
        assert pod.status.container_statuses[0].state.running is not None

    def test_survives_a_pod_the_api_has_barely_filled_in(self, kubectl_process):
        # [FUNCTION DOC] A raw pod with only name, uid and an empty status parses; the phase
        # reads as "Unknown", the pod is not ready with 0 restarts, and there are no container
        # statuses.
        """A pod one poll old carries almost no status, and crashing here would end the run's reporting."""
        kubectl_process.answers["pods"] = {"items": [{"metadata": {"name": "trainer-0", "uid": "u"}, "status": {}}]}

        pod = pod_facts.selected_pods("rl", "app=x")[0]

        assert pod_facts.phase_of_pod(pod) == "Unknown"
        assert (pod_facts.is_pod_ready(pod), pod_facts.restarts_of_pod(pod)) == (False, 0)
        assert pod.status.container_statuses == []

    def test_reports_no_pods_before_the_scheduler_has_acted(self, kubectl_process):
        # [FUNCTION DOC] With no configured answer (empty command output), the listing is an
        # empty list.
        """helm returns before any pod exists, and kubectl answers that with nothing at all."""
        assert pod_facts.selected_pods("rl", "app=x") == []


class TestPodEvents:
    # [CLASS DOC] Tests for listing Warning events of the given pods: only events whose
    # involved object uid matches the uid of one of the given pods are returned, one cluster query
    # is made with a field selector, and a missing count defaults to 1.
    def test_reads_the_events_of_the_pods_the_release_owns(self, kubectl_process):
        # [FUNCTION DOC] Two events in the answer (for "mine-0" and "someone-elses-0") and one
        # given pod "mine-0": only the event of "mine-0" is returned.
        """The launcher watches one release, and another run's failures would be a false alarm."""
        kubectl_process.answers["events"] = {"items": [_raw_event("mine-0"), _raw_event("someone-elses-0")]}

        events = pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0", uid="uid-mine-0")])

        assert [event.involved_object.name for event in events] == ["mine-0"]

    def test_asks_the_api_only_for_warnings_about_pods(self, kubectl_process):
        # [FUNCTION DOC] The command sent has the field selector option followed by the value
        # "involvedObject.kind=Pod,type=Warning".
        """A namespace also records node and job events, none of which explain a pod that will not start."""
        kubectl_process.answers["events"] = {"items": [_raw_event("mine-0")]}

        pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0", uid="uid-mine-0")])

        command = kubectl_process.commands[0]
        assert command[command.index("--field-selector") + 1] == "involvedObject.kind=Pod,type=Warning"

    def test_carries_the_reason_and_the_message_the_summary_prints(self, kubectl_process):
        # [FUNCTION DOC] An event with reason "Failed", message "denied" and count 3 is returned
        # with those values and type "Warning".
        """A reason without its message says a pod failed but not what a user has to fix."""
        kubectl_process.answers["events"] = {
            "items": [_raw_event("mine-0", reason="Failed", message="denied", count=3)]
        }

        event = pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0", uid="uid-mine-0")])[0]

        assert (event.reason, event.message, event.count, event.type) == ("Failed", "denied", 3, "Warning")

    def test_survives_an_event_the_api_left_half_empty(self, kubectl_process):
        # [FUNCTION DOC] An event whose count is null is returned with count 1.
        """A freshly created event has no count yet, and a crash here would hide the pod it describes."""
        kubectl_process.answers["events"] = {"items": [_raw_event("mine-0", count=None)]}

        event = pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0", uid="uid-mine-0")])[0]

        assert event.count == 1

    def test_does_not_ask_about_a_release_with_no_pods_yet(self, kubectl_process):
        # [FUNCTION DOC] With an empty pod list the result is empty and no command was run at
        # all.
        """Before the scheduler acts there is nothing to explain, and the call would only cost a round trip."""
        assert pod_facts.pod_events(namespace="rl", pods=[]) == []
        assert kubectl_process.commands == []

    def test_leaves_out_the_warning_of_a_pod_that_was_replaced(self, kubectl_process):
        # [FUNCTION DOC] An event whose involved object has the old uid does not match a pod of
        # the same name with a new uid (matching is by uid): the result is empty.
        """A replacement pod must not inherit a warning emitted by the old instance with the same name."""
        kubectl_process.answers["events"] = {"items": [_raw_event("mine-0", uid="uid-old")]}

        assert pod_facts.pod_events(namespace="rl", pods=[make_pod(name="mine-0", uid="uid-new")]) == []


class TestObservedPod:
    # [CLASS DOC] Tests for observing one pod by workload name in a namespace, returning a record
    # with its phase and startup failure reason, or none when the pod does not exist.
    def test_reads_the_phase_of_the_orchestrator_pod(self, kubectl_process):
        # [FUNCTION DOC] A raw pod in phase Failed is observed with phase "Failed".
        """The launcher decides a run lost its orchestrator from this, so it must read the real phase."""
        kubectl_process.answers["pod"] = _raw_pod("r-orchestrator-0", phase="Failed")

        observed = pod_facts.observed_pod("rl", "r-orchestrator")

        assert observed is not None and observed.phase == "Failed"

    def test_reports_a_pod_that_is_not_there_as_absent(self, kubectl_process):
        # [FUNCTION DOC] With no configured answer the observed pod is none.
        """A pod between deletion and recreation must read as gone, not as an error the launcher dies on."""
        assert pod_facts.observed_pod("rl", "r-orchestrator") is None

    def test_reports_a_container_that_cannot_start(self, kubectl_process):
        # [FUNCTION DOC] A pending pod whose container is waiting with reason
        # "CrashLoopBackOff" has startup failure "CrashLoopBackOff".
        """A restartPolicy Always orchestrator never reaches Failed, so only the waiting reason says it broke."""
        kubectl_process.answers["pod"] = _raw_pod(
            "r-orchestrator-0",
            phase="Pending",
            containers=[
                {"name": "orchestrator", "restartCount": 0, "state": {"waiting": {"reason": "CrashLoopBackOff"}}}
            ],
        )

        assert pod_facts.observed_pod("rl", "r-orchestrator").startup_failure == "CrashLoopBackOff"

    def test_reads_an_ordinary_wait_as_no_failure(self, kubectl_process):
        # [FUNCTION DOC] A container waiting with reason "PodInitializing" gives startup failure
        # none.
        """A container waiting on initialization is a normal start, and ending the run there is wrong."""
        kubectl_process.answers["pod"] = _raw_pod(
            "r-orchestrator-0",
            phase="Pending",
            containers=[
                {"name": "orchestrator", "restartCount": 0, "state": {"waiting": {"reason": "PodInitializing"}}}
            ],
        )

        assert pod_facts.observed_pod("rl", "r-orchestrator").startup_failure is None


class TestPodQueries:
    # [CLASS DOC] Test of the exact commands issued by the two pod queries.
    def test_pod_queries_preserve_their_namespace_selector_and_workload_identity(
        self, kubectl_process: FakeKubectlProcess
    ) -> None:
        # [FUNCTION DOC] Runs the selected-pods query (namespace "training", selector on the
        # instance label "run-a") and the observed-pod query (namespace "control", workload
        # "run-a-orchestrator"), with canned answers for a trainer pod and a succeeded
        # orchestrator pod. Asserts the returned pod name and phase and that exactly two commands
        # were issued: the pods listing with namespace, JSON output, ignore-not-found and the
        # selector, and the single-pod get for the pod named workload plus "-0" in its
        # namespace, with JSON output and ignore-not-found. Argument order is fixed.
        """Pod queries must preserve their namespace, selector, and workload identity at the API boundary."""
        kubectl_process.answers["pods"] = {"items": [_raw_pod("run-a-trainer-0")]}
        kubectl_process.answers["pod"] = _raw_pod("run-a-orchestrator-0", phase="Succeeded")

        pods = pod_facts.selected_pods(namespace="training", selector="app.kubernetes.io/instance=run-a")
        observed = pod_facts.observed_pod(namespace="control", workload="run-a-orchestrator")

        assert [pod.metadata.name for pod in pods] == ["run-a-trainer-0"]
        assert observed is not None and observed.phase == "Succeeded"
        assert kubectl_process.commands == [
            [
                "kubectl",
                "get",
                "pods",
                "--namespace",
                "training",
                "--output",
                "json",
                "--ignore-not-found",
                "--selector",
                "app.kubernetes.io/instance=run-a",
            ],
            [
                "kubectl",
                "get",
                "pod",
                "run-a-orchestrator-0",
                "--namespace",
                "control",
                "--output",
                "json",
                "--ignore-not-found",
            ],
        ]
