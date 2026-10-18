# [MODULE DOC] Tests for parallel execution of preflight permission checkers: a phase of N
# checkers must be asked in one parallel pass (proved with a barrier that only opens when all N
# calls are in flight together), makes exactly N calls and returns N outcomes, asks every
# resource and verb pair of the rule set (subresources are asked using the parent resource name
# and a subresource option), and an empty phase asks nothing. The kubectl raw runner is replaced
# by a thread safe recording fake that answers yes.
from __future__ import annotations

import subprocess
import threading

import pytest

from miles.utils.external_utils.miles_workbench.preflight import checkers as checkers_module
from miles.utils.external_utils.miles_workbench.preflight.checkers import (
    ResourceVerb,
    ResourceVerbAvailabilityChecker,
    Status,
    expand_resource_verbs,
    parallel_execute_checkers,
)

NAMESPACE = "rl"

_RULES = {
    "pods": ("create", "delete", "get", "list", "patch", "update", "watch"),
    "services": ("create", "delete", "get"),
    "pods/exec": ("create",),
}


class _RecordingKubectl:
    # [CLASS DOC] HELPER: Fake raw kubectl runner for concurrency tests. Records the arguments of
    # every call (guarded by a lock) and makes each call wait on a barrier sized to the expected
    # number of simultaneous callers; the barrier times out after 30 seconds, so a serial
    # implementation fails by a broken barrier error instead of hanging.
    def __init__(self, *, meeting: int) -> None:
        # [FUNCTION DOC] HELPER: Starts with no recorded calls, a lock, and a barrier for the given
        # number of parties.
        self.calls: list[list[str]] = []
        self._lock = threading.Lock()
        self._meeting = threading.Barrier(meeting, timeout=30.0)

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] HELPER: Records the arguments under the lock, waits at the barrier until
        # all parties arrive, then returns a successful process with output yes.
        with self._lock:
            self.calls.append(list(args))
        self._meeting.wait()
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="yes\n", stderr="")


def _verb_checkers(rules: dict[str, tuple[str, ...]]) -> list[ResourceVerbAvailabilityChecker]:
    # [FUNCTION DOC] HELPER: Expands the rules mapping (resource to verbs) into resource and verb
    # pairs and builds one availability checker per pair for the test namespace.
    return [ResourceVerbAvailabilityChecker(NAMESPACE, verb) for verb in expand_resource_verbs(rules)]


def _record(monkeypatch: pytest.MonkeyPatch, *, meeting: int) -> _RecordingKubectl:
    # [FUNCTION DOC] HELPER: Creates a recording fake expecting the given number of simultaneous
    # callers, installs it as the checkers module's raw kubectl runner, and returns it.
    kubectl = _RecordingKubectl(meeting=meeting)
    monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(kubectl))
    return kubectl


class TestAPhaseIsAskedConcurrently:
    # [CLASS DOC] Tests of the parallel checker execution function, which takes a list of checkers
    # and returns the outcomes (each with its checker and result).
    def test_a_phase_is_asked_in_one_parallel_pass(self, monkeypatch):
        # [FUNCTION DOC] With a barrier sized to the number of checkers, running them all
        # succeeds only if every call is in flight at once; every outcome's status is pass.
        """A plan runs to several hundred rules, and asking them in turn makes the wait a round trip
        each; the barrier only opens once that many callers are in flight together."""
        checkers = _verb_checkers(_RULES)
        _record(monkeypatch, meeting=len(checkers))

        outcomes = parallel_execute_checkers(checkers)

        assert [outcome.result.status for outcome in outcomes] == [Status.PASS] * len(checkers)

    def test_a_phase_of_n_checkers_makes_exactly_one_round_of_calls(self, monkeypatch):
        # [FUNCTION DOC] The number of kubectl calls and the number of outcomes both equal the
        # number of checkers.
        """Nothing is remembered between checkers, so a phase must be one round: N checkers, N calls, all in flight."""
        checkers = _verb_checkers(_RULES)
        kubectl = _record(monkeypatch, meeting=len(checkers))

        outcomes = parallel_execute_checkers(checkers)

        assert len(kubectl.calls) == len(checkers)
        assert len(outcomes) == len(checkers)

    def test_every_rule_is_still_asked_of_the_cluster(self, monkeypatch):
        # [FUNCTION DOC] The set of (verb, resource) pairs found at positions 2 and 3 of the
        # recorded kubectl calls equals the set built from the rules, with subresource names cut at
        # the slash (so the pod exec rule is asked as pods).
        """Answering in parallel must not drop or merge a rule: each verb needs its own answer."""
        checkers = _verb_checkers(_RULES)
        kubectl = _record(monkeypatch, meeting=len(checkers))

        parallel_execute_checkers(checkers)

        asked = {(call[2], call[3]) for call in kubectl.calls}
        assert asked == {(verb, resource.partition("/")[0]) for resource, verbs in _RULES.items() for verb in verbs}

    def test_a_subresource_is_asked_for_by_name(self, monkeypatch):
        # [FUNCTION DOC] A create permission on the pod exec subresource yields a first call that
        # contains the subresource option for exec.
        """kubectl reads pods/exec as the pods resource unless the subresource is named separately."""
        checker = ResourceVerbAvailabilityChecker(NAMESPACE, ResourceVerb(verb="create", resource="pods/exec"))
        kubectl = _record(monkeypatch, meeting=1)

        parallel_execute_checkers([checker])

        assert "--subresource=exec" in kubectl.calls[0]

    def test_an_empty_phase_asks_nothing(self, monkeypatch):
        # [FUNCTION DOC] An empty checker list returns an empty list and makes no kubectl call
        # (the barrier is sized for one party, so any call would be visible).
        """A plan that grants nothing leaves a phase empty, and an empty pool must not be opened for it."""
        kubectl = _record(monkeypatch, meeting=1)

        assert parallel_execute_checkers([]) == []
        assert kubectl.calls == []
