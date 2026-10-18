# [MODULE DOC]
# Tests for the pod manipulation helpers of the ft fault-injection support code: computing the
# label selector for a cell's pods, deleting exactly one random pod of a cell within a release, and
# sending a kill signal to processes inside a pod via an exec call. The subprocess runner used by
# both the pod manipulation module and the kubectl read helpers is replaced with fakes that emulate
# kubectl get (returning space-separated pod names for the release in the selector) and record
# every command; no real cluster is touched. Module constants fix a cell id, two releases, a
# namespace and the pods each release holds.
import random
import subprocess

import pytest
from tests.e2e.ft.conftest_ft.fault_injection import pod_manipulation

from miles.utils.test_utils import kubectl_reads

_CELL_ID = "actor-3"
_RELEASE = "miles-run-abc123"
_OTHER_RELEASE = "miles-run-def456"
_NAMESPACE = "miles-e2e"

_PODS_OF_RELEASE: dict[str, str] = {
    _RELEASE: "actor-3-0 actor-3-1",
    _OTHER_RELEASE: "other-actor-3-0 other-actor-3-1",
}


def _completed(stdout: str = "", *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] HELPER: Builds a finished-process result with empty args and the given stdout,
    # return code (default 0) and stderr.
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _pods_matching(selector: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the space-separated pod names of the first release whose
    # instance label equals that release in the selector text; empty string if none matches.
    for release, pods in _PODS_OF_RELEASE.items():
        if f"{kubectl_reads.INSTANCE_LABEL}={release}" in selector:
            return pods
    return ""


def _fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] HELPER: Fake process runner. For any kubectl subcommand other than get it
    # returns an empty success. For get it reads the value after the selector flag and returns the
    # pods of the matching release.
    if argv[1] != "get":
        return _completed("")
    return _completed(_pods_matching(argv[argv.index("--selector") + 1]))


def _patch_run_process(monkeypatch, fake) -> None:
    # [FUNCTION DOC] HELPER: Replaces the process runner in both the pod manipulation module and
    # the kubectl reads module with the given fake.
    monkeypatch.setattr(pod_manipulation, "run_process", fake)
    monkeypatch.setattr(kubectl_reads, "run_process", fake)


class TestComputeCellPodSelector:
    # [CLASS DOC] Tests for building the label selector of a cell's pods.
    def test_the_selector_names_the_release_the_pool_and_the_cell_index(self) -> None:
        # [FUNCTION DOC] Builds the release selector with the cell labels for actor-3 and asserts it
        # contains the instance label for the release, the pool label for actor, and group index 3.
        """Deleting by cell id needs the labels the run's pods actually carry, not a name guess."""
        selector = kubectl_reads.compute_release_selector(
            release=_RELEASE, extra_labels=pod_manipulation._compute_cell_labels(_CELL_ID)
        )

        assert f"app.kubernetes.io/instance={_RELEASE}" in selector
        assert "miles.radixark.io/pool=actor" in selector
        assert "group-index=3" in selector


class TestDeleteOnePodOfCell:
    # [CLASS DOC] Tests for deleting one random pod of a cell, with a faked process runner.
    def test_only_one_of_the_cells_pods_is_deleted(self, monkeypatch) -> None:
        # [FUNCTION DOC] Records every kubectl call, deletes one pod of the cell with a seeded
        # generator, and asserts the returned pod is one of the two cell pods and that exactly one
        # delete call was made: delete pod in the namespace without waiting, for that pod.
        """An outsider kills a pod, not the whole cell: deleting them all would mimic the heal path."""
        calls: list[list[str]] = []

        def fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Records the command then defers to the shared fake runner.
            calls.append(argv)
            return _fake_run_process(argv, **kwargs)

        _patch_run_process(monkeypatch, fake_run_process)

        deleted = pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert deleted in ("actor-3-0", "actor-3-1")
        assert [call for call in calls if call[1] == "delete"] == [
            ["kubectl", "delete", "pod", "--namespace", _NAMESPACE, "--wait=false", deleted]
        ]

    def test_a_second_release_with_the_same_topology_is_never_touched(self, monkeypatch) -> None:
        # [FUNCTION DOC] With two releases each holding a pod set for the same cell id, deletes a
        # pod for the first release and asserts the result belongs to it and not to the other.
        """Regression: a pool-and-index-only selector deletes another run's pod and injects nothing here."""
        _patch_run_process(monkeypatch, _fake_run_process)

        deleted = pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert deleted in _PODS_OF_RELEASE[_RELEASE].split()
        assert deleted not in _PODS_OF_RELEASE[_OTHER_RELEASE].split()

    def test_a_kubectl_call_that_never_returns_cannot_outlive_the_soak(self, monkeypatch) -> None:
        # [FUNCTION DOC] Records the timeout argument of every runner call during a deletion and
        # asserts at least one call happened and every timeout is a floating point number (bounded).
        """An unbounded kubectl would let the injector thread survive the join and race the witness."""
        timeouts: list[object] = []

        def fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Records the timeout keyword; returns the release's pods for a
            # get call and an empty success otherwise.
            timeouts.append(kwargs.get("timeout"))
            return _completed(_PODS_OF_RELEASE[_RELEASE] if argv[1] == "get" else "")

        _patch_run_process(monkeypatch, fake_run_process)

        pod_manipulation.delete_one_pod_of_cell(
            namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
        )

        assert timeouts and all(isinstance(timeout, float) for timeout in timeouts), timeouts

    def test_a_cell_without_pods_fails_loudly(self, monkeypatch) -> None:
        # [FUNCTION DOC] Makes every runner call return empty output; asserts the deletion raises an
        # assertion error saying there is nothing to delete.
        """Silently injecting nothing would let the soak pass while proving nothing."""
        _patch_run_process(monkeypatch, lambda argv, **kwargs: _completed(""))

        with pytest.raises(AssertionError, match="nothing to delete"):
            pod_manipulation.delete_one_pod_of_cell(
                namespace=_NAMESPACE, release=_RELEASE, cell_id=_CELL_ID, rng=random.Random(0)
            )


_POD_NAME = "rollout-engine-0-0"
_PATTERN = "sglang::"


def _fake_kubectl(monkeypatch: pytest.MonkeyPatch, respond) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Installs a process runner fake that records each command and answers
    # with the result of the respond callback applied to it. Returns the live list of recorded commands.
    calls: list[list[str]] = []

    def fake_run_process(argv, *, capture_output, check, input=None, timeout=None):
        # [FUNCTION DOC] HELPER: Records the command and returns the response callback's result.
        calls.append(argv)
        return respond(argv)

    _patch_run_process(monkeypatch, fake_run_process)
    return calls


class TestSigkillProcessPatternsInPod:
    # [CLASS DOC] Tests for killing processes matching a pattern inside a pod's container.
    def test_it_signals_a_process_inside_the_pod_rather_than_deleting_it(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Runs the kill against a fake that always succeeds. Asserts the first command
        # starts with kubectl exec, contains the process pattern, and has no delete word.
        """Deleting the pod is a different fault: this one leaves the pod and crashes what runs in it."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed())

        pod_manipulation.sigkill_process_patterns_in_pod(
            namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
        )

        assert calls[0][:2] == ["kubectl", "exec"]
        assert _PATTERN in calls[0]
        assert "delete" not in calls[0]

    def test_it_names_the_container_it_reaches_into(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Asserts the value after the container flag of the first command is the
        # requested container name.
        """A pod can hold sidecars, and killing a process in the wrong one is not the fault under test."""
        calls = _fake_kubectl(monkeypatch, lambda argv: _completed())

        pod_manipulation.sigkill_process_patterns_in_pod(
            namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
        )

        assert calls[0][calls[0].index("--container") + 1] == "engine"

    def test_matching_no_process_is_a_failure_rather_than_a_crash_nobody_caused(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Makes the exec return code 1 (nothing matched); asserts an assertion error
        # whose message says no process matched.
        """pkill exits 1 when it matched nothing, which is indistinguishable from a kill that worked."""
        _fake_kubectl(monkeypatch, lambda argv: _completed(returncode=1))

        with pytest.raises(AssertionError, match="No process matching"):
            pod_manipulation.sigkill_process_patterns_in_pod(
                namespace=_NAMESPACE, pod_name=_POD_NAME, container="engine", process_pattern=_PATTERN
            )
