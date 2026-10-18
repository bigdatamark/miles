# [MODULE DOC] Unit tests for the kubectl read helpers of the test utilities: building the label
# selector that scopes reads to one release, and the generic read of objects of a release, whose
# external process runner is replaced by a recorder so that no kubectl is executed. Module
# constants: a sample release name and namespace.
import subprocess
from typing import NamedTuple

import pytest

from miles.utils.test_utils import kubectl_reads
from miles.utils.test_utils.kubectl_reads import (
    KUBECTL_TIMEOUT_SECONDS,
    compute_release_selector,
    read_objects_of_release,
)

_RELEASE = "miles-run-abc123"
_NAMESPACE = "miles-e2e"


class _Call(NamedTuple):
    # [CLASS DOC] HELPER: Record of one intercepted process run: the argument vector and the
    # keyword arguments it was called with.
    argv: list[str]
    kwargs: dict[str, object]


def _record_run_process(monkeypatch: pytest.MonkeyPatch, *, stdout: str) -> list[_Call]:
    # [FUNCTION DOC] HELPER: Replaces the process runner in the kubectl reads module with a fake
    # (reverted at teardown) that appends a call record for every invocation and returns a
    # completed process with exit code 0, the given stdout and empty stderr. Returns the list that
    # accumulates the call records.
    calls: list[_Call] = []

    def fake_run_process(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] HELPER: Records the call and returns the canned successful result.
        calls.append(_Call(argv=argv, kwargs=kwargs))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(kubectl_reads, "run_process", fake_run_process)
    return calls


class TestComputeReleaseSelector:
    # [CLASS DOC] Tests of the label selector builder.
    def test_the_release_alone_selects_by_the_instance_label(self) -> None:
        # [FUNCTION DOC] With only a release, the selector equals the standard instance label
        # equals the release name.
        """Every object a run installs carries this label, so it is what scopes a read to one release."""
        assert compute_release_selector(release=_RELEASE) == f"app.kubernetes.io/instance={_RELEASE}"

    def test_extra_labels_are_appended_after_the_release(self) -> None:
        # [FUNCTION DOC] With two extra labels, the selector is the release label followed by the
        # extra labels in order, all comma separated.
        """Narrowing labels must add to the release scope, never replace it and reach another run."""
        selector = compute_release_selector(release=_RELEASE, extra_labels=["pool=actor", "group-index=3"])

        assert selector == f"app.kubernetes.io/instance={_RELEASE},pool=actor,group-index=3"


class TestReadObjectsOfRelease:
    # [CLASS DOC] Tests of the release read helper using the recorded fake process runner.
    def test_it_builds_the_kubectl_get_argv_from_its_arguments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Reads pods of the sample release with an output format and one extra
        # label; asserts exactly one process call whose argument vector is the get command, the
        # kind, the namespace flag and value, the selector flag and the composed selector, and the
        # output flag and the given output format, in that order.
        """The read is only correct if kind, namespace, selector and output land in the flags kubectl reads."""
        calls = _record_run_process(monkeypatch, stdout="")

        read_objects_of_release(
            kind="pod",
            release=_RELEASE,
            namespace=_NAMESPACE,
            output="jsonpath={.items[*].metadata.name}",
            extra_labels=["pool=actor"],
        )

        assert [call.argv for call in calls] == [
            [
                "kubectl",
                "get",
                "pod",
                "--namespace",
                _NAMESPACE,
                "--selector",
                f"app.kubernetes.io/instance={_RELEASE},pool=actor",
                "--output",
                "jsonpath={.items[*].metadata.name}",
            ]
        ]

    def test_it_returns_the_stdout_of_the_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] The fake runner returns a fixed stdout; asserts the helper returns that
        # text unchanged.
        """Callers parse the object list out of stdout, so anything else would silently read nothing."""
        _record_run_process(monkeypatch, stdout="actor-3-0 actor-3-1")

        assert (
            read_objects_of_release(kind="pod", release=_RELEASE, namespace=_NAMESPACE, output="name")
            == "actor-3-0 actor-3-1"
        )

    def test_the_read_is_captured_checked_and_bounded_by_the_kubectl_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Asserts the single process call received exactly the keyword arguments
        # capture output enabled, check enabled (raise on failure), and the timeout constant.
        """An unbounded or unchecked kubectl would let a hung or failed read look like an empty cluster."""
        calls = _record_run_process(monkeypatch, stdout="")

        read_objects_of_release(kind="pod", release=_RELEASE, namespace=_NAMESPACE, output="name")

        assert [call.kwargs for call in calls] == [
            dict(capture_output=True, check=True, timeout=KUBECTL_TIMEOUT_SECONDS)
        ]
