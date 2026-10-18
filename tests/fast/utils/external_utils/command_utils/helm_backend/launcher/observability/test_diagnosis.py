# [MODULE DOC] Unit tests for the failure diagnosis collector of the Helm launcher observability
# code. The collector (for a namespace and an output directory, optionally a label selector) lists
# the pods, then writes into a new uniquely named directory ("miles-diagnosis-" plus namespace and
# a timestamp with microseconds) one log file, one describe file and one previous-container log
# file per pod plus a file of namespace events, by running kubectl style commands. It returns a
# result with the directory, a tuple of descriptions of missing evidence and a completeness flag
# (complete only when nothing is missing and at least one pod was found). A previous-log request
# failing because the container never restarted is expected and not a gap; any other error is a
# gap and its text is written into the previous-log file. Command execution and pod listing are
# replaced by fakes via monkeypatching; files go to temporary directories.
import re
import subprocess
from datetime import datetime
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import diagnosis

_NEVER_RESTARTED = 'Error from server (BadRequest): previous terminated container "app" in pod "trainer-0" not found'
_API_SERVER_BLINKED = "Error from server: etcdserver: request timed out"


def _collect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    pods: list[str] | None,
    previous_failure: str | None = None,
) -> diagnosis.Diagnosis:
    # [FUNCTION DOC] HELPER: Runs the diagnosis collector for namespace "rl" into the temporary
    # directory with fakes installed. The fake command runner returns exit code 1, empty output
    # and the given previous_failure text as error output for any command containing the
    # "--previous" option when previous_failure is not none, and otherwise exit code 0 with the
    # output "captured" and a newline. The fake pod name lookup returns the given list of pod
    # names (or none to simulate a failed listing). Returns the diagnosis result.
    def run_process(command: list[str], capture_output: bool, check: bool) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: The fake command runner described in the enclosing helper.
        if "--previous" in command and previous_failure is not None:
            return subprocess.CompletedProcess(args=command, returncode=1, stdout="", stderr=previous_failure)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="captured\n", stderr="")

    monkeypatch.setattr(diagnosis, "run_process", run_process)
    monkeypatch.setattr(diagnosis, "_pod_names", lambda *, namespace, selector: pods)
    return diagnosis.collect_diagnosis(namespace="rl", output_dir=tmp_path)


class TestPreviousLogsOfAPod:
    # [CLASS DOC] Tests for how the previous-container log request of each pod is judged and
    # stored.
    def test_a_container_that_never_restarted_leaves_the_diagnosis_complete(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] When the previous-log request fails with the "previous terminated
        # container not found" error, the result is complete and has no missing items.
        """Most pods of a healthy-looking failure never restarted, and that is not evidence going missing."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_NEVER_RESTARTED)

        assert collected.is_complete
        assert collected.missing == ()

    def test_a_container_that_never_restarted_gets_no_previous_log_file(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] In the same situation, no "trainer-0.previous.log" file is created in
        # the result directory.
        """A file holding nothing but the api server's refusal reads as a crash log that says nothing."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_NEVER_RESTARTED)

        assert not (collected.directory / "trainer-0.previous.log").exists()

    def test_an_api_error_is_reported_rather_than_read_as_a_pod_that_never_restarted(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] When the previous-log request fails with a different error (an etcd
        # timeout), the result is not complete and its missing list contains the entry
        # "previous logs of trainer-0".
        """The crash log is the whole point of the diagnosis, and losing it silently is what hid the crash."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_API_SERVER_BLINKED)

        assert not collected.is_complete
        assert "previous logs of trainer-0" in collected.missing

    def test_an_api_error_is_written_down_where_the_crash_log_would_be(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] In that case the "trainer-0.previous.log" file exists and contains the
        # error text.
        """Whoever reads the directory has to be able to see why the file is not the log they wanted."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"], previous_failure=_API_SERVER_BLINKED)

        assert _API_SERVER_BLINKED in (collected.directory / "trainer-0.previous.log").read_text()

    def test_a_collection_that_captured_everything_is_complete(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] When every command succeeds, the result is complete and the previous
        # log file holds the command output "captured" plus newline.
        """A diagnosis reported as incomplete sends its reader looking for evidence that is right there."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert collected.is_complete
        assert (collected.directory / "trainer-0.previous.log").read_text() == "captured\n"


class TestThePodsACollectionCovers:
    # [CLASS DOC] Tests for how the set of pods affects completeness and which per-pod files are
    # written.
    def test_a_run_with_no_pods_left_is_not_a_complete_diagnosis(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] An empty pod list makes the result incomplete, with the missing entry
        # "pods of the run in namespace rl".
        """A directory holding nothing but namespace events answered no question about the failed run."""
        collected = _collect(monkeypatch, tmp_path, pods=[])

        assert not collected.is_complete
        assert "pods of the run in namespace rl" in collected.missing

    def test_a_pod_listing_that_failed_is_reported_as_missing(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] A failed listing (none instead of a list) reports the missing entry
        # "pod listing in namespace rl" and does not also report the "pods of the run" entry.
        """Not knowing which pods exist is a different gap from knowing there are none."""
        collected = _collect(monkeypatch, tmp_path, pods=None)

        assert "pod listing in namespace rl" in collected.missing
        assert "pods of the run in namespace rl" not in collected.missing

    def test_every_pod_of_the_run_is_captured(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] With two pods, both the log file of "trainer-0" and the describe file
        # of "engine-0" exist and the result is complete.
        """A split run fails in one of its deployments, and each pod's log is a candidate answer."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0", "engine-0"])

        assert (collected.directory / "trainer-0.log").exists()
        assert (collected.directory / "engine-0.describe.txt").exists()
        assert collected.is_complete


class TestTheDirectoryACollectionGetsForItself:
    # [CLASS DOC] Tests for the naming and creation of the per-collection directory inside the
    # output directory.
    def test_two_collections_of_one_second_do_not_share_a_directory(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Two consecutive collections (likely within the same second) produce
        # different directories.
        """A relaunch diagnoses twice within a second, and one directory means one set of files."""
        first = _collect(monkeypatch, tmp_path, pods=["trainer-0"])
        second = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert first.directory != second.directory

    def test_the_name_carries_the_microsecond_the_collection_started_at(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] The directory name matches "miles-diagnosis-rl-" plus 8 digits, a
        # dash, 6 digits, a dash and 6 digits (date, time, microseconds).
        """Second resolution is what let two collections land in one directory in the first place."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert re.fullmatch(r"miles-diagnosis-rl-\d{8}-\d{6}-\d{6}", collected.directory.name)

    def test_a_collection_into_a_directory_that_exists_fails_loudly(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] Freezes the module's clock to a fixed time (2026-01-01 00:00:00 plus 5
        # microseconds) so two collections want the same directory name. The second raises a
        # file-exists error instead of reusing the directory.
        """Silently reusing it interleaves two runs' evidence, which is worse than collecting nothing."""
        monkeypatch.setattr(diagnosis, "datetime", SimpleNamespace(now=lambda: datetime(2026, 1, 1, 0, 0, 0, 5)))

        _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        with pytest.raises(FileExistsError):
            _collect(monkeypatch, tmp_path, pods=["trainer-0"])

    def test_the_name_says_which_namespace_was_diagnosed(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] The directory name starts with "miles-diagnosis-rl-" and the directory
        # is a direct child of the output directory.
        """One output directory holds the diagnoses of every run a user launched from this machine."""
        collected = _collect(monkeypatch, tmp_path, pods=["trainer-0"])

        assert collected.directory.name.startswith("miles-diagnosis-rl-")
        assert collected.directory.parent == tmp_path


class TestCollectDiagnosis:
    # [CLASS DOC] A fuller test of the collector with the real pod listing path (through the
    # command wrapper) and fake command output.
    def test_collection_preserves_outputs_and_reports_a_failed_pod_description(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Fakes the command wrapper's process runner (used for the pod listing) to
        # record commands and return a JSON list with one pod "trainer-0" (uid "trainer-uid"), and
        # the diagnosis module's runner so that the describe pod command fails (exit 1, with
        # "describe stdout" and "describe stderr") while event and log commands succeed with
        # "<artifact> stdout" and "<artifact> stderr" lines. Runs the collector for namespace
        # "training" with selector "run=selected". Asserts: exactly one listing command was run,
        # with the pods get arguments, namespace, JSON output, ignore-not-found and the selector;
        # the events file holds the events stdout and stderr text concatenated; the pod log file
        # holds the logs stdout and stderr; the describe file holds the failing command's stdout
        # and stderr (output is kept even on failure); and the missing list is exactly the single
        # entry "describe of trainer-0".
        """Collection keeps command output and reports a failed describe without losing other artifacts."""
        kubectl_commands: list[list[str]] = []

        def fake_kubectl_process(
            command: list[str], *, capture_output: bool, check: bool, input: str | None, timeout: float | None
        ) -> CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake kubectl runner for the pod listing: records the command
            # and returns exit 0 with a JSON document listing the single pod "trainer-0" with its
            # uid, and empty error output.
            kubectl_commands.append(command)
            return CompletedProcess(
                command,
                returncode=0,
                stdout='{"items":[{"metadata":{"name":"trainer-0","uid":"trainer-uid"}}]}',
                stderr="",
            )

        def fake_run_process(command: list[str], *, capture_output: bool, check: bool) -> CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake runner for the artifact commands: a describe pod command
            # (second and third words "describe" and "pod") returns exit 1 with describe output;
            # any other command returns exit 0 with "events stdout/stderr" lines for a get events
            # command and "logs stdout/stderr" lines otherwise.
            if command[1:3] == ["describe", "pod"]:
                return CompletedProcess(command, returncode=1, stdout="describe stdout\n", stderr="describe stderr\n")
            artifact = "events" if command[1:3] == ["get", "events"] else "logs"
            return CompletedProcess(
                command,
                returncode=0,
                stdout=f"{artifact} stdout\n",
                stderr=f"{artifact} stderr\n",
            )

        monkeypatch.setattr(command_wrapper, "run_process", fake_kubectl_process)
        monkeypatch.setattr(diagnosis, "run_process", fake_run_process)

        result = diagnosis.collect_diagnosis(namespace="training", output_dir=tmp_path, selector="run=selected")

        assert kubectl_commands == [
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
                "run=selected",
            ]
        ]
        assert (result.directory / "events.txt").read_text() == "events stdout\nevents stderr\n"
        assert (result.directory / "trainer-0.log").read_text() == "logs stdout\nlogs stderr\n"
        assert (result.directory / "trainer-0.describe.txt").read_text() == "describe stdout\ndescribe stderr\n"
        assert result.missing == ("describe of trainer-0",)
