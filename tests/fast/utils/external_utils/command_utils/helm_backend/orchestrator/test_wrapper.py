# [MODULE DOC] Tests for the in-pod orchestrator wrapper's main function, which runs the
# orchestration script as a subprocess while publishing its progress in a state file and, when
# given an uninstall manifest, creating a cluster job that uninstalls the release once the run
# has a verdict. Verified behaviors: state published as started then exited with the script's exit
# code (127 when the script cannot be launched, 128 plus signal number for signals, 1 when an
# unexpected error escapes or a previous first attempt left only a started state), a SIGTERM to
# the pod produces a verdict and a SystemExit and never creates the uninstall job, a recorded
# verdict is not re-run and is kept through teardown (a terminal state without an exit code is
# treated as corrupt and re-run), the wrapper stays alive after the script (the keep-alive step),
# a missing command is an assertion failure, the state file location, and the uninstall job
# protocol (create at once on any run end, retry with the configured sleeps after refusal or
# exception, replace a finished job left by a previous launch, accept an existing unfinished job,
# stop when superseded by a newer generation marker, skip when no manifest was given). Two
# autouse fixtures disable the keep-alive wait and replace sleeping with a recorder.
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import pytest

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.naming import RunFiles, _orchestrator_state_path
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import state as orchestrator_state
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import wrapper as orchestrator_wrapper

MANIFEST = "/etc/miles-uninstall/uninstall-job.yaml"


def _state(path):
    # [FUNCTION DOC] HELPER: Reads the file at the path and returns its json decoded mapping.
    return json.loads(path.read_text())


@pytest.fixture(autouse=True)
def _no_keep_alive(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture; replaces the wrapper's keep-alive wait with a no-op
    # so main returns right after the run for every test (tests may override it).
    monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", lambda: None)


@pytest.fixture(autouse=True)
def slept(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture; replaces the wrapper's sleep with a recorder and
    # returns the (live) list of requested durations so tests can assert on retry waits.
    waits = []
    monkeypatch.setattr(orchestrator_wrapper, "sleep", lambda seconds: waits.append(seconds))
    return waits


@pytest.fixture
def kubectl_calls(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture; replaces the kubectl wrapper's low level runner with a fake
    # that records each argument list and always succeeds with empty output, and returns the
    # (live) list of recorded argument lists.
    calls = []

    def fake_run(arguments, **kwargs):
        # [FUNCTION DOC] HELPER: Records the arguments as a list and returns a successful
        # completed process with empty output.
        calls.append(list(arguments))
        return subprocess.CompletedProcess(args=list(arguments), returncode=0, stdout="", stderr="")

    monkeypatch.setattr(Kubectl, "_run", staticmethod(fake_run))
    return calls


def _refusal(stderr: str) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Returns a failed completed process (exit code 1, empty output) with
    # the given error text, imitating a refused kubectl call.
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


class TestMain:
    # [CLASS DOC] Tests of the wrapper main function that mostly run a real python subprocess as
    # the orchestration script (the current interpreter), with a state file in tmp_path. Main takes
    # an argument list and returns the exit code to use.
    def test_an_unexpected_runner_exception_publishes_failure_before_reraising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kubectl_calls: list[list[str]]
    ) -> None:
        # [FUNCTION DOC] The subprocess launcher is replaced by one raising a runtime error. Main
        # must re-raise that same error object, leave an exited state with exit code 1, and have
        # created the uninstall job exactly once (one kubectl create call with the manifest).
        """An unexpected launch error records failure and arms cleanup before escaping."""
        state_file = tmp_path / "orchestrator.state"
        error = RuntimeError("runner failed")

        def fail_to_launch(command: list[str], *, envs: dict[str, str]) -> NoReturn:
            # [FUNCTION DOC] HELPER: Fake launcher that always raises the prepared error.
            raise error

        monkeypatch.setattr(orchestrator_wrapper, "launch_bound_subprocess", fail_to_launch)

        with pytest.raises(RuntimeError) as raised:
            orchestrator_wrapper.main(
                ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", "unused"]
            )

        assert raised.value is error
        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 1)
        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_publishes_a_successful_exit_code(self, tmp_path):
        # [FUNCTION DOC] A script that does nothing: main returns 0 and the state file is exited
        # with exit code 0.
        """The launcher reads this file to learn the run finished, so a clean run must record zero."""
        state_file = tmp_path / "orchestrator.state"

        code = orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert code == 0
        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 0)

    def test_publishes_a_failing_exit_code(self, tmp_path):
        # [FUNCTION DOC] A script exiting with 7: main returns 7 and the state records 7.
        """A failed experiment has to reach the caller's shell, not just the pod logs."""
        state_file = tmp_path / "orchestrator.state"

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(7)"]
        )

        assert code == 7
        assert _state(state_file)["exit_code"] == 7

    def test_marks_the_run_started_before_the_script_runs(self, tmp_path):
        # [FUNCTION DOC] The script copies the state file to another path while running; that
        # copy has status started, while the final state file is exited.
        """Otherwise a launcher polling an absent file cannot tell a slow start from a lost run."""
        state_file = tmp_path / "orchestrator.state"
        observed = tmp_path / "observed"
        script = f"import shutil; shutil.copyfile({str(state_file)!r}, {str(observed)!r})"

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", script])

        assert json.loads(observed.read_text())["status"] == "started"
        assert _state(state_file)["status"] == "exited"

    def test_records_a_verdict_when_the_script_cannot_be_launched(self, tmp_path):
        # [FUNCTION DOC] The command is a nonexistent executable: state is exited with code 127.
        """A wrapper that died silently would leave the launcher polling a started run forever."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", str(tmp_path / "not-an-executable")])

        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 127)

    def test_publishes_a_verdict_when_the_pod_is_asked_to_stop(self, tmp_path):
        # [FUNCTION DOC] The script sends SIGTERM to its parent (the wrapper) and sleeps. Main
        # raises a system exit and the state is exited with code 128 plus the SIGTERM number.
        """A signal kills python without raising, and a restart would then silently re-run the training."""
        state_file = tmp_path / "orchestrator.state"
        script = "import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"

        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", script])

        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 128 + signal.SIGTERM)

    def test_keeps_the_verdict_when_the_pod_is_torn_down_afterwards(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] The script exits 7; the keep-alive step is replaced by one sending SIGTERM
        # to the current process. Main raises a system exit, but the state still records 7.
        """A helm uninstall during keep-alive would otherwise overwrite the run's real exit code with 143."""
        state_file = tmp_path / "orchestrator.state"

        def stop_pod():
            # [FUNCTION DOC] HELPER: Fake keep-alive: sends SIGTERM to the current process.
            os.kill(os.getpid(), signal.SIGTERM)

        monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", stop_pod)
        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(
                ["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(7)"]
            )

        assert _state(state_file)["exit_code"] == 7

    def test_does_not_run_a_script_whose_verdict_is_already_recorded(self, tmp_path):
        # [FUNCTION DOC] After a first run exits with 4, a second main call with a script that
        # would create a marker file returns 4 and the marker file does not exist (script skipped).
        """The pod restarts on its own, and rerunning training that already finished would lose its result."""
        state_file = tmp_path / "orchestrator.state"
        ran = tmp_path / "ran"
        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(4)"])

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", f"open({str(ran)!r}, 'w').close()"]
        )

        assert code == 4
        assert not ran.exists()

    def test_keeps_an_inherited_verdict_when_the_restarted_pod_is_torn_down(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] After a first run exits with 4, a second main call whose keep-alive sends
        # SIGTERM to the process raises a system exit; the state still records exit code 4.
        """A pod rebuilt after its run finished skips the script, and its teardown must not rewrite the result."""
        state_file = tmp_path / "orchestrator.state"
        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(4)"])

        monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", lambda: os.kill(os.getpid(), signal.SIGTERM))
        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert _state(state_file)["exit_code"] == 4

    def test_reruns_a_script_whose_recorded_verdict_carries_no_exit_code(self, tmp_path):
        # [FUNCTION DOC] The state file is pre-written as exited with a null exit code (corrupt).
        # Main runs the script (exit 5), returns 5 and the state records 5.
        """A terminal state with no code is a corrupt file, and returning it would exit the pod as a success."""
        state_file = tmp_path / "orchestrator.state"
        state_file.write_text(json.dumps({"status": "exited", "exit_code": None, "timestamp": 0.0}))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(5)"]
        )

        assert code == 5
        assert _state(state_file)["exit_code"] == 5

    def test_reports_a_signalled_script_as_a_shell_exit_code(self, tmp_path):
        # [FUNCTION DOC] A script that kills itself with SIGKILL: main returns 128 plus the SIGKILL
        # number and the state records the same value.
        """A negative return code is not an exit status, and the launcher passes this straight to its caller."""
        state_file = tmp_path / "orchestrator.state"
        script = "import os, signal; os.kill(os.getpid(), signal.SIGKILL)"

        code = orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", script])

        assert code == 128 + signal.SIGKILL
        assert _state(state_file)["exit_code"] == 128 + signal.SIGKILL

    def test_stays_alive_after_the_script_finishes(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] A recording keep-alive replacement is called exactly once after a
        # successful script.
        """The pod's logs die with its process, so exiting would take the run's own output with it."""
        state_file = tmp_path / "orchestrator.state"
        stayed = []
        monkeypatch.setattr(orchestrator_wrapper, "_keep_alive", lambda: stayed.append(True))

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert stayed == [True]

    def test_requires_a_command(self, tmp_path):
        # [FUNCTION DOC] Main given only the state file option raises an assertion error.
        """A wrapper with nothing to run is a chart bug, and must fail loudly at startup."""
        with pytest.raises(AssertionError):
            orchestrator_wrapper.main(["--state-file", str(tmp_path / "orchestrator.state")])

    def test_ignores_a_started_verdict_of_its_own_first_attempt(self, tmp_path):
        # [FUNCTION DOC] The state file already holds a started state (a previous attempt died).
        # Main does not run the script (marker file absent), returns 1 and records exited with 1.
        """Nothing else distinguishes a restarted pod from a first start, so the exit file has to."""
        state_file = tmp_path / "orchestrator.state"
        ran = tmp_path / "ran"
        orchestrator_state.OrchestratorState(status=orchestrator_state.OrchestratorStatus.STARTED).write(state_file)

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--", sys.executable, "-c", f"open({str(ran)!r}, 'w').close()"]
        )

        assert code == 1
        assert not ran.exists()
        assert (_state(state_file)["status"], _state(state_file)["exit_code"]) == ("exited", 1)

    def test_writes_where_the_run_directory_layout_says(self, tmp_path):
        # [FUNCTION DOC] With the state file path computed from the run directory layout helpers
        # (nested directories not yet existing), main creates the file there.
        """The chart passes a path derived from the shared root, so both sides agree without extra config."""
        state_file = _orchestrator_state_path(
            RunFiles.run_dir(shared_root=tmp_path, run_id="260101-000000-000"), "260101-000000-000001"
        )

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert state_file.is_file()


class TestUninstallJob:
    # [CLASS DOC] Tests of the uninstall job creation protocol, run with an uninstall manifest
    # path option. The cluster is faked either at the kubectl runner level (recording argument
    # lists, or scripting answers popped in order) or at the create-if-absent level.
    def test_stops_retrying_when_a_new_generation_takes_over(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Job creation always raises a runtime error, and the (replaced) sleep
        # drops the superseded marker for this state file. Main returns 0 and creation was
        # attempted exactly once, so the takeover stopped further retries.
        """A takeover during retry backoff prevents recreating the uninstall job."""
        state_file = tmp_path / "orchestrator.state"
        attempts: list[str] = []

        def fail_to_create(manifest_path: str) -> bool:
            # [FUNCTION DOC] HELPER: Fake create-if-absent: records the manifest path and raises a
            # runtime error saying the apiserver is unavailable.
            attempts.append(manifest_path)
            raise RuntimeError("apiserver unavailable")

        def take_over(seconds: int) -> None:
            # [FUNCTION DOC] HELPER: Fake sleep: writes the superseded marker file for the state
            # file, imitating a newer launch taking over.
            RunFiles.superseded_marker(state_file=state_file).write_text("new generation\n")

        monkeypatch.setattr(Kubectl, "create_if_absent", staticmethod(fail_to_create))
        monkeypatch.setattr(orchestrator_wrapper, "sleep", take_over)

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 0
        assert attempts == [MANIFEST]

    def test_retries_when_creating_the_uninstall_job_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slept: list[int]
    ) -> None:
        # [FUNCTION DOC] Job creation raises on the first attempt and returns true on the second.
        # Main returns 0, the state records 0, two attempts were made with the manifest, and the
        # recorded sleeps are only the first configured retry sleep.
        """A transient create exception is retried without changing the run verdict."""
        state_file = tmp_path / "orchestrator.state"
        attempts: list[str] = []

        def create_after_exception(manifest_path: str) -> bool:
            # [FUNCTION DOC] HELPER: Fake create-if-absent: records the path, raises a runtime
            # error on the first call and returns true afterwards.
            attempts.append(manifest_path)
            if len(attempts) == 1:
                raise RuntimeError("apiserver unavailable")
            return True

        monkeypatch.setattr(Kubectl, "create_if_absent", staticmethod(create_after_exception))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 0
        assert _state(state_file)["exit_code"] == 0
        assert attempts == [MANIFEST, MANIFEST]
        assert slept == [orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS[0]]

    def test_creates_the_job_as_soon_as_the_run_has_a_verdict(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] After a successful script, the only kubectl call is a create with the
        # manifest.
        """The wrapper could die during the grace period, so the job carries the delay and is created at once."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_creates_the_job_for_a_failed_run_too(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] A script exiting with 7 still leads to exactly one create call.
        """A failed run holds just as many gpus as a successful one, and its logs stay in the run directory."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(
            [
                "--state-file",
                str(state_file),
                "--uninstall-manifest",
                MANIFEST,
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ]
        )

        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_creates_the_job_when_the_script_could_not_be_launched(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] A nonexistent executable still leads to exactly one create call.
        """That verdict ends the run as surely as a failing script, and the release must go the same way."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(
            [
                "--state-file",
                str(state_file),
                "--uninstall-manifest",
                MANIFEST,
                "--",
                str(tmp_path / "not-an-executable"),
            ]
        )

        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_never_creates_the_job_when_the_pod_is_asked_to_stop(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] The script SIGTERMs the wrapper: main raises a system exit and no
        # kubectl call was made.
        """A relaunch SIGTERMs the orchestrator it replaces, and uninstalling then would tear down the new run."""
        state_file = tmp_path / "orchestrator.state"
        script = "import os, signal, time; os.kill(os.getppid(), signal.SIGTERM); time.sleep(30)"

        with pytest.raises(SystemExit):
            orchestrator_wrapper.main(
                ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", script]
            )

        assert kubectl_calls == []

    def test_creates_the_job_a_restarted_pod_found_no_trace_of(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] A first run (no manifest) exits with 4 and makes no kubectl call of its
        # own; a second call with the manifest returns the inherited 4 and then creates the job once.
        """The wrapper may have written its verdict and then died before creating the job."""
        state_file = tmp_path / "orchestrator.state"
        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "raise SystemExit(4)"])

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 4
        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_creates_the_job_for_a_run_a_restart_cut_short(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] A pre-existing started state: main returns 1 without running the script
        # and creates the job once.
        """A pod that restarted mid-training reports a failure, and that failure ends the release too."""
        state_file = tmp_path / "orchestrator.state"
        orchestrator_state.OrchestratorState(status=orchestrator_state.OrchestratorStatus.STARTED).write(state_file)

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 1
        assert kubectl_calls == [["create", "-f", MANIFEST]]

    def test_replaces_the_finished_job_a_previous_launch_of_this_release_left(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Scripted kubectl answers in order: create refused as already existing, a
        # status query printing a succeeded count of 1 (job finished), then a success for replace.
        # Main returns 0, all answers were consumed, and the last call is a forced replace with
        # the manifest.
        """A completed job holds the name without ever running again, so the release would never be uninstalled."""
        state_file = tmp_path / "orchestrator.state"
        answers = [
            _refusal('Error from server (AlreadyExists): jobs "u" already exists'),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="1,", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        calls: list[list[str]] = []

        def fake_run(arguments, **kwargs):
            # [FUNCTION DOC] HELPER: Fake kubectl runner: records the argument list and returns
            # the next scripted answer (raises an index error if exhausted).
            calls.append(list(arguments))
            return answers.pop(0)

        monkeypatch.setattr(Kubectl, "_run", staticmethod(fake_run))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert (code, answers) == (0, [])
        assert calls[-1] == ["replace", "--force", "-f", MANIFEST]

    def test_treats_a_job_someone_already_created_as_success(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Scripted answers: create refused as already existing, then a status query
        # with an empty (unfinished) result. Main returns 0 and both answers were consumed (no
        # replace, no retry).
        """A second attempt is normal after a restart, and it must not turn a finished run into a failure."""
        state_file = tmp_path / "orchestrator.state"
        answers = [
            _refusal('Error from server (AlreadyExists): jobs "u" already exists'),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=",", stderr=""),
        ]
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: answers.pop(0)))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert (code, answers) == (0, [])

    def test_retries_when_the_cluster_cannot_say_whether_the_job_it_holds_has_finished(
        self, tmp_path, monkeypatch, slept
    ):
        # [FUNCTION DOC] Scripted answers: already exists, then the status query fails with a
        # timeout (so the attempt is retried after the first configured sleep), already exists
        # again, a finished status of 1, then replace succeeds. Main returns 0, all answers are
        # consumed, exactly the first retry sleep was recorded, and the last call is a forced
        # replace.
        """Reading that as unfinished would adopt a job that already ran and leave the release installed."""
        state_file = tmp_path / "orchestrator.state"
        answers = [
            _refusal('Error from server (AlreadyExists): jobs "u" already exists'),
            _refusal("Error from server: etcdserver: request timed out"),
            _refusal('Error from server (AlreadyExists): jobs "u" already exists'),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="1,", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
        ]
        calls: list[list[str]] = []

        def fake_run(arguments, **kwargs):
            # [FUNCTION DOC] HELPER: Fake kubectl runner: records the argument list and returns
            # the next scripted answer.
            calls.append(list(arguments))
            return answers.pop(0)

        monkeypatch.setattr(Kubectl, "_run", staticmethod(fake_run))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert (code, answers) == (0, [])
        assert slept == [orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS[0]]
        assert calls[-1] == ["replace", "--force", "-f", MANIFEST]

    def test_keeps_the_run_alive_when_every_attempt_is_refused(self, tmp_path, monkeypatch, slept):
        # [FUNCTION DOC] Every kubectl call is refused as forbidden. Main still returns 0 with the
        # state recording 0, and the recorded sleeps equal the full configured retry sleep list.
        """The run's own outcome is what the launcher waits for; a leaked release is the smaller problem."""
        state_file = tmp_path / "orchestrator.state"
        refusal = _refusal("Error from server: forbidden")
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: refusal))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert code == 0
        assert _state(state_file)["exit_code"] == 0
        assert slept == list(orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS)

    def test_retries_a_create_the_cluster_refused_once(self, tmp_path, monkeypatch, slept):
        # [FUNCTION DOC] Scripted answers: one refusal then one success. Main returns 0, answers
        # consumed, and a single first-retry sleep recorded.
        """An apiserver that was busy for a moment must not cost the release its only cleaner."""
        state_file = tmp_path / "orchestrator.state"
        answers = [_refusal("Error from server: try again"), subprocess.CompletedProcess(args=[], returncode=0)]
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: answers.pop(0)))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert (code, answers) == (0, [])
        assert slept == [orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS[0]]

    def test_stops_retrying_as_soon_as_the_job_is_there(self, tmp_path, monkeypatch, slept):
        # [FUNCTION DOC] Scripted answers: a generic refusal, then already exists, then an
        # unfinished status. Main returns 0, answers consumed, one first-retry sleep recorded.
        """A retry that finds the job someone else created has nothing left to do."""
        state_file = tmp_path / "orchestrator.state"
        answers = [
            _refusal("Error from server: try again"),
            _refusal('Error from server (AlreadyExists): jobs "u" already exists'),
            subprocess.CompletedProcess(args=[], returncode=0, stdout=",", stderr=""),
        ]
        monkeypatch.setattr(Kubectl, "_run", staticmethod(lambda arguments, **kwargs: answers.pop(0)))

        code = orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert (code, answers) == (0, [])
        assert slept == [orchestrator_wrapper._UNINSTALL_JOB_RETRY_SLEEPS[0]]

    def test_touches_no_cluster_when_no_manifest_was_rendered(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] Without the manifest option no kubectl call is made.
        """A cluster with no account for it renders no manifest, and the run must not try to uninstall itself."""
        state_file = tmp_path / "orchestrator.state"

        orchestrator_wrapper.main(["--state-file", str(state_file), "--", sys.executable, "-c", "pass"])

        assert kubectl_calls == []

    def test_leaves_the_uninstall_to_the_launch_that_took_the_run_over(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] A superseded marker naming the new state file exists for this state file;
        # with the manifest option, no kubectl call is made.
        """The release already carries the replacement, so uninstalling it now tears the new run down."""
        state_file = tmp_path / "orchestrator.state"
        RunFiles.superseded_marker(state_file=state_file).write_text(f"{tmp_path / 'orchestrator-new.state'}\n")

        orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert kubectl_calls == []

    def test_a_superseded_generation_still_publishes_its_own_verdict(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] With the superseded marker present and a script exiting 7, main returns 7
        # and the state records 7.
        """Losing the run is not a reason to leave the state file of this generation unwritten."""
        state_file = tmp_path / "orchestrator.state"
        RunFiles.superseded_marker(state_file=state_file).write_text("taken over\n")

        code = orchestrator_wrapper.main(
            [
                "--state-file",
                str(state_file),
                "--uninstall-manifest",
                MANIFEST,
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ]
        )

        assert code == 7
        assert _state(state_file)["exit_code"] == 7

    def test_a_marker_for_another_generation_defuses_nothing(self, tmp_path, kubectl_calls):
        # [FUNCTION DOC] A superseded marker for a different state file does not stop job
        # creation: the first three arguments of each kubectl call equal one create call.
        """Every generation writes its own state file, and only its own marker speaks for it."""
        state_file = tmp_path / "orchestrator.state"
        RunFiles.superseded_marker(state_file=tmp_path / "someone-else.state").write_text("taken over\n")

        orchestrator_wrapper.main(
            ["--state-file", str(state_file), "--uninstall-manifest", MANIFEST, "--", sys.executable, "-c", "pass"]
        )

        assert [call[:3] for call in kubectl_calls] == [["create", "-f", MANIFEST]]
