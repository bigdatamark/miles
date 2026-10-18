# [MODULE DOC] Tests for the command actor, which launches a shell command as a bound subprocess
# and ties its own process lifetime to it: the actor process exits (through the process exit call,
# replaced here by a recorder) with the subprocess's exit code once it ends, using 128 plus the
# signal number for a signal-killed subprocess; a deliberate shutdown terminates the subprocess
# without the crash exit; kill-subprocess simulates a crash; fault injection supports only a kill of
# the whole process tree; and the node ip reporting honours the host ip override. Most tests spawn
# real shell subprocesses and use waits of up to ten seconds.
import os
import shlex
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from miles.utils.http_utils import MILES_HOST_IP_ENV
from miles.utils.misc import get_current_node_ip
from miles.utils.test_utils import fault_injector
from miles.utils.workers import process_utils
from miles.utils.workers.command_actor import CommandActor

_DRAINING_SOURCE = """
import os, signal, sys, time

def drain(*_):
    time.sleep(1)
    with open(sys.argv[2], "w") as f:
        f.write("drained")
    sys.exit(0)

signal.signal(signal.SIGTERM, drain)
os.close(os.open(sys.argv[1], os.O_CREAT | os.O_WRONLY))
time.sleep(300)
"""


class _FakeExit:
    # [CLASS DOC] HELPER: Replaces the process exit function with a recorder for the test's duration:
    # exit codes are collected in a list and a thread event is set on each exit call, so the test
    # process is not killed.
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] HELPER: Creates the event and code list and patches the process exit function
        # of the os module with this object's recorder (undone by the monkeypatch at test end).
        self.event = threading.Event()
        self.codes: list[int] = []
        monkeypatch.setattr(os, "_exit", self._exit)

    def _exit(self, code: int) -> None:
        # [FUNCTION DOC] HELPER: Appends the code and sets the event; does not exit.
        self.codes.append(code)
        self.event.set()

    def wait(self) -> None:
        # [FUNCTION DOC] HELPER: Waits up to 10 seconds for an exit call, asserting that one happened.
        assert self.event.wait(timeout=10)


class TestRun:
    # [CLASS DOC] Tests the run method, which launches the command in a subprocess and a watcher thread.
    def test_launches_subprocess_with_envs_merged_over_parent_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # [FUNCTION DOC] Sets two parent variables, then runs a shell command that prints three
        # variables (one given only in envs, one only in the parent, one in both) to a file. After the
        # fake exit, asserts the file holds the envs value, the parent value, and the envs value
        # overriding the parent's.
        """The command sees the given envs, inherited parent vars, and envs override same-named parent vars."""
        fake_exit = _FakeExit(monkeypatch)
        monkeypatch.setenv("COMMAND_ACTOR_PARENT_VAR", "parent")
        monkeypatch.setenv("COMMAND_ACTOR_OVERRIDDEN_VAR", "from-parent")
        output_path = tmp_path / "output.txt"

        CommandActor().run(
            cmd=(
                'printf "%s %s %s" "$COMMAND_ACTOR_TEST_VAR" "$COMMAND_ACTOR_PARENT_VAR" '
                f'"$COMMAND_ACTOR_OVERRIDDEN_VAR" > {output_path}'
            ),
            envs={"COMMAND_ACTOR_TEST_VAR": "hello", "COMMAND_ACTOR_OVERRIDDEN_VAR": "from-envs"},
        )
        fake_exit.wait()

        assert output_path.read_text() == "hello parent from-envs"

    def test_rejects_second_run(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A second run on the same actor raises an assertion error; waits for the first
        # command's exit before finishing.
        """Calling run a second time on the same actor is rejected."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="true", envs={})

        with pytest.raises(AssertionError):
            actor.run(cmd="true", envs={})
        fake_exit.wait()

    def test_a_launch_error_propagates_without_consuming_the_actor(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Inside a nested monkeypatch context, the bound-subprocess launcher raises an
        # os error and the thread class is replaced by a recorder. Running raises the os error and no
        # thread is created. After the context ends, a real run of "exit 7" succeeds and ends with exit
        # code 7 (the actor was not marked as used by the failed launch).
        """A failed launch surfaces to the caller and leaves the actor free to launch again."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        thread_creations: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def _fail_to_launch(argv: list[str], *, envs: dict[str, str]) -> None:
            # [FUNCTION DOC] HELPER: Always raises an os error "cannot launch".
            raise OSError("cannot launch")

        class _FakeThread:
            # [CLASS DOC] HELPER: Thread stand-in that only records construction arguments.
            def __init__(self, *args: object, **kwargs: object) -> None:
                # [FUNCTION DOC] HELPER: Appends the positional and keyword arguments to the shared list.
                thread_creations.append((args, kwargs))

            def start(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        with pytest.MonkeyPatch.context() as failing_launch:
            failing_launch.setattr(process_utils, "launch_bound_subprocess", _fail_to_launch)
            failing_launch.setattr(threading, "Thread", _FakeThread)
            with pytest.raises(OSError):
                actor.run(cmd="true", envs={})

        assert thread_creations == []

        actor.run(cmd="exit 7", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [7]


class TestLifecycleBinding:
    # [CLASS DOC] Tests how the actor process exit follows the subprocess's end.
    def test_exits_actor_process_with_zero_on_subprocess_success(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A command "true" leads to exactly one exit call with code 0.
        """The actor process exits with code 0 when the subprocess succeeds."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd="true", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [0]

    def test_exits_actor_process_with_subprocess_returncode_on_failure(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] A command "exit 7" leads to exit code 7.
        """The actor process exits with the subprocess returncode when it fails."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd="exit 7", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [7]

    def test_exits_actor_process_with_128_plus_signal_on_signal_killed_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] A command that sends terminate to its own shell leads to exit code 128 plus
        # the terminate signal number.
        """A signal-killed subprocess (negative returncode) maps to exit code 128+signal, as a shell reports it."""
        fake_exit = _FakeExit(monkeypatch)

        CommandActor().run(cmd='kill -TERM "$$"', envs={})
        fake_exit.wait()

        assert fake_exit.codes == [128 + signal.SIGTERM]

    def test_does_not_exit_while_subprocess_is_running(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        # [FUNCTION DOC] Runs a shell loop that waits for a flag file. For 0.3 seconds no exit happens;
        # after creating the file the exit happens with code 0.
        """The actor keeps running until the subprocess actually exits."""
        fake_exit = _FakeExit(monkeypatch)
        flag_path = tmp_path / "flag"

        CommandActor().run(cmd=f'while [ ! -f "{flag_path}" ]; do sleep 0.01; done', envs={})

        assert not fake_exit.event.wait(timeout=0.3)
        flag_path.touch()
        fake_exit.wait()
        assert fake_exit.codes == [0]


class TestShutdown:
    # [CLASS DOC] Tests the deliberate shutdown method, which must stop the subprocess without
    # triggering the crash exit.
    def test_shutdown_stops_the_subprocess_without_the_crash_exit(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Runs a long sleep, shuts down, waits for the private process handle to end
        # (10 seconds), and asserts no exit call occurs within 0.5 seconds.
        """A deliberate shutdown must not look like a crash to the babysitter."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="sleep 300", envs={})

        actor.shutdown()

        actor._process.wait(timeout=10)
        assert not fake_exit.event.wait(timeout=0.5)

    def test_shutdown_before_run_is_a_noop(self):
        # [FUNCTION DOC] Shutting down an actor that never ran raises nothing.
        """Tearing down an actor that never launched anything is safe."""
        CommandActor().shutdown()

    def test_shutdown_before_run_does_not_disable_a_later_run(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Shuts down first, then runs "exit 7"; the crash exit still happens with code 7.
        """A shutdown of a never-launched actor must not suppress the crash exit of a later run."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()

        actor.shutdown()
        actor.run(cmd="exit 7", envs={})
        fake_exit.wait()

        assert fake_exit.codes == [7]

    def test_shutdown_twice_is_safe(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Shuts down a running sleep twice (waiting for the process in between);
        # no error and no exit call within 0.5 seconds.
        """A second shutdown of an already-stopped subprocess is a harmless no-op."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="sleep 300", envs={})

        actor.shutdown()
        actor._process.wait(timeout=10)
        actor.shutdown()

        assert not fake_exit.event.wait(timeout=0.5)

    def test_shutdown_after_the_subprocess_exited_on_its_own_is_safe(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] After a command "true" has finished (exit observed), shutdown raises nothing.
        """Shutting down after a natural subprocess exit must not raise."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="true", envs={})
        fake_exit.wait()

        actor.shutdown()

    def test_an_exec_command_finishes_its_sigterm_handling_before_shutdown_returns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ):
        # [FUNCTION DOC] Runs a python command via exec whose terminate handler sleeps one second and
        # then writes a marker file; waits for its ready file, calls shutdown, and asserts the marker
        # file already says "drained" when shutdown returned (shutdown waited for the grace period).
        """Shutdown's SIGTERM reaches the exec'd command itself, which gets its grace period to drain."""
        _FakeExit(monkeypatch)
        ready_path, drained_path = tmp_path / "ready", tmp_path / "drained"
        actor = CommandActor()
        actor.run(cmd=_exec_command(_DRAINING_SOURCE, ready_path, drained_path), envs={})
        _wait_for_path(ready_path)

        actor.shutdown()

        assert drained_path.read_text() == "drained"


class TestKillSubprocess:
    # [CLASS DOC] Tests the kill-subprocess method used to simulate a crash.
    def test_killing_the_subprocess_surfaces_as_the_actor_crash_exit(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Kills a running sleep; the exit call happens with 128 plus the kill signal.
        """A killed subprocess must take the actor down, as a real crash would."""
        fake_exit = _FakeExit(monkeypatch)
        actor = CommandActor()
        actor.run(cmd="sleep 300", envs={})

        actor.kill_subprocess()

        fake_exit.wait()
        assert fake_exit.codes == [128 + signal.SIGKILL]

    def test_kill_before_run_is_rejected(self):
        # [FUNCTION DOC] Killing before any run raises an assertion error.
        """A crash injection into an actor with no subprocess is a caller bug."""
        with pytest.raises(AssertionError):
            CommandActor().kill_subprocess()


class TestInjectFault:
    # [CLASS DOC] Tests fault injection, which supports only the sigkill mode and kills the whole
    # process tree of the subprocess, never the actor process itself.
    def test_a_sigkill_reaches_the_worker_process_group(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the single-process kill replaced by one that fails and the tree kill
        # replaced by a recorder, injecting sigkill into an actor holding a fake process (pid 4321)
        # records pid 4321 as killed via the tree kill.
        """Crashing a worker must include every subprocess that belongs to that worker."""
        killed: list[int] = []
        monkeypatch.setattr(process_utils, "kill_process", _refuse_to_kill)
        monkeypatch.setattr(process_utils, "kill_process_tree", lambda process: killed.append(process.pid))
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        actor.inject_fault("sigkill")

        assert killed == [4321]

    def test_a_sigkill_does_not_leave_a_spawned_engine_process_behind(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Runs a shell command that starts a background sleep (its pid saved to a file)
        # and waits. After injecting sigkill, waits for the fake exit and then for the background
        # child to disappear (10 seconds). The tree is killed again in cleanup.
        """A worker crash must not orphan an engine child that can retain its GPU allocation."""
        fake_exit = _FakeExit(monkeypatch)
        child_pid_path = tmp_path / "child-pid"
        actor = CommandActor()
        actor.run(cmd=f"sleep 300 & printf '%s' \"$!\" > {shlex.quote(str(child_pid_path))}; wait", envs={})
        _wait_for_path(child_pid_path)
        child_pid = int(child_pid_path.read_text())

        try:
            actor.inject_fault("sigkill")

            fake_exit.wait()
            _wait_for_process_exit(child_pid)
        finally:
            process_utils.kill_process_tree(actor._process)

    @pytest.mark.parametrize("mode", ["exit", "segfault", "deadlock"])
    def test_every_other_failure_mode_is_rejected(self, monkeypatch: pytest.MonkeyPatch, mode: str):
        # [FUNCTION DOC] Parametrized over exit, segfault and deadlock; with the tree kill replaced by
        # one that fails, injecting the mode raises an assertion error saying only sigkill is supported.
        """A process exits, segfaults and deadlocks from the inside; no signal an outsider sends reproduces that."""
        monkeypatch.setattr(process_utils, "kill_process_tree", _refuse_to_kill)
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        with pytest.raises(AssertionError, match="only sigkill"):
            actor.inject_fault(mode)

    def test_an_unknown_mode_is_rejected(self):
        # [FUNCTION DOC] An unrecognized mode name raises a value error.
        """A misspelt mode must not be waved through as some default crash."""
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        with pytest.raises(ValueError):
            actor.inject_fault("nuke")

    def test_the_actor_process_survives_the_injection(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] The process-level fault injector is replaced by one that fails the test if
        # called; injecting sigkill must only kill the (stubbed) subprocess tree and return normally.
        """Production loses the engine, not its supervisor, so crashing the actor would be the wrong fault."""
        monkeypatch.setattr(fault_injector, "inject_fault", _refuse_to_inject)
        monkeypatch.setattr(process_utils, "kill_process_tree", lambda process: None)
        actor = CommandActor()
        actor._process = _FakeProcess(pid=4321)

        actor.inject_fault("sigkill")

    def test_an_actor_without_a_subprocess_is_rejected(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Injecting sigkill into an actor that never ran raises an assertion error
        # mentioning "no subprocess" (and never falls back to the process-level injector).
        """Falling back to killing the actor would inject a fault production never produces."""
        monkeypatch.setattr(fault_injector, "inject_fault", _refuse_to_inject)

        with pytest.raises(AssertionError, match="no subprocess"):
            CommandActor().inject_fault("sigkill")


class _FakeProcess:
    # [CLASS DOC] HELPER: Minimal process stand-in carrying only a pid.
    def __init__(self, *, pid: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the pid.
        self.pid = pid


def _refuse_to_inject(mode: str) -> None:
    # [FUNCTION DOC] HELPER: Replacement for the process-level fault injector that always raises an
    # assertion error saying the actor process must not crash itself, naming the mode.
    raise AssertionError(f"the actor process must not crash itself, but {mode} was injected into it")


def _refuse_to_kill(process) -> None:
    # [FUNCTION DOC] HELPER: Replacement kill function that always raises an assertion error naming
    # the pid of the process it was asked to kill.
    raise AssertionError(f"no kill was expected, but pid {process.pid} was killed")


def _wait_for_path(path: Path) -> None:
    # [FUNCTION DOC] HELPER: Polls every 0.01 seconds for up to 10 seconds until the path exists, then
    # asserts that it exists.
    deadline = time.monotonic() + 10
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def _exec_command(source: str, *args: Path) -> str:
    # [FUNCTION DOC] HELPER: Builds a shell command string that replaces the shell with the current
    # python interpreter running the given source text with the given paths as arguments (each
    # shell-quoted). Returns the string.
    # Mirrors how the worker manager hands a spec's command to its actor.
    return f"exec {shlex.join([sys.executable, '-c', source, *map(str, args)])}"


def _wait_for_process_exit(pid: int) -> None:
    # [FUNCTION DOC] HELPER: Polls every 0.01 seconds for up to 10 seconds by sending the null signal
    # to the pid; returns once the process no longer exists. Raises an assertion error saying the
    # process survived the worker fault if the deadline passes.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.01)
    raise AssertionError(f"process {pid} survived the worker fault")


class TestNodeAddress:
    # [CLASS DOC] Tests the node ip reported by the actor.
    def test_the_node_ip_the_actor_reports_honours_the_host_ip_override(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the host ip environment variable set to 10.20.30.40, the actor's private
        # node ip getter returns that value.
        """The address a worker publishes to its peers is the one its deployment pinned through MILES_HOST_IP."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.20.30.40")

        assert CommandActor()._get_node_ip() == "10.20.30.40"

    def test_the_node_ip_the_actor_reports_falls_back_to_the_ray_node_ip(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the override variable removed, the getter returns the current node ip
        # as computed by the miscellaneous utility.
        """Without an override the worker publishes the address ray placed it on."""
        monkeypatch.delenv(MILES_HOST_IP_ENV, raising=False)

        assert CommandActor()._get_node_ip() == get_current_node_ip()
