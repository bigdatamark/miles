# [MODULE DOC] Tests for the worker process utilities: launching a subprocess bound to its parent
# (new session, trampoline wrapper, exit hook), killing a process or its whole group, and the
# terminate-then-kill escalation. Tests start real python child processes that sleep or ignore
# SIGTERM, use files in a temp directory to learn pids, and poll with a deadline to observe death.
# Linux-only behaviors (parent-death signal) are skipped on other platforms.
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import miles.utils.workers.process_utils as process_utils
from miles.utils.workers.process_utils import (
    kill_process,
    kill_process_tree,
    launch_bound_subprocess,
    terminate_process_tree,
)

_SLEEP_FOREVER = "import time; time.sleep(300)"


def _is_alive(pid: int) -> bool:
    # [FUNCTION DOC] HELPER: True when sending the null signal to the pid succeeds, false when no
    # such process exists. A zombie still counts as alive; a permission error propagates.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_until(predicate, *, deadline_seconds: float = 15.0) -> bool:
    # [FUNCTION DOC] HELPER: Polls the predicate every 50 ms until true or until the deadline
    # (default 15 seconds); returns the predicate's final value (one last evaluation after the
    # deadline), never raising.
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _read_when_present(path: Path, *, deadline_seconds: float = 15.0) -> str:
    # [FUNCTION DOC] HELPER: Waits until the file exists and is non-empty (assertion error on
    # timeout) and returns its text.
    assert _wait_until(lambda: path.exists() and path.read_text() != "", deadline_seconds=deadline_seconds)
    return path.read_text()


class TestTerminateProcessTree:
    # [CLASS DOC] Tests of terminate_process_tree: SIGTERM to the whole process group, then SIGKILL
    # after an optional timeout, also for descendants that outlive the group leader.
    def test_kills_the_child(self):
        # [FUNCTION DOC] Launches a sleeping bound child, checks it is alive, terminates the tree,
        # and asserts the pid is no longer alive.
        """A launched sleeper is gone after terminate_process_tree."""
        process = launch_bound_subprocess([sys.executable, "-c", _SLEEP_FOREVER], envs={})
        assert _is_alive(process.pid)
        terminate_process_tree(process)
        assert not _is_alive(process.pid)

    def test_kills_the_grandchild_too(self, tmp_path):
        # [FUNCTION DOC] The launched child spawns a sleeping grandchild and writes its pid to a
        # file; after terminating the tree, asserts the grandchild eventually dies.
        """Killing the group takes down processes the child itself spawned."""
        pid_file = tmp_path / "grandchild.pid"
        child_code = (
            "import subprocess, sys, time\n"
            f"grand = subprocess.Popen([sys.executable, '-c', {_SLEEP_FOREVER!r}])\n"
            f"open({str(pid_file)!r}, 'w').write(str(grand.pid))\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", child_code], envs={})
        grandchild_pid = int(_read_when_present(pid_file))

        terminate_process_tree(process)
        assert _wait_until(lambda: not _is_alive(grandchild_pid))

    def test_sigkills_grandchild_after_leader_exits(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Leader sleeps and spawns a grandchild that ignores SIGTERM and records its
        # pid. Terminating with a 0.1 second kill timeout must kill the grandchild within 1 second
        # (SIGKILL escalation once the leader dies from SIGTERM). A finally block force-kills the
        # grandchild so a failure leaks nothing.
        """An ignored SIGTERM is escalated after the process-group leader exits."""
        pid_file = tmp_path / "grandchild.pid"
        grandchild_code = (
            "import os, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(300)\n"
        )
        leader_code = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", leader_code], envs={})
        grandchild_pid = int(_read_when_present(pid_file))

        try:
            terminate_process_tree(process, sigkill_timeout=0.1)
            assert _wait_until(lambda: not _is_alive(grandchild_pid), deadline_seconds=1.0)
        finally:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_sigkills_grandchild_when_leader_already_exited(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] The leader spawns a SIGTERM-ignoring grandchild, waits for its pid file and
        # exits on its own; the test waits for the leader to be reaped, then terminates the tree
        # with a 0.1 second timeout and asserts the grandchild dies within 1 second (an exited
        # leader must not block group cleanup). The grandchild is force-killed in a finally block.
        """An exited process-group leader does not prevent cleanup of a surviving descendant."""
        pid_file = tmp_path / "grandchild.pid"
        grandchild_code = (
            "import os, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(300)\n"
        )
        leader_code = (
            "import subprocess, sys, time\n"
            "from pathlib import Path\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            f"pid_file = Path({str(pid_file)!r})\n"
            "while not pid_file.exists():\n"
            "    time.sleep(0.01)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", leader_code], envs={})
        grandchild_pid = int(_read_when_present(pid_file))
        process.wait(timeout=15)

        try:
            terminate_process_tree(process, sigkill_timeout=0.1)
            assert _wait_until(lambda: not _is_alive(grandchild_pid), deadline_seconds=1.0)
        finally:
            try:
                os.kill(grandchild_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_escalates_to_sigkill_when_sigterm_is_ignored(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A child that ignores SIGTERM signals readiness through a file; terminating
        # with a 0.1 second timeout must leave its return code at minus SIGKILL.
        """A process that ignores SIGTERM is terminated with SIGKILL after the timeout."""
        ready_file = tmp_path / "ready"
        code = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"open({str(ready_file)!r}, 'w').write('ready')\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={})
        _read_when_present(ready_file)

        terminate_process_tree(process, sigkill_timeout=0.1)

        assert process.returncode == -signal.SIGKILL

    def _launch_slow_sigterm_exiter(self, tmp_path: Path) -> subprocess.Popen:
        # [FUNCTION DOC] HELPER: Launches a bound child whose SIGTERM handler sleeps one second and
        # then hard-exits with status 7, waits until it has written its ready file, and returns the
        # process object.
        ready_file = tmp_path / "ready"
        code = (
            "import os, signal, time\n"
            "def handle(signum, frame):\n"
            "    time.sleep(1.0)\n"
            "    os._exit(7)\n"
            "signal.signal(signal.SIGTERM, handle)\n"
            f"open({str(ready_file)!r}, 'w').write('ready')\n"
            "time.sleep(300)\n"
        )
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={})
        _read_when_present(ready_file)
        return process

    def test_a_generous_sigkill_timeout_lets_a_slow_sigterm_handler_exit_cleanly(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With a 30 second kill timeout the slow handler finishes; asserts the child's
        # return code is 7.
        """A slow SIGTERM handler reaches its own exit code when the grace period is long enough."""
        process = self._launch_slow_sigterm_exiter(tmp_path)

        terminate_process_tree(process, sigkill_timeout=30.0)

        assert process.returncode == 7

    def test_a_short_sigkill_timeout_escalates_before_the_sigterm_handler_finishes(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With a 0.05 second kill timeout the slow handler is cut short; asserts the
        # return code is minus SIGKILL.
        """The same slow SIGTERM handler is cut short by SIGKILL when the grace period is tiny."""
        process = self._launch_slow_sigterm_exiter(tmp_path)

        terminate_process_tree(process, sigkill_timeout=0.05)

        assert process.returncode == -signal.SIGKILL

    def test_noop_on_already_exited_process(self):
        # [FUNCTION DOC] Launches a child that exits immediately, waits for it, then terminates the
        # tree twice; asserts no exception is raised (only the absence of an error is checked).
        """Terminating an exited process is safe and idempotent."""
        process = launch_bound_subprocess([sys.executable, "-c", "pass"], envs={})
        process.wait(timeout=15)
        terminate_process_tree(process)
        terminate_process_tree(process)


class TestKillProcessTree:
    # [CLASS DOC] Tests of kill_process_tree, the immediate SIGKILL of the process group (crash
    # simulation).
    def test_kills_the_child_without_a_term_grace_period(self):
        # [FUNCTION DOC] The child ignores SIGTERM; after kill_process_tree its return code must be
        # minus SIGKILL (no SIGTERM phase).
        """The kill is a crash simulation, so the child gets SIGKILL, not SIGTERM."""
        trap_term = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"
        process = launch_bound_subprocess([sys.executable, "-c", trap_term], envs={})
        assert _is_alive(process.pid)

        kill_process_tree(process)

        process.wait(timeout=15)
        assert process.returncode == -signal.SIGKILL


class TestKillProcess:
    # [CLASS DOC] Tests of kill_process, which signals a single pid via the operating system call.
    def test_kills_only_the_requested_process_with_sigkill(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Builds a bare process object with a fake pid (bypassing its constructor),
        # replaces the kill call with a recorder, and asserts exactly one call with that pid and
        # SIGKILL.
        """Killing one process sends SIGKILL to its exact pid through the OS boundary."""
        process = subprocess.Popen.__new__(subprocess.Popen)
        process.pid = 12345
        sent_signals: list[tuple[int, int]] = []

        monkeypatch.setattr(
            process_utils.os, "kill", lambda pid, signal_number: sent_signals.append((pid, signal_number))
        )

        kill_process(process)

        assert sent_signals == [(process.pid, signal.SIGKILL)]

    def test_an_already_exited_process_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] The patched kill call raises a process-lookup error; kill_process must
        # swallow it and return normally.
        """Killing a process that has already exited returns normally."""
        process = subprocess.Popen.__new__(subprocess.Popen)
        process.pid = 12345

        def raise_process_lookup_error(pid: int, signal_number: int) -> None:
            # [FUNCTION DOC] HELPER: Fake kill that always raises a process-lookup error.
            raise ProcessLookupError

        monkeypatch.setattr(process_utils.os, "kill", raise_process_lookup_error)

        kill_process(process)


class TestLaunchBoundSubprocess:
    # [CLASS DOC] Tests of launch_bound_subprocess: environment merging, wrapping in the trampoline,
    # absence of pre-exec python code, in-place exec and automatic death with the parent (normal
    # exit via an exit hook, SIGKILL via the parent-death signal on linux).
    def test_envs_are_merged_over_the_parent_environment(self, tmp_path):
        # [FUNCTION DOC] Launches a child that writes the value of a custom variable passed in envs
        # to a file; asserts the file holds that value (envs layered on the inherited environment).
        """Passed envs reach the child on top of the inherited environment."""
        out_file = tmp_path / "env.txt"
        code = f"import os; open({str(out_file)!r}, 'w').write(os.environ['MILES_BOUND_TEST_VAR'])"
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={"MILES_BOUND_TEST_VAR": "yes"})
        process.wait(timeout=15)
        assert out_file.read_text() == "yes"

    def _launch_intermediate_parent(
        self, tmp_path, *, stay_alive: bool, argv: list[str] | None = None
    ) -> tuple[subprocess.Popen, int]:
        # [FUNCTION DOC] HELPER: Starts an intermediate python parent process that itself calls
        # launch_bound_subprocess for argv (default: a long sleeper), writes the bound child's pid
        # to a file, and then sleeps 300 seconds if stay_alive is true (otherwise exits at once).
        # Waits for the pid file and returns the parent process object and the bound child's pid.
        pid_file = tmp_path / "bound_child.pid"
        argv = argv or [sys.executable, "-c", _SLEEP_FOREVER]
        parent_code = (
            "import sys, time\n"
            "from miles.utils.workers.process_utils import launch_bound_subprocess\n"
            f"process = launch_bound_subprocess({argv!r}, envs={{}})\n"
            f"open({str(pid_file)!r}, 'w').write(str(process.pid))\n" + ("time.sleep(300)\n" if stay_alive else "")
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_code])
        bound_child_pid = int(_read_when_present(pid_file))
        return parent, bound_child_pid

    def test_child_dies_when_parent_exits_normally(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] An intermediate parent launches a bound child that spawns a grandchild
        # (pid written to a file), waits for that file and exits normally. Asserts the grandchild
        # dies within the deadline (via the parent's exit hook reaping the group). The finally block
        # kills the parent if still alive and force-kills the grandchild's process group.
        """The atexit hook reaps a bound child's descendant on normal parent exit."""
        grandchild_pid_file = tmp_path / "bound_grandchild.pid"
        grandchild_code = (
            "import os, time\n"
            f"open({str(grandchild_pid_file)!r}, 'w').write(str(os.getpid()))\n"
            "time.sleep(300)\n"
        )
        child_code = (
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            "time.sleep(300)\n"
        )
        parent_code = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "from miles.utils.workers.process_utils import launch_bound_subprocess\n"
            f"launch_bound_subprocess([sys.executable, '-c', {child_code!r}], envs={{}})\n"
            f"pid_file = Path({str(grandchild_pid_file)!r})\n"
            "while not pid_file.exists():\n"
            "    time.sleep(0.01)\n"
        )
        parent = subprocess.Popen([sys.executable, "-c", parent_code])
        grandchild_pid = int(_read_when_present(grandchild_pid_file))

        try:
            parent.wait(timeout=15)
            assert _wait_until(lambda: not _is_alive(grandchild_pid))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=15)
            try:
                process_group_id = os.getpgid(grandchild_pid)
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_no_python_runs_between_the_fork_and_the_exec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Replaces process creation and the exit-hook registration with fakes, calls
        # launch_bound_subprocess, and asserts the creation keyword arguments lack a pre-exec
        # function and request a new session.
        """preexec_fn runs python in a forked child of a threaded actor, which can deadlock before exec."""
        popen_kwargs: list[dict[str, object]] = []

        def fake_popen(argv: list[str], **kwargs: object) -> object:
            # [FUNCTION DOC] HELPER: Fake process constructor; records the keyword arguments and
            # returns a plain object.
            popen_kwargs.append(kwargs)
            return object()

        monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(process_utils.atexit, "register", lambda function, *args: function)

        process_utils.launch_bound_subprocess([sys.executable, "-c", "pass"], envs={})

        assert "preexec_fn" not in popen_kwargs[0]
        assert popen_kwargs[0]["start_new_session"] is True

    def test_the_launched_command_is_wrapped_in_the_trampoline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Fakes the current pid (12345), process creation and the exit-hook
        # registration; launches a shell command and asserts the recorded argument vector is the
        # python executable, module flag, the trampoline module name, the pid as text, then the
        # original command words.
        """The parent pid is captured before spawning and handed to the trampoline as an argument."""
        launched_argvs: list[list[str]] = []

        def fake_popen(argv: list[str], **kwargs: object) -> object:
            # [FUNCTION DOC] HELPER: Fake process constructor; records the argument vector and
            # returns a plain object.
            launched_argvs.append(argv)
            return object()

        monkeypatch.setattr(process_utils.os, "getpid", lambda: 12345)
        monkeypatch.setattr(process_utils.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(process_utils.atexit, "register", lambda function, *args: function)

        process_utils.launch_bound_subprocess(["/bin/sh", "-c", "true"], envs={})

        assert launched_argvs == [
            [sys.executable, "-m", "miles.utils.workers.process_trampoline", "12345", "/bin/sh", "-c", "true"]
        ]

    def test_the_trampoline_execs_the_real_command_in_place(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] The child writes its own pid and its argument-zero to a file. Asserts the
        # pid equals the launched process object's pid (no extra wrapper process remains) and that
        # argument zero is the inline-code flag.
        """An exec replaces the trampoline, so the reported pid must be the command's own process."""
        out_file = tmp_path / "pid.txt"
        code = f"import os, sys; open({str(out_file)!r}, 'w').write(f'{{os.getpid()}} {{sys.argv[0]}}')"
        process = launch_bound_subprocess([sys.executable, "-c", code], envs={})
        process.wait(timeout=15)

        reported_pid, argv0 = _read_when_present(out_file).split(" ")
        assert int(reported_pid) == process.pid
        assert argv0 == "-c"

    @pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is linux-only")
    def test_the_trampoline_exits_when_the_parent_is_already_gone(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Linux only. Runs the trampoline directly with expected parent pid 1 (which
        # is not the real parent) and a command that would create a marker file; asserts return code
        # 1 and that the marker file does not exist.
        """Between the fork and the prctl the parent may die, and then nothing would ever reap the child."""
        marker = tmp_path / "ran"
        code = f"open({str(marker)!r}, 'w').write('ran')"
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "miles.utils.workers.process_trampoline",
                "1",
                sys.executable,
                "-c",
                code,
            ]
        )
        process.wait(timeout=15)

        assert process.returncode == 1
        assert not marker.exists()

    @pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is linux-only")
    def test_child_dies_when_parent_is_sigkilled(self, tmp_path):
        # [FUNCTION DOC] Linux only. Launches an intermediate parent that stays alive with a bound
        # sleeping child, SIGKILLs the parent, and asserts the bound child dies within the deadline
        # (parent-death signal; no exit hook can run on SIGKILL).
        """The parent-death signal reaps the bound child even on SIGKILL."""
        parent, bound_child_pid = self._launch_intermediate_parent(tmp_path, stay_alive=True)
        assert _is_alive(bound_child_pid)

        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=15)
        assert _wait_until(lambda: not _is_alive(bound_child_pid))

    @pytest.mark.skipif(sys.platform != "linux", reason="PDEATHSIG is linux-only")
    def test_a_command_its_shell_execs_dies_when_parent_is_sigkilled(self, tmp_path):
        # [FUNCTION DOC] Linux only. The bound command is a shell that execs a python sleeper which
        # writes its pid to a file. After SIGKILLing the intermediate parent, asserts that exec'd
        # command process dies within the deadline; the finally block force-kills it.
        """exec makes the command itself replace `/bin/sh`, so it inherits the binding instead of being orphaned."""
        command_pid_file = tmp_path / "command.pid"
        command_code = (
            f"import os, time; open({str(command_pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(300)"
        )
        shell_command = f"exec {shlex.join([sys.executable, '-c', command_code])}"
        parent, _ = self._launch_intermediate_parent(tmp_path, stay_alive=True, argv=["/bin/sh", "-c", shell_command])
        command_pid = int(_read_when_present(command_pid_file))

        try:
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait(timeout=15)
            assert _wait_until(lambda: not _is_alive(command_pid))
        finally:
            try:
                os.kill(command_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
