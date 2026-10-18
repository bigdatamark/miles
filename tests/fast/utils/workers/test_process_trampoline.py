# [MODULE DOC] Tests for the process trampoline module, a tiny launcher that arms the Linux
# parent-death signal (SIGKILL), verifies the expected parent is still the parent, and then
# replaces itself with the real command. Three layers: in-process tests of main with the C library,
# parent-pid lookup, exec and exit all faked (checking call order and arguments); real subprocess
# tests of the exit status and argument passing; and a source scan asserting the module imports
# only a few standard library modules. The trampoline's command line is the expected parent pid
# followed by the command and its arguments.
from __future__ import annotations

import ast
import ctypes
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import miles.utils.workers.process_trampoline as process_trampoline

_TRAMPOLINE_MODULE = "miles.utils.workers.process_trampoline"


@dataclass
class _TrampolineRun:
    # [CLASS DOC] HELPER: Record of what a faked trampoline run did: ordered event names, the
    # option/signal pairs passed to the process-control call, the program/argument-vector pairs
    # passed to exec, and the exit status if the process ended (none when exec was reached).
    events: list[str] = field(default_factory=list)
    prctl_calls: list[tuple[int, int]] = field(default_factory=list)
    exec_calls: list[tuple[str, list[str]]] = field(default_factory=list)
    exit_status: int | None = None


def _run_main(
    *,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    platform: str = "linux",
    current_parent_pid: int,
    exec_error: OSError | None = None,
) -> _TrampolineRun:
    # [FUNCTION DOC] HELPER: Runs the trampoline main in-process with the platform name, the
    # argument vector, the C library loader, the parent-pid lookup, exec and the hard exit all
    # replaced by recorders (via the monkeypatch, restored at test end). The fake parent pid is the
    # given current_parent_pid; exec optionally raises the given operating system error; the fake
    # hard exit records an event and raises a system exit with the status. Returns the record, with
    # the exit status taken from a system exit (non-integer codes count as 1).
    run = _TrampolineRun()

    class _FakeLibc:
        # [CLASS DOC] HELPER: Fake C library exposing only the process-control call.
        def prctl(self, option: int, signal_number: int) -> int:
            # [FUNCTION DOC] HELPER: Records a "prctl" event and the option/signal pair; returns 0
            # (success).
            run.events.append("prctl")
            run.prctl_calls.append((option, signal_number))
            return 0

    def fake_getppid() -> int:
        # [FUNCTION DOC] HELPER: Records a "getppid" event and returns the configured parent pid.
        run.events.append("getppid")
        return current_parent_pid

    def fake_execvp(file: str, args: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Records an "execvp" event and the program/arguments pair, then
        # raises the configured operating system error if there is one; otherwise returns (unlike
        # the real exec, which never returns).
        run.events.append("execvp")
        run.exec_calls.append((file, args))
        if exec_error is not None:
            raise exec_error

    def fake_exit(status: int) -> None:
        # [FUNCTION DOC] HELPER: Records an "exit" event and raises a system exit with the status
        # (the real hard exit would end the process immediately).
        run.events.append("exit")
        raise SystemExit(status)

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(ctypes, "CDLL", lambda name, use_errno=False: _FakeLibc())
    monkeypatch.setattr(os, "getppid", fake_getppid)
    monkeypatch.setattr(os, "execvp", fake_execvp)
    monkeypatch.setattr(os, "_exit", fake_exit)

    try:
        process_trampoline.main()
    except SystemExit as error:
        run.exit_status = error.code if isinstance(error.code, int) else 1

    return run


def _run_trampoline_process(*, expected_parent_pid: int, argv: list[str]) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] HELPER: Runs the real trampoline module in a child python process with the
    # expected parent pid followed by argv; captures output as text, 60 second timeout, no check on
    # the return code. Returns the completed process.
    return subprocess.run(
        [sys.executable, "-m", _TRAMPOLINE_MODULE, str(expected_parent_pid), *argv],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class TestTrampolineMain:
    # [CLASS DOC] In-process tests of the trampoline main function using the faked environment from
    # _run_main; they check the sequence and arguments of the low-level calls.
    def test_the_death_signal_is_armed_with_sigkill_on_linux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] On linux with a matching parent, asserts exactly one process-control call
        # was made with option 1 (the set-parent-death-signal option) and SIGKILL.
        """On linux the trampoline asks the kernel for a SIGKILL when its parent dies."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            current_parent_pid=4242,
        )

        assert run.prctl_calls == [(1, signal.SIGKILL)]

    def test_the_real_command_is_exec_ed_with_its_own_argv(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Command line has the pid then a shell command with an empty last word;
        # asserts exec is called once with the program and an argument vector equal to the whole
        # command (program included as first element, empty argument kept) and no exit happened.
        """Everything after the expected parent pid becomes the exec'd command, argv0 included."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/sh", "-c", "sleep 1", ""],
            current_parent_pid=4242,
        )

        assert run.exec_calls == [("/bin/sh", ["/bin/sh", "-c", "sleep 1", ""])]
        assert run.exit_status is None

    def test_the_death_signal_is_armed_before_the_parent_is_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Asserts the event order is arm the signal, look up the parent pid, exec.
        """Checking the parent first would leave a window where the death signal is never armed."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            current_parent_pid=4242,
        )

        assert run.events == ["prctl", "getppid", "execvp"]

    def test_a_changed_parent_makes_it_exit_instead_of_exec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Expected parent 4242 but the current parent is 1; asserts exit status 1 and
        # that exec was never called.
        """A child reparented before the signal was armed would never be reaped, so it must not run the command."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            current_parent_pid=1,
        )

        assert run.exit_status == 1
        assert run.exec_calls == []

    def test_a_failed_exec_is_reported_and_exits_with_the_shell_convention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Exec raises a file-not-found error for a missing program; asserts the last
        # two events are exec then exit and the exit status is 127 (shell convention for command
        # not found).
        """A missing or unrunnable command must not look like the real command dying with status 1."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "1234", "no-such-binary", "--flag"],
            current_parent_pid=1234,
            exec_error=FileNotFoundError(2, "No such file or directory"),
        )

        assert run.events[-2:] == ["execvp", "exit"]
        assert run.exit_status == 127

    def test_off_linux_it_execs_without_arming_or_checking_anything(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Platform is darwin and the parent differs; asserts no process-control call,
        # the only event is exec, and exec received the program and its argument vector.
        """PR_SET_PDEATHSIG is linux-only, so elsewhere the trampoline degrades to a plain exec."""
        run = _run_main(
            monkeypatch=monkeypatch,
            argv=["trampoline", "4242", "/bin/echo", "hi"],
            platform="darwin",
            current_parent_pid=1,
        )

        assert run.prctl_calls == []
        assert run.events == ["execvp"]
        assert run.exec_calls == [("/bin/echo", ["/bin/echo", "hi"])]


class TestTrampolineProcess:
    # [CLASS DOC] Tests running the real trampoline module in a child python process, with the
    # test process as the expected parent.
    def test_the_exit_status_is_the_real_commands_own(self) -> None:
        # [FUNCTION DOC] Runs a python one-liner that exits with status 7 through the trampoline and
        # asserts the child's return code is 7 (the command replaced the trampoline).
        """A trampoline that ran the command as a child instead of exec'ing would report its own status."""
        result = _run_trampoline_process(
            expected_parent_pid=os.getpid(),
            argv=[sys.executable, "-c", "raise SystemExit(7)"],
        )

        assert result.returncode == 7

    def test_the_command_arguments_arrive_unchanged(self) -> None:
        # [FUNCTION DOC] Passes arguments including spaces, an empty string, a dollar-sign word and a
        # double quote to a python snippet that prints its arguments as JSON; asserts success and
        # that the printed list equals the passed list exactly.
        """Joining the argv into a shell string would expand, split or drop these arguments."""
        arguments = ["--flag", "a b", "", "$HOME", 'quote"d']
        code = "import json, sys; print(json.dumps(sys.argv[1:]))"
        result = _run_trampoline_process(
            expected_parent_pid=os.getpid(),
            argv=[sys.executable, "-c", code, *arguments],
        )

        assert result.returncode == 0
        assert json.loads(result.stdout) == arguments


class TestTrampolineModule:
    # [CLASS DOC] Static checks on the trampoline module source.
    def test_it_imports_nothing_but_the_standard_library_modules_it_needs(self) -> None:
        # [FUNCTION DOC] Parses the module source into a syntax tree, collects the top-level package
        # name of every plain import and from-import (relative imports without a module are
        # skipped), and asserts the set is exactly the C-types, operating system, signal and system
        # modules (no project code).
        """The trampoline runs before every launched command, so importing miles code would be a new failure mode."""
        source = Path(process_trampoline.__file__).read_text()

        imported_modules: set[str] = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.add(node.module.split(".")[0])

        assert imported_modules == {"ctypes", "os", "signal", "sys"}
