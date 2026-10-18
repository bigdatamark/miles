# [MODULE DOC] Tests for the process supervisor (command line entry point plus supervise function)
# that runs N copies of a command inside one pod through torchelastic and forwards signals.
# The file has two layers. Unit tests replace torchelastic's start_processes with a fake context
# (_FakeContext) and check argument parsing, environment/log wiring, exit-code mapping (a signalled
# death becomes 128 plus the signal number, otherwise the earliest recorded failure decides) and the
# teardown (close called with the forwarded signal and the grace period). Integration tests launch
# the real supervisor module in a fresh session as a subprocess and supervise this very file run as a
# script (the "child", see _child_main), which reports its identity into a temp directory so tests
# can assert on indices, pids, process groups, recorded signals and SIGKILL escalation.
# When run as a script this file is the supervised child; it exits with the status from _child_main.
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

import pytest
from torch.distributed.elastic.multiprocessing import DefaultLogsSpecs
from torch.distributed.elastic.multiprocessing.api import RunProcsResult, SignalException
from torch.distributed.elastic.multiprocessing.errors import ProcessFailure

import miles.utils.workers.process_supervisor as supervisor_module
from miles.utils.workers.process_supervisor import (
    _DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS,
    _SIGNALS_TO_HANDLE_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
    _exit_code_from_result,
    _exit_code_from_wait_result,
    _report_signal_received_during_teardown,
    main,
    supervise,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
CHILD_SCRIPT = Path(__file__).resolve()
SUPERVISOR_EXIT_TIMEOUT_SECONDS = 30.0
SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS = 10.0
GRACE_PERIOD_SECONDS = 0.5
GRACE_PERIOD_UPPER_BOUND_SLACK_SECONDS = 10.0
DOUBLE_SIGNAL_GRACE_PERIOD_SECONDS = 3.0
SECOND_SIGNAL_DELAY_SECONDS = 0.5
DOCUMENTED_SUBPROCESS_INDEX_ENV_VAR = "MILES_SUPERVISOR_SUBPROCESS_INDEX"
DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS = 20.0
KUBERNETES_DEFAULT_GRACE_PERIOD_SECONDS = 30.0
CHILD_PID = 4002
OTHER_CHILD_PID = CHILD_PID + 1
RECORDED_SUPERVISOR_EXIT_CODE = 17

ENV_MARKER_VAR = "MILES_TEST_ENV_MARKER"
STDOUT_MARKER = "child-stdout-"
STDERR_MARKER = "child-stderr-"

MODE_SLEEP = "sleep"
MODE_EXIT = "exit"
MODE_SUICIDE = "suicide"
MODE_IGNORE_SIGTERM = "ignore-sigterm"
MODE_RECORD_SIGNAL = "record-signal"

REPORT_WAIT_TIMEOUT_SECONDS = 30.0
_GRANDCHILD = "import sys, time\nopen(sys.argv[1], 'w').close()\nwhile True: time.sleep(3600)"


@pytest.mark.usefixtures("unconfigured_root_logging")
class TestArgumentValidation:
    # [CLASS DOC] Command line parsing of the supervisor main function. Every rejection is expected
    # to terminate with a system exit whose code is 2 (the argument parser convention); root
    # logging setup is stubbed out by the unconfigured_root_logging fixture.
    def test_a_missing_separator_is_rejected(self, capsys):
        # [FUNCTION DOC] Calls main with a valid subprocess count flag but no double-dash separator
        # and no command; expects a system exit with code 2 and the message that the command must
        # be given after the separator on standard error.
        """The supervised command must be introduced by '--', even when the flags parse cleanly."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "2"])

        assert exc_info.value.code == 2
        assert "must be given after '--'" in capsys.readouterr().err

    def test_a_command_given_without_the_separator_is_rejected(self):
        # [FUNCTION DOC] Passes a command ("sleep 1") directly after the count flag without the
        # separator; expects a system exit with code 2 instead of the words being parsed as flags.
        """A command written without '--' would otherwise be parsed as stray flags."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "2", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_an_empty_command_is_rejected(self):
        # [FUNCTION DOC] Passes the separator with nothing after it; expects a system exit, code 2.
        """A '--' with nothing after it is an error rather than a supervisor with nothing to run."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "2", "--"])

        assert exc_info.value.code == 2

    @pytest.mark.parametrize("count", ["0", "-1", "-8"])
    def test_a_non_positive_subprocess_count_is_rejected(self, count):
        # [FUNCTION DOC] Parametrized over the counts zero, minus one and minus eight (strings, as
        # given on the command line); each must make main exit with code 2 at parse time.
        """Supervising zero or fewer subprocesses is meaningless and is rejected up front."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", count, "--", "sleep", "1"])

        assert exc_info.value.code == 2

    @pytest.mark.parametrize("grace_period", ["inf", "-inf", "nan"])
    def test_a_non_finite_grace_period_is_rejected(self, grace_period):
        # [FUNCTION DOC] Parametrized over infinity, negative infinity and not-a-number given as the
        # termination grace period flag; each must make main exit with code 2.
        """A non-finite deadline would either disable the SIGKILL escalation or never compare true."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "1", "--termination-grace-period-seconds", grace_period, "--", "sleep", "1"])

        assert exc_info.value.code == 2

    @pytest.mark.parametrize("grace_period", ["-1", "-0.001", "-3600"])
    def test_a_negative_grace_period_is_rejected(self, grace_period):
        # [FUNCTION DOC] Parametrized over three negative finite values; each must make main exit
        # with code 2 (a negative period is not treated as "kill immediately").
        """A negative grace period is a typo rather than a request to SIGKILL immediately."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "1", "--termination-grace-period-seconds", grace_period, "--", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_a_missing_subprocess_count_is_rejected(self):
        # [FUNCTION DOC] Calls main with only the separator and a command; expects exit code 2
        # because the subprocess count flag is required.
        """There is no sensible default for how many ranks the pod holds."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_a_non_numeric_subprocess_count_is_rejected(self):
        # [FUNCTION DOC] Gives the word "two" as the count; expects exit code 2 from the parser.
        """A count that is not an integer fails at parse time rather than at spawn time."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--num-subprocesses", "two", "--", "sleep", "1"])

        assert exc_info.value.code == 2

    def test_help_is_printed_without_a_supervised_command(self, capsys):
        # [FUNCTION DOC] Calls main with the help flag; expects a system exit with code 0 and the
        # help text on standard output mentioning the subprocess index environment variable name.
        """Asking for help must not require inventing a command to supervise."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])

        assert exc_info.value.code == 0
        assert SUBPROCESS_INDEX_ENV_VAR in capsys.readouterr().out

    def test_a_zero_grace_period_is_accepted(self, monkeypatch):
        # [FUNCTION DOC] Replaces supervise with a recorder, runs main with a grace period of "0"
        # and asserts the recorded keyword argument is the float zero (no error is raised).
        """Escalating immediately is a legitimate request, unlike a negative period."""
        created = _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--termination-grace-period-seconds", "0", "--", "sleep", "1"])

        assert created[0]["termination_grace_period_seconds"] == 0.0


@pytest.mark.usefixtures("unconfigured_root_logging")
class TestMain:
    # [CLASS DOC] Checks the main wrapper: parse arguments, build the supervise call, return its
    # result. Uses a recorder in place of supervise that returns a fixed exit code.
    def test_the_parsed_arguments_reach_the_supervisor_and_its_code_is_returned(self, monkeypatch):
        # [FUNCTION DOC] Runs main with count 2, grace period 1.5 and the command "sleep 1"; asserts
        # main returns the recorder's fixed exit code and that supervise was called exactly once
        # with the command as a list, the integer count and the float grace period.
        """main is a thin wrapper: it parses, constructs, and propagates the exit code."""
        created = _record_supervisor(monkeypatch)

        exit_code = main(["--num-subprocesses", "2", "--termination-grace-period-seconds", "1.5", "--", "sleep", "1"])

        assert exit_code == RECORDED_SUPERVISOR_EXIT_CODE
        assert created == [{"command": ["sleep", "1"], "num_subprocesses": 2, "termination_grace_period_seconds": 1.5}]

    def test_a_separator_inside_the_supervised_command_is_left_alone(self, monkeypatch):
        # [FUNCTION DOC] Gives a command that itself contains a second separator; asserts the
        # recorded command keeps it, i.e. only the first separator splits flags from the command.
        """Only the first '--' splits, so the supervised command may contain its own."""
        created = _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--", "bash", "-c", "--", "payload"])

        assert created[0]["command"] == ["bash", "-c", "--", "payload"]

    def test_the_grace_period_defaults_when_the_flag_is_absent(self, monkeypatch):
        # [FUNCTION DOC] Omits the grace period flag and asserts the recorded value equals the
        # documented default of 20 seconds.
        """The default must expire before the Kubernetes termination grace period it runs inside."""
        created = _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--", "sleep", "1"])

        assert created[0]["termination_grace_period_seconds"] == DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS

    def test_the_process_arguments_are_used_when_no_argv_is_given(self, monkeypatch):
        # [FUNCTION DOC] Sets the process argument vector to a valid supervisor invocation, calls
        # main with no argument at all and asserts the command reached the recorded supervise call.
        # Mutates the global argument vector through the monkeypatch (restored at test end).
        """The container entrypoint calls main with no arguments at all."""
        created = _record_supervisor(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["process_supervisor", "--num-subprocesses", "1", "--", "sleep", "1"])

        main()

        assert created[0]["command"] == ["sleep", "1"]

    def test_an_explicitly_empty_argv_is_not_confused_with_no_argv(self, monkeypatch):
        # [FUNCTION DOC] Sets a valid process argument vector but calls main with an empty list;
        # expects a system exit with code 2, proving an empty list does not fall back to the
        # process arguments. supervise is deliberately not stubbed (it must never be reached).
        """Falling back to sys.argv here would make main run whatever the caller was invoked with."""
        monkeypatch.setattr(sys, "argv", ["process_supervisor", "--num-subprocesses", "1", "--", "sleep", "1"])

        with pytest.raises(SystemExit) as exc_info:
            main([])

        assert exc_info.value.code == 2

    def test_the_function_default_grace_period_matches_the_documented_one(self):
        # [FUNCTION DOC] Inspects the signature of supervise and asserts the default of its grace
        # period parameter equals the documented 20 seconds (same as the command line default).
        """Callers that call supervise directly must get the same default as the CLI."""
        parameters = inspect.signature(supervise).parameters

        assert parameters["termination_grace_period_seconds"].default == DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS


class TestExitCodeFromWaitResult:
    # [CLASS DOC] Pure-function tests for converting a Popen-style wait status (negative number for
    # death by signal) into a shell-style exit code.
    def test_a_normal_exit_keeps_its_code(self):
        # [FUNCTION DOC] Asserts the conversion maps the status 7 to 7.
        """Popen reports a normal exit as the exit code itself."""
        assert _exit_code_from_wait_result(7) == 7

    def test_a_successful_exit_stays_zero(self):
        # [FUNCTION DOC] Asserts the conversion maps the status 0 to 0.
        """Zero must not be mistaken for a signalled death."""
        assert _exit_code_from_wait_result(0) == 0

    def test_the_highest_exit_code_is_left_alone(self):
        # [FUNCTION DOC] Asserts the conversion maps the status 255 to 255 (no translation at the top
        # of the exit code range).
        """Exit code 255 is the top of the range and must not be translated."""
        assert _exit_code_from_wait_result(255) == 255

    @pytest.mark.parametrize("signum", [signal.SIGKILL, signal.SIGTERM, signal.SIGINT, signal.SIGABRT])
    def test_a_signalled_exit_becomes_128_plus_that_signal(self, signum):
        # [FUNCTION DOC] Parametrized over SIGKILL, SIGTERM, SIGINT and SIGABRT; feeds the negated
        # signal number and asserts the result is 128 plus the signal number.
        """Popen reports a signalled death as a negative signal number, which no shell understands."""
        assert _exit_code_from_wait_result(-int(signum)) == 128 + signum


class TestExitCodeFromResult:
    # [CLASS DOC] Pure-function tests for choosing the supervisor exit code from a torchelastic run
    # result built with _run_result and _failure (failures keyed by local rank, each carrying an
    # exit code and a whole-second timestamp).
    def test_a_run_without_failures_is_a_success(self):
        # [FUNCTION DOC] A result with an empty failure mapping yields exit code 0.
        """A batch where every rank exited zero produces no failures at all."""
        assert _exit_code_from_result(_run_result()) == 0

    def test_the_failure_torchelastic_recorded_first_decides(self):
        # [FUNCTION DOC] Two failures: rank 1 exit code 7 at timestamp 20 and rank 0 exit code 9 at
        # timestamp 10 (inserted rank 1 first). Asserts the result is 7, i.e. the failure that
        # torchelastic listed first (not the lowest rank, not the smallest timestamp): the code
        # takes the first entry in the failure mapping's own iteration order and ignores timestamps.
        """torchelastic records the rank that actually failed before the peers it then tears down."""
        failures = {
            1: _failure(local_rank=1, exitcode=7, timestamp=20),
            0: _failure(exitcode=9, timestamp=10),
        }
        assert _exit_code_from_result(_run_result(failures)) == 7

    def test_a_signalled_cause_beats_a_peer_that_traps_the_teardown_signal(self):
        # [FUNCTION DOC] Two failures with the same timestamp: rank 0 killed by SIGSEGV (listed
        # first) and rank 1 exiting with code 42. Asserts the result is 128 plus SIGSEGV.
        # KNOWN ISSUE: the name and docstring claim a signalled cause is preferred over a peer, but
        # the code has no such rule; it just takes the first entry of the failure mapping, so the
        # test passes only because the signalled rank is listed first.
        """A peer turning SIGTERM into a plain exit code ties on the second-resolution timestamp."""
        failures = {
            0: _failure(exitcode=-int(signal.SIGSEGV), timestamp=10),
            1: _failure(local_rank=1, exitcode=42, timestamp=10),
        }
        assert _exit_code_from_result(_run_result(failures)) == 128 + signal.SIGSEGV

    def test_a_signalled_earliest_failure_maps_to_128_plus_the_signal(self):
        # [FUNCTION DOC] A single failure by SIGKILL yields 128 plus SIGKILL (137).
        """A rank dying of SIGKILL surfaces as 137, the shell convention."""
        failures = {0: _failure(exitcode=-int(signal.SIGKILL), timestamp=10)}
        assert _exit_code_from_result(_run_result(failures)) == 128 + signal.SIGKILL

    def test_the_highest_exit_code_is_reported_unchanged(self):
        # [FUNCTION DOC] A single failure with exit code 255 yields 255 (not masked to seven bits).
        """Exit code 255 must not be masked down into the low seven bits."""
        failures = {0: _failure(exitcode=255, timestamp=10)}
        assert _exit_code_from_result(_run_result(failures)) == 255


class TestSpawning:
    # [CLASS DOC] Verifies what supervise hands to torchelastic's start_processes, using a recording
    # replacement (_record_start_processes) that returns a fake context finishing cleanly.
    @pytest.mark.parametrize("count", [0, -1])
    def test_a_non_positive_subprocess_count_is_refused(self, count):
        # [FUNCTION DOC] Parametrized over zero and minus one; calling supervise directly with such
        # a count must raise a value error before anything is spawned.
        """A count derived from an empty config would otherwise report success without running anything."""
        with pytest.raises(ValueError):
            supervise(command=["sleep", "1"], num_subprocesses=count)

    def test_every_subprocess_is_started_with_its_own_index(self, monkeypatch):
        # [FUNCTION DOC] Supervises 3 copies and asserts the recorded environments mapping is exactly
        # rank 0, 1, 2 each holding only the subprocess index variable set to its own index string.
        """Each rank has to be told which of the N copies it is, through the environment."""
        started = _record_start_processes(monkeypatch)

        supervise(command=["sleep", "1"], num_subprocesses=3)

        assert started[0]["envs"] == {
            0: {SUBPROCESS_INDEX_ENV_VAR: "0"},
            1: {SUBPROCESS_INDEX_ENV_VAR: "1"},
            2: {SUBPROCESS_INDEX_ENV_VAR: "2"},
        }

    def test_the_supervised_command_is_spawned_directly(self, monkeypatch):
        # [FUNCTION DOC] Supervises 2 copies of "sleep 1"; asserts the entrypoint is the program name
        # alone and each rank's arguments are a tuple holding the remaining words (no wrapper).
        """No wrapper sits between torchelastic and the command, so the pid it logs is the command's."""
        started = _record_start_processes(monkeypatch)

        supervise(command=["sleep", "1"], num_subprocesses=2)

        assert started[0]["entrypoint"] == "sleep"
        assert started[0]["args"] == {0: ("1",), 1: ("1",)}

    def test_both_standard_streams_are_teed_back_with_rank_prefixes(self, monkeypatch):
        # [FUNCTION DOC] Supervises 2 copies and asserts the log specs object is torchelastic's
        # default log specs, the line prefixes are "[rank0] " and "[rank1] ", and, when the specs
        # are resolved for two ranks, both ranks tee both standard output and standard error.
        """The pod log interleaves every rank's output, so each line must say which rank wrote it."""
        started = _record_start_processes(monkeypatch)

        supervise(command=["sleep", "1"], num_subprocesses=2)

        assert isinstance(started[0]["logs_specs"], DefaultLogsSpecs)
        assert started[0]["log_line_prefixes"] == {0: "[rank0] ", 1: "[rank1] "}
        destinations = started[0]["logs_specs"].reify({0: {}, 1: {}})
        assert sorted(destinations.tee_stdouts) == [0, 1]
        assert sorted(destinations.tee_stderrs) == [0, 1]

    def test_the_signals_torchelastic_handles_are_pinned_before_it_reads_them(self, monkeypatch):
        # [FUNCTION DOC] Pre-sets torchelastic's signals-to-handle environment variable to SIGHUP,
        # supervises one copy, and asserts the value seen inside start_processes (captured at call
        # time by the recorder) is "SIGTERM,SIGINT", i.e. supervise overrides it before spawning.
        """torchelastic reads this variable while spawning, so pinning it afterwards would be a no-op."""
        started = _record_start_processes(monkeypatch)
        monkeypatch.setenv(_SIGNALS_TO_HANDLE_ENV_VAR, "SIGHUP")

        supervise(command=["sleep", "1"], num_subprocesses=1)

        assert started[0]["signals_to_handle"] == "SIGTERM,SIGINT"

    def test_each_spawned_subprocess_is_logged_under_its_own_index(self, monkeypatch, caplog):
        # [FUNCTION DOC] Runs supervise over a fake context with two pids and captures info logs of
        # the supervisor module; asserts a "Spawned subprocess index=<i> pid=<pid>" line per rank.
        """An operator maps a rank to a pid using this line and nothing else."""
        context = _fake_context(num_subprocesses=2, wait_outcome=_run_result())

        with caplog.at_level(logging.INFO, logger=supervisor_module.__name__):
            _run(monkeypatch, context=context, num_subprocesses=2)

        assert f"Spawned subprocess index=0 pid={CHILD_PID}" in caplog.text
        assert f"Spawned subprocess index=1 pid={OTHER_CHILD_PID}" in caplog.text


class TestSubprocessExit:
    # [CLASS DOC] Verifies how supervise turns the outcome of waiting on the (fake) torchelastic
    # context into an exit code, log lines and a teardown call.
    def test_every_subprocess_exiting_cleanly_yields_zero(self, monkeypatch):
        # [FUNCTION DOC] The fake context's wait returns a result without failures; supervise
        # returns 0.
        """torchelastic waits for the whole batch, and a clean batch is a clean pod."""
        context = _fake_context(wait_outcome=_run_result())

        assert _run(monkeypatch, context=context) == 0

    def test_the_exit_code_of_the_earliest_failure_is_returned(self, monkeypatch):
        # [FUNCTION DOC] Two failures with the same timestamp: rank 1 exiting 7 (listed first) and
        # rank 0 killed by SIGTERM; with two subprocesses supervise must return 7, the code of the
        # first entry of the failure mapping (timestamps are not consulted).
        """The rank that died first is the cause; the ones torn down after it are consequences."""
        failures = {
            1: _failure(local_rank=1, exitcode=7, timestamp=10),
            0: _failure(exitcode=-int(signal.SIGTERM), timestamp=10),
        }
        context = _fake_context(num_subprocesses=2, wait_outcome=_run_result(failures))

        assert _run(monkeypatch, context=context, num_subprocesses=2) == 7

    def test_a_subprocess_killed_by_a_signal_is_reported_as_128_plus_the_signal(self, monkeypatch):
        # [FUNCTION DOC] One failure by SIGKILL; supervise returns 128 plus SIGKILL.
        """Popen reports a signalled death as a negative number, which no shell would understand."""
        context = _fake_context(wait_outcome=_run_result({0: _failure(exitcode=-int(signal.SIGKILL))}))

        assert _run(monkeypatch, context=context) == 128 + signal.SIGKILL

    def test_an_unexpected_failure_tears_the_batch_down_before_propagating(self, monkeypatch, caplog):
        # [FUNCTION DOC] The fake wait raises a runtime error; supervise (grace period 2 seconds)
        # must re-raise it, call close exactly once with SIGTERM and 2.0 seconds, and log at error
        # level with exception information attached.
        """Leaving the tee threads running would hang the container instead of failing the pod."""
        context = _fake_context(wait_outcome=RuntimeError("torchelastic bookkeeping blew up"))

        with caplog.at_level(logging.ERROR, logger=supervisor_module.__name__), pytest.raises(RuntimeError):
            _run(monkeypatch, context=context, termination_grace_period_seconds=2.0)

        assert context.close_calls == [(signal.SIGTERM, 2.0)]
        assert any(record.exc_info is not None for record in caplog.records)

    def test_a_failure_that_is_not_an_exception_still_tears_the_batch_down(self, monkeypatch):
        # [FUNCTION DOC] The fake wait raises a keyboard interrupt (a base exception, not a regular
        # exception); supervise must let it propagate but still call close once with SIGTERM and
        # the 2.0 second grace period.
        """SystemExit and KeyboardInterrupt skip an `except Exception`, leaving the tee threads alive."""
        context = _fake_context(wait_outcome=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            _run(monkeypatch, context=context, termination_grace_period_seconds=2.0)

        assert context.close_calls == [(signal.SIGTERM, 2.0)]

    def test_every_failure_is_logged_with_its_index_and_pid(self, monkeypatch, caplog):
        # [FUNCTION DOC] One failure (rank 0, exit code 7) at warning level; asserts the log holds
        # "Subprocess index=0 pid=<pid> exited with 7".
        """An operator reads these lines to find out which rank went down and how."""
        context = _fake_context(wait_outcome=_run_result({0: _failure(exitcode=7)}))

        with caplog.at_level(logging.WARNING, logger=supervisor_module.__name__):
            _run(monkeypatch, context=context)

        assert f"Subprocess index=0 pid={CHILD_PID} exited with 7" in caplog.text


class TestSignalForwarding:
    # [CLASS DOC] Verifies the in-process handling of a forwarded signal: torchelastic raises a
    # signal exception from wait (or from spawning), and supervise tears the batch down with that
    # signal. Signal handlers are restored after each test by the autouse restored_signal_handlers.
    def test_a_forwarded_signal_decides_the_exit_code(self, monkeypatch):
        # [FUNCTION DOC] wait raises a signal exception for SIGTERM; supervise returns 128 plus
        # SIGTERM (143).
        """A pod terminated by the kubelet exits the way a shell would report the signal."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))

        assert _run(monkeypatch, context=context) == 128 + signal.SIGTERM

    def test_the_received_signal_and_the_grace_period_shape_the_teardown(self, monkeypatch):
        # [FUNCTION DOC] wait raises a signal exception for SIGINT with a 2.0 second grace period;
        # asserts close is called once with SIGINT (not SIGTERM) and 2.0.
        """Forwarding SIGTERM for a SIGINT would deny the subprocesses their own Ctrl-C handling."""
        context = _fake_context(wait_outcome=SignalException("got SIGINT", sigval=signal.SIGINT))

        _run(monkeypatch, context=context, termination_grace_period_seconds=2.0)

        assert context.close_calls == [(signal.SIGINT, 2.0)]

    def test_further_forwarded_signals_are_reported_instead_of_interrupting(self, monkeypatch):
        # [FUNCTION DOC] After a SIGTERM-driven teardown, asserts that the handlers installed for
        # SIGTERM and SIGINT while close ran were both the module's report-only handler, then calls
        # that handler once for SIGINT with standard error redirected to the null device, checking
        # it returns normally instead of raising.
        """torchelastic leaves its handler installed, so a repeated signal would raise inside close."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))

        _run(monkeypatch, context=context)

        installed = context.handlers_during_close[0]
        assert installed == (_report_signal_received_during_teardown, _report_signal_received_during_teardown)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            installed[0](int(signal.SIGINT), None)

    def test_the_repeat_notice_reaches_stderr_without_taking_a_lock(self, capfd):
        # [FUNCTION DOC] Reads the source text of the report-only handler, invokes it for SIGTERM,
        # and asserts file-descriptor-level standard error contains "SIGTERM while already tearing
        # down" and that the handler source never mentions a logger (it must write directly).
        """A handler that goes through logging drops the line, or nests, when signals arrive in bursts."""
        source = inspect.getsource(_report_signal_received_during_teardown)

        _report_signal_received_during_teardown(int(signal.SIGTERM), None)

        assert "SIGTERM while already tearing down" in capfd.readouterr().err
        assert "logger" not in source

    def test_the_handlers_are_replaced_before_anything_else_in_the_teardown(self, monkeypatch):
        # [FUNCTION DOC] Replaces the supervisor logger's info method with a recorder that notes the
        # current SIGTERM handler, runs a SIGTERM-driven teardown, and asserts the handler seen at
        # the last info log is already the report-only handler (swap happens before logging).
        """Every instruction between torchelastic raising and the swap is exposed to a second signal."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))
        handlers_when_logging = []

        def record(message, *args) -> None:
            # [FUNCTION DOC] HELPER: Stand-in for the logger info method; ignores the message and
            # appends the current SIGTERM handler to the enclosing list. Returns nothing.
            handlers_when_logging.append(signal.getsignal(signal.SIGTERM))

        monkeypatch.setattr(supervisor_module.logger, "info", record)

        _run(monkeypatch, context=context)

        assert handlers_when_logging[-1] is _report_signal_received_during_teardown

    def test_a_signal_arriving_while_spawning_still_decides_the_exit_code(self, monkeypatch, caplog):
        # [FUNCTION DOC] start_processes itself raises a signal exception for SIGTERM (simulating a
        # signal during spawning). Asserts supervise returns 143, logs "Received SIGTERM while still
        # spawning" and does not log the "forwarding it to every subprocess" line (nothing to
        # forward to yet).
        """torchelastic installs its handler before the last rank is spawned, so the signal lands here."""

        def start_processes(**kwargs) -> object:
            # [FUNCTION DOC] HELPER: Replacement for start_processes that always raises the signal
            # exception for SIGTERM; accepts and ignores any keyword arguments.
            raise SignalException("got SIGTERM", sigval=signal.SIGTERM)

        monkeypatch.setattr(supervisor_module, "start_processes", start_processes)

        with caplog.at_level(logging.INFO, logger=supervisor_module.__name__):
            assert supervise(command=["true"], num_subprocesses=2) == 128 + signal.SIGTERM

        assert "Received SIGTERM while still spawning" in caplog.text
        assert "forwarding it to every subprocess" not in caplog.text

    def test_the_forwarding_is_announced_in_the_log(self, monkeypatch, caplog):
        # [FUNCTION DOC] SIGTERM-driven teardown at info level; asserts the log contains "Received
        # SIGTERM, forwarding it to every subprocess".
        """An operator reconstructs why a pod died from this line."""
        context = _fake_context(wait_outcome=SignalException("got SIGTERM", sigval=signal.SIGTERM))

        with caplog.at_level(logging.INFO, logger=supervisor_module.__name__):
            _run(monkeypatch, context=context)

        assert "Received SIGTERM, forwarding it to every subprocess" in caplog.text


def _run(
    monkeypatch,
    *,
    context,
    num_subprocesses: int = 1,
    termination_grace_period_seconds: float = 1.0,
) -> int:
    # [FUNCTION DOC] HELPER: Makes start_processes in the supervisor module return the given fake
    # context, then calls supervise with the command "true", the given count (default 1) and grace
    # period (default 1.0 second). Returns supervise's exit code; propagates whatever it raises.
    monkeypatch.setattr(supervisor_module, "start_processes", lambda **kwargs: context)
    return supervise(
        command=["true"],
        num_subprocesses=num_subprocesses,
        termination_grace_period_seconds=termination_grace_period_seconds,
    )


def _record_start_processes(monkeypatch) -> list[dict]:
    # [FUNCTION DOC] HELPER: Replaces start_processes in the supervisor module with a recorder and
    # returns the list it appends to (one dict per call: all keyword arguments plus the value of
    # torchelastic's signals-to-handle environment variable observed at call time, or none).
    started: list[dict] = []

    def start_processes(**kwargs) -> object:
        # [FUNCTION DOC] HELPER: Recording stand-in; appends the keyword arguments and the current
        # environment value, and returns a fake context sized to the environments mapping whose wait
        # yields a clean result.
        started.append({**kwargs, "signals_to_handle": os.environ.get(_SIGNALS_TO_HANDLE_ENV_VAR)})
        return _fake_context(num_subprocesses=len(kwargs["envs"]), wait_outcome=_run_result())

    monkeypatch.setattr(supervisor_module, "start_processes", start_processes)
    return started


class TestHappyPath:
    # [CLASS DOC] End-to-end test with real processes: the supervisor subprocess runs copies of this
    # file as the child.
    def test_every_copy_runs_to_completion_and_the_supervisor_exits_zero(self, launch_supervisor):
        # [FUNCTION DOC] Launches 3 children in exit mode with code 0 that wait for all 3 reports.
        # Asserts the supervisor exits 0, the reports are keyed "0", "1", "2" and each reports the
        # exit mode.
        """Each copy sees its own index, finishes with exit code 0, and the supervisor reports success."""
        run = launch_supervisor(
            num_subprocesses=3,
            child_args=["--mode", "exit", "--exit-code", "0", "--wait-for-reports", "3"],
        )

        reports = run.wait_for_children(3)

        assert run.wait_for_exit() == 0
        assert sorted(reports) == ["0", "1", "2"]
        assert _modes(reports) == {"0": "exit", "1": "exit", "2": "exit"}


class TestSubprocessLaunching:
    # [CLASS DOC] Real-process tests of how the supervisor spawns the children (indices, sessions,
    # environment, stream prefixing, spawn failure) plus two contract checks on constants and
    # logging configuration that need no process.
    def test_every_subprocess_gets_its_own_index_and_session(self, launch_supervisor):
        # [FUNCTION DOC] Launches 3 sleeping children and waits for their reports. Asserts indices
        # are exactly "0", "1", "2", the pids are all distinct, and every child is the leader of its
        # own process group and session (group id and session id equal its pid).
        """Every index in 0..N-1 shows up exactly once, on a distinct pid that leads its own session."""
        run = launch_supervisor(num_subprocesses=3, child_args=["--mode", "sleep"])

        reports = run.wait_for_children(3)

        assert sorted(reports) == ["0", "1", "2"]
        assert len({report["pid"] for report in reports.values()}) == 3
        assert all(report["process_group_id"] == report["pid"] for report in reports.values())
        assert all(report["session_id"] == report["pid"] for report in reports.values())

    def test_the_external_contract_stays_where_consumers_expect_it(self):
        # [FUNCTION DOC] Asserts the index environment variable name equals the documented
        # MILES_SUPERVISOR_SUBPROCESS_INDEX, the default grace period equals the documented 20
        # seconds, and that default is smaller than Kubernetes' own 30 second default.
        """Both values are part of the pod contract, so drifting from them is a breaking change."""
        assert SUBPROCESS_INDEX_ENV_VAR == DOCUMENTED_SUBPROCESS_INDEX_ENV_VAR
        assert _DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS == DOCUMENTED_DEFAULT_GRACE_PERIOD_SECONDS
        assert _DEFAULT_TERMINATION_GRACE_PERIOD_SECONDS < KUBERNETES_DEFAULT_GRACE_PERIOD_SECONDS

    def test_an_index_inherited_from_the_parent_is_overwritten(self, launch_supervisor):
        # [FUNCTION DOC] Launches 2 children with the index variable pre-set to "poison" in the
        # supervisor's environment; asserts the reported indices are still "0" and "1".
        """A stale index left in the pod environment must not leak into any subprocess."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "sleep"],
            extra_env={DOCUMENTED_SUBPROCESS_INDEX_ENV_VAR: "poison"},
        )

        reports = run.wait_for_children(2)

        assert sorted(reports) == ["0", "1"]

    def test_both_standard_streams_reach_the_container_log_with_rank_prefixes(self, launch_supervisor):
        # [FUNCTION DOC] Launches 2 children that print one marker line to each standard stream and
        # exit; captures the supervisor's own output and asserts standard output contains the
        # "[rank<i>] " prefixed output marker and standard error the prefixed error marker for both
        # ranks.
        """Container logs interleave every rank's streams, so each line carries the rank that wrote it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "exit", "--emit-stream-markers", "--wait-for-reports", "2"],
            capture_output=True,
        )

        stdout, stderr = run.process.communicate(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)

        for index in ("0", "1"):
            assert f"[rank{index}] {STDOUT_MARKER}{index}" in stdout
            assert f"[rank{index}] {STDERR_MARKER}{index}" in stderr

    def test_a_single_subprocess_is_supervised_like_any_other_count(self, launch_supervisor):
        # [FUNCTION DOC] One child exits with code 5; asserts the supervisor exits 5 and exactly one
        # report (index "0") exists.
        """The degenerate one-rank-per-pod case still spawns, supervises and propagates."""
        run = launch_supervisor(
            num_subprocesses=1,
            child_args=["--mode", "exit", "--exit-code", "5"],
        )

        assert run.wait_for_exit() == 5
        assert sorted(run.reports()) == ["0"]

    def test_the_supervisor_environment_reaches_every_subprocess(self, launch_supervisor):
        # [FUNCTION DOC] Sets a marker environment variable for the supervisor, launches 2 sleeping
        # children and asserts both report seeing the marker value (environment is inherited).
        """The index var is added to the inherited environment rather than replacing it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "sleep"],
            extra_env={ENV_MARKER_VAR: "inherited"},
        )

        reports = run.wait_for_children(2)

        assert [report["env_marker"] for report in reports.values()] == ["inherited", "inherited"]

    def test_keeps_running_while_every_subprocess_is_alive(self, launch_supervisor):
        # [FUNCTION DOC] Launches 2 sleeping children, sleeps half a second after both report, and
        # asserts the supervisor process has not exited (poll still returns nothing).
        """The supervisor does not exit on its own as long as no subprocess has exited."""
        run = launch_supervisor(num_subprocesses=2, child_args=["--mode", "sleep"])
        run.wait_for_children(2)

        time.sleep(0.5)

        assert run.process.poll() is None

    def test_a_command_that_cannot_be_spawned_at_all_exits_nonzero(self, launch_supervisor):
        # [FUNCTION DOC] Supervises a non-existent executable path; asserts the supervisor exits
        # with code 1 and no child ever reported.
        """A typo in the entrypoint must fail the pod immediately rather than hang it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=[],
            command=[str(REPO_ROOT / "definitely-not-an-executable")],
        )

        assert run.wait_for_exit() == 1
        assert run.reports() == {}

    def test_the_container_log_is_configured_to_be_readable(self, monkeypatch):
        # [FUNCTION DOC] Replaces the logging basic configuration function with a recorder and stubs
        # supervise; runs main and asserts basic configuration was called exactly once at info level
        # with a format containing the time, the "process_supervisor" name and the level name.
        """These lines are interleaved with every rank's own output, so they must stay identifiable."""
        configured: list[dict] = []
        monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: configured.append(kwargs))
        _record_supervisor(monkeypatch)

        main(["--num-subprocesses", "1", "--", "sleep", "1"])

        assert [kwargs["level"] for kwargs in configured] == [logging.INFO]
        assert all(part in configured[0]["format"] for part in ("%(asctime)s", "process_supervisor", "%(levelname)s"))


class TestSupervisedSubprocessExit:
    # [CLASS DOC] Real-process tests of what happens when one child exits or dies while the others
    # keep running.
    def test_propagates_the_exit_code_of_the_one_subprocess_that_exits(self, launch_supervisor):
        # [FUNCTION DOC] Launches 3 children where only index 1 exits with code 7 (the others sleep)
        # and all record any signal. Asserts the supervisor exits 7, the reported modes are sleep,
        # exit, sleep, both survivors eventually record SIGTERM (forwarded teardown), and all three
        # pids end up not alive.
        """The exit code of the rank that dies is what the container exits with."""
        run = launch_supervisor(
            num_subprocesses=3,
            child_args=[
                "--mode",
                "exit",
                "--only-index",
                "1",
                "--exit-code",
                "7",
                "--record-signal",
                "--wait-for-reports",
                "3",
            ],
        )
        reports = run.wait_for_children(3)

        assert run.wait_for_exit() == 7
        assert _modes(reports) == {"0": "sleep", "1": "exit", "2": "sleep"}
        wait_until(
            lambda: {path.name for path in run.report_dir.glob("signal-*")} >= {"signal-0", "signal-2"},
            message="the survivors to record the forwarded SIGTERM",
        )
        assert int((run.report_dir / "signal-0").read_text()) == signal.SIGTERM
        assert int((run.report_dir / "signal-2").read_text()) == signal.SIGTERM
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be killed")

    def test_a_successful_exit_leaves_the_others_running(self, launch_supervisor):
        # [FUNCTION DOC] Only index 0 exits with code 0 (index 1 sleeps). After index 0 is gone and
        # 2.5 more seconds pass, asserts the supervisor is still running and index 1 is still alive.
        """A rank finishing cleanly is not a failure, so the rest of the batch keeps running without it."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "exit", "--only-index", "0", "--exit-code", "0", "--wait-for-reports", "2"],
        )
        reports = run.wait_for_children(2)
        wait_until(lambda: not is_alive(reports["0"]["pid"]), message="the exiting subprocess to be gone")

        time.sleep(2.5)

        assert run.process.poll() is None
        assert is_alive(reports["1"]["pid"])

    def test_a_subprocess_killed_by_a_signal_maps_to_128_plus_the_signal(self, launch_supervisor):
        # [FUNCTION DOC] Index 0 kills itself with SIGKILL, index 1 sleeps. Asserts the supervisor
        # exits 137 (128 plus SIGKILL) and the modes reported are suicide and sleep.
        """A subprocess dying from SIGKILL surfaces as exit code 137, the shell convention."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "suicide", "--only-index", "0", "--wait-for-reports", "2"],
        )
        reports = run.wait_for_children(2)

        assert run.wait_for_exit() == 128 + signal.SIGKILL
        assert _modes(reports) == {"0": "suicide", "1": "sleep"}


class TestSignalHandling:
    # [CLASS DOC] Real-process tests sending signals to the supervisor and checking forwarding,
    # process-group scope, SIGKILL escalation after the grace period and robustness.
    def test_sigterm_is_forwarded_to_every_subprocess(self, launch_supervisor):
        # [FUNCTION DOC] Sends SIGTERM to the supervisor with 2 signal-recording children; asserts
        # exit code 143 and that both children recorded SIGTERM.
        """SIGTERM sent to the supervisor reaches every subprocess rather than stopping at it."""
        run = launch_supervisor(num_subprocesses=2, child_args=["--mode", "record-signal"])
        run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        assert _recorded_signals(run) == {"0": signal.SIGTERM, "1": signal.SIGTERM}

    def test_sigint_is_forwarded_to_every_subprocess(self, launch_supervisor):
        # [FUNCTION DOC] Sends SIGINT to the supervisor; asserts exit code 130 and that both
        # children recorded SIGINT (not SIGTERM).
        """SIGINT is forwarded as SIGINT rather than translated to SIGTERM, and yields exit code 130."""
        run = launch_supervisor(num_subprocesses=2, child_args=["--mode", "record-signal"])
        run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGINT)

        assert run.wait_for_exit() == 128 + signal.SIGINT
        assert _recorded_signals(run) == {"0": signal.SIGINT, "1": signal.SIGINT}

    def test_grandchildren_of_a_running_subprocess_die_with_its_group(self, launch_supervisor):
        # [FUNCTION DOC] One sleeping child spawns a grandchild and reports its pid. After SIGTERM
        # to the supervisor, asserts exit code 143 and that the grandchild eventually dies.
        """Forwarding is per process group, so a process spawned by a subprocess is taken down too."""
        run = launch_supervisor(num_subprocesses=1, child_args=["--mode", "sleep", "--spawn-grandchild"])
        reports = run.wait_for_children(1)
        grandchild_pid = reports["0"]["grandchild_pid"]

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        wait_until(lambda: not is_alive(grandchild_pid), message="the grandchild to be killed")

    def test_a_subprocess_ignoring_sigterm_is_killed_after_the_grace_period(self, launch_supervisor):
        # [FUNCTION DOC] Two children ignore SIGTERM; the supervisor has a 0.5 second grace period.
        # After SIGTERM, asserts exit code 143, elapsed time between the grace period and the grace
        # period plus 10 seconds, and that both children are eventually dead (SIGKILL escalation).
        """A subprocess that ignores SIGTERM is SIGKILLed once the grace period expires."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "ignore-sigterm"],
            termination_grace_period_seconds=GRACE_PERIOD_SECONDS,
        )
        reports = run.wait_for_children(2)
        started_at = time.monotonic()

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        elapsed = time.monotonic() - started_at
        assert GRACE_PERIOD_SECONDS <= elapsed <= GRACE_PERIOD_SECONDS + GRACE_PERIOD_UPPER_BOUND_SLACK_SECONDS
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be killed")

    def test_an_unreadable_torchelastic_error_file_fails_the_pod_instead_of_hanging_it(self, launch_supervisor):
        # [FUNCTION DOC] Index 0 writes invalid JSON to torchelastic's error file and exits 1 while
        # index 1 sleeps. Asserts the supervisor exits with a non-zero code (instead of hanging) and
        # both children end up gone.
        """torchelastic reads that file while bookkeeping a failure, before it stops the tee threads."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=[
                "--mode",
                "exit",
                "--only-index",
                "0",
                "--exit-code",
                "1",
                "--corrupt-error-file",
                "--wait-for-reports",
                "2",
            ],
        )
        reports = run.wait_for_children(2)

        assert run.wait_for_exit() != 0
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be gone")

    def test_an_inherited_signal_list_cannot_disable_the_forwarding(self, launch_supervisor):
        # [FUNCTION DOC] Starts the supervisor with torchelastic's signals-to-handle variable set to
        # SIGUSR1; after SIGTERM asserts exit code 143 and both children recorded SIGTERM.
        """A stale value for torchelastic's own variable would kill the supervisor without forwarding."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "record-signal"],
            extra_env={_SIGNALS_TO_HANDLE_ENV_VAR: "SIGUSR1"},
        )
        run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGTERM)

        assert run.wait_for_exit() == 128 + signal.SIGTERM
        assert _recorded_signals(run) == {"0": signal.SIGTERM, "1": signal.SIGTERM}

    def test_a_second_signal_during_the_grace_period_does_not_abort_the_teardown(self, launch_supervisor):
        # [FUNCTION DOC] Two SIGTERM-ignoring children, 3 second grace period, supervisor output
        # captured. Sends SIGTERM twice, half a second apart, then waits for the supervisor. Asserts
        # return code 143, that standard error says "Received SIGTERM while already tearing down",
        # and that both children are eventually killed.
        """Two SIGTERMs are ordinary, and the second must not leave the batch half torn down."""
        run = launch_supervisor(
            num_subprocesses=2,
            child_args=["--mode", "ignore-sigterm"],
            termination_grace_period_seconds=DOUBLE_SIGNAL_GRACE_PERIOD_SECONDS,
            capture_output=True,
        )
        reports = run.wait_for_children(2)

        os.kill(run.process.pid, signal.SIGTERM)
        time.sleep(SECOND_SIGNAL_DELAY_SECONDS)
        os.kill(run.process.pid, signal.SIGTERM)
        _, stderr = run.process.communicate(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)

        assert run.process.returncode == 128 + signal.SIGTERM
        assert "Received SIGTERM while already tearing down" in stderr
        for index, report in reports.items():
            wait_until(lambda pid=report["pid"]: not is_alive(pid), message=f"subprocess {index} to be killed")


@pytest.fixture
def launch_supervisor(tmp_path: Path) -> Iterator[Callable[..., SupervisorRun]]:
    # [FUNCTION DOC] HELPER: Fixture yielding a launch function; the temp directory is the shared
    # report directory. After the test, teardown walks every launched run: sends SIGTERM to the
    # supervisor's process group and waits up to 10 seconds, then SIGKILLs that group, SIGKILLs the
    # group of every reporting child and any grandchild, waits (30 second timeout) for the
    # supervisor and closes any captured output pipes. Ignores already-gone processes.
    runs: list[SupervisorRun] = []

    def launch(
        *,
        num_subprocesses: int,
        child_args: list[str],
        termination_grace_period_seconds: float = 5.0,
        command: list[str] | None = None,
        extra_env: dict[str, str] | None = None,
        capture_output: bool = False,
    ) -> SupervisorRun:
        # [FUNCTION DOC] HELPER: Starts the supervisor module as a subprocess in a new session from
        # the repository root with the repository on the python path and extra environment merged
        # over the current one. The supervised command defaults to this file run with the report
        # directory and child_args; an explicit command replaces it. The grace period defaults to 5
        # seconds. Standard output and error are piped (text mode) only when capture_output is true,
        # otherwise inherited. Registers and returns the run record.
        supervised = command or [sys.executable, str(CHILD_SCRIPT), "--report-dir", str(tmp_path), *child_args]
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "miles.utils.workers.process_supervisor",
                "--num-subprocesses",
                str(num_subprocesses),
                "--termination-grace-period-seconds",
                str(termination_grace_period_seconds),
                "--",
                *supervised,
            ],
            cwd=REPO_ROOT,
            env={**_repo_environment(), **(extra_env or {})},
            start_new_session=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            text=True,
        )
        run = SupervisorRun(process=process, report_dir=tmp_path)
        runs.append(run)
        return run

    yield launch

    for run in runs:
        signal_process_group(run.process.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            run.process.wait(timeout=SUPERVISOR_SHUTDOWN_TIMEOUT_SECONDS)

        signal_process_group(run.process.pid, signal.SIGKILL)
        for report in run.reports().values():
            signal_process_group(report["pid"], signal.SIGKILL)
            if "grandchild_pid" in report:
                kill_process(report["grandchild_pid"])
        run.process.wait(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)
        for stream in (run.process.stdout, run.process.stderr):
            if stream is not None and not stream.closed:
                stream.close()


@pytest.fixture(autouse=True)
def restored_signal_handlers() -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Autouse fixture; saves the current SIGTERM and SIGINT handlers before
    # each test and reinstalls them afterwards, since supervise swaps handlers in-process.
    saved = {forwarded: signal.getsignal(forwarded) for forwarded in (signal.SIGTERM, signal.SIGINT)}

    yield

    for forwarded, handler in saved.items():
        signal.signal(forwarded, handler)


@pytest.fixture
def unconfigured_root_logging(monkeypatch) -> None:
    # [FUNCTION DOC] HELPER: Fixture making the logging basic configuration function a no-op for
    # the test, so calling main does not reconfigure the root logger of the test session.
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)


@dataclass
class SupervisorRun:
    # [CLASS DOC] HELPER: Handle on one launched supervisor: the Popen object and the directory
    # where children write their reports and signal records.
    process: subprocess.Popen
    report_dir: Path

    def wait_for_children(self, count: int) -> dict[str, dict]:
        # [FUNCTION DOC] HELPER: Polls until exactly count child reports exist (30 second timeout,
        # raising an assertion error otherwise) and returns them, keyed by index string.
        wait_until(lambda: len(self.reports()) == count, message=f"{count} children to report in")
        return self.reports()

    def reports(self) -> dict[str, dict]:
        # [FUNCTION DOC] HELPER: Reads every child report file in the report directory (sorted by
        # file name) and returns a mapping from the index string in the file name to the decoded
        # JSON content. Reads the file system only.
        reports = {}
        for path in sorted(self.report_dir.glob("child-*.json")):
            reports[path.stem.removeprefix("child-")] = json.loads(path.read_text())
        return reports

    def wait_for_exit(self) -> int:
        # [FUNCTION DOC] HELPER: Waits up to 30 seconds for the supervisor to exit and returns its
        # return code; raises a timeout error from the process wait otherwise.
        return self.process.wait(timeout=SUPERVISOR_EXIT_TIMEOUT_SECONDS)


def _repo_environment() -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns a copy of the current environment whose python path has the
    # repository root prepended (joined with the existing python path, which may leave a trailing
    # empty entry when none was set).
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]),
    }


def wait_until(predicate: Callable[[], bool], *, message: str, timeout: float = 30.0) -> None:
    # [FUNCTION DOC] HELPER: Polls the predicate every 20 milliseconds until it is true; returns
    # then. After timeout seconds (default 30) raises an assertion error naming the message.
    # KNOWN ISSUE: near-duplicate of _wait_until below, which raises a timeout error instead and
    # has a fixed timeout.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)

    raise AssertionError(f"Timed out after {timeout}s waiting for {message}")


def is_alive(pid: int) -> bool:
    # [FUNCTION DOC] HELPER: True when a process with this pid exists and is not a zombie. Sends the
    # null signal; a missing process gives false, a permission error counts as alive (and is
    # returned true without the zombie check).
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not is_zombie(pid)


def is_zombie(pid: int) -> bool:
    # [FUNCTION DOC] HELPER: True when the process is in the zombie state. Uses the proc file
    # system status file when a proc directory exists (a vanished file means not a zombie),
    # otherwise falls back to running the ps command and checking the state column.
    proc_status = Path(f"/proc/{pid}/status")
    if proc_status.parent.parent.is_dir():
        try:
            return "State:\tZ" in proc_status.read_text()
        except FileNotFoundError:
            return False

    listing = subprocess.run(["ps", "-o", "state=", "-p", str(pid)], capture_output=True, text=True)
    return listing.stdout.strip().startswith("Z")


def signal_process_group(pid: int, sent_signal: signal.Signals) -> None:
    # [FUNCTION DOC] HELPER: Sends the signal to the process group whose id is pid; silently
    # ignores a missing group or a permission error.
    try:
        os.killpg(pid, sent_signal)
    except (ProcessLookupError, PermissionError):
        pass


def kill_process(pid: int) -> None:
    # [FUNCTION DOC] HELPER: Sends SIGKILL to a single process; silently ignores a missing process
    # or a permission error.
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _fake_context(
    *,
    num_subprocesses: int = 1,
    wait_outcome: RunProcsResult | BaseException | None = None,
) -> _FakeContext:
    # [FUNCTION DOC] HELPER: Builds a fake torchelastic context with num_subprocesses ranks whose
    # pids are consecutive starting at 4002; wait will return or raise the given outcome.
    pids = {index: CHILD_PID + index for index in range(num_subprocesses)}
    return _FakeContext(pids_by_index=pids, wait_outcome=wait_outcome)


def _run_result(failures: dict[int, ProcessFailure] | None = None) -> RunProcsResult:
    # [FUNCTION DOC] HELPER: Builds a torchelastic run result from the failures mapping (empty when
    # none is given).
    return RunProcsResult(failures=failures or {})


def _failure(*, local_rank: int = 0, exitcode: int, timestamp: int = 0) -> ProcessFailure:
    # [FUNCTION DOC] HELPER: Builds a torchelastic process failure record for the rank (pid is
    # 4002 plus the rank, error file is the null device) with the given exit code, then overrides
    # its timestamp (whole seconds) with the given value.
    failure = ProcessFailure(
        local_rank=local_rank,
        pid=CHILD_PID + local_rank,
        exitcode=exitcode,
        error_file=os.devnull,
    )
    failure.timestamp = timestamp
    return failure


def _record_supervisor(monkeypatch) -> list[dict]:
    # [FUNCTION DOC] HELPER: Replaces supervise in the supervisor module with a recorder and
    # returns the list of recorded keyword-argument dicts.
    created: list[dict] = []

    def supervise(**kwargs) -> int:
        # [FUNCTION DOC] HELPER: Recorder; appends the keyword arguments and returns the fixed exit
        # code 17.
        created.append(kwargs)
        return RECORDED_SUPERVISOR_EXIT_CODE

    monkeypatch.setattr(supervisor_module, "supervise", supervise)
    return created


def _modes(reports: dict[str, dict]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Maps each report's index to the mode recorded in that report.
    return {index: report["mode"] for index, report in reports.items()}


def _recorded_signals(run: SupervisorRun) -> dict[str, int]:
    # [FUNCTION DOC] HELPER: Waits until as many signal record files exist as there are child
    # reports, then returns a mapping from index string to the recorded signal number.
    expected = len(run.reports())
    wait_until(
        lambda: len(list(run.report_dir.glob("signal-*"))) == expected,
        message="every subprocess to record the signal it received",
    )
    return {path.name.removeprefix("signal-"): int(path.read_text()) for path in run.report_dir.glob("signal-*")}


@dataclass
class _FakeContext:
    # [CLASS DOC] HELPER: Stand-in for torchelastic's process context. Holds a pid per rank, the
    # configured outcome of wait, and records every close call plus the SIGTERM/SIGINT handlers
    # that were installed at the moment of each close call.
    pids_by_index: dict[int, int]
    wait_outcome: RunProcsResult | BaseException | None = None
    close_calls: list[tuple[signal.Signals | None, float]] = field(default_factory=list)
    handlers_during_close: list[tuple[object, object]] = field(default_factory=list)

    def pids(self) -> dict[int, int]:
        # [FUNCTION DOC] HELPER: Returns the mapping from rank to pid.
        return self.pids_by_index

    def wait(self) -> RunProcsResult | None:
        # [FUNCTION DOC] HELPER: Raises the configured outcome if it is an exception (including
        # base exceptions such as a keyboard interrupt); otherwise returns it (may be none).
        if isinstance(self.wait_outcome, BaseException):
            raise self.wait_outcome
        return self.wait_outcome

    def close(self, death_sig: signal.Signals | None = None, timeout: float = 30) -> None:
        # [FUNCTION DOC] HELPER: Records the signal and timeout of the call and the currently
        # installed SIGTERM and SIGINT handlers; kills nothing. Returns nothing.
        self.close_calls.append((death_sig, timeout))
        self.handlers_during_close.append((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)))


def _child_main() -> int:
    # [FUNCTION DOC] HELPER: Entry point of the supervised child (this file run as a script). Parses
    # the command line, reads its index from the supervisor's environment variable (the word
    # "missing" if unset) and decides its mode: the requested mode, or sleep when an only-index
    # restriction excludes this index. Builds a report (index, pid, process group id, session id,
    # mode, value of the test marker environment variable). Then, in order: optionally corrupts the
    # torchelastic error file (writes truncated JSON), optionally prints one marker line to each
    # standard stream, ignores SIGTERM in that mode, installs the signal recorder for SIGTERM and
    # SIGINT in record mode or when asked, optionally spawns a grandchild (only if selected),
    # publishes the report atomically, and if selected and asked waits until enough sibling reports
    # exist. Exit mode publishes an "exited-at" file and returns the requested exit code; suicide
    # mode kills itself with SIGKILL; every other mode sleeps forever.
    # KNOWN ISSUE: the final return of zero is unreachable because the sleep never returns.
    args = _build_parser().parse_args()
    report_dir = Path(args.report_dir)
    index = os.environ.get(SUBPROCESS_INDEX_ENV_VAR, "missing")
    selected = args.only_index is None or args.only_index == index
    mode = args.mode if selected else MODE_SLEEP

    report: dict[str, object] = {
        "index": index,
        "pid": os.getpid(),
        "process_group_id": os.getpgrp(),
        "session_id": os.getsid(0),
        "mode": mode,
        "env_marker": os.environ.get(ENV_MARKER_VAR),
    }

    if args.corrupt_error_file:
        Path(os.environ["TORCHELASTIC_ERROR_FILE"]).write_text('{"message": ')

    if args.emit_stream_markers:
        print(f"{STDOUT_MARKER}{index}", flush=True)
        print(f"{STDERR_MARKER}{index}", file=sys.stderr, flush=True)

    if mode == MODE_IGNORE_SIGTERM:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if mode == MODE_RECORD_SIGNAL or args.record_signal:
        recorder = _make_signal_recorder(report_dir=report_dir, index=index)
        signal.signal(signal.SIGTERM, recorder)
        signal.signal(signal.SIGINT, recorder)
    if args.spawn_grandchild and selected:
        report["grandchild_pid"] = _spawn_grandchild(report_dir=report_dir, index=index)

    _write_report(report_dir=report_dir, index=index, report=report)
    if selected and args.wait_for_reports is not None:
        _wait_for_reports(report_dir=report_dir, count=args.wait_for_reports)

    if mode == MODE_EXIT:
        _publish_atomically(path=report_dir / f"exited-at-{index}", content=repr(time.monotonic()))
        return args.exit_code
    if mode == MODE_SUICIDE:
        os.kill(os.getpid(), signal.SIGKILL)

    _sleep_forever()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Builds the child's argument parser. Required: report directory and
    # mode (one of sleep, exit, suicide, ignore-sigterm, record-signal). Optional: only-index
    # string, record-signal flag, spawn-grandchild flag, exit code integer (default 0),
    # wait-for-reports integer (default none), emit-stream-markers flag, corrupt-error-file flag.
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[MODE_SLEEP, MODE_EXIT, MODE_SUICIDE, MODE_IGNORE_SIGTERM, MODE_RECORD_SIGNAL],
    )
    parser.add_argument("--only-index", default=None)
    parser.add_argument("--record-signal", action="store_true")
    parser.add_argument("--spawn-grandchild", action="store_true")
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--wait-for-reports", type=int, default=None)
    parser.add_argument("--emit-stream-markers", action="store_true")
    parser.add_argument("--corrupt-error-file", action="store_true")
    return parser


def _wait_for_reports(*, report_dir: Path, count: int) -> None:
    # [FUNCTION DOC] HELPER: Blocks the child until at least count child report files exist in the
    # report directory (30 second timeout, raising a timeout error).
    _wait_until(
        lambda: len(list(report_dir.glob("child-*.json"))) >= count,
        message=f"{count} sibling reports",
    )


def _spawn_grandchild(*, report_dir: Path, index: str) -> int:
    # [FUNCTION DOC] HELPER: Starts a python subprocess that creates a ready file named for the
    # index and then sleeps forever (inheriting this process's group); waits until the ready file
    # exists (timeout error after 30 seconds) and returns the grandchild's pid.
    ready_path = report_dir / f"grandchild-ready-{index}"
    grandchild = subprocess.Popen([sys.executable, "-c", _GRANDCHILD, str(ready_path)])
    _wait_until(ready_path.exists, message="the grandchild to report ready")
    return grandchild.pid


def _wait_until(predicate: Callable[[], bool], *, message: str) -> None:
    # [FUNCTION DOC] HELPER: Child-side polling loop (every 20 ms, fixed 30 second limit); raises a
    # timeout error naming the message when it expires.
    deadline = time.monotonic() + REPORT_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)

    raise TimeoutError(f"Timed out waiting for {message}")


def _make_signal_recorder(*, report_dir: Path, index: str) -> Callable[[int, FrameType | None], None]:
    # [FUNCTION DOC] HELPER: Returns a signal handler closure bound to the report directory and
    # index.
    def handler(signum: int, frame: FrameType | None) -> None:
        # [FUNCTION DOC] HELPER: Publishes the received signal number (as text) atomically into the
        # file named for the index, then exits the process normally with status 0.
        _publish_atomically(path=report_dir / f"signal-{index}", content=str(signum))
        sys.exit(0)

    return handler


def _write_report(*, report_dir: Path, index: str, report: dict[str, object]) -> None:
    # [FUNCTION DOC] HELPER: Atomically writes the report as JSON to the child report file of the
    # index in the report directory.
    _publish_atomically(path=report_dir / f"child-{index}.json", content=json.dumps(report))


def _publish_atomically(*, path: Path, content: str) -> None:
    # [FUNCTION DOC] HELPER: Writes the text to a hidden temporary sibling file and renames it over
    # the target path so readers never see partial content.
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(content)
    os.replace(temp_path, path)


def _sleep_forever() -> None:
    # [FUNCTION DOC] HELPER: Sleeps in one-hour steps forever; only a signal or kill ends it.
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(_child_main())
