# [MODULE DOC] Unit tests for the log follower of the Helm launcher observability code. The
# follower polls the pods selected by a namespace and label selector, and for each container run
# (identified by pod uid and container id) starts a "kubectl logs" subprocess whose lines are
# logged as "[pod/container] text". It follows only live containers, reads the previous life of a
# container that restarted (marked "(previous)") but not when it was already followed live,
# never starts a second stream for a container it already reads, resumes a dropped stream from the
# last timestamp seen using a since-time option, reports a failed stream as "dropped" together with
# its error output (but stays silent for streams it stopped itself), reports process start
# failures once without retrying, and kills subprocesses that ignore termination after a grace
# period. Tests also cover the RFC 3339 timestamp parser and the kubectl logs command builder.
# The subprocess start function, pod lookup and polling interval are replaced via monkeypatching;
# fake processes yield canned lines.
import logging
import subprocess
import time

from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.observability.conftest import (
    make_container,
    make_pod,
    wait_for,
)

from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import log_follower, polling


def _pod(name="trainer-0", uid="u", container_id="docker://a", previous_container_id="", running=True):
    # [FUNCTION DOC] HELPER: Builds a pod (see the shared pod builder) with a single container
    # named "app" having the given container id, previous container id and running flag.
    return make_pod(
        name=name,
        uid=uid,
        containers=(
            make_container(
                name="app",
                container_id=container_id,
                previous_container_id=previous_container_id,
                running=running,
            ),
        ),
    )


class FakeProcess:
    # [CLASS DOC] HELPER: Fake subprocess handle for a log stream. Its standard output is a
    # generator that yields the canned lines and then, if blocking, sleeps in 0.01 second steps
    # until the process is killed or terminated (simulating a live follow); if not blocking the
    # output ends after the lines. poll returns 0 once killed, otherwise none (still running).
    def __init__(self, command: list[str], lines: list[str], blocking: bool, returncode: int = 0) -> None:
        # [FUNCTION DOC] HELPER: Stores the command, the return code to report, a killed flag
        # (false) and the output generator.
        self.command = command
        self.killed = False
        self.returncode = returncode
        self.stdout = self._lines(lines, blocking)

    def poll(self) -> int | None:
        # [FUNCTION DOC] HELPER: Returns 0 when killed, none while running.
        return 0 if self.killed else None

    def wait(self, timeout: float | None = None) -> int:
        # [FUNCTION DOC] HELPER: Returns 0 immediately regardless of the timeout.
        return 0

    def terminate(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the process as killed.
        self.killed = True

    def kill(self) -> None:
        # [FUNCTION DOC] HELPER: Marks the process as killed.
        self.killed = True

    def _lines(self, lines: list[str], blocking: bool):
        # [FUNCTION DOC] HELPER: Generator yielding the canned lines, then (if blocking) sleeping
        # in 0.01 second steps until the killed flag is set.
        yield from lines
        while blocking and not self.killed:
            time.sleep(0.01)


class FakeKubectl:
    # [CLASS DOC] HELPER: Replacement for the subprocess start function of the follower module.
    # Every start records the command, writes the configured error text into the error stream
    # passed by the caller, and returns a new fake process with the canned lines, blocking mode and
    # return code. Records all commands and all created processes.
    def __init__(self, monkeypatch, lines: list[str], blocking: bool, returncode: int = 0, error: str = "") -> None:
        # [FUNCTION DOC] HELPER: Stores the configuration, creates empty command and process
        # lists, and installs itself as the follower module's process start function.
        self.commands: list[list[str]] = []
        self.processes: list[FakeProcess] = []
        self._lines = lines
        self._blocking = blocking
        self._returncode = returncode
        self._error = error
        monkeypatch.setattr(log_follower.subprocess, "Popen", self._popen)

    def _popen(self, command, **kwargs) -> FakeProcess:
        # [FUNCTION DOC] HELPER: Fake process start: records the command, writes the error text
        # to the "stderr" keyword argument (which must be a writable file object), creates a fake
        # process with a copy of the canned lines, records and returns it.
        self.commands.append(command)
        kwargs["stderr"].write(self._error)
        process = FakeProcess(command, list(self._lines), self._blocking, returncode=self._returncode)
        self.processes.append(process)
        return process


def _followed(monkeypatch, pods, lines=(), blocking: bool = True, returncode: int = 0, error: str = "") -> FakeKubectl:
    # [FUNCTION DOC] HELPER: Installs the fake process start (with the given lines, blocking
    # mode, return code and error text), makes the follower's pod lookup return the given pods
    # regardless of namespace and selector, and shortens the polling interval to 0.01 seconds.
    # Returns the fake so tests can inspect started commands and processes.
    fake = FakeKubectl(monkeypatch, list(lines), blocking, returncode=returncode, error=error)
    monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)
    monkeypatch.setattr(polling, "POLL_INTERVAL_SECONDS", 0.01)
    return fake


class TestWithLogFollowing:
    # [CLASS DOC] Tests for the follower object and the context manager that runs it in the
    # background for the duration of a with block.
    def test_a_process_start_failure_is_reported_without_a_retry_storm(self, monkeypatch, caplog):
        # [FUNCTION DOC] The process start function raises an operating system error ("kubectl
        # is unavailable") and the pod lookup returns one pod. Two reconcile calls on a follower
        # lead to exactly one start attempt, and a warning "[trainer-0/app] could not be followed
        # (kubectl is unavailable)" is logged.
        """A stream that cannot start reports the failure without retrying the same container."""
        attempts = 0

        def fail_to_start(command, **kwargs):
            # [FUNCTION DOC] HELPER: Counts the attempt and raises an operating system error.
            nonlocal attempts
            attempts += 1
            raise OSError("kubectl is unavailable")

        monkeypatch.setattr(log_follower.subprocess, "Popen", fail_to_start)
        monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: [_pod()])
        follower = log_follower._LogFollower(namespace="rl", selector="app=x")

        with caplog.at_level(logging.WARNING, logger=log_follower.__name__):
            follower.reconcile()
            follower.reconcile()

        assert attempts == 1
        assert "[trainer-0/app] could not be followed (kubectl is unavailable)" in caplog.text

    def test_a_failed_log_stream_reports_its_error_and_is_restarted(self, monkeypatch, caplog):
        # [FUNCTION DOC] The fake process ends at once with return code 2 and writes "server
        # rejected stream" to its error stream. After the first reconcile, the test joins the
        # stream's reader thread (1 second max), asserts it ended, and reconciles again. Asserts
        # two commands were started in total and the warning "[trainer-0/app] dropped: server
        # rejected stream" was logged.
        """A dropped kubectl stream reports stderr and is retried by later reconciliation."""
        fake = _followed(monkeypatch, [_pod()], blocking=False, returncode=2, error="server rejected stream")
        follower = log_follower._LogFollower(namespace="rl", selector="app=x")

        with caplog.at_level(logging.WARNING, logger=log_follower.__name__):
            follower.reconcile()
            stream = next(iter(follower._streams.values()))
            assert stream._thread is not None
            stream._thread.join(timeout=1.0)
            assert not stream._thread.is_alive()
            follower.reconcile()

        assert len(fake.commands) == 2
        assert "[trainer-0/app] dropped: server rejected stream" in caplog.text

    def test_an_unresponsive_stream_is_killed_after_the_stop_grace_period(self, monkeypatch):
        # [FUNCTION DOC] Uses a fake process that never exits: it ignores termination, its wait
        # with a timeout raises the subprocess timeout error, and kill sets a flag. Builds the
        # container run of one pod, creates a stream for it, calls stop, and asserts the process
        # was killed.
        """A kubectl process that ignores termination is killed after the stop grace period."""

        class UnresponsiveProcess:
            # [CLASS DOC] HELPER: Fake process that never ends by itself: no output lines, poll
            # always reports still running, terminate does nothing, a wait with a timeout raises
            # the timeout error, and kill records that it happened.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the killed flag (false), return code 0 and an
                # empty output iterator.
                self.killed = False
                self.returncode = 0
                self.stdout = iter(())

            def poll(self) -> None:
                # [FUNCTION DOC] HELPER: Always reports that the process is still running.
                return None

            def wait(self, timeout: float | None = None) -> int:
                # [FUNCTION DOC] HELPER: With a timeout given, raises the subprocess timeout
                # error; without one returns 0.
                if timeout is not None:
                    raise subprocess.TimeoutExpired(cmd="kubectl logs", timeout=timeout)
                return 0

            def terminate(self) -> None:
                # [FUNCTION DOC] HELPER: Ignores the request.
                pass

            def kill(self) -> None:
                # [FUNCTION DOC] HELPER: Records that the process was killed.
                self.killed = True

        process = UnresponsiveProcess()
        monkeypatch.setattr(log_follower.subprocess, "Popen", lambda command, **kwargs: process)
        run = next(iter(log_follower.container_runs([_pod()]).values()))
        stream = log_follower._LogStream(namespace="rl", run=run)

        stream.stop()

        assert process.killed

    def test_a_line_without_a_timestamp_is_emitted_intact(self, monkeypatch, caplog):
        # [FUNCTION DOC] A stream yielding one line with no leading timestamp, ending at once
        # (not blocking), is logged intact as "[trainer-0/app] plain message with spaces". The
        # test waits for a second start (the reconnect), and asserts that none of the started
        # commands carries the since-time option, because no timestamp could be parsed.
        """A log line without an ISO timestamp is emitted intact without inventing a resume cursor."""
        fake = _followed(monkeypatch, [_pod()], lines=["plain message with spaces\n"], blocking=False)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: len(fake.commands) >= 2)

        assert "[trainer-0/app] plain message with spaces" in caplog.text
        assert all("--since-time" not in command for command in fake.commands)

    def test_follows_the_container_a_pod_is_running(self, monkeypatch, caplog):
        # [FUNCTION DOC] A stream yielding one timestamped line "hello" is logged as
        # "[trainer-0/app] hello" (the timestamp is stripped). The first started command begins
        # with kubectl, logs and the pod name, and includes the follow option.
        """A run's own output is what a user launched the launcher to read."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "hello" in caplog.text)

        assert "[trainer-0/app] hello" in caplog.text
        assert fake.commands[0][:3] == ["kubectl", "logs", "trainer-0"]
        assert "--follow" in fake.commands[0]

    def test_reads_the_life_a_container_already_crashed_out_of(self, monkeypatch, caplog):
        # [FUNCTION DOC] A pod whose container id differs from its recorded previous
        # (terminated) container id causes a read of the previous life: the log gets the marker
        # "(previous)" and at least one started command includes the previous option.
        """The log of the container that died is the only record of why the pod will not start."""
        pods = [_pod(container_id="docker://b", previous_container_id="docker://a")]
        fake = _followed(monkeypatch, pods, lines=["2026-08-10T00:00:00.1Z boom\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "(previous)" in caplog.text)

        assert any("--previous" in command for command in fake.commands)

    def test_does_not_start_a_second_stream_for_a_container_it_already_reads(self, monkeypatch, caplog):
        # [FUNCTION DOC] With a live blocking stream and many polls (waiting 0.1 seconds more
        # after the first line), exactly one command was started.
        """Every poll sees the same container, and a stream per poll would print the log many times over."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "hello" in caplog.text)
                time.sleep(0.1)

        assert len(fake.commands) == 1

    def test_asks_only_for_what_it_has_not_read_when_a_stream_drops(self, monkeypatch, caplog):
        # [FUNCTION DOC] A stream yielding one timestamped line then ending cleanly (return code
        # 0, non-blocking) is restarted; the last started command carries the since-time option
        # with the last line's timestamp.
        """A reconnect that started over would reprint the whole log every time the api server blinked."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"], blocking=False)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: len(fake.commands) >= 2)

        assert "--since-time" in fake.commands[-1]
        assert "2026-08-10T00:00:00.1Z" in fake.commands[-1]

    def test_stops_every_stream_when_the_caller_leaves(self, monkeypatch, caplog):
        # [FUNCTION DOC] After leaving the with block while a blocking stream is live, every
        # created fake process has been killed.
        """kubectl logs --follow never returns on its own, so a leaked one outlives the whole launcher."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"])

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: bool(fake.processes))

        assert all(process.killed for process in fake.processes)

    def test_does_not_reconnect_to_a_container_that_ended(self, monkeypatch, caplog):
        # [FUNCTION DOC] A pod whose container is not running (it ended) is read once: after the
        # line "bye" appears and 0.1 seconds more pass, exactly one command was started.
        """kubectl returns at once on a dead container, so retrying it would spawn a process every poll."""
        fake = _followed(monkeypatch, [_pod(running=False)], lines=["2026-08-10T00:00:00.1Z bye\n"], blocking=False)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "bye" in caplog.text)
                time.sleep(0.1)

        assert len(fake.commands) == 1

    def test_does_not_read_a_crashed_life_it_already_followed_live(self, monkeypatch, caplog):
        # [FUNCTION DOC] First the pod is reported with container id "a" (followed live). Then
        # the pod lookup is swapped to report container id "b" with previous container id "a"
        # (the container crashed and restarted). The test waits for the second start and an extra
        # 0.1 seconds, and asserts no started command used the previous option, since that life
        # was already followed live.
        # KNOWN ISSUE: the selected-pods function is patched twice at the start (once by the
        # shared helper and once again right after it) with equivalent lambdas, so the second
        # patch before the with block is redundant.
        """Replaying an hour of log under a (previous) prefix buries the crash the user came to read."""
        running = [_pod(container_id="docker://a")]
        crashed = [_pod(container_id="docker://b", previous_container_id="docker://a")]
        pods = running
        fake = _followed(monkeypatch, pods, lines=["2026-08-10T00:00:00.1Z hello\n"])
        monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: bool(fake.commands))
                pods = crashed
                monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)
                wait_for(lambda: len(fake.commands) >= 2)
                time.sleep(0.1)

        assert not any("--previous" in command for command in fake.commands)

    def test_picks_a_container_back_up_after_its_stream_failed(self, monkeypatch, caplog):
        # [FUNCTION DOC] A stream that yields a line and ends with return code 1 is started
        # again by later polls: at least two commands were started.
        """One blink of the api server would otherwise hide the whole rest of that container's log."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"], blocking=False, returncode=1)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: len(fake.commands) >= 2)

        assert len(fake.commands) >= 2

    def test_says_a_failed_stream_dropped_rather_than_stopped(self, monkeypatch, caplog):
        # [FUNCTION DOC] A stream ending with return code 1 is reported with the text
        # "[trainer-0/app] dropped" and never with the text "stopped:"; at least one process was
        # created.
        """A stream that is picked up again is not a stream that stopped, and the log has to say which it was."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"], blocking=False, returncode=1)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "dropped" in caplog.text)

        assert "[trainer-0/app] dropped" in caplog.text
        assert "stopped:" not in caplog.text
        assert fake.processes

    def test_resumes_a_failed_stream_from_where_it_had_read_to(self, monkeypatch, caplog):
        # [FUNCTION DOC] After a stream fails with return code 1 having read one timestamped
        # line, the restart command carries the since-time option with that line's timestamp.
        # KNOWN ISSUE: this test is almost identical to the test of resuming after a clean drop,
        # differing only in the return code, so it largely duplicates it.
        """Restarting from the beginning would reprint everything the user has already read."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"], blocking=False, returncode=1)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: len(fake.commands) >= 2)

        assert "--since-time" in fake.commands[-1]
        assert "2026-08-10T00:00:00.1Z" in fake.commands[-1]

    def test_says_nothing_about_a_stream_the_caller_tore_down(self, monkeypatch, caplog):
        # [FUNCTION DOC] A blocking stream that would exit with return code 1 is torn down when
        # leaving the with block: every process is killed and the log never contains "dropped".
        """The launcher kills every stream on its way out, and every one of them exits non-zero for it."""
        fake = _followed(monkeypatch, [_pod()], lines=["2026-08-10T00:00:00.1Z hello\n"], returncode=1)

        with caplog.at_level(logging.INFO, logger=log_follower.__name__):
            with log_follower.with_log_following(namespace="rl", selector="app=x"):
                wait_for(lambda: "hello" in caplog.text)

        assert all(process.killed for process in fake.processes)
        assert "dropped" not in caplog.text


class TestParseRfc3339:
    # [CLASS DOC] Tests for the parser of the leading timestamp token of a log line; it returns a
    # parsed time for RFC 3339 stamps with zero to nine fractional digits and a trailing "Z", and
    # none for anything else.
    def test_reads_the_nine_digit_fraction_kubectl_stamps_a_line_with(self):
        # [FUNCTION DOC] A stamp with nine fractional digits parses (is not none).
        """A stamp this parser rejects stays in the line and never advances the point a reconnect resumes from."""
        assert log_follower._parse_rfc3339("2026-08-12T09:00:00.123456789Z") is not None

    def test_pads_a_fraction_shorter_than_it_supports(self):
        # [FUNCTION DOC] A stamp with a two digit fraction parses.
        """kubectl trims trailing zeros, so the same stream carries stamps of several fractional widths."""
        assert log_follower._parse_rfc3339("2026-08-12T09:00:00.12Z") is not None

    def test_reads_a_stamp_that_carries_no_fraction(self):
        # [FUNCTION DOC] A stamp without any fraction parses.
        """A line stamped on a whole second is an ordinary line, not one to be left with its stamp."""
        assert log_follower._parse_rfc3339("2026-08-12T09:00:00Z") is not None

    def test_answers_nothing_for_a_line_that_starts_with_a_word(self):
        # [FUNCTION DOC] The word "INFO" is not a timestamp and gives none.
        """The first token of an untimestamped line is log text, and taking it as a stamp would eat it."""
        assert log_follower._parse_rfc3339("INFO") is None


class TestLogsCommand:
    # [CLASS DOC] Tests for the kubectl logs command builder of the command wrapper class.
    def test_asks_for_the_timestamps_a_reconnect_needs(self):
        # [FUNCTION DOC] A follow request without a since time includes the timestamps option
        # and not the since-time option.
        """--since-time is the only way back to where a dropped stream stopped, and it reads these stamps."""
        command = Kubectl.logs_command(
            namespace="rl", target="trainer-0", container="app", follow=True, previous=False, since_time=None
        )

        assert "--timestamps" in command
        assert "--since-time" not in command

    def test_does_not_follow_a_container_that_already_ended(self):
        # [FUNCTION DOC] A request with follow off, previous on and a since time "t" has no
        # follow option and ends with the arguments since-time and "t".
        """--follow on a dead container returns at once; asking plainly is what actually fetches its log."""
        command = Kubectl.logs_command(
            namespace="rl", target="trainer-0", container="app", follow=False, previous=True, since_time="t"
        )

        assert "--follow" not in command
        assert command[-2:] == ["--since-time", "t"]
