# [MODULE DOC] Unit tests for the audit event logger module: the logger class that appends one
# JSON line per event to a file (the file is opened for every write, writes are thread safe and
# flushed, the directory is created on demand, timestamp and source are filled automatically, and
# a structured "audit op=event" log message is echoed), the process-wide get/set functions, the
# reader that loads all JSON lines files of a directory (malformed lines skipped with a warning),
# preservation of non-finite metric values, the context mechanism that injects fields such as the
# rollout id and attempt into logged events (nested scopes, restoration on exit), and the method
# decorator that computes such context from method arguments (sync and async, no-op if no logger
# is set). All files live in temporary directories.
import asyncio
import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

import miles.utils.audit_utils.event_logger.logger as event_logger_module
from miles.utils.audit_utils.event_logger.logger import (
    EventLogger,
    event_logger_context,
    get_event_logger,
    read_events,
    set_event_logger,
)
from miles.utils.audit_utils.event_logger.models import MetricEvent, WitnessAllocateIdEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainProcessIdentity

_TEST_SOURCE = SimpleProcessIdentity(component="main")


def _make_logger(log_dir: Path, file_name: str = "events.jsonl") -> EventLogger:
    # [FUNCTION DOC] HELPER: Creates an event logger writing to the given file name (default
    # "events.jsonl") in the given directory, with the main-component test source identity.
    return EventLogger(log_dir=log_dir, file_name=file_name, source=_TEST_SOURCE)


_EVENT_CLS = WitnessAllocateIdEvent
_EVENT_PARTIAL: dict = dict(
    rollout_id=0, attempt=0, witness_id_to_sample_index={10: 0, 11: 1, 12: 2}, counter_after=13
)

_LOGGER_MODULE_LOGGER = "miles.utils.audit_utils.event_logger.logger"


def _structured_messages(caplog) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the formatted messages of the captured log records that
    # came from the event logger module's logger, in order.
    return [record.getMessage() for record in caplog.records if record.name == _LOGGER_MODULE_LOGGER]


class TestEventLoggerWritesJsonl:
    # [CLASS DOC] Tests of the basic output format: one JSON object per line.
    def test_writes_multiple_events(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Logs two events and closes; the file has exactly two lines, each valid
        # JSON containing "timestamp" and "type" keys.
        logger = _make_logger(tmp_path, file_name="test.jsonl")

        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.log(
            WitnessAllocateIdEvent, dict(rollout_id=1, attempt=0, witness_id_to_sample_index={0: 0}, counter_after=1)
        )
        logger.close()

        lines = (tmp_path / "test.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "timestamp" in parsed
            assert "type" in parsed


class TestEventLoggerStructuredPrint:
    # [CLASS DOC] Tests of the log message echoing each logged event.
    def test_printed_event_uses_audit_structured_log_tag(self, tmp_path: Path, caplog) -> None:
        # [FUNCTION DOC] Logging one event at info level yields exactly one message from the
        # module logger, starting with "audit op=event event=WitnessAllocateIdEvent " and
        # containing "counter_after=13".
        """A printed event is echoed as one audit-tagged op=event record naming the event class and its fields."""
        logger = _make_logger(tmp_path)

        with caplog.at_level(logging.INFO, logger=_LOGGER_MODULE_LOGGER):
            logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()

        messages = _structured_messages(caplog)
        assert len(messages) == 1
        assert messages[0].startswith("audit op=event event=WitnessAllocateIdEvent ")
        assert "counter_after=13" in messages[0]


class TestEventLoggerAutoFillsMetadata:
    # [CLASS DOC] Tests that the logger fills the timestamp and source fields of events itself.
    def test_timestamp_is_utc_and_recent(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Records the current UTC time before and after logging, parses the
        # stored timestamp (a trailing "Z" is read as UTC) and asserts it lies between the two.
        logger = _make_logger(tmp_path)

        before = datetime.now(timezone.utc)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        after = datetime.now(timezone.utc)
        logger.close()

        line = (tmp_path / "events.jsonl").read_text().strip()
        parsed = json.loads(line)
        ts = datetime.fromisoformat(parsed["timestamp"].replace("Z", "+00:00"))
        assert before <= ts <= after

    def test_source_auto_filled(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A logger created with a training identity (component "actor", cell 2,
        # rank 3) and the default file name writes events whose source carries exactly those
        # values.
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=3)
        logger = EventLogger(log_dir=tmp_path, source=source)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["source"]["component"] == "actor"
        assert parsed["source"]["cell_index"] == 2
        assert parsed["source"]["rank_within_cell"] == 3


class TestEventLoggerThreadSafety:
    # [CLASS DOC] Tests that concurrent writers do not lose or corrupt lines.
    def test_concurrent_writes_no_data_loss(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Eight threads each log 50 events through one logger; after joining and
        # closing, the file has exactly 400 lines and every line is valid JSON.
        logger = _make_logger(tmp_path)
        num_threads = 8
        events_per_thread = 50

        def writer() -> None:
            # [FUNCTION DOC] HELPER: Thread body: logs the standard test event 50 times
            # (the events-per-thread constant of the enclosing test).
            for _ in range(events_per_thread):
                logger.log(_EVENT_CLS, _EVENT_PARTIAL)

        threads = [threading.Thread(target=writer) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        logger.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == num_threads * events_per_thread

        for line in lines:
            json.loads(line)


class TestSetGetEventLogger:
    # [CLASS DOC] Tests of the process-wide logger registry (set and get functions).
    def test_set_then_get(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] After setting a logger, getting returns the very same object; the test
        # closes it and resets the registry to none.
        logger = _make_logger(tmp_path)
        set_event_logger(logger)
        assert get_event_logger() is logger
        logger.close()
        set_event_logger(None)

    def test_replace_logger(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Setting a second logger replaces the first: getting returns the
        # second. Closes both and resets the registry to none.
        logger1 = _make_logger(tmp_path, file_name="a.jsonl")
        logger2 = _make_logger(tmp_path, file_name="b.jsonl")
        set_event_logger(logger1)
        set_event_logger(logger2)
        assert get_event_logger() is logger2
        logger1.close()
        logger2.close()
        set_event_logger(None)


class TestGetEventLoggerRaisesWhenNotSet:
    # [CLASS DOC] Tests the error raised when getting a logger that was never set.
    def test_raises_runtime_error(self) -> None:
        # [FUNCTION DOC] Saves the module's private logger variable, sets it to none directly,
        # asserts getting raises a runtime error matching "EventLogger not initialized", and
        # restores the saved value in a finally block.
        original = event_logger_module._event_logger
        event_logger_module._event_logger = None
        try:
            with pytest.raises(RuntimeError, match="EventLogger not initialized"):
                get_event_logger()
        finally:
            event_logger_module._event_logger = original


class TestEventLoggerFlushOnEachWrite:
    # [CLASS DOC] Tests that every event reaches the file immediately.
    def test_readable_before_close(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] After one log call and before closing, the file already has non-empty
        # content.
        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)

        content = (tmp_path / "events.jsonl").read_text()
        assert len(content.strip()) > 0
        logger.close()


class TestEventLoggerCreatesDirectory:
    # [CLASS DOC] Tests that missing parent directories of the log directory are created.
    def test_creates_nested_dir(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A logger on a three-level nested path that does not exist yet writes
        # an event; the events file exists afterwards.
        nested = tmp_path / "a" / "b" / "c"
        logger = _make_logger(nested)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()
        assert (nested / "events.jsonl").exists()


class TestEventLoggerFilePerWrite:
    # [CLASS DOC] Tests that the file is reopened on every write rather than kept open.
    def test_log_after_events_file_removed_writes_to_fresh_file(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Logs once, deletes the file, logs again; the file then exists and has
        # exactly one line (the second event). The logger is not closed in this test.
        """The file is opened per write, so a checkpoint restore can swap it between events."""
        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)

        path = tmp_path / "events.jsonl"
        path.unlink()

        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        assert len(path.read_text().strip().split("\n")) == 1


class TestReadEvents:
    # [CLASS DOC] Tests of the directory reader, which parses every JSON lines file into event
    # objects.
    def test_malformed_line_skipped_with_warning(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] After one valid event, appends a line of invalid JSON to the file; the
        # reader returns exactly one event (the warning is not asserted despite the name). The
        # reader is imported again locally, redundantly with the module-level import.
        from miles.utils.audit_utils.event_logger.logger import read_events

        logger = _make_logger(tmp_path)
        logger.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger.close()

        with open(tmp_path / "events.jsonl", "a") as f:
            f.write("this is not valid json\n")

        events = read_events(tmp_path)
        assert len(events) == 1

    def test_reads_multiple_jsonl_files(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two files with one and two events respectively produce three events
        # in total. The reader is imported again locally, redundantly.
        from miles.utils.audit_utils.event_logger.logger import read_events

        logger_a = EventLogger(log_dir=tmp_path, file_name="a.jsonl", source=_TEST_SOURCE)
        logger_a.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_a.close()

        logger_b = EventLogger(log_dir=tmp_path, file_name="b.jsonl", source=_TEST_SOURCE)
        logger_b.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_b.log(_EVENT_CLS, _EVENT_PARTIAL)
        logger_b.close()

        events = read_events(tmp_path)
        assert len(events) == 3


class TestEventLoggerKeepsNonFiniteMetrics:
    # [CLASS DOC] Tests that not-a-number metric values survive writing and reading.
    def test_a_nan_metric_reads_back_as_a_nan(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Logs a metric event (with echo printing disabled) whose loss metric is
        # not-a-number, reads the directory back and asserts the single event's metric is
        # not-a-number (rather than missing or null).
        """Written as json null it reads back absent, so a diverged run would look like one that never reported."""
        logger = _make_logger(tmp_path)
        logger.log(MetricEvent, dict(rollout_id=0, attempt=0, metrics={"train/loss": math.nan}), print_log=False)
        logger.close()

        [event] = read_events(tmp_path)

        assert math.isnan(event.metrics["train/loss"])


class TestWithContext:
    # [CLASS DOC] Tests of the context manager that supplies default field values (rollout id,
    # attempt) for events logged inside its scope; nested scopes override and restore.
    def test_injects_context_fields_into_logged_event(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Inside a scope with rollout id 5 and attempt 7, a metric event logged
        # with only metrics is stored with those two fields and the given metrics.
        """Fields from with_context are merged into events logged inside the scope."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 5, "attempt": 7}):
            logger.log(MetricEvent, dict(metrics={"loss": 1.0}))
        logger.close()

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["rollout_id"] == 5
        assert parsed["attempt"] == 7
        assert parsed["metrics"] == {"loss": 1.0}

    def test_context_not_applied_outside_scope(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] An event logged after an (empty) scope has exited has a null rollout
        # id.
        """Events logged after the context scope exits do not carry context fields."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 5}):
            pass
        logger.log(MetricEvent, dict(metrics={}))
        logger.close()

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["rollout_id"] is None

    def test_nesting_overrides_then_restores(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Outer scope sets rollout 1 and attempt 1; an inner scope sets attempt
        # 2. The event logged inside the inner scope has (1, 2); the one logged after the inner
        # scope but inside the outer has (1, 1).
        """A nested with_context overrides outer fields, then restores them on exit."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 1, "attempt": 1}):
            with logger.with_context({"attempt": 2}):
                logger.log(MetricEvent, dict(metrics={"k": "inner"}))
            logger.log(MetricEvent, dict(metrics={"k": "outer"}))
        logger.close()

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        inner = json.loads(lines[0])
        outer = json.loads(lines[1])
        assert (inner["rollout_id"], inner["attempt"]) == (1, 2)
        assert (outer["rollout_id"], outer["attempt"]) == (1, 1)

    def test_context_var_empty_after_exit(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] After two nested scopes have exited, the logger's private context
        # variable resolves to an empty mapping (default empty).
        """After all with_context scopes exit the underlying contextvar resolves to empty."""
        logger = _make_logger(tmp_path)
        with logger.with_context({"rollout_id": 1}):
            with logger.with_context({"attempt": 2}):
                pass
        assert logger._context_var.get({}) == {}


class TestEventLoggerContextDecorator:
    # [CLASS DOC] Tests of the method decorator that takes a function mapping the method's own
    # arguments to context fields and runs the method inside the corresponding logger scope. When
    # no logger is set, the function is not even evaluated and the method runs plainly; async
    # methods are supported and keep the scope across awaits.
    def test_uninitialized_runs_without_context(self) -> None:
        # [FUNCTION DOC] With no logger set, a decorated function is called, returns its result
        # (6 for input 3) and is invoked exactly once.
        """When the logger is uninitialized the wrapper invokes the method directly."""
        set_event_logger(None)
        calls: list[tuple] = []

        @event_logger_context(lambda obj, x: {"rollout_id": x})
        def method(obj: object, x: int) -> int:
            # [FUNCTION DOC] HELPER: Records its two arguments in the enclosing list and returns
            # the second argument doubled.
            calls.append((obj, x))
            return x * 2

        try:
            assert method(object(), 3) == 6
            assert len(calls) == 1
        finally:
            set_event_logger(None)

    def test_ctx_fn_not_called_when_uninitialized(self) -> None:
        # [FUNCTION DOC] With no logger set, the context function is never called (its call
        # record stays empty) while the decorated method still returns its input 9.
        """ctx_fn is skipped entirely when the event logger is not initialized."""
        set_event_logger(None)
        ctx_calls: list[int] = []

        def ctx_fn(obj: object, x: int) -> dict:
            # [FUNCTION DOC] HELPER: Records the second argument and returns a context mapping
            # with that value as the rollout id.
            ctx_calls.append(x)
            return {"rollout_id": x}

        @event_logger_context(ctx_fn)
        def method(obj: object, x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns its second argument unchanged.
            return x

        try:
            assert method(object(), 9) == 9
            assert ctx_calls == []
        finally:
            set_event_logger(None)

    def test_initialized_injects_fields_from_method_args(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] With a logger set, a decorated method of a local worker class, called
        # with 42, logs a metric event through the global logger; the stored event has rollout
        # id 42. The logger is closed and the registry reset afterwards.
        """When initialized, ctx_fn output is injected into events logged by the method."""
        logger = _make_logger(tmp_path)
        set_event_logger(logger)

        class Worker:
            # [CLASS DOC] HELPER: Local test class with one decorated method whose context is
            # the rollout id taken from its argument.
            @event_logger_context(lambda self, rollout_id: {"rollout_id": rollout_id})
            def run(self, rollout_id: int) -> None:
                # [FUNCTION DOC] HELPER: Logs a metric event (metrics with ok true) through the
                # global event logger.
                get_event_logger().log(MetricEvent, dict(metrics={"ok": True}))

        try:
            Worker().run(42)
            logger.close()
        finally:
            set_event_logger(None)

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["rollout_id"] == 42

    def test_ctx_fn_receives_method_args(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] The context function (taking object, rollout id and a keyword-only
        # attempt) is called with exactly the method's arguments, here (3, attempt 8); the
        # recorded call equals (3, 8) and the stored event has rollout id 3 and attempt 8.
        """ctx_fn is called with exactly the same positional/keyword args as the method."""
        logger = _make_logger(tmp_path)
        set_event_logger(logger)
        seen: list[tuple] = []

        def ctx_fn(obj: object, rollout_id: int, *, attempt: int) -> dict:
            # [FUNCTION DOC] HELPER: Records the rollout id and attempt and returns them as a
            # context mapping.
            seen.append((rollout_id, attempt))
            return {"rollout_id": rollout_id, "attempt": attempt}

        @event_logger_context(ctx_fn)
        def method(obj: object, rollout_id: int, *, attempt: int) -> None:
            # [FUNCTION DOC] HELPER: Logs an empty metric event through the global logger.
            get_event_logger().log(MetricEvent, dict(metrics={}))

        try:
            method(object(), 3, attempt=8)
            logger.close()
        finally:
            set_event_logger(None)

        assert seen == [(3, 8)]
        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert (parsed["rollout_id"], parsed["attempt"]) == (3, 8)

    async def test_initialized_injects_fields_into_events_an_async_method_logs(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Async test: a decorated async method yields to the event loop once
        # (sleep of zero) and then logs a metric event; the stored event still has rollout id 42,
        # showing the scope survives the suspension.
        """An awaited method keeps the context across its suspension points, or its events lose their identity."""
        logger = _make_logger(tmp_path)
        set_event_logger(logger)

        class Worker:
            # [CLASS DOC] HELPER: Local test class with one decorated async method whose context
            # is the rollout id taken from its argument.
            @event_logger_context(lambda self, rollout_id: {"rollout_id": rollout_id})
            async def run(self, rollout_id: int) -> None:
                # [FUNCTION DOC] HELPER: Awaits a zero-length sleep, then logs a metric event
                # (metrics with ok true) through the global logger.
                await asyncio.sleep(0)
                get_event_logger().log(MetricEvent, dict(metrics={"ok": True}))

        try:
            await Worker().run(42)
            logger.close()
        finally:
            set_event_logger(None)

        parsed = json.loads((tmp_path / "events.jsonl").read_text().strip())
        assert parsed["rollout_id"] == 42

    async def test_uninitialized_awaits_an_async_method_without_context(self) -> None:
        # [FUNCTION DOC] Async test: with no logger set, a decorated async method can be awaited
        # and returns its result (6 for input 3) without failing.
        """Decorating a method must not make it fail wherever the event logger was never set up."""
        set_event_logger(None)

        @event_logger_context(lambda obj, x: {"rollout_id": x})
        async def method(obj: object, x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns its second argument doubled.
            return x * 2

        try:
            assert await method(object(), 3) == 6
        finally:
            set_event_logger(None)
