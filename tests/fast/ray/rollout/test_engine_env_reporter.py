# [MODULE DOC]
# Tests of the engine env reporter. It reads an engine cell's server info through an api client and
# records it as an engine env report event in the audit event log, at most once per configured
# interval per cell, with credentials redacted. Covers the interval and retry cadence (including a
# shorter retry after a failed reading), redaction of nested secrets, tolerance of unreachable or
# hanging engines, and running without an event logger. Uses a fake api client, an event logger that
# writes into a temporary directory, and a fake monotonic clock patched into the reporter module.
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from miles.ray.rollout import engine_env_reporter as engine_env_reporter_module
from miles.ray.rollout.engine_env_reporter import RETRY_INTERVAL_SECONDS, EngineEnvReporter
from miles.utils.audit_utils.event_logger.logger import EventLogger, read_events, set_event_logger
from miles.utils.audit_utils.event_logger.models import EngineEnvReportEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity

_SAMPLE_SERVER_INFO: dict[str, Any] = {
    "model_path": "/models/qwen",
    "api_key": "hunter2",
    "admin_api_key": "hunter3",
    "ssl_keyfile_password": "hunter4",
    "version": "0.5.0",
    "internal_states": [
        {
            "last_gen_throughput": 1.0,
            "api_key": "hunter2",
            "admin_api_key": "hunter3",
            "env_vars": {"HF_TOKEN": "hunter5", "CUDA_VISIBLE_DEVICES": "0"},
        },
        {"last_gen_throughput": 2.0, "api_key": "hunter2", "env_vars": {"RANK": "1"}},
    ],
}


class _FakeApiClient:
    # [CLASS DOC] HELPER: Fake engine api client. Counts server info requests and either raises the
    # configured error or returns the configured payload (default: the module sample payload with
    # secrets in it).
    def __init__(self, *, server_info: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        # [FUNCTION DOC] HELPER: Starts the call counter at zero, stores the error (settable later by
        # tests) and the payload, defaulting to the shared sample payload when none is given.
        self.call_count = 0
        self.error = error
        self._server_info = _SAMPLE_SERVER_INFO if server_info is None else server_info

    async def get_server_info(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Increments the call counter, raises the configured error if any,
        # otherwise returns the configured payload (the same object each time, not a copy).
        self.call_count += 1
        if self.error is not None:
            raise self.error
        return self._server_info


@pytest.fixture()
def event_log_dir(tmp_path: Path) -> Path:
    # [FUNCTION DOC] HELPER: Fixture. Installs a global event logger writing to the temporary
    # directory with an inference controller process identity as source, yields that directory, and
    # clears the global event logger after the test.
    set_event_logger(EventLogger(log_dir=tmp_path, source=SimpleProcessIdentity(component="inference_controller")))
    yield tmp_path
    set_event_logger(None)


@pytest.fixture()
def fake_clock(monkeypatch) -> list[float]:
    # [FUNCTION DOC] HELPER: Fixture. Replaces the time reference in the reporter module with a simple
    # object whose monotonic function returns the first element of a one-element list that starts at
    # 1000.0, and returns that list so tests advance time by adding to its first element.
    now = [1000.0]
    monkeypatch.setattr(engine_env_reporter_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


async def _report(reporter: EngineEnvReporter, api_client: _FakeApiClient, *, cell_id: str = "a") -> None:
    # [FUNCTION DOC] HELPER: Asks the reporter to report if due for the given cell (default a), using a
    # server url of http, the cell id and port 30000, and the given api client.
    await reporter.report_if_due(cell_id=cell_id, server_url=f"http://{cell_id}:30000", api_client=api_client)


def _events(log_dir: Path) -> list[EngineEnvReportEvent]:
    # [FUNCTION DOC] HELPER: Reads and returns the events recorded in the given event log directory.
    return read_events(log_dir)


class TestReportIfDue:
    # [CLASS DOC] Checks when the reporter reads and records a cell's environment: the first call,
    # the interval, a non-positive interval, and a fresh reporter for a restarted cell.
    async def test_records_one_event_for_the_cell(self, event_log_dir: Path) -> None:
        # [FUNCTION DOC] One report with an hour interval must write exactly one event: for cell a,
        # with the server url used in the call, and with the source set to the installed logger's
        # inference controller identity.
        """An engine's environment is only knowable once it answers, so the reading is the record."""
        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient())

        events = _events(event_log_dir)
        assert [event.cell_id for event in events] == ["a"]
        assert events[0].server_url == "http://a:30000"
        assert events[0].source == SimpleProcessIdentity(component="inference_controller")

    async def test_reads_only_once_within_the_interval(self, event_log_dir: Path, fake_clock) -> None:
        # [FUNCTION DOC] Reports, advances the fake clock by one second less than the interval, reports
        # again, and asserts the api client was asked only once.
        """The sweep runs every few seconds, and re-dumping a whole environment each time would flood the log."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += 3599.0
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_reads_again_after_the_interval(self, event_log_dir: Path, fake_clock) -> None:
        # [FUNCTION DOC] Reports, advances the fake clock by exactly the interval, reports again, and
        # asserts two reads (the boundary is inclusive).
        """An engine's environment can be changed under it, so one reading at startup is not enough."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += 3600.0
        await _report(reporter, api_client)

        assert api_client.call_count == 2

    async def test_a_non_positive_interval_reads_a_cell_once(self, event_log_dir: Path, fake_clock) -> None:
        # [FUNCTION DOC] With an interval of zero, reports, advances the clock by a very large amount,
        # reports again, and asserts only one read (zero means only at startup).
        """--env-report-interval-seconds of zero means 'only at startup', for engines as much as for processes."""
        reporter, api_client = EngineEnvReporter(interval_seconds=0.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += 100000.0
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_a_restarted_cell_is_read_again_immediately(self, event_log_dir: Path, fake_clock) -> None:
        # [FUNCTION DOC] Reports with two different reporter instances (as after a cell is rebuilt)
        # against the same api client without advancing the clock, and asserts two reads.
        """Healing builds a new cell, whose reporter is new too, which is the case this audit exists for."""
        api_client = _FakeApiClient()

        await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)
        await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)

        assert api_client.call_count == 2


class TestRedaction:
    # [CLASS DOC] Checks that recorded server info has secret values replaced by a redaction marker
    # (prefix redacted-sha256 followed by a digest) while everything else is kept.
    async def test_the_recorded_server_info_carries_no_secret(self, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Reports the sample payload and asserts the serialized event contains none of
        # the secret text, and that the top-level api key, admin api key and ssl key file password, plus
        # the first internal state's api key, admin api key and HF token environment variable, all start
        # with the redaction marker.
        """/server_info returns the engine's api keys and its whole sglang environment, verbatim."""
        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient())

        event = _events(event_log_dir)[0]
        server_info = event.server_info
        assert "hunter" not in event.model_dump_json()
        assert server_info["api_key"].startswith("redacted-sha256:")
        assert server_info["admin_api_key"].startswith("redacted-sha256:")
        assert server_info["ssl_keyfile_password"].startswith("redacted-sha256:")
        assert server_info["internal_states"][0]["env_vars"]["HF_TOKEN"].startswith("redacted-sha256:")
        assert server_info["internal_states"][0]["api_key"].startswith("redacted-sha256:")
        assert server_info["internal_states"][0]["admin_api_key"].startswith("redacted-sha256:")

    async def test_the_recorded_server_info_keeps_everything_that_is_not_a_secret(self, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Reports the sample payload and asserts non-secret values survive unchanged:
        # model path, version, visible devices and rank environment variables of the internal states,
        # and the second state's generation throughput.
        """A redaction that also drops the model path and the engine's rank records nothing worth auditing."""
        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient())

        server_info = _events(event_log_dir)[0].server_info
        assert server_info["model_path"] == "/models/qwen"
        assert server_info["version"] == "0.5.0"
        assert server_info["internal_states"][0]["env_vars"]["CUDA_VISIBLE_DEVICES"] == "0"
        assert server_info["internal_states"][1]["env_vars"]["RANK"] == "1"
        assert server_info["internal_states"][1]["last_gen_throughput"] == 2.0


class TestDegradation:
    # [CLASS DOC] Checks behavior with older engines, unreachable or hanging engines, the failure retry
    # cadence, and a missing event logger.
    async def test_an_engine_without_the_env_var_gate_is_still_recorded(self, event_log_dir: Path) -> None:
        # [FUNCTION DOC] Reports a payload from an older engine (internal states without an environment
        # variables entry) and asserts the event is still recorded: model path kept, api key redacted,
        # and the internal state's other field kept.
        """An older engine has no env_vars key at all, and its server args are still worth recording."""
        server_info = {
            "model_path": "/models/qwen",
            "api_key": "hunter2",
            "internal_states": [{"waiting_queue": 0, "api_key": "hunter2"}],
        }

        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(server_info=server_info))

        recorded = _events(event_log_dir)[0].server_info
        assert recorded["model_path"] == "/models/qwen"
        assert recorded["api_key"].startswith("redacted-sha256:")
        assert recorded["internal_states"][0]["waiting_queue"] == 0

    async def test_an_engine_without_the_env_var_gate_still_has_its_nested_keys_hidden(
        self, event_log_dir: Path
    ) -> None:
        # [FUNCTION DOC] Reports a payload that has only internal states (with an api key, no
        # environment variables entry) and asserts the secret text is absent from the serialized event
        # while the nested api key carries the redaction marker.
        """Every internal state repeats the whole ServerArgs, credentials included, gate or no gate."""
        server_info = {"internal_states": [{"waiting_queue": 0, "api_key": "hunter2"}]}

        await _report(EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(server_info=server_info))

        event = _events(event_log_dir)[0]
        assert "hunter2" not in event.model_dump_json()
        assert event.server_info["internal_states"][0]["api_key"].startswith("redacted-sha256:")

    async def test_an_unreachable_engine_only_warns(self, event_log_dir: Path, caplog) -> None:
        # [FUNCTION DOC] An api client that raises a timeout error must not make the report raise.
        # Asserts a warning from the reporter logger saying the engine env of cell a could not be
        # recorded, that the first log record carries exception info, and that no event was written.
        """This audit rides on the controller's tick, which must not die because one engine timed out."""
        api_client = _FakeApiClient(error=TimeoutError("no answer"))

        with caplog.at_level(logging.WARNING, logger="miles.ray.rollout.engine_env_reporter"):
            await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)

        assert "Failed to record the engine env of cell a" in caplog.text
        assert caplog.records[0].exc_info is not None
        assert _events(event_log_dir) == []

    async def test_a_hanging_engine_is_deadlined_and_retried_at_failure_cadence(
        self,
        event_log_dir: Path,
        fake_clock: list[float],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # [FUNCTION DOC] Uses an api client whose server info request blocks forever on an event, and
        # shrinks the reporter's server info timeout to 0.01 seconds. Starts the report as a task,
        # yields once and asserts it is still running, then waits up to one second for it to finish.
        # Asserts the failure warning, a timeout error as the logged exception, and no event written.
        # Then checks the failure retry cadence: advancing the fake clock to just before the retry
        # interval gives no new request (one call total); advancing a bit past it gives a second call.
        """A hanging server-info request must time out and follow the short failure retry cadence."""

        class _HangingApiClient(_FakeApiClient):
            # [CLASS DOC] HELPER: Fake api client whose server info request counts the call and then
            # waits on an event that is never set, so it never completes.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base fake and creates the never-set event.
                super().__init__()
                self.blocked = asyncio.Event()

            async def get_server_info(self) -> dict[str, Any]:
                # [FUNCTION DOC] HELPER: Counts the call, then waits forever on the event (the return
                # of the payload after it is unreachable in these tests).
                self.call_count += 1
                await self.blocked.wait()
                return self._server_info

        monkeypatch.setattr(engine_env_reporter_module, "SERVER_INFO_TIMEOUT_SECONDS", 0.01)
        reporter = EngineEnvReporter(interval_seconds=3600.0)
        api_client = _HangingApiClient()

        with caplog.at_level(logging.WARNING, logger="miles.ray.rollout.engine_env_reporter"):
            report_task = asyncio.create_task(_report(reporter, api_client))
            await asyncio.sleep(0)
            assert not report_task.done()
            await asyncio.wait_for(report_task, timeout=1.0)

        assert "Failed to record the engine env of cell a" in caplog.text
        assert isinstance(caplog.records[0].exc_info[1], TimeoutError)
        assert _events(event_log_dir) == []

        fake_clock[0] += RETRY_INTERVAL_SECONDS - 0.01
        await _report(reporter, api_client)
        assert api_client.call_count == 1

        fake_clock[0] += 0.02
        await _report(reporter, api_client)
        assert api_client.call_count == 2

    async def test_a_failed_reading_is_not_retried_on_every_tick(self, event_log_dir: Path, fake_clock) -> None:
        # [FUNCTION DOC] After a failing read, advances the clock by five seconds (less than the retry
        # interval), reports again, and asserts only the first request was made.
        """The sweep runs every few seconds, and a wedged engine would warn on each of them."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(error=RuntimeError("boom"))

        await _report(reporter, api_client)
        fake_clock[0] += 5.0
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_a_failed_reading_is_retried_long_before_the_next_interval(
        self, event_log_dir: Path, fake_clock
    ) -> None:
        # [FUNCTION DOC] After a failing read, advances the clock by exactly the retry interval, clears
        # the client's error, reports again, and asserts a second request was made and one event was
        # recorded for cell a.
        """A cell that answers nothing once would otherwise have no environment recorded for a whole hour."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient(error=RuntimeError("boom"))

        await _report(reporter, api_client)
        fake_clock[0] += RETRY_INTERVAL_SECONDS
        api_client.error = None
        await _report(reporter, api_client)

        assert api_client.call_count == 2
        assert [event.cell_id for event in _events(event_log_dir)] == ["a"]

    async def test_a_recorded_cell_is_not_retried_at_the_failure_cadence(
        self, event_log_dir: Path, fake_clock
    ) -> None:
        # [FUNCTION DOC] After a successful read, advances the clock by the retry interval (far less
        # than the hour interval), reports again, and asserts no second request was made.
        """The short retry exists for failures only; a cell that answered waits out the whole interval."""
        reporter, api_client = EngineEnvReporter(interval_seconds=3600.0), _FakeApiClient()

        await _report(reporter, api_client)
        fake_clock[0] += RETRY_INTERVAL_SECONDS
        await _report(reporter, api_client)

        assert api_client.call_count == 1

    async def test_a_run_without_an_event_logger_does_not_raise(self, fake_clock) -> None:
        # [FUNCTION DOC] Clears the global event logger, reports once, and asserts no exception and
        # that the api client was still asked once.
        """Not every run has an event dir, and the reporter must not raise when there is none."""
        set_event_logger(None)
        api_client = _FakeApiClient()

        await _report(EngineEnvReporter(interval_seconds=3600.0), api_client)

        assert api_client.call_count == 1
