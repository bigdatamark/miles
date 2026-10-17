# [MODULE DOC] Periodic reporting of an inference engine's environment (its server info, redacted)
# into the audit event log. Constants: a 10 second timeout for fetching server info and a 60 second
# retry interval after a failed report.
import asyncio
import logging
import math
import time
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import EngineEnvReportEvent
from miles.utils.env_report.redaction import redact_server_info

logger = logging.getLogger(__name__)
SERVER_INFO_TIMEOUT_SECONDS = 10.0
RETRY_INTERVAL_SECONDS = 60.0


class EngineEnvReporter:
    # [CLASS DOC] Rate-limits environment reports for engines. Holds the configured success
    # interval and a monotonic-clock deadline for the next report (unset until the first attempt).
    # State is shared across all engines passed to it, i.e. one deadline per reporter instance.

    def __init__(self, *, interval_seconds: float) -> None:
        # [FUNCTION DOC] Store the interval in seconds between successful reports (zero or negative
        # means report only once successfully) and clear the next-due deadline.
        self._interval_seconds = interval_seconds
        self._next_due: float | None = None

    async def report_if_due(self, *, cell_id: str, server_url: str, api_client: SGLangApiClient) -> None:
        # [FUNCTION DOC] If the next-due deadline has not passed (monotonic clock), return at once.
        # Otherwise attempt a report for the cell and set the next deadline to now plus the delay
        # that depends on success (see delay helper). Never raises for report failures.
        now = time.monotonic()
        if self._next_due is not None and now < self._next_due:
            return

        reported = await self._report(cell_id=cell_id, server_url=server_url, api_client=api_client)
        self._next_due = now + self._delay_after(reported=reported)

    def _delay_after(self, *, reported: bool) -> float:
        # [FUNCTION DOC] HELPER: Seconds until the next attempt: the fixed retry interval after a
        # failure; after success the configured interval, or infinity (never again) if that
        # interval is not positive.
        if not reported:
            return RETRY_INTERVAL_SECONDS
        return self._interval_seconds if self._interval_seconds > 0 else math.inf

    async def _report(self, *, cell_id: str, server_url: str, api_client: SGLangApiClient) -> bool:
        # [FUNCTION DOC] HELPER: Fetch server info from the engine's api client with a timeout, then,
        # only if the event logger is initialized, log an engine env report event (cell id, server
        # url, redacted server info) without echoing it to the console log. Returns true on success
        # (including when no event logger exists, in which case nothing is logged), false on any
        # exception (a warning with traceback is logged). The server url is not used for the fetch;
        # it is only recorded.
        try:
            server_info = await asyncio.wait_for(api_client.get_server_info(), timeout=SERVER_INFO_TIMEOUT_SECONDS)
            if is_event_logger_initialized():
                get_event_logger().log(
                    EngineEnvReportEvent,
                    {"cell_id": cell_id, "server_url": server_url, "server_info": redact_server_info(server_info)},
                    print_log=False,
                )
            return True
        except Exception:
            logger.warning("Failed to record the engine env of cell %s", cell_id, exc_info=True)
            return False
