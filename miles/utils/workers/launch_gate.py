# [MODULE DOC]
# Client side of the launch gate: a worker that must wait before starting its real work is held
# by a gate service, and this module opens that gate by posting an activation request to it,
# retrying on transient network errors until a deadline.
import httpx

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.retry_utils import retry_until_deadline

GATE_PORT_NAME = "gate"

_INITIAL_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 5.0
_ATTEMPT_TIMEOUT_SECONDS = 30.0


async def activate_launch_gate(gate_url: str, timeout: float = 1800.0) -> None:
    # [FUNCTION DOC]
    # Activate the gate at the given base url by posting an empty json body to its activate path,
    # using the shared general http client. Retries on http errors and operating-system errors
    # with delays growing from 1 second up to 5 seconds, for at most the timeout in seconds
    # (default 1800, i.e. 30 minutes), then lets the last error propagate. Each attempt is bounded
    # by 30 seconds or the remaining time, whichever is smaller. A non-success status counts as an
    # http error and is retried.
    async def _activate(remaining_seconds: float) -> None:
        # [FUNCTION DOC] HELPER: One attempt: post the activation request with a timeout of the
        # smaller of the per-attempt limit and the remaining seconds, and raise on an error status.
        response = await GeneralHttpClientProvider.client().post(
            f"{gate_url}/gate/activate",
            json={},
            timeout=min(_ATTEMPT_TIMEOUT_SECONDS, remaining_seconds),
        )
        response.raise_for_status()

    await retry_until_deadline(
        _activate,
        total_seconds=timeout,
        retry_on=(httpx.HTTPError, OSError),
        initial_delay=_INITIAL_DELAY_SECONDS,
        max_delay=_MAX_DELAY_SECONDS,
        log_fields=dict(gate_url=gate_url),
    )
