# [MODULE DOC]
# HTTP client side of the NeMo Skills evaluation delegate. A single client class turns the
# configured Skills datasets into a JSON request (rollout id, inference router URL and a list of
# per-benchmark parameter mappings), posts it to the configured server URL with a timeout and
# bounded retries with exponential backoff, and returns the server's raw metrics together with the
# full response. Uses a persistent requests session.
import logging
import time
from typing import Any

import requests
from examples.experimental.eval.eval_delegate import EvalClient, EvalDelegateError
from examples.experimental.eval.nemo_skills.skills_config import SkillsEvalEnvConfig

logger = logging.getLogger(__name__)


class SkillsEvalClient(EvalClient):
    # [CLASS DOC]
    # Evaluation client for the skills environment. State: the environment record, the router URL
    # without a trailing slash, the server endpoint URL (trailing slash removed), the request
    # timeout in seconds (float), the retry attempt count (at least one), the extra HTTP headers
    # and a requests session reused across calls.
    """HTTP client that proxies evaluation requests to the NeMo Skills server."""

    def __init__(self, config: SkillsEvalEnvConfig, router_url: str):
        # [FUNCTION DOC] Initialises from the environment record and router URL. The client name
        # is the record's name or "skills" if empty. Normalises the URLs, converts the timeout to
        # float, clamps the retry count to at least one, copies headers (none treated as empty) and
        # creates a session.
        super().__init__(config.name or "skills")
        self._config = config
        self._router_url = router_url.rstrip("/")
        self._endpoint = (config.url or "").rstrip("/")
        self._timeout_secs = float(config.timeout_secs)
        self._max_retries = max(1, int(config.max_retries))
        self._headers = dict(config.headers or {})
        self._session = requests.Session()

    @classmethod
    def from_config(cls, config: SkillsEvalEnvConfig, router_url: str):
        # [FUNCTION DOC] Factory used by the delegate aggregator: returns none if the record has no
        # server URL, else a new client.
        if not config.url:
            return None
        return cls(config, router_url)

    def evaluate(self, args, rollout_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        # [FUNCTION DOC] Runs a delegated evaluation for the rollout id. With no configured
        # datasets, logs a warning and returns a pair of two empty mappings without any request.
        # Otherwise builds the payload, posts it (with retries) and returns the response's raw
        # metrics entry together with the whole response mapping.
        # KNOWN ISSUE: if the server answers with an empty body the request helper returns an empty
        # mapping and indexing the raw metrics key then raises a key error. Also the environment
        # record's defaults are never put into the payload, although the server accepts a defaults
        # field and uses it as fallback for unset dataset parameters.
        if not self._config.datasets:
            logger.warning("No Skills datasets configured; skipping delegate evaluation.")
            return {}, {}

        payload = self._build_payload(args, rollout_id)
        response = self._request(payload)
        metrics = response["raw_metrics"]
        return metrics, response

    def _build_payload(self, args, rollout_id: int) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Builds the request body: the rollout id, the router URL and a
        # benchmarks list holding each dataset record's payload mapping (non-none fields only);
        # datasets with an empty payload are dropped. The launch args are unused.
        benchmarks = [cfg.to_payload() for cfg in self._config.datasets]
        benchmarks = [cfg for cfg in benchmarks if cfg]
        return {
            "rollout_id": rollout_id,
            "router_url": self._router_url,
            "benchmarks": benchmarks,
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Posts the payload as JSON to the endpoint with the configured
        # headers and timeout, up to the retry count. A success status with an empty body returns
        # an empty mapping; otherwise returns the decoded JSON. On any request-library exception
        # (including bad status via raise-for-status and timeouts) logs a warning and, if attempts
        # remain, sleeps min(2 to the attempt number, 30) seconds. After the last failure raises a
        # delegate error chained to the last exception. Note a JSON decode failure in a successful
        # response is not caught here unless the library treats it as a request exception.
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = self._session.post(
                    self._endpoint,
                    json=payload,
                    timeout=self._timeout_secs,
                    headers=self._headers,
                )
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "Skills eval delegate request failed (attempt %s/%s): %s", attempt, self._max_retries, exc
                )
                if attempt < self._max_retries:
                    time.sleep(min(2**attempt, 30))
        raise EvalDelegateError("Skills evaluation request failed") from last_error
