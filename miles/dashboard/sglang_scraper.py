# [MODULE DOC] Periodic scraper of Prometheus-format inference-engine (sglang) metrics.
# Defines the whitelist of metric names kept, the label names preserved (engine type and data
# parallel rank), a scrape-mode enum (router or direct) and the SglangScraper class that runs a
# daemon thread fetching text-format metrics over HTTP, parsing them with the prometheus client
# parser and passing one engine sample record per accepted sample to a sink callable. The HTTP
# getter is injectable for tests; the default uses the requests library (imported lazily).
"""Scraper turning sglang engine metrics into dashboard ``EngineSample`` records.

Two fetch modes (design doc §6.6):

- ``router``: one ``GET {router_addr}/engine_metrics`` per tick. The sgl
  router fans the request out to every registered worker and labels each
  sample with ``worker_addr`` (sgl-model-gateway ``get_engine_metrics``).
- ``direct``: concurrent ``GET {addr}/metrics`` against every engine address
  from the topology. Needed under ``--use-miles-router``, whose router has no
  aggregation endpoint; here the scraper attaches the address itself.

Metric names are normalized (``sglang:num_running_reqs`` from a raw engine
equals ``sglang_num_running_reqs`` from the router aggregation), filtered by
a whitelist, and reduced to the label subset the dashboard uses. Records are
handed to a sink callable (the collector passes ``store.append``), so this
module has no collector or Ray dependency and is fully unit-testable.

Failure policy: a failed tick (or a failed engine in direct mode) leaves a
gap in the series — values are never fabricated — and warnings are
rate-limited to one per ``WARN_INTERVAL_SECONDS`` (pattern from
``examples/infra_features/random_async/random_async_sglang_metrics.py``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import EngineSample

logger = logging.getLogger(__name__)

DEFAULT_METRIC_WHITELIST = (
    "sglang_num_running_reqs",
    "sglang_num_queue_reqs",
    "sglang_gen_throughput",
    "sglang_token_usage",
    "sglang_cache_hit_rate",
    # PD disaggregation; these families simply don't exist when PD is off
    "sglang_num_prefill_prealloc_queue_reqs",
    "sglang_num_prefill_inflight_queue_reqs",
    "sglang_num_decode_prealloc_queue_reqs",
    "sglang_num_decode_transfer_queue_reqs",
    "sglang_kv_transfer_speed_gb_s",
    "sglang_kv_transfer_latency_ms",
    # cumulative families (counters + histogram _sum/_count): stored raw,
    # served as per-interval rates/means by Store.engine_series
    "sglang_prompt_tokens_total",
    "sglang_generation_tokens_total",
    "sglang_num_requests_total",
    "sglang_num_aborted_requests_total",
    "sglang_time_to_first_token_seconds_sum",
    "sglang_time_to_first_token_seconds_count",
    "sglang_inter_token_latency_seconds_sum",
    "sglang_inter_token_latency_seconds_count",
    "sglang_time_per_output_token_seconds_sum",
    "sglang_time_per_output_token_seconds_count",
    "sglang_e2e_request_latency_seconds_sum",
    "sglang_e2e_request_latency_seconds_count",
)

# label subset passed through to EngineSample.labels: engine_type (PD
# prefill/decode) and dp_rank (dp-attention engines export one sample per
# dp rank; dropping the label would fold the ranks into one series)
KEPT_LABELS = ("engine_type", "dp_rank")


class ScrapeMode(StrEnum):
    # [CLASS DOC] String enum of where metrics are fetched from: "router" (one aggregated request
    # to the router) or "direct" (one request per engine address).
    ROUTER = "router"
    DIRECT = "direct"


def _default_http_get(url: str, timeout: float) -> str:
    # [FUNCTION DOC] HELPER: Performs an HTTP GET with the given timeout in seconds using the
    # requests library (imported on call) and returns the response body text. Raises the library's
    # error for non-success status codes, timeouts and connection failures. Network I/O.
    import requests

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


class SglangScraper:
    # [CLASS DOC] Background scraper that turns engine metric text into engine sample records.
    # Lifecycle: start launches one daemon thread; stop signals it and waits up to timeout plus
    # one second. Each tick (every interval seconds, but never sleeping less than half a second
    # after a tick) scrapes once with the tick start time as the sample timestamp; any exception
    # from a tick is swallowed and reported through a rate-limited warning, leaving a gap.
    # Router mode makes one request to the router's aggregated endpoint; direct mode requests every
    # engine address (from a provider callable evaluated each tick) concurrently with at most 8
    # worker threads, and skips engines whose request fails.

    def __init__(
        self,
        sink: Callable[[EngineSample], None],
        *,
        mode: ScrapeMode,
        router_addr: str | None = None,
        engine_addrs: Callable[[], Iterable[str]] | None = None,
        interval: float = 2.0,
        timeout: float = 5.0,
        whitelist: Iterable[str] = DEFAULT_METRIC_WHITELIST,
        http_get: Callable[[str, float], str] = _default_http_get,
    ):
        # [FUNCTION DOC] Stores configuration. sink receives each sample record; mode selects the
        # fetch strategy; router_addr (base address, required in router mode) and engine_addrs (a
        # callable returning the current engine base addresses, required in direct mode) are
        # checked with assertions; interval and timeout are in seconds (defaults 2 and 5);
        # whitelist is the iterable of normalized metric names to keep (stored as a frozen set);
        # http_get is the fetch function taking a URL and timeout and returning text. Creates the
        # stop event and a rate-limited warner; the thread is not started.
        if mode == ScrapeMode.ROUTER:
            assert router_addr is not None, "router mode needs router_addr"
        else:
            assert engine_addrs is not None, "direct mode needs an engine_addrs provider"
        self._sink = sink
        self.mode = mode
        self.router_addr = router_addr
        self.engine_addrs = engine_addrs
        self.interval = interval
        self.timeout = timeout
        self.whitelist = frozenset(whitelist)
        self._http_get = http_get
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)

    # ------------------------------ lifecycle -------------------------------

    def start(self) -> None:
        # [FUNCTION DOC] Starts the scrape loop in a named daemon thread. Asserts it was not
        # started before (so a stopped scraper cannot be restarted).
        assert self._thread is None, "scraper already started"
        self._thread = threading.Thread(target=self._run, name="dashboard-sglang-scraper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # [FUNCTION DOC] Sets the stop event and, if started, joins the thread for at most timeout
        # plus one second (an in-flight tick may outlive the join). Safe to call without start.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout + 1)

    def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body. Until stopped: record the tick start time, run one
        # scrape with it (catching all exceptions into a rate-limited warning), then wait on the
        # stop event for the interval minus elapsed time, with a floor of 0.5 seconds.
        while not self._stop_event.is_set():
            started = time.time()
            try:
                self.scrape_once(started)
            except Exception:
                self._warn("sglang metrics scrape failed; leaving a gap")
            self._stop_event.wait(max(0.5, self.interval - (time.time() - started)))

    # ------------------------------- one tick -------------------------------

    def scrape_once(self, now: float) -> int:
        # [FUNCTION DOC] Fetches all sources, ingests each response stamping samples with now
        # (epoch seconds), and returns the total number of samples sent to the sink. Exceptions
        # from the router fetch or parsing propagate to the caller.
        """Fetch, parse and sink one round of samples. Returns the count."""
        count = 0
        for addr, text in self._fetch():
            count += self._ingest(addr, text, now)
        return count

    def _fetch(self) -> list[tuple[str | None, str]]:
        # [FUNCTION DOC] HELPER: Returns a list of pairs (engine address or None, metrics text).
        # Router mode: a single pair with no address from the router's engine-metrics endpoint
        # (errors propagate). Direct mode: evaluates the address provider, fetches each engine's
        # metrics endpoint in a thread pool of min(8, number of engines, at least 1) workers, and
        # drops engines whose fetch raised (warning rate-limited); order follows the address order.
        if self.mode == ScrapeMode.ROUTER:
            # addr None: the router aggregation carries worker_addr labels
            return [(None, self._http_get(f"{self.router_addr}/engine_metrics", self.timeout))]

        addrs = list(self.engine_addrs())

        def fetch_one(addr: str) -> tuple[str | None, str] | None:
            # [FUNCTION DOC] HELPER: Fetches one engine's metrics text and returns it with the
            # address; on any exception emits a warning and returns None.
            try:
                return addr, self._http_get(f"{addr}/metrics", self.timeout)
            except Exception:
                self._warn(f"scrape of engine {addr} failed; skipping this tick")
                return None

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(addrs)))) as pool:
            return [result for result in pool.map(fetch_one, addrs) if result is not None]

    def _ingest(self, addr: str | None, text: str, now: float) -> int:
        # [FUNCTION DOC] HELPER: Parses Prometheus text with the client library's parser (imported
        # on call) and sends an engine sample to the sink for every sample whose normalized name
        # (colons replaced by underscores) is in the whitelist. Address is the given engine
        # address, or in router mode the sample's worker address label; a sample lacking it is
        # dropped with a warning. Labels are reduced to the kept label names (engine type, dp
        # rank) and the value is converted to a float. Returns the number of samples sent.
        # Parse errors propagate.
        from prometheus_client.parser import text_string_to_metric_families

        count = 0
        for family in text_string_to_metric_families(text):
            for sample in family.samples:
                # raw engines export "sglang:x", the router aggregation "sglang_x";
                # histogram _sum/_count are sample names within a base-named family
                name = sample.name.replace(":", "_")
                if name not in self.whitelist:
                    continue
                sample_addr = addr if addr is not None else sample.labels.get("worker_addr")
                if sample_addr is None:
                    self._warn(f"router sample for {name} lacks worker_addr; dropping")
                    continue
                labels = {k: v for k, v in sample.labels.items() if k in KEPT_LABELS}
                self._sink(
                    EngineSample(ts=now, addr=sample_addr, metric=name, labels=labels, value=float(sample.value))
                )
                count += 1
        return count

    def _warn(self, message: str) -> None:
        # [FUNCTION DOC] HELPER: Forwards a message to the rate-limited warner (logging side effect).
        self._warner.warn(message)
