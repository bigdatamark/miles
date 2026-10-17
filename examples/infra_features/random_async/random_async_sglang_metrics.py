# [MODULE DOC] Background reporter that periodically scrapes the SGLang router's engine metrics
# endpoint (text in the OpenMetrics format), converts selected series into a small set of stable
# aggregate numbers, merges in client-side request statistics recorded by the rollout code, and
# logs everything to the active Weights and Biases run under a fixed key prefix.
# Two parts: (1) a thread-safe module-level accumulator of per-request statistics (a window that
# is emptied each scrape, plus cumulative totals) fed by the record function; (2) a reporter
# class with its own daemon thread. The scrape interval is read from an environment variable at
# import (default 30 seconds). Errors while scraping are logged at most once per 5 minutes and
# never stop the loop; a missing wandb run is such an error.
"""Example-local SGLang metrics scraper for random_async.

Miles can forward raw SGLang OpenMetrics to W&B, but this helper logs a small
set of stable aggregate keys under ``random_async_sglang/`` for the stress test.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

METRICS_INTERVAL_SECONDS = float(os.environ.get("RANDOM_ASYNC_SGLANG_METRICS_INTERVAL_SECONDS", "30"))

_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')
_AGENT_METRICS_LOCK = threading.Lock()
_AGENT_WINDOW: dict[str, float] = {
    "output_tokens": 0.0,
    "prompt_tokens": 0.0,
    "cached_tokens": 0.0,
    "perfect_cacheable_tokens": 0.0,
    "request_time": 0.0,
    "max_request_time": 0.0,
    "request_count": 0.0,
}
_AGENT_CUMULATIVE: dict[str, float] = {
    "output_tokens": 0.0,
    "prompt_tokens": 0.0,
    "cached_tokens": 0.0,
    "perfect_cacheable_tokens": 0.0,
    "request_count": 0.0,
}


@dataclass(frozen=True)
class MetricSample:
    # [CLASS DOC] HELPER: Immutable record of one parsed metric line: the metric name (colons
    # replaced by underscores), its label mapping from name to unescaped value, and the numeric value.
    name: str
    labels: dict[str, str]
    value: float


def record_agent_request(
    *,
    output_tokens: int,
    prompt_tokens: int,
    cached_tokens: int,
    perfect_cacheable_tokens: int,
    request_time: float,
) -> None:
    # [FUNCTION DOC] Called by the rollout after each engine request. Under the module lock, adds
    # the output, prompt, cached and ideal-cacheable token counts and the request duration to the
    # current window sums, raises the window's max request time if exceeded, and increments the
    # window request count. Keyword-only arguments; no return.
    with _AGENT_METRICS_LOCK:
        _AGENT_WINDOW["output_tokens"] += output_tokens
        _AGENT_WINDOW["prompt_tokens"] += prompt_tokens
        _AGENT_WINDOW["cached_tokens"] += cached_tokens
        _AGENT_WINDOW["perfect_cacheable_tokens"] += perfect_cacheable_tokens
        _AGENT_WINDOW["request_time"] += request_time
        _AGENT_WINDOW["max_request_time"] = max(_AGENT_WINDOW["max_request_time"], request_time)
        _AGENT_WINDOW["request_count"] += 1


def _pop_agent_metrics() -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Atomically copy and zero the request window. Returns an empty mapping
    # if no requests were recorded. Otherwise adds the window sums to the cumulative totals and
    # returns four client-side metrics: ideal cacheable tokens divided by prompt tokens (perfect
    # cache hit rate), actual cached tokens divided by ideal cacheable tokens (each ratio is 0
    # when its denominator is 0), average request time and max request time.
    # KNOWN ISSUE: the cumulative totals are updated here (outside the lock) but nothing ever
    # reads or reports them.
    with _AGENT_METRICS_LOCK:
        window = dict(_AGENT_WINDOW)
        for key in _AGENT_WINDOW:
            _AGENT_WINDOW[key] = 0.0

    if window["request_count"] == 0:
        return {}

    _AGENT_CUMULATIVE["output_tokens"] += window["output_tokens"]
    _AGENT_CUMULATIVE["prompt_tokens"] += window["prompt_tokens"]
    _AGENT_CUMULATIVE["cached_tokens"] += window["cached_tokens"]
    _AGENT_CUMULATIVE["perfect_cacheable_tokens"] += window["perfect_cacheable_tokens"]
    _AGENT_CUMULATIVE["request_count"] += window["request_count"]

    prompt_tokens = window["prompt_tokens"]
    cached_tokens = window["cached_tokens"]
    perfect_cacheable_tokens = window["perfect_cacheable_tokens"]
    request_count = window["request_count"]
    return {
        "random_async_sglang/agent_perfect_cache_hit_rate": (
            perfect_cacheable_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
        ),
        "random_async_sglang/agent_cache_to_perfect_cache_ratio": (
            cached_tokens / perfect_cacheable_tokens if perfect_cacheable_tokens > 0 else 0.0
        ),
        "random_async_sglang/agent_avg_request_time": window["request_time"] / request_count,
        "random_async_sglang/agent_max_request_time": window["max_request_time"],
    }


def _parse_labels(raw: str | None) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Parse the inside of a label brace group (name equals quoted value,
    # comma separated) into a mapping; quoted values may contain escaped characters, which are
    # decoded through a unicode-escape pass. Empty or missing input gives an empty mapping.
    if not raw:
        return {}
    return {
        match.group(1): bytes(match.group(2), "utf-8").decode("unicode_escape") for match in _LABEL_RE.finditer(raw)
    }


def parse_openmetrics(text: str) -> list[MetricSample]:
    # [FUNCTION DOC] Parse OpenMetrics or Prometheus text into metric samples. Each line is
    # stripped; blank lines and comment lines (starting with a hash) and lines not matching the
    # pattern of name, optional labels and a numeric value are silently skipped (so lines with
    # timestamps or non-numeric values like NaN are dropped). Names have colons replaced by
    # underscores. Returns samples in input order.
    samples: list[MetricSample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _METRIC_RE.match(line)
        if match is None:
            continue
        samples.append(
            MetricSample(
                name=match.group("name").replace(":", "_"),
                labels=_parse_labels(match.group("labels")),
                value=float(match.group("value")),
            )
        )
    return samples


def _matches_engine_type(labels: dict[str, str], engine_type: str | None) -> bool:
    # [FUNCTION DOC] HELPER: True when no engine type filter is given, else whether the sample's
    # engine type label equals it. Raises a key error if the filter is given but the sample has
    # no engine type label.
    if engine_type is None:
        return True
    return labels["engine_type"] == engine_type


def _sum(samples: list[MetricSample], name: str, *, engine_type: str | None = None, mode: str | None = None) -> float:
    # [FUNCTION DOC] HELPER: Sum the values of all samples with the given name, optionally
    # restricted to an engine type label and/or a mode label (a sample lacking the mode label
    # never matches when a mode is given). Returns 0 if none match.
    total = 0.0
    for sample in samples:
        if sample.name != name:
            continue
        if not _matches_engine_type(sample.labels, engine_type):
            continue
        if mode is not None and sample.labels.get("mode") != mode:
            continue
        total += sample.value
    return total


def _has(samples: list[MetricSample], name: str, *, engine_type: str | None = None, mode: str | None = None) -> bool:
    # [FUNCTION DOC] HELPER: Whether any sample has the given name and passes the same optional
    # engine type and mode filters as the sum helper.
    return any(
        sample.name == name
        and _matches_engine_type(sample.labels, engine_type)
        and (mode is None or sample.labels.get("mode") == mode)
        for sample in samples
    )


def _avg(samples: list[MetricSample], name: str, *, engine_type: str | None = None) -> float | None:
    # [FUNCTION DOC] HELPER: Arithmetic mean of the values of samples with the given name
    # (optionally filtered by engine type); returns none if there are none.
    values = [
        sample.value for sample in samples if sample.name == name and _matches_engine_type(sample.labels, engine_type)
    ]
    if not values:
        return None
    return sum(values) / len(values)


class SGLangMetricsReporter:
    # [CLASS DOC] Periodic scraper and logger. State: router base URL without trailing slash,
    # prefill and decode GPU counts (each at least 1, used to normalize per-GPU rates), a stop
    # event, the thread, an exposed failure attribute, the previous cumulative token counters and
    # timestamp (for rate computation) and the time of the last logged scrape failure.
    # KNOWN ISSUE: the failure attribute is never assigned by this class (scrape errors are only
    # logged), so the rollout worker's check of it can never fire.

    def __init__(self, router_url: str, prefill_num_gpus: int, decode_num_gpus: int):
        # [FUNCTION DOC] Store the URL (trailing slash stripped), clamp both GPU counts to at
        # least 1 and initialize empty rate-tracking state. Does not start the thread.
        self.router_url = router_url.rstrip("/")
        self.prefill_num_gpus = max(1, prefill_num_gpus)
        self.decode_num_gpus = max(1, decode_num_gpus)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.failure: BaseException | None = None
        self._previous_realtime_tokens: dict[str, float] = {}
        self._previous_time: float | None = None
        self._last_failure_log_time = 0.0

    def start(self) -> None:
        # [FUNCTION DOC] Start the daemon scrape thread unless one is already alive; logs at info.
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="random-async-sglang-metrics", daemon=True)
        self._thread.start()
        logger.info("Started random_async SGLang metrics reporter for %s", self.router_url)

    def stop(self) -> None:
        # [FUNCTION DOC] Signal the thread to stop and wait up to 5 seconds for it to finish.
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body. Until stopped: fetch the router's engine metrics
        # endpoint with a 10 second timeout (an error status raises), parse the text, build the
        # engine metrics from it using the scrape start time as the clock, merge in the client-side
        # request metrics, and log the merged mapping to wandb if non-empty. Any exception is
        # swallowed; a warning with traceback is logged at most once per 300 seconds. Then waits
        # the interval minus elapsed time, but never less than 1 second, or until stop.
        session = requests.Session()
        while not self._stop_event.is_set():
            start = time.time()
            metrics: dict[str, float] = {}
            try:
                response = session.get(f"{self.router_url}/engine_metrics", timeout=10)
                response.raise_for_status()
                samples = parse_openmetrics(response.text)
                metrics.update(self._build_metrics(samples, start))
                metrics.update(_pop_agent_metrics())
                if metrics:
                    self._log_to_wandb(metrics)
            except Exception:
                now = time.time()
                if now - self._last_failure_log_time > 300:
                    logger.warning("random_async SGLang metrics scrape failed; continuing", exc_info=True)
                    self._last_failure_log_time = now
            elapsed = time.time() - start
            self._stop_event.wait(max(1.0, METRICS_INTERVAL_SECONDS - elapsed))

    def _build_metrics(self, samples: list[MetricSample], now: float) -> dict[str, float]:
        # [FUNCTION DOC] HELPER: Turn parsed samples into a mapping of prefixed metric keys to
        # floats; mutates the stored previous counters and timestamp. Only metrics whose source
        # series are present are emitted. Emitted: (a) decode generation throughput summed over
        # decode engines divided by decode GPUs; (b) mean prefill cache hit rate; (c) for each of
        # prefill and decode engine types, token usage, queued requests and running requests, each
        # summed across engines; (d) a fixed list of router-wide series (prefill/decode queue
        # lengths, KV transfer speed, latency, bootstrap, alloc, total size, retracted requests,
        # bootstrap/transfer failures and prefill retries), each summed across all series;
        # (e) token rates: the cumulative realtime token counters for decode, prefill compute and
        # prefill cache modes are compared with the previous scrape; the delta over elapsed time
        # (at least one microsecond) gives tokens per second, skipped for a mode with no previous
        # value or a negative delta (counter reset); decode rate is per decode GPU, prefill
        # compute, prefill cache and their total (only if positive) are per prefill GPU. The first
        # call only records baselines. Always records the current counters and time.
        metrics: dict[str, float] = {}

        if _has(samples, "sglang_gen_throughput", engine_type="decode"):
            decode_gen_throughput = _sum(samples, "sglang_gen_throughput", engine_type="decode")
            metrics["random_async_sglang/sglang_reported_decode_gen_throughput_per_decode_gpu"] = (
                decode_gen_throughput / self.decode_num_gpus
            )

        prefill_cache_hit_rate = _avg(samples, "sglang_cache_hit_rate", engine_type="prefill")
        if prefill_cache_hit_rate is not None:
            metrics["random_async_sglang/sglang_cache_hit_rate"] = prefill_cache_hit_rate

        for engine_type in ("prefill", "decode"):
            for name, suffix in (
                ("sglang_token_usage", "token_usage"),
                ("sglang_num_queue_reqs", "queue_reqs"),
                ("sglang_num_running_reqs", "running_reqs"),
            ):
                if _has(samples, name, engine_type=engine_type):
                    value = _sum(samples, name, engine_type=engine_type)
                    metrics[f"random_async_sglang/{engine_type}_{suffix}"] = value

        for name, key in (
            ("sglang_num_prefill_prealloc_queue_reqs", "prefill_prealloc_queue_reqs"),
            ("sglang_num_prefill_inflight_queue_reqs", "prefill_inflight_queue_reqs"),
            ("sglang_num_decode_prealloc_queue_reqs", "decode_prealloc_queue_reqs"),
            ("sglang_num_decode_transfer_queue_reqs", "decode_transfer_queue_reqs"),
            ("sglang_kv_transfer_speed_gb_s", "kv_transfer_speed_gb_s"),
            ("sglang_kv_transfer_latency_ms", "kv_transfer_latency_ms"),
            ("sglang_kv_transfer_bootstrap_ms", "kv_transfer_bootstrap_ms"),
            ("sglang_kv_transfer_alloc_ms", "kv_transfer_alloc_ms"),
            ("sglang_kv_transfer_total_mb", "kv_transfer_total_mb"),
            ("sglang_num_retracted_reqs", "retracted_reqs"),
            ("sglang_num_bootstrap_failed_reqs_total", "bootstrap_failed_reqs_total"),
            ("sglang_num_transfer_failed_reqs_total", "transfer_failed_reqs_total"),
            ("sglang_num_prefill_retries_total", "prefill_retries_total"),
        ):
            if _has(samples, name):
                value = _sum(samples, name)
                metrics[f"random_async_sglang/{key}"] = value

        realtime_tokens = {
            mode: _sum(samples, "sglang_realtime_tokens_total", mode=mode)
            for mode in ("decode", "prefill_compute", "prefill_cache")
        }
        if self._previous_time is not None:
            dt = max(now - self._previous_time, 1e-6)
            tps_by_mode: dict[str, float] = {}
            for mode, value in realtime_tokens.items():
                previous = self._previous_realtime_tokens.get(mode)
                if previous is None:
                    continue
                delta = value - previous
                if delta >= 0:
                    tps_by_mode[mode] = delta / dt
            if "decode" in tps_by_mode:
                metrics["random_async_sglang/decode_tps_per_decode_gpu"] = tps_by_mode["decode"] / self.decode_num_gpus
            if "prefill_compute" in tps_by_mode:
                metrics["random_async_sglang/prefill_without_cache_tps_per_prefill_gpu"] = (
                    tps_by_mode["prefill_compute"] / self.prefill_num_gpus
                )
            if "prefill_cache" in tps_by_mode:
                metrics["random_async_sglang/prefill_cache_read_tps_per_prefill_gpu"] = (
                    tps_by_mode["prefill_cache"] / self.prefill_num_gpus
                )
            prefill_total_tps = tps_by_mode.get("prefill_compute", 0.0) + tps_by_mode.get("prefill_cache", 0.0)
            if prefill_total_tps > 0:
                metrics["random_async_sglang/prefill_total_tps_per_prefill_gpu"] = (
                    prefill_total_tps / self.prefill_num_gpus
                )
        self._previous_realtime_tokens = realtime_tokens
        self._previous_time = now

        return metrics

    def _log_to_wandb(self, metrics: dict[str, float]) -> None:
        # [FUNCTION DOC] HELPER: Import wandb lazily and log the mapping to the active run.
        # Raises a runtime error if no run is active.
        import wandb

        if wandb.run is None:
            raise RuntimeError("random_async SGLang metrics reporter requires an active W&B run")
        wandb.log(metrics)
