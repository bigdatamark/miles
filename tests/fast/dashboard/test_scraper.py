# [MODULE DOC]
# Tests for the sglang metrics scraper using canned prometheus text pages (router and direct
# engine fixtures from the fixtures directory, read at import time) and fake http getters.
# Covers router mode (worker address lifted into the engine address, whitelist filtering, label
# filtering), direct mode (colon names normalised, per-engine address attached, failed engines
# skipped), rate-limited warnings, dropped router samples without a worker address, whitelist
# override, constructor prerequisites per mode, the default whitelist contents and the data
# parallel rank label.
import logging
from pathlib import Path

import pytest

from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST, ScrapeMode, SglangScraper
from miles.dashboard.store import EngineSample

FIXTURES = Path(__file__).parent / "fixtures"
ROUTER_TEXT = (FIXTURES / "engine_metrics_router.txt").read_text()
DIRECT_TEXT = (FIXTURES / "engine_metrics_direct.txt").read_text()

ENGINE_A = "http://10.0.0.1:15000"
ENGINE_B = "http://10.0.0.1:15004"


def router_scraper(sink, http_get, **kwargs):
    # [FUNCTION DOC] HELPER: Builds a scraper in router mode pointed at "http://router:3000" with
    # the given sample sink and http getter; extra keyword arguments (e.g. whitelist) pass through.
    return SglangScraper(sink, mode=ScrapeMode.ROUTER, router_addr="http://router:3000", http_get=http_get, **kwargs)


def test_router_mode_lifts_worker_addr_and_filters(caplog):
    # [FUNCTION DOC] One router scrape at time 123 requests only the router's engine metrics
    # url, returns and emits 5 samples (2 running, 2 throughput, 1 token usage; an unlisted
    # metric is dropped), all stamped 123. The engine address of each sample is the worker
    # address label lifted from the page (running values 42 and 7 for engines A and B). Only the
    # engine type label survives; model name and worker address labels are removed.
    # KNOWN ISSUE: the caplog fixture parameter is unused.
    records: list[EngineSample] = []
    urls: list[str] = []

    def fake_get(url, timeout):
        # [FUNCTION DOC] HELPER: Records the requested url and returns the router fixture text.
        urls.append(url)
        return ROUTER_TEXT

    scraper = router_scraper(records.append, fake_get)
    count = scraper.scrape_once(now=123.0)

    assert urls == ["http://router:3000/engine_metrics"]
    assert count == len(records) == 5  # 2 running + 2 throughput + 1 token_usage; unlisted name dropped
    assert {r.metric for r in records} == {"sglang_num_running_reqs", "sglang_gen_throughput", "sglang_token_usage"}
    assert all(r.ts == 123.0 for r in records)

    running = {r.addr: r.value for r in records if r.metric == "sglang_num_running_reqs"}
    assert running == {ENGINE_A: 42.0, ENGINE_B: 7.0}

    # only the engine_type label survives; model_name and worker_addr are dropped
    token_usage = next(r for r in records if r.metric == "sglang_token_usage")
    assert token_usage.labels == {"engine_type": "prefill"}
    assert all("worker_addr" not in r.labels and "model_name" not in r.labels for r in records)


def test_direct_mode_normalizes_colon_names_and_attaches_addr():
    # [FUNCTION DOC] Direct mode with one engine: only that engine's metrics url is requested
    # (any other url fails the test). The tick emits 2 samples (an uptime metric is filtered out)
    # with colon-style names normalised to underscore names, the engine address attached, and
    # values 13.0 running requests and 0.71 cache hit rate.
    records: list[EngineSample] = []
    scraper = SglangScraper(
        records.append,
        mode=ScrapeMode.DIRECT,
        engine_addrs=lambda: [ENGINE_A],
        http_get=lambda url, timeout: DIRECT_TEXT if url == f"{ENGINE_A}/metrics" else pytest.fail(url),
    )
    count = scraper.scrape_once(now=5.0)

    assert count == 2  # uptime is filtered out
    assert {(r.addr, r.metric, r.value) for r in records} == {
        (ENGINE_A, "sglang_num_running_reqs", 13.0),
        (ENGINE_A, "sglang_cache_hit_rate", 0.71),
    }


def test_direct_mode_skips_failed_engine(caplog):
    # [FUNCTION DOC] Two engines, the second's request raising a connection error. The tick still
    # returns 2 samples (from engine A only) and exactly one warning containing "skipping this
    # tick" is logged.
    records: list[EngineSample] = []

    def flaky_get(url, timeout):
        # [FUNCTION DOC] HELPER: Raises a connection error for engine B's url; otherwise returns
        # the direct fixture text.
        if ENGINE_B in url:
            raise ConnectionError("engine restarting")
        return DIRECT_TEXT

    scraper = SglangScraper(
        records.append,
        mode=ScrapeMode.DIRECT,
        engine_addrs=lambda: [ENGINE_A, ENGINE_B],
        http_get=flaky_get,
    )
    with caplog.at_level(logging.WARNING):
        count = scraper.scrape_once(now=1.0)

    assert count == 2  # engine A's samples still land
    assert {r.addr for r in records} == {ENGINE_A}
    assert len([r for r in caplog.records if "skipping this tick" in r.message]) == 1


def test_warnings_are_rate_limited(caplog):
    # [FUNCTION DOC] With an always-failing engine, two ticks in a row log exactly one warning
    # (second suppressed in the window). After resetting the warner's window, a third tick logs
    # a second warning.
    scraper = SglangScraper(
        lambda r: None,
        mode=ScrapeMode.DIRECT,
        engine_addrs=lambda: [ENGINE_A],
        http_get=lambda url, timeout: (_ for _ in ()).throw(ConnectionError("down")),
    )
    with caplog.at_level(logging.WARNING):
        scraper.scrape_once(now=1.0)
        scraper.scrape_once(now=2.0)  # within the suppression window
    assert len(caplog.records) == 1

    scraper._warner.reset_window_for_test()  # window elapses
    with caplog.at_level(logging.WARNING):
        scraper.scrape_once(now=3.0)
    assert len(caplog.records) == 2


def test_router_sample_without_worker_addr_is_dropped(caplog):
    # [FUNCTION DOC] A router page whose only sample lacks the worker address label yields 0
    # samples, nothing emitted, and a warning whose message says the sample "lacks worker_addr".
    records: list[EngineSample] = []
    text = '# TYPE sglang_num_running_reqs gauge\nsglang_num_running_reqs{model_name="q"} 3.0\n'
    scraper = router_scraper(records.append, lambda url, timeout: text)
    with caplog.at_level(logging.WARNING):
        assert scraper.scrape_once(now=1.0) == 0
    assert records == []
    assert "lacks worker_addr" in caplog.records[0].message


def test_whitelist_override():
    # [FUNCTION DOC] A router scraper given a whitelist of only gpu generation throughput emits
    # only samples of that metric.
    records: list[EngineSample] = []
    scraper = router_scraper(records.append, lambda url, timeout: ROUTER_TEXT, whitelist=("sglang_gen_throughput",))
    scraper.scrape_once(now=1.0)
    assert {r.metric for r in records} == {"sglang_gen_throughput"}


def test_mode_prerequisites():
    # [FUNCTION DOC] Router mode without a router address raises an assertion error mentioning
    # router_addr; direct mode without an engine address provider raises one mentioning
    # engine_addrs.
    with pytest.raises(AssertionError, match="router_addr"):
        SglangScraper(lambda r: None, mode=ScrapeMode.ROUTER)
    with pytest.raises(AssertionError, match="engine_addrs"):
        SglangScraper(lambda r: None, mode=ScrapeMode.DIRECT)


def test_default_whitelist_covers_design_set():
    # [FUNCTION DOC] The default whitelist contains running requests and the prefill-decode
    # kv transfer speed metric.
    assert "sglang_num_running_reqs" in DEFAULT_METRIC_WHITELIST
    assert "sglang_kv_transfer_speed_gb_s" in DEFAULT_METRIC_WHITELIST  # PD set present


def test_dp_rank_label_survives():
    # [FUNCTION DOC] A direct page with two samples differing in dp_rank (and tp_rank) labels
    # yields 2 samples keeping the dp_rank label (values 11.0 for "0" and 1.0 for "1") and
    # dropping the tp_rank and model name labels.
    records: list[EngineSample] = []
    text = (
        "# TYPE sglang:num_running_reqs gauge\n"
        'sglang:num_running_reqs{model_name="q",engine_type="unified",dp_rank="0",tp_rank="0"} 11.0\n'
        'sglang:num_running_reqs{model_name="q",engine_type="unified",dp_rank="1",tp_rank="1"} 1.0\n'
    )
    scraper = SglangScraper(
        records.append, mode=ScrapeMode.DIRECT, engine_addrs=lambda: [ENGINE_A], http_get=lambda url, timeout: text
    )
    assert scraper.scrape_once(now=1.0) == 2
    assert {(r.labels["dp_rank"], r.value) for r in records} == {("0", 11.0), ("1", 1.0)}
    assert all("tp_rank" not in r.labels and "model_name" not in r.labels for r in records)
