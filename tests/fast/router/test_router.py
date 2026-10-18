# [MODULE DOC] Tests for the miles router, a load-balancing reverse proxy over SGLang workers.
# Covers: client construction from the config; worker registration and removal over HTTP (query
# parameter or JSON body, idempotency, bookkeeping cleanup, listing); least-loaded worker selection
# that skips quarantined workers; request-slot release after deregistration; the background health
# check loop (interval, consecutive-failure quarantine, handling of results for workers removed
# during a probe, re-adding quarantined urls); and proxying through a mock SGLang worker. Real local
# HTTP servers on background threads are used for the integration cases (router ports from 20000 and
# worker ports from 30000); loop timing is controlled with fake sleep clocks and scripted health
# results.
import asyncio
from collections.abc import Callable
from urllib.parse import urlencode

import httpx
import pytest
import requests
from fastapi import Request

from miles.router import router as router_module
from miles.router.config import MilesRouterConfig
from miles.router.router import MilesRouter
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.mock_sglang_server import MockSGLangServer, default_process_fn
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer


def make_router_config(router_port: int, **overrides) -> MilesRouterConfig:
    # [FUNCTION DOC] HELPER: Builds a router config for the loopback address and the given port with
    # defaults: health check interval 1.0 s, failure threshold 3, 100 max connections, no timeout.
    # Keyword overrides replace defaults. Returns the validated config.
    defaults = dict(
        host="127.0.0.1",
        port=router_port,
        health_check_interval=1.0,
        health_check_failure_threshold=3,
        max_connections=100,
        timeout=None,
    )
    defaults.update(overrides)
    return MilesRouterConfig(**defaults)


def create_mock_worker(start_port: int = 30000) -> MockSGLangServer:
    # [FUNCTION DOC] HELPER: Finds an available port at or after the start port and returns a mock
    # SGLang server (not yet started) for model Qwen/Qwen3-0.6B on the loopback address using the
    # default processing function and zero latency.
    port = find_available_port(start_port)
    return MockSGLangServer(
        model_name="Qwen/Qwen3-0.6B",
        process_fn=default_process_fn,
        host="127.0.0.1",
        port=port,
        latency=0.0,
    )


def make_add_worker_request(worker_url: str) -> Request:
    # [FUNCTION DOC] HELPER: Builds a bare POST web request for the add-worker path whose query
    # string carries the url-encoded worker url, with no headers and no body, to call the router's
    # add-worker handler directly without HTTP.
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/add_worker",
        "query_string": urlencode({"url": worker_url}).encode(),
        "headers": [],
    }
    return Request(scope)


class RouterEnv:
    # [CLASS DOC] HELPER: Pair of a router and the background HTTP server running its application.
    def __init__(self, router: MilesRouter, server: UvicornThreadServer):
        # [FUNCTION DOC] HELPER: Stores the router and server.
        self.router = router
        self.server = server

    @property
    def url(self) -> str:
        # [FUNCTION DOC] HELPER: Property: the server's base URL.
        return self.server.url


@pytest.fixture
def router_env():
    # [FUNCTION DOC] HELPER: Fixture: creates a router on a free port starting at 20000 (quiet mode),
    # starts its application on a background thread, yields a router environment handle, and stops
    # the server after the test.
    config = make_router_config(find_available_port(20000))
    router = MilesRouter(config, verbose=False)
    server = UvicornThreadServer(router.app, host=config.host, port=config.port)
    server.start()
    yield RouterEnv(router, server)
    server.stop()


@pytest.fixture
def mock_worker():
    # [FUNCTION DOC] HELPER: Fixture: starts one mock SGLang worker, yields it, stops it afterwards.
    server = create_mock_worker()
    server.start()
    yield server
    server.stop()


@pytest.fixture
def mock_worker_factory():
    # [FUNCTION DOC] HELPER: Fixture yielding a factory function that starts and returns a new mock
    # worker each call, using search start ports 30000 plus 100 per worker already created. All
    # created workers are stopped after the test.
    servers = []

    def _create():
        # [FUNCTION DOC] HELPER: Creates, starts and records one mock worker and returns it.
        start_port = 30000 + len(servers) * 100
        server = create_mock_worker(start_port)
        server.start()
        servers.append(server)
        return server

    yield _create
    for s in servers:
        s.stop()


@pytest.fixture
def router_factory():
    # [FUNCTION DOC] HELPER: Fixture yielding a factory function that builds routers (quiet mode, not
    # served over HTTP) from a config on a free port starting at 20000 with keyword overrides.
    def _create(**overrides) -> MilesRouter:
        # [FUNCTION DOC] HELPER: Builds one router with the config overrides and returns it.
        config = make_router_config(find_available_port(20000), **overrides)
        return MilesRouter(config, verbose=False)

    return _create


class TestMilesRouterInitialization:
    # [CLASS DOC] Tests of router construction.
    async def test_client_uses_configured_connection_limit_and_timeout(self):
        # [FUNCTION DOC] A router built with max connections 7 and timeout 3.5 holds an async HTTP
        # client whose connection pool maximum is 7 (read from private transport internals) and whose
        # connect, read, write and pool timeouts are all 3.5. The client is closed afterwards.
        """The router builds its HTTP client with the connection limit and timeout taken from the config."""
        config = make_router_config(20000, max_connections=7, timeout=3.5)

        router = MilesRouter(config, verbose=False)

        try:
            assert isinstance(router.client, httpx.AsyncClient)
            assert router.client._transport._pool._max_connections == 7
            assert router.client.timeout.connect == 3.5
            assert router.client.timeout.read == 3.5
            assert router.client.timeout.write == 3.5
            assert router.client.timeout.pool == 3.5
        finally:
            await router.client.aclose()


class TestWorkerManagement:
    # [CLASS DOC] Tests of the router's add, remove and list worker endpoints against a served
    # router. Workers are only registered by url, never contacted.
    def test_add_worker_via_query_param(self, router_env: RouterEnv):
        # [FUNCTION DOC] Posting a worker url as a query parameter returns status success and
        # registers it with a request count of 0.
        worker_url = "http://127.0.0.1:30001"
        r = requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url in router_env.router.worker_request_counts
        assert router_env.router.worker_request_counts[worker_url] == 0

    def test_add_worker_via_body(self, router_env: RouterEnv):
        # [FUNCTION DOC] Posting the url in a JSON body returns status success and registers it.
        worker_url = "http://127.0.0.1:30002"
        r = requests.post(f"{router_env.url}/add_worker", json={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url in router_env.router.worker_request_counts

    def test_add_worker_duplicate(self, router_env: RouterEnv):
        # [FUNCTION DOC] Registering the same url twice succeeds both times and leaves a single
        # registered worker.
        worker_url = "http://127.0.0.1:30003"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        assert len(router_env.router.worker_request_counts) == 1
        assert worker_url in router_env.router.worker_request_counts

    def test_add_worker_missing_url(self, router_env: RouterEnv):
        # [FUNCTION DOC] An empty JSON body is rejected with status 400 and a body containing an
        # error entry.
        r = requests.post(f"{router_env.url}/add_worker", json={}, timeout=5.0)
        assert r.status_code == 400
        assert "error" in r.json()

    def test_list_workers(self, router_env: RouterEnv):
        # [FUNCTION DOC] After adding two workers, the list endpoint returns exactly their urls
        # (compared as sets).
        worker_urls = ["http://127.0.0.1:30001", "http://127.0.0.1:30002"]
        for url in worker_urls:
            requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0)

        r = requests.get(f"{router_env.url}/list_workers", timeout=5.0)
        r.raise_for_status()
        assert set(r.json()["urls"]) == set(worker_urls)

    def test_remove_worker_via_query_param(self, router_env: RouterEnv):
        # [FUNCTION DOC] Removing a registered url by query parameter returns status success and
        # drops it from the request counts.
        """A disposed cell must be able to deregister, or its dead url is routed to forever."""
        worker_url = "http://127.0.0.1:30011"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        r = requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0)
        r.raise_for_status()

        assert r.json()["status"] == "success"
        assert worker_url not in router_env.router.worker_request_counts

    def test_remove_worker_via_body(self, router_env: RouterEnv):
        # [FUNCTION DOC] Removing by JSON body works like add: the url leaves the request counts.
        """Removal accepts the same body form as add_worker so callers need no special case."""
        worker_url = "http://127.0.0.1:30012"
        requests.post(f"{router_env.url}/add_worker", json={"url": worker_url}, timeout=5.0).raise_for_status()

        requests.post(f"{router_env.url}/remove_worker", json={"url": worker_url}, timeout=5.0).raise_for_status()

        assert worker_url not in router_env.router.worker_request_counts

    def test_remove_worker_clears_the_failure_and_dead_bookkeeping(self, router_env: RouterEnv):
        # [FUNCTION DOC] For a worker given a failure count of 3 and placed in the dead set, removal
        # clears both the failure count entry and the dead-set membership.
        """Leaving a removed url quarantined would keep it in the containers this fix drains."""
        worker_url = "http://127.0.0.1:30013"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        router_env.router.worker_failure_counts[worker_url] = 3
        router_env.router.dead_workers.add(worker_url)

        requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        assert worker_url not in router_env.router.worker_failure_counts
        assert worker_url not in router_env.router.dead_workers

    def test_remove_worker_is_idempotent(self, router_env: RouterEnv):
        # [FUNCTION DOC] Removing a url that was never registered returns status 200.
        """Teardown can race a health-check eviction, and a second removal must not error."""
        worker_url = "http://127.0.0.1:30014"

        r = requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0)

        assert r.status_code == 200

    def test_remove_worker_missing_url(self, router_env: RouterEnv):
        # [FUNCTION DOC] Removal with an empty JSON body returns status 400 and an error entry.
        """Without a url there is nothing to remove, and silently succeeding would hide the caller bug."""
        r = requests.post(f"{router_env.url}/remove_worker", json={}, timeout=5.0)

        assert r.status_code == 400
        assert "error" in r.json()

    def test_a_removed_worker_disappears_from_list_workers(self, router_env: RouterEnv):
        # [FUNCTION DOC] With two workers registered and one removed, the list endpoint returns only
        # the kept url.
        """list_workers is what the rollout side aborts against, so dead urls must leave it."""
        kept, removed = "http://127.0.0.1:30015", "http://127.0.0.1:30016"
        for url in (kept, removed):
            requests.post(f"{router_env.url}/add_worker", params={"url": url}, timeout=5.0).raise_for_status()

        requests.post(f"{router_env.url}/remove_worker", params={"url": removed}, timeout=5.0).raise_for_status()

        r = requests.get(f"{router_env.url}/list_workers", timeout=5.0)
        r.raise_for_status()
        assert r.json()["urls"] == [kept]


class TestLoadBalancing:
    # [CLASS DOC] Tests of the router's private url selection, called directly with hand-set
    # request counts.
    def test_use_url_selects_min_load(self, router_factory):
        # [FUNCTION DOC] With counts 5, 2 and 8, selection returns the worker with count 2 and
        # increments its count to 3.
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 5, "http://w2:8000": 2, "http://w3:8000": 8}

        selected = router._use_url()
        assert selected == "http://w2:8000"
        assert router.worker_request_counts["http://w2:8000"] == 3

    def test_use_url_excludes_dead_workers(self, router_factory):
        # [FUNCTION DOC] The least loaded worker (count 1) is in the dead set, so selection returns
        # the next least loaded (count 3) and increments it to 4.
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 5, "http://w2:8000": 1, "http://w3:8000": 3}
        router.dead_workers = {"http://w2:8000"}

        selected = router._use_url()
        assert selected == "http://w3:8000"
        assert router.worker_request_counts["http://w3:8000"] == 4

    def test_use_url_raises_when_all_dead(self, router_factory):
        # [FUNCTION DOC] With the only worker dead, selection raises a runtime error saying there
        # are no healthy workers.
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 0}
        router.dead_workers = {"http://w1:8000"}

        with pytest.raises(RuntimeError, match="No healthy workers"):
            router._use_url()


class TestFinishingARequestAfterDeregistration:
    # [CLASS DOC] Tests of the router's private request-finish bookkeeping when the worker may have
    # been deregistered or replaced in the meantime.
    def test_a_finished_request_decrements_its_workers_count(self, router_factory):
        # [FUNCTION DOC] Finishing a request for a registered worker with count 2 leaves count 1.
        """The ordinary path still hands the slot back to a registered worker."""
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 2}

        router._finish_url("http://w1:8000")

        assert router.worker_request_counts == {"http://w1:8000": 1}

    def test_a_request_that_outlives_its_workers_registration_finishes_quietly(self, router_factory):
        # [FUNCTION DOC] After selecting a worker and then deregistering it, finishing the request
        # does not raise and does not re-create the worker's entry.
        """Deregistering a worker mid-request must not turn the already proxied response into a 500."""
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 0, "http://w2:8000": 0}
        worker_url = router._use_url()
        router.worker_request_counts.pop(worker_url)

        router._finish_url(worker_url)

        assert worker_url not in router.worker_request_counts

    def test_a_stale_finish_does_not_charge_a_replacement_that_took_the_same_url(self, router_factory):
        # [FUNCTION DOC] After selecting a worker, deregistering it and re-registering the same url
        # with count 0, finishing the old request leaves the count at 0 (not negative), and the url
        # is still selectable.
        """A request from the previous registration must not drive the re-registered worker's count negative."""
        router = router_factory()
        router.worker_request_counts = {"http://w1:8000": 0}
        worker_url = router._use_url()
        router.worker_request_counts.pop(worker_url)
        router.worker_request_counts[worker_url] = 0

        router._finish_url(worker_url)

        assert router.worker_request_counts[worker_url] == 0
        assert router._use_url() == worker_url


class FakeSleepClock:
    # [CLASS DOC] HELPER: Replacement for the async sleep function that returns immediately, records
    # each requested duration, optionally calls a callback, and raises an asyncio cancellation once
    # a given number of sleeps has been made (used to stop the endless health check loop).
    def __init__(self, *, stop_after: int, on_sleep: Callable[[], None] | None = None):
        # [FUNCTION DOC] HELPER: Stores the stop count and optional callback, and an empty list of
        # sleep durations.
        self.sleeps: list[float] = []
        self._stop_after = stop_after
        self._on_sleep = on_sleep

    async def sleep(self, duration: float) -> None:
        # [FUNCTION DOC] HELPER: Records the duration, calls the callback if any, and raises an
        # asyncio cancellation when the number of recorded sleeps reaches the stop count; otherwise
        # returns without waiting.
        self.sleeps.append(duration)
        if self._on_sleep is not None:
            self._on_sleep()
        if len(self.sleeps) >= self._stop_after:
            raise asyncio.CancelledError


class ScriptedWorkerHealth:
    # [CLASS DOC] HELPER: Fake health probe that returns scripted booleans in order, whichever url
    # is probed, and records the probed urls.
    def __init__(self, results: list[bool]):
        # [FUNCTION DOC] HELPER: Stores a copy of the results queue and an empty probed-url list.
        self.checked_urls: list[str] = []
        self._results = list(results)

    async def check(self, url: str) -> tuple[str, bool]:
        # [FUNCTION DOC] HELPER: Records the url and returns the pair (url, next scripted result),
        # removing the result from the queue. Raises an index error when the script is exhausted.
        self.checked_urls.append(url)
        return url, self._results.pop(0)


class DeregisteringWorkerHealth:
    # [CLASS DOC] HELPER: Fake health probe with per-url scripted results that, for the urls in a
    # given set, simulates a worker being removed from the router while its probe is in flight.
    def __init__(self, router: MilesRouter, *, results: dict[str, bool], deregister: set[str]):
        # [FUNCTION DOC] HELPER: Stores the router, the mapping from url to result, the set of urls
        # to deregister, and an empty probed-url list.
        self.checked_urls: list[str] = []
        self._router = router
        self._results = results
        self._deregister = deregister

    async def check(self, url: str) -> tuple[str, bool]:
        # [FUNCTION DOC] HELPER: Records the url. If it is in the deregister set, removes it from
        # the router's request counts and failure counts and the dead set (tolerating absence).
        # Returns (url, scripted result for that url); a missing url raises a key error.
        self.checked_urls.append(url)
        if url in self._deregister:
            self._router.worker_request_counts.pop(url, None)
            self._router.worker_failure_counts.pop(url, None)
            self._router.dead_workers.discard(url)
        return url, self._results[url]


# TODO: extract main body inside `_health_check_loop`, then can test that function
class TestHealthCheck:
    # [CLASS DOC] Tests of the router's private health probe and background health check loop. The
    # loop tests replace the router's sleep (via the router module's asyncio) and its probe, and
    # run the loop until the fake clock cancels it.
    def test_check_worker_health_success(self, router_factory, mock_worker: MockSGLangServer):
        # [FUNCTION DOC] Probing a running mock worker returns its url and a true health flag.
        router = router_factory()
        url, healthy = asyncio.run(router._check_worker_health(mock_worker.url))
        assert url == mock_worker.url
        assert healthy is True

    def test_check_worker_health_failure(self, router_factory):
        # [FUNCTION DOC] Probing an unreachable address (port 59999) returns the url and a false
        # health flag instead of raising.
        router = router_factory()
        url, healthy = asyncio.run(router._check_worker_health("http://127.0.0.1:59999"))
        assert url == "http://127.0.0.1:59999"
        assert healthy is False

    def test_health_check_loop_waits_for_configured_interval(self, router_factory, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With interval 7.5 s, one worker and three healthy scripted probes, running
        # the loop until the fake clock cancels it after three sleeps records the durations 7.5,
        # 7.5, 7.5 (the cancellation propagates out of the loop).
        """The background health check loop waits the configured interval before every round."""
        router = router_factory(health_check_interval=7.5)
        router.worker_request_counts = {"http://w1:8000": 0}
        router._check_worker_health = ScriptedWorkerHealth([True, True, True]).check
        clock = FakeSleepClock(stop_after=3)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert clock.sleeps == [7.5, 7.5, 7.5]

    def test_only_configured_consecutive_failures_quarantine_worker(
        self, router_factory, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] With failure threshold 4 and probe results fail, fail, fail, succeed, fail,
        # fail, fail, fail, the dead set (snapshotted at each sleep) stays empty for the first eight
        # rounds and contains the worker only at the ninth sleep, showing a success resets the
        # failure streak and quarantine needs four consecutive failures. The final failure count is
        # 4.
        """A worker is quarantined only once it fails the configured number of checks in a row, and a success resets the count."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=4)
        router.worker_request_counts = {worker_url: 0}
        router._check_worker_health = ScriptedWorkerHealth(
            [False, False, False, True, False, False, False, False]
        ).check
        dead_worker_snapshots: list[set[str]] = []
        clock = FakeSleepClock(stop_after=9, on_sleep=lambda: dead_worker_snapshots.append(set(router.dead_workers)))
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert dead_worker_snapshots == [set(), set(), set(), set(), set(), set(), set(), set(), {worker_url}]
        assert router.worker_failure_counts[worker_url] == 4


class TestStaleHealthResults:
    # [CLASS DOC] Tests of how the health check loop treats probe results for workers that were
    # removed or replaced while the probe was in flight, and how registration treats quarantined
    # urls.
    def test_a_probe_result_for_a_worker_removed_meanwhile_is_dropped(
        self, router_factory, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] The probe removes the worker from the router's bookkeeping and then reports
        # a failure (threshold 1). After the loop runs, the dead set is empty and the worker has no
        # failure count entry.
        """A failure observed on a deregistered url would quarantine the replacement that later takes it."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router.worker_failure_counts = {worker_url: 0}

        async def _check_and_remove(url: str) -> tuple[str, bool]:
            # [FUNCTION DOC] HELPER: Removes the url from the router's request and failure counts and
            # reports it unhealthy (returns the url and false).
            router.worker_request_counts.pop(url, None)
            router.worker_failure_counts.pop(url, None)
            return url, False

        router._check_worker_health = _check_and_remove
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.dead_workers == set()
        assert worker_url not in router.worker_failure_counts

    def test_a_probe_result_for_the_current_registration_still_counts(
        self, router_factory, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] With threshold 1 and a failing probe for a still-registered worker, the loop
        # quarantines it: the dead set contains exactly that worker.
        """Dropping every result would leave a genuinely dead worker in the routing pool forever."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router._check_worker_health = ScriptedWorkerHealth([False]).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.dead_workers == {worker_url}

    def test_readding_a_quarantined_url_puts_it_back_in_the_pool(self, router_env: RouterEnv):
        # [FUNCTION DOC] Over HTTP: add then remove a worker, mark its url dead directly (simulating a
        # late probe), then add it again; the url is no longer in the dead set.
        """A stale probe can quarantine a url after its removal, and the replacement engine must not inherit that."""
        worker_url = "http://127.0.0.1:30017"
        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        requests.post(f"{router_env.url}/remove_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()
        router_env.router.dead_workers.add(worker_url)

        requests.post(f"{router_env.url}/add_worker", params={"url": worker_url}, timeout=5.0).raise_for_status()

        assert worker_url not in router_env.router.dead_workers

    def test_a_healthy_probe_result_for_a_removed_worker_does_not_revive_its_bookkeeping(
        self, router_factory: Callable[..., MilesRouter], monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] The probe deregisters the worker while reporting it healthy (threshold 1).
        # After the loop runs, both the request counts and the failure counts are empty, so a stale
        # success does not recreate a failure counter entry.
        """A success observed on a deregistered url must not put the url back into the failure bookkeeping either."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router.worker_failure_counts = {worker_url: 0}
        router._check_worker_health = DeregisteringWorkerHealth(
            router, results={worker_url: True}, deregister={worker_url}
        ).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.worker_request_counts == {}
        assert router.worker_failure_counts == {}

    def test_a_stale_result_does_not_abandon_the_rest_of_the_round(
        self, router_factory: Callable[..., MilesRouter], monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] Two workers both probed as failing (threshold 1); the probe deregisters the
        # first. After the loop, only the second worker is quarantined and the first has no failure
        # count entry, so a stale result skips only itself.
        """One deregistered url must only skip its own result, while the other workers of the same round are still judged."""
        removed_url, kept_url = "http://w1:8000", "http://w2:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {removed_url: 0, kept_url: 0}
        router.worker_failure_counts = {removed_url: 0, kept_url: 0}
        router._check_worker_health = DeregisteringWorkerHealth(
            router, results={removed_url: False, kept_url: False}, deregister={removed_url}
        ).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())

        assert router.dead_workers == {kept_url}
        assert removed_url not in router.worker_failure_counts

    def test_a_url_freed_during_a_probe_is_routable_again_once_its_replacement_registers(
        self, router_factory: Callable[..., MilesRouter], monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] The probe deregisters the worker while reporting it failing (threshold 1).
        # After the loop is cancelled, the worker is registered again by calling the router's
        # add-worker handler directly; the url is then selectable (selection returns it).
        """A dispose racing a probe must not stop the engine that later takes the same address from receiving traffic."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_interval=0.01, health_check_failure_threshold=1)
        router.worker_request_counts = {worker_url: 0}
        router.worker_failure_counts = {worker_url: 0}
        router._check_worker_health = DeregisteringWorkerHealth(
            router, results={worker_url: False}, deregister={worker_url}
        ).check
        clock = FakeSleepClock(stop_after=2)
        monkeypatch.setattr(router_module.asyncio, "sleep", clock.sleep)

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(router._health_check_loop())
        asyncio.run(router.add_worker(make_add_worker_request(worker_url)))

        assert router._use_url() == worker_url

    def test_a_duplicate_registration_does_not_lift_a_live_workers_quarantine(
        self, router_factory: Callable[..., MilesRouter]
    ):
        # [FUNCTION DOC] For a registered worker with count 2, failure count 3 and quarantined,
        # calling the add-worker handler again for the same url leaves it quarantined with failure
        # count 3 and request count 2 (unchanged).
        """Only a fresh registration clears quarantine, because reconnecting a still-registered dead worker needs a weight resync."""
        worker_url = "http://w1:8000"
        router = router_factory(health_check_failure_threshold=3)
        router.worker_request_counts = {worker_url: 2}
        router.worker_failure_counts = {worker_url: 3}
        router.dead_workers = {worker_url}

        asyncio.run(router.add_worker(make_add_worker_request(worker_url)))

        assert router.dead_workers == {worker_url}
        assert router.worker_failure_counts[worker_url] == 3
        assert router.worker_request_counts[worker_url] == 2


class TestProxyIntegration:
    # [CLASS DOC] End-to-end tests of proxying through a served router to mock SGLang workers.
    def test_proxy_forwards_request(self, router_env: RouterEnv, mock_worker: MockSGLangServer):
        # [FUNCTION DOC] With one registered mock worker, a generate request with input ids and
        # logprob flag succeeds, the response contains a text field, and the worker's request log
        # holds exactly one entry equal to the sent payload (forwarded unchanged).
        requests.post(f"{router_env.url}/add_worker", params={"url": mock_worker.url}, timeout=5.0).raise_for_status()

        payload = {"input_ids": [1, 2, 3], "return_logprob": True}
        r = requests.post(f"{router_env.url}/generate", json=payload, timeout=10.0)
        r.raise_for_status()

        assert "text" in r.json()
        assert len(mock_worker.request_log) == 1
        assert mock_worker.request_log[0] == payload

    def test_proxy_multi_worker(self, router_env: RouterEnv, mock_worker_factory):
        # [FUNCTION DOC] With two registered mock workers, four generate requests all succeed, and
        # across both workers' request logs exactly four requests were received, each equal to the
        # payload. The split between workers is not asserted.
        worker1, worker2 = mock_worker_factory(), mock_worker_factory()
        requests.post(f"{router_env.url}/add_worker", params={"url": worker1.url}, timeout=5.0)
        requests.post(f"{router_env.url}/add_worker", params={"url": worker2.url}, timeout=5.0)

        payload = {"input_ids": [1, 2, 3], "return_logprob": True}
        for _ in range(4):
            requests.post(f"{router_env.url}/generate", json=payload, timeout=10.0).raise_for_status()

        all_requests = worker1.request_log + worker2.request_log
        assert len(all_requests) == 4
        assert all(req == payload for req in all_requests)

    def test_proxy_health_endpoint(self, router_env: RouterEnv, mock_worker: MockSGLangServer):
        # [FUNCTION DOC] With a mock worker registered, the router's health endpoint succeeds and
        # reports status "ok".
        requests.post(f"{router_env.url}/add_worker", params={"url": mock_worker.url}, timeout=5.0)

        r = requests.get(f"{router_env.url}/health", timeout=5.0)
        r.raise_for_status()
        assert r.json()["status"] == "ok"
