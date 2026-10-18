# [MODULE DOC]
# Tests for the SGLang router API client (worker registration and removal) and its legacy-API
# decision. The shared HTTP client provider is patched with a recorder (as in the engine API
# client tests) and the installed router package's version attribute is patched to simulate
# router versions. Covers: legacy query-string endpoints (router 0.2.1 and below, or the miles
# router), JSON-body registration on the modern API (with the bootstrap port only for prefill
# workers), removal by percent-encoded URL for versions from 0.2.2 up to but excluding 0.3.0 and
# by looked-up worker id from 0.3.0, tolerance of unknown workers and lookup/transport failures
# (warning logs) while status errors from the actual delete or registration propagate, rejection
# of prefill/decode workers on the legacy API, and a total (not connect-only) timeout on every
# request. Skipped when the router package is not installed.
from argparse import Namespace

import httpx
import pytest

sglang_router = pytest.importorskip("sglang_router")

from miles.backends.sglang_utils.sglang_router_api_client import (  # noqa: E402
    ROUTER_REQUEST_TIMEOUT,
    SGLangRouterApiClient,
)
from miles.utils.http_utils import GeneralHttpClientProvider  # noqa: E402

ROUTER_URL = "http://router-host:9000"
WORKER_URL = "http://fake-host:1234"


class _FakeResponse:
    # [CLASS DOC] HELPER: Fake HTTP response with a JSON payload (default a mapping with ok true)
    # and a status code (default 200).
    def __init__(self, payload: dict | None = None, status_code: int = 200):
        # [FUNCTION DOC] HELPER: Stores the payload (default ok true) and the status code.
        self._payload = payload if payload is not None else {"ok": True}
        self.status_code = status_code

    def raise_for_status(self):
        # [FUNCTION DOC] HELPER: Raises an HTTP status error (without request or response objects)
        # for status codes 400 and above; otherwise does nothing.
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=None, response=None)

    def json(self):
        # [FUNCTION DOC] HELPER: Returns the payload.
        return self._payload


class _Recorder:
    # [CLASS DOC] HELPER: Fake async HTTP client recording (verb, url, keyword arguments) per call
    # and replaying queued responses, defaulting to an OK response.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with empty call and response lists.
        self.calls: list[tuple[str, str, dict]] = []
        self.responses: list[_FakeResponse] = []

    def install(self, monkeypatch, responses: list[_FakeResponse] | None = None):
        # [FUNCTION DOC] HELPER: Stores a copy of the queued responses and patches (via
        # monkeypatch) the shared client provider to return this recorder.
        self.responses = list(responses or [])
        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: self)

    async def get(self, url, **kwargs):
        # [FUNCTION DOC] HELPER: Records a "get" call and returns the next response.
        return self._record("get", url, kwargs)

    async def post(self, url, **kwargs):
        # [FUNCTION DOC] HELPER: Records a "post" call and returns the next response.
        return self._record("post", url, kwargs)

    async def delete(self, url, **kwargs):
        # [FUNCTION DOC] HELPER: Records a "delete" call and returns the next response.
        return self._record("delete", url, kwargs)

    def _record(self, verb: str, url: str, kwargs: dict) -> _FakeResponse:
        # [FUNCTION DOC] HELPER: Appends the call record; returns the first queued response if any
        # (removing it) else a default OK response.
        self.calls.append((verb, url, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse()


@pytest.fixture
def recorder(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Creates a recorder with no queued responses, installs it as
    # the shared client, and returns it.
    rec = _Recorder()
    rec.install(monkeypatch)
    return rec


@pytest.fixture
def client():
    # [FUNCTION DOC] HELPER: Fixture. Returns a router API client for the fake router URL.
    return SGLangRouterApiClient(router_url=ROUTER_URL)


async def test_add_worker_uses_the_query_string_api_when_legacy(client, recorder):
    # [FUNCTION DOC] Registering a regular worker with the legacy flag makes exactly one POST to
    # the router's /add_worker with the worker URL as query parameter and the router timeout.
    """Routers <= 0.2.1 and the miles router only understand /add_worker?url=."""
    await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True)

    assert recorder.calls == [
        ("post", f"{ROUTER_URL}/add_worker?url={WORKER_URL}", {"timeout": ROUTER_REQUEST_TIMEOUT})
    ]


async def test_add_worker_rejects_pd_disaggregation_on_the_legacy_api(client, recorder):
    # [FUNCTION DOC] Registering a prefill worker on the legacy API raises an assertion error
    # "pd disaggregation is not supported".
    """The legacy API has no worker_type concept, so prefill/decode workers must be refused."""
    with pytest.raises(AssertionError, match="pd disaggregation is not supported"):
        await client.add_worker(worker_url=WORKER_URL, worker_type="prefill", use_legacy_api=True)


async def test_add_worker_posts_the_worker_payload_on_the_modern_api(client, recorder):
    # [FUNCTION DOC] Registering a regular worker on the modern API makes one POST to /workers
    # with JSON {url, worker_type "regular"}.
    """Modern routers take a JSON body on /workers."""
    await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=False)

    assert len(recorder.calls) == 1
    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{ROUTER_URL}/workers")
    assert kwargs["json"] == {"url": WORKER_URL, "worker_type": "regular"}


async def test_add_worker_includes_the_bootstrap_port_for_prefill_workers(client, recorder):
    # [FUNCTION DOC] A prefill worker with bootstrap port 8998 on the modern API: one POST whose
    # JSON has the URL, worker type prefill and the bootstrap port.
    """PD disaggregation needs the prefill worker's bootstrap port registered with the router."""
    await client.add_worker(worker_url=WORKER_URL, worker_type="prefill", use_legacy_api=False, bootstrap_port=8998)

    assert len(recorder.calls) == 1
    assert recorder.calls[0][2]["json"] == {
        "url": WORKER_URL,
        "worker_type": "prefill",
        "bootstrap_port": 8998,
    }


async def test_remove_worker_uses_the_query_string_api_when_legacy(client, recorder):
    # [FUNCTION DOC] Removing with the legacy flag makes exactly one POST to /remove_worker with
    # the worker URL as query parameter and the router timeout.
    """Legacy routers unregister via /remove_worker?url=."""
    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=True)

    assert recorder.calls == [
        ("post", f"{ROUTER_URL}/remove_worker?url={WORKER_URL}", {"timeout": ROUTER_REQUEST_TIMEOUT})
    ]


async def test_remove_worker_deletes_by_url_on_pre_0_3_routers(client, recorder, monkeypatch):
    # [FUNCTION DOC] With router version patched to 0.2.5 and the modern flag, removal makes one
    # DELETE to /workers/ plus the percent-encoded worker URL, with the router timeout.
    """Routers in [0.2.2, 0.3.0) address workers by percent-encoded url."""
    monkeypatch.setattr(sglang_router, "__version__", "0.2.5")

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert recorder.calls == [
        ("delete", f"{ROUTER_URL}/workers/http%3A%2F%2Ffake-host%3A1234", {"timeout": ROUTER_REQUEST_TIMEOUT})
    ]


async def test_remove_worker_resolves_the_worker_id_on_modern_routers(client, monkeypatch):
    # [FUNCTION DOC] Router version 0.3.1; the list response holds the worker with id "w-7".
    # Removal makes a GET of /workers then a DELETE of /workers/w-7.
    """Routers >= 0.3.0 address workers by id, so the url must be resolved first."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-7"}]})])

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert [(verb, url) for verb, url, _kwargs in rec.calls] == [
        ("get", f"{ROUTER_URL}/workers"),
        ("delete", f"{ROUTER_URL}/workers/w-7"),
    ]


async def test_remove_worker_tolerates_an_unknown_worker(client, monkeypatch):
    # [FUNCTION DOC] Router 0.3.1 whose worker list is empty: removal completes with only the GET
    # (no DELETE) and does not raise.
    """Shutdown must not fail when the router no longer knows the worker."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": []})])

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert [verb for verb, _url, _kwargs in rec.calls] == ["get"]


async def test_add_worker_propagates_router_errors(client, monkeypatch):
    # [FUNCTION DOC] A 500 response on the legacy registration raises an HTTP status error.
    """A router that rejects the registration must surface, not be swallowed."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=True)


async def test_remove_worker_propagates_router_errors(client, monkeypatch):
    # [FUNCTION DOC] A 500 response on the legacy removal raises an HTTP status error.
    """Unregistration errors must surface on the legacy path too."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=True)


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
async def test_add_worker_rejects_pd_worker_types_before_any_request(client, monkeypatch, worker_type):
    # [FUNCTION DOC] Parametrized over prefill and decode. On the legacy API the registration
    # raises an assertion error "pd disaggregation is not supported" and no request is recorded.
    """The legacy API has no worker_type, so a pd worker must be refused without registering it."""
    rec = _Recorder()
    rec.install(monkeypatch)

    with pytest.raises(AssertionError, match="pd disaggregation is not supported"):
        await client.add_worker(worker_url=WORKER_URL, worker_type=worker_type, use_legacy_api=True)

    assert rec.calls == []


@pytest.mark.parametrize("worker_type", ["regular", "decode"])
async def test_add_worker_omits_bootstrap_port_for_non_prefill_workers(client, recorder, worker_type):
    # [FUNCTION DOC] Parametrized over regular and decode. On the modern API with a bootstrap
    # port supplied, one POST is made whose JSON has only the URL and worker type (no port).
    """Only a prefill worker exposes a bootstrap port for the decode side to dial."""
    await client.add_worker(worker_url=WORKER_URL, worker_type=worker_type, use_legacy_api=False, bootstrap_port=8998)

    assert len(recorder.calls) == 1
    assert recorder.calls[0][2]["json"] == {"url": WORKER_URL, "worker_type": worker_type}


@pytest.mark.parametrize(
    "version, expected_verb", [("0.2.2", "delete"), ("0.2.9", "delete"), ("0.3.0", "get"), ("0.3.1", "get")]
)
async def test_remove_worker_switches_api_exactly_at_0_3_0(client, monkeypatch, version, expected_verb):
    # [FUNCTION DOC] Parametrized over versions 0.2.2, 0.2.9 (expect the first request to be a
    # DELETE by URL) and 0.3.0, 0.3.1 (expect a GET to look up the worker id). The list response
    # holds one worker with id "w-1".
    """0.3.0 is where workers stop being addressable by url and must be looked up by id."""
    monkeypatch.setattr(sglang_router, "__version__", version)
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-1"}]})])

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert rec.calls[0][0] == expected_verb


async def test_add_worker_propagates_a_failing_modern_response(client, monkeypatch):
    # [FUNCTION DOC] A 500 response on the modern registration raises an HTTP status error.
    """A router rejecting the registration must surface, not be swallowed."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=False)


async def test_remove_worker_propagates_a_failing_url_addressed_delete(client, monkeypatch):
    # [FUNCTION DOC] Router 0.2.5 and a 500 response on the URL-addressed DELETE: raises an HTTP
    # status error.
    """Unregistration errors surface on the url-addressed path too."""
    monkeypatch.setattr(sglang_router, "__version__", "0.2.5")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(status_code=500)])

    with pytest.raises(httpx.HTTPStatusError):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)


async def test_remove_worker_propagates_a_failing_id_addressed_delete(client, monkeypatch):
    # [FUNCTION DOC] Router 0.3.1: the lookup succeeds (worker id w-1), the DELETE returns 500:
    # raises an HTTP status error (the delete is outside the lookup's tolerant error handling).
    """The id-addressed delete is outside the lookup's broad except, so its status must surface."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(
        monkeypatch,
        responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-1"}]}), _FakeResponse(status_code=500)],
    )

    with pytest.raises(httpx.HTTPStatusError):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)


async def test_remove_worker_warns_about_an_unknown_worker(client, monkeypatch, caplog):
    # [FUNCTION DOC] Router 0.3.1 with an empty worker list and warning-level log capture: removal
    # logs a message containing "not found in router".
    """Shutdown tolerates a worker the router forgot, but says so."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": []})])

    with caplog.at_level("WARNING"):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert "not found in router" in caplog.text


async def test_remove_worker_tolerates_a_broken_worker_lookup(client, monkeypatch, caplog):
    # [FUNCTION DOC] Router 0.3.1 whose list GET raises a connection error: removal does not raise
    # and logs "Failed to fetch workers list".
    """A router that cannot list its workers must not block the engine's shutdown."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")

    class _Refusing:
        # [CLASS DOC] HELPER: Fake client whose GET always raises a connection error.
        async def get(self, url, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises a connection error "router down".
            raise httpx.ConnectError("router down")

    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Refusing())

    with caplog.at_level("WARNING"):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert "Failed to fetch workers list" in caplog.text


async def test_remove_worker_tolerates_a_transport_error_during_id_delete(client, monkeypatch, caplog):
    # [FUNCTION DOC] Router 0.3.1 whose list answers with worker id w-1 but whose DELETE raises a
    # connection error: removal does not raise; the attempted delete URL list is exactly
    # /workers/w-1, and a warning containing "Failed to fetch workers list or remove worker" is logged.
    """A router that answers the lookup but drops the delete connection must not block the engine's shutdown."""
    monkeypatch.setattr(sglang_router, "__version__", "0.3.1")

    class _FailingDelete:
        # [CLASS DOC] HELPER: Fake client whose GET returns a one-worker list and whose DELETE
        # records the URL then raises a connection error.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Starts an empty list of attempted delete URLs.
            self.deleted_urls: list[str] = []

        async def get(self, url, **kwargs):
            # [FUNCTION DOC] HELPER: Returns a fake response listing the worker with id "w-1".
            return _FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-1"}]})

        async def delete(self, url, **kwargs):
            # [FUNCTION DOC] HELPER: Records the URL, then raises a connection error "router down".
            self.deleted_urls.append(url)
            raise httpx.ConnectError("router down")

    fake = _FailingDelete()
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: fake)

    with caplog.at_level("WARNING"):
        await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert fake.deleted_urls == [f"{ROUTER_URL}/workers/w-1"]
    assert "Failed to fetch workers list or remove worker" in caplog.text


def test_the_router_timeout_bounds_the_whole_request_not_just_the_connect():
    # [FUNCTION DOC] Asserts the module's router request timeout has a connect timeout of 10.0
    # seconds and non-none read, write and pool timeouts.
    """The shared client reads without a deadline, so control-plane RPCs must carry their own total timeout."""
    assert ROUTER_REQUEST_TIMEOUT.connect == 10.0
    assert ROUTER_REQUEST_TIMEOUT.read is not None
    assert ROUTER_REQUEST_TIMEOUT.write is not None
    assert ROUTER_REQUEST_TIMEOUT.pool is not None


@pytest.mark.parametrize("use_legacy_api", [True, False])
async def test_add_worker_bounds_the_registration_request(client, recorder, use_legacy_api):
    # [FUNCTION DOC] Parametrized over the legacy flag. Registration makes one request carrying
    # the router timeout keyword.
    """A router that accepts the connection but never answers would wedge the weight-update window forever."""
    await client.add_worker(worker_url=WORKER_URL, worker_type="regular", use_legacy_api=use_legacy_api)

    assert [kwargs.get("timeout") for _verb, _url, kwargs in recorder.calls] == [ROUTER_REQUEST_TIMEOUT]


@pytest.mark.parametrize("version", ["0.2.5", "0.3.1"])
async def test_remove_worker_bounds_every_request_it_makes(client, monkeypatch, version):
    # [FUNCTION DOC] Parametrized over router versions 0.2.5 and 0.3.1 with a one-worker list
    # response. Removal makes at least one request and every request carries the router timeout.
    """Both the worker lookup and the delete must be bounded, or teardown hangs on an unanswered router."""
    monkeypatch.setattr(sglang_router, "__version__", version)
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse({"workers": [{"url": WORKER_URL, "id": "w-1"}]})])

    await client.remove_worker(worker_url=WORKER_URL, use_legacy_api=False)

    assert rec.calls
    assert all(kwargs.get("timeout") == ROUTER_REQUEST_TIMEOUT for _verb, _url, kwargs in rec.calls)


@pytest.mark.parametrize(
    "version, use_miles_router, expected",
    [
        ("0.2.1", False, True),
        ("0.2.2", False, False),
        ("0.3.1", False, False),
        ("0.3.1", True, True),
    ],
)
def test_legacy_router_api_decision(version, use_miles_router, expected, monkeypatch):
    # [FUNCTION DOC] Parametrized over (router version, use miles router, expected): 0.2.1 plain ->
    # legacy; 0.2.2 and 0.3.1 plain -> not legacy; 0.3.1 with the miles router -> legacy. Patches
    # the router package's version and asserts the legacy-API decision function (given an
    # arguments namespace with the miles-router flag) returns exactly the expected boolean.
    """0.2.1 is the last version with the query-string API; --use-miles-router pins it too."""
    from miles.backends.sglang_utils.sglang_router_api_client import use_legacy_router_api

    monkeypatch.setattr(sglang_router, "__version__", version)

    assert use_legacy_router_api(Namespace(use_miles_router=use_miles_router)) is expected
