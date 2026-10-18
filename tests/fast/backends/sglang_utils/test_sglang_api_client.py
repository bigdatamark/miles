# [MODULE DOC]
# Tests for the async SGLang HTTP API client (and its module-level health helpers). The shared
# HTTP client provider is patched with a recorder that stores every call (verb, URL, keyword
# arguments) and replays queued fake responses, so tests assert the wire contract: endpoint
# paths, JSON payload shapes (including omitted-when-absent optional fields and renamed keys),
# query parameters, timeouts, bearer-token headers, retry/back-off loops (with sleep patched),
# error propagation, swallowed errors, and that every public method is a coroutine function.
# Test functions are plain async functions (no marker), relying on the project's async test mode.
import asyncio
import inspect
import json

import httpx
import pytest

from miles.backends.sglang_utils import sglang_api_client
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.utils.http_utils import GeneralHttpClientProvider

SERVER_URL = "http://fake-host:1234"


class _FakeResponse:
    # [CLASS DOC] HELPER: Fake HTTP response with a status code, JSON payload (default a mapping
    # with ok true), text, and raw content bytes (default: the payload encoded as JSON text).
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = "", body: bytes | None = None):
        # [FUNCTION DOC] HELPER: Stores the status code, payload (default ok true), text; the
        # content is the JSON encoding of the payload unless explicit body bytes are given (an
        # empty body means no content).
        self.status_code = status_code
        self._payload = payload if payload is not None else {"ok": True}
        self.text = text
        self.content = json.dumps(self._payload).encode() if body is None else body

    def raise_for_status(self):
        # [FUNCTION DOC] HELPER: Raises an HTTP status error (without request or response objects)
        # when the status code is 400 or higher; otherwise does nothing.
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"status {self.status_code}", request=None, response=None)

    def json(self):
        # [FUNCTION DOC] HELPER: Returns the payload; raises a value error when the content is
        # empty (mimicking JSON decoding failure on an empty body).
        if not self.content:
            raise ValueError("a response with no body carries no json")
        return self._payload


class _Recorder:
    # [CLASS DOC] HELPER: Fake async HTTP client that records every request as (verb, url, keyword
    # arguments) and returns queued fake responses in order, falling back to a default OK response.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with empty call and response lists.
        self.calls: list[tuple[str, str, dict]] = []
        self.responses: list[_FakeResponse] = []

    def install(self, monkeypatch, responses: list[_FakeResponse] | None = None):
        # [FUNCTION DOC] HELPER: Stores a copy of the queued responses (default none) and patches
        # (via monkeypatch) the shared client provider so it returns this recorder.
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
        # [FUNCTION DOC] HELPER: Appends (verb, url, keyword arguments) to the call list; pops and
        # returns the first queued response if any, else a default OK fake response.
        self.calls.append((verb, url, kwargs))
        if self.responses:
            return self.responses.pop(0)
        return _FakeResponse()


@pytest.fixture
def recorder(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Creates a recorder with no queued responses, installs it as
    # the shared client (patched), and returns it.
    rec = _Recorder()
    rec.install(monkeypatch)
    return rec


@pytest.fixture
def client():
    # [FUNCTION DOC] HELPER: Fixture. Returns an API client for the fake server URL with no API key.
    return SGLangApiClient(server_url=SERVER_URL)


async def test_post_methods_hit_the_server_url_with_expected_payload(client, recorder):
    # [FUNCTION DOC] Calls the tensor weight update (one tensor, "direct" load format, flush on),
    # the weight-version update, and begin/end weight update. Asserts four POSTs to the matching
    # endpoints in order; the first payload has the tensors, load format, flush flag and selector
    # "all"; the second has the new version and abort-all-requests false.
    """Every POST-based method targets ``<server_url>/<endpoint>`` and sends the documented payload."""
    await client.update_weights_from_tensor(serialized_named_tensors=["a"], load_format="direct", flush_cache=True)
    await client.update_weight_version("run-0001")
    await client.begin_weight_update()
    await client.end_weight_update()

    assert [(verb, url) for verb, url, _ in recorder.calls] == [
        ("post", f"{SERVER_URL}/update_weights_from_tensor"),
        ("post", f"{SERVER_URL}/update_weight_version"),
        ("post", f"{SERVER_URL}/begin_weight_update"),
        ("post", f"{SERVER_URL}/end_weight_update"),
    ]
    assert recorder.calls[0][2]["json"] == {
        "serialized_named_tensors": ["a"],
        "load_format": "direct",
        "flush_cache": True,
        "selector": "all",
    }
    assert recorder.calls[1][2]["json"] == {"new_version": "run-0001", "abort_all_requests": False}


async def test_update_weights_from_tensor_omits_weight_version_when_not_given(client, recorder):
    # [FUNCTION DOC] A tensor update without a weight version sends a payload with no weight
    # version key.
    """``weight_version`` stays out of the payload unless the caller passes one."""
    await client.update_weights_from_tensor(serialized_named_tensors=["a"])

    assert "weight_version" not in recorder.calls[0][2]["json"]


async def test_check_weights_renames_skip_list_to_skip_tensor_list(client, recorder):
    # [FUNCTION DOC] Weight check with action reset_tensors and a skip list of one name: asserts
    # one POST to the weights-checker endpoint whose payload has the action, allow-quant-error
    # false, selector "all" and the skip list under the server's key name skip_tensor_list.
    """sglang's CheckWeightsReqInput expects ``skip_tensor_list``, not ``skip_list``."""
    await client.check_weights(action="reset_tensors", skip_list=["lm_head"])

    verb, url, kwargs = recorder.calls[0]
    assert (verb, url) == ("post", f"{SERVER_URL}/weights_checker")
    assert kwargs["json"] == {
        "action": "reset_tensors",
        "allow_quant_error": False,
        "selector": "all",
        "skip_tensor_list": ["lm_head"],
    }


async def test_pull_weights_forwards_the_explicit_checkpoint_dirs(client, recorder):
    # [FUNCTION DOC] Pull-weights with target version 7, local dir and source dir: the payload
    # contains exactly those three values under their names.
    """The client holds no args, so both checkpoint dirs arrive as explicit parameters."""
    await client.pull_weights(target_version=7, local_checkpoint_dir="/local", source_dir="/shared")

    assert recorder.calls[0][2]["json"] == {
        "local_checkpoint_dir": "/local",
        "source_dir": "/shared",
        "target_version": 7,
    }


async def test_get_methods_hit_the_documented_endpoints(client, recorder):
    # [FUNCTION DOC] Calls health-generate, get-server-info and get-parallelism-info (rank 3).
    # Asserts three GETs to /health_generate, /server_info and /parallelism_config in order, and
    # that the last call carries query params with rank 3.
    """GET-based methods keep their (non-uniform) endpoint names."""
    await client.health_generate()
    await client.get_server_info()
    await client.get_parallelism_info(rank=3)

    assert [(verb, url) for verb, url, _ in recorder.calls] == [
        ("get", f"{SERVER_URL}/health_generate"),
        ("get", f"{SERVER_URL}/server_info"),
        ("get", f"{SERVER_URL}/parallelism_config"),
    ]
    assert recorder.calls[2][2]["params"] == {"rank": 3}


async def test_get_weight_version_falls_back_to_the_legacy_endpoint(client, monkeypatch):
    # [FUNCTION DOC] First response (model-info) has status 404, second (legacy endpoint) has
    # weight version "v3". Asserts the result is "v3" and the requested URLs are /model_info then
    # /get_weight_version.
    """Old sglang builds only serve /get_weight_version, so a non-200 /model_info must fall through."""
    rec = _Recorder()
    rec.install(
        monkeypatch, responses=[_FakeResponse(status_code=404), _FakeResponse(payload={"weight_version": "v3"})]
    )

    assert await client.get_weight_version() == "v3"
    assert [url for _verb, url, _kwargs in rec.calls] == [
        f"{SERVER_URL}/model_info",
        f"{SERVER_URL}/get_weight_version",
    ]


async def test_release_memory_occupation_flushes_the_cache_first(client, recorder):
    # [FUNCTION DOC] Releasing memory for the weights tag: asserts a GET to /flush_cache is made
    # first and then a POST to /release_memory_occupation.
    """Offload is only safe once the working queue is drained."""
    await client.release_memory_occupation(tags=["weights"])

    assert [(verb, url) for verb, url, _ in recorder.calls] == [
        ("get", f"{SERVER_URL}/flush_cache"),
        ("post", f"{SERVER_URL}/release_memory_occupation"),
    ]


async def test_destroy_weights_update_group_swallows_request_errors(client, monkeypatch):
    # [FUNCTION DOC] The shared client's POST raises a connection error; destroying group
    # "group-0" returns none instead of raising.
    """A freshly created engine has no group yet; failing to destroy it must not propagate."""

    class _Raising:
        # [CLASS DOC] HELPER: Fake client whose POST always raises a connection error.
        async def post(self, url, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises a connection error "no such group".
            raise httpx.ConnectError("no such group")

    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Raising())

    assert await client.destroy_weights_update_group("group-0") is None


async def test_flush_cache_sleeps_between_pending_request_retries(client, monkeypatch):
    # [FUNCTION DOC] Replaces the async sleep with a recorder and queues 60 responses with status
    # 400. Flushing the cache must raise a timeout error "Timeout while flushing cache" and the
    # recorded sleeps must number exactly 60 (a back-off after every attempt, including 400s).
    """Regression test for the fully_async weight-update crash: sglang
    returns 400 (not an exception) while requests are still pending, so the
    retry loop must back off on THAT path too, or all 60 "attempts" burn
    through in a fraction of a second -- nowhere near enough time for
    in-flight generation to drain -- and flush_cache raises TimeoutError
    almost immediately after pause_generation instead of after ~60s."""
    sleep_calls = []

    async def fake_sleep(seconds):
        # [FUNCTION DOC] HELPER: Records the requested sleep duration without sleeping.
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=400) for _ in range(60)])

    with pytest.raises(TimeoutError, match="Timeout while flushing cache"):
        await client.flush_cache()

    assert len(sleep_calls) == 60, (
        f"expected the loop to back off on every one of its 60 attempts, got {len(sleep_calls)} sleeps "
        "-- a 400 response (pending requests) must not skip the retry delay"
    )


async def test_flush_cache_retries_a_refused_connection(client, monkeypatch):
    # [FUNCTION DOC] The first GET raises a connection error, the second succeeds (sleep patched
    # to a no-op). Flushing the cache returns normally after exactly 2 attempts.
    """A server that is briefly unreachable mid-offload must not fail the whole weight update."""

    class _RefusingThenServing:
        # [CLASS DOC] HELPER: Fake client whose first GET raises a connection error and later
        # GETs return an OK response; counts attempts.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Starts the attempt counter at 0.
            self.attempts = 0

        async def get(self, url, **kwargs):
            # [FUNCTION DOC] HELPER: Counts the attempt; raises a connection error on the first
            # attempt, otherwise returns an OK fake response.
            self.attempts += 1
            if self.attempts == 1:
                raise httpx.ConnectError("connection refused")
            return _FakeResponse()

    serving = _RefusingThenServing()
    monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: serving)
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

    await client.flush_cache()

    assert serving.attempts == 2


# Table of (callable taking the client, endpoint name, expected JSON payload) for the POST
# methods not covered by dedicated tests: LoRA adapter loads (tensors and distributed, with
# "torch." stripped from dtype names), adapter registration, memory resume, weights from disk,
# weights-update-group init and destroy, distributed weight update, pause/continue generation,
# and profile start/stop.
_REMAINING_POST_CASES = [
    (
        lambda c: c.load_lora_adapter_from_tensors(
            lora_name="l", config_dict={"r": 8}, serialized_named_tensors=["t"]
        ),
        "load_lora_adapter_from_tensors",
        {"lora_name": "l", "config_dict": {"r": 8}, "serialized_named_tensors": ["t"], "pinned": False},
    ),
    (
        lambda c: c.load_lora_adapter_from_distributed(
            lora_name="l", config_dict={"r": 8}, names=["w"], dtypes=["torch.bfloat16"], shapes=[[1]], group_name="g"
        ),
        "load_lora_adapter_from_distributed",
        {
            "lora_name": "l",
            "config_dict": {"r": 8},
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "pinned": False,
            "upsert": False,
        },
    ),
    (
        lambda c: c.register_lora_adapter("l", {"r": 8}),
        "register_lora_adapter",
        {"lora_name": "l", "config_dict": {"r": 8}, "pinned": False},
    ),
    (
        lambda c: c.resume_memory_occupation(tags=["weights"]),
        "resume_memory_occupation",
        {"tags": ["weights"]},
    ),
    (
        lambda c: c.update_weights_from_disk(model_path="/ckpt", load_format="direct", weight_version="v1"),
        "update_weights_from_disk",
        {"model_path": "/ckpt", "load_format": "direct", "weight_version": "v1"},
    ),
    (
        lambda c: c.init_weights_update_group("addr", 1234, 8, 9, "g", "nccl"),
        "init_weights_update_group",
        {
            "master_address": "addr",
            "master_port": 1234,
            "rank_offset": 8,
            "world_size": 9,
            "group_name": "g",
            "backend": "nccl",
        },
    ),
    (lambda c: c.destroy_weights_update_group("g"), "destroy_weights_update_group", {"group_name": "g"}),
    (
        lambda c: c.update_weights_from_distributed(
            names=["w"], dtypes=["torch.bfloat16"], shapes=[[1]], group_name="g", flush_cache=True
        ),
        "update_weights_from_distributed",
        {
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "flush_cache": True,
            "selector": "all",
        },
    ),
    (lambda c: c.pause_generation(mode="abort"), "pause_generation", {"mode": "abort"}),
    (lambda c: c.continue_generation(), "continue_generation", {}),
    (
        lambda c: c.start_profile(output_dir="/out", num_steps=3),
        "start_profile",
        {
            "output_dir": "/out",
            "start_step": None,
            "num_steps": 3,
            "activities": None,
            "profile_by_stage": False,
            "with_stack": None,
            "record_shapes": None,
        },
    ),
    (lambda c: c.stop_profile(), "stop_profile", {}),
]


@pytest.mark.parametrize("call, endpoint, expected_payload", _REMAINING_POST_CASES)
async def test_every_remaining_post_method_wire_contract(client, recorder, call, endpoint, expected_payload):
    # [FUNCTION DOC] Parametrized over the table above. Awaits the call and asserts exactly one
    # request was made: a POST to the server URL plus the endpoint, with the expected JSON payload.
    """Each remaining POST method posts its documented payload to its own endpoint."""
    await call(client)

    verb, url, kwargs = recorder.calls[0]
    assert (verb, url, kwargs["json"]) == ("post", f"{SERVER_URL}/{endpoint}", expected_payload)
    assert len(recorder.calls) == 1


async def test_get_remote_instance_transfer_engine_info_unwraps_the_response(client, monkeypatch):
    # [FUNCTION DOC] With a response holding the transfer-engine info field {a: 1}, the getter
    # for rank 2 returns the inner mapping {a: 1}, and the request carried params with rank 2.
    """The method returns the inner field, not the whole JSON body."""
    rec = _Recorder()
    rec.install(monkeypatch, responses=[_FakeResponse(payload={"remote_instance_transfer_engine_info": {"a": 1}})])

    assert await client.get_remote_instance_transfer_engine_info(rank=2) == {"a": 1}
    assert rec.calls[0][2]["params"] == {"rank": 2}


async def test_every_public_method_is_a_coroutine_function():
    # [FUNCTION DOC] Collects the non-dunder callable attributes of the client class and asserts
    # every one is a coroutine function (the list of non-async names is empty).
    # KNOWN ISSUE: private underscore-prefixed callables (such as the shared request helper) are
    # included in the check too, so any private synchronous helper would fail this test even
    # though only public methods are meant to be covered.
    """The client is async-only: a caller that forgets to await must never silently fire nothing."""
    methods = [
        name for name in dir(SGLangApiClient) if not name.startswith("__") and callable(getattr(SGLangApiClient, name))
    ]
    non_async = [name for name in methods if not inspect.iscoroutinefunction(getattr(SGLangApiClient, name))]

    assert non_async == []


class TestProbeServerHealthy:
    # [CLASS DOC] Tests for the module-level single-shot health probe function.
    """``probe_server_healthy`` is a single bounded shot, because it runs inside the locked tick sweep."""

    async def test_a_healthy_server_probes_true(self, monkeypatch):
        # [FUNCTION DOC] One OK response: the probe (with API key "k") returns true after a single
        # GET to /health_generate.
        """A ready engine is what moves the cell out of the initializing state."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is True
        assert [url for _verb, url, _kwargs in rec.calls] == [f"{SERVER_URL}/health_generate"]

    async def test_a_server_that_is_still_loading_probes_false(self, monkeypatch):
        # [FUNCTION DOC] A 503 response: the probe returns false (no exception).
        """A non-200 means not ready yet, not a failure to report upwards."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(status_code=503)])

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is False

    async def test_a_server_that_is_not_listening_probes_false(self, monkeypatch):
        # [FUNCTION DOC] The shared client's GET raises a connection error: the probe returns false.
        """The port only opens minutes after launch, so a refused connection is the normal case."""

        class _Refusing:
            # [CLASS DOC] HELPER: Fake client whose GET always raises a connection error.
            async def get(self, url, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises a connection error "connection refused".
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Refusing())

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is False

    async def test_the_probe_is_bounded_so_a_wedged_engine_cannot_stall_the_sweep(self, monkeypatch):
        # [FUNCTION DOC] After a probe with no explicit timeout, the recorded GET carries a timeout
        # keyword of 5.0 seconds (the default bound).
        """The shared http client has no read timeout, so this call must carry its own."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k")

        assert rec.calls[0][2]["timeout"] == 5.0

    async def test_a_socket_level_os_error_probes_false(self, monkeypatch):
        # [FUNCTION DOC] The shared client's GET raises a plain operating-system error (name
        # resolution failed): the probe returns false.
        """A bare ``OSError`` (DNS/socket failure outside httpx's hierarchy) is an unhealthy probe, not a crash."""

        class _Failing:
            # [CLASS DOC] HELPER: Fake client whose GET always raises an operating-system error.
            async def get(self, url, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises an OS error "name resolution failed".
                raise OSError("name resolution failed")

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _Failing())

        assert await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k") is False

    async def test_a_caller_supplied_timeout_replaces_the_default(self, monkeypatch):
        # [FUNCTION DOC] A probe with timeout 0.25: the recorded GET carries timeout 0.25.
        """A caller that knows its own sweep budget must be able to tighten the bound."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="k", timeout=0.25)

        assert rec.calls[0][2]["timeout"] == 0.25

    async def test_it_authenticates_like_every_other_call(self, monkeypatch):
        # [FUNCTION DOC] A probe with API key "secret": the recorded GET's authorization header is
        # "Bearer secret".
        """A probe rejected for missing auth would look exactly like an engine that never starts."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        await sglang_api_client.probe_server_healthy(server_url=SERVER_URL, api_key="secret")

        assert rec.calls[0][2]["headers"]["Authorization"] == "Bearer secret"


class TestWaitServerHealthy:
    # [CLASS DOC] Tests for the module-level function that polls health, then flush-cache, until
    # both answer successfully.
    """``wait_server_healthy`` polls until the server answers on health and flush_cache."""

    async def test_it_polls_health_then_flush_cache(self, monkeypatch):
        # [FUNCTION DOC] Responses: 503, OK, OK, with sleep patched to a no-op. Asserts the GET
        # URLs are, in order, /health_generate twice then /flush_cache once.
        """Readiness means both the health endpoint and a drained working queue."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(status_code=503), _FakeResponse(), _FakeResponse()])
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        await sglang_api_client.wait_server_healthy(server_url=SERVER_URL, api_key="k")

        assert [url for _verb, url, _kwargs in rec.calls] == [
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/flush_cache",
        ]

    async def test_both_phases_retry_transport_errors_and_non_200_responses(self, monkeypatch):
        # [FUNCTION DOC] Scripts six outcomes: health connect error, health 503, health OK, then
        # flush connect error, flush 500, flush OK; sleep is patched to record durations. Asserts
        # the requested URLs are three health then three flush-cache GETs, and the sleeps are
        # exactly [2, 2, 2, 2] (one 2-second back-off after each of the four failures).
        """Health and flush_cache each back off and retry on an httpx error as well as on a non-200 body."""

        class _Sequenced:
            # [CLASS DOC] HELPER: Fake client replaying a scripted sequence of outcomes (responses
            # to return or exceptions to raise) for GET calls and recording the URLs.
            def __init__(self, outcomes):
                # [FUNCTION DOC] HELPER: Copies the scripted outcome list and starts an empty URL list.
                self.outcomes = list(outcomes)
                self.urls: list[str] = []

            async def get(self, url, **kwargs):
                # [FUNCTION DOC] HELPER: Records the URL, pops the first scripted outcome, raises
                # it if it is an exception, otherwise returns it.
                self.urls.append(url)
                outcome = self.outcomes.pop(0)
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

        sequenced = _Sequenced(
            [
                httpx.ConnectError("health refused"),
                _FakeResponse(status_code=503),
                _FakeResponse(),
                httpx.ConnectError("flush refused"),
                _FakeResponse(status_code=500),
                _FakeResponse(),
            ]
        )
        sleep_calls = []

        async def recording_sleep(seconds):
            # [FUNCTION DOC] HELPER: Records the requested duration without sleeping.
            sleep_calls.append(seconds)

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: sequenced)
        monkeypatch.setattr(asyncio, "sleep", recording_sleep)

        await sglang_api_client.wait_server_healthy(server_url=SERVER_URL, api_key="k")

        assert sequenced.urls == [
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/health_generate",
            f"{SERVER_URL}/flush_cache",
            f"{SERVER_URL}/flush_cache",
            f"{SERVER_URL}/flush_cache",
        ]
        assert sleep_calls == [2, 2, 2, 2]


class TestMakeRequest:
    # [CLASS DOC] Tests of the client's shared private POST helper, exercised through begin-weight-update.
    """``_make_request`` is the shared POST path behind most of the client's methods."""

    async def test_it_returns_the_decoded_json_body(self, client, monkeypatch):
        # [FUNCTION DOC] With a response payload {success: true, message: "done"}, begin-weight-update
        # returns exactly that mapping.
        """Callers read fields out of the result, so the raw response object is not an acceptable return value."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(payload={"success": True, "message": "done"})])

        assert await client.begin_weight_update() == {"success": True, "message": "done"}

    async def test_a_failed_request_raises_with_the_server_response_body_attached(self, client, monkeypatch):
        # [FUNCTION DOC] A 500 response with text "cuda oom": begin-weight-update raises an HTTP
        # status error whose exception notes are exactly one note "response.text='cuda oom'".
        """sglang puts the real reason in the body, so the note must carry it into the traceback."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=500, text="cuda oom")])

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.begin_weight_update()

        assert excinfo.value.__notes__ == ["response.text='cuda oom'"]


class TestHealthGenerate:
    # [CLASS DOC] Tests for the client-bound health-generate method.
    """``health_generate`` is the client-bound counterpart of the module-level probe."""

    async def test_it_returns_true_and_forwards_a_custom_timeout(self, client, monkeypatch):
        # [FUNCTION DOC] With an OK response and timeout 1.5, returns true and the recorded GET
        # carries timeout 1.5.
        """The shared http client has no read timeout, so the caller's bound must reach the wire."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse()])

        assert await client.health_generate(timeout=1.5) is True
        assert rec.calls[0][2]["timeout"] == 1.5

    async def test_a_non_success_status_propagates(self, client, monkeypatch):
        # [FUNCTION DOC] A 503 response makes health-generate raise an HTTP status error.
        """An unhealthy engine must raise rather than report ``True``."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=503)])

        with pytest.raises(httpx.HTTPStatusError):
            await client.health_generate()


class TestInformationGetters:
    # [CLASS DOC] Tests for the read-only getter methods' endpoints, parameters, bounds and bodies.
    """The three read-only GET methods each carry their own endpoint, params and bound."""

    async def test_remote_instance_transfer_engine_info_targets_its_endpoint_with_a_bound(self, client, monkeypatch):
        # [FUNCTION DOC] Fetching the transfer-engine info for rank 2: asserts a GET to
        # /get_remote_instance_transfer_engine_info with params rank 2 and timeout 5.0.
        """Endpoint name and the 5-second bound are part of this method's wire contract."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"remote_instance_transfer_engine_info": {"a": 1}})])

        await client.get_remote_instance_transfer_engine_info(rank=2)

        verb, url, kwargs = rec.calls[0]
        assert (verb, url) == ("get", f"{SERVER_URL}/get_remote_instance_transfer_engine_info")
        assert (kwargs["params"], kwargs["timeout"]) == ({"rank": 2}, 5.0)

    async def test_parallelism_info_returns_the_whole_json_body(self, client, monkeypatch):
        # [FUNCTION DOC] For rank 1 the getter returns the entire response mapping (tp size 4, dp
        # size 2), sending params rank 1 and timeout 5.0.
        """Unlike the transfer-engine getter, this one does not unwrap a field."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"tp_size": 4, "dp_size": 2})])

        assert await client.get_parallelism_info(rank=1) == {"tp_size": 4, "dp_size": 2}
        assert (rec.calls[0][2]["params"], rec.calls[0][2]["timeout"]) == ({"rank": 1}, 5.0)

    async def test_server_info_returns_the_whole_json_body_without_params(self, client, monkeypatch):
        # [FUNCTION DOC] Server info returns the entire response mapping, sends no params keyword,
        # and carries timeout 5.0.
        """``/server_info`` is rank-independent, so it must not send a rank parameter."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"version": "0.4.0"})])

        assert await client.get_server_info() == {"version": "0.4.0"}
        assert "params" not in rec.calls[0][2]
        assert rec.calls[0][2]["timeout"] == 5.0

    async def test_server_info_uses_the_clients_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A client built with API key "secret" calls server info: asserts a GET to
        # /server_info with authorization header "Bearer secret".
        """A protected server must receive the client's API key as a bearer token."""
        rec = _Recorder()
        rec.install(monkeypatch)
        client = SGLangApiClient(server_url=SERVER_URL, api_key="secret")

        await client.get_server_info()

        verb, url, kwargs = rec.calls[0]
        assert (verb, url) == ("get", f"{SERVER_URL}/server_info")
        assert kwargs["headers"]["Authorization"] == "Bearer secret"


# Table of (name, callable taking the client) for methods that build their own request and do not
# go through the shared private POST helper: the three getters, pause/continue generation and
# start/stop profile.
_DIRECT_HTTP_METHODS = [
    ("get_remote_instance_transfer_engine_info", lambda c: c.get_remote_instance_transfer_engine_info(rank=0)),
    ("get_parallelism_info", lambda c: c.get_parallelism_info(rank=0)),
    ("get_server_info", lambda c: c.get_server_info()),
    ("pause_generation", lambda c: c.pause_generation()),
    ("continue_generation", lambda c: c.continue_generation()),
    ("start_profile", lambda c: c.start_profile()),
    ("stop_profile", lambda c: c.stop_profile()),
]

# Table of (name, callable taking the client) for every method that talks to the engine, used to
# check each sends the bearer token: abort-all, health, the three getters, weight version, flush
# cache, pause/continue generation and start/stop profile.
_AUTHENTICATED_CALLS = [
    ("abort_all_requests", lambda c: c.abort_all_requests()),
    ("health_generate", lambda c: c.health_generate()),
    ("get_server_info", lambda c: c.get_server_info()),
    ("get_parallelism_info", lambda c: c.get_parallelism_info(rank=0)),
    ("get_remote_instance_transfer_engine_info", lambda c: c.get_remote_instance_transfer_engine_info(rank=0)),
    ("get_weight_version", lambda c: c.get_weight_version()),
    ("flush_cache", lambda c: c.flush_cache()),
    ("pause_generation", lambda c: c.pause_generation()),
    ("continue_generation", lambda c: c.continue_generation()),
    ("start_profile", lambda c: c.start_profile()),
    ("stop_profile", lambda c: c.stop_profile()),
]


class TestEveryRequestIsAuthenticated:
    # [CLASS DOC] Test that every request path carries the API key as a bearer token.
    """An engine started with --sglang-api-key answers 401 to any request that carries no bearer token."""

    @pytest.mark.parametrize("name, call", _AUTHENTICATED_CALLS, ids=[name for name, _call in _AUTHENTICATED_CALLS])
    async def test_a_client_holding_an_api_key_sends_a_bearer_token(self, monkeypatch, name, call):
        # [FUNCTION DOC] Parametrized over the authenticated-call table. With a single canned
        # response containing a weight version and an (empty) transfer-engine info field, calls the
        # method on a client with API key "secret" and asserts the first recorded request's
        # authorization header is "Bearer secret". The name parameter is only used for the test id.
        """Every endpoint the client speaks to sits behind the same key, whatever helper builds the request."""
        rec = _Recorder()
        rec.install(
            monkeypatch,
            responses=[_FakeResponse(payload={"weight_version": "v1", "remote_instance_transfer_engine_info": {}})],
        )

        await call(SGLangApiClient(server_url=SERVER_URL, api_key="secret"))

        assert rec.calls[0][2]["headers"]["Authorization"] == "Bearer secret"


class TestMethodsThatBypassMakeRequest:
    # [CLASS DOC] Test that methods issuing their own HTTP requests still check the status code.
    @pytest.mark.parametrize("name, call", _DIRECT_HTTP_METHODS, ids=[name for name, _call in _DIRECT_HTTP_METHODS])
    async def test_methods_that_bypass_make_request_still_propagate_non_success_status(
        self, client, monkeypatch, name, call
    ):
        # [FUNCTION DOC] Parametrized over the direct-method table. A single 500 response makes
        # the method raise an HTTP status error. The name parameter is only used for the test id.
        """These methods build their own request, so each must keep its own ``raise_for_status``."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(status_code=500)])

        with pytest.raises(httpx.HTTPStatusError):
            await call(client)


class TestLoadLoraAdapterFromTensors:
    # [CLASS DOC] Tests for the tensor-based LoRA adapter load method's transport arguments.
    """The two transports are mutually exclusive and each shapes the payload differently."""

    async def test_supplying_both_transports_is_rejected(self, client, recorder):
        # [FUNCTION DOC] Passing both a serialized blob and a named-tensors list raises a value
        # error matching "exactly one of"; asserts no request was sent.
        """An ambiguous call must fail loudly instead of silently dropping one transport."""
        with pytest.raises(ValueError, match="exactly one of"):
            await client.load_lora_adapter_from_tensors(
                lora_name="l",
                config_dict={"r": 8},
                serialized_tensors="blob",
                serialized_named_tensors=["t"],
            )

        assert recorder.calls == []

    async def test_supplying_neither_transport_is_rejected(self, client, recorder):
        # [FUNCTION DOC] Passing neither transport raises the same value error; no request is sent.
        """A call with no weights at all must fail before it reaches the server."""
        with pytest.raises(ValueError, match="exactly one of"):
            await client.load_lora_adapter_from_tensors(lora_name="l", config_dict={"r": 8})

        assert recorder.calls == []

    async def test_the_serialized_tensors_transport_forwards_every_optional_field(self, client, recorder):
        # [FUNCTION DOC] Loads with a serialized blob, direct load format, pinned, added-tokens
        # config, upsert and expected checksums. Asserts the entire recorded call: one POST to
        # /load_lora_adapter_from_tensors with the full JSON payload and headers of JSON content
        # type and authorization "Bearer None".
        # KNOWN ISSUE: with no API key the client sends the literal header value "Bearer None",
        # and this test pins that behavior.
        """The whole-adapter transport plus all optional fields must reach the wire verbatim."""
        await client.load_lora_adapter_from_tensors(
            lora_name="l",
            config_dict={"r": 8},
            serialized_tensors="blob",
            load_format="direct",
            pinned=True,
            added_tokens_config={"tok": 1},
            upsert=True,
            expected_checksums={"w": "abc"},
        )

        assert recorder.calls == [
            (
                "post",
                f"{SERVER_URL}/load_lora_adapter_from_tensors",
                {
                    "json": {
                        "lora_name": "l",
                        "config_dict": {"r": 8},
                        "pinned": True,
                        "serialized_tensors": "blob",
                        "upsert": True,
                        "load_format": "direct",
                        "added_tokens_config": {"tok": 1},
                        "expected_checksums": {"w": "abc"},
                    },
                    "headers": {
                        "Content-Type": "application/json; charset=utf-8",
                        "Authorization": "Bearer None",
                    },
                },
            )
        ]


class TestLoadLoraAdapterFromDistributed:
    # [CLASS DOC] Test for the distributed LoRA adapter load method's optional fields.
    async def test_load_lora_adapter_from_distributed_forwards_non_default_adapter_options(self, client, recorder):
        # [FUNCTION DOC] Loads with pinned, added-tokens config and upsert set. Asserts the
        # payload has all of them plus names, dtypes (with "torch." prefix removed), shapes and
        # group name.
        """``pinned``, ``upsert`` and ``added_tokens_config`` all belong in the distributed payload."""
        await client.load_lora_adapter_from_distributed(
            lora_name="l",
            config_dict={"r": 8},
            names=["w"],
            dtypes=["torch.bfloat16"],
            shapes=[[1]],
            group_name="g",
            pinned=True,
            added_tokens_config={"tok": 1},
            upsert=True,
        )

        assert recorder.calls[0][2]["json"] == {
            "lora_name": "l",
            "config_dict": {"r": 8},
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "pinned": True,
            "upsert": True,
            "added_tokens_config": {"tok": 1},
        }


class TestFlushCacheTimeoutMessage:
    # [CLASS DOC] Tests of the content of the timeout error raised by the flush-cache retry loop.
    """The timeout is the only diagnostic the caller gets, so it must name the last failure."""

    async def test_it_reports_the_last_non_200_response_body(self, client, monkeypatch):
        # [FUNCTION DOC] 60 responses with status 400 and text "3 requests pending" (sleep patched):
        # flushing raises a timeout error whose message includes that text.
        """A 400 carries the reason (pending requests) in its body, not in an exception."""
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)
        _Recorder().install(
            monkeypatch, responses=[_FakeResponse(status_code=400, text="3 requests pending") for _ in range(60)]
        )

        with pytest.raises(TimeoutError, match="3 requests pending"):
            await client.flush_cache()

    async def test_it_reports_the_last_exception_message(self, client, monkeypatch):
        # [FUNCTION DOC] Every GET raises a connection error "connection refused" (sleep patched):
        # flushing raises a timeout error whose message includes that text.
        """When the server is unreachable the exception text is the only clue available."""

        class _AlwaysRaising:
            # [CLASS DOC] HELPER: Fake client whose GET always raises a connection error.
            async def get(self, url, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises a connection error "connection refused".
                raise httpx.ConnectError("connection refused")

        monkeypatch.setattr(GeneralHttpClientProvider, "client", lambda: _AlwaysRaising())
        monkeypatch.setattr(asyncio, "sleep", _noop_sleep)

        with pytest.raises(TimeoutError, match="connection refused"):
            await client.flush_cache()


class TestGetWeightVersion:
    # [CLASS DOC] Tests for the weight-version getter's endpoint order and failure behavior.
    """The client probes the new endpoint first and only then the legacy one."""

    async def test_it_prefers_model_info_and_never_touches_the_legacy_endpoint(self, client, monkeypatch):
        # [FUNCTION DOC] An OK model-info response with weight version "v9": the getter returns
        # "v9" after a single GET to /model_info.
        """A modern sglang answers /model_info, so the legacy call must not be made at all."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(payload={"weight_version": "v9"})])

        assert await client.get_weight_version() == "v9"
        assert [url for _verb, url, _kwargs in rec.calls] == [f"{SERVER_URL}/model_info"]

    async def test_it_raises_when_neither_endpoint_answers(self, client, monkeypatch):
        # [FUNCTION DOC] Two 404 responses: the getter raises an HTTP status error after trying
        # /model_info and then /get_weight_version.
        """Returning ``None`` here would silently mark every weight version as unknown."""
        rec = _Recorder()
        rec.install(monkeypatch, responses=[_FakeResponse(status_code=404), _FakeResponse(status_code=404)])

        with pytest.raises(httpx.HTTPStatusError):
            await client.get_weight_version()

        assert [url for _verb, url, _kwargs in rec.calls] == [
            f"{SERVER_URL}/model_info",
            f"{SERVER_URL}/get_weight_version",
        ]


class TestWeightControlPayloads:
    # [CLASS DOC] Tests of optional fields in the payloads of the weight-control methods.
    """Optional weight-control fields must be included when given and omitted when not."""

    async def test_tensor_update_forwards_a_weight_version_and_a_non_default_selector(self, client, recorder):
        # [FUNCTION DOC] Tensor update with weight version "run-7" and selector "draft": payload
        # has the tensors, load format none, flush false, selector draft and the weight version.
        """Multi-model engines address one submodel at a time and stamp the resulting version."""
        await client.update_weights_from_tensor(
            serialized_named_tensors=["a"], weight_version="run-7", selector="draft"
        )

        assert recorder.calls[0][2]["json"] == {
            "serialized_named_tensors": ["a"],
            "load_format": None,
            "flush_cache": False,
            "selector": "draft",
            "weight_version": "run-7",
        }

    @pytest.mark.parametrize("load_format", [None, "flattened_bucket"])
    async def test_distributed_update_forwards_a_weight_version_and_a_non_default_selector(
        self, client, recorder, load_format
    ):
        # [FUNCTION DOC] Parametrized over load format none and "flattened_bucket". Distributed
        # update with weight version "run-7" and selector "draft": payload has names, dtypes (with
        # "torch." prefix removed), shapes, group name, flush false, selector, weight version, and
        # the load format key only when the load format is not none.
        """The distributed path carries the same optional fields as the tensor path."""
        await client.update_weights_from_distributed(
            names=["w"],
            dtypes=["torch.bfloat16"],
            shapes=[[1]],
            group_name="g",
            weight_version="run-7",
            selector="draft",
            load_format=load_format,
        )

        assert recorder.calls[0][2]["json"] == {
            "names": ["w"],
            "dtypes": ["bfloat16"],
            "shapes": [[1]],
            "group_name": "g",
            "flush_cache": False,
            "selector": "draft",
            "weight_version": "run-7",
            **({"load_format": load_format} if load_format is not None else {}),
        }

    async def test_disk_update_omits_the_optional_fields_when_not_given(self, client, recorder):
        # [FUNCTION DOC] A disk update with only the model path sends a payload with just the
        # model path (no load format or weight version keys, not even null).
        """sglang rejects an explicit ``load_format=None``, so the key must be absent, not null."""
        await client.update_weights_from_disk(model_path="/ckpt")

        assert recorder.calls[0][2]["json"] == {"model_path": "/ckpt"}

    async def test_check_weights_omits_the_skip_list_when_not_given(self, client, recorder):
        # [FUNCTION DOC] A weight check with action "check", allow-quant-error true and selector
        # "draft" and no skip list sends a payload with exactly those three keys.
        """Without a skip list the request must not carry an empty ``skip_tensor_list``."""
        await client.check_weights(action="check", allow_quant_error=True, selector="draft")

        assert recorder.calls[0][2]["json"] == {
            "action": "check",
            "allow_quant_error": True,
            "selector": "draft",
        }

    async def test_begin_weight_update_forwards_a_custom_selector(self, client, recorder):
        # [FUNCTION DOC] Begin-weight-update with selector "draft" sends selector draft and
        # sync-base true.
        """The session must open on the same submodel the update will write to."""
        await client.begin_weight_update(selector="draft")

        assert recorder.calls[0][2]["json"] == {"selector": "draft", "sync_base": True}

    async def test_end_weight_update_posts_an_empty_payload(self, client, recorder):
        # [FUNCTION DOC] End-weight-update sends a payload with only expected LoRA checksums set to none.
        # KNOWN ISSUE: the test name says "empty payload" but the asserted payload is not empty;
        # it holds one key with a null value.
        """Closing the session takes no selector: post-processing covers the full model."""
        await client.end_weight_update()

        assert recorder.calls[0][2]["json"] == {"expected_lora_checksums": None}

    async def test_update_weight_version_keeps_in_flight_requests_by_default(self, client, recorder):
        # [FUNCTION DOC] Updating the version to "run-7" sends new version and abort-all false.
        """main never aborted generation on a version bump, and the broadcast path relies on that default."""
        await client.update_weight_version("run-7")

        assert recorder.calls[0][2]["json"] == {"new_version": "run-7", "abort_all_requests": False}

    async def test_update_weight_version_can_abort_all_requests_explicitly(self, client, recorder):
        # [FUNCTION DOC] Passing abort-all-requests true sends abort-all true.
        """An explicit opt-in must still reach the server for callers that do want the abort."""
        await client.update_weight_version("run-7", abort_all_requests=True)

        assert recorder.calls[0][2]["json"] == {"new_version": "run-7", "abort_all_requests": True}

    async def test_release_memory_occupation_forwards_its_tags(self, client, recorder):
        # [FUNCTION DOC] Releasing memory for tags weights and kv_cache: the second recorded call
        # (after the cache flush) carries a payload with those tags.
        """Multi-stage offload releases weights and kv_cache separately."""
        await client.release_memory_occupation(tags=["weights", "kv_cache"])

        assert recorder.calls[1][2]["json"] == {"tags": ["weights", "kv_cache"]}


class TestStartProfile:
    # [CLASS DOC] Test for the start-profile method's payload.
    async def test_start_profile_forwards_every_non_default_profile_option(self, client, recorder):
        # [FUNCTION DOC] Starting a profile with output dir, start step 5, 3 steps, CPU and GPU
        # activities, by-stage, with-stack and record-shapes: the payload contains all seven values.
        """Every profiling knob is optional, so each one must survive the trip to the wire."""
        await client.start_profile(
            output_dir="/out",
            start_step=5,
            num_steps=3,
            activities=["CPU", "GPU"],
            profile_by_stage=True,
            with_stack=True,
            record_shapes=True,
        )

        assert recorder.calls[0][2]["json"] == {
            "output_dir": "/out",
            "start_step": 5,
            "num_steps": 3,
            "activities": ["CPU", "GPU"],
            "profile_by_stage": True,
            "with_stack": True,
            "record_shapes": True,
        }


async def _noop_sleep(seconds):
    # [FUNCTION DOC] HELPER: Replacement for the async sleep that returns immediately and ignores
    # the duration.
    return None


class TestABodylessSuccess:
    # [CLASS DOC] Tests for abort-all-requests, including empty-body success responses and timeouts.
    async def test_a_success_carrying_no_body_answers_none_instead_of_raising(self, client, monkeypatch):
        # [FUNCTION DOC] A 200 response with an empty body: abort-all returns none (no JSON error).
        """The engine answers some endpoints with 200 and an empty body, and parsing that as json raises."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(body=b"")])

        assert await client.abort_all_requests() is None

    async def test_a_success_carrying_a_body_is_still_parsed(self, client, monkeypatch):
        # [FUNCTION DOC] A 200 response with JSON payload {ok: true}: abort-all returns that mapping.
        """Every other endpoint answers with json the callers read, so the empty-body case must not swallow it."""
        _Recorder().install(monkeypatch, responses=[_FakeResponse(payload={"ok": True})])

        assert await client.abort_all_requests() == {"ok": True}

    async def test_aborting_every_request_reaches_the_engines_abort_endpoint(self, client, recorder):
        # [FUNCTION DOC] Abort-all sends a POST to /abort_request with payload abort_all true.
        """A take-over aborts what the previous script left generating, and no other endpoint does that."""
        await client.abort_all_requests()

        assert recorder.calls[0][0] == "post"
        assert recorder.calls[0][1] == f"{SERVER_URL}/abort_request"
        assert recorder.calls[0][2]["json"] == {"abort_all": True}

    async def test_aborting_every_request_carries_the_budget_the_caller_gives(self, client, recorder):
        # [FUNCTION DOC] Abort-all with timeout 12.0: the recorded request carries timeout 12.0.
        """A wedged tokenizer manager is exactly what a take-over aborts, and this client waits forever by default."""
        await client.abort_all_requests(timeout=12.0)

        assert recorder.calls[0][2]["timeout"] == 12.0

    async def test_aborting_every_request_without_a_budget_leaves_the_client_default(self, client, recorder):
        # [FUNCTION DOC] Abort-all without a timeout: the recorded request has no timeout keyword.
        """Every other endpoint of this client is unbounded, so naming a timeout stays the caller's choice."""
        await client.abort_all_requests()

        assert "timeout" not in recorder.calls[0][2]
