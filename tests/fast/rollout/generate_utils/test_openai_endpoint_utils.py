# [MODULE DOC] Tests of the client side of the session server: the OpenAI endpoint tracer
# (session creation and sample collection) and the underlying no-retry binary post primitive.
# Creation tests replace the module's HTTP post function with fakes that record calls; they cover
# the posted body (evaluation flag plus the sampling fields the session fills: temperature, top-p,
# integer top-k), reading the server instance id from arguments, random selection among several
# session server instances (and the matching cluster-side versus agent-side external URLs),
# the error for missing instances, validation of sampling replay before a session is allocated,
# and choosing the wire field set by session server version. Collection tests fake the binary post
# and the delete call and cover the call order, error propagation with cleanup, tolerance of delete
# failures, not blocking the event loop while decoding, and the v2 wire with agent metadata.
# Primitive tests install a fake HTTP client in the HTTP utilities module.
"""Tests for OpenAIEndpointTracer (session-server client side).

The sample-assembly and TITO multi-turn merge tests live in
tests/fast/rollout/session/test_samples.py (assembly) and
test_samples_codec.py (wire codec), next to the functions.
The collect_samples tests here lock the client's HTTP behavior deltas vs the
old collect_records path: single POST with no retries, non-2xx raises with the
body text, timeout raises (instead of silently ABORTing), and the session
DELETE is attempted on every path.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

import miles.utils.http_utils as http_utils
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import (
    COMPUTED_FIELDS,
    COMPUTED_FIELDS_V2,
    ROLLOUT_SAMPLING_MASK_FIELDS,
    decode_samples_and_merge_input_sample,
    encode_samples,
)
from miles.rollout.session.types import SessionServerInstance
from miles.utils.http_utils import post_bytes_no_retry
from miles.utils.types import Sample


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("create_kwargs", "expected_payload"),
    [
        ({}, {"evaluation": False}),
        ({"evaluation": False}, {"evaluation": False}),
        ({"evaluation": True}, {"evaluation": True}),
        # Only the sampling fields the session fills travel; None and other keys stay behind.
        (
            {"sampling_params": {"temperature": 0.6, "top_p": 0.9, "top_k": 20, "max_new_tokens": 8}},
            {"evaluation": False, "temperature": 0.6, "top_p": 0.9, "top_k": 20},
        ),
        # An eval dataset YAML can spell top_k as 40.0; the session body carries the integer.
        (
            {"sampling_params": {"temperature": 1.0, "top_p": 1.0, "top_k": 40.0}},
            {"evaluation": False, "temperature": 1.0, "top_p": 1.0, "top_k": 40},
        ),
    ],
)
async def test_create_reads_session_server_instance_id_from_args(monkeypatch, create_kwargs, expected_payload):
    # [FUNCTION DOC] Parametrized over create keyword arguments (default, evaluation false or
    # true, sampling parameters of which only temperature, top-p and top-k travel, and a float
    # top-k that must be sent as an integer). A fake post checks the posted body equals the
    # expectation including value types (a float must not leak through since 40.0 equals 40),
    # the action is a plain post and the URL is the sessions endpoint of the single instance.
    # Arguments hold one instance with an id and the replay flag and rollout sampling values
    # derived from the expectation. Asserts the tracer's base URL ends with the returned session
    # id, its server id is the instance address, its instance id is read locally from the
    # arguments, and only the one post was made (no health probe).
    calls: list[tuple[str, str]] = []

    async def fake_post(url: str, payload: dict, action: str = "post"):
        # [FUNCTION DOC] HELPER: Fake post; records the action and URL, asserts the payload,
        # value types, action and URL as described by the test, and returns session id
        # "session-123".
        calls.append((action, url))
        assert payload == expected_payload
        # 40.0 == 40 in Python, so the equality above cannot see a float leaking through
        assert {key: type(value) for key, value in payload.items()} == {
            key: type(value) for key, value in expected_payload.items()
        }
        assert action == "post"
        assert url == "http://127.0.0.1:12345/sessions"
        return {"session_id": "session-123"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    args = SimpleNamespace(
        session_server_instances=[SessionServerInstance(addr="127.0.0.1:12345", instance_id="server-instance-123")],
        use_sampling_support_replay=expected_payload.get("top_p", 1.0) < 1.0 or expected_payload.get("top_k", -1) > 0,
        rollout_temperature=expected_payload.get("temperature", 1.0),
        rollout_top_p=expected_payload.get("top_p", 1.0),
        rollout_top_k=expected_payload.get("top_k", -1),
    )
    tracer = await OpenAIEndpointTracer.create(args, **create_kwargs)

    assert tracer.base_url == "http://127.0.0.1:12345/sessions/session-123"
    assert tracer.session_server_id == "127.0.0.1:12345"
    assert tracer.session_server_instance_id == "server-instance-123"
    # No /health probe: the id is read locally, create() issues only the POST.
    assert calls == [("post", "http://127.0.0.1:12345/sessions")]


@pytest.mark.asyncio
async def test_create_without_instance_id_on_args(monkeypatch):
    # [FUNCTION DOC] Creates a tracer from arguments whose single instance has no instance id and
    # asserts the tracer's instance id is none.
    async def fake_post(url: str, payload: dict, action: str = "post"):
        # [FUNCTION DOC] HELPER: Fake post returning session id "session-123".
        return {"session_id": "session-123"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    args = SimpleNamespace(
        session_server_instances=[SessionServerInstance(addr="127.0.0.1:12345")],
        use_sampling_support_replay=False,
    )
    tracer = await OpenAIEndpointTracer.create(args)

    assert tracer.session_server_instance_id is None


@pytest.mark.asyncio
async def test_create_distributes_sessions_across_port_range(monkeypatch):
    # [FUNCTION DOC] Fakes the plain post (returning a numbered session id for session creation
    # and an empty mapping otherwise) and the binary post (recording the call and returning an
    # encoded empty "no records" reply). With four local instances on consecutive ports, creates
    # a tracer 32 times, clearing the call log each time; for every tracer asserts the chosen port
    # is one of the four, that collecting samples triggers exactly the calls: create session at
    # the chosen port, post samples to the tracer's base URL, delete the session (the tracer's
    # base URL), and that the base URL is under the chosen port's sessions path. Finally asserts
    # more than one port was chosen overall (probabilistic: a single port repeats 32 times with
    # chance about 1e-4 per port).
    """With a multi-port range, sessions land on more than one instance, and every
    request of a session (create, samples POST, DELETE) hits the port chosen
    at create time — the URL is the router."""
    calls: list[tuple[str, str]] = []

    async def fake_post(url: str, payload: dict, action: str = "post"):
        # [FUNCTION DOC] HELPER: Fake post; records the action and URL. A post to a sessions
        # collection URL returns a session id numbered by the number of recorded calls; any other
        # call (the delete) returns an empty mapping.
        calls.append((action, url))
        if action == "post" and url.endswith("/sessions"):
            return {"session_id": f"session-{len(calls)}"}
        return {}

    async def fake_post_bytes(url, payload, *, timeout):
        # [FUNCTION DOC] HELPER: Fake binary post; records the URL and returns an encoded reply
        # with no samples and the reason "no_records".
        calls.append(("post_bytes", url))
        return encode_samples([], {}, "no_records")

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)

    ports = [12345, 12346, 12347, 12348]
    args = SimpleNamespace(
        session_server_instances=[SessionServerInstance(addr=f"127.0.0.1:{port}") for port in ports],
        use_sampling_support_replay=False,
    )

    chosen_ports = set()
    for _ in range(32):
        calls.clear()
        tracer = await OpenAIEndpointTracer.create(args)
        port = int(tracer.session_server_id.rsplit(":", 1)[1])
        assert port in ports
        chosen_ports.add(port)

        await tracer.collect_samples(Sample(), max_seq_len=None)
        prefix = f"http://127.0.0.1:{port}"
        assert [url for _, url in calls] == [
            f"{prefix}/sessions",
            f"{tracer.base_url}/samples",
            tracer.base_url,
        ]
        assert tracer.base_url.startswith(f"{prefix}/sessions/")

    # 32 uniform picks over 4 ports miss a given port with p = (3/4)^32 ≈ 1e-4.
    assert len(chosen_ports) > 1


class TestOpenAIEndpointTracerCreate:
    # [CLASS DOC] Tests of the tracer's create factory: instance selection, internal versus
    # external addresses, missing instances and early sampling validation.
    @pytest.mark.asyncio
    async def test_create_routes_to_selected_address_across_multiple_hosts(self, monkeypatch):
        # [FUNCTION DOC] Forces the random instance choice to pick the second of two instances on
        # different hosts sharing a port, then creates a tracer. Asserts the session creation post
        # went to the second host's sessions URL only, and the tracer's server id, base URL and
        # instance id all come from the second instance.
        """create() sends the session POST to the whole selected host:port and reads that host's instance id, even when both hosts share a port."""
        posted: list[str] = []

        async def fake_post(url: str, payload: dict, action: str = "post"):
            # [FUNCTION DOC] HELPER: Fake post; records the URL and returns session id
            # "session-abc".
            posted.append(url)
            return {"session_id": "session-abc"}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
        monkeypatch.setattr(
            "miles.rollout.generate_utils.openai_endpoint_utils.random.choice", lambda instances: instances[1]
        )

        args = SimpleNamespace(
            session_server_instances=[
                SessionServerInstance(addr="10.0.0.1:5005", instance_id="instance-a"),
                SessionServerInstance(addr="10.0.0.2:5005", instance_id="instance-b"),
            ],
            use_sampling_support_replay=False,
        )
        tracer = await OpenAIEndpointTracer.create(args)

        assert posted == ["http://10.0.0.2:5005/sessions"]
        assert tracer.session_server_id == "10.0.0.2:5005"
        assert tracer.base_url == "http://10.0.0.2:5005/sessions/session-abc"
        assert tracer.session_server_instance_id == "instance-b"

    @pytest.mark.asyncio
    async def test_agent_url_names_the_same_instance_from_outside_the_cluster(self, monkeypatch):
        # [FUNCTION DOC] Two instances each have an internal address and an external address;
        # the random choice is forced to the second. Asserts the session is created over the
        # internal (cluster) address of the chosen instance, the tracer's base URL uses the
        # internal address, and its agent base URL uses the same instance's external address.
        """The agent's URL and the driver's URL are the chosen record's two views of one
        instance, so a session is never opened on one instance and dialed on another."""
        posted: list[str] = []

        async def fake_post(url: str, payload: dict, action: str = "post"):
            # [FUNCTION DOC] HELPER: Fake post; records the URL and returns session id
            # "session-abc".
            posted.append(url)
            return {"session_id": "session-abc"}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
        monkeypatch.setattr(
            "miles.rollout.generate_utils.openai_endpoint_utils.random.choice", lambda instances: instances[1]
        )

        args = SimpleNamespace(
            session_server_instances=[
                SessionServerInstance(addr="10.0.0.1:5005", external_addr="100.64.0.1:5005"),
                SessionServerInstance(addr="10.0.0.2:5005", external_addr="100.64.0.2:5005"),
            ],
            use_sampling_support_replay=False,
        )
        tracer = await OpenAIEndpointTracer.create(args)

        # The session is opened over the cluster network, not the external one.
        assert posted == ["http://10.0.0.2:5005/sessions"]
        assert tracer.base_url == "http://10.0.0.2:5005/sessions/session-abc"
        assert tracer.agent_base_url == "http://100.64.0.2:5005/sessions/session-abc"

    @pytest.mark.asyncio
    async def test_agent_url_falls_back_to_the_cluster_address(self, monkeypatch):
        # [FUNCTION DOC] With a single instance lacking an external address, asserts the tracer's
        # agent base URL equals its base URL.
        """Without an external address, the agent dials the address the driver dials."""

        async def fake_post(url: str, payload: dict, action: str = "post"):
            # [FUNCTION DOC] HELPER: Fake post returning session id "session-abc".
            return {"session_id": "session-abc"}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

        args = SimpleNamespace(
            session_server_instances=[SessionServerInstance(addr="10.0.0.1:5005")],
            use_sampling_support_replay=False,
        )
        tracer = await OpenAIEndpointTracer.create(args)

        assert tracer.agent_base_url == tracer.base_url

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "instances_kwargs", [{}, {"session_server_instances": None}, {"session_server_instances": []}]
    )
    async def test_create_without_session_server_instances_raises_before_post(self, monkeypatch, instances_kwargs):
        # [FUNCTION DOC] Parametrized over the instances attribute absent, none or empty. Asserts
        # create raises a runtime error saying the session server instances setting is not set
        # (pointing at the session server option) and that no post was issued.
        """create() raises a RuntimeError pointing at --use-session-server and issues no HTTP request when session_server_instances is absent, null or empty."""
        posted: list[str] = []

        async def fake_post(url: str, payload: dict, action: str = "post"):
            # [FUNCTION DOC] HELPER: Fake post; records the URL and returns session id
            # "session-abc" (never expected to be called).
            posted.append(url)
            return {"session_id": "session-abc"}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

        with pytest.raises(RuntimeError, match="session_server_instances is not set"):
            await OpenAIEndpointTracer.create(SimpleNamespace(**instances_kwargs))

        assert posted == []

    @pytest.mark.asyncio
    async def test_create_validates_sampling_replay_before_allocating_session(self, monkeypatch):
        # [FUNCTION DOC] With support replay on and the training temperature 1.0, create is given
        # sampling parameters with temperature 0.7. Asserts a value error saying the temperature
        # does not match the training temperature, and that no session was created (no post).
        posted: list[str] = []

        async def fake_post(url: str, payload: dict, action: str = "post"):
            # [FUNCTION DOC] HELPER: Fake post; records the URL and returns session id
            # "session-abc" (never expected to be called).
            posted.append(url)
            return {"session_id": "session-abc"}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)
        args = SimpleNamespace(
            session_server_instances=[SessionServerInstance(addr="127.0.0.1:12345")],
            use_sampling_support_replay=True,
            rollout_temperature=1.0,
            rollout_top_p=0.95,
            rollout_top_k=32,
        )

        with pytest.raises(ValueError, match="does not match the training temperature"):
            await OpenAIEndpointTracer.create(
                args,
                sampling_params={"temperature": 0.7, "top_p": 0.95, "top_k": 32},
            )

        assert posted == []


# ── collect_samples client behavior ──


def _tracer() -> OpenAIEndpointTracer:
    # [FUNCTION DOC] HELPER: Returns a tracer for a fixed local router URL and session id "sid-1".
    return OpenAIEndpointTracer(router_url="http://127.0.0.1:12345", session_id="sid-1")


def _computed_reply_payload() -> bytes:
    # [FUNCTION DOC] HELPER: Encodes a reply with one completed sample (tokens 1, 2, 10, response
    # "r" of length 1, loss mask [1], log probability -0.5) and session metadata containing a
    # maximum trim tokens entry of 1, with no empty reason. Returns the bytes.
    sample = Sample()
    sample.tokens = [1, 2, 10]
    sample.response = "r"
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [-0.5]
    sample.status = Sample.Status.COMPLETED
    return encode_samples([sample], {"max_trim_tokens": 1}, None)


class _CollectCalls:
    # [CLASS DOC] HELPER: Installs fakes for the binary post and the plain post that sample
    # collection uses, recording an ordered list of call descriptions ("POST url" and
    # "DELETE url"). The binary post returns, or raises if it is an exception, the configured
    # outcome; the delete returns an empty mapping or raises the configured delete error.
    """Patches the two HTTP primitives collect_samples uses and records order."""

    def __init__(self, monkeypatch, *, post_outcome, delete_outcome=None):
        # [FUNCTION DOC] HELPER: Creates the call record and patches the tracer module's binary
        # post and plain post functions through the monkeypatch fixture.
        self.calls: list[str] = []

        async def fake_post_bytes(url, payload, *, timeout):
            # [FUNCTION DOC] HELPER: Records a POST entry, asserts the body is exactly the maximum
            # sequence length 7, then raises the outcome if it is an exception or returns it.
            self.calls.append(f"POST {url}")
            assert payload == {"max_seq_len": 7}
            if isinstance(post_outcome, Exception):
                raise post_outcome
            return post_outcome

        async def fake_post(url, payload, action="post"):
            # [FUNCTION DOC] HELPER: Asserts the action is delete, records a DELETE entry, raises
            # the configured delete error if it is an exception, otherwise returns an empty
            # mapping.
            assert action == "delete"
            self.calls.append(f"DELETE {url}")
            if isinstance(delete_outcome, Exception):
                raise delete_outcome
            return {}

        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)
        monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)


@pytest.mark.asyncio
async def test_collect_samples_single_post_then_delete(monkeypatch):
    # [FUNCTION DOC] Collects with a successful reply and maximum sequence length 7. Asserts the
    # call order is one samples post then one session delete, the decoded sample has the encoded
    # tokens and completed status, and the session metadata is the encoded one.
    calls = _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload())
    result = await _tracer().collect_samples(Sample(), max_seq_len=7)

    assert calls.calls == [
        "POST http://127.0.0.1:12345/sessions/sid-1/samples",
        "DELETE http://127.0.0.1:12345/sessions/sid-1",
    ]
    (sample,) = result.samples
    assert sample.tokens == [1, 2, 10] and sample.status == Sample.Status.COMPLETED
    assert result.session_metadata == {"max_trim_tokens": 1}


@pytest.mark.asyncio
async def test_collect_samples_decoding_does_not_block_event_loop(monkeypatch):
    # [FUNCTION DOC] Replaces the decoding function with one that signals it started and then
    # blocks its thread on a second event (up to 1 second, raising a timeout error otherwise)
    # before delegating to the real decoder. Runs collection as a task and spins the event loop
    # (yielding each iteration, surfacing any task failure) until decoding has started, which only
    # works if decoding runs off the event loop thread; then releases the block and asserts one
    # sample is returned.
    _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload())
    decode_started = threading.Event()
    release_decode = threading.Event()

    def blocking_decode(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Signals that decoding started, waits up to 1 second for the
        # release signal (raising a timeout error saying decoding blocked the event loop if not
        # released), then returns the real decoder's result for the same arguments.
        decode_started.set()
        if not release_decode.wait(timeout=1.0):
            raise TimeoutError("sample decoding blocked the event loop")
        return decode_samples_and_merge_input_sample(*args, **kwargs)

    monkeypatch.setattr(
        "miles.rollout.generate_utils.openai_endpoint_utils.decode_samples_and_merge_input_sample",
        blocking_decode,
    )

    collect_task = asyncio.create_task(_tracer().collect_samples(Sample(), max_seq_len=7))
    while not decode_started.is_set():
        if collect_task.done():
            await collect_task
        await asyncio.sleep(0)
    release_decode.set()

    result = await collect_task
    assert len(result.samples) == 1


@pytest.mark.asyncio
async def test_collect_samples_non_2xx_raises_with_body_and_still_deletes(monkeypatch):
    # [FUNCTION DOC] The binary post raises a runtime error carrying a 422 body text. Asserts
    # collection propagates that error and that the last recorded call is still the session
    # delete.
    calls = _CollectCalls(monkeypatch, post_outcome=RuntimeError("422: trim_count 2 exceeds allowed=1"))
    with pytest.raises(RuntimeError, match="trim_count 2 exceeds allowed=1"):
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert calls.calls[-1] == "DELETE http://127.0.0.1:12345/sessions/sid-1"


@pytest.mark.asyncio
async def test_collect_samples_timeout_raises_and_still_deletes(monkeypatch):
    # [FUNCTION DOC] The binary post raises a timeout error. Asserts collection propagates it
    # (instead of returning empty records and silently aborting the sample) and that the last
    # call is still the session delete.
    # The old collect_records swallowed the timeout and returned empty records
    # (silently ABORTing the sample); the samples path must raise it.
    calls = _CollectCalls(monkeypatch, post_outcome=asyncio.TimeoutError())
    with pytest.raises(asyncio.TimeoutError):
        await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert calls.calls[-1] == "DELETE http://127.0.0.1:12345/sessions/sid-1"


@pytest.mark.asyncio
async def test_collect_samples_delete_failure_is_tolerated(monkeypatch):
    # [FUNCTION DOC] The session delete raises a runtime error after a successful post. Asserts
    # collection still returns the one decoded sample.
    _CollectCalls(monkeypatch, post_outcome=_computed_reply_payload(), delete_outcome=RuntimeError("delete boom"))
    result = await _tracer().collect_samples(Sample(), max_seq_len=7)
    assert len(result.samples) == 1


# ── post_bytes_no_retry primitive ──


class _FakeResponse:
    # [CLASS DOC] HELPER: Fake HTTP response exposing a status code, raw content bytes and text.
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        # [FUNCTION DOC] HELPER: Stores the status code, content (default empty bytes) and text
        # (default empty).
        self.status_code = status_code
        self.content = content
        self.text = text


class _FakeClient:
    # [CLASS DOC] HELPER: Fake asynchronous HTTP client with a queue of outcomes (responses or
    # exceptions) and a counter of how many posts were made.
    def __init__(self, responses):
        # [FUNCTION DOC] HELPER: Copies the outcome queue and zeroes the post counter.
        self.responses = list(responses)
        self.post_count = 0

    async def post(self, url, json=None):
        # [FUNCTION DOC] HELPER: Counts the post, pops the first queued outcome and raises it if
        # it is an exception, otherwise returns it. The URL and body are ignored. Raises an index
        # error if the queue is empty.
        self.post_count += 1
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_post_bytes_no_retry_returns_raw_bytes(monkeypatch):
    # [FUNCTION DOC] Installs a fake client returning a 200 response with binary content and
    # asserts the primitive returns exactly those bytes after one post.
    client = _FakeClient([_FakeResponse(200, content=b"\x00\x01binary")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    assert await post_bytes_no_retry("http://x/samples", {}, timeout=5) == b"\x00\x01binary"
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_does_not_retry_and_carries_body(monkeypatch):
    # [FUNCTION DOC] The fake client queues a 422 response with body text and a second outcome
    # that would raise if consumed. Asserts the primitive raises a runtime error containing the
    # status and body text and that only one post was made (no retry).
    # Two queued outcomes; a retrying client would consume both. It must not.
    client = _FakeClient([_FakeResponse(422, text="cursor 3 != len(accumulated_token_ids) 4"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(RuntimeError, match="422.*cursor 3"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1


@pytest.mark.asyncio
async def test_post_bytes_no_retry_transport_error_propagates_once(monkeypatch):
    # [FUNCTION DOC] The fake client first raises a connection error. Asserts the primitive
    # propagates it unchanged after exactly one post.
    client = _FakeClient([ConnectionError("boom"), RuntimeError("late")])
    monkeypatch.setattr(http_utils, "_http_client", client)
    with pytest.raises(ConnectionError, match="boom"):
        await post_bytes_no_retry("http://x/samples", {}, timeout=5)
    assert client.post_count == 1


# ── v2 wire (--use-session-server v2): metadata channel + extended fields ──


@pytest.mark.asyncio
async def test_collect_samples_v2_payload_carries_metadata_and_decodes_extras(monkeypatch):
    # [FUNCTION DOC] Encodes a reply with the version-2 field set for a sample with reward 0.75
    # and leaf metadata.
    # KNOWN ISSUE: the version-2 field tuple is imported again locally although it is already
    # imported at module level (redundant import).
    # A tracer configured with the v2 wire
    # fields collects for an input sample with environment metadata and agent metadata holding a
    # reward. Asserts the posted body has the maximum sequence length and a metadata entry equal to
    # the agent metadata, the decoded sample has reward 0.75 and its metadata is the input sample's
    # metadata merged with the server's leaf metadata.
    """v2 pin: the collect body gains the "metadata" key only when the caller
    passes agent metadata, and the v2 field tuple overlays reward + merged
    metadata; the v1 pin above (`payload == {"max_seq_len": 7}`) stays."""
    from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2

    sample = Sample()
    sample.tokens = [1, 2, 10]
    sample.response = "r"
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [-0.5]
    sample.status = Sample.Status.COMPLETED
    sample.reward = 0.75
    sample.metadata = {"leaf": {"node_id": 1}}
    payload = encode_samples([sample], {"max_trim_tokens": 1}, None, fields=COMPUTED_FIELDS_V2)

    seen = []

    async def fake_post_bytes(url, body, *, timeout):
        # [FUNCTION DOC] HELPER: Fake binary post; records the posted body and returns the
        # prepared encoded reply.
        seen.append(body)
        return payload

    async def fake_post(url, body, action="post"):
        # [FUNCTION DOC] HELPER: Fake plain post for the session delete; asserts the action is
        # delete and returns an empty mapping.
        assert action == "delete"
        return {}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post_bytes_no_retry", fake_post_bytes)
    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    tracer = OpenAIEndpointTracer(
        router_url="http://127.0.0.1:12345", session_id="sid-1", samples_wire_fields=COMPUTED_FIELDS_V2
    )
    input_sample = Sample()
    input_sample.metadata = {"env": "keep-me"}
    result = await tracer.collect_samples(input_sample, max_seq_len=7, agent_metadata={"reward": 0.75})

    assert seen == [{"max_seq_len": 7, "metadata": {"reward": 0.75}}]
    (decoded,) = result.samples
    assert decoded.reward == 0.75
    assert decoded.metadata == {"env": "keep-me", "leaf": {"node_id": 1}}


@pytest.mark.asyncio
async def test_create_selects_wire_fields_by_session_server_version(monkeypatch):
    # [FUNCTION DOC] With a fake session creation post, asserts the tracer's wire field set by
    # configuration: session server flag true gives the base field set; version "v2" gives the v2
    # set; with bounded sampling (top-p 0.95 and top-k 32, or top-k alone) the sampling mask fields
    # are appended to the set for either version; and an evaluation tracer gets the plain base set
    # even with bounded sampling.
    async def fake_post(url: str, payload: dict, action: str = "post"):
        # [FUNCTION DOC] HELPER: Fake post returning session id "sid-x".
        return {"session_id": "sid-x"}

    monkeypatch.setattr("miles.rollout.generate_utils.openai_endpoint_utils.post", fake_post)

    def args(version, top_p=1.0, top_k=-1):
        # [FUNCTION DOC] HELPER: Builds an arguments namespace with one local instance, the given
        # session server setting, a replay flag that is true when top-p is below 1 or top-k is
        # positive, and rollout temperature 1.0 with the given top-p and top-k (defaults 1.0 and
        # -1, meaning unbounded).
        return SimpleNamespace(
            session_server_instances=[SessionServerInstance(addr="127.0.0.1:7000")],
            use_session_server=version,
            use_sampling_support_replay=top_p < 1.0 or top_k > 0,
            rollout_temperature=1.0,
            rollout_top_p=top_p,
            rollout_top_k=top_k,
        )

    def sampling_params(top_p, top_k):
        # [FUNCTION DOC] HELPER: Returns request sampling parameters with temperature 1.0 and the
        # given top-p and top-k.
        return {"temperature": 1.0, "top_p": top_p, "top_k": top_k}

    assert (await OpenAIEndpointTracer.create(args(True))).samples_wire_fields == COMPUTED_FIELDS
    assert (await OpenAIEndpointTracer.create(args("v2"))).samples_wire_fields == COMPUTED_FIELDS_V2
    assert (
        await OpenAIEndpointTracer.create(args(True, 0.95, 32), sampling_params=sampling_params(0.95, 32))
    ).samples_wire_fields == (COMPUTED_FIELDS + ROLLOUT_SAMPLING_MASK_FIELDS)
    assert (
        await OpenAIEndpointTracer.create(args("v2", 0.95, 32), sampling_params=sampling_params(0.95, 32))
    ).samples_wire_fields == (COMPUTED_FIELDS_V2 + ROLLOUT_SAMPLING_MASK_FIELDS)
    assert (
        await OpenAIEndpointTracer.create(args(True, top_k=32), sampling_params=sampling_params(1.0, 32))
    ).samples_wire_fields == (COMPUTED_FIELDS + ROLLOUT_SAMPLING_MASK_FIELDS)
    assert (
        await OpenAIEndpointTracer.create(
            args(True, 0.95, 32), evaluation=True, sampling_params=sampling_params(0.95, 32)
        )
    ).samples_wire_fields == COMPUTED_FIELDS
