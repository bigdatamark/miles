# [MODULE DOC] Tests of the session server's samples operation (assembling training Samples from a
# session's recorded turns). Drives the session core's sample collection in-process with a real
# tokenizer (Qwen3-0.6B) and records injected directly into the registry (a two-turn fixture with
# routed-experts (R3) payloads, cache statistics and weight versions), a backend that fails the test
# if it is ever called, and the real route setup served via a test client. Constants: 3 layers and
# top-k 2 for the R3 tensors. Golden tests assert exact Sample fields (tokens, response text,
# loss mask, log-probs, status, weight version spans, R3, prefix-cache info, template-field overlay,
# metadata application order, lifecycle), truncation by maximum sequence length, speculative-decoding
# info, debug messages, metadata equality with the get-session reply, the additional-R3 mode
# (in-place weight updates, patches rebuild the full tensor, gaps give 422), the empty-reply reasons,
# the broken-chain 422 lane, and the HTTP route order and 404 mapping.
"""Samples-op tests: golden assembled Samples plus the op-level error/route contracts.

Drives `SessionCore.collect_samples` in-process against a real tokenizer (the
`test_sessions.py` precedent), with records injected via the registry — the
broken-chain and R3 fixtures cannot be produced through the chat path. The
HTTP surface (route registration order, 404 mapping) is exercised through the
real `setup_session_routes` app with a `TestClient`.

The golden tests assert the exact `Sample` field values derivable from the
two-turn records fixture — through `collect_samples` → `decode_samples_and_merge_input_sample`
overlay → the driver-side metadata application `agentic_tool_call.generate`
performs — including the template-field overlay and the metadata application
order (agent metadata overrides the input's keys; session metadata, applied
last, overrides the agent's).
"""

import json
import uuid

import numpy as np
import pybase64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_samples import _make_record

from miles.rollout.session.core import SessionCore
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.sessions import setup_session_routes
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

NUM_LAYERS = 3
TOPK = 2


def _make_config(**overrides):
    # [FUNCTION DOC] HELPER: Builds a session server config for Qwen3-0.6B with 30 s timeout, no chat
    # template override, thinking disabled, default token-in-token-out model, a random instance id,
    # 3 layers, router top-k 2, no speculative decoding, plus keyword overrides.
    return make_session_server_config(
        timeout=30,
        hf_checkpoint="Qwen/Qwen3-0.6B",
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="default",
        instance_id=uuid.uuid4().hex,
        num_layers=NUM_LAYERS,
        moe_router_topk=TOPK,
        sglang_speculative_algorithm=None,
        **overrides,
    )


_CONFIG = _make_config()


class _UnusedBackend:
    # [CLASS DOC] HELPER: Backend stub whose proxy always fails the test.
    """collect_samples never proxies; any backend call is a test bug."""

    async def do_proxy(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying sample collection must not
        # touch the proxy backend.
        raise AssertionError("collect_samples must not touch the proxy backend")


def _build_core(config=None, use_addition_r3: bool = False) -> SessionCore:
    # [FUNCTION DOC] HELPER: Builds a session core without running the full route setup: loads the
    # tokenizer named by the config (with its chat template path, remote code trusted), builds the
    # token-in-token-out tokenizer with the config's template kwargs, a session registry around both,
    # and a session core over the unused backend, registry and config using the config's instance id
    # and the additional-R3 flag. The config defaults to the module-level one. May download or read
    # model files.
    # Mirrors setup_session_routes (sessions.py): tokenizer + registry + core.
    config = config if config is not None else _CONFIG
    tokenizer = load_tokenizer(
        config.hf_checkpoint, chat_template_path=config.chat_template_path, trust_remote_code=True
    )
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=config.tito_model,
        chat_template_kwargs=config.apply_chat_template_kwargs,
    )
    registry = SessionRegistry(tokenizer, tito_tokenizer=tito_tokenizer)
    return SessionCore(_UnusedBackend(), registry, config, config.instance_id, use_addition_r3=use_addition_r3)


@pytest.fixture(scope="module")
def core():
    # [FUNCTION DOC] HELPER: Module-scoped fixture: a session core with default settings.
    return _build_core()


@pytest.fixture(scope="module")
def addition_core():
    # [FUNCTION DOC] HELPER: Module-scoped fixture: a session core with additional R3 enabled.
    return _build_core(use_addition_r3=True)


@pytest.fixture(scope="module")
def spec_core():
    # [FUNCTION DOC] HELPER: Module-scoped fixture: a session core whose config has the EAGLE
    # speculative decoding algorithm.
    return _build_core(_CONFIG.model_copy(update={"sglang_speculative_algorithm": "EAGLE"}))


# ── fixtures: a two-turn trajectory with R3 / cache stats / weight versions ──


def _r3_b64(num_tokens: int, seed: int) -> str:
    # [FUNCTION DOC] HELPER: Builds a routed-experts payload: consecutive 32-bit integers starting at
    # the seed, count tokens times layers times top-k, as raw bytes encoded in base64 text.
    arr = np.arange(seed, seed + num_tokens * NUM_LAYERS * TOPK, dtype=np.int32)
    return pybase64.b64encode(arr.tobytes()).decode("ascii")


def _two_turn_records():
    # [FUNCTION DOC] HELPER: Builds two session records. Turn 1: prompt [1, 2, 3], output [10, 11],
    # log-probs [-0.125, -0.25], 0 cached of 3 prompt tokens, weight version "w1", full R3 for 4
    # tokens seeded 0. Turn 2: prompt [1, 2, 3, 10, 11, 20, 21], output [30, 31], log-probs
    # [-0.5, -1.0], 5 cached of 7 prompt tokens, weight version "w2", full R3 for 8 tokens seeded 100.
    # R3 buffer length per record = (len(prompt) + len(output) - 1) * layers * topk.
    return [
        _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.125, -0.25],
            cached_tokens=0,
            prompt_tokens=3,
            weight_version="w1",
            routed_experts=_r3_b64(4, seed=0),
        ),
        _make_record(
            prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
            output_token_ids=[30, 31],
            output_log_probs=[-0.5, -1.0],
            cached_tokens=5,
            prompt_tokens=7,
            weight_version="w2",
            routed_experts=_r3_b64(8, seed=100),
        ),
    ]


_ACCUMULATED = [1, 2, 3, 10, 11, 20, 21, 30, 31]


def _input_sample() -> Sample:
    # [FUNCTION DOC] HELPER: Builds the driver-side input sample that the server reply is overlaid on:
    # group 4, index 9, a one-message prompt "hi", label "lbl", reward 2.5, metadata with task,
    # shared_key "from-input" and lifecycle "stale", routing key "routing-sid", training metadata
    # (loss ppo) and a generate function path.
    return Sample(
        group_index=4,
        index=9,
        prompt=[{"role": "user", "content": "hi"}],
        label="lbl",
        reward=2.5,
        metadata={"task": "t1", "shared_key": "from-input", "lifecycle": "stale"},
        routing_key="routing-sid",
        train_metadata={"loss": "ppo"},
        generate_function_path="gen.fn",
    )


# Overlapping keys lock the application order: agent overrides the input's
# shared_key; session_metadata (applied last) overrides the agent's
# max_trim_tokens plant.
_AGENT_METADATA = {"shared_key": "from-agent", "agent_only": 1, "max_trim_tokens": "agent-plant"}


async def _make_session(core, records, accumulated) -> str:
    # [FUNCTION DOC] HELPER: Creates a session through the core, appends each given record directly to
    # the registry session, and, unless accumulated is none, appends a copy of the accumulated token
    # ids as a trajectory entry. Returns the session id.
    response = await core.create_session()
    sid = json.loads(response.body)["session_id"]
    session = core.registry.sessions[sid]
    for record in records:
        session.append_record(record)
    if accumulated is not None:
        session.trajectory_token_ids.append(list(accumulated))
    return sid


async def _collect_via_op(core, sid, *, max_seq_len=None):
    # [FUNCTION DOC] HELPER: Calls the core's sample collection for the session with an optional
    # maximum sequence length and returns the pair (status code, raw response body bytes).
    response = await core.collect_samples(sid, max_seq_len=max_seq_len)
    return response.status_code, response.body


def _new_pipeline(payload, input_sample):
    # [FUNCTION DOC] HELPER: Decodes the payload and overlays it on the input sample; applies the
    # fixed agent metadata to every sample's metadata (overriding same keys), then, if any samples
    # exist, expects exactly one and applies the session metadata last (overriding agent keys).
    # Returns (samples, reply).
    """What collect_samples() does after the cutover: overlay + driver-side metadata."""
    reply = decode_samples_and_merge_input_sample(payload, input_sample)
    samples = reply.samples
    for s in samples:
        s.metadata.update(_AGENT_METADATA)
    if samples:
        (sample,) = samples
        sample.metadata.update(reply.session_metadata)
    return samples, reply


# ── golden assembly: exact expected Samples for the two-turn fixture ──


def _expected_r3(seed: int, num_tokens: int):
    # [FUNCTION DOC] HELPER: The expected R3 array: consecutive 32-bit integers from the seed,
    # shaped (tokens, layers, top-k).
    return np.arange(seed, seed + num_tokens * NUM_LAYERS * TOPK, dtype=np.int32).reshape(num_tokens, NUM_LAYERS, TOPK)


async def test_assembled_sample_golden(core):
    # [FUNCTION DOC] Collects samples for the two-turn session and runs the overlay pipeline,
    # expecting exactly one sample with: tokens equal to the accumulated ids; response text the
    # concatenated decodings of the three response spans; response length 6; loss mask [1, 1, 0, 0,
    # 1, 1] (environment tokens between turns masked); log-probs [-0.125, -0.25, 0, 0, -0.5, -1.0];
    # completed status; weight version spans for "w1" (tokens 3 to 5) and "w2" (7 to 9); R3 equal to
    # the last turn's tensor; prefix cache info (5 cached of 10 total prompt tokens); and the input
    # sample's prompt, label, reward, routing key and training metadata preserved. Metadata: task
    # kept, shared key from the agent, accumulated ids present, and a lifecycle with two segments
    # (turn 1, then turn 2 with the previous end time). Empty reason is none.
    """Turns merge into one trajectory Sample; env tokens between turns get
    zero loss/logprob, and the last turn's R3 is kept."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid)
    assert status == 200
    samples, reply = _new_pipeline(payload, _input_sample())
    (m,) = samples
    tokenizer = core.registry.tokenizer

    assert m.tokens == _ACCUMULATED
    assert m.response == tokenizer.decode([10, 11]) + tokenizer.decode([20, 21]) + tokenizer.decode([30, 31])
    assert m.response_length == 6
    assert m.loss_mask == [1, 1, 0, 0, 1, 1]
    assert m.rollout_log_probs == [-0.125, -0.25, 0.0, 0.0, -0.5, -1.0]
    assert m.status == Sample.Status.COMPLETED
    assert m.weight_versions == [
        WeightVersionsPerCall(spans=[WeightVersionSpan(version="w1", abs_start=3, abs_end=5)]),
        WeightVersionsPerCall(spans=[WeightVersionSpan(version="w2", abs_start=7, abs_end=9)]),
    ]
    assert np.array_equal(m.rollout_routed_experts, _expected_r3(100, 8))
    assert m.prefix_cache_info.to_dict() == {"cached_tokens": 5, "total_prompt_tokens": 10}
    assert m.prompt == [{"role": "user", "content": "hi"}]
    assert m.label == "lbl"
    assert m.reward == 2.5
    assert m.routing_key == "routing-sid"
    assert m.train_metadata == {"loss": "ppo"}
    assert m.metadata["task"] == "t1"
    assert m.metadata["shared_key"] == "from-agent"
    assert m.metadata["accumulated_token_ids"] == _ACCUMULATED
    assert m.metadata["lifecycle"] == [
        {"t0": None, "t1": 0.0, "turn": 1},
        {"t0": None, "t1": 0.0, "turn": 2, "prev_t1": 0.0},
    ]
    assert reply.empty_reason is None


async def test_spec_info_crosses_samples_wire(spec_core):
    # [FUNCTION DOC] A single record with 7 output tokens whose meta info carries speculative stats
    # (3 correct drafts, 5 proposed, 2 verify rounds) is collected via a speculative-decoding core:
    # the sample's speculative info dictionary has those three values plus completion tokens 7.
    output_token_ids = [10, 11, 12, 13, 14, 15, 16]
    record = _make_record(prompt_token_ids=[1, 2, 3], output_token_ids=output_token_ids)
    record.response["choices"][0]["meta_info"].update(
        {"spec_num_correct_drafts": 3, "spec_num_proposed_drafts": 5, "spec_verify_ct": 2}
    )
    sid = await _make_session(spec_core, [record], [1, 2, 3, *output_token_ids])

    status, payload = await _collect_via_op(spec_core, sid)
    assert status == 200
    (sample,), _ = _new_pipeline(payload, _input_sample())

    assert sample.spec_info.to_dict() == {
        "spec_num_correct_drafts": 3,
        "spec_num_proposed_drafts": 5,
        "spec_verify_ct": 2,
        "completion_tokens": 7,
    }


async def test_truncation_golden(core):
    # [FUNCTION DOC] With a maximum sequence length of 8, the last sample is truncated: truncated
    # status, first 8 accumulated tokens, loss mask [1, 1, 0, 0, 1], log-probs trimmed to five
    # entries, R3 trimmed by one row, and lifecycle segments for turns 1 and 2.
    """max_seq_len=8 strips one output token off the second turn (a turn-level
    budget applied before merge): the final sample ends TRUNCATED at 8 tokens
    with its per-token fields (including R3) trimmed in lockstep."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, max_seq_len=8)
    assert status == 200
    samples, _ = _new_pipeline(payload, _input_sample())

    last = samples[-1]
    assert last.status == Sample.Status.TRUNCATED
    assert last.tokens == _ACCUMULATED[:8]
    assert last.loss_mask == [1, 1, 0, 0, 1]
    assert last.rollout_log_probs == [-0.125, -0.25, 0.0, 0.0, -0.5]
    assert np.array_equal(last.rollout_routed_experts, _expected_r3(100, 8)[:-1])
    assert [segment["turn"] for segment in last.metadata["lifecycle"]] == [1, 2]


async def test_debug_messages_cross_samples_wire():
    # [FUNCTION DOC] With a debug trajectory save path configured (never written), the single decoded
    # sample's metadata messages equal the last record's request messages plus its response message.
    core = _build_core(_make_config(save_debug_trajectory_data="/unused/{rollout_id}.jsonl"))
    records = _two_turn_records()
    sid = await _make_session(core, records, _ACCUMULATED)
    _, payload = await _collect_via_op(core, sid)
    (sample,) = decode_samples_and_merge_input_sample(payload, _input_sample()).samples

    assert sample.metadata["messages"] == records[-1].request["messages"] + [
        records[-1].response["choices"][0]["message"]
    ]


async def test_session_metadata_matches_get_session(core):
    # [FUNCTION DOC] The session metadata in the samples reply equals the metadata in the core's
    # get-session reply (status 200) and contains the accumulated token ids.
    """The samples reply and the records GET must expose the same metadata dict
    (both are built by the extracted _session_metadata helper)."""
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED)
    _, payload = await _collect_via_op(core, sid)
    reply = decode_samples_and_merge_input_sample(payload, Sample())

    response = await core.get_session(sid)
    assert response.status_code == 200
    assert reply.session_metadata == json.loads(response.body)["metadata"]
    assert reply.session_metadata["accumulated_token_ids"] == _ACCUMULATED


# ── additional R3 (in-place weight updates): patches rebuild the full tensor ──


def _r3_slice_b64(seed: int, start_row: int, end_row: int) -> str:
    # [FUNCTION DOC] HELPER: Base64 payload of the integer stream rows from start (inclusive) to end
    # (exclusive), where each row has layers times top-k consecutive values, with the stream starting
    # at the seed.
    """Rows [start_row, end_row) of the same arange stream `_r3_b64(seed=...)`
    encodes in full, so addition patches rebuild exactly `_expected_r3`."""
    arr = np.arange(seed + start_row * NUM_LAYERS * TOPK, seed + end_row * NUM_LAYERS * TOPK, dtype=np.int32)
    return pybase64.b64encode(arr.tobytes()).decode("ascii")


def _two_turn_addition_records(turn2_start_len: int = 4):
    # [FUNCTION DOC] HELPER: The two-turn records with additional-R3 payloads: turn 1 carries rows 0
    # to 4 with start length 0; turn 2 carries rows from the given start (default 4) to 8 with that
    # start length; other fields as in the main fixture minus cache stats and weight versions. A
    # start other than 4 creates an overlap or gap.
    # The _two_turn_records trajectory with addition-R3 payloads: each record
    # carries only rows [start, len(prompt)+len(output)-1) plus the request
    # offset that produced them (turn 1: rows [0,4); turn 2: rows [4,8)).
    return [
        _make_record(
            prompt_token_ids=[1, 2, 3],
            output_token_ids=[10, 11],
            output_log_probs=[-0.125, -0.25],
            routed_experts=_r3_slice_b64(100, 0, 4),
            routed_experts_start_len=0,
        ),
        _make_record(
            prompt_token_ids=[1, 2, 3, 10, 11, 20, 21],
            output_token_ids=[30, 31],
            output_log_probs=[-0.5, -1.0],
            routed_experts=_r3_slice_b64(100, turn2_start_len, 8),
            routed_experts_start_len=turn2_start_len,
        ),
    ]


async def test_addition_assembled_sample_matches_full_reference(addition_core):
    # [FUNCTION DOC] With additional R3 patches, collection returns 200 and one sample with the same
    # tokens, loss mask, log-probs and completed status as the golden test, and an R3 array of 32-bit
    # integers equal to the full tensor.
    """Addition patches must reconstruct byte-for-byte the tensor the full-R3
    fixture assembles for the same tokens (test_assembled_sample_golden)."""
    sid = await _make_session(addition_core, _two_turn_addition_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(addition_core, sid)
    assert status == 200
    samples, _ = _new_pipeline(payload, _input_sample())
    (m,) = samples

    assert m.tokens == _ACCUMULATED
    assert m.loss_mask == [1, 1, 0, 0, 1, 1]
    assert m.rollout_log_probs == [-0.125, -0.25, 0.0, 0.0, -0.5, -1.0]
    assert m.status == Sample.Status.COMPLETED
    assert m.rollout_routed_experts.dtype == np.int32
    assert np.array_equal(m.rollout_routed_experts, _expected_r3(100, 8))


async def test_addition_truncation_golden(addition_core):
    # [FUNCTION DOC] With a maximum sequence length of 8 and additional R3, the last sample is
    # truncated to the first 8 tokens and its R3 has exactly 7 rows (tokens minus one).
    """max_seq_len=8 strips one token off the merged sample; R3 is materialized
    for exactly len(tokens) - 1 rows."""
    sid = await _make_session(addition_core, _two_turn_addition_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(addition_core, sid, max_seq_len=8)
    assert status == 200
    samples, _ = _new_pipeline(payload, _input_sample())

    last = samples[-1]
    assert last.status == Sample.Status.TRUNCATED
    assert last.tokens == _ACCUMULATED[:8]
    assert np.array_equal(last.rollout_routed_experts, _expected_r3(100, 8)[:-1])


async def test_addition_turn_boundary_truncation_uses_required_prefix(addition_core):
    # [FUNCTION DOC] With a maximum sequence length of 5 (which drops the second turn), the last
    # sample keeps the first 5 tokens and R3 equal to the first turn's 4 rows.
    """A max_seq_len boundary may drop the original last turn entirely."""
    sid = await _make_session(addition_core, _two_turn_addition_records(), _ACCUMULATED)
    status, payload = await _collect_via_op(addition_core, sid, max_seq_len=5)
    assert status == 200
    samples, _ = _new_pipeline(payload, _input_sample())

    last = samples[-1]
    assert last.tokens == _ACCUMULATED[:5]
    assert np.array_equal(last.rollout_routed_experts, _expected_r3(100, 4))


async def test_addition_gap_returns_422(addition_core):
    # [FUNCTION DOC] Turn 2's patch starts at row 5 although turn 1 kept only 4 rows. Collection
    # returns status 422 with a body mentioning additional R3, and the core's health check still
    # returns 200 (the failure does not break the server).
    # Turn 2 starts at row 5 while turn 1 retained only 4 rows: the assembler
    # rejects the gap as a 422 instead of assembling a corrupt tensor.
    sid = await _make_session(addition_core, _two_turn_addition_records(turn2_start_len=5), _ACCUMULATED)
    status, payload = await _collect_via_op(addition_core, sid)
    assert status == 422
    assert "additional R3" in payload.decode()

    health = await addition_core.health()
    assert health.status_code == 200


# ── empty_reason discriminator ──


async def test_no_records_reply(core):
    # [FUNCTION DOC] A session with no records and no accumulated ids returns 200 and a decoded reply
    # with no samples and empty reason "no_records".
    sid = await _make_session(core, [], None)
    status, payload = await _collect_via_op(core, sid)
    assert status == 200
    reply = decode_samples_and_merge_input_sample(payload, Sample())
    assert reply.samples == [] and reply.empty_reason == "no_records"


async def test_all_truncated_reply(core):
    # [FUNCTION DOC] With a maximum sequence length of 2, every turn is dropped: the reply is 200 with
    # no samples and empty reason "all_truncated".
    # max_seq_len=2 < the first turn's prompt+1: truncate_samples_by_total_tokens
    # drops every turn -> empty samples with the all_truncated reason; the old
    # pipeline returns [] on the same fixture (today's ABORTED path).
    records = _two_turn_records()
    sid = await _make_session(core, records, _ACCUMULATED)
    status, payload = await _collect_via_op(core, sid, max_seq_len=2)
    assert status == 200
    reply = decode_samples_and_merge_input_sample(payload, Sample())
    assert reply.samples == [] and reply.empty_reason == "all_truncated"


# ── the 422 lane ──


async def test_broken_chain_returns_422_and_server_survives(core):
    # [FUNCTION DOC] Accumulated ids with one extra token (99) that no record produced make collection
    # return 422 with a body mentioning the cursor, and the core's health check still returns 200.
    # The accumulated sequence carries one token the records never produced ->
    # the cursor consistency assert fires -> 422 with the assertion text, and
    # the server keeps serving (the failure never escapes as an unhandled 500).
    sid = await _make_session(core, _two_turn_records(), _ACCUMULATED + [99])
    status, payload = await _collect_via_op(core, sid)
    assert status == 422
    assert "cursor" in payload.decode()

    health = await core.health()
    assert health.status_code == 200


# ── the HTTP surface: route order and error mapping through the real app ──


@pytest.fixture(scope="module")
def app_client():
    # [FUNCTION DOC] HELPER: Module-scoped fixture: builds a web app, installs the real session routes
    # with the unused backend and the module config, and yields a test client for it (closed at
    # teardown).
    app = FastAPI()
    setup_session_routes(app, _UnusedBackend(), _CONFIG)
    with TestClient(app) as client:
        yield client


def test_missing_session_returns_404(app_client):
    # [FUNCTION DOC] Posting to the samples path of a random unknown session id returns 404 with an
    # error message containing "not found".
    response = app_client.post(f"/sessions/{uuid.uuid4().hex}/samples", content=b'{"max_seq_len":null}')
    assert response.status_code == 404
    assert "not found" in response.json()["error"]


def test_samples_route_registered_before_catch_all_proxy(app_client):
    # [FUNCTION DOC] For a fresh session, posting to its samples path returns 200 with an octet-stream
    # content type and a decodable reply whose empty reason is "no_records", proving the samples route
    # takes precedence over the catch-all proxy route (which would hit the failing backend).
    # The catch-all session_proxy would forward the request to the inference
    # backend (_UnusedBackend raises); the samples route must win instead and
    # answer with a decodable empty reply for a fresh session.
    sid = app_client.post("/sessions").json()["session_id"]
    response = app_client.post(f"/sessions/{sid}/samples", content=b'{"max_seq_len":null}')
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    reply = decode_samples_and_merge_input_sample(response.content, Sample())
    assert reply.empty_reason == "no_records", "catch-all session_proxy swallowed the samples route"
