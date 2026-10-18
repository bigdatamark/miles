# [MODULE DOC] Unit tests for the Anthropic-style session verification agent used by end-to-end
# tests: a scripted six-turn conversation (three tool-use turns each followed by a text turn)
# against a fake HTTP client, canonical record checking against a session snapshot (including
# superseded retry leaves), the retrying of incomplete responses, the generate wrapper that
# injects model settings and writes verification metrics to a file named by an environment
# variable, capability checks for intermediate system messages per model family, and rejection of
# unsupported model families. Fixtures here build canned response bodies, matching canonical
# assistant messages and a matching session snapshot (records plus a metadata tree).
import asyncio
import json
from types import SimpleNamespace

import pytest

from miles.rollout.base_types import GenerateFnOutput
from miles.utils.test_utils import anthropic_session_verify_agent
from miles.utils.types import Sample


class _Response:
    # [CLASS DOC] HELPER: Minimal fake HTTP response exposing a status code, the JSON text of the
    # body, and a method returning the body mapping.
    def __init__(self, body: dict, status_code: int = 200):
        # [FUNCTION DOC] HELPER: Stores the body, the status code (default 200) and the
        # serialized body text.
        self._body = body
        self.status_code = status_code
        self.text = json.dumps(body)

    def json(self) -> dict:
        # [FUNCTION DOC] HELPER: Returns the stored body mapping.
        return self._body


def _response_fixtures():
    # [FUNCTION DOC] HELPER: Builds the canned data for the six-turn scenario and returns three
    # items: the list of three tool use blocks (get_weather calls for Beijing, Shanghai and
    # London, with ids numbered from 1); the list of six response bodies alternating a tool-use
    # message (a thinking block plus the tool use, stop reason tool use) and a text message (a
    # thinking block plus an answer text, stop reason end turn); and the six canonical assistant
    # messages that the session is expected to record for them (tool call messages with empty
    # content and a function call whose arguments are the serialized input, or text messages),
    # each carrying the reasoning text.
    tool_uses = [
        {
            "type": "tool_use",
            "id": f"call_weather_{index}",
            "name": "get_weather",
            "input": {"location": location},
        }
        for index, location in enumerate(("Beijing", "Shanghai", "London"), start=1)
    ]
    tool_bodies = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": f"Think about turn {index}."},
                tool_use,
            ],
            "stop_reason": "tool_use",
        }
        for index, tool_use in enumerate(tool_uses, start=1)
    ]
    text_bodies = [
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": f"Use result {index}."},
                {"type": "text", "text": f"Weather answer {index}."},
            ],
            "stop_reason": "end_turn",
        }
        for index in range(1, 4)
    ]
    bodies = [item for pair in zip(tool_bodies, text_bodies, strict=True) for item in pair]

    canonical_messages = []
    for response_index, body in enumerate(bodies):
        reasoning = next(block["thinking"] for block in body["content"] if block["type"] == "thinking")
        if response_index % 2 == 0:
            tool_use = tool_uses[response_index // 2]
            message = {
                "content": "",
                "reasoning_content": reasoning,
                "tool_calls": [
                    {
                        "id": tool_use["id"],
                        "type": "function",
                        "function": {
                            "name": tool_use["name"],
                            "arguments": json.dumps(tool_use["input"]),
                        },
                    }
                ],
            }
        else:
            text = next(block["text"] for block in body["content"] if block["type"] == "text")
            message = {"content": text, "reasoning_content": reasoning}
        canonical_messages.append({"role": "assistant", **message})
    return tool_uses, bodies, canonical_messages


def _snapshot(canonical_messages, *, include_system: bool):
    # [FUNCTION DOC] HELPER: Builds the fake session snapshot mapping with records and metadata
    # for the six canonical messages. For each record it tracks the request message roles so far
    # (a user message at each even record, an optional system message before the third turn's
    # user message when the flag is set, and assistant plus tool messages after each odd record),
    # reuses earlier assistant messages in the request history, gives increasing input id lists
    # and one completion token id per response, and sets the chat completions path and a response
    # id per record. Metadata holds zero trim tokens, the accumulated ids (last request ids plus
    # one more), and a linear tree of six nodes with a single leaf at the last node whose path
    # covers all nodes. Returns the mapping with keys records and metadata.
    records = []
    history_roles = []
    for record_index, response_message in enumerate(canonical_messages):
        turn_index = record_index // 2
        if record_index % 2 == 0:
            if turn_index == 2 and include_system:
                history_roles.append("system")
            history_roles.append("user")
        else:
            history_roles.extend(["assistant", "tool"])

        prior_assistants = iter(canonical_messages[:record_index])
        request_messages = [
            dict(next(prior_assistants)) if role == "assistant" else {"role": role}
            for role in ["system", *history_roles]
        ]
        input_ids = list(range(1, record_index * 2 + 2))
        completion_id = record_index * 2 + 2
        records.append(
            {
                "path": "/v1/chat/completions",
                "request": {"input_ids": input_ids, "messages": request_messages},
                "response": {
                    "id": f"response_{record_index}",
                    "choices": [
                        {
                            "message": response_message,
                            "meta_info": {"output_token_logprobs": [[-0.1, completion_id, None]]},
                        }
                    ],
                },
            }
        )
        if record_index % 2 == 1:
            history_roles.append("assistant")

    final_ids = records[-1]["request"]["input_ids"] + [len(records) * 2]
    metadata = {
        "max_trim_tokens": 0,
        "accumulated_token_ids": final_ids,
        "tree": {
            "nodes": [
                {
                    "id": index,
                    "parent": None if index == 0 else index - 1,
                    "response_id": f"response_{index}",
                }
                for index in range(6)
            ],
            "leaves": [{"node_id": 5, "path_node_ids": [0, 1, 2, 3, 4, 5]}],
        },
    }
    return {"records": records, "metadata": metadata}


@pytest.mark.parametrize(
    ("tito_model", "route_supports", "expectation", "include_system"),
    [
        ("qwen3", True, "required", True),
        ("qwen35", True, "forbidden", False),
        ("qwen35", False, "forbidden", False),
    ],
)
def test_run_agent_runs_six_turns_and_checks_canonical_records(
    monkeypatch, tito_model, route_supports, expectation, include_system
):
    # [FUNCTION DOC] Parametrized over (qwen3 with route support, intermediate system required,
    # system message included), (qwen35 with support, forbidden, no system) and (qwen35 without
    # support, forbidden, no system). Replaces the HTTP client class in the agent module with a
    # fake that serves canned responses in order on posts, a health response advertising the
    # route's intermediate system capability, and the snapshot at the session address. Runs the
    # agent in a fresh event loop with request settings (max tokens 2048, temperature 0.2, a stop
    # string) and metadata (model, tito model, expectation). Asserts: six posts, all to the
    # messages route; the tool choice alternates forcing the weather tool and none; model, max
    # tokens, non-streaming and stop sequences are forwarded; the first request's tool schema
    # requires a location; tool result shapes (a string result, a list-of-text result, and a
    # recovery text result without an error flag); the system message appears in the tail of the
    # fifth request only when included; the thinking block of the first answer is replayed; and
    # the returned summary mapping (endpoint, expected driver events, counts of requests 6, tool
    # uses 3, tool results 3, text turns 3, string results 2, list results 1, and whether an
    # intermediate system message was used).
    posted = []
    _, response_bodies, canonical_messages = _response_fixtures()
    snapshot = _snapshot(canonical_messages, include_system=include_system)

    class FakeAsyncClient:
        # [CLASS DOC] HELPER: Fake async HTTP client usable as an async context manager. It
        # asserts the construction timeout is 180 seconds. A post records the url and JSON
        # payload and returns the next canned response body. A get on the health url returns an
        # ok status with the route capability flag; a get on the session url returns the
        # snapshot; any other url fails an assertion.
        def __init__(self, *, timeout):
            # [FUNCTION DOC] HELPER: Asserts the timeout keyword equals 180.
            assert timeout == 180

        async def __aenter__(self):
            # [FUNCTION DOC] HELPER: Returns the client itself.
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # [FUNCTION DOC] HELPER: Does not suppress exceptions (returns none).
            return None

        async def post(self, url, *, json):
            # [FUNCTION DOC] HELPER: Picks the canned body indexed by the number of posts so far,
            # records the (url, payload) pair, and returns it wrapped in a fake response.
            body = response_bodies[len(posted)]
            posted.append((url, json))
            return _Response(body)

        async def get(self, url):
            # [FUNCTION DOC] HELPER: Health url gives an ok response carrying the capability
            # flag; the session url gives the snapshot; anything else raises an assertion error.
            if url == "http://session/health":
                return _Response({"status": "ok", "anthropic_intermediate_system_supported": route_supports})
            assert url == "http://session"
            return _Response(snapshot)

    monkeypatch.setattr(anthropic_session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        anthropic_session_verify_agent.run_agent(
            "http://session",
            prompt=None,
            request_kwargs={"max_tokens": 2048, "temperature": 0.2, "stop": "<stop>"},
            metadata={
                "anthropic_model": "/models/test",
                "tito_model": tito_model,
                "anthropic_intermediate_system_expectation": expectation,
            },
        )
    )

    assert len(posted) == 6
    assert {url for url, _ in posted} == {"http://session/v1/messages"}
    assert [payload["tool_choice"] for _, payload in posted] == [
        {"type": "tool", "name": "get_weather"},
        {"type": "none"},
    ] * 3
    assert all(payload["model"] == "/models/test" for _, payload in posted)
    assert all(payload["max_tokens"] == 2048 for _, payload in posted)
    assert all(payload["stream"] is False for _, payload in posted)
    assert all(payload["stop_sequences"] == ["<stop>"] for _, payload in posted)
    assert posted[0][1]["tools"][0]["input_schema"]["required"] == ["location"]

    first_result = posted[1][1]["messages"][-1]["content"][0]
    assert first_result["type"] == "tool_result"
    assert isinstance(first_result["content"], str)
    second_result = posted[3][1]["messages"][-1]["content"][0]
    assert second_result["content"][0]["type"] == "text"
    third_result = posted[5][1]["messages"][-1]["content"][0]
    assert third_result["content"] == anthropic_session_verify_agent._TOOL_RECOVERY_TEXT
    assert "is_error" not in third_result
    tail_roles = [message["role"] for message in posted[4][1]["messages"][-2:]]
    assert ("system" in tail_roles) is include_system
    assert posted[4][1]["messages"][-1]["content"][0]["type"] == "text"
    assert posted[1][1]["messages"][1]["content"][0]["type"] == "thinking"

    expected_events = anthropic_session_verify_agent._expected_driver_events(include_system=include_system)
    assert result == {
        "endpoint": "anthropic",
        "driver_events": expected_events,
        "request_count": 6,
        "tool_use_count": 3,
        "tool_result_count": 3,
        "text_turn_count": 3,
        "tool_result_string_count": 2,
        "tool_result_list_count": 1,
        "intermediate_system_used": include_system,
    }


def test_canonical_records_allow_superseded_retry_leaves():
    # [FUNCTION DOC] Takes a snapshot whose last record has response id response_7 and replaces
    # its tree with the linear chain of five nodes plus three alternative leaves (two retry
    # branches that were superseded and the real final response), all hanging off node 4. Asserts
    # the canonical record check passes without error, i.e. superseded retry leaves are allowed
    # as long as the final response's leaf is consistent.
    tool_uses, _, canonical_messages = _response_fixtures()
    snapshot = _snapshot(canonical_messages, include_system=True)
    snapshot["records"][-1]["response"]["id"] = "response_7"
    snapshot["metadata"]["tree"] = {
        "nodes": [
            {
                "id": index,
                "parent": None if index == 0 else index - 1,
                "response_id": f"response_{index}",
            }
            for index in range(5)
        ]
        + [
            {"id": 5, "parent": 4, "response_id": "retry_1"},
            {"id": 6, "parent": 4, "response_id": "retry_2"},
            {"id": 7, "parent": 4, "response_id": "response_7"},
        ],
        "leaves": [
            {"node_id": 5, "path_node_ids": [0, 1, 2, 3, 4, 5]},
            {"node_id": 6, "path_node_ids": [0, 1, 2, 3, 4, 6]},
            {"node_id": 7, "path_node_ids": [0, 1, 2, 3, 4, 7]},
        ],
    }

    anthropic_session_verify_agent._assert_canonical_records(
        snapshot,
        [[tool_use] for tool_use in tool_uses],
        include_system=True,
    )


def test_post_complete_retries_max_tokens():
    # [FUNCTION DOC] The fake client first returns a text message cut off by the max tokens stop
    # reason, then a complete one. Asserts the post-and-complete helper returns the second body
    # (stop reason end turn) and posted the same url and payload twice.
    responses = iter(
        [
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "truncated"}],
                    "stop_reason": "max_tokens",
                }
            ),
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "complete"}],
                    "stop_reason": "end_turn",
                }
            ),
        ]
    )
    posted = []

    class FakeAsyncClient:
        # [CLASS DOC] HELPER: Fake client with only a post method that serves the scripted
        # responses in order.
        async def post(self, url, *, json):
            # [FUNCTION DOC] HELPER: Records the (url, payload) pair and returns the next
            # scripted response.
            posted.append((url, json))
            return next(responses)

    body = asyncio.run(
        anthropic_session_verify_agent._post_complete(
            FakeAsyncClient(),
            "http://session/v1/messages",
            {"messages": []},
            label="turn",
            assert_response=anthropic_session_verify_agent._assert_anthropic_text_response,
        )
    )

    assert body["stop_reason"] == "end_turn"
    assert posted == [
        ("http://session/v1/messages", {"messages": []}),
        ("http://session/v1/messages", {"messages": []}),
    ]


def test_post_complete_retries_empty_text_response():
    # [FUNCTION DOC] The fake client first returns a message whose text is empty (end turn), then
    # a complete one. Asserts the helper retries and returns the second body's content.
    responses = iter(
        [
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "stop_reason": "end_turn",
                }
            ),
            _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "complete"}],
                    "stop_reason": "end_turn",
                }
            ),
        ]
    )

    class FakeAsyncClient:
        # [CLASS DOC] HELPER: Fake client with only a post method that serves the scripted
        # responses in order.
        async def post(self, url, *, json):
            # [FUNCTION DOC] HELPER: Returns the next scripted response.
            return next(responses)

    body = asyncio.run(
        anthropic_session_verify_agent._post_complete(
            FakeAsyncClient(),
            "http://session/v1/messages",
            {"messages": []},
            label="turn",
            assert_response=anthropic_session_verify_agent._assert_anthropic_text_response,
        )
    )

    assert body["content"] == [{"type": "text", "text": "complete"}]


def test_post_complete_raises_after_bounded_incomplete_responses():
    # [FUNCTION DOC] The fake client always returns an empty text message. Asserts the helper
    # raises an assertion error after giving up, and that the number of posts equals the module's
    # maximum incomplete turn retries plus one (the first try).
    class FakeAsyncClient:
        # [CLASS DOC] HELPER: Fake client that counts posts in a class level attempt counter and
        # always answers with an empty text end-turn message.
        attempts = 0

        async def post(self, url, *, json):
            # [FUNCTION DOC] HELPER: Increments the attempt counter on the instance and returns
            # the empty text response.
            self.attempts += 1
            return _Response(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "stop_reason": "end_turn",
                }
            )

    client = FakeAsyncClient()
    with pytest.raises(AssertionError):
        asyncio.run(
            anthropic_session_verify_agent._post_complete(
                client,
                "http://session/v1/messages",
                {"messages": []},
                label="turn",
                assert_response=anthropic_session_verify_agent._assert_anthropic_text_response,
            )
        )

    assert client.attempts == anthropic_session_verify_agent._MAX_ANTHROPIC_INCOMPLETE_TURN_RETRIES + 1


def _successful_sample() -> Sample:
    # [FUNCTION DOC] HELPER: Returns a sample whose metadata looks like a successful six-turn
    # anthropic run: endpoint, expected driver events with the system message, counts (6
    # requests, 3 tool uses, 3 tool results, 3 text turns, 2 string results, 1 list result), the
    # intermediate system flag set, a leaf path of node ids ending at 7, and no session mismatch
    # entries.
    return Sample(
        metadata={
            "endpoint": "anthropic",
            "driver_events": anthropic_session_verify_agent._expected_driver_events(include_system=True),
            "request_count": 6,
            "tool_use_count": 3,
            "tool_result_count": 3,
            "text_turn_count": 3,
            "tool_result_string_count": 2,
            "tool_result_list_count": 1,
            "intermediate_system_used": True,
            "leaf": {"path_node_ids": [0, 1, 2, 3, 4, 7]},
            "tito_session_mismatch": [],
        }
    )


def test_generate_injects_model_and_writes_tito_metrics(monkeypatch, tmp_path):
    # [FUNCTION DOC] Points the metrics path environment variable at a temporary file and
    # replaces the wrapped base generate function with a fake that checks the sample it receives
    # already carries the injected model path, tito model and intermediate system expectation (all
    # taken from the input arguments), then returns the successful sample. Asserts the output
    # holds that sample, the input sample's metadata holds exactly the three injected entries, and
    # the metrics file has exactly one JSON line with the expected driver events, no assistant
    # mismatch, and zero total and hard mismatches.
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    returned_sample = _successful_sample()

    async def fake_base_generate(input):
        # [FUNCTION DOC] HELPER: Asserts the injected metadata on the received sample and returns
        # an output holding the successful sample.
        assert input.sample.metadata["anthropic_model"] == "/models/test"
        assert input.sample.metadata["tito_model"] == "qwen3"
        assert input.sample.metadata["anthropic_intermediate_system_expectation"] == "required"
        return GenerateFnOutput(samples=[returned_sample])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_sample = Sample()
    input_value = SimpleNamespace(
        sample=input_sample,
        args=SimpleNamespace(
            hf_checkpoint="/models/test",
            tito_model="qwen3",
            anthropic_intermediate_system_expectation="required",
        ),
    )

    output = asyncio.run(anthropic_session_verify_agent.generate(input_value))

    assert output.samples == [returned_sample]
    assert input_sample.metadata == {
        "anthropic_model": "/models/test",
        "tito_model": "qwen3",
        "anthropic_intermediate_system_expectation": "required",
    }
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["driver_events"] == anthropic_session_verify_agent._expected_driver_events(include_system=True)
    assert metric["had_assistant_mismatch"] is False
    assert metric["total_mismatches"] == 0
    assert metric["hard_mismatch_count"] == 0


def test_generate_records_hard_mismatch_without_dropping_sample(monkeypatch, tmp_path):
    # [FUNCTION DOC] The returned sample carries one special token type mismatch entry (segment
    # 4). Asserts generate still returns the sample, and the single metrics line reports one hard
    # mismatch, the list of its types, and the example entry with segment index 4.
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    returned_sample = _successful_sample()
    returned_sample.metadata["tito_session_mismatch"] = [
        {
            "type": "special_token_type",
            "segment_index": 4,
            "detail": "special token differs",
        }
    ]

    async def fake_base_generate(input):
        # [FUNCTION DOC] HELPER: Returns an output holding the prepared sample.
        return GenerateFnOutput(samples=[returned_sample])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            hf_checkpoint="/models/test",
            tito_model="qwen3",
            anthropic_intermediate_system_expectation="required",
        ),
    )

    output = asyncio.run(anthropic_session_verify_agent.generate(input_value))

    assert output.samples == [returned_sample]
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["hard_mismatch_count"] == 1
    assert metric["hard_mismatch_types"] == ["special_token_type"]
    assert metric["hard_mismatch_example"]["segment_index"] == 4


def test_generate_journals_stale_single_turn_metadata(monkeypatch, tmp_path):
    # [FUNCTION DOC] The returned sample's request count is 1 instead of 6 (stale single-turn
    # metadata). Asserts generate raises an assertion error saying the request count was 1 and 6
    # was expected, and that the metrics file still gets two lines: the sample's record with zero
    # hard mismatches, then an error record whose verification error names the generate stage.
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _successful_sample()
    sample.metadata["request_count"] = 1

    async def fake_base_generate(input):
        # [FUNCTION DOC] HELPER: Returns an output holding the prepared stale sample.
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            hf_checkpoint="/models/test",
            tito_model="qwen3",
            anthropic_intermediate_system_expectation="required",
        ),
    )

    with pytest.raises(AssertionError, match="request_count=1, expected 6"):
        asyncio.run(anthropic_session_verify_agent.generate(input_value))

    sample_record, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert sample_record["hard_mismatch_count"] == 0
    assert error_record["verification_error"]["stage"] == "anthropic_session_verify_agent.generate"


@pytest.mark.parametrize(
    ("tito_model", "route_supports", "expected"),
    [
        ("qwen3", True, True),
        ("qwen3", False, False),
        ("qwen35", True, False),
        ("qwen35", False, False),
    ],
)
def test_intermediate_system_family_gate(tito_model, route_supports, expected):
    # [FUNCTION DOC] Parametrized truth table: the intermediate system message is verified only
    # for the qwen3 family and only when the route supports it; qwen35 never verifies it.
    assert (
        anthropic_session_verify_agent._verify_intermediate_system(tito_model, route_supports=route_supports)
        is expected
    )


@pytest.mark.parametrize(
    ("tito_model", "route_supports", "expectation"),
    [("qwen3", False, "required"), ("qwen35", True, "required")],
)
def test_run_agent_rejects_intermediate_system_capability_drift_before_post(
    monkeypatch, tmp_path, tito_model, route_supports, expectation
):
    # [FUNCTION DOC] Parametrized over two contradictory pairs (qwen3 with an unsupported route
    # while required, and qwen35 with a supporting route while required). The fake client serves
    # only the health check and fails on any post. Asserts the agent raises an assertion error
    # saying the capability did not match the per-model contract before sending any request, and
    # that exactly one error record naming the run agent stage is written to the metrics file.
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))

    class FakeAsyncClient:
        # [CLASS DOC] HELPER: Fake async context manager client that serves the health route and
        # raises on any post.
        def __init__(self, *, timeout):
            # [FUNCTION DOC] HELPER: Asserts the timeout keyword equals 180.
            assert timeout == 180

        async def __aenter__(self):
            # [FUNCTION DOC] HELPER: Returns the client itself.
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # [FUNCTION DOC] HELPER: Does not suppress exceptions.
            return None

        async def get(self, url):
            # [FUNCTION DOC] HELPER: Asserts the url is the health url and returns an ok response
            # with the route capability flag.
            assert url == "http://session/health"
            return _Response({"status": "ok", "anthropic_intermediate_system_supported": route_supports})

        async def post(self, url, *, json):
            # [FUNCTION DOC] HELPER: Always raises an assertion error because no request may be
            # sent.
            raise AssertionError("request must not be sent after a capability mismatch")

    monkeypatch.setattr(anthropic_session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(AssertionError, match="did not match the per-model E2E contract"):
        asyncio.run(
            anthropic_session_verify_agent.run_agent(
                "http://session",
                prompt=None,
                request_kwargs={"max_tokens": 128},
                metadata={
                    "anthropic_model": "/models/test",
                    "tito_model": tito_model,
                    "anthropic_intermediate_system_expectation": expectation,
                },
            )
        )

    [record] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert record["verification_error"]["stage"] == "anthropic_session_verify_agent.run_agent"


@pytest.mark.parametrize("tito_model", ["minimax_m25", "minimax_m27"])
def test_minimax_is_rejected_before_base_generate(monkeypatch, tito_model):
    # [FUNCTION DOC] Parametrized over two minimax tito models. Asserts generate raises an
    # assertion error saying the model is not supported, and that the wrapped base generate was
    # never called.
    called = False

    async def fake_base_generate(input):
        # [FUNCTION DOC] HELPER: Marks that it was called (outer flag) and returns an output with
        # no samples.
        nonlocal called
        called = True
        return GenerateFnOutput(samples=[])

    monkeypatch.setattr(anthropic_session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(hf_checkpoint="/models/test", tito_model=tito_model),
    )

    with pytest.raises(AssertionError, match="does not support tito_model"):
        asyncio.run(anthropic_session_verify_agent.generate(input_value))
    assert called is False


@pytest.mark.parametrize("tito_model", ["minimax_m25", "minimax_m27"])
def test_run_agent_rejects_minimax_before_http(tito_model):
    # [FUNCTION DOC] Parametrized over two minimax tito models. Runs the agent with no HTTP
    # client patched and asserts it raises an assertion error about the unsupported model before
    # any network access is attempted.
    with pytest.raises(AssertionError, match="does not support tito_model"):
        asyncio.run(
            anthropic_session_verify_agent.run_agent(
                "http://session",
                prompt=None,
                request_kwargs={"max_tokens": 128},
                metadata={"anthropic_model": "/models/test", "tito_model": tito_model},
            )
        )
