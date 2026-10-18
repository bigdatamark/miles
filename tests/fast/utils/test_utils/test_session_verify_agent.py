# [MODULE DOC] Tests for the OpenAI-style session verification agent used by end-to-end tests:
# schedule selection of driver actions per allowed append role set (assistant input always last,
# rollback kept, system reminder excluded for some families), per-family fixed template roles,
# running the agent against a fake HTTP client and fake chat function (assistant input injection,
# rollback regeneration), retrying length-truncated completions with a distinct seed, and the
# generate wrapper that journals verification metrics (hard mismatches, coverage failures, errors)
# to a file named by an environment variable.
import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.e2e.sglang.test_session_server_multi_role._common import ModelConfig

from miles.rollout.base_types import GenerateFnOutput
from miles.utils.chat_template_utils.tito_tokenizer import VALID_APPEND_ROLES
from miles.utils.test_utils import session_verify_agent
from miles.utils.test_utils.session_verify_agent import (
    ASSISTANT_INPUT_FOLLOWUP_TEXT,
    ASSISTANT_INPUT_TEXTS,
    DriverAction,
    fixed_template_append_roles,
    run_agent,
    select_schedule,
)
from miles.utils.types import Sample


def test_all_role_schedule_places_assistant_input_last():
    # [FUNCTION DOC] With all valid append roles and 2 cycles, the schedule ends with the single
    # assistant input action and contains a rollback before it.
    schedule = select_schedule(VALID_APPEND_ROLES, cycles=2)

    assert schedule[-1] is DriverAction.ASSISTANT_INPUT
    assert schedule.count(DriverAction.ASSISTANT_INPUT) == 1
    assert DriverAction.ROLLBACK in schedule[:-1]


def test_model_config_uses_family_fixed_template_capability():
    # [FUNCTION DOC] For a qwen3 model config, the fixed template append roles of its tito model
    # equal the full valid append role set.
    cfg = ModelConfig(
        model_name="model",
        reasoning_parser="reasoning",
        tool_call_parser="tool",
        tito_model="qwen3",
    )

    assert fixed_template_append_roles(cfg.tito_model) == VALID_APPEND_ROLES


def test_assistant_input_appends_two_text_messages_then_user(monkeypatch):
    # [FUNCTION DOC] Patches the HTTP client, the chat function (records a deep copy of each
    # message list and returns a generated assistant text) and the schedule (only the assistant
    # input action). Runs the agent for qwen3. Asserts the second request's roles are system,
    # user, assistant, assistant, assistant, user; the two injected assistant contents equal the
    # module's assistant input texts; the last message is the follow-up user text; and the result
    # has driver events initial then append-assistant, 2 assistant inputs and 1 user message.
    calls = []

    class FakeAsyncClient:
        # [CLASS DOC] HELPER: Fake async context manager client that only stores its timeout.
        def __init__(self, *, timeout):
            # [FUNCTION DOC] HELPER: Stores the timeout.
            self.timeout = timeout

        async def __aenter__(self):
            # [FUNCTION DOC] HELPER: Returns the client.
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # [FUNCTION DOC] HELPER: Does not suppress exceptions.
            return None

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        # [FUNCTION DOC] HELPER: Records a deep copy of the messages and returns a chat response
        # whose single choice is an assistant message with text generated response N (N being the
        # call count) and no tool calls.
        calls.append(deepcopy(messages))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"generated response {len(calls)}",
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    monkeypatch.setattr(
        session_verify_agent,
        "select_schedule",
        lambda allowed_roles, *, cycles: [DriverAction.ASSISTANT_INPUT],
    )

    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={},
            metadata={"tito_model": "qwen3"},
        )
    )

    assert [message["role"] for message in calls[1]] == [
        "system",
        "user",
        "assistant",
        "assistant",
        "assistant",
        "user",
    ]
    assert [message["content"] for message in calls[1][-3:-1]] == list(ASSISTANT_INPUT_TEXTS)
    assert calls[1][-1] == {"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT}
    assert result["driver_events"] == ["initial", "append_assistant"]
    assert result["assistant_input_count"] == 2
    assert result["user_count"] == 1


def test_chat_complete_retries_length_with_distinct_seed(monkeypatch):
    # [FUNCTION DOC] The fake chat first answers with a length-truncated choice, then a stopped
    # one. Asserts the completion helper returns the stopped response and that the two attempts
    # used seeds 7 and 1,000,007 (seed plus one million on retry).
    attempts = []
    responses = iter(
        [
            {"choices": [{"finish_reason": "length"}]},
            {"choices": [{"finish_reason": "stop"}]},
        ]
    )

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        # [FUNCTION DOC] HELPER: Records the request keyword arguments and returns the next
        # scripted response.
        attempts.append(request_kwargs)
        return next(responses)

    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)

    response = asyncio.run(
        session_verify_agent._chat_complete(
            None,
            "http://session",
            [],
            {"seed": 7},
            label="turn",
        )
    )

    assert response["choices"][0]["finish_reason"] == "stop"
    assert [kwargs["seed"] for kwargs in attempts] == [7, 1_000_007]


def test_minimax_schedule_excludes_system_and_keeps_assistant_rollback():
    # [FUNCTION DOC] The minimax m27 family allows roles tool, user, assistant; its 1-cycle
    # schedule has no system reminder, ends with assistant input and contains a rollback.
    roles = fixed_template_append_roles("minimax_m27")
    schedule = select_schedule(roles, cycles=1)

    assert roles == ("tool", "user", "assistant")
    assert DriverAction.SYSTEM_REMINDER not in schedule
    assert schedule[-1] is DriverAction.ASSISTANT_INPUT
    assert DriverAction.ROLLBACK in schedule[:-1]


def test_qwen35_schedule_excludes_system_and_keeps_assistant_rollback():
    # [FUNCTION DOC] Same expectations as the minimax test, for the qwen35 family.
    roles = fixed_template_append_roles("qwen35")
    schedule = select_schedule(roles, cycles=1)

    assert roles == ("tool", "user", "assistant")
    assert DriverAction.SYSTEM_REMINDER not in schedule
    assert schedule[-1] is DriverAction.ASSISTANT_INPUT
    assert DriverAction.ROLLBACK in schedule[:-1]


def test_assistant_input_generated_response_is_rolled_back_and_regenerated(monkeypatch):
    # [FUNCTION DOC] Same fakes as the assistant input test, with a schedule of assistant input
    # then rollback. Asserts the injected request ends with the follow-up user text; the rollback
    # request equals the injected request exactly (the generated response was popped, injected
    # assistants stay) and contains no generated response 2; and the result has driver events
    # initial, append-assistant, rollback with counts 2 assistant inputs, 1 user, 1 rollback.
    calls = []

    class FakeAsyncClient:
        # [CLASS DOC] HELPER: Fake async context manager client that only stores its timeout.
        def __init__(self, *, timeout):
            # [FUNCTION DOC] HELPER: Stores the timeout.
            self.timeout = timeout

        async def __aenter__(self):
            # [FUNCTION DOC] HELPER: Returns the client.
            return self

        async def __aexit__(self, exc_type, exc, tb):
            # [FUNCTION DOC] HELPER: Does not suppress exceptions.
            return None

    async def fake_chat(client, base_url, messages, request_kwargs, *, label):
        # [FUNCTION DOC] HELPER: Records a deep copy of the messages and returns an assistant
        # response with text generated response N and no tool calls.
        calls.append(deepcopy(messages))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": f"generated response {len(calls)}",
                        "tool_calls": [],
                    }
                }
            ]
        }

    monkeypatch.setattr(session_verify_agent.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(session_verify_agent, "_chat", fake_chat)
    monkeypatch.setattr(
        session_verify_agent,
        "select_schedule",
        lambda allowed_roles, *, cycles: [DriverAction.ASSISTANT_INPUT, DriverAction.ROLLBACK],
    )

    result = asyncio.run(
        run_agent(
            "http://session",
            prompt=None,
            request_kwargs={},
            metadata={"tito_model": "qwen3"},
        )
    )

    injected_request = calls[1]
    rollback_request = calls[2]

    assert injected_request[-1] == {"role": "user", "content": ASSISTANT_INPUT_FOLLOWUP_TEXT}
    # The rollback retries the exact injected request: the popped generated
    # response is gone, the injected assistants stay as prompt history.
    assert rollback_request == injected_request
    assert all(message.get("content") != "generated response 2" for message in rollback_request)
    assert result["driver_events"] == ["initial", "append_assistant", "rollback"]
    assert result["assistant_input_count"] == 2
    assert result["user_count"] == 1
    assert result["rollback_count"] == 1


def _qwen3_verified_sample(mismatches):
    # [FUNCTION DOC] HELPER: Builds a sample whose metadata holds a full qwen3 driver event list
    # (initial, append tool, user, system, assistant, rollback) and the given session mismatch
    # list.
    return Sample(
        metadata={
            "driver_events": [
                "initial",
                "append_tool",
                "append_user",
                "append_system",
                "append_assistant",
                "rollback",
            ],
            "tito_session_mismatch": mismatches,
        }
    )


def test_generate_records_hard_mismatch_without_dropping_sample(monkeypatch, tmp_path):
    # [FUNCTION DOC] Points the metrics path at a temporary file and wraps the module's low-level
    # write to record payloads. The base generate returns a sample with two mismatches (a special
    # token count one and an assistant text one). Asserts generate returns that sample and writes
    # one metrics line with hard mismatch count 1, types list of the special token count, the
    # example entry, the assistant mismatch flag true, and that the single raw write equals that
    # JSON plus a newline, encoded.
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    writes = []
    real_write = session_verify_agent.os.write

    def tracked_write(fd, payload):
        # [FUNCTION DOC] HELPER: Records the payload and delegates to the real write; returns its
        # result.
        writes.append(payload)
        return real_write(fd, payload)

    monkeypatch.setattr(session_verify_agent.os, "write", tracked_write)
    returned_sample = _qwen3_verified_sample(
        [
            {
                "type": "special_token_count",
                "segment_index": -1,
                "detail": "segment count differs: expected 99, got 97",
            },
            {
                "type": "assistant_text",
                "segment_index": 3,
                "expected_text": "expected",
                "actual_text": "actual",
            },
        ]
    )

    async def fake_base_generate(input):
        # [FUNCTION DOC] HELPER: Returns an output holding the prepared sample.
        return GenerateFnOutput(samples=[returned_sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            tito_model="qwen3",
            session_verify_cycles=3,
            tool_call_failure_mode="rollback",
        ),
    )

    output = asyncio.run(session_verify_agent.generate(input_value))

    assert output.samples == [returned_sample]
    [metric] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert metric["hard_mismatch_count"] == 1
    assert metric["hard_mismatch_types"] == ["special_token_count"]
    assert metric["hard_mismatch_example"] == {
        "type": "special_token_count",
        "segment_index": -1,
        "detail": "segment count differs: expected 99, got 97",
    }
    assert metric["had_assistant_mismatch"] is True
    assert writes == [(json.dumps(metric) + "\n").encode()]


def test_hard_mismatch_still_raises_without_metrics_sidecar(monkeypatch):
    # [FUNCTION DOC] With the metrics path variable unset, verifying a sample with a non
    # assistant text mismatch raises an assertion error saying forbidden mismatches.
    monkeypatch.delenv("MILES_SESSION_VERIFY_METRICS_PATH", raising=False)
    sample = _qwen3_verified_sample([{"type": "non_assistant_text"}])

    with pytest.raises(AssertionError, match="forbidden mismatches"):
        session_verify_agent._verify_tito_samples(
            [sample],
            [sample.metadata["driver_events"]],
            allowed_roles=list(VALID_APPEND_ROLES),
        )


def test_run_agent_journals_assertion_before_retry(monkeypatch, tmp_path):
    # [FUNCTION DOC] The fixed-template-roles lookup is patched to raise an assertion error with
    # message journal me. Asserts the agent re-raises it and that exactly one record was written
    # to the metrics file with a verification error naming the run agent stage and that message.
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))

    def reject_template(_tito_model):
        # [FUNCTION DOC] HELPER: Always raises an assertion error with message journal me.
        raise AssertionError("journal me")

    monkeypatch.setattr(session_verify_agent, "fixed_template_append_roles", reject_template)

    with pytest.raises(AssertionError, match="journal me"):
        asyncio.run(
            run_agent(
                "http://session",
                prompt=None,
                request_kwargs={},
                metadata={"tito_model": "qwen3"},
            )
        )

    [record] = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert record["verification_error"]["stage"] == "session_verify_agent.run_agent"
    assert record["verification_error"]["message"] == "journal me"


def test_generate_journals_coverage_failure_after_hard_mismatch(monkeypatch, tmp_path):
    # [FUNCTION DOC] The returned sample has a special token count mismatch and lacks the
    # rollback driver event. Asserts generate raises an assertion error about missing required
    # driver events, and the metrics file has two records: the sample record with hard mismatch
    # count 1, then an error record naming the generate stage.
    metrics_path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv("MILES_SESSION_VERIFY_METRICS_PATH", str(metrics_path))
    sample = _qwen3_verified_sample([{"type": "special_token_count", "detail": "count differs"}])
    sample.metadata["driver_events"].remove("rollback")

    async def fake_base_generate(input):
        # [FUNCTION DOC] HELPER: Returns an output holding the prepared sample.
        return GenerateFnOutput(samples=[sample])

    monkeypatch.setattr(session_verify_agent, "_base_generate", fake_base_generate)
    input_value = SimpleNamespace(
        sample=Sample(),
        args=SimpleNamespace(
            tito_model="qwen3",
            session_verify_cycles=3,
            tool_call_failure_mode="rollback",
        ),
    )

    with pytest.raises(AssertionError, match="missing required driver events"):
        asyncio.run(session_verify_agent.generate(input_value))

    sample_record, error_record = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert sample_record["hard_mismatch_count"] == 1
    assert error_record["verification_error"]["stage"] == "session_verify_agent.generate"
