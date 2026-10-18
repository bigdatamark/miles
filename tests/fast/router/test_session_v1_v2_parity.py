# [MODULE DOC] Parity tests between the v1 and v2 session servers. A mock SGLang backend serves
# scripted responses (keyed by marker text found in the prompt) to a batch of 16 scripted
# agentic trajectories that include tool calls, rollbacks/retries, system reminders, user
# follow-ups and assistant-carried input. The tests assert that the v2 server yields training
# samples bitwise equal to v1's, that an aborted generation is not committed, and that the
# bitwise sample comparator is sensitive to signed zero and to exported turn arguments.
# Module constants: model name, batch size 16, nine backend requests per trajectory, the table of
# scripted responses (each tagged with a distinct weight version) and the list of weight versions
# expected to be selected into the samples.
import uuid
from copy import deepcopy

import pytest
import requests
from tests.session_parity_utils import (
    SESSION_PARITY_SEED,
    V1,
    V2,
    _serve_session,
    _training_metadata_projection,
    assert_agentic_retry_trajectory_parity,
    assert_sample_bitwise_equal,
    run_agentic_retry_trajectories,
)

from miles.utils.test_utils.mock_sglang_server import (
    MockSGLangServer,
    ProcessResult,
    ProcessResultMetaInfo,
    with_mock_server,
)
from miles.utils.test_utils.session_verify_agent import (
    ASSISTANT_INPUT_FOLLOWUP_TEXT,
    FORCE_FINAL_TEXT,
    MOCK_TOOL_RESULTS,
    SYSTEM_REMINDER_TEXT,
    USER_FOLLOWUP_TEXT,
    build_initial_messages,
)
from miles.utils.types import Sample

_MODEL = "Qwen/Qwen3-0.6B"
_BATCH_SIZE = 16
_REQUESTS_PER_TRAJECTORY = 9
_AGENT_RESPONSES = {
    "initial": ProcessResult(
        'I will check Beijing.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"location": "Beijing"}}\n</tool_call>',
        meta_info=ProcessResultMetaInfo(weight_version="w0"),
    ),
    "beijing_result": ProcessResult(
        "Beijing is 22 degrees Celsius and sunny.",
        meta_info=ProcessResultMetaInfo(weight_version="w1"),
    ),
    "shanghai_call": ProcessResult(
        'I will check Shanghai.\n<tool_call>\n{"name": "get_weather", '
        '"arguments": {"location": "Shanghai"}}\n</tool_call>',
        meta_info=ProcessResultMetaInfo(weight_version="w2"),
    ),
    "shanghai_result": ProcessResult(
        "Shanghai is 15 degrees Celsius and cloudy.",
        meta_info=ProcessResultMetaInfo(weight_version="w3"),
    ),
    "rollback_retry": ProcessResult(
        "I will answer in one sentence.",
        meta_info=ProcessResultMetaInfo(weight_version="w4"),
    ),
    "force_final": ProcessResult(
        "<final_answer>Beijing is sunny and Shanghai is cloudy.</final_answer>",
        meta_info=ProcessResultMetaInfo(weight_version="w7"),
    ),
    "assistant_input": ProcessResult(
        "<final_answer>Beijing was sunny and Shanghai was rainy.</final_answer>",
        meta_info=ProcessResultMetaInfo(weight_version="w8"),
    ),
}
_SELECTED_WEIGHT_VERSIONS = ["w0", "w1", "w2", "w3", "w4", "w7", "w8"]


@pytest.mark.parametrize("version", [V1, V2])
def test_aborted_generation_is_not_committed(version):
    # [FUNCTION DOC] For each server version: starts a mock backend whose generation always ends
    # with an abort finish reason, serves a session on it, creates a session over HTTP and posts a
    # one-message chat request. Asserts the response is HTTP 503 with the body "upstream
    # generation aborted before completion" and that the session afterwards has no records.
    def abort_generation(_prompt: str) -> ProcessResult:
        # [FUNCTION DOC] HELPER: Mock generation callback; ignores the prompt and returns a
        # partial text with the abort finish reason.
        return ProcessResult(text="partial response", finish_reason="abort")

    with with_mock_server(model_name=_MODEL, process_fn=abort_generation) as backend:
        with _serve_session(backend_url=backend.url, hf_checkpoint=_MODEL, version=version) as args:
            session_url = args.session_server_instances[0].url
            session_id = requests.post(f"{session_url}/sessions", timeout=5.0).json()["session_id"]
            response = requests.post(
                f"{session_url}/sessions/{session_id}/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
                timeout=10.0,
            )

            assert response.status_code == 503
            assert response.json() == {"error": "upstream generation aborted before completion"}
            session = requests.get(f"{session_url}/sessions/{session_id}", timeout=5.0).json()
            assert session["records"] == []


@pytest.mark.parametrize("random_tool_ids", [False, True])
def test_agentic_v2_same_prompt_retries_match_v1_training_payload_bitwise(random_tool_ids, monkeypatch):
    # [FUNCTION DOC] Optionally (parameter random_tool_ids) patches the mock server so every
    # returned tool call gets a fresh random id. Runs the scripted agent batch against v1 and v2,
    # then asserts both produce 16 runs and, per index: the first sample's index and rollout id
    # equal the position, the per-call weight-version spans equal the expected selected versions
    # (one version per call) for both versions, and the full retry-trajectory parity check
    # passes. The patch is undone by monkeypatch at test end.
    if random_tool_ids:
        original = MockSGLangServer._compute_chat_completions_response

        def response_with_random_tool_ids(self, payload):
            # [FUNCTION DOC] HELPER: Wraps the original mock chat-completion computation and
            # overwrites each choice's tool call ids with a call_ prefix plus a random 32-hex
            # token; returns the modified response.
            response = original(self, payload)
            for choice in response["choices"]:
                for call in choice["message"].get("tool_calls") or []:
                    call["id"] = f"call_{uuid.uuid4().hex}"
            return response

        monkeypatch.setattr(MockSGLangServer, "_compute_chat_completions_response", response_with_random_tool_ids)

    v1_runs = _run_scripted_agents(V1)
    v2_runs = _run_scripted_agents(V2)

    assert len(v1_runs) == len(v2_runs) == _BATCH_SIZE
    for index, (v1, v2) in enumerate(zip(v1_runs, v2_runs, strict=True)):
        assert v1.samples[0].index == v2.samples[0].index == index
        assert v1.samples[0].rollout_id == v2.samples[0].rollout_id == index
        expected_weight_versions = [[version] for version in _SELECTED_WEIGHT_VERSIONS]
        assert [[span.version for span in call.spans] for call in v1.samples[0].weight_versions] == (
            expected_weight_versions
        )
        assert [[span.version for span in call.spans] for call in v2.samples[0].weight_versions] == (
            expected_weight_versions
        )
        assert_agentic_retry_trajectory_parity(v1, v2)


def test_sample_bitwise_comparator_distinguishes_signed_zero():
    # [FUNCTION DOC] Comparing samples with rewards 0.0 and negative 0.0 must raise an assertion
    # error saying the reward is not bitwise equal.
    with pytest.raises(AssertionError, match="sample.reward is not bitwise equal"):
        assert_sample_bitwise_equal(Sample(reward=0.0), Sample(reward=-0.0))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("tools", 0, "function", "name"), "get_time"),
        (("chat_template_kwargs", "enable_thinking"), True),
        (("temperature",), 1),
    ],
)
def test_turn_args_parity_checks_exported_request_values(path, value):
    # [FUNCTION DOC] Parametrized over (key path, new value) cases: a tool function name, a chat
    # template kwarg and the temperature (0 versus 1). Builds sample metadata holding turn args,
    # deep-copies it, changes the value at the path in the copy, and asserts the bitwise sample
    # comparator, using the training metadata projection, raises an assertion error.
    metadata = {
        "turn_args": {
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
        }
    }
    changed = deepcopy(metadata)
    target = changed["turn_args"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        assert_sample_bitwise_equal(
            Sample(metadata=metadata), Sample(metadata=changed), metadata_projection=_training_metadata_projection
        )


def _run_scripted_agents(version: str):
    # [FUNCTION DOC] HELPER: Runs the scripted agentic retry trajectories for the given server
    # version. Builds 16 input samples (index and rollout id equal to position, initial messages
    # as prompt, reward 0.25, metadata marking the source as parity), starts a mock backend with a
    # 0.05 second latency and the prompt-keyed response function, and runs the trajectories.
    # Asserts the backend logged 16 times nine requests, that all requests carried the shared
    # parity seed, and that all 16 trajectories were in flight concurrently. Returns the
    # per-trajectory results.
    input_samples = [
        Sample(
            index=index,
            rollout_id=index,
            prompt=build_initial_messages(),
            reward=0.25,
            metadata={"source": "parity"},
        )
        for index in range(_BATCH_SIZE)
    ]
    with with_mock_server(model_name=_MODEL, process_fn=_process_agent_prompt, latency=0.05) as backend:
        results = run_agentic_retry_trajectories(
            backend_url=backend.url,
            hf_checkpoint=_MODEL,
            version=version,
            input_samples=input_samples,
        )
        assert len(backend.request_log) == _BATCH_SIZE * _REQUESTS_PER_TRAJECTORY
        assert {request["seed"] for request in backend.request_log} == {SESSION_PARITY_SEED}
        assert backend.max_concurrent == _BATCH_SIZE
    return results


def _process_agent_prompt(prompt: str) -> ProcessResult:
    # [FUNCTION DOC] HELPER: Mock backend callback choosing a scripted response by marker text
    # found in the rendered prompt, checked in this priority order: assistant-input follow-up,
    # force-final, system reminder (retry response), second tool result (Shanghai result), user
    # follow-up (Shanghai call), first tool result (Beijing result); otherwise the initial
    # response. Returns the matching scripted result.
    if ASSISTANT_INPUT_FOLLOWUP_TEXT in prompt:
        return _AGENT_RESPONSES["assistant_input"]
    if FORCE_FINAL_TEXT in prompt:
        return _AGENT_RESPONSES["force_final"]
    if SYSTEM_REMINDER_TEXT in prompt:
        return _AGENT_RESPONSES["rollback_retry"]
    if MOCK_TOOL_RESULTS[1] in prompt:
        return _AGENT_RESPONSES["shanghai_result"]
    if USER_FOLLOWUP_TEXT in prompt:
        return _AGENT_RESPONSES["shanghai_call"]
    if MOCK_TOOL_RESULTS[0] in prompt:
        return _AGENT_RESPONSES["beijing_result"]
    return _AGENT_RESPONSES["initial"]
