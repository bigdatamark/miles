# [MODULE DOC] End-to-end smoke tests of the session server with each bundled fixed chat template
# (seven model configurations: names, checkpoints, bundled template paths and token-in-token-out
# model names are listed in a module-level tuple). For each, a scripted backend replays a long chain
# trajectory through a real session server and the test checks the session's accumulated token ids
# against a locally computed template rendering. Needs network or cached model access for the
# tokenizers (a config whose tokenizer cannot be loaded is skipped) and opens local ports.
"""Representative session-layer smoke tests for miles-maintained fixed templates.

These tests intentionally stay narrow:

- only bundled fixed templates maintained by miles
- only tool-only multi-turn session flow
- only session/TITO plumbing + mismatch taxonomy checks

Detailed template correctness remains covered by the lower-level chat-template
tests in ``tests/fast/utils/chat_template_utils/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import requests
from tests.fast.router.session_pretokenized_test_utils import (
    ScriptedBackendTurn,
    ScriptedChatBackend,
    compute_local_session_mismatch,
    fetch_session_payload,
    forbidden_mismatches,
    load_test_tokenizer,
    make_router_env,
    teardown_router_env,
)

from miles.utils.chat_template_utils import TITOTokenizerType, resolve_fixed_chat_template
from miles.utils.test_utils.mock_trajectories import LongChainTrajectory, build_trajectory


@dataclass(frozen=True)
class FixedTemplateSmokeConfig:
    # [CLASS DOC] HELPER: Immutable description of one smoke case: a display name, the Hugging Face
    # checkpoint to load the tokenizer from, the path to the bundled fixed chat template, and the
    # token-in-token-out tokenizer type name.
    name: str
    hf_checkpoint: str
    chat_template_path: str
    tito_model: str


FIXED_TEMPLATE_SMOKE_CONFIGS: tuple[FixedTemplateSmokeConfig, ...] = (
    FixedTemplateSmokeConfig(
        name="qwen3-fixed",
        hf_checkpoint="Qwen/Qwen3-0.6B",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN3)[0],
        tito_model=TITOTokenizerType.QWEN3.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3.5-fixed",
        hf_checkpoint="Qwen/Qwen3.5-0.8B",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN35)[0],
        tito_model=TITOTokenizerType.QWEN35.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3.6-fixed",
        hf_checkpoint="Qwen/Qwen3.6-35B-A3B",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN36)[0],
        tito_model=TITOTokenizerType.QWEN36.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3.8-small-fixed",
        hf_checkpoint="Qwen/Qwen3.8-27B",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN38_SMALL)[0],
        tito_model=TITOTokenizerType.QWEN38_SMALL.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen4-exp-fixed",
        hf_checkpoint="Qwen/Qwen3.8-Flash-Next",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWEN4_EXP)[0],
        tito_model=TITOTokenizerType.QWEN4_EXP.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3-thinking2507-fixed",
        hf_checkpoint="Qwen/Qwen3-4B-Thinking-2507",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWENNEXT)[0],
        tito_model=TITOTokenizerType.QWENNEXT.value,
    ),
    FixedTemplateSmokeConfig(
        name="qwen3-next-thinking-fixed",
        hf_checkpoint="Qwen/Qwen3-Next-80B-A3B-Thinking",
        chat_template_path=resolve_fixed_chat_template(TITOTokenizerType.QWENNEXT)[0],
        tito_model=TITOTokenizerType.QWENNEXT.value,
    ),
)


def _get_followup_messages_after_assistant(full_messages: list[dict], assistant_idx: int) -> list[dict]:
    # [FUNCTION DOC] HELPER: Returns the messages that follow the assistant message at the given index,
    # up to but excluding the next assistant message (or the end), in order.
    followup = []
    i = assistant_idx + 1
    while i < len(full_messages) and full_messages[i]["role"] != "assistant":
        followup.append(full_messages[i])
        i += 1
    return followup


def _remap_followup_messages(followup_msgs: list[dict], response_tool_calls: list[dict]) -> list[dict]:
    # [FUNCTION DOC] HELPER: Copies each follow-up message; the Nth tool message (in order) gets its
    # tool call id replaced by the id of the Nth tool call the backend actually returned, if that call
    # exists (extra tool messages keep their original id). Non-tool messages are copied unchanged.
    # Returns the new list; inputs are not mutated.
    remapped = []
    tool_idx = 0
    for msg in followup_msgs:
        new_msg = dict(msg)
        if msg["role"] == "tool":
            if tool_idx < len(response_tool_calls):
                new_msg["tool_call_id"] = response_tool_calls[tool_idx]["id"]
            tool_idx += 1
        remapped.append(new_msg)
    return remapped


@pytest.mark.parametrize("config", FIXED_TEMPLATE_SMOKE_CONFIGS, ids=[c.name for c in FIXED_TEMPLATE_SMOKE_CONFIGS])
def test_bundled_fixed_template_session_smoke(config: FixedTemplateSmokeConfig):
    # [FUNCTION DOC] Parametrized over the seven fixed-template configs. Asserts the config has a
    # bundled template path; loads the tokenizer (skipping the case on a value or OS error). Builds a
    # long chain trajectory, scripts one backend turn per trajectory turn (the returned message gets
    # empty-string content when missing; the render message is the original), starts the backend and
    # a session server for the config. Creates a session; then for each turn sends the accumulated
    # messages (the first turn's request messages for turn 0) with the trajectory tools to the
    # session chat endpoint, asserting 200 and one choice, and for later turns that the backend
    # received explicit input ids. After each turn, fetches the session payload and asserts: the tree
    # has turn-number-plus-one nodes; the server's recorded mismatch list equals the locally
    # computed mismatch of the accumulated token ids against a fresh rendering of messages plus the
    # assistant reply; and no forbidden mismatch types occur. Then the assistant reply and the
    # trajectory's follow-up messages (tool results remapped to the returned tool call ids) are
    # appended for the next turn. Finally the records count equals the turn count, and all records
    # have status 200 and the chat completions path. The router environment is torn down in a final
    # cleanup step.
    assert config.chat_template_path is not None, f"{config.name} should resolve to a bundled fixed template"

    try:
        tokenizer = load_test_tokenizer(config.hf_checkpoint, config.chat_template_path)
    except (ValueError, OSError) as exc:
        pytest.skip(f"Cannot load tokenizer for {config.hf_checkpoint}: {exc}")

    trajectory = build_trajectory(tokenizer, LongChainTrajectory)
    scripted_turns = [
        ScriptedBackendTurn(
            response_message={**turn.assistant_message, "content": turn.assistant_message.get("content") or ""},
            render_message=turn.assistant_message,
        )
        for turn in trajectory.turns
    ]
    backend = ScriptedChatBackend(tokenizer, scripted_turns)
    backend.start()
    env = make_router_env(
        backend,
        hf_checkpoint=config.hf_checkpoint,
        chat_template_path=config.chat_template_path,
        tito_model=config.tito_model,
    )

    try:
        backend.reset_stats()
        session_id = requests.post(f"{env.url}/sessions", timeout=5.0).json()["session_id"]
        assistant_indices = [i for i, m in enumerate(trajectory.full_messages) if m["role"] == "assistant"]

        accumulated_messages: list[dict] = []
        for turn_idx, turn in enumerate(trajectory.turns):
            messages = list(accumulated_messages) if turn_idx > 0 else list(turn.request_messages)
            if turn_idx == 0:
                accumulated_messages = list(messages)

            payload = {"messages": messages, "tools": trajectory.tools}
            response = requests.post(
                f"{env.url}/sessions/{session_id}/v1/chat/completions",
                json=payload,
                timeout=10.0,
            )
            assert response.status_code == 200, f"{config.name} turn {turn_idx} failed: {response.text}"

            body = response.json()
            assert len(body["choices"]) == 1
            if turn_idx > 0:
                assert "input_ids" in backend.request_log[turn_idx], f"{config.name} turn {turn_idx} missing input_ids"

            assistant_msg = body["choices"][0]["message"]
            session_messages = list(messages) + [assistant_msg]

            session_payload = fetch_session_payload(env.url, session_id)
            metadata = session_payload["metadata"]
            assert len(metadata["tree"]["nodes"]) == turn_idx + 1
            remote_mismatch = metadata.get("tito_session_mismatch", [])
            local_mismatch = compute_local_session_mismatch(
                tokenizer,
                tito_model=config.tito_model,
                messages=session_messages,
                accumulated_token_ids=metadata["accumulated_token_ids"],
                tools=trajectory.tools,
            )
            assert remote_mismatch == local_mismatch
            assert (
                forbidden_mismatches(remote_mismatch) == []
            ), f"{config.name} turn {turn_idx} has forbidden mismatch types: {remote_mismatch}"

            accumulated_messages.append(assistant_msg)
            ass_idx = assistant_indices[turn_idx]
            followup_msgs = _get_followup_messages_after_assistant(trajectory.full_messages, ass_idx)
            response_tool_calls = assistant_msg.get("tool_calls") or []
            accumulated_messages.extend(_remap_followup_messages(followup_msgs, response_tool_calls))

        final_session_payload = fetch_session_payload(env.url, session_id)
        records = final_session_payload["records"]
        assert len(records) == len(trajectory.turns)
        assert all(r["status_code"] == 200 for r in records)
        assert all(r["path"] == "/v1/chat/completions" for r in records)
    finally:
        teardown_router_env(env)
