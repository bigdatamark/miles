# [MODULE DOC] Tests that preparing token ids and request args for a session request is atomic with
# respect to session history. Uses a registry with the stub tokenizer from the linear trajectory
# tests. Case 1: any failure during preparation (invalid input ids, invalid template kwargs, invalid
# adapter path, model-side argument resolution failure, or a render/merge failure) leaves every
# session attribute (except its lock) exactly as before, for both one and two stored checkpoints.
# Case 2: a valid retry renders from the selected earlier checkpoint while the session still holds
# its full history, and the session ends rolled back to that checkpoint.
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.router.test_linear_trajectory import _make_registry

from miles.rollout.session.errors import MessageValidationError, TokenizationError
from miles.rollout.session.types import SessionRecord


USER = {"role": "user", "content": "first"}
ASSISTANT = {"role": "assistant", "content": "answer"}
TOOL = {"role": "tool", "content": "result", "tool_call_id": "call_1"}


def _snapshot(session):
    # [FUNCTION DOC] HELPER: Returns a deep copy of the session's instance attributes, excluding the
    # attribute named lock, for before/after state comparison.
    return deepcopy({key: value for key, value in vars(session).items() if key != "lock"})


def _commit(session, messages, prompt, completion):
    # [FUNCTION DOC] HELPER: Commits a finished turn: stores the messages plus the fixed assistant
    # reply with the given prompt and completion token ids (no trimming), then appends a successful
    # completions-request record (timestamp 1.0, empty response) carrying the messages.
    session.update_pretokenized_state(messages, ASSISTANT, prompt, completion, max_trim_tokens=0)
    session.append_record(
        SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": messages},
            response={},
        )
    )


@pytest.mark.parametrize("checkpoint_count", [1, 2])
@pytest.mark.parametrize("failure", ["input_ids", "kwargs", "lora", "model", "render"])
def test_failed_retry_preparation_preserves_history(checkpoint_count, failure):
    # [FUNCTION DOC] Parametrized over one or two stored checkpoints and five failure kinds. Builds a
    # session with one committed turn (plus a second with a tool result when two), snapshots it, and
    # prepares a request that repeats the last committed messages with one defect: explicit input ids
    # (message "input_ids ... is not accepted"), chat template kwargs that are a list instead of an
    # object, a LoRA path (not accepted), the tokenizer's argument resolution raising a value error
    # ("model rejected request"), or the render step failing (the first-turn render when one
    # checkpoint, the token merge when two) with a tokenization error "render failed". The expected
    # error type is a message validation error except for the render case. After the exception the
    # session snapshot must equal the one taken before.
    registry = _make_registry()
    session = registry.get_session(registry.create_session())
    _commit(session, [USER], [1], [2])
    messages = [USER]
    if checkpoint_count == 2:
        messages = [USER, ASSISTANT, TOOL]
        _commit(session, messages, [1, 2, 3], [4])
    before = _snapshot(session)
    client_args = {"messages": messages}
    error_type = MessageValidationError
    if failure == "input_ids":
        client_args["input_ids"] = [999]
        error = "input_ids=.*is not accepted"
    elif failure == "kwargs":
        client_args["chat_template_kwargs"] = []
        error = "chat_template_kwargs must be an object"
    elif failure == "lora":
        client_args["lora_path"] = "wrong-adapter"
        error = "lora_path=.*is not accepted"
    elif failure == "model":
        error = "model rejected request"
        registry.tito_tokenizer.resolve_request_args = MagicMock(side_effect=ValueError(error))
    else:
        error = "render failed"
        error_type = TokenizationError
        render = "apply_chat_template" if checkpoint_count == 1 else "merge_tokens"
        setattr(registry.tito_tokenizer, render, MagicMock(side_effect=TokenizationError(error)))

    with pytest.raises(error_type, match=error):
        session.prepare_token_ids_and_request_args(
            client_args, config=make_session_server_config(), tito_tokenizer=registry.tito_tokenizer
        )

    assert _snapshot(session) == before


def test_valid_retry_renders_selected_checkpoint_before_rollback():
    # [FUNCTION DOC] Commits two turns, snapshotting the state after the first (the checkpoint) and
    # before the retry. A retry replaces the tool message's content. The merge step is mocked to
    # assert, at call time, that the session still equals its full pre-retry state, that the old
    # messages are the checkpoint's messages, that the pretokenized ids are [1, 2] and that the new
    # messages are the retry messages, and returns [1, 2, 5]. After preparing, the body's input ids
    # are [1, 2, 5] and the session equals the checkpoint snapshot (rolled back).
    registry = _make_registry()
    session = registry.get_session(registry.create_session())
    _commit(session, [USER], [1], [2])
    checkpoint = _snapshot(session)
    _commit(session, [USER, ASSISTANT, TOOL], [1, 2, 3], [4])
    before = _snapshot(session)
    retry = [USER, ASSISTANT, {**TOOL, "content": "retry"}]

    def render(*, old_messages, new_messages, pretokenized_token_ids, template_args):
        # [FUNCTION DOC] HELPER: Mock merge behavior: asserts that the session is still in its full
        # pre-retry state and that the old messages, pretokenized ids and new messages are the
        # expected ones, then returns the token ids [1, 2, 5]. Template args are not checked.
        assert _snapshot(session) == before
        assert old_messages == checkpoint["messages"]
        assert pretokenized_token_ids == [1, 2]
        assert new_messages == retry
        return [1, 2, 5]

    registry.tito_tokenizer.merge_tokens = MagicMock(side_effect=render)
    prepared = session.prepare_token_ids_and_request_args(
        {"messages": retry}, config=make_session_server_config(), tito_tokenizer=registry.tito_tokenizer
    )

    assert prepared.body["input_ids"] == [1, 2, 5]
    assert _snapshot(session) == checkpoint
