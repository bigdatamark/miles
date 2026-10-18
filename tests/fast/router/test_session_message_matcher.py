# [MODULE DOC] Wiring tests for the configurable session message matcher at the session layer, with
# a recording stub tokenizer (first-turn sentinel; merge appends token 99 and records the old and new
# messages it was given). Covers: version 1 rollback decisions and appended-role validation under the
# strict default, loose tool-call and role-and-content-only matchers; that the tokenizer receives the
# stored prefix verbatim plus the raw replay suffix; that commits after a loose accept keep the stored
# spelling; the version 2 attach-point search under strict versus loose matching and its tokenizer
# input; and that the registry holds the strict matcher by default or an injected one. Matcher
# semantics themselves are tested elsewhere.
"""Wiring tests for the configurable session message matcher.

The hub owns matcher semantics and the runtime contract wrapper
(``test_message_matcher_hub``); this file verifies the session layer honors
one resolved matcher end to end: v1 rollback detection and append
validation, v2 attach search, the effective history handed to TITO, and
registry ownership.
"""

from typing import Any

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.linear_trajectory import LinearTrajectory, SessionRegistry
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.session_state import (
    SessionStateV2,
    attach_point_for_request,
    prepare_token_ids_and_request_args,
)
from miles.utils.chat_template_utils.message_matcher_hub import (
    loose_tool_call_message_matches,
    role_content_only_message_matches,
    strict_message_matches,
)
from miles.utils.chat_template_utils.tito_tokenizer import ALL_APPEND_ROLES, FixedTemplate, TITOTokenizer


def _prepare_token_ids(session, request_messages, tools=None, *, tito_tokenizer, message_matcher=None) -> list[int]:
    # [FUNCTION DOC] HELPER: Version 1 helper. Wraps the messages (plus tools when given) as client
    # args and calls the session's prepare method with a default config, the tokenizer and optional
    # matcher; this may roll the session back or raise validation errors as a side effect. Returns the
    # prepared body's input ids.
    """v1: rollback + request args + render for a messages-only request; returns ``input_ids``."""
    client_args = {"messages": request_messages}
    if tools is not None:
        client_args["tools"] = tools
    prepared = session.prepare_token_ids_and_request_args(
        client_args,
        config=make_session_server_config(),
        tito_tokenizer=tito_tokenizer,
        message_matcher=message_matcher,
    )
    return prepared.body["input_ids"]


def _prepare_token_ids_v2(state, request_messages, tools=None, *, tito_tokenizer, message_matcher=None) -> list[int]:
    # [FUNCTION DOC] HELPER: Version 2 helper. Same as the version 1 helper but calls the module-level
    # prepare function with the session state; it returns a pair of prepared request and parent node,
    # and only the prepared body's input ids are returned.
    """v2: attach + request args + render for a messages-only request; returns ``input_ids``."""
    client_args = {"messages": request_messages}
    if tools is not None:
        client_args["tools"] = tools
    prepared, _parent = prepare_token_ids_and_request_args(
        state,
        client_args,
        config=make_session_server_config(),
        tito_tokenizer=tito_tokenizer,
        message_matcher=message_matcher,
    )
    return prepared.body["input_ids"]


_FIRST_TURN_TOKENS = [0]

USER = {"role": "user", "content": "q"}
STORED_ASSISTANT = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call-A",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"city":"Paris","units":"metric"}'},
        }
    ],
}
# Same call under JSON-object equivalence, reserialized with different key
# order and spacing: strict rejects it, loose_tool_call accepts it.
REPLAYED_ASSISTANT = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {
            "id": "call-A",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"units": "metric", "city": "Paris"}'},
        }
    ],
}
TOOL_RESULT = {"role": "tool", "content": "ok", "tool_call_id": "call-A"}


class _RecordingTITOTokenizer(TITOTokenizer):
    # [CLASS DOC] HELPER: Stub token-in-token-out tokenizer allowing all append roles. Rendering
    # returns the first-turn sentinel; merging records the old and new messages and returns the
    # pretokenized ids plus one extra token 99.
    """Mock that records merge_tokens inputs and skips real tokenization."""

    FIXED_TEMPLATE = FixedTemplate(allowed_append_roles=ALL_APPEND_ROLES)

    def __init__(self):
        # [FUNCTION DOC] HELPER: Calls the base constructor with no real tokenizer and the assistant
        # start string "<|im_start|>assistant", and starts an empty list of recorded merge calls.
        super().__init__(tokenizer=None, assistant_start_str="<|im_start|>assistant")
        self.merge_calls: list[dict[str, Any]] = []

    def create_comparator(self):
        # [FUNCTION DOC] HELPER: Returns none.
        return None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool = False,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Ignores all arguments and returns a copy of the first-turn sentinel.
        return list(_FIRST_TURN_TOKENS)

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Records a mapping with the old and new messages (same objects, not
        # copies) and returns a new list of the pretokenized ids with token 99 appended.
        self.merge_calls.append({"old_messages": old_messages, "new_messages": new_messages})
        return list(pretokenized_token_ids) + [99]


class _ToolOnlyRecordingTITOTokenizer(_RecordingTITOTokenizer):
    # [CLASS DOC] HELPER: The recording stub restricted to appended tool messages only.
    FIXED_TEMPLATE = FixedTemplate(allowed_append_roles=frozenset({"tool"}))


def _session_with_one_checkpoint(tito: _RecordingTITOTokenizer) -> LinearTrajectory:
    # [FUNCTION DOC] HELPER: Creates a registry (no real tokenizer) around the stub tokenizer and one
    # new session; prepares ids for the first user message, then commits the stored tool-calling
    # assistant reply with prompt ids [0], completion ids [1] and no trimming. Returns the session
    # holding one checkpoint.
    registry = SessionRegistry(tokenizer=None, tito_tokenizer=tito)
    session = registry.get_session(registry.create_session())
    _prepare_token_ids(session, [USER], tito_tokenizer=tito)
    session.update_pretokenized_state([USER], STORED_ASSISTANT, [0], [1], 0)
    return session


class TestV1ReplayMatching:
    # [CLASS DOC] Tests of version 1 session behavior when a request replays the stored assistant
    # message with re-serialized tool arguments (same call, different key order and spacing).
    def test_strict_default_rolls_back_on_reserialized_arguments(self):
        # [FUNCTION DOC] Under the default strict matcher, replaying the call with re-serialized
        # arguments diverges from the stored message: the session rolls back to empty (zero
        # assistants) and the prepared ids are the first-turn sentinel.
        tito = _RecordingTITOTokenizer()
        session = _session_with_one_checkpoint(tito)

        result = _prepare_token_ids(session, [USER, REPLAYED_ASSISTANT, TOOL_RESULT], tito_tokenizer=tito)

        assert session.num_assistant == 0
        assert result == _FIRST_TURN_TOKENS

    def test_loose_matcher_keeps_the_checkpoint_and_reuses_stored_tokens(self):
        # [FUNCTION DOC] With the loose tool-call matcher the checkpoint is kept (one assistant) and
        # the prepared ids are the stored ids [0, 1] plus the stub's merge token 99.
        tito = _RecordingTITOTokenizer()
        session = _session_with_one_checkpoint(tito)

        result = _prepare_token_ids(
            session,
            [USER, REPLAYED_ASSISTANT, TOOL_RESULT],
            tito_tokenizer=tito,
            message_matcher=loose_tool_call_message_matches,
        )

        assert session.num_assistant == 1
        assert result == [0, 1, 99]

    def test_tito_receives_the_stored_prefix_verbatim_plus_the_raw_replay_suffix(self):
        # [FUNCTION DOC] After a loose-accepted replay there is exactly one merge call: the old
        # messages are the stored user and assistant messages; the new messages contain the very same
        # stored assistant object (not the replayed spelling) at index 1 and the raw tool result
        # object at index 2.
        tito = _RecordingTITOTokenizer()
        session = _session_with_one_checkpoint(tito)

        _prepare_token_ids(
            session,
            [USER, REPLAYED_ASSISTANT, TOOL_RESULT],
            tito_tokenizer=tito,
            message_matcher=loose_tool_call_message_matches,
        )

        (call,) = tito.merge_calls
        assert call["old_messages"] == [USER, STORED_ASSISTANT]
        assert call["new_messages"][1] is STORED_ASSISTANT
        assert call["new_messages"][2] is TOOL_RESULT

    def test_role_content_only_accepts_dropped_reasoning_and_different_calls(self):
        # [FUNCTION DOC] A replayed assistant message with the same role and content but no tool calls
        # and a null reasoning field is accepted by the role-and-content-only matcher: one assistant
        # is kept and the prepared ids are [0, 1, 99].
        tito = _RecordingTITOTokenizer()
        session = _session_with_one_checkpoint(tito)
        replayed = {"role": "assistant", "content": "", "reasoning_content": None}

        result = _prepare_token_ids(
            session,
            [USER, replayed, TOOL_RESULT],
            tito_tokenizer=tito,
            message_matcher=role_content_only_message_matches,
        )

        assert session.num_assistant == 1
        assert result == [0, 1, 99]

    def test_loose_matcher_does_not_bypass_appended_role_validation(self):
        # [FUNCTION DOC] With a tool-only template, a loose-accepted replay that also appends a user
        # message raises a message validation error mentioning the appended message at index 3.
        tito = _ToolOnlyRecordingTITOTokenizer()
        session = _session_with_one_checkpoint(tito)

        with pytest.raises(MessageValidationError, match="appended message at index 3"):
            _prepare_token_ids(
                session,
                [USER, REPLAYED_ASSISTANT, TOOL_RESULT, {"role": "user", "content": "next"}],
                tito_tokenizer=tito,
                message_matcher=loose_tool_call_message_matches,
            )

    def test_commit_after_loose_accept_preserves_stored_spelling(self):
        # [FUNCTION DOC] After a loose-accepted replay, committing a new assistant reply (completion
        # [7]) leaves stored messages as user, the stored assistant object (identity preserved),
        # tool result, and the new assistant.
        tito = _RecordingTITOTokenizer()
        session = _session_with_one_checkpoint(tito)
        replay = [USER, REPLAYED_ASSISTANT, TOOL_RESULT]
        tokens = _prepare_token_ids(
            session, replay, tito_tokenizer=tito, message_matcher=loose_tool_call_message_matches
        )
        next_assistant = {"role": "assistant", "content": "done"}

        session.update_pretokenized_state(replay, next_assistant, tokens, [7], 0)

        assert session.messages == [USER, STORED_ASSISTANT, TOOL_RESULT, next_assistant]
        assert session.messages[1] is STORED_ASSISTANT

    def test_canonical_replay_still_strict_matches_after_loose_commit(self):
        # [FUNCTION DOC] After a loose-accepted replay and commit, a next request using the stored
        # (canonical) spelling plus the new assistant and a new user message passes the default
        # strict matcher: two assistants are kept and the prepared ids are [0, 1, 99, 7, 99].
        tito = _RecordingTITOTokenizer()
        session = _session_with_one_checkpoint(tito)
        replay = [USER, REPLAYED_ASSISTANT, TOOL_RESULT]
        tokens = _prepare_token_ids(
            session, replay, tito_tokenizer=tito, message_matcher=loose_tool_call_message_matches
        )
        next_assistant = {"role": "assistant", "content": "done"}
        session.update_pretokenized_state(replay, next_assistant, tokens, [7], 0)

        result = _prepare_token_ids(
            session,
            [USER, STORED_ASSISTANT, TOOL_RESULT, next_assistant, {"role": "user", "content": "next"}],
            tito_tokenizer=tito,
        )

        assert session.num_assistant == 2
        assert result == [0, 1, 99, 7, 99]


def _state_with_one_node() -> tuple[SessionStateV2, Any]:
    # [FUNCTION DOC] HELPER: Creates a fresh version 2 session state and adds one root node to its tree
    # holding the user and stored assistant messages, token ids [0, 1], completion span (1, 2),
    # commit time 0.0, response id "resp-0", an empty successful record and finish reason stop.
    # Returns the pair (state, node).
    state = SessionStateV2()
    record = SessionRecord(
        timestamp=0.0, method="POST", path="/v1/chat/completions", request={}, response={}, status_code=200
    )
    node = state.tree.create_node(
        None,
        delta_messages=[USER, STORED_ASSISTANT],
        token_ids=[0, 1],
        completion_span=(1, 2),
        committed_at=0.0,
        response_id="resp-0",
        record=record,
        finish_reason="stop",
    )
    return state, node


class TestV2ReplayMatching:
    # [CLASS DOC] Tests of version 2 attach-point behavior for the same re-serialized replay.
    def test_strict_default_starts_a_new_root_on_reserialized_arguments(self):
        # [FUNCTION DOC] Under the default strict matcher, the attach point for the replay has no node
        # (it would start a new root).
        state, _ = _state_with_one_node()

        attach = attach_point_for_request(state, [USER, REPLAYED_ASSISTANT, TOOL_RESULT])

        assert attach.node is None

    def test_loose_matcher_attaches_at_the_stored_node(self):
        # [FUNCTION DOC] With the loose tool-call matcher the attach point is the stored node.
        state, node = _state_with_one_node()

        attach = attach_point_for_request(
            state, [USER, REPLAYED_ASSISTANT, TOOL_RESULT], message_matcher=loose_tool_call_message_matches
        )

        assert attach.node is node

    def test_tito_receives_the_stored_path_verbatim_plus_the_raw_replay_suffix(self):
        # [FUNCTION DOC] Preparing the loose replay for version 2 results in exactly one merge call
        # with old messages equal to the stored user and assistant, new messages holding the stored
        # assistant object at index 1 and the tool result object at index 2, and prepared ids
        # [0, 1, 99].
        state, _ = _state_with_one_node()
        tito = _RecordingTITOTokenizer()
        replay = [USER, REPLAYED_ASSISTANT, TOOL_RESULT]
        result = _prepare_token_ids_v2(
            state, replay, tools=None, tito_tokenizer=tito, message_matcher=loose_tool_call_message_matches
        )

        (call,) = tito.merge_calls
        assert call["old_messages"] == [USER, STORED_ASSISTANT]
        assert call["new_messages"][1] is STORED_ASSISTANT
        assert call["new_messages"][2] is TOOL_RESULT
        assert result == [0, 1, 99]


class TestRegistryOwnership:
    # [CLASS DOC] Tests of which message matcher the session registry holds.
    def test_defaults_to_the_strict_matcher(self):
        # [FUNCTION DOC] A registry built without a matcher holds the strict matcher function.
        registry = SessionRegistry(tokenizer=None, tito_tokenizer=_RecordingTITOTokenizer())

        assert registry.message_matcher is strict_message_matches

    def test_holds_the_injected_matcher(self):
        # [FUNCTION DOC] A registry built with the loose tool-call matcher holds exactly it.
        registry = SessionRegistry(
            tokenizer=None,
            tito_tokenizer=_RecordingTITOTokenizer(),
            message_matcher=loose_tool_call_message_matches,
        )

        assert registry.message_matcher is loose_tool_call_message_matches
