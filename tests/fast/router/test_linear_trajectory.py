# [MODULE DOC] Unit tests for the session registry and the linear trajectory (pretokenized state)
# logic, run without an HTTP server or real tokenizer. A stub token-in-token-out tokenizer renders
# first turns as a fixed one-token sentinel, adds no incremental tokens and merges by returning the
# stored ids unchanged; registries are built with a configurable set of roles allowed to be appended
# between assistant turns. Tests cover: session create / get / remove / append record; multi-turn
# update of messages and token ids; prefix mismatch detection; append-role capability checks per
# template; rollback to earlier assistant checkpoints (including to empty, bounded rollback steps,
# record truncation and injected assistants not counting as checkpoints); and the session mismatch
# computation against a comparator. Shared message constants (system, user, assistant, tool turns)
# are defined mid-file.
"""Unit tests for SessionRegistry and LinearTrajectory.

Tests the session registry CRUD and the trajectory pretokenized state management
logic in isolation (no HTTP server, no real tokenizer).
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.types import SessionRecord
from miles.utils.chat_template_utils.tito_tokenizer import ALL_APPEND_ROLES, FixedTemplate, TITOTokenizer


def _prepare_token_ids(session, request_messages, tools=None, *, tito_tokenizer, message_matcher=None) -> list[int]:
    # [FUNCTION DOC] HELPER: Wraps a messages-only request (plus tools when given) into client args
    # and asks the session to prepare its token ids and request args, using a default session server
    # config, the given tokenizer and optional message matcher. This performs any rollback and state
    # validation the session does as a side effect (and may raise its message validation errors).
    # Returns the input ids from the prepared request body.
    """Rollback + request args + render for a messages-only request; returns ``input_ids``."""
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


_MOCK_FIRST_TURN_TOKENS = [0]


class _MockTITOTokenizer(TITOTokenizer):
    # [CLASS DOC] HELPER: Stub token-in-token-out tokenizer. Merging returns a copy of the stored
    # token ids, additional-message tokenization returns nothing, rendering returns a copy of the
    # sentinel list, and no comparator is created. Real tokenizer operations are never used.
    """Stub for unit tests: returns pretokenized_token_ids unchanged (no
    incremental tokens), renders first-turn prompts as a fixed sentinel, and
    skips real tokenizer operations.
    """

    def create_comparator(self):
        # [FUNCTION DOC] HELPER: Returns none (tests install their own comparator when needed).
        return None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool = False,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Ignores all arguments and returns a fresh copy of the one-token
        # first-turn sentinel list.
        return list(_MOCK_FIRST_TURN_TOKENS)

    def tokenize_additional_messages(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Ignores all arguments and returns an empty list (no incremental
        # tokens for newly appended messages).
        return []

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Returns a copy of the pretokenized token ids, ignoring messages
        # and template args.
        return list(pretokenized_token_ids)


def _make_registry(allowed_append_roles: frozenset[str] = ALL_APPEND_ROLES) -> SessionRegistry:
    # [FUNCTION DOC] HELPER: Creates a subclass of the stub tokenizer whose fixed template allows
    # appending only the given roles (default all four), instantiates it with no real tokenizer and
    # the assistant start string "<|im_start|>assistant", and returns a session registry using it.
    configured_mock_type = type(
        "_ConfiguredMockTITOTokenizer",
        (_MockTITOTokenizer,),
        {"FIXED_TEMPLATE": FixedTemplate(allowed_append_roles=allowed_append_roles)},
    )
    mock_tito = configured_mock_type(tokenizer=None, assistant_start_str="<|im_start|>assistant")
    return SessionRegistry(tokenizer=None, tito_tokenizer=mock_tito)


@pytest.fixture
def registry():
    # [FUNCTION DOC] HELPER: Fixture: registry whose template allows all append roles.
    """Default FixedTemplate supports the maximal four-role surface."""
    return _make_registry()


@pytest.fixture
def registry_tool_only():
    # [FUNCTION DOC] HELPER: Fixture: registry whose template allows appending only tool messages.
    """Registry whose test FixedTemplate is restricted to tool messages."""
    return _make_registry(frozenset({"tool"}))


@pytest.fixture
def registry_with_system():
    # [FUNCTION DOC] HELPER: Fixture: registry allowing appended tool and system messages.
    """Registry whose test FixedTemplate supports tool and system."""
    return _make_registry(frozenset({"tool", "system"}))


@pytest.fixture
def registry_with_user():
    # [FUNCTION DOC] HELPER: Fixture: registry allowing appended tool and user messages.
    """Registry whose test FixedTemplate supports tool and user."""
    return _make_registry(frozenset({"tool", "user"}))


@pytest.fixture
def registry_with_assistant():
    # [FUNCTION DOC] HELPER: Fixture: registry allowing appended tool, user and assistant messages.
    """Registry whose test FixedTemplate supports injected assistant input."""
    return _make_registry(frozenset({"tool", "user", "assistant"}))


class TestSessionCRUD:
    # [CLASS DOC] Tests of creating, fetching, removing and appending records to sessions.
    def test_create_session(self, registry: SessionRegistry):
        # [FUNCTION DOC] A created session id is a non-null 32-character string registered in the
        # registry's sessions mapping.
        session_id = registry.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in registry.sessions

    def test_get_session(self, registry: SessionRegistry):
        # [FUNCTION DOC] A fetched new session has an empty record list.
        session_id = registry.create_session()
        session = registry.get_session(session_id)
        assert session.records == []

    def test_get_session_not_found(self, registry: SessionRegistry):
        # [FUNCTION DOC] Fetching an unknown session id raises a session-not-found error.
        with pytest.raises(SessionNotFoundError):
            registry.get_session("nonexistent")

    def test_remove_session(self, registry: SessionRegistry):
        # [FUNCTION DOC] Removing a session deletes it from the sessions mapping; removing it again
        # raises a session-not-found error.
        session_id = registry.create_session()
        registry.remove_session(session_id)  # no raise = success
        assert session_id not in registry.sessions
        with pytest.raises(SessionNotFoundError):
            registry.remove_session(session_id)

    def test_append_record(self, registry: SessionRegistry):
        # [FUNCTION DOC] Appending a session record to a fetched session makes the record list have
        # length 1 with the same path.
        session_id = registry.create_session()
        record = SessionRecord(
            timestamp=0.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": [{"role": "user", "content": "hello"}]},
            response={"choices": []},
        )

        session = registry.get_session(session_id)
        session.append_record(record)

        assert len(session.records) == 1
        assert session.records[0].path == record.path

    def test_append_record_missing_session(self, registry: SessionRegistry):
        # [FUNCTION DOC] Fetching the session "missing" raises a session-not-found error.
        # KNOWN ISSUE: despite its name, the test never appends a record; it only fetches a missing
        # session, duplicating the get-session-not-found test.
        with pytest.raises(SessionNotFoundError):
            registry.get_session("missing")


# ---------------------------------------------------------------------------
# Messages for multi-turn pretokenized tests
# ---------------------------------------------------------------------------

SYS_MSG = {"role": "system", "content": "You are a helpful assistant."}
USER_MSG = {"role": "user", "content": "What's the weather in Beijing?"}
ASSISTANT_MSG_1 = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}
    ],
}
TOOL_MSG_1 = {"role": "tool", "content": '{"temperature": 25}', "tool_call_id": "call_1"}
ASSISTANT_MSG_2 = {
    "role": "assistant",
    "content": "It's 25\u00b0C in Beijing. Let me also check Shanghai.",
    "tool_calls": [
        {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Shanghai"}'}}
    ],
}
TOOL_MSG_2 = {"role": "tool", "content": '{"temperature": 30}', "tool_call_id": "call_2"}
ASSISTANT_MSG_FINAL = {"role": "assistant", "content": "Beijing is 25\u00b0C and Shanghai is 30\u00b0C."}
RETRY_SYS_MSG = {"role": "system", "content": "Please try using the tools to answer."}


class TestSingleUserTurnPretokenized:
    # [CLASS DOC] Tests of the session's pretokenized state across turns: preparing request token
    # ids from stored state, and committing a finished turn (messages plus prompt and completion
    # ids). The stub tokenizer returns stored ids unchanged on later turns, so the prepared ids equal
    # the stored token ids.
    """Test prepare_token_ids_and_request_args and update_pretokenized_state across turns."""

    def test_first_turn_renders_from_scratch(self, registry: SessionRegistry):
        # [FUNCTION DOC] With nothing stored, preparing the ids for [system, user] yields the
        # first-turn sentinel list.
        """First turn has no prior token_ids, so prepare renders from scratch."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        messages = [SYS_MSG, USER_MSG]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry.tito_tokenizer)
        assert result == _MOCK_FIRST_TURN_TOKENS

    def test_two_turn_trajectory(self, registry: SessionRegistry):
        # [FUNCTION DOC] Turn 1 prepares the sentinel ids, then commits an assistant tool-call turn
        # with prompt ids 1-5 and completion ids 10-12: stored messages are [system, user, assistant]
        # and token ids are the concatenation. Turn 2 (tool result appended) prepares exactly the
        # stored ids; committing the final answer extends messages to five and token ids with the
        # new prompt tail (20, 21) and completion (30-32).
        """Full 2-turn: user -> assistant(tool_call) -> tool -> final answer."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # --- Turn 1: [sys, user] -> assistant with tool_call ---
        turn1_messages = [SYS_MSG, USER_MSG]
        assert (
            _prepare_token_ids(session, turn1_messages, tito_tokenizer=registry.tito_tokenizer)
            == _MOCK_FIRST_TURN_TOKENS
        )

        turn1_prompt_ids = [1, 2, 3, 4, 5]
        turn1_completion_ids = [10, 11, 12]
        session.update_pretokenized_state(
            turn1_messages, ASSISTANT_MSG_1, turn1_prompt_ids, turn1_completion_ids, max_trim_tokens=0
        )

        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12]

        # --- Turn 2: [sys, user, assistant, tool] -> final answer ---
        turn2_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, turn2_messages, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 4, 5, 10, 11, 12]

        turn2_prompt_ids = [1, 2, 3, 4, 5, 10, 11, 12, 20, 21]
        turn2_completion_ids = [30, 31, 32]
        session.update_pretokenized_state(
            turn2_messages, ASSISTANT_MSG_FINAL, turn2_prompt_ids, turn2_completion_ids, max_trim_tokens=0
        )

        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_FINAL]
        assert session.token_ids == [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 32]

    def test_three_turn_trajectory(self, registry: SessionRegistry):
        # [FUNCTION DOC] Commits three turns (two tool-call turns and a final answer), each time
        # first preparing the ids, which must equal the previously stored token ids. Ends with seven
        # stored messages and the full concatenated token ids.
        """Full 3-turn: user -> ass(tool) -> tool -> ass(tool) -> tool -> final."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11]

        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0
        )

        # Turn 3
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = _prepare_token_ids(session, t3_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11, 20, 21, 30, 31]

        session.update_pretokenized_state(
            t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert len(session.messages) == 7  # sys, user, ass1, tool1, ass2, tool2, final
        assert session.token_ids == [1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 50, 51]

    def test_prefix_mismatch_raises(self, registry: SessionRegistry):
        # [FUNCTION DOC] Committing a second turn whose prompt ids do not start with the stored token
        # ids raises a tokenization error mentioning a pretokenized prefix mismatch.
        """update_pretokenized_state asserts stored token_ids is prefix of new."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        with pytest.raises(TokenizationError, match="pretokenized prefix mismatch"):
            session.update_pretokenized_state(
                [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1],
                ASSISTANT_MSG_FINAL,
                [9, 9, 9, 20, 21],  # does NOT start with [1,2,3,10,11]
                [30],
                max_trim_tokens=0,
            )

    def test_session_not_found_raises(self, registry: SessionRegistry):
        # [FUNCTION DOC] Fetching an unknown session raises a session-not-found error whose message
        # says the session was not found. KNOWN ISSUE: duplicates the registry-level not-found test.
        with pytest.raises(SessionNotFoundError, match="session not found"):
            registry.get_session("nonexistent")

    def test_no_system_message(self, registry: SessionRegistry):
        # [FUNCTION DOC] A trajectory starting directly with a user message commits and the next
        # turn (assistant, tool) prepares the stored ids [1, 2, 10].
        """Works without system message (system is optional)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        msgs = [USER_MSG]
        session.update_pretokenized_state(msgs, ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2_msgs = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 10]

    def test_multiple_system_messages_at_start(self, registry: SessionRegistry):
        # [FUNCTION DOC] Two system messages before the user message: the first turn prepares the
        # sentinel, committing stores all four messages, and the next turn prepares the stored ids.
        """Multiple system messages before the user message are allowed (part of stored prefix)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        extra_sys = {"role": "system", "content": "Extra instructions."}
        msgs = [SYS_MSG, extra_sys, USER_MSG]
        result = _prepare_token_ids(session, msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == _MOCK_FIRST_TURN_TOKENS  # first turn, no prior tokens

        session.update_pretokenized_state(msgs, ASSISTANT_MSG_1, [1, 2, 3, 4], [10, 11], max_trim_tokens=0)
        assert session.messages == [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1]

        t2_msgs = [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 4, 10, 11]


# ---------------------------------------------------------------------------
# TestAppendRole* — FixedTemplate capability tests
#
# Each class exercises an actual append behavior against either the default
# four-role template capability or a synthetic restricted template.
# ---------------------------------------------------------------------------


class TestAppendRoleDefault:
    # [CLASS DOC] Tests that the all-roles registry accepts appended user and assistant messages.
    """The default FixedTemplate supports all four roles."""

    def test_default_template_allows_user_append(self, registry: SessionRegistry):
        # [FUNCTION DOC] After one committed turn, a request appending a tool and an extra user
        # message prepares ids without error (the result is a list).
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

    def test_default_template_allows_assistant_append(self, registry: SessionRegistry):
        # [FUNCTION DOC] After one committed turn, a request appending a tool and an injected
        # assistant message prepares ids without error.
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "assistant", "content": "injected"}]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)


class TestAppendRoleToolOnly:
    # [CLASS DOC] Tests of a registry whose template permits only appended tool messages.
    """A template explicitly restricted to tool appends."""

    def test_tool_append_allowed(self, registry_tool_only: SessionRegistry):
        # [FUNCTION DOC] Appending a tool message after a committed turn prepares ids without error.
        sid = registry_tool_only.create_session()
        session = registry_tool_only.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry_tool_only.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_rejected(self, registry_tool_only: SessionRegistry):
        # [FUNCTION DOC] Appending a system message after the tool message raises a message
        # validation error naming role system and the allowed roles.
        sid = registry_tool_only.create_session()
        session = registry_tool_only.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            _prepare_token_ids(session, messages, tito_tokenizer=registry_tool_only.tito_tokenizer)

    def test_user_append_rejected(self, registry_tool_only: SessionRegistry):
        # [FUNCTION DOC] Appending a user message after the tool message raises a message validation
        # error naming role user and the allowed roles.
        sid = registry_tool_only.create_session()
        session = registry_tool_only.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            _prepare_token_ids(session, messages, tito_tokenizer=registry_tool_only.tito_tokenizer)


class TestAppendRoleAssistant:
    # [CLASS DOC] Tests of a registry whose template permits tool, user and injected assistant
    # appends.
    """A template supporting tool, user, and injected assistant appends.

    Injected assistant input joins the prompt region of the next sample (the
    loss mask only ever covers generated response tokens)."""

    def test_assistant_append_allowed(self, registry_with_assistant: SessionRegistry):
        # [FUNCTION DOC] Appending a tool, an injected assistant and a user message after a committed
        # turn prepares ids without error.
        sid = registry_with_assistant.create_session()
        session = registry_with_assistant.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [
            SYS_MSG,
            USER_MSG,
            ASSISTANT_MSG_1,
            TOOL_MSG_1,
            {"role": "assistant", "content": "injected"},
            {"role": "user", "content": "next"},
        ]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry_with_assistant.tito_tokenizer)
        assert isinstance(result, list)


class TestAppendRoleToolSystem:
    # [CLASS DOC] Tests of a registry whose template permits tool and system appends.
    """A template supporting tool and system appends."""

    def test_tool_append_allowed(self, registry_with_system: SessionRegistry):
        # [FUNCTION DOC] Appending a tool message prepares ids without error.
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_allowed(self, registry_with_system: SessionRegistry):
        # [FUNCTION DOC] Appending a tool and a retry system message prepares exactly the stored ids
        # [1, 2, 3, 10, 11].
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11]

    def test_system_then_assistant_trajectory(self, registry_with_system: SessionRegistry):
        # [FUNCTION DOC] A three-turn flow where turn 2 appends tool plus retry system message and
        # turn 3 appends another tool: each preparation returns a list, and after committing turn 2
        # the stored messages include the retry system message followed by the second assistant.
        """Full trajectory with a retry system message between tool-call turns."""
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)

        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0
        )
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2]

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = _prepare_token_ids(session, t3_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_append_rejected(self, registry_with_system: SessionRegistry):
        # [FUNCTION DOC] Appending a user message raises a message validation error naming role user
        # and the allowed roles.
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            _prepare_token_ids(session, messages, tito_tokenizer=registry_with_system.tito_tokenizer)


class TestAppendRoleToolUser:
    # [CLASS DOC] Tests of a registry whose template permits tool and user appends.
    """A template supporting tool and user appends."""

    def test_tool_append_allowed(self, registry_with_user: SessionRegistry):
        # [FUNCTION DOC] Appending a tool message prepares ids without error.
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_append_allowed(self, registry_with_user: SessionRegistry):
        # [FUNCTION DOC] Appending a tool and a follow-up user message prepares ids without error.
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "follow-up"}]
        result = _prepare_token_ids(session, messages, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_then_assistant_trajectory(self, registry_with_user: SessionRegistry):
        # [FUNCTION DOC] Three turns: turn 2 appends tool plus a user follow-up, commits the second
        # assistant (stored messages end with follow-up then assistant), and turn 3 appends another
        # tool; each preparation returns a list.
        """Full trajectory: tool → user follow-up → assistant → tool → final."""
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)

        # Turn 1: [sys, user] -> assistant(tool_call)
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: append tool + user follow-up -> assistant(tool_call)
        follow_up = {"role": "user", "content": "Also check Shanghai."}
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up]
        result = _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0
        )
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up, ASSISTANT_MSG_2]

        # Turn 3: append tool after the second assistant
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = _prepare_token_ids(session, t3_msgs, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_rejected(self, registry_with_user: SessionRegistry):
        # [FUNCTION DOC] Appending a system message raises a message validation error naming role
        # system and the allowed roles.
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            _prepare_token_ids(session, messages, tito_tokenizer=registry_with_user.tito_tokenizer)


class TestRollback:
    # [CLASS DOC] Tests of rolling the session back to an earlier generated-assistant checkpoint
    # when a request diverges from the stored messages. Rollback is bounded to a maximum number of
    # discarded assistant checkpoints (one step in these tests); exceeding it raises a message
    # validation error and leaves the state untouched.
    """Tests for session rollback to a previous assistant checkpoint."""

    def test_rollback_to_first_assistant(self, registry: SessionRegistry):
        # [FUNCTION DOC] After two committed turns, a request that keeps the first assistant but
        # replaces the tool message rolls back to checkpoint 0: one assistant, one trajectory entry,
        # token ids [1, 2, 3, 10, 11], messages [system, user, assistant 1].
        """After 2 completions, rolling back to the first assistant checkpoint works."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1: [sys, user] -> assistant1
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> assistant2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0
        )

        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2

        # Rollback: send [sys, user, asst1, NEW_tool] - diverges after asst1
        new_tool = {"role": "tool", "content": '{"temperature": 99}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        result = _prepare_token_ids(session, rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # State should be rolled back to checkpoint 0
        assert session.num_assistant == 1
        assert len(session.trajectory_token_ids) == 1
        assert session.token_ids == [1, 2, 3, 10, 11]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_multi_step_rollback_raises(self, registry: SessionRegistry):
        # [FUNCTION DOC] After three committed turns, a request diverging after the first assistant
        # would discard two assistants and raises a message validation error mentioning the maximum
        # rollback steps; messages, trajectory token ids, records and the assistant count are
        # unchanged from a snapshot taken before.
        """Rollback that discards >1 assistant raises MessageValidationError and leaves state unchanged."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(
            t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0
        )

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        _prepare_token_ids(session, t3_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(
            t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert session.num_assistant == 3

        # Snapshot state before attempted rollback
        prev_messages = list(session.messages)
        prev_token_ids = list(session.trajectory_token_ids)
        prev_records = list(session.records)
        prev_num_assistant = session.num_assistant

        # Attempt rollback to checkpoint 0 (discard 2 assistants) — should fail
        new_tool = {"role": "tool", "content": '{"alt": true}', "tool_call_id": "call_1"}
        with pytest.raises(MessageValidationError, match="exceeds max_assistant_rollback_steps"):
            _prepare_token_ids(
                session, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool], tito_tokenizer=registry.tito_tokenizer
            )

        # State must be unchanged
        assert session.messages == prev_messages
        assert session.trajectory_token_ids == prev_token_ids
        assert session.records == prev_records
        assert session.num_assistant == prev_num_assistant

    def test_rollback_then_continue_full_trajectory(self, registry: SessionRegistry):
        # [FUNCTION DOC] After rolling back to the first assistant with a different tool message, a
        # new turn (prompt ids [1, 2, 3, 10, 11, 40, 41], completion [50, 51]) commits cleanly: two
        # assistants, two trajectory entries, the new token ids, and messages ending with the new
        # tool and final assistant.
        """Rollback and then complete a full new trajectory from the checkpoint."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        session.update_pretokenized_state(t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20], [30], max_trim_tokens=0)

        # Rollback to asst1, send different tool
        new_tool = {"role": "tool", "content": '{"retry": true}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        result = _prepare_token_ids(session, rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # Continue: complete a new turn from the rolled-back state
        session.update_pretokenized_state(
            rollback_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 40, 41], [50, 51], max_trim_tokens=0
        )

        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2
        assert session.token_ids == [1, 2, 3, 10, 11, 40, 41, 50, 51]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool, ASSISTANT_MSG_FINAL]

    def test_rollback_fewer_messages_than_stored(self, registry_with_system: SessionRegistry):
        # [FUNCTION DOC] Stored state has five messages; a retry request with only four messages
        # (ending in a retry system message) still triggers rollback to the first assistant:
        # assistant count 1 and messages [system, user, assistant 1].
        """Rollback triggered when request has strictly fewer messages than stored."""
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)

        # Turn 1: [sys, user] -> asst1
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> asst2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        session.update_pretokenized_state(t2_msgs, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        # stored messages: [sys, user, asst1, tool1, asst2] (5 messages)

        # Agent retries with only [sys, user, asst1, sys_retry] (4 messages)
        retry_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, RETRY_SYS_MSG]
        result = _prepare_token_ids(session, retry_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

        assert session.num_assistant == 1
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_rollback_to_second_assistant(self, registry: SessionRegistry):
        # [FUNCTION DOC] After three committed turns, a request keeping the first two assistants but
        # changing the second tool message rolls back one step to the second checkpoint: two
        # assistants, two trajectory entries, token ids [1, 2, 10, 20, 30], and the five matching
        # messages.
        """Rollback to the second checkpoint (skipping the third)."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # 3 completions
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare_token_ids(session, t2, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)

        t3 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        _prepare_token_ids(session, t3, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t3, ASSISTANT_MSG_FINAL, [1, 2, 10, 20, 30, 40], [50], max_trim_tokens=0)

        assert session.num_assistant == 3

        # Rollback: keep up to asst2, diverge at tool2
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_2"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, new_tool]
        result = _prepare_token_ids(session, rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        assert session.num_assistant == 2
        assert len(session.trajectory_token_ids) == 2
        assert session.token_ids == [1, 2, 10, 20, 30]
        assert session.messages == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2]

    def test_no_rollback_when_append_only(self, registry: SessionRegistry):
        # [FUNCTION DOC] A request that only appends a tool message leaves the state untouched: one
        # assistant, one trajectory entry, token ids [1, 2, 10].
        """Normal append-only flow does not trigger rollback."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Append tool - not a rollback
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare_token_ids(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # State should NOT have been rolled back
        assert session.num_assistant == 1
        assert len(session.trajectory_token_ids) == 1
        assert session.token_ids == [1, 2, 10]

    def test_rollback_no_assistant_in_prefix_resets_session(self, registry: SessionRegistry):
        # [FUNCTION DOC] A request diverging at the user message (only the system message matches,
        # no assistant in the common prefix) resets the session to empty: the prepared ids are the
        # first-turn sentinel, and messages, trajectory ids, records and assistant count are empty or
        # zero.
        """No assistant in the matched prefix rolls back to the empty checkpoint."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Diverge at user message (index 1) - only sys matched, no assistant
        new_msgs = [SYS_MSG, {"role": "user", "content": "different question"}]
        result = _prepare_token_ids(session, new_msgs, tito_tokenizer=registry.tito_tokenizer)

        assert result == _MOCK_FIRST_TURN_TOKENS
        assert session.messages == []
        assert session.trajectory_token_ids == []
        assert session.records == []
        assert session.num_assistant == 0

    def test_rollback_regenerates_verbatim_first_turn(self, registry: SessionRegistry):
        # [FUNCTION DOC] After committing a first turn, resending the identical single-message
        # request resets the session to empty and prepares the first-turn sentinel (no error). A new
        # completion then commits on top of the reset state: one assistant, token ids [1, 2, 99], and
        # messages [user, final assistant].
        """Re-sending the first turn verbatim regenerates it instead of failing.

        This is the shape a retrying HTTP client produces: the completion was
        already stored, then the byte-identical single-message request arrives
        again, so stored continues past the request with no assistant to
        roll back to.
        """
        sid = registry.create_session()
        session = registry.get_session(sid)

        turn1 = [USER_MSG]
        assert _prepare_token_ids(session, turn1, tito_tokenizer=registry.tito_tokenizer) == _MOCK_FIRST_TURN_TOKENS
        session.update_pretokenized_state(turn1, ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        assert session.num_assistant == 1

        result = _prepare_token_ids(session, turn1, tito_tokenizer=registry.tito_tokenizer)

        assert result == _MOCK_FIRST_TURN_TOKENS
        assert session.messages == []
        assert session.trajectory_token_ids == []
        assert session.num_assistant == 0

        # The regenerated turn commits cleanly on top of the reset state.
        session.update_pretokenized_state(turn1, ASSISTANT_MSG_FINAL, [1, 2], [99], max_trim_tokens=0)
        assert session.num_assistant == 1
        assert session.token_ids == [1, 2, 99]
        assert session.messages == [USER_MSG, ASSISTANT_MSG_FINAL]

    def test_rollback_to_empty_beyond_one_assistant_raises(self, registry: SessionRegistry):
        # [FUNCTION DOC] With two committed assistants, resending the first request (which would
        # discard both) raises a message validation error mentioning the maximum rollback steps; the
        # assistant count stays 2 and token ids stay [1, 2, 10, 20, 30].
        """Resetting to empty is still bounded by MAX_ASSISTANT_ROLLBACK_STEPS."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        turn1 = [USER_MSG]
        session.update_pretokenized_state(turn1, ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        turn2 = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare_token_ids(session, turn2, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(turn2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        assert session.num_assistant == 2

        # Discarding both assistants exceeds the single-step budget.
        with pytest.raises(MessageValidationError, match="exceeds max_assistant_rollback_steps"):
            _prepare_token_ids(session, turn1, tito_tokenizer=registry.tito_tokenizer)

        assert session.num_assistant == 2
        assert session.token_ids == [1, 2, 10, 20, 30]

    def test_rollback_records_truncated(self, registry: SessionRegistry):
        # [FUNCTION DOC] Two turns each with an appended record (timestamps 1.0 and 2.0); after a
        # rollback to checkpoint 0 only the first record remains.
        """Records are truncated in sync with trajectory_token_ids on rollback."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        r1 = SessionRecord(
            timestamp=1.0, method="POST", path="/v1/chat/completions", status_code=200, request={}, response={}
        )
        session.append_record(r1)

        # Turn 2
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare_token_ids(session, t2, tito_tokenizer=registry.tito_tokenizer)
        session.update_pretokenized_state(t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        r2 = SessionRecord(
            timestamp=2.0, method="POST", path="/v1/chat/completions", status_code=200, request={}, response={}
        )
        session.append_record(r2)

        assert len(session.records) == 2

        # Rollback to checkpoint 0
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        _prepare_token_ids(
            session, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool], tito_tokenizer=registry.tito_tokenizer
        )

        assert len(session.records) == 1
        assert session.records[0].timestamp == 1.0

    def test_injected_assistants_are_not_generated_checkpoints(self, registry_with_assistant: SessionRegistry):
        # [FUNCTION DOC] Scenario with two injected assistant messages and a user follow-up between
        # generated turns. After committing turn 1 and turn 2 (with records), stored messages are the
        # injected request plus assistant 2, trajectory ids hold two entries, generated checkpoint
        # message ends are [3, 7] and the assistant count is 2 (injected assistants do not count).
        # Resending the exact turn-2 request rolls back exactly one step (the absent generated
        # response): messages, trajectory, records and checkpoint ends return to the turn-1 state
        # (ends [3], count 1). A regenerated assistant then commits again, giving ends [3, 7], count
        # 2, token ids equal to the second prompt plus the new completion, and records of turn 1 and
        # the regenerated turn.
        """Only backend-generated responses create checkpoints: injected
        assistants are prompt history, and rollback counts generated
        checkpoints, never assistant roles."""
        sid = registry_with_assistant.create_session()
        session = registry_with_assistant.get_session(sid)

        first_request = [SYS_MSG, USER_MSG]
        first_tokens = [1, 2, 3, 10, 11]
        session.update_pretokenized_state(
            first_request,
            ASSISTANT_MSG_1,
            prompt_token_ids=[1, 2, 3],
            completion_token_ids=[10, 11],
            max_trim_tokens=0,
        )
        first_record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": first_request},
            response={"message": ASSISTANT_MSG_1},
        )
        session.append_record(first_record)

        injected_assistant_1 = {"role": "assistant", "content": "Injected context one."}
        injected_assistant_2 = {"role": "assistant", "content": "Injected context two."}
        follow_up = {"role": "user", "content": "Use both injected facts."}
        injected_request = [
            SYS_MSG,
            USER_MSG,
            ASSISTANT_MSG_1,
            injected_assistant_1,
            injected_assistant_2,
            follow_up,
        ]
        second_prompt_tokens = first_tokens + [20, 21, 22]
        _prepare_token_ids(
            session,
            injected_request,
            tito_tokenizer=registry_with_assistant.tito_tokenizer,
        )
        session.update_pretokenized_state(
            injected_request,
            ASSISTANT_MSG_2,
            prompt_token_ids=second_prompt_tokens,
            completion_token_ids=[30, 31],
            max_trim_tokens=0,
        )
        second_record = SessionRecord(
            timestamp=2.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": injected_request},
            response={"message": ASSISTANT_MSG_2},
        )
        session.append_record(second_record)

        assert session.messages == injected_request + [ASSISTANT_MSG_2]
        assert session.trajectory_token_ids == [first_tokens, second_prompt_tokens + [30, 31]]
        assert session.records == [first_record, second_record]
        assert session.generated_checkpoint_message_ends == [3, 7]
        assert session.num_assistant == 2

        # Resend the exact request that produced assistant2. The absent generated
        # response is one rollback step; the two injected assistants are prompt
        # history, not additional checkpoints.
        _prepare_token_ids(
            session,
            injected_request,
            tito_tokenizer=registry_with_assistant.tito_tokenizer,
        )

        assert session.messages == first_request + [ASSISTANT_MSG_1]
        assert session.trajectory_token_ids == [first_tokens]
        assert session.records == [first_record]
        assert session.generated_checkpoint_message_ends == [3]
        assert session.num_assistant == 1

        regenerated_assistant = {"role": "assistant", "content": "Regenerated response."}
        regenerated_tokens = second_prompt_tokens + [40, 41]
        session.update_pretokenized_state(
            injected_request,
            regenerated_assistant,
            prompt_token_ids=second_prompt_tokens,
            completion_token_ids=[40, 41],
            max_trim_tokens=0,
        )
        regenerated_record = SessionRecord(
            timestamp=3.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": injected_request},
            response={"message": regenerated_assistant},
        )
        session.append_record(regenerated_record)

        assert session.messages == injected_request + [regenerated_assistant]
        assert session.trajectory_token_ids == [first_tokens, regenerated_tokens]
        assert session.token_ids == regenerated_tokens
        assert session.records == [first_record, regenerated_record]
        assert session.generated_checkpoint_message_ends == [3, 7]
        assert session.num_assistant == 2


class TestUpdatePretokenizedStateMissingSession:
    # [CLASS DOC] Intended to test failure on unknown sessions.
    """update_pretokenized_state raises SessionNotFoundError for unknown session."""

    def test_raises_on_missing_session(self, registry: SessionRegistry):
        # [FUNCTION DOC] Fetching an unknown session raises a session-not-found error whose message
        # says the session was not found.
        # KNOWN ISSUE: the class docstring says it tests the state update on an unknown session, but
        # the test only fetches the session from the registry and never calls the update.
        with pytest.raises(SessionNotFoundError, match="session not found"):
            registry.get_session("nonexistent")


class TestComputeSessionMismatch:
    # [CLASS DOC] Tests of the registry's session mismatch computation: it renders the stored
    # messages with the chat template (no generation prompt, tokenized, using the tip turn's recorded
    # template args), compares the rendering against the stored token ids with the registry's
    # comparator, and returns the mismatches as plain mappings. Tests replace the template renderer
    # and comparator with mocks.
    """Tests for compute_session_mismatch."""

    def test_raises_for_missing_session(self, registry: SessionRegistry):
        # [FUNCTION DOC] Fetching an unknown session raises a session-not-found error.
        # KNOWN ISSUE: never calls the mismatch computation; it only fetches a missing session.
        with pytest.raises(SessionNotFoundError):
            registry.get_session("nonexistent")

    def test_returns_none_for_empty_token_ids(self, registry: SessionRegistry):
        # [FUNCTION DOC] A fresh session with no stored token ids yields none.
        sid = registry.create_session()
        session = registry.get_session(sid)
        assert registry.compute_session_mismatch(session) is None

    def test_returns_empty_list_when_no_mismatch(self, registry: SessionRegistry):
        # [FUNCTION DOC] With one committed turn (ids [1, 2, 3, 10, 11]), a mocked renderer returning
        # the same ids and a mocked comparator returning no mismatches, the result is an empty list.
        # The comparator is called once with (rendered ids, stored ids), and the renderer once with
        # the stored messages, no generation prompt, tokenize on, and empty template args.
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Simulate: template returns same IDs as stored
        registry.tito_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3, 10, 11])

        # Need a real comparator; replace the None one
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        registry.comparator = mock_comparator

        result = registry.compute_session_mismatch(session)
        assert result == []
        mock_comparator.compare_sequences.assert_called_once_with([1, 2, 3, 10, 11], [1, 2, 3, 10, 11])
        registry.tito_tokenizer.apply_chat_template.assert_called_once_with(
            session.messages,
            add_generation_prompt=False,
            tokenize=True,
            template_args={},
        )

    def test_returns_mismatch_dicts(self, registry: SessionRegistry):
        # [FUNCTION DOC] With a mocked comparator returning one fake mismatch object (position 2),
        # the result is a list of its dictionary form with position 2 and detail "mismatch".
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        registry.tito_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 99, 10, 11])

        @dataclass
        class FakeMismatch:
            # [CLASS DOC] HELPER: Fake mismatch record with an integer position.
            position: int

            def to_dict(self):
                # [FUNCTION DOC] HELPER: Returns a mapping with the position and detail "mismatch".
                return {"position": self.position, "detail": "mismatch"}

        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = [FakeMismatch(position=2)]
        registry.comparator = mock_comparator

        result = registry.compute_session_mismatch(session)
        assert result == [{"position": 2, "detail": "mismatch"}]

    def test_raises_tokenization_error_on_exception(self, registry: SessionRegistry):
        # [FUNCTION DOC] When the renderer raises a runtime error ("tokenizer failed"), the
        # computation raises a tokenization error containing that message.
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        registry.tito_tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("tokenizer failed"))

        with pytest.raises(TokenizationError, match="tokenizer failed"):
            registry.compute_session_mismatch(session)

    def test_renders_with_the_tools_the_tip_recorded(self, registry: SessionRegistry):
        # [FUNCTION DOC] The committed turn recorded tools in its turn args, while an appended record
        # carries different tools. The renderer must receive the turn args' tools as template args
        # (not the record's), and generation prompt off.
        sid = registry.create_session()
        session = registry.get_session(sid)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        session.update_pretokenized_state(
            [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0, turn_args={"tools": tools}
        )
        # A record carrying other tools does not matter: the tip's template args do.
        record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"tools": [{"type": "function", "function": {"name": "get_time"}}]},
            response={},
        )
        session.append_record(record)

        mock_tokenize = MagicMock(return_value=[1, 2, 10])
        registry.tito_tokenizer.apply_chat_template = mock_tokenize
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        registry.comparator = mock_comparator

        registry.compute_session_mismatch(session)

        _, kwargs = mock_tokenize.call_args
        assert kwargs["template_args"] == {"tools": tools}
        assert kwargs["add_generation_prompt"] is False

    def test_renders_with_the_kwargs_the_tip_recorded(self, registry: SessionRegistry):
        # [FUNCTION DOC] The committed turn recorded a temperature and chat template kwargs
        # (reasoning effort low). The computation returns an empty list and the renderer receives
        # exactly the chat template kwargs mapping as template args.
        sid = registry.create_session()
        session = registry.get_session(sid)
        session.update_pretokenized_state(
            [SYS_MSG, USER_MSG],
            ASSISTANT_MSG_1,
            [1, 2, 3],
            [10, 11],
            max_trim_tokens=0,
            turn_args={"temperature": 0.7, "chat_template_kwargs": {"reasoning_effort": "low"}},
        )

        mock_tokenize = MagicMock(return_value=[1, 2, 3, 10, 11])
        registry.tito_tokenizer.apply_chat_template = mock_tokenize
        registry.comparator = MagicMock(compare_sequences=MagicMock(return_value=[]))

        assert registry.compute_session_mismatch(session) == []
        assert mock_tokenize.call_args.kwargs["template_args"] == {"reasoning_effort": "low"}


def test_committed_full_turn_args_are_isolated_from_later_request_mutation(registry):
    # [FUNCTION DOC] Commits a turn with a request args mapping (temperature, messages, tools,
    # template kwargs with a nested list, input ids), then mutates every part of the original
    # mapping in place (including nested values). The session's stored turn args must still equal
    # the original values, showing the commit made a deep copy.
    session = registry.get_session(registry.create_session())
    request_args = {
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"function": {"name": "f"}}],
        "chat_template_kwargs": {"nested": [1]},
        "input_ids": [1, 2],
    }
    session.update_pretokenized_state(
        [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [3], max_trim_tokens=0, turn_args=request_args
    )
    request_args["temperature"] = 0.1
    request_args["messages"][0]["content"] = "changed"
    request_args["tools"][0]["function"]["name"] = "changed"
    request_args["chat_template_kwargs"]["nested"].append(2)
    request_args["input_ids"].append(9)
    assert session.turn_args == {
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"function": {"name": "f"}}],
        "chat_template_kwargs": {"nested": [1]},
        "input_ids": [1, 2],
    }


def test_mismatch_does_not_resolve_defaults_for_an_empty_committed_record(registry):
    # [FUNCTION DOC] A turn committed with empty turn args, while the tokenizer has default chat
    # template kwargs (thinking enabled), computes a mismatch (mocked comparator, empty result) whose
    # renderer call still receives empty template args: the tokenizer defaults are not merged in.
    session = registry.get_session(registry.create_session())
    session.update_pretokenized_state([SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1], [2], max_trim_tokens=0, turn_args={})
    registry.tito_tokenizer.chat_template_kwargs = {"enable_thinking": True}
    registry.tito_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2])
    registry.comparator = MagicMock(compare_sequences=MagicMock(return_value=[]))
    assert registry.compute_session_mismatch(session) == []
    assert registry.tito_tokenizer.apply_chat_template.call_args.kwargs["template_args"] == {}
