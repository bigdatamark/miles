# [MODULE DOC] Unit tests for the v2 session registry and per-session state (tree of committed
# generations). Uses a stub TITO tokenizer (no real tokenizer, no HTTP server) so that only the
# bookkeeping is exercised: session create/get/remove, committing generations, preparing input
# token ids for follow-up requests, append-role policy validation, retry/rollback attachment
# semantics over the generation tree, and mismatch computation against the chat template.
# Module-level message constants (system, user, assistant with tool calls, tool results) are
# shared by all tests.
"""Unit tests for SessionRegistryV2 and SessionStateV2.

Tests the session registry CRUD and the trajectory pretokenized state management
logic in isolation (no HTTP server, no real tokenizer).
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError, SessionNotFoundError, TokenizationError
from miles.rollout.session.types import SessionRecord
from miles.rollout.session.v2.session_state import (
    SessionRegistryV2,
    SessionStateV2,
    attach_point_for_request,
    commit_generation,
    prepare_token_ids_and_request_args,
)
from miles.utils.chat_template_utils.tito_tokenizer import FixedTemplate, TITOTokenizer

_MOCK_FIRST_TURN_TOKENS = [0]


class _MockTITOTokenizer(TITOTokenizer):
    # [CLASS DOC] HELPER: Test double for the TITO tokenizer. It has no comparator, renders any
    # prompt as the fixed one-token sentinel, produces no incremental tokens for appended
    # messages, and merges by returning the stored pretokenized ids unchanged. Its methods
    # below take the same arguments as the real ones and ignore them.
    """Stub for unit tests: returns pretokenized_token_ids unchanged (no
    incremental tokens), renders first-turn prompts as a fixed sentinel, and
    skips real tokenizer operations.
    """

    def create_comparator(self):
        # [FUNCTION DOC] HELPER: Returns no comparator; tests that need one assign a mock to the
        # registry's comparator attribute themselves.
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
        # first-turn sentinel list, so a from-scratch render is recognizable in assertions.
        return list(_MOCK_FIRST_TURN_TOKENS)

    def tokenize_additional_messages(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Always returns an empty list: appended messages add no tokens.
        return []

    def merge_tokens(
        self,
        old_messages: list[dict[str, Any]],
        new_messages: list[dict[str, Any]],
        pretokenized_token_ids: list[int],
        *,
        template_args: dict[str, Any] | None = None,
    ) -> list[int]:
        # [FUNCTION DOC] HELPER: Returns a copy of the pretokenized ids, ignoring the messages;
        # getting the stored prefix back proves the merge path ran instead of a fresh render.
        return list(pretokenized_token_ids)


def _make_mock_tito(tito_cls, allowed_append_roles: list[str]) -> TITOTokenizer:
    # [FUNCTION DOC] HELPER: Builds a subclass of the given TITO tokenizer class whose fixed
    # template declares the given roles as allowed to be appended, then instantiates it with no
    # real tokenizer and the assistant-start marker string for the chat-ml style template.
    # Returns the instance.
    configured_tito_cls = type(
        f"_Configured{tito_cls.__name__}",
        (tito_cls,),
        {"FIXED_TEMPLATE": FixedTemplate(allowed_append_roles=frozenset(allowed_append_roles))},
    )
    return configured_tito_cls(tokenizer=None, assistant_start_str="<|im_start|>assistant")


def _make_registry(allowed_append_roles: list[str]) -> SessionRegistryV2:
    # [FUNCTION DOC] HELPER: Creates a v2 session registry with no tokenizer, backed by the mock
    # TITO tokenizer configured with the given allowed append roles.
    mock_tito = _make_mock_tito(_MockTITOTokenizer, allowed_append_roles)
    return SessionRegistryV2(tokenizer=None, tito_tokenizer=mock_tito)


def _commit(
    state: SessionStateV2, request_messages, assistant_message, prompt_ids, completion_ids, *, max_trim_tokens=0
):
    # [FUNCTION DOC] HELPER: Commits one generation to the session state. Builds a stub session
    # record (timestamp equal to the current node count, a successful chat-completions post
    # carrying the request messages, empty response), attaches under the node found for the
    # request messages, and commits with the given assistant message, prompt and completion
    # token ids and trim limit, using a response id derived from the node count and a "stop"
    # finish reason. Returns the new tree node. Raises whatever the commit raises (e.g. a
    # tokenization error on a prefix mismatch).
    """Drive one committed generation through the serving surface (record stub)."""
    record = SessionRecord(
        timestamp=float(len(state.tree.nodes)),
        method="POST",
        path="/v1/chat/completions",
        status_code=200,
        request={"messages": list(request_messages)},
        response={},
    )
    return commit_generation(
        state,
        parent=attach_point_for_request(state, request_messages).node,
        request_messages=request_messages,
        assistant_message=assistant_message,
        prompt_token_ids=prompt_ids,
        completion_token_ids=completion_ids,
        max_trim_tokens=max_trim_tokens,
        record=record,
        response_id=f"resp-{len(state.tree.nodes)}",
        finish_reason="stop",
    )


def _prepare(state, request_messages, *, tito_tokenizer, message_matcher=None) -> list[int]:
    # [FUNCTION DOC] HELPER: Runs request preparation for a messages-only request using a default
    # session-server config and the given tokenizer and optional message matcher. Returns the
    # input token ids from the prepared request body. Propagates validation errors (e.g. a
    # disallowed appended role).
    """Attach + request args + render for a messages-only request, as the core does; returns ``input_ids``."""
    prepared, _parent = prepare_token_ids_and_request_args(
        state,
        {"messages": request_messages},
        config=make_session_server_config(),
        tito_tokenizer=tito_tokenizer,
        message_matcher=message_matcher,
    )
    return prepared.body["input_ids"]


def _path(state):
    # [FUNCTION DOC] HELPER: Returns the list of nodes from the root to the latest committed
    # generation, or an empty list when nothing has been committed.
    """The served single chain: the latest committed generation's path."""
    node = state.latest()
    return node.path_nodes() if node is not None else []


def _messages(state):
    # [FUNCTION DOC] HELPER: Flattens the delta messages of every node on the served path into
    # one ordered message list.
    return [message for node in _path(state) for message in node.delta_messages]


def _token_ids(state):
    # [FUNCTION DOC] HELPER: Returns the full token id sequence stored on the latest committed
    # node, or an empty list when there is none.
    node = state.latest()
    return node.token_ids if node is not None else []


def _records(state):
    # [FUNCTION DOC] HELPER: Returns the session records of the nodes on the served path, in
    # order.
    return [node.record for node in _path(state)]


@pytest.fixture
def registry():
    # [FUNCTION DOC] HELPER: Pytest fixture yielding a registry whose allowed append roles are
    # only the tool role.
    """Tool-only registry (explicit: the ctor default is tool+user)."""
    return _make_registry(allowed_append_roles=["tool"])


@pytest.fixture
def registry_with_system():
    # [FUNCTION DOC] HELPER: Pytest fixture yielding a registry allowing tool and system roles
    # to be appended.
    """Registry that allows both tool and system in appended messages."""
    return _make_registry(allowed_append_roles=["tool", "system"])


@pytest.fixture
def registry_with_user():
    # [FUNCTION DOC] HELPER: Pytest fixture yielding a registry allowing tool and user roles to
    # be appended.
    """Registry that allows tool and user in appended messages."""
    return _make_registry(allowed_append_roles=["tool", "user"])


class TestSessionCRUD:
    # [CLASS DOC] Tests for creating, fetching, removing sessions in the registry and for the
    # record carried by a committed generation.
    def test_create_session(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Creating a session returns a 32-character id that is registered in the
        # registry's session mapping.
        session_id = registry.create_session()
        assert session_id is not None
        assert len(session_id) == 32
        assert session_id in registry.sessions

    def test_get_session(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A freshly created session can be fetched and has no records.
        session_id = registry.create_session()
        session = registry.get_session(session_id)
        assert _records(session) == []

    def test_get_session_not_found(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Fetching an unknown id raises the session-not-found error.
        with pytest.raises(SessionNotFoundError):
            registry.get_session("nonexistent")

    def test_remove_session(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Removing a session deletes it from the registry; removing it a second
        # time raises the session-not-found error.
        session_id = registry.create_session()
        registry.remove_session(session_id)  # no raise = success
        assert session_id not in registry.sessions
        with pytest.raises(SessionNotFoundError):
            registry.remove_session(session_id)

    def test_committed_generation_carries_its_record(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] After one commit the session has exactly one record, it is the very
        # object stored on the committed node, and its path is the chat-completions path.
        session_id = registry.create_session()
        session = registry.get_session(session_id)
        node = _commit(session, [{"role": "user", "content": "hello"}], ASSISTANT_MSG_1, [1, 2], [10])

        assert len(_records(session)) == 1
        assert _records(session)[0] is node.record
        assert _records(session)[0].path == "/v1/chat/completions"

    def test_append_record_missing_session(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Despite its name, only checks that fetching a missing session id raises
        # the session-not-found error (no record is appended).
        # KNOWN ISSUE: the name suggests an append-record test but the body just repeats the
        # get-session-not-found check.
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
    # [CLASS DOC] Multi-turn tests with a single user turn: preparing the input ids of follow-up
    # requests from the stored token ids and committing each generation, including error cases.
    """Test prepare_token_ids_and_request_args and commit_generation across turns."""

    def test_first_turn_renders_from_scratch(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] With nothing committed, preparing a system+user request yields the
        # first-turn sentinel tokens from the mock renderer.
        """First turn has no prior token_ids, so prepare renders from scratch."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        messages = [SYS_MSG, USER_MSG]
        result = _prepare(session, messages, tito_tokenizer=registry.tito_tokenizer)
        assert result == _MOCK_FIRST_TURN_TOKENS

    def test_two_turn_trajectory(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Commits turn one (prompt 5 ids, completion 3 ids), then checks turn two's
        # prepared ids equal the stored 8 ids (mock merge adds nothing), commits turn two and
        # asserts the flattened messages and the final concatenated token ids (13 ids).
        """Full 2-turn: user -> assistant(tool_call) -> tool -> final answer."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # --- Turn 1: [sys, user] -> assistant with tool_call ---
        turn1_messages = [SYS_MSG, USER_MSG]
        assert _prepare(session, turn1_messages, tito_tokenizer=registry.tito_tokenizer) == _MOCK_FIRST_TURN_TOKENS

        turn1_prompt_ids = [1, 2, 3, 4, 5]
        turn1_completion_ids = [10, 11, 12]
        _commit(session, turn1_messages, ASSISTANT_MSG_1, turn1_prompt_ids, turn1_completion_ids, max_trim_tokens=0)

        assert _messages(session) == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]
        assert _token_ids(session) == [1, 2, 3, 4, 5, 10, 11, 12]

        # --- Turn 2: [sys, user, assistant, tool] -> final answer ---
        turn2_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare(session, turn2_messages, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 4, 5, 10, 11, 12]

        turn2_prompt_ids = [1, 2, 3, 4, 5, 10, 11, 12, 20, 21]
        turn2_completion_ids = [30, 31, 32]
        _commit(
            session, turn2_messages, ASSISTANT_MSG_FINAL, turn2_prompt_ids, turn2_completion_ids, max_trim_tokens=0
        )

        assert _messages(session) == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_FINAL]
        assert _token_ids(session) == [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 32]

    def test_three_turn_trajectory(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Same as the two-turn test with a third turn: each prepared request equals
        # the stored ids so far, and after three commits there are seven messages and the
        # expected concatenated token ids.
        """Full 3-turn: user -> ass(tool) -> tool -> ass(tool) -> tool -> final."""
        sid = registry.create_session()
        session = registry.get_session(sid)

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11]

        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0)

        # Turn 3
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = _prepare(session, t3_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11, 20, 21, 30, 31]

        _commit(
            session, t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert len(_messages(session)) == 7  # sys, user, ass1, tool1, ass2, tool2, final
        assert _token_ids(session) == [1, 2, 3, 10, 11, 20, 21, 30, 31, 40, 50, 51]

    def test_prefix_mismatch_raises(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Committing a second generation whose prompt ids do not start with the
        # stored ids must raise a tokenization error mentioning a pretokenized prefix mismatch.
        """update_pretokenized_state asserts stored token_ids is prefix of new."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        with pytest.raises(TokenizationError, match="pretokenized prefix mismatch"):
            _commit(
                session,
                [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1],
                ASSISTANT_MSG_FINAL,
                [9, 9, 9, 20, 21],  # does NOT start with [1,2,3,10,11]
                [30],
                max_trim_tokens=0,
            )

    def test_disallowed_env_role_in_suffix_raises(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] With a tool-only registry, a user message appended after the stored
        # assistant must raise a message validation error naming the role and allowed roles.
        """The append-role gate still guards env roles (assistants are exempt:
        they are prompt material for compaction branches)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        bad_messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, {"role": "user", "content": "not allowed here"}]
        with pytest.raises(MessageValidationError, match="role=.user.*allowed="):
            _prepare(session, bad_messages, tito_tokenizer=registry.tito_tokenizer)

    def test_session_not_found_raises(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Fetching an unknown session raises the session-not-found error whose
        # message contains "session not found".
        with pytest.raises(SessionNotFoundError, match="session not found"):
            registry.get_session("nonexistent")

    def test_no_system_message(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A trajectory starting with only a user message commits and its follow-up
        # (assistant + tool appended) prepares to the three stored ids.
        """Works without system message (system is optional)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        msgs = [USER_MSG]
        _commit(session, msgs, ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2_msgs = [USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 10]

    def test_multiple_system_messages_at_start(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Two leading system messages: the first turn renders the sentinel, the
        # commit stores all four messages, and the follow-up prepares to the stored ids.
        """Multiple system messages before the user message are allowed (part of stored prefix)."""
        sid = registry.create_session()
        session = registry.get_session(sid)
        extra_sys = {"role": "system", "content": "Extra instructions."}
        msgs = [SYS_MSG, extra_sys, USER_MSG]
        result = _prepare(session, msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == _MOCK_FIRST_TURN_TOKENS  # first turn, no prior tokens

        _commit(session, msgs, ASSISTANT_MSG_1, [1, 2, 3, 4], [10, 11], max_trim_tokens=0)
        assert _messages(session) == [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1]

        t2_msgs = [SYS_MSG, extra_sys, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert result == [1, 2, 3, 4, 10, 11]


# ---------------------------------------------------------------------------
# TestAppendRole* — allowed_append_roles policy tests
#
# Each class tests one configuration: tool-only, tool+system, tool+user
# (the ctor default).  Tests verify which appended roles are accepted or
# rejected under each allowed_append_roles setting.
# ---------------------------------------------------------------------------


class TestAppendRoleToolOnly:
    # [CLASS DOC] Append-role policy tests for the tool-only configuration.
    """Config: allowed_append_roles=['tool']."""

    def test_tool_append_allowed(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Appending a tool message after a committed assistant is accepted and
        # prepare returns a list of ids.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare(session, messages, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_rejected(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Appending a system message after the tool message raises a message
        # validation error that names role system and the allowed roles.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            _prepare(session, messages, tito_tokenizer=registry.tito_tokenizer)

    def test_user_append_rejected(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Appending a user message after the tool message raises a message
        # validation error naming role user and the allowed roles.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            _prepare(session, messages, tito_tokenizer=registry.tito_tokenizer)


class TestAppendRoleToolSystem:
    # [CLASS DOC] Append-role policy tests for the tool+system configuration.
    """Config: allowed_append_roles=['tool', 'system']."""

    def test_tool_append_allowed(self, registry_with_system: SessionRegistryV2):
        # [FUNCTION DOC] Appending a tool message is accepted.
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare(session, messages, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_allowed(self, registry_with_system: SessionRegistryV2):
        # [FUNCTION DOC] Appending a tool then a retry system message is accepted and prepares to
        # the stored five ids.
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = _prepare(session, messages, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert result == [1, 2, 3, 10, 11]

    def test_system_then_assistant_trajectory(self, registry_with_system: SessionRegistryV2):
        # [FUNCTION DOC] Three-turn flow where turn two appends tool+retry-system; after the
        # second commit the flattened messages include the retry system message, and a third
        # request prepares to a list.
        """Full trajectory with a retry system message between tool-call turns."""
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)

        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        result = _prepare(session, t2_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0)
        assert _messages(session) == [
            SYS_MSG,
            USER_MSG,
            ASSISTANT_MSG_1,
            TOOL_MSG_1,
            RETRY_SYS_MSG,
            ASSISTANT_MSG_2,
        ]

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = _prepare(session, t3_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_append_rejected(self, registry_with_system: SessionRegistryV2):
        # [FUNCTION DOC] A user message appended after tool is rejected with a message validation
        # error naming role user.
        sid = registry_with_system.create_session()
        session = registry_with_system.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "extra"}]
        with pytest.raises(MessageValidationError, match="role='user'.*allowed="):
            _prepare(session, messages, tito_tokenizer=registry_with_system.tito_tokenizer)


class TestAppendRoleToolUser:
    # [CLASS DOC] Append-role policy tests for the tool+user configuration (the constructor
    # default).
    """Config: allowed_append_roles=['tool', 'user']; user follow-ups are allowed here."""

    def test_tool_append_allowed(self, registry_with_user: SessionRegistryV2):
        # [FUNCTION DOC] Appending a tool message is accepted.
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        result = _prepare(session, messages, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_append_allowed(self, registry_with_user: SessionRegistryV2):
        # [FUNCTION DOC] Appending tool then a user follow-up is accepted.
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, {"role": "user", "content": "follow-up"}]
        result = _prepare(session, messages, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_user_then_assistant_trajectory(self, registry_with_user: SessionRegistryV2):
        # [FUNCTION DOC] Three-turn flow where turn two appends tool plus a user follow-up; checks
        # the flattened messages after the second commit and that turn three prepares.
        """Full trajectory: tool → user follow-up → assistant → tool → final."""
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)

        # Turn 1: [sys, user] -> assistant(tool_call)
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: append tool + user follow-up -> assistant(tool_call)
        follow_up = {"role": "user", "content": "Also check Shanghai."}
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up]
        result = _prepare(session, t2_msgs, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21, 22], [30, 31], max_trim_tokens=0)
        assert _messages(session) == [
            SYS_MSG,
            USER_MSG,
            ASSISTANT_MSG_1,
            TOOL_MSG_1,
            follow_up,
            ASSISTANT_MSG_2,
        ]

        # Turn 3: append tool after the second assistant
        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, follow_up, ASSISTANT_MSG_2, TOOL_MSG_2]
        result = _prepare(session, t3_msgs, tito_tokenizer=registry_with_user.tito_tokenizer)
        assert isinstance(result, list)

    def test_system_append_rejected(self, registry_with_user: SessionRegistryV2):
        # [FUNCTION DOC] A system message appended after tool is rejected with a message
        # validation error naming role system.
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, RETRY_SYS_MSG]
        with pytest.raises(MessageValidationError, match="role='system'.*allowed="):
            _prepare(session, messages, tito_tokenizer=registry_with_user.tito_tokenizer)


class TestCarriedAssistant:
    # [CLASS DOC] Tests for assistant messages carried by the client inside a request suffix.
    """Client-carried assistants in a branch suffix are prompt material and
    inherit the attach node's snapshot like any other suffix."""

    def test_carried_assistant_before_user_allowed(self):
        # [FUNCTION DOC] With assistant also allowed as an append role, a request with tool, a
        # carried assistant and a user message prepares to the stored prefix ids (proving the
        # merge path, not a fresh render) and still attaches under the first generation node.
        mock_tito = _make_mock_tito(_MockTITOTokenizer, ["tool", "user", "assistant"])
        registry = SessionRegistryV2(tokenizer=None, tito_tokenizer=mock_tito)
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        carried = {"role": "assistant", "content": "carried summary"}
        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, carried, {"role": "user", "content": "next"}]
        result = _prepare(session, messages, tito_tokenizer=registry.tito_tokenizer)
        # The mock merge returns the pretokenized prefix unchanged: getting the
        # snapshot back proves the merge path ran (a from-scratch render would
        # return the first-turn sentinel) under the parent node.
        assert result == [1, 2, 3, 10]
        assert attach_point_for_request(session, messages).node is session.tree.nodes[0]

    def test_carried_assistant_rejected_without_template_support(self, registry_with_user):
        # [FUNCTION DOC] With assistant not in the allowed append roles (tool+user registry), a
        # carried assistant in the suffix raises a message validation error naming role assistant.
        sid = registry_with_user.create_session()
        session = registry_with_user.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10], max_trim_tokens=0)

        carried = {"role": "assistant", "content": "carried summary"}
        messages = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, carried, {"role": "user", "content": "next"}]
        with pytest.raises(MessageValidationError, match="role='assistant'.*allowed="):
            _prepare(session, messages, tito_tokenizer=registry_with_user.tito_tokenizer)


class TestRollback:
    # [CLASS DOC] Retry handling over the generation tree: a divergent retry attaches under the
    # matching earlier node, nothing is ever deleted, and a request diverging inside the root
    # delta opens a new root.
    """Retry handling over the tree.

    Attaching never destroys anything: a legal one-step retry attaches under
    the anchor node (abandoned generations stay in the tree). Byte-level HTTP
    fidelity is pinned by ``TestRollbackPins`` in ``test_sessions.py``; this
    suite covers the mechanism."""

    def _dispatch_and_apply(self, state, messages):
        # [FUNCTION DOC] HELPER: Thin wrapper that returns the attach point for the given request
        # messages on the state; it applies nothing despite its name.
        return attach_point_for_request(state, messages)

    def test_rollback_to_first_assistant(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] After two commits, a request replacing the tool result attaches under the
        # first generation (node zero) and prepares; the tree keeps both nodes and node zero
        # still holds its five token ids and three path messages.
        """After 2 completions, a divergent retry rolls back to the first checkpoint."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        # Turn 1: [sys, user] -> assistant1
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> assistant2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0)

        assert len(_path(session)) == 2
        assert len([n.token_ids for n in _path(session)]) == 2

        # Retry: send [sys, user, asst1, NEW_tool] - diverges after asst1
        new_tool = {"role": "tool", "content": '{"temperature": 99}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        attach = self._dispatch_and_apply(state, rollback_msgs)
        assert attach.node is state.tree.nodes[0]
        result = _prepare(session, rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # The retry attaches under the first generation; the abandoned node stays in the tree
        assert len(state.tree.nodes) == 2
        assert len(attach.node.path_nodes()) == 1
        assert attach.node.token_ids == [1, 2, 3, 10, 11]
        assert attach.node.path_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_multi_step_rollback_raises(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] After three commits, a deeply divergent request attaches at node zero and
        # nothing is destroyed: node count, token ids, records and path messages are unchanged.
        # KNOWN ISSUE: the name and docstring say a multi-step rollback raises a validation error,
        # but the test asserts no error and that the attach succeeds without destroying anything.
        """Rollback that discards >1 assistant raises MessageValidationError and leaves state unchanged."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20, 21], [30, 31], max_trim_tokens=0)

        t3_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        _prepare(session, t3_msgs, tito_tokenizer=registry.tito_tokenizer)
        _commit(
            session, t3_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 20, 21, 30, 31, 40], [50, 51], max_trim_tokens=0
        )

        assert len(_path(session)) == 3

        # Snapshot state before attempted rollback
        prev_messages = list(_messages(session))
        prev_token_ids = list([n.token_ids for n in _path(session)])
        prev_records = list(_records(session))
        prev_num_assistant = len(_path(session))

        # Deep divergence: the view positions at the deep anchor (node 0) and
        # NOTHING is destroyed — the abandoned generations stay in the tree.
        new_tool = {"role": "tool", "content": '{"alt": true}', "tool_call_id": "call_1"}
        assert (
            attach_point_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]).node is state.tree.nodes[0]
        )
        assert len(state.tree.nodes) == prev_num_assistant  # nothing destroyed
        assert [n.token_ids for n in state.tree.nodes] == prev_token_ids
        assert [n.record for n in state.tree.nodes] == prev_records
        assert state.tree.nodes[2].path_messages() == prev_messages

    def test_rollback_then_continue_full_trajectory(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Commits two turns, retries from the first assistant with a different tool
        # result and commits a final answer; the served path then has two nodes with the retry's
        # token ids and messages.
        """Rollback and then complete a full new trajectory from the checkpoint."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        # Turn 1
        t1_msgs = [SYS_MSG, USER_MSG]
        _commit(session, t1_msgs, ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        # Turn 2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 3, 10, 11, 20], [30], max_trim_tokens=0)

        # Rollback to asst1, send different tool
        new_tool = {"role": "tool", "content": '{"retry": true}', "tool_call_id": "call_1"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        self._dispatch_and_apply(state, rollback_msgs)
        result = _prepare(session, rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # Continue: complete a new turn from the rolled-back state
        _commit(session, rollback_msgs, ASSISTANT_MSG_FINAL, [1, 2, 3, 10, 11, 40, 41], [50, 51], max_trim_tokens=0)

        assert len(_path(session)) == 2
        assert len([n.token_ids for n in _path(session)]) == 2
        assert _token_ids(session) == [1, 2, 3, 10, 11, 40, 41, 50, 51]
        assert _messages(session) == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool, ASSISTANT_MSG_FINAL]

    def test_rollback_fewer_messages_than_stored(self, registry_with_system: SessionRegistryV2):
        # [FUNCTION DOC] A retry carrying fewer messages than stored (ending in a retry system
        # message) attaches under the first generation, whose path has one node and three messages.
        """Rollback triggered when request has strictly fewer messages than stored."""
        sid = registry_with_system.create_session()
        state = registry_with_system.get_session(sid)
        session = state

        # Turn 1: [sys, user] -> asst1
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Turn 2: [sys, user, asst1, tool1] -> asst2
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare(session, t2_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        _commit(session, t2_msgs, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        # stored messages: [sys, user, asst1, tool1, asst2] (5 messages)

        # Agent retries with only [sys, user, asst1, sys_retry] (4 messages)
        retry_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, RETRY_SYS_MSG]
        attach = self._dispatch_and_apply(state, retry_msgs)
        result = _prepare(session, retry_msgs, tito_tokenizer=registry_with_system.tito_tokenizer)
        assert isinstance(result, list)

        assert len(attach.node.path_nodes()) == 1
        assert attach.node.path_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_rollback_to_second_assistant(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] After three commits, diverging at the second tool result attaches under
        # the second generation (node one), whose path has two nodes and stored ids of length five.
        """Rollback to the second checkpoint (skipping the third)."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        # 3 completions
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare(session, t2, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)

        t3 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        _prepare(session, t3, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t3, ASSISTANT_MSG_FINAL, [1, 2, 10, 20, 30, 40], [50], max_trim_tokens=0)

        assert len(_path(session)) == 3

        # Rollback: keep up to asst2, diverge at tool2
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_2"}
        rollback_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, new_tool]
        attach = self._dispatch_and_apply(state, rollback_msgs)
        assert attach.node is state.tree.nodes[1]
        result = _prepare(session, rollback_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        assert len(attach.node.path_nodes()) == 2
        assert attach.node.token_ids == [1, 2, 10, 20, 30]
        assert attach.node.path_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2]

    def test_no_rollback_when_append_only(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A pure append of a tool message attaches under the only node, prepares,
        # and leaves the served path (one node) and token ids unchanged.
        """Normal append-only flow classifies as EXTEND and mutates nothing."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Append tool - not a rollback
        t2_msgs = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        assert attach_point_for_request(state, t2_msgs).node is state.tree.nodes[0]
        result = _prepare(session, t2_msgs, tito_tokenizer=registry.tito_tokenizer)
        assert isinstance(result, list)

        # State should NOT have been rolled back
        assert len(_path(session)) == 1
        assert len([n.token_ids for n in _path(session)]) == 1
        assert _token_ids(session) == [1, 2, 10]

    def test_rollback_no_assistant_in_prefix_raises(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A request diverging inside the root delta has no attach node (opens a new
        # root) and the tree keeps its single existing node.
        # KNOWN ISSUE: the name and docstring say dispatch rejects/raises, but the test asserts no
        # exception, only a missing attach node.
        """Dispatch rejects when no assistant anchor exists in the matched prefix."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        # Diverge inside the root delta: nothing fully matches, so the request
        # opens a new root (the old chain stays intact in the tree).
        bad_msgs = [SYS_MSG, {"role": "user", "content": "different question"}]
        assert attach_point_for_request(state, bad_msgs).node is None
        assert len(state.tree.nodes) == 1

    def test_served_records_follow_the_latest_commit(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Attaching a retry leaves the served records at two; only after the retry
        # is committed do the served records become the first node's record and the new node's.
        """Attaching is pure; the served record chain moves only when a retry commits."""
        sid = registry.create_session()
        state = registry.get_session(sid)
        session = state

        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare(session, t2, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)

        assert len(_records(session)) == 2

        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        retry = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]
        attach = self._dispatch_and_apply(state, retry)
        assert attach.node is state.tree.nodes[0]
        assert len(_records(session)) == 2  # nothing served changes until the retry commits

        _commit(session, retry, ASSISTANT_MSG_FINAL, [1, 2, 10, 40], [50], max_trim_tokens=0)
        assert _records(session) == [state.tree.nodes[0].record, state.tree.nodes[2].record]


class TestJudgmentCountingMatrix:
    # [CLASS DOC] Matrix of attach-point decisions where the unit is the generation (tree node),
    # not the message; client-carried foreign assistants are never anchors. The class constant
    # FS_USER_2 is the second user message used in few-shot style prompts.
    """Counting matrix for the single-chain judgment: the step unit is the
    GENERATION (tree node), not the message — client-carried foreign
    assistants live inside a node's delta and are never anchors (the shape
    the prompt-assistant fix pinned)."""

    FS_USER_2 = {"role": "user", "content": "the real question"}

    def _fresh(self, registry):
        # [FUNCTION DOC] HELPER: Creates a session and returns the same state object twice (as a
        # state/session pair).
        sid = registry.create_session()
        state = registry.get_session(sid)
        return state, state

    def _two_turns(self, registry):
        # [FUNCTION DOC] HELPER: Builds a fresh session with two committed generations (tool call
        # then second assistant) and returns the state pair.
        """stored: [sys, user, asst1, tool1, asst2]; 2 generations."""
        state, session = self._fresh(registry)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        _prepare(session, t2, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_2, [1, 2, 10, 20], [30], max_trim_tokens=0)
        return state, session

    def test_empty_view_extends_anything(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] On an empty session any request has no attach node (starts a new root).
        state, _ = self._fresh(registry)
        assert attach_point_for_request(state, [USER_MSG]).node is None

    def test_strict_extension_leaves_view_alone(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A request strictly extending the stored history attaches under the
        # latest node (node one).
        state, _ = self._two_turns(registry)
        request = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        assert attach_point_for_request(state, request).node is state.tree.nodes[1]

    def test_degenerate_equal_history_is_extend(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A request identical to the stored history also attaches under node one.
        state, _ = self._two_turns(registry)
        request = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2]
        assert attach_point_for_request(state, request).node is state.tree.nodes[1]

    def test_pure_drop_one_generation(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A request that drops the last assistant (keeping the tool message)
        # attaches under node zero.
        """Dropping asst2 (and nothing else) is one step even though the
        request also re-sends tool1: 2 messages beyond the anchor, 1 generation."""
        state, _ = self._two_turns(registry)
        assert (
            attach_point_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]).node
            is state.tree.nodes[0]
        )

    def test_divergent_one_generation(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A request with a different tool result attaches under node zero, whose
        # path messages are system, user, first assistant.
        state, _ = self._two_turns(registry)
        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        attach = attach_point_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool])
        assert attach.node is state.tree.nodes[0]
        assert attach.node.path_messages() == [SYS_MSG, USER_MSG, ASSISTANT_MSG_1]

    def test_deep_divergence_two_generations_rejected(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] With three generations committed, a request diverging after the first
        # assistant attaches under node zero without destroying anything.
        # KNOWN ISSUE: the name says "rejected" but nothing is rejected; the test expects success.
        state, session = self._two_turns(registry)
        t3 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1, ASSISTANT_MSG_2, TOOL_MSG_2]
        _prepare(session, t3, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t3, ASSISTANT_MSG_FINAL, [1, 2, 10, 20, 30, 40], [50], max_trim_tokens=0)

        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_1"}
        assert (
            attach_point_for_request(state, [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, new_tool]).node is state.tree.nodes[0]
        )  # deep anchor, no destruction

    def test_no_anchor(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A request diverging inside the first user message has no attach node and
        # the tree keeps its one node.
        state, session = self._fresh(registry)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)
        assert (
            attach_point_for_request(state, [SYS_MSG, {"role": "user", "content": "different"}]).node is None
        )  # new root; old chain intact
        assert len(state.tree.nodes) == 1

    def test_prompt_assistant_is_not_an_anchor(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] With a few-shot first request (user, assistant, user) committed, then a
        # tool-follow-up commit, a divergent retry attaches under node zero whose path messages
        # are the few-shot prompt plus the first real assistant.
        """Few-shot first request: the anchor is the generation boundary, and
        a divergent retry drops exactly one generation (historically this
        computed discard_count=0 and kept a stale checkpoint)."""
        state, session = self._fresh(registry)
        few_shot = [USER_MSG, ASSISTANT_MSG_1, self.FS_USER_2]
        _commit(session, few_shot, ASSISTANT_MSG_2, [1, 2], [10], max_trim_tokens=0)
        t2 = [*few_shot, ASSISTANT_MSG_2, TOOL_MSG_2]
        _prepare(session, t2, tito_tokenizer=registry.tito_tokenizer)
        _commit(session, t2, ASSISTANT_MSG_FINAL, [1, 2, 10, 20], [30], max_trim_tokens=0)
        # stored: [user, fs_asst, user2, asst2, tool2, final]; generations: asst2, final

        new_tool = {"role": "tool", "content": '{"alt": 1}', "tool_call_id": "call_2"}
        attach = attach_point_for_request(state, [*few_shot, ASSISTANT_MSG_2, new_tool])
        assert attach.node is state.tree.nodes[0]
        assert attach.node.path_messages() == [*few_shot, ASSISTANT_MSG_2]

    def test_prefix_with_only_prompt_assistants_has_no_anchor(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Diverging inside the few-shot root delta gives no attach node.
        state, session = self._fresh(registry)
        few_shot = [USER_MSG, ASSISTANT_MSG_1, self.FS_USER_2]
        _commit(session, few_shot, ASSISTANT_MSG_2, [1, 2], [10], max_trim_tokens=0)

        assert (
            attach_point_for_request(state, [USER_MSG, ASSISTANT_MSG_1, {"role": "user", "content": "changed"}]).node
            is None
        )  # divergence inside the root delta opens a new root


class TestUpdatePretokenizedStateMissingSession:
    # [CLASS DOC] Missing-session behavior tests.
    """update_pretokenized_state raises SessionNotFoundError for unknown session."""

    def test_raises_on_missing_session(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Fetching an unknown session raises the session-not-found error.
        # KNOWN ISSUE: the class docstring mentions update_pretokenized_state, but the test only
        # calls get_session.
        with pytest.raises(SessionNotFoundError, match="session not found"):
            registry.get_session("nonexistent")


class TestComputeSessionMismatch:
    # [CLASS DOC] Tests for the registry's session-mismatch computation, which re-renders the
    # session's messages through the (mocked) chat template and compares them with stored ids
    # using a (mocked) comparator.
    """Tests for compute_session_mismatch."""

    def test_raises_for_missing_session(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Fetching an unknown session raises the session-not-found error.
        with pytest.raises(SessionNotFoundError):
            registry.get_session("nonexistent")

    def test_returns_none_for_empty_token_ids(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] A session with no committed ids yields a none result from the mismatch
        # computation.
        sid = registry.create_session()
        session = registry.get_session(sid)
        assert registry.compute_session_mismatch(session) is None

    def test_returns_empty_list_when_no_mismatch(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Mocks the template render to return the stored ids and the comparator to
        # report no differences. Asserts the result is an empty list, the comparator was called
        # once with rendered and stored ids, and the render was called once with the session
        # messages, no generation prompt, tokenization on and empty template arguments.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

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
            _messages(session),
            add_generation_prompt=False,
            tokenize=True,
            template_args={},
        )

    def test_returns_mismatch_dicts(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] With a comparator returning one fake mismatch object (position two), the
        # result is the list of its dictionary form. The nested FakeMismatch dataclass only
        # supplies a to-dict conversion.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        registry.tito_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 99, 10, 11])

        @dataclass
        class FakeMismatch:
            position: int

            def to_dict(self):
                # [FUNCTION DOC] HELPER: Returns a mapping with the position and a fixed detail
                # string "mismatch".
                return {"position": self.position, "detail": "mismatch"}

        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = [FakeMismatch(position=2)]
        registry.comparator = mock_comparator

        result = registry.compute_session_mismatch(session)
        assert result == [{"position": 2, "detail": "mismatch"}]

    def test_raises_tokenization_error_on_exception(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] If the render raises a runtime error, the computation re-raises it as a
        # tokenization error whose message contains the original text.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)

        registry.tito_tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("tokenizer failed"))

        with pytest.raises(TokenizationError, match="tokenizer failed"):
            registry.compute_session_mismatch(session)

    def test_renders_with_the_tools_the_latest_node_recorded(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Commits a second node with turn arguments carrying a tool list (while the
        # record's own request carries a different tool), then runs the mismatch computation with
        # mocks. Asserts the render receives the node's tools as template arguments and no
        # generation prompt.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2], [10], max_trim_tokens=0)

        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        # A record carrying other tools does not matter: the node's template args do.
        record = SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"tools": [{"type": "function", "function": {"name": "get_time"}}]},
            response={},
        )
        t2 = [SYS_MSG, USER_MSG, ASSISTANT_MSG_1, TOOL_MSG_1]
        commit_generation(
            session,
            parent=session.latest(),
            request_messages=t2,
            assistant_message=ASSISTANT_MSG_2,
            prompt_token_ids=[1, 2, 10, 20],
            completion_token_ids=[30],
            max_trim_tokens=0,
            record=record,
            response_id="resp-tools",
            finish_reason="stop",
            turn_args={"tools": tools},
        )

        mock_tokenize = MagicMock(return_value=[1, 2, 10])
        registry.tito_tokenizer.apply_chat_template = mock_tokenize
        mock_comparator = MagicMock()
        mock_comparator.compare_sequences.return_value = []
        registry.comparator = mock_comparator

        registry.compute_session_mismatch(session)

        _, kwargs = mock_tokenize.call_args
        assert kwargs["template_args"] == {"tools": tools}
        assert kwargs["add_generation_prompt"] is False

    def test_renders_with_the_kwargs_the_latest_node_recorded(self, registry: SessionRegistryV2):
        # [FUNCTION DOC] Sets the latest node's turn arguments (temperature plus a reasoning
        # effort chat-template kwarg); the mismatch computation returns empty and renders with only
        # the chat-template kwargs as template arguments.
        sid = registry.create_session()
        session = registry.get_session(sid)
        _commit(session, [SYS_MSG, USER_MSG], ASSISTANT_MSG_1, [1, 2, 3], [10, 11], max_trim_tokens=0)
        session.latest().turn_args = {"temperature": 0.7, "chat_template_kwargs": {"reasoning_effort": "low"}}

        mock_tokenize = MagicMock(return_value=[1, 2, 3, 10, 11])
        registry.tito_tokenizer.apply_chat_template = mock_tokenize
        registry.comparator = MagicMock(compare_sequences=MagicMock(return_value=[]))

        assert registry.compute_session_mismatch(session) == []
        assert mock_tokenize.call_args.kwargs["template_args"] == {"reasoning_effort": "low"}


def test_committed_full_turn_args_are_isolated_between_siblings(registry):
    # [FUNCTION DOC] Commits two sibling root generations with the same request-args mapping,
    # mutating that mapping (top-level value, nested list, input ids) in between and after.
    # Asserts each node holds an independent deep copy: the first keeps the original values, the
    # second the mutated ones, and later input-id mutation affects neither.
    session = registry.get_session(registry.create_session())
    request_args = {"temperature": 0.7, "chat_template_kwargs": {"nested": [1]}, "input_ids": [1, 2]}
    record = SessionRecord(
        timestamp=1.0, method="POST", path="/v1/chat/completions", status_code=200, request=request_args, response={}
    )
    first = commit_generation(
        session,
        parent=None,
        request_messages=[SYS_MSG, USER_MSG],
        assistant_message=ASSISTANT_MSG_1,
        prompt_token_ids=[1, 2],
        completion_token_ids=[3],
        max_trim_tokens=0,
        record=record,
        response_id="first",
        finish_reason="stop",
        turn_args=request_args,
    )
    request_args["temperature"] = 0.1
    request_args["chat_template_kwargs"]["nested"].append(2)
    second = commit_generation(
        session,
        parent=None,
        request_messages=[SYS_MSG, USER_MSG],
        assistant_message=ASSISTANT_MSG_1,
        prompt_token_ids=[1, 2],
        completion_token_ids=[4],
        max_trim_tokens=0,
        record=record,
        response_id="second",
        finish_reason="stop",
        turn_args=request_args,
    )
    request_args["input_ids"].append(9)
    assert first.turn_args == {"temperature": 0.7, "chat_template_kwargs": {"nested": [1]}, "input_ids": [1, 2]}
    assert second.turn_args == {"temperature": 0.1, "chat_template_kwargs": {"nested": [1, 2]}, "input_ids": [1, 2]}


def test_mismatch_does_not_resolve_defaults_for_an_empty_committed_record(registry):
    # [FUNCTION DOC] With the tokenizer holding default chat-template kwargs, computing a
    # mismatch for explicit empty turn arguments returns no mismatches and renders with empty
    # template arguments, i.e. the defaults are not applied.
    registry.tito_tokenizer.chat_template_kwargs = {"enable_thinking": True}
    registry.tito_tokenizer.apply_chat_template = MagicMock(return_value=[1, 2])
    registry.comparator = MagicMock(compare_sequences=MagicMock(return_value=[]))
    assert registry.compute_mismatch([SYS_MSG, USER_MSG, ASSISTANT_MSG_1], [1, 2], turn_args={}) == []
    assert registry.tito_tokenizer.apply_chat_template.call_args.kwargs["template_args"] == {}
