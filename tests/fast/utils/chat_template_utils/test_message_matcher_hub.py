# [MODULE DOC] Contract tests for the session message matcher hub
# (miles.utils.chat_template_utils.message_matcher_hub). A matcher decides whether a stored
# message equals a replayed one. Three built-in policies are nested by strictness: strict (full
# comparison with empty-value and wire-index normalization), loose tool call (additionally treats
# tool call arguments as equal when they decode to equal JSON objects, with empty arguments
# normalized, but still comparing ids, types, names, unknown fields, call count/order and
# reasoning content) and role/content only (compares only role and content with empty-value
# equivalence). Also tested: resolving matchers by alias or dotted path, the contract wrapper
# (must return a real bool, exceptions wrapped), the append-only assertion using a configurable
# matcher, compatibility re-exports, lazy import of the function registry, and absence of import
# cycles (checked in subprocesses). Test helper functions build assistant messages with tool calls.
"""Contract tests for session message matching policies."""

from __future__ import annotations

import copy
import operator
import subprocess
import sys
from decimal import Decimal
from typing import Any

import pytest

import miles.utils.chat_template_utils as chat_template_utils
from miles.utils.chat_template_utils import message_matcher_hub, template
from miles.utils.chat_template_utils.message_matcher_hub import (
    SessionMessageMatcherError,
    loose_tool_call_message_matches,
    resolve_session_message_matcher,
    role_content_only_message_matches,
    strict_message_matches,
)
from miles.utils.chat_template_utils.message_matcher_hub.funcs import _validated

_MISSING = object()


def _tool_call(
    arguments: Any = _MISSING,
    *,
    call_id: str = "call-A",
    call_type: str = "function",
    name: str = "lookup",
    index: Any = _MISSING,
    call_extra: dict[str, Any] | None = None,
    function_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build one tool call dictionary with an id (default "call-A"), a type
    # (default "function") and a function entry with the name (default "lookup"). The arguments
    # entry is included only if arguments were supplied (a sentinel distinguishes missing from
    # null). Extra function entries are merged in; an index is added only if supplied; extra
    # call-level entries are merged last.
    function = {"name": name}
    if arguments is not _MISSING:
        function["arguments"] = arguments
    if function_extra:
        function.update(function_extra)

    call = {"id": call_id, "type": call_type, "function": function}
    if index is not _MISSING:
        call["index"] = index
    if call_extra:
        call.update(call_extra)
    return call


def _assistant(
    arguments: Any = _MISSING,
    *,
    calls: list[Any] | None = None,
    content: Any = "",
    reasoning_content: Any = _MISSING,
    **call_kwargs: Any,
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build an assistant message with the given content (default empty)
    # and tool calls: the explicit list when given, otherwise a single tool call built from the
    # arguments and the remaining keyword options. Reasoning content is added only when supplied.
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": calls if calls is not None else [_tool_call(arguments, **call_kwargs)],
    }
    if reasoning_content is not _MISSING:
        message["reasoning_content"] = reasoning_content
    return message


def test_strict_preserves_existing_empty_and_wire_index_normalization() -> None:
    # [FUNCTION DOC] A stored message (tool call index 0, null reasoning) and a replayed one (index
    # 91, no reasoning, different extra wire metadata) are equal under the strict matcher.
    stored = _assistant('{"city":"Paris"}', index=0, reasoning_content=None)
    replayed = _assistant('{"city":"Paris"}', index=91)
    stored["wire_metadata"] = {"source": "stored"}
    replayed["wire_metadata"] = {"source": "replayed"}

    assert strict_message_matches(stored, replayed)


@pytest.mark.parametrize(
    ("stored_arguments", "replayed_arguments"),
    [
        pytest.param(None, "", id="none-empty-string"),
        pytest.param("", {}, id="empty-string-dict"),
        pytest.param({}, "{}", id="dict-json-string"),
        pytest.param("{}", "{ }", id="json-whitespace"),
    ],
)
def test_loose_tool_call_normalizes_empty_argument_objects(stored_arguments: Any, replayed_arguments: Any) -> None:
    # [FUNCTION DOC] Parametrized over four pairs of empty-argument spellings (null versus empty
    # string, empty string versus empty dict, empty dict versus "{}", "{}" versus "{ }"). Asserts
    # the strict matcher rejects the pair while the loose matcher accepts it.
    stored = _assistant(stored_arguments)
    replayed = _assistant(replayed_arguments)

    assert not strict_message_matches(stored, replayed)
    assert loose_tool_call_message_matches(stored, replayed)


def test_loose_tool_call_normalizes_nested_json_objects_without_mutating_array_order() -> None:
    # [FUNCTION DOC] Stored arguments are a JSON string with nested objects, an escaped non-ASCII
    # character and number 1e0; replayed arguments are the equivalent dictionary with different key
    # order, float 1.0 and the literal character. Strict rejects, loose accepts.
    stored = _assistant('{"outer":{"b":2,"a":[1,{"text":"\\u00e9"}]},"number":1e0}')
    replayed = _assistant(
        {
            "number": 1.0,
            "outer": {"a": [1.0, {"text": "é"}], "b": 2},
        }
    )

    assert not strict_message_matches(stored, replayed)
    assert loose_tool_call_message_matches(stored, replayed)


@pytest.mark.parametrize(
    ("stored_arguments", "replayed_arguments"),
    [
        pytest.param('{"value":true}', '{"value":1}', id="boolean-number"),
        pytest.param('{"value":null}', '{"value":"null"}', id="null-string"),
        pytest.param('{"value":[1,2]}', '{"value":[2,1]}', id="array-order"),
        pytest.param('{"value":{"x":1}}', '{"value":[["x",1]]}', id="object-array"),
        pytest.param('{"value":1}', '{"value":2}', id="number-value"),
    ],
)
def test_loose_tool_call_preserves_json_types_values_and_array_order(
    stored_arguments: Any, replayed_arguments: Any
) -> None:
    # [FUNCTION DOC] Parametrized over argument pairs that differ semantically (boolean versus
    # number, null versus string, array order, object versus array of pairs, different numbers).
    # Strict and loose both reject, while the role/content-only matcher still accepts.
    stored = _assistant(stored_arguments)
    replayed = _assistant(replayed_arguments)

    assert not strict_message_matches(stored, replayed)
    assert not loose_tool_call_message_matches(stored, replayed)
    assert role_content_only_message_matches(stored, replayed)


@pytest.mark.parametrize(
    ("stored_arguments", "replayed_arguments"),
    [
        pytest.param(" ", "{}", id="whitespace-only"),
        pytest.param("{invalid}", "{ invalid }", id="invalid-json"),
        pytest.param('{"x":1,"x":2}', '{"x":1, "x":2}', id="duplicate-keys"),
        pytest.param('{"x":NaN}', '{"x": NaN}', id="nan"),
        pytest.param('{"x":Infinity}', '{"x": Infinity}', id="positive-infinity"),
        pytest.param('{"x":-Infinity}', '{"x": -Infinity}', id="negative-infinity"),
        pytest.param("[]", "[ ]", id="top-level-array"),
        pytest.param("1", "1.0", id="top-level-number"),
        pytest.param({"bad": {1, 2}}, {"bad": {2, 3}}, id="non-json-dict-value"),
        pytest.param({1: "value"}, {"1": "value"}, id="non-string-dict-key"),
    ],
)
def test_loose_tool_call_invalid_or_non_object_arguments_use_type_sensitive_raw_comparison(
    stored_arguments: Any, replayed_arguments: Any
) -> None:
    # [FUNCTION DOC] Parametrized over argument pairs that are not normalizable JSON objects
    # (whitespace only, invalid JSON, duplicate keys, NaN and infinities, top-level array or
    # number, dictionaries with non-JSON values or non-string keys). Asserts the loose matcher
    # rejects each pair (raw, type-sensitive comparison, no normalization).
    stored = _assistant(stored_arguments)
    replayed = _assistant(replayed_arguments)

    assert not loose_tool_call_message_matches(stored, replayed)


def test_loose_tool_call_preserves_large_finite_json_numbers() -> None:
    # [FUNCTION DOC] Arguments with 1e999 and 10e998 (equal exact numbers beyond float range) are
    # accepted as equal by the loose matcher.
    stored = _assistant('{"value":1e999}')
    replayed = _assistant('{"value":10e998}')

    assert loose_tool_call_message_matches(stored, replayed)


def test_loose_tool_call_rejects_decimal_from_python_dict() -> None:
    # [FUNCTION DOC] A JSON string argument with number 1 versus a dictionary holding a decimal
    # object 1 (not a JSON type): the loose matcher rejects.
    stored = _assistant('{"value":1}')
    replayed = _assistant({"value": Decimal("1")})

    assert not loose_tool_call_message_matches(stored, replayed)


def test_loose_tool_call_deep_json_falls_back_without_raising() -> None:
    # [FUNCTION DOC] Two argument strings nested 2000 array levels deep and differing only in
    # whitespace. With the interpreter recursion limit temporarily lowered to 500 (always restored),
    # asserts the loose matcher returns exactly false instead of raising a recursion error.
    stored_arguments = '{"x":' + "[" * 2000 + "0" + "]" * 2000 + "}"
    replayed_arguments = '{"x": ' + "[" * 2000 + "0" + "]" * 2000 + "}"

    original_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(500)
        result = loose_tool_call_message_matches(
            _assistant(stored_arguments),
            _assistant(replayed_arguments),
        )
    finally:
        sys.setrecursionlimit(original_limit)

    assert result is False


@pytest.mark.parametrize(
    "empty_arguments",
    [
        pytest.param(None, id="none"),
        pytest.param("", id="empty-string"),
        pytest.param({}, id="dict"),
        pytest.param("{}", id="json-string"),
    ],
)
def test_loose_tool_call_does_not_equate_missing_arguments_with_empty_object(
    empty_arguments: Any,
) -> None:
    # [FUNCTION DOC] Parametrized over four empty spellings; a tool call with no arguments entry
    # at all never matches one with an empty spelling under the loose matcher.
    assert not loose_tool_call_message_matches(
        _assistant(_MISSING),
        _assistant(empty_arguments),
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        pytest.param(("id",), "call-B", id="call-id"),
        pytest.param(("type",), "provider-function", id="call-type"),
        pytest.param(("function", "name"), "other", id="function-name"),
        pytest.param(("provider",), "other", id="unknown-call-field"),
        pytest.param(("function", "provider"), "other", id="unknown-function-field"),
    ],
)
def test_loose_tool_call_still_compares_call_identity_and_unknown_fields(
    path: tuple[str, ...], replacement: Any
) -> None:
    # [FUNCTION DOC] Parametrized over a field path in the replayed tool call (call id, call type,
    # function name, unknown call-level field, unknown function-level field) and a replacement
    # value. Starts from equivalent messages (JSON string versus dictionary arguments, same unknown
    # fields), changes just that field in the replayed copy and asserts the loose matcher rejects.
    stored = _assistant(
        '{"x":1}',
        call_extra={"provider": "stored"},
        function_extra={"provider": "stored"},
    )
    replayed = _assistant(
        {"x": 1},
        call_extra={"provider": "stored"},
        function_extra={"provider": "stored"},
    )
    target = replayed["tool_calls"][0]
    if len(path) == 1:
        target[path[0]] = replacement
    else:
        target[path[0]][path[1]] = replacement

    assert not loose_tool_call_message_matches(stored, replayed)


def test_loose_tool_call_requires_unknown_fields_to_be_present() -> None:
    # [FUNCTION DOC] Deleting the unknown call-level field from the replayed tool call makes the
    # loose matcher reject.
    stored = _assistant('{"x":1}', call_extra={"provider": "same"})
    replayed = _assistant({"x": 1}, call_extra={"provider": "same"})
    del replayed["tool_calls"][0]["provider"]

    assert not loose_tool_call_message_matches(stored, replayed)


def test_loose_tool_call_preserves_tool_call_count_and_order() -> None:
    # [FUNCTION DOC] Stored has two tool calls; a reordered replay and a replay with only the first
    # call are both rejected by the loose matcher.
    stored = _assistant(
        calls=[
            _tool_call('{"x":1}', call_id="call-A", name="first"),
            _tool_call('{"x":2}', call_id="call-B", name="second"),
        ]
    )
    reordered = _assistant(
        calls=[
            _tool_call({"x": 2}, call_id="call-B", name="second"),
            _tool_call({"x": 1}, call_id="call-A", name="first"),
        ]
    )
    truncated = _assistant(calls=[_tool_call({"x": 1}, call_id="call-A", name="first")])

    assert not loose_tool_call_message_matches(stored, reordered)
    assert not loose_tool_call_message_matches(stored, truncated)


def test_loose_tool_call_does_not_relax_reasoning_content() -> None:
    # [FUNCTION DOC] Stored has non-empty reasoning content, replay has none: loose rejects while
    # role/content-only accepts.
    stored = _assistant('{"x":1}', reasoning_content="kept")
    replayed = _assistant({"x": 1})

    assert not loose_tool_call_message_matches(stored, replayed)
    assert role_content_only_message_matches(stored, replayed)


def _projected_message(field: str, value: Any) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build a user message with content "same", then set the named field to
    # the value, or remove the field entirely when the value is the missing sentinel. Note that for
    # the role field the base role is user, so the result may lack a role.
    message: dict[str, Any] = {"role": "user", "content": "same"}
    if value is _MISSING:
        message.pop(field)
    else:
        message[field] = value
    return message


def test_role_content_only_uses_the_existing_empty_value_class_for_each_projected_field() -> None:
    # [FUNCTION DOC] For each of role and content, over all pairs of empty values (missing, null,
    # empty string, empty list), the role/content-only matcher accepts.
    empty_values = (_MISSING, None, "", [])
    for field in ("role", "content"):
        for stored_value in empty_values:
            for replayed_value in empty_values:
                assert role_content_only_message_matches(
                    _projected_message(field, stored_value),
                    _projected_message(field, replayed_value),
                )


@pytest.mark.parametrize(
    ("field", "stored_value", "replayed_value"),
    [
        pytest.param("role", "assistant", "user", id="role"),
        pytest.param("content", "text", " text", id="leading-space"),
        pytest.param("content", "Text", "text", id="case"),
        pytest.param("content", "é", "e\u0301", id="unicode-normalization"),
        pytest.param("content", '{"x":1}', '{"x": 1}', id="json-format"),
        pytest.param("content", ["a", "b"], ["b", "a"], id="content-list-order"),
    ],
)
def test_role_content_only_does_not_add_other_normalization(
    field: str, stored_value: Any, replayed_value: Any
) -> None:
    # [FUNCTION DOC] Parametrized over differing role or content values (different role, leading
    # space, case, Unicode normalization form, JSON spacing, list order); the role/content-only
    # matcher rejects each (no extra normalization).
    assert not role_content_only_message_matches(
        _projected_message(field, stored_value),
        _projected_message(field, replayed_value),
    )


def test_role_content_only_ignores_every_non_projected_field_even_when_tool_calls_are_malformed() -> None:
    # [FUNCTION DOC] Two assistant messages with empty (null versus empty string) content but
    # completely different values for every other field (reasoning, tool calls including a
    # malformed string, names, ids, date, location, tools, response format, unknown); the
    # role/content-only matcher accepts.
    stored = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "stored reasoning",
        "tool_calls": [_tool_call('{"x":1}', call_id="A")],
        "name": "stored-name",
        "tool_call_id": "A",
        "current_date": "2026-01-01",
        "current_location": "Paris",
        "tools": [{"name": "stored"}],
        "response_format": {"type": "json_object"},
        "unknown": {"stored": True},
    }
    replayed = {
        "role": "assistant",
        "content": None,
        "reasoning_content": None,
        "tool_calls": "malformed and deliberately not parsed",
        "name": "replayed-name",
        "tool_call_id": "B",
        "current_date": "2030-12-31",
        "current_location": "Tokyo",
        "tools": None,
        "response_format": "text",
        "unknown": ["replayed"],
    }

    assert role_content_only_message_matches(stored, replayed)


def test_builtin_match_sets_are_monotonic() -> None:
    # [FUNCTION DOC] Over all ordered pairs of a small message set (user messages, several
    # assistant tool-call variants, a malformed one), asserts strict implies loose and loose implies
    # role/content-only (match sets are nested).
    messages = [
        {"role": "user", "content": "same"},
        {"role": "user", "content": "different"},
        _assistant('{"x":1}', index=0, reasoning_content=None),
        _assistant({"x": 1}, index=7),
        _assistant('{"x":2}', call_id="call-B", reasoning_content="changed"),
        {"role": "assistant", "content": "", "tool_calls": "malformed"},
    ]

    for stored in messages:
        for replayed in messages:
            strict = strict_message_matches(stored, replayed)
            loose = loose_tool_call_message_matches(stored, replayed)
            role_content = role_content_only_message_matches(stored, replayed)
            assert not strict or loose
            assert not loose or role_content


def test_matchers_do_not_mutate_either_input() -> None:
    # [FUNCTION DOC] Runs all three matchers in both argument orders on messages with nested
    # arguments and unknown fields, then asserts both messages equal their deep copies.
    stored = _assistant(
        {"nested": {"b": [2, 1], "a": True}},
        index=0,
        call_extra={"unknown": {"items": [1, 2]}},
    )
    replayed = _assistant(
        '{"nested":{"a":true,"b":[2,1]}}',
        index=5,
        call_extra={"unknown": {"items": [1, 2]}},
    )
    stored_before = copy.deepcopy(stored)
    replayed_before = copy.deepcopy(replayed)

    for matcher in (
        strict_message_matches,
        loose_tool_call_message_matches,
        role_content_only_message_matches,
    ):
        matcher(stored, replayed)
        matcher(replayed, stored)

    assert stored == stored_before
    assert replayed == replayed_before


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        pytest.param("strict", strict_message_matches, id="strict"),
        pytest.param("loose_tool_call", loose_tool_call_message_matches, id="loose-tool-call"),
        pytest.param("role_content_only", role_content_only_message_matches, id="role-content-only"),
    ],
)
def test_resolver_maps_builtin_aliases_by_identity(selector: str, expected: Any) -> None:
    # [FUNCTION DOC] Parametrized over the three alias names; the resolved matcher (a wrapper) wraps
    # exactly the corresponding built-in function object.
    assert resolve_session_message_matcher(selector).__wrapped__ is expected


def test_resolver_loads_a_synchronous_dotted_path() -> None:
    # [FUNCTION DOC] Resolving the dotted path of the standard equality operator function yields a
    # wrapper around that very function.
    assert resolve_session_message_matcher("operator.eq").__wrapped__ is operator.eq


class TestValidatedMessageMatcher:
    # [CLASS DOC] Tests of the contract wrapper that validates custom matcher results.
    def test_passes_through_exact_bool_results(self):
        # [FUNCTION DOC] Wrapped matchers returning true or false yield exactly those booleans.
        assert _validated(lambda stored, replayed: True)({}, {}) is True
        assert _validated(lambda stored, replayed: False)({}, {}) is False

    def test_wraps_matcher_exceptions(self):
        # [FUNCTION DOC] A matcher that raises a runtime error makes the wrapper raise the matcher
        # error type with a message saying the matcher raised an exception.
        def broken(stored, replayed):
            # [FUNCTION DOC] HELPER: Matcher that always raises a runtime error "boom".
            raise RuntimeError("boom")

        with pytest.raises(SessionMessageMatcherError, match="raised an exception"):
            _validated(broken)({}, {})

    def test_rejects_truthy_non_bool_results(self):
        # [FUNCTION DOC] A matcher returning the integer 1 makes the wrapper raise the matcher error
        # saying it must return bool, got int.
        with pytest.raises(SessionMessageMatcherError, match="must return bool, got int"):
            _validated(lambda stored, replayed: 1)({}, {})

    def test_resolver_applies_the_contract_check_to_custom_paths(self):
        # [FUNCTION DOC] Resolving the dotted path of the path-join function and calling it with two
        # strings raises the matcher error saying it must return bool, got str.
        wrapped = resolve_session_message_matcher("os.path.join")
        with pytest.raises(SessionMessageMatcherError, match="must return bool, got str"):
            wrapped("a", "b")


@pytest.mark.parametrize("selector", ["Strict", "LOOSE_TOOL_CALL", "role-content-only"])
def test_builtin_aliases_are_exact_and_case_sensitive(selector: str) -> None:
    # [FUNCTION DOC] Parametrized over wrongly cased or hyphenated aliases; resolving raises a
    # value error whose message lists the three valid aliases.
    with pytest.raises(ValueError) as exc_info:
        resolve_session_message_matcher(selector)

    message = str(exc_info.value)
    assert "strict" in message
    assert "loose_tool_call" in message
    assert "role_content_only" in message


@pytest.mark.parametrize(
    "selector",
    [
        pytest.param("missing_package.matcher", id="missing-module"),
        pytest.param("operator.missing_matcher", id="missing-attribute"),
        pytest.param("json.decoder.WHITESPACE", id="not-callable"),
        pytest.param("asyncio.sleep", id="async-callable"),
    ],
)
def test_resolver_wraps_invalid_custom_paths(selector: str) -> None:
    # [FUNCTION DOC] Parametrized over a missing module, a missing attribute, a non-callable and an
    # asynchronous callable; each raises a value error saying it failed to resolve the session
    # message matcher option.
    with pytest.raises(ValueError, match="failed to resolve --session-message-matcher"):
        resolve_session_message_matcher(selector)


def test_append_only_assert_defaults_to_strict_and_honors_the_configured_matcher() -> None:
    # [FUNCTION DOC] Stored has one assistant message (JSON string arguments), replay has the same
    # with dictionary arguments plus an allowed tool message. By default the append-only assertion
    # raises a value error about a mismatch at index 0; with the loose matcher passed it succeeds.
    stored = [_assistant('{"city":"Paris"}')]
    replayed = [_assistant({"city": "Paris"}), {"role": "tool", "content": "ok"}]

    with pytest.raises(ValueError, match="message mismatch at index 0"):
        message_matcher_hub.assert_messages_append_only_with_allowed_role(stored, replayed, {"tool"})

    message_matcher_hub.assert_messages_append_only_with_allowed_role(
        stored, replayed, {"tool"}, message_matcher=loose_tool_call_message_matches
    )


def test_append_only_assert_still_validates_suffix_roles_under_a_loose_matcher() -> None:
    # [FUNCTION DOC] With the loose matcher, an appended system message when only tool is allowed
    # still raises a value error about the appended message at index 1.
    stored = [_assistant('{"city":"Paris"}')]
    replayed = [_assistant({"city": "Paris"}), {"role": "system", "content": "injected"}]

    with pytest.raises(ValueError, match="appended message at index 1"):
        message_matcher_hub.assert_messages_append_only_with_allowed_role(
            stored, replayed, {"tool"}, message_matcher=loose_tool_call_message_matches
        )


def test_compatibility_exports_are_direct_aliases() -> None:
    # [FUNCTION DOC] Asserts the strict matcher and the append-only assertion are the same objects
    # when reached through the template module, the package and the hub; and that the package does
    # not export the loose matcher, role/content-only matcher or the resolver.
    assert template.strict_message_matches is message_matcher_hub.strict_message_matches
    assert chat_template_utils.strict_message_matches is message_matcher_hub.strict_message_matches
    assert (
        template.assert_messages_append_only_with_allowed_role
        is message_matcher_hub.assert_messages_append_only_with_allowed_role
    )
    assert (
        chat_template_utils.assert_messages_append_only_with_allowed_role
        is message_matcher_hub.assert_messages_append_only_with_allowed_role
    )
    assert not hasattr(chat_template_utils, "loose_tool_call_message_matches")
    assert not hasattr(chat_template_utils, "role_content_only_message_matches")
    assert not hasattr(chat_template_utils, "resolve_session_message_matcher")


def test_hub_import_keeps_the_function_registry_lazy_until_a_custom_path_is_resolved() -> None:
    # [FUNCTION DOC] Runs a fresh interpreter subprocess (120 second timeout, must exit
    # successfully) that checks: the function registry module is not loaded initially nor after
    # importing the hub nor after resolving the built-in strict alias; it is loaded after resolving a
    # custom dotted path; and the broader misc utilities module stays unloaded.
    code = """
import importlib
import sys

assert "miles.utils.function_registry" not in sys.modules
hub = importlib.import_module("miles.utils.chat_template_utils.message_matcher_hub")
assert "miles.utils.function_registry" not in sys.modules
assert hub.resolve_session_message_matcher("strict").__wrapped__ is hub.strict_message_matches
assert "miles.utils.function_registry" not in sys.modules
hub.resolve_session_message_matcher("operator.eq")
assert "miles.utils.function_registry" in sys.modules
assert "miles.utils.misc" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize(
    "first_import",
    [
        "miles.utils.chat_template_utils.message_matcher_hub",
        "miles.utils.chat_template_utils.template",
        "miles.utils.chat_template_utils",
        "miles.utils.chat_template_utils.tito_tokenizer",
    ],
)
def test_supported_import_orders_have_no_cycle(first_import: str) -> None:
    # [FUNCTION DOC] Parametrized over four first-imported modules (hub, template, package, TITO
    # tokenizer). In a fresh interpreter subprocess (120 second timeout) imports that module first
    # and then all four; the subprocess must exit successfully (no import cycle error).
    modules = (
        "miles.utils.chat_template_utils.message_matcher_hub",
        "miles.utils.chat_template_utils.template",
        "miles.utils.chat_template_utils",
        "miles.utils.chat_template_utils.tito_tokenizer",
    )
    code = f"import importlib\nimportlib.import_module({first_import!r})\nmodules = {modules!r}\nfor module in modules:\n    importlib.import_module(module)\n"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=120)
