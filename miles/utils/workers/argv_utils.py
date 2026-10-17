# [MODULE DOC] Helpers for building and checking command lines of worker processes. Covers: recovering
# the interpreter flags the current process was started with; passing a pydantic config as one JSON
# command-line option with a round-trip check; rendering a mapping of field values into argparse-style
# tokens for a given parser (skipping defaults, with a re-parse check that the result reproduces the
# expected object); parsing a command text against a parser while reporting declared destinations;
# coercing and validating a mapping of overrides against a parser's declared types, choices and
# allowed names; and context managers that temporarily suppress a parser's help option or relax its
# required arguments (both mutate private parser state and restore it). Light import footprint is
# required of this module (heavy packages are imported lazily).
import argparse
import contextlib
import dataclasses
import json
import shlex
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, NamedTuple, TypeVar

from miles.utils.pydantic_utils import FrozenStrictBaseModel

CONFIG_JSON_FLAG = "--config-json"

_INTERPRETER_SHORT_FLAGS_TAKING_A_VALUE = frozenset({"X", "W", "Q"})

_ConfigT = TypeVar("_ConfigT", bound=FrozenStrictBaseModel)
_ArgsT = TypeVar("_ArgsT")


def python_argv_prefix() -> list[str]:
    # [FUNCTION DOC] Returns the interpreter executable followed by the interpreter options the current
    # process was launched with (from the original argument vector), stopping at the script, module
    # or code selector. Algorithm over the tokens after the executable: stop at the first token that
    # does not start with a dash, or is a lone dash or double dash. For a single-dash cluster, scan
    # its letters: on the code (c) or module (m) selector, keep the preceding letters as one token if
    # any exist (e.g. a cluster that begins with other flags) and finish; on a letter of the
    # value-taking set (X, W, Q) keep the whole token, and if that letter is the last character also
    # keep the next token as its value, then stop scanning the cluster; a cluster with no special
    # letters is kept whole. Long options are kept as is; the hash-based pyc check option also keeps
    # its following value. Raises a stop-iteration error if a value-taking option has no value token.
    prefix = [sys.executable]
    tokens = iter(sys.orig_argv[1:])
    for token in tokens:
        if not token.startswith("-") or token in ("-", "--"):
            break
        if not token.startswith("--"):
            for index, flag in enumerate(token[1:], start=1):
                if flag in ("c", "m"):
                    # Keep flags before a clustered selector, e.g. -u in -ucCODE.
                    if index > 1:
                        prefix.append(token[:index])
                    return prefix
                if flag in _INTERPRETER_SHORT_FLAGS_TAKING_A_VALUE:
                    prefix.append(token)
                    if index == len(token) - 1:
                        prefix.append(next(tokens))
                    # The remainder is an option value, not more short flags.
                    break
            else:
                prefix.append(token)
            continue
        prefix.append(token)
        if token == "--check-hash-based-pycs":
            prefix.append(next(tokens))
    return prefix


# ==================== config argv ====================


def config_to_argv(config: FrozenStrictBaseModel) -> list[str]:
    # [FUNCTION DOC] Returns a two-element token list: the config-json flag and the config serialized as
    # a JSON string. Asserts that parsing the tokens back with the config's own class yields an equal
    # object (assertion error otherwise).
    argv = [CONFIG_JSON_FLAG, config.model_dump_json()]

    parsed = parse_config_argv(type(config), argv)
    assert parsed == config, f"config argv roundtrip mismatch: {parsed!r} != {config!r}"
    return argv


def parse_config_argv(config_cls: type[_ConfigT], argv: list[str] | None) -> _ConfigT:
    # [FUNCTION DOC] Parses a required config-json option from the tokens (none means the process
    # arguments; a missing option makes the parser exit) and validates its JSON text into an instance
    # of the given config class (validation errors propagate).
    parser = argparse.ArgumentParser()
    parser.add_argument(CONFIG_JSON_FLAG, required=True)
    args = parser.parse_args(argv)
    return config_cls.model_validate_json(args.config_json)


def dataclass_to_values(args_obj: object) -> dict[str, object]:
    # [FUNCTION DOC] Returns a mapping from each field name of the record (dataclass or msgspec struct,
    # class or instance) to the attribute value of the given object.
    return {name: getattr(args_obj, name) for name in _record_field_names(args_obj)}


def render_cli_argv(
    input_values: Mapping[str, object],
    *,
    expected_obj: _ArgsT,
    make_parser: Callable[[], argparse.ArgumentParser],
    from_parsed: Callable[[argparse.Namespace], _ArgsT],
    always_render_fields: Sequence[str] = (),
    field_to_dest: Mapping[str, str] | None = None,
    uncompared_fields: frozenset[str] = frozenset(),
) -> list[str]:
    # [FUNCTION DOC] Renders field values as a token list for the parser built by the factory.
    # Step one: for each always-render field name in order, renders the input value, or the expected
    # object's attribute when the input is absent or none. Step two: for every other input field
    # (skipping always-rendered ones and none values): finds the parser action (field name mapped to
    # destination through the optional mapping; assertion error if no option exists), skips values the
    # action cannot express (zero-argument flags whose value is not the flag's constant) and values
    # equal to the action's default, and renders the rest. Finally re-parses the tokens with a fresh
    # parser (a parser exit becomes an assertion error showing the command), converts with the
    # from-parsed callable and asserts no field of the expected object differs, ignoring the
    # uncompared fields (assertion error lists mismatches). Returns the tokens.
    actions_by_dest = _actions_by_dest(make_parser())
    field_to_dest = field_to_dest or {}

    def action_of(field_name: str) -> argparse.Action:
        # [FUNCTION DOC] HELPER: Resolves the parser action for a field name using the mapping.
        return _resolve_action(actions_by_dest, field_name=field_name, field_to_dest=field_to_dest or {})

    def render(field_name: str, value: object) -> list[str]:
        # [FUNCTION DOC] HELPER: Renders one field's value into tokens through its action.
        return _render_action_argv(action_of(field_name), value)

    argv = [
        token
        for name in always_render_fields
        for token in render(
            name,
            (
                input_values[name]
                if name in input_values and input_values[name] is not None
                else getattr(expected_obj, name)
            ),
        )
    ]
    for name, value in input_values.items():
        if name in always_render_fields or value is None:
            continue
        action = action_of(name)
        if not _is_renderable(action, value):
            continue
        if value == action.default:
            continue
        argv.extend(render(name, value))

    parsed = from_parsed(_parse_without_exiting(make_parser(), argv))
    mismatch = _describe_mismatch(parsed, expected_obj, uncompared_fields=uncompared_fields)
    assert not mismatch, f"cli argv roundtrip mismatch on {mismatch}"
    return argv


def _record_field_names(record) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Returns field names in declaration order: from the dataclass fields when
    # the argument is a dataclass (class or instance), otherwise from the msgspec struct fields,
    # importing msgspec lazily (fails with an import error or type error for other kinds).
    """Accept dataclass and msgspec Struct classes or instances across SGLang versions.

    msgspec is imported inside the branch that needs it: this module is on the light
    worker entrypoint's import path, whose footprint tests/fast/utils/workers/import_probe.py
    pins.
    """
    if dataclasses.is_dataclass(record):
        return tuple(field.name for field in dataclasses.fields(record))

    import msgspec

    return tuple(field.name for field in msgspec.structs.fields(record))


def _parse_without_exiting(parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Parses the tokens; converts the parser's exit request into an assertion
    # error that shows the shell-quoted command (the parser still prints its usage error to standard
    # error first).
    try:
        return parser.parse_args(argv)
    except SystemExit as exiting:
        raise AssertionError(f"the argument parser rejects the rendered {shlex.join(argv)}") from exiting


def _describe_mismatch(parsed: _ArgsT, wanted: _ArgsT, *, uncompared_fields: frozenset[str]) -> str:
    # [FUNCTION DOC] HELPER: Compares every field of the wanted record except the uncompared ones and
    # returns a comma-separated description of each differing field with both values; empty string
    # means equal.
    return ", ".join(
        f"{name}: parsed {getattr(parsed, name)!r} != wanted {getattr(wanted, name)!r}"
        for name in _record_field_names(wanted)
        if name not in uncompared_fields and getattr(parsed, name) != getattr(wanted, name)
    )


def _actions_by_dest(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    # [FUNCTION DOC] HELPER: Maps destination name to the first parser action registered with it
    # (reads the parser's private action list).
    actions_by_dest: dict[str, argparse.Action] = {}
    for action in parser._actions:
        actions_by_dest.setdefault(action.dest, action)
    return actions_by_dest


def _resolve_action(
    actions_by_dest: dict[str, argparse.Action],
    *,
    field_name: str,
    field_to_dest: Mapping[str, str],
) -> argparse.Action:
    # [FUNCTION DOC] HELPER: Maps the field name to a destination (identity by default) and returns that
    # destination's action if it has at least one option string; otherwise raises an assertion error
    # suggesting a destination mapping entry.
    dest = field_to_dest.get(field_name, field_name)
    action = actions_by_dest.get(dest)
    if action is not None and action.option_strings:
        return action

    raise AssertionError(
        f"{field_name!r} cannot be rendered: the parser registers no option for dest {dest!r}. "
        f"Add an entry to field_to_dest, or pass the value through the native passthrough path."
    )


def _is_renderable(action: argparse.Action, value: object) -> bool:
    # [FUNCTION DOC] HELPER: False for none; true for boolean-optional actions; otherwise true when
    # the action takes arguments, or (zero-argument flag) when the value equals the flag's constant.
    if value is None:
        return False
    if isinstance(action, argparse.BooleanOptionalAction):
        return True
    return action.nargs != 0 or value == action.const


def _render_action_argv(action: argparse.Action, value: object) -> list[str]:
    # [FUNCTION DOC] HELPER: Renders one value into tokens by action kind. Boolean-optional: the
    # positive or negative option string by truthiness. Zero-argument flag: nothing if the value equals
    # the default; else asserts the value equals the flag's constant and emits the long flag.
    # Append action: the flag repeated before the tokens of each item (list or tuple items spread
    # into several tokens). Variable or fixed-count arguments: the flag followed by key=value tokens
    # for a mapping, or the string form of each item. Otherwise a list, tuple or mapping is emitted as
    # one JSON string after the flag, and anything else as its string form. Uses the first long option
    # string, falling back to the first option string.
    if isinstance(action, argparse.BooleanOptionalAction):
        return [_boolean_option_string(action, value=bool(value))]

    if action.nargs == 0:
        if value == action.default:
            return []
        flag = _long_option_string(action)
        assert (
            value == action.const
        ), f"{flag} cannot be rendered: the CLI only has a flag for {action.const!r}, not {value!r}"
        return [flag]

    flag = _long_option_string(action)

    if isinstance(action, argparse._AppendAction):
        argv: list[str] = []
        for item in value:
            argv.append(flag)
            argv.extend(_scalar_tokens(item))
        return argv

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        if isinstance(value, dict):
            return [flag, *(f"{key}={item}" for key, item in value.items())]
        return [flag, *(str(item) for item in value)]

    if isinstance(value, dict | list | tuple):
        return [flag, json.dumps(value)]

    return [flag, str(value)]


def _scalar_tokens(item: object) -> list[str]:
    # [FUNCTION DOC] HELPER: A list or tuple becomes one string token per element; anything else a
    # single string token.
    if isinstance(item, list | tuple):
        return [str(element) for element in item]
    return [str(item)]


def _long_option_string(action: argparse.Action) -> str:
    # [FUNCTION DOC] HELPER: The first option string starting with a double dash, else the first option
    # string.
    long_options = [option for option in action.option_strings if option.startswith("--")]
    return long_options[0] if long_options else action.option_strings[0]


def _boolean_option_string(action: argparse.Action, *, value: bool) -> str:
    # [FUNCTION DOC] HELPER: For true returns the first option string not starting with the negation
    # prefix; for false the first starting with it; asserts one exists (assertion error otherwise).
    negative = [option for option in action.option_strings if option.startswith("--no-")]
    positive = [option for option in action.option_strings if not option.startswith("--no-")]
    if value:
        assert positive, f"{action.dest!r} cannot be rendered: no positive option string"
        return positive[0]
    assert negative, f"{action.dest!r} cannot be rendered: no negative option string"
    return negative[0]


# ==================== parser reflection ====================


def parse_declared_args(text: str, *, parser: argparse.ArgumentParser) -> dict[str, object]:
    # [FUNCTION DOC] Splits the text with shell rules and parses it leniently: required arguments are
    # temporarily made optional, and unknown tokens are collected. Asserts there are no unknown tokens
    # and that every double-dash token is a registered option string (so values starting with a double
    # dash would be misread). Returns a mapping from destination to parsed value for only the options
    # that appear in the text (abbreviated options are rejected by the exact-match assertion).
    tokens = shlex.split(text)
    with with_relax_parser_required_args(parser):
        namespace, unknown = parser.parse_known_args(tokens)
    assert not unknown, f"the argument parser does not declare {unknown} of {text!r}"

    action_by_option_string = parser._option_string_actions
    dests = []
    for token in tokens:
        if not token.startswith("--"):
            continue
        assert token in action_by_option_string, f"the argument parser does not declare {token!r}"
        dests.append(action_by_option_string[token].dest)
    return {dest: getattr(namespace, dest) for dest in dests}


def declared_arg_dests(parser: argparse.ArgumentParser) -> frozenset[str]:
    # [FUNCTION DOC] Returns the set of destination names of all the parser's actions (including help).
    return frozenset(action.dest for action in parser._actions)


def coerce_dict_to_args(
    values: Mapping[str, Any], *, parser: argparse.ArgumentParser, allowed_names: frozenset[str], context: str
) -> dict[str, Any]:
    # [FUNCTION DOC] Validates and types a mapping of override names to values against the parser.
    # Names may be option names (dashes as underscores) or destinations; each is mapped to its
    # destination. Asserts no two names map to the same destination (the message says only the last
    # would survive). Allowed names are mapped the same way. Each value is then coerced (see the
    # coerce helper) and the result is a mapping from destination to coerced value in input order. The
    # context string prefixes messages.
    dest_of_option_name = _compute_dest_of_option_names(parser)
    arg_specs = _compute_arg_specs(parser)
    allowed_dests = frozenset(dest_of_option_name.get(name, name) for name in allowed_names)

    dest_of_name = {name: dest_of_option_name.get(name, name) for name in values}
    aliased = {dest: names for dest, names in _invert_dict(dest_of_name).items() if len(names) > 1}
    assert not aliased, f"{context} names one argument twice: {aliased}; only the last of each group survives"

    return {
        (dest := dest_of_name[name]): _coerce_value(
            value, dest=dest, spec=arg_specs.get(dest), allowed_dests=allowed_dests, context=context
        )
        for name, value in values.items()
    }


def _invert_dict(mapping: dict[str, str]) -> dict[str, list[str]]:
    # [FUNCTION DOC] HELPER: Maps each value to the list of keys that have it, in key order.
    inverted: dict[str, list[str]] = {}
    for key, value in mapping.items():
        inverted.setdefault(value, []).append(key)
    return inverted


def _coerce_value(
    value: Any, *, dest: str, spec: "_ArgSpec | None", allowed_dests: frozenset[str], context: str
) -> Any:
    # [FUNCTION DOC] HELPER: Asserts, in order: the destination is allowed; the value is not none; the
    # parser declares an argument for it. A boolean-typed argument requires a real boolean and returns
    # it. A list value requires the argument to take several values and returns the list of coerced
    # elements. Anything else is coerced as one scalar. Every failure is an assertion error whose
    # message includes the context.
    assert dest in allowed_dests, (
        f"{context} sets {dest!r}, which it may not override; only these are allowed: {sorted(allowed_dests)}. "
        f"Everything else is read from the base command line, so setting it here would be silently ignored"
    )
    assert value is not None, f"{context} sets {dest!r} with no value"
    assert spec is not None, (
        f"{dest!r} is allowed, but the argument parser declares no such argument, so the value {value!r} "
        f"of {context} cannot be typed"
    )

    if spec.type is bool:
        assert isinstance(value, bool), f"{context} sets {dest!r} to {value!r}, which is not a boolean"
        return value

    if isinstance(value, list):
        assert _takes_several_values(
            spec
        ), f"{context} sets {dest!r} to the list {value!r}, but the command line takes a single value there"
        return [_coerce_one_value(item, dest=dest, spec=spec, context=context) for item in value]

    return _coerce_one_value(value, dest=dest, spec=spec, context=context)


def _takes_several_values(spec: "_ArgSpec") -> bool:
    # [FUNCTION DOC] HELPER: True when the argument takes zero-or-more, one-or-more, or an integer
    # count above one of values.
    return spec.nargs in ("*", "+") or (isinstance(spec.nargs, int) and spec.nargs > 1)


def _coerce_one_value(value: Any, *, dest: str, spec: "_ArgSpec", context: str) -> Any:
    # [FUNCTION DOC] HELPER: Asserts the value is an integer, float or string and not a boolean;
    # converts it through the argument's type from its string form; asserts the converted value is in
    # the argument's choices when choices exist. Returns the converted value.
    assert not isinstance(value, bool) and isinstance(
        value, (int, float, str)
    ), f"{context} sets {dest!r} to {value!r}, which is not a {spec.type.__name__}"
    coerced = _coerce_scalar(value, dest=dest, spec=spec, context=context)
    assert (
        spec.choices is None or coerced in spec.choices
    ), f"{context} sets {dest!r} to {value!r}, but the command line only accepts {list(spec.choices)}"
    return coerced


def _coerce_scalar(value: int | float | str, *, dest: str, spec: "_ArgSpec", context: str) -> Any:
    # [FUNCTION DOC] HELPER: Applies the argument's type callable to the string form of the value; a
    # value error from it becomes an assertion error saying the command line would reject it. Other
    # exception kinds from custom type callables propagate.
    try:
        return spec.type(str(value))
    except ValueError as exception:
        raise AssertionError(
            f"{context} sets {dest!r} to {value!r}, which the command line would reject: {exception}"
        ) from exception


class _ArgSpec(NamedTuple):
    # [CLASS DOC] HELPER: Summary of one parser argument: destination, value type callable, tuple of
    # allowed choices (or none), and the number-of-values specification (or none).
    dest: str
    type: type
    choices: tuple[Any, ...] | None
    nargs: int | str | None


def _compute_arg_specs(parser: argparse.ArgumentParser) -> dict[str, _ArgSpec]:
    # [FUNCTION DOC] HELPER: Maps destination to a spec for each action (later actions with the same
    # destination overwrite earlier ones).
    return {action.dest: _compute_arg_spec(action) for action in parser._actions}


def _compute_dest_of_option_names(parser: argparse.ArgumentParser) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Maps each double-dash option string, with the dashes prefix removed and
    # remaining hyphens turned into underscores, to the destination of its action.
    return {
        option.removeprefix("--").replace("-", "_"): action.dest
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }


def _compute_arg_spec(action: argparse.Action) -> _ArgSpec:
    # [FUNCTION DOC] HELPER: Builds a spec from the action's destination, inferred type, choices
    # (converted to a tuple) and nargs.
    choices = None if action.choices is None else tuple(action.choices)
    return _ArgSpec(dest=action.dest, type=_compute_arg_type(action), choices=choices, nargs=action.nargs)


@contextlib.contextmanager
def with_suppressed_parser_help(parser: argparse.ArgumentParser) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that removes the help actions' option strings from the parser's
    # private option lookup table for the duration (so those flags are treated as unknown), and
    # restores them afterwards even on errors. Mutates parser internals; not thread-safe.
    suppressed = {
        option: action
        for option, action in parser._option_string_actions.items()
        if isinstance(action, argparse._HelpAction)
    }
    for option in suppressed:
        del parser._option_string_actions[option]
    try:
        yield
    finally:
        parser._option_string_actions.update(suppressed)


@contextlib.contextmanager
def with_relax_parser_required_args(parser: argparse.ArgumentParser) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that marks every currently required action as optional for the
    # duration and restores the flag afterwards even on errors. Mutates the parser's actions; not
    # thread-safe.
    required = [action for action in parser._actions if action.required]
    for action in required:
        action.required = False
    try:
        yield
    finally:
        for action in required:
            action.required = True


def _compute_arg_type(action: argparse.Action) -> type:
    # [FUNCTION DOC] HELPER: Boolean for store-true, store-false and boolean-optional actions; string
    # when the action has no type; otherwise the action's type callable.
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        return bool
    if action.type is None:
        return str
    return action.type
