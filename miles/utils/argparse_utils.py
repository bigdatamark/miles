# [MODULE DOC] Bridge between flat dataclasses and argparse. A generic converter class registers
# one command-line option per dataclass field on a parser (with an optional name prefix), rebuilds
# a dataclass instance from a parsed namespace, and renders an instance back to a command-line
# string. Also provides a context manager that temporarily overrides namespace attributes.
# Supported field types: string, integer, float, path, boolean, and each of the non-boolean ones
# combined with None. Anything else raises a type error at registration time.

"""Dataclass ↔ argparse bridge.

Supported field types: str, int, float, Path, bool, and their ``X | None``
variants (except ``bool | None`` which is not supported).
"""

from __future__ import annotations

import argparse
import dataclasses
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Generic, TypeVar, get_type_hints

T = TypeVar("T")

_SCALAR_TYPES: dict[type, type] = {str: str, int: int, float: float, Path: Path}


def _is_bool(tp: type) -> bool:
    # [FUNCTION DOC] HELPER: True only when the annotation is exactly the boolean type.
    return tp is bool


def _is_optional(tp: type) -> tuple[bool, type | None]:
    # [FUNCTION DOC] HELPER: Detects an "X or None" union written with the pipe syntax. Returns a
    # pair (true, X) when the annotation is such a union with exactly one non-None member;
    # otherwise (false, None). Note that unions spelled with the typing Optional helper are not
    # recognized (only the new-style union type is), and that they therefore later fail as
    # unsupported types.
    if not isinstance(tp, types.UnionType):
        return False, None

    args: tuple[type, ...] = tp.__args__
    non_none: list[type] = [a for a in args if a is not type(None)]
    if type(None) not in args or len(non_none) != 1:
        return False, None

    return True, non_none[0]


def _resolve_default(field: dataclasses.Field[object]) -> object:
    # [FUNCTION DOC] HELPER: Returns the field's plain default if it has one; else calls the
    # default factory (with no arguments) and returns its result; else returns the dataclass
    # missing sentinel.
    """Return the effective default for a dataclass field, or ``MISSING``."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return dataclasses.MISSING


class DataclassArgparseBridge(Generic[T]):
    # [CLASS DOC] Converter bound to one dataclass type. Constructor inputs: the dataclass type
    # (must be a dataclass, else type error), a required keyword-only prefix string, and an
    # optional group title (defaults to the class name followed by "args"). It caches the resolved
    # type hints of the dataclass. Flag name for a field: replace underscores with hyphens, and
    # prepend two dashes plus the prefix and a hyphen when the prefix is non-empty. The namespace
    # attribute name (dest) is prefix, underscore, field name, or just the field name when the
    # prefix is empty.
    """Bi-directional converter: dataclass ↔ argparse.

    *prefix* controls the CLI flag prefix: ``"script"`` → ``--script-field-name``,
    ``""`` → ``--field-name``.
    """

    def __init__(
        self,
        dataclass_type: type[T],
        *,
        prefix: str,
        group_title: str | None = None,
    ) -> None:
        if not dataclasses.is_dataclass(dataclass_type):
            raise TypeError(f"{dataclass_type!r} is not a dataclass")

        self._cls: type[T] = dataclass_type
        self._prefix: str = prefix
        self._group_title: str = group_title or f"{dataclass_type.__name__} args"
        self._hints: dict[str, type] = get_type_hints(dataclass_type)

    def _flag(self, field_name: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the command-line flag for a field per the class rule.
        stem: str = field_name.replace("_", "-")
        if self._prefix:
            return f"--{self._prefix}-{stem}"
        return f"--{stem}"

    def _dest(self, field_name: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the namespace attribute name for a field per the class
        # rule.
        if self._prefix:
            return f"{self._prefix}_{field_name}"
        return field_name

    def register_on_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # [FUNCTION DOC] Adds an argument group (with the stored title) to the parser and one
        # option per dataclass field, in field order. Booleans become store-true flags defaulting
        # to false (the dataclass default is ignored). A scalar-or-None field becomes an option of
        # the scalar type defaulting to None (the dataclass default is ignored). A plain scalar
        # field becomes an option of that type, using the dataclass default (plain or factory
        # result) when there is one and being required otherwise. Raises a type error for an
        # unsupported optional inner type or any other unsupported annotation. Mutates and returns
        # the same parser. An optional boolean is rejected via the unsupported optional inner type
        # error.
        group: argparse._ArgumentGroup = parser.add_argument_group(self._group_title)

        for field in dataclasses.fields(self._cls):
            flag: str = self._flag(field.name)
            dest: str = self._dest(field.name)
            tp: type = self._hints[field.name]

            if _is_bool(tp):
                group.add_argument(flag, dest=dest, action="store_true", default=False)
                continue

            is_opt, inner = _is_optional(tp)
            if is_opt:
                if inner not in _SCALAR_TYPES:
                    raise TypeError(f"Unsupported optional inner type {inner!r} for field {field.name}")
                group.add_argument(flag, dest=dest, type=_SCALAR_TYPES[inner], default=None)
                continue

            if tp in _SCALAR_TYPES:
                default: object = _resolve_default(field)
                kwargs: dict[str, object] = {"dest": dest, "type": _SCALAR_TYPES[tp]}
                if default is not dataclasses.MISSING:
                    kwargs["default"] = default
                else:
                    kwargs["required"] = True
                group.add_argument(flag, **kwargs)  # type: ignore[arg-type]
                continue

            raise TypeError(f"Unsupported field type {tp!r} for field {field.name}")

        return parser

    def from_namespace(self, namespace: argparse.Namespace) -> T:
        # [FUNCTION DOC] Builds a dataclass instance by reading, for every field, the namespace
        # attribute with the field's dest name. Raises an attribute error if one is missing.
        kwargs: dict[str, object] = {}
        for field in dataclasses.fields(self._cls):
            kwargs[field.name] = getattr(namespace, self._dest(field.name))
        return self._cls(**kwargs)  # type: ignore[call-arg]

    def to_cli_args(self, instance: T) -> str:
        # [FUNCTION DOC] Renders an instance as a space-joined command-line string in field order.
        # A boolean field emits its bare flag only when true. Any other field emits the flag, a
        # space and the value's string form unless the value is None (skipped). Values are not
        # shell-quoted, so values containing spaces or special characters will break when the
        # string is later split.
        parts: list[str] = []

        for field in dataclasses.fields(self._cls):  # type: ignore[arg-type]
            value: object = getattr(instance, field.name)
            flag: str = self._flag(field.name)
            tp: type = self._hints[field.name]

            if _is_bool(tp):
                if value:
                    parts.append(flag)
            elif value is not None:
                parts.append(f"{flag} {value}")

        return " ".join(parts)


@contextmanager
def inplace_modify_args(args: argparse.Namespace, overrides: dict[str, object]) -> Iterator[None]:
    # [FUNCTION DOC] Context manager that mutates the namespace in place. On entry it records the
    # current value of every overridden key (attribute error if a key does not exist yet) and sets
    # the new values. On exit (even after an exception in the body) it checks each key still holds
    # the override value and then restores the old value; a mismatch triggers an assertion failure
    # with a message naming the key. Yields nothing. Not thread safe.
    # KNOWN ISSUE: the assertion aborts the restore loop midway, leaving the remaining keys
    # unrestored, and it can mask an exception raised in the body.
    """Temporarily set attributes on ``args``, restoring the originals on exit."""
    old_values = {key: getattr(args, key) for key in overrides}
    for key, value in overrides.items():
        setattr(args, key, value)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            current = getattr(args, key)
            assert current == overrides[key], (
                f"args.{key} was modified inside the inplace_modify_args block "
                f"(expected {overrides[key]!r}, found {current!r}); restoring would clobber it"
            )
            setattr(args, key, old_value)
