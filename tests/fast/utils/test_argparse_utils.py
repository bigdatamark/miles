# [MODULE DOC]
# Tests for the dataclass-to-argparse bridge and the in-place argument override context manager.
# The bridge registers a frozen dataclass's fields as command line options (with an optional name
# prefix and an argument group title), builds a dataclass instance back from a parsed namespace, and
# serializes an instance to a command line string. Covers required vs defaulted fields, booleans as
# flags, optional values, round trips, and validation of unsupported types. The override context
# manager temporarily changes attributes of a namespace and restores them on exit.
"""Tests for DataclassArgparseBridge."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from miles.utils.argparse_utils import DataclassArgparseBridge, inplace_modify_args


@dataclasses.dataclass(frozen=True)
class _SampleArgs:
    # [CLASS DOC] HELPER: Frozen sample dataclass used with the bridge: three required fields (string,
    # integer, float), three defaulted fields (string "default", integer 10, float 0.5), a boolean
    # flag defaulting to false, and three optional fields (string, integer, float) defaulting to none.
    mandatory_str: str
    mandatory_int: int
    mandatory_float: float
    defaulted_str: str = "default"
    defaulted_int: int = 10
    defaulted_float: float = 0.5
    flag: bool = False
    optional_str: str | None = None
    optional_int: int | None = None
    optional_float: float | None = None


_BRIDGE: DataclassArgparseBridge[_SampleArgs] = DataclassArgparseBridge(
    _SampleArgs,
    prefix="sample",
    group_title="sample args",
)


@dataclasses.dataclass(frozen=True)
class _NoPrefixArgs:
    # [CLASS DOC] HELPER: Frozen sample dataclass with a required string name and a boolean debug flag
    # defaulting to false; used with a bridge having an empty option prefix.
    name: str
    debug: bool = False


_NO_PREFIX_BRIDGE: DataclassArgparseBridge[_NoPrefixArgs] = DataclassArgparseBridge(
    _NoPrefixArgs,
    prefix="",
)


class TestRegisterOnParser:
    # [CLASS DOC] Tests of registering dataclass fields as parser options.
    def test_required_fields_are_required(self) -> None:
        # [FUNCTION DOC] Parsing an empty argument list after registering the sample fields exits the
        # parser (system exit), since the required fields are missing.
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_all_field_types(self) -> None:
        # [FUNCTION DOC] Passes every field with the sample prefix (options named with the prefix and
        # dashes) and asserts each namespace attribute (prefix plus underscored field name) holds the
        # given value converted to its declared type; the flag becomes true.
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(
            [
                "--sample-mandatory-str",
                "hello",
                "--sample-mandatory-int",
                "42",
                "--sample-mandatory-float",
                "3.14",
                "--sample-defaulted-str",
                "custom",
                "--sample-defaulted-int",
                "99",
                "--sample-defaulted-float",
                "1.5",
                "--sample-flag",
                "--sample-optional-str",
                "mytag",
                "--sample-optional-int",
                "100",
                "--sample-optional-float",
                "0.9",
            ]
        )

        assert namespace.sample_mandatory_str == "hello"
        assert namespace.sample_mandatory_int == 42
        assert namespace.sample_mandatory_float == 3.14
        assert namespace.sample_defaulted_str == "custom"
        assert namespace.sample_defaulted_int == 99
        assert namespace.sample_defaulted_float == 1.5
        assert namespace.sample_flag is True
        assert namespace.sample_optional_str == "mytag"
        assert namespace.sample_optional_int == 100
        assert namespace.sample_optional_float == 0.9

    def test_defaults_are_applied(self) -> None:
        # [FUNCTION DOC] Passing only the three required options leaves defaulted fields at their
        # defaults, the flag false and all optional fields none.
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(
            [
                "--sample-mandatory-str",
                "test",
                "--sample-mandatory-int",
                "1",
                "--sample-mandatory-float",
                "2.0",
            ]
        )

        assert namespace.sample_defaulted_str == "default"
        assert namespace.sample_defaulted_int == 10
        assert namespace.sample_defaulted_float == 0.5
        assert namespace.sample_flag is False
        assert namespace.sample_optional_str is None
        assert namespace.sample_optional_int is None
        assert namespace.sample_optional_float is None

    def test_no_prefix(self) -> None:
        # [FUNCTION DOC] With an empty prefix the options are plain field names (name and debug) and
        # the namespace attributes carry no prefix.
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _NO_PREFIX_BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(["--name", "foo", "--debug"])
        assert namespace.name == "foo"
        assert namespace.debug is True


class TestFromNamespace:
    # [CLASS DOC] Tests of building a dataclass instance from a parsed namespace.
    def test_constructs_dataclass_from_namespace(self) -> None:
        # [FUNCTION DOC] Parses a mix of required, flag and one optional option, builds the instance,
        # and asserts the fields carry the parsed values, defaults and none for unspecified optionals.
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(
            [
                "--sample-mandatory-str",
                "world",
                "--sample-mandatory-int",
                "7",
                "--sample-mandatory-float",
                "1.0",
                "--sample-flag",
                "--sample-optional-str",
                "x",
            ]
        )

        instance: _SampleArgs = _BRIDGE.from_namespace(namespace)

        assert instance.mandatory_str == "world"
        assert instance.mandatory_int == 7
        assert instance.mandatory_float == 1.0
        assert instance.defaulted_str == "default"
        assert instance.flag is True
        assert instance.optional_str == "x"
        assert instance.optional_int is None

    def test_no_prefix_from_namespace(self) -> None:
        # [FUNCTION DOC] With the no-prefix bridge, parses a name only and asserts the built instance
        # has that name and a false debug flag.
        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _NO_PREFIX_BRIDGE.register_on_parser(parser)

        namespace: argparse.Namespace = parser.parse_args(["--name", "bar"])
        instance: _NoPrefixArgs = _NO_PREFIX_BRIDGE.from_namespace(namespace)

        assert instance.name == "bar"
        assert instance.debug is False


class TestToCliArgs:
    # [CLASS DOC] Tests of serializing a dataclass instance to a command line string.
    def test_serializes_all_types(self) -> None:
        # [FUNCTION DOC] Serializes a fully populated sample instance and asserts the string contains
        # each prefixed option followed by its value, and the bare flag option for the true boolean.
        instance: _SampleArgs = _SampleArgs(
            mandatory_str="hello",
            mandatory_int=42,
            mandatory_float=3.14,
            flag=True,
            optional_str="mytag",
            optional_int=100,
            optional_float=0.9,
        )

        cli: str = _BRIDGE.to_cli_args(instance)

        assert "--sample-mandatory-str hello" in cli
        assert "--sample-mandatory-int 42" in cli
        assert "--sample-mandatory-float 3.14" in cli
        assert "--sample-flag" in cli
        assert "--sample-optional-str mytag" in cli
        assert "--sample-optional-int 100" in cli
        assert "--sample-optional-float 0.9" in cli

    def test_skips_none_and_false(self) -> None:
        # [FUNCTION DOC] Serializes an instance with only the required fields set; asserts the string
        # omits the flag option (false boolean) and all three optional options (none values).
        instance: _SampleArgs = _SampleArgs(
            mandatory_str="test",
            mandatory_int=1,
            mandatory_float=2.0,
        )

        cli: str = _BRIDGE.to_cli_args(instance)

        assert "--sample-flag" not in cli
        assert "--sample-optional-str" not in cli
        assert "--sample-optional-int" not in cli
        assert "--sample-optional-float" not in cli

    def test_no_prefix_serialization(self) -> None:
        # [FUNCTION DOC] With the no-prefix bridge, an instance with name "foo" and debug true
        # serializes to a string containing "--name foo" and the bare debug option.
        instance: _NoPrefixArgs = _NoPrefixArgs(name="foo", debug=True)
        cli: str = _NO_PREFIX_BRIDGE.to_cli_args(instance)

        assert "--name foo" in cli
        assert "--debug" in cli


class TestRoundTrip:
    # [CLASS DOC] Tests that serializing then parsing then rebuilding yields an equal instance.
    def test_to_cli_args_then_parse_back(self) -> None:
        # [FUNCTION DOC] A fully populated instance is serialized to a string, split on whitespace,
        # parsed by a fresh parser with the fields registered, and rebuilt; the result equals the
        # original.
        original: _SampleArgs = _SampleArgs(
            mandatory_str="roundtrip",
            mandatory_int=99,
            mandatory_float=2.5,
            defaulted_str="custom",
            defaulted_int=77,
            defaulted_float=1.1,
            flag=True,
            optional_str="t",
            optional_int=50,
            optional_float=0.1,
        )

        cli: str = _BRIDGE.to_cli_args(original)

        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)
        namespace: argparse.Namespace = parser.parse_args(cli.split())

        restored: _SampleArgs = _BRIDGE.from_namespace(namespace)
        assert restored == original

    def test_round_trip_with_defaults(self) -> None:
        # [FUNCTION DOC] Same round trip for an instance with only required fields set (defaults and
        # none optionals elsewhere); the rebuilt instance equals the original.
        original: _SampleArgs = _SampleArgs(
            mandatory_str="minimal",
            mandatory_int=1,
            mandatory_float=2.0,
        )

        cli: str = _BRIDGE.to_cli_args(original)

        parser: argparse.ArgumentParser = argparse.ArgumentParser()
        _BRIDGE.register_on_parser(parser)
        namespace: argparse.Namespace = parser.parse_args(cli.split())

        restored: _SampleArgs = _BRIDGE.from_namespace(namespace)
        assert restored == original


class TestValidation:
    # [CLASS DOC] Tests of input validation by the bridge.
    def test_rejects_non_dataclass(self) -> None:
        # [FUNCTION DOC] Constructing the bridge with a non-dataclass type (the string type) raises a
        # type error matching "not a dataclass".
        with pytest.raises(TypeError, match="not a dataclass"):
            DataclassArgparseBridge(str, prefix="x")

    def test_rejects_unsupported_type(self) -> None:
        # [FUNCTION DOC] A dataclass with a list-of-strings field constructs a bridge fine, but
        # registering it on a parser raises a type error matching "Unsupported field type".
        @dataclasses.dataclass(frozen=True)
        class _Bad:
            # [CLASS DOC] HELPER: Frozen dataclass with one list-of-strings field (default empty
            # list), a type the bridge does not support.
            data: list[str] = dataclasses.field(default_factory=list)

        bridge: DataclassArgparseBridge[_Bad] = DataclassArgparseBridge(_Bad, prefix="bad")
        parser: argparse.ArgumentParser = argparse.ArgumentParser()

        with pytest.raises(TypeError, match="Unsupported field type"):
            bridge.register_on_parser(parser)


class TestInplaceModifyArgs:
    # [CLASS DOC] Tests of the in-place override context manager on a namespace.
    def test_overrides_inside_and_restores_on_exit(self) -> None:
        # [FUNCTION DOC] Overrides two boolean attributes to false; inside the block they are false
        # and an untouched attribute keeps its value; after the block the originals (true) are back.
        """Overridden attributes are visible inside the block and restored afterwards."""
        args = argparse.Namespace(no_load_optim=True, finetune=True, lr=1.0)

        with inplace_modify_args(args, dict(no_load_optim=False, finetune=False)):
            assert args.no_load_optim is False
            assert args.finetune is False
            assert args.lr == 1.0

        assert args.no_load_optim is True
        assert args.finetune is True

    def test_restores_on_exception(self) -> None:
        # [FUNCTION DOC] A runtime error raised inside the block propagates and the original value
        # (true) is restored.
        """Originals are restored even when the block raises."""
        args = argparse.Namespace(flag=True)

        with pytest.raises(RuntimeError):
            with inplace_modify_args(args, dict(flag=False)):
                raise RuntimeError("boom")

        assert args.flag is True

    def test_empty_overrides_is_noop(self) -> None:
        # [FUNCTION DOC] With an empty override mapping the attribute is unchanged inside and after
        # the block.
        """An empty override dict changes nothing."""
        args = argparse.Namespace(flag=True)

        with inplace_modify_args(args, {}):
            assert args.flag is True

        assert args.flag is True

    def test_unknown_attribute_raises(self) -> None:
        # [FUNCTION DOC] Overriding an attribute missing from an empty namespace raises an attribute
        # error when entering the block.
        """Overriding an attribute the namespace does not have fails loudly."""
        args = argparse.Namespace()

        with pytest.raises(AttributeError):
            with inplace_modify_args(args, dict(missing=1)):
                pass

    def test_mutation_inside_block_fails_on_exit(self) -> None:
        # [FUNCTION DOC] Overrides a flag to false and then sets it back to true inside the block;
        # leaving the block raises an assertion error matching "modified inside".
        """An attribute mutated inside the block is detected instead of silently clobbered."""
        args = argparse.Namespace(flag=True)

        with pytest.raises(AssertionError, match="modified inside"):
            with inplace_modify_args(args, dict(flag=False)):
                args.flag = True
