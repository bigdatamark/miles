# [MODULE DOC] Unit tests for the dataclass command-line decorator, which turns a function taking
# one dataclass argument into a command whose options come from the dataclass fields. Tests use
# the typer command runner to invoke small commands built per test, and cover: environment
# variable defaults with a configurable prefix (default prefix for script variables), option
# overrides, help text from field metadata, boolean on/off flags, float, optional string and enum
# fields, default factories, required fields, return value passthrough, the dataclass instance
# type received, wrapped function attributes, errors for non-dataclass annotations, and
# environment variable naming. Module level: a shared runner and several small private dataclasses
# (and one string enum) that serve as test inputs. The inspect module is imported at the bottom of
# the file for the signature test.
import dataclasses
import enum

import pytest
import typer
from typer.testing import CliRunner

from miles.utils.typer_utils import dataclass_cli

runner = CliRunner()

# ---------------------------------------------------------------------------
# Test dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _SimpleArgs:
    # [CLASS DOC] HELPER: Required string name plus an integer count defaulting to 1.
    name: str
    count: int = 1


@dataclasses.dataclass
class _BoolArgs:
    # [CLASS DOC] HELPER: A single boolean verbose flag defaulting to false.
    verbose: bool = False


@dataclasses.dataclass
class _MetaHelpArgs:
    # [CLASS DOC] HELPER: An integer level (default 0) carrying help text in its metadata.
    level: int = dataclasses.field(
        default=0,
        metadata={"help": "Verbosity level"},
    )


@dataclasses.dataclass
class _MetaHelpAndFlagArgs:
    # [CLASS DOC] HELPER: A boolean verbose flag (default false) carrying help text in metadata.
    verbose: bool = dataclasses.field(
        default=False,
        metadata={"help": "Enable verbose output"},
    )


@dataclasses.dataclass
class _MultiFieldArgs:
    # [CLASS DOC] HELPER: Host string (default localhost), port integer (default 8080) and debug
    # boolean (default false).
    host: str = "localhost"
    port: int = 8080
    debug: bool = False


@dataclasses.dataclass
class _AllRequiredArgs:
    # [CLASS DOC] HELPER: Two required fields: a string and an integer, with no defaults.
    first: str
    second: int


@dataclasses.dataclass
class _SingleFieldArgs:
    # [CLASS DOC] HELPER: One string field defaulting to the word default.
    value: str = "default"


@dataclasses.dataclass
class _FloatArgs:
    # [CLASS DOC] HELPER: One float field rate defaulting to 0.01.
    rate: float = 0.01


@dataclasses.dataclass
class _OptionalStrArgs:
    # [CLASS DOC] HELPER: One optional string tag defaulting to none.
    tag: str | None = None


class _Color(str, enum.Enum):
    # [CLASS DOC] HELPER: String enum with members red, green and blue (values are the lowercase
    # color names).
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclasses.dataclass
class _EnumArgs:
    # [CLASS DOC] HELPER: One enum field color defaulting to red.
    color: _Color = _Color.RED


@dataclasses.dataclass
class _FieldWithDefaultNoMeta:
    # [CLASS DOC] HELPER: An integer field declared through the field helper with default 5 and
    # no metadata.
    count: int = dataclasses.field(default=5)


@dataclasses.dataclass
class _DefaultFactoryArgs:
    # [CLASS DOC] HELPER: A string field whose default comes from a factory returning a,b,c.
    items: str = dataclasses.field(default_factory=lambda: "a,b,c")


@dataclasses.dataclass
class _MixedMetaArgs:
    # [CLASS DOC] HELPER: Four fields mixing styles: a plain string default, an integer with help
    # metadata, a string declared through the field helper without metadata, and a boolean with
    # help metadata.
    plain: str = "hello"
    with_help: int = dataclasses.field(
        default=0,
        metadata={"help": "A number"},
    )
    with_flag: str = dataclasses.field(
        default="x",
    )
    with_both: bool = dataclasses.field(
        default=False,
        metadata={"help": "Toggle it"},
    )


@dataclasses.dataclass
class _EmptyMetaArgs:
    # [CLASS DOC] HELPER: A string field with default ok and an explicitly empty metadata mapping.
    value: str = dataclasses.field(default="ok", metadata={})


@dataclasses.dataclass
class _MultipleRequiredArgs:
    # [CLASS DOC] HELPER: Three required string fields alpha, beta and gamma.
    alpha: str
    beta: str
    gamma: str


# ---------------------------------------------------------------------------
# Bare decorator: @dataclass_cli
# ---------------------------------------------------------------------------


class TestBareDecorator:
    # [CLASS DOC] Tests of the decorator used without parentheses (default environment variable
    # prefix for script options). Each test builds its own app and command, and the command
    # prints the received fields separated by a bar.
    def test_env_vars(self) -> None:
        # [FUNCTION DOC] Invokes with no options but with script-prefixed name and count
        # environment variables; asserts exit code 0 and the output shows the environment values.
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, [], env={"MILES_SCRIPT_NAME": "EnvName", "MILES_SCRIPT_COUNT": "10"})
        assert result.exit_code == 0
        assert "EnvName|10" in result.stdout

    def test_cli_flag_overrides_env_var(self) -> None:
        # [FUNCTION DOC] Sets the name through the environment and count through the command line
        # flag; asserts exit code 0 and that the output uses the environment name with the flag
        # count 999.
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, ["--count", "999"], env={"MILES_SCRIPT_NAME": "EnvName"})
        assert result.exit_code == 0
        assert "EnvName|999" in result.stdout

    def test_default_value_used_when_omitted(self) -> None:
        # [FUNCTION DOC] Passes only the name flag; asserts exit code 0 and that the count takes
        # its default of 1.
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, ["--name", "Alice"])
        assert result.exit_code == 0
        assert "Alice|1" in result.stdout

    def test_all_flags_explicit(self) -> None:
        # [FUNCTION DOC] Passes both flags; asserts exit code 0 and the given values in output.
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, ["--name", "Bob", "--count", "42"])
        assert result.exit_code == 0
        assert "Bob|42" in result.stdout

    def test_missing_required_field_fails(self) -> None:
        # [FUNCTION DOC] Passes no options and no environment; asserts a non-zero exit code since
        # the name is required.
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name field.
            print(args.name)

        result = runner.invoke(app, [])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Parameterized decorator: @dataclass_cli(env_var_prefix=...)
# ---------------------------------------------------------------------------


class TestParameterizedDecorator:
    # [CLASS DOC] Tests of the decorator called with an explicit environment variable prefix
    # (empty disables environment variables).
    def test_empty_prefix_disables_env_vars(self) -> None:
        # [FUNCTION DOC] With an empty prefix, a script-prefixed environment variable must be
        # ignored; asserts a non-zero exit code because the required name is still missing.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(
            app,
            [],
            env={"MILES_SCRIPT_NAME": "ShouldBeIgnored"},
        )
        assert result.exit_code != 0

    def test_custom_prefix(self) -> None:
        # [FUNCTION DOC] With a custom prefix, environment variables using that prefix supply the
        # name and count; asserts exit code 0 and the values in the output.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="MY_APP_")
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, [], env={"MY_APP_NAME": "Custom", "MY_APP_COUNT": "77"})
        assert result.exit_code == 0
        assert "Custom|77" in result.stdout

    def test_custom_prefix_ignores_default_prefix(self) -> None:
        # [FUNCTION DOC] With a custom prefix, a variable using the default script prefix is not
        # read; asserts a non-zero exit code (name missing).
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="MY_APP_")
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, [], env={"MILES_SCRIPT_NAME": "Wrong"})
        assert result.exit_code != 0

    def test_default_prefix_when_called_with_parens(self) -> None:
        # [FUNCTION DOC] Uses the decorator with empty parentheses and script-prefixed
        # environment variables; asserts it behaves like the bare form (exit code 0, values read).
        """@dataclass_cli() with no args should behave same as bare @dataclass_cli."""
        app = typer.Typer()

        @app.command()
        @dataclass_cli()
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the name and count fields separated by a bar.
            print(f"{args.name}|{args.count}")

        result = runner.invoke(app, [], env={"MILES_SCRIPT_NAME": "ViaParens", "MILES_SCRIPT_COUNT": "3"})
        assert result.exit_code == 0
        assert "ViaParens|3" in result.stdout


# ---------------------------------------------------------------------------
# Field metadata: help
# ---------------------------------------------------------------------------


class TestFieldMetadata:
    # [CLASS DOC] Tests of field metadata handling: help text from metadata, combined with flags,
    # empty metadata, and mixtures.
    def test_help_appears_in_help_text(self) -> None:
        # [FUNCTION DOC] Requests the help screen; asserts exit code 0 and that the field's help
        # text appears.
        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _MetaHelpArgs) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Verbosity level" in result.stdout

    def test_help_and_flag_together(self) -> None:
        # [FUNCTION DOC] With environment variables disabled, asserts the help screen shows the
        # boolean field's help text, and that passing the verbose flag sets the field to true.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MetaHelpAndFlagArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the verbose field.
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Enable verbose output" in result.stdout

        result = runner.invoke(app, ["--verbose"])
        assert result.exit_code == 0
        assert "verbose=True" in result.stdout

    def test_empty_metadata_treated_as_no_metadata(self) -> None:
        # [FUNCTION DOC] A field with an empty metadata mapping uses its default when omitted;
        # asserts exit code 0 and the default value in the output.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EmptyMetaArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the value field.
            print(f"value={args.value}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "value=ok" in result.stdout

    def test_mixed_metadata_fields(self) -> None:
        # [FUNCTION DOC] Asserts the help screen shows both help texts of the mixed dataclass,
        # and that passing the three option flags (with a number, a string and the boolean flag)
        # yields the combined output with the plain field at its default.
        """Dataclass with a mix of plain fields and help-annotated fields."""
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MixedMetaArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the four fields separated by bars.
            print(f"{args.plain}|{args.with_help}|{args.with_flag}|{args.with_both}")

        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "A number" in result.stdout
        assert "Toggle it" in result.stdout

        result = runner.invoke(app, ["--with-help", "7", "--with-flag", "y", "--with-both"])
        assert result.exit_code == 0
        assert "hello|7|y|True" in result.stdout


# ---------------------------------------------------------------------------
# Bool fields
# ---------------------------------------------------------------------------


class TestBoolFields:
    # [CLASS DOC] Tests of boolean fields becoming on/off flag pairs.
    def test_bool_default_false(self) -> None:
        # [FUNCTION DOC] No flags: asserts the field is false.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the verbose field.
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "verbose=False" in result.stdout

    def test_bool_flag_enables(self) -> None:
        # [FUNCTION DOC] The positive flag sets the field to true.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the verbose field.
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, ["--verbose"])
        assert result.exit_code == 0
        assert "verbose=True" in result.stdout

    def test_bool_no_flag_disables(self) -> None:
        # [FUNCTION DOC] The negated flag (no-verbose) is accepted and leaves the field false.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the verbose field.
            print(f"verbose={args.verbose}")

        result = runner.invoke(app, ["--no-verbose"])
        assert result.exit_code == 0
        assert "verbose=False" in result.stdout


# ---------------------------------------------------------------------------
# Various field types
# ---------------------------------------------------------------------------


class TestFieldTypes:
    # [CLASS DOC] Tests of non-trivial field types: float, optional string and enum.
    def test_float_field(self) -> None:
        # [FUNCTION DOC] Asserts the float default 0.01 when omitted and 0.5 when given.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _FloatArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the rate field.
            print(f"rate={args.rate}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "rate=0.01" in result.stdout

        result = runner.invoke(app, ["--rate", "0.5"])
        assert result.exit_code == 0
        assert "rate=0.5" in result.stdout

    def test_optional_str_default_none(self) -> None:
        # [FUNCTION DOC] An optional string omitted on the command line is none.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _OptionalStrArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the tag field.
            print(f"tag={args.tag}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "tag=None" in result.stdout

    def test_optional_str_provided(self) -> None:
        # [FUNCTION DOC] An optional string given on the command line carries the value.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _OptionalStrArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the tag field.
            print(f"tag={args.tag}")

        result = runner.invoke(app, ["--tag", "v1"])
        assert result.exit_code == 0
        assert "tag=v1" in result.stdout

    def test_enum_field_default(self) -> None:
        # [FUNCTION DOC] An omitted enum option takes its default member (value red).
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the value of the color field.
            print(f"color={args.color.value}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "color=red" in result.stdout

    def test_the_argument_table_prints_an_enum_value_without_its_python_class_name(self) -> None:
        # [FUNCTION DOC] Runs a command that prints nothing itself with no options; asserts the
        # output (an argument table printed by the decorator) contains a row for color with the
        # plain value red and does not contain the enum class-qualified member name.
        """The argument table prints an enum default using its wire value."""
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "| color | red" in result.stdout
        assert "_Color.RED" not in result.stdout

    def test_enum_field_override(self) -> None:
        # [FUNCTION DOC] Passing the color option blue sets that member.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the value of the color field.
            print(f"color={args.color.value}")

        result = runner.invoke(app, ["--color", "blue"])
        assert result.exit_code == 0
        assert "color=blue" in result.stdout

    def test_enum_invalid_value_fails(self) -> None:
        # [FUNCTION DOC] Passing a color that is not a member (yellow) gives a non-zero exit code.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        result = runner.invoke(app, ["--color", "yellow"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# dataclasses.field without metadata / with default_factory
# ---------------------------------------------------------------------------


class TestDataclassField:
    # [CLASS DOC] Tests of fields declared through the dataclass field helper.
    def test_field_with_default_no_metadata(self) -> None:
        # [FUNCTION DOC] A field with a default and no metadata uses that default (5).
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _FieldWithDefaultNoMeta) -> None:
            # [FUNCTION DOC] HELPER: Prints the count field.
            print(f"count={args.count}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "count=5" in result.stdout

    def test_field_with_default_factory_override(self) -> None:
        # [FUNCTION DOC] Overrides a default-factory field explicitly on the command line and
        # asserts the given value is received; the omitted-option path is not tested here
        # because the factory default is displayed as a placeholder in the signature.
        """default_factory defaults show as <factory> in the signature,
        so only the explicit-override path is reliable."""
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _DefaultFactoryArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the items field.
            print(f"items={args.items}")

        result = runner.invoke(app, ["--items", "x,y"])
        assert result.exit_code == 0
        assert "items=x,y" in result.stdout


# ---------------------------------------------------------------------------
# Single field / many required fields
# ---------------------------------------------------------------------------


class TestFieldCounts:
    # [CLASS DOC] Tests of dataclasses with one field and with several required fields.
    def test_single_field_dataclass(self) -> None:
        # [FUNCTION DOC] A one-field dataclass takes its default when omitted.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SingleFieldArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the value field.
            print(f"value={args.value}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "value=default" in result.stdout

    def test_multiple_required_all_provided(self) -> None:
        # [FUNCTION DOC] All three required options given: exit code 0 and joined values shown.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultipleRequiredArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the three fields separated by bars.
            print(f"{args.alpha}|{args.beta}|{args.gamma}")

        result = runner.invoke(app, ["--alpha", "a", "--beta", "b", "--gamma", "c"])
        assert result.exit_code == 0
        assert "a|b|c" in result.stdout

    def test_multiple_required_missing_any_fails(self) -> None:
        # [FUNCTION DOC] Omitting the third required option gives a non-zero exit code.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultipleRequiredArgs) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        result = runner.invoke(app, ["--alpha", "a", "--beta", "b"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Multiple fields
# ---------------------------------------------------------------------------


class TestMultipleFields:
    # [CLASS DOC] Tests of dataclasses with several fields of mixed kinds.
    def test_all_defaults(self) -> None:
        # [FUNCTION DOC] No options: all defaults (localhost, 8080, false) are used.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultiFieldArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints host, port and debug separated by colons.
            print(f"{args.host}:{args.port}:{args.debug}")

        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "localhost:8080:False" in result.stdout

    def test_override_some(self) -> None:
        # [FUNCTION DOC] Overriding the port and enabling debug leaves the host at its default.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _MultiFieldArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints host, port and debug separated by colons.
            print(f"{args.host}:{args.port}:{args.debug}")

        result = runner.invoke(app, ["--port", "9090", "--debug"])
        assert result.exit_code == 0
        assert "localhost:9090:True" in result.stdout

    def test_all_required_both_provided(self) -> None:
        # [FUNCTION DOC] Both required options given: exit code 0 and the values shown.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _AllRequiredArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the two fields separated by a bar.
            print(f"{args.first}|{args.second}")

        result = runner.invoke(app, ["--first", "a", "--second", "2"])
        assert result.exit_code == 0
        assert "a|2" in result.stdout

    def test_all_required_missing_one_fails(self) -> None:
        # [FUNCTION DOC] Omitting the second required option gives a non-zero exit code.
        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _AllRequiredArgs) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        result = runner.invoke(app, ["--first", "a"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Return value passthrough
# ---------------------------------------------------------------------------


class TestReturnValue:
    # [CLASS DOC] Tests that the decorated function's return value is passed through when called
    # directly with keyword arguments (no command runner).
    def test_return_value_is_forwarded(self) -> None:
        # [FUNCTION DOC] Calls a decorated function with name and count keywords and asserts the
        # returned string is forwarded unchanged.
        @dataclass_cli(env_var_prefix="")
        def compute(args: _SimpleArgs) -> str:
            # [FUNCTION DOC] HELPER: Returns the name field prefixed with result and a colon.
            return f"result:{args.name}"

        value = compute(name="test", count=1)
        assert value == "result:test"

    def test_return_none(self) -> None:
        # [FUNCTION DOC] A decorated function returning nothing yields none when called directly.
        @dataclass_cli(env_var_prefix="")
        def noop(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        value = noop(name="x", count=0)
        assert value is None


# ---------------------------------------------------------------------------
# Dataclass instance correctness
# ---------------------------------------------------------------------------


class TestDataclassInstance:
    # [CLASS DOC] Tests of the argument object the command receives: its type and the python
    # types of its fields. Commands append the received object to a list for inspection.
    def test_receives_correct_dataclass_type(self) -> None:
        # [FUNCTION DOC] Invokes with name and count; asserts exactly one call, the argument is
        # an instance of the simple dataclass, with name check and count 7 (converted integer).
        received: list = []

        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Appends the received argument object to the outer list.
            received.append(args)

        runner.invoke(app, ["--name", "check", "--count", "7"])
        assert len(received) == 1
        assert isinstance(received[0], _SimpleArgs)
        assert received[0].name == "check"
        assert received[0].count == 7

    def test_bool_field_receives_python_bool(self) -> None:
        # [FUNCTION DOC] The verbose flag arrives as the true boolean object.
        received: list = []

        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _BoolArgs) -> None:
            # [FUNCTION DOC] HELPER: Appends the received argument object to the outer list.
            received.append(args)

        runner.invoke(app, ["--verbose"])
        assert received[0].verbose is True

    def test_enum_field_receives_enum_instance(self) -> None:
        # [FUNCTION DOC] The color option green arrives as the green enum member itself.
        received: list = []

        app = typer.Typer()

        @app.command()
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _EnumArgs) -> None:
            # [FUNCTION DOC] HELPER: Appends the received argument object to the outer list.
            received.append(args)

        runner.invoke(app, ["--color", "green"])
        assert received[0].color is _Color.GREEN


# ---------------------------------------------------------------------------
# Wrapped function attributes
# ---------------------------------------------------------------------------


class TestWrappedAttributes:
    # [CLASS DOC] Tests that the decorator preserves the wrapped function's name, qualified name
    # and docstring, and replaces the dataclass parameter with its fields in the signature.
    def test_preserves_name(self) -> None:
        # [FUNCTION DOC] The wrapper keeps the name and a qualified name ending in the name.
        @dataclass_cli
        def my_func(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Function under decoration; does nothing.
            """My docstring."""

        assert my_func.__name__ == "my_func"
        assert my_func.__qualname__.endswith("my_func")

    def test_preserves_docstring(self) -> None:
        # [FUNCTION DOC] The wrapper keeps the original docstring text.
        @dataclass_cli
        def my_func(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Function under decoration; does nothing.
            """My docstring."""

        assert my_func.__doc__ == "My docstring."

    def test_preserves_name_parameterized(self) -> None:
        # [FUNCTION DOC] The parameterized form also keeps the name and docstring.
        @dataclass_cli(env_var_prefix="")
        def another_func(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Function under decoration; does nothing.
            """Another doc."""

        assert another_func.__name__ == "another_func"
        assert another_func.__doc__ == "Another doc."

    def test_no_docstring(self) -> None:
        # [FUNCTION DOC] A function without a docstring still has no docstring after decoration.
        @dataclass_cli
        def no_doc(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Function under decoration; does nothing.
            pass

        assert no_doc.__doc__ is None

    def test_has_signature(self) -> None:
        # [FUNCTION DOC] The wrapper's signature lists the dataclass fields (name, count) and no
        # longer the single args parameter.
        @dataclass_cli(env_var_prefix="")
        def cmd(args: _SimpleArgs) -> None:
            # [FUNCTION DOC] HELPER: Function under decoration; does nothing.
            pass

        sig = inspect.signature(cmd)
        param_names = list(sig.parameters.keys())
        assert "name" in param_names
        assert "count" in param_names
        assert "args" not in param_names


# ---------------------------------------------------------------------------
# Error: non-dataclass annotation
# ---------------------------------------------------------------------------


class TestErrorCases:
    # [CLASS DOC] Tests that decorating a function whose parameter is annotated with a
    # non-dataclass type fails at decoration time.
    def test_non_dataclass_raises(self) -> None:
        # [FUNCTION DOC] Bare form: decorating raises an assertion error.
        class NotADataclass:
            # [CLASS DOC] HELPER: Plain class that is not a dataclass.
            pass

        with pytest.raises(AssertionError):

            @dataclass_cli
            def bad(args: NotADataclass) -> None:
                # [FUNCTION DOC] HELPER: Never decorated successfully; does nothing.
                pass

    def test_non_dataclass_raises_parameterized(self) -> None:
        # [FUNCTION DOC] Parameterized form: decorating raises an assertion error.
        class NotADataclass:
            # [CLASS DOC] HELPER: Plain class that is not a dataclass.
            pass

        with pytest.raises(AssertionError):

            @dataclass_cli(env_var_prefix="")
            def bad(args: NotADataclass) -> None:
                # [FUNCTION DOC] HELPER: Never decorated successfully; does nothing.
                pass


# ---------------------------------------------------------------------------
# Env var naming: uppercase conversion
# ---------------------------------------------------------------------------


class TestEnvVarNaming:
    # [CLASS DOC] Tests of environment variable naming and of default factory fields.
    def test_underscore_field_name_uppercased(self) -> None:
        # [FUNCTION DOC] A field with underscores in its name is read from the environment
        # variable made of the default prefix and the upper-cased field name; asserts the value
        # from the environment is used.
        @dataclasses.dataclass
        class _SnakeArgs:
            # [CLASS DOC] HELPER: One string field my_long_name defaulting to the word default.
            my_long_name: str = "default"

        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _SnakeArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the my_long_name field.
            print(f"val={args.my_long_name}")

        result = runner.invoke(app, [], env={"MILES_SCRIPT_MY_LONG_NAME": "from_env"})
        assert result.exit_code == 0
        assert "val=from_env" in result.stdout

    def test_a_field_with_a_default_factory_is_usable_from_the_command_line(self) -> None:
        # [FUNCTION DOC] An integer field whose default comes from a factory returning 7 works
        # with the default prefix: omitting the option gives 7 and passing 9 gives 9. Guards
        # against the dataclass factory placeholder leaking into the option's declared default.
        """Click type-casts the declared default, so dataclasses' factory sentinel must never reach it."""

        @dataclasses.dataclass
        class _FactoryArgs:
            # [CLASS DOC] HELPER: One integer count field with a factory producing 7.
            count: int = dataclasses.field(default_factory=lambda: 7)

        app = typer.Typer()

        @app.command()
        @dataclass_cli
        def cmd(args: _FactoryArgs) -> None:
            # [FUNCTION DOC] HELPER: Prints the count field.
            print(f"count={args.count}")

        assert "count=7" in runner.invoke(app, []).stdout
        assert "count=9" in runner.invoke(app, ["--count", "9"]).stdout


# ---------------------------------------------------------------------------
# import needed for signature inspection
# ---------------------------------------------------------------------------
import inspect
