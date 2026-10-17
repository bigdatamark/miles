# [MODULE DOC] Utilities that turn a dataclass into a typer command line interface. A decorator makes a
# function whose first parameter is a dataclass into a typer-compatible command whose options mirror
# the dataclass fields (with per-field environment variable binding and help text from field
# metadata); a second entry point builds a dataclass purely from environment variables by running the
# same click machinery with empty arguments. Tuple fields of the form "tuple of T, variable length" are
# exposed as repeatable list options and converted back to tuples.
import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Annotated, Any, TypeVar, overload

import typer

_F = TypeVar("_F", bound=Callable[..., object])

SCRIPT_ENV_VAR_PREFIX = "MILES_SCRIPT_"


@overload
def dataclass_cli(func: _F) -> _F: ...


@overload
def dataclass_cli(
    func: None = None,
    *,
    env_var_prefix: str = SCRIPT_ENV_VAR_PREFIX,
) -> Callable[[_F], _F]: ...


def dataclass_cli(
    func: _F | None = None,
    *,
    env_var_prefix: str = SCRIPT_ENV_VAR_PREFIX,
) -> _F | Callable[[_F], _F]:
    # [FUNCTION DOC] Decorator usable bare or with an env var prefix keyword. When called without the
    # function, returns a partial that will decorate later with the given prefix. Otherwise returns
    # the wrapped command (see the wrap helper). The first parameter of the function must be annotated
    # with a dataclass type.
    """Turn a function whose first param is a dataclass into a typer-compatible CLI.

    Modified from https://github.com/fastapi/typer/issues/154#issuecomment-1544876144

    Supports field ``metadata`` keys:
    - ``"help"``: passed as ``help=`` to ``typer.Option``

    Usage::

        @app.command()
        @dataclass_cli                              # bare — uses MILES_SCRIPT_ env prefix
        def cmd(args: MyArgs): ...

        @app.command()
        @dataclass_cli(env_var_prefix="")            # no env-var binding
        def cmd(args: MyArgs): ...
    """
    if func is None:
        return functools.partial(dataclass_cli, env_var_prefix=env_var_prefix)  # type: ignore[return-value]

    return _wrap(func, env_var_prefix=env_var_prefix)


def _resolve_default(field: dataclasses.Field, param: inspect.Parameter) -> object:
    # [FUNCTION DOC] HELPER: Returns the result of calling the field's default factory if it has one,
    # else the signature parameter's default.
    """Call a default_factory now; click would otherwise type-cast dataclasses' sentinel."""
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return param.default


def dataclass_from_env(
    dataclass_cls: type, *, env_var_prefix: str = SCRIPT_ENV_VAR_PREFIX, overrides: Mapping[str, object] | None = None
) -> Any:
    # [FUNCTION DOC] Builds an instance of the dataclass without any command-line input. Creates a
    # typer app with one command whose signature is the generated option list (with environment
    # variables bound by the prefix), runs it with empty arguments in non-standalone mode (so click
    # errors such as missing required options raise exceptions instead of exiting), and returns the
    # instance. Overrides (a mapping from field name to value) win over parsed values. Does not print
    # the argument table.
    """Build the dataclass from the environment alone, letting click read it exactly as it would on the cli."""
    built: list[Any] = []

    def build(**kwargs: object) -> None:
        # [FUNCTION DOC] HELPER: Command body: builds the dataclass from the parsed options merged with
        # the overrides and stores it for the caller.
        built.append(_build(dataclass_cls, {**kwargs, **(overrides or {})}))

    build.__signature__ = inspect.Signature(_cli_parameters(dataclass_cls, env_var_prefix=env_var_prefix))
    build.__name__ = dataclass_cls.__name__

    app = typer.Typer(add_completion=False)
    app.command()(build)
    typer.main.get_command(app)(args=[], standalone_mode=False)

    return built[0]


def _wrap(func: _F, *, env_var_prefix: str) -> _F:
    # [FUNCTION DOC] HELPER: Finds the dataclass type from the type hint of the function's first
    # parameter and returns a keyword-only command whose signature lists the dataclass fields as typer
    # options, copying the function's doc, name and qualified name. When invoked it builds the
    # dataclass from the keywords, prints the argument table to standard output, and calls the original
    # function with the instance as its only argument (so other parameters of the function are not
    # supported) and returns its result.
    hints: dict[str, type] = typing.get_type_hints(func)
    first_param_name: str = next(iter(inspect.signature(func).parameters))
    dataclass_cls: type = hints[first_param_name]

    def wrapped(**kwargs: object) -> object:
        # [FUNCTION DOC] HELPER: The generated command body described in the enclosing function.
        data: object = _build(dataclass_cls, kwargs)
        _print_arguments(data)
        return func(data)

    wrapped.__signature__ = inspect.Signature(_cli_parameters(dataclass_cls, env_var_prefix=env_var_prefix))
    wrapped.__doc__ = func.__doc__
    wrapped.__name__ = func.__name__  # type: ignore[attr-defined]
    wrapped.__qualname__ = func.__qualname__  # type: ignore[attr-defined]

    return wrapped  # type: ignore[return-value]


def _cli_parameters(dataclass_cls: type, *, env_var_prefix: str) -> list[inspect.Parameter]:
    # [FUNCTION DOC] HELPER: Asserts the type is a dataclass. Takes the parameters of its generated
    # constructor signature (dropping a leading self) and rewrites each: annotation becomes the
    # resolved type hint (variable-length tuples as lists) annotated with a typer option carrying an
    # environment variable name (prefix plus upper-cased field name, only when the prefix is
    # non-empty) and the field's help metadata when present; the default becomes the default factory's
    # result when one exists (otherwise the parameter's own default), with tuples converted to lists.
    # Returns the list of parameters in field order.
    assert dataclasses.is_dataclass(dataclass_cls)

    old_parameters: list[inspect.Parameter] = list(inspect.signature(dataclass_cls.__init__).parameters.values())
    if old_parameters and old_parameters[0].name == "self":
        del old_parameters[0]

    resolved_hints: dict[str, type] = typing.get_type_hints(dataclass_cls)
    fields_by_name: dict[str, dataclasses.Field] = {  # type: ignore[type-arg]
        f.name: f for f in dataclasses.fields(dataclass_cls)
    }

    new_parameters: list[inspect.Parameter] = []
    for param in old_parameters:
        field: dataclasses.Field = fields_by_name[param.name]  # type: ignore[type-arg]

        typer_kwargs: dict[str, object] = {}
        if env_var_prefix:
            typer_kwargs["envvar"] = f"{env_var_prefix}{param.name.upper()}"
        if "help" in field.metadata:
            typer_kwargs["help"] = field.metadata["help"]

        resolved_type: type = resolved_hints.get(param.name, param.annotation)
        new_annotation = Annotated[_repeatable_as_list(resolved_type), typer.Option(**typer_kwargs)]
        default = _resolve_default(field, param)

        new_parameters.append(
            param.replace(annotation=new_annotation, default=list(default) if isinstance(default, tuple) else default)
        )
    return new_parameters


def _repeatable_as_list(annotation: type) -> type:
    # [FUNCTION DOC] HELPER: Converts an annotation of a variable-length tuple of one element type into
    # a list of that type; returns every other annotation unchanged.
    """click has no variadic tuple, so a repeatable option reaches it as the list it is spelled with."""
    if typing.get_origin(annotation) is tuple and typing.get_args(annotation)[1:] == (Ellipsis,):
        return list[typing.get_args(annotation)[0]]  # type: ignore[misc, return-value]
    return annotation


def _build(dataclass_cls: type, kwargs: dict[str, object]) -> Any:
    # [FUNCTION DOC] HELPER: Constructs the dataclass from the keyword mapping, converting list values
    # back to tuples for fields whose type hint is a tuple (of any form).
    hints: dict[str, type] = typing.get_type_hints(dataclass_cls)
    return dataclass_cls(
        **{
            name: tuple(value) if typing.get_origin(hints[name]) is tuple and isinstance(value, list) else value
            for name, value in kwargs.items()
        }
    )


def _print_arguments(data: object) -> None:
    # [FUNCTION DOC] HELPER: Prints an ASCII table of the dataclass instance's fields to standard
    # output: a header row, then one row per field with the name left-aligned to the longest name and
    # the value (enumerations shown by their value) cut to 47 characters plus an ellipsis when longer
    # than 50. Fails with a value error from the maximum computation if the dataclass has no fields.
    fields = dataclasses.fields(data)
    max_key_len = max(len(f.name) for f in fields)
    sep = "+" + "-" * (max_key_len + 2) + "+" + "-" * 52 + "+"
    print(sep)
    print(f"| {'Argument':<{max_key_len}} | {'Value':<50} |")
    print(sep)
    for f in fields:
        val_raw = getattr(data, f.name)
        val = str(val_raw.value if isinstance(val_raw, Enum) else val_raw)
        if len(val) > 50:
            val = val[:47] + "..."
        print(f"| {f.name:<{max_key_len}} | {val:<50} |")
    print(sep)
