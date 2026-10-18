# [MODULE DOC] Unit tests for the worker argument-vector utilities. Areas: reading field names and
# values from dataclass and msgspec record types; converting a frozen strict config model to a single
# json command-line flag and back (with a mandatory roundtrip check); rendering a command line from
# constructor input values by comparing against a parser's defaults and verifying with a parse
# roundtrip (always-render fields, field to destination aliases, boolean optional, json valued and
# append options, uncompared fields); building the interpreter flag prefix for re-executing python
# (tested with real subprocesses); coercing a dictionary of yaml values to typed arguments using a
# parser; listing declared argument destinations; parsing a model script's declared arguments; and
# context managers that temporarily relax required arguments or suppress help in a parser.
import argparse
import contextlib
import dataclasses
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
import pytest
from pydantic import ValidationError

from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers import argv_utils
from miles.utils.workers.argv_utils import (
    CONFIG_JSON_FLAG,
    _record_field_names,
    coerce_dict_to_args,
    config_to_argv,
    dataclass_to_values,
    declared_arg_dests,
    parse_config_argv,
    parse_declared_args,
    render_cli_argv,
    with_relax_parser_required_args,
    with_suppressed_parser_help,
)


@pytest.mark.parametrize(
    "record_factory", [dataclasses.make_dataclass, msgspec.defstruct], ids=["dataclass", "msgspec"]
)
def test_record_fields_support_classes_and_instances(record_factory):
    # [FUNCTION DOC] Parametrized over a dataclass factory and a msgspec struct factory. Builds a
    # record type with fields port (int) and host (str) and an instance. Asserts the private field name
    # reader returns the ordered names for both the type and the instance, and the values extractor
    # returns a mapping of field names to values for the instance.
    record_type = record_factory("Args", [("port", int), ("host", str)])
    record = record_type(port=30000, host="localhost")

    assert _record_field_names(record_type) == ("port", "host")
    assert _record_field_names(record) == ("port", "host")
    assert dataclass_to_values(record) == {"port": 30000, "host": "localhost"}


class _DemoConfig(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Frozen strict config model with a text, an integer, a float, a boolean, an
    # optional float, an optional list of strings and an optional mapping, all required (no defaults).
    text: str
    count: int
    ratio: float
    enabled: bool
    maybe_timeout: float | None
    tags: list[str] | None
    options: dict[str, Any] | None


def _make_demo_config(**overrides) -> _DemoConfig:
    # [FUNCTION DOC] HELPER: Builds a demo config with defaults (text "hello world", count 3, ratio
    # 0.5, enabled true, no timeout, tags a and b, options holding a nested mapping with a list) and
    # applies keyword overrides on top.
    kwargs = dict(
        text="hello world",
        count=3,
        ratio=0.5,
        enabled=True,
        maybe_timeout=None,
        tags=["a", "b"],
        options={"nested": {"k": [1, 2]}},
    )
    kwargs.update(overrides)
    return _DemoConfig(**kwargs)


class TestConfigToArgv:
    # [CLASS DOC] Tests converting a config object to an argument vector and parsing it back.
    def test_roundtrip_preserves_every_field_type(self):
        # [FUNCTION DOC] Parsing the rendered argv of a demo config gives an equal config.
        """str, int, float, bool, None, list, and nested dict all survive."""
        config = _make_demo_config()
        assert parse_config_argv(_DemoConfig, config_to_argv(config)) == config

    @pytest.mark.parametrize(
        "text",
        ["with space", 'quo"te', "single'quote", "中文字符", "line\nbreak", "--looks-like-a-flag", ""],
    )
    def test_roundtrip_survives_hostile_strings(self, text):
        # [FUNCTION DOC] Parametrized over strings with spaces, quotes, non-ascii, a newline, a
        # flag-like string and the empty string; the text field survives the roundtrip.
        """Quoting-hostile string values survive the argv boundary."""
        config = _make_demo_config(text=text)
        assert parse_config_argv(_DemoConfig, config_to_argv(config)).text == text

    def test_roundtrip_preserves_none_versus_value(self):
        # [FUNCTION DOC] The optional timeout stays nothing when nothing and stays 30.0 when set.
        """None and a real value on a nullable field stay distinguishable."""
        assert parse_config_argv(_DemoConfig, config_to_argv(_make_demo_config())).maybe_timeout is None
        config = _make_demo_config(maybe_timeout=30.0)
        assert parse_config_argv(_DemoConfig, config_to_argv(config)).maybe_timeout == 30.0

    def test_argv_is_a_flag_value_pair(self):
        # [FUNCTION DOC] The rendered argv has exactly two items, the config json flag first.
        """The rendered argv is exactly the config-json flag plus its payload."""
        argv = config_to_argv(_make_demo_config())
        assert argv[0] == CONFIG_JSON_FLAG
        assert len(argv) == 2

    def test_production_roundtrip_check_cannot_be_skipped(self, monkeypatch):
        # [FUNCTION DOC] With the module's config parser replaced by one returning a different config
        # (count 999), rendering raises an assertion error mentioning a roundtrip mismatch.
        """A parse that fails to reproduce the config aborts the render."""
        monkeypatch.setattr(argv_utils, "parse_config_argv", lambda config_cls, argv: _make_demo_config(count=999))
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            config_to_argv(_make_demo_config())

    def test_real_worker_configs_roundtrip(self):
        # [FUNCTION DOC] Builds a real router config and a real session server config with fully
        # specified fields (including nested kwargs, optional nothing values and dotted function paths),
        # and asserts each survives render and parse unchanged.
        """The miles router and session server configs survive the boundary."""
        router_config = MilesRouterConfig(
            host="127.0.0.1",
            port=30080,
            max_connections=256,
            timeout=None,
            health_check_interval=10.0,
            health_check_failure_threshold=3,
        )
        assert parse_config_argv(MilesRouterConfig, config_to_argv(router_config)) == router_config

        session_config = SessionServerConfig(
            host="127.0.0.1",
            port=30100,
            instance_id="abc",
            backend_url="http://127.0.0.1:30000",
            timeout=600.0,
            hf_checkpoint="/fake/model",
            chat_template_path=None,
            tito_model="qwen3",
            apply_chat_template_kwargs={"enable_thinking": False},
            use_rollout_routing_replay=True,
            use_rollout_indexer_replay=False,
            use_sampling_support_replay=False,
            sglang_speculative_algorithm=None,
            num_layers=None,
            moe_router_topk=None,
            save_debug_trajectory_data=None,
            lora_rank=0,
            lora_adapter_path=None,
            lora_train_only=False,
            use_session_server="v2",
            session_message_matcher="strict",
            pause_generation_mode=None,
            session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_same_prompt_retries",
            session_sample_postprocessor_path=("miles.rollout.session.v2.postprocessor_hub.default_postprocess"),
            rollout_top_logprobs_num=128,
            rollout_sampling_logprobs_mode="selected",
            rollout_temperature=0.7,
        )
        assert parse_config_argv(SessionServerConfig, config_to_argv(session_config)) == session_config


class TestParseConfigArgv:
    # [CLASS DOC] Tests the strict config argv parser.
    def test_none_argv_parses_the_process_arguments(self, monkeypatch):
        # [FUNCTION DOC] With the process argv replaced by a program name plus the rendered config, a
        # parse with no explicit argv gives the original config.
        """A None argv reads the payload from the process command line."""
        config = _make_demo_config()
        monkeypatch.setattr(sys, "argv", ["prog", *config_to_argv(config)])
        assert parse_config_argv(_DemoConfig, None) == config

    def test_missing_flag_is_rejected(self):
        # [FUNCTION DOC] An empty argv makes the parser exit (system exit).
        """An argv without the config-json flag fails to parse."""
        with pytest.raises(SystemExit):
            parse_config_argv(_DemoConfig, [])

    def test_unknown_flag_is_rejected(self):
        # [FUNCTION DOC] A valid argv plus an extra unknown flag makes the parser exit.
        """Stray extra flags fail to parse instead of being ignored."""
        argv = config_to_argv(_make_demo_config())
        with pytest.raises(SystemExit):
            parse_config_argv(_DemoConfig, [*argv, "--unknown", "1"])

    def test_invalid_json_is_rejected(self):
        # [FUNCTION DOC] A payload "not json" raises a pydantic validation error.
        """A payload that is not valid JSON fails validation loudly."""
        with pytest.raises(ValidationError):
            parse_config_argv(_DemoConfig, [CONFIG_JSON_FLAG, "not json"])

    def test_extra_json_fields_are_rejected(self):
        # [FUNCTION DOC] A json payload with an inserted unknown field raises a validation error.
        """A payload with unknown fields violates the strict schema."""
        payload = _make_demo_config().model_dump_json().replace("{", '{"unknown_field": 1, ', 1)
        with pytest.raises(ValidationError):
            parse_config_argv(_DemoConfig, [CONFIG_JSON_FLAG, payload])


@dataclasses.dataclass
class _DemoArgs:
    # [CLASS DOC] HELPER: Dataclass of demo arguments: name, count, ratio, verbose and enabled flags,
    # a list of items, a mapping, and an optional cli-filled text whose dataclass default (nothing)
    # differs from the parser default.
    name: str = "default-name"
    count: int = 0
    ratio: float = 1.0
    verbose: bool = False
    enabled: bool = True
    items: list[str] = dataclasses.field(default_factory=list)
    mapping: dict[str, str] = dataclasses.field(default_factory=dict)
    cli_filled: str | None = None


def _make_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Builds a parser for the demo arguments: name, count (int), ratio (float),
    # verbose (store true), enabled (store true with default true), items and mapping (zero or more
    # values, default empty list) and cli-filled (default "filled-by-cli").
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="default-name")
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enabled", action="store_true", default=True)
    parser.add_argument("--items", nargs="*", default=[])
    parser.add_argument("--mapping", nargs="*", default=[])
    parser.add_argument("--cli-filled", default="filled-by-cli")
    return parser


def _from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
    # [FUNCTION DOC] HELPER: Converts a parsed namespace to demo arguments: copies the scalar fields,
    # copies the items to a list, and turns each "key=value" mapping token into an entry by splitting
    # at the first equals sign.
    return _DemoArgs(
        name=parsed.name,
        count=parsed.count,
        ratio=parsed.ratio,
        verbose=parsed.verbose,
        enabled=parsed.enabled,
        items=list(parsed.items),
        mapping=dict(item.split("=", 1) for item in parsed.mapping),
        cli_filled=parsed.cli_filled,
    )


def _render(args_obj: _DemoArgs) -> list[str]:
    # [FUNCTION DOC] HELPER: Renders a command line for the demo arguments with the demo parser and
    # converter, using the object itself as the expected result and input values from the input
    # values helper.
    return render_cli_argv(
        _input_values(args_obj), expected_obj=args_obj, make_parser=_make_parser, from_parsed=_from_parsed
    )


def _parse(argv: list[str]) -> _DemoArgs:
    # [FUNCTION DOC] HELPER: Parses an argv with the demo parser and converts it to demo arguments.
    return _from_parsed(_make_parser().parse_args(argv))


def _make_cli_default_args(**overrides) -> _DemoArgs:
    # [FUNCTION DOC] HELPER: Parses an empty command line (so cli defaults apply) and then sets the
    # given attributes on the result. Returns it.
    args_obj = _parse([])
    for name, value in overrides.items():
        setattr(args_obj, name, value)
    return args_obj


def _input_values(args_obj: _DemoArgs) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Returns the object's values as a mapping, with the mapping field turned
    # into a list of "key=value" strings (the shape the parser takes).
    values = dataclass_to_values(args_obj)
    values["mapping"] = [f"{key}={value}" for key, value in args_obj.mapping.items()]
    return values


def _from_parsed_drifting(parsed: argparse.Namespace) -> _DemoArgs:
    # [FUNCTION DOC] HELPER: Like the normal converter, but always replaces the ratio with 99.0,
    # imitating a field rewritten on every parse.
    return dataclasses.replace(_from_parsed(parsed), ratio=99.0)


def _render_drifting(args_obj: _DemoArgs, **overrides) -> list[str]:
    # [FUNCTION DOC] HELPER: Renders like the normal helper but with the drifting converter; extra
    # keyword arguments are forwarded to the renderer.
    return render_cli_argv(
        _input_values(args_obj),
        expected_obj=args_obj,
        make_parser=_make_parser,
        from_parsed=_from_parsed_drifting,
        **overrides,
    )


class TestRenderCliArgv:
    # [CLASS DOC] Tests the command line renderer, which emits only inputs that differ from parser
    # defaults and then verifies by parsing the result and comparing with the expected object.
    def test_cli_defaults_render_to_an_empty_argv(self):
        # [FUNCTION DOC] An object parsed from an empty command line renders to an empty list.
        """An object matching the CLI defaults needs no flags at all."""
        assert _render(_parse([])) == []

    def test_scalar_bool_list_and_dict_fields_roundtrip(self):
        # [FUNCTION DOC] An object with changed name, count, ratio, verbose flag, items and mapping
        # renders an argv that contains the verbose flag and parses back to an equal object.
        """Every rendered field kind survives parse back to an equal object."""
        args_obj = _make_cli_default_args(
            name="other",
            count=3,
            ratio=0.5,
            verbose=True,
            items=["a", "b"],
            mapping={"k1": "v1", "k2": "v2"},
        )
        argv = _render(args_obj)
        assert "--verbose" in argv
        assert _parse(argv) == args_obj

    def test_a_variadic_dict_renders_key_value_tokens(self):
        # [FUNCTION DOC] A mapping given directly as input for a zero-or-more option renders as the
        # option followed by key=value tokens.
        """A dict handed to an nargs option becomes key=value tokens, not JSON."""
        args_obj = _make_cli_default_args(mapping={"k1": "v1", "k2": "v2"})
        argv = render_cli_argv(
            {"mapping": {"k1": "v1", "k2": "v2"}},
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=_from_parsed,
        )
        assert argv == ["--mapping", "k1=v1", "k2=v2"]

    def test_cli_only_defaults_are_not_rendered(self):
        # [FUNCTION DOC] An object parsed with defaults except count 3 renders no cli-filled option.
        """A field keeping its CLI default (even when it differs from the
        dataclass default) stays off the command line."""
        argv = _render(_make_cli_default_args(count=3))
        assert "--cli-filled" not in argv

    def test_none_constructor_inputs_are_left_for_the_cli_to_normalize(self):
        # [FUNCTION DOC] An input of nothing for the items field renders nothing (an empty argv).
        """A nullable input can normalize to a collection without being rendered."""
        args_obj = _parse([])
        argv = render_cli_argv(
            {"items": None},
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=_from_parsed,
        )
        assert argv == []

    def test_constructor_values_are_rendered_before_post_parse_normalization(self):
        # [FUNCTION DOC] With a converter that halves count when verbose is set, the raw input count of
        # 6 is rendered as 6 (not the normalized 3). Asserts the parsed count is 6 and converting the
        # parse of the rendered argv equals the expected object.
        """Raw values are not normalized twice when from_parsed rewrites them."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            # [FUNCTION DOC] HELPER: Normal conversion, then halves count (integer division) if verbose.
            args_obj = _from_parsed(parsed)
            if args_obj.verbose:
                args_obj.count //= 2
            return args_obj

        input_values = {**_input_values(_make_cli_default_args(verbose=True)), "count": 6}
        args_obj = from_parsed(_make_parser().parse_args(["--verbose", "--count", "6"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert _make_parser().parse_args(argv).count == 6
        assert from_parsed(_make_parser().parse_args(argv)) == args_obj

    def test_each_non_default_input_is_rendered_in_one_pass(self):
        # [FUNCTION DOC] With a converter that sets count to 3 when ratio is the default, inputs of
        # count 3 and ratio 0.5 render exactly the count and ratio options (in that order), and the
        # parse of that argv equals the expected object.
        """Interacting non-default inputs are both emitted without reconciliation."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            # [FUNCTION DOC] HELPER: Normal conversion, then sets count to 3 if the ratio is 1.0.
            args_obj = _from_parsed(parsed)
            if args_obj.ratio == 1.0:
                args_obj.count = 3
            return args_obj

        input_values = {**_input_values(_make_cli_default_args(count=3)), "ratio": 0.5}
        args_obj = from_parsed(_make_parser().parse_args(["--count", "3", "--ratio", "0.5"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert argv == ["--count", "3", "--ratio", "0.5"]
        assert from_parsed(_make_parser().parse_args(argv)) == args_obj

    def test_raw_parser_default_is_omitted_before_post_parse_normalization(self):
        # [FUNCTION DOC] With a converter that multiplies the ratio by 0.3 when verbose, an input
        # ratio equal to the raw parser default of 1.0 is omitted; only the verbose flag is rendered
        # and the parse of that argv converts to the expected object.
        """A raw default is omitted so post-parse normalization runs exactly once."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            # [FUNCTION DOC] HELPER: Normal conversion, then scales the ratio by 0.3 if verbose.
            args_obj = _from_parsed(parsed)
            if args_obj.verbose:
                args_obj.ratio *= 0.3
            return args_obj

        input_values = {**_input_values(_make_cli_default_args(verbose=True)), "ratio": 1.0}
        expected_obj = from_parsed(_make_parser().parse_args(["--verbose"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=expected_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert argv == ["--verbose"]
        assert from_parsed(_make_parser().parse_args(argv)) == expected_obj

    def test_expected_object_is_constructed_only_once_during_render(self):
        # [FUNCTION DOC] Counts calls of the converter during one render with count 3; the argv is the
        # count option and the converter ran exactly once (the final verification parse).
        """The renderer performs one final conversion and never reconciles iteratively."""
        conversion_count = 0

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            # [FUNCTION DOC] HELPER: Counts the call and delegates to the normal converter.
            nonlocal conversion_count
            conversion_count += 1
            return _from_parsed(parsed)

        expected_obj = _from_parsed(_make_parser().parse_args(["--count", "3"]))
        argv = render_cli_argv(
            {**_input_values(expected_obj), "count": 3},
            expected_obj=expected_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert argv == ["--count", "3"]
        assert conversion_count == 1

    def test_default_store_true_value_is_expressed_by_omitting_the_flag(self):
        # [FUNCTION DOC] With a converter that derives verbose from count being zero, a false verbose
        # input is not rendered (no verbose flag), and the parse of the argv converts to the expected
        # object.
        """A default False remains implicit when other inputs change its resolved baseline."""

        def from_parsed(parsed: argparse.Namespace) -> _DemoArgs:
            # [FUNCTION DOC] HELPER: Normal conversion, then sets verbose to whether count is zero.
            args_obj = _from_parsed(parsed)
            args_obj.verbose = args_obj.count == 0
            return args_obj

        input_values = _input_values(_make_cli_default_args(count=1))
        args_obj = from_parsed(_make_parser().parse_args(["--count", "1"]))
        argv = render_cli_argv(
            input_values,
            expected_obj=args_obj,
            make_parser=_make_parser,
            from_parsed=from_parsed,
        )
        assert "--verbose" not in argv
        assert from_parsed(_make_parser().parse_args(argv)) == args_obj

    def test_unrenderable_false_on_a_true_default_flag_fails_loudly(self):
        # [FUNCTION DOC] Wanting enabled false when its store-true option defaults to true cannot be
        # expressed; rendering raises an assertion error saying enabled parsed true, wanted false.
        """A store-true flag whose CLI default is True cannot express False, and the roundtrip
        must refuse the argv rather than let the value disappear from it."""
        with pytest.raises(AssertionError, match="enabled: parsed True != wanted False"):
            _render(_make_cli_default_args(enabled=False))

    def test_roundtrip_mismatch_aborts_the_render(self):
        # [FUNCTION DOC] A converter that returns an object with count 999 regardless of the argv
        # makes rendering raise an assertion error mentioning a roundtrip mismatch.
        """A from_parsed that fails to reproduce the object aborts the render."""
        args_obj = _make_cli_default_args(count=3)
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            render_cli_argv(
                _input_values(args_obj),
                expected_obj=args_obj,
                make_parser=_make_parser,
                from_parsed=lambda parsed: _make_cli_default_args(count=999),
            )

    def test_a_value_the_parser_refuses_is_raised_rather_than_exited(self):
        # [FUNCTION DOC] A parser with a restricted choice option and an input outside the choices:
        # rendering raises an assertion error saying the parser rejects the rendered option and value,
        # instead of letting the parser exit the process.
        """argparse answers a value it will not accept by exiting the process. This renders inside the
        worker that launches the command, so exiting takes it down past everything that reports a
        failure, and the run waits out its whole timeout on an engine nobody ever started."""

        def make_parser() -> argparse.ArgumentParser:
            # [FUNCTION DOC] HELPER: Parser with a single flavour option restricted to vanilla.
            parser = argparse.ArgumentParser()
            parser.add_argument("--flavour", choices=["vanilla"], default="vanilla")
            return parser

        with pytest.raises(AssertionError, match="rejects the rendered --flavour durian"):
            render_cli_argv(
                {"flavour": "durian"},
                expected_obj=make_parser().parse_args([]),
                make_parser=make_parser,
                from_parsed=lambda parsed: parsed,
            )

    def test_a_field_the_parser_rewrites_blocks_the_render(self):
        # [FUNCTION DOC] With the drifting converter and no excused fields, rendering raises an
        # assertion error mentioning a roundtrip mismatch.
        """This is the failure that uncompared_fields exists to excuse."""
        args_obj = _make_cli_default_args(count=3)
        with pytest.raises(AssertionError, match="roundtrip mismatch"):
            _render_drifting(args_obj)

    def test_an_uncompared_field_is_excused_from_the_roundtrip(self):
        # [FUNCTION DOC] With the ratio named in the uncompared fields set, the drifting converter no
        # longer blocks rendering and the argv is the count option only.
        """Some upstream fields are rewritten on every parse and can never be made to match."""
        args_obj = _make_cli_default_args(count=3)
        argv = _render_drifting(args_obj, uncompared_fields=frozenset({"ratio"}))
        assert argv == ["--count", "3"]


@dataclasses.dataclass
class _AliasArgs:
    # [CLASS DOC] HELPER: Dataclass whose field names differ from the flags of its parser: a server
    # certificate path, a list of prefill url and port tuples, a boolean with default true, an
    # optional mapping and a plain text.
    server_cert_path: str | None = None
    prefill_urls: list[tuple] = dataclasses.field(default_factory=list)
    dllm_fdfo: bool = True
    mm_process_config: dict[str, Any] | None = None
    plain: str = "default"


def _make_alias_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Builds a parser with router-prefixed options: tls cert path (default
    # nothing), prefill (append action taking one or more values, default empty), dllm fdfo (boolean
    # optional action, default true), mm process config (json-decoded value, default nothing) and a
    # plain option with default "default".
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-tls-cert-path", default=None)
    parser.add_argument("--router-prefill", action="append", nargs="+", default=[])
    parser.add_argument("--router-dllm-fdfo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--router-mm-process-config", type=json.loads, default=None)
    parser.add_argument("--router-plain", default="default")
    return parser


def _alias_from_parsed(parsed: argparse.Namespace) -> _AliasArgs:
    # [FUNCTION DOC] HELPER: Converts a parsed alias namespace to alias arguments, turning each
    # prefill entry into a tuple of the url and the integer bootstrap port.
    return _AliasArgs(
        server_cert_path=parsed.router_tls_cert_path,
        prefill_urls=[(url, int(bootstrap_port)) for url, bootstrap_port in parsed.router_prefill],
        dllm_fdfo=parsed.router_dllm_fdfo,
        mm_process_config=parsed.router_mm_process_config,
        plain=parsed.router_plain,
    )


_ALIAS_FIELD_TO_DEST = {
    "server_cert_path": "router_tls_cert_path",
    "prefill_urls": "router_prefill",
    "dllm_fdfo": "router_dllm_fdfo",
    "mm_process_config": "router_mm_process_config",
    "plain": "router_plain",
}


def _render_alias(args_obj: _AliasArgs) -> list[str]:
    # [FUNCTION DOC] HELPER: Renders a command line for alias arguments with the alias parser and
    # converter and the field to destination mapping.
    return render_cli_argv(
        dataclass_to_values(args_obj),
        expected_obj=args_obj,
        make_parser=_make_alias_parser,
        from_parsed=_alias_from_parsed,
        field_to_dest=_ALIAS_FIELD_TO_DEST,
    )


class TestRenderCliArgvAgainstTheRealParserShape:
    # [CLASS DOC] Tests rendering against a parser with aliased, boolean optional, json and append options.
    """The renderer must take flag names and value shapes from the parser, not from field names."""

    def test_a_mapped_field_renders_the_dest_it_points_at(self):
        # [FUNCTION DOC] A changed plain field renders the plain router option and its value.
        """The mapping is the only thing that connects a field name to a flag."""
        argv = _render_alias(_AliasArgs(plain="other"))
        assert argv == ["--router-plain", "other"]

    def test_aliased_field_renders_the_registered_flag(self):
        # [FUNCTION DOC] A certificate path renders the tls cert path option with the path.
        """A field name that differs from its flag renders the flag the parser actually accepts."""
        argv = _render_alias(_AliasArgs(server_cert_path="/certs/a.pem"))
        assert argv == ["--router-tls-cert-path", "/certs/a.pem"]

    def test_boolean_optional_action_can_express_false(self):
        # [FUNCTION DOC] A false value for a boolean optional option renders its negative form.
        """A BooleanOptionalAction defaulting to True renders its negative option."""
        argv = _render_alias(_AliasArgs(dllm_fdfo=False))
        assert argv == ["--no-router-dllm-fdfo"]

    def test_json_valued_option_renders_a_single_json_token(self):
        # [FUNCTION DOC] A nested mapping renders the option followed by one json text token.
        """A dict option parsed by json.loads renders one JSON document, not key=value pairs."""
        argv = _render_alias(_AliasArgs(mm_process_config={"image": {"max_pixels": 1}}))
        assert argv == ["--router-mm-process-config", '{"image": {"max_pixels": 1}}']

    def test_append_action_repeats_the_flag_per_entry(self):
        # [FUNCTION DOC] Two prefill entries (url, port tuples) render the prefill option twice, each
        # followed by the url and the port as text.
        """An append option renders once per entry, spreading each entry's tokens."""
        argv = _render_alias(_AliasArgs(prefill_urls=[("http://a:1", 9000), ("http://b:2", 9001)]))
        assert argv == ["--router-prefill", "http://a:1", "9000", "--router-prefill", "http://b:2", "9001"]

    @pytest.mark.parametrize(
        "args_obj",
        [
            _AliasArgs(server_cert_path="/certs/a.pem"),
            _AliasArgs(dllm_fdfo=False),
            _AliasArgs(mm_process_config={"image": {"max_pixels": 1}}),
            _AliasArgs(prefill_urls=[("http://a:1", 9000)]),
        ],
        ids=["aliased", "boolean-optional-false", "json-dict", "append-list"],
    )
    def test_every_shape_survives_the_production_roundtrip(self, args_obj: _AliasArgs):
        # [FUNCTION DOC] Parametrized over four shapes; parsing the rendered argv with the alias
        # parser and converter gives an equal object.
        """Each shape parses back to an equal object, which is what the production assert enforces."""
        assert _alias_from_parsed(_make_alias_parser().parse_args(_render_alias(args_obj))) == args_obj

    def test_a_field_with_no_registered_option_fails_loudly(self):
        # [FUNCTION DOC] A dataclass whose changed field has no option in the parser makes rendering
        # raise an assertion error saying it cannot be rendered.
        """An unrenderable field is rejected instead of being rendered as a guessed flag."""

        @dataclasses.dataclass
        class _UnknownArgs:
            # [CLASS DOC] HELPER: Dataclass with a single field the parser does not know.
            not_on_the_cli: str = "default"

        args_obj = _UnknownArgs(not_on_the_cli="other")
        with pytest.raises(AssertionError, match="cannot be rendered"):
            render_cli_argv(
                dataclass_to_values(args_obj),
                expected_obj=args_obj,
                make_parser=_make_alias_parser,
                from_parsed=lambda parsed: _UnknownArgs(),
            )


@dataclasses.dataclass
class _RequiredDemoArgs:
    # [CLASS DOC] HELPER: Dataclass with a required model text and an integer count defaulting to zero.
    model: str
    count: int = 0


def _make_required_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Parser with a required model option and an integer count option
    # defaulting to zero.
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=0)
    return parser


def _from_parsed_required(parsed: argparse.Namespace) -> _RequiredDemoArgs:
    # [FUNCTION DOC] HELPER: Converts a parsed namespace to the required demo arguments.
    return _RequiredDemoArgs(model=parsed.model, count=parsed.count)


def _render_required(args_obj: _RequiredDemoArgs, **overrides) -> list[str]:
    # [FUNCTION DOC] HELPER: Renders a command line for required demo arguments with the required
    # parser and converter; extra keyword arguments are forwarded to the renderer.
    return render_cli_argv(
        dataclass_to_values(args_obj),
        expected_obj=args_obj,
        make_parser=_make_required_parser,
        from_parsed=_from_parsed_required,
        **overrides,
    )


class TestRenderCliArgvAlwaysRenderFields:
    # [CLASS DOC] Tests the always-render fields option, which forces named fields onto the command
    # line even when they equal the parser default; forced fields are emitted before the others.
    def test_an_always_render_field_is_emitted_exactly_once(self):
        # [FUNCTION DOC] Forcing the model field while count is also non-default emits the model option
        # exactly once and the parse of the argv equals the object.
        """An explicit policy field appears once even when another value is rendered."""
        args_obj = _RequiredDemoArgs(model="m", count=3)
        argv = _render_required(args_obj, always_render_fields=("model",))
        assert argv.count("--model") == 1
        assert _from_parsed_required(_make_required_parser().parse_args(argv)) == args_obj

    def test_an_always_render_field_is_emitted_even_at_its_own_default(self):
        # [FUNCTION DOC] Forcing the model field with a default count renders exactly the model option
        # and its value.
        """The explicit-output policy is independent of the parser default."""
        assert _render_required(_RequiredDemoArgs(model="m"), always_render_fields=("model",)) == ["--model", "m"]

    def test_an_unspecified_always_render_field_uses_the_resolved_value(self):
        # [FUNCTION DOC] An input of nothing for the forced model field falls back to the expected
        # object's value, giving the model option and "m".
        """A raw None falls back to the value resolved by the target constructor."""
        argv = render_cli_argv(
            {"model": None},
            expected_obj=_RequiredDemoArgs(model="m"),
            make_parser=_make_required_parser,
            from_parsed=_from_parsed_required,
            always_render_fields=("model",),
        )
        assert argv == ["--model", "m"]

    def test_an_always_render_field_is_emitted_at_the_parser_default(self):
        # [FUNCTION DOC] Forcing count (value equal to its default zero) renders the count option
        # first and then the required model option.
        """A value equal to the parser default is still emitted when the field is always rendered."""
        args_obj = _RequiredDemoArgs(model="m", count=0)
        assert _render_required(args_obj, always_render_fields=("count",)) == ["--count", "0", "--model", "m"]

    def test_an_always_render_field_missing_from_inputs_uses_the_expected_value(self):
        # [FUNCTION DOC] Inputs hold only count 3 while model is forced; the model value comes from
        # the expected object, rendered first, then the count.
        """An always-rendered field absent from the inputs falls back to the expected object."""
        args_obj = _RequiredDemoArgs(model="m", count=3)
        argv = render_cli_argv(
            {"count": 3},
            expected_obj=args_obj,
            make_parser=_make_required_parser,
            from_parsed=_from_parsed_required,
            always_render_fields=("model",),
        )
        assert argv == ["--model", "m", "--count", "3"]

    def test_an_always_render_field_prefers_the_raw_input_value(self):
        # [FUNCTION DOC] With a converter that halves count and a forced count whose raw input is 6
        # (expected object holds the normalized 3), the argv renders 6 (the raw value) then model.
        """A normalized expected value cannot replace its raw constructor input."""

        def from_parsed(parsed: argparse.Namespace) -> _RequiredDemoArgs:
            # [FUNCTION DOC] HELPER: Converts, halving the count with integer division.
            return _RequiredDemoArgs(model=parsed.model, count=parsed.count // 2)

        argv = render_cli_argv(
            {"model": "m", "count": 6},
            expected_obj=_RequiredDemoArgs(model="m", count=3),
            make_parser=_make_required_parser,
            from_parsed=from_parsed,
            always_render_fields=("count",),
        )
        assert argv == ["--count", "6", "--model", "m"]


_REPO_ROOT = Path(__file__).parents[4]


class TestPythonArgvPrefix:
    # [CLASS DOC] Tests the function returning the interpreter flags needed to re-execute python like
    # the current process (executable plus flags before the program selector). Runs real python
    # subprocesses with the repository root on the python path and checks the printed json prefix.
    @pytest.mark.parametrize("selector,kept", [("-c", []), ("-uc", ["-u"]), ("-OOuc", ["-OOu"])])
    def test_attached_command_does_not_replace_the_child_entrypoint(self, selector, kept):
        # [FUNCTION DOC] Parametrized over attached command selectors (plain, with an unbuffered flag,
        # with two optimize flags and unbuffered). The prefix keeps the executable, the no-bytecode
        # flag and any attached flags without the command letter; running a child with the prefix plus
        # its own command prints "child entrypoint".
        prefix = _run_prefix_printing_command([sys.executable, "-B", selector + _PRINT_PREFIX_SOURCE])
        assert prefix == [sys.executable, "-B", *kept]
        completed = subprocess.run(
            [*prefix, "-c", "print('child entrypoint')"], capture_output=True, check=True, text=True
        )
        assert completed.stdout.strip() == "child entrypoint"

    @pytest.mark.parametrize("selector,kept", [("-m", []), ("-um", ["-u"])])
    def test_attached_module_is_not_forwarded(self, tmp_path, selector, kept):
        # [FUNCTION DOC] Writes a probe module file in a temporary directory (added to the python
        # path) and runs it with an attached module selector, with and without an unbuffered flag; the
        # prefix keeps the executable, the no-bytecode flag and the kept flags, but not the module name
        # or later parent arguments.
        (tmp_path / "prefix_probe.py").write_text(_PRINT_PREFIX_SOURCE)
        prefix = _run_prefix_printing_command(
            [sys.executable, "-B", selector + "prefix_probe", "--parent-only"], extra_python_path=tmp_path
        )
        assert prefix == [sys.executable, "-B", *kept]

    @pytest.mark.parametrize(
        "flags", [["-Wignore::DeprecationWarning"], ["-Ximporttime"], ["-uW", "ignore"], ["-uX", "dev"]]
    )
    def test_short_option_values_are_not_scanned_as_entrypoint_flags(self, flags):
        # [FUNCTION DOC] Parametrized over attached and combined warning and extended-option flags;
        # the prefix is the executable followed by exactly those flags.
        assert self._run_prefix_under(flags) == [sys.executable, *flags]

    def _run_prefix_under(self, interpreter_flags: list[str]) -> list[str]:
        # [FUNCTION DOC] HELPER: Runs a python subprocess with the given interpreter flags and a
        # command that imports the prefix function from the repository, prints its result as json.
        # The subprocess inherits the environment with the python path set to the repository root; a
        # nonzero exit raises. Returns the decoded list.
        script = "import json, sys; from miles.utils.workers.argv_utils import python_argv_prefix; print(json.dumps(python_argv_prefix()))"
        completed = subprocess.run(
            [sys.executable, *interpreter_flags, "-c", script],
            capture_output=True,
            check=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        )
        return json.loads(completed.stdout)

    def test_a_plain_interpreter_yields_only_the_executable(self):
        # [FUNCTION DOC] With no flags the prefix is just the executable.
        """Nothing to preserve means the prefix is exactly what the old hardcoded rebuild produced."""
        assert self._run_prefix_under([]) == [sys.executable]

    def test_optimization_and_unbuffered_flags_are_preserved(self):
        # [FUNCTION DOC] Optimize and unbuffered flags are preserved in order.
        """A re-executed child that drops -O runs with assertions back on, silently changing its semantics."""
        assert self._run_prefix_under(["-O", "-u"]) == [sys.executable, "-O", "-u"]

    def test_a_flag_taking_a_separate_value_keeps_its_value(self):
        # [FUNCTION DOC] The extended-option flag and its separate value are both preserved.
        """-X and its value are one option, so splitting them would feed the value to the module as an argument."""
        assert self._run_prefix_under(["-X", "faulthandler"]) == [sys.executable, "-X", "faulthandler"]

    def test_the_double_dash_terminator_is_not_forwarded(self, tmp_path: Path):
        # [FUNCTION DOC] Writes a script printing the prefix and runs it as: executable, optimize flag,
        # double dash, script path. The prefix is the executable and the optimize flag only.
        """Forwarding -- would swallow the -m the caller appends, turning the module name into a script path."""
        script = tmp_path / "print_prefix.py"
        script.write_text(
            "import json\n"
            "from miles.utils.workers.argv_utils import python_argv_prefix\n"
            "print(json.dumps(python_argv_prefix()))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-O", "--", str(script)],
            capture_output=True,
            check=True,
            text=True,
            env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        )

        assert json.loads(completed.stdout) == [sys.executable, "-O"]

    def test_the_script_selector_and_everything_after_it_is_dropped(self):
        # [FUNCTION DOC] The prefix computed under an optimize flag contains neither the command nor
        # the module selector.
        """The prefix is only the interpreter part; the caller supplies its own -m and module arguments."""
        prefix = self._run_prefix_under(["-O"])

        assert "-c" not in prefix and "-m" not in prefix

    def test_a_warning_filter_flag_keeps_its_value(self):
        # [FUNCTION DOC] A warning flag with a separate filter value keeps both in the prefix when
        # the program is given by a command.
        """-W and its filter are one option, so dropping the filter both loses it and feeds it to the module."""
        assert _run_prefix_printing_command([sys.executable, "-W", "ignore", "-c", _PRINT_PREFIX_SOURCE]) == [
            sys.executable,
            "-W",
            "ignore",
        ]

    def test_a_hash_based_pyc_flag_keeps_its_long_form_value(self):
        # [FUNCTION DOC] The long option for checking hash-based bytecode files with its separate
        # value is preserved with both parts.
        """A long option taking a separate value is the case a short-flag-only scan silently mangles."""
        assert _run_prefix_printing_command(
            [sys.executable, "--check-hash-based-pycs", "always", "-c", _PRINT_PREFIX_SOURCE]
        ) == [sys.executable, "--check-hash-based-pycs", "always"]

    def test_flags_after_a_valued_flag_are_still_collected_in_order(self):
        # [FUNCTION DOC] Optimize, extended-option with value, then unbuffered are all collected in
        # order.
        """Consuming a flag's value must not stop the scan, or every later flag is lost from the child."""
        assert self._run_prefix_under(["-O", "-X", "faulthandler", "-u"]) == [
            sys.executable,
            "-O",
            "-X",
            "faulthandler",
            "-u",
        ]

    def test_a_module_run_stops_before_the_module_selector(self, tmp_path: Path):
        # [FUNCTION DOC] A probe module run with a separate module selector gives the executable and
        # optimize flag only.
        """Forwarding the parent's -m and module name would launch the parent's module instead of the child's."""
        module_path = tmp_path / "print_argv_prefix_module.py"
        module_path.write_text(_PRINT_PREFIX_SOURCE)

        prefix = _run_prefix_printing_command(
            [sys.executable, "-O", "-m", "print_argv_prefix_module"],
            extra_python_path=tmp_path,
        )

        assert prefix == [sys.executable, "-O"]

    def test_a_script_path_ends_the_prefix_and_its_arguments_are_not_absorbed(self, tmp_path: Path):
        # [FUNCTION DOC] A script run with trailing arguments that look like flags gives the
        # executable and optimize flag only.
        """Script arguments that look like flags must not be mistaken for interpreter flags of the child."""
        script_path = tmp_path / "print_argv_prefix_script.py"
        script_path.write_text(_PRINT_PREFIX_SOURCE)

        prefix = _run_prefix_printing_command([sys.executable, "-O", str(script_path), "-u", "--verbose"])

        assert prefix == [sys.executable, "-O"]

    def test_reading_the_program_from_stdin_ends_the_prefix(self):
        # [FUNCTION DOC] Running with a lone dash selector and the program text on standard input
        # gives the executable and optimize flag only.
        """The stdin selector is not an interpreter flag, so forwarding it would make the child read stdin too."""
        prefix = _run_prefix_printing_command([sys.executable, "-O", "-"], stdin_text=_PRINT_PREFIX_SOURCE)

        assert prefix == [sys.executable, "-O"]


_PRINT_PREFIX_SOURCE = (
    "import json\n"
    "from miles.utils.workers.argv_utils import python_argv_prefix\n"
    "print(json.dumps(python_argv_prefix()))\n"
)


def _run_prefix_printing_command(
    command: list[str],
    *,
    extra_python_path: Path | None = None,
    stdin_text: str | None = None,
) -> list[str]:
    # [FUNCTION DOC] HELPER: Runs the given command as a subprocess (stdout and stderr captured, a
    # nonzero exit raises) with the program text optionally supplied on standard input. The python
    # path is the repository root, followed by an optional extra directory. Returns the standard
    # output decoded as json (a list of strings).
    python_path_entries = [str(_REPO_ROOT)] + ([str(extra_python_path)] if extra_python_path is not None else [])
    completed = subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
        input=stdin_text,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(python_path_entries)},
    )
    return json.loads(completed.stdout)


class TestCoerceDictToArgs:
    # [CLASS DOC] Tests coercion of a dictionary of yaml values to typed values using a parser's
    # declared argument types, choices and arities, keyed by argument destinations. Values outside
    # the allowed names, unknown names, none values and type mismatches raise assertion errors.
    def _parser(self) -> argparse.ArgumentParser:
        # [FUNCTION DOC] HELPER: Builds a parser with swiglu (store true), bias (boolean optional),
        # normalization (text), num layers (int), lr (float), megatron to hf mode (choices raw or
        # bridge), disable bias linear (store false with destination add_bias_linear), spec (zero or
        # more values) and window size (exactly two ints).
        parser = argparse.ArgumentParser()
        parser.add_argument("--swiglu", action="store_true")
        parser.add_argument("--bias", action=argparse.BooleanOptionalAction)
        parser.add_argument("--normalization")
        parser.add_argument("--num-layers", type=int)
        parser.add_argument("--lr", type=float)
        parser.add_argument("--megatron-to-hf-mode", choices=["raw", "bridge"])
        parser.add_argument("--disable-bias-linear", dest="add_bias_linear", action="store_false")
        parser.add_argument("--spec", nargs="*")
        parser.add_argument("--window-size", nargs=2, type=int)
        return parser

    def _coerce(self, values: dict, *, allowed: set[str] | None = None) -> dict:
        # [FUNCTION DOC] HELPER: Coerces the values with the test parser, allowing the given names (all
        # keys of the values when not given) and using the context text "the overlay". Returns the
        # coerced mapping.
        allowed_names = frozenset(allowed if allowed is not None else values)
        return coerce_dict_to_args(values, parser=self._parser(), allowed_names=allowed_names, context="the overlay")

    def test_a_yaml_scalar_is_coerced_the_way_the_command_line_would(self):
        # [FUNCTION DOC] Strings "12" and "1e-5" become the integer 12 and the float 1e-5; a text stays
        # a text.
        """The overlay never reaches argparse, so this is the only place its strings become typed values."""
        assert self._coerce({"num_layers": "12", "lr": "1e-5", "normalization": "RMSNorm"}) == {
            "num_layers": 12,
            "lr": 1e-5,
            "normalization": "RMSNorm",
        }

    def test_a_float_written_where_an_int_is_declared_is_refused(self):
        # [FUNCTION DOC] The float 1.9 for an integer argument raises an assertion error saying the
        # parser would reject it.
        """int(1.9) would silently train 1 layer fewer than the config asked for."""
        with pytest.raises(AssertionError, match="would reject"):
            self._coerce({"num_layers": 1.9})

    def test_a_value_outside_the_declared_choices_is_refused(self):
        # [FUNCTION DOC] A value not among the declared choices raises an assertion error saying the
        # argument only accepts the choices.
        """argparse would reject it on the command line, and the overlay must not be the softer door."""
        with pytest.raises(AssertionError, match="only accepts"):
            self._coerce({"megatron_to_hf_mode": "bridged"})

    @pytest.mark.parametrize("flag", ["swiglu", "bias"])
    def test_a_boolean_flag_takes_a_boolean(self, flag):
        # [FUNCTION DOC] Parametrized over a store-true and a boolean optional flag; a true value is
        # kept and the text "yes" raises an assertion error saying it is not a boolean.
        """store_true and BooleanOptionalAction both carry no value on the command line."""
        assert self._coerce({flag: True}) == {flag: True}

        with pytest.raises(AssertionError, match="not a boolean"):
            self._coerce({flag: "yes"})

    def test_an_option_spelled_unlike_its_destination_is_keyed_by_the_destination(self):
        # [FUNCTION DOC] The option spelling disable_bias_linear with value false is returned under
        # the destination name add_bias_linear.
        """--disable-bias-linear writes add_bias_linear, and setting the spelling would reach no argument."""
        assert self._coerce({"disable_bias_linear": False}) == {"add_bias_linear": False}

    def test_two_names_for_one_argument_are_refused(self) -> None:
        # [FUNCTION DOC] Giving both the spelling and the destination of one argument raises an
        # assertion error saying it names one argument twice.
        """An option spelling and its destination must not silently compete for one resulting value."""
        with pytest.raises(AssertionError, match="names one argument twice"):
            self._coerce({"disable_bias_linear": False, "add_bias_linear": True})

    def test_a_name_outside_the_allowed_set_is_refused(self):
        # [FUNCTION DOC] Setting lr when only num_layers is allowed raises an assertion error saying
        # it may not override.
        """Everything else is read from the base command line, so overriding it here would be ignored."""
        with pytest.raises(AssertionError, match="may not override"):
            self._coerce({"lr": 1.0}, allowed={"num_layers"})

    def test_a_name_the_parser_does_not_declare_is_refused(self):
        # [FUNCTION DOC] An allowed name with no parser argument raises an assertion error saying the
        # parser declares no such argument.
        """An allowed name with no argument behind it cannot be typed, and would land as a stray attribute."""
        with pytest.raises(AssertionError, match="declares no such argument"):
            self._coerce({"made_up": 1})

    def test_a_value_of_none_is_refused(self):
        # [FUNCTION DOC] A none value raises an assertion error saying there is no value.
        """A yaml key with no value is a typo, not a request to unset the argument."""
        with pytest.raises(AssertionError, match="no value"):
            self._coerce({"lr": None})

    def test_a_list_reaches_an_argument_taking_several_values(self):
        # [FUNCTION DOC] A list of two strings for the zero-or-more option is returned unchanged.
        """An argument taking several values arrives from yaml as a list, which used to be refused outright."""
        assert self._coerce({"spec": ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]}) == {
            "spec": ["miles_plugins.models.glm5.glm5", "get_glm5_spec"]
        }

    def test_every_element_of_a_list_is_typed_by_the_declared_argument(self):
        # [FUNCTION DOC] The list "128", 0 for the two-int option becomes the integers 128 and 0.
        """--window-size takes a pair of ints, and an untyped overlay would hand the model two strings."""
        assert self._coerce({"window_size": ["128", 0]}) == {"window_size": [128, 0]}

    def test_a_list_written_where_a_single_value_is_declared_is_refused(self):
        # [FUNCTION DOC] A list for a single-valued option raises an assertion error saying it takes
        # a single value.
        """The command line takes one value there, so the list could never be rendered back onto it."""
        with pytest.raises(AssertionError, match="takes a single value"):
            self._coerce({"num_layers": [12, 24]})


class TestDeclaredArgDests:
    # [CLASS DOC] Tests listing the destinations of a parser's declared arguments.
    def test_every_declared_argument_is_reported_under_its_destination(self):
        # [FUNCTION DOC] A parser without help with num layers and disable bias linear (destination
        # add_bias_linear) reports the frozen set of the two destinations.
        """The whitelist is intersected with this, so a spelling rather than a destination admits nothing."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--num-layers", type=int)
        parser.add_argument("--disable-bias-linear", dest="add_bias_linear", action="store_false")

        assert declared_arg_dests(parser) == frozenset({"num_layers", "add_bias_linear"})

    def test_an_argument_this_parser_leaves_out_is_absent(self):
        # [FUNCTION DOC] An argument that the parser does not declare is absent from the result.
        """A run whose parser never declares an argument cannot be asked to override it."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--num-layers", type=int)

        assert "num_experts" not in declared_arg_dests(parser)


class TestParseDeclaredArgs:
    # [CLASS DOC] Tests parsing a model script's argument text into the mapping of only the
    # arguments it names, typed by the given parser.
    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        # [FUNCTION DOC] HELPER: Builds a parser with num layers (int), norm epsilon (float),
        # normalization (text), swiglu and add qkv bias (both store true).
        parser = argparse.ArgumentParser()
        parser.add_argument("--num-layers", type=int)
        parser.add_argument("--norm-epsilon", type=float)
        parser.add_argument("--normalization", type=str)
        parser.add_argument("--swiglu", action="store_true")
        parser.add_argument("--add-qkv-bias", action="store_true")
        return parser

    def test_every_argument_a_model_script_names_becomes_an_override(self):
        # [FUNCTION DOC] Swiglu, num layers 24 and normalization RMSNorm are all returned, typed.
        """A policy names its architecture in full, so nothing a model script declares may be dropped."""
        parsed = parse_declared_args("--swiglu --num-layers 24 --normalization RMSNorm", parser=self._parser())

        assert parsed == {"swiglu": True, "num_layers": 24, "normalization": "RMSNorm"}

    def test_a_value_is_typed_by_the_parser_rather_than_by_its_spelling(self):
        # [FUNCTION DOC] The text 1e-6 becomes the float 1e-6 and 24 the integer 24.
        """The overrides travel through yaml, where "1e-6" would otherwise arrive at megatron as a string."""
        parsed = parse_declared_args("--norm-epsilon 1e-6 --num-layers 24", parser=self._parser())

        assert parsed == {"norm_epsilon": 1e-6, "num_layers": 24}

    def test_an_argument_the_model_script_leaves_out_is_not_an_override(self):
        # [FUNCTION DOC] Only num layers is returned when only it is named (unnamed arguments with
        # defaults are not included).
        """Overriding an argument to its default would claim the model script names it when it does not."""
        parsed = parse_declared_args("--num-layers 24", parser=self._parser())

        assert parsed == {"num_layers": 24}

    def test_an_argument_no_parser_declares_fails_loudly(self):
        # [FUNCTION DOC] An option the parser does not know raises an assertion error saying it does
        # not declare it.
        """A model script this run cannot parse must not quietly yield a shorter override set."""
        with pytest.raises(AssertionError, match="does not declare"):
            parse_declared_args("--rotary-base 1000000", parser=self._parser())

    def test_an_argument_the_run_is_required_to_carry_is_not_demanded_of_the_model_script(self):
        # [FUNCTION DOC] With an extra required option on the parser that the model text does not
        # mention, parsing still succeeds and returns only num layers.
        """The real parser marks run-level arguments required, and argparse answers a missing one by
        exiting the process, so a model script that names only its architecture kills the launcher."""
        parser = self._parser()
        parser.add_argument("--rollout-batch-size", type=int, required=True)

        assert parse_declared_args("--num-layers 24", parser=parser) == {"num_layers": 24}

    def test_the_parser_still_demands_it_of_the_run_afterwards(self):
        # [FUNCTION DOC] After parsing the model text, a normal parse of the parser without the
        # required option still exits (the requirement was restored).
        """Relaxing it for one parse must not disarm the check for every later caller of the parser."""
        parser = self._parser()
        parser.add_argument("--rollout-batch-size", type=int, required=True)

        parse_declared_args("--num-layers 24", parser=parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--num-layers", "24"])


class TestWithRelaxParserRequiredArgs:
    # [CLASS DOC] Tests the context manager that temporarily makes required arguments optional.
    def test_lets_a_parser_read_argv_that_omits_its_required_arguments(self):
        # [FUNCTION DOC] Inside the context, a known-args parse without the required option succeeds
        # and the optional value is read.
        """A throwaway parser is asked what it declares, not to validate a run, so required must not fire."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--needed", required=True)
        parser.add_argument("--optional", default="d")

        with with_relax_parser_required_args(parser):
            namespace, _ = parser.parse_known_args(["--optional", "v"])

        assert namespace.optional == "v"

    def test_prints_nothing_while_the_requirement_is_relaxed(self):
        # [FUNCTION DOC] With standard error redirected to a buffer, parsing an empty argv inside the
        # context writes nothing.
        """argparse writes a whole usage screen before exiting, which a caller in a loop turns into a flood."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--needed", required=True)
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr), with_relax_parser_required_args(parser):
            parser.parse_known_args([])

        assert stderr.getvalue() == ""

    def test_puts_the_requirement_back_so_a_real_parse_still_refuses(self):
        # [FUNCTION DOC] After leaving the context, parsing an empty argv exits again.
        """The relaxation is for one read; a run that genuinely omits the argument must still be rejected."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--needed", required=True)

        with with_relax_parser_required_args(parser):
            parser.parse_known_args([])

        with pytest.raises(SystemExit):
            parser.parse_known_args([])

    def test_puts_the_requirement_back_even_when_the_parse_raises(self):
        # [FUNCTION DOC] An exception raised inside the context (suppressed by the test) still leaves
        # the parser with a required action.
        """An exception mid-read must not leave the process with a parser that validates nothing."""
        parser = argparse.ArgumentParser()
        parser.add_argument("--needed", required=True)

        with contextlib.suppress(RuntimeError), with_relax_parser_required_args(parser):
            raise RuntimeError("boom")

        assert [action for action in parser._actions if action.required]


class TestWithSuppressedParserHelp:
    # [CLASS DOC] Tests the context manager that temporarily removes the help options from a parser
    # so a probe parse leaves them as unknown extras instead of printing help and exiting.
    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        # [FUNCTION DOC] HELPER: Builds a parser (with default help) with a required needed option
        # and an optional option defaulting to "d".
        parser = argparse.ArgumentParser()
        parser.add_argument("--needed", required=True)
        parser.add_argument("--optional", default="d")
        return parser

    def test_the_probe_parse_neither_prints_help_nor_exits(self):
        # [FUNCTION DOC] With both relaxations active and standard output redirected, a known-args
        # parse of the help flag returns the flag as an extra and prints nothing.
        """miles --help printed the probe's relaxed usage, which lists what the run requires as optional."""
        parser = self._parser()
        stdout = io.StringIO()

        with (
            contextlib.redirect_stdout(stdout),
            with_relax_parser_required_args(parser),
            with_suppressed_parser_help(parser),
        ):
            _namespace, extras = parser.parse_known_args(["--help"])

        assert extras == ["--help"]
        assert stdout.getvalue() == ""

    def test_the_arguments_the_probe_is_after_are_still_read(self):
        # [FUNCTION DOC] Inside the context, normal options are still parsed (optional equals "v")
        # and the help flag is left in the extras.
        """The probe exists to find the user-provided functions, which it must still parse out of argv."""
        parser = self._parser()

        with with_suppressed_parser_help(parser):
            namespace, extras = parser.parse_known_args(["--needed", "n", "--optional", "v", "--help"])

        assert namespace.optional == "v"
        assert extras == ["--help"]

    def test_the_real_parse_still_answers_help(self):
        # [FUNCTION DOC] After the context, a parse with the help flag exits (system exit).
        """Help moves to the parse that knows what the run requires; it is not taken away from the user."""
        parser = self._parser()

        with with_suppressed_parser_help(parser):
            parser.parse_known_args(["--needed", "n"])

        with pytest.raises(SystemExit):
            parser.parse_known_args(["--needed", "n", "--help"])

    def test_both_spellings_of_help_are_gone_only_for_the_duration(self):
        # [FUNCTION DOC] Inside the context neither -h nor --help is registered; after it both are
        # registered again.
        """argparse registers -h and --help separately, and leaving either behind still runs the help action."""
        parser = self._parser()

        with with_suppressed_parser_help(parser):
            suppressed = set(parser._option_string_actions)

        assert {"-h", "--help"}.isdisjoint(suppressed)
        assert {"-h", "--help"} <= set(parser._option_string_actions)

    def test_help_is_restored_even_when_the_parse_raises(self):
        # [FUNCTION DOC] An exception inside the context (suppressed by the test) still leaves both
        # help spellings registered afterwards.
        """A probe that fails mid-read must not leave the process with a parser that cannot print help."""
        parser = self._parser()

        with contextlib.suppress(RuntimeError), with_suppressed_parser_help(parser):
            raise RuntimeError("boom")

        assert {"-h", "--help"} <= set(parser._option_string_actions)

    def test_the_other_options_are_left_registered_while_help_is_suppressed(self):
        # [FUNCTION DOC] Inside the context, the needed and optional options are still registered.
        """Only the help action is taken out; removing more would make the probe read a different command line."""
        parser = self._parser()

        with with_suppressed_parser_help(parser):
            inside = set(parser._option_string_actions)

        assert {"--needed", "--optional"} <= inside

    def test_a_parser_that_declares_no_help_is_left_alone(self):
        # [FUNCTION DOC] For a parser built without help, the context works: the optional value is
        # parsed, there are no extras, and only the optional option is registered.
        """Every parser the probe is handed is not required to carry a help action at all."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--optional", default="d")

        with with_suppressed_parser_help(parser):
            namespace, extras = parser.parse_known_args(["--optional", "v"])

        assert namespace.optional == "v"
        assert extras == []
        assert set(parser._option_string_actions) == {"--optional"}
