# [MODULE DOC]
# Tests for rendering SGLang server arguments to a command-line argv and parsing them back
# (server_args_to_argv / parse_server_args_argv). Round-trip tests compute the engine's real
# server-args mapping for many scenarios (regular, prefill, decode, multi-node, dtype and
# parallel sizes, DP attention, overrides, LoRA, IPv6 rendezvous address, colocated prefill CUDA
# graph backend, prefixed pass-through options) and assert that every field of the parsed record
# equals the intended one. A sweep test checks that every field of SGLang's server-args record
# resolves to a CLI action that round-trips a value of its own shape, with a documented list of
# fields known to have no renderable CLI marked as strict expected failures. An autouse fixture
# restores the process environment because SGLang writes tuning switches into it when validating
# server arguments. Skipped when SGLang is not installed.
from __future__ import annotations

import argparse
import functools
import json
import os
from argparse import Namespace
from typing import Any

import pytest
from tests.fast.backends.sglang_utils.conftest import make_engine_args as _args
from tests.fast.backends.sglang_utils.conftest import tiny_model_path

pytest.importorskip("sglang")

from sglang.srt.server_args import ServerArgs

from miles.backends.sglang_utils.server_args_utils import parse_server_args_argv, server_args_to_argv
from miles.backends.sglang_utils.sglang_engine import _compute_server_args
from miles.utils.workers.argv_utils import _actions_by_dest, _record_field_names, _render_action_argv, _resolve_action


@pytest.fixture(autouse=True)
def _assert_sweep_restores_env():
    # [FUNCTION DOC] HELPER: Autouse fixture. Snapshots a copy of the process environment, yields
    # to the test, then clears the environment and restores the snapshot, so environment changes
    # made by SGLang's server-args validation do not leak into other tests.
    # sglang writes tuning switches into the process environment while it validates a ServerArgs, and
    # this file builds one per cli option, so anything reading os.environ afterwards reads the sweep
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


_FIELDS_WITHOUT_A_RENDERABLE_CLI: dict[str, str] = {
    "custom_sigquit_handler": "A Python-only callable hook; sglang registers no CLI option for it.",
    "stat_loggers": "A Python-only injection point; sglang registers no CLI option for it.",
    "uses_mamba_radix_cache": "Derived inside __post_init__; sglang registers no CLI option for it.",
    "_speculative_draft_quantization_explicitly_set": (
        "Derived inside __post_init__ and declared Arg(no_cli=True); sglang registers no CLI option for it."
    ),
    "_radix_eviction_policy_explicitly_set": (
        "Derived during resolution and declared Arg(no_cli=True); sglang registers no CLI option for it."
    ),
    "_swa_full_tokens_ratio_explicitly_set": (
        "Derived by the cache hook and declared Arg(no_cli=True); sglang registers no CLI option for it."
    ),
    "speculative_boundary_reduction": (
        "Derived from boundary_reduction and declared Arg(no_cli=True); sglang registers no CLI option for it."
    ),
    "grpc_worker_threads": (
        "Env-only (SGLANG_GRPC_WORKER_THREADS) and declared Arg(no_cli=True); sglang registers no CLI option for it."
    ),
    "cuda_graph_config": (
        "The CLI parses a validated per-phase JSON object while ServerArgs holds a CudaGraphConfig "
        "instance, so no generically rendered token parses back to the same value."
    ),
}


def _server_args(
    *,
    worker_type: str = "regular",
    node_rank: int = 0,
    dist_init_addr: str = "10.0.0.1:20000",
    args: Namespace | None = None,
    sglang_overrides: dict | None = None,
    disaggregation_bootstrap_port: int | None = None,
    num_gpus_per_engine: int = 1,
    random_seed: int = 0,
) -> dict:
    # [FUNCTION DOC] HELPER: Keyword-only. Computes the engine's server-args mapping through the
    # private computation function with fixed values (gated launch port 20034, NCCL port 20031,
    # host 10.0.0.1, port 30000, base GPU id 0, engine info bootstrap port 20033) and the given
    # worker type (default regular), node rank (0), rendezvous address (10.0.0.1:20000), args
    # (default: the shared engine-args factory), SGLang overrides, disaggregation bootstrap port,
    # GPUs per engine (1) and random seed (0). The device is forced to "cuda" unless the overrides
    # say otherwise, because SGLang cannot probe an accelerator on CPU-only runners. Returns the
    # mapping.
    # ServerArgs probes the local accelerator when no device is given, which a CPU-only
    # CI runner cannot answer. Production resolves it to the engine's own device the same way.
    overrides = {"device": "cuda", **(sglang_overrides or {})}
    server_args_dict = _compute_server_args(
        args or _args(),
        node_rank=node_rank,
        gated_launch_port=20034,
        dist_init_addr=dist_init_addr,
        nccl_port=20031,
        host="10.0.0.1",
        port=30000,
        worker_type=worker_type,
        disaggregation_bootstrap_port=disaggregation_bootstrap_port,
        base_gpu_id=0,
        engine_info_bootstrap_port=20033,
        sglang_overrides=overrides,
        num_gpus_per_engine=num_gpus_per_engine,
        random_seed=random_seed,
    )
    return server_args_dict


def _assert_roundtrips(server_args_dict: dict) -> None:
    # [FUNCTION DOC] HELPER: Renders the mapping to argv, parses it back to a record, builds the
    # wanted record from the mapping with the device filled from the parsed record when the
    # mapping has none, and asserts that the list of field names whose parsed and wanted values
    # differ is empty. Raises an assertion error listing the differing field names otherwise.
    """Every field the launched process ends up with matches what miles asked for.

    ``device`` is the one field miles fills in before rendering, so an unset input is
    compared against the accelerator the argv carries."""
    parsed = parse_server_args_argv(server_args_to_argv(server_args_dict))
    device = server_args_dict.get("device") or parsed.device
    wanted = ServerArgs(**{**server_args_dict, "device": device})
    differing = [name for name in _record_field_names(wanted) if getattr(parsed, name) != getattr(wanted, name)]
    assert differing == []


class TestServerArgsToArgv:
    # [CLASS DOC] Round-trip tests of the argv rendering for the engine's computed server args.
    def test_a_regular_engine_launch_roundtrips(self):
        # [FUNCTION DOC] Default computed arguments: the rendered argv has no disaggregation-mode
        # flag, and the argv round-trips.
        """The exact ServerArgs the launch computes survives the argv boundary."""
        server_args = _server_args()
        assert "--disaggregation-mode" not in server_args_to_argv(server_args)
        _assert_roundtrips(server_args)

    def test_the_identity_flags_are_always_rendered_exactly_once(self):
        # [FUNCTION DOC] The trust-remote-code, model-path, host, port and device flags each occur
        # exactly once in the rendered argv of the default arguments.
        """Model path, addressing, and device must stay explicit even at CLI defaults."""
        argv = server_args_to_argv(_server_args())
        for flag in ("--trust-remote-code", "--model-path", "--host", "--port", "--device"):
            assert argv.count(flag) == 1

    def test_an_unspecified_device_renders_the_auto_detected_accelerator(self, monkeypatch):
        # [FUNCTION DOC] Patches the renderer module's device detection to report "cuda" and sets
        # the mapping's device to none. Asserts the mapping keeps none, the argv carries "cuda"
        # after the device flag, and the round trip holds.
        """An unset low-level device renders the accelerator chosen by ServerArgs instead of the text None."""
        monkeypatch.setattr("miles.backends.sglang_utils.server_args_utils.get_device", lambda: "cuda")
        server_args = _server_args()
        server_args["device"] = None
        argv = server_args_to_argv(server_args)

        assert server_args["device"] is None
        assert argv[argv.index("--device") + 1] == "cuda"
        _assert_roundtrips(server_args)

    def test_a_prefill_worker_roundtrips(self):
        # [FUNCTION DOC] A prefill worker with bootstrap port 20090: the mapping's disaggregation
        # mode is "prefill", the argv contains the disaggregation-mode flag, and the round trip holds.
        """PD-disaggregation prefill fields survive the argv boundary."""
        server_args = _server_args(worker_type="prefill", disaggregation_bootstrap_port=20090)
        assert server_args["disaggregation_mode"] == "prefill"
        assert "--disaggregation-mode" in server_args_to_argv(server_args)
        _assert_roundtrips(server_args)

    def test_a_decode_worker_roundtrips(self):
        # [FUNCTION DOC] A decode worker: the mapping's disaggregation mode is "decode" and the
        # round trip holds.
        """PD-disaggregation decode fields survive the argv boundary."""
        server_args = _server_args(worker_type="decode")
        assert server_args["disaggregation_mode"] == "decode"
        _assert_roundtrips(server_args)

    def test_a_multi_node_rank_roundtrips(self):
        # [FUNCTION DOC] Node rank 1 of an engine spanning 16 GPUs (8 per node): the mapping has 2
        # nodes and node rank 1, and the round trip holds.
        """nnodes, node_rank and tp_size of a multi-node engine survive the boundary."""
        server_args = _server_args(
            node_rank=1,
            num_gpus_per_engine=16,
            args=_args(rollout_num_gpus_per_engine=16),
        )
        assert server_args["nnodes"] == 2 and server_args["node_rank"] == 1
        _assert_roundtrips(server_args)

    def test_dtype_and_parallel_sizes_roundtrip(self):
        # [FUNCTION DOC] fp16 with data-parallel size 2 and expert-parallel size 2: the mapping's
        # dtype is "float16" and the round trip holds.
        """fp16 and dp/pp/ep sizes land in the argv and parse back."""
        server_args = _server_args(args=_args(fp16=True, sglang_dp_size=2, sglang_ep_size=2))
        assert server_args["dtype"] == "float16"
        _assert_roundtrips(server_args)

    def test_dp_attention_inputs_are_passed_through_raw(self):
        # [FUNCTION DOC] 2-GPU engine with data-parallel size 2, DP attention on, chunked prefill
        # size 4096 and schedule conservativeness 1.0. Builds the expected record directly from
        # the mapping. Asserts: the schedule-conservativeness flag is absent from argv (value at
        # its default), the chunked-prefill flag carries 4096, the parsed and expected records
        # agree on both values (1.0 and 4096), and the full round trip holds.
        """sglang applies the interacting DP-attention defaults when the engine resolves its
        arguments, so the boundary carries the raw inputs unchanged."""
        server_args = _server_args(
            args=_args(
                rollout_num_gpus_per_engine=2,
                sglang_dp_size=2,
                sglang_enable_dp_attention=True,
                sglang_chunked_prefill_size=4096,
                sglang_schedule_conservativeness=1.0,
            ),
            num_gpus_per_engine=2,
        )
        expected = ServerArgs(**server_args)
        argv = server_args_to_argv(server_args)
        parsed = parse_server_args_argv(argv)

        assert "--schedule-conservativeness" not in argv
        assert argv[argv.index("--chunked-prefill-size") + 1] == "4096"
        assert parsed.schedule_conservativeness == expected.schedule_conservativeness == 1.0
        assert parsed.chunked_prefill_size == expected.chunked_prefill_size == 4096
        _assert_roundtrips(server_args)

    def test_sglang_overrides_roundtrip(self):
        # [FUNCTION DOC] Overrides of static memory fraction 0.5 and log level "warning" appear in
        # the mapping (fraction asserted) and survive the round trip.
        """User overrides merged into the dict survive the argv boundary."""
        server_args = _server_args(sglang_overrides={"mem_fraction_static": 0.5, "log_level": "warning"})
        assert server_args["mem_fraction_static"] == 0.5
        _assert_roundtrips(server_args)

    def test_lora_fields_roundtrip(self):
        # [FUNCTION DOC] LoRA rank 8 with a target-modules argument: the mapping has LoRA enabled
        # and the round trip holds.
        """enable_lora, ranks and the target-modules list survive the boundary."""
        server_args = _server_args(args=_args(lora_rank=8, target_modules=["linear_qkv"]))
        assert server_args["enable_lora"]
        _assert_roundtrips(server_args)

    def test_an_ipv6_dist_init_addr_roundtrips(self):
        # [FUNCTION DOC] A bracketed IPv6 rendezvous address with port 20000 survives the round trip.
        """The bracketed v6 rendezvous address survives the argv boundary."""
        server_args = _server_args(dist_init_addr="[fd00::1]:20000")
        _assert_roundtrips(server_args)

    def test_a_colocate_prefill_cuda_graph_backend_roundtrips(self):
        # [FUNCTION DOC] An override disabling the prefill CUDA graph backend: the rendered argv
        # must not contain the CUDA-graph-config flag (that derived object has no faithful CLI
        # spelling) and the round trip holds.
        """Colocate forces the prefill cuda graph backend off, which sglang folds into a
        derived config object that has no faithful command-line spelling."""
        server_args = _server_args(sglang_overrides={"cuda_graph_backend_prefill": "disabled"})
        argv = server_args_to_argv(server_args)
        assert "--cuda-graph-config" not in argv
        _assert_roundtrips(server_args)

    def test_lora_adapter_paths_roundtrip(self):
        # [FUNCTION DOC] LoRA rank 8 with an adapter path in rollout-only debug mode (so the
        # engine loads the adapter itself): the name=path adapter mapping survives the round trip.
        """The name=path lora mapping survives the argv boundary."""
        server_args = _server_args(
            args=_args(
                lora_rank=8, target_modules=["linear_qkv"], lora_adapter_path="/fake/adapter", debug_rollout_only=True
            )
        )
        _assert_roundtrips(server_args)


class TestSglangPrefixedPassthrough:
    # [CLASS DOC] Tests that options given with the sglang prefix reach the engine's server args.
    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            pytest.param("fp4_gemm_runner_backend", "marlin", id="cli-name-renamed-field"),
            pytest.param("preferred_sampling_params", {"temperature": 0.7}, id="json-parsed-field"),
            pytest.param("dllm_fdfo", False, id="boolean-optional-action-set-to-false"),
            pytest.param("disable_cuda_graph", True, id="field-reachable-only-by-a-legacy-alias"),
        ],
    )
    def test_a_prefixed_user_flag_reaches_the_engine_command(self, field_name: str, value: Any) -> None:
        # [FUNCTION DOC] Parametrized over four fields with unusual option shapes: FP4 GEMM runner
        # backend ("marlin", renamed CLI option), preferred sampling params (a JSON-parsed
        # mapping), a boolean-optional field set to false, and the disable-CUDA-graph field
        # reachable only through a legacy alias. Sets the run argument with the sglang prefix,
        # asserts the computed mapping holds the value, and the round trip holds.
        """A --sglang-<field> value lands on the engine ServerArgs and survives the argv boundary."""
        server_args = _server_args(args=_args(**{f"sglang_{field_name}": value}))
        assert server_args[field_name] == value
        _assert_roundtrips(server_args)


class TestEveryServerArgsFieldIsRenderable:
    # [CLASS DOC] Sweep over every field of SGLang's server-args record.
    @pytest.mark.parametrize(
        "field_name",
        [
            (
                pytest.param(
                    name,
                    marks=pytest.mark.xfail(reason=_FIELDS_WITHOUT_A_RENDERABLE_CLI[name], strict=True),
                    id=name,
                )
                if name in _FIELDS_WITHOUT_A_RENDERABLE_CLI
                else pytest.param(name, id=name)
            )
            for name in _record_field_names(ServerArgs)
        ],
    )
    def test_a_field_renders_to_argv_that_parses_back_to_the_same_value(self, field_name: str) -> None:
        # [FUNCTION DOC] Parametrized over all field names of the real server-args record at
        # collection time; fields listed in the known-unrenderable table are marked as strict
        # expected failures with their reason. Builds the real parser, resolves the field's CLI
        # action (no field-to-destination renames), takes the baseline default for its
        # destination, and asserts that at least one candidate value renders and parses back to
        # itself (the first such value is not none); the failure message names the field and its
        # first option string.
        """Every ServerArgs field resolves to a CLI action that round-trips a value of its own shape."""
        parser = _make_server_args_parser()
        action = _resolve_action(_actions_by_dest(parser), field_name=field_name, field_to_dest={})
        default_value = getattr(_baseline_namespace(), action.dest, None)

        accepted = _first_roundtripping_value(action=action, default_value=default_value)

        assert (
            accepted is not None
        ), f"{field_name!r} renders to argv that {action.option_strings[0]!r} cannot parse back"


def _make_server_args_parser() -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Returns a fresh parser with all of SGLang's server-args CLI options.
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    return parser


@functools.lru_cache(maxsize=1)
def _baseline_namespace() -> Namespace:
    # [FUNCTION DOC] HELPER: Cached (one entry). Parses only the model-path argument with the
    # server-args parser and returns the namespace, which holds every option's default.
    return _make_server_args_parser().parse_args(_model_argv())


def _model_argv() -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the minimal argv: the model-path flag and the tiny model path.
    return ["--model-path", str(tiny_model_path())]


def _first_roundtripping_value(*, action: argparse.Action, default_value: object) -> object | None:
    # [FUNCTION DOC] HELPER: Walks the sweep candidates for the action in order. For each, renders
    # the action's argv for that value, parses it (after the model argv) with a fresh parser
    # (skipping the candidate if the parser exits), and returns the first candidate whose parsed
    # value for the action's destination equals the candidate. Returns none if no candidate works.
    for value in _sweep_candidates(action=action, default_value=default_value):
        argv = _render_action_argv(action, value)
        try:
            namespace = _make_server_args_parser().parse_args([*_model_argv(), *argv])
        except SystemExit:
            continue
        if getattr(namespace, action.dest) == value:
            return value
    return None


def _sweep_candidates(*, action: argparse.Action, default_value: object) -> list[object]:
    # [FUNCTION DOC] HELPER: Returns candidate non-default values by action kind, in this order:
    # boolean-optional -> the negation of the default; zero-argument flag -> its constant; JSON
    # object type -> a one-key mapping; JSON list type (recognized by the type function's name)
    # -> a one-element list of a string; list-valued nargs (star, plus, fixed count) -> one-element
    # lists of each scalar candidate computed with no default; otherwise the scalar candidates.
    if isinstance(action, argparse.BooleanOptionalAction):
        return [not bool(default_value)]

    if action.nargs == 0:
        return [action.const]

    if action.type is json.loads:
        return [{"sweep-key": "sweep-value"}]

    if getattr(action.type, "__name__", "") == "json_list_type":
        return [["sweep-value"]]

    if action.nargs in ("*", "+") or isinstance(action.nargs, int):
        return [[element] for element in _scalar_candidates(action=action, default_value=None)]

    return _scalar_candidates(action=action, default_value=default_value)


def _scalar_candidates(*, action: argparse.Action, default_value: object) -> list[object]:
    # [FUNCTION DOC] HELPER: Returns scalar candidates: with choices, the first choice differing
    # from the default (or the default if none differs); integer type -> default plus one if the
    # default is an integer else 1; float type -> default plus 1.0 if a float else 1.0; string or
    # untyped -> "sweep-value" (or "other-sweep-value" when that is the default); any other type
    # -> the three values 1, "sweep-value" and 1.0 to try in turn.
    if action.choices:
        return [next((choice for choice in action.choices if choice != default_value), default_value)]

    if action.type is int:
        return [default_value + 1 if isinstance(default_value, int) else 1]

    if action.type is float:
        return [default_value + 1.0 if isinstance(default_value, float) else 1.0]

    if action.type in (str, None):
        return ["other-sweep-value" if default_value == "sweep-value" else "sweep-value"]

    return [1, "sweep-value", 1.0]


def test_the_sweep_leaves_the_environment_as_it_found_it():
    # [FUNCTION DOC] Asserts the SGLang custom all-reduce v2 tuning switch is absent from the
    # process environment (the autouse fixture restores the environment after each test, so a
    # leaked switch from an earlier sweep case would show up here).
    """A tuning switch left behind is read as the machine's own by every later test that consults it."""
    assert os.environ.get("SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2") is None
