# [MODULE DOC] CPU tests of the DeepSeek sparse attention (DSA) command-line arguments for the
# GLM-5 / DeepSeek-V3.2 model family: the argument registrar's defaults, the normalization function
# that maps a miles-implementation argument set to Megatron's native DSA settings from a Hugging
# Face config (indexer heads, head dim, top-k, rope interleaving, top-k frequency/skip offset,
# fixed norm epsilon, kernel backend, loss coefficient, top-k backend, spec), its idempotence, and
# its early value errors for incompatible configurations (negative loss coefficient, bridge mode,
# wrong spec, contiguous CP, non-allgather CP communication, unsupported model type); plus the
# shared default-argument pass (cp communication defaults) and the debug worker's registration of
# these arguments, both exercised by executing only the relevant source pieces with stubs.
import argparse
import ast
import importlib.util
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE
from miles_plugins.models.deepseek_v4.arguments import add_dsv4_arguments
from miles_plugins.models.glm5.arguments import (
    MEGATRON_DSA_SPEC,
    MILES_DSA_SPEC,
    add_dsa_arguments,
    normalize_dsa_args,
)


def _args(**overrides):
    # [FUNCTION DOC] HELPER: Builds an argument namespace for the normalizer with defaults: a request
    # for the native implementation (implementation "megatron", the miles DSA spec list, raw conversion mode,
    # CP size 1, no all-gather CP, cudnn kernel backend, no loss coefficient, torch top-k backend, no
    # CP communication type). Keyword overrides replace entries (dictionary merge). Returns the namespace.
    values = dict(
        dsa_impl="megatron",
        spec=list(MILES_DSA_SPEC),
        megatron_to_hf_mode="raw",
        context_parallel_size=1,
        allgather_cp=False,
        dsa_kernel_backend="cudnn",
        dsa_indexer_loss_coeff=None,
        miles_dsa_topk_backend="torch",
        cp_comm_type=None,
    )
    return Namespace(**(values | overrides))


def _hf_config(**overrides):
    # [FUNCTION DOC] HELPER: Builds a fake Hugging Face config namespace with model type
    # deepseek_v32, 64 indexer heads, indexer head dim 128 and indexer top-k 2048; keyword overrides
    # replace entries.
    values = dict(model_type="deepseek_v32", index_n_heads=64, index_head_dim=128, index_topk=2048)
    return SimpleNamespace(**(values | overrides))


def test_default_preserves_the_existing_miles_path():
    # [FUNCTION DOC] Parses an empty command line with the registrar: asserts the only resulting
    # fields are implementation "miles", top-k backend "torch" and CP communication type none; then
    # normalizing with no HF config leaves the namespace unchanged.
    args = add_dsa_arguments(ArgumentParser()).parse_args([])
    assert vars(args) == {"dsa_impl": "miles", "miles_dsa_topk_backend": "torch", "cp_comm_type": None}
    before = vars(args).copy()
    normalize_dsa_args(args, None)
    assert vars(args) == before


@pytest.mark.parametrize(
    ("hf_overrides", "interleaved", "frequency", "offset"),
    [
        ({"model_type": "deepseek_v32"}, False, 1, 0),
        ({"model_type": "glm_moe_dsa", "indexer_rope_interleave": True}, True, 1, 0),
        (
            {
                "model_type": "glm_moe_dsa",
                "indexer_rope_interleave": True,
                "index_topk_freq": 4,
                "index_skip_topk_offset": 2,
            },
            True,
            4,
            2,
        ),
    ],
    ids=["deepseek-v32", "glm5", "glm52-shared-indices"],
)
def test_native_dsa_preserves_checkpoint_indexer_conventions(hf_overrides, interleaved, frequency, offset):
    # [FUNCTION DOC] Parametrized over three HF configs (deepseek_v32; glm_moe_dsa with interleaved
    # indexer rope; glm_moe_dsa with top-k frequency 4 and skip offset 2). After normalization asserts:
    # the spec is the native Megatron DSA spec; experimental attention variant "dsa" and experimental
    # mode on; indexer heads 64, head dim 128, top-k 2048; rope interleave flags equal the config; top-k
    # frequency and skip offset equal the config (defaults 1 and 0); no activation rotation; norm
    # epsilon 1e-6 in fp32; kernel backend unchanged; loss coefficient defaulted to 0.0; top-k backend
    # carried over; and that normalizing a second time changes nothing.
    args = _args()
    hf_config = _hf_config(**hf_overrides)
    normalize_dsa_args(args, hf_config)

    assert args.spec == list(MEGATRON_DSA_SPEC)
    assert args.experimental_attention_variant == "dsa"
    assert args.enable_experimental is True
    assert (args.dsa_indexer_n_heads, args.dsa_indexer_head_dim, args.dsa_indexer_topk) == (64, 128, 2048)
    assert args.dsa_indexer_rope_interleaved is interleaved
    assert args.indexer_rope_interleave is interleaved
    assert (args.dsa_indexer_topk_freq, args.dsa_indexer_skip_topk_offset) == (frequency, offset)
    assert args.dsa_indexer_rotate_activation is False
    assert args.dsa_indexer_k_norm_epsilon == 1e-6
    assert args.dsa_indexer_k_norm_fp32 is True
    assert args.dsa_kernel_backend == "cudnn"
    assert args.dsa_indexer_loss_coeff == 0.0
    assert args.dsa_indexer_topk_backend == "torch"

    # Checkpoint conversion and training can both normalize an already native spec.
    before = vars(args).copy()
    normalize_dsa_args(args, hf_config)
    assert vars(args) == before


@pytest.mark.parametrize("loss_coeff", [0.0, 0.001])
def test_explicit_indexer_training_objective_is_preserved(loss_coeff):
    # [FUNCTION DOC] For explicit loss coefficients 0.0 and 0.001, normalization leaves the value
    # unchanged.
    args = _args(dsa_indexer_loss_coeff=loss_coeff)
    normalize_dsa_args(args, _hf_config())
    assert args.dsa_indexer_loss_coeff == loss_coeff


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dsa_indexer_loss_coeff": -0.001}, "--dsa-indexer-loss-coeff must be non-negative"),
        ({"megatron_to_hf_mode": "bridge"}, "requires --megatron-to-hf-mode raw"),
        ({"spec": None}, "requires the shared DeepSeek-V3.2/GLM DSA spec"),
        ({"spec": ["miles_plugins.models.deepseek_v4", "get_dsv4_spec"]}, "requires the shared"),
        ({"context_parallel_size": 2, "allgather_cp": True}, "uses zigzag CP token partitioning"),
        ({"cp_comm_type": ["p2p"]}, "requires --cp-comm-type allgather"),
        ({"cp_comm_type": ["allgather", "a2a"]}, "requires --cp-comm-type allgather"),
    ],
    ids=[
        "negative-loss",
        "bridge",
        "no-spec",
        "v4-spec",
        "contiguous-cp",
        "explicit-p2p",
        "mixed-cp",
    ],
)
def test_incompatible_native_configuration_fails_early(overrides, message):
    # [FUNCTION DOC] Parametrized over seven incompatible argument sets (negative loss coefficient;
    # bridge conversion mode; no spec; the DeepSeek-V4 spec; contiguous CP with all-gather CP;
    # explicit p2p CP communication; mixed communication types). Asserts normalization raises a value
    # error whose message matches the expected text.
    with pytest.raises(ValueError, match=message):
        normalize_dsa_args(_args(**overrides), _hf_config())


def test_unsupported_checkpoint_cannot_select_native_dsa():
    # [FUNCTION DOC] An HF config with model type deepseek_v4 makes normalization raise a value
    # error saying that type is not supported.
    with pytest.raises(ValueError, match="does not support model_type='deepseek_v4'"):
        normalize_dsa_args(_args(), _hf_config(model_type="deepseek_v4"))


def test_native_cp_preserves_per_layer_allgather_communication_with_zigzag_partitioning():
    # [FUNCTION DOC] With CP size 4 and a per-layer list of two allgather communication types,
    # normalization keeps the list as is and leaves the all-gather-CP flag false.
    args = _args(context_parallel_size=4, cp_comm_type=["allgather", "allgather"])
    normalize_dsa_args(args, _hf_config())
    assert args.cp_comm_type == ["allgather", "allgather"]
    assert args.allgather_cp is False


def test_native_dsa_keeps_the_requested_topk_backend():
    # [FUNCTION DOC] Parses command-line flags selecting the native implementation and the
    # flashinfer top-k backend, merges the parsed values into the default test namespace, normalizes,
    # and asserts the indexer top-k backend is flashinfer.
    parsed = add_dsa_arguments(ArgumentParser()).parse_args(
        ["--dsa-impl", "megatron", "--miles-dsa-topk-backend", "flashinfer"]
    )
    args = _args(**vars(parsed))
    normalize_dsa_args(args, _hf_config())
    assert args.dsa_indexer_topk_backend == "flashinfer"


@pytest.fixture
def megatron_defaults(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Using the monkeypatch fixture, puts stub entries in the module
    # table for the tokenizer vocabulary-padding helper (returns the size unchanged), Megatron's
    # training arguments module (placeholder parse and validate functions) and the HF config loader
    # (returns the default fake config). Then loads the Megatron backend's arguments source file by
    # path (four directory levels above this test) under a private module name and executes it. Returns
    # the module's default-argument-setting function.
    # Exercise the real normalization entrypoint without importing GPU startup dependencies.
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.tokenizers.utils.build_tokenizer",
        SimpleNamespace(vocab_size_with_padding=lambda size, args: size),
    )
    monkeypatch.setitem(
        sys.modules, "megatron.training.arguments", SimpleNamespace(parse_args=None, validate_args=None)
    )
    monkeypatch.setitem(
        sys.modules, "miles.utils.hf_utils.config", SimpleNamespace(load_hf_config=lambda path: _hf_config())
    )
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/arguments.py"
    spec = importlib.util.spec_from_file_location("dsa_megatron_arguments_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.set_default_megatron_args


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["p2p"]),
        (["--cp-comm-type", "a2a"], ["a2a"]),
        (["--dsa-impl", "megatron"], ["allgather"]),
        (["--dsa-impl", "megatron", "--cp-comm-type", "allgather"], ["allgather"]),
    ],
    ids=["default-miles", "explicit-miles-cp", "default-native", "explicit-native-cp"],
)
def test_shared_parser_normalizes_omitted_cp_without_overriding_explicit_settings(megatron_defaults, argv, expected):
    # [FUNCTION DOC] Parametrized over four command lines (default; explicit a2a; native with no
    # explicit CP communication; native with explicit allgather). Builds a parser with the CP
    # communication option (default list with p2p) plus the DSA arguments, merges its parsed values and
    # some extra training fields into the default namespace, runs the default-setting function, and
    # asserts the CP communication type: p2p for the default miles path, a2a when explicit, allgather
    # for native DSA (also when given explicitly).
    parser = ArgumentParser()
    parser.add_argument("--cp-comm-type", nargs="+", default=["p2p"])
    add_dsa_arguments(parser)
    args = _args(
        **vars(parser.parse_args(argv)),
        optimizer="adam",
        fp16=False,
        seq_length=None,
        vocab_size=None,
        tokenizer_model=None,
        tokenizer_type=None,
        hf_checkpoint="/model",
    )
    megatron_defaults(args)
    assert args.cp_comm_type == expected


@pytest.mark.parametrize("argv,implementation", [([], "miles"), (["--dsa-impl", "megatron"], "megatron")])
def test_worker_parser_registers_dsa_arguments(argv, implementation):
    # [FUNCTION DOC] Parametrized over default and native implementation selections. Extracts only the
    # worker's argument-registration function from the worker entry module's source by parsing and
    # executing its definition with the needed helpers supplied, runs it on a parser that already has
    # the CP communication option, parses a command line with script options (HF checkpoint, token ids
    # file, flashinfer top-k backend) plus the parametrized flags, and asserts: implementation as
    # expected, top-k backend flashinfer, CP communication type none (the worker clears the default),
    # and the script-args bridge reads the HF checkpoint as the given path.
    # Exercise the real registrar without importing the worker's GPU startup stack.
    path = Path(__file__).resolve().parents[4] / "miles/utils/debug_utils/run_megatron/worker/main.py"
    tree = ast.parse(path.read_text())
    registrar = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_register_worker_arguments"
    )
    namespace = {
        "argparse": argparse,
        "WORKER_SCRIPT_ARGS_BRIDGE": WORKER_SCRIPT_ARGS_BRIDGE,
        "add_dsv4_arguments": add_dsv4_arguments,
        "add_dsa_arguments": add_dsa_arguments,
    }
    exec(compile(ast.Module(body=[registrar], type_ignores=[]), str(path), "exec"), namespace)
    parser = ArgumentParser()
    parser.add_argument("--cp-comm-type", nargs="+", default=["p2p"])
    namespace["_register_worker_arguments"](parser)
    args = parser.parse_args(
        [
            "--script-hf-checkpoint",
            "/hf",
            "--script-token-ids-file",
            "/tokens.json",
            "--miles-dsa-topk-backend",
            "flashinfer",
        ]
        + argv
    )
    assert args.dsa_impl == implementation
    assert args.miles_dsa_topk_backend == "flashinfer"
    assert args.cp_comm_type is None
    assert WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(args).hf_checkpoint == Path("/hf")
