# [MODULE DOC] Unit tests for the worker script argument dataclass of the run_megatron debug
# utility and its bridge object that converts between the dataclass, command line strings and a
# parsed argument namespace. Tests use only paths that are never touched on disk.
import dataclasses
from pathlib import Path

import pytest

from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE, WorkerScriptArgs


class TestWorkerScriptArgs:
    # [CLASS DOC] Tests of the immutable worker argument record (required: Hugging Face
    # checkpoint path and token ids file path) and its command line round trip.
    def test_frozen(self) -> None:
        # [FUNCTION DOC] Assigning to a field of an instance raises the dataclass frozen
        # instance error.
        args = WorkerScriptArgs(hf_checkpoint=Path("/hf"), token_ids_file=Path("/tokens.json"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            args.role = "critic"  # type: ignore[misc]

    def test_defaults(self) -> None:
        # [FUNCTION DOC] With only the two required fields, role defaults to "actor", top k to 0,
        # run backward to false, and the reference load path, source patcher config and both
        # routing replay paths to none.
        args = WorkerScriptArgs(hf_checkpoint=Path("/hf"), token_ids_file=Path("/tokens.json"))
        assert args.role == "actor"
        assert args.top_k == 0
        assert args.run_backward is False
        assert args.ref_load is None
        assert args.source_patcher_config is None
        assert args.routing_replay_dump_path is None
        assert args.routing_replay_load_path is None

    def test_to_cli_args_roundtrip(self) -> None:
        # [FUNCTION DOC] Serializes a record (critic role, backward on, top k 5) with the bridge
        # to a command line string, registers the bridge's options on a fresh argument parser,
        # parses the string split on whitespace, rebuilds a record from the namespace, and
        # asserts it equals the original.
        original = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            role="critic",
            run_backward=True,
            top_k=5,
        )
        cli_str = WORKER_SCRIPT_ARGS_BRIDGE.to_cli_args(original)

        import argparse

        parser = argparse.ArgumentParser()
        WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser(parser)
        namespace = parser.parse_args(cli_str.split())
        reconstructed = WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(namespace)

        assert reconstructed == original

    def test_none_fields_omitted(self) -> None:
        # [FUNCTION DOC] Fields that are none (reference load, source patcher config, both
        # routing replay paths) do not appear in the command line string; the options use a
        # "script" prefix in their names.
        args = WorkerScriptArgs(hf_checkpoint=Path("/hf"), token_ids_file=Path("/tokens.json"))
        cli_str = WORKER_SCRIPT_ARGS_BRIDGE.to_cli_args(args)
        assert "--script-ref-load" not in cli_str
        assert "--script-source-patcher-config" not in cli_str
        assert "--script-routing-replay-dump-path" not in cli_str
        assert "--script-routing-replay-load-path" not in cli_str

    def test_bool_serialization(self) -> None:
        # [FUNCTION DOC] A true boolean field is serialized as a bare flag
        # (the run-backward option) appearing in the command line string.
        args = WorkerScriptArgs(
            hf_checkpoint=Path("/hf"),
            token_ids_file=Path("/tokens.json"),
            run_backward=True,
        )
        cli_str = WORKER_SCRIPT_ARGS_BRIDGE.to_cli_args(args)
        assert "--script-run-backward" in cli_str
