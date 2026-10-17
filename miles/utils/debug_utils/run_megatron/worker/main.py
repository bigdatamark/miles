# [MODULE DOC] Worker entry point launched by torchrun on every rank. Flow: parse Megatron plus
# script arguments, initialize distributed Megatron, optionally apply source patches, set up
# routing replay before building the model, build and load the model, register the dumper on every
# model chunk, load replay data, read the token ids file and build the batch, run one forward
# (and optionally backward) step capturing the output logits, write logprob output and print
# top-k predictions on the last pipeline stage if requested, save replay data, finalize the
# dumper, then barrier and tear down the process group. Rank 0 prints progress lines.
# KNOWN ISSUE: the docstring below says the script is launched by "cli.py run" and consumes
# MODEL_ARGS from a shell script; the launcher is now the cli package's run command.
"""Torchrun worker script for standalone Megatron forward/backward.

This script is launched by ``cli.py run`` via ``torchrun`` and runs inside
each GPU process.  It uses Megatron's argparse (to consume ``MODEL_ARGS``
from the shell script) plus a handful of custom arguments.

Not intended to be run directly — use ``python -m miles.utils.debug_utils.run_megatron run`` instead.
"""

import argparse
import json
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.source_patcher import apply_patches_from_config

from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.debug_utils.run_megatron.worker.batch import loss_func, prepare_batch
from miles.utils.debug_utils.run_megatron.worker.output import compute_and_save_output_info
from miles.utils.debug_utils.run_megatron.worker.replay import (
    load_replay_data,
    save_replay_data,
    setup_replay_before_model,
)
from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE, WorkerScriptArgs
from miles.utils.debug_utils.run_megatron.worker.top_k_print import print_top_k
from miles_plugins.models.deepseek_v4.arguments import add_dsv4_arguments
from miles_plugins.models.glm5.arguments import add_dsa_arguments


def main() -> None:
    # [FUNCTION DOC] Orchestrates the worker as described in the module doc. Details: the replay
    # data loader is told whether sequence parallelism is on (defaulting to off when the argument
    # is missing); the batch uses the micro batch size and this rank's context-parallel rank and
    # size; logprob files are written only on the last pipeline stage when a logprob output
    # directory is set and logits were captured; top-k printing happens under the same stage and
    # capture conditions when top-k is positive. Ends with a distributed barrier and destruction
    # of the process group.
    args, script = _parse_args()
    _initialize_megatron(args)

    rank: int = dist.get_rank()
    if rank == 0:
        _print_config(args, script)

    if script.source_patcher_config:
        _apply_source_patches(script.source_patcher_config)

    setup_replay_before_model(script)
    model: list[Any] = _build_and_load_model(args, script)

    for m in model:
        dumper.register_non_intrusive_dumper(m)

    load_replay_data(script, rank=rank, sequence_parallel=getattr(args, "sequence_parallel", False))

    token_ids: list[int] = json.loads(script.token_ids_file.read_text())
    batch: dict[str, torch.Tensor] = prepare_batch(
        token_ids=token_ids,
        batch_size=args.micro_batch_size,
        cp_rank=get_parallel_state().cp.rank,
        cp_size=get_parallel_state().cp.size,
    )

    if rank == 0:
        print(f"[worker] input_ids shape={batch['input_ids'].shape}", flush=True)

    captured_logits: torch.Tensor | None = _run_forward_backward(
        args=args,
        script=script,
        model=model,
        batch=batch,
    )

    is_last_pp_stage: bool = mpu.is_pipeline_last_stage()

    if script.logprob_output is not None and captured_logits is not None and is_last_pp_stage:
        compute_and_save_output_info(
            logits=captured_logits,
            labels=batch["labels"],
            position_ids=batch["position_ids"],
            output_dir=script.logprob_output,
        )

    if script.top_k > 0 and captured_logits is not None and is_last_pp_stage:
        print_top_k(
            logits=captured_logits,
            input_ids=batch["input_ids"],
            top_k=script.top_k,
            tokenizer_path=script.hf_checkpoint,
        )

    save_replay_data(script, rank=rank)
    _finalize_dumper()

    if rank == 0:
        print("[worker] Done.", flush=True)

    dist.barrier()
    dist.destroy_process_group()


def _register_worker_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # [FUNCTION DOC] HELPER: Extra-arguments provider for Megatron's parser. Registers the script
    # option group via the bridge and the DeepSeek V4 and DSA plugin arguments (so model-argument
    # strings that mention them still parse). Returns the same parser.
    """Worker arguments plus the plugin arguments the model scripts pass through."""
    WORKER_SCRIPT_ARGS_BRIDGE.register_on_parser(parser)
    add_dsv4_arguments(parser)
    add_dsa_arguments(parser)
    return parser


def _parse_args() -> tuple[argparse.Namespace, WorkerScriptArgs]:
    # [FUNCTION DOC] HELPER: Parses the process command line with Megatron's argument parser plus
    # the worker arguments, extracts the script options dataclass from the namespace, and, if a
    # reference checkpoint path is set, stores it (as text) as the namespace's load path. Returns
    # (namespace, script options).
    args: argparse.Namespace = parse_args(extra_args_provider=_register_worker_arguments)
    script_args: WorkerScriptArgs = WORKER_SCRIPT_ARGS_BRIDGE.from_namespace(args)

    if script_args.ref_load is not None:
        args.load = str(script_args.ref_load)

    return args, script_args


def _initialize_megatron(args: argparse.Namespace) -> None:
    # [FUNCTION DOC] HELPER: Side effects: initializes the NCCL process group; selects the CUDA
    # device from the LOCAL_RANK environment variable (default 0); sets the namespace's HF
    # checkpoint path (as text) from the script option; fills defaults for megatron-to-HF mode
    # ("raw"), batch-size decrease (false) and deterministic collectives (false) only when absent;
    # applies miles' default Megatron arguments, runs Megatron's argument validation, and runs
    # miles' initializer.
    torch.distributed.init_process_group(backend="nccl")
    local_rank: int = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    args.hf_checkpoint = str(args.script_hf_checkpoint)
    args.__dict__.setdefault("megatron_to_hf_mode", "raw")
    args.__dict__.setdefault("decrease_batch_size_if_needed", False)
    args.__dict__.setdefault("debug_deterministic_collective", False)
    set_default_megatron_args(args)
    validate_args(args)

    init(args)


def _build_and_load_model(args: argparse.Namespace, script: WorkerScriptArgs) -> list[Any]:
    # [FUNCTION DOC] HELPER: Builds the model chunks for the script's role with Megatron's model
    # builder, wrapping in distributed data parallel only when running backward (to avoid
    # allocating the large gradient buffer). If the namespace has a load path, loads the checkpoint
    # into the model only (no optimizer or scheduler). Puts every chunk in training mode (dropout
    # was set to zero on the command line). Returns the list of chunks.
    model_provider: Callable[..., Any] = get_model_provider_func(args, role=script.role)
    # Forward-only runs skip DDP wrapping so the distributed-optimizer grad buffer
    # (tens-to-hundreds of GB for large MoE models) is never allocated -> avoids OOM.
    model: list[Any] = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=script.run_backward)

    if args.load is not None:
        load_checkpoint(
            model,
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
        )

    for m in model:
        m.train()
    return model


def _apply_source_patches(config_path: Path) -> None:
    # [FUNCTION DOC] HELPER: Reads the YAML file and applies the source patches it describes via
    # sglang's patcher, making the dumper object available to patched code through an extra import
    # line. Modifies loaded code in the process; prints a confirmation.
    yaml_content: str = config_path.read_text()
    apply_patches_from_config(
        yaml_content,
        extra_imports=["from sglang.srt.debug_utils.dumper import dumper"],
    )
    print(f"[worker] Applied source patches from {config_path}", flush=True)


def _run_forward_backward(
    args: argparse.Namespace,
    script: WorkerScriptArgs,
    model: list[Any],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor | None:
    # [FUNCTION DOC] HELPER: Runs one micro batch through Megatron's pipeline schedule (a single
    # microbatch fed from a one-element iterator, forward only unless backward was requested) with
    # a step function that fetches the batch, runs the model chunk with gathered output across
    # tensor parallel ranks, stores a detached copy of the output in a local list and returns the
    # output with the loss callback bound to the labels. Rank 0 prints the returned losses if any.
    # Returns the first captured output, or none if the step function never ran (for example on
    # pipeline stages that do not produce logits it still captures whatever the chunk returns).
    """Run forward (and optionally backward) pass, returning captured logits."""
    forward_backward_func: Callable[..., Any] = get_forward_backward_func()
    captured: list[torch.Tensor] = []

    def forward_step_func(
        data_iterator: Any,
        model_chunk: Any,
    ) -> tuple[torch.Tensor, partial[tuple[torch.Tensor, dict[str, Any]]]]:
        # [FUNCTION DOC] HELPER: Schedule step callback. Takes the next batch from the data
        # iterator, calls the model chunk with input ids, position ids, the optional attention mask
        # and gathered output, appends the detached output to the enclosing capture list, and
        # returns the output together with the loss function partially applied to the labels.
        data: dict[str, torch.Tensor] = next(data_iterator)
        output: torch.Tensor = model_chunk(
            input_ids=data["input_ids"],
            position_ids=data["position_ids"],
            attention_mask=data.get("attention_mask"),
            runtime_gather_output=True,
        )
        captured.append(output.detach())
        return output, partial(loss_func, data["labels"])

    losses: Any = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=iter([batch]),
        model=model,
        num_microbatches=1,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        forward_only=not script.run_backward,
    )

    rank: int = dist.get_rank()
    if rank == 0 and losses:
        print(f"[worker rank={rank}] losses={losses}", flush=True)

    return captured[0] if captured else None


def _print_config(args: argparse.Namespace, script: WorkerScriptArgs) -> None:
    # [FUNCTION DOC] HELPER: Prints three flushed lines: sequence length and micro batch size; the
    # tensor, pipeline, context, expert and expert-tensor parallel sizes; backward flag and role.
    print(f"[worker] seq_length={args.seq_length}, micro_batch_size={args.micro_batch_size}", flush=True)
    print(
        f"[worker] tp={args.tensor_model_parallel_size}, pp={args.pipeline_model_parallel_size}, "
        f"cp={args.context_parallel_size}, ep={args.expert_model_parallel_size}, "
        f"etp={args.expert_tensor_parallel_size}",
        flush=True,
    )
    print(f"[worker] run_backward={script.run_backward}, role={script.role}", flush=True)


def _finalize_dumper() -> None:
    # [FUNCTION DOC] HELPER: Only when the DUMPER_ENABLE environment variable equals "1": advances
    # the dumper by one step and then turns the dumper off. Otherwise does nothing.
    """Step + disable dumper after forward/backward."""
    if os.environ.get("DUMPER_ENABLE", "0") == "1":
        dumper.step()
        dumper.configure(enable=False)


if __name__ == "__main__":
    main()
