# [MODULE DOC] Command-line arguments for the torchtitan backend: an arguments dataclass extending
# the FSDP arguments with torchtitan model selection and parallel sizes, parser builders, and a
# validation function that rejects unsupported combinations early.
from dataclasses import dataclass

import torch

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser


@dataclass
class TorchtitanArgs(FSDPArgs):
    # [CLASS DOC] Dataclass of torchtitan options added to the FSDP options. Fields and defaults:
    # model name "qwen3" and model flavor "0.6B" (the name selects a torchtitan model package and
    # the flavor is passed to its model registry); sequence length 4096 (tokens, must cover prompt
    # plus response, see validation); tensor, pipeline and expert model parallel sizes, all 1.
    # torchtitan.models.<name>.model_registry(<flavor>)
    titan_model_name: str = "qwen3"
    titan_model_flavor: str = "0.6B"

    seq_length: int = 4096

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1


def build_torchtitan_parser(extra_args_provider=None):
    # [FUNCTION DOC] Builds an argument parser from the torchtitan arguments dataclass with the
    # description "torchtitan Training (miles)", letting an optional provider callback add more
    # arguments. Returns the parser.
    return build_dataclass_parser(TorchtitanArgs, "torchtitan Training (miles)", extra_args_provider)


def load_torchtitan_args(extra_args_provider=None):
    # [FUNCTION DOC] Builds the torchtitan parser (with the optional extra provider) and returns the
    # arguments loaded from it by the shared FSDP loader (which parses the process command line).
    return load_args_from_parser(build_torchtitan_parser(extra_args_provider))


def validate_torchtitan_args(args) -> None:
    # [FUNCTION DOC] Raises a value error for the first unsupported setting; returns nothing.
    # Checks in order: installed torch (major and minor numbers of its version) is at least 2.13;
    # context parallelism is not used with the qwen3_5 model; sequence length: if a rollout maximum
    # context length is set the sequence length must be at least that, otherwise it must be strictly
    # greater than the rollout maximum response length (rotary embeddings are built for the
    # sequence length); no reference update interval; no fp16; the learning-rate decay style is
    # constant, linear or cosine; no warmup fraction, warmup-stable-decay iterations or decay
    # iterations (only warmup iterations are supported); no saving of debug train data.
    torch_version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if torch_version < (2, 13):
        raise ValueError(f"the torchtitan backend needs torch>=2.13; this environment runs {torch.__version__}")
    if args.context_parallel_size != 1 and args.titan_model_name == "qwen3_5":
        raise ValueError("torchtitan does not support context parallelism for qwen3_5")

    if args.rollout_max_context_len is not None:
        if args.seq_length < args.rollout_max_context_len:
            raise ValueError(
                f"--seq-length {args.seq_length} is shorter than "
                f"--rollout-max-context-len {args.rollout_max_context_len}: torchtitan builds its "
                "rotary embeddings for the former, so a longer sequence would index past them"
            )
    elif args.seq_length <= args.rollout_max_response_len:
        raise ValueError(
            f"--seq-length {args.seq_length} leaves no room for a prompt ahead of "
            f"--rollout-max-response-len {args.rollout_max_response_len}: torchtitan builds its "
            "rotary embeddings for the former, and a prompt-plus-response beyond them asserts "
            "inside the rope kernel"
        )

    if args.ref_update_interval is not None:
        raise ValueError("--ref-update-interval is not supported by the torchtitan backend")
    if args.fp16:
        raise ValueError("the torchtitan backend trains in bf16 mixed precision; --fp16 is not supported")
    if args.lr_decay_style not in ("constant", "linear", "cosine"):
        raise ValueError(
            f"torchtitan's LR schedule has no {args.lr_decay_style!r} decay; use constant, linear or cosine"
        )
    if args.lr_warmup_fraction is not None or args.lr_wsd_decay_iters is not None or args.lr_decay_iters is not None:
        raise ValueError(
            "torchtitan's LR schedule takes --lr-warmup-iters only; fraction, WSD and decay-iters are unsupported"
        )
    if args.save_debug_train_data is not None:
        raise ValueError("--save-debug-train-data is not wired up for the torchtitan backend")
