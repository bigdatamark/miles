# [MODULE DOC]
# Small slot-related helpers for multi-adapter LoRA: a factory building the Megatron-Bridge
# multi-adapter LoRA transform from the launch args, and a function trimming rank-padded adapter
# tensors (all slots are allocated at the maximum rank) down to a slot's actual rank for export.
"""Slot mechanics for multi-LoRA: model instrumentation, shard addressing, and
per-slot optimizer state hygiene. Kept deliberately free of any control-plane
or data-plane logic."""

import logging
from argparse import Namespace

import torch

logger = logging.getLogger(__name__)


def create_multi_lora_instance(args: Namespace, *, target_modules):
    # [FUNCTION DOC] Returns a Bridge multi-adapter LoRA transform for the target modules, with
    # the number of adapters from args.multi_lora_n_adapters, rank from lora_rank, alpha from
    # lora_alpha, dropout (default 0.0), and A/B initialization methods (defaults xavier and zero)
    # read leniently from args. The Bridge import is local.
    from megatron.bridge.peft.multi_lora import MultiLoRA

    return MultiLoRA(
        target_modules=target_modules,
        n_adapters=args.multi_lora_n_adapters,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=getattr(args, "lora_dropout", 0.0),
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )


def slice_lora_to_rank(hf_name: str, tensor: torch.Tensor, adapter_rank: int) -> torch.Tensor:
    # [FUNCTION DOC]
    # Chooses the rank axis by the tensor name: names containing "lora_A" have the rank on the
    # second-to-last axis, names containing "lora_B" on the last axis (axes counted from the end so
    # extra leading expert axes are untouched); other names are returned unchanged. If the adapter
    # rank is below the tensor's size on that axis, asserts the discarded padding region is
    # entirely zero (assertion error with name, max value, shape and rank otherwise) and returns the
    # narrowed view (not a copy) of the first adapter-rank entries; if not smaller, returns the
    # tensor unchanged. The "lora_A" check takes precedence if both substrings occur.
    """Trim a max-rank-padded LoRA tensor to ``adapter_rank`` on the rank axis, addressed
    from the end so packed grouped-expert exports are not sliced on the expert axis."""
    if "lora_A" in hf_name:
        rank_dim = tensor.ndim - 2
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_A padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    if "lora_B" in hf_name:
        rank_dim = tensor.ndim - 1
        if adapter_rank < tensor.shape[rank_dim]:
            remainder = tensor.narrow(rank_dim, adapter_rank, tensor.shape[rank_dim] - adapter_rank)
            assert remainder.abs().max() == 0, (
                f"lora_B padded dims are non-zero: {hf_name}, "
                f"max={remainder.abs().max().item():.6e}, shape={tensor.shape}, rank={adapter_rank}"
            )
            return tensor.narrow(rank_dim, 0, adapter_rank)
        return tensor
    return tensor
