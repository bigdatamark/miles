# [MODULE DOC] Prints, for each input position, the top-k most probable next tokens with their
# probabilities, one rank at a time (ranks take turns using barriers so output is not
# interleaved). Used for eyeballing forward-pass outputs.
"""Top-K prediction printing for debugging forward pass outputs."""

import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


def print_top_k(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
    tokenizer_path: Path,
) -> None:
    # [FUNCTION DOC] Keyword-only inputs: logits (batch x positions x vocabulary), input ids, k and
    # tokenizer location. Loads the tokenizer with remote code trusted (lazy import). The padding
    # id is the tokenizer's pad id, or the end-of-sequence id when the pad id is missing or zero,
    # then prints the predictions for all ranks. Must be called by every rank, since it uses
    # barriers.
    # KNOWN ISSUE: a pad id of zero is treated as missing because of truthiness, so the
    # end-of-sequence id is used instead.
    """Load tokenizer and print top-k predictions across all ranks."""
    from miles.utils.processing_utils import load_tokenizer

    tokenizer: Any = load_tokenizer(str(tokenizer_path), trust_remote_code=True)
    pad_token_id: int | None = tokenizer.pad_token_id or tokenizer.eos_token_id

    _print_top_predictions_all_ranks(
        logits=logits,
        input_ids=input_ids,
        top_k=top_k,
        tokenizer=tokenizer,
        pad_token_id=pad_token_id,
    )


def _print_top_predictions_all_ranks(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
    tokenizer: object,
    pad_token_id: int | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Gets rank and world size (0 and 1 if not distributed). Rank 0 prints
    # a banner with k and world size. Then for each rank in order: everyone synchronizes at a
    # barrier (when distributed); the matching rank prints its predictions and flushes. A final
    # barrier follows. Output of different ranks is therefore ordered by rank.
    """Print top-k predictions from all ranks sequentially (rank 0 first, then rank 1, etc.)."""
    rank, world_size = _get_dist_info()

    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Top-{top_k} Predictions (all ranks)")
        print(f"World size: {world_size}")
        print(f"{'=' * 80}")

    for r in range(world_size):
        _maybe_barrier()
        if rank == r:
            _print_top_predictions_for_rank(
                logits=logits,
                input_ids=input_ids,
                top_k=top_k,
                tokenizer=tokenizer,
                rank=rank,
                pad_token_id=pad_token_id,
            )
            sys.stdout.flush()

    _maybe_barrier()


def _print_top_predictions_for_rank(
    *,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    top_k: int,
    tokenizer: object,
    rank: int,
    pad_token_id: int | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Prints a header with the rank and local sequence length. For each
    # batch row (with a "Batch n" line when the batch has more than one row) and each position
    # (skipping positions whose input token equals the padding id when one is known): softmax over
    # the vocabulary, top-k probabilities and indices, then one line with the zero-padded position,
    # the input token's decoded text in quotes, an arrow, and comma-separated decoded predictions
    # each followed by its probability with three decimals.
    """Print top-k predictions for this rank, one line per position."""
    batch_size: int = logits.shape[0]
    seq_length: int = logits.shape[1]

    print(f"\n--- Rank {rank} (seq_len={seq_length}) ---")
    for b in range(batch_size):
        if batch_size > 1:
            print(f"  Batch {b}:")
        for pos in range(seq_length):
            if pad_token_id is not None and input_ids[b, pos].item() == pad_token_id:
                continue

            input_token: int = input_ids[b, pos].item()
            probs: torch.Tensor = torch.softmax(logits[b, pos], dim=-1)
            top_probs: torch.Tensor
            top_indices: torch.Tensor
            top_probs, top_indices = torch.topk(probs, top_k)

            input_str: str = _decode_token(tokenizer, token_id=input_token)
            preds: str = ", ".join(
                f"{_decode_token(tokenizer, token_id=idx.item())}({prob.item():.3f})"
                for prob, idx in zip(top_probs, top_indices, strict=True)
            )
            print(f"pos[{pos:3d}] {input_str!r:12s} -> {preds}")


def _decode_token(tokenizer: object, *, token_id: int) -> str:
    # [FUNCTION DOC] HELPER: Decodes a single token id with the tokenizer, or, if no tokenizer is
    # given, returns the letter t followed by the id.
    return tokenizer.decode([token_id]) if tokenizer else f"t{token_id}"


def _get_dist_info() -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Returns (rank, world size) from torch distributed, or (0, 1) when it
    # is not initialized.
    """Return (rank, world_size), defaulting to (0, 1) if not initialized."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _maybe_barrier() -> None:
    # [FUNCTION DOC] HELPER: Blocks until all ranks arrive, only when torch distributed is
    # initialized; otherwise does nothing.
    if dist.is_initialized():
        dist.barrier()
