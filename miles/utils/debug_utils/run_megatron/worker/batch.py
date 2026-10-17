# [MODULE DOC] Builds the single input batch for the standalone Megatron worker from a list of
# token ids (replicated across the batch dimension, optionally sliced for context parallelism,
# with next-token labels) and provides the cross-entropy loss used by the pipeline schedule.
"""Input batch preparation and loss function for standalone Megatron forward/backward."""

from types import SimpleNamespace
from typing import Any

import torch


def prepare_batch(
    *,
    token_ids: list[int],
    batch_size: int,
    cp_rank: int = 0,
    cp_size: int = 1,
    device: str | torch.device = "cuda",
) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] Keyword-only inputs: token id list, batch size, this rank's context-parallel
    # rank and size (defaults 0 and 1), device (default cuda). Creates long tensors for the token
    # ids and for positions 0..length-1 on the device and keeps an unsliced copy of the tokens.
    # When the context-parallel size exceeds 1, slices tokens and positions for this rank with the
    # shared context-parallel slicer (imported lazily; padding value 0, "bshd" layout, maximum
    # length equal to the sequence length, using a minimal stand-in for parallel state). Adds a
    # leading batch dimension and expands it to the batch size as views (rows share memory, no
    # copies). Labels come from the label builder. Returns a mapping with input ids, position ids,
    # labels, the unsliced global input ids and an attention mask entry that is always none (the
    # model builds its own causal mask).
    # KNOWN ISSUE: the docstring describes an explicit causal attention mask of shape batch x 1 x
    # length x length, but the code returns none for it.
    """Build the batch dict for Megatron forward from pre-tokenized token IDs.

    Returns a dict containing:
    - input_ids: [batch_size, local_seq_len]
    - position_ids: [batch_size, local_seq_len]
    - attention_mask: [batch_size, 1, local_seq_len, local_seq_len] causal mask
    - labels: [batch_size, local_seq_len] (CP-aware next-token labels)
    - global_input_ids: [batch_size, full_seq_len] (unsliced, for reference)
    """
    seq_length: int = len(token_ids)

    token_tensor: torch.Tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    position_tensor: torch.Tensor = torch.arange(seq_length, dtype=torch.long, device=device)

    # Keep global copy before CP slicing (needed for CP-aware labels)
    global_tokens: torch.Tensor = token_tensor.clone()

    if cp_size > 1:
        from miles.backends.training_utils.data.context_parallel import slice_with_cp

        cp_kwargs: dict[str, object] = dict(
            pad_value=0,
            parallel_state=SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size)),
            qkv_format="bshd",
            max_seq_len=seq_length,
        )
        token_tensor = slice_with_cp(token_tensor, **cp_kwargs)
        position_tensor = slice_with_cp(position_tensor, **cp_kwargs)

    input_ids: torch.Tensor = token_tensor.unsqueeze(0).expand(batch_size, -1)
    position_ids: torch.Tensor = position_tensor.unsqueeze(0).expand(batch_size, -1)
    global_input_ids: torch.Tensor = global_tokens.unsqueeze(0).expand(batch_size, -1)

    labels: torch.Tensor = _build_labels(
        input_ids=input_ids,
        position_ids=position_ids,
        global_input_ids=global_input_ids,
        cp_size=cp_size,
    )

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": None,
        "labels": labels,
        "global_input_ids": global_input_ids,
    }


def _build_labels(
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    global_input_ids: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Context parallel size above 1: a position is valid when it is less
    # than the global length minus one; its label is the global token at position plus one (valid
    # positions) gathered from the global ids, invalid positions gather index 0 and are then
    # overwritten with -100. Otherwise (no context parallelism): labels are the input ids shifted
    # left by one with a final column of -100. The ignore value is -100 in both cases. The input
    # ids argument is only used in the non-parallel branch.
    """Build next-token prediction labels, handling CP zigzag slicing.

    With CP>1, each rank sees non-contiguous positions (zigzag pattern).
    We use position_ids to gather the correct next-token from global_input_ids.
    Positions at the end of the global sequence get label -100 (ignored by loss).
    """
    if cp_size > 1:
        global_seq_len: int = global_input_ids.shape[1]
        valid_mask: torch.Tensor = position_ids < global_seq_len - 1
        next_pos: torch.Tensor = torch.where(valid_mask, position_ids + 1, torch.zeros_like(position_ids))
        labels: torch.Tensor = global_input_ids.gather(1, next_pos)
        labels = labels.masked_fill(~valid_mask, -100)
        return labels
    else:
        batch_size: int = input_ids.shape[0]
        return torch.cat(
            [input_ids[:, 1:], torch.full((batch_size, 1), -100, device=input_ids.device, dtype=input_ids.dtype)],
            dim=1,
        )


def loss_func(
    labels: torch.Tensor,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    # [FUNCTION DOC] Loss callback for the schedule (called with labels bound first, then the model
    # output). Converts the output to float32, flattens logits to rows of vocabulary size and
    # labels to a vector, and computes mean cross entropy ignoring label -100. Returns (loss, a
    # mapping with the detached loss under the loss key). If all labels are ignored the mean is
    # not a number.
    """Cross-entropy loss for forward-backward pipeline schedule.

    Uses ignore_index=-100 to handle CP-aware label masking.
    """
    logits: torch.Tensor = output_tensor.float()
    vocab_size: int = logits.size(-1)

    loss: torch.Tensor = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss, {"loss": loss.detach()}
