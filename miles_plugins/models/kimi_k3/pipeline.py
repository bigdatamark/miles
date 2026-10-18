# [MODULE DOC] HELPER: Pure tensor helpers for transporting the attention-residual state across
# pipeline stages: row-count rule for the snapshot bank, packing of (prefix sum, bank) into one tensor
# along the hidden dim, and the inverse with a width check.
"""Stage-boundary packing for pipeline-parallel Kimi K3.

Megatron's p2p carries one hidden-states tensor, so the stage-exit layer packs
`[prefix_sum, snapshot bank]` along the hidden dim and the stage-entry layer unpacks it.
"""

import torch


def bank_num_rows(layer_idx: int, block_size: int) -> int:
    # [FUNCTION DOC] HELPER: Asserts the zero-based layer index is positive; returns the ceiling of
    # layer index over block size, i.e. the number of block-write layers (indices that are multiples of
    # the block size, starting at zero) strictly before that layer.
    """Snapshot rows present before global layer ``layer_idx`` executes."""
    assert layer_idx > 0
    return (layer_idx + block_size - 1) // block_size


def pack_stage_boundary(prefix_sum: torch.Tensor, block_residual: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Asserts the bank has at least one row. Flattens the bank's last two dims
    # (rows, hidden) into one and concatenates it after the prefix sum along the last dim, giving width
    # (1 + rows) times hidden.
    assert block_residual.shape[-2] > 0, "stage boundary before the first snapshot write"
    return torch.cat((prefix_sum, block_residual.flatten(-2)), dim=-1)


def unpack_stage_boundary(packed: torch.Tensor, hidden_size: int, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Asserts the last dim equals (1 + rows) times hidden size (message shows
    # the mismatch). Returns the first hidden-size channels as the prefix sum and the rest reshaped to
    # (rows, hidden) as the bank, both made contiguous.
    expected = (1 + num_rows) * hidden_size
    assert (
        packed.shape[-1] == expected
    ), f"stage-boundary payload width {packed.shape[-1]} != (1 + {num_rows}) * {hidden_size}"
    prefix_sum = packed[..., :hidden_size].contiguous()
    block_residual = packed[..., hidden_size:].unflatten(-1, (num_rows, hidden_size)).contiguous()
    return prefix_sum, block_residual
