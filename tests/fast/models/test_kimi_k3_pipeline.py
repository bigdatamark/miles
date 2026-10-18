# [MODULE DOC]
# Cpu tests of the Kimi K3 pipeline-parallel stage boundary helpers: packing a prefix sum and a bank
# of block residuals into one payload tensor and unpacking it (exactly, including gradients), and the
# row count of the bank as a function of layer index and block size.
import pytest
import torch

from miles_plugins.models.kimi_k3.pipeline import bank_num_rows, pack_stage_boundary, unpack_stage_boundary


def test_pack_unpack_is_exact_in_both_directions() -> None:
    # [FUNCTION DOC] Seeds the generator, packs a bfloat16 prefix sum (5x2x16) with a block residual bank
    # (5x2x3x16) and asserts the payload shape is 5x2x64 (hidden width times 1 plus 3 rows). Unpacking
    # with width 16 and 3 rows reproduces both inputs exactly. Gradient check: with differentiable
    # float32 inputs (4x1x8 and 4x1x2x8), a pack then unpack, then backward with random output
    # gradients, yields input gradients exactly equal to those output gradients. Finally, unpacking a
    # payload whose width is 3 times 16 instead of 4 times 16 raises an assertion error mentioning
    # stage-boundary payload width.
    """A transposed unflatten mixes bank rows into prefix_sum with no shape error."""
    torch.manual_seed(0)
    prefix_sum = torch.randn(5, 2, 16, dtype=torch.bfloat16)
    block_residual = torch.randn(5, 2, 3, 16, dtype=torch.bfloat16)

    packed = pack_stage_boundary(prefix_sum, block_residual)
    assert packed.shape == (5, 2, 4 * 16)

    prefix_out, bank_out = unpack_stage_boundary(packed, 16, 3)
    torch.testing.assert_close(prefix_out, prefix_sum, rtol=0, atol=0)
    torch.testing.assert_close(bank_out, block_residual, rtol=0, atol=0)

    grad_prefix_sum = torch.randn(4, 1, 8, requires_grad=True)
    grad_block_residual = torch.randn(4, 1, 2, 8, requires_grad=True)
    prefix_out, bank_out = unpack_stage_boundary(pack_stage_boundary(grad_prefix_sum, grad_block_residual), 8, 2)
    grad_prefix = torch.randn_like(prefix_out)
    grad_bank = torch.randn_like(bank_out)
    torch.autograd.backward([prefix_out, bank_out], [grad_prefix, grad_bank])

    torch.testing.assert_close(grad_prefix_sum.grad, grad_prefix, rtol=0, atol=0)
    torch.testing.assert_close(grad_block_residual.grad, grad_bank, rtol=0, atol=0)

    with pytest.raises(AssertionError, match="stage-boundary payload width"):
        unpack_stage_boundary(torch.zeros(2, 1, 3 * 16), 16, 3)


def test_bank_num_rows_matches_write_schedule() -> None:
    # [FUNCTION DOC] With block size 12: for each layer index from 1 to 92, the row count must equal the
    # number of earlier layers (indices below it) that are multiples of the block size, i.e. the rows
    # written before that layer. Also for each last layer of a stage from 0 to 91, the row count at the
    # next layer index must equal that layer index integer-divided by the block size, plus one.
    """One row off between the write schedule and the receiver reinterprets the payload."""
    block_size = 12
    for layer_idx in range(1, 93):
        rows_written_before = sum(1 for w in range(layer_idx) if w % block_size == 0)
        assert bank_num_rows(layer_idx, block_size) == rows_written_before

    for last_layer_of_stage in range(92):
        rows_after_exit = last_layer_of_stage // block_size + 1
        assert bank_num_rows(last_layer_of_stage + 1, block_size) == rows_after_exit
