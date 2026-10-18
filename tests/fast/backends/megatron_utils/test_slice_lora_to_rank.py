# [MODULE DOC]
# Tests for the LoRA rank-slicing helper of the Megatron backend: LoRA A and B tensors padded to a
# maximum rank are trimmed on the rank axis to the adapter's real rank, for plain two-dimensional
# tensors and for packed grouped-expert three-dimensional tensors; padding that holds nonzero
# values is refused; full-rank tensors and non-LoRA names pass through as the same object.
"""slice_lora_to_rank trims max-rank-padded LoRA tensors to the adapter's real rank;
used by weight-sync and HF PEFT export (PEFT rejects tensors padded past the declared rank)."""

import pytest
import torch

from miles.backends.megatron_utils.lora.slots import slice_lora_to_rank


def _padded(shape, live_rows=None, live_cols=None):
    # [FUNCTION DOC] HELPER: Returns a zero tensor of the given shape, with the first live_rows
    # rows set to ones when given and the first live_cols columns set to ones when given.
    t = torch.zeros(shape)
    if live_rows is not None:
        t[:live_rows] = 1.0
    if live_cols is not None:
        t[:, :live_cols] = 1.0
    return t


def test_lora_a_is_sliced_on_the_rank_dim():
    # [FUNCTION DOC] A 32-by-8 LoRA A tensor with the first 16 rows live is sliced to rank 16:
    # asserts shape (16, 8) and equality with the first 16 rows (rank is the first dimension).
    tensor = _padded((32, 8), live_rows=16)
    out = slice_lora_to_rank("base_model.q_proj.lora_A.weight", tensor, 16)
    assert out.shape == (16, 8)
    assert torch.equal(out, tensor[:16])


def test_lora_b_is_sliced_on_the_rank_dim():
    # [FUNCTION DOC] An 8-by-32 LoRA B tensor with the first 16 columns live is sliced to rank 16:
    # asserts shape (8, 16) and equality with the first 16 columns (rank is the last dimension).
    tensor = _padded((8, 32), live_cols=16)
    out = slice_lora_to_rank("base_model.q_proj.lora_B.weight", tensor, 16)
    assert out.shape == (8, 16)
    assert torch.equal(out, tensor[:, :16])


def test_nonzero_padding_is_rejected():
    # [FUNCTION DOC] An all-ones 32-by-8 LoRA A tensor sliced to rank 16 raises an assertion error
    # saying the padded dims are non-zero.
    # Live values beyond the adapter's rank mean the pad rows were trained —
    # slicing would silently drop signal, so it must hard-fail instead.
    tensor = torch.ones(32, 8)
    with pytest.raises(AssertionError, match="padded dims are non-zero"):
        slice_lora_to_rank("x.lora_A.weight", tensor, 16)


def test_full_rank_tensor_passes_through():
    # [FUNCTION DOC] A 16-by-8 tensor already at rank 16 is returned as the very same object.
    tensor = torch.ones(16, 8)
    assert slice_lora_to_rank("x.lora_A.weight", tensor, 16) is tensor


def test_non_lora_names_pass_through():
    # [FUNCTION DOC] A tensor whose name is not a LoRA weight name is returned as the same object,
    # even when its size exceeds the requested rank.
    tensor = torch.ones(32, 8)
    assert slice_lora_to_rank("x.some_other.weight", tensor, 16) is tensor


# Packed grouped-expert exports ([E, rank, in] / [E, out, rank]) must be sliced on the
# rank axis — dim 0/1 slicing would drop experts while reporting the requested shape.


def test_packed_expert_lora_a_is_sliced_on_the_rank_dim():
    # [FUNCTION DOC] A packed expert LoRA A tensor of shape (4 experts, 32, 8) with the first 16
    # ranks live is sliced to rank 16: asserts shape (4, 16, 8) and equality with the first 16
    # entries along the middle axis (experts are kept).
    # 4 experts, max rank 32, in_features 8; only the first 16 ranks are live.
    tensor = torch.zeros(4, 32, 8)
    tensor[:, :16] = 1.0
    out = slice_lora_to_rank("base_model.experts.gate_proj.lora_A.weight", tensor, 16)
    assert out.shape == (4, 16, 8)
    assert torch.equal(out, tensor[:, :16])


def test_packed_expert_lora_b_is_sliced_on_the_rank_dim():
    # [FUNCTION DOC] A packed expert LoRA B tensor of shape (4, 8, 32) with the first 16 ranks
    # live is sliced to rank 16: asserts shape (4, 8, 16) and equality with the first 16 entries
    # along the last axis.
    tensor = torch.zeros(4, 8, 32)
    tensor[..., :16] = 1.0
    out = slice_lora_to_rank("base_model.experts.gate_proj.lora_B.weight", tensor, 16)
    assert out.shape == (4, 8, 16)
    assert torch.equal(out, tensor[..., :16])


def test_packed_expert_nonzero_padding_is_rejected():
    # [FUNCTION DOC] An all-ones packed expert LoRA A tensor sliced to rank 16 raises an assertion
    # error saying the padded dims are non-zero.
    tensor = torch.ones(4, 32, 8)
    with pytest.raises(AssertionError, match="padded dims are non-zero"):
        slice_lora_to_rank("x.experts.gate_proj.lora_A.weight", tensor, 16)


def test_packed_expert_fewer_experts_than_rank_is_not_confused():
    # [FUNCTION DOC] Packed expert LoRA A of shape (2 experts, 32, 8) with 4 live ranks sliced to
    # rank 4: asserts shape (2, 4, 8), which only holds if the rank axis (not the expert axis)
    # is sliced.
    # 2 experts, max rank 32, sliced to 4: dim-0 slicing would silently pass the shape check.
    tensor = torch.zeros(2, 32, 8)
    tensor[:, :4] = 1.0
    out = slice_lora_to_rank("x.experts.gate_proj.lora_A.weight", tensor, 4)
    assert out.shape == (2, 4, 8)
