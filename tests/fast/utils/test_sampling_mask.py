# [MODULE DOC] Unit tests for the rollout sampling mask container, which stores per-token lists of
# allowed token ids in a flattened compressed-row layout (an id array plus an offsets array) and
# selects masks for given response positions. Tests cover construction from nested lists, input
# validation, ownership of storage, and the private mask selection routine.
import pytest
import torch

from miles.utils.sampling_mask import RolloutSamplingMask


def test_rollout_sampling_mask_builds_private_int32_storage():
    # [FUNCTION DOC] Builds a mask from three lists, selects all three positions, and asserts: the
    # length is 3; ids are 32-bit integers and lengths are 64-bit; flattened ids and per-token
    # lengths match the input; and the public ids and offsets attributes do not exist (storage is
    # private).
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7]])

    ids, lengths = mask._select_masks(range(3))

    assert len(mask) == 3
    assert ids.dtype == torch.int32
    assert lengths.dtype == torch.long
    assert ids.tolist() == [1, 3, 4, 2, 5, 6, 7]
    assert lengths.tolist() == [2, 2, 3]
    assert not hasattr(mask, "ids")
    assert not hasattr(mask, "offsets")


def test_rollout_sampling_mask_requires_non_empty_mask():
    # [FUNCTION DOC] Asserts building from lists where one entry is empty raises a value error
    # saying every response token needs a non-empty sampling mask.
    with pytest.raises(ValueError, match="every response token needs a non-empty sampling mask"):
        RolloutSamplingMask.from_mask_list([[], [1]])


def test_rollout_sampling_mask_owns_input_storage():
    # [FUNCTION DOC] Constructs a mask directly from a 16-bit id tensor and a 32-bit offsets
    # tensor, then mutates both source tensors in place. Asserts the selected ids and lengths still
    # reflect the original values, proving the mask copied its inputs.
    ids = torch.tensor([5, 7], dtype=torch.int16)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    mask = RolloutSamplingMask(ids=ids, offsets=offsets)

    ids[0] = 99
    offsets[1] = 99
    selected_ids, lengths = mask._select_masks(range(2))

    assert selected_ids.tolist() == [5, 7]
    assert lengths.tolist() == [1, 1]


def test_rollout_sampling_mask_validates_csr_offsets():
    # [FUNCTION DOC] Asserts that offsets which do not end at the flattened id count raise a value
    # error saying offsets must start at zero and end at the flattened id count.
    with pytest.raises(ValueError, match="offsets must start at zero and end at the flattened id count"):
        RolloutSamplingMask(ids=torch.tensor([0, 1]), offsets=torch.tensor([0, 1]))


@pytest.mark.parametrize("ids", [torch.tensor([1.5]), torch.tensor([True]), torch.tensor([[1]])])
def test_rollout_sampling_mask_rejects_non_integer_or_non_1d_tensor_ids(ids):
    # [FUNCTION DOC] Parametrized over a float tensor, a boolean tensor and a two-dimensional
    # integer tensor. Asserts construction raises a value error saying ids must be
    # one-dimensional integers.
    with pytest.raises(ValueError, match="must be one-dimensional integers"):
        RolloutSamplingMask(ids=ids, offsets=[0, 1])


def test_select_masks_slices_a_contiguous_range_and_reports_lengths():
    # [FUNCTION DOC] Selects the contiguous range of positions 1 and 2 from a four-entry mask and
    # asserts the returned ids are those two entries concatenated and the lengths are 2 and 3.
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7], [8]])

    ids, lengths = mask._select_masks(range(1, 3))

    assert ids.tolist() == [4, 2, 5, 6, 7]
    assert lengths.tolist() == [2, 3]


def test_select_masks_concatenates_disjoint_runs_in_order():
    # [FUNCTION DOC] Selects positions 0, 1 and 3 (given as a tensor, skipping position 2) and
    # asserts ids are concatenated in the requested order with lengths 2, 2 and 1.
    mask = RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7], [8]])

    ids, lengths = mask._select_masks(torch.tensor([0, 1, 3]))

    assert ids.tolist() == [1, 3, 4, 2, 8]
    assert lengths.tolist() == [2, 2, 1]


@pytest.mark.parametrize("token_indices", [range(0), torch.empty(0, dtype=torch.long)])
def test_select_masks_handles_no_tokens(token_indices):
    # [FUNCTION DOC] Parametrized over an empty range and an empty long tensor. Asserts selection
    # returns empty ids and empty lengths without error.
    mask = RolloutSamplingMask.from_mask_list([[1]])

    ids, lengths = mask._select_masks(token_indices)

    assert ids.numel() == 0
    assert lengths.numel() == 0


@pytest.mark.parametrize("token_indices", [range(1, 2), [1]])
def test_select_masks_rejects_out_of_range_position(token_indices):
    # [FUNCTION DOC] Parametrized over a range and a list, both pointing at position 1 of a
    # one-entry mask. Asserts a value error saying response indices must lie in the half-open
    # interval from zero to the mask length.
    mask = RolloutSamplingMask.from_mask_list([[0]])

    with pytest.raises(ValueError, match=r"response indices must be in \[0, 1\)"):
        mask._select_masks(token_indices)
