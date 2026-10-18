# [MODULE DOC] CPU tests for rollout sampling-support masks: rebuilding mask objects from the
# transport batch fields (ids plus offsets; both required), building a local boolean mask for
# selected response rows and one tensor-parallel vocabulary shard (with errors for out-of-range
# indices, row misalignment and malformed tensor indices, and no selection work for empty rows), the
# true-on-policy log-prob calculation that masks the log-prob to the support but keeps full-vocab
# entropy and whose gradient equals the support-normalized one, per-response sampling support in the
# log-prob/entropy extraction (and its length-mismatch error), and that response chunk iteration keeps
# global response indices for allgather and zigzag context parallel splits.
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data.sampling_mask import build_local_sampling_mask, get_rollout_sampling_masks
from miles.backends.training_utils.loss.hub import logit_processors
from miles.backends.training_utils.loss.hub.math_utils import _calculate_log_probs_and_entropy_true_on_policy
from miles.utils.sampling_mask import RolloutSamplingMask


def test_get_rollout_sampling_masks_reconstructs_transport_batch():
    # [FUNCTION DOC] Test: ids [1,4,2] with offsets [0,2,3] rebuild one mask whose internal tensors
    # are the same ids and offsets.
    masks = get_rollout_sampling_masks(
        {
            "rollout_sampling_mask_ids": [torch.tensor([1, 4, 2], dtype=torch.int32)],
            "rollout_sampling_mask_offsets": [torch.tensor([0, 2, 3], dtype=torch.int64)],
        }
    )

    ids, offsets = masks[0]._as_tensors()
    assert ids.tolist() == [1, 4, 2]
    assert offsets.tolist() == [0, 2, 3]


def test_get_rollout_sampling_masks_requires_complete_transport_batch():
    # [FUNCTION DOC] Test: a batch with only the ids field raises a value error saying both wire
    # fields are required.
    with pytest.raises(ValueError, match="requires both sampling-mask wire fields"):
        get_rollout_sampling_masks({"rollout_sampling_mask_ids": [torch.tensor([1])]})


def test_build_local_sampling_mask_selects_original_response_rows_and_tp_shard():
    # [FUNCTION DOC] Test: local logits of shape 2 by 4 (a vocabulary shard of width 4, tensor
    # parallel rank 1 so global ids 4 to 7), a mask over three response positions with supports
    # [1,3], [4,2], [5,6,7], and response indices [2,0]. The boolean result is row 0 (position 2:
    # ids 5,6,7 -> local columns 1,2,3) = [F,T,T,T] and row 1 (position 0: ids 1,3, outside the
    # shard) all false.
    logits = torch.zeros(2, 4)
    mask = build_local_sampling_mask(
        logits,
        sampling_mask=RolloutSamplingMask.from_mask_list([[1, 3], [4, 2], [5, 6, 7]]),
        response_indices=[2, 0],
        tp_rank=1,
    )

    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [False, True, True, True],
                [False, False, False, False],
            ]
        ),
    )


def test_build_local_sampling_mask_rejects_out_of_range_response_index():
    # [FUNCTION DOC] Test: a mask with one position and response index 1 raises a value error
    # saying response indices must be in [0, 1).
    with pytest.raises(ValueError, match=r"response indices must be in \[0, 1\)"):
        build_local_sampling_mask(
            torch.zeros(1, 4),
            sampling_mask=RolloutSamplingMask.from_mask_list([[0]]),
            response_indices=[1],
            tp_rank=0,
        )


def test_build_local_sampling_mask_rejects_row_misalignment():
    # [FUNCTION DOC] Test: two logits rows but only one response index raises a value error saying
    # sampling-mask rows must align with logits (indices=1, logits=2).
    with pytest.raises(ValueError, match="sampling-mask rows must align with logits: indices=1, logits=2"):
        build_local_sampling_mask(
            torch.zeros(2, 4),
            sampling_mask=RolloutSamplingMask.from_mask_list([[0], [1]]),
            response_indices=[0],
            tp_rank=0,
        )


def test_build_local_sampling_mask_skips_selection_for_empty_local_rows(monkeypatch):
    # [FUNCTION DOC] Test: with the mask object's internal selection method replaced by one that
    # fails, zero logits rows and an empty index range give a boolean result of shape (0, 4) without
    # invoking the selection.
    def unexpected_selection(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Replacement that always raises an assertion error (selection must
        # not happen for empty local rows).
        raise AssertionError("empty local rows must not select sampling-mask ids")

    monkeypatch.setattr(RolloutSamplingMask, "_select_masks", unexpected_selection)

    mask = build_local_sampling_mask(
        torch.zeros(0, 4),
        sampling_mask=RolloutSamplingMask.from_mask_list([[0]]),
        response_indices=range(0),
        tp_rank=0,
    )

    assert mask.shape == (0, 4)
    assert mask.dtype == torch.bool


@pytest.mark.parametrize(
    "response_indices",
    [torch.tensor(0), torch.empty(0, dtype=torch.float32), torch.empty((0, 1), dtype=torch.long)],
)
def test_build_local_sampling_mask_validates_malformed_tensor_indices(response_indices):
    # [FUNCTION DOC] Test (parametrized: scalar tensor, empty float tensor, empty two-dimensional
    # integer tensor): each malformed index tensor raises a value error saying indices must be
    # one-dimensional integers.
    with pytest.raises(ValueError, match="must be one-dimensional integers"):
        build_local_sampling_mask(
            torch.zeros(0, 4),
            sampling_mask=RolloutSamplingMask.from_mask_list([[0]]),
            response_indices=response_indices,
            tp_rank=0,
        )


def test_true_on_policy_masks_logprob_but_keeps_full_vocab_entropy():
    # [FUNCTION DOC] Test: one row of logits [2,1,0,-1], token 0, support mask [T,F,T,F]. The
    # log-prob equals log-softmax after filling unsupported logits with negative infinity, taken at
    # the token; the entropy equals the entropy of the full (unmasked) distribution.
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], requires_grad=True)
    tokens = torch.tensor([0])
    sampling_mask = torch.tensor([[True, False, True, False]])

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        sampling_mask=sampling_mask,
    )

    expected_masked_logprob = torch.log_softmax(logits.masked_fill(~sampling_mask, float("-inf")), dim=-1)[0, 0]
    full_log_probs = torch.log_softmax(logits, dim=-1)
    expected_entropy = -(full_log_probs.exp() * full_log_probs).sum(dim=-1)
    torch.testing.assert_close(log_probs, expected_masked_logprob.unsqueeze(0))
    torch.testing.assert_close(entropy, expected_entropy)


def test_sampling_support_normalization_has_the_expected_actor_gradient():
    # [FUNCTION DOC] Test: two float64 rows with tokens 0 and 1; row 0's support is {0,2}, row 1's
    # support is {1} only. Backpropagating the sum of log-probs gives, for row 0, gradient 1 - p0
    # at column 0 and -p2 at column 2 (p being the softmax restricted to the support), and zeros
    # elsewhere; row 1 (single-element support, log-prob exactly zero) has zero gradient.
    logits = torch.tensor(
        [[2.0, 1.0, 0.0, -1.0], [-1.0, 2.0, 1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    tokens = torch.tensor([0, 1])
    sampling_mask = torch.tensor([[True, False, True, False], [False, True, False, False]])

    log_probs, _ = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        sampling_mask=sampling_mask,
    )
    log_probs.sum().backward()

    first_support_probs = torch.softmax(logits.detach()[0, [0, 2]], dim=-1)
    expected_grad = torch.zeros_like(logits)
    expected_grad[0, 0] = 1 - first_support_probs[0]
    expected_grad[0, 2] = -first_support_probs[1]

    torch.testing.assert_close(logits.grad, expected_grad)


def test_get_log_probs_and_entropy_applies_per_response_sampling_support(monkeypatch):
    # [FUNCTION DOC] Test: fake single-rank parallel state, true-on-policy args, logits of shape 1 by
    # 3 by 4, a sequence of 3 tokens [2,0,3] with response length 2 and per-position supports
    # {0,2} and {1,3}. The extracted log-probs equal, for the two response rows (positions 0 and
    # 1), the log-softmax restricted to each support evaluated at the target token (tokens 0 and 3
    # => support entries 0 and 1 respectively).
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0, group=None),
        cp=SimpleNamespace(rank=0, size=1),
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=True,
        bf16=False,
        fp16=False,
        log_probs_chunk_size=-1,
        vocab_size=4,
        allgather_cp=False,
        debug_unified_grad_fused_logprob=False,
    )
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )

    result = logit_processors.get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([2, 0, 3])],
        total_lengths=[3],
        response_lengths=[2],
        rollout_sampling_mask=[RolloutSamplingMask.from_mask_list([[0, 2], [1, 3]])],
    )

    expected = torch.stack(
        [
            torch.log_softmax(logits[0, 0, [0, 2]], dim=-1)[0],
            torch.log_softmax(logits[0, 1, [1, 3]], dim=-1)[1],
        ]
    )
    torch.testing.assert_close(result["log_probs"][0], expected)


def test_get_log_probs_and_entropy_rejects_mask_shorter_than_response(monkeypatch):
    # [FUNCTION DOC] Test: a mask with 1 position for a response of length 2 raises a value error
    # saying the sampling-mask length 1 differs from the response length 2.
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(rank=0, group=None),
        cp=SimpleNamespace(rank=0, size=1),
    )
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(qkv_format="thd", rollout_temperature=1.0, true_on_policy_mode=False, allgather_cp=False)

    with pytest.raises(ValueError, match="sampling-mask length 1 != response length 2"):
        logit_processors.get_log_probs_and_entropy(
            torch.zeros(1, 3, 4),
            args=args,
            unconcat_tokens=[torch.tensor([2, 0, 3])],
            total_lengths=[3],
            response_lengths=[2],
            rollout_sampling_mask=[RolloutSamplingMask.from_mask_list([[0]])],
        )


@pytest.mark.parametrize(("cp_rank", "expected_indices"), [(0, [0, 1]), (1, [2, 3])])
def test_allgather_cp_response_rows_keep_global_response_indices(monkeypatch, cp_rank, expected_indices):
    # [FUNCTION DOC] Test (parametrized: cp rank 0 expects response indices [0,1]; rank 1 expects
    # [2,3]): with allgather context parallel of size 2, total length 6, response length 4 and 3
    # local logits rows, the first response chunk (requested with indices) has those global
    # response indices, tokens equal 2 plus each index, and as many logits rows as indices.
    parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=True,
    )

    response_chunks = list(
        logit_processors._iter_response_chunks(
            torch.zeros(1, 3, 4),
            args=args,
            unconcat_tokens=[torch.arange(6)],
            total_lengths=[6],
            response_lengths=[4],
            include_response_indices=True,
        )
    )
    logits_chunk, tokens_chunk, response_indices = response_chunks[0]

    assert list(response_indices) == expected_indices
    assert tokens_chunk.tolist() == [2 + index for index in expected_indices]
    assert logits_chunk.size(0) == len(expected_indices)


@pytest.mark.parametrize(("cp_rank", "expected_indices"), [(0, [4]), (1, [0, 1, 2, 3])])
def test_zigzag_cp_response_rows_keep_global_response_indices(monkeypatch, cp_rank, expected_indices):
    # [FUNCTION DOC] Test (parametrized: rank 0 expects [4]; rank 1 expects [0,1,2,3]): with zigzag
    # context parallel of size 2, total length 8, response length 5 and 4 logits rows, the first
    # response chunk has those global response indices, tokens equal 3 plus each index, and as many
    # logits rows as indices. The parallel state is patched in both the log-prob module and the
    # context-parallel module.
    parallel_state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=2))
    monkeypatch.setattr(logit_processors, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    args = SimpleNamespace(
        qkv_format="thd",
        rollout_temperature=1.0,
        true_on_policy_mode=False,
        allgather_cp=False,
    )

    response_chunks = list(
        logit_processors._iter_response_chunks(
            torch.zeros(1, 4, 4),
            args=args,
            unconcat_tokens=[torch.arange(8)],
            total_lengths=[8],
            response_lengths=[5],
            include_response_indices=True,
        )
    )
    logits_chunk, tokens_chunk, response_indices = response_chunks[0]

    assert list(response_indices) == expected_indices
    assert tokens_chunk.tolist() == [3 + index for index in expected_indices]
    assert logits_chunk.size(0) == len(expected_indices)
