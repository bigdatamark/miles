# [MODULE DOC] CPU unit tests for the true-on-policy log-probability and entropy computation:
# truncation of padded vocabulary columns before the log-softmax, gathering of tensor-parallel
# shards, chunked evaluation, gradient equivalence with an unchunked reference, and the backward
# split of a replicated gradient across tensor-parallel ranks. Expected values are computed with
# plain torch operations.
import torch

from miles.backends.training_utils.loss.hub.math_utils import (
    _calculate_log_probs_and_entropy_true_on_policy,
    _prepare_true_on_policy_full_logits,
    _split_replicated_loss_gather_grad,
    calculate_log_probs_and_entropy,
)


def test_true_on_policy_logprobs_tp1_truncate_after_real_vocab():
    # [FUNCTION DOC] Uses half-precision logits of width 6 with a real vocabulary size of 4 (the
    # last two columns hold large padding values) and no parallel group. Asserts the selected
    # token log-probs and the entropy equal those computed with a log-softmax over only the first
    # four columns.
    logits = torch.tensor(
        [
            [1.0, 0.0, -1.0, 3.0, 40.0, 50.0],
            [2.0, 1.0, 0.5, -0.5, 60.0, 70.0],
        ],
        dtype=torch.float16,
    )
    tokens = torch.tensor([3, 0], dtype=torch.long)

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        vocab_size=4,
    )

    expected_log_probs_full = torch.log_softmax(logits[:, :4], dim=-1)
    expected_log_probs = expected_log_probs_full.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(expected_log_probs_full.exp() * expected_log_probs_full).sum(dim=-1)

    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)


def test_true_on_policy_fake_tp_vocab_gather_truncates_before_log_softmax():
    # [FUNCTION DOC] Builds two half-precision logit shards of width 4 each (padded columns hold
    # large values), gathers them with a real vocabulary size of 6, and computes log-probs from
    # the gathered logits. Asserts the gathered logits equal the concatenated shards truncated to
    # 6 columns, and the selected log-probs match a log-softmax over those 6 columns.
    shard_0 = torch.tensor(
        [
            [5.0, 1.0, -2.0, 0.0],
            [0.0, 3.0, -4.0, 1.0],
        ],
        dtype=torch.float16,
    )
    shard_1 = torch.tensor(
        [
            [2.0, 4.0, 30.0, 40.0],
            [-1.0, 2.0, 50.0, 60.0],
        ],
        dtype=torch.float16,
    )
    tokens = torch.tensor([5, 4], dtype=torch.long)

    gathered_logits = _prepare_true_on_policy_full_logits((shard_0, shard_1), vocab_size=6)
    log_probs, _ = _calculate_log_probs_and_entropy_true_on_policy(
        gathered_logits,
        tokens,
        None,
        vocab_size=6,
    )

    expected_full_logits = torch.cat([shard_0, shard_1], dim=-1)[:, :6]
    expected_log_probs = torch.log_softmax(expected_full_logits, dim=-1)
    expected_selected = expected_log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    torch.testing.assert_close(gathered_logits, expected_full_logits)
    torch.testing.assert_close(log_probs, expected_selected)


def test_true_on_policy_logprobs_honor_chunk_size(monkeypatch):
    # [FUNCTION DOC] Computes reference log-probs and entropy for random 5 by 7 logits that
    # require gradients. Patches the torch log-softmax (reverted at teardown) with a wrapper
    # recording the row count of each call, then runs the public log-prob and entropy function in
    # true-on-policy mode with chunk size 2 and entropy gradients disabled. Asserts the recorded
    # chunk sizes are 2, 2 and 1, the log-probs require grad while the entropy does not, and both
    # values match the references.
    logits = torch.randn(5, 7, dtype=torch.float32, requires_grad=True)
    tokens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    expected_log_probs_full = torch.log_softmax(logits, dim=-1)
    expected_log_probs = expected_log_probs_full.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    expected_entropy = -(expected_log_probs_full.exp() * expected_log_probs_full).sum(dim=-1)

    chunk_rows = []
    original_log_softmax = torch.log_softmax

    def record_chunk_rows(input_: torch.Tensor, dim: int) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Wrapper around the original log-softmax that appends the number of
        # rows (size of the first dimension) of each input to the recorded list, then delegates to
        # the original with the same dimension and returns its result unchanged.
        chunk_rows.append(input_.size(0))
        return original_log_softmax(input_, dim=dim)

    monkeypatch.setattr(torch, "log_softmax", record_chunk_rows)
    log_probs, entropy = calculate_log_probs_and_entropy(
        logits,
        tokens,
        None,
        with_entropy=True,
        entropy_requires_grad=False,
        chunk_size=2,
        true_on_policy=True,
    )

    assert chunk_rows == [2, 2, 1]
    assert log_probs.requires_grad
    assert not entropy.requires_grad
    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)


def test_true_on_policy_chunked_entropy_preserves_gradients():
    # [FUNCTION DOC] Computes log-probs and entropy with chunk size 2 and without chunking on a
    # detached copy of random double-precision logits, back-propagates the sum of both outputs in
    # each case, and asserts the values and the logit gradients agree between the chunked and
    # unchunked runs.
    logits = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    tokens = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    reference_logits = logits.detach().clone().requires_grad_(True)

    log_probs, entropy = _calculate_log_probs_and_entropy_true_on_policy(
        logits,
        tokens,
        None,
        with_entropy=True,
        chunk_size=2,
    )
    reference_log_probs, reference_entropy = _calculate_log_probs_and_entropy_true_on_policy(
        reference_logits,
        tokens,
        None,
        with_entropy=True,
    )
    (log_probs.sum() + entropy.sum()).backward()
    (reference_log_probs.sum() + reference_entropy.sum()).backward()

    torch.testing.assert_close(log_probs, reference_log_probs)
    torch.testing.assert_close(entropy, reference_entropy)
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_true_on_policy_replicated_loss_gather_backward_splits_without_tp_sum():
    # [FUNCTION DOC] Takes a 2 by 6 gradient and splits it for rank 0 and rank 1 of a world of 2
    # with local width 3. Asserts each rank gets its own contiguous column block unchanged (no
    # summation across ranks, since the loss is replicated).
    grad_output = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    rank0_grad = _split_replicated_loss_gather_grad(
        grad_output,
        rank=0,
        world_size=2,
        local_last_dim=3,
    )
    rank1_grad = _split_replicated_loss_gather_grad(
        grad_output,
        rank=1,
        world_size=2,
        local_last_dim=3,
    )

    torch.testing.assert_close(rank0_grad, grad_output[:, :3])
    torch.testing.assert_close(rank1_grad, grad_output[:, 3:])
