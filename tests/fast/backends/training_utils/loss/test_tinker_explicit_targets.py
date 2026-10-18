# [MODULE DOC] CPU tests that the tinker-style loss scoring treats the batch's target tokens as
# explicit labels to gather log-probs for (independent of the input sequence), and that shifted
# targets (the sequence minus its first token) reproduce ordinary next-token scoring. The log-prob
# kernel is replaced by a naive log-softmax gather, and a trivial parallel state is installed.
"""target_tokens are explicit gather labels; shifted targets reproduce next-token scoring."""

import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss.hub import math_utils, tinker_losses

VOCAB = 32


def _naive_compute_log_probs(logits, tokens, _tp_group, *, sampling_mask=None, debug_unified_grad_fused_logprob=False):
    # [FUNCTION DOC] HELPER: Reference replacement for the log-prob kernel: log-softmax over the
    # last (vocab) axis, then gathers the entry of each token id (token tensor given an added
    # trailing axis). Returns the gathered log-probs with a trailing axis of size 1. The tensor
    # parallel group, sampling mask and debug flag arguments are accepted and ignored.
    return torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1))


def _target_logprobs(batch, logits):
    # [FUNCTION DOC] HELPER: Builds args (not true-on-policy, temperature 1.0, chunk size 64) and
    # calls the tinker loss module's private target log-prob routine on the batch and logits.
    # Returns its per-sample list of log-prob tensors.
    args = make_args(true_on_policy_mode=False, rollout_temperature=1.0, log_probs_chunk_size=64)
    return tinker_losses._target_logprobs(args, batch, logits)


def _batch(sequences: list[list[int]], targets: list[list[int]]) -> dict:
    # [FUNCTION DOC] HELPER: Builds a batch dictionary from lists of token-id sequences and target
    # lists: tensors of the sequences, the targets as given, total lengths (sequence lengths), and
    # response lengths (target lengths).
    return {
        "unconcat_tokens": [torch.tensor(sequence) for sequence in sequences],
        "target_tokens": targets,
        "total_lengths": [len(sequence) for sequence in sequences],
        "response_lengths": [len(target) for target in targets],
    }


def test_targets_replace_the_sequence_as_gather_labels(monkeypatch):
    # [FUNCTION DOC] Test: two sequences (lengths 5 and 3) with targets (3 and 2) deliberately
    # different from the sequence tokens, random logits for 8 packed positions. For each sample the
    # returned log-probs equal the log-softmax of the packed logits at row (offset + length - target
    # count - 1 + i) gathered at the i-th target label, where offset is the start of the sample in
    # the packed axis (i.e. predictions come from the positions just before the response).
    monkeypatch.setattr(math_utils, "compute_log_probs", _naive_compute_log_probs)
    make_parallel_state()
    generator = torch.Generator().manual_seed(0)
    sequences = [[1, 2, 3, 4, 5], [6, 7, 8]]
    targets = [[9, 10, 11], [12, 13]]  # rl_loop-style: labels differ from the sequence
    logits = torch.randn((1, 8, VOCAB), generator=generator)

    log_probs = _target_logprobs(_batch(sequences, targets), logits)

    flat = torch.log_softmax(logits.squeeze(0), dim=-1)
    offset = 0
    for sequence, target, log_prob in zip(sequences, targets, log_probs, strict=True):
        start = offset + len(sequence) - len(target) - 1
        expected = [flat[start + i, label] for i, label in enumerate(target)]
        torch.testing.assert_close(log_prob, torch.stack(expected))
        offset += len(sequence)


def test_shifted_targets_degenerate_to_next_token_scoring(monkeypatch):
    # [FUNCTION DOC] Test: one random sequence of 6 tokens with targets equal to the sequence minus
    # its first token; the log-probs equal the log-softmax of all logit rows except the last,
    # gathered at the shifted tokens (standard next-token scoring).
    monkeypatch.setattr(math_utils, "compute_log_probs", _naive_compute_log_probs)
    make_parallel_state()
    generator = torch.Generator().manual_seed(1)
    sequence = torch.randint(0, VOCAB, (6,), generator=generator).tolist()
    logits = torch.randn((1, 6, VOCAB), generator=generator)

    (log_probs,) = _target_logprobs(_batch([sequence], [sequence[1:]]), logits)

    expected = torch.log_softmax(logits.squeeze(0)[:-1], dim=-1).gather(-1, torch.tensor(sequence[1:]).unsqueeze(-1))
    torch.testing.assert_close(log_probs, expected.squeeze(-1))
