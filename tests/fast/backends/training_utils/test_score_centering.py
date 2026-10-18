# [MODULE DOC] CPU (and optional CUDA) tests of the score centering loss and its helpers.
# Score centering corrects the sampled-token policy gradient with a baseline built from the
# retained head (top-k) of the rollout distribution; the importance mode is none, tis (clipped at
# an upper bound) or mis (kept only inside a band). Tests cover: invalid or NaN importance bounds and
# ratios failing clearly; zero advantage ignoring non-finite ratios, log-probs and head candidates;
# shape validation; finite gradients when a weight is zero and a log-prob is negative infinity;
# gradient equality against an independent dense-distribution reconstruction (CPU and CUDA);
# a zero gradient for a constant reward over the full distribution; no correction for matched
# policies or unclipped importance sampling; finite, detached tiny-tail metrics with a fixed key set;
# and the selected log-prob helpers (chunking, padding ids, temperature, vocabulary slicing, entropy,
# numeric gradient check, empty responses).
"""Compare the implemented gradient with independent dense-distribution oracles."""

import math

import pytest
import torch

from miles.backends.training_utils.loss.hub.score_centering import (
    ScoreCenteringInputs,
    importance_weights,
    score_centering_loss,
    selected_log_probs,
    selected_log_probs_and_entropy,
)


@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("tis", {"tis_clip": 0.0}),
        ("tis", {"tis_clip": float("inf")}),
        ("mis", {"mis_low": 0.0}),
        ("mis", {"mis_low": -1.0}),
        ("mis", {"mis_low": 2.0, "mis_high": 1.0}),
        ("mis", {"mis_high": float("nan")}),
    ],
)
def test_invalid_importance_bounds_fail_clearly(mode: str, kwargs: dict[str, float]) -> None:
    # [FUNCTION DOC] Test (parametrized over tis clip 0 or infinity, mis low 0 or negative, mis
    # low above high, mis high NaN): both the importance weights helper and the score centering
    # loss (with a minimal one-sample, one-candidate input) raise a value error whose message
    # mentions clip or bounds.
    with pytest.raises(ValueError, match="clip|bounds"):
        importance_weights(torch.tensor([0.0]), mode, **kwargs)
    with pytest.raises(ValueError, match="clip|bounds"):
        score_centering_loss(
            ScoreCenteringInputs(
                train_log_probs=torch.tensor([-1.0]),
                train_head_log_probs=torch.tensor([[-1.0]]),
                rollout_log_probs=torch.tensor([-1.0]),
                rollout_head_log_probs=torch.tensor([[-1.0]]),
                head_mask=torch.tensor([[True]]),
                advantages=torch.ones(1),
                mode=mode,
                **kwargs,
            ),
        )


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_nan_importance_ratio_fails_clearly(mode: str) -> None:
    # [FUNCTION DOC] Test (parametrized over tis and mis): a NaN log ratio passed to the weights
    # helper raises a value error "log-ratio contains NaN"; so does the loss when train and rollout
    # sample log-probs are both negative infinity (infinity minus infinity gives NaN) with unit
    # advantage.
    with pytest.raises(ValueError, match="log-ratio contains NaN"):
        importance_weights(torch.tensor([float("nan")]), mode)
    with pytest.raises(ValueError, match="log-ratio contains NaN"):
        score_centering_loss(
            ScoreCenteringInputs(
                train_log_probs=torch.tensor([float("-inf")]),
                train_head_log_probs=torch.tensor([[-1.0]]),
                rollout_log_probs=torch.tensor([float("-inf")]),
                rollout_head_log_probs=torch.tensor([[-1.0]]),
                head_mask=torch.tensor([[True]]),
                advantages=torch.ones(1),
                mode=mode,
            ),
        )


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_zero_advantage_ignores_nan_importance_ratio(mode: str) -> None:
    # [FUNCTION DOC] Test (tis, mis): the same negative-infinity pair with zero advantage does not
    # raise; the loss is zero and the logged importance weight is finite.
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=torch.tensor([float("-inf")]),
            train_head_log_probs=torch.tensor([[-1.0]]),
            rollout_log_probs=torch.tensor([float("-inf")]),
            rollout_head_log_probs=torch.tensor([[-1.0]]),
            head_mask=torch.tensor([[True]]),
            advantages=torch.zeros(1),
            mode=mode,
        ),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert torch.isfinite(metrics.importance_weight).all()


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_zero_advantage_preserves_finite_importance_weight(mode: str) -> None:
    # [FUNCTION DOC] Test (tis, mis): with finite log-probs whose ratio is 0.8 and zero advantage,
    # the logged importance weight is 0.8 (still reported although it does not affect the loss).
    _, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=torch.tensor([-1.0]),
            train_head_log_probs=torch.tensor([[-1.0]]),
            rollout_log_probs=torch.tensor([-1.0 - math.log(0.8)]),
            rollout_head_log_probs=torch.tensor([[-1.0]]),
            head_mask=torch.tensor([[True]]),
            advantages=torch.zeros(1),
            mode=mode,
        ),
    )
    torch.testing.assert_close(metrics.importance_weight, torch.tensor([0.8]))


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("bad_side", ["train", "rollout"])
def test_zero_advantage_ignores_nan_head_candidate(mode: str, bad_side: str) -> None:
    # [FUNCTION DOC] Test (modes none, tis, mis; NaN placed in the train or the rollout head
    # candidates): with zero advantage the loss is zero and every logged metric is finite.
    train_head = torch.tensor([[float("nan") if bad_side == "train" else -1.0, -1.5]])
    rollout_head = torch.tensor([[float("nan") if bad_side == "rollout" else -1.0, -1.5]])
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=torch.tensor([-1.0]),
            train_head_log_probs=train_head,
            rollout_log_probs=torch.tensor([-1.0]),
            rollout_head_log_probs=rollout_head,
            head_mask=torch.tensor([[True, True]]),
            advantages=torch.zeros(1),
            mode=mode,
        ),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
    assert all(torch.isfinite(value).all() for value in metrics.as_log_dict().values())


@pytest.mark.parametrize("bad_index", [1, 2, 3, 4, 5])
def test_score_centering_loss_rejects_broadcastable_shapes(bad_index: int) -> None:
    # [FUNCTION DOC] Test (parametrized over input positions 1 to 5): starting from consistent
    # positional inputs (train sample log-prob, train head, rollout sample log-prob, rollout head,
    # head mask, advantages), one input is made shape-inconsistent but still broadcastable (head
    # tensors and mask truncated to one candidate, or sample-level tensors given an extra leading
    # axis); the loss raises a value error saying the sample tensors must be consistent.
    # KNOWN ISSUE: index 0 (the train sample log-prob) is never perturbed, so that input's shape
    # check is untested.
    values = [
        torch.tensor([-1.0]),
        torch.tensor([[-1.0, -2.0]]),
        torch.tensor([-1.0]),
        torch.tensor([[-1.0, -2.0]]),
        torch.tensor([[True, True]]),
        torch.ones(1),
    ]
    values[bad_index] = values[bad_index][..., :1] if bad_index in (1, 3, 4) else values[bad_index][None]
    with pytest.raises(ValueError, match="sample tensors must be"):
        score_centering_loss(ScoreCenteringInputs(*values))


@pytest.mark.parametrize("mode", ["tis", "mis"])
@pytest.mark.parametrize("nonfinite", ["sample", "head"])
def test_zero_weight_negative_infinity_has_finite_loss_and_gradient(mode: str, nonfinite: str) -> None:
    # [FUNCTION DOC] Test (tis, mis; negative infinity placed in the train sample log-prob or in
    # the train head log-prob): float64 inputs with matching rollout probabilities 0.3 / 0.2 and
    # unit advantage. Backpropagating the summed loss yields a finite loss and finite gradients for
    # both train tensors, and the gradient of the infinite one is exactly zero.
    sample = torch.tensor(
        [float("-inf") if nonfinite == "sample" else torch.log(torch.tensor(0.3)).item()],
        dtype=torch.float64,
        requires_grad=True,
    )
    head = torch.tensor(
        [[float("-inf") if nonfinite == "head" else torch.log(torch.tensor(0.3)).item()]],
        dtype=torch.float64,
        requires_grad=True,
    )
    loss, _ = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=sample,
            train_head_log_probs=head,
            rollout_log_probs=torch.tensor([0.3], dtype=torch.float64).log(),
            rollout_head_log_probs=torch.tensor([[0.2]], dtype=torch.float64).log(),
            head_mask=torch.tensor([[True]]),
            advantages=torch.ones(1, dtype=torch.float64),
            mode=mode,
        ),
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(sample.grad).all()
    assert torch.isfinite(head.grad).all()
    if nonfinite == "sample":
        torch.testing.assert_close(sample.grad, torch.zeros_like(sample.grad))
    else:
        torch.testing.assert_close(head.grad, torch.zeros_like(head.grad))


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_zero_advantage_skips_nonfinite_log_probabilities(mode: str) -> None:
    # [FUNCTION DOC] Test (none, tis, mis): train sample and head log-probs both negative infinity
    # with zero advantage give a zero loss (no NaN).
    loss, _ = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=torch.tensor([float("-inf")]),
            train_head_log_probs=torch.tensor([[float("-inf")]]),
            rollout_log_probs=torch.tensor([-1.0]),
            rollout_head_log_probs=torch.tensor([[-1.0]]),
            head_mask=torch.tensor([[True]]),
            advantages=torch.zeros(1),
            mode=mode,
        ),
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))


@pytest.mark.parametrize("mode", ["tis", "mis"])
def test_masked_logit_from_selected_probability_helper_has_finite_gradient(mode: str) -> None:
    # [FUNCTION DOC] Test (tis, mis): logits [0, -inf] pass through the selected-log-prob helper
    # (ids [0,1]) to produce the sample log-prob and one head candidate (log-prob negative
    # infinity) feeding the loss with rollout probabilities 0.8 / 0.2 and unit advantage. The loss
    # and the logits gradient after backward are both finite.
    logits = torch.tensor([[0.0, float("-inf")]], dtype=torch.float64, requires_grad=True)
    selected, _ = selected_log_probs_and_entropy(logits, torch.tensor([[0, 1]]))
    loss, _ = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=selected[:, 0],
            train_head_log_probs=selected[:, 1:],
            rollout_log_probs=torch.tensor([0.8], dtype=torch.float64).log(),
            rollout_head_log_probs=torch.tensor([[0.2]], dtype=torch.float64).log(),
            head_mask=torch.tensor([[True]]),
            advantages=torch.ones(1, dtype=torch.float64),
            mode=mode,
        ),
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
@pytest.mark.parametrize("k", [2, 7])
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_gradient_matches_reconstructed_full_distribution(mode: str, k: int, device: str) -> None:
    # [FUNCTION DOC] Test (modes none/tis/mis, retained head size 2 or 7, device cpu or cuda when
    # available): float64, seeded. Logits are 4 rows by vocab 7 requiring grad; the rollout
    # distribution q is a sharpened softmax of random values; the head is q's top-k; the sampled
    # token is q's least likely one (outside the head when k < 7); advantages [1.7,-0.8,0,0.2].
    # The loss gradient w.r.t. logits is compared (absolute and relative tolerance 1e-12) with an
    # independent reference built from the dense distribution: tail mass of q spread proportionally
    # to the trainer distribution p (scale rho = (1 - head mass of q) / (1 - head mass of p), floors
    # 1e-6) to form q_hat, with head entries replaced by q's; ratio p/q_hat; weights 1 (none),
    # ratio clamped above at 2 (tis), or ratio kept only within [0.5, 5] else 0 (mis); same for the
    # sampled token; reference loss = -advantage * (sampled weight * log p[sampled] - sum over
    # vocab of q_hat * weight * log p), differentiated with autograd.
    generator = torch.Generator().manual_seed(27)
    logits = torch.randn(4, 7, generator=generator, dtype=torch.float64).to(device).requires_grad_()
    q = torch.softmax(torch.randn(4, 7, generator=generator, dtype=torch.float64).to(device) * 2, -1)
    q_head, ids = q.topk(k, dim=-1)
    sampled = q.argmin(-1)  # outside the retained head when k < vocabulary size
    advantage = torch.tensor([1.7, -0.8, 0.0, 0.2], dtype=torch.float64, device=device)
    logp = logits.log_softmax(-1)
    loss, _ = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=logp.gather(-1, sampled[:, None]).squeeze(-1),
            train_head_log_probs=logp.gather(-1, ids),
            rollout_log_probs=q.gather(-1, sampled[:, None]).squeeze(-1).log(),
            rollout_head_log_probs=q_head.log(),
            head_mask=torch.ones_like(ids, dtype=torch.bool),
            advantages=advantage,
            mode=mode,
        ),
    )
    actual_grad = torch.autograd.grad(loss.sum(), logits)[0]

    reference_logits = logits.detach().clone().requires_grad_()
    reference_logp = reference_logits.log_softmax(-1)
    p = reference_logp.detach().exp()
    rho = (1 - q_head.sum(-1)).clamp_min(1e-6) / (1 - p.gather(-1, ids).sum(-1)).clamp_min(1e-6)
    q_hat = p * rho[:, None]
    q_hat.scatter_(-1, ids, q_head)
    ratio = p / q_hat
    sampled_ratio = p.gather(-1, sampled[:, None]).squeeze(-1) / q.gather(-1, sampled[:, None]).squeeze(-1)
    if mode == "none":
        weights, sample_weights = torch.ones_like(ratio), torch.ones_like(sampled_ratio)
    elif mode == "tis":
        weights, sample_weights = ratio.clamp_max(2), sampled_ratio.clamp_max(2)
    else:
        weights = torch.where((ratio >= 0.5) & (ratio <= 5), ratio, 0.0)
        sample_weights = torch.where((sampled_ratio >= 0.5) & (sampled_ratio <= 5), sampled_ratio, 0.0)
    reference = -advantage * (
        sample_weights * reference_logp.gather(-1, sampled[:, None]).squeeze(-1)
        - (q_hat * weights * reference_logp).sum(-1)
    )
    expected_grad = torch.autograd.grad(reference.sum(), reference_logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_full_distribution_has_zero_constant_reward_gradient(mode: str) -> None:
    # [FUNCTION DOC] Test (none/tis/mis): a 4-way distribution where the head is the whole
    # vocabulary; every vocabulary item is treated as the sampled token with unit advantage. The
    # q-weighted sum of per-token losses has (absolute tolerance 1e-12) zero gradient w.r.t. the
    # logits: a constant reward gives no learning signal in expectation.
    logits = torch.tensor([0.7, -1.2, 2.3, 0.0], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([0.1, 0.5, 0.15, 0.25], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    loss, _ = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=logp,
            train_head_log_probs=logp.expand(4, 4),
            rollout_log_probs=q.log(),
            rollout_head_log_probs=q.log().expand(4, 4),
            head_mask=torch.ones(4, 4, dtype=torch.bool),
            advantages=torch.ones(4),
            mode=mode,
        ),
    )
    gradient = torch.autograd.grad((q * loss).sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)


def test_matched_policies_have_no_correction() -> None:
    # [FUNCTION DOC] Test: trainer and rollout distributions identical (0.7, 0.2, 0.1, head = first
    # two, sampled = third), advantage 2, default mode: the logged correction is zero and the
    # loss equals minus advantage times the sampled log-prob.
    logp = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64).log().requires_grad_()
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=logp[:, 2],
            train_head_log_probs=logp[:, :2],
            rollout_log_probs=logp[:, 2].detach(),
            rollout_head_log_probs=logp[:, :2].detach(),
            head_mask=torch.ones(1, 2, dtype=torch.bool),
            advantages=torch.tensor([2.0]),
        ),
    )
    torch.testing.assert_close(metrics.correction, torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(loss, -2 * logp[:, 2])


def test_unclipped_importance_sampling_has_no_correction() -> None:
    # [FUNCTION DOC] Test: trainer p = (0.55, 0.3, 0.15) and rollout q = (0.2, 0.5, 0.3) with head =
    # first two, sampled = third, unit advantage, tis mode with a loose clip of 10 (no clipping
    # occurs): the logged correction is zero within 1e-14.
    p = torch.tensor([[0.55, 0.3, 0.15]], dtype=torch.float64)
    q = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.float64)
    _, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=p[:, 2].log(),
            train_head_log_probs=p[:, :2].log(),
            rollout_log_probs=q[:, 2].log(),
            rollout_head_log_probs=q[:, :2].log(),
            head_mask=torch.ones(1, 2, dtype=torch.bool),
            advantages=torch.ones(1),
            mode="tis",
            tis_clip=10,
        ),
    )
    torch.testing.assert_close(metrics.correction, torch.zeros(1, dtype=torch.float64), atol=1e-14, rtol=0)


@pytest.mark.parametrize("mode", ["none", "tis", "mis"])
def test_tiny_tails_and_padding_are_finite_and_detached(mode: str) -> None:
    # [FUNCTION DOC] Test (none/tis/mis): trainer logits [40,-40,-80] (extreme tail), rollout head
    # log-probs [0, -inf] with the second candidate masked out as padding, sampled rollout log-prob
    # -0.1, unit advantage, all requiring grad. After backward: loss and logits gradient are finite,
    # every logged metric is finite, the rollout log-probs and advantage receive no gradient
    # (detached inside), and the logged dictionary has exactly the keys correction, train head mass,
    # rollout head mass, tail ratio, importance weight (prefixed with the sc marker) in that order,
    # each being the corresponding metrics object itself and not requiring grad.
    logits = torch.tensor([[40.0, -40.0, -80.0]], requires_grad=True)
    head = logits.log_softmax(-1)[:, :2]
    q = torch.tensor([[0.0, -torch.inf]], requires_grad=True)
    sampled_q = torch.tensor([-0.1], requires_grad=True)
    advantage = torch.ones(1, requires_grad=True)
    loss, metrics = score_centering_loss(
        ScoreCenteringInputs(
            train_log_probs=head[:, 0],
            train_head_log_probs=head,
            rollout_log_probs=sampled_q,
            rollout_head_log_probs=q,
            head_mask=torch.tensor([[True, False]]),
            advantages=advantage,
            mode=mode,
        ),
    )
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all()
    assert all(torch.isfinite(value).all() for value in metrics.as_log_dict().values())
    assert q.grad is None and sampled_q.grad is None and advantage.grad is None
    logged = metrics.as_log_dict()
    expected = {
        "sc_correction": metrics.correction,
        "sc_train_head_mass": metrics.train_head_mass,
        "sc_rollout_head_mass": metrics.rollout_head_mass,
        "sc_tail_ratio": metrics.tail_ratio,
        "sc_importance_weight": metrics.importance_weight,
    }
    assert list(logged) == list(expected)
    assert all(logged[key] is value and not value.requires_grad for key, value in expected.items())


@pytest.mark.parametrize("chunk_size", [-1, 1, 3])
def test_selected_log_probs_and_gradients_match_dense(chunk_size: int) -> None:
    # [FUNCTION DOC] Test (chunk size -1 meaning unchunked, 1, 3): float64 logits of width 9 but
    # vocabulary size 7 (padded columns), ids of shape 5 by 4 including -1 padding ids, temperature
    # 0.7. Result equals the dense reference (logits cut to the vocabulary, divided by temperature,
    # log-softmax, gathered at clamped ids, padding ids set to zero); the gradient of a random
    # weighted sum equals the reference gradient; gradients on the padded vocabulary columns are
    # exactly zero.
    logits = torch.randn(5, 9, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 3, -1, 0], [1, 2, 6, 2], [3, 3, 3, 3], [6, -1, 1, 2], [5, 2, 4, 1]])
    weights = torch.randn(5, 4, dtype=torch.float64)
    actual = selected_log_probs(logits, ids, vocab_size=7, temperature=0.7, chunk_size=chunk_size)
    reference = (logits[:, :7] / 0.7).log_softmax(-1).gather(-1, ids.clamp_min(0)).masked_fill(ids < 0, 0)
    torch.testing.assert_close(actual, reference)
    actual_grad = torch.autograd.grad((actual * weights).sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad((reference * weights).sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert (actual_grad[:, 7:] == 0).all()


def test_entropy_uses_same_unpadded_distribution_and_gradient() -> None:
    # [FUNCTION DOC] Test: with vocabulary size 7 of width-9 logits, temperature 0.7 and entropy
    # requested, the entropy equals that of the dense unpadded tempered distribution, and the
    # gradient of (sum of selected log-probs minus 0.2 times summed entropy) equals the dense
    # reference gradient.
    logits = torch.randn(3, 9, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 2], [1, 4], [3, 6]])
    selected, entropy = selected_log_probs_and_entropy(logits, ids, vocab_size=7, temperature=0.7, with_entropy=True)
    logp = (logits[:, :7] / 0.7).log_softmax(-1)
    expected_entropy = -(logp.exp() * logp).sum(-1)
    torch.testing.assert_close(entropy, expected_entropy)
    actual_grad = torch.autograd.grad(selected.sum() - 0.2 * entropy.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(logp.gather(-1, ids).sum() - 0.2 * expected_entropy.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("vocab_size", [0, -1])
def test_selected_log_probs_rejects_invalid_vocabulary_size(vocab_size: int) -> None:
    # [FUNCTION DOC] Test (vocabulary size 0 or -1): raises a value error mentioning a valid
    # vocabulary size.
    logits = torch.randn(2, 4)
    token_ids = torch.tensor([[0], [1]])
    with pytest.raises(ValueError, match="valid vocabulary size"):
        selected_log_probs(logits, token_ids, vocab_size=vocab_size)


def test_selected_log_probs_gradcheck_and_empty_response() -> None:
    # [FUNCTION DOC] Test: numeric gradient check passes on float64 logits (2 by 4) with ids
    # including a padding id; with zero rows the result has shape (0, 3) and backward leaves a
    # zero gradient on the original logits.
    logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 2, -1], [1, 3, 1]])
    assert torch.autograd.gradcheck(lambda x: selected_log_probs(x, ids), (logits,))
    empty = logits[:0]
    result = selected_log_probs(empty, ids[:0])
    assert result.shape == (0, 3)
    result.sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
