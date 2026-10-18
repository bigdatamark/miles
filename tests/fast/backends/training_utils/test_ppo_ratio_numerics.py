# [MODULE DOC] CPU numerics tests for the PPO policy loss and approximate KL helpers, using their
# eager (undecorated, un-compiled) variants: extreme or non-finite log ratios with zero advantages
# stay finite (policy loss and clip fraction; low-variance KL), and for normal log ratios the
# results equal the straightforward unclamped formulas.
import torch

from miles.backends.training_utils.loss.hub.math_utils import compute_approx_kl, compute_policy_loss

compute_approx_kl_eager = compute_approx_kl.__wrapped__
compute_policy_loss_eager = compute_policy_loss.__wrapped__


def test_policy_loss_extreme_log_ratios_with_zero_advantages_stay_finite():
    # [FUNCTION DOC] Test: log ratios -1000, 1000, NaN, +inf, -inf with zero advantages and clip
    # epsilons 0.2/0.2 give all-finite policy losses and clip fractions, with losses equal to zero.
    ppo_kl = torch.tensor([-1000.0, 1000.0, float("nan"), float("inf"), float("-inf")])
    advantages = torch.zeros_like(ppo_kl)

    pg_losses, clipfrac = compute_policy_loss_eager(ppo_kl, advantages, eps_clip=0.2, eps_clip_high=0.2)

    assert torch.isfinite(pg_losses).all().item()
    assert torch.isfinite(clipfrac).all().item()
    torch.testing.assert_close(pg_losses, torch.zeros_like(pg_losses))


def test_policy_loss_matches_unclamped_ratio_for_normal_log_ratios():
    # [FUNCTION DOC] Test: for log ratios [-0.1, 0, 0.1] and advantages [1, -2, 0.5], the loss equals
    # the maximum of -ratio*advantage and -clamped_ratio*advantage with ratio = exp(-log ratio) and
    # clamp bounds 1-0.2 and 1+0.2, and the clip fraction equals 1 where the clamped term is
    # strictly greater, else 0.
    ppo_kl = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32)
    advantages = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32)
    eps_clip = 0.2
    eps_clip_high = 0.2

    pg_losses, clipfrac = compute_policy_loss_eager(
        ppo_kl,
        advantages,
        eps_clip=eps_clip,
        eps_clip_high=eps_clip_high,
    )

    ratio = (-ppo_kl).exp()
    expected_losses1 = -ratio * advantages
    expected_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    expected_losses = torch.maximum(expected_losses1, expected_losses2)
    expected_clipfrac = torch.gt(expected_losses2, expected_losses1).float()

    torch.testing.assert_close(pg_losses, expected_losses)
    torch.testing.assert_close(clipfrac, expected_clipfrac)


def test_low_var_kl_extreme_log_ratios_stay_finite():
    # [FUNCTION DOC] Test: log-probs -1000, 1000, NaN, +inf, -inf against zero base log-probs give an
    # all-finite low-variance KL.
    log_probs = torch.tensor([-1000.0, 1000.0, float("nan"), float("inf"), float("-inf")])
    log_probs_base = torch.zeros_like(log_probs)

    kl = compute_approx_kl_eager(log_probs, log_probs_base, kl_loss_type="low_var_kl")

    assert torch.isfinite(kl).all().item()


def test_low_var_kl_matches_unclamped_formula_for_normal_log_ratios():
    # [FUNCTION DOC] Test: for log-probs [-0.1, 0, 0.1] against zero base, low-variance KL equals
    # exp(r) - 1 - r where r is the negated log-prob difference.
    log_probs = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float32)
    log_probs_base = torch.zeros_like(log_probs)

    kl = compute_approx_kl_eager(log_probs, log_probs_base, kl_loss_type="low_var_kl")

    log_ratio = -(log_probs - log_probs_base)
    expected_kl = log_ratio.exp() - 1 - log_ratio
    torch.testing.assert_close(kl, expected_kl)
