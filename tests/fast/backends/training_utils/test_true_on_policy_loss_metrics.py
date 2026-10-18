# [MODULE DOC] CPU tests of the policy loss function's metrics and gradient flow with its helper
# functions replaced by fakes (tensor-parallel state, local response masks, effective-sample-size
# ratio, log-prob/entropy extraction, policy-loss kernel): the train-rollout log-prob absolute
# difference uses the trainer's scores whichever baseline is used; gradients flow only through the
# current policy logits (never through stored old/rollout/reference/advantage inputs); a custom
# correction function can run without trainer-scored log-probs; and non-finite values that are zero
# weighted or masked (entropy NaN, KL NaN with zero coefficient, masked NaN log-probs) do not poison
# the loss or metrics.
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.loss.hub import losses as loss_utils


def _make_args(*, use_rollout_logprobs: bool) -> Namespace:
    # [FUNCTION DOC] HELPER: Returns policy-loss args for a ppo estimator with the given
    # rollout-log-probs flag and everything else off or neutral: no support replay, no skipping of
    # actor forward, no opsm, no mismatch metrics, no tis, clip epsilons 0.2/0.2, no custom paths,
    # per-sample loss, packed format, zero entropy and KL coefficients, k1 KL, temperature 1,
    # unchunked, not true-on-policy, no allgather, no entropy observation.
    return Namespace(
        use_rollout_logprobs=use_rollout_logprobs,
        use_sampling_support_replay=False,
        skip_actor_forward_only=False,
        use_opsm=False,
        advantage_estimator="ppo",
        get_mismatch_metrics=False,
        use_tis=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        calculate_per_token_loss=False,
        qkv_format="thd",
        entropy_coef=0.0,
        use_kl_loss=False,
        use_unbiased_kl=False,
        kl_loss_type="k1",
        kl_loss_coef=0.0,
        rollout_temperature=1.0,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        allgather_cp=False,
        observe_training_entropy=False,
    )


def _make_batch(*, old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor) -> dict:
    # [FUNCTION DOC] HELPER: Builds a one-sample batch: advantages [1.0, -0.5], the given old-policy
    # and rollout log-probs, tokens [7,8,9], response length 2, total length 3, all-ones loss mask.
    return {
        "advantages": [torch.tensor([1.0, -0.5], dtype=torch.float32)],
        "log_probs": [old_log_probs],
        "rollout_log_probs": [rollout_log_probs],
        "unconcat_tokens": [torch.tensor([7, 8, 9], dtype=torch.long)],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 1.0], dtype=torch.float32)],
    }


def _patch_single_rank_loss_helpers(monkeypatch):
    # [FUNCTION DOC] HELPER: Patches the loss module so that the local response loss mask getter
    # returns the masks as given and the effective-sample-size contribution returns a tensor of 1.0
    # shaped like a scalar of the same type as the KL.
    monkeypatch.setattr(
        loss_utils,
        "get_local_response_loss_masks",
        lambda total_lengths, response_lengths, loss_masks, qkv_format="thd", max_seq_lens=None: loss_masks,
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_ess_ratio_contribution",
        lambda *, ppo_kl, **kwargs: ppo_kl.new_tensor(1.0),
    )


@pytest.mark.parametrize(
    ("use_rollout_logprobs", "train_log_probs", "old_log_probs", "rollout_log_probs", "expected_abs_diff"),
    [
        (
            False,
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.45,
        ),
        (
            True,
            torch.tensor([0.50, 1.00], dtype=torch.float32),
            torch.tensor([0.10, 0.20], dtype=torch.float32),
            torch.tensor([0.40, 0.80], dtype=torch.float32),
            0.45,
        ),
    ],
)
def test_train_rollout_logprob_abs_diff_uses_trainer_scores(
    monkeypatch,
    use_rollout_logprobs: bool,
    train_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    expected_abs_diff: float,
):
    # [FUNCTION DOC] Test (parametrized: old-policy baseline with trainer scores equal to rollout
    # [0.4,0.8] and old [0.1,0.2]; rollout baseline with trainer scores [0.5,1.0] and rollout
    # [0.4,0.8]): with tensor-parallel state, mask helpers, log-prob extraction (returns the trainer
    # scores and zero entropy) and the policy-loss kernel (zeros) patched, the policy loss is finite
    # and the train-rollout log-prob absolute difference metric equals 0.45 -- the mean absolute
    # difference between the trainer scores and the rollout log-probs, independent of which baseline
    # the loss uses.
    args = _make_args(use_rollout_logprobs=use_rollout_logprobs)
    batch = _make_batch(old_log_probs=old_log_probs, rollout_log_probs=rollout_log_probs)

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [train_log_probs.clone()],
            "entropy": [torch.zeros_like(train_log_probs)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], torch.tensor(expected_abs_diff))


@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_policy_loss_only_backpropagates_through_current_policy(monkeypatch, use_rollout_logprobs: bool):
    # [FUNCTION DOC] Test (parametrized over the rollout-baseline flag): old-policy log-probs,
    # rollout log-probs and advantages are computed from leaf tensors requiring grad (the old
    # entry is removed in rollout-baseline mode); the fake extraction returns the sine of the first
    # two logits. After the loss backward, the current logits have a nonzero gradient while the
    # three source leaves have none.
    args = _make_args(use_rollout_logprobs=use_rollout_logprobs)
    old_source = torch.tensor([0.10, 0.20], requires_grad=True)
    rollout_source = torch.tensor([0.12, 0.22], requires_grad=True)
    advantage_source = torch.tensor([1.0, 2.0], requires_grad=True)
    current_logits = torch.tensor([[[0.15], [0.25], [0.0]]], requires_grad=True)
    batch = _make_batch(
        old_log_probs=old_source.sin(),
        rollout_log_probs=rollout_source.sin(),
    )
    if use_rollout_logprobs:
        batch.pop("log_probs")
    batch["advantages"] = [advantage_source.square()]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)

    def fake_get_log_probs_and_entropy(logits, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake extraction; returns log-probs equal to the sine of the first
        # two flattened logits (keeps the graph to the logits).
        return {"log_probs": [logits.flatten()[:2].sin()]}

    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        fake_get_log_probs_and_entropy,
    )

    loss, _ = loss_utils.policy_loss_function(
        args,
        batch,
        logits=current_logits,
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    assert current_logits.grad is not None
    assert torch.count_nonzero(current_logits.grad) > 0
    assert old_source.grad is None
    assert rollout_source.grad is None
    assert advantage_source.grad is None


def test_kl_loss_does_not_backpropagate_through_reference_scores(monkeypatch):
    # [FUNCTION DOC] Test: KL loss on with coefficient 0.5 and reference log-probs computed from a
    # leaf tensor requiring grad; after the loss backward the current logits have a nonzero
    # gradient and the reference leaf has none.
    args = _make_args(use_rollout_logprobs=False)
    args.use_kl_loss = True
    args.kl_loss_coef = 0.5
    reference_source = torch.tensor([0.30, 0.40], requires_grad=True)
    current_logits = torch.tensor([[[0.15], [0.25], [0.0]]], requires_grad=True)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20]),
        rollout_log_probs=torch.tensor([0.12, 0.22]),
    )
    batch["ref_log_probs"] = [reference_source.sin()]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda logits, *args, **kwargs: {"log_probs": [logits.flatten()[:2].sin()]},
    )

    loss, _ = loss_utils.policy_loss_function(
        args,
        batch,
        logits=current_logits,
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    assert current_logits.grad is not None
    assert torch.count_nonzero(current_logits.grad) > 0
    assert reference_source.grad is None


def test_custom_tis_can_ignore_missing_trainer_scored_log_probs(monkeypatch):
    # [FUNCTION DOC] Test: rollout-baseline mode with tis on, a custom correction function path,
    # no stored old-policy log-probs; the function loader is patched to return a fake that asserts
    # the train log-probs argument is none and no rollout log-probs require grad, and returns the
    # pg loss, masks and empty metrics. The sample-mean reducer getter is patched too. The loss is
    # computed and backpropagated, giving the logits a nonzero gradient.
    args = _make_args(use_rollout_logprobs=True)
    args.use_tis = True
    args.custom_tis_function_path = "tests.custom_tis"
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20]),
        rollout_log_probs=torch.tensor([0.12, 0.22]),
    )
    batch.pop("log_probs")

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda logits, *args, **kwargs: {"log_probs": [logits.flatten()[:2].sin()]},
    )

    def custom_tis(**kwargs):
        # [FUNCTION DOC] HELPER: Fake correction function; asserts train log-probs is none and no
        # rollout log-prob tensor requires grad; returns (pg loss, loss masks, empty metrics).
        assert kwargs["train_log_probs"] is None
        assert all(not tensor.requires_grad for tensor in kwargs["rollout_log_probs"])
        return kwargs["pg_loss"], kwargs["loss_masks"], {}

    monkeypatch.setattr(loss_utils, "load_function", lambda path: custom_tis)
    monkeypatch.setattr(
        loss_utils,
        "get_sum_of_sample_mean",
        lambda *args, **kwargs: lambda tensor: tensor.float().mean(),
    )

    logits = torch.tensor([[[0.15], [0.25], [0.0]]], requires_grad=True)
    loss, _ = loss_utils.policy_loss_function(
        args,
        batch,
        logits=logits,
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) > 0


def test_zero_weighted_entropy_nan_does_not_poison_policy_loss(monkeypatch):
    # [FUNCTION DOC] Test: with entropy coefficient 0 the fake extraction is asserted to be called
    # with entropy disabled (so entropy NaNs cannot arise); the policy-loss kernel is zeros. The
    # loss is finite and the entropy loss metric is exactly 0.
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)

    def fake_get_log_probs_and_entropy(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Fake extraction; asserts the entropy flag is false and returns
        # fixed log-probs [0.10, 0.20] with no entropy.
        assert kwargs["with_entropy"] is False
        return {"log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)]}

    monkeypatch.setattr(loss_utils, "get_log_probs_and_entropy", fake_get_log_probs_and_entropy)
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    torch.testing.assert_close(metrics["entropy_loss"], torch.tensor(0.0))


def test_zero_weighted_kl_nan_does_not_poison_policy_loss(monkeypatch):
    # [FUNCTION DOC] Test: KL loss on with coefficient 0 and NaN reference log-probs; the loss is
    # finite and the logged KL loss metric is finite.
    args = _make_args(use_rollout_logprobs=False)
    args.use_kl_loss = True
    args.kl_loss_coef = 0.0
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, 0.20], dtype=torch.float32),
    )
    batch["ref_log_probs"] = [torch.tensor([float("nan"), float("nan")], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, 0.20], dtype=torch.float32)],
        },
    )
    monkeypatch.setattr(
        loss_utils,
        "compute_policy_loss",
        lambda ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None: (
            torch.zeros_like(ppo_kl),
            torch.zeros_like(ppo_kl),
        ),
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: tensor.float().mean(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["kl_loss"])


def test_masked_nonfinite_ppo_terms_do_not_poison_policy_loss(monkeypatch):
    # [FUNCTION DOC] Test: the second token has NaN old/rollout/trainer log-probs but loss mask 0;
    # with a masked-sum reducer, the loss and the PPO KL metric are finite.
    args = _make_args(use_rollout_logprobs=False)
    batch = _make_batch(
        old_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
        rollout_log_probs=torch.tensor([0.10, float("nan")], dtype=torch.float32),
    )
    batch["loss_masks"] = [torch.tensor([1.0, 0.0], dtype=torch.float32)]

    monkeypatch.setattr(
        loss_utils,
        "get_parallel_state",
        lambda: SimpleNamespace(tp=SimpleNamespace(group=None)),
    )
    _patch_single_rank_loss_helpers(monkeypatch)
    monkeypatch.setattr(
        loss_utils,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [torch.tensor([0.10, float("nan")], dtype=torch.float32)],
        },
    )

    loss, metrics = loss_utils.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 8), dtype=torch.float32),
        sum_of_sample_mean=lambda tensor: (tensor * batch["loss_masks"][0]).sum(),
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["ppo_kl"])
