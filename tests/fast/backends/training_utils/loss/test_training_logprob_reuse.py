# [MODULE DOC] CPU tests for reusing training log-probs in the policy loss when the actor
# forward-only pass is skipped: the loss, gradient and metrics with reuse must be bit-identical to
# an explicit detached baseline (grpo, gspo and grpo with importance sampling; the gspo case also
# checks per-sample gathers); with rollout log-probs used as the old policy the metrics still
# reflect a nonzero old/new gap; train-rollout metrics use trainer scores independent of the loss
# baseline; missing old-policy log-probs are rejected with specific errors; advantage computation
# synthesizes zero-KL reference scores only on the last pipeline stage and detaches fixed scores;
# the dispatcher passes only the args to the loss function (also not applying the skip to the value
# loss); and mismatch-metric correction functions receive detached actor log-probs. A module-scoped
# fixture ensures a one-rank Gloo process group.
from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss import objective as loss_module
from miles.backends.training_utils.loss.hub import losses as losses_module
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.loss.hub.losses import policy_loss_function
from miles.backends.training_utils.loss.objective import compute_advantages_and_returns
from miles.utils.ft_utils.process_group_utils import GroupInfo

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state, make_rollout_data


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    # [FUNCTION DOC] HELPER: Module-scoped fixture; if a process group already exists, does nothing;
    # otherwise initializes a one-rank Gloo group with a file rendezvous in a temp dir and destroys
    # it afterwards (also on failure).
    if dist.is_initialized():
        yield
        return

    rendezvous = tmp_path_factory.mktemp("training-logprob-reuse") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _run_policy_loss(args, batch, inputs, *, skip_actor_forward_only):
    # [FUNCTION DOC] HELPER: Sets the skip-actor-forward-only flag on args (mutation), clones the
    # policy logits with gradient enabled, builds the sum-of-sample-mean reducer from the batch,
    # runs the policy loss, backpropagates. Returns (detached loss, metrics, clone of logits grad).
    args.skip_actor_forward_only = skip_actor_forward_only
    logits = deep_clone(inputs["policy_logits"])
    logits.requires_grad_(True)
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens"),
    )
    loss, metrics = policy_loss_function(
        args,
        batch,
        logits,
        reducer,
    )
    loss.backward()
    return loss.detach(), metrics, logits.grad.clone()


@pytest.mark.parametrize(
    ("advantage_estimator", "use_tis"),
    [
        ("grpo", False),
        ("gspo", False),
        ("grpo", True),
    ],
)
def test_reused_training_log_probs_match_an_explicit_detached_baseline(
    process_group,
    advantage_estimator,
    use_tis,
    monkeypatch,
):
    # [FUNCTION DOC] Test (parametrized over grpo, gspo, grpo with importance sampling): computes the
    # policy loss with the batch's old-policy log-probs removed and the skip flag on (log-probs are
    # reused from the training forward), and again with explicit detached log-probs computed
    # separately and the skip flag off. Asserts the losses and logits gradients are exactly equal,
    # the gradient is nonzero, metrics have the same keys and exactly equal values, and ppo_kl and
    # clip fraction are zero. For gspo, the per-sample gather function is spied and must be called
    # once per sample in the reuse pass.
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        advantage_estimator=advantage_estimator,
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_tis=use_tis,
    )
    inputs = make_inputs(
        seed=1234,
        batch_size=3,
        prompt_lens=[20, 64, 40],
        response_lens=[10, 48, 32],
        vocab_size=128,
        args=args,
    )

    gather_spy = Mock(wraps=losses_module.all_gather_with_cp)
    if advantage_estimator == "gspo":
        monkeypatch.setattr(losses_module, "all_gather_with_cp", gather_spy)

    reuse_batch = make_batch(inputs, "policy_loss")
    del reuse_batch["log_probs"]
    reuse_loss, reuse_metrics, reuse_grad = _run_policy_loss(
        args,
        reuse_batch,
        inputs,
        skip_actor_forward_only=True,
    )
    if advantage_estimator == "gspo":
        assert gather_spy.call_count == len(inputs["response_lens"])

    baseline_batch = make_batch(inputs, "policy_loss")
    baseline_batch["log_probs"] = [
        log_prob.detach()
        for log_prob in get_log_probs_and_entropy(
            deep_clone(inputs["policy_logits"]),
            args=args,
            unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
            total_lengths=list(inputs["total_lens"]),
            response_lengths=list(inputs["response_lens"]),
            with_entropy=False,
        )["log_probs"]
    ]
    baseline_loss, baseline_metrics, baseline_grad = _run_policy_loss(
        args,
        baseline_batch,
        inputs,
        skip_actor_forward_only=False,
    )

    assert torch.equal(reuse_loss, baseline_loss)
    assert torch.equal(reuse_grad, baseline_grad)
    assert torch.count_nonzero(reuse_grad).item() > 0
    assert reuse_metrics.keys() == baseline_metrics.keys()
    for key in reuse_metrics:
        assert torch.equal(reuse_metrics[key], baseline_metrics[key]), key
    assert reuse_metrics["ppo_kl"].item() == 0.0
    assert reuse_metrics["pg_clipfrac"].item() == 0.0


def test_skip_actor_forward_only_preserves_rollout_log_probs_as_old_policy(process_group):
    # [FUNCTION DOC] Test: with rollout log-probs used as the old policy and set to the trainer's
    # log-probs plus 0.25, the loss with the skip flag on and off is exactly equal in loss, gradient
    # and all metrics, while ppo_kl, train-rollout log-prob absolute difference and train-rollout
    # KL are all positive (the old-policy gap is preserved, not zeroed).
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_rollout_logprobs=True,
    )
    inputs = make_inputs(seed=17, batch_size=2, prompt_lens=[4, 5], response_lens=[3, 4], vocab_size=16, args=args)
    batch = make_batch(inputs, "policy_loss")
    training_log_probs = get_log_probs_and_entropy(
        deep_clone(inputs["policy_logits"]),
        args=args,
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        total_lengths=list(inputs["total_lens"]),
        response_lengths=list(inputs["response_lens"]),
        with_entropy=False,
    )["log_probs"]
    batch["rollout_log_probs"] = [log_prob.detach() + 0.25 for log_prob in training_log_probs]
    del batch["log_probs"]

    skip_loss, skip_metrics, skip_grad = _run_policy_loss(
        args,
        batch,
        inputs,
        skip_actor_forward_only=True,
    )
    baseline_loss, baseline_metrics, baseline_grad = _run_policy_loss(
        args,
        batch,
        inputs,
        skip_actor_forward_only=False,
    )

    assert torch.equal(skip_loss, baseline_loss)
    assert torch.equal(skip_grad, baseline_grad)
    assert skip_metrics.keys() == baseline_metrics.keys()
    for key in skip_metrics:
        assert torch.equal(skip_metrics[key], baseline_metrics[key]), key
    assert skip_metrics["ppo_kl"].item() > 0
    assert skip_metrics["train_rollout_logprob_abs_diff"].item() > 0
    assert skip_metrics["train_rollout_kl"].item() > 0


@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_train_rollout_metrics_use_trainer_scores_independent_of_loss_baseline(process_group, use_rollout_logprobs):
    # [FUNCTION DOC] Test (parametrized over the rollout-log-probs-as-baseline flag): batch with
    # trainer log-probs and rollout log-probs equal to them plus 0.25. The reported train-rollout
    # absolute difference is approximately 0.25 and train-rollout KL is positive in both settings.
    make_parallel_state()
    args = make_args(
        entropy_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_rollout_logprobs=use_rollout_logprobs,
    )
    inputs = make_inputs(seed=23, batch_size=1, prompt_lens=[4], response_lens=[3], vocab_size=16, args=args)
    batch = make_batch(inputs, "policy_loss")
    trainer_scores = get_log_probs_and_entropy(
        deep_clone(inputs["policy_logits"]),
        args=args,
        unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
        total_lengths=list(inputs["total_lens"]),
        response_lengths=list(inputs["response_lens"]),
        with_entropy=False,
    )["log_probs"]
    batch["log_probs"] = [scores.detach() for scores in trainer_scores]
    batch["rollout_log_probs"] = [scores.detach() + 0.25 for scores in trainer_scores]

    _, metrics, _ = _run_policy_loss(args, batch, inputs, skip_actor_forward_only=False)

    assert metrics["train_rollout_logprob_abs_diff"].item() == pytest.approx(0.25)
    assert metrics["train_rollout_kl"].item() > 0


@pytest.mark.parametrize(
    ("skip_actor_forward_only", "use_rollout_logprobs", "remove_old"),
    [
        (False, False, True),
        (True, True, True),
    ],
)
def test_policy_loss_rejects_missing_old_policy_log_probs(
    skip_actor_forward_only,
    use_rollout_logprobs,
    remove_old,
):
    # [FUNCTION DOC] Test (parametrized: normal estimator without skip, and skip with rollout
    # baseline; in both the old-policy entry is removed): the policy loss raises a value error
    # mentioning old-policy log-probs when the baseline is trainer log-probs, and an assertion
    # error saying rollout log-probs must be provided when the rollout baseline is used.
    # KNOWN ISSUE: the remove flag is true in every parametrization, so the no-removal branch is
    # never exercised.
    make_parallel_state()
    args = make_args(
        entropy_coef=0.0,
        observe_training_entropy=False,
        skip_actor_forward_only=skip_actor_forward_only,
        true_on_policy_mode=False,
        use_rollout_logprobs=use_rollout_logprobs,
    )
    inputs = make_inputs(seed=7, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "policy_loss")
    old_key = "rollout_log_probs" if use_rollout_logprobs else "log_probs"
    if remove_old:
        del batch[old_key]
    reducer = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
    )

    expected_error = AssertionError if use_rollout_logprobs else ValueError
    expected_message = "rollout_log_probs must be provided" if use_rollout_logprobs else "old-policy log-probs"
    with pytest.raises(expected_error, match=expected_message):
        policy_loss_function(
            args,
            batch,
            inputs["policy_logits"],
            reducer,
        )


@pytest.mark.parametrize("is_pp_last_stage", [False, True])
def test_skip_actor_forward_only_synthesizes_zero_kl_only_on_the_last_pipeline_stage(is_pp_last_stage):
    # [FUNCTION DOC] Test (parametrized over last-stage or not): with the skip flag on and KL
    # coefficient 0, rollout data lacking log-probs, reference log-probs and values. On a non-last
    # stage no advantages or returns are produced. On the last stage, advantages and returns equal
    # exactly those from a baseline run with the skip flag off on rollout data that still has
    # log-probs and reference log-probs (only values removed).
    parallel_state = make_parallel_state()
    parallel_state.is_pp_last_stage = is_pp_last_stage
    args = make_args(kl_coef=0.0, skip_actor_forward_only=True, true_on_policy_mode=False)
    inputs = make_inputs(seed=11, batch_size=2, prompt_lens=[4, 6], response_lens=[3, 5], vocab_size=16, args=args)
    reused = make_rollout_data(inputs)
    for key in ("log_probs", "ref_log_probs", "values"):
        del reused[key]

    compute_advantages_and_returns(args, reused)

    if not is_pp_last_stage:
        assert "advantages" not in reused
        assert "returns" not in reused
        return

    baseline = make_rollout_data(inputs)
    del baseline["values"]
    args.skip_actor_forward_only = False
    compute_advantages_and_returns(args, baseline)
    for key in ("advantages", "returns"):
        for actual, expected in zip(reused[key], baseline[key], strict=True):
            assert torch.equal(actual, expected)


def test_skip_actor_forward_only_detaches_fixed_scores_without_precomputed_log_probs():
    # [FUNCTION DOC] Test: last stage, skip flag on, rollout data without log-probs or values and
    # with reference and teacher log-probs computed from leaf tensors requiring grad. After the
    # advantage computation both entries have no gradient history and do not require grad.
    parallel_state = make_parallel_state()
    parallel_state.is_pp_last_stage = True
    args = make_args(kl_coef=0.0, skip_actor_forward_only=True, true_on_policy_mode=False)
    inputs = make_inputs(seed=17, batch_size=1, prompt_lens=[4], response_lens=[3], vocab_size=16, args=args)
    rollout_data = make_rollout_data(inputs)
    reference_source = torch.tensor([0.2, 0.4, 0.6], requires_grad=True)
    teacher_source = torch.tensor([0.1, 0.3, 0.5], requires_grad=True)
    rollout_data["ref_log_probs"] = [reference_source.sin()]
    rollout_data["teacher_log_probs"] = [teacher_source.cos()]
    del rollout_data["log_probs"]
    del rollout_data["values"]

    compute_advantages_and_returns(args, rollout_data)

    for key in ("ref_log_probs", "teacher_log_probs"):
        assert rollout_data[key][0].grad_fn is None
        assert rollout_data[key][0].requires_grad is False


@pytest.mark.parametrize("skip_actor_forward_only", [False, True])
def test_loss_dispatcher_uses_args_without_extra_policy_kwargs(monkeypatch, skip_actor_forward_only):
    # [FUNCTION DOC] Test (parametrized over the skip flag): the loss-function getter is replaced
    # to return a mock policy loss. After calling the dispatcher, the mock was called with the args
    # object as the first positional argument and no keyword arguments.
    make_parallel_state()
    args = make_args(
        observe_training_entropy=False,
        skip_actor_forward_only=skip_actor_forward_only,
        true_on_policy_mode=False,
    )
    inputs = make_inputs(seed=19, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "policy_loss")
    logits = deep_clone(inputs["policy_logits"])
    policy_loss = Mock(return_value=(logits.sum(), {"loss": logits.new_zeros(())}))
    monkeypatch.setattr(loss_module, "get_loss_function", lambda _args, _loss_fn=None: policy_loss)

    loss_module.loss_function(
        args,
        batch,
        1,
        logits,
    )

    assert policy_loss.call_args.args[0] is args
    assert policy_loss.call_args.kwargs == {}


def test_loss_dispatcher_does_not_apply_actor_skip_to_value_loss(monkeypatch):
    # [FUNCTION DOC] Test: with the value loss type and the skip flag on, the dispatcher calls the
    # (mocked) value loss with the args object as first positional argument and no keyword
    # arguments.
    make_parallel_state()
    args = make_args(loss_type="value_loss", skip_actor_forward_only=True)
    inputs = make_inputs(seed=23, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "value_loss")
    logits = deep_clone(inputs["value_logits"])
    value_loss = Mock(return_value=(logits.sum(), {"loss": logits.new_zeros(())}))
    monkeypatch.setattr(loss_module, "get_loss_function", lambda _args, _loss_fn=None: value_loss)

    loss_module.loss_function(args, batch, 1, logits)

    assert value_loss.call_args.args[0] is args
    assert value_loss.call_args.kwargs == {}


@pytest.mark.parametrize("skip_actor_forward_only", [False, True])
def test_mismatch_metrics_keep_actor_log_probs_as_training_source(process_group, monkeypatch, skip_actor_forward_only):
    # [FUNCTION DOC] Test (parametrized over the skip flag): with a custom correction function path
    # and mismatch metrics on, the function loader is replaced by a fake correction function. The
    # fake asserts the train log-probs it receives exist, do not require grad, and equal the
    # expected ones (the batch's own when not skipping; freshly computed from the logits when
    # skipping and the batch's were removed). It returns the pg loss and loss masks unchanged with
    # empty metrics. The test just runs the policy loss under each setting.
    parallel_state = make_parallel_state()
    parallel_state.tp = GroupInfo(rank=0, size=1, group=dist.group.WORLD)
    args = make_args(
        custom_tis_function_path="tests.fake_tis",
        entropy_coef=0.0,
        get_mismatch_metrics=True,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        use_rollout_logprobs=True,
    )
    inputs = make_inputs(seed=29, batch_size=1, prompt_lens=[3], response_lens=[2], vocab_size=8, args=args)
    batch = make_batch(inputs, "policy_loss")
    if skip_actor_forward_only:
        del batch["log_probs"]
        expected_train_log_probs = get_log_probs_and_entropy(
            deep_clone(inputs["policy_logits"]),
            args=args,
            unconcat_tokens=deep_clone(inputs["unconcat_tokens"]),
            total_lengths=list(inputs["total_lens"]),
            response_lengths=list(inputs["response_lens"]),
            with_entropy=False,
        )["log_probs"]
    else:
        expected_train_log_probs = batch["log_probs"]

    def fake_tis_function(**kwargs):
        # [FUNCTION DOC] HELPER: Fake correction function; asserts the train log-probs keyword is
        # present, none require grad and each equals the expected train log-probs exactly. Returns
        # the pg loss, loss masks and an empty metrics dictionary.
        assert kwargs["train_log_probs"] is not None
        assert all(not log_prob.requires_grad for log_prob in kwargs["train_log_probs"])
        for actual, expected in zip(kwargs["train_log_probs"], expected_train_log_probs, strict=True):
            assert torch.equal(actual, expected)
        return kwargs["pg_loss"], kwargs["loss_masks"], {}

    monkeypatch.setattr(losses_module, "load_function", lambda _path: fake_tis_function)

    _run_policy_loss(args, batch, inputs, skip_actor_forward_only=skip_actor_forward_only)
