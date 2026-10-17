# [MODULE DOC] The loss functions used in training and the dispatcher that picks one. Every loss
# function has the same signature (arguments, mini-batch dictionary, logits, a context-parallel-
# aware sample-mean reducer) and returns (scalar loss with gradient, dictionary of detached metric
# scalars). Provided: policy loss (PPO-style clipped, with GSPO sequence-level KL, the optional
# OPSM mask, off-policy importance sampling correction, entropy and KL terms,
# sampling-support replay and many diagnostics), clipped value loss, and supervised negative
# log-likelihood; score-centering and user-defined losses are dispatched too. Loss functions pull
# response-aligned log-probs or values from the logit processors.
from argparse import Namespace
from collections.abc import Callable
from typing import Protocol

import torch

from miles.backends.training_utils.data.context_parallel import (
    all_gather_with_cp,
    get_local_response_loss_masks,
    get_sum_of_sample_mean,
)
from miles.backends.training_utils.data.sampling_mask import get_rollout_sampling_masks
from miles.backends.training_utils.loss.hub.corrections import vanilla_tis_function
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy, get_values
from miles.backends.training_utils.loss.hub.math_utils import (
    compute_approx_kl,
    compute_ess_ratio_contribution,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
)
from miles.backends.training_utils.loss.hub.score_centering_loss import score_centering_loss_function
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.function_registry import load_function
from miles.utils.types import RolloutBatch


class LossFunction(Protocol):
    # [CLASS DOC] Structural type (protocol) describing what every loss function must accept and
    # return; it has no behavior of its own and exists for type checking and documentation.
    """Common signature of the per-loss-type functions dispatched by `get_loss_function`."""

    def __call__(
        self,
        args: Namespace,
        batch: RolloutBatch,
        logits: torch.Tensor,
        sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # [FUNCTION DOC] Protocol method with no implementation (the body is only documentation):
        # takes arguments, the mini-batch dictionary, float32 logits and the sample-mean reducer,
        # and returns (scalar loss with gradient, dictionary of detached scalar metrics).
        """Compute a scalar loss with gradient + a dict of detached scalar metrics.

        Args:
            args: Configuration. Each loss type reads its own subset of flags
                (PPO eps, KL coefficients, value clip, TIS settings, OPSM, etc.).
            batch: Mini-batch (`miles.utils.types.RolloutBatch`). Loss-type-
                dependent keys; common ones are `unconcat_tokens`, `response_lengths`,
                `total_lengths`, `loss_masks`, optional `max_seq_lens` (required for
                qkv_format="bshd"). Per-implementation docstrings list extras.
            logits: Float32. Last dim is vocab_size (policy/sft) or 1 (value).
                Outer shape is `[1, T, ...]` for qkv_format="thd" (T = sum of
                total_lengths) or `[B, max_seq_len, ...]` for "bshd".
            sum_of_sample_mean: CP-aware reducer; takes a flat per-token tensor
                and returns a scalar, sample-mean-weighted by `loss_masks`.

        Returns:
            `(loss, metrics)`:
              * `loss`: scalar tensor with grad, un-rescaled (the dispatcher
                applies Megatron scaling on top).
              * `metrics`: dict of detached 0-d scalars; surfaced under `train/`
                in the training log / wandb. Keys per loss type are documented
                on each implementation.
        """
        ...


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Policy-gradient loss with many optional terms; returns (loss, metrics).
    # Inputs from the batch: advantages (detached), optional rollout log-probs and reference
    # log-probs (detached), optional trainer-scored old log-probs, lengths, loss masks, unconcat
    # tokens, optional max sequence lengths, optional on-policy-distillation reverse KL, rollout
    # mask sums, and sampling-mask fields.
    # Validation: with the use-rollout-logprobs flag, rollout log-probs are required (assertion);
    # else if the actor forward-only pass is not skipped, old log-probs are required (value error).
    # Steps: (1) current log-probs (and entropy if the entropy coefficient is nonzero or entropy
    # observation is on) from the logits, using sampling masks when sampling-support replay is on.
    # (2) Old log-probs: the detached current ones if actor forward-only is skipped, else the batch's
    # trainer-scored ones; the baseline for the ratio is the rollout log-probs when the flag says so,
    # else those. (3) If the OPSM option or GSPO is on, all-gathers current and old log-probs
    # across CP to full responses. (4) Optional OPSM mask and clip fraction from the full log-probs
    # and advantages. (5) KL term ppo_kl: for GSPO the sequence-level KL helper over full and local
    # log-probs; otherwise per-token old minus current. Then the log-probs are concatenated. (6) Local
    # (CP-sliced) loss masks; ppo_kl and advantages have non-finite values replaced by zero and are
    # forced to zero on inactive tokens. (7) Clipped policy loss and clip fraction from ppo_kl,
    # advantages and clip epsilons (low, high, and the optional dual-clip constant). (8) Optional
    # debug dump when a dump-details setting exists. (9) OPSM mask multiplies the loss. (10) If
    # mismatch metrics or TIS are on: keeps the original reducer for metrics; loads the user
    # correction function or asserts trainer and rollout log-probs exist for the built-in one;
    # computes the importance ratio exp(-ppo_kl); calls the correction with arguments, loss, trainer
    # and rollout log-probs, masks, lengths, parallel state and max lengths, getting back the new
    # loss, modified masks and metrics; rebuilds the reducer from the modified masks but keeps the
    # original rollout mask sums as denominators. (11) Loss reducer is the user-supplied one (with
    # lengths, the modified masks if corrections ran else the batch masks, and the per-token flag)
    # or the default reducer. (12) Effective-sample-size ratio contribution computed from ppo_kl.
    # (13) Reduces the loss, clip fraction and ppo_kl. (14) Entropy: reduces it and subtracts
    # entropy coefficient times it from the loss when the coefficient is nonzero (else reports it
    # detached). (15) KL-to-reference loss when enabled: asserts reference log-probs exist, optional
    # importance ratio exp(current - old) for the unbiased variant, approximate KL of the configured
    # type with non-finite values zeroed outside active tokens, reduced and added with its
    # coefficient if nonzero. (16) If there are no local tokens, adds zero times the logit sum so
    # gradients still flow. (17) When rollout log-probs exist: diagnostics of the absolute
    # difference and low-variance KL between rollout and independently scored trainer log-probs,
    # reduced over active tokens.
    # Reported metrics (detached): loss, pg_loss, entropy_loss, pg_clipfrac, ppo_kl, ess_ratio;
    # plus rollout diagnostics, kl_loss, ois and the correction's metrics (reduced with the original
    # masks), opsm_clipfrac, and opd_reverse_kl when present.
    # KNOWN ISSUE: the docstring's metric list omits ess_ratio and several optional keys; the loss
    # is aliased to the policy loss, so the in-place zero-gradient addition changes both.
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    parallel_state = get_parallel_state()
    advantages_list = [advantage.detach() for advantage in batch["advantages"]]
    advantages = torch.cat(advantages_list, dim=0)
    rollout_old_log_probs = (
        [log_prob.detach() for log_prob in batch["rollout_log_probs"]]
        if batch.get("rollout_log_probs") is not None
        else None
    )
    reference_log_probs = (
        [log_prob.detach() for log_prob in batch["ref_log_probs"]] if batch.get("ref_log_probs") is not None else None
    )
    if args.use_rollout_logprobs:
        assert (
            rollout_old_log_probs is not None
        ), "rollout_log_probs must be provided when --use-rollout-logprobs is set"
    elif not args.skip_actor_forward_only and batch.get("log_probs") is None:
        raise ValueError("policy loss requires old-policy log-probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    calculate_entropy = args.entropy_coef != 0 or args.observe_training_entropy
    rollout_sampling_mask = get_rollout_sampling_masks(batch) if args.use_sampling_support_replay else None

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=calculate_entropy,
        entropy_requires_grad=args.entropy_coef != 0,
        max_seq_lens=max_seq_lens,
        rollout_sampling_mask=rollout_sampling_mask,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if args.skip_actor_forward_only:
        trainer_scored_log_probs = [log_prob.detach() for log_prob in log_probs]
    else:
        trainer_scored_log_probs = (
            [log_prob.detach() for log_prob in batch["log_probs"]] if batch.get("log_probs") is not None else None
        )
    if args.use_rollout_logprobs:
        old_log_probs = rollout_old_log_probs
    else:
        old_log_probs = trainer_scored_log_probs
    train_log_probs_list = log_probs
    old_log_probs_list = old_log_probs

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        if args.skip_actor_forward_only and not args.use_rollout_logprobs:
            full_old_log_probs = [full_log_prob.detach() for full_log_prob in full_log_probs]
        else:
            full_old_log_probs = [
                all_gather_with_cp(old_log_prob, total_length, response_length)
                for old_log_prob, total_length, response_length in zip(
                    old_log_probs, total_lengths, response_lengths, strict=False
                )
            ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=advantages_list,
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    local_loss_mask_list = get_local_response_loss_masks(
        total_lengths,
        response_lengths,
        batch["loss_masks"],
        args.qkv_format,
        max_seq_lens,
    )
    local_loss_masks = torch.cat(local_loss_mask_list, dim=0).to(device=ppo_kl.device)
    active_tokens = local_loss_masks.bool()
    ppo_kl = torch.where(
        active_tokens,
        torch.nan_to_num(ppo_kl, nan=0.0, posinf=0.0, neginf=0.0),
        ppo_kl.new_zeros(()),
    )
    advantages = torch.where(
        active_tokens,
        torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0),
        advantages.new_zeros(()),
    )

    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl, advantages, args.eps_clip, args.eps_clip_high, getattr(args, "eps_clip_c", None)
    )

    if getattr(args, "dump_details", None) is not None:
        from miles.backends.training_utils.metrics.debug_dump import maybe_dump_policy_loss_debug

        maybe_dump_policy_loss_debug(
            args=args,
            batch=batch,
            train_log_probs=train_log_probs_list,
            old_log_probs=old_log_probs_list,
            rollout_log_probs=rollout_old_log_probs,
            advantages=advantages_list,
            local_loss_masks=local_loss_mask_list,
            ppo_kl=ppo_kl,
            pg_loss=pg_loss,
        )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            assert trainer_scored_log_probs is not None, "log_probs must be provided for built-in TIS"
            assert rollout_old_log_probs is not None, "rollout_log_probs must be provided for built-in TIS"
            tis_func = vanilla_tis_function

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": trainer_scored_log_probs,
            "rollout_log_probs": rollout_old_log_probs,
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "parallel_state": parallel_state,
            "max_seq_lens": max_seq_lens,
        }

        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] modified masks correct the numerator only;
        # denominators stay the precomputed rollout_mask_sums.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
            denominators=batch.get("rollout_mask_sums", None),
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if args.custom_pg_loss_reducer_function_path is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    # ESS (Effective Sample Size) ratio from per-token IS weights
    # w = π_new/π_old = exp(-ppo_kl).  A value of 1.0 is on-policy; near 0
    # means the per-token weights are highly concentrated.
    ess_ratio_sum = compute_ess_ratio_contribution(
        ppo_kl=ppo_kl,
        loss_masks=batch["loss_masks"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        calculate_per_token_loss=args.calculate_per_token_loss,
    )

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    entropy_loss = pg_loss.new_zeros(())
    loss = pg_loss
    if calculate_entropy:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
        if args.entropy_coef != 0:
            loss = pg_loss - args.entropy_coef * entropy_loss
        else:
            entropy_loss = entropy_loss.detach()

    if args.use_kl_loss:
        assert reference_log_probs is not None, "ref_log_probs must be provided when --use-kl-loss is set"
        ref_log_probs = torch.cat(reference_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl = torch.where(
            active_tokens,
            torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0),
            kl.new_zeros(()),
        )
        kl_loss = sum_of_sample_mean(kl)

        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly; fp32 sum avoids fp16 inf -> nan
    if log_probs.numel() == 0:
        loss += 0 * logits.sum(dtype=torch.float32)

    train_rollout_logprob_abs_diff = None
    train_rollout_kl = None
    if rollout_old_log_probs:
        # The loss baseline may be rollout log-probs. Diagnostics must use an
        # independently scored trainer policy, even in that case.
        train_scored_log_probs = (
            torch.cat(trainer_scored_log_probs, dim=0) if trainer_scored_log_probs is not None else log_probs.detach()
        )
        rollout_log_probs = torch.cat(rollout_old_log_probs, dim=0)
        abs_diff = (train_scored_log_probs - rollout_log_probs).abs()
        abs_diff = torch.where(
            active_tokens,
            torch.nan_to_num(abs_diff, nan=0.0, posinf=0.0, neginf=0.0),
            abs_diff.new_zeros(()),
        )
        train_rollout_logprob_abs_diff = sum_of_sample_mean(abs_diff)

        # KL(rollout || train) at sampled tokens via Schulman k3 with per-token clamp [-10, 10]
        rollout_train_kl = compute_approx_kl(rollout_log_probs, train_scored_log_probs, kl_loss_type="low_var_kl")
        rollout_train_kl = torch.where(
            active_tokens,
            torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
            rollout_train_kl.new_zeros(()),
        )
        train_rollout_kl = sum_of_sample_mean(rollout_train_kl)

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
        "ess_ratio": ess_ratio_sum.squeeze(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if train_rollout_kl is not None:
        reported_loss["train_rollout_kl"] = train_rollout_kl.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metrics if available
    if batch.get("opd_reverse_kl") is not None:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # [FUNCTION DOC] PPO value loss. Old values and returns come from the batch (concatenated over
    # samples); current values are extracted from the value-head logits (flattened). Clip fraction
    # flag: tokens where the absolute change from old values exceeds the clip value from arguments.
    # Clipped prediction = old + clamp(new - old, +-clip); per-token loss = max of squared error of
    # the clipped prediction and of the unclipped prediction against the returns. Both the loss and
    # the (float) clip flags are reduced with the sample-mean reducer. If there are no values, adds
    # zero times the value sum to keep gradients flowing. Returns (loss, metrics with detached
    # value_loss and value_clipfrac).
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Negative log-likelihood of the response tokens: gets per-token log-probs (no
    # entropy) from the logits, concatenates them, and returns minus their reduced sample mean. With
    # no local tokens adds zero times the float32 logit sum so gradients flow. Returns (loss,
    # mapping with one detached entry "loss").
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly; fp32 sum avoids fp16 inf -> nan
    if log_probs.numel() == 0:
        loss += 0 * logits.sum(dtype=torch.float32)

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def get_loss_function(args: Namespace, loss_fn: str | None = None) -> LossFunction:
    # [FUNCTION DOC] Dispatcher. If a named loss function is given (the Tinker-style path), looks it
    # up in the Tinker loss table (imported lazily; key error if unknown) and returns it. Otherwise
    # switches on the loss type in arguments: score centering, policy loss, value loss, SFT loss, or
    # custom loss (loads a function from the configured path); any other value raises a value error.
    if loss_fn is not None:
        from miles.backends.training_utils.loss.hub.tinker_losses import TINKER_LOSS_FUNCTIONS

        return TINKER_LOSS_FUNCTIONS[loss_fn]
    match args.loss_type:
        case "score_centering":
            return score_centering_loss_function
        case "policy_loss":
            return policy_loss_function
        case "value_loss":
            return value_loss_function
        case "sft_loss":
            return sft_loss_function
        case "custom_loss":
            return load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")
