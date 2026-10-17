# [MODULE DOC] Mismatch importance sampling (MIS) utilities for correcting the gap between the
# training engine's and the inference engine's token probabilities, plus metrics describing that
# gap. Three public entry points share one core: the core function (no distributed
# dependencies, unit-testable on CPU) works on per-sequence one dimensional tensors; a context
# parallel wrapper first gathers full sequences from the CP ranks and afterwards slices results
# back to this rank (Megatron utilities imported lazily); a FSDP wrapper just concatenates.
# Conventions: all inputs are lists with one tensor per sequence; log probabilities are per
# response token; metrics are mappings from name to a list of per-sequence tensors that have the
# same shape as the log probabilities (see the metrics append helper) so the trainer's mean
# reducer can aggregate them honoring the loss mask. Ratios are exponentiated differences of log
# probabilities clamped to plus or minus 20 to avoid overflow. Weight handling modes: truncate
# (cap above), clip (cap both sides) or mask (turn out-of-range tokens off in the loss mask),
# at token, sequence (sum of log ratios) or geometric (mean of log ratios) level.
from typing import Any

import torch


# NOTE:
# - `compute_mis_weights` is a lightweight, standalone function that is useful to unit-test on CPU.
# - `compute_mis_weights_with_cp` depends on Megatron context-parallel utilities, which are heavy and may not be
#   available in minimal environments.
# To keep `mis.py` importable for unit tests, we lazily import CP utilities inside `compute_mis_weights_with_cp`.


def masked_sum(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Sum of x weighted elementwise by the mask, as a scalar tensor; if
    # expand is true the scalar is broadcast to the shape of x.
    result = (x * loss_mask).sum()
    return result.expand_as(x) if expand else result


def masked_mean(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Masked sum divided by the mask total, with the divisor floored at 1 so
    # an empty mask gives 0. Scalar tensor, optionally broadcast to the shape of x.
    result = masked_sum(x, loss_mask) / torch.clamp_min(loss_mask.sum(), 1)
    return result.expand_as(x) if expand else result


def masked_min(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Minimum of x over positions where the mask is nonzero (the mask is
    # converted to boolean); when no position is selected returns a zero scalar of x's dtype and
    # device. Optionally broadcast to the shape of x.
    """Masked min over valid tokens (loss_mask == 1). Returns 0 when mask is empty."""
    mask = loss_mask.bool()
    if mask.any():
        result = x[mask].min()
    else:
        result = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return result.expand_as(x) if expand else result


def masked_max(x: torch.Tensor, loss_mask: torch.Tensor, expand: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Same as the masked minimum but takes the maximum; zero scalar when the
    # mask selects nothing.
    """Masked max over valid tokens (loss_mask == 1). Returns 0 when mask is empty."""
    mask = loss_mask.bool()
    if mask.any():
        result = x[mask].max()
    else:
        result = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return result.expand_as(x) if expand else result


def metrics_append(metrics: dict[str, list[torch.Tensor]], key: str, value: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Append a detached copy of the tensor to the list stored under the key
    # in the metrics mapping, creating the list if absent. Mutates the mapping. The tensor must
    # have the shape of the sequence's log probabilities (sequence-level values are expanded).
    """

    Every metrics-dict value is a list of 1D tensor, i.e., [torch.Tensor] with shapes exactly the same as log_probs.

    All metrics will be aggregated and averaged by `sum_of_sample_mean` and divided by DP size automatically
    - If calculate_per_token_loss=False (default), the final results will first be averaged in each sequence,
      then across all the sequences in the global batch.
    - If calculate_per_token_loss=True, the final results will be the mean of all the tokens in the global batch.

    No need to specifically handle loss_mask, sum_of_sample_mean automatically ignores statistics where loss_mask = 0.

    e.g.
    For token-level metric:
        value = [
            [0.1, 0.2],
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.6]
        ]
        When calculate_per_token_loss = False (default):
            result = (0.1 + 0.2) / 2 + (0.1 + 0.2 + 0.3 + 0.4 + 0.5) / 5 + (0.6) / 1 = 0.15 + 0.3 + 0.6 = 1.05 / 3 = 0.35
        When calculate_per_token_loss = True:
            result = (0.1 + 0.2 + 0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6) / 8 = 2.4 / 8 = 0.3
    For sequence-level metric:
        original sequence lengths = [2, 5, 2]
        We should expand the metrics to the length of each sequence:
        value = [
            [2, 2],
            [5, 5, 5, 5, 5],
            [1, 1]
        ]
        When calculate_per_token_loss = False (default):
            result = (2 + 2) / 2 + (5 + 5 + 5 + 5 + 5) / 5 + (1 + 1) / 2 = 2 + 5 + 1 = 8 / 3 = 2.6665
        Note that for sequence-level, calculating token-level loss is invalid; thus, calculate_per_token_loss should always be False.
    """
    if key not in metrics:
        metrics[key] = []
    metrics[key].append(value.clone().detach())


def calculate_veto_mask(
    log_ratio: torch.Tensor,
    loss_mask: torch.Tensor,
    veto_threshold: float | None,
    metrics: dict[str, list[torch.Tensor]],
    *,
    metric_prefix: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] Sequence-level veto for one sequence. Input: per-token log ratio, loss mask,
    # an optional threshold on the ratio (not its log), the metrics mapping and a metric key
    # prefix. Without a threshold returns all ones. Otherwise a token is catastrophic if its log
    # ratio is below the log of the threshold and it is unmasked; if any token of the sequence is
    # catastrophic the whole sequence gets zeros, else ones (float tensor of the log ratio's
    # shape). Records two metrics: the per-token catastrophic indicator and the sequence-wide
    # indicator broadcast to the mask's shape. The threshold must be positive.
    if veto_threshold is None:
        return torch.ones_like(log_ratio)
    log_veto_threshold = torch.log(torch.tensor(veto_threshold, device=log_ratio.device))
    # For each sequence, if it has any catastrophic tokens, return 0 for the sequence
    catastrophic_tokens = (log_ratio < log_veto_threshold) & loss_mask.bool()
    has_catastrophic = catastrophic_tokens.any()
    veto_mask = (~has_catastrophic).float().expand_as(log_ratio)

    metrics_append(metrics, f"{metric_prefix}catastrophic_token_fraction", catastrophic_tokens.int())
    metrics_append(metrics, f"{metric_prefix}catastrophic_seq_fraction", has_catastrophic.int().expand_as(loss_mask))
    return veto_mask


def truncate(
    weights: torch.Tensor,
    loss_mask: torch.Tensor,
    metrics: dict[str, list[torch.Tensor]],
    upper_bound: float,
    *,
    metric_prefix: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] Truncated importance sampling for one sequence: records the indicator of
    # weights above the upper bound (before clamping), then clamps weights into zero to upper
    # bound and zeroes masked positions. Asserts the bound is given. Returns the weights.
    assert upper_bound is not None
    metrics_append(metrics, f"{metric_prefix}truncate_fraction", (weights > upper_bound).int())
    return weights.clamp(0, upper_bound) * loss_mask


def clip(
    weights: torch.Tensor,
    loss_mask: torch.Tensor,
    metrics: dict[str, list[torch.Tensor]],
    lower_bound: float,
    upper_bound: float,
    *,
    metric_prefix: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] Clipped importance sampling for one sequence: asserts both bounds are given
    # and the lower is strictly below the upper; records indicators of weights below the lower
    # bound and above the upper bound (before clamping); clamps weights into the interval and
    # zeroes masked positions. Returns the weights.
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics_append(metrics, f"{metric_prefix}clip_fraction_low", (weights < lower_bound).int())
    metrics_append(metrics, f"{metric_prefix}clip_fraction_high", (weights > upper_bound).int())
    return weights.clamp(lower_bound, upper_bound) * loss_mask


def mask(
    weights: torch.Tensor,
    loss_mask: torch.Tensor,
    metrics: dict[str, list[torch.Tensor]],
    lower_bound: float,
    upper_bound: float,
    *,
    metric_prefix: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    # [FUNCTION DOC] Masking (rejection) for one sequence: asserts valid bounds as for clipping;
    # records indicators of weights below the lower and above the upper bound; builds a new
    # loss mask equal to the input mask where the weight lies inside the closed interval and zero
    # elsewhere. The weights are only zeroed at masked-out padding, not changed at rejected
    # positions. Returns the pair of weights and new mask (float).
    assert lower_bound is not None and upper_bound is not None and lower_bound < upper_bound
    metrics_append(metrics, f"{metric_prefix}mask_fraction_low", (weights < lower_bound).int())
    metrics_append(metrics, f"{metric_prefix}mask_fraction_high", (weights > upper_bound).int())
    in_range = (weights >= lower_bound) & (weights <= upper_bound)
    modified_mask = loss_mask * in_range.float()
    # Zero out padding in weights but preserve values at non-rejected positions
    weights = weights * loss_mask
    return weights, modified_mask


def compute_mis_weights(
    args,
    *,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, list[torch.Tensor]]]:
    # [FUNCTION DOC] Core MIS computation over a list of sequences (docstring below lists the
    # inputs). Steps: derive bounds (TIS lower bound defaults to the reciprocal of the TIS upper
    # bound; rejection-sampling bounds default to the TIS bounds); assert the three lists have
    # equal length and matching shapes per sequence; record perplexity/divergence metrics for each
    # sequence; if TIS is disabled return no weights, the original masks and the metrics. Then per
    # sequence (mask converted to float): raw log ratio = training minus rollout log probability;
    # TIS ratio at the configured level (token: as is; sequence: masked sum broadcast; geometric:
    # masked mean broadcast; anything else raises a value error), clamped to plus or minus 20 and
    # exponentiated, recorded before bounding, then bounded by the configured mode (truncate,
    # clip or mask, else value error) and recorded after bounding. If rejection sampling is on,
    # its ratio (reusing the TIS ratio when levels are equal) is clamped and exponentiated, then
    # used to mask the working loss mask with its bounds; an optional veto threshold additionally
    # zeroes whole sequences having a token whose raw ratio is below the threshold. Records the
    # mean of the weights; weights and masks are detached and collected.
    # If batch normalization is on, divide all weights by their mean (token level: total weight
    # over total original mask count; sequence level: mean of per-sequence masked means; other
    # levels raise a value error) unless the mean is not above 1e-8, in which case weights are
    # unchanged; the factor is recorded per sequence. Finally records, per sequence, masked mean,
    # min and max of final weights broadcast to token shape. Returns weights, modified masks and
    # metrics.
    # KNOWN ISSUE: when TIS is disabled the function returns none for the weights although the
    # annotated type is a list, and everything after the early return that checks the use-TIS
    # flag again (including the else branch giving unit weights, and the rejection sampling
    # path being reachable only with TIS) is dead or unreachable without TIS.
    # KNOWN ISSUE: the geometric level is accepted for the log ratio but batch normalization
    # rejects it with a value error.
    """
    Compute the importance sampling (IS) weights and metrics between the inference and training engine.
    Args:
        train_log_probs: List of log probs from training backend. 1D tensor each. Lengths can be different.
        rollout_log_probs: List of log probs from inference backend. 1D tensor each.
        loss_masks: List of loss masks. 1D tensor each.
            Note that for single turn RL, the loss_mask is [1] * response_length tensor for each sequence
            For multi-turn RL, the tool response will be marked as 0 in the loss_mask.

    Returns:
        weights: List of importance sampling weights (safety-bounded; zeroed at padding only). 1D tensor each.
        modified_response_masks: List of rejection masks to apply in aggregation (mask mode + veto). 1D tensor each.
        metrics: The metrics for the importance sampling weights, a dict of list[torch.Tensor]. 1D tensor each.
    """

    metrics: dict[str, list[torch.Tensor]] = {}

    tis_lower_bound = args.tis_lower_bound if args.tis_lower_bound is not None else 1.0 / args.tis_upper_bound
    rs_lower_bound = args.rs_lower_bound if args.rs_lower_bound is not None else tis_lower_bound
    rs_upper_bound = args.rs_upper_bound if args.rs_upper_bound is not None else args.tis_upper_bound

    # Validate input lists have same length and each sequence has matching shapes
    assert (
        len(train_log_probs) == len(rollout_log_probs) == len(loss_masks)
    ), f"Input lists must have the same number of sequences: {len(train_log_probs)} vs {len(rollout_log_probs)} vs {len(loss_masks)}"

    for i, (train, rollout, loss_mask) in enumerate(zip(train_log_probs, rollout_log_probs, loss_masks, strict=False)):
        assert (
            train.shape == rollout.shape == loss_mask.shape
        ), f"Sequence {i}: shapes must match - train: {train.shape}, rollout: {rollout.shape}, loss_mask: {loss_mask.shape}"

    SAFETY_BOUND = 20.0  # Add a safety bound to avoid exp overflow
    all_weights = []
    all_modified_masks = []

    def compute_log_ratio(raw_log_diff: torch.Tensor, mask: torch.Tensor, level: str) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Aggregate a per-token log difference to the requested level:
        # token returns it unchanged; sequence returns the masked sum broadcast over tokens;
        # geometric returns the masked mean broadcast over tokens; other levels raise a value error.
        if level == "token":
            return raw_log_diff
        elif level == "sequence":
            return masked_sum(raw_log_diff, mask, expand=True)
        elif level == "geometric":
            return masked_mean(raw_log_diff, mask, expand=True)
        else:
            raise ValueError(f"Invalid level: {level}")

    for train_log_prob, rollout_log_prob, loss_mask in zip(
        train_log_probs, rollout_log_probs, loss_masks, strict=False
    ):
        add_ppl_metrics(train_log_prob, rollout_log_prob, loss_mask, metrics)

    # only calculate mismatch metrics if TIS is not used
    if not args.use_tis:
        return None, loss_masks, metrics

    # handle each sequence independently
    for train_log_prob, rollout_log_prob, loss_mask in zip(
        train_log_probs, rollout_log_probs, loss_masks, strict=False
    ):
        loss_mask = loss_mask.float()
        raw_log_ratio_diff = train_log_prob - rollout_log_prob
        modified_mask = loss_mask.clone().float()

        # IS (Importance Sampling): Modify IS weights
        if args.use_tis:
            log_ratio_tis = compute_log_ratio(raw_log_ratio_diff, loss_mask, args.tis_level)
            log_ratio_safe = torch.clamp(log_ratio_tis, min=-SAFETY_BOUND, max=SAFETY_BOUND)
            weights = torch.exp(log_ratio_safe)
            metrics_append(metrics, "tis_weight_before_bound", weights)

            if args.tis_mode == "truncate":
                weights = truncate(weights, loss_mask, metrics, args.tis_upper_bound, metric_prefix="tis_")
            elif args.tis_mode == "clip":
                weights = clip(
                    weights, loss_mask, metrics, tis_lower_bound, args.tis_upper_bound, metric_prefix="tis_"
                )
            elif args.tis_mode == "mask":
                weights, modified_mask = mask(
                    weights, loss_mask, metrics, tis_lower_bound, args.tis_upper_bound, metric_prefix="tis_"
                )
            else:
                raise ValueError(f"Unsupported tis_mode: {args.tis_mode}")

            metrics_append(metrics, "tis_weight_after_bound", weights)
        else:
            weights = torch.ones_like(raw_log_ratio_diff)

        # RS (Rejection Sampling): Modify mask
        if args.use_rs:
            if args.use_tis and args.rs_level == args.tis_level:
                log_ratio_rs = log_ratio_tis
            else:
                log_ratio_rs = compute_log_ratio(raw_log_ratio_diff, loss_mask, args.rs_level)

            log_ratio_safe_rs = torch.clamp(log_ratio_rs, min=-SAFETY_BOUND, max=SAFETY_BOUND)
            rs_weights = torch.exp(log_ratio_safe_rs)

            # Apply mask-based rejection sampling
            _, modified_mask = mask(
                rs_weights, modified_mask, metrics, rs_lower_bound, rs_upper_bound, metric_prefix="rs_"
            )

            # Veto on raw per-token ratios (sequence-wise rejection)
            if args.rs_veto_threshold is not None:
                veto_mask = calculate_veto_mask(
                    raw_log_ratio_diff, loss_mask, args.rs_veto_threshold, metrics, metric_prefix="rs_"
                )
                modified_mask = modified_mask * veto_mask

        metrics_append(metrics, "is_ratio_mean_after_tis_rs", weights)

        weights = weights.detach()
        modified_mask = modified_mask.detach()
        all_weights.append(weights)
        all_modified_masks.append(modified_mask)

    # Apply batch normalization if enabled (normalize to mean=1.0 across entire batch)
    if args.tis_batch_normalize:
        # Compute mean based on TIS aggregation level
        tis_level = args.tis_level if args.use_tis else "token"
        if tis_level == "token":
            # Token-level: normalize over all token weights
            total_weights_sum = sum(masked_sum(w, m) for w, m in zip(all_weights, loss_masks, strict=False))
            total_mask_count = sum(m.sum() for m in loss_masks)
            weights_mean = total_weights_sum / torch.clamp_min(total_mask_count, 1)
        elif tis_level == "sequence":
            # Sequence-level: normalize over sequence weights (one weight per sequence)
            # For each sequence, compute mean over valid tokens (they all have the same weight)
            # then average across sequences
            seq_weights_means = [masked_mean(w, m) for w, m in zip(all_weights, loss_masks, strict=False)]
            weights_mean = sum(seq_weights_means) / len(seq_weights_means)
        else:
            raise ValueError(f"Unsupported tis_level: {tis_level}")

        # Normalize to mean=1.0 (avoid division by zero)
        if weights_mean > 1e-8:
            all_weights = [w / weights_mean for w in all_weights]
            for w in all_weights:
                metrics_append(metrics, "batch_norm_factor", weights_mean.expand_as(w))
        else:
            for w in all_weights:
                metrics_append(metrics, "batch_norm_factor", torch.ones_like(w))

    # Final weight stats (after optional batch normalization).
    # NOTE: These are expanded to token-shape so that the existing mean-reducer can aggregate them.
    for w, m in zip(all_weights, loss_masks, strict=False):
        m = m.float()
        metrics_append(metrics, "is_ratio_mean_final", masked_mean(w, m, expand=True))
        metrics_append(metrics, "is_ratio_min_final", masked_min(w, m, expand=True))
        metrics_append(metrics, "is_ratio_max_final", masked_max(w, m, expand=True))

    return all_weights, all_modified_masks, metrics


def compute_mis_weights_with_cp(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    # [FUNCTION DOC] Context-parallel wrapper around the core computation. Lazily imports the
    # gather and slice helpers. Reads the tensor layout format from args; for the padded batch
    # format requires the optional per-sequence maximum lengths keyword (assertion error if
    # absent). Gathers the rollout and training log probabilities of every sequence from all CP
    # ranks into full-length tensors (using total and response lengths), runs the core on the
    # full tensors with the given (already full-length) loss masks, then slices each weight
    # tensor and each metric back to this rank's shard and concatenates along the first dimension.
    # If weights exist, multiplies the policy-gradient loss by the concatenated weights. Metric
    # keys are returned prefixed with the letters mis and an underscore. The modified masks are
    # returned as produced by the core, NOT sliced to this rank.
    # Returns the loss, the modified masks (list) and the flattened metrics mapping.
    """
    Compute the importance sampling (IS) weights and metrics with context parallel.
    Args:
        train_log_probs: List of log probs from training backend on this cp rank. 1D tensor each. Lengths can be different.
        rollout_log_probs: List of log probs from inference backend on this cp rank. 1D tensor each.
        loss_masks: List of loss masks. 1D tensor each.
        total_lengths: List of total lengths.
        response_lengths: List of response lengths.
    Returns:
        pg_loss: Policy gradient loss with IS weights applied (flattened along dim=0).
        modified_masks: List of modified response masks with rejection applied (one per sequence).
        is_metrics: The metrics for the importance sampling weights, a dict of flattened tensors.
    """
    # Lazy import to avoid importing Megatron dependencies when only `compute_mis_weights` is used.
    from miles.backends.training_utils.data.context_parallel import all_gather_with_cp, slice_log_prob_with_cp

    qkv_format = args.qkv_format
    max_seq_lens = kwargs.get("max_seq_lens", None)
    if qkv_format == "bshd":
        assert max_seq_lens is not None, "max_seq_lens is required when qkv_format is bshd"

    # Gather cp slice from other cp ranks
    def gather_log_probs(log_probs):
        # [FUNCTION DOC] HELPER: For each sequence's local CP shard, gather the full-length
        # response log probabilities across CP ranks (with the per-sequence max length when the
        # padded layout is used, else none). Returns a list of tensors.
        return [
            all_gather_with_cp(
                log_prob,
                total_length,
                response_length,
                qkv_format,
                max_seq_lens[i] if max_seq_lens is not None else None,
            )
            for i, (log_prob, total_length, response_length) in enumerate(
                zip(log_probs, total_lengths, response_lengths, strict=False)
            )
        ]

    full_rollout_log_probs = gather_log_probs(rollout_log_probs)
    full_old_log_probs = gather_log_probs(train_log_probs)

    # Main logic for is (decoupled)
    is_weights, modified_masks, is_metrics = compute_mis_weights(
        args=args,
        train_log_probs=full_old_log_probs,
        rollout_log_probs=full_rollout_log_probs,
        loss_masks=loss_masks,
    )

    # Slice out the value shards for this CP rank and concat them into a 1D tensor along dim=0 for loss.py computation.
    def slice_cp_and_concat(
        values: list[torch.Tensor], total_lengths: list[int], response_lengths: list[int]
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Slice each full-length per-sequence tensor to this CP rank's
        # shard (same layout rules as the gather) and concatenate the shards along the first
        # dimension into one tensor. The length lists are passed explicitly and shadow the outer
        # ones with the same contents.
        values = [
            slice_log_prob_with_cp(
                values[i],
                total_lengths[i],
                response_lengths[i],
                qkv_format,
                max_seq_lens[i] if max_seq_lens is not None else None,
            )
            for i in range(len(values))
        ]
        return torch.cat(values, dim=0)

    result_metrics = {}
    if is_weights is not None:
        is_weights = slice_cp_and_concat(is_weights, total_lengths, response_lengths)
        pg_loss = pg_loss * is_weights

    for key, values in is_metrics.items():
        key_name = f"mis_{key}"
        values = slice_cp_and_concat(values, total_lengths, response_lengths)
        result_metrics[key_name] = values

    return pg_loss, modified_masks, result_metrics


def add_ppl_metrics(
    train_log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    loss_mask: torch.Tensor,
    metrics: dict[str, list[torch.Tensor]],
):
    # [FUNCTION DOC] Record divergence and perplexity metrics for one sequence into the metrics
    # mapping (all expanded to the per-token shape). Training and rollout perplexity and their
    # logs from masked mean log probabilities; direct KL estimate (rollout minus training log
    # probability, per token); the K3 estimator (ratio minus log ratio minus 1, per token,
    # without clamping); log perplexity difference (rollout mean minus training mean), its
    # absolute value and the perplexity ratio (its exponential); token-level chi-squared
    # divergence (masked mean of squared ratios minus 1, ratio from a log ratio clamped to plus
    # or minus 20); sequence-level chi-squared (exponential of twice the clamped masked sum of
    # log ratios, minus 1). No return.
    loss_mask = loss_mask.float()

    mean_log_prob_training = masked_mean(train_log_prob, loss_mask, expand=True)
    training_log_ppl = -mean_log_prob_training
    training_ppl = torch.exp(training_log_ppl)
    metrics_append(metrics, "training_log_ppl", training_log_ppl)
    metrics_append(metrics, "training_ppl", training_ppl)

    # 2. Rollout policy perplexity metrics
    mean_log_prob_rollout = masked_mean(rollout_log_prob, loss_mask, expand=True)
    rollout_log_ppl = -mean_log_prob_rollout
    rollout_ppl = torch.exp(rollout_log_ppl)
    metrics_append(metrics, "rollout_log_ppl", rollout_log_ppl)
    metrics_append(metrics, "rollout_ppl", rollout_ppl)

    # 3a. kl: Direct estimator for KL(π_rollout || π_training)
    # This is the standard KL divergence: E[log(π_rollout) - log(π_training)]
    # Positive value means rollout policy is more confident than training policy
    kl_per_token = rollout_log_prob - train_log_prob
    metrics_append(metrics, "kl", kl_per_token)

    # 3b. K3 KL estimator for improved stability
    # More stable for small KL values using: E[exp(log_ratio) - log_ratio - 1]
    # Formula: KL ≈ E[r - log(r) - 1] where r = π_training/π_rollout
    log_ratio = train_log_prob - rollout_log_prob
    k3_kl_matrix = torch.exp(log_ratio) - log_ratio - 1
    metrics_append(metrics, "k3_kl", k3_kl_matrix)

    # 3c. Log PPL difference (sequence-level perplexity difference)
    # log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    # Since ppl = exp(-log_prob), we have:
    #   log(ppl_ratio) = log(training_ppl/rollout_ppl) = log_ppl_diff
    # Positive value means training assigns lower probability (higher PPL) than rollout
    log_ppl_diff = mean_log_prob_rollout - mean_log_prob_training
    metrics_append(metrics, "log_ppl_diff", log_ppl_diff)
    metrics_append(metrics, "log_ppl_abs_diff", log_ppl_diff.abs())

    # 3d. PPL ratio (how much higher is training PPL vs rollout PPL)
    # For numerical stability, compute in log space using log_ppl_diff
    # Note: log_ppl_diff = log(ppl_ratio), so ppl_ratio = exp(log_ppl_diff)
    ppl_ratio = torch.exp(log_ppl_diff)
    metrics_append(metrics, "ppl_ratio", ppl_ratio)

    # 4a. Token-level chi-squared divergence
    # χ²(π_training || π_rollout) = E[ρ²] - 1, where ρ = π_training / π_rollout
    # This measures the second moment of the importance weights
    SAFETY_BOUND = 20.0
    log_ratio = train_log_prob - rollout_log_prob
    log_ratio_safe = torch.clamp(log_ratio, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    rho_token = torch.exp(log_ratio_safe)  # ρ = π_training / π_rollout
    rho_squared_token = rho_token.square()
    chi2_token_value = masked_mean(rho_squared_token, loss_mask) - 1.0
    chi2_token = chi2_token_value.expand_as(train_log_prob)
    metrics_append(metrics, "chi2_token", chi2_token)

    # 4b. Sequence-level chi-squared divergence
    # Computes (Π ρ_t)² - 1 for the entire sequence
    # This captures the squared product of importance ratios
    log_ratio_sum = masked_sum(log_ratio, loss_mask, expand=True)
    log_ratio_sum_safe = torch.clamp(log_ratio_sum, min=-SAFETY_BOUND, max=SAFETY_BOUND)
    rho_squared_seq = torch.exp(2.0 * log_ratio_sum_safe)  # (Π ρ_t)²
    chi2_seq = rho_squared_seq - 1.0
    metrics_append(metrics, "chi2_seq", chi2_seq)


def compute_mis_weights_fsdp(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    # [FUNCTION DOC] FSDP wrapper (no context parallelism): runs the core computation; if weights
    # exist, concatenates them across sequences and multiplies the already flattened policy
    # gradient loss by them; concatenates every metric across sequences under the mis prefix.
    # Extra keyword arguments are ignored. Returns the loss, the modified masks (list per
    # sequence) and the flattened metrics.
    """Compute masked importance sampling weights for FSDP. No context parallelism.

    Args:
        args: Arguments containing MIS settings (use_tis, tis_mode, etc.)
        pg_loss: Policy gradient loss, flattened tensor [total_tokens]
        train_log_probs: Training log probs, list of 1D tensors per sequence
        rollout_log_probs: Rollout log probs, list of 1D tensors per sequence
        loss_masks: Loss masks, list of 1D tensors per sequence
        **kwargs: Additional arguments (cp_rank, cp_size, etc.) for compatibility

    Returns:
        pg_loss: Policy gradient loss with IS weights applied
        modified_masks: Modified loss masks after rejection sampling
        mis_metrics: Metrics dict with flattened tensors
    """
    is_weights, modified_masks, is_metrics = compute_mis_weights(
        args=args,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )

    result_metrics = {}
    if is_weights is not None:
        is_weights_flat = torch.cat(is_weights, dim=0)
        pg_loss = pg_loss * is_weights_flat

    for key, values in is_metrics.items():
        result_metrics[f"mis_{key}"] = torch.cat(values, dim=0)

    return pg_loss, modified_masks, result_metrics
