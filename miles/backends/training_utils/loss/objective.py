# [MODULE DOC]
# Training objective orchestration. Two stages: (1) compute_advantages_and_returns turns rollout
# log probs / rewards / values into per-sample advantages and returns (with KL shaping, optional
# on-policy distillation penalty and optional whitening) stored back into the rollout data; (2)
# loss_function dispatches to the configured loss from the loss hub, then rescales it for Megatron's
# gradient accumulation and packs a logging mapping. Also a helper to emit an audit event with the
# computed advantages.
from argparse import Namespace
import torch
from torch.utils.checkpoint import checkpoint

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks, get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub.advantages import compute_advantages, normalize_advantages
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy, get_values  # noqa: F401
from miles.backends.training_utils.loss.hub.losses import get_loss_function
from miles.backends.training_utils.loss.hub.math_utils import compute_approx_kl
from miles.backends.training_utils.loss.hub.opd import apply_opd_kl_to_advantages
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import TrainAdvantageComputationEvent
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.types import RolloutBatch


def _detach_rollout_tensor_list(rollout_data: RolloutBatch, key: str) -> list[torch.Tensor] | None:
    # [FUNCTION DOC] HELPER: If the rollout data has a list of tensors under the key, replace it in
    # place by the detached tensors (dropping autograd history) and return the new list; if the key
    # is absent or none, return none and leave the data unchanged.
    tensors = rollout_data.get(key)
    if tensors is None:
        return None

    detached_tensors = [tensor.detach() for tensor in tensors]
    rollout_data[key] = detached_tensors
    return detached_tensors


def compute_advantages_and_returns(
    args: Namespace,
    rollout_data: RolloutBatch,
) -> None:
    # [FUNCTION DOC] Fills "advantages" and "returns" in the rollout data. Steps: read the log-prob
    # list (rollout log probs when the use-rollout-logprobs flag is on, else trainer log probs),
    # reference log probs, rewards, values, lengths and masks. Return without changes if this is not
    # the last pipeline stage, or if there are neither log probs nor values (unless the actor
    # forward pass is skipped and rollout log probs are not used). Detach the stored log-prob lists
    # (log probs, rollout, reference, teacher) in the rollout data. KL: if there are neither log
    # probs nor values, zeros shaped like the local response masks; if the kl coefficient is 0 or
    # log probs are empty, zeros shaped like the log probs (or values); otherwise the approximate
    # KL (type from args) between log probs and reference log probs per sample. Then compute
    # advantages and returns via the estimator selected in args, apply the OPD penalty when enabled,
    # and optionally whiten advantages across the data-parallel group. KNOWN ISSUE: the empty-log-prob
    # test is falsy for an empty list, so an empty list of log probs with a nonzero coefficient also
    # takes the zero-KL branch (probably intended).
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". On-policy distillation (OPD) is applied
    orthogonally on top of any estimator via `args.use_opd`. When
    `args.normalize_advantages` is True, advantages are whitened across the
    data-parallel group using masked statistics.

    Early returns on intermediate pipeline stages; the last stage also
    returns early when neither `log_probs` nor `values` is present, unless
    it is explicitly allowed to derive zero-KL shapes without the standalone
    actor pass.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    allow_missing_log_probs = args.skip_actor_forward_only and not args.use_rollout_logprobs
    log_probs_key = "rollout_log_probs" if args.use_rollout_logprobs else "log_probs"
    log_probs: list[torch.Tensor] = rollout_data.get(log_probs_key)
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # under --use-rollout-logprobs every stage holds log-probs, so ask the parallel state, not the tensors
    if not get_parallel_state().is_pp_last_stage:
        return
    if log_probs is None and values is None and not allow_missing_log_probs:
        return

    # This is the authoritative persistence boundary: scores produced before
    # the policy update are fixed training data and must not retain a graph.
    _detach_rollout_tensor_list(rollout_data, "log_probs")
    _detach_rollout_tensor_list(rollout_data, "rollout_log_probs")
    _detach_rollout_tensor_list(rollout_data, "ref_log_probs")
    _detach_rollout_tensor_list(rollout_data, "teacher_log_probs")
    log_probs = rollout_data.get(log_probs_key)
    ref_log_probs = rollout_data.get("ref_log_probs")

    if log_probs is None and values is None:
        local_masks = get_local_response_loss_masks(
            total_lengths, response_lengths, loss_masks, args.qkv_format, max_seq_lens
        )
        kl = [torch.zeros_like(mask, dtype=torch.float32) for mask in local_masks]
    elif args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    advantages, returns = compute_advantages(
        args=args,
        kl=kl,
        rewards=rewards,
        log_probs=log_probs,
        loss_masks=loss_masks,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        values=values,
    )

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    if args.normalize_advantages:
        advantages = normalize_advantages(args, advantages, loss_masks, total_lengths, response_lengths, max_seq_lens)

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    apply_megatron_loss_scaling: bool = False,
    num_rollouts: int | None = None,
) -> tuple[torch.Tensor, int | torch.Tensor, dict]:
    # [FUNCTION DOC] Compute the loss for one micro-batch and rescale it. Steps: number of tokens =
    # sum over samples of the loss mask sum clamped to at least 1; build the per-sample-mean reducer
    # (honoring per-token loss mode, qkv format, optional padded lengths and optional rollout mask
    # sums as denominators); look up the loss function from args and the batch's loss function name;
    # call it directly or through activation checkpointing when recompute-loss is on. With CP > 1
    # and all-gather-CP, add a zero-valued term depending on the logits so autograd visits the whole
    # graph on every rank. The normalizer is num_rollouts if given, else the dynamic global batch
    # size from the batch (asserting the dynamic-batch flag matches presence of that key) or the
    # configured global batch size; it is forced to 1 when multi-LoRA is enabled. Scaling: without
    # per-token loss, with Megatron scaling: loss * num_microbatches / normalizer * the DP (times CP,
    # or DP only for true-on-policy with Ulysses CP) size; without Megatron scaling: loss /
    # normalizer * intra-DP size. With per-token loss: multiplied by CP size only under Megatron
    # scaling. A per-datum entry is popped from the log mapping only if a Tinker loss name is set.
    # Returns (loss, token count tensor or tensor 1, mapping with optional per-datum, "keys" and
    # "values" = tensor of [sample count (or token count if per-token), metrics in key order]).
    # KNOWN ISSUE: the docstring says the normalizer is a plain int 1 when not per-token, but a
    # tensor is returned in both cases; the comment about Tinker losses carrying their own
    # normalization sits above code that checks multi-LoRA, not the Tinker loss name. The tensor
    # constructor is applied to a tensor token count (copy warning). The docstring omits that the
    # parameter apply_megatron_loss_scaling exists.
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects "policy_loss", "score_centering", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, or the Tinker loss specified by
    `batch["loss_fn"]`. Computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).
        num_rollouts: This step's rollout count (total across DP), used as
            the loss normalizer; None falls back to the legacy batch/args value.

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
          Tinker losses may also return "per_datum" outputs.
    """
    parallel_state = get_parallel_state()
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
        denominators=batch.get("rollout_mask_sums", None),
    )

    func = get_loss_function(args, batch.get("loss_fn"))

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # Forces autograd to traverse the full graph on every rank to avoid hang.
    # fp32 sum: an fp16 logits sum can overflow to inf, and 0 * inf is nan.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum(dtype=torch.float32)

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if num_rollouts is not None:
        global_batch_size = num_rollouts
    else:
        assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)
        global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    # Tinker losses already carry the client's normalization
    loss_normalizer = 1 if is_multi_lora_enabled(args) else global_batch_size
    if not args.calculate_per_token_loss:
        if apply_megatron_loss_scaling:
            loss_parallel_size = (
                parallel_state.intra_dp.size
                if args.true_on_policy_mode and parallel_state.is_ulysses_cp
                else parallel_state.intra_dp_cp.size
            )
            loss = loss * num_microbatches / loss_normalizer * loss_parallel_size
        else:
            loss = loss / loss_normalizer * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            loss = loss * parallel_state.cp.size

    per_datum = log.pop("per_datum", None) if batch.get("loss_fn") is not None else None
    return (
        loss,
        torch.tensor(num_tokens if args.calculate_per_token_loss else 1, device=logits.device),
        {
            **({"per_datum": per_datum} if per_datum is not None else {}),
            "keys": list(log.keys()),
            "values": torch.tensor(
                [num_samples if not args.calculate_per_token_loss else num_tokens] + list(log.values()),
                device=logits.device,
            ),
        },
    )


def log_train_advantage_computation_event(rollout_data: RolloutBatch) -> None:
    # [FUNCTION DOC] Audit hook: if the event logger is initialized and the rollout data has both
    # advantages and witness ids, log one advantage-computation event with both converted to nested
    # Python lists (no console print). Otherwise does nothing.
    if not is_event_logger_initialized():
        return

    advantages = rollout_data.get("advantages")
    witness_ids = rollout_data.get("witness_ids")
    if advantages is None or witness_ids is None:
        return

    get_event_logger().log(
        TrainAdvantageComputationEvent,
        dict(
            advantages=[x.tolist() for x in advantages],
            witness_ids=[x.tolist() for x in witness_ids],
        ),
        print_log=False,
    )
