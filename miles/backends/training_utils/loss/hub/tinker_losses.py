# [MODULE DOC]
# Loss functions for Tinker-style client-defined training (cross entropy, importance sampling, PPO,
# CISPO, DRO). Each loss function has the standard loss signature (args, batch, logits, reducer)
# and returns (scalar loss, logging mapping). The loss is the plain sum over datums and response
# tokens of a per-token objective multiplied by the local context-parallel loss mask; the reducer
# argument is unused. The log mapping holds the detached loss and a per-datum list with, for each
# datum, its sample index, full-length response log probs and its loss (gathered over the CP group
# so they are identical on every CP rank). Config constants hold default thresholds, overridable by
# the batch's loss function config mapping. A registry maps loss name to function.
"""Sum Tinker losses over datums without server-side normalization.

Client loss inputs carry normalization; the trainer accumulates raw sums.
The SDK represents custom-loss gradients as `weights = -dL/dlogprob` with `cross_entropy`.
Per-datum outputs are gathered across the CP group; the loss stays a local-shard sum.
"""

from argparse import Namespace
from collections.abc import Callable

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_local_response_loss_masks, slice_log_prob_with_cp
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

PPO_DEFAULTS = {"clip_low_threshold": 0.8, "clip_high_threshold": 1.2}
CISPO_DEFAULTS = {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
DRO_DEFAULTS = {"beta": 0.05}


def _target_logprobs(args: Namespace, batch: RolloutBatch, logits: torch.Tensor) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Log probs of the client-supplied target tokens. For each sample the
    # token sequence is rebuilt by replacing its trailing positions (as many as the targets) with the
    # target tokens, then the shared log-prob routine is run without entropy. Returns the list of
    # per-sample local response log probs.
    # Tinker targets are explicit labels: splice them over the response region of the gather sequence
    label_tokens = [
        torch.cat([tokens[: len(tokens) - len(targets)], _as_tensor_like(targets, tokens)])
        for tokens, targets in zip(batch["unconcat_tokens"], batch["target_tokens"], strict=True)
    ]
    outputs = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=label_tokens,
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    return outputs["log_probs"]


def _as_tensor_like(values, reference: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Convert a list or tensor to a tensor with the dtype and device of the
    # reference tensor.
    return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)


def _local_response_values(args: Namespace, batch: RolloutBatch, values: list) -> list:
    # [FUNCTION DOC] HELPER: For each full-response-length client vector, return this context-parallel
    # rank's slice using the shared CP slicing helper with the qkv format and optional per-sample
    # padded length. Lists must be equally long as the batch lengths (strict zip raises).
    """This rank's zigzag CP shard of each full-length client vector; `rollout_log_probs` arrives pre-sliced."""
    max_seq_lens = batch.get("max_seq_lens", None)
    return [
        slice_log_prob_with_cp(
            value,
            total_length,
            response_length,
            args.qkv_format,
            max_seq_lens[i] if max_seq_lens is not None else None,
        )
        for i, (value, total_length, response_length) in enumerate(
            zip(values, batch["total_lengths"], batch["response_lengths"], strict=True)
        )
    ]


def _local_response_masks(args: Namespace, batch: RolloutBatch, log_probs: list[torch.Tensor]) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Local CP slices of the batch loss masks converted to the dtype/device
    # of the matching log prob tensors (one per sample, strict pairing).
    """This CP rank's slice of each loss mask; a DP-padding datum is all zeros and must not reach the objective."""
    local_masks = get_local_response_loss_masks(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )
    return [_as_tensor_like(mask, log_prob) for mask, log_prob in zip(local_masks, log_probs, strict=True)]


def _gather_per_datum_outputs(
    args: Namespace,
    batch: RolloutBatch,
    log_probs: list[torch.Tensor],
    per_datum_losses: list[torch.Tensor],
) -> list[dict]:
    # [FUNCTION DOC] HELPER: Build the per-datum output list. Returns empty list when there are no
    # losses. With CP size > 1, each datum's local log probs are scattered into a zero tensor of full
    # response length at their global positions. Per-datum losses (stacked) and all log probs are
    # concatenated into one flat tensor, summed across the CP group (all-reduce) so every rank has
    # full values, moved to CPU, then split back. Returns one mapping per datum with keys sample
    # index (from the batch), full log probs and loss. Detached throughout.
    """Per-datum outputs reported back to the client; identical on every CP rank."""
    if not per_datum_losses:
        return []
    parallel_state = get_parallel_state()
    logprobs = [log_prob.detach() for log_prob in log_probs]
    if parallel_state.cp.size > 1:
        response_lengths = batch["response_lengths"]
        local_positions = _local_response_values(
            args, batch, [torch.arange(n, device=logprobs[0].device) for n in response_lengths]
        )
        logprobs = [
            logprob.new_zeros(n).index_copy_(0, positions, logprob)
            for logprob, positions, n in zip(logprobs, local_positions, response_lengths, strict=True)
        ]
    losses_and_logprobs = torch.cat([torch.stack([local_loss.detach() for local_loss in per_datum_losses]), *logprobs])
    if parallel_state.cp.size > 1:
        dist.all_reduce(losses_and_logprobs, group=parallel_state.cp.group)
    losses_and_logprobs = losses_and_logprobs.cpu()
    num_datums = len(logprobs)
    full_losses = [full_loss.clone() for full_loss in losses_and_logprobs[:num_datums].unbind()]
    full_logprobs = [
        logprob.clone()
        for logprob in losses_and_logprobs[num_datums:].split([logprob.numel() for logprob in logprobs])
    ]
    return [
        {"sample_index": sample_index, "logprobs": full_logprob, "loss": full_loss}
        for sample_index, full_logprob, full_loss in zip(
            batch["sample_indices"], full_logprobs, full_losses, strict=True
        )
    ]


def _sum_loss_and_outputs(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    log_probs: list[torch.Tensor],
    per_datum_losses: list[torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    # [FUNCTION DOC] HELPER: Sum the per-datum losses into the scalar loss (if any datum has response
    # tokens), otherwise use a zero that is still connected to the logits graph so backward works.
    # Returns (loss, mapping with detached loss and the per-datum outputs).
    if any(log_prob.numel() for log_prob in log_probs):
        loss = torch.stack(per_datum_losses).sum()
    else:
        loss = logits[..., :0].sum(dtype=torch.float32)  # zero, but still connected to logits for backward
    per_datum = _gather_per_datum_outputs(args, batch, log_probs, per_datum_losses)
    return loss, {"loss": loss.detach(), "per_datum": per_datum}


def cross_entropy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    # [FUNCTION DOC] Cross entropy: per datum loss = negative sum over local response tokens of
    # client weight * target log prob * loss mask. (Client weights are the negated gradients.)
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = [
        -(_as_tensor_like(weights, log_prob) * log_prob * mask).sum()
        for log_prob, weights, mask in zip(
            log_probs,
            _local_response_values(args, batch, batch["loss_weights"]),
            _local_response_masks(args, batch, log_probs),
            strict=True,
        )
    ]
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def importance_sampling_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    # [FUNCTION DOC] Importance sampling: per datum loss = negative sum of ratio * advantage * mask,
    # where ratio = exp(target log prob - sampling log prob), advantages are CP-sliced client
    # values and rollout log probs are already CP-local.
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        per_datum_losses.append(-(ratio *_as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def ppo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    # [FUNCTION DOC] PPO: thresholds from the batch config (defaults 0.8 and 1.2). Per datum loss =
    # negative sum of mask * min(ratio * advantage, clamp(ratio, low, high) * advantage), ratio as in
    # importance sampling.
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", PPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", PPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        advantages = _as_tensor_like(advantage, log_prob)
        objective = torch.minimum(ratio * advantages, torch.clamp(ratio, clip_low, clip_high) * advantages)
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def cispo_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    # [FUNCTION DOC] CISPO: thresholds from the batch config (defaults 0.0 and 4.0). Per datum loss =
    # negative sum of mask * advantage * log prob * coefficient, where the coefficient is the ratio
    # clamped to the thresholds and detached (no gradient through the ratio).
    config = batch.get("loss_fn_config") or {}
    clip_low = config.get("clip_low_threshold", CISPO_DEFAULTS["clip_low_threshold"])
    clip_high = config.get("clip_high_threshold", CISPO_DEFAULTS["clip_high_threshold"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        ratio = torch.exp(log_prob - _as_tensor_like(sampling_log_prob, log_prob))
        coefficient = torch.clamp(ratio, clip_low, clip_high).detach()
        per_datum_losses.append(-(coefficient * log_prob * _as_tensor_like(advantage, log_prob) * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


def dro_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict]:
    # [FUNCTION DOC] DRO: beta from the batch config (default 0.05). Per datum loss = negative sum of
    # mask * (log prob * advantage - 0.5 * beta * (log prob - sampling log prob)^2).
    config = batch.get("loss_fn_config") or {}
    beta = config.get("beta", DRO_DEFAULTS["beta"])
    log_probs = _target_logprobs(args, batch, logits)
    per_datum_losses = []
    for log_prob, sampling_log_prob, advantage, mask in zip(
        log_probs,
        batch["rollout_log_probs"],
        _local_response_values(args, batch, batch["advantages"]),
        _local_response_masks(args, batch, log_probs),
        strict=True,
    ):
        divergence = log_prob - _as_tensor_like(sampling_log_prob, log_prob)
        objective = log_prob * _as_tensor_like(advantage, log_prob) - 0.5 * beta * divergence**2
        per_datum_losses.append(-(objective * mask).sum())
    return _sum_loss_and_outputs(args, batch, logits, log_probs, per_datum_losses)


TINKER_LOSS_FUNCTIONS = {
    "cross_entropy": cross_entropy_loss_function,
    "importance_sampling": importance_sampling_loss_function,
    "ppo": ppo_loss_function,
    "cispo": cispo_loss_function,
    "dro": dro_loss_function,
}
