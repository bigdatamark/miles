# [MODULE DOC] Conversion of generated rollout samples into training data and splitting that data
# across data-parallel ranks. Steps: (1) turn samples into a mapping of column name to per-sample
# lists (tokens, lengths, rewards, loss masks and optional extras such as log probs, top-k,
# sampling masks, routed experts, multimodal inputs, teacher log probs); (2) post-process rewards
# (optional per-rollout group normalization); (3) split into per-rank shards, either with a
# micro-batch schedule precomputed on the rollout side or with a legacy partition (balanced by
# sequence length or round-robin), and put the shards into the object store; (4) the train-side
# completion step for a received shard. Also defines the dtype table of tensor columns and the
# object store value spec (codec per column).
import logging
from typing import Any

import torch

from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils import object_store
from miles.utils.dp_schedule import build_dp_schedule, has_full_schedule_config
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.object_store import ValueSpec
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.timer import Timer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# Table from column name to the element dtype name used when encoding that column as a typed
# ragged tensor list in the object store. Every key here also gets the typed-ragged codec below.
ROLLOUT_DATA_TENSOR_DTYPES = {
    "tokens": "int32",
    "target_tokens": "int32",
    "loss_masks": "int32",
    "rollout_log_probs": "float32",
    "rollout_topk_token_ids": "int32",
    "rollout_topk_lengths": "int32",
    "rollout_topk_log_probs": "float32",
    "rollout_sampling_mask_ids": "int32",
    "rollout_sampling_mask_offsets": "int64",
    "teacher_log_probs": "float32",
    "opd_reverse_kl": "float32",
    "rollout_routed_experts": "int32",
    "rollout_indexer_topk": "int32",
}

# Object store encoding spec per column: typed ragged for every tensor column in the dtype table;
# fixed-dtype arrays for index/length/reward style columns; message-pack ragged for prompt, metadata
# and weight versions; a ragged tensor dict for multimodal inputs; automatic codec for the rest.
ROLLOUT_DATA_VALUE_SPEC: dict[str, ValueSpec] = {
    **{field: ValueSpec(codec="typed_ragged") for field in ROLLOUT_DATA_TENSOR_DTYPES},
    "partition": ValueSpec(codec="ndarray", dtype="int64"),
    "seq_witness_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "response_lengths": ValueSpec(codec="ndarray", dtype="int64"),
    "rewards": ValueSpec(codec="ndarray", dtype="float32"),
    "truncated": ValueSpec(codec="ndarray", dtype="int64"),
    "round_number": ValueSpec(codec="ndarray", dtype="int64"),
    "sample_indices": ValueSpec(codec="ndarray", dtype="int64"),
    "rollout_ids": ValueSpec(codec="ndarray", dtype="int64"),
    "rollout_mask_sums": ValueSpec(codec="ndarray", dtype="int64"),
    "multimodal_train_inputs": ValueSpec(codec="ragged_tensor_dict"),
    "prompt": ValueSpec(codec="msgpack_ragged"),
    "metadata": ValueSpec(codec="msgpack_ragged"),
    "weight_versions": ValueSpec(codec="msgpack_ragged"),
    "raw_reward": ValueSpec(codec="auto"),
    "total_lengths": ValueSpec(codec="auto"),
    "dynamic_global_batch_size": ValueSpec(codec="auto"),
    "num_microbatches": ValueSpec(codec="auto"),
    "micro_batch_indices": ValueSpec(codec="auto"),
    "num_rollouts": ValueSpec(codec="auto"),
}


def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
    metadata: dict[str, Any],
    custom_convert_samples_to_train_data_func,
    custom_reward_post_process_func,
):
    # [FUNCTION DOC] Build the training data mapping (column name to per-sample lists, plus a few
    # batch-level values) from generated samples. If a custom conversion function is given it is
    # called with (args, samples) and its result returned unchanged. Otherwise: post-process rewards
    # (custom hook or default; yields raw and training rewards, each of sample count, asserted).
    # Base columns: tokens, response lengths, rewards, raw reward, truncated flag (1 for truncated
    # status), sample indices, rollout ids (rollout id, else sample index). Loss masks: for each
    # sample, if missing, a mask of ones of the response length is created ON THE SAMPLE (mutation);
    # its length must equal the response length (assertion error); samples flagged for removal get
    # their mask overwritten with zeros (mutation). Per-rollout mask sums are derived. Optional
    # columns decided by the FIRST sample unless noted: raw reward from metadata overrides raw
    # reward; round number from metadata; rollout log probs. When top-k log probs are requested
    # (args), every sample must have top-k token ids, top-k log probs and rollout log probs (value
    # error naming the field and sample index); in CI test mode each sample is validated for k; the
    # top-k token ids and log probs columns are added. If ANY sample has a sampling mask: each
    # sample is validated, must have one (value error otherwise), and its id / offset tensors are
    # collected; when top-k is on, the loss type is score centering and the sampling logprobs mode
    # is support, the candidate ids are compacted. Routed experts column when the first sample has
    # it, else value error if routing replay is enabled; same for indexer top-k with indexer replay.
    # Train metadata column when the first sample has it; multimodal inputs when any sample has
    # them; weight versions (list of dictionaries per call) when any sample has any; teacher log
    # probs and on-policy-distillation reverse KL when the first sample has them. Passes through
    # prompt group sizes from the metadata mapping when present. Finally asserts that the
    # dynamic-global-batch-size launch flag is on exactly when the metadata supplies a value, and
    # copies the value. Returns the mapping.
    """
    Convert inference generated samples to training data.
    """
    if (f := custom_convert_samples_to_train_data_func) is not None:
        return f(args, samples)

    raw_rewards, rewards = _post_process_rewards(
        args,
        samples,
        custom_reward_post_process_func=custom_reward_post_process_func,
        prompt_group_sizes=metadata.get("prompt_group_sizes"),
    )

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
        "rollout_ids": [s.rollout_id if s.rollout_id is not None else s.index for s in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    train_data["rollout_mask_sums"] = _compute_rollout_mask_sums(train_data["rollout_ids"], loss_masks)

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if k := args.rollout_top_logprobs_num:
        for sample in samples:
            for field in ("rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"):
                if getattr(sample, field) is None:
                    raise ValueError(
                        f"Rollout top-k logprobs collection requires {field}; "
                        f"sample_index={sample.index}. Custom rollout producers must record probabilities during generation."
                    )
        if args.ci_test:
            for sample in samples:
                validate_rollout_topk_logprobs_sample(sample, k)
        train_data["rollout_topk_token_ids"] = [sample.rollout_topk_token_ids for sample in samples]
        train_data["rollout_topk_log_probs"] = [sample.rollout_topk_log_probs for sample in samples]

    has_sampling_mask = any(sample.rollout_sampling_mask is not None for sample in samples)
    if has_sampling_mask:
        sampling_mask_ids = []
        sampling_mask_offsets = []
        for position, sample in enumerate(samples):
            sample.validate()
            if sample.rollout_sampling_mask is None:
                raise ValueError(
                    "sampling-mask data must be present for every training sample; "
                    f"missing at position={position}, sample_index={sample.index}, status={sample.status}"
                )
            ids, offsets = sample.rollout_sampling_mask._as_tensors()

            sampling_mask_ids.append(ids)
            sampling_mask_offsets.append(offsets)

        train_data["rollout_sampling_mask_ids"] = sampling_mask_ids
        train_data["rollout_sampling_mask_offsets"] = sampling_mask_offsets
        if k and args.loss_type == "score_centering" and args.rollout_sampling_logprobs_mode == "support":
            _compact_candidate_ids(train_data)

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]
    elif getattr(args, "use_rollout_routing_replay", False):
        raise ValueError(
            "--use-rollout-routing-replay is set but the rollout samples carry no "
            "rollout_routed_experts: the engine response meta_info lacked 'routed_experts'."
        )

    if samples[0].rollout_indexer_topk is not None:
        train_data["rollout_indexer_topk"] = [sample.rollout_indexer_topk for sample in samples]
    elif getattr(args, "use_rollout_indexer_replay", False):
        raise ValueError(
            "--use-rollout-indexer-replay is set but the rollout samples carry no "
            "rollout_indexer_topk: the engine response meta_info lacked 'indexer_topk'."
        )

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if any(sample.weight_versions for sample in samples):
        train_data["weight_versions"] = [[call.to_dicts() for call in sample.weight_versions] for sample in samples]

    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    if (prompt_group_sizes := metadata.get("prompt_group_sizes")) is not None:
        train_data["prompt_group_sizes"] = prompt_group_sizes

    if samples[0].opd_reverse_kl is not None:
        train_data["opd_reverse_kl"] = [sample.opd_reverse_kl for sample in samples]

    x = metadata.get("dynamic_global_batch_size")
    assert args.use_dynamic_global_batch_size == (x is not None)
    if x is not None:
        train_data["dynamic_global_batch_size"] = x

    return train_data


def _compact_candidate_ids(data: dict) -> None:
    # [FUNCTION DOC] HELPER: Optimization that stops duplicating the dense top-k token ids when they
    # coincide with the sampling-mask support ids. Mutates the data mapping. For each sample (parallel
    # lists of dense top-k ids, top-k log probs, support ids, support offsets) it verifies the layout
    # is exactly compactable: dense ids are two-dimensional with the same shape as the log probs and
    # one row per offset interval; valid (non-negative) entries form a left-aligned prefix per row
    # with padding equal to minus one; the per-row valid counts equal the support interval lengths
    # for non-empty rows; the valid dense ids equal the corresponding support ids; and rows with no
    # valid ids have all log probs negative infinity. If ANY check fails for ANY sample, it returns
    # with the data completely unchanged (all or nothing). On success it stores a per-sample list of
    # valid counts under the top-k lengths column and deletes the dense top-k token ids column.
    """Share ordered support IDs in training batches; preserve other producer layouts."""
    lengths = []
    for dense, logps, support, offsets in zip(
        data["rollout_topk_token_ids"],
        data["rollout_topk_log_probs"],
        data["rollout_sampling_mask_ids"],
        data["rollout_sampling_mask_offsets"],
        strict=True,
    ):
        dense, logps = torch.as_tensor(dense), torch.as_tensor(logps)
        if dense.ndim != 2 or dense.shape != logps.shape or dense.size(0) != offsets.numel() - 1:
            return
        valid = dense >= 0
        count = valid.sum(-1, dtype=torch.int32)
        columns = torch.arange(dense.size(1))
        prefix = columns < count[:, None]
        if not torch.equal(valid, prefix) or not (dense[~prefix] == -1).all():
            return
        if not (count[count > 0] == offsets.diff()[count > 0]).all():
            return
        positions = offsets[:-1, None] + columns
        if not torch.equal(dense[prefix], support[positions[prefix]]):
            return
        if not torch.isneginf(logps[count == 0]).all():
            return
        lengths.append(count)
    data["rollout_topk_lengths"] = lengths
    del data["rollout_topk_token_ids"]


def _compute_rollout_mask_sums(rollout_ids: list[int], loss_masks: list[list[int]]) -> list[int]:
    # [FUNCTION DOC] HELPER: Sum each sample's loss mask per rollout id, then return for every sample
    # the total of its rollout (in input order). Lists must have equal length (strict zip).
    """Whole-rollout loss-mask total per sample: every sibling of one rollout carries
    the sum over all of that rollout's samples, so the loss reducer reconstructs one
    token-weighted mean per rollout even when siblings land in different micro-batches."""
    totals: dict[int, int] = {}
    for rid, mask in zip(rollout_ids, loss_masks, strict=True):
        totals[rid] = totals.get(rid, 0) + sum(mask)
    return [totals[rid] for rid in rollout_ids]


def _reward_group_segments(args: Any, samples: list[Sample], prompt_group_sizes: list[int] | None) -> list[list[int]]:
    # [FUNCTION DOC] HELPER: Partition row indices into prompt reward groups, trying in order:
    # (1) explicit prompt group sizes (multi-LoRA): asserts they sum to the sample count, and makes
    # consecutive index ranges, skipping groups of size zero; (2) group index on every sample (all
    # must be non-null): rows grouped by integer group index in first-appearance order; (3) legacy
    # fixed fan-out: if the sample count equals samples per prompt times rollout batch size, contiguous
    # blocks of samples per prompt; (4) otherwise a single group of all rows.
    """Return the flattened row indices for each prompt reward group."""
    # Multi-LoRA records explicit prompt boundaries before flattening.
    if prompt_group_sizes is not None:
        assert sum(prompt_group_sizes) == len(
            samples
        ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(samples)} rewards"
        groups: list[list[int]] = []
        start = 0
        for size in prompt_group_sizes:
            end = start + size
            if size > 0:
                groups.append(list(range(start, end)))
            start = end
        return groups

    # Standard rollout samples carry their prompt identity in `group_index`.
    group_indices = [sample.group_index for sample in samples]
    if all(group_index is not None for group_index in group_indices):
        segments_by_group_index: dict[int, list[int]] = {}
        for segment_index, group_index in enumerate(group_indices):
            segments_by_group_index.setdefault(int(group_index), []).append(segment_index)
        return list(segments_by_group_index.values())

    # Legacy fixed-fanout batches store each prompt's segments contiguously.
    expected_samples = args.n_samples_per_prompt * args.rollout_batch_size
    if len(samples) == expected_samples:
        return [
            list(range(start, start + args.n_samples_per_prompt))
            for start in range(0, len(samples), args.n_samples_per_prompt)
        ]
    # Without prompt identities or a complete fixed layout, use one reward group.
    return [list(range(len(samples)))]


def _normalize_rewards_by_rollout(
    args: Any,
    samples: list[Sample],
    raw_rewards: list[float],
    prompt_group_sizes: list[int] | None,
) -> list[float]:
    # [FUNCTION DOC] HELPER: Group-baseline reward normalization where all siblings of one rollout
    # share one reward. Returns a list of floats, one per sample (empty for no samples). For each
    # prompt group (see segment helper): group rows by rollout key (rollout id, else sample index,
    # else the row position); all rows of a rollout must have identical raw rewards (value error
    # otherwise). The per-rollout rewards are centered by subtracting their mean; if the estimator
    # is grpo or gspo, std normalization is on, and the group has more than one rollout, they are
    # divided by (std + 1e-6) when std is positive. The normalized value is broadcast to each
    # rollout's rows. Single-rollout groups end up centered to zero.
    """Normalize one shared reward per rollout, then broadcast it to siblings."""
    if not samples:
        return []

    normalized_rewards = torch.empty(len(raw_rewards), dtype=torch.float)
    for prompt_segments in _reward_group_segments(args, samples, prompt_group_sizes):
        segments_by_rollout_key: dict[int | tuple[str, int], list[int]] = {}
        for segment_index in prompt_segments:
            sample = samples[segment_index]
            if sample.rollout_id is not None:
                rollout_key = sample.rollout_id
            elif sample.index is not None:
                rollout_key = sample.index
            else:
                rollout_key = ("row", segment_index)
            segments_by_rollout_key.setdefault(rollout_key, []).append(segment_index)

        rollout_segment_groups = list(segments_by_rollout_key.items())
        shared_rewards: list[float] = []
        for rollout_key, rollout_segments in rollout_segment_groups:
            sibling_rewards = [raw_rewards[segment_index] for segment_index in rollout_segments]
            if any(reward != sibling_rewards[0] for reward in sibling_rewards[1:]):
                raise ValueError(
                    f"all samples in rollout {rollout_key!r} must share one reward; "
                    f"rows {rollout_segments} have rewards {sibling_rewards}"
                )
            shared_rewards.append(sibling_rewards[0])

        rollout_rewards = torch.tensor(shared_rewards, dtype=torch.float)
        normalized_rollout_rewards = rollout_rewards - rollout_rewards.mean()
        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization and len(rollout_rewards) > 1:
            rollout_std = rollout_rewards.std()
            if rollout_std > 0:
                normalized_rollout_rewards = normalized_rollout_rewards / (rollout_std + 1e-6)

        for (_, rollout_segments), normalized_reward in zip(
            rollout_segment_groups, normalized_rollout_rewards.tolist(), strict=True
        ):
            for segment_index in rollout_segments:
                normalized_rewards[segment_index] = normalized_reward

    return normalized_rewards.tolist()


def _post_process_rewards(
    args,
    samples: list[Sample] | list[list[Sample]],
    custom_reward_post_process_func,
    prompt_group_sizes: list[int] | None = None,
):
    # [FUNCTION DOC] HELPER: Returns a pair (raw rewards, training rewards). A custom post-process
    # function, if given, is called with (args, samples) and its result returned as is. Otherwise raw
    # rewards are each sample's reward value under args; if the estimator is grpo, gspo or
    # reinforce plus plus baseline and reward normalization is on, the second element is the
    # rollout-normalized rewards; else it is the same list as raw (identical object).
    if (f := custom_reward_post_process_func) is not None:
        return f(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        normalized_rewards = _normalize_rewards_by_rollout(args, samples, raw_rewards, prompt_group_sizes)
        return raw_rewards, normalized_rewards

    return raw_rewards, raw_rewards


def split_train_data_by_dp(args, data: dict[str, Any], train_parallel_config: dict | None):
    # [FUNCTION DOC] Split the batch into one shard per data-parallel rank and store each in the object
    # store using the shared value spec. Uses the scheduled split when the rollout side can schedule,
    # otherwise the legacy split with the data-parallel size from the config (a missing config or
    # missing size raises there). Returns a list of object store references, one per rank in rank
    # order. Mutates the input mapping by adding total lengths.
    """Split the train data across DP ranks and put the shards into the object store.

    When the training backend can consume a rollout-side schedule, the shards
    also carry the precomputed micro-batch layout; otherwise this falls back to
    the legacy split (the training side schedules locally)."""
    if can_schedule_on_rollout_side(args, data, train_parallel_config):
        shards = split_train_data_by_dp_scheduled_raw(args, data, train_parallel_config=train_parallel_config)
    else:
        shards = split_train_data_by_dp_raw(args, data, dp_size=train_parallel_config["dp_size"])
    store = object_store.get_instance()
    return [store.put(value=shard, value_spec=ROLLOUT_DATA_VALUE_SPEC) for shard in shards]


def can_schedule_on_rollout_side(args, data: dict[str, Any], train_parallel_config: dict | None) -> bool:
    # [FUNCTION DOC] True only if all hold: the parallel config has everything needed for a full
    # schedule; multi-LoRA is not enabled; the data has no multimodal inputs; the data has rollout
    # ids; and the number of distinct rollout ids is at least the global batch size (the dynamic one
    # from the data if present, else from args).
    """Whether the rollout side can precompute the full DP/mbs schedule."""
    if not has_full_schedule_config(train_parallel_config):
        return False
    if is_multi_lora_enabled(args):
        return False
    if "multimodal_train_inputs" in data:
        return False
    if "rollout_ids" not in data:
        return False
    global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    return len(set(data["rollout_ids"])) >= global_batch_size


def split_train_data_by_dp_scheduled_raw(
    args, data: dict[str, Any], *, train_parallel_config: dict
) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Compute total lengths (token count per sample; stored into the input mapping,
    # mutation), choose the global batch size (dynamic one from data if present, else args), and ask
    # the schedule builder for per-rank partitions, per-rank micro-batch indices, the number of
    # micro-batches and the number of rollouts, using rollout ids as grouping. Logs these. Packages
    # shards from the partitions and adds to each: the shared micro-batch count, that rank's micro-
    # batch indices and the shared rollout count. Returns the list of shard mappings in rank order.
    """DP split with the micro-batch schedule precomputed on the rollout side."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    global_batch_size = data.get("dynamic_global_batch_size", args.global_batch_size)
    partitions, micro_batch_indices, num_microbatches, num_rollouts = build_dp_schedule(
        args,
        train_parallel_config,
        total_lengths,
        global_batch_size=global_batch_size,
        rollout_indices=data["rollout_ids"],
    )
    logger.info(
        f"Rollout-side DP schedule: num_samples={len(total_lengths)}, "
        f"num_rollouts={num_rollouts}, num_microbatches={num_microbatches}"
    )

    shards = _package_shards(args, data, partitions)
    for rank, shard in enumerate(shards):
        shard["num_microbatches"] = num_microbatches
        shard["micro_batch_indices"] = micro_batch_indices[rank]
        shard["num_rollouts"] = num_rollouts
    return shards


def split_train_data_by_dp_raw(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Legacy split (training side schedules micro-batches itself). Stores total lengths
    # into the input mapping (mutation). Partitions: with data balancing on, sequence-length-balanced
    # equal-size partitions; otherwise round-robin (rank i gets rows i, i + dp, ...). If the data
    # has adapter slots (multi-LoRA), each partition is sorted by adapter slot so micro-batches are
    # contiguous per slot. Returns the packaged shards, one per rank.
    """Split the train data by data parallel size."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    # Multi-LoRA: sort partitions by adapter slot so each microbatch is
    # contiguous-by-slot (required by the per-adapter token-count math).
    adapter_slots = data.get("adapter_slots")
    if adapter_slots is not None:
        partitions = [sorted(p, key=lambda i: adapter_slots[i]) for p in partitions]

    return _package_shards(args, data, partitions)


def _package_shards(args, data: dict[str, Any], partitions) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Build one shard mapping per partition. Each shard holds the partition
    # itself, then for a fixed list of per-sample columns (those present in the data) the values
    # selected by the partition's row indices in partition order; and, for batch-level columns that
    # the train side splits itself (raw reward, total lengths, dynamic global batch size, prompt
    # group sizes, loss function and its config), the full unsplit value. If the shard has adapter
    # slots, it also records the configured number of adapters. Note raw reward and total lengths
    # remain batch-global and are reordered later by the train-side completion step.
    """Package one rollout_data shard per DP rank from precomputed partitions."""
    shards = []

    for i in range(len(partitions)):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        for key in [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "rollout_ids",
            "rollout_mask_sums",
            "rollout_log_probs",
            "rollout_topk_token_ids",
            "rollout_topk_lengths",
            "rollout_topk_log_probs",
            "rollout_sampling_mask_ids",
            "rollout_sampling_mask_offsets",
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "opd_reverse_kl",
            "seq_witness_ids",
            "weight_versions",
            "adapter_slots",
            "loss_weights",
            "advantages",
            "target_tokens",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
            "dynamic_global_batch_size",
            "prompt_group_sizes",
            "loss_fn",
            "loss_fn_config",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        if "adapter_slots" in rollout_data:
            rollout_data["n_adapters"] = args.multi_lora_n_adapters
        shards.append(rollout_data)
    return shards


def process_rollout_data_shard(args, rollout_data):
    # [FUNCTION DOC] Finish a received shard on the training side: remove the partition entry
    # (mutation), publish the batch-global total lengths on the global timer object as its sequence
    # lengths, then replace total lengths in the shard by those of this shard's rows in partition
    # order. Returns the same mapping. The args parameter is unused.
    """Train-side completion of the DP split: drop the ``partition`` key and
    reorder the batch-global ``total_lengths`` into this shard's row order."""
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
