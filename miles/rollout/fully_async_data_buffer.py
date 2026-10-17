# [MODULE DOC] Buffering between producers (the fully-async rollout worker) and consumers (training
# steps). Contains: the CLI argument registration for per-model custom buffer paths; the abstract
# buffer contract (put, get, metrics); a default FIFO buffer that filters groups on put (aborted,
# missing reward, dynamic filter) and on get (stale), applies a capacity bound and collects staleness
# and filter metrics; and a composite that holds one default (or custom) buffer per trained policy and
# routes groups by the trainer model id stamped on each sample. A group is a list of samples (or of
# lists of samples).
"""Data buffer between fully-async rollout production and training consumption.

``DataBuffer`` is the contract (put / get / get_metrics); ``DefaultDataBuffer``
is the built-in implementation, replaceable via ``--custom-async-data-buffer-path``.
Every group-level decision lives here — what to keep, what to hand to
``--async-unused-samples-handler`` — so a custom buffer owns all of it. Only
``--rollout-sample-filter-path`` stays outside: it runs on the assembled batch.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter, iter_samples
from miles.rollout.filter_hub.common_filters import (
    GroupWeightVersionStats,
    apply_aborted_filter,
    apply_missing_reward_filter,
    group_staleness,
    group_weight_version_stats,
)
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# A finished group is list[Sample], or list[list[Sample]] when a generate function
# returns multiple samples per trajectory (e.g. multi-agent).
Group = list[Sample | list[Sample]]

DATA_BUFFER_PATH_PER_MODEL_FLAG = "--custom-async-data-buffer-path-per-model"


def add_data_buffer_arguments(parser: ArgumentParser) -> None:
    # [FUNCTION DOC] Register the per-model custom buffer path option on the argument parser: a list of
    # "MODEL_ID=PATH" strings, default none, with help text explaining that named models use the given
    # class and other models keep the built-in one. Mutates the parser; returns nothing.
    parser.add_argument(
        DATA_BUFFER_PATH_PER_MODEL_FLAG,
        type=str,
        nargs="+",
        default=None,
        metavar="MODEL_ID=PATH",
        help=(
            "Per policy form of --custom-async-data-buffer-path, e.g. "
            "--custom-async-data-buffer-path-per-model solver=pkg.SolverBuffer. A run training several "
            "policies composes one buffer per policy (see DefaultMultiDataBuffer); each model id named "
            "here gets that class instead of the built-in one, and every model id left out keeps it. "
            "The model ids are the --megatron-config ones."
        ),
    )


# =================================== shared ===================================


def first_sample(group: Group) -> Sample:
    # [FUNCTION DOC] The first sample of a group, descending one level if the first element is a list.
    # An empty group raises an index error.
    return group[0][0] if isinstance(group[0], list) else group[0]


@dataclass(frozen=True)
# ================================== contract ==================================


class DataBufferConstructorInput:
    # [CLASS DOC] Frozen constructor input of buffers: the launch args and the callback applied to the
    # prompt group of every group that is not trained on (it either drops it or recycles it for
    # regeneration, depending on the configured handler). KNOWN ISSUE: the frozen dataclass decorator
    # is separated from the class by a banner comment; harmless since comments are legal there.
    args: Namespace
    unused_handler_fn: Callable[[list[Sample]], None]  # --async-unused-samples-handler, applied to unused groups


@dataclass
class DataBufferInput:
    # [CLASS DOC] Mutable entry stored in buffers: the original prompt samples (kept so the prompt can be
    # resubmitted) and the finished group generated from them.
    prompt_group: list[Sample]  # resubmittable, for recycling
    group: Group  # finished samples


class DataBuffer(ABC):
    # [CLASS DOC] Abstract contract of buffers: asynchronous put and get, synchronous per-policy metrics.
    """Store for finished groups between rollout production and training consumption.

    The producer puts each finished group as it completes; the consumer gets one
    group at a time; get_metrics is collected once per training step. Storage,
    ordering, and filtering are invisible to callers — an implementation is free
    to reject a group on put, on get, or not at all.
    """

    @abstractmethod
    async def put(self, input: DataBufferInput) -> None:
        # [FUNCTION DOC] Abstract: accept a finished group; implementations may store, reject or evict,
        # and may block the producer when full.
        """Accept a finished group; may store it, reject it, or evict to make room."""

    @abstractmethod
    async def get(self, **context) -> DataBufferInput:
        # [FUNCTION DOC] Abstract: wait for and return one group to train on. The context keywords carry
        # at least the trainer model id and, for the default buffer, the current weight version.
        """Return one group to train on, waiting until one is available.

        ``context`` is the extra information for sample processing at get() time,
        including the ``trainer_model_id`` whose groups are asked for.
        """

    @abstractmethod
    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        # [FUNCTION DOC] Abstract: return the mapping of metric name to number for one policy since the
        # previous call, resetting its window counters.
        """Report the metrics of one policy since its previous call (its window counters reset here)."""


# ============================= one policy buffer ==============================


class DefaultDataBuffer(DataBuffer):
    # [CLASS DOC] Single-policy FIFO buffer. State: args; the list of entries (oldest first); the capacity
    # (floor of capacity factor times rollout batch size, at least 1); the unused-group callback; the
    # dynamic filter function loaded from args (may be none); one async condition guarding the list and
    # signalling both producers (space) and consumers (data); the last seen weight version; a metric
    # gatherer for filter drops; and window counters/lists for metrics (aborted groups, stale groups,
    # consumed staleness, newest lag, version span, token-weighted lag sums, sample coverage), all reset
    # by the metrics call.
    """FIFO buffer of finished groups, filtering out what training should not see.

    Rejected on put, because the verdict is fixed once the group is generated:

    - aborted groups (the generate function gave up, e.g. an agentic collect timeout)
    - groups with a missing reward
    - groups ``--dynamic-sampling-filter-path`` does not keep

    Rejected on get, because staleness depends on when the group is consumed:

    - groups beyond ``--max-weight-staleness``

    Dataflow control options:

    (1) capacity: ``--async-data-buffer-capacity-factor`` bounds the buffer at
        floor(factor * rollout_batch_size) groups; when full, put blocks until
        training consumes.
    (2) unused handling: ``--async-unused-samples-handler`` decides what happens
        to aborted and stale groups: drop discards them, retry recycles their
        prompts for regeneration. Missing-reward and custom-filter rejections
        are discarded directly.
    """

    def __init__(self, input: DataBufferConstructorInput):
        # [FUNCTION DOC] Store args, create the empty buffer, compute capacity as the integer part of
        # capacity factor times rollout batch size (assertion errors if the factor is not positive or
        # the capacity is below 1), keep the unused-group callback, load the dynamic filter function by
        # its configured path, create the condition (needs a running or later-running event loop on
        # first use), and zero all metric state.
        args = input.args
        self._args = args

        self._buffer: list[DataBufferInput] = []
        assert args.async_data_buffer_capacity_factor > 0
        self._capacity = int(args.async_data_buffer_capacity_factor * args.rollout_batch_size)
        assert self._capacity >= 1

        self._unused_handler_fn = input.unused_handler_fn
        self._dynamic_filter = load_function(args.dynamic_sampling_filter_path)
        self._cond = asyncio.Condition()
        self._current_version: int | None = None

        self._metric_gatherer = MetricGatherer()
        self._metric_aborted_groups = 0
        self._metric_stale_groups = 0
        self._metric_consumed_staleness: list[int] = []
        self._metric_selected_newest_lag: list[int] = []
        self._metric_selected_version_span: list[int] = []
        self._metric_selected_token_lag_sum = 0.0
        self._metric_selected_versioned_tokens = 0
        self._metric_selected_samples = 0
        self._metric_selected_versioned_samples = 0

    async def put(self, input: DataBufferInput) -> None:
        # [FUNCTION DOC] Run the put-time filters (a rejected group is not stored and put returns at once).
        # Otherwise, under the condition, wait while the buffer is at capacity (blocking the producer),
        # append the entry at the tail and wake all waiters.
        if not self._preput_filter(input):
            return

        async with self._cond:
            while len(self._buffer) >= self._capacity:
                await self._cond.wait()
            self._buffer.append(input)
            self._cond.notify_all()

    def _preput_filter(self, input: DataBufferInput) -> bool:
        # [FUNCTION DOC] HELPER: Return whether to store the group. In order: (1) aborted filter: if the
        # group has an aborted sample, count it, hand the prompt group to the unused callback (so it may
        # be recycled) and reject; (2) missing reward filter: reject and count the drop reason, WITHOUT
        # calling the unused callback (prompt is discarded); (3) record the group's rewards in the
        # unfiltered reward statistic; (4) dynamic filter (if any): reject and count the reason, also
        # without the unused callback. Otherwise accept.
        output = apply_aborted_filter(self._args, input.group)
        if not output.keep:
            self._metric_aborted_groups += 1
            self._unused_handler_fn(input.prompt_group)
            return False

        output = apply_missing_reward_filter(self._args, input.group)
        if not output.keep:
            self._metric_gatherer.on_dynamic_filter_drop(reason=output.reason)
            return False

        self._metric_gatherer.on_group_before_dynamic_filter(self._args, input.group)
        output = call_dynamic_filter(self._dynamic_filter, self._args, input.group)
        if not output.keep:
            self._metric_gatherer.on_dynamic_filter_drop(reason=output.reason)
            return False
        return True

    async def get(self, current_version: int | None = None, **_) -> DataBufferInput:
        # [FUNCTION DOC] Wait for and return the oldest acceptable group. Remembers the given current
        # weight version if not none (used later by the metrics for buffered-group staleness). Loop under
        # the condition: wait until the buffer is non-empty; pop the oldest entry and wake producers;
        # compute the group's version statistics and staleness against the version PASSED TO THIS CALL
        # (not the remembered one). If staleness is known: if a maximum staleness is configured and
        # exceeded, log, count a stale group, hand the prompt group to the unused callback and continue
        # with the next entry; else record the staleness for metrics. Record selected-version metrics
        # and return the entry. Other context keywords are ignored.
        if current_version is not None:
            self._current_version = current_version
        async with self._cond:
            while True:
                while not self._buffer:
                    await self._cond.wait()
                entry = self._buffer.pop(0)
                self._cond.notify_all()  # wake producers blocked on a full buffer

                version_stats = group_weight_version_stats(entry.group)
                staleness = version_stats.oldest_lag(current_version)
                if staleness is not None:
                    if self._args.max_weight_staleness is not None and staleness > self._args.max_weight_staleness:
                        logger.info(f"Filtered stale group ({staleness=} > max={self._args.max_weight_staleness})")
                        self._metric_stale_groups += 1
                        self._unused_handler_fn(entry.prompt_group)
                        continue
                    self._metric_consumed_staleness.append(staleness)
                self._record_selected_version_stats(version_stats, current_version)
                return entry

    def _record_selected_version_stats(
        self,
        stats: GroupWeightVersionStats,
        current_version: int | None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Update window metrics for a group handed to training: add its sample and
        # versioned-sample counts; if oldest and newest versions are known append the newest-minus-oldest
        # span; if the newest lag against the current version is known append it; if the token-weighted
        # lag is known add lag times versioned tokens to the lag sum and the tokens to the token count.
        self._metric_selected_samples += stats.sample_count
        self._metric_selected_versioned_samples += stats.versioned_sample_count

        if stats.oldest_version is not None and stats.newest_version is not None:
            self._metric_selected_version_span.append(stats.newest_version - stats.oldest_version)

        newest_lag = stats.newest_lag(current_version)
        if newest_lag is not None:
            self._metric_selected_newest_lag.append(newest_lag)

        token_weighted_lag = stats.token_weighted_lag(current_version)
        if token_weighted_lag is not None:
            self._metric_selected_token_lag_sum += token_weighted_lag * stats.versioned_token_count
            self._metric_selected_versioned_tokens += stats.versioned_token_count

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        # [FUNCTION DOC] Build the metrics mapping (keys prefixed rollout/fully_async/): queue size, aborted
        # and stale group counts, the gatherer's filter drop and unfiltered reward metrics; when data
        # exists: mean and max consumed staleness, mean and max post-generation staleness (newest lag),
        # mean and max generation version span, token-weighted staleness, weight-version sample
        # coverage; and, from the groups currently buffered, mean and max staleness against the
        # remembered current version. Then resets every window metric (new gatherer, empty lists, zero
        # counters). The trainer model id is ignored. Returns the mapping.
        prefix = "rollout/fully_async/"
        metrics = {
            f"{prefix}queue_size": len(self._buffer),
            f"{prefix}aborted_groups_filtered": self._metric_aborted_groups,
            f"{prefix}stale_groups_filtered": self._metric_stale_groups,
            **self._metric_gatherer.collect(),
        }
        if consumed := self._metric_consumed_staleness:
            metrics[f"{prefix}avg_staleness"] = sum(consumed) / len(consumed)
            metrics[f"{prefix}max_staleness"] = max(consumed)
        if newest_lag := self._metric_selected_newest_lag:
            metrics[f"{prefix}avg_post_generation_staleness"] = sum(newest_lag) / len(newest_lag)
            metrics[f"{prefix}max_post_generation_staleness"] = max(newest_lag)
        if version_span := self._metric_selected_version_span:
            metrics[f"{prefix}avg_generation_version_span"] = sum(version_span) / len(version_span)
            metrics[f"{prefix}max_generation_version_span"] = max(version_span)
        if self._metric_selected_versioned_tokens:
            metrics[f"{prefix}token_weighted_staleness"] = (
                self._metric_selected_token_lag_sum / self._metric_selected_versioned_tokens
            )
        if self._metric_selected_samples:
            metrics[f"{prefix}weight_version_sample_coverage"] = (
                self._metric_selected_versioned_samples / self._metric_selected_samples
            )
        buffered = [
            s for entry in self._buffer if (s := group_staleness(entry.group, self._current_version)) is not None
        ]
        if buffered:
            metrics[f"{prefix}buffer_avg_staleness"] = sum(buffered) / len(buffered)
            metrics[f"{prefix}buffer_max_staleness"] = max(buffered)

        self._metric_gatherer = MetricGatherer()
        self._metric_consumed_staleness = []
        self._metric_selected_newest_lag = []
        self._metric_selected_version_span = []
        self._metric_selected_token_lag_sum = 0.0
        self._metric_selected_versioned_tokens = 0
        self._metric_selected_samples = 0
        self._metric_selected_versioned_samples = 0
        self._metric_aborted_groups = self._metric_stale_groups = 0
        return metrics


# ============================ multi policy buffer =============================


class DefaultMultiDataBuffer(DataBuffer):
    # [CLASS DOC] Composite buffer holding one inner buffer per trained policy model id (the model ids of
    # the Megatron config), routing puts by each sample's trainer model id and gets/metrics by the
    # requested id. Inner buffers are the built-in default unless the per-model option names another
    # class.
    """One plain ``DefaultDataBuffer`` per policy model, composed.

    Each policy consumes at its own pace, so each gets its own capacity, staleness accounting and
    metrics, and the single-policy buffer stays untouched.
    """

    def __init__(self, input: DataBufferConstructorInput):
        # [FUNCTION DOC] Parse the per-model path option, read the model ids from the resolved Megatron
        # config, assert that every named model id is one of them (assertion error listing the unknown
        # ids), and build one inner buffer per model id from the same constructor input: the class
        # loaded from the model's path, or the default buffer when none is given.
        paths = _parse_data_buffer_paths(input.args.custom_async_data_buffer_path_per_model)
        model_ids = resolve_megatron_config(input.args).model_ids
        assert not (unknown := sorted(set(paths) - set(model_ids))), (
            f"{DATA_BUFFER_PATH_PER_MODEL_FLAG} names {unknown}, which train no policy of this run "
            f"({sorted(model_ids)})"
        )
        self._inners: dict[str, DataBuffer] = {
            model_id: (load_function(paths.get(model_id)) or DefaultDataBuffer)(input) for model_id in model_ids
        }

    async def put(self, input: DataBufferInput) -> None:
        # [FUNCTION DOC] Split the group by trainer model id (see split helper) and put each part into its
        # policy's inner buffer sequentially. A full inner buffer blocks the whole put, so it delays
        # other policies' parts too (acknowledged by the TODO).
        # TODO: a full inner blocks the one producer for every policy; give each policy its own dispatcher
        for trainer_model_id, entry in _split_by_trainer_model_id(input).items():
            await self._inner_of(trainer_model_id).put(entry)

    async def get(self, trainer_model_id: str | None = None, **context) -> DataBufferInput:
        # [FUNCTION DOC] Delegate to the inner buffer of the model id (assertion error if unknown),
        # passing the model id and the other context along.
        return await self._inner_of(trainer_model_id).get(trainer_model_id=trainer_model_id, **context)

    def get_metrics(self, trainer_model_id: str | None = None) -> dict[str, float]:
        # [FUNCTION DOC] Return the metrics of the model's inner buffer (resetting its window).
        return self._inner_of(trainer_model_id).get_metrics(trainer_model_id=trainer_model_id)

    def _inner_of(self, trainer_model_id: str | None) -> DataBuffer:
        # [FUNCTION DOC] HELPER: The inner buffer for the model id; assertion error if the id (including
        # none) trains no policy of this run.
        assert trainer_model_id in self._inners, (
            f"trainer_model_id {trainer_model_id!r} trains no policy of this run ({sorted(self._inners)}), so "
            f"its groups would queue up in a buffer nobody drains"
        )
        return self._inners[trainer_model_id]


# TODO: a policy absent from a trajectory shortens its group below n_samples_per_prompt, which the drain refuses
def _parse_data_buffer_paths(values: Iterable[str] | None) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Parse "MODEL_ID=PATH" entries (none = no entries) into a mapping from model
    # id to path. Each part is whitespace-stripped (split at the first equals sign). Raises value error
    # for an entry lacking the separator or with an empty id or path, and for a repeated model id.
    ans: dict[str, str] = {}
    for value in values or []:
        model_id, separator, path = value.partition("=")
        model_id, path = model_id.strip(), path.strip()
        if not separator or not model_id or not path:
            raise ValueError(f"Invalid {DATA_BUFFER_PATH_PER_MODEL_FLAG} entry {value!r}; expected MODEL_ID=PATH.")
        if model_id in ans:
            raise ValueError(f"Duplicate model id {model_id!r} in {DATA_BUFFER_PATH_PER_MODEL_FLAG}.")
        ans[model_id] = path
    return ans


def _split_by_trainer_model_id(input: DataBufferInput) -> dict[str, DataBufferInput]:
    # [FUNCTION DOC] HELPER: Collect the distinct trainer model ids of the group's samples in order of
    # first appearance (assertion error if any sample has none). Returns a mapping from id to a buffer
    # entry with the SAME (shared) prompt group and a group filtered to that id's samples (see filter
    # helper). The prompt group is therefore recycled per policy part if each is unused.
    trainer_model_ids = list(dict.fromkeys(sample.trainer_model_id for sample in iter_samples(input.group)))
    assert None not in trainer_model_ids, (
        f"a multi policy run routes every group by the policy it belongs to, so the generate function must stamp "
        f"every sample with its trainer_model_id, but this group carries {trainer_model_ids}"
    )
    return {
        trainer_model_id: DataBufferInput(
            prompt_group=input.prompt_group, group=_filter_group(input.group, trainer_model_id=trainer_model_id)
        )
        for trainer_model_id in trainer_model_ids
    }


def _filter_group(group: Group, *, trainer_model_id: str) -> Group:
    # [FUNCTION DOC] HELPER: Keep only samples of the given trainer model id. A plain element is kept if
    # it matches; a nested list is replaced by the list of its matching samples and dropped entirely if
    # none match. Order is preserved.
    ans: Group = []
    for item in group:
        if isinstance(item, list):
            if kept := [sample for sample in item if sample.trainer_model_id == trainer_model_id]:
                ans.append(kept)
        else:
            if item.trainer_model_id == trainer_model_id:
                ans.append(item)
    return ans
