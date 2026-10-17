# [MODULE DOC] Fully asynchronous rollout function: a background worker task on the rollout event loop
# keeps up to a bounded number of prompt groups generating at once and pushes each finished group into
# a data buffer; each training call only drains the required number of groups from the buffer, sorts
# them by sample index, applies an optional batch filter and returns them with buffer metrics.
# Evaluation either runs on a dedicated generation state if one is passed, or on the shared engines
# while pausing producer submissions. A "no progress" warning is logged every 30 s while waiting for
# groups.
"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging
from dataclasses import replace

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


class FullyAsyncRolloutFn(BaseRolloutFn):
    # [CLASS DOC] Stateful rollout function. State: args; the data source; one generation state shared
    # by producer and shared-engine eval; the submission scheduler (default sample-level backfill); the
    # handler for unused groups (recycle for "retry", no-op for "drop"); an optional batch sample
    # filter; the worker task (created lazily by the first training call); a cache of evaluation
    # prompt datasets; an event that producers wait on (cleared while a shared-engine eval runs); and
    # the output buffer (created with the worker). Exposes a static hook to register extra CLI
    # arguments (the buffer options).
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    add_arguments = staticmethod(add_data_buffer_arguments)

    def __init__(self, input: RolloutFnConstructorInput):
        # [FUNCTION DOC] Initialize the base, keep args and data source, create the generation state,
        # the submission scheduler (default kind: sample), assert the unused-samples handler option is
        # "retry" or "drop" (assertion error otherwise) and derive the unused handler, load the optional
        # batch sample filter by path, and set up empty worker/cache/output state with the producer gate
        # initially open. Must be constructed where an event loop can be attached (creates an event).
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else (lambda prompt_group: None)
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        # [FUNCTION DOC] Evaluation inputs go to the eval path. For training: on the first call, create
        # the output buffer (the custom class from the configured path, else the multi-policy default for
        # a multi-policy run, else the single default; constructed with args and the unused handler),
        # start the long-lived worker task on the current loop and log. Then drain the required groups
        # from the buffer and return the training output.
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            default_buffer_cls = (
                DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
            )
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
            self._output = buffer_cls(
                DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
            )
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def dispose(self) -> None:
        # [FUNCTION DOC] If a worker was started, cancel it and wait for it to finish, scheduling that on
        # the worker's own event loop in a thread-safe way (the caller may be on another loop) and
        # awaiting the result here. Nothing happens if no worker exists. The buffer is not disposed.
        if (worker := self._worker) is None:
            return
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_end_worker(worker), worker.get_loop()))

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        # [FUNCTION DOC] HELPER: Evaluate all eval datasets and return them as an evaluation output (data
        # only). If the input carries a generation state (dedicated eval engines), use it directly
        # without touching the producer. Otherwise: log and clear the producer gate so no new groups are
        # submitted, run the datasets on this function's own state (shared engines), and always reopen
        # the gate afterwards (even on error), logging. In-flight groups continue generating.
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        # [FUNCTION DOC] HELPER: Cap on concurrently generating groups: if a maximum concurrent sample
        # count is configured, it divided (integer division, floored) by samples per prompt, at least 1;
        # else the rollout batch size.
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> tuple[asyncio.Task, list[Sample]]:
        # [FUNCTION DOC] HELPER: Draw one prompt group from the data source, notify the scheduler of the
        # submission, and start generating it as a background task. Returns (task, prompt group).
        samples = self.data_source.get_samples(1)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        return asyncio.create_task(self._generate_group(prompt_group)), prompt_group

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        # [FUNCTION DOC] HELPER: Generate and score one group with the shared state, a copy of the state's
        # sampling parameters, not evaluation, and the scheduler's per-sample completion callback.
        # Returns a buffer entry pairing the original prompt group with the generated group.
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(prompt_group=prompt_group, group=result)

    async def _worker_loop(self) -> None:
        # [FUNCTION DOC] HELPER: Producer loop, never returns normally (ends by cancellation or by an
        # exception, which fails the next consumer wait). Forever: wait for the producer gate; while the
        # scheduler reports capacity for the current number of active groups under the group budget,
        # submit new groups; wait for scheduler-defined progress; for every finished task, convert it to a
        # buffer entry (see collect helper) and put it into the output buffer, which may block on a full
        # buffer. A failed generation task raises here and kills the worker.
        active: dict[asyncio.Task, list[Sample]] = {}
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                task, prompt_group = self._submit_one_group()
                active[task] = prompt_group
            done, _ = await self._scheduler.wait_for_progress(set(active))
            for task in done:
                entry = self._collect_group_result(task, active.pop(task))
                await self._output.put(entry)

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        # [FUNCTION DOC] HELPER: For a finished task that was not cancelled, return its result (re-raising
        # the task's exception if it failed). For a cancelled task, log a warning with the sample indices
        # and return an entry whose group is copies of the prompt samples with aborted status, so the
        # buffer's aborted filter hands the prompt to the unused handler.
        if not task.cancelled():
            return task.result()

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
        )

    # -------------------------- consumer --------------------------

    async def _next_group(self, *, current_version: int | None, trainer_model_id: str | None) -> DataBufferInput:
        # [FUNCTION DOC] HELPER: Fetch one group from the buffer for the trainer model with the current
        # weight version, while watching the worker. Loops waiting on either the fetch or the worker with
        # a 30 s timeout. If the worker finished first (checked before the fetch result): raise runtime
        # error if it was cancelled (disposed), re-raise its exception if it failed, else raise runtime
        # error that it exited without error. If the fetch finished, return its result. On timeout log a
        # no-progress warning and keep waiting. On exit cancels the pending fetch if not done.
        queue_get = asyncio.create_task(
            self._output.get(current_version=current_version, trainer_model_id=trainer_model_id)
        )
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    if self._worker.cancelled():
                        raise RuntimeError("fully-async rollout was disposed while a step waited for groups")
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        # [FUNCTION DOC] HELPER: Collect exactly the rollout batch size of groups for one training step.
        # Asserts a global dataset is used. Repeatedly fetches the next group (with the input's weight
        # version and trainer model id), asserting its length equals samples per prompt (assertion error
        # otherwise; that can fail in multi-policy runs where a policy is absent from a trajectory),
        # logging a preview of the first one once. After enough groups, logs the last group's preview,
        # sorts groups ascending by the first sample's index, applies the optional batch sample filter
        # (which may mutate the list in place), and returns a training output with the groups and the
        # buffer's metrics for the trainer model.
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        do_print = True

        while len(data) < target_data_size:
            entry = await self._next_group(
                current_version=input.weight_version, trainer_model_id=input.trainer_model_id
            )
            assert len(entry.group) == args.n_samples_per_prompt

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        return RolloutFnTrainOutput(samples=data, metrics=self._output.get_metrics(input.trainer_model_id))

    def _recycle(self, prompt_group: list[Sample]) -> None:
        # [FUNCTION DOC] HELPER: Make an unused prompt group available for regeneration: reset each sample
        # for retry (mutation) and add the group back to the data source. Requires a data source that
        # accepts samples; the plain dataset source raises there.
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])


async def _end_worker(worker: asyncio.Task) -> None:
    # [FUNCTION DOC] HELPER: Cancel the worker task and wait until it is done, without re-raising its
    # cancellation or error (waiting on a set of tasks only waits).
    worker.cancel()
    await asyncio.wait({worker})
