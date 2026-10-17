# [MODULE DOC]
# Shared machinery for the inference-based rollout functions (train and eval): a generation state
# object holding tokenizer/processor/concurrency/sampling defaults, the per-sample and per-group
# generate-then-reward coroutines, the sampling-parameter builder, and the class-based rollout
# function that dispatches to the train or eval implementation. The train and eval modules import
# from here, so this module imports them lazily to avoid a cycle.
import asyncio
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from miles.rollout.base_types import (
    BaseRolloutFn,
    GenerateFnInput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.generate_hub.single_turn import generate
from miles.rollout.generate_utils.generate_endpoint_utils import policy_uses_routing_key
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class GenerateState:
    # [CLASS DOC]
    # Long-lived state shared by all generation calls of one rollout function: launch args,
    # tokenizer, multimodal processor, a semaphore bounding concurrent generate calls, default
    # training sampling parameters, the resolved default generate function, and an aborted flag
    # that is set when the rollout is being torn down and cleared by reset.
    def __init__(self, args: Namespace) -> None:
        # [FUNCTION DOC]
        # Build the state from launch args. Loads the tokenizer (with the configured chat template
        # path, trusting remote code) and processor from the HF checkpoint (disk/network I/O).
        # The semaphore capacity is the per-server concurrency times the number of rollout engines
        # (rollout GPUs divided by GPUs per engine, integer division). Default sampling parameters
        # come from the rollout temperature, top-p, top-k and max response length. The generate
        # function is loaded from the custom generate function path, falling back to the built-in
        # single-turn generate when that path yields nothing. Finally calls reset.
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.generate_fn_semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = compute_sampling_params(
            args,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
        )

        self.generate_function = load_generate_function(args.custom_generate_function_path) or generate

        self.reset()

    def reset(self) -> None:
        # [FUNCTION DOC] Clear the aborted flag so a new rollout or eval can generate.
        self.aborted = False


async def generate_and_rm(
    state: GenerateState,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # [FUNCTION DOC]
    # Generate one sample (possibly resumed from a partial rollout) and compute its reward.
    # Returns the resulting sample, or a list of samples when the generate function returns
    # several (multi-agent). Steps:
    # 1. Partial rollout with off-policy masking enabled: if the sample already has response
    #    tokens, zero its whole loss mask so previously generated tokens are not trained on.
    # 2. If the sample is already completed or truncated, assert it has a response (and a reward
    #    unless group reward is on) and return it untouched.
    # 3. For training (not evaluation) create a lifecycle sink and report attempt start; the sink
    #    reports attempt end in a finally block, even on error or abort.
    # 4. Under the state's semaphore: if the state is aborted, mark the sample aborted and return
    #    it. Otherwise report generation start, resolve the generate function (the sample's own
    #    generate function path overrides the state default), call it with the state, sample, a
    #    deep copy of the sampling params and the evaluation flag, and take its output samples.
    # 5. With group reward on, return without computing rewards (done per group later).
    # 6. For a list result: return early if any entry is aborted; otherwise compute rewards in a
    #    batch only for entries whose reward is unset, writing them in place.
    # 7. For a single result: return if aborted; otherwise compute the reward only if unset.
    # The type annotation says the input may be a list, but the code reads response length and
    # status from it, so only a single sample is valid input.
    args = state.args

    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    # dashboard lifecycle probe: the semaphore wait is the queue; attempt_end fires before reward
    sink = None if evaluation else TrajectoryLifecycle().sink
    if sink is not None:
        sink.attempt_start(sample)

    # generate
    log_prefix = f"[sample={getattr(sample, 'index', '?')}]"
    logger.debug(f"{log_prefix} Waiting for semaphore...")
    try:
        async with state.generate_fn_semaphore:
            if state.aborted:
                sample.status = Sample.Status.ABORTED
                return sample

            logger.debug(f"{log_prefix} Acquired semaphore, calling generate_function")
            if sink is not None:
                sink.gen_start(sample)
            # per-sample override, e.g. an eval dataset naming its own generate function
            generate_fn = load_generate_function(sample.generate_function_path) or state.generate_function
            output = await generate_fn(
                GenerateFnInput(
                    state=state,
                    sample=sample,
                    sampling_params=deepcopy(sampling_params),
                    evaluation=evaluation,
                )
            )
            sample = output.samples
            logger.debug(f"{log_prefix} generate_function returned")
    finally:
        if sink is not None:
            sink.attempt_end(sample)

    # TODO change to `if not args.group_rm: do reward model` for more clarity after the refactor below
    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # TODO: unify the two branches into one if we decide to use list as output type
    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        await batched_async_rm(args, samples_need_reward, inplace_set_reward_field=True)
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    logger.debug(f"{log_prefix} generate_and_rm complete")
    return sample


async def generate_and_rm_group(
    state: GenerateState,
    group: list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
    sample_done_callback: Callable[[], None] | None = None,
) -> list[Sample]:
    # [FUNCTION DOC]
    # Generate all samples of one prompt group concurrently and return the group (the gathered
    # results, one entry per sample in order). Returns the input group unchanged if the state is
    # already aborted, and returns the gathered results without group reward if the state became
    # aborted during generation.
    # Steps: when the routing policy needs a routing key, assign a fresh random UUID string to any
    # sample lacking one. For each sample start a task running the per-sample generate-and-reward
    # coroutine with a copy of the sampling params; with deterministic inference enabled, set the
    # sampling seed to the rollout seed plus the sample's index in the group. If a done callback is
    # given it is attached to every task and fires on success, error and cancellation. If gathering
    # raises (including cancellation), all tasks are cancelled and awaited to settle before the
    # exception is re-raised. Finally, with group reward on, rewards are computed for the whole
    # group in a batch, written in place.
    args = state.args

    if state.aborted:
        return group

    if policy_uses_routing_key(args):
        for sample in group:
            if sample.routing_key is None:
                sample.routing_key = str(uuid.uuid4())

    log_prefix = f"[group indices={[getattr(s, 'index', '?') for s in group]}]"
    logger.debug(f"{log_prefix} Starting group with {len(group)} samples")
    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            current_sampling_params["sampling_seed"] = args.rollout_seed + idx
        task = asyncio.create_task(generate_and_rm(state, sample, current_sampling_params, evaluation=evaluation))
        if sample_done_callback is not None:
            # fires on success, exception, and cancellation, so in-flight accounting is conserved
            task.add_done_callback(lambda _task: sample_done_callback())
        tasks.append(task)

    try:
        group = await asyncio.gather(*tasks)
    except BaseException:
        # cancel siblings and let them settle: the group returns only after every
        # sample task is done, so no orphan generation or in-flight credit outlives it
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    logger.debug(f"{log_prefix} [group] All {len(group)} samples completed")
    if state.aborted:
        return group

    if args.group_rm:
        await batched_async_rm(args, group, inplace_set_reward_field=True)

    return group


def compute_sampling_params(
    args,
    *,
    # after unifying configuration, this can be further refactored
    temperature,
    top_p,
    top_k,
    max_new_tokens,
):
    # [FUNCTION DOC]
    # Build the sampling-parameter mapping sent to the engine. Takes temperature, top-p, top-k and
    # max new tokens as explicit keyword arguments (so train and eval can differ) and fills the
    # rest from launch args: stop strings, stop token ids and the skip-special-tokens flag. Always
    # sets no-stop-trim to true and spaces-between-special-tokens to false. Returns a fresh dict.
    return dict(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )


class InferenceRolloutFn(BaseRolloutFn):
    # [CLASS DOC]
    # The default class-based rollout function. Owns the data source, a generation state and a
    # cache of tokenized eval datasets; routes each call to the training or evaluation path.
    def __init__(self, input: RolloutFnConstructorInput):
        # [FUNCTION DOC] Initialize the base class, keep the data source, create the generation
        # state from the launch args (loads tokenizer/processor), and start an empty eval prompt
        # dataset cache.
        super().__init__(input)
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        self.eval_prompt_dataset_cache = {}

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        # [FUNCTION DOC] Dispatch: await the eval path when the input is an evaluation request,
        # otherwise the train path, and return its output.
        if input.evaluation:
            return await self._call_eval(input)
        return await self._call_train(input)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        # [FUNCTION DOC] HELPER: Run one training rollout (module imported lazily to avoid an
        # import cycle). Calls the async train generator with the state, rollout id and the data
        # source's sample getter, then returns the aborted partial sample groups it reports back to
        # the data source (side effect on the data buffer) and returns the train output.
        from miles.rollout.inference_rollout.inference_rollout_train import generate_rollout_async

        output, aborted_samples = await generate_rollout_async(
            self.state, input.rollout_id, self.data_source.get_samples
        )
        self.data_source.add_samples(aborted_samples)
        return output

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        # [FUNCTION DOC] HELPER: Run evaluation over all configured eval datasets (module imported
        # lazily to avoid an import cycle). Uses the generation state supplied on the input if any,
        # else this object's own state, runs all datasets with the shared prompt cache (which gets
        # filled), and returns an eval output whose data maps dataset name to its result mapping.
        # Local: inference_rollout_eval imports GenerateState from this module.
        from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets

        state = input.generate_state or self.state
        results = await run_eval_datasets(state, self.eval_prompt_dataset_cache)
        return RolloutFnEvalOutput(data=results)
