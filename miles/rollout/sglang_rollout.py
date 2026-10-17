# [MODULE DOC]
# Legacy (function-style) default rollout implementation on top of the SGLang router. It exposes
# generate_rollout (the entry point called with args, rollout id, data source and an evaluation
# flag) and get_model_url (router url lookup for named models). Contents: a process-wide
# singleton generation state (tokenizer, processor, concurrency semaphore, default sampling
# parameters, deterministic seeds, data-parallel rank balancing, pending tasks, abort flag); the
# default single-sample token-in token-out generate function (multimodal, LoRA, OPD top-k, replay
# options, partial rollout resume); per-sample and per-group generate-and-reward coroutines; abort
# of in-flight requests; the training rollout loop with dynamic sampling filters, optional filter
# hooks, and log-prob recomputation; and the evaluation rollout over configured datasets.
# The newer class-based implementation lives in the inference rollout package. Module global:
# a cache of tokenized eval prompt datasets keyed by dataset config and tokenizer settings.
import asyncio
import copy
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.rollout.base_types import GenerateFnInput, RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
)
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.utils import dumper_utils
from miles.utils.async_utils import run
from miles.utils.data import Dataset
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.function_registry import load_function
from miles.utils.http_utils import get, post, router_worker_base_urls
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.misc import SingletonMeta, call_agent_abort_hook
from miles.utils.processing_utils import (
    call_processor,
    encode_image_for_rollout_engine,
    extract_multimodal_train_inputs,
    load_processor,
    load_tokenizer,
)
from miles.utils.types import Sample

from .generate_utils.generate_endpoint_utils import (
    compute_routing_headers,
    get_indexer_topk_from_response,
    policy_uses_routing_key,
)
from .generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from .generate_utils.sample_utils import reward_log_summary, sample_text_preview
from .generate_utils.sampling_mask import append_sampling_metadata, should_return_sampling_mask
from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    # [FUNCTION DOC] Build "http://ip:port" plus the endpoint for the model's own router if the
    # launch arg mapping from model name to (ip, port) is non-empty and contains the name;
    # otherwise use the default router ip and port. Pure.
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = args.sglang_model_routers
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    # [CLASS DOC]
    # Process-wide singleton (constructing it again returns the first instance, ignoring new
    # args). Holds: launch args, tokenizer, processor, a semaphore bounding concurrent generations,
    # default sampling parameters, optional per-sample-in-group sampling seeds (only with
    # deterministic inference), data-parallel rank usage counts and the last chosen rank, the
    # count of submitted-but-unfiltered groups, the set of pending group tasks, and the aborted
    # flag.
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # [FUNCTION DOC] Load tokenizer and processor from the HF checkpoint (trusting remote code,
        # using the chat template path); semaphore size is per-server concurrency times the number
        # of engines (rollout GPUs divided by GPUs per engine, integer division); default sampling
        # parameters come from the rollout launch args (no-stop-trim true, spaces between special
        # tokens false); with deterministic inference, group seeds are the rollout seed plus each
        # index in the group (one per sample per prompt); data-parallel counts start at zero for the
        # configured number of data-parallel ranks (at least one); then reset.
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        # [FUNCTION DOC] Context manager yielding a data-parallel rank with the fewest in-flight
        # generations: picks randomly (numpy) among ranks tied at the minimum count, increments its
        # count and records it as the current rank, and on exit decrements the count (asserting it
        # stays non-negative), even on error. The yielded rank is not otherwise used by callers.
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        # [FUNCTION DOC] Clear per-rollout state: remaining batch size to zero, no pending tasks,
        # not aborted.
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        # [FUNCTION DOC] Start one asyncio task per group running the group generate-and-reward
        # coroutine in training mode with a copy of the default sampling parameters; add the tasks
        # to the pending set and add the number of groups to the remaining batch size.
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    evaluation: bool = False,
) -> Sample:
    # [FUNCTION DOC]
    # Default generate function: one request to the router generate endpoint, appending the
    # result to the sample (mutated in place and returned). The sampling params mapping is also
    # mutated (max new tokens reduced when resuming).
    # Preconditions: in CI test mode the prompt must be a string; sample status must be pending or
    # aborted (assertion errors otherwise).
    # Prompt ids: if a processor exists and the prompt is a message list or the sample has any
    # non-null multimodal inputs, run the processor and take the first row of input ids, storing
    # the multimodal training inputs on the sample; else encode the text prompt without special
    # tokens. For a resumed (partial) sample the token budget is reduced by the tokens already
    # generated (tokens beyond the prompt); a negative budget fails an assertion; a zero budget
    # marks the sample truncated and returns without a request.
    # Payload: sampling params; log-prob return on; the sampling-support flag when required; for
    # on-policy distillation with a positive top-k and a strategy other than only-teacher, request
    # that many top log-probs; for training, the rollout top-k log-prob request fields; the LoRA
    # adapter name when LoRA rollout is on; routing replay and indexer replay flags; images (each
    # encoded for the engine) and audio (base64 data urls) if the sample has them; input ids are
    # the sample's existing tokens when it already has a response, else the prompt ids (also
    # initializing the sample's tokens when empty). Routing headers are computed from the sample.
    # After the post: OPD student top log-probs from the response are appended to the sample
    # metadata list; output tokens and log-probs are read from the output token log-prob pairs (or
    # empty); with the sampling mask, the mask is appended and log-probs are replaced by sampling
    # log-probs. The sample gets extended tokens, response length and response text; for training,
    # top-k log-prob data is appended; if the sample already has a loss mask (partial rollout with
    # off-policy masking, asserted) ones are appended for new tokens; rollout log-probs are
    # extended. Routed experts, when returned (base64 int32), are decoded to an array reshaped as
    # tokens by layers by top-k, where token count is prompt tokens plus new tokens minus one;
    # an extra trailing token row from the engine's stop edge is dropped; sizes are asserted and a
    # payload of all zeros is rejected as a capture failure. Indexer top-k is decoded if present.
    # Finally the sample's status, weight versions etc. are updated from the response meta info.
    # KNOWN ISSUE: the routed-experts size assertion message uses the sample's token count after
    # extension, labeled "unexpanded"; harmless but misleading.
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor and (
        isinstance(sample.prompt, (list, tuple))
        or (sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()))
    ):
        processor_output = call_processor(state.processor, sample.prompt, sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        prompt_ids = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
        sample.multimodal_train_inputs = extract_multimodal_train_inputs(processor_output)
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    return_sampling_mask = should_return_sampling_mask(args, sampling_params, evaluation=evaluation)

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if return_sampling_mask:
        payload["return_sampling_mask"] = True
    opd_top_k = getattr(args, "opd_log_prob_top_k", 0) or 0
    opd_top_k_strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if getattr(args, "use_opd", False) and opd_top_k > 0 and opd_top_k_strategy != "only-teacher":
        payload["top_logprobs_num"] = opd_top_k
    if not evaluation:
        configure_rollout_topk_logprobs_request(args, payload)

    if lora_rollout_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True
    if getattr(args, "use_rollout_indexer_replay", False):
        payload["return_indexer_topk"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    if sample.multimodal_inputs and sample.multimodal_inputs.get("audios"):
        import base64 as _b64

        payload["audio_data"] = [
            f"data:audio;base64,{_b64.b64encode(a).decode('ascii')}" for a in sample.multimodal_inputs["audios"]
        ]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    headers = compute_routing_headers(args, sample)

    output = await post(url, payload, headers=headers)
    if getattr(args, "use_opd", False) and opd_top_k > 0 and opd_top_k_strategy != "only-teacher":
        output_top_logprobs = output.get("meta_info", {}).get("output_top_logprobs")
        if output_top_logprobs is not None:
            sample.metadata.setdefault("opd_student_top_logprobs", [])
            sample.metadata["opd_student_top_logprobs"].extend(output_top_logprobs)

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    if payload.get("return_sampling_mask", False):
        new_response_log_probs = append_sampling_metadata(
            sample,
            new_response_tokens,
            output["meta_info"],
            sampling_logprobs_mode=payload.get("sampling_logprobs_mode", "selected"),
        )

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]
    if not evaluation:
        append_rollout_topk_logprobs(
            sample,
            output["meta_info"],
            args.rollout_top_logprobs_num,
            sampling_logprobs_mode=payload.get("sampling_logprobs_mode", "selected"),
        )

    # When partial rollout and masking off policy is enabled, update the loss mask
    if sample.loss_mask is not None:
        assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
        sample.loss_mask += [1] * len(new_response_tokens)

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        _re = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        )
        _ntok = int(output["meta_info"]["prompt_tokens"]) + len(new_response_tokens) - 1
        _topk = _re.size // max(1, _ntok * args.num_layers)
        if _re.size == (_ntok + 1) * args.num_layers * max(1, _topk):
            # stop-edge: sglang also forwarded the final token; its routing rows
            # feed no training position - drop the tail position.
            _re = _re[: _ntok * args.num_layers * _topk]
        assert _re.size == _ntok * args.num_layers * _topk, (
            f"routed_experts buffer {_re.size} != ntok({_ntok}) x layers({args.num_layers}) x topk({_topk}); "
            f"prompt_tokens={output['meta_info'].get('prompt_tokens')} response={len(new_response_tokens)} "
            f"unexpanded_tokens={len(sample.tokens)}"
        )
        assert _re.size == 0 or _re.any(), (
            "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
            "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
        )
        sample.rollout_routed_experts = _re.reshape(_ntok, args.num_layers, _topk)
    if "indexer_topk" in output["meta_info"]:
        sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample)

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # [FUNCTION DOC]
    # Generate one sample and compute its reward; returns the sample, or a list of samples when a
    # custom generate function returns several. The input is really a single sample despite the
    # annotation.
    # Steps: for partial rollout with off-policy masking, zero the loss mask of existing response
    # tokens. A completed or truncated sample is returned after asserting it has a response (and a
    # reward unless group reward is on). Training mode reports attempt start to the lifecycle
    # sink. Under the singleton state's semaphore: if aborted, mark aborted, report attempt end
    # and return. Report generation start; inside a data-parallel rank context, pick the generate
    # function path from the sample (per-sample override) else the launch arg; a loaded custom
    # function is called with state, sample, sampling params (not copied) and the evaluation flag,
    # and its output samples are used; else the default generate function runs. Report attempt end
    # after generation. With group reward, return now. List result: return early if any is aborted;
    # else batch-compute rewards for those without one and assign them positionally (a length
    # mismatch is silently truncated). Single result: return if aborted; else compute the reward
    # if unset.
    # KNOWN ISSUE: attempt end is not reported if generation raises (the inference version uses a
    # finally block); the aborted-early path reports attempt end but then skips the final report.
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # dashboard lifecycle probe (design §18.3): the semaphore wait IS the
    # queue; attempt_end fires once generation is over, before reward
    sink = None if evaluation else TrajectoryLifecycle().sink
    if sink is not None:
        sink.attempt_start(sample)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            if sink is not None:
                sink.attempt_end(sample)
            return sample
        if sink is not None:
            sink.gen_start(sample)

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            generate_fn = load_generate_function(custom_func_path) if custom_func_path else None
            if generate_fn is not None:
                output = await generate_fn(
                    GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params, evaluation=evaluation)
                )
                sample = output.samples
            else:
                sample = await generate(args, sample, sampling_params, evaluation=evaluation)

    if sink is not None:
        sink.attempt_end(sample)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    # [FUNCTION DOC]
    # Generate all samples of a prompt group concurrently and return the gathered group (list in
    # input order). Returns the group unchanged if the state is aborted. With routing-key policies,
    # assigns a random UUID string routing key to samples lacking one. Each sample runs the
    # generate-and-reward coroutine with its own copy of sampling params; with deterministic
    # inference the sampling seed is the state's group seed at the sample's index. After gathering
    # (no cancellation of siblings if one fails, unlike the inference version), if not aborted
    # and group reward is on, rewards are computed for the whole group and assigned positionally.
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique routing_key for each sample in the group (routing-key policies only)
    if policy_uses_routing_key(args):
        for sample in group:
            if sample.routing_key is None:
                sample.routing_key = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    # [FUNCTION DOC]
    # Abort in-flight generation (network I/O) and drain pending tasks. Asserts the state was not
    # already aborted, sets the aborted flag, lists worker urls from the router (list-workers
    # endpoint for old router versions or the in-repo router, else the workers endpoint),
    # posts an abort-all request to every worker concurrently (failures logged as warnings), and
    # awaits the user agent abort hook. Then repeatedly waits for pending tasks until none remain.
    # With partial rollout on, each finished group is collected after tagging samples that have a
    # response and no start rollout id with the current rollout id in their metadata; otherwise
    # the groups are discarded. Returns the collected groups (empty when partial rollout is off).
    # A task that raised would propagate its error when its result is read (partial rollout only).
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    urls = router_worker_base_urls(urls)

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=True):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # Let the agent integration tear down its in-flight trials so they stop hitting
    # SGLang, instead of running on until their own max_seq_len / timeout.
    await call_agent_abort_hook(args)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    # [FUNCTION DOC]
    # Training rollout. Returns (train output with kept groups and metrics, aborted partial
    # groups). Asserts the global dataset mode. Steps: configure the dumper on the engines; load
    # the optional dynamic sampling filter; create a metric gatherer; target group count is the
    # rollout batch size. Loop until enough groups are kept: while submitted-but-unresolved groups
    # are fewer than the target, fetch over-sampling-batch-size groups from the data source and
    # submit them; wait for at least one task to finish; for each finished task (errors
    # propagate, unlike the inference version) log the first group once, assert its size equals
    # samples-per-prompt, record it in all groups, feed metrics, apply the dynamic filter (a
    # dropped group decrements the remaining count so that a replacement is fetched and is reported
    # to the metrics), and keep it while fewer than target are kept.
    # Then log the last sample, abort leftover requests, assert the kept count, sort kept and all
    # groups by the first sample's index, reset the state, run the optional sample filter and
    # all-samples hooks (they may mutate in place; the latter also gets the data source), then
    # recompute rollout log-probs through a prefill pass at the default model url for all kept
    # samples (a no-op unless configured).
    # KNOWN ISSUE: kept groups never decrement the remaining count, so extra groups finishing
    # after the target is reached are only dropped silently by the length check.
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            metric_gatherer.on_group_before_dynamic_filter(args, group)
            filter_output = apply_preput_filters(args, dynamic_filter, group)
            if not filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
        sample_text_preview(sample),
        str(sample.label)[:100],
        reward_log_summary(sample.reward),
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if (x := args.rollout_sample_filter_path) is not None:
        filter_func = load_function(x)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if (x := args.rollout_all_samples_process_path) is not None:
        process_func = load_function(x)
        process_func(args, all_samples, data_source)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [sample for group in data for sample in group],
        url=get_model_url(args, "default"),
        sampling_params=state.sampling_params,
    )

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    # [FUNCTION DOC] Evaluate all datasets from the launch args (none if absent or empty)
    # concurrently and merge their result mappings (later wins on duplicate names). Asserts group
    # reward is off. Returns a pair: the eval output object holding the merged mapping from
    # dataset name to its rewards, truncated flags and samples, and an empty list (no aborted
    # samples). The rollout id is unused.
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    # [FUNCTION DOC]
    # Evaluate one dataset; returns a mapping with the dataset name as sole key and a mapping of
    # parallel lists (rewards, truncated flags, samples) sorted by sample index. Asserts group
    # reward is off. The tokenized prompt dataset is looked up in the module-level cache by the
    # dataset's cache key plus checkpoint, chat-template flag and template path, and built on a
    # miss (loads tokenizer and processor; disk I/O). Sampling params come from the dataset's own
    # temperature, top-p, top-k and max response length, plus the stop settings from launch args.
    # Every prompt is repeated n times (deep copies with a running index, injected dataset
    # metadata, the dataset's custom generate function path if any, a random routing key for
    # routing-key policies, and with deterministic inference a seed of rollout seed plus repeat
    # number); each runs in evaluation mode as its own task. Results are collected as tasks
    # complete with a progress bar, logging the first one, flattening list results. The reward
    # key is the eval reward key else the general one; if set, each reward is indexed by it.
    # The rollout id is unused.
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            if policy_uses_routing_key(args):
                sample.routing_key = str(uuid.uuid4())
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logged_sample = sample[0] if isinstance(sample, list) else sample
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(logged_sample.prompt) + logged_sample.response]} "
                f"reward={logged_sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    # [FUNCTION DOC]
    # Synchronous public entry point. Asserts the global dataset mode. Evaluation: run the eval
    # coroutine to completion on the shared async runner and return its output object (the empty
    # aborted list is discarded). Training: run the async training rollout with the data source's
    # sample getter, give the aborted partial groups back to the data source, and return the train
    # output. The third parameter is the data source (the docstring calls it a data buffer).
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
