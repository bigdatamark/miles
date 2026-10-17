# [MODULE DOC]
# Evaluation rollout for the inference-based rollout function. Loads (and caches) each configured
# eval dataset, generates n samples per prompt using per-dataset sampling settings, computes
# rewards, and returns per-dataset rewards, truncation flags and the samples themselves. Group
# reward models are not supported for evaluation.
import asyncio
import copy
import logging
import uuid
from typing import Any

from tqdm import tqdm

from miles.rollout.generate_utils.generate_endpoint_utils import policy_uses_routing_key
from miles.rollout.inference_rollout.inference_rollout_common import (
    GenerateState,
    compute_sampling_params,
    generate_and_rm,
)
from miles.utils.data import Dataset
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.misc import as_completed_async
from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def run_eval_datasets(
    state: GenerateState,
    prompt_dataset_cache: dict[Any, Dataset],
) -> dict[str, dict[str, Any]]:
    # [FUNCTION DOC]
    # Evaluate all datasets listed in the launch args, concurrently. Asserts group reward is off.
    # Each dataset is run by the single-dataset coroutine sharing the prompt dataset cache (which
    # is mutated). Returns one merged mapping from dataset name to that dataset's result mapping
    # (rewards, truncated flags, samples); on duplicate names the later dataset wins.
    args = state.args
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in args.eval_datasets:
        coros.append(eval_rollout_single_dataset(state, dataset_cfg, prompt_dataset_cache))
    results_list = await asyncio.gather(*coros)
    return {k: v for r in results_list for k, v in r.items()}


async def eval_rollout_single_dataset(
    state: GenerateState,
    dataset_cfg: EvalDatasetConfig,
    prompt_dataset_cache: dict[Any, Dataset],
) -> dict[str, dict[str, list[Any]]]:
    # [FUNCTION DOC]
    # Evaluate one dataset and return a mapping with the single key of the dataset name, whose
    # value holds three parallel lists sorted by sample index: rewards, truncated flags, samples.
    # Steps:
    # 1. Assert group reward is off (the same check as the caller).
    # 2. Look up the tokenized prompt dataset in the cache by the dataset config's cache key plus
    #    checkpoint, apply-chat-template flag and chat template path; on a miss load tokenizer and
    #    processor and build the dataset with the eval max prompt length and the config's input,
    #    label, metadata and tool keys (disk I/O), then store it.
    # 3. Build base sampling params from the dataset's own temperature, top-p, top-k and max
    #    response length.
    # 4. For every prompt and every repeat (samples per eval prompt), deep-copy the prompt sample,
    #    assign a running index, inject the dataset config's metadata, set the sample's generate
    #    function path from the config, give it a random routing key if the routing policy needs
    #    one, and with deterministic inference use a copy of the params with the seed set to the
    #    rollout seed plus the repeat number. Start a generate-and-reward task in evaluation mode.
    # 5. Collect results as tasks finish with a progress bar; log the first result once; flatten
    #    list results into the output; sort by index.
    # 6. The reward key is the eval reward key, else the general reward key; if set, the reward of
    #    each sample is indexed by it, otherwise the reward is used directly.
    args = state.args
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    if cache_key not in prompt_dataset_cache:
        tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        prompt_dataset_cache[cache_key] = Dataset(
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
    dataset = prompt_dataset_cache[cache_key]

    base_sampling_params = compute_sampling_params(
        args,
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
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
            sample.generate_function_path = dataset_cfg.custom_generate_function_path
            if policy_uses_routing_key(args):
                sample.routing_key = str(uuid.uuid4())
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        state,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    async for sample in as_completed_async(tasks):
        if do_print:
            # TODO improve this after enhancing samples' type
            s = (sample[0] if len(sample) > 0 else None) if isinstance(sample, list) else sample
            if s is not None:
                logger.info(
                    "eval_rollout_single_dataset example data: "
                    f"{[str(s.prompt) + s.response]} "
                    f"reward={s.reward}"
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
