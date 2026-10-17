# [MODULE DOC] Custom multi-turn generate function for the geo3k vision-language example. For one
# sample it builds a pluggable interaction environment (module path or file path from the args, by
# default the geo3k environment), then alternates between asking the SGLang router to continue the
# token sequence (with image data and log probs) and feeding the model response to the environment,
# which returns an observation encoded into tokens (and optionally images) appended to the same
# sequence. Model tokens get loss mask 1 and real log probs; environment tokens get loss mask 0 and
# zero log probs. A token budget derived from the context limit stops runaway episodes. Multimodal
# training inputs of all turns are concatenated per key (tensors only).
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from examples.geo3k_vlm.multi_turn.base_env import BaseInteractionEnv

# When executed as a module: python -m examples.geo3k_vlm.multi_turn.rollout
from miles.rollout.sglang_rollout import GenerateState
from miles.utils.http_utils import post
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.geo3k_vlm.multi_turn.env_geo3k"

# Dummy messages used for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    # [FUNCTION DOC] HELPER: Loads the environment module. The target is the given path/name or the
    # default geo3k module name. If the target is an existing file with the .py suffix, imports it
    # from that file under the name "rollout_env_<file stem>" (registered in the module table,
    # raising an import error if a loader cannot be made); otherwise imports it by dotted module name.
    # Returns the module.
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    # [FUNCTION DOC] HELPER: Calls the module's environment factory with sample and args as keyword
    # arguments; if that raises a type error, calls it again positionally. Raises a value error if
    # the factory is not callable. Returns the environment.
    # KNOWN ISSUE: a type error raised inside the factory body (not only a signature mismatch) also
    # triggers the positional retry and may then be masked or reported confusingly.
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        # Fallback to positional signature
        return build_fn(sample, args)


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    # [FUNCTION DOC] HELPER: Encodes one observation message (possibly with image content parts)
    # into token ids that can be appended to a running sequence. Tool descriptions come from the
    # sample metadata (if any). With chat templating: renders a fixed dummy conversation (system and
    # user) without generation prompt to learn how many tokens the template preamble takes, and
    # renders the dummy conversation plus the message with a generation prompt; the preamble token
    # count is later cut from the front of the encoded ids. Without chat templating the formatted
    # prompt is a list holding just the message. With a processor: extracts images and videos from
    # the message, runs the processor on the text and media, takes the first row of input ids and
    # keeps all other processor outputs except attention mask as multimodal training inputs (none if
    # empty). Without a processor: tokenizes the text without special tokens. Image data for the
    # engine is each extracted image encoded for the rollout engine. Returns a four-item tuple:
    # token ids, encoded images list (possibly empty), multimodal inputs mapping (images, videos) or
    # none, and multimodal training inputs or none.
    # KNOWN ISSUE: with chat templating off and no processor, the tokenizer is given a list holding
    # a message mapping, which it cannot encode.
    """
    Encode a single observation turn that may include images/videos in the content list.
    Trim out the system/tool preamble added by the chat template so only the observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        # Convert content-embedded images/videos into multimodal inputs for the processor.
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    return prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    # [FUNCTION DOC] HELPER: Merges per-turn multimodal training input mappings. Empty list gives
    # none. Collects, per key, the non-none values from all non-empty chunks in order, and
    # concatenates along the first dimension those keys whose values are all tensors; keys with any
    # non-tensor value are dropped. Returns the merged mapping, which may be empty if nothing
    # qualified.
    """
    Merge per-turn multimodal_train_inputs with a single concat per key.

    Note: Only torch.Tensor values are merged; non-tensor fields are ignored by design.
    """
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


def _initialize_resources(args: Any, sample: Sample):
    # [FUNCTION DOC] HELPER: Prepares everything for one episode. Loads the environment module named
    # by the args' interaction environment path option; reads the turn limit from the args and raises
    # a value error if it is none; creates the generation state and the router generate url; makes
    # sure the sample has a metadata mapping; builds the environment. Returns a five-item tuple:
    # environment, module, config mapping (max turns), state, url.
    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)
    config = {"max_turns": max_turns}
    return env, env_module, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    # [FUNCTION DOC] HELPER: Tokenizes the sample's prompt. With a processor: runs it on the prompt
    # text and the sample's multimodal inputs, uses the first row of input ids, and stores the other
    # outputs (minus attention mask) on the sample as multimodal training inputs (none if empty);
    # otherwise tokenizes without special tokens. Image data is each image of the sample's
    # multimodal inputs encoded for the engine. Returns the prompt ids, the image data list and the
    # sample's multimodal training inputs. Mutates the sample (multimodal training inputs).
    if processor:
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]
    return prompt_ids, image_data, sample.multimodal_train_inputs


def _prepare_start_state(sample: Sample, state, args: Any, sampling_params: dict):
    # [FUNCTION DOC] HELPER: Initializes the sample's running state and the token budget. Tokenizes
    # the prompt; the training-inputs buffer starts with the initial multimodal training inputs if
    # any. If the sample has no tokens they become the prompt ids; the existing response tokens are
    # whatever follows the prompt ids (none if the sample is shorter than the prompt). Ensures the
    # loss mask and rollout log probs lists exist and sets the response length. Budget: if a
    # context-length limit is set, the limit minus the current token count; else if the sampling
    # params have a max-new-tokens value, that value minus the current token count; else none.
    # Returns the current image data, the response token list, the budget and the buffer.
    # KNOWN ISSUE: in the fallback the new-token cap is reduced by the total token count (prompt
    # included), although a cap on new tokens should not count the prompt.
    prompt_ids, image_data, init_mm_train = _prepare_initial_inputs(sample, state.processor, state.tokenizer)
    current_image_data = image_data
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)
    return current_image_data, response_tokens, budget, multimodal_train_inputs_buffer


async def _run_inference_step(url: str, tokens: list[int], sampling_params: dict, image_data, tokenizer):
    # [FUNCTION DOC] HELPER: Sends one generation request with the full token sequence so far, the
    # sampling params and log probs enabled (plus image data when there is any) to the router.
    # Returns a four-item tuple: response text, new token ids and new log probs (taken from the
    # per-token log prob list in the response; both empty if it is missing), and the finish reason
    # type. The tokenizer argument is unused.
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type


def _process_env_step(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    # [FUNCTION DOC] HELPER: Feeds the response to the environment. If the environment reports done,
    # returns five items: four nones and true. Otherwise formats the observation as a user message,
    # encodes it (honoring the args' chat template flags and the sample metadata tools), removes a
    # leading beginning-of-sequence token if present, and returns the observation token ids, encoded
    # image data, multimodal inputs, multimodal training inputs and false. The info mapping from the
    # environment is ignored.
    observation, done, _ = env.step(response_text)
    if done:
        return None, None, None, None, True

    next_user_message = env.format_observation(observation)
    obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs = (
        _encode_observation_for_generation(
            tokenizer,
            processor,
            next_user_message,
            sample_metadata,
            args.apply_chat_template,
            args.apply_chat_template_kwargs,
        )
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]

    return obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, False


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    # [FUNCTION DOC] HELPER: Appends the tokens to both the sample tokens and the response token
    # list, appends the loss mask value once per added token, extends the rollout log probs with the
    # given values (caller supplies one per token), and updates the sample's response length to the
    # response token count. Mutates in place.
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    sample: Sample,
    current_image_data,
    obs_image_data,
    obs_multimodal_inputs,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer: list[dict | None],
):
    # [FUNCTION DOC] HELPER: Folds an observation's media into the running state and returns the
    # updated image data list (current plus the observation's, when any). Sample multimodal inputs:
    # if the sample has none, they become the observation's; if both are mappings, for each key of
    # the observation with a non-none list value whose counterpart in the sample is also a list, the
    # list is extended in place (other keys, including keys absent from the sample, are ignored);
    # if the sample's inputs are not a mapping or the observation's is not, they are replaced by the
    # observation's. The observation's multimodal training inputs, if any, are appended to the
    # buffer.
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _should_stop_on_finish(sample: Sample, finish_type: str) -> bool:
    # [FUNCTION DOC] HELPER: Sets the sample status and returns true for finish reasons that end the
    # episode: length -> truncated, abort -> aborted. Other reasons return false and change nothing.
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return True
        case "abort":
            sample.status = Sample.Status.ABORTED
            return True
    return False


def _update_budget(budget, consumed: int):
    # [FUNCTION DOC] HELPER: Returns the budget minus the consumed token count, or none if there is
    # no budget.
    if budget is None:
        return None
    return budget - consumed


def _finalize_sample(sample: Sample, tokenizer, response_tokens, multimodal_train_inputs_buffer):
    # [FUNCTION DOC] HELPER: Completes the sample: merges the buffered multimodal training inputs,
    # decodes the response tokens to text keeping special tokens, sets the response length, and
    # sets the status to completed if it is still unset. Returns the sample.
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    # [FUNCTION DOC] Custom generate function for multi-turn interaction; returns the filled sample.
    # Asserts partial rollout is off. Initializes the environment, state and url, copies the
    # sampling params, prepares the start state and resets the environment. If the initial budget is
    # not positive the sample is marked truncated and returned immediately (without finalization).
    # Then for up to the maximum number of turns: when a budget exists, sets the request's max new
    # tokens to the remaining budget; generates; appends the model tokens with loss mask 1 and the
    # real log probs; reduces the budget; stops on a length or abort finish; stops as truncated if the
    # budget is used up; passes the response to the environment: if done, marks the sample completed
    # and stops; else appends the observation tokens with loss mask 0 and zero log probs, reduces the
    # budget by their count, merges the new media into the state, stops as truncated if the budget is
    # used up, and marks the sample completed after the last allowed turn. Always closes the
    # environment at the end, ignoring close errors. Returns the sample after finalization (merged
    # multimodal training inputs, decoded response, response length, default status completed).
    # KNOWN ISSUE: the environment module is loaded and the environment built on every sample, and
    # the early return for a non-positive budget skips the finalization step.
    """Custom multi-turn rollout that interacts with a pluggable environment."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    env, env_module, config, state, url = _initialize_resources(args, sample)
    sampling_params = sampling_params.copy()
    current_image_data, response_tokens, budget, multimodal_train_inputs_buffer = _prepare_start_state(
        sample, state, args, sampling_params
    )
    try:
        env.reset()
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        cur_sampling_params = sampling_params
        for turn_idx in range(config["max_turns"]):
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget

            response_text, new_response_tokens, new_response_log_probs, finish_type = await _run_inference_step(
                url, sample.tokens, cur_sampling_params, current_image_data, state.tokenizer
            )
            _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            budget = _update_budget(budget, len(new_response_tokens))

            if _should_stop_on_finish(sample, finish_type):
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            obs_prompt_ids, obs_image_data, obs_multimodal_inputs, obs_multimodal_train_inputs, done = (
                _process_env_step(env, response_text, state.tokenizer, state.processor, args, sample.metadata)
            )
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            budget = _update_budget(budget, len(obs_prompt_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                multimodal_train_inputs_buffer,
            )

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= config["max_turns"]:
                sample.status = Sample.Status.COMPLETED
                break

        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
    finally:
        try:
            env.close()
        except Exception:
            pass
