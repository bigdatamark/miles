# [MODULE DOC] HELPER: Glue between SGLang's generate endpoint and Miles samples. Builds prompt token
# ids (text, chat template or multimodal processor), routing headers for key-based router policies,
# the request payload (token budget, log probs, replay flags, LoRA, images, top-k), and applies a
# response to a sample (tokens, log probs, loss mask, response text, routed experts, indexer top-k).
# Also decodes the base64 integer buffers the engine returns for expert routing and indexer top-k.
"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
)
from miles.rollout.generate_utils.sampling_mask import append_sampling_metadata, should_return_sampling_mask
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.processing_utils import encode_image_for_rollout_engine, extract_multimodal_train_inputs
from miles.utils.types import Sample


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    # [FUNCTION DOC] Turn the sample's prompt into token ids using the generation state's tokenizer and
    # optional processor. Multimodal path: if the state has a processor and the sample has multimodal
    # inputs with at least one non-none value, run the processor on the prompt text plus those inputs,
    # take the first row of its input ids, and STORE the extracted training-side multimodal inputs on
    # the sample (mutation; TODO in code to move this elsewhere); the tools argument is ignored there.
    # Text path: if the prompt is not a string (a chat message list) render it with the tokenizer's chat
    # template (not tokenized, generation prompt added, tools passed through); then encode the text
    # WITHOUT adding special tokens. Returns the processor's ids or a list of integer ids.
    prompt = sample.prompt

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]

        # TODO shall we move it to other places? then can make this function immutable
        sample.multimodal_train_inputs = extract_multimodal_train_inputs(processor_output)

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


def policy_uses_routing_key(args) -> bool:
    # [FUNCTION DOC] True when the configured router policy routes by a per-request key: consistent
    # hashing or manual.
    return args.sglang_router_policy in ("consistent_hashing", "manual")


def compute_routing_headers(args, sample: Sample) -> dict[str, str] | None:
    # [FUNCTION DOC] HTTP headers for routing a sample's requests. If the router policy needs a key and
    # the sample has none, raises value error naming the policy and sample index. If the sample has a
    # routing key (regardless of policy), returns a mapping with the routing key header set to it;
    # otherwise returns none.
    if policy_uses_routing_key(args) and not sample.routing_key:
        raise ValueError(
            f"router policy {args.sglang_router_policy} routes by X-SMG-Routing-Key, "
            f"but sample (index={sample.index}) has no routing_key set"
        )
    if sample.routing_key:
        return {"X-SMG-Routing-Key": sample.routing_key}
    return None


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
    *,
    evaluation: bool = False,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    # [FUNCTION DOC] Build the generate request. Returns (payload, None) or (None, truncated status) when
    # no new tokens are allowed. Works on a deep copy of the sampling parameters (input unchanged).
    # Token budget: the maximum new tokens from the parameters (removed from the copy), defaulting to the
    # launch maximum response length; if a launch maximum context length is set (non-zero), capped at
    # that minus the number of input ids; a budget of zero or less yields no payload. Payload: input
    # ids; sampling parameters with the final max new tokens; per-token log probs on; flags for
    # returning routed experts and indexer top-k from launch args; a sampling-mask request flag when
    # required by the sampling mask rules; the LoRA adapter name when LoRA rollout is enabled; images
    # from multimodal inputs encoded for the engine. For non-evaluation requests, top-k candidate log
    # probs settings are added (which may mutate the payload's sampling parameters and keys and raise
    # on invalid settings).
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    return_sampling_mask = should_return_sampling_mask(args, sampling_params, evaluation=evaluation)

    payload = {
        "input_ids": input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
        "return_routed_experts": args.use_rollout_routing_replay,
        "return_indexer_topk": args.use_rollout_indexer_replay,
    }
    if return_sampling_mask:
        payload["return_sampling_mask"] = True
    if lora_rollout_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME
    if image_data := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    if not evaluation:
        configure_rollout_topk_logprobs_request(args, payload)
    return payload, None


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # [FUNCTION DOC] Apply one engine response to the sample in place. Steps: on the first turn (no
    # response yet and no tokens) set the sample's tokens to the request's input ids. Read new response
    # tokens and log probs from the response's per-token log prob list (pairs of log prob and token id;
    # empty if absent). If a sampling mask was requested, let the sampling metadata helper record it
    # and take the log probs it returns (they replace the plain ones). Append new tokens to the tokens
    # (new list, no re-tokenization), increase the response length by their count, append the response
    # text, append log probs to the sample's rollout log probs (created if none). If top-k candidates
    # were requested (top log prob count in the payload, or support mode) append them. If requested,
    # extend the loss mask with ones for the new tokens (creating it if none). Then OVERWRITE (not
    # append; TODO in code about multi-turn) the sample's routed experts and indexer top-k from the
    # response, and finally let the sample update itself from the response meta info (status, weight
    # versions, speculative and prefix cache info; TODO to unify). Key errors if required response
    # fields are absent.
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

    if x := output["meta_info"].get("output_token_logprobs"):
        new_response_tokens = [item[1] for item in x]
        new_response_log_probs = [item[0] for item in x]
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

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs
    if payload.get("top_logprobs_num") or payload.get("sampling_logprobs_mode") == "support":
        append_rollout_topk_logprobs(
            sample,
            output["meta_info"],
            args.rollout_top_logprobs_num,
            sampling_logprobs_mode=payload.get("sampling_logprobs_mode", "selected"),
        )

    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask += [1] * len(new_response_tokens)

    # TODO handle multi-turn cases (may need concat instead of assignment)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, output, len(sample.tokens) - 1)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample)

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def _decode_topk_buffer(info: str, num_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    # [FUNCTION DOC] HELPER: Decode a base64 text of 32-bit integers into an array shaped (tokens,
    # layers, top-k). If the token count is zero or negative, returns an empty array of shape
    # (0, layers, max(0, top-k)) without checking the buffer. A top-k of minus one means "infer it":
    # buffer length divided by (tokens times layers). Reshape fails (value error) if the buffer size
    # does not match.
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:  # indexer: topk dim recovered from buffer length
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk)


def get_routed_experts_from_response(args, output, num_tokens: int):
    # [FUNCTION DOC] Decode the response's routed experts (expert ids chosen per token per layer) for
    # the given token count, with the layer count from args and the expert count inferred. Returns none
    # if the response has none. Asserts the decoded array is not entirely zeros (which means the engine
    # did not capture them, e.g. with a router-bypassing MoE backend); an empty array is allowed.
    info = output["meta_info"].get("routed_experts")
    if info is None:
        return None
    routed_experts = _decode_topk_buffer(info, num_tokens, args.num_layers, -1)
    assert routed_experts.size == 0 or routed_experts.any(), (
        "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
        "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
    )
    return routed_experts


def get_indexer_topk_from_response(args, output, sample):
    # [FUNCTION DOC] Decode the response's indexer top-k (sparse attention indices per token per stream)
    # for all sample tokens except the last. Returns none if absent. Asserts the response also reports
    # the number of indexer streams (layers), and, when the launch args state an expected stream count,
    # that it matches (otherwise replay would map streams to wrong layers). The top-k width is
    # inferred from the buffer size.
    info = output["meta_info"].get("indexer_topk")
    if info is None:
        return None
    num_layers = output["meta_info"].get("indexer_topk_num_layers")
    assert num_layers is not None, (
        "Server returned indexer_topk without indexer_topk_num_layers; "
        "sglang-miles must include the layer count in meta_info."
    )
    expected_num_streams = getattr(args, "rollout_indexer_topk_num_streams", None)
    assert expected_num_streams is None or num_layers == expected_num_streams, (
        f"Server returned indexer_topk with {num_layers} streams but the model has "
        f"{expected_num_streams} indexer layers; replaying it would map streams to the wrong layers."
    )
    return _decode_topk_buffer(info, len(sample.tokens) - 1, num_layers, -1)
