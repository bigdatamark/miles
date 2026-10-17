# [MODULE DOC] Single-turn generation: one request to the router's generate endpoint for one sample,
# with support for resuming a partially generated sample (partial rollout).
"""
Simple single-turn generation.
"""

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample


async def generate(input: GenerateFnInput, url: str | None = None) -> GenerateFnOutput:
    # [FUNCTION DOC] Generate a completion for one sample, mutating the input sample IN PLACE (no copy)
    # and returning it as the output. Asserts the sample status is pending or aborted. The target URL is
    # the optional url argument, else the router's generate endpoint from args. Prompt ids come from the
    # sample. Partial rollout resume: if the sample already has response text, continue from its
    # existing tokens and reduce the MUTATED input sampling parameters' maximum new tokens by the
    # number of tokens already generated (assertion error if that goes negative); if exactly zero
    # remain, mark the sample truncated and return without a request. Otherwise start from the prompt
    # ids. Builds the request payload (with multimodal inputs and the evaluation flag); if none can be
    # built, sets the sample status to the returned halting status and returns. Else posts it with
    # routing headers and updates the sample from the response (no loss mask update).
    args = input.args
    sample = input.sample
    sampling_params = input.sampling_params
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"
    url = url or f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)

    # Handle Partial Rollout resuming
    if len(sample.response) > 0:
        input_ids = sample.tokens
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

        assert sampling_params["max_new_tokens"] >= 0
        if sampling_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        input_ids = prompt_ids

    payload, halt_status = compute_request_payload(
        args,
        input_ids=input_ids,
        sampling_params=sampling_params,
        multimodal_inputs=sample.multimodal_inputs,
        evaluation=input.evaluation,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    output = await post(url, payload, headers=compute_routing_headers(args, sample))
    await update_sample_from_response(args, sample, payload=payload, output=output)

    return GenerateFnOutput(samples=sample)
