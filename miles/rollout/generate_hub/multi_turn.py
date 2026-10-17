# [MODULE DOC] Multi-turn generation with tool calling: repeatedly query the policy through the router's
# generate endpoint, parse tool calls from the reply, run the tools and append their responses to the
# token sequence, until the model stops calling tools, hits a length/abort stop, a request cannot be
# built, or the turn limit is reached. Defines its command line options (turn limit, tool specs path,
# tool call parser name, tool execution function path) as an add-arguments hook on the function.
"""
Simple multi-turn generation with tool calling.
"""

import argparse
import time
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.function_registry import load_function
from miles.utils.http_utils import post
from miles.utils.lifecycle import TrajectoryLifecycle


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # [FUNCTION DOC] Run a multi-turn tool-calling episode on a deep copy of the input sample and
    # return it as the single output sample. Setup: asserts partial rollout is off; router generate URL
    # from args; loads the tool execution function and tool specs by their paths and creates the tool
    # call parser (named in args) for those specs; if debug trajectory saving is enabled, starts a
    # recorded conversation from the prompt (when the prompt is a message list, else empty). Initial
    # tokens: prompt ids computed with the tool specs. Loop up to the maximum number of turns: build the
    # request payload from current tokens, sampling parameters and the evaluation flag (if no payload
    # can be built, set the sample status to the returned halting status and stop); post it with routing
    # headers; unless evaluating, record a generation span (turn number, completion token count) on the
    # trajectory timeline; update the sample from the response, extending response tokens, log probs
    # and the loss mask (loss mask enabled); if recording, append the assistant text and store a
    # snapshot of the conversation under messages in the sample metadata; stop if the finish reason is
    # abort or length; parse tool calls from the text and stop if none; otherwise execute them
    # (recording a tool span), append the tool responses to the sample tokens (tokenizer-aware, non-model
    # tokens masked out), and extend the recorded conversation. Reaching the turn limit simply ends the
    # loop without setting a status.
    # ----------------------- Setup -------------------------

    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    assert not args.partial_rollout, "Partial rollout is not supported"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    execute_tool_function = load_function(args.generate_execute_tool_function_path)

    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    # tool-call markup stays inline in the assistant text (what the model emitted)
    record_trajectory = args.save_debug_trajectory_data is not None
    trajectory = (list(sample.prompt) if isinstance(sample.prompt, list) else []) if record_trajectory else None

    # ----------------------- Initial prompts -------------------------

    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)

    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        # ----------------------- Call inference endpoint -------------------------

        payload, halt_status = compute_request_payload(
            args,
            sample.tokens,
            input.sampling_params,
            evaluation=input.evaluation,
        )
        if payload is None:
            sample.status = halt_status
            break

        gen_t0 = time.time()
        output = await post(url, payload, headers=compute_routing_headers(args, sample))
        sink = None if input.evaluation else TrajectoryLifecycle().sink
        if sink is not None:
            tokens = output.get("meta_info", {}).get("completion_tokens", "")
            sink.gen_span(sample, gen_t0, time.time(), turn=_turn + 1, detail=str(tokens))
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        if record_trajectory:
            trajectory.append({"role": "assistant", "content": output["text"]})
            sample.metadata["messages"] = list(trajectory)  # snapshot: trajectory keeps growing in later turns

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            break

        # ----------------------- Execute tools -------------------------

        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            break

        tool_t0 = time.time()
        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        if sink is not None:
            sink.tool_span(sample, tool_t0, time.time(), turn=_turn + 1, detail=f"{len(tool_calls)} calls")
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)
        if record_trajectory:
            trajectory.extend(tool_messages)
            sample.metadata["messages"] = list(trajectory)

    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    # [FUNCTION DOC] HELPER: Register this generate function's options: maximum turns (integer, default
    # 16), tool specs path, tool call parser name, and tool execution function path (all text, no
    # default). Attached below as the function's add-arguments hook.
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)


generate.add_arguments = _add_arguments
