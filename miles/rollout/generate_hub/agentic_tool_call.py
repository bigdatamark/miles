# [MODULE DOC] Generic agentic generate function. Flow: start a traced session (session server instance
# chosen by the tracer), run a user-supplied agent function (inline or in a child process) against the
# session's OpenAI-compatible URL, then collect the training samples the session server assembled from
# the recorded model calls. Two session server protocol versions: v1 (a single sample per call; agent
# metadata and session metadata are merged into it driver-side) and v2 (possibly several samples per
# call; agent metadata travels through the collect call; speculative decoding metrics are carried in
# sample metadata). Failures to collect or empty collections yield an aborted copy of the input sample.
# Also defines command line options (agent function path, run mode, maximum sequence length) and the
# conversion of internal sampling parameters to chat completion request keyword arguments.
"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Collecting the worker-assembled training samples (the session server
     converts records to samples, truncates and merges in the owning worker)
  3. Driver-side metadata application (agent_metadata, session_metadata)
  4. Where each agent call runs (--custom-agent-function-mode)

Agent function contract:
  async def my_agent(
      base_url: str,
      prompt: ...,
      request_kwargs: dict,
      metadata: dict,       # sample.metadata — env-specific fields
      **kwargs,
  ) -> dict | None:
      ...

  Returning None means no extra metadata to attach.
  Returning a dict merges it into every sample's metadata, so downstream
  reward models (--custom-rm-path) can read whatever the agent left there.

  Under ``subproc`` (the default) each call runs in a child process of its own:
  its arguments and return value must be picklable, and module-level state lasts
  one call. Limits shared across calls go through ``miles.rollout.agentic.rollout_limits``.
"""

import argparse
import logging
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import httpx
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.agentic.agent_function import AGENT_FUNCTION_MODES, call_agent_function
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # [FUNCTION DOC] Generate for one sample by running the agent. Preconditions (assertion errors):
    # partial rollout is off; session server instances are configured (the session server flag is on).
    # Version v2 is chosen when the session server flag equals "v2"; speculative metrics are collected
    # only for v2 with a speculative algorithm. Steps: (1) create the tracing session; (2) load the
    # agent function by path (assertion error if missing); (3) build the metadata given to the agent: a
    # copy of the sample metadata plus the maximum sequence length (if set), the session server
    # instance id (if the tracer has one) and the session server id (always, from the tracer since the
    # owning address is per session); (4) call the agent with base url, prompt, chat request keyword
    # arguments derived from sampling parameters, and metadata, in the configured mode; an ordinary
    # exception is logged as a warning and swallowed (agent metadata stays none); (5) regardless of agent
    # outcome, collect samples from the session (v2 also sends the agent metadata); only timeouts and
    # transport errors during collection are swallowed (marking collection failed), other errors (such as
    # error responses) propagate. If collection failed, or returned no samples (logging whether all were
    # truncated or no model calls were recorded): return an aborted deep copy of the input sample (a
    # list for v2, a bare sample for v1), first removing any speculative metrics entry from its
    # metadata when those are collected. (6) With speculative metrics: take the session's metrics from
    # the session metadata, raise value error if their session id differs from the tracer's or their
    # metrics are none, then replace the entry in every sample's metadata with the session-level one.
    # (7) v2 with several samples: give every sample the same rollout id (input rollout id, else input
    # index; assertion error if both unset) so they aggregate as one rollout (FIXME in code about index
    # handling). (8) v1: merge the agent metadata (or nothing) into every sample's metadata. (9) If the
    # agent reported total tool time under its agent metrics, set it as the non-generation time of every
    # sample. (10) v2 returns the sample list; v1 requires exactly one sample (unpack error otherwise),
    # merges the session metadata into it and returns it.
    assert not input.args.partial_rollout, "Partial rollout is not supported"
    assert getattr(input.args, "session_server_instances", None), (
        "agentic_tool_call.generate requires session_server_instances. "
        "Pass --use-session-server to start the session server."
    )
    use_v2 = getattr(input.args, "use_session_server", None) == "v2"
    collect_spec_metrics = use_v2 and input.args.sglang_speculative_algorithm is not None
    tracer = await OpenAIEndpointTracer.create(
        input.args, evaluation=input.evaluation, sampling_params=input.sampling_params
    )

    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}
    if tracer.session_server_instance_id:
        metadata = {**metadata, "session_server_instance_id": tracer.session_server_instance_id}

    log_prefix = f"[session={tracer.session_id}]"

    # From the tracer, not args: with multiple instances the owning ip:port is per-session.
    metadata = {**metadata, "session_server_id": tracer.session_server_id}

    agent_metadata = None
    collect_failed = False
    t_start = time.monotonic()
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await call_agent_function(
            custom_agent_function,
            mode=input.args.custom_agent_function_mode,
            base_url=tracer.agent_base_url,
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
        )
        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic()-t_start:.1f}s")
    except Exception as e:
        logger.warning(f"{log_prefix} Agent function failed: {e}", exc_info=True)

    finally:
        # Collect even if the agent failed.
        logger.debug(f"{log_prefix} Calling collect_samples...")
        collect_kwargs = {"max_seq_len": max_seq_len}
        if use_v2:
            collect_kwargs["agent_metadata"] = agent_metadata
        try:
            result = await tracer.collect_samples(input.sample, **collect_kwargs)
        # Costs this sample, not the run; a non-2xx still raises RuntimeError.
        except (TimeoutError, httpx.TransportError) as e:
            collect_failed = True
            logger.warning(f"{log_prefix} Failed collecting samples: {e!r}", exc_info=True)
        else:
            logger.debug(
                f"{log_prefix} collect_samples done: {len(result.samples)} samples, "
                f"total_time={time.monotonic()-t_start:.1f}s"
            )

    if collect_failed:
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        if collect_spec_metrics:
            sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
        return GenerateFnOutput(samples=[sample] if use_v2 else sample)

    if not result.samples:
        if result.empty_reason == "all_truncated":
            logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        else:
            logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        if collect_spec_metrics:
            sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
        return GenerateFnOutput(samples=[sample] if use_v2 else sample)

    session_rollout_metrics = None
    if collect_spec_metrics:
        session_rollout_metrics = result.session_metadata[SESSION_ROLLOUT_METRICS_KEY]
        if session_rollout_metrics["session_id"] != tracer.session_id:
            raise ValueError(
                "session_rollout_metrics.session_id does not match the collected session: "
                f"{session_rollout_metrics['session_id']!r} != {tracer.session_id!r}"
            )
        if session_rollout_metrics["metrics"] is None:
            raise ValueError("a successful session collect must carry metrics")

    samples = result.samples
    if collect_spec_metrics:
        for sample in samples:
            sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
            sample.metadata[SESSION_ROLLOUT_METRICS_KEY] = session_rollout_metrics
    if use_v2 and len(samples) > 1:
        # FIXME: handle sample index issues.
        rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
        assert rollout_id is not None, "v2 agentic samples require input Sample.rollout_id or Sample.index"
        for sample in samples:
            sample.rollout_id = rollout_id
    if not use_v2:
        # v1: the agent's metadata is applied driver-side. Under v2 it traveled
        # through collect_samples and came back applied by the server-side
        # merge (per-sample metadata/reward on the wire) — no overlay here.
        for s in samples:
            s.metadata.update(agent_metadata or {})

    # If the agent function reports wall-clock time spent outside policy generation
    # (env/tool steps), surface it on Sample.non_generation_time so throughput
    # accounting subtracts it.
    ngt = ((agent_metadata or {}).get("agent_metrics") or {}).get("total_tool_time")
    if ngt is not None:
        for s in samples:
            s.non_generation_time = ngt

    if use_v2:
        return GenerateFnOutput(samples=samples)

    (sample,) = samples
    sample.metadata.update(result.session_metadata)
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    # [FUNCTION DOC] HELPER: Register this generate function's options on the parser: the agent function
    # path (text, no default), the run mode (choices inline or subproc, default subproc) and the
    # maximum sequence length in tokens (integer, default none; truncation happens in the session
    # server and the value is also forwarded to the agent). Attached below as the generate function's
    # add-arguments hook.
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument(
        "--custom-agent-function-mode",
        choices=AGENT_FUNCTION_MODES,
        default="subproc",
        help="Where each agent call runs: subproc, a child process of its own so one episode's "
        "blocking step or crash cannot stall the others; or inline, on the rollout's event loop.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        dest="max_seq_len",
        help="Max sequence length in tokens (prompt + completion, including env responses) "
        "per session. Truncation happens inside the session server during sample assembly; "
        "also forwarded to the Harbor agent server (as max_seq_len) to abort the trial early.",
    )


generate.add_arguments = _add_arguments


# Process keys to match ChatCompletionRequest input
def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] Convert internal sampling parameters into keyword arguments for a chat completion
    # request. On a copy: rename maximum new tokens to maximum tokens, minimum new tokens to minimum
    # tokens and sampling seed to seed; a renamed key is only set if the destination key is not already
    # present (the existing destination wins), and the source key is always removed. Then keep only keys
    # that are fields of the chat completion request model, excluding model and messages, and drop
    # entries whose value is none. Returns the new mapping; the input is not modified.
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
