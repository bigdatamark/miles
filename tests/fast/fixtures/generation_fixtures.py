# [MODULE DOC]
# Fixtures and helpers for testing generate functions (old sglang rollout, single turn, multi turn and
# agentic tool call variants) against a mock sglang server. Provides: command line assembly per
# variant, a sample factory, an environment dataclass and result dataclass, a synchronous runner that
# invokes the variant's generate function, a parsed-arguments factory, a context manager that starts
# a session server in a background thread, and the pytest fixture that wires the mock server, the
# arguments and (for agentic variants) module-level knobs of the mock tools module. Constants: default
# model name, default mock response text (a boxed answer), default sampling parameters, and a mapping
# from variant name to the dotted path of its generate function.
"""
Fixtures to test custom-generate-function
"""

from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import pytest

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.rollout.session.config import compute_session_server_config
from miles.rollout.session.server import SessionServer
from miles.rollout.session.types import SessionServerInstance
from miles.utils.async_utils import run
from miles.utils.http_utils import find_available_port, init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, ProcessResultMetaInfo, with_mock_server
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

MODEL_NAME = "Qwen/Qwen3-0.6B"
RESPONSE_TEXT = "\\boxed{8}"


DEFAULT_SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}

VARIANT_TO_GENERATE_FN_PATH = {
    "old_sglang_rollout": "miles.rollout.sglang_rollout.generate",
    "single_turn": "miles.rollout.generate_hub.single_turn.generate",
    "multi_turn": "miles.rollout.generate_hub.multi_turn.generate",
    "agentic_tool_call": "miles.rollout.generate_hub.agentic_tool_call.generate",
}


def extra_argv_for_variant(
    variant: str,
    *,
    custom_generate_function_path: str | None = None,
    generate_max_turns: int = 16,
    generate_tool_specs_path: str = "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
    generate_tool_call_parser: str = "qwen25",
    generate_execute_tool_function_path: str = "miles.utils.test_utils.mock_tools.execute_tool_call",
    custom_agent_function_path: str = "miles.utils.test_utils.mock_tools.run_agentic_tool_call",
) -> list[str]:
    # [FUNCTION DOC] Builds the variant-specific command line tokens. Always starts with the custom
    # generate function path flag, valued by the override if given, else the variant's mapped path (an
    # unknown variant without an override raises a key error). For multi turn, appends the max turns
    # (as text), tool specs path, execute-tool function path and tool call parser flags. For agentic tool
    # call, appends the custom agent function path, forces the agent function mode to inline (the mock
    # agent is steered through module state a subprocess could not see), and enables the session server
    # at version v2 with the qwen3 tito model. Other variants add nothing. Returns the token list.
    argv = [
        "--custom-generate-function-path",
        custom_generate_function_path or VARIANT_TO_GENERATE_FN_PATH[variant],
    ]

    if variant == "multi_turn":
        argv += [
            "--generate-max-turns",
            str(generate_max_turns),
            "--generate-tool-specs-path",
            generate_tool_specs_path,
            "--generate-execute-tool-function-path",
            generate_execute_tool_function_path,
        ]
        argv += ["--generate-tool-call-parser", generate_tool_call_parser]
    elif variant == "agentic_tool_call":
        argv += ["--custom-agent-function-path", custom_agent_function_path]
        # these tests steer the mock agent through module state, which a subproc call cannot see
        argv += ["--custom-agent-function-mode", "inline"]
        argv += ["--use-session-server", "v2", "--tito-model", "qwen3"]

    return argv


def listify(x):
    # [FUNCTION DOC] HELPER: Returns the input unchanged if it is a list, otherwise a new one-element
    # list containing it (tuples are wrapped, not converted).
    return x if isinstance(x, list) else [x]


def make_sample(
    *,
    prompt: str | list[dict] = "What is 1+7?",
    tokens: list[int] | None = None,
    response: str = "",
    response_length: int = 0,
    status: Sample.Status = Sample.Status.PENDING,
    multimodal_inputs: dict | None = None,
) -> Sample:
    # [FUNCTION DOC] HELPER: Builds a sample from keyword-only arguments: prompt (text or chat message
    # list, default a short arithmetic question), token ids (none becomes an empty list), response text,
    # response length, status (default pending) and optional multimodal inputs. All other sample fields
    # keep their defaults.
    return Sample(
        prompt=prompt,
        tokens=tokens or [],
        response=response,
        response_length=response_length,
        status=status,
        multimodal_inputs=multimodal_inputs,
    )


@dataclass
class GenerateEnv:
    # [CLASS DOC] HELPER: Environment yielded by the generation fixture: the parsed arguments and the
    # mock sglang server object (exposing a request log, a url and a port).
    args: Namespace
    mock_server: Any


@dataclass
class GenerateResult:
    # [CLASS DOC] HELPER: Outcome of one generate run: the resulting sample or list of samples, and the
    # list of request payloads the mock server received during that run.
    sample: Sample | list[Sample]
    requests: list[dict]


def run_generate(
    env: GenerateEnv,
    sample: Sample,
    sampling_params: dict[str, Any] | None = None,
    *,
    variant: str = "single_turn",
) -> GenerateResult:
    # [FUNCTION DOC] Clears the mock server's request log, runs the async call helper to completion with
    # the environment's args, the sample and the sampling params (default: the module default params),
    # and returns a result with the generated sample(s) and a copy of the requests logged during the
    # run. Side effects: mutates the sample in place (via the generate function) and the mock log.
    env.mock_server.request_log.clear()
    result_sample = run(
        _call_generate(
            env.args,
            sample,
            sampling_params or DEFAULT_SAMPLING_PARAMS,
            variant=variant,
        )
    )
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


async def _call_generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    variant: str = "single_turn",
) -> Sample:
    # [FUNCTION DOC] HELPER: Loads the variant's generate function by its dotted path, creates a
    # generation state from the args, wraps the sample and a copy of the sampling params in a generate
    # input (not evaluation), awaits the function and returns the output's samples field (a sample, or a
    # list of samples for multi sample variants, despite the single-sample return annotation).
    generate_fn = load_generate_function(VARIANT_TO_GENERATE_FN_PATH[variant])
    state = GenerateState(args)
    input = GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params.copy(), evaluation=False)
    output = await generate_fn(input)
    return output.samples


def make_args(
    *,
    variant: str,
    router_port: int,
    use_rollout_routing_replay: bool = False,
    sglang_speculative_algorithm: str | None = None,
    model_name: str = MODEL_NAME,
    extra_argv: list[str] | None = None,
    custom_generate_function_path: str | None = None,
    generate_max_turns: int = 16,
    generate_tool_specs_path: str = "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
    generate_tool_call_parser: str = "qwen25",
    generate_execute_tool_function_path: str = "miles.utils.test_utils.mock_tools.execute_tool_call",
    rollout_max_context_len: int | None = None,
    chat_template_path: str | None = None,
    num_layers: int | None = None,
    moe_router_topk: int | None = None,
) -> Namespace:
    # [FUNCTION DOC] Builds a parsed argument namespace for a generation test. Steps: assemble a command
    # line with fixed values (program name pytest, fsdp train backend, ci test mode, batch and rollout
    # counts of one, one rollout gpu and one gpu per engine, the model as hf checkpoint, a null prompt
    # data path, math reward type, router ip 127.0.0.1 with the given port, response length 16); add the
    # optional chat template path, routing replay flag, speculative algorithm and max context length; add
    # the variant tokens (see the variant command line helper; generate tool settings are forwarded);
    # append any extra tokens; then parse with the real parser while the process argv is temporarily
    # patched. Afterwards sets the num layers and moe router topk attributes directly when given (they are
    # not command line flags in production), and initializes the global http client from the args
    # (side effect). Returns the namespace.
    argv = [
        "pytest",
        "--train-backend",
        "fsdp",
        "--ci-test",
        "--rollout-batch-size",
        "1",
        "--num-rollout",
        "1",
        "--rollout-num-gpus",
        "1",
        "--rollout-num-gpus-per-engine",
        "1",
        "--hf-checkpoint",
        model_name,
        "--prompt-data",
        "/dev/null",
        "--rm-type",
        "math",
        "--sglang-router-ip",
        "127.0.0.1",
        "--sglang-router-port",
        str(router_port),
        "--rollout-max-response-len",
        "16",
    ]
    if chat_template_path:
        argv.extend(["--chat-template-path", chat_template_path])
    if use_rollout_routing_replay:
        argv.append("--use-rollout-routing-replay")
    if sglang_speculative_algorithm:
        argv.extend(["--sglang-speculative-algorithm", sglang_speculative_algorithm])
    if rollout_max_context_len is not None:
        argv.extend(["--rollout-max-context-len", str(rollout_max_context_len)])

    argv.extend(
        extra_argv_for_variant(
            variant,
            custom_generate_function_path=custom_generate_function_path,
            generate_max_turns=generate_max_turns,
            generate_tool_specs_path=generate_tool_specs_path,
            generate_tool_call_parser=generate_tool_call_parser,
            generate_execute_tool_function_path=generate_execute_tool_function_path,
        )
    )

    if extra_argv:
        argv.extend(extra_argv)

    from miles.utils.arguments import parse_args

    with patch("sys.argv", argv):
        args = parse_args()

    # R3 decode shape overrides — not CLI flags (derived from the model config
    # in production). Applied here, before with_session_server copies args into
    # the worker namespace, because sample assembly runs inside the worker.
    if num_layers is not None:
        args.num_layers = num_layers
    if moe_router_topk is not None:
        args.moe_router_topk = moe_router_topk

    init_http_client(args)
    return args


@contextmanager
def _noop_port(port: int):
    # [FUNCTION DOC] HELPER: Context manager that does nothing and yields the port.
    """No-op context manager that just yields the given port."""
    yield port


@contextmanager
def with_session_server(
    backend_url: str,
    args: Namespace,
    *,
    port: int,
):
    # [FUNCTION DOC] Context manager that runs a session server proxying to the backend url on
    # localhost at the given port, and yields the port. Steps: set an instance id from the run uuid with
    # a zero suffix; publish a single session server instance record (address and id) on the args; compute
    # the server config from the args (the host, port, instance id and backend url); build the server and
    # start its web app in a background uvicorn thread; yield; on exit (even on error) stop the thread.
    # Side effect: mutates the args. Any decode-shape args must be set before this runs, since the config
    # is computed here.
    # Mirror wait_session_server_ready (router_manager.py): publish the instance record,
    # id included, that OpenAIEndpointTracer.create picks from.
    instance_id = f"{args.run_uuid}-0"
    args.session_server_instances = [SessionServerInstance(addr=f"127.0.0.1:{port}", instance_id=instance_id)]
    # Sample assembly runs inside the server, so the R3 decode shape args
    # must reach the server config (set them via args_kwargs BEFORE the
    # server starts; assigning to the driver args afterwards has no effect).
    config = compute_session_server_config(
        args,
        host="127.0.0.1",
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
    )
    session_server = SessionServer(config)

    server = UvicornThreadServer(session_server.app, host="127.0.0.1", port=port)
    server.start()

    try:
        yield port
    finally:
        server.stop()


@pytest.fixture
def generation_env(request, variant):
    # [FUNCTION DOC] Pytest fixture that yields a generation environment. It needs a variant fixture
    # from the requesting test and optionally an indirect parameter mapping with an args-kwargs mapping
    # (forwarded to the args factory except model name and agentic return metadata, which are consumed
    # here) and a process-fn-kwargs mapping (mock response text, finish reason, cached tokens, weight
    # version, routed experts and speculative decoding counters). Steps: import the mock tools module
    # lazily; clear all singleton instances; start the mock sglang server with a process function
    # building its response from those kwargs; pick the router port (a free port from 31000 for agentic
    # variants, else the mock server port); build the args; for agentic variants run a session server in
    # front of the mock and set the mock tools module's max turns and return metadata globals; yield the
    # environment. Teardown resets those two globals to none and clears singletons again. Nested
    # function: process function that maps the kwargs to a process result.
    # tests/conftest.py imports this fixture for every test; load the tokenizer-backed helper only when it is used.
    from miles.utils.test_utils import mock_tools

    SingletonMeta.clear_all_instances()
    params = getattr(request, "param", {})
    args_kwargs = params.get("args_kwargs", {})
    model_name = args_kwargs.get("model_name", MODEL_NAME)
    custom_generate_function_path = VARIANT_TO_GENERATE_FN_PATH[variant]

    def process_fn(_):
        # [FUNCTION DOC] HELPER: Ignores the request payload and returns a process result built from
        # the process-fn-kwargs mapping: text (default the module response text), finish reason
        # (default stop), cached tokens (default 0), and meta info with weight version, routed experts
        # and the three speculative decoding counters (all default none).
        x = params.get("process_fn_kwargs", {})
        return ProcessResult(
            text=x.get("response_text", RESPONSE_TEXT),
            finish_reason=x.get("finish_reason", "stop"),
            cached_tokens=x.get("cached_tokens", 0),
            meta_info=ProcessResultMetaInfo(
                weight_version=x.get("weight_version"),
                routed_experts=x.get("routed_experts"),
                spec_num_correct_drafts=x.get("spec_num_correct_drafts"),
                spec_num_proposed_drafts=x.get("spec_num_proposed_drafts"),
                spec_verify_ct=x.get("spec_verify_ct"),
            ),
        )

    is_agentic = variant.startswith("agentic_tool_call")

    with with_mock_server(model_name=model_name, process_fn=process_fn) as mock_server:
        server_port = find_available_port(31000) if is_agentic else mock_server.port
        _FIXTURE_ONLY_KEYS = {"model_name", "agentic_return_metadata"}
        other_args_kwargs = {k: v for k, v in args_kwargs.items() if k not in _FIXTURE_ONLY_KEYS}
        args = make_args(
            variant=variant,
            router_port=server_port,
            model_name=model_name,
            custom_generate_function_path=custom_generate_function_path,
            **other_args_kwargs,
        )

        # Agentic variants need a SessionServer for TITO session tracking;
        # non-agentic variants talk directly to the mock sglang server.
        cm = with_session_server(mock_server.url, args, port=server_port) if is_agentic else _noop_port(server_port)

        with cm:
            if is_agentic:
                mock_tools.AGENTIC_MAX_TURNS = args_kwargs.get("generate_max_turns")
                mock_tools.AGENTIC_RETURN_METADATA = args_kwargs.get("agentic_return_metadata")
            yield GenerateEnv(args=args, mock_server=mock_server)

    mock_tools.AGENTIC_MAX_TURNS = None
    mock_tools.AGENTIC_RETURN_METADATA = None
    SingletonMeta.clear_all_instances()
