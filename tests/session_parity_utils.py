# [MODULE DOC] HELPER: Shared helpers for comparing the v1 and v2 session-server implementations
# on the same agent workload. A public runner starts a real session server (v1 or v2) in a
# background thread against a given inference backend, runs a batch of concurrent agentic
# weather-tool trajectories through the production generate function, snapshots the session
# state just before sample collection, and returns one run record per input sample. A public
# assertion function checks the expected retry/rollback topology and requires that the v1 and v2
# training samples and session metadata are bitwise identical (floats compared by packed bytes,
# arrays by raw bytes, runtime timing values masked). Version names, the sampling seed, timeouts
# and the expected agent counters are module constants.
"""Shared v1/v2 session sample parity helpers."""

from __future__ import annotations

import asyncio
import dataclasses
import enum
import json
import math
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import agentic_tool_call
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.rollout.session.samples.codec import SamplesReply
from miles.rollout.session.server import SessionServer
from miles.rollout.session.types import SessionServerInstance
from miles.utils import http_utils
from miles.utils.http_utils import find_available_port
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

V1 = "v1"
V2 = "v2"
SESSION_PARITY_SEED = 20260803

_CHAT_TIMEOUT_SECS = 120.0
_PICKER_PATH = "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries"
_POSTPROCESSOR_PATH = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"
_RUNTIME_LIFECYCLE_KEYS = frozenset({"t0", "t1", "req_ts", "prev_t1"})
_EXPECTED_AGENT_METADATA = {
    "driver_events": [
        "initial",
        "append_tool",
        "append_user",
        "append_tool",
        "append_system",
        "rollback",
        "rollback",
        "force_final",
        "append_assistant",
    ],
    "rollback_count": 2,
    "user_count": 2,
    "system_count": 1,
    "assistant_input_count": 2,
    "tool_result_count": 2,
    "tool_call_count": 2,
}


@dataclasses.dataclass(frozen=True)
class SessionParityRun:
    # [CLASS DOC] Immutable result of one trajectory run: the session version used, the training
    # samples collected, the session metadata mapping from the collection reply, the session
    # snapshot fetched just before collection, and the reply's reason for being empty (none when
    # samples exist).
    version: str
    samples: list[Sample]
    session_metadata: dict[str, Any]
    pre_collect: dict[str, Any]
    empty_reason: str | None


def run_agentic_retry_trajectories(
    *,
    backend_url: str,
    hf_checkpoint: str,
    version: str,
    input_samples: list[Sample],
) -> list[SessionParityRun]:
    # [FUNCTION DOC] Runs all input samples concurrently through one session-server version.
    # Inputs (keyword only): inference backend URL, tokenizer checkpoint, version string and
    # the list of input samples. Raises a value error for any version other than v1 or v2.
    # Starts a session server for the duration of the block, runs and collects everything in one
    # new event loop, then returns one run record per input sample in the collection order
    # (which follows the input order). Side effects: local port, background server thread,
    # network calls to the backend.
    """Run a concurrent weather-agent batch through one session version."""
    if version not in (V1, V2):
        raise ValueError(f"unknown session version: {version}")

    with _serve_session(backend_url=backend_url, hf_checkpoint=hf_checkpoint, version=version) as args:
        collected = asyncio.run(_run_and_collect(args=args, hf_checkpoint=hf_checkpoint, input_samples=input_samples))

    return [
        SessionParityRun(
            version=version,
            samples=samples,
            session_metadata=reply.session_metadata,
            pre_collect=pre_collect,
            empty_reason=reply.empty_reason,
        )
        for pre_collect, reply, samples in collected
    ]


async def _run_and_collect(
    *,
    args: SimpleNamespace,
    hf_checkpoint: str,
    input_samples: list[Sample],
) -> list[tuple[dict[str, Any], SamplesReply, list[Sample]]]:
    # [FUNCTION DOC] HELPER: Drives the agent for every input sample concurrently and gathers
    # outputs. Uses one HTTP client with no timeout and substitutes it as the shared client of
    # the HTTP utilities module. Temporarily replaces the tracer's sample-collection method with
    # a wrapper that, before delegating to the original, fetches the session's state from the
    # tracer's base URL (asserting a 200 status) and stores that JSON together with the real
    # reply, keyed by the identity of the input sample. For each input sample the metadata gets
    # the tito model, verification cycle count and tool-call failure mode from the args; the
    # generate input uses greedy temperature zero, the fixed sampling seed, model name equal to
    # the checkpoint and 128 new tokens, evaluation off. Runs the agentic tool-call generate
    # function. Returns, per input sample in order, a triple of the pre-collection snapshot, the
    # collection reply and the output samples (a single sample is wrapped in a list).
    async with httpx.AsyncClient(timeout=None) as client:
        with patch.object(http_utils, "_http_client", client):
            original_collect = OpenAIEndpointTracer.collect_samples
            collected: dict[int, tuple[dict[str, Any], SamplesReply]] = {}

            async def collect_with_snapshot(
                tracer: OpenAIEndpointTracer,
                collected_input_sample: Sample,
                *,
                max_seq_len: int | None,
                agent_metadata: dict | None = None,
            ) -> SamplesReply:
                # [FUNCTION DOC] HELPER: Replacement for the tracer's collect method. Fetches
                # the session snapshot (assert status 200), calls the original method with the
                # same sequence-length limit and agent metadata, records snapshot plus reply
                # under the identity of the input sample, and returns the reply.
                response = await client.get(tracer.base_url)
                assert response.status_code == 200, response.text
                reply = await original_collect(
                    tracer,
                    collected_input_sample,
                    max_seq_len=max_seq_len,
                    agent_metadata=agent_metadata,
                )
                collected[id(collected_input_sample)] = (response.json(), reply)
                return reply

            async def generate_one(input_sample: Sample):
                # [FUNCTION DOC] HELPER: Runs the agent for one sample (see the enclosing
                # function for settings), mutating its metadata in place, and returns the pair
                # of the input sample and the generate output.
                input_sample.metadata.update(
                    {
                        "tito_model": args.tito_model,
                        "session_verify_cycles": args.session_verify_cycles,
                        "tool_call_failure_mode": args.tool_call_failure_mode,
                    }
                )
                generate_input = GenerateFnInput(
                    state=SimpleNamespace(args=args),
                    sample=input_sample,
                    sampling_params={
                        "model": hf_checkpoint,
                        "temperature": 0,
                        "sampling_seed": SESSION_PARITY_SEED,
                        "max_new_tokens": 128,
                    },
                    evaluation=False,
                )
                output = await agentic_tool_call.generate(generate_input)
                return input_sample, output

            with patch.object(OpenAIEndpointTracer, "collect_samples", collect_with_snapshot):
                outputs = await asyncio.gather(*(generate_one(sample) for sample in input_samples))

    results = []
    for input_sample, output in outputs:
        samples = output.samples if isinstance(output.samples, list) else [output.samples]
        pre_collect, reply = collected[id(input_sample)]
        results.append((pre_collect, reply, samples))
    return results


def assert_agentic_retry_trajectory_parity(v1: SessionParityRun, v2: SessionParityRun) -> None:
    # [FUNCTION DOC] Asserts, for the pair of runs of the same trajectory: the versions are v1
    # and v2; neither is empty; each has exactly one sample and seven request/response records;
    # the weather tool round trip holds for both snapshots; the v2 metadata tree has nine nodes
    # with the expected parent links (node 3 has children 4, 5 and 6; 6 leads to 7 then 8) and
    # three leaves with the expected root-to-leaf paths; the v2 sample's selected leaf is node 8
    # with parent 7, the expected path and the response id of that node; the v2 session agent
    # metadata equals the expected agent counters and event sequence, and both samples carry
    # those counters; the maximum-trim-token values of the v1 sample and v2 session agree; the v1
    # session metadata equals the v2 session metadata minus its agent and tree entries, bitwise;
    # finally both samples are bitwise equal with metadata compared after projection.
    # Raises assertion errors on any mismatch.
    """Assert agent coverage, retry topology, and training-payload parity."""
    assert v1.version == V1
    assert v2.version == V2
    assert v1.empty_reason is None
    assert v2.empty_reason is None
    assert len(v1.samples) == 1
    assert len(v2.samples) == 1
    assert len(v1.pre_collect["records"]) == 7
    assert len(v2.pre_collect["records"]) == 7
    _assert_weather_tool_round_trip(v1.pre_collect)
    _assert_weather_tool_round_trip(v2.pre_collect)

    tree = v2.pre_collect["metadata"]["tree"]
    assert [(node["id"], node["parent"]) for node in tree["nodes"]] == [
        (0, None),
        (1, 0),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 6),
        (8, 7),
    ]
    assert [(leaf["node_id"], leaf["path_node_ids"]) for leaf in tree["leaves"]] == [
        (4, [0, 1, 2, 3, 4]),
        (5, [0, 1, 2, 3, 5]),
        (8, [0, 1, 2, 3, 6, 7, 8]),
    ]

    selected_leaf = v2.samples[0].metadata["leaf"]
    assert selected_leaf["node_id"] == 8
    assert selected_leaf["parent"] == 7
    assert selected_leaf["path_node_ids"] == [0, 1, 2, 3, 6, 7, 8]
    assert selected_leaf["response_id"] == tree["nodes"][8]["response_id"]
    assert v2.session_metadata["agent"] == _EXPECTED_AGENT_METADATA
    for key, value in _EXPECTED_AGENT_METADATA.items():
        assert v1.samples[0].metadata[key] == value
        assert v2.samples[0].metadata[key] == value
    assert v1.samples[0].metadata["max_trim_tokens"] == v2.session_metadata["max_trim_tokens"]

    v2_linear_metadata = {key: value for key, value in v2.session_metadata.items() if key not in ("agent", "tree")}
    _assert_bits_equal(v1.session_metadata, v2_linear_metadata, path="session_metadata")
    assert_sample_bitwise_equal(
        v1.samples[0],
        v2.samples[0],
        metadata_projection=_training_metadata_projection,
    )


def assert_sample_bitwise_equal(
    left: Sample,
    right: Sample,
    *,
    metadata_projection: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    # [FUNCTION DOC] Asserts two samples are bitwise identical. Requires the same type and that
    # neither carries attributes beyond the declared dataclass fields. Each declared field is
    # compared with the bit-exact comparison; the metadata field is first passed through the
    # optional projection function (applied to both sides) when one is given. Raises assertion
    # errors naming the field path.
    """Compare every declared Sample field without float tolerance."""
    assert type(left) is type(right)
    left_extras = set(vars(left)) - {field.name for field in dataclasses.fields(left)}
    right_extras = set(vars(right)) - {field.name for field in dataclasses.fields(right)}
    assert left_extras == right_extras == set()

    for field in dataclasses.fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if field.name == "metadata" and metadata_projection is not None:
            left_value = metadata_projection(left_value)
            right_value = metadata_projection(right_value)
        _assert_bits_equal(left_value, right_value, path=f"sample.{field.name}")


@contextmanager
def _serve_session(*, backend_url: str, hf_checkpoint: str, version: str) -> Iterator[SimpleNamespace]:
    # [FUNCTION DOC] HELPER: Context manager that serves one session-server instance. Picks a
    # free port from 31000 upward, names the instance after the version, builds an args namespace
    # (120 second router timeout, thinking disabled in the chat template, the qwen3 tito model,
    # no replay features, retract pause mode, one local instance, an inline custom agent function
    # from the session verification test utilities, one verification cycle, rollback on tool-call
    # failure, the v2 picker that drops same-prompt retries and the default v2 postprocessor)
    # and the matching server config, creates the server app and runs it on a background
    # uvicorn thread. Yields the args namespace; stops the server in a final block.
    port = find_available_port(31000)
    instance_id = f"session-parity-{version}"
    session_addr = f"127.0.0.1:{port}"
    args = SimpleNamespace(
        miles_router_timeout=_CHAT_TIMEOUT_SECS,
        hf_checkpoint=hf_checkpoint,
        chat_template_path=None,
        apply_chat_template_kwargs={"enable_thinking": False},
        tito_model="qwen3",
        sglang_speculative_algorithm=None,
        use_session_server=version,
        use_rollout_routing_replay=False,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        pause_generation_mode="retract",
        session_server_ip="127.0.0.1",
        session_server_instances=[SessionServerInstance(addr=session_addr, instance_id=instance_id)],
        save_debug_trajectory_data=None,
        custom_agent_function_path="miles.utils.test_utils.session_verify_agent.run_agent",
        custom_agent_function_mode="inline",
        partial_rollout=False,
        max_seq_len=None,
        session_verify_cycles=1,
        tool_call_failure_mode="rollback",
        session_sample_picker_path=_PICKER_PATH,
        session_sample_postprocessor_path=_POSTPROCESSOR_PATH,
    )
    config = make_session_server_config(
        host=args.session_server_ip,
        port=port,
        instance_id=instance_id,
        backend_url=backend_url,
        timeout=args.miles_router_timeout,
        hf_checkpoint=args.hf_checkpoint,
        apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        tito_model=args.tito_model,
        use_session_server=args.use_session_server,
        pause_generation_mode=args.pause_generation_mode,
        session_sample_picker_path=args.session_sample_picker_path,
        session_sample_postprocessor_path=args.session_sample_postprocessor_path,
    )
    app = SessionServer(config).app
    server = UvicornThreadServer(app, host=args.session_server_ip, port=port)
    server.start()
    try:
        yield args
    finally:
        server.stop()


def _training_metadata_projection(metadata: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a deep copy of the sample metadata with the leaf and
    # maximum-trim-token entries removed (they are v2-only) and, when a lifecycle entry exists,
    # replaced by a list of segments with runtime timestamps masked (a single mapping becomes a
    # one-element list).
    projected = deepcopy(metadata)
    projected.pop("leaf", None)
    projected.pop("max_trim_tokens", None)
    lifecycle = projected.get("lifecycle")
    if lifecycle is not None:
        segments = lifecycle if isinstance(lifecycle, list) else [lifecycle]
        projected["lifecycle"] = [_project_lifecycle_segment(segment) for segment in segments]
    return projected


def _assert_weather_tool_round_trip(snapshot: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Checks the weather agent's recorded traffic. Every request declares
    # exactly one tool named get_weather. Records 0 and 2 must each hold a single tool call to
    # that tool whose JSON arguments equal a location of Beijing (record 0) or Shanghai
    # (record 2); the last message of the following request (record 1 or 3) must be a tool-role
    # message whose call id equals that tool call's id.
    records = snapshot["records"]
    for record in records:
        assert [tool["function"]["name"] for tool in record["request"]["tools"]] == ["get_weather"]

    for call_index, result_index, location in ((0, 1, "Beijing"), (2, 3, "Shanghai")):
        [tool_call] = records[call_index]["response"]["choices"][0]["message"]["tool_calls"]
        assert tool_call["function"]["name"] == "get_weather"
        assert json.loads(tool_call["function"]["arguments"]) == {"location": location}

        tool_result = records[result_index]["request"]["messages"][-1]
        assert tool_result["role"] == "tool"
        assert tool_result["tool_call_id"] == tool_call["id"]


def _project_lifecycle_segment(segment: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Copies a lifecycle segment and replaces each non-none value of the
    # runtime timestamp keys (start, end, request time, previous end) by a placeholder string,
    # after asserting each is exactly an integer or float type and finite.
    projected = dict(segment)
    for key in _RUNTIME_LIFECYCLE_KEYS & projected.keys():
        if projected[key] is not None:
            assert type(projected[key]) in (int, float)
            assert math.isfinite(projected[key])
            projected[key] = "<runtime>"
    return projected


def _assert_bits_equal(left: Any, right: Any, *, path: str) -> None:
    # [FUNCTION DOC] HELPER: Converts both values to their canonical bit trees and raises an
    # assertion error stating the path is not bitwise equal if the trees differ. Conversion
    # errors propagate.
    left_bits = _bit_tree(left, path=path)
    right_bits = _bit_tree(right, path=path)
    if left_bits != right_bits:
        raise AssertionError(f"{path} is not bitwise equal")


def _bit_tree(value: Any, *, path: str) -> Any:
    # [FUNCTION DOC] HELPER: Recursively converts a value into a hashable-comparable tree that
    # distinguishes values by exact type and bits. Dataclasses (raising an assertion error on
    # undeclared extra attributes) become a tagged tuple of type and ordered field trees; numpy
    # arrays must be non-object and C-contiguous and become dtype string, shape and raw bytes;
    # numpy scalars become dtype and bytes; enums become type and value tree; dicts become the
    # items (key tree, value tree) sorted by the text form of the key tree, so insertion order
    # is ignored; lists and tuples keep their type and element order; floats are packed as
    # big-endian 8-byte doubles (so negative zero and NaN payloads are distinguished); none and
    # exact bool, int, str, bytes types are kept as (type, value). Anything else raises a type
    # error naming the path. Subclasses of int, str, etc. are rejected by the exact-type check.
    if dataclasses.is_dataclass(value):
        declared = {field.name for field in dataclasses.fields(value)}
        extras = set(vars(value)) - declared
        if extras:
            raise AssertionError(f"{path} has undeclared fields: {sorted(extras)}")
        return (
            "dataclass",
            type(value),
            tuple(
                (field.name, _bit_tree(getattr(value, field.name), path=f"{path}.{field.name}"))
                for field in dataclasses.fields(value)
            ),
        )
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise AssertionError(f"{path} has object dtype")
        if not value.flags.c_contiguous:
            raise AssertionError(f"{path} is not C-contiguous")
        return "ndarray", value.dtype.str, value.shape, value.tobytes(order="C")
    if isinstance(value, np.generic):
        return "numpy-scalar", value.dtype.str, value.tobytes()
    if isinstance(value, enum.Enum):
        return "enum", type(value), _bit_tree(value.value, path=f"{path}.value")
    if isinstance(value, dict):
        items = [
            (_bit_tree(key, path=f"{path}.<key>"), _bit_tree(item, path=f"{path}[{key!r}]"))
            for key, item in value.items()
        ]
        return "dict", tuple(sorted(items, key=lambda pair: repr(pair[0])))
    if isinstance(value, (list, tuple)):
        return type(value), tuple(_bit_tree(item, path=f"{path}[{index}]") for index, item in enumerate(value))
    if isinstance(value, float):
        return "float64", struct.pack(">d", value)
    if value is None or type(value) in (bool, int, str, bytes):
        return type(value), value
    raise TypeError(f"unsupported value at {path}: {type(value).__name__}")
