# [MODULE DOC] Command-line benchmark script (manual, needs a tokenizer checkpoint and the full
# Miles stack). Flow: load a tokenizer and a token-in-token-out tokenizer wrapper, synthesize
# per-turn request and response bodies (repeated single-token text of the requested sizes plus a
# base64 filler blob standing in for routed-expert data, either accumulated over the sequence or
# only the per-turn delta), spawn several mock backend processes and several real session
# server processes on consecutive local ports, then replay the requests with one or more load
# generator processes while a background sampler tracks peak resident memory of the server
# process trees only. Prints a text report (latency percentiles, throughput, errors, memory)
# and optionally writes the result mapping as JSON. All Miles imports are deferred into
# functions so spawned child processes that re-import this module stay cheap. Flags: sessions,
# turns, input/output tokens, r3 scale (bytes per token), incremental r3, checkpoint, tito
# model, server implementation version, append role, chat template path, JSON output path,
# raw-sample inclusion, server instance count, records fetch, inference and tool intervals,
# mock backend process count, driver process count.
"""HTTP end-to-end benchmark for Session Server per-turn overhead.

Stands up N independent single-process session servers (`run_session_server`,
the production chassis; N=1 is exactly today's single-instance deployment),
points them at a mock R3 backend that replays pre-built synthetic response
bytes, and drives `sessions` concurrent sessions x `turns` sequential chat
turns over loopback HTTP. Each session picks one instance at create time and
stays on it — the URL is the router, mirroring `OpenAIEndpointTracer.create`.
The server does its real work — request/response `json.loads`, validate, TITO
tokenization, record store — while the mock backend removes model generation
from the measurement. Reports per-chat reply latency, throughput, and peak
server-group RSS.

Run it directly:

  python tests/manual/session/bench_session_server_overhead.py \
      --sessions 32 --turns 50 --r3-scale 1000 --session-server-instances 16

Add `--incremental-r3` to mock a backend that returns only the new per-turn
`routed_experts` payload instead of the full accumulated sequence payload;
add `--get-records` to also exercise the full-records GET (read path).
Use `--append-role tool` to make turns after the first append tool responses
to assistant tool calls instead of user messages. This selects the benchmark
traffic shape; the TITO family's fixed template owns the allowed role surface.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import multiprocessing
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# No miles imports at module level, deliberately: multiprocessing "spawn"
# children (the load generators and the mock backend) re-import this script as
# __mp_main__, and the miles chain drags in transformers (~10s per child). All
# miles imports live inside the driver-side functions that use them.
if TYPE_CHECKING:
    from miles.rollout.session.config import SessionServerConfig

DEFAULT_HF_CHECKPOINT = "Qwen/Qwen3-0.6B"
DEFAULT_TITO_MODEL = "qwen3"
DEFAULT_APPEND_ROLE = "user"

# Tool-append turns declare this in every request: some encoders (DeepSeek V4)
# render assistant history differently when no tools are declared, which breaks
# the TITO incremental diff's prefix stability server-side.
BENCH_TOOL = {
    "type": "function",
    "function": {
        "name": "bench_tool",
        "description": "Synthetic benchmark tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}


@dataclass(frozen=True)
class TurnSpec:
    # [CLASS DOC] HELPER: Immutable record of one synthetic conversation turn: the serialized
    # request JSON bytes, the serialized backend response JSON bytes, the expected prompt token
    # ids, counts of content input tokens, content output tokens and completion tokens, the
    # number of routed-expert tokens carried this turn and for the full sequence, the raw byte
    # size of that payload and its encoded character length.
    request_body: bytes
    response_body: bytes
    expected_prompt_token_ids: list[int]
    content_input_tokens: int
    content_output_tokens: int
    completion_tokens: int
    r3_token_count: int
    full_sequence_r3_token_count: int
    r3_raw_bytes: int
    r3_json_chars: int


def _positive_int(value: str) -> int:
    # [FUNCTION DOC] HELPER: Command-line argument type. Parses an integer and rejects zero or
    # negative values by raising the argument parser's type error naming the offending value.
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _non_negative_int(value: str) -> int:
    # [FUNCTION DOC] HELPER: Command-line argument type. Parses an integer, rejecting negatives
    # with the argument parser's type error.
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def _non_negative_float(value: str) -> float:
    # [FUNCTION DOC] HELPER: Command-line argument type. Parses a float, rejecting negatives
    # with the argument parser's type error.
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative number, got {value!r}")
    return parsed


def _pct(values: list[float], q: float) -> float:
    # [FUNCTION DOC] HELPER: Nearest-rank percentile. Returns not-a-number for an empty list;
    # otherwise sorts the values and returns the element at rank ceil(q times count) minus one,
    # clamped into the valid index range. q is a fraction between 0 and 1.
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _summary(values: list[float]) -> dict[str, float]:
    # [FUNCTION DOC] HELPER: Returns a mapping with mean, median (50th), 95th and 99th percentile
    # and maximum of the values, keyed with an "_ms" suffix (values are assumed milliseconds).
    # Mean and maximum are not-a-number for an empty list.
    return {
        "mean_ms": statistics.mean(values) if values else float("nan"),
        "p50_ms": _pct(values, 0.50),
        "p95_ms": _pct(values, 0.95),
        "p99_ms": _pct(values, 0.99),
        "max_ms": max(values) if values else float("nan"),
    }


def _find_repeatable_token_id(tokenizer) -> int:
    # [FUNCTION DOC] HELPER: Searches a fixed list of candidate texts (space-x, space-a,
    # space-the, space-token, space-zero, capital A), encoding each without special tokens; for
    # each resulting token id, decodes it (keeping special tokens) and re-encodes; the first id
    # that round-trips to exactly itself is returned. Raises a runtime error if none does.
    for text in (" x", " a", " the", " token", " 0", "A"):
        for token_id in tokenizer.encode(text, add_special_tokens=False):
            decoded = tokenizer.decode([token_id], skip_special_tokens=False)
            if tokenizer.encode(decoded, add_special_tokens=False) == [token_id]:
                return token_id
    raise RuntimeError("could not find a repeatable one-token text unit for this tokenizer")


def _make_text_with_token_count(tokenizer, token_id: int, token_count: int) -> tuple[str, list[int]]:
    # [FUNCTION DOC] HELPER: Builds text that tokenizes to exactly the requested number of copies
    # of one token. Zero gives an empty string and empty list. Otherwise decodes the repeated id
    # list, re-encodes the text without special tokens, and raises a runtime error (reporting
    # requested and actual counts) if the length changed. Returns (text, re-encoded ids).
    if token_count == 0:
        return "", []
    token_ids = [token_id] * token_count
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    roundtrip_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(roundtrip_ids) != token_count:
        raise RuntimeError(
            "repeatable token changed length after decode/encode roundtrip: "
            f"requested={token_count}, actual={len(roundtrip_ids)}"
        )
    return text, roundtrip_ids


def _make_r3_blob(raw_bytes: int) -> str:
    # [FUNCTION DOC] HELPER: Makes a filler payload. Zero bytes gives an empty string. Otherwise
    # repeats the byte sequence 0..250 until at least the requested length, truncates to exactly
    # raw_bytes and returns it base64-encoded as ASCII text (about 4/3 the raw size).
    if raw_bytes == 0:
        return ""
    pattern = bytes(range(251))
    repeats = math.ceil(raw_bytes / len(pattern))
    raw = (pattern * repeats)[:raw_bytes]
    return base64.b64encode(raw).decode("ascii")


def _completion_token_ids(
    tito_tokenizer,
    tokenizer,
    messages: list[dict[str, Any]],
    assistant_message: dict[str, Any],
    tools: list[dict[str, Any]] | None,
):
    # [FUNCTION DOC] HELPER: Computes the token ids the assistant message adds to the prompt.
    # Renders the prompt text (with generation prompt) and the full text with the assistant
    # message appended (no generation prompt), both as text with the tokenizer's default template
    # arguments for the given tools. Raises a runtime error if the full text does not start with
    # the prompt text. Returns the encoding (no special tokens) of the remaining suffix.
    template_args = tito_tokenizer.default_template_args(tools)
    prompt_text = tito_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, template_args=template_args
    )
    full_text = tito_tokenizer.apply_chat_template(
        messages + [assistant_message],
        add_generation_prompt=False,
        tokenize=False,
        template_args=template_args,
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError("assistant response does not extend the rendered prompt")
    return tokenizer.encode(full_text[len(prompt_text) :], add_special_tokens=False)


def _build_response_body(
    assistant_message: dict[str, Any],
    completion_token_ids: list[int],
    r3_blob: str,
) -> bytes:
    # [FUNCTION DOC] HELPER: Serializes a fake chat-completion response to compact JSON bytes.
    # One choice with the given assistant message, finish reason "stop" and meta info holding the
    # completion token count, per-token [logprob, token id] pairs and the routed-experts blob.
    # The synthetic logprob of the i-th token is minus ((i mod 1024) + 1) / 1024. Other fields
    # are constants (id, object type, created zero, model name "synthetic").
    output_token_logprobs = [
        [-((idx % 1024) + 1) / 1024.0, token_id] for idx, token_id in enumerate(completion_token_ids)
    ]
    response = {
        "id": "synthetic-session-overhead",
        "object": "chat.completion",
        "created": 0,
        "model": "synthetic",
        "choices": [
            {
                "index": 0,
                "message": assistant_message,
                "finish_reason": "stop",
                "meta_info": {
                    "completion_tokens": len(completion_token_ids),
                    "output_token_logprobs": output_token_logprobs,
                    "routed_experts": r3_blob,
                },
            }
        ],
    }
    return json.dumps(response, separators=(",", ":")).encode()


def _build_turn_specs(
    tokenizer,
    tito_tokenizer,
    turns: int,
    input_tokens: int,
    output_tokens: int,
    r3_scale: int,
    incremental_r3: bool,
    tool_appends: bool,
):
    # [FUNCTION DOC] HELPER: Generates the list of turn records for one synthetic conversation.
    # Picks a repeatable token, then per turn: builds input and output text of the requested
    # token counts; the input message is a user message, or for turns after the first in tool
    # mode a tool message answering the previous call id (call ids are numbered with five
    # digits); the assistant message carries a single no-argument tool call in tool mode. The
    # request messages are copies of the history plus the input message; tools are declared (one
    # fixed benchmark tool) only in tool mode. Computes prompt token ids (tokenized chat template
    # with generation prompt) and completion token ids. Full-sequence routed-expert token count is
    # max(0, prompt + completion - 1); in incremental mode the turn's count is the increase over
    # the previous turn (a decrease raises a runtime error), otherwise the full count. Raw bytes
    # are count times scale; the filler blob is generated; request body is compact JSON of the
    # messages (plus tools when present), response body from the builder. History then grows by
    # the request messages and the assistant message. Returns the list in turn order.
    token_id = _find_repeatable_token_id(tokenizer)
    history: list[dict[str, Any]] = []
    specs: list[TurnSpec] = []
    previous_full_sequence_r3_token_count = 0

    for turn_idx in range(turns):
        input_text, input_content_ids = _make_text_with_token_count(tokenizer, token_id, input_tokens)
        output_text, output_content_ids = _make_text_with_token_count(tokenizer, token_id, output_tokens)

        if tool_appends and history:
            input_message: dict[str, Any] = {
                "role": "tool",
                "content": input_text,
                "tool_call_id": f"call_bench_{turn_idx - 1:05d}",
                "name": "bench_tool",
            }
        else:
            input_message = {"role": "user", "content": input_text}
        assistant_message: dict[str, Any] = {"role": "assistant", "content": output_text}
        if tool_appends:
            # The appended tool turn must answer a prior assistant tool_call,
            # or templates have no call boundary to render the tool turn against.
            assistant_message["tool_calls"] = [
                {
                    "id": f"call_bench_{turn_idx:05d}",
                    "type": "function",
                    "function": {"name": "bench_tool", "arguments": {}},
                }
            ]
        request_messages = [dict(message) for message in history] + [input_message]
        tools = [BENCH_TOOL] if tool_appends else None

        prompt_token_ids = tito_tokenizer.apply_chat_template(
            request_messages,
            add_generation_prompt=True,
            tokenize=True,
            template_args=tito_tokenizer.default_template_args(tools),
        )
        completion_token_ids = _completion_token_ids(
            tito_tokenizer, tokenizer, request_messages, assistant_message, tools
        )

        full_sequence_r3_token_count = max(0, len(prompt_token_ids) + len(completion_token_ids) - 1)
        if incremental_r3:
            if full_sequence_r3_token_count < previous_full_sequence_r3_token_count:
                raise RuntimeError(
                    "full-sequence r3 token count decreased between turns: "
                    f"previous={previous_full_sequence_r3_token_count}, current={full_sequence_r3_token_count}"
                )
            r3_token_count = full_sequence_r3_token_count - previous_full_sequence_r3_token_count
        else:
            r3_token_count = full_sequence_r3_token_count
        r3_raw_bytes = r3_token_count * r3_scale
        r3_blob = _make_r3_blob(r3_raw_bytes)

        payload: dict[str, Any] = {"messages": request_messages}
        if tools is not None:
            payload["tools"] = tools
        request_body = json.dumps(payload, separators=(",", ":")).encode()
        response_body = _build_response_body(assistant_message, completion_token_ids, r3_blob)
        specs.append(
            TurnSpec(
                request_body=request_body,
                response_body=response_body,
                expected_prompt_token_ids=prompt_token_ids,
                content_input_tokens=len(input_content_ids),
                content_output_tokens=len(output_content_ids),
                completion_tokens=len(completion_token_ids),
                r3_token_count=r3_token_count,
                full_sequence_r3_token_count=full_sequence_r3_token_count,
                r3_raw_bytes=r3_raw_bytes,
                r3_json_chars=len(r3_blob),
            )
        )

        history = request_messages + [assistant_message]
        previous_full_sequence_r3_token_count = full_sequence_r3_token_count

    return specs


def _fmt_ms_stats(stats: dict[str, float]) -> str:
    # [FUNCTION DOC] HELPER: Formats a summary mapping as one line: mean, p50, p95, p99 and max,
    # each with three decimals and a millisecond suffix.
    return (
        f"mean={stats['mean_ms']:.3f}ms  p50={stats['p50_ms']:.3f}ms  "
        f"p95={stats['p95_ms']:.3f}ms  p99={stats['p99_ms']:.3f}ms  max={stats['max_ms']:.3f}ms"
    )


def _write_json(result: dict[str, Any], path: str) -> None:
    # [FUNCTION DOC] HELPER: Writes a copy of the result mapping to the path as indented JSON,
    # first dropping the raw-samples entry if its value is none. Overwrites the file and prints
    # a line naming the path. Raises on I/O failure or non-serializable values.
    payload = dict(result)
    if payload.get("raw_samples_ms") is None:
        payload.pop("raw_samples_ms", None)
    with Path(path).open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] wrote {path}")


# ---------------------------------------------------------------------------
# Drive the REAL deployed session server over HTTP.
#
# Stands up the actual server the rollout path runs (`run_session_server`, one
# independent process per port), points it at a mock backend that returns
# pre-built large-R3 response bytes, and drives the production-shaped workload
# over loopback HTTP: `sessions` concurrent sessions, each doing `turns`
# sequential chat turns. So the server does the real json.loads / validate /
# record work end to end — one instance pays it on one event loop under one
# GIL; N instances spread the sessions across N processes with the client
# picking an instance per session (no router hop). We measure per-request
# reply latency and aggregate, plus peak RSS of the whole server process group.
# ---------------------------------------------------------------------------


def _build_server_config(
    bench_args,
    *,
    chat_template_path,
    chat_template_kwargs,
    ip: str,
    port: int,
    instance_idx: int,
    backend_url: str,
) -> SessionServerConfig:
    # [FUNCTION DOC] HELPER: Builds the server configuration for one instance. Inputs: parsed
    # benchmark arguments, chat template path and kwargs (may be none), bind host, port, the
    # instance index (instance id is "bench-i" plus the index) and the backend URL. Fixed
    # choices: routing replay on, indexer replay and sampling-support replay off, no speculative
    # algorithm, layers, router top-k, debug data or LoRA (rank zero, not train-only), 600 second
    # timeout, strict session message matching, no sample picker or postprocessor, no top
    # logprobs, "selected" sampling logprobs mode, temperature 1.0. From the benchmark arguments:
    # checkpoint, tito model and server implementation; the pause mode is "in_place" when
    # incremental routed-expert payloads are on, else "retract". Imports the config class lazily.
    """The config carrying exactly what one session-server instance reads.

    `use_rollout_routing_replay=True` makes the server inject
    `return_routed_experts=True` upstream and exercise the R3-strip path, i.e.
    the production-shaped large-R3 scenario the overhead doc is about.

    `pause_generation_mode` follows `--incremental-r3` so the delta-shaped
    payloads run against a server that actually requests additional R3
    (`routed_experts_start_len` on every upstream chat request).

    `use_session_server` selects which implementation is measured, so it comes
    from the command line rather than being pinned here; the sample picker and
    postprocessor stay unset, matching `SessionServerConfig.from_args` defaults.
    """
    from miles.rollout.session.config import SessionServerConfig

    return SessionServerConfig(
        hf_checkpoint=bench_args.hf_checkpoint,
        chat_template_path=chat_template_path,
        apply_chat_template_kwargs=chat_template_kwargs,
        tito_model=bench_args.tito_model,
        use_rollout_routing_replay=True,
        use_rollout_indexer_replay=False,
        use_sampling_support_replay=False,
        sglang_speculative_algorithm=None,
        num_layers=None,
        moe_router_topk=None,
        save_debug_trajectory_data=None,
        lora_rank=0,
        lora_adapter_path=None,
        lora_train_only=False,
        timeout=600.0,
        backend_url=backend_url,
        host=ip,
        port=port,
        instance_id=f"bench-i{instance_idx}",
        use_session_server=bench_args.use_session_server,
        session_message_matcher="strict",
        pause_generation_mode="in_place" if bench_args.incremental_r3 else "retract",
        session_sample_picker_path=None,
        session_sample_postprocessor_path=None,
        rollout_top_logprobs_num=0,
        rollout_sampling_logprobs_mode="selected",
        rollout_temperature=1.0,
    )


def _start_backend(
    response_bodies: list[bytes], ip: str, inference_interval: float = 0.0, procs: int = 1
) -> tuple[list[multiprocessing.process.BaseProcess], str]:
    # [FUNCTION DOC] HELPER: Starts the mock backend. Finds a free port from 28000 upward, spawns
    # the given number of non-daemon processes (spawn start method) all running the backend
    # server with the response bodies, host, shared port and inference interval, then waits up to
    # 120 seconds for the port to answer for each process. Returns the process list and the base
    # URL of the form http plus host and port. Side effects: child processes, network listening.
    # The backend module is imported by bare name, relying on the script directory being first
    # on the module search path.
    """Spawn the mock R3 backend as `procs` processes; return (procs, base_url).

    All shards bind the same port via SO_REUSEPORT (see `run_mock_r3_backend`)
    so the workers keep a single backend URL while the kernel spreads their
    connections across shards — the mock's stateless turn mapping makes
    any-request-to-any-shard safe. Each shard holds its own copy of
    `response_bodies`. `inference_interval` (seconds) is the simulated
    generation delay the mock waits before returning each chat response.
    """
    # `_mock_r3_backend` lives beside this script; `sys.path[0]` is this
    # script's dir, which the spawn child inherits, so a top-level import resolves
    # in both parent and child without a tests/manual/session package.
    from _mock_r3_backend import run_mock_r3_backend

    from miles.utils.http_utils import find_available_port, wait_for_server_ready

    ctx = multiprocessing.get_context("spawn")
    port = find_available_port(28000)
    backend_procs = []
    for _ in range(procs):
        proc = ctx.Process(
            target=run_mock_r3_backend,
            args=(response_bodies, ip, port, inference_interval),
            name="bench-mock-r3-backend",
            daemon=False,
        )
        proc.start()
        backend_procs.append(proc)
    # The port probe passes once ANY shard listens; shards bind before serving
    # and workers only open connections after the (much later) server readiness,
    # so all shards are accepting by the time load arrives.
    for proc in backend_procs:
        wait_for_server_ready(ip, port, proc, timeout=120.0)
    return backend_procs, f"http://{ip}:{port}"


def _proc_tree_rss_bytes(root_pids: list[int]) -> int:
    # [FUNCTION DOC] HELPER: Sums resident set size in bytes over each root process and all its
    # recursive children, counting each pid once. Roots that no longer exist are skipped; a
    # process that vanishes or denies access while read contributes zero. Returns the total.
    """Summed RSS of each pid in `root_pids` and all its descendants.

    For OLD this is the single server process tree; for NEW it is the router
    and worker process trees. The bench driver process is deliberately excluded
    so the OLD-vs-NEW RSS comparison reflects only the server, not the driver's
    tokenizer + synthetic-body memory.
    """
    import psutil

    seen: set[int] = set()
    total = 0
    for root_pid in root_pids:
        try:
            root = psutil.Process(root_pid)
        except psutil.NoSuchProcess:
            continue
        for p in [root] + root.children(recursive=True):
            if p.pid in seen:
                continue
            seen.add(p.pid)
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    return total


class _RSSSampler:
    # [CLASS DOC] HELPER: Context manager that records the peak summed memory of given process
    # trees. Entering starts a daemon thread that samples every interval seconds (default 0.25);
    # exiting stops it, waits up to 2 seconds, and takes one last sample. The result is the
    # peak_bytes attribute (starts at zero).
    """Background thread sampling peak RSS of one or more process trees."""

    def __init__(self, root_pids: list[int], interval: float = 0.25):
        # [FUNCTION DOC] HELPER: Stores the root pids (entries equal to none are dropped), the
        # interval, a stop event, the not-yet-started thread and a zero peak.
        import threading

        self._root_pids = [p for p in root_pids if p is not None]
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bench-rss-sampler", daemon=True)
        self.peak_bytes = 0

    def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body: until stopped, updates the peak with the current
        # summed memory, then waits for the interval (waking early when stopped).
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, _proc_tree_rss_bytes(self._root_pids))
            self._stop.wait(self._interval)

    def __enter__(self) -> _RSSSampler:
        # [FUNCTION DOC] HELPER: Starts the sampling thread and returns the sampler itself.
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        # [FUNCTION DOC] HELPER: Signals the thread to stop, joins it with a 2 second timeout and
        # takes a final sample into the peak. Does not suppress exceptions.
        self._stop.set()
        self._thread.join(timeout=2.0)
        # One final reading in case the peak landed between samples.
        self.peak_bytes = max(self.peak_bytes, _proc_tree_rss_bytes(self._root_pids))


def _merge_agg(into: dict[str, Any], src: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Adds the source aggregate into the destination in place. The
    # first-errors list is appended only up to a total of ten entries (extra source entries
    # dropped); every other key is added numerically, treating a missing destination key as zero.
    for k, v in src.items():
        if k == "first_errors":
            room = 10 - len(into["first_errors"])
            if room > 0:
                into["first_errors"].extend(v[:room])
        else:
            into[k] = into.get(k, 0) + v


def _drive_workload(
    base_urls: list[str],
    request_bodies: list[bytes],
    num_sessions: int,
    *,
    get_records: bool,
    tool_interval: float,
    driver_procs: int,
):
    # [FUNCTION DOC] HELPER: Runs the load. With one driver process (or fewer) it runs all
    # sessions in this process's event loop and returns the samples, aggregate and elapsed wall
    # seconds. Otherwise it splits the session count into near-equal shards (the remainder spread
    # one each over the first shards; empty shards dropped), spawns one non-daemon load generator
    # per shard, collects exactly one result per shard from a queue (2400 second timeout each)
    # before joining, merges latency samples and aggregates, joins with a 120 second timeout, and
    # in a final block terminates every child. Wall time spans spawn to completion of joins.
    # Raises the queue timeout error if a generator never reports. Generators are imported
    # lazily by bare module name from the script directory.
    """Drive the HTTP workload; return (samples, agg, wall_s).

    `driver_procs == 1` runs all sessions in this process's event loop.
    `driver_procs > 1` shards the sessions across that many spawned load
    generators (each its own event loop + httpx client) so a single loop's
    connection-churn ceiling does not cap the achievable session count — see
    `_bench_load_generator`. The generators run NO tokenizer (they replay
    pre-built request bodies), so they are cheap to spawn and are deliberately
    excluded from the server RSS measurement.
    """
    # Imported here (not at module top) so the spawn children that target the
    # mock backend / session server do not transitively import httpx eagerly.
    from _bench_load_generator import lg_drive_all, load_generator_entry

    if driver_procs <= 1:
        wall_start = time.perf_counter()
        samples, agg = asyncio.run(lg_drive_all(base_urls, request_bodies, num_sessions, get_records, tool_interval))
        return samples, agg, time.perf_counter() - wall_start

    base, rem = divmod(num_sessions, driver_procs)
    shards = [base + (1 if i < rem else 0) for i in range(driver_procs)]
    shards = [s for s in shards if s > 0]

    ctx = multiprocessing.get_context("spawn")
    result_q: Any = ctx.Queue()
    procs: list[multiprocessing.process.BaseProcess] = []
    samples = {"reply_latency_ms": []}
    if get_records:
        samples["get_records_ms"] = []
    agg: dict[str, Any] = {
        "completed_turns": 0,
        "chat_server_errors": 0,
        "chat_transport_errors": 0,
        "get_ok": 0,
        "get_server_errors": 0,
        "get_transport_errors": 0,
        "sessions_ok": 0,
        "first_errors": [],
    }
    wall_start = time.perf_counter()
    try:
        for n in shards:
            p = ctx.Process(
                target=load_generator_entry,
                args=(base_urls, request_bodies, n, get_records, tool_interval, result_q),
                name="bench-loadgen",
                daemon=False,
            )
            p.start()
            procs.append(p)
        # Drain all results BEFORE joining (a large queued item would otherwise
        # block the feeder and deadlock the join).
        for _ in shards:
            s, a = result_q.get(timeout=2400)
            samples["reply_latency_ms"].extend(s["reply_latency_ms"])
            if get_records:
                samples["get_records_ms"].extend(s.get("get_records_ms", []))
            _merge_agg(agg, a)
        for p in procs:
            p.join(timeout=120)
    finally:
        for p in procs:
            _terminate_proc(p)
    wall_s = time.perf_counter() - wall_start
    return samples, agg, wall_s


def run_http_bench(args) -> dict[str, Any]:
    # [FUNCTION DOC] Orchestrates a full run from parsed arguments and returns the result mapping.
    # Chat template selection: explicit path wins (no kwargs); tito model "default" uses none;
    # otherwise the fixed template for the tito model is resolved. Loads the tokenizer (remote
    # code trusted) and tito tokenizer, builds the turn specs, starts the mock backend, picks a
    # free base port from 33000 and requires every further consecutive port to be free (else
    # raises a runtime error), spawns one session server process per instance before waiting on
    # any, waits up to 180 seconds per server, then drives the workload under the memory sampler
    # that watches only the server processes. In a final block all server then backend processes
    # are terminated. Totals: turns = sessions times turns; throughput token counts are scaled by
    # the fraction of completed turns (zero when no turns) times sessions; the raw routed-expert
    # byte estimate counts every turn for every session. The result holds configuration echoes,
    # error counters, throughput per second (not-a-number if wall time is not positive), the
    # memory peak, per-turn spec summaries, latency summaries per sample kind, and optionally the
    # raw samples.
    # KNOWN ISSUE: the total and retained routed-expert byte estimates are identical and are not
    # scaled by the completed fraction, so they over-report when sessions fail.
    from miles.utils.chat_template_utils import get_tito_tokenizer, resolve_fixed_chat_template
    from miles.utils.http_utils import find_available_port, is_port_available, wait_for_server_ready
    from miles.utils.processing_utils import load_tokenizer

    if args.chat_template_path is not None:
        chat_template_path = args.chat_template_path
        chat_template_kwargs = None
    elif args.tito_model == "default":
        chat_template_path = None
        chat_template_kwargs = None
    else:
        chat_template_path, chat_template_kwargs = resolve_fixed_chat_template(args.tito_model)

    # Build the synthetic per-turn request/response bodies once (CPU, off the clock).
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=chat_template_path, trust_remote_code=True)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=args.tito_model,
        chat_template_kwargs=chat_template_kwargs,
    )
    specs = _build_turn_specs(
        tokenizer,
        tito_tokenizer,
        turns=args.turns,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        r3_scale=args.r3_scale,
        incremental_r3=args.incremental_r3,
        tool_appends=args.append_role == "tool",
    )
    response_bodies = [spec.response_body for spec in specs]

    ip = "127.0.0.1"
    instances = args.session_server_instances
    backend_procs, backend_url = _start_backend(
        response_bodies, ip, inference_interval=args.inference_interval, procs=args.mock_backend_procs
    )

    server_procs: list[multiprocessing.process.BaseProcess] = []
    try:
        from miles.rollout.session.server import run_session_server

        # Consecutive ports from one free base, mirroring production's
        # `--session-server-port START --session-server-workers N` deployment.
        base_port = find_available_port(33000)
        ports = list(range(base_port, base_port + instances))
        for port in ports[1:]:
            if not is_port_available(port):
                raise RuntimeError(f"port {port} in the bench range {ports[0]}..{ports[-1]} is already in use")
        base_urls = [f"http://{ip}:{port}" for port in ports]

        # Spawn every instance before waiting on any (matches
        # wait_session_server_ready): N instances pay ~one transformers import of
        # wall-time, not N.
        ctx = multiprocessing.get_context("spawn")
        for idx, port in enumerate(ports):
            server_config = _build_server_config(
                args,
                chat_template_path=chat_template_path,
                chat_template_kwargs=chat_template_kwargs,
                ip=ip,
                port=port,
                instance_idx=idx,
                backend_url=backend_url,
            )
            proc = ctx.Process(
                target=run_session_server,
                args=(server_config,),
                name=f"bench-session-server-{idx}",
                daemon=False,
            )
            proc.start()
            server_procs.append(proc)
        for port, proc in zip(ports, server_procs, strict=True):
            wait_for_server_ready(ip, port, proc, timeout=180.0)
        # Measure ONLY the server group, not the bench driver process which
        # holds the tokenizer + synthetic response bodies.
        server_root_pids = [p.pid for p in server_procs]

        request_bodies = [spec.request_body for spec in specs]
        with _RSSSampler(server_root_pids) as rss:
            samples, agg, wall_s = _drive_workload(
                base_urls,
                request_bodies,
                args.sessions,
                get_records=args.get_records,
                tool_interval=args.tool_interval,
                driver_procs=args.bench_driver_procs,
            )
        peak_rss_bytes = rss.peak_bytes
    finally:
        for proc in server_procs:
            _terminate_proc(proc)
        for proc in backend_procs:
            _terminate_proc(proc)

    total_turns = args.sessions * args.turns
    completed_turns = agg.get("completed_turns", total_turns)
    # token-throughput is reported on COMPLETED work (frac=1 for clean runs, so
    # this matches the strict-mode numbers; honest when sessions fail partway).
    frac = (completed_turns / total_turns) if total_turns else 0.0
    content_tokens = (
        frac * args.sessions * sum(spec.content_input_tokens + spec.content_output_tokens for spec in specs)
    )
    output_tokens = frac * args.sessions * sum(spec.content_output_tokens for spec in specs)
    completion_tokens = frac * args.sessions * sum(spec.completion_tokens for spec in specs)
    # Records are never pruned (the training data path reads them via GET
    # /sessions/{id}), so the retained R3 equals everything transferred.
    total_r3_raw_bytes = args.sessions * sum(spec.r3_raw_bytes for spec in specs)

    return {
        "mode": "http",
        "session_server_instances": instances,
        "bench_driver_procs": args.bench_driver_procs,
        "mock_backend_procs": args.mock_backend_procs,
        "get_records": bool(args.get_records),
        "inference_interval_s": args.inference_interval,
        "tool_interval_s": args.tool_interval,
        "sessions": args.sessions,
        "turns_per_session": args.turns,
        "total_turns": total_turns,
        "completed_turns": completed_turns,
        "sessions_ok": agg.get("sessions_ok"),
        "chat_server_errors": agg.get("chat_server_errors"),
        "chat_transport_errors": agg.get("chat_transport_errors"),
        "get_ok": agg.get("get_ok"),
        "get_server_errors": agg.get("get_server_errors"),
        "get_transport_errors": agg.get("get_transport_errors"),
        "first_errors": agg.get("first_errors", []),
        "input_tokens_per_turn": args.input_tokens,
        "output_tokens_per_turn": args.output_tokens,
        "r3_mode": "incremental" if args.incremental_r3 else "accumulated",
        "r3_scale_raw_bytes_per_token": args.r3_scale,
        "hf_checkpoint": args.hf_checkpoint,
        "tito_model": args.tito_model,
        "append_role": args.append_role,
        "chat_template_path": chat_template_path,
        "chat_template_kwargs": chat_template_kwargs,
        "wall_s": wall_s,
        "throughput_turns_per_s": completed_turns / wall_s if wall_s > 0 else float("nan"),
        "throughput_content_tokens_per_s": content_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_completion_tokens_per_s": completion_tokens / wall_s if wall_s > 0 else float("nan"),
        "throughput_output_content_tokens_per_s": output_tokens / wall_s if wall_s > 0 else float("nan"),
        "total_r3_raw_bytes_transfer_estimate": total_r3_raw_bytes,
        "retained_r3_raw_bytes_estimate": total_r3_raw_bytes,
        "peak_rss_bytes": peak_rss_bytes,
        "turn_specs": [
            {
                "turn_index": idx,
                "prompt_tokens": len(spec.expected_prompt_token_ids),
                "completion_tokens": spec.completion_tokens,
                "content_input_tokens": spec.content_input_tokens,
                "content_output_tokens": spec.content_output_tokens,
                "r3_token_count": spec.r3_token_count,
                "full_sequence_r3_token_count": spec.full_sequence_r3_token_count,
                "r3_raw_bytes": spec.r3_raw_bytes,
                "r3_json_chars": spec.r3_json_chars,
                "request_body_bytes": len(spec.request_body),
                "response_body_bytes": len(spec.response_body),
            }
            for idx, spec in enumerate(specs)
        ],
        "metrics": {name: _summary(values) for name, values in samples.items()},
        "raw_samples_ms": samples if args.include_raw_samples else None,
    }


def _terminate_proc(proc: multiprocessing.process.BaseProcess) -> None:
    # [FUNCTION DOC] HELPER: Best-effort stop of a child process: if alive, sends terminate and
    # waits up to 5 seconds; if still alive with a known pid, force-kills it and waits up to 2
    # seconds. Value, lookup and attribute errors are swallowed (e.g. never-started processes).
    import signal as _signal

    try:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        if proc.is_alive() and proc.pid is not None:
            os.kill(proc.pid, _signal.SIGKILL)
            proc.join(timeout=2.0)
    except (ValueError, ProcessLookupError, AttributeError):
        pass


def _fmt_http_block(result: dict[str, Any]) -> str:
    # [FUNCTION DOC] HELPER: Renders the result mapping as a multi-line report framed by
    # 72-character rules: header with instance, driver and backend process counts; workload
    # shape; interval and routed-expert settings; tokenizer; final-turn sizes; transfer estimate
    # in MiB; session and turn success counts and chat error counts; records-fetch counts only if
    # that option was on; up to five first errors if any; chat latency stats and (if sampled)
    # records latency stats; wall clock; throughputs; peak memory in GiB. Returns the text.
    metrics = result["metrics"]
    last_spec = result["turn_specs"][-1]
    lines = [
        "=" * 72,
        f"Session Server HTTP end-to-end benchmark (instances={result['session_server_instances']}, "
        f"driver_procs={result.get('bench_driver_procs', 1)}, backend_procs={result.get('mock_backend_procs', 1)})",
        "=" * 72,
        f"  sessions x turns             : {result['sessions']} x {result['turns_per_session']} "
        f"({result['total_turns']} turns)",
        f"  content tokens / turn         : input={result['input_tokens_per_turn']}  "
        f"output={result['output_tokens_per_turn']}",
        f"  inference / tool interval     : {result['inference_interval_s']}s / {result['tool_interval_s']}s",
        f"  r3 mode                       : {result['r3_mode']}",
        f"  r3 raw bytes / token          : {result['r3_scale_raw_bytes_per_token']}",
        f"  tokenizer / TITO              : {result['hf_checkpoint']} / {result['tito_model']}",
        f"  final-turn prompt/completion  : {last_spec['prompt_tokens']} / {last_spec['completion_tokens']} tokens",
        f"  final-turn response body      : {last_spec['response_body_bytes'] / 1024:.1f} KiB",
        f"  total r3 transfer estimate    : {result['total_r3_raw_bytes_transfer_estimate'] / 1024 / 1024:.1f} MiB raw",
        "-" * 72,
        f"  sessions ok / total           : {result.get('sessions_ok')} / {result['sessions']}    "
        f"completed turns: {result.get('completed_turns')} / {result['total_turns']}",
        f"  chat errors (srv / transport) : {result.get('chat_server_errors')} / {result.get('chat_transport_errors')}",
        *(
            [
                f"  GET ok / errors (srv / trans) : {result.get('get_ok')} / "
                f"{result.get('get_server_errors')} / {result.get('get_transport_errors')}"
            ]
            if result.get("get_records")
            else []
        ),
        *([f"  first errors                  : {result['first_errors'][:5]}"] if result.get("first_errors") else []),
        "-" * 72,
        f"  reply latency (per chat)      : {_fmt_ms_stats(metrics['reply_latency_ms'])}",
        *(
            [f"  GET full-records latency      : {_fmt_ms_stats(metrics['get_records_ms'])}"]
            if "get_records_ms" in metrics
            else []
        ),
        "-" * 72,
        f"  wall clock                    : {result['wall_s']:.3f}s",
        f"  throughput                    : {result['throughput_turns_per_s']:.1f} turns/s",
        f"  content-token throughput      : {result['throughput_content_tokens_per_s']:.1f} tokens/s",
        f"  completion-token throughput   : {result['throughput_completion_tokens_per_s']:.1f} tokens/s",
        f"  peak RSS (server proc group)  : {result['peak_rss_bytes'] / 1024 / 1024 / 1024:.2f} GiB",
        "=" * 72,
    ]
    return "\n".join(lines)


def main() -> None:
    # [FUNCTION DOC] Script entry point: defines and parses the command-line options (defaults:
    # 32 sessions, 4 turns, 64 input and 64 output tokens, r3 scale 1000, checkpoint
    # Qwen3-0.6B, tito model "qwen3", server version "v2", append role "user", 1 server
    # instance, zero intervals, 8 mock backend processes, 1 driver process), runs the benchmark,
    # prints the report, and writes JSON when an output path was given.
    parser = argparse.ArgumentParser(description="HTTP end-to-end Session Server overhead benchmark")
    parser.add_argument("--sessions", type=_positive_int, default=32, help="number of sessions to create")
    parser.add_argument("--turns", type=_positive_int, default=4, help="turns per session")
    parser.add_argument("--input-tokens", type=_non_negative_int, default=64, help="new user-content tokens per turn")
    parser.add_argument(
        "--output-tokens",
        type=_non_negative_int,
        default=64,
        help="assistant-content tokens per turn",
    )
    parser.add_argument(
        "--r3-scale",
        type=_non_negative_int,
        default=1000,
        help="raw routed_experts bytes per r3 token",
    )
    parser.add_argument(
        "--incremental-r3",
        action="store_true",
        help="mock routed_experts as only the per-turn delta instead of the full accumulated sequence",
    )
    parser.add_argument("--hf-checkpoint", default=DEFAULT_HF_CHECKPOINT, help="tokenizer checkpoint or local path")
    parser.add_argument("--tito-model", default=DEFAULT_TITO_MODEL, help="TITO tokenizer family")
    parser.add_argument(
        "--use-session-server",
        choices=["v1", "v2"],
        default="v2",
        help="session server implementation to measure",
    )
    parser.add_argument(
        "--append-role",
        choices=["user", "tool"],
        default=DEFAULT_APPEND_ROLE,
        help="role used for benchmark turns after the first; selects traffic shape only",
    )
    parser.add_argument("--chat-template-path", default=None, help="explicit chat template path")
    parser.add_argument("--json-out", default=None, help="persist the run as a JSON artifact")
    parser.add_argument(
        "--include-raw-samples", action="store_true", help="include every per-turn sample in JSON output"
    )
    parser.add_argument(
        "--session-server-instances",
        type=_positive_int,
        default=1,
        help="number of independent single-process session servers on consecutive ports; "
        "each session picks one at create time (the URL is the router)",
    )
    parser.add_argument(
        "--get-records",
        action="store_true",
        help="per session, GET the full (retained-R3) records before delete; off by default "
        "because the hundreds-of-MiB read is the read path, not the latency/throughput headline",
    )
    parser.add_argument(
        "--inference-interval",
        type=_non_negative_float,
        default=0.0,
        help="seconds the mock backend waits before returning each chat response "
        "(simulated generation time). Default 0 = respond immediately (continuous back-to-back).",
    )
    parser.add_argument(
        "--tool-interval",
        type=_non_negative_float,
        default=0.0,
        help="seconds a session idles after each chat response before sending its next "
        "turn (simulated tool/env step producing the next input). Default 0 = fire turns back-to-back.",
    )
    parser.add_argument(
        "--mock-backend-procs",
        type=_positive_int,
        default=8,
        help="shard the mock backend across this many SO_REUSEPORT processes on one port. A single "
        "process saturates one core streaming large-R3 bodies to 16 workers and caps the measurement; "
        "the default keeps the mock off the critical path. Set 1 to reproduce older single-process runs.",
    )
    parser.add_argument(
        "--bench-driver-procs",
        type=_positive_int,
        default=1,
        help="shard the client load across this many load-generator processes (each its "
        "own event loop + httpx client). Default 1. Raise it to drive many hundreds of sessions without a "
        "single event loop's connection-churn ceiling; mirrors the real multi-worker rollout.",
    )
    args = parser.parse_args()

    result = run_http_bench(args)
    print(_fmt_http_block(result))
    if args.json_out:
        _write_json(result, args.json_out)


if __name__ == "__main__":
    main()
