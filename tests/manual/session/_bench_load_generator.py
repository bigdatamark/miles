# [MODULE DOC] HELPER: Load-generator half of the session-server HTTP benchmark (a private
# module of the manual benchmark). Defines the per-session driver, a single-event-loop driver
# for many concurrent sessions, and a process entry point that runs one shard and returns its
# results through a queue. Results are a pair: latency sample lists (chat reply latency, and
# optionally records-read latency, in milliseconds) and an aggregate mapping of summed counters
# (completed turns, classified server and transport errors for chat and for the records read,
# sessions that completed without any chat error, and up to five first-error strings).
"""Multi-process load generator for the session-server HTTP benchmark.

Lives beside `bench_session_server_overhead` so it is importable by qualified
module name in spawn children (`sys.path[0]` is the script dir, inherited by
the spawn bootstrap) — the same mechanism `_mock_r3_backend` relies on.

Why a separate, multi-process driver
------------------------------------
A single asyncio event loop + one httpx client cannot cleanly drive ~1000
concurrent sessions over loopback: it cycles the ephemeral-port range and resets
pooled connections under its own scheduling pressure, surfacing as
`httpx.ReadError` on a chat POST — which is NOT retried (chat mutates session
state) — and aborting the whole run, even though the server stays healthy
(every answered request is 200, no worker death, no OOM). Sharding the sessions
across `--bench-driver-procs` OS processes (each its own loop + client) keeps
per-loop connection volume low and mirrors the real rollout, where many workers
drive sessions rather than one process.

Robust per-session driving
---------------------------
Each session is driven defensively: a failure is CLASSIFIED (server non-2xx vs
client transport reset, for chat vs the GET-records read) and tallied, never
raised. One reset therefore does not abort a generator — it becomes a measurable
rate. That is what lets the bench answer "does it stay stable" instead of dying
on the first hiccup.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

_STATUS_KEYS = (
    "completed_turns",
    "chat_server_errors",
    "chat_transport_errors",
    "get_ok",
    "get_server_errors",
    "get_transport_errors",
)


def _transport_error_types() -> tuple:
    # [FUNCTION DOC] HELPER: Lazily imports the HTTP client library and returns a tuple of its
    # client-side failure classes: read, write, connect and remote-protocol errors plus pool,
    # connect, read and write timeouts. Used as an exception filter.
    """Client-side connection/transport faults (NOT server non-2xx responses)."""
    import httpx

    return (
        httpx.ReadError,
        httpx.WriteError,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.WriteTimeout,
    )


async def _idempotent_request(client, method: str, url: str):
    # [FUNCTION DOC] HELPER: Sends one body-less request through the client and returns the
    # response. If a read error, connect error or remote-protocol error is raised, the identical
    # request is sent exactly once more; a second failure propagates. Timeouts and non-2xx
    # responses are not retried.
    """Issue an idempotent control-plane request, retrying ONCE on a connection-
    level transport fault with a fresh connection.

    A reset surfaces client-side BEFORE any handler ran, so the request had no
    effect — reissuing it on a new connection is correct transport-fault
    handling, not retrying a server-side failure (a non-2xx is surfaced, not
    retried). Only create/get/delete come through here; chat is never retried
    (it mutates session state).
    """
    import httpx

    try:
        return await client.request(method, url)
    except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError):
        return await client.request(method, url)


def _empty_status() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a fresh per-session status mapping with every counter key
    # set to zero and a first-error entry set to none.
    st: dict[str, Any] = {k: 0 for k in _STATUS_KEYS}
    st["first_error"] = None
    return st


async def drive_one_session(
    client, base_url, request_bodies, samples, *, get_records, tool_interval=0.0
) -> dict[str, Any]:
    # [FUNCTION DOC] Drives one session lifecycle against the server. Steps: create the session
    # (idempotent retry once; a transport fault or non-200 counts as a chat error and returns
    # early, with the failure noted as the first error); read the session id from the response;
    # for each prebuilt request body, post it to the session's chat-completions path with a JSON
    # content type, record the elapsed milliseconds into the reply-latency sample list, stop on
    # a transport fault or non-200 (counted as chat errors, the first-error text including the
    # turn index and up to 120 characters of the body), otherwise count a completed turn and,
    # except after the last turn, sleep for the tool interval when it is positive. Optionally
    # fetch the session records (retry once), recording latency on 200 and counting ok or
    # server or transport errors. Finally delete the session, ignoring transport faults.
    # Inputs: client, base URL, list of body bytes, shared samples mapping, keyword flags for
    # records fetch and tool interval seconds (default zero). Returns the status mapping; never
    # raises for classified faults. Mutates the samples mapping.
    # KNOWN ISSUE: a create failure is tallied under the chat error counters, not a separate one,
    # and a non-200 delete response is ignored without being counted.
    """create -> N x chat (-> GET records) -> delete for one session, robustly.

    Never raises: returns a per-session status dict (completed turns + classified
    error counts). Records per-chat reply latency in `samples['reply_latency_ms']`
    and, when `get_records`, the GET latency in `samples['get_records_ms']`.
    A chat transport reset / non-2xx ends that session (its state is unknown / a
    server error is real) but does not abort the generator. `tool_interval` is
    the idle a session waits after each chat (except the last) before its next
    turn (simulated tool/env step).
    """
    transport_errors = _transport_error_types()
    st = _empty_status()

    try:
        r = await _idempotent_request(client, "POST", f"{base_url}/sessions")
    except transport_errors as e:
        st["chat_transport_errors"] += 1
        st["first_error"] = f"create transport {type(e).__name__}"
        return st
    if r.status_code != 200:
        st["chat_server_errors"] += 1
        st["first_error"] = f"create {r.status_code}"
        return st
    sid = r.json()["session_id"]

    last_idx = len(request_bodies) - 1
    for idx, body in enumerate(request_bodies):
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                f"{base_url}/sessions/{sid}/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
        except transport_errors as e:
            st["chat_transport_errors"] += 1
            st["first_error"] = st["first_error"] or f"chat[{idx}] transport {type(e).__name__}"
            return st
        samples["reply_latency_ms"].append((time.perf_counter() - t0) * 1000)
        if resp.status_code != 200:
            st["chat_server_errors"] += 1
            st["first_error"] = st["first_error"] or f"chat[{idx}] {resp.status_code} {resp.text[:120]}"
            return st
        st["completed_turns"] += 1
        if tool_interval > 0 and idx != last_idx:
            await asyncio.sleep(tool_interval)

    if get_records:
        t0 = time.perf_counter()
        g = None
        try:
            g = await _idempotent_request(client, "GET", f"{base_url}/sessions/{sid}")
        except transport_errors as e:
            st["get_transport_errors"] += 1
            st["first_error"] = st["first_error"] or f"get transport {type(e).__name__}"
        if g is not None:
            if g.status_code == 200:
                samples["get_records_ms"].append((time.perf_counter() - t0) * 1000)
                st["get_ok"] += 1
            else:
                st["get_server_errors"] += 1
                st["first_error"] = st["first_error"] or f"get {g.status_code}"

    try:
        await _idempotent_request(client, "DELETE", f"{base_url}/sessions/{sid}")
    except transport_errors:
        pass
    return st


def _empty_agg() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a fresh aggregate mapping: every counter key zero, a
    # sessions-ok count of zero and an empty list of first errors.
    agg: dict[str, Any] = {k: 0 for k in _STATUS_KEYS}
    agg["sessions_ok"] = 0
    agg["first_errors"] = []
    return agg


async def lg_drive_all(base_urls, request_bodies, num_sessions, get_records, tool_interval):
    # [FUNCTION DOC] Creates the sample lists (reply latency, plus records latency when records
    # are fetched) and one shared HTTP client with a 600 second timeout, connection limits of
    # twice the session count (at least 2) for both total and keep-alive connections, and a
    # keep-alive expiry of 2 seconds (shorter than the server's window so the client drops idle
    # connections first). Sends a health request to every base URL, then runs all sessions
    # concurrently, each choosing a random base URL. Aggregates: sums every counter over the
    # per-session statuses, counts sessions with no chat server or transport errors as ok, and
    # keeps the first-error text of at most the first five sessions that have one. Returns
    # (samples, aggregate). Network side effects against the given servers.
    """Drive `num_sessions` sessions concurrently in ONE event loop/client.

    Each session picks one instance URL at create time and stays on it for its
    whole lifecycle — the URL is the router, mirroring
    `OpenAIEndpointTracer.create`. Returns (samples, agg). Used directly for the
    single-process path and as the body of each spawned load generator.
    """
    import httpx

    samples: dict[str, list] = {"reply_latency_ms": []}
    if get_records:
        samples["get_records_ms"] = []
    # keepalive_expiry < the server's keep-alive window so the CLIENT drops an
    # idle connection first (a fresh connection, not a reset) rather than reusing
    # one the server already closed.
    limits = httpx.Limits(
        max_connections=max(2, num_sessions * 2),
        max_keepalive_connections=max(2, num_sessions * 2),
        keepalive_expiry=2.0,
    )
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0), limits=limits) as client:
        for base_url in base_urls:
            await client.get(f"{base_url}/health")
        statuses = await asyncio.gather(
            *(
                drive_one_session(
                    client,
                    random.choice(base_urls),
                    request_bodies,
                    samples,
                    get_records=get_records,
                    tool_interval=tool_interval,
                )
                for _ in range(num_sessions)
            )
        )

    agg = _empty_agg()
    for s in statuses:
        for k in _STATUS_KEYS:
            agg[k] += s[k]
        if s["chat_server_errors"] == 0 and s["chat_transport_errors"] == 0:
            agg["sessions_ok"] += 1
        if s["first_error"] and len(agg["first_errors"]) < 5:
            agg["first_errors"].append(s["first_error"])
    return samples, agg


def load_generator_entry(base_urls, request_bodies, num_sessions, get_records, tool_interval, result_queue) -> None:
    # [FUNCTION DOC] Process entry point for one load-generator shard. Tries to set the process
    # title to a benchmark load-generator label (any failure, including a missing library, is
    # swallowed), runs the all-sessions driver in a new event loop, and puts the resulting
    # (samples, aggregate) pair on the result queue. Returns nothing.
    """spawn `Process` target: drive a shard of sessions, put (samples, agg) on
    `result_queue`. Drives no tokenizer — it only replays pre-built request
    bodies — so generator startup is cheap."""
    try:
        import setproctitle

        setproctitle.setproctitle("miles-bench-loadgen")
    except Exception:
        pass
    samples, agg = asyncio.run(lg_drive_all(base_urls, request_bodies, num_sessions, get_records, tool_interval))
    result_queue.put((samples, agg))
