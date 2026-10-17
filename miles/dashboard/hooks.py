# [MODULE DOC]
# Process-side producers feeding the collector actor. A phase sink turns Timer intervals into phase
# events stamped with the process identity (node, GPUs, rank, role) and batches them; a trajectory sink
# turns per-sample lifecycle callbacks into trajectory events; registration functions attach sinks,
# announce the router and the engine topology, and report the data buffer length. Batching: a buffer
# is flushed when it holds 64 events or 2 seconds have passed since the last flush. Module globals
# keep the single phase sink, the last engine fingerprint and a warner. Every entry point is best
# effort and swallows errors (with rate-limited warnings).
"""Process-side hooks feeding the dashboard collector.

These are the functions the (tiny) core wiring calls: a Timer event sink on
every training rank, and engine-topology registration from the rollout
manager. Every entry point is a no-op when the dashboard is disabled or the
collector cannot be reached, and every sink swallows its own exceptions with
rate-limited warnings — a deliberate exception to fail-loud (design doc
§6.3): observability must never be able to kill a training step.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass

from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import (
    DataBufferSample,
    EngineInfo,
    PhaseEvent,
    Role,
    TopologySnapshot,
    TrajectoryEvent,
    TrajectoryEventKind,
)
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.timer import Timer
from miles.utils.workers.worker_provider.base import BaseWorkerProvider

logger = logging.getLogger(__name__)

BATCH_MAX_EVENTS = 64
BATCH_MAX_SECONDS = 2.0


@dataclass
class _Identity:
    # [CLASS DOC] HELPER: Process identity: node IP string, list of integer GPU ids assigned to the
    # process, and distributed rank (-1 when torch distributed is not initialized).
    node: str
    gpus: list[int]
    rank: int


def _default_resolve_identity() -> _Identity:
    # [FUNCTION DOC] HELPER: Builds the identity from the current node IP, the GPU ids Ray assigned to
    # this process (as ints) and the torch distributed rank if initialized, else -1. Imports ray and
    # torch lazily.
    import ray
    import torch.distributed as dist

    from miles.utils.misc import get_current_node_ip

    return _Identity(
        node=get_current_node_ip(),
        gpus=[int(gpu) for gpu in ray.get_gpu_ids()],
        rank=dist.get_rank() if dist.is_initialized() else -1,
    )


# test seams; production always uses the defaults
_resolve_identity = _default_resolve_identity


class PhaseSink:
    # [CLASS DOC] Timer event sink bound to a collector handle and a process role. Callable with
    # (name, start, end) for closed intervals; begin() reports an interval opening. Buffers closed
    # events under a lock, flushes by size/age on arrival plus a lazily started background thread that
    # flushes idle buffers every 2 seconds. Identity is resolved lazily and re-resolved for training
    # processes until a real rank is known.
    """Timer event sink: stamps process identity onto intervals and pushes
    them to the collector in batches (fire-and-forget)."""

    def __init__(self, handle, role: str):
        # [FUNCTION DOC] Store the handle and role; empty buffer; last flush time now; no identity
        # yet; fresh warner; no flusher thread yet.
        self._handle = handle
        self.role = role
        self._buffer: list[PhaseEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._identity: _Identity | None = None
        self._warner = RateLimitedWarner(logger)
        self._flusher = None

    def begin(self, name: str, t0: float) -> None:
        # [FUNCTION DOC] Resolves identity under the lock (re-resolving when the stored rank is
        # negative and the role is training), builds a phase event with the open-end marker as end time
        # and immediately sends it (as a one-element list) to the collector without buffering. Any
        # exception is swallowed with a rate-limited warning.
        """Timer.start notification: push an OPEN interval immediately (no
        batching — starts are rare and while-open visibility is the point).
        The closing event supersedes it on the read side."""
        try:
            with self._lock:
                if self._identity is None or (self._identity.rank < 0 and self.role == Role.TRAIN):
                    self._identity = _resolve_identity()
                identity = self._identity
            event = PhaseEvent(
                name=name,
                t0=t0,
                t1=PhaseEvent.OPEN_T1,
                node=identity.node,
                gpus=identity.gpus,
                rank=identity.rank,
                role=self.role,
            )
            self._handle.push_phases.remote([event])
        except Exception:
            self._warner.warn("dashboard phase sink failed; dropping events")

    def __call__(self, name: str, t0: float, t1: float) -> None:
        # [FUNCTION DOC] Closed-interval notification. Under the lock: resolve identity as in begin,
        # append the phase event and take a batch if due. Then make sure the background flusher
        # exists and send the batch (if any) to the collector asynchronously. Exceptions are swallowed
        # with a rate-limited warning (events may be lost).
        try:
            with self._lock:
                # lazy: torch.distributed is usually not initialized yet when
                # the sink attaches; re-resolve until a real rank appears
                if self._identity is None or (self._identity.rank < 0 and self.role == Role.TRAIN):
                    self._identity = _resolve_identity()
                identity = self._identity
                self._buffer.append(
                    PhaseEvent(
                        name=name,
                        t0=t0,
                        t1=t1,
                        node=identity.node,
                        gpus=identity.gpus,
                        rank=identity.rank,
                        role=self.role,
                    )
                )
                batch = self._take_batch_if_due()
            self._ensure_flusher()
            if batch:
                self._handle.push_phases.remote(batch)
        except Exception:
            self._warner.warn("dashboard phase sink failed; dropping events")

    def _take_batch_if_due(self) -> list[PhaseEvent] | None:
        # [FUNCTION DOC] HELPER: Call with the lock held. Returns none if the buffer has fewer than 64
        # events and less than 2 seconds passed since the last flush; otherwise empties the buffer,
        # resets the last-flush time and returns the events.
        if len(self._buffer) < BATCH_MAX_EVENTS and time.monotonic() - self._last_flush < BATCH_MAX_SECONDS:
            return None
        batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        return batch

    def flush(self) -> None:
        # [FUNCTION DOC] Takes the whole buffer under the lock and sends it to the collector if
        # non-empty; swallows exceptions with a rate-limited warning. Does not update the last-flush
        # time.
        try:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if batch:
                self._handle.push_phases.remote(batch)
        except Exception:
            self._warner.warn("dashboard phase sink flush failed; dropping events")

    def _ensure_flusher(self) -> None:
        # [FUNCTION DOC] HELPER: Starts, once, a daemon thread "dashboard-phase-flush" that loops
        # forever: sleep 2 seconds, and flush if the buffer is non-empty. Needed because batches are
        # only taken when new events arrive. KNOWN ISSUE: the check-and-start is not protected by the
        # lock so concurrent first calls could start two threads; the thread never stops (even after
        # the sink is detached); and the function re-imports the threading module redundantly.
        """Background flush: _take_batch_if_due only runs when a new event
        arrives, so on a process that goes idle (train ranks during a
        shared-engine eval) the closing events of its last phases sit in the
        buffer indefinitely and the UI shows them as still-open."""
        if self._flusher is not None:
            return
        import threading

        def _loop():
            # [FUNCTION DOC] HELPER: Thread body described by the enclosing function; never returns.
            while True:
                time.sleep(BATCH_MAX_SECONDS)
                with self._lock:
                    due = bool(self._buffer)
                if due:
                    self.flush()

        self._flusher = threading.Thread(target=_loop, daemon=True, name="dashboard-phase-flush")
        self._flusher.start()


class TrajectorySink:
    # [CLASS DOC] Collects trajectory events (attempt start/end, generation start/end, tool start/end)
    # per sample into a lock-protected buffer with the same batching rule as the phase sink, and sends
    # batches to the collector asynchronously. Public emitters are called by core code through the
    # trajectory lifecycle seam. KNOWN ISSUE: the docstring names a dashboard lifecycle module
    # accessor, but the code registers the sink on the shared utils lifecycle object instead.
    """Per-sample lifecycle events from the rollout process, batched to the
    collector (fire-and-forget). Core code reaches it via
    ``miles.dashboard.lifecycle.trajectory_sink()`` — duck-typed so the core
    call sites never import this module."""

    def __init__(self, handle):
        # [FUNCTION DOC] Store the handle; empty buffer; last flush time now; fresh warner.
        self._handle = handle
        self._buffer: list[TrajectoryEvent] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._warner = RateLimitedWarner(logger)

    # ------------------------- core-facing emitters -------------------------

    def attempt_start(self, sample) -> None:
        # [FUNCTION DOC] Emit an attempt-start event for the sample (timestamp now, turn -1).
        self._emit(TrajectoryEventKind.ATTEMPT_START, sample)

    def gen_start(self, sample) -> None:
        # [FUNCTION DOC] Emit a generation-start event for the sample (timestamp now, turn -1).
        self._emit(TrajectoryEventKind.GEN_START, sample)

    def attempt_end(self, result) -> None:
        # [FUNCTION DOC] Input is a sample or a list of per-turn samples. For each sample, reads the
        # lifecycle metadata (a mapping, or a list of mappings for merged samples; ignored when
        # absent). For each segment: if it has a previous-end time and an arrival time (request
        # timestamp, else its start time), emits a tool span from the previous end to that time for
        # the segment's turn labelled "agent gap"; if it has a start time, emits a generation-start at
        # that time; always emits a generation-end at the segment's end time. After all samples,
        # emits an attempt-end event for the last sample whose detail is the status value text (empty
        # if no status). KNOWN ISSUE: unlike the emit helper, this method has no exception guard, so a
        # segment missing the turn or end-time keys would raise into the caller despite the module
        # promise that sinks never raise.
        """``result`` is the Sample or the list of per-turn Samples a generate
        function returned; per-turn samples share the identity and carry their
        own lifecycle metadata segments (agentic endpoint path)."""
        samples = result if isinstance(result, list) else [result]
        for sample in samples:
            value = (sample.metadata or {}).get("lifecycle")
            # one dict per turn-sample; a TITO-merged sample carries the list
            for segment in value if isinstance(value, list) else [value] if value else []:
                gap_end = segment.get("req_ts") or segment.get("t0")
                if segment.get("prev_t1") is not None and gap_end is not None:
                    # the gap between chat calls is agent-side work (tool
                    # execution, environment steps) — design §18.3. req_ts
                    # (server-edge arrival) bounds it exactly; without it the
                    # gen start approximates and absorbs engine queueing
                    self.tool_span(sample, segment["prev_t1"], gap_end, turn=segment["turn"], detail="agent gap")
                if segment.get("t0") is not None:
                    self._emit(TrajectoryEventKind.GEN_START, sample, ts=segment["t0"], turn=segment["turn"])
                self._emit(TrajectoryEventKind.GEN_END, sample, ts=segment["t1"], turn=segment["turn"])
        last = samples[-1]
        status = last.status.value if getattr(last, "status", None) is not None else ""
        self._emit(TrajectoryEventKind.ATTEMPT_END, last, detail=status)

    def gen_span(self, sample, t0: float, t1: float, turn: int, detail: str = "") -> None:
        # [FUNCTION DOC] Emit a generation-start at t0 and a generation-end at t1 (the detail goes on
        # the end event only) for the given turn.
        self._emit(TrajectoryEventKind.GEN_START, sample, ts=t0, turn=turn)
        self._emit(TrajectoryEventKind.GEN_END, sample, ts=t1, turn=turn, detail=detail)

    def tool_span(self, sample, t0: float, t1: float, turn: int, detail: str = "") -> None:
        # [FUNCTION DOC] Emit a tool-start at t0 and a tool-end at t1, both carrying the detail text,
        # for the given turn.
        self._emit(TrajectoryEventKind.TOOL_START, sample, ts=t0, turn=turn, detail=detail)
        self._emit(TrajectoryEventKind.TOOL_END, sample, ts=t1, turn=turn, detail=detail)

    # ------------------------------ internals -------------------------------

    def _emit(self, kind: str, sample, *, ts: float | None = None, turn: int = -1, detail: str = "") -> None:
        # [FUNCTION DOC] HELPER: Builds one trajectory event: timestamp (default now), kind, sample
        # index and group index (-1 when unset), turn (default -1), weight version = the last version
        # among the sample's weight version spans as text (empty if none), and detail. Appends it under
        # the lock, takes a batch if due, and sends the batch asynchronously. Any exception is
        # swallowed with a rate-limited warning.
        try:
            versions = [span.version for span in getattr(sample, "all_weight_version_spans", None) or []]
            event = TrajectoryEvent(
                ts=time.time() if ts is None else ts,
                kind=kind,
                sample_index=sample.index if sample.index is not None else -1,
                group_index=sample.group_index if sample.group_index is not None else -1,
                turn=turn,
                weight_version=str(versions[-1]) if versions else "",
                detail=detail,
            )
            with self._lock:
                self._buffer.append(event)
                batch = self._take_batch_if_due()
            if batch:
                self._handle.push_trajectories.remote(batch)
        except Exception:
            self._warner.warn("dashboard trajectory sink failed; dropping events")

    def _take_batch_if_due(self) -> list[TrajectoryEvent] | None:
        # [FUNCTION DOC] HELPER: Same rule as the phase sink version: with the lock held, returns none
        # unless the buffer has 64 events or 2 seconds have elapsed since the last take; otherwise
        # empties and returns the buffer and resets the timer. KNOWN ISSUE: duplicated code with the
        # phase sink; there is no background flusher here, so events in an idle process stay buffered
        # until the next emit or an explicit flush.
        if len(self._buffer) < BATCH_MAX_EVENTS and time.monotonic() - self._last_flush < BATCH_MAX_SECONDS:
            return None
        batch, self._buffer = self._buffer, []
        self._last_flush = time.monotonic()
        return batch

    def flush(self) -> None:
        # [FUNCTION DOC] Takes the whole buffer under the lock and sends it to the collector if
        # non-empty; swallows exceptions with a rate-limited warning.
        try:
            with self._lock:
                batch, self._buffer = self._buffer, []
            if batch:
                self._handle.push_trajectories.remote(batch)
        except Exception:
            self._warner.warn("dashboard trajectory sink flush failed; dropping events")


_phase_sink: PhaseSink | None = None
_engines_fingerprint: tuple | None = None
_warner = RateLimitedWarner(logger)


def attach_phase_sink(handle, role: str) -> None:
    # [FUNCTION DOC] Creates the process's single phase sink for the handle and role and appends it to
    # the global Timer's event sinks; does nothing if one is already attached (the new handle and
    # role are then ignored).
    global _phase_sink
    if _phase_sink is not None:
        return  # one sink per process
    _phase_sink = PhaseSink(handle, role)
    Timer().event_sinks.append(_phase_sink)


def attach_trajectory_sink(handle) -> None:
    # [FUNCTION DOC] Sets a new trajectory sink on the shared trajectory lifecycle object unless one
    # is already set.
    seam = TrajectoryLifecycle()
    if seam.sink is not None:
        return  # one sink per process
    seam.sink = TrajectorySink(handle)


def detach_and_flush() -> None:
    # [FUNCTION DOC] If a phase sink exists: removes it from the Timer sinks (if present) and flushes
    # it; clears the module's sink and engine fingerprint. If the lifecycle object has a trajectory
    # sink, flushes it, then clears it. The background flusher thread of the phase sink keeps running.
    global _phase_sink, _engines_fingerprint
    if _phase_sink is not None:
        if _phase_sink in Timer().event_sinks:
            Timer().event_sinks.remove(_phase_sink)
        _phase_sink.flush()
    _phase_sink = None
    _engines_fingerprint = None
    seam = TrajectoryLifecycle()
    if seam.sink is not None:
        seam.sink.flush()
    seam.sink = None


def register_train_actor(args) -> None:
    # [FUNCTION DOC] No-op if the dashboard flag is off. Otherwise resolves the collector handle
    # (may wait up to 60 seconds; returns if unavailable) and attaches a phase sink with the training
    # role.
    """Called by TrainActor.init on EVERY rank — init_tracking only runs on
    the megatron main rank, but per-GPU phase lanes need every rank's Timer."""
    if not args.use_miles_dashboard:
        return
    from miles.dashboard import backend

    handle = backend.resolve_collector()
    if handle is None:
        return
    attach_phase_sink(handle, Role.TRAIN)


def register_router(args) -> None:
    # [FUNCTION DOC] No-op if the dashboard flag is off or the collector cannot be resolved.
    # Asserts the router IP in args is set (fail loudly: it indicates wrong call order). Then sends
    # the collector the router URL "http://<ip>:<port>" asynchronously (exceptions swallowed with a
    # rate-limited warning that engine metrics will be missing).
    """Called by the rollout manager AFTER start_rollout_servers: only then are
    ``args.sglang_router_ip/port`` filled in. init_tracking runs earlier in
    __init__, so the backend cannot register the router at init time."""
    if not args.use_miles_dashboard:
        return
    from miles.dashboard import backend

    handle = backend.resolve_collector()
    if handle is None:
        return
    # a None ip here is a wiring-order bug, not runtime flakiness: fail loud
    assert args.sglang_router_ip is not None, "register_router must run after start_rollout_servers"
    try:
        handle.set_router.remote(f"http://{args.sglang_router_ip}:{args.sglang_router_port}")
    except Exception:
        _warner.warn("dashboard router registration failed; engine metrics will be missing")


async def register_engines(servers, *, provider: BaseWorkerProvider) -> None:
    # [FUNCTION DOC] Uses the already-resolved collector handle only (returns if none). Collects the
    # alive engine cells and their worker infos from the provider, and computes a fingerprint of
    # (worker name, generation, primary host, primary port) over all workers. If it equals the last
    # sent one, returns. Else builds engine infos (probing GPU UUIDs), sends a topology snapshot
    # (timestamp now) to the collector asynchronously and stores the fingerprint (only after success).
    # Exceptions are swallowed with a rate-limited warning.
    """Called at the top of every InferenceController.prepare_rollout(): pushes an engine
    topology snapshot whenever the set of engine actors changed (startup,
    fault-tolerance recovery). Steady state costs one worker-manager round trip
    and a local tuple compare."""
    global _engines_fingerprint
    from miles.dashboard import backend

    handle = backend.current_collector()
    if handle is None:
        return
    try:
        cells = _alive_engine_cells(servers)
        worker_infos_per_cell = _collect_worker_infos(cells, provider=provider)
        fingerprint = tuple(
            (info.name, info.generation, info.self_addrs["primary"].host, info.self_addrs["primary"].port)
            for worker_infos in worker_infos_per_cell
            for info in worker_infos
        )
        if fingerprint == _engines_fingerprint:
            return
        engines = await _compute_engine_infos(cells, worker_infos_per_cell, provider=provider)
        handle.update_topology.remote(TopologySnapshot(ts=time.time(), engines=engines))
        _engines_fingerprint = fingerprint
    except Exception:
        _warner.warn("dashboard engine registration failed; topology may be stale")


def report_data_buffer(length: int | None) -> None:
    # [FUNCTION DOC] Returns if the length is none or no collector handle is resolved. Otherwise sends
    # a data-buffer sample (timestamp now, length) asynchronously; exceptions are swallowed with a
    # rate-limited warning.
    """Called at the top of every ``RolloutExecutor.get()`` alongside
    ``register_engines``, with ``getattr(data_source, "get_buffer_length",
    lambda: None)()``. A no-op for ``length is None`` — most data sources
    (plain ``RolloutDataSource``) never buffer samples across steps."""
    if length is None:
        return
    from miles.dashboard import backend

    handle = backend.current_collector()
    if handle is None:
        return
    try:
        handle.push_data_buffer.remote(DataBufferSample(ts=time.time(), length=length))
    except Exception:
        _warner.warn("dashboard data-buffer report failed")


def _alive_engine_cells(servers) -> list:
    # [FUNCTION DOC] HELPER: Walks the mapping of servers, then each server's mapping of cells, and
    # returns the cells whose "pending weights or serving" property is true (in iteration order).
    """A multi-node engine's cell holds ``num_nodes`` actors; only the first
    (master) owns the router-visible URL. Cells that are not alive are skipped
    until recovery completes."""
    cells = []
    for server in servers.values():
        for cell in server.server_cells.values():
            if cell.is_pending_weights_or_serving:
                cells.append(cell)
    return cells


def _collect_worker_infos(cells, *, provider: BaseWorkerProvider) -> list[list]:
    # [FUNCTION DOC] HELPER: Asks the provider for the worker infos of the cells' ids; returns one
    # list per cell, aligned with the input order.
    return provider.get_worker_infos(cell_ids=[cell.meta.cell_id for cell in cells])


async def _no_gpu_uuids(info) -> list[str | None]:
    # [FUNCTION DOC] HELPER: Coroutine returning a list of none, one per GPU id of the worker info
    # (used when no handle exists to probe).
    return [None] * len(info.gpu_ids)


async def _compute_engine_infos(cells, worker_infos_per_cell, *, provider: BaseWorkerProvider) -> list[EngineInfo]:
    # [FUNCTION DOC] HELPER: Flattens worker infos, fetches actor handles for them from the provider,
    # and concurrently probes each worker's GPU UUIDs through its handle (or none-lists when no handle
    # exists). Then, per cell in order (strict pairing of cells and worker info lists; a length
    # mismatch raises a value error), builds one engine info: address = the cell's server URL, worker
    # type from the cell metadata, engine rank = the cell's position, GPU list of [primary host with
    # square brackets stripped, GPU id] for every worker GPU, and the flattened UUIDs in the same
    # order.
    flat_infos = [info for worker_infos in worker_infos_per_cell for info in worker_infos]
    handles = provider.get_handles_of_worker_infos(flat_infos)
    probed_uuids = iter(
        await asyncio.gather(
            *[
                (
                    handle._get_gpu_uuids(gpu_ids=info.gpu_ids)
                    if (handle := handles.get(info.name)) is not None
                    else _no_gpu_uuids(info)
                )
                for info in flat_infos
            ]
        )
    )

    engines = []
    for engine_rank, (cell, worker_infos) in enumerate(zip(cells, worker_infos_per_cell, strict=True)):
        worker_gpu_uuids = [next(probed_uuids) for _ in worker_infos]
        engines.append(
            EngineInfo(
                addr=cell.server_url,
                worker_type=cell.meta.worker_type,
                engine_rank=engine_rank,
                gpus=[
                    [info.self_addrs["primary"].host.strip("[]"), gpu_id]
                    for info in worker_infos
                    for gpu_id in info.gpu_ids
                ],
                gpu_uuids=[uuid for uuids in worker_gpu_uuids for uuid in uuids],
            )
        )
    return engines
