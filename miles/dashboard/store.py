# [MODULE DOC] Storage and query layer for dashboard telemetry, backed by append-only JSONL files.
# Defines (1) the record types, one dataclass per stream, with the Stream enum naming each stream
# and its file; (2) a partition reader for hour-partitioned streams; (3) MetricStore, with a write
# side (buffer, flush, meta, lane catalog) used by the live collector and a read side (load,
# follow, timeline/metric/engine queries) used by the server and advisory code; and (4) two
# downsampling helpers.
# On disk, under the store directory: meta.json, one {stream}.jsonl per low-rate stream (metrics,
# topology, data_buffer), a {stream}/{YYYYMMDD_HH}.jsonl partition directory per high-rate stream
# (gpu_util, engine_series, phases, trajectories, gpu_processes; hour in UTC), and
# lane_catalog.json. Timestamps are epoch seconds (floats).
# Concurrency: queries run concurrently from a thread pool; the partition reader and the engine
# frame cache use locks, while the plain record lists rely on append-only list semantics.
"""File-backed store for miles dashboard time series.

The live collector buffers records via ``append()`` and persists them with
``flush()`` as append-only JSONL streams under ``{dump_details}/dashboard/``.
The dashboard server reads the same directory with ``MetricStore.load()`` and
picks up appended lines incrementally with ``follow()`` — a plain byte-offset
tail, correct because every stream is append-only.

Each stream holds one record type (one JSON object per line); see the
``Record`` subclasses for the schemas.

The two high-rate streams (``gpu_util``, ``engine_series``) are held in memory
as columnar polars frames instead of dataclass lists (design doc §17):
~16 B/row instead of ~600 B, vectorized parsing and numpy queries.

High-rate streams (those two plus ``phases``) write hourly partition files
``{stream}/{YYYYMMDD_HH}.jsonl`` (phases keyed by END hour) and parse lazily:
open reads nothing from them; a windowed query parses only the hours it
touches, through an LRU block cache with per-file byte-offset tail refresh.
A v1 flat ``{stream}.jsonl`` still reads as one "legacy" partition. Lane
metadata for selection resolution is folded into ``lane_catalog.json`` at
flush time so it never needs a stream scan.
"""

from __future__ import annotations

import calendar
import io
import json
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

import numpy as np
import polars as pl


class Stream(StrEnum):
    # [CLASS DOC] String enum naming the eight telemetry streams: metrics, phases, topology,
    # gpu_util, engine_series, trajectories, gpu_processes and data_buffer. The value is the
    # stream's file stem and partition directory name.
    METRICS = "metrics"
    PHASES = "phases"
    TOPOLOGY = "topology"
    GPU_UTIL = "gpu_util"
    ENGINE_SERIES = "engine_series"
    TRAJECTORIES = "trajectories"
    GPU_PROCESSES = "gpu_processes"
    DATA_BUFFER = "data_buffer"

    @property
    def filename(self) -> str:
        # [FUNCTION DOC] Property: the flat file name of the stream, its value plus the
        # ".jsonl" suffix.
        return f"{self.value}.jsonl"


@dataclass
class Record:
    # [CLASS DOC] Base dataclass of all stream records. Each subclass sets the class-level stream it
    # belongs to; one record serializes to one JSON object per line. Subclasses (other than the
    # phase event) carry a timestamp field named ts, which the default timestamps method uses.
    stream: ClassVar[Stream]

    def to_dict(self) -> dict:
        # [FUNCTION DOC] Returns the record as a plain nested mapping of its dataclass fields
        # (class-level stream marker excluded); this is the JSON line payload.
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Record:
        # [FUNCTION DOC] Builds a record from a mapping by passing every key as a constructor
        # argument. Raises a type error for unknown or missing keys (callers treat that as
        # corruption). Subclasses with nested records override it.
        return cls(**data)

    def timestamps(self) -> tuple[float, ...]:
        # [FUNCTION DOC] Returns the timestamps this record contributes to the global time range:
        # by default just its single ts field.
        return (self.ts,)


@dataclass
class MetricsRecord(Record):
    # [CLASS DOC] Metrics stream record: ts, the name of the step axis the values were logged
    # against (step_key, may be None), the integer step (may be None) and a mapping from metric
    # key to value.
    """One ``tracking_utils.log()`` payload."""

    stream: ClassVar[Stream] = Stream.METRICS
    ts: float
    step_key: str | None
    step: int | None
    metrics: dict[str, Any]


class Role(StrEnum):
    # [CLASS DOC] String enum of process roles for phase events: train, rollout_manager, and
    # derived (synthesized on the read side, e.g. the initialize band, never written).
    TRAIN = "train"
    ROLLOUT_MANAGER = "rollout_manager"
    DERIVED = "derived"  # synthesized by the read side, not observed


# phase name synthesized per lane for [meta.start_ts, first observed event)
INITIALIZE_PHASE = "initialize"


@dataclass
class PhaseEvent(Record):
    # [CLASS DOC] Phases stream record: phase name, start t0 and end t1 (epoch seconds), node,
    # list of GPU ids on that node, process rank, and role (a Role value as a string). An event
    # with t1 equal to the open sentinel -1 marks a still-running interval. Has no ts field; it
    # overrides the timestamps method.
    """One completed timer interval on one process (Timer sink event).

    ``rollout_manager`` events carry no GPUs (the manager is a driver-side
    process); the timeline queries expand them onto rollout-engine lanes."""

    stream: ClassVar[Stream] = Stream.PHASES
    name: str
    t0: float
    t1: float
    node: str
    gpus: list[int]
    rank: int
    role: str  # a Role value

    # t1 sentinel: the interval was still running when this event was written
    # (Timer.start emits it so long phases are visible before they end); the
    # closing event with the real t1 supersedes it on the read side
    OPEN_T1: ClassVar[float] = -1.0

    @property
    def open(self) -> bool:
        # [FUNCTION DOC] Property: true when the end time equals the open sentinel (-1).
        return self.t1 == self.OPEN_T1

    def timestamps(self) -> tuple[float, ...]:
        # [FUNCTION DOC] Returns only the start for an open event, else both start and end.
        return (self.t0,) if self.open else (self.t0, self.t1)


@dataclass
class EngineInfo:
    # [CLASS DOC] Not a stream record: description of one rollout engine inside a topology
    # snapshot. Fields: address, worker type, engine rank, the list of (node ip, gpu id) pairs it
    # occupies, and a parallel list of GPU UUIDs (entries may be None).
    addr: str
    worker_type: str
    engine_rank: int
    gpus: list[list]  # [node_ip, gpu_id] pairs
    gpu_uuids: list[str | None]


@dataclass
class TopologySnapshot(Record):
    # [CLASS DOC] Topology stream record: timestamp plus the full list of engines at that moment.
    # It replaces the previous snapshot, so each is valid until the next one.
    """Full engine topology; snapshot N is valid until snapshot N+1."""

    stream: ClassVar[Stream] = Stream.TOPOLOGY
    ts: float
    engines: list[EngineInfo]

    @classmethod
    def from_dict(cls, data: dict) -> TopologySnapshot:
        # [FUNCTION DOC] Builds a snapshot from a mapping with ts and engines, converting each
        # engine mapping into an engine info. Raises a key error or type error on bad shape.
        return cls(ts=data["ts"], engines=[EngineInfo(**e) for e in data["engines"]])


@dataclass
class DataBufferSample(Record):
    # [CLASS DOC] Data buffer stream record: timestamp and the integer number of samples held in
    # the rollout data source's buffer.
    """One report of ``RolloutDataSourceWithBuffer.get_buffer_length()``
    (design doc's "show the data status in databuffer" ask). Appended once
    per ``RolloutExecutor.get()`` call — a no-op for plain
    ``RolloutDataSource`` runs, which never buffer samples across steps.
    Low-rate and unpartitioned like ``TopologySnapshot``: only the latest
    value matters for display."""

    stream: ClassVar[Stream] = Stream.DATA_BUFFER
    ts: float
    length: int


class TrajectoryEventKind(StrEnum):
    # [CLASS DOC] String enum of sample lifecycle event kinds: attempt start/end, generation
    # start/end and tool-call start/end.
    ATTEMPT_START = "attempt_start"
    GEN_START = "gen_start"
    GEN_END = "gen_end"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    ATTEMPT_END = "attempt_end"


@dataclass
class TrajectoryEvent(Record):
    # [CLASS DOC] Trajectories stream record: ts, kind (a TrajectoryEventKind value as string),
    # sample index, group index (-1 if none), turn number, weight version string and a free-text
    # detail (for attempt end, the final status).
    """One lifecycle boundary of one sample's rollout (design §18.3).

    ``sample_index`` is the run-global ``Sample.index`` — the same key the
    dump join uses, so the read side can resolve a consuming step's batch to
    its events. ``turn`` is 1-based for gen/tool segments and -1 for attempt
    boundaries; ``weight_version`` is the engine-reported version of the
    segment ("" when unknown)."""

    stream: ClassVar[Stream] = Stream.TRAJECTORIES
    ts: float
    kind: str  # a TrajectoryEventKind value
    sample_index: int
    group_index: int  # -1 when the sample carries none
    turn: int
    weight_version: str
    detail: str


@dataclass
class GpuSample(Record):
    # [CLASS DOC] GPU utilization stream record: ts, node, gpu index, utilization percent, memory
    # used in MB and power in watts (all integers except ts and node).
    """One NVML sample of one GPU."""

    stream: ClassVar[Stream] = Stream.GPU_UTIL
    ts: float
    node: str
    gpu: int
    util: int
    mem_mb: int
    power_w: int


@dataclass
class EngineSample(Record):
    # [CLASS DOC] Engine series stream record: ts, engine address, normalized metric name, a
    # mapping of kept labels (strings) and the float value.
    """One scraped sglang engine metric value."""

    stream: ClassVar[Stream] = Stream.ENGINE_SERIES
    ts: float
    addr: str
    metric: str
    labels: dict[str, str]
    value: float


@dataclass
class GpuProcessSample(Record):
    # [CLASS DOC] GPU process stream record: ts, node, gpu index, process id, process name and
    # memory in MB used by that process on that GPU.
    """One NVML per-process memory reading on one GPU, at a coarser cadence
    than GpuSample (design doc's GPU/CPU/disk memory-breakdown TODO — this
    covers the GPU side: who is actually holding the VRAM, not just the
    per-GPU aggregate)."""

    stream: ClassVar[Stream] = Stream.GPU_PROCESSES
    ts: float
    node: str
    gpu: int
    pid: int
    name: str
    mem_mb: int


# Module-level mapping from each stream to its record class, built from every record class's
# stream marker.
_RECORD_TYPE_OF_STREAM: dict[Stream, type[Record]] = {
    cls.stream: cls
    for cls in (
        MetricsRecord,
        PhaseEvent,
        TopologySnapshot,
        GpuSample,
        EngineSample,
        TrajectoryEvent,
        GpuProcessSample,
        DataBufferSample,
    )
}


@dataclass
class Meta:
    # [CLASS DOC] Run metadata persisted as meta.json: run name, collector start time (epoch
    # seconds), a mapping of the training arguments snapshot, and a schema version (default 1,
    # not checked anywhere on load). FILENAME is the file name inside the store directory.
    run_name: str
    start_ts: float
    args: dict[str, Any]
    schema_version: int = 1

    FILENAME: ClassVar[str] = "meta.json"


def _hour_key(ts: float) -> str:
    # [FUNCTION DOC] HELPER: Formats an epoch timestamp as the UTC hour partition stem
    # year-month-day underscore hour (eight digits, underscore, two digits).
    return time.strftime("%Y%m%d_%H", time.gmtime(ts))


def _hour_bounds(key: str) -> tuple[float, float]:
    # [FUNCTION DOC] HELPER: Inverse of the hour key: parses it as a UTC hour and returns the
    # epoch start of that hour and start plus 3600 seconds. Raises a value error on a malformed key.
    start = calendar.timegm(time.strptime(key, "%Y%m%d_%H"))
    return float(start), float(start + 3600)


class _PartitionReader:
    # [CLASS DOC] HELPER: Lazy reader over one stream's partition directory plus an optional legacy
    # flat file. Parsing is delegated to injected callables: parse (raw bytes of complete lines,
    # path, byte offset -> block), concat (list of blocks -> one block; empty list gives an empty
    # block), length (block -> row count) and line_stamps (decoded line mapping -> tuple of
    # timestamps). Parsed blocks are cached per partition key in an LRU of max_blocks entries
    # (evicting also forgets the byte offset, so a later read reparses the whole file). A block is
    # refreshed by reading from its stored byte offset and consuming only complete lines. A lock
    # serializes block fills because concurrent fills from one stale offset would duplicate rows.
    """One hourly-partitioned JSONL stream on the read side.

    Files parse lazily at first windowed read into an LRU block cache; a
    cached block picks up appended bytes through a per-file byte offset with
    the same complete-line rule as ``MetricStore.follow``. A v1 flat
    ``{stream}.jsonl`` participates as one "legacy" partition spanning all
    time, so every pre-partition dump dir keeps working.
    """

    LEGACY_KEY: ClassVar[str] = "legacy"

    def __init__(
        self,
        dir: Path,
        legacy_path: Path,
        *,
        parse: Callable[[bytes, Path, int], Any],
        concat: Callable[[list], Any],
        length: Callable[[Any], int],
        line_stamps: Callable[[dict], tuple],
        max_blocks: int,
    ):
        # [FUNCTION DOC] HELPER: Stores the partition directory, the legacy flat file path and the
        # injected callables, and creates an empty ordered block cache, an offset table and a lock.
        # No file is touched.
        self.dir = dir
        self.legacy_path = legacy_path
        self._parse = parse
        self._concat = concat
        self._length = length
        self._line_stamps = line_stamps
        self._max_blocks = max_blocks
        self._blocks: OrderedDict[str, Any] = OrderedDict()
        self._offsets: dict[str, int] = {}
        self._lock = threading.Lock()

    def files(self) -> list[tuple[str, Path]]:
        # [FUNCTION DOC] HELPER: Lists (key, path) pairs: the legacy file first under the key
        # "legacy" if it exists, then every jsonl file in the partition directory (if it is a
        # directory) keyed by file stem, sorted by stem (chronological). Scans the disk each call.
        out = [(self.LEGACY_KEY, self.legacy_path)] if self.legacy_path.exists() else []
        if self.dir.is_dir():
            out.extend(sorted((path.stem, path) for path in self.dir.glob("*.jsonl")))
        return out

    def has_data(self) -> bool:
        # [FUNCTION DOC] HELPER: True if any partition or legacy file has non-zero size.
        return any(path.stat().st_size > 0 for _, path in self.files())

    def window(self, t0: float | None, t1: float | None) -> Any:
        # [FUNCTION DOC] HELPER: Returns one concatenated block of all non-empty blocks of the
        # partitions that may intersect the window. A dated partition is skipped when its hour
        # ends at or before t0 or starts after t1; the legacy partition is always included. Blocks
        # are loaded or tail-refreshed as needed. The result is NOT trimmed to the exact window;
        # callers filter rows themselves.
        """Concatenated block over the partitions intersecting [t0, t1]."""
        blocks = []
        for key, path in self.files():
            if key != self.LEGACY_KEY:
                lo, hi = _hour_bounds(key)
                if (t0 is not None and hi <= t0) or (t1 is not None and lo > t1):
                    continue
            blocks.append(self._block(key, path))
        return self._concat([block for block in blocks if self._length(block)])

    def refresh_cached(self) -> int:
        # [FUNCTION DOC] HELPER: For each currently cached key whose file still exists, tail-reads
        # appended complete lines into the block. Returns the total rows added. Partitions never
        # read are skipped, and cached keys whose file vanished are left untouched.
        """Tail-refresh every cached block; returns rows appended. Partitions
        never read remain lazy — they parse fully at their first read."""
        paths = dict(self.files())
        added = 0
        for key in list(self._blocks):
            if key in paths:
                before = self._length(self._blocks[key])
                self._block(key, paths[key])
                added += self._length(self._blocks[key]) - before
        return added

    def _block(self, key: str, path: Path) -> Any:
        # [FUNCTION DOC] HELPER: Returns the up-to-date block for one partition, under the lock.
        # If cached and the file is not larger than the stored offset, only marks it most recently
        # used. Otherwise reads bytes from the stored offset (0 if none) to the end, keeps only
        # up to the last newline, parses those complete lines with the injected parser (errors
        # propagate) and concatenates them after the cached block, advancing the offset. If there
        # is no complete line and the key is uncached, an empty block is cached. Finally the key
        # becomes most recently used and the least recently used blocks beyond the cap are evicted
        # together with their offsets.
        # read+parse release the GIL; two fills from one stale offset append the bytes twice
        with self._lock:
            offset = self._offsets.get(key, 0)
            if key in self._blocks and path.stat().st_size <= offset:
                self._blocks.move_to_end(key)
                return self._blocks[key]
            with open(path, "rb") as f:
                f.seek(offset)
                chunk = f.read()
            end = chunk.rfind(b"\n")
            if end >= 0:
                new = self._parse(chunk[: end + 1], path, offset)
                pieces = [self._blocks[key], new] if key in self._blocks else [new]
                self._blocks[key] = self._concat([piece for piece in pieces if self._length(piece)])
                self._offsets[key] = offset + end + 1
            elif key not in self._blocks:
                self._blocks[key] = self._concat([])
                self._offsets[key] = offset
            self._blocks.move_to_end(key)
            while len(self._blocks) > self._max_blocks:
                evicted, _ = self._blocks.popitem(last=False)
                self._offsets.pop(evicted, None)
            return self._blocks[key]

    def edge_stamps(self) -> list[float]:
        # [FUNCTION DOC] HELPER: Cheaply finds timestamps bounding the stream without a full scan.
        # Takes the non-empty files, reads the first line of the first (oldest) file and the last
        # complete line within the final 64 KiB of the last file, decodes each as JSON and applies
        # the injected line-stamp function. A line that fails to decode (partial write) is ignored.
        # Returns the stamps (empty list if no data). Assumes files are ordered chronologically and
        # lines roughly time-ordered.
        """Global-range candidates from the first line of the oldest file and
        the last complete line of the newest — no full scan."""
        files = [(key, path) for key, path in self.files() if path.stat().st_size > 0]
        if not files:
            return []
        lines = []
        with open(files[0][1], "rb") as f:
            lines.append(f.readline())
        with open(files[-1][1], "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read()
        complete = tail[: tail.rfind(b"\n") + 1]
        if complete:
            lines.append(complete.splitlines()[-1])
        stamps: list[float] = []
        for line in lines:
            try:
                stamps.extend(self._line_stamps(json.loads(line)))
            except json.JSONDecodeError:
                pass  # partial line mid-append: no complete record to stamp yet
        return stamps


class MetricStore:
    # [CLASS DOC] The telemetry store, used for writing (collector) and reading (server).
    # Class constants: COLUMNAR_STREAMS (gpu_util, engine_series) are held as polars frames with
    # the column schemas in FRAME_SCHEMAS (engine labels become a JSON string column);
    # PARTITIONED_STREAMS (those two plus phases, trajectories, gpu_processes) are stored as hourly
    # files and parsed lazily with at most PARTITION_CACHE_BLOCKS (24) cached hours; MAX_WINDOW_S is
    # the largest allowed query window (4 hours); CATALOG_FILENAME names the lane catalog file.
    # Instance state: meta (or None); records, a per-stream list for the non-partitioned streams
    # (metrics, topology, data_buffer) filled by load/follow; write buffers per stream; byte
    # offsets for tailing; one partition reader per partitioned stream; the in-memory lane catalog
    # accumulator of the write side; and a lock-guarded cache of the full engine frame with
    # per-metric partitions, invalidated when partition file sizes change.
    # Write side and read side are meant for different processes (or the same, in tests).
    COLUMNAR_STREAMS: ClassVar[tuple[Stream, ...]] = (Stream.GPU_UTIL, Stream.ENGINE_SERIES)
    PARTITIONED_STREAMS: ClassVar[tuple[Stream, ...]] = (
        Stream.GPU_UTIL,
        Stream.ENGINE_SERIES,
        Stream.PHASES,
        Stream.TRAJECTORIES,
        Stream.GPU_PROCESSES,
    )
    MAX_WINDOW_S: ClassVar[float] = 4 * 3600.0
    PARTITION_CACHE_BLOCKS: ClassVar[int] = 24
    CATALOG_FILENAME: ClassVar[str] = "lane_catalog.json"
    FRAME_SCHEMAS: ClassVar[dict[Stream, dict[str, Any]]] = {
        Stream.GPU_UTIL: dict(
            ts=pl.Float64, node=pl.String, gpu=pl.Int64, util=pl.Int64, mem_mb=pl.Int64, power_w=pl.Int64
        ),
        # labels dicts are canonicalized to a JSON string column at parse time
        Stream.ENGINE_SERIES: dict(
            ts=pl.Float64, addr=pl.String, metric=pl.String, labels_json=pl.String, value=pl.Float64
        ),
    }

    def __init__(self, dir: Path | str):
        # [FUNCTION DOC] Initializes empty state for the store rooted at dir (no I/O; the directory
        # need not exist yet). Creates record lists and byte offsets only for the non-partitioned
        # streams, buffers for all streams, one partition reader per partitioned stream, and the
        # engine cache fields (version, full frame, per-metric partition cache) plus their lock.
        self.dir = Path(dir)
        self.meta: Meta | None = None
        self.records: dict[Stream, list[Record]] = {s: [] for s in Stream if s not in self.PARTITIONED_STREAMS}
        self._buffers: dict[Stream, list[Record]] = {s: [] for s in Stream}
        self._offsets: dict[Stream, int] = {s: 0 for s in Stream if s not in self.PARTITIONED_STREAMS}
        self._readers: dict[Stream, _PartitionReader] = {s: self._make_reader(s) for s in self.PARTITIONED_STREAMS}
        self._catalog: dict[str, dict] | None = None  # write-side lane catalog accumulator
        # full-window engine query cache, versioned by partition file sizes so
        # appended scrapes invalidate it; the lock is load-bearing because sync
        # endpoints serve concurrently from FastAPI's threadpool
        self._engine_cache_lock = threading.Lock()
        self._engine_cache_version: tuple | None = None
        self._engine_frame_cache: pl.DataFrame | None = None
        self._engine_parts_cache: dict[tuple[str, bool], dict[tuple[str, str], pl.DataFrame]] = {}

    def _make_reader(self, stream: Stream) -> _PartitionReader:
        # [FUNCTION DOC] HELPER: Builds the partition reader for one stream. Columnar streams use
        # the columnar parser, a concatenation that returns the empty typed frame for no blocks, a
        # single block unchanged, else a non-rechunked concat (rechunked if over 256 chunks), the
        # frame height as length, and the ts field as the line stamp. Object streams (phases,
        # trajectories, gpu_processes; asserted) use the object-line parser, list concatenation,
        # list length, and the record's own timestamps (so open phase intervals skip the sentinel)
        # as line stamps. Both use the partition cache size constant.
        if stream in self.COLUMNAR_STREAMS:
            empty = pl.DataFrame(schema=self.FRAME_SCHEMAS[stream])

            def concat_frames(blocks: list, empty: pl.DataFrame = empty) -> pl.DataFrame:
                # [FUNCTION DOC] HELPER: Merges frame blocks: empty typed frame for none, the
                # block itself for one, otherwise a concat without rechunking; the result is
                # rechunked when it has more than 256 chunks to bound fragmentation.
                if not blocks:
                    return empty
                merged = blocks[0] if len(blocks) == 1 else pl.concat(blocks, rechunk=False)
                # bound chunk fragmentation from long tail sessions
                return merged.rechunk() if merged.n_chunks() > 256 else merged

            return _PartitionReader(
                self.dir / stream.value,
                self.dir / stream.filename,
                parse=lambda raw, path, offset, stream=stream: self._parse_columnar(stream, raw, path, offset),
                concat=concat_frames,
                length=lambda frame: frame.height,
                line_stamps=lambda data: (data["ts"],),
                max_blocks=self.PARTITION_CACHE_BLOCKS,
            )
        assert stream in (Stream.PHASES, Stream.TRAJECTORIES, Stream.GPU_PROCESSES), stream
        record_type = _RECORD_TYPE_OF_STREAM[stream]
        return _PartitionReader(
            self.dir / stream.value,
            self.dir / stream.filename,
            parse=lambda raw, path, offset, record_type=record_type: self._parse_object_lines(
                record_type, raw, path, offset
            ),
            concat=lambda blocks: [event for block in blocks for event in block],
            length=len,
            # the record's own timestamps() skips the open-interval t1 sentinel
            line_stamps=lambda data, record_type=record_type: record_type.from_dict(data).timestamps(),
            max_blocks=self.PARTITION_CACHE_BLOCKS,
        )

    # ------------------------------ write side ------------------------------

    def append(self, record: Record) -> None:
        # [FUNCTION DOC] Adds a record to its stream's in-memory write buffer (no I/O until flush).
        self._buffers[record.stream].append(record)

    def buffered_count(self, stream: Stream) -> int:
        # [FUNCTION DOC] Returns how many unflushed records the stream's buffer holds.
        return len(self._buffers[stream])

    def drop_oldest_buffered(self, stream: Stream, *, keep_ratio: float = 0.9) -> int:
        # [FUNCTION DOC] Deletes the oldest records from the stream's buffer, keeping about
        # keep_ratio of them: drops max(1, floor(length times one minus keep_ratio)). Returns that
        # count; mutates the buffer.
        # KNOWN ISSUE: the minimum of one means an empty buffer still reports one dropped record
        # although nothing was removed.
        """Drop the oldest buffered (not yet flushed) records of one stream and
        return how many were dropped. Lets the collector bound memory when the
        disk stays unwritable instead of growing without limit."""
        buffer = self._buffers[stream]
        dropped = max(1, int(len(buffer) * (1 - keep_ratio)))
        del buffer[:dropped]
        return dropped

    def write_meta(self, meta: Meta) -> None:
        # [FUNCTION DOC] Sets the store's meta, creates the directory (with parents) and writes
        # meta.json as indented JSON, overwriting any existing file. Disk I/O.
        self.meta = meta
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / Meta.FILENAME).write_text(json.dumps(asdict(meta), indent=1))

    def flush(self) -> None:
        # [FUNCTION DOC] Persists all buffers. Creates the store directory, merges buffered
        # records into the lane catalog, then for each non-empty buffer: partitioned streams are
        # grouped by UTC hour of their timestamp (phase events use the end time, or the start for
        # open events) and each group is appended to its hour file in the stream's subdirectory
        # (groups written in sorted key order); other streams are appended to their flat file.
        # Records are written as compact JSON, one per line. A buffer is cleared only after its
        # stream was fully written; on an I/O error the failing buffer and later ones stay buffered,
        # and any hour files already appended in the failing stream would be duplicated on retry.
        self.dir.mkdir(parents=True, exist_ok=True)
        self._merge_catalog()
        for stream, buffer in self._buffers.items():
            if not buffer:
                continue
            if stream in self.PARTITIONED_STREAMS:
                subdir = self.dir / stream.value
                subdir.mkdir(exist_ok=True)
                groups: dict[str, list[Record]] = {}
                for record in buffer:
                    # phases key by END hour (the completion append is what
                    # lands on disk, keeping the window lower bound exact);
                    # OPEN markers have no end yet and key by their start
                    if stream is Stream.PHASES:
                        ts = record.t0 if record.open else record.t1
                    else:
                        ts = record.ts
                    groups.setdefault(_hour_key(ts), []).append(record)
                for key, group in sorted(groups.items()):
                    with open(subdir / f"{key}.jsonl", "a") as f:
                        f.write("".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in group))
            else:
                with open(self.dir / stream.filename, "a") as f:
                    f.write("".join(json.dumps(r.to_dict(), separators=(",", ":")) + "\n" for r in buffer))
            buffer.clear()

    def _merge_catalog(self) -> None:
        # [FUNCTION DOC] HELPER: Updates the lane catalog from the buffered records. Loads the
        # existing catalog file on first use (empty if absent). Lanes are keyed by node name, a
        # colon, and the gpu index. Train phase events add the train role and, for non-negative
        # ranks, the rank to each of their GPUs; topology snapshots add the rollout role and the
        # engine address to each engine GPU; GPU utilization samples only ensure the lane exists.
        # If anything changed, the three lists of every lane are sorted and the file is rewritten
        # atomically (temporary file then replace). Not thread-safe; called from flush.
        """Fold buffered records into ``lane_catalog.json``: per-lane train
        ranks, engine addrs and roles, so selection resolution and heatmap
        rows never need a stream scan. Atomic rewrite, only when changed."""
        if self._catalog is None:
            path = self.dir / self.CATALOG_FILENAME
            self._catalog = json.loads(path.read_text())["lanes"] if path.exists() else {}
        lanes = self._catalog
        dirty = False

        def entry(node: str, gpu: int) -> dict:
            # [FUNCTION DOC] HELPER: Returns the catalog entry for a lane, creating one with empty
            # ranks, engine address and role lists (and marking the catalog dirty) if missing.
            nonlocal dirty
            key = f"{node}:{gpu}"
            if key not in lanes:
                lanes[key] = dict(ranks=[], engine_addrs=[], roles=[])
                dirty = True
            return lanes[key]

        def add(values: list, value) -> None:
            # [FUNCTION DOC] HELPER: Appends the value to the list unless already present,
            # marking the catalog dirty when it did.
            nonlocal dirty
            if value not in values:
                values.append(value)
                dirty = True

        for event in self._buffers[Stream.PHASES]:
            if event.role != Role.TRAIN:
                continue
            for gpu in event.gpus:
                lane = entry(event.node, gpu)
                add(lane["roles"], Role.TRAIN.value)
                if event.rank >= 0:
                    add(lane["ranks"], event.rank)
        for snapshot in self._buffers[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    lane = entry(node, gpu)
                    add(lane["roles"], "rollout")
                    add(lane["engine_addrs"], engine.addr)
        for sample in self._buffers[Stream.GPU_UTIL]:
            entry(sample.node, sample.gpu)
        if dirty:
            for lane in lanes.values():
                lane["ranks"].sort()
                lane["engine_addrs"].sort()
                lane["roles"].sort()
            tmp = self.dir / (self.CATALOG_FILENAME + ".tmp")
            tmp.write_text(json.dumps(dict(version=1, lanes=lanes)))
            os.replace(tmp, self.dir / self.CATALOG_FILENAME)

    # ------------------------------ read side -------------------------------

    @classmethod
    def load(cls, dir: Path | str) -> MetricStore:
        # [FUNCTION DOC] Opens an existing store directory for reading: constructs the store,
        # reads meta.json if present, then performs an initial follow to ingest the non-partitioned
        # streams. Returns the store. Raises a value error on a corrupt complete line.
        store = cls(dir)
        meta_path = store.dir / Meta.FILENAME
        if meta_path.exists():
            store.meta = Meta(**json.loads(meta_path.read_text()))
        store.follow()
        return store

    def follow(self) -> int:
        # [FUNCTION DOC] Ingests newly appended lines. For each non-partitioned stream whose file
        # exists: reads from the saved byte offset, keeps bytes up to the last newline, decodes
        # each line into the stream's record type and appends it to the in-memory list; a bad line
        # raises a value error naming the file, offset and line prefix, in which case the offset is
        # not advanced (records of earlier lines in that chunk were already appended and will be
        # duplicated on a retry). Then tail-refreshes every cached partition block. Returns the
        # number of rows added across both. The meta record is not re-read.
        """Read records appended since the last load()/follow(). Returns the count.

        Only complete lines (terminated by a newline) are consumed, so a
        crash-interrupted partial write at the tail is left for a later call
        once its writer completes it. A malformed *complete* line is real
        corruption and raises — for hour-partitioned streams that happens at
        the first windowed read of the bad partition, since those parse
        lazily; here only already-cached partition blocks refresh.
        """
        num_new = 0
        for stream in self.records:
            path = self.dir / stream.filename
            if not path.exists():
                continue
            with open(path, "rb") as f:
                f.seek(self._offsets[stream])
                chunk = f.read()
            end = chunk.rfind(b"\n")
            if end < 0:
                continue
            complete = chunk[: end + 1]
            record_type = _RECORD_TYPE_OF_STREAM[stream]
            for line in complete.splitlines():
                try:
                    self.records[stream].append(record_type.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    raise ValueError(
                        f"corrupt record in {path} near byte {self._offsets[stream]}: {line[:200]!r}"
                    ) from e
                num_new += 1
            self._offsets[stream] += len(complete)
        for reader in self._readers.values():
            num_new += reader.refresh_cached()
        return num_new

    def _parse_columnar(self, stream: Stream, raw: bytes, path: Path, offset: int) -> pl.DataFrame:
        # [FUNCTION DOC] HELPER: Parses newline-delimited JSON bytes into a polars frame with the
        # stream's fixed schema (column set must match exactly, then each column is cast to its
        # type). For engine series, the labels mapping column is converted to a JSON string column
        # named labels_json (the constant empty-object string if all labels are empty or the column
        # is absent) and the original column dropped. Any parse or schema error is re-raised as a
        # value error mentioning the file path and byte offset. path and offset are only used for
        # the message.
        schema = self.FRAME_SCHEMAS[stream]
        try:
            frame = pl.read_ndjson(io.BytesIO(raw))
            if stream is Stream.ENGINE_SERIES:
                # all-empty labels: read_ndjson drops the column entirely
                dtype = frame.schema.get("labels")
                empty = dtype is None or dtype == pl.Null or (isinstance(dtype, pl.Struct) and not dtype.fields)
                # struct schemas unify per chunk (same-writer key order is stable,
                # so equal label dicts encode to equal strings within a store)
                encoded = pl.lit("{}") if empty else pl.col("labels").struct.json_encode()
                frame = frame.with_columns(encoded.alias("labels_json")).drop("labels", strict=False)
            if set(frame.columns) != set(schema):
                raise ValueError(f"fields {sorted(frame.columns)} != schema {sorted(schema)}")
            return frame.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])
        except (pl.exceptions.PolarsError, ValueError) as e:
            raise ValueError(f"corrupt record in {path} near byte {offset}: {e}") from e

    def _parse_object_lines(self, record_type: type[Record], raw: bytes, path: Path, offset: int) -> list[Record]:
        # [FUNCTION DOC] HELPER: Decodes each line of the bytes as JSON into a record of the given
        # type, returning the list in order. A decode, type or key error becomes a value error
        # naming the file, byte offset (of the chunk start, not the line) and the first 200 bytes
        # of the line.
        events = []
        for line in raw.splitlines():
            try:
                events.append(record_type.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                raise ValueError(f"corrupt record in {path} near byte {offset}: {line[:200]!r}") from e
        return events

    def trajectory_events(self, t0: float | None = None, t1: float | None = None) -> list[TrajectoryEvent]:
        # [FUNCTION DOC] Returns trajectory events whose ts lies inside the inclusive window (open
        # ends allowed), in file order (not sorted). Loads the touched hour partitions.
        """Lifecycle events in [t0, t1], keyed by emission ts (no slack here —
        the consuming-step join and its lookback window live in the reader)."""
        return self._window_records(self._readers[Stream.TRAJECTORIES].window(t0, t1), t0, t1)

    @staticmethod
    def _window_records(events: list[TrajectoryEvent], t0: float | None, t1: float | None) -> list[TrajectoryEvent]:
        # [FUNCTION DOC] HELPER: Static. Filters records having a ts field to the inclusive window
        # (a missing bound is open). Despite the annotation it is also used for GPU process records.
        return [e for e in events if (t0 is None or e.ts >= t0) and (t1 is None or e.ts <= t1)]

    def trajectory_lanes(
        self,
        *,
        t0: float | None = None,
        t1: float | None = None,
        sample_indices: set[int] | None = None,
    ) -> list[dict]:
        # [FUNCTION DOC] Groups the window's trajectory events by sample index (optionally limited
        # to the given set) and builds one lane mapping per sample, sorted by first event time.
        # Per sample, events are sorted by ts and walked in order: attempt start remembers the
        # start; attempt end appends an attempt window (start possibly None), records the event's
        # detail as status (keeping the previous when empty), and force-closes every still-open
        # segment at that time, filling a missing weight version from the attempt-end event;
        # gen/tool start opens a segment keyed by (kind, turn); gen/tool end closes the matching
        # open segment (taking the end event's version if non-empty) or, if none is open, adds a
        # segment with unknown start. An attempt still running at the end is added with unknown
        # end. If any generation segment has a positive turn, coarse generation segments (negative
        # turn) are removed. Lane fields: sample index, group index of the first event, first and
        # last event ts, segments (kind, t0, t1, turn, weight version), attempts, status, and the
        # sorted distinct numeric weight versions seen. Unknown event kinds raise a value error.
        # KNOWN ISSUE: only segments are paired by (kind, turn); a second start with the same key
        # replaces the open entry so the first segment is left without an end.
        """Per-sample lifecycle assembly for the batch-anatomy swimlanes:
        gen/tool segments paired by (kind, turn), attempt windows, and the
        version span observed across segments. A start without an end (still
        running, or clipped by the window) yields t1=None and vice versa."""
        grouped: dict[int, list[TrajectoryEvent]] = {}
        for event in self.trajectory_events(t0, t1):
            if sample_indices is not None and event.sample_index not in sample_indices:
                continue
            grouped.setdefault(event.sample_index, []).append(event)

        span_kinds = {
            TrajectoryEventKind.GEN_START: ("gen", True),
            TrajectoryEventKind.GEN_END: ("gen", False),
            TrajectoryEventKind.TOOL_START: ("tool", True),
            TrajectoryEventKind.TOOL_END: ("tool", False),
        }
        lanes = []
        for index, events in grouped.items():
            events.sort(key=lambda e: e.ts)
            segments: list[dict] = []
            open_spans: dict[tuple[str, int], dict] = {}
            attempts: list[dict] = []
            attempt_t0: float | None = None
            status = ""
            versions = sorted({int(e.weight_version) for e in events if e.weight_version.isdigit()})
            for event in events:
                if event.kind == TrajectoryEventKind.ATTEMPT_START:
                    attempt_t0 = event.ts
                elif event.kind == TrajectoryEventKind.ATTEMPT_END:
                    attempts.append(dict(t0=attempt_t0, t1=event.ts))
                    attempt_t0 = None
                    status = event.detail or status
                    # an attempt ending closes whatever it left open: the
                    # single-turn path emits no gen_end (generation ends WITH
                    # the attempt), and an abort mid-turn stops right here —
                    # dangling spans must not render to the consume line
                    for segment in open_spans.values():
                        segment["t1"] = event.ts
                        # coarse gen spans open before the first turn finished,
                        # so their start event predates any weight_version; the
                        # attempt_end event carries the version that generated them
                        if not segment["weight_version"]:
                            segment["weight_version"] = event.weight_version
                    open_spans.clear()
                else:
                    base, is_start = span_kinds[TrajectoryEventKind(event.kind)]
                    if is_start:
                        segment = dict(
                            kind=base, t0=event.ts, t1=None, turn=event.turn, weight_version=event.weight_version
                        )
                        open_spans[(base, event.turn)] = segment
                        segments.append(segment)
                    else:
                        segment = open_spans.pop((base, event.turn), None)
                        if segment is None:
                            segments.append(
                                dict(
                                    kind=base,
                                    t0=None,
                                    t1=event.ts,
                                    turn=event.turn,
                                    weight_version=event.weight_version,
                                )
                            )
                        else:
                            segment["t1"] = event.ts
                            segment["weight_version"] = event.weight_version or segment["weight_version"]
            if attempt_t0 is not None:
                attempts.append(dict(t0=attempt_t0, t1=None))  # attempt still running
            if any(segment["turn"] > 0 for segment in segments if segment["kind"] == "gen"):
                # per-turn spans (multi_turn / agentic) supersede the coarse
                # attempt-level gen span the generate_and_rm probe emits
                segments = [s for s in segments if not (s["kind"] == "gen" and s["turn"] < 0)]
            lanes.append(
                dict(
                    sample_index=index,
                    group_index=events[0].group_index,
                    first_ts=events[0].ts,
                    last_ts=events[-1].ts,
                    segments=segments,
                    attempts=attempts,
                    status=status,
                    versions=versions,
                )
            )
        lanes.sort(key=lambda lane: lane["first_ts"])
        return lanes

    def phase_events(self, t0: float | None = None, t1: float | None = None) -> list[PhaseEvent]:
        # [FUNCTION DOC] Public accessor: all phase events that may matter for a window (see the
        # private helper), including open markers, without lane expansion or de-duplication.
        """Raw phase events, open markers included — advisory heuristics need
        the unexpanded events rather than the lane-resolved view."""
        return self._phase_events(t0, t1)

    def _phase_events(self, t0: float | None, t1: float | None) -> list[PhaseEvent]:
        # [FUNCTION DOC] HELPER: Returns phase events from all partitions up to t1 plus the maximum
        # window (no upper limit when t1 is None). Phases are partitioned by end hour, so an
        # interval overlapping the window can live in a later partition; the lower bound is not
        # applied at all, so older partitions are loaded too. The result is not trimmed to the
        # window. KNOWN ISSUE: the t0 parameter is unused.
        upper = None if t1 is None else t1 + self.MAX_WINDOW_S
        return self._readers[Stream.PHASES].window(None, upper)

    def has_stream(self, stream: Stream) -> bool:
        # [FUNCTION DOC] True if the stream has any data: for partitioned streams any non-empty
        # file on disk (checked live), otherwise any in-memory record loaded so far.
        if stream in self.PARTITIONED_STREAMS:
            return self._readers[stream].has_data()
        return bool(self.records[stream])

    def iter_records(self, stream: Stream) -> list[Record]:
        # [FUNCTION DOC] Materializes every record of a stream as record objects (despite the name,
        # returns a list). GPU utilization and engine series rows are rebuilt from frames (engine
        # labels decoded from JSON with null entries removed); phases, trajectories and GPU
        # processes come from the full-range partitions; other streams copy the in-memory list.
        # Loads everything, so meant for tests and tools.
        """Records as dataclass objects — slow materialization for tests/tools."""
        if stream is Stream.GPU_UTIL:
            frame = self._readers[stream].window(None, None)
            return [GpuSample(**row) for row in frame.iter_rows(named=True)]
        if stream is Stream.ENGINE_SERIES:
            return [
                EngineSample(
                    ts=row["ts"],
                    addr=row["addr"],
                    metric=row["metric"],
                    labels={k: v for k, v in json.loads(row["labels_json"]).items() if v is not None},
                    value=row["value"],
                )
                for row in self._readers[stream].window(None, None).iter_rows(named=True)
            ]
        if stream is Stream.PHASES:
            return list(self._phase_events(None, None))
        if stream is Stream.TRAJECTORIES:
            return self.trajectory_events()
        if stream is Stream.GPU_PROCESSES:
            return self._readers[stream].window(None, None)
        return list(self.records[stream])

    @staticmethod
    def _window(frame: pl.DataFrame, t0: float | None, t1: float | None) -> pl.DataFrame:
        # [FUNCTION DOC] HELPER: Static. Keeps frame rows whose ts column is within the inclusive
        # window; a missing bound is open.
        if t0 is not None:
            frame = frame.filter(pl.col("ts") >= t0)
        if t1 is not None:
            frame = frame.filter(pl.col("ts") <= t1)
        return frame

    # ------------------------------- queries --------------------------------

    def metric_keys(self) -> list[str]:
        # [FUNCTION DOC] Returns the sorted union of metric keys over all loaded metrics records.
        keys: set[str] = set()
        for record in self.records[Stream.METRICS]:
            keys.update(record.metrics.keys())
        return sorted(keys)

    def step_keys(self) -> list[str]:
        # [FUNCTION DOC] Returns the sorted distinct non-null step axis names in the metrics records.
        return sorted({r.step_key for r in self.records[Stream.METRICS] if r.step_key is not None})

    def step_key_of_metric_key(self) -> dict[str, str]:
        # [FUNCTION DOC] Returns a mapping from metric key to the step axis name of the LAST record
        # that contained it; records without a step key are skipped, so keys logged only that way
        # are absent.
        ans: dict[str, str] = {}
        for record in self.records[Stream.METRICS]:
            if record.step_key is None:
                continue
            for key in record.metrics:
                ans[key] = record.step_key
        return ans

    def metric_series(
        self, keys: list[str], *, x_key: str, t0: float | None = None, t1: float | None = None
    ) -> dict[str, dict[str, list]]:
        # [FUNCTION DOC] Returns, for each requested key (all present even if empty), a mapping
        # with parallel lists x (step), y (value) and ts (record time). Only records whose step key
        # equals x_key and whose ts is in the optional inclusive window contribute, in file order.
        """Series for each key, restricted to records logged against ``x_key``
        (e.g. "rollout/step") so values from different step axes never mix."""
        out: dict[str, dict[str, list]] = {k: {"x": [], "y": [], "ts": []} for k in keys}
        for record in self.records[Stream.METRICS]:
            if record.step_key != x_key:
                continue
            if (t0 is not None and record.ts < t0) or (t1 is not None and record.ts > t1):
                continue
            for k in keys:
                if k in record.metrics:
                    series = out[k]
                    series["x"].append(record.step)
                    series["y"].append(record.metrics[k])
                    series["ts"].append(record.ts)
        return out

    def time_range(self) -> tuple[float, float] | None:
        # [FUNCTION DOC] Returns (earliest, latest) timestamp over everything known, or None if
        # empty. Uses every timestamp of the in-memory records plus the cheap edge stamps of each
        # partitioned stream (first line of the oldest file, last complete line of the newest), so
        # it assumes partitioned data is roughly time-ordered at the file edges.
        stamps = [ts for records in self.records.values() for record in records for ts in record.timestamps()]
        for reader in self._readers.values():
            stamps.extend(reader.edge_stamps())
        if not stamps:
            return None
        return min(stamps), max(stamps)

    # --------------------------- timeline queries ---------------------------

    def lanes(self) -> list[dict]:
        # [FUNCTION DOC] Returns the lane list reduced to node, gpu and global index, taken from
        # the lane index (so sorted by node then gpu).
        """Every (node, gpu) seen in any stream — one timeline lane each."""
        return [dict(node=entry["node"], gpu=entry["gpu"], index=entry["index"]) for entry in self.lane_index()]

    def topology_windows(self) -> list[dict]:
        # [FUNCTION DOC] Sorts topology snapshots by ts and returns one mapping each with t0 (its
        # ts), t1 (next snapshot's ts or None for the last) and the engines as plain mappings.
        """Engine topology snapshots with validity windows: snapshot N is valid
        [ts_N, ts_{N+1}); the last one is open-ended (t1=None)."""
        snapshots = sorted(self.records[Stream.TOPOLOGY], key=lambda s: s.ts)
        return [
            dict(
                t0=snapshot.ts,
                t1=snapshots[i + 1].ts if i + 1 < len(snapshots) else None,
                engines=[asdict(engine) for engine in snapshot.engines],
            )
            for i, snapshot in enumerate(snapshots)
        ]

    def latest_data_buffer_length(self) -> int | None:
        # [FUNCTION DOC] Returns the length of the last data buffer record in file order, or None.
        """Most recent ``RolloutDataSourceWithBuffer`` buffer length, or None
        when the run's data source never reported one (plain RolloutDataSource,
        or a dump predating this probe)."""
        records = self.records[Stream.DATA_BUFFER]
        return records[-1].length if records else None

    def phases_by_lane(
        self, *, t0: float | None = None, t1: float | None = None, lanes: set[tuple[str, int]] | None = None
    ) -> list[dict]:
        # [FUNCTION DOC] Returns phase interval mappings (name, t0, t1, node, gpu, rank, role),
        # sorted by node, gpu and start, for intervals overlapping the window, optionally limited
        # to a set of (node, gpu) lanes.
        # Steps: (1) load events; an open marker is dropped if a closed event with the same node,
        # rank, name and start exists, otherwise its end is set to the newest known data timestamp
        # (but not before its start). (2) Train-side events are emitted once per own GPU.
        # Rollout-manager events are expanded per topology window: clipped to the window bounds,
        # skipped if the clipped span is empty, and emitted for every GPU of every engine of that
        # window; if the window's engines list no GPUs (external engines), all known lanes whose
        # node matches an engine address host are used. (3) If meta is known and the window starts
        # at or before the run start, a derived "initialize" interval (rank -1) is added per lane
        # from the run start to the lane's earliest emitted start, when that is later and overlaps
        # the window.
        # KNOWN ISSUE: the branch for a still-open manager event (negative end) is dead code,
        # because open events were already given a non-negative end in step (1).
        """Phase intervals resolved onto (node, gpu) lanes.

        Train-side events carry their own GPUs. ``rollout_manager`` events have
        none — the manager is a GPU-less driver process — so they are expanded
        onto every GPU of every rollout engine, clipped at topology-window
        boundaries (an engine restart mid-rollout splits the painted interval).
        """

        def overlaps(a0: float, a1: float) -> bool:
            # [FUNCTION DOC] HELPER: True if the interval from a0 to a1 starts before the query end
            # and ends after the query start (strict inequalities; missing bounds always pass).
            return (t1 is None or a0 < t1) and (t0 is None or a1 > t0)

        # resolve OPEN intervals (Timer.start markers): a closing event with
        # the real t1 supersedes its open twin; a still-open one is clipped to
        # the newest data timestamp so it renders as a growing in-progress band
        events = list(self._phase_events(t0, t1))
        closed = {(e.node, e.rank, e.name, e.t0) for e in events if not e.open}
        edge = (self.time_range() or (0.0, 0.0))[1]
        events = [
            replace(e, t1=max(edge, e.t0)) if e.open else e
            for e in events
            if not (e.open and (e.node, e.rank, e.name, e.t0) in closed)
        ]

        out = []
        windows = self.topology_windows()
        for event in events:
            if not overlaps(event.t0, event.t1):
                continue
            if event.role == Role.ROLLOUT_MANAGER:
                for window in windows:
                    clip0 = max(event.t0, window["t0"])
                    if event.t1 < 0:
                        # still-open interval: extends to the window edge (the
                        # renderer draws open t1=-1 through to "now")
                        clip1 = -1.0 if window["t1"] is None else window["t1"]
                    else:
                        clip1 = event.t1 if window["t1"] is None else min(event.t1, window["t1"])
                    if clip1 >= 0 and clip0 >= clip1:
                        continue
                    covered = {(node, gpu) for engine in window["engines"] for node, gpu in engine["gpus"]}
                    if not covered:
                        # External engines register without GPU identity; fall
                        # back to every known lane on the engines' nodes.
                        engine_nodes = {engine["addr"].split("//")[-1].split(":")[0] for engine in window["engines"]}
                        covered = {
                            (lane["node"], lane["gpu"]) for lane in self.lanes() if lane["node"] in engine_nodes
                        }
                    if lanes is not None:
                        covered &= lanes
                    for node, gpu in sorted(covered):
                        out.append(
                            dict(
                                name=event.name,
                                t0=clip0,
                                t1=clip1,
                                node=node,
                                gpu=gpu,
                                rank=event.rank,
                                role=event.role,
                            )
                        )
            else:
                for gpu in event.gpus:
                    if lanes is not None and (event.node, gpu) not in lanes:
                        continue
                    out.append(
                        dict(
                            name=event.name,
                            t0=event.t0,
                            t1=event.t1,
                            node=event.node,
                            gpu=gpu,
                            rank=event.rank,
                            role=event.role,
                        )
                    )
        # synthesize the initialize band: from collector start (meta.start_ts)
        # to each lane's first observed event — model loading / engine startup
        # has GPU util but no Timer instrumentation. Only when the window
        # covers the run start: a later window's first event is not the
        # lane's first event ever
        if self.meta is not None and (t0 is None or t0 <= self.meta.start_ts):
            first_event: dict[tuple[str, int], float] = {}
            for event in out:
                key = (event["node"], event["gpu"])
                first_event[key] = min(first_event.get(key, float("inf")), event["t0"])
            for (node, gpu), first_t0 in first_event.items():
                if lanes is not None and (node, gpu) not in lanes:
                    continue
                if first_t0 > self.meta.start_ts and overlaps(self.meta.start_ts, first_t0):
                    out.append(
                        dict(
                            name=INITIALIZE_PHASE,
                            t0=self.meta.start_ts,
                            t1=first_t0,
                            node=node,
                            gpu=gpu,
                            rank=-1,
                            role=Role.DERIVED,
                        )
                    )
        out.sort(key=lambda e: (e["node"], e["gpu"], e["t0"]))
        return out

    def gpu_series(
        self,
        *,
        t0: float | None = None,
        t1: float | None = None,
        max_points: int = 2000,
        lanes: set[tuple[str, int]] | None = None,
    ) -> dict[str, dict]:
        # [FUNCTION DOC] Returns a mapping from lane key (node, colon, gpu) to parallel lists ts,
        # util, mem_mb and power_w, sorted by lane and by time. Rows are the window's GPU samples
        # (exact inclusive filtering), optionally restricted to the lane set. Per lane at most
        # max_points (at least 2) samples are kept by min/max bucket downsampling on utilization;
        # the same indices are applied to all four arrays.
        """Per-lane NVML series, min/max-downsampled on util (the primary
        signal); mem/power take the same indices so all arrays stay aligned."""
        frame = self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
        out = {}
        for (node, gpu), part in sorted(frame.partition_by(["node", "gpu"], as_dict=True).items()):
            if lanes is not None and (node, gpu) not in lanes:
                continue
            part = part.sort("ts")
            util = part["util"].to_numpy()
            indices, _ = minmax_downsample(np.arange(part.height), util, max_points)
            out[f"{node}:{gpu}"] = dict(
                ts=part["ts"].to_numpy()[indices].tolist(),
                util=util[indices].tolist(),
                mem_mb=part["mem_mb"].to_numpy()[indices].tolist(),
                power_w=part["power_w"].to_numpy()[indices].tolist(),
            )
        return out

    def gpu_processes(
        self, *, t0: float | None = None, t1: float | None = None, lanes: set[tuple[str, int]] | None = None
    ) -> list[dict]:
        # [FUNCTION DOC] Returns per-process memory readings (ts, node, gpu, pid, name, mem_mb) in
        # the inclusive window, optionally restricted to a lane set, in file order, no downsampling.
        """Per-process memory snapshots in [t0, t1] (coarser cadence than
        ``gpu_series``): the timeline hover resolves the nearest snapshot per
        lane client-side, same as it already does for the util/mem series."""
        events = self._window_records(self._readers[Stream.GPU_PROCESSES].window(t0, t1), t0, t1)
        return [
            dict(ts=e.ts, node=e.node, gpu=e.gpu, pid=e.pid, name=e.name, mem_mb=e.mem_mb)
            for e in events
            if lanes is None or (e.node, e.gpu) in lanes
        ]

    # cumulative engine families are stored raw; the catalog exposes them as
    # per-interval derived series instead
    ENGINE_RATE_METRICS: ClassVar[dict[str, str]] = {
        "sglang_prompt_tokens_per_s": "sglang_prompt_tokens_total",
        "sglang_generation_tokens_per_s": "sglang_generation_tokens_total",
        "sglang_requests_per_s": "sglang_num_requests_total",
        "sglang_aborted_requests_per_s": "sglang_num_aborted_requests_total",
    }
    ENGINE_MEAN_METRICS: ClassVar[dict[str, str]] = {
        "sglang_ttft_mean_s": "sglang_time_to_first_token_seconds",
        "sglang_inter_token_latency_mean_s": "sglang_inter_token_latency_seconds",
        "sglang_time_per_output_token_mean_s": "sglang_time_per_output_token_seconds",
        "sglang_e2e_latency_mean_s": "sglang_e2e_request_latency_seconds",
    }
    _CUMULATIVE_SUFFIXES: ClassVar[tuple[str, ...]] = ("_total", "_sum", "_count")

    # engine-level aggregation across dp ranks: intensive metrics average,
    # everything else (queue depths, throughputs, cumulative counters) sums
    ENGINE_MEAN_AGGREGATED_METRICS: ClassVar[frozenset[str]] = frozenset(
        {"sglang_token_usage", "sglang_cache_hit_rate", "sglang_kv_transfer_latency_ms"}
    )

    def engine_metric_names(self) -> list[str]:
        # [FUNCTION DOC] Returns the sorted catalog of servable engine metric names: raw names not
        # ending in a cumulative suffix (total, sum, count), plus each derived rate name whose base
        # counter was seen, plus each derived mean name whose base has both sum and count seen.
        # Loads the full engine frame through the cache.
        """Distinct scraped engine metrics — the L0 sglang category catalog."""
        frame, _ = self._engine_cache()
        raw = set(frame.get_column("metric").unique())
        names = {name for name in raw if not name.endswith(self._CUMULATIVE_SUFFIXES)}
        names.update(derived for derived, base in self.ENGINE_RATE_METRICS.items() if base in raw)
        names.update(
            derived
            for derived, base in self.ENGINE_MEAN_METRICS.items()
            if base + "_sum" in raw and base + "_count" in raw
        )
        return sorted(names)

    def engine_series(
        self,
        metric: str,
        *,
        t0: float | None = None,
        t1: float | None = None,
        max_points: int = 2000,
        per_dp_rank: bool = False,
    ) -> list[dict]:
        # [FUNCTION DOC] Returns a list of series mappings (addr, labels, ts, value lists), one per
        # engine address and label set, each reduced to at most max_points (at least 2) points.
        # Derived rate names yield per-second increases of the base counter; derived mean names
        # yield increase of the sum over increase of the count. Other metrics are served raw using
        # evenly spaced index downsampling. The per_dp_rank flag is passed down to the grouping.
        """One series per (engine addr, label set) for the given metric.

        By default per-dp-rank samples (dp-attention engines export one per
        rank) are folded into one engine-level series — summed, or averaged
        for ``ENGINE_MEAN_AGGREGATED_METRICS``; ``per_dp_rank=True`` keeps
        them apart, with ``dp_rank`` in each series' labels.

        Derived names (``ENGINE_RATE_METRICS`` / ``ENGINE_MEAN_METRICS``) are
        computed from adjacent raw cumulative samples; a counter reset or an
        interval without completions leaves a gap, never a fabricated value."""
        if metric in self.ENGINE_RATE_METRICS:
            parts = self._engine_parts(self.ENGINE_RATE_METRICS[metric], t0, t1, per_dp_rank=per_dp_rank)
            return self._derived_series(parts, None, max_points)
        if metric in self.ENGINE_MEAN_METRICS:
            base = self.ENGINE_MEAN_METRICS[metric]
            return self._derived_series(
                self._engine_parts(base + "_sum", t0, t1, per_dp_rank=per_dp_rank),
                self._engine_parts(base + "_count", t0, t1, per_dp_rank=per_dp_rank),
                max_points,
            )
        out = []
        for (addr, labels_json), part in self._engine_parts(metric, t0, t1, per_dp_rank=per_dp_rank).items():
            ts, values = stride_downsample(part["ts"].to_numpy(), part["value"].to_numpy(), max_points)
            out.append(dict(addr=addr, labels=_engine_labels(labels_json), ts=ts.tolist(), value=values.tolist()))
        return out

    def _engine_parts(
        self, metric: str, t0: float | None, t1: float | None, *, per_dp_rank: bool = False
    ) -> dict[tuple[str, str], pl.DataFrame]:
        # [FUNCTION DOC] HELPER: Returns, for one raw metric, a mapping from (address, labels JSON
        # string) to a frame sorted by ts. With no window the memoized full-range result is used.
        # Otherwise the window's partitions are loaded, filtered exactly, restricted to the metric,
        # and (unless per_dp_rank) data-parallel ranks are folded; groups are ordered by key.
        if t0 is None and t1 is None:
            return self._engine_parts_cached(metric, per_dp_rank)
        frame = self._window(self._readers[Stream.ENGINE_SERIES].window(t0, t1), t0, t1).filter(
            pl.col("metric") == metric
        )
        if not per_dp_rank:
            frame = self._fold_dp_ranks(frame, metric)
        return {
            key: part.sort("ts")
            for key, part in sorted(frame.partition_by(["addr", "labels_json"], as_dict=True).items())
        }

    def _fold_dp_ranks(self, frame: pl.DataFrame, metric: str) -> pl.DataFrame:
        # [FUNCTION DOC] HELPER: Collapses rows that differ only by data-parallel rank. First each
        # distinct labels string is rewritten with the rank label and null-valued labels removed
        # (keys sorted), then rows are grouped by timestamp, address, metric and labels, and the
        # value is averaged for metrics in the mean-aggregated set, else summed. An empty frame
        # is returned unchanged. Output columns: ts, addr, metric, labels_json, value.
        """Engine-level view: collapse per-dp-rank samples — including
        same-tick duplicates from dumps written before ``dp_rank`` was kept —
        into one value per (addr, ts). Ranks share a scrape ts (one fetch per
        engine yields every rank's sample), so grouping on ts is exact."""
        if frame.is_empty():
            return frame
        # also drop nulls: chunk-level struct unification pads absent labels
        # with null, which would otherwise split one logical series
        stripped = {
            labels_json: json.dumps(
                {k: v for k, v in json.loads(labels_json).items() if k != "dp_rank" and v is not None},
                sort_keys=True,
            )
            for labels_json in frame["labels_json"].unique()
        }
        frame = frame.with_columns(pl.col("labels_json").replace_strict(stripped))
        agg = pl.col("value").mean() if metric in self.ENGINE_MEAN_AGGREGATED_METRICS else pl.col("value").sum()
        return frame.group_by(["ts", "addr", "metric", "labels_json"]).agg(agg.alias("value"))

    def _engine_cache(self) -> tuple[pl.DataFrame, dict[tuple[str, bool], dict[tuple[str, str], pl.DataFrame]]]:
        # [FUNCTION DOC] HELPER: Under the cache lock, computes a version from the (key, size) of
        # every engine partition file; if it differs from the cached version, reloads the full
        # range frame and resets the per-metric partition cache, setting the version last so a
        # failed reload is retried. Returns the frame and the partition cache dictionary.
        """The full-window frame and its partition cache, returned as a pair so a
        caller racing a rebuild files partitions under the frame they came from."""
        reader = self._readers[Stream.ENGINE_SERIES]
        with self._engine_cache_lock:
            version = tuple((key, path.stat().st_size) for key, path in reader.files())
            if version != self._engine_cache_version:
                self._engine_frame_cache = reader.window(None, None)
                self._engine_parts_cache = {}
                self._engine_cache_version = version  # last: a failed rebuild must not look cached
            return self._engine_frame_cache, self._engine_parts_cache

    def _engine_parts_cached(self, metric: str, per_dp_rank: bool) -> dict[tuple[str, str], pl.DataFrame]:
        # [FUNCTION DOC] HELPER: Returns the per-(address, labels) frames of one metric over the
        # full range, memoized per (metric, per_dp_rank) in the cache that belongs to the current
        # full frame. Computed like the windowed variant without a window filter.
        frame, cache = self._engine_cache()
        cache_key = (metric, per_dp_rank)
        parts = cache.get(cache_key)
        if parts is None:
            frame = frame.filter(pl.col("metric") == metric)
            if not per_dp_rank:
                frame = self._fold_dp_ranks(frame, metric)
            parts = {
                key: part.sort("ts")
                for key, part in sorted(frame.partition_by(["addr", "labels_json"], as_dict=True).items())
            }
            cache[cache_key] = parts
        return parts

    def _derived_series(self, num_parts, den_parts, max_points: int) -> list[dict]:
        # [FUNCTION DOC] HELPER: Turns cumulative counter frames into series. Inputs are mappings
        # from (address, labels) to frames with ts and value, for the numerator and, optionally,
        # the denominator. For each numerator group: with a denominator, the matching group is
        # required (else skipped) and both are inner-joined on identical timestamps; without one,
        # the time difference is the denominator. At least two points are needed. Consecutive
        # differences are computed; intervals with a negative numerator difference (counter reset)
        # or a non-positive denominator difference are dropped, so gaps remain. Each kept point is
        # stamped with the later timestamp and its value is difference ratio; the points are
        # downsampled evenly to at most max_points. Groups with no points are omitted. Returns
        # mappings of addr, labels (nulls removed), ts and value lists.
        """Per-interval derivative: rate when ``den_parts`` is None (dt as the
        denominator), else delta(num)/delta(den) joined on scrape timestamps."""
        out = []
        for (addr, labels_json), num in num_parts.items():
            ts = num["ts"].to_numpy()
            values = num["value"].to_numpy()
            if den_parts is None:
                den_ts, den_values = ts, None
            else:
                den = den_parts.get((addr, labels_json))
                if den is None:
                    continue
                den_ts, den_values = den["ts"].to_numpy(), den["value"].to_numpy()
                shared, num_idx, den_idx = np.intersect1d(ts, den_ts, return_indices=True)
                ts, values, den_values = shared, values[num_idx], den_values[den_idx]
            if len(ts) < 2:
                continue
            d_num = np.diff(values)
            d_den = np.diff(ts) if den_values is None else np.diff(den_values)
            keep = (d_num >= 0) & (d_den > 0)
            point_ts, point_values = ts[1:][keep], d_num[keep] / d_den[keep]
            if not len(point_ts):
                continue
            point_ts, point_values = stride_downsample(point_ts, point_values, max_points)
            out.append(
                dict(addr=addr, labels=_engine_labels(labels_json), ts=point_ts.tolist(), value=point_values.tolist())
            )
        return out

    def lane_index(self) -> list[dict]:
        # [FUNCTION DOC] Returns one mapping per lane (node, gpu, ranks, engine_addrs, roles, and a
        # global index), sorted by node then gpu, index being the sorted position. If the catalog
        # file exists it is read (every call, uncached) and the key split at its last colon.
        # Otherwise lanes are derived by scanning: GPUs seen in GPU samples, phase events and
        # topology snapshots; train phase events add the train role and non-negative ranks,
        # topology engines add the rollout role and address. In the scan fallback the lane order
        # follows the sorted set of (node, gpu) pairs.
        """Per-lane metadata for selection resolution: train ranks observed on
        the lane, engine addrs that ever covered it, and derived roles.

        Reads ``lane_catalog.json`` (merge-written at flush time) when
        present; a legacy dir without one falls back to a full-stream scan.
        """
        path = self.dir / self.CATALOG_FILENAME
        if path.exists():
            lanes = json.loads(path.read_text())["lanes"]
            out = []
            for key, entry in lanes.items():
                node, _, gpu = key.rpartition(":")
                out.append(
                    dict(
                        node=node,
                        gpu=int(gpu),
                        ranks=entry["ranks"],
                        engine_addrs=entry["engine_addrs"],
                        roles=entry["roles"],
                    )
                )
            out.sort(key=lambda entry: (entry["node"], entry["gpu"]))
            for i, entry in enumerate(out):
                entry["index"] = i
            return out

        seen: set[tuple[str, int]] = set(
            self._readers[Stream.GPU_UTIL].window(None, None).select("node", "gpu").unique().iter_rows()
        )
        events = self._phase_events(None, None)
        for event in events:
            for gpu in event.gpus:
                seen.add((event.node, gpu))
        for snapshot in self.records[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    seen.add((node, gpu))
        info: dict[tuple[str, int], dict] = {
            (node, gpu): dict(node=node, gpu=gpu, ranks=set(), engine_addrs=set(), roles=set())
            for node, gpu in sorted(seen)
        }
        for event in events:
            if event.role != Role.TRAIN:
                continue
            for gpu in event.gpus:
                entry = info.get((event.node, gpu))
                if entry is not None:
                    entry["roles"].add(Role.TRAIN.value)
                    if event.rank >= 0:
                        entry["ranks"].add(event.rank)
        for snapshot in self.records[Stream.TOPOLOGY]:
            for engine in snapshot.engines:
                for node, gpu in engine.gpus:
                    entry = info.get((node, gpu))
                    if entry is not None:
                        entry["roles"].add("rollout")
                        entry["engine_addrs"].add(engine.addr)
        return [
            dict(
                node=entry["node"],
                gpu=entry["gpu"],
                index=i,
                ranks=sorted(entry["ranks"]),
                engine_addrs=sorted(entry["engine_addrs"]),
                roles=sorted(entry["roles"]),
            )
            for i, entry in enumerate(info.values())
        ]

    def resolve_lanes(self, grammar: str | None) -> set[tuple[str, int]] | None:
        # [FUNCTION DOC] Returns None (meaning all lanes) for a missing, blank or "all" grammar;
        # otherwise the union of the lanes matched by each comma-separated selector, as a set of
        # (node, gpu) pairs. Integer ranges are inclusive. Rank and global-index selectors may be
        # a single number or low-high; node matches exactly; gpu takes node and index split at the
        # last colon and is added even if the lane is unknown; engine matches by substring of any
        # engine address; every selects lanes whose global index is divisible by the stride; role
        # selects train or rollout lanes. Empty tokens are skipped. Raises a value error for a
        # token without a value, an unknown kind, a missing node in a gpu selector, a stride not
        # positive or an invalid role; non-numeric numbers raise the usual conversion value error.
        """Parse the lane-selection grammar into a lane set (None = all lanes).

        Comma-separated selectors: ``rank:5`` / ``rank:0-7`` (train ranks),
        ``g:5`` / ``g:0-31`` (global lane numbers), ``node:<ip>``,
        ``gpu:<node>:<index>``, ``engine:<addr substring>``,
        ``role:train`` / ``role:rollout``, ``every:<stride>`` (equidistant
        global-index sampling), or ``all``. Unknown selector syntax
        raises; a valid selector matching nothing selects nothing.
        """
        if grammar is None or grammar.strip() in ("", "all"):
            return None
        index = self.lane_index()
        selected: set[tuple[str, int]] = set()
        for raw_token in grammar.split(","):
            token = raw_token.strip()
            if not token:
                continue
            kind, _, value = token.partition(":")
            if not value:
                raise ValueError(f"bad lane selector {token!r}: expected kind:value")
            if kind == "rank":
                lo, _, hi = value.partition("-")
                ranks = set(range(int(lo), int(hi if hi else lo) + 1))
                selected |= {(e["node"], e["gpu"]) for e in index if ranks & set(e["ranks"])}
            elif kind == "g":
                # cluster-global lane numbers (the g{index} labels on every view)
                lo, _, hi = value.partition("-")
                positions = set(range(int(lo), int(hi if hi else lo) + 1))
                selected |= {(e["node"], e["gpu"]) for e in index if e["index"] in positions}
            elif kind == "node":
                selected |= {(e["node"], e["gpu"]) for e in index if e["node"] == value}
            elif kind == "gpu":
                node, _, gpu = value.rpartition(":")
                if not node:
                    raise ValueError(f"bad lane selector {token!r}: expected gpu:<node>:<index>")
                selected.add((node, int(gpu)))
            elif kind == "engine":
                selected |= {(e["node"], e["gpu"]) for e in index if any(value in a for a in e["engine_addrs"])}
            elif kind == "every":
                stride = int(value)
                if stride <= 0:
                    raise ValueError(f"bad lane selector {token!r}: stride must be positive")
                selected |= {(e["node"], e["gpu"]) for e in index if e["index"] % stride == 0}
            elif kind == "role":
                if value not in ("train", "rollout"):
                    raise ValueError(f"bad lane selector {token!r}: role must be train or rollout")
                selected |= {(e["node"], e["gpu"]) for e in index if value in e["roles"]}
            else:
                raise ValueError(f"unknown lane selector kind {kind!r} in {token!r}")
        return selected

    HEATMAP_MAGNITUDE_FIELDS: ClassVar[tuple[str, ...]] = ("util", "mem_mb", "power_w")

    def heatmap(
        self,
        metric: str,
        *,
        t0: float | None = None,
        t1: float | None = None,
        x_buckets: int = 1200,
        lanes: set[tuple[str, int]] | None = None,
    ) -> dict:
        # [FUNCTION DOC] Builds the carpet. Returns a mapping with rows (per-lane node, gpu, global
        # index, roles; or per-sample indices for lifecycle), x_buckets, t0, t1, metric, values
        # (raw bytes of a rows-by-x_buckets unsigned 8-bit matrix, row-major), scale and palette.
        # x_buckets must be at least 2 (assertion). With no data at all, returns empty rows and
        # bytes with t0 and t1 zero. Missing bounds default to the overall time range; end must be
        # after start (assertion). The metric "lifecycle" delegates to the lifecycle carpet and
        # ignores the lane set. Rows are the lane index entries filtered by the lane set.
        # Magnitude metrics (util, mem_mb, power_w): GPU samples are bucketed by time (clamped to
        # the edge buckets), the per-cell maximum is kept, and cells are scaled to 0-255 by 100 for
        # util or by the observed peak (1.0 if no samples) otherwise, capped at 255; cells without
        # samples stay 0; scale holds the maximum, palette is None.
        # Metric "phase": cell value is the 1-based id of the phase name (sorted names; palette
        # element 0 is the empty string for none); idle phases (initialize, train_wait, sleep) are
        # painted first so any active phase in the same cell overwrites them; each interval fills
        # its clipped bucket range inclusively; scale is None. Other metrics raise a value error.
        """Rank-carpet matrix: one uint8 cell per (lane, time bucket).

        Magnitude metrics take the bucket MAX (activity must not average away);
        ``phase`` paints the covering phase id with non-idle winning over idle.
        Returns rows metadata + the raw bytes; the server frames it as binary.
        """
        assert x_buckets >= 2, f"{x_buckets=}"
        window = self.time_range()
        if window is None:
            return dict(
                rows=[], x_buckets=x_buckets, t0=0.0, t1=0.0, metric=metric, values=b"", scale=None, palette=None
            )
        t0 = window[0] if t0 is None else t0
        t1 = window[1] if t1 is None else t1
        assert t1 > t0, f"empty heatmap window [{t0}, {t1})"
        if metric == "lifecycle":
            return self._lifecycle_heatmap(t0, t1, x_buckets)
        rows = [
            dict(node=entry["node"], gpu=entry["gpu"], index=entry["index"], roles=entry["roles"])
            for entry in self.lane_index()
            if lanes is None or (entry["node"], entry["gpu"]) in lanes
        ]
        row_of = {(lane["node"], lane["gpu"]): i for i, lane in enumerate(rows)}
        matrix = np.zeros((len(rows), x_buckets), dtype=np.uint8)
        span = t1 - t0

        def bucket(ts: float) -> int:
            # [FUNCTION DOC] HELPER: Maps a timestamp to its column index in the window, truncating
            # toward zero and clamping to the range 0 to x_buckets minus 1.
            return min(x_buckets - 1, max(0, int((ts - t0) / span * x_buckets)))

        if metric in self.HEATMAP_MAGNITUDE_FIELDS:
            frame = self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
            values = np.zeros((len(rows), x_buckets))
            filled = np.zeros((len(rows), x_buckets), dtype=bool)
            for (node, gpu), part in frame.partition_by(["node", "gpu"], as_dict=True).items():
                row = row_of.get((node, gpu))
                if row is None:
                    continue
                buckets = np.clip(((part["ts"].to_numpy() - t0) / span * x_buckets).astype(np.int64), 0, x_buckets - 1)
                np.maximum.at(values[row], buckets, part[metric].to_numpy().astype(float))
                filled[row, buckets] = True
            peak = float(values[filled].max()) if filled.any() else 1.0
            scale = dict(max=100.0 if metric == "util" else peak)
            cell = np.minimum(255, (values[filled] / scale["max"] * 255).astype(np.int64))
            matrix[filled] = cell.astype(np.uint8)
            return dict(
                rows=rows,
                x_buckets=x_buckets,
                t0=t0,
                t1=t1,
                metric=metric,
                values=matrix.tobytes(),
                scale=scale,
                palette=None,
            )

        if metric != "phase":
            raise ValueError(f"unknown heatmap metric {metric!r}")
        idle = {INITIALIZE_PHASE, "train_wait", "sleep"}
        events = self.phases_by_lane(t0=t0, t1=t1, lanes=lanes)
        names = sorted({e["name"] for e in events})
        palette = [""] + names  # id 0 = no phase observed
        phase_id = {name: i + 1 for i, name in enumerate(names)}
        # idle painted first so any active phase in the same bucket wins
        for event in sorted(events, key=lambda e: e["name"] not in idle):
            row = row_of.get((event["node"], event["gpu"]))
            if row is None:
                continue
            b0 = bucket(max(event["t0"], t0))
            b1 = bucket(min(event["t1"], t1))
            matrix[row, b0 : b1 + 1] = phase_id[event["name"]]
        return dict(
            rows=rows,
            x_buckets=x_buckets,
            t0=t0,
            t1=t1,
            metric=metric,
            values=matrix.tobytes(),
            scale=None,
            palette=palette,
        )

    LIFECYCLE_PALETTE: ClassVar[tuple[str, ...]] = ("", "queue", "generating", "tool_wait")
    LIFECYCLE_MAX_ROWS: ClassVar[int] = 1024  # y-cap: thousands of samples must not explode the canvas

    def _lifecycle_heatmap(self, t0: float, t1: float, x_buckets: int) -> dict:
        # [FUNCTION DOC] HELPER: Builds the lifecycle carpet for the given window. Takes the
        # trajectory lanes in the window, keeps the first LIFECYCLE_MAX_ROWS (1024) in order. Each
        # row is filled first with the queue state (id 1) across every attempt window, then with
        # generating (2) or tool wait (3) across each segment, so segments overwrite the queue
        # state; a missing start or end (None or zero) extends to the window edge. Cell id 0 means
        # nothing. Returns rows (sample index and group index), the uncapped lane count as
        # rows_total, x_buckets, t0, t1, metric lifecycle, values bytes, scale None and the palette
        # list (empty, queue, generating, tool_wait).
        """Run-wide trajectory carpet: y = samples ordered by first event,
        color = lifecycle state. Queue paints attempt windows first so gen and
        tool segments win the shared buckets. Rows cap at LIFECYCLE_MAX_ROWS
        (submit order, earliest kept); ``rows_total`` reports the uncapped
        count so the frontend can say "showing N/M"."""
        all_lanes = self.trajectory_lanes(t0=t0, t1=t1)
        lanes = all_lanes[: self.LIFECYCLE_MAX_ROWS]
        matrix = np.zeros((len(lanes), x_buckets), dtype=np.uint8)
        span = t1 - t0
        state_id = {name: i for i, name in enumerate(self.LIFECYCLE_PALETTE)}

        def bucket(ts: float) -> int:
            # [FUNCTION DOC] HELPER: Same mapping as in the main carpet: timestamp to a column index
            # clamped to the valid range.
            return min(x_buckets - 1, max(0, int((ts - t0) / span * x_buckets)))

        for row, lane in enumerate(lanes):
            for attempt in lane["attempts"]:
                matrix[row, bucket(attempt["t0"] or t0) : bucket(attempt["t1"] or t1) + 1] = state_id["queue"]
            for segment in lane["segments"]:
                cell = state_id["generating" if segment["kind"] == "gen" else "tool_wait"]
                matrix[row, bucket(segment["t0"] or t0) : bucket(segment["t1"] or t1) + 1] = cell
        rows = [dict(sample_index=lane["sample_index"], group_index=lane["group_index"]) for lane in lanes]
        return dict(
            rows=rows,
            rows_total=len(all_lanes),
            x_buckets=x_buckets,
            t0=t0,
            t1=t1,
            metric="lifecycle",
            values=matrix.tobytes(),
            scale=None,
            palette=list(self.LIFECYCLE_PALETTE),
        )

    def fleet(self, *, t0: float | None = None, t1: float | None = None, x_buckets: int = 600) -> dict:
        # [FUNCTION DOC] Builds the fleet overview from the phase carpet over all lanes. Result:
        # t0, t1 (resolved window), x_buckets, lane count, phase palette, composition (a mapping
        # from phase name, "none" for the empty id, to a list per bucket of the fraction of lanes
        # in that phase, rounded to 4 decimals; empty if no lanes) and band (lists per bucket of
        # p10, p50, p90 and min of per-lane utilization, null for buckets without samples). The
        # band takes the maximum utilization per lane per bucket first, then quantiles across lanes.
        # With no data, the empty carpet has a zero window and no lanes. Asserts x_buckets >= 2.
        """Scale-invariant fleet overview: per-bucket phase composition
        (fraction of lanes in each phase) and the util distribution band
        across all lanes. Payload size is O(x_buckets), never O(lanes)."""
        phase = self.heatmap("phase", t0=t0, t1=t1, x_buckets=x_buckets)
        palette = phase["palette"] or []
        n_rows = len(phase["rows"])
        composition = {}
        if n_rows and palette:
            matrix = np.frombuffer(phase["values"], dtype=np.uint8).reshape(n_rows, x_buckets)
            for pid, name in enumerate(palette):
                fractions = (matrix == pid).sum(axis=0) / n_rows
                composition[name or "none"] = [round(float(v), 4) for v in fractions]

        band = dict(p10=[None] * x_buckets, p50=[None] * x_buckets, p90=[None] * x_buckets, min=[None] * x_buckets)
        rt0, rt1 = phase["t0"], phase["t1"]
        frame = self._window(self._readers[Stream.GPU_UTIL].window(rt0, rt1), rt0, rt1)
        if len(frame) and rt1 > rt0:
            # exact percentiles from the raw stream: carpet cells cannot
            # distinguish "no sample" from 0% util
            span = rt1 - rt0
            per_bucket = (
                frame.with_columns(
                    bucket=((pl.col("ts") - rt0) / span * x_buckets).floor().clip(0, x_buckets - 1).cast(pl.Int32)
                )
                .group_by("bucket", "node", "gpu")
                .agg(pl.col("util").max().alias("u"))
                .group_by("bucket")
                .agg(
                    p10=pl.col("u").quantile(0.1),
                    p50=pl.col("u").quantile(0.5),
                    p90=pl.col("u").quantile(0.9),
                    min=pl.col("u").min(),
                )
            )
            for row in per_bucket.iter_rows(named=True):
                for key in ("p10", "p50", "p90", "min"):
                    band[key][row["bucket"]] = float(row[key])

        return dict(
            t0=rt0, t1=rt1, x_buckets=x_buckets, lanes=n_rows, palette=palette, composition=composition, band=band
        )

    def outliers(
        self, criterion: str, *, t0: float | None = None, t1: float | None = None, top_k: int = 16
    ) -> list[dict]:
        # [FUNCTION DOC] Returns up to top_k mappings (node, gpu, score). Criterion "lowest_util"
        # scores each lane by mean utilization over the window and returns the lowest first.
        # Criterion "slowest_phase:<name>" scores each lane by the mean duration of intervals of
        # that phase name and returns the highest first. Ties are ordered by node and gpu in the
        # tuple sort. Any other criterion raises a value error.
        """Candidate lanes for the quick-pick buttons; the user confirms the
        selection — the machine only proposes."""
        if criterion == "lowest_util":
            frame = self._window(self._readers[Stream.GPU_UTIL].window(t0, t1), t0, t1)
            grouped = frame.group_by("node", "gpu").agg(pl.col("util").mean().alias("score"))
            scored = [(row["score"], row["node"], row["gpu"]) for row in grouped.iter_rows(named=True)]
            scored.sort()
        elif criterion.startswith("slowest_phase:"):
            phase_name = criterion.removeprefix("slowest_phase:")
            durations: dict[tuple[str, int], list[float]] = {}
            for event in self.phases_by_lane(t0=t0, t1=t1):
                if event["name"] != phase_name:
                    continue
                durations.setdefault((event["node"], event["gpu"]), []).append(event["t1"] - event["t0"])
            scored = [(sum(v) / len(v), node, gpu) for (node, gpu), v in durations.items()]
            scored.sort(reverse=True)
        else:
            raise ValueError(f"unknown outlier criterion {criterion!r}")
        return [dict(node=node, gpu=gpu, score=score) for score, node, gpu in scored[:top_k]]

    def bubbles(self) -> list[dict]:
        # [FUNCTION DOC] For each metrics record whose step key ends with the rollout step suffix,
        # looks up the step time and the wait-time ratio metrics (exact key first, else the first
        # key ending with a slash and that name); records with neither are skipped. Returns
        # mappings of step, ts, step_time and wait_ratio (either may be None), in file order.
        """Per-step bubble summary strip: step time and wait ratio from the
        perf metrics, with the wall-clock ts as the timeline zoom anchor."""
        out = []
        for record in self.records[Stream.METRICS]:
            if record.step_key is None or not record.step_key.endswith("rollout/step"):
                continue
            step_time = _metric_by_suffix(record.metrics, "perf/step_time")
            wait_ratio = _metric_by_suffix(record.metrics, "perf/wait_time_ratio")
            if step_time is None and wait_ratio is None:
                continue
            out.append(dict(step=record.step, ts=record.ts, step_time=step_time, wait_ratio=wait_ratio))
        return out


def _metric_by_suffix(metrics: dict[str, float], base_key: str) -> float | None:
    # [FUNCTION DOC] HELPER: Returns the value for the exact key if present and not null,
    # otherwise the first value (in mapping order) whose key ends with a slash plus the base key,
    # otherwise None.
    if (value := metrics.get(base_key)) is not None:
        return value
    for key, value in metrics.items():
        if key.endswith(f"/{base_key}"):
            return value
    return None


# ------------------------------ downsampling --------------------------------


def _engine_labels(labels_json: str) -> dict:
    # [FUNCTION DOC] HELPER: Decodes a labels JSON string into a mapping, dropping null values.
    return {k: v for k, v in json.loads(labels_json).items() if v is not None}


def stride_downsample(xs, ys, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    # [FUNCTION DOC] Reduces two parallel sequences to at most max_points (asserted at least 2)
    # by picking evenly spaced indices from first to last element inclusive (truncated to
    # integers). Returns the two arrays unchanged if already short enough. Returns arrays.
    xs, ys = np.asarray(xs), np.asarray(ys)
    assert max_points >= 2, f"{max_points=}"
    if len(xs) <= max_points:
        return xs, ys
    idx = np.linspace(0, len(xs) - 1, max_points).astype(np.int64)
    return xs[idx], ys[idx]


def minmax_downsample(xs, ys, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    # [FUNCTION DOC] Splits the sequences into max_points divided by two (integer division)
    # equal-index buckets and keeps, per non-empty bucket, the positions of the minimum and the
    # maximum of ys (the first occurrence of each), returned in original order without duplicates.
    # Asserts max_points >= 2; returns inputs unchanged (as arrays) when already short enough.
    # Output length is at most 2 times the bucket count.
    """Bucketed downsampling keeping each bucket's min and max, so spikes and
    dips survive. Output length is at most ``max_points``."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    assert max_points >= 2, f"{max_points=}"
    if len(xs) <= max_points:
        return xs, ys
    num_buckets = max_points // 2
    edges = np.linspace(0, len(xs), num_buckets + 1).astype(np.int64)
    keep: set[int] = set()
    for b in range(num_buckets):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if lo >= hi:
            continue
        segment = ys[lo:hi]
        keep.add(lo + int(np.argmin(segment)))
        keep.add(lo + int(np.argmax(segment)))
    idx = sorted(keep)
    return xs[idx], ys[idx]
