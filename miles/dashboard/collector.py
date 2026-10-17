# [MODULE DOC]
# Ray-free ingest hub of the dashboard. A collector object owns a metric store (JSONL streams on
# disk), a lock-protected set of in-memory buffers, a periodic flush thread, optional per-node GPU
# sampler actors, an optional engine metrics scraper and optional Prometheus forwarding. Module
# level: the named-actor constant, two small callable sink classes that push GPU records back into the
# collector actor, default implementations (plus replaceable module variables used as test seams)
# for listing GPU nodes, spawning and killing samplers, and the configuration record.
"""The dashboard ingest hub.

Producers (Timer sinks on every rank, the rollout manager's hooks, per-node
GPU samplers) push records here; the collector buffers them, appends to the
JSONL streams under ``{dump_details}/dashboard/`` on a flush cadence, runs
the sglang scraper thread once a router is registered, and optionally
forwards a latest-value snapshot to the existing Prometheus collector for
external Grafana.

This class is deliberately Ray-free: the backend glue (``backend.py``) wraps
it in a named Ray actor pinned to the driver node and spawns the per-node
sampler actors — the collector itself only ever sees plain method calls, so
every behavior here is unit-testable. Producers call in fire-and-forget
style; nothing in the training path ever waits on this class.

Failure policy: if the disk write fails (disk full, NFS hiccup) the error is
logged LOUDLY on every flush attempt — never masked — while ingestion keeps
running with bounded buffers (oldest records dropped past the cap) so a disk
problem cannot OOM the driver.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from miles.dashboard.sglang_scraper import DEFAULT_METRIC_WHITELIST, ScrapeMode, SglangScraper
from miles.dashboard.store import (
    DataBufferSample,
    EngineInfo,
    EngineSample,
    GpuProcessSample,
    GpuSample,
    Meta,
    MetricsRecord,
    MetricStore,
    PhaseEvent,
    Record,
    TopologySnapshot,
    TrajectoryEvent,
)

logger = logging.getLogger(__name__)

COLLECTOR_ACTOR_NAME = "miles_dashboard_collector"


class _SelfGpuPush:
    # [CLASS DOC] HELPER: Picklable callable holding the collector actor handle. Called by a sampler
    # with a node name and a batch of GPU samples; sends them asynchronously (fire-and-forget) to the
    # collector's GPU sample ingestion method.
    """GpuSampler sink pushing back into this collector actor (the sampler
    runs on its own node; the handle crosses the process boundary)."""

    def __init__(self, handle):
        # [FUNCTION DOC] HELPER: Store the actor handle.
        self._handle = handle

    def __call__(self, node: str, batch: list[GpuSample]) -> None:
        # [FUNCTION DOC] HELPER: Asynchronously forward (node, batch) to the collector; no result.
        self._handle.push_gpu_samples.remote(node, batch)


class _SelfGpuProcessPush:
    # [CLASS DOC] HELPER: Same pattern as the GPU sample pusher, for per-process GPU memory records.
    """Same as ``_SelfGpuPush`` for the per-process memory breakdown stream."""

    def __init__(self, handle):
        # [FUNCTION DOC] HELPER: Store the actor handle.
        self._handle = handle

    def __call__(self, node: str, batch: list[GpuProcessSample]) -> None:
        # [FUNCTION DOC] HELPER: Asynchronously forward (node, batch) to the collector's GPU process
        # ingestion method; no result.
        self._handle.push_gpu_processes.remote(node, batch)


def _default_list_gpu_nodes() -> list[tuple[str, str]]:
    # [FUNCTION DOC] HELPER: Returns (node id, node manager address) for each alive Ray node that has
    # at least one GPU resource. Returns an empty list without importing Ray if Ray is not already
    # imported or not initialized in this process (unit tests, tooling).
    # no initialized ray in this process = not running as the collector actor
    # (unit tests, tooling): nothing to reconcile, and never pay the import
    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return []
    return [
        (node["NodeID"], node["NodeManagerAddress"])
        for node in ray.nodes()
        if node.get("Alive") and node.get("Resources", {}).get("GPU", 0) > 0
    ]


def _default_spawn_sampler(node_id: str, node_ip: str, interval: float):
    # [FUNCTION DOC] HELPER: Creates a GPU sampler Ray actor with no CPU reservation, hard-pinned to
    # the node, wired to push samples and process records back into the current (collector) actor, and
    # starts it (blocking). Returns the handle if sampling started; otherwise kills the actor and
    # returns none.
    """Spawn + start a GpuSampler pinned to one node; None when NVML is
    unavailable there (recorded so the node is not retried every tick)."""
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from miles.dashboard.gpu_sampler import GpuSampler

    handle = (
        ray.remote(GpuSampler)
        .options(
            num_cpus=0,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        )
        .remote(
            push=_SelfGpuPush(ray.get_runtime_context().current_actor),
            node=node_ip,
            interval=interval,
            push_processes=_SelfGpuProcessPush(ray.get_runtime_context().current_actor),
        )
    )
    if ray.get(handle.start.remote()):
        return handle
    ray.kill(handle)  # NVML unavailable; the sampler already warned
    return None


def _default_kill_sampler(handle) -> None:
    # [FUNCTION DOC] HELPER: Kill the sampler Ray actor.
    import ray

    ray.kill(handle)


# test seams; production always uses the defaults
_list_gpu_nodes = _default_list_gpu_nodes
_spawn_sampler = _default_spawn_sampler
_kill_sampler = _default_kill_sampler


@dataclass
class CollectorConfig:
    # [CLASS DOC] Plain configuration record: output directory, run name, start timestamp, snapshot of
    # selected run args, flush interval (default 5.0 s), GPU sample interval (1.0 s), engine scrape
    # interval (2.0 s), scrape mode string ("auto" resolves to direct when the router is set; else a
    # scrape-mode value), tuple of metric names to scrape (default whitelist), and Prometheus
    # forwarding flag (default off).
    dashboard_dir: str  # {dump_details}/dashboard
    run_name: str
    start_ts: float
    args_snapshot: dict[str, Any] = field(default_factory=dict)
    flush_interval_seconds: float = 5.0
    gpu_sample_interval_seconds: float = 1.0
    scrape_interval_seconds: float = 2.0
    scrape_mode: str = "auto"  # "auto" or a ScrapeMode value; auto resolves at set_router()
    metric_whitelist: tuple[str, ...] = DEFAULT_METRIC_WHITELIST
    forward_prometheus: bool = False


class DashboardCollector:
    # [CLASS DOC] Thread-safe ingest hub (one lock guards the store and latest-value caches). Ingestion
    # methods are cheap and callable from any producer. A background thread (started by the start
    # method) periodically reconciles GPU samplers, folds scraped external engines into the topology and
    # flushes buffered records to disk. Per-stream buffers are capped at 500000 records (oldest dropped
    # first, counted and reported at the next flush) so persistent disk failure cannot exhaust memory.
    # State: store, topology last seen, scraper, latest-value caches for Prometheus, address sets for
    # scraped and actor-registered engines, and the node-id to sampler-handle mapping (none for nodes
    # without GPU telemetry).
    # bounded ingest buffers: past this many buffered records per stream the
    # oldest are dropped (only reachable when flushing to disk keeps failing)
    MAX_BUFFERED_PER_STREAM: ClassVar[int] = 500_000

    def __init__(
        self,
        *,
        config: CollectorConfig,
        prometheus_handle_factory=None,  # () -> handle with .update.remote(dict), or None
        scraper_http_get=None,  # test hook, forwarded to SglangScraper
    ):
        # [FUNCTION DOC] Keyword-only. Creates the store on the configured directory and immediately
        # writes run metadata (run name, start timestamp, args snapshot) to disk. Initializes the
        # lock, drop counter, caches and empty sampler mapping. The optional Prometheus factory is a
        # callable returning an actor handle with a remote update method, or none. The optional HTTP
        # get function is a test hook passed to the scraper.
        self.config = config
        self._store = MetricStore(config.dashboard_dir)
        self._store.write_meta(Meta(run_name=config.run_name, start_ts=config.start_ts, args=config.args_snapshot))
        self._lock = threading.Lock()
        self._dropped_since_flush = 0
        self._last_topology: TopologySnapshot | None = None
        self._scraper: SglangScraper | None = None
        self._scraper_http_get = scraper_http_get
        self._prometheus_handle_factory = prometheus_handle_factory
        # latest-value caches for the Prometheus forwarding snapshot; kept
        # separately because store buffers empty out on every flush
        self._latest_gpu: dict[tuple[str, int], GpuSample] = {}
        self._latest_running_reqs: dict[str, tuple[float, float]] = {}
        self._latest_phase_seconds: dict[str, float] = {}
        self._scraped_engine_addrs: set[str] = set()
        self._actor_engine_addrs: set[str] = set()
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None
        # per-node GPU samplers, keyed by ray NodeID and reconciled on every
        # flush tick: nodes joining AFTER startup (late rollout nodes, node
        # restarts) get a sampler too — a one-shot spawn at init cannot.
        # None marks a node where NVML is unavailable (never retried).
        self._samplers: dict[str, Any] = {}

    # ------------------------------ lifecycle -------------------------------

    def ping(self) -> bool:
        # [FUNCTION DOC] Liveness check for the actor creator; always true.
        return True

    def start(self) -> None:
        # [FUNCTION DOC] Starts the daemon flush thread named "dashboard-flush"; asserts it was not
        # started before.
        assert self._flush_thread is None, "collector already started"
        self._flush_thread = threading.Thread(target=self._run_flush_loop, name="dashboard-flush", daemon=True)
        self._flush_thread.start()

    def shutdown(self) -> None:
        # [FUNCTION DOC] Stops the scraper if any, signals the flush loop to stop and joins it with a
        # timeout of the flush interval plus one second, kills all live sampler actors and clears the
        # mapping, then performs one final flush (which logs but does not raise on disk errors).
        if self._scraper is not None:
            self._scraper.stop()
        self._stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=self.config.flush_interval_seconds + 1)
        for handle in self._samplers.values():
            if handle is not None:
                _kill_sampler(handle)
        self._samplers.clear()
        self.flush()

    # ------------------------------- ingestion ------------------------------

    def push_metrics(self, record: MetricsRecord) -> None:
        # [FUNCTION DOC] Ingest one metrics record.
        self._append(record)

    def push_phases(self, batch: list[PhaseEvent]) -> None:
        # [FUNCTION DOC] Ingest each phase event of the batch (also updates the latest phase duration
        # cache, using end minus start).
        for event in batch:
            self._append(event)

    def push_trajectories(self, batch: list[TrajectoryEvent]) -> None:
        # [FUNCTION DOC] Ingest each trajectory event of the batch.
        for event in batch:
            self._append(event)

    def push_gpu_samples(self, node: str, batch: list[GpuSample]) -> None:
        # [FUNCTION DOC] Ingest each GPU sample (also updates the latest sample per node and GPU).
        # KNOWN ISSUE: the node argument is unused; each sample carries its own node.
        for sample in batch:
            self._append(sample)

    def push_gpu_processes(self, node: str, batch: list[GpuProcessSample]) -> None:
        # [FUNCTION DOC] Ingest each per-process GPU memory record. KNOWN ISSUE: the node argument is
        # unused; each record carries its own node.
        for sample in batch:
            self._append(sample)

    def push_data_buffer(self, sample: DataBufferSample) -> None:
        # [FUNCTION DOC] Ingest one data-buffer length sample.
        self._append(sample)

    def update_topology(self, snapshot: TopologySnapshot) -> None:
        # [FUNCTION DOC] Registers the engine topology. Addresses of engines not marked external are
        # remembered permanently as actor-known. The snapshot is merged with engines known only from
        # scraping (see the merge helper). Under the lock, if an earlier topology exists and its engine
        # list equals the new one, nothing is recorded (steady-state re-registration); otherwise the
        # new snapshot becomes the latest and is appended to the store.
        # an addr the actor path ever registered must not be resurrected as
        # "external" after a restart retires it — scrape memory is a fallback
        # for engines the actor path NEVER knew, not a liveness source
        self._actor_engine_addrs.update(e.addr for e in snapshot.engines if e.worker_type != "external")
        snapshot = self._with_external_engines(snapshot)
        with self._lock:
            if self._last_topology is not None and self._last_topology.engines == snapshot.engines:
                return  # steady-state re-registration; only changes are recorded
            self._last_topology = snapshot
        self._append(snapshot)

    def _with_external_engines(self, snapshot: TopologySnapshot) -> TopologySnapshot:
        # [FUNCTION DOC] HELPER: Adds a synthetic engine entry (worker type "external", rank -1, empty
        # GPU lists) for every scraped address that is neither in the snapshot nor ever known to the
        # actor path, in sorted address order. Returns the original snapshot if none are needed,
        # otherwise a new snapshot with the same timestamp. Reads the scraped address set without
        # locking.
        """Merge in engines known only from scraping: externally launched
        sglang servers have no miles engine actor, so actor registration
        never sees them — but every scraped sample carries their addr. Node
        is the addr host; GPU placement is unknown (gpus=[]), which the
        frontend resolves by node match."""
        covered = {engine.addr for engine in snapshot.engines} | self._actor_engine_addrs
        synthetic = [
            EngineInfo(addr=addr, worker_type="external", engine_rank=-1, gpus=[], gpu_uuids=[])
            for addr in sorted(self._scraped_engine_addrs)
            if addr not in covered
        ]
        if not synthetic:
            return snapshot
        return TopologySnapshot(ts=snapshot.ts, engines=snapshot.engines + synthetic)

    def _sync_external_topology(self) -> None:
        # [FUNCTION DOC] HELPER: Under the lock, computes scraped addresses missing from the last
        # topology and from the actor-known set, and copies the last topology's engine list. Outside
        # the lock, if any are missing, re-registers a fresh-timestamp topology with the same engines,
        # which causes the merge to add the external ones.
        """Flush-loop step (outside the ingest lock): fold newly scraped
        engine addrs into the topology. _update_latest only RECORDS addrs —
        it runs under self._lock, and update_topology takes the same lock."""
        with self._lock:
            covered = set() if self._last_topology is None else {e.addr for e in self._last_topology.engines}
            missing = self._scraped_engine_addrs - covered - self._actor_engine_addrs
            base = [] if self._last_topology is None else list(self._last_topology.engines)
        if missing:
            self.update_topology(TopologySnapshot(ts=time.time(), engines=base))

    def set_router(self, router_addr: str) -> None:
        # [FUNCTION DOC] Scrape mode: "auto" becomes direct; otherwise the configured mode value. If
        # a scraper already exists with the same router address and mode, does nothing. Otherwise
        # detaches the old scraper under the lock, stops it outside the lock (to avoid a deadlock with
        # its sink), builds a new scraper (sink = the append method, mode, router address, a callable
        # giving current engine addresses, interval and whitelist from config, plus the test HTTP get
        # function if provided), stores it under the lock, starts it and logs. Raises a value error if
        # the configured mode is not a valid scrape mode.
        """Register the sglang router and start (or re-point) the scraper."""
        mode = ScrapeMode.DIRECT if self.config.scrape_mode == "auto" else ScrapeMode(self.config.scrape_mode)
        # never hold the lock while stopping a scraper: its thread may be
        # blocked on the same lock inside the _append sink (deadlock)
        with self._lock:
            previous = self._scraper
            if previous is not None and previous.router_addr == router_addr and previous.mode == mode:
                return
            self._scraper = None
        if previous is not None:
            previous.stop()
        kwargs = dict(
            mode=mode,
            router_addr=router_addr,
            engine_addrs=self._current_engine_addrs,
            interval=self.config.scrape_interval_seconds,
            whitelist=self.config.metric_whitelist,
        )
        if self._scraper_http_get is not None:
            kwargs["http_get"] = self._scraper_http_get
        scraper = SglangScraper(self._append, **kwargs)
        with self._lock:
            self._scraper = scraper
        scraper.start()
        logger.info("dashboard scraper started in %s mode against %s", mode, router_addr)

    def _current_engine_addrs(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Addresses of the engines in the last registered topology (under the
        # lock); empty list if none registered. Passed to the scraper as its engine list provider.
        with self._lock:
            if self._last_topology is None:
                return []
            return [engine.addr for engine in self._last_topology.engines]

    def _append(self, record: Record) -> None:
        # [FUNCTION DOC] HELPER: Under the lock: if the record's stream buffer has reached the cap,
        # drops its oldest buffered records and adds their count to the drop counter; then appends the
        # record to the store and updates the latest-value caches.
        with self._lock:
            if self._store.buffered_count(record.stream) >= self.MAX_BUFFERED_PER_STREAM:
                self._dropped_since_flush += self._store.drop_oldest_buffered(record.stream)
            self._store.append(record)
            self._update_latest(record)

    def _update_latest(self, record: Record) -> None:
        # [FUNCTION DOC] HELPER: Called with the lock held. GPU sample: remember per (node, gpu).
        # Engine sample: remember its address as scraped (only records, no topology update, because
        # that would re-take the lock); for the running-requests metric keep, per address, the latest
        # timestamp and the sum of values sharing that timestamp (summing per-dp-rank series; a new
        # timestamp restarts the sum). Phase event: remember its duration (end minus start) by name.
        if isinstance(record, GpuSample):
            self._latest_gpu[(record.node, record.gpu)] = record
        elif isinstance(record, EngineSample):
            self._scraped_engine_addrs.add(record.addr)
            if record.metric == "sglang_num_running_reqs":
                ts, total = self._latest_running_reqs.get(record.addr, (None, 0.0))
                self._latest_running_reqs[record.addr] = (
                    record.ts,
                    (total if ts == record.ts else 0.0) + record.value,
                )
        elif isinstance(record, PhaseEvent):
            self._latest_phase_seconds[record.name] = record.t1 - record.t0

    # -------------------------------- flushing ------------------------------

    def _run_flush_loop(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body. Until stopped: wait one flush interval (or the stop
        # signal), reconcile samplers and sync external topology (any exception is logged with
        # traceback and retried next tick), then flush. Note a final flush still occurs after the stop
        # signal wakes the wait.
        while not self._stop_event.is_set():
            self._stop_event.wait(self.config.flush_interval_seconds)
            try:
                self._reconcile_samplers()
                self._sync_external_topology()
            except Exception:
                logger.exception("dashboard sampler reconcile failed; will retry next tick")
            self.flush()

    def _reconcile_samplers(self) -> None:
        # [FUNCTION DOC] HELPER: Lists alive GPU nodes via the seam; forgets samplers for nodes no
        # longer alive; for each alive node without an entry (including failed ones removed by the
        # previous step only if the node disappeared), spawns a sampler via the seam with the
        # configured sample interval and stores the handle (or none when GPU telemetry is
        # unavailable, which prevents retries). Not under the lock; runs only on the flush thread.
        """Diff alive GPU nodes against owned samplers; spawn the missing.

        Runs on the flush cadence (a ray.nodes() call is cheap), so late-
        joining nodes start reporting util within one flush interval. A node
        restart changes its NodeID, which reads as gone + new -> respawn."""
        alive = _list_gpu_nodes()
        alive_ids = {node_id for node_id, _ in alive}
        for node_id in list(self._samplers):
            if node_id not in alive_ids:
                del self._samplers[node_id]  # node gone; actor died with it
        for node_id, node_ip in alive:
            if node_id not in self._samplers:
                self._samplers[node_id] = _spawn_sampler(node_id, node_ip, self.config.gpu_sample_interval_seconds)

    def flush(self) -> None:
        # [FUNCTION DOC] Under the lock: if records were dropped, logs an error with the count and
        # resets it; flushes the store to disk, and on an OS error logs the exception loudly and
        # returns (records stay buffered, and Prometheus forwarding is skipped for this tick). Outside
        # the lock, if forwarding is enabled and a factory is given, resolves the Prometheus handle
        # (may be none) and sends it the latest-value snapshot asynchronously. Other exceptions
        # propagate.
        with self._lock:
            if self._dropped_since_flush:
                logger.error(
                    "dashboard collector dropped %d records since the last flush "
                    "(buffers past the cap — is the disk full?)",
                    self._dropped_since_flush,
                )
                self._dropped_since_flush = 0
            try:
                self._store.flush()
            except OSError:
                # deliberately loud on EVERY failed flush: a disk problem must
                # surface, not be masked; buffers stay bounded via _append
                logger.exception("dashboard flush to %s failed; records stay buffered", self.config.dashboard_dir)
                return
        if self.config.forward_prometheus and self._prometheus_handle_factory is not None:
            handle = self._prometheus_handle_factory()
            if handle is not None:
                handle.update.remote(self._prometheus_snapshot())

    # -------------------------- prometheus forwarding -----------------------

    def _prometheus_snapshot(self) -> dict[str, float]:
        # [FUNCTION DOC] HELPER: Under the lock builds a flat mapping of gauge name to float: per GPU
        # utilization ("dashboard/gpu_<node>_<gpu>_util") and memory in MB ("..._mem_mb"), per engine
        # running requests ("dashboard/engine_<addr>_running_reqs") and per phase last duration
        # ("dashboard/phase_<name>_seconds"). Node and address text is sanitized by removing the http
        # scheme prefix and replacing dots and colons with underscores. Phase names are not sanitized.
        """Latest-value gauges for external Grafana. Keys avoid characters the
        prometheus collector's sanitizer does not handle ('.', ':')."""

        def safe(text: str) -> str:
            # [FUNCTION DOC] HELPER: Strip an "http://" prefix occurrence and replace dots and colons
            # with underscores.
            return text.replace("http://", "").replace(".", "_").replace(":", "_")

        with self._lock:
            snapshot = {
                f"dashboard/gpu_{safe(node)}_{gpu}_util": float(sample.util)
                for (node, gpu), sample in self._latest_gpu.items()
            }
            snapshot |= {
                f"dashboard/gpu_{safe(node)}_{gpu}_mem_mb": float(sample.mem_mb)
                for (node, gpu), sample in self._latest_gpu.items()
            }
            snapshot |= {
                f"dashboard/engine_{safe(addr)}_running_reqs": value
                for addr, (_, value) in self._latest_running_reqs.items()
            }
            snapshot |= {
                f"dashboard/phase_{name}_seconds": seconds for name, seconds in self._latest_phase_seconds.items()
            }
        return snapshot
