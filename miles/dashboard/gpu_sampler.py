# [MODULE DOC]
# Per-node GPU telemetry sampler. Two thin provider classes adapt the NVIDIA (pynvml) and AMD (amdsmi)
# management libraries to one interface of three methods: initialize (returns device handles and
# UUIDs), read one device (utilization percent, used memory in MiB, power in watts) and read the
# processes holding GPU memory (pid, name, MiB). The sampler class picks a provider (injected or
# auto-detected, NVIDIA first), samples all devices on a background thread, buffers records and pushes
# batches via injected callables. All library failures degrade gracefully (warning, skipped tick).
"""Per-node GPU utilization sampler feeding the dashboard timeline.

One instance runs per GPU node (the collector spawns it as a Ray actor with
``NodeAffinitySchedulingStrategy``; the class itself is plain Python and
unit-testable). A daemon thread samples every SMI-visible device on the node at
``interval`` seconds — physical device order, independent of CUDA/HIP process
visibility variables — buffers locally, and hands batches to the injected
``push(node, batch)`` callable (the collector wraps its own Ray handle) every
``FLUSH_INTERVAL_SECONDS``, so there is roughly one RPC per node per flush
rather than one per sample.

Degradation: NVML is preferred and AMD SMI is the fallback; neither being
usable (no pynvml/amdsmi, driver mismatch) disables the sampler with a single
warning — the timeline just lacks the util band. A device that fails mid-run
(e.g. during a GPU reset) is skipped for that tick with rate-limited warnings;
the other devices keep reporting.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections.abc import Callable
from typing import ClassVar

from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import GpuProcessSample, GpuSample

logger = logging.getLogger(__name__)


def _text(value) -> str:
    # [FUNCTION DOC] HELPER: Decode bytes to text, or convert any other value to its string form.
    return value.decode() if isinstance(value, bytes) else str(value)


# Providers share initialize/read_device/read_processes; initialization errors enable auto-detection fallback.


class _NvmlProvider:
    # [CLASS DOC] HELPER: Provider over the NVML Python module (name "NVML", module name "pynvml").
    name = "NVML"
    module = "pynvml"

    def __init__(self, api):
        # [FUNCTION DOC] HELPER: Store the NVML-like module object (real or injected fake).
        self._api = api

    def initialize(self) -> tuple[list, list[str]]:
        # [FUNCTION DOC] HELPER: Initializes the library, counts devices (raises a runtime error if
        # zero) and returns the list of device handles (by physical index) plus their UUID strings.
        self._api.nvmlInit()
        count = self._api.nvmlDeviceGetCount()
        if count == 0:
            raise RuntimeError("no NVML devices")
        handles = [self._api.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        return handles, [_text(self._api.nvmlDeviceGetUUID(handle)) for handle in handles]

    def read_device(self, handle) -> tuple[int, int, int]:
        # [FUNCTION DOC] HELPER: Returns (GPU utilization percent, used memory in MiB computed by
        # shifting bytes right 20 bits, power in whole watts from milliwatts by integer division).
        # Any library error propagates.
        util = int(self._api.nvmlDeviceGetUtilizationRates(handle).gpu)
        mem_mb = int(self._api.nvmlDeviceGetMemoryInfo(handle).used) >> 20
        power_w = int(self._api.nvmlDeviceGetPowerUsage(handle)) // 1000
        return util, mem_mb, power_w

    def read_processes(self, handle) -> list[tuple[int, str, int]]:
        # [FUNCTION DOC] HELPER: For each compute process on the device returns (pid, process name,
        # GPU memory in MiB, zero when unreported). If the name lookup fails (process gone), the name
        # is "pid <pid>". Only compute processes are listed.
        processes = []
        for proc in self._api.nvmlDeviceGetComputeRunningProcesses(handle):
            pid = int(proc.pid)
            try:
                name = _text(self._api.nvmlSystemGetProcessName(pid))
            except Exception:
                name = f"pid {pid}"  # process exited between enumeration and lookup, or name unavailable
            processes.append((pid, name, int(proc.usedGpuMemory or 0) >> 20))
        return processes


class _AmdSmiProvider:
    # [CLASS DOC] HELPER: Provider over the AMD SMI Python module (name "AMD SMI", module name
    # "amdsmi"), with the same three-method interface as the NVML provider.
    name = "AMD SMI"
    module = "amdsmi"

    def __init__(self, api):
        # [FUNCTION DOC] HELPER: Store the AMD SMI-like module object.
        self._api = api

    def initialize(self) -> tuple[list, list[str]]:
        # [FUNCTION DOC] HELPER: Initializes the library, gets processor handles (raises a runtime
        # error if there are none) and returns them in slot order with their device UUID strings.
        self._api.amdsmi_init()
        handles = self._api.amdsmi_get_processor_handles()
        if not handles:
            raise RuntimeError("no AMD SMI devices")
        # SMI slot order matches Ray; partitioned MI300+ cards may repeat physical-GPU UUIDs.
        return list(handles), [_text(self._api.amdsmi_get_gpu_device_uuid(handle)) for handle in handles]

    def read_device(self, handle) -> tuple[int, int, int]:
        # [FUNCTION DOC] HELPER: Returns (graphics activity percent, VRAM used already in MiB, socket
        # power in watts). Power read errors of any kind are swallowed and reported as 0 watts.
        util = int(self._api.amdsmi_get_gpu_activity(handle)["gfx_activity"])
        # Unlike NVML, AMD SMI reports VRAM usage in MiB already.
        mem_mb = int(self._api.amdsmi_get_gpu_vram_usage(handle)["vram_used"])
        try:
            power_w = _amd_socket_power(self._api.amdsmi_get_power_info(handle))
        except Exception:
            power_w = 0
        return util, mem_mb, power_w

    def read_processes(self, handle) -> list[tuple[int, str, int]]:
        # [FUNCTION DOC] HELPER: For each listed process computes VRAM in MiB by shifting a byte count
        # right 20 bits and skips processes holding zero VRAM (bystander pids). Returns (pid, name or
        # "pid <pid>" when the name is missing or "N/A", MiB). The per-process VRAM value is treated
        # as bytes here (unlike the device-level value, which is already MiB).
        processes = []
        for proc in self._api.amdsmi_get_gpu_process_list(handle):
            mem_mb = int((proc.get("memory_usage") or {}).get("vram_mem") or 0) >> 20
            if mem_mb == 0:
                continue  # KFD lists bystander pids holding no VRAM; NVML reports only compute processes
            pid = int(proc["pid"])
            name = proc.get("name")
            processes.append((pid, str(name) if name and name != "N/A" else f"pid {pid}", mem_mb))
        return processes


def _amd_socket_power(power: dict) -> int:
    # [FUNCTION DOC] HELPER: Given the power info mapping, tries the keys socket power, current
    # socket power and average socket power in that order and returns the first value that is not the
    # string "N/A" and lies in [0, 65535) as an int. Raises a value error if none qualifies (a
    # non-numeric non-"N/A" value would raise a type error on comparison).
    # Prefer MI300+ socket power, then split fields; unavailable sensors use "N/A" or 0xFFFF.
    for field in ("socket_power", "current_socket_power", "average_socket_power"):
        value = power.get(field, "N/A")
        if value != "N/A" and 0 <= value < 0xFFFF:
            return int(value)
    raise ValueError(f"AMD SMI socket power unavailable: {power!r}")


class GpuSampler:
    # [CLASS DOC] Samples every GPU visible to the management library on this node. Class constants:
    # flush interval 5 seconds and process-sample interval 5 seconds. State: push callables, node name,
    # sample interval, provider, device handles, UUIDs, two lock-protected buffers (device samples and
    # process samples), stop event, thread, a rate-limited warner and a public availability flag. Plain
    # Python (no Ray) so it is testable; the collector wraps it in an actor.
    FLUSH_INTERVAL_SECONDS: ClassVar[float] = 5.0
    # per-process memory breakdown is a coarser, heavier SMI call (enumerates
    # every process) than the plain util/mem read, so it samples on its own,
    # slower cadence rather than every `interval` tick
    PROCESS_SAMPLE_INTERVAL_SECONDS: ClassVar[float] = 5.0

    def __init__(
        self,
        *,
        push: Callable[[str, list[GpuSample]], None],
        node: str,
        interval: float = 1.0,
        nvml=None,
        push_processes: Callable[[str, list[GpuProcessSample]], None] | None = None,
        amdsmi=None,
    ):
        # [FUNCTION DOC] Keyword-only. push: callable taking (node, list of GPU samples); node: node
        # name string; interval: seconds between samples (default 1.0, must be positive, assertion
        # error otherwise); nvml / amdsmi: optional injected library modules (at most one, assertion
        # error otherwise); push_processes: optional callable for per-process records (without it
        # the process sampling never runs). Initializes the provider at construction; the result is
        # stored as the availability flag.
        assert interval > 0, f"{interval=}"
        assert nvml is None or amdsmi is None, "inject only one GPU telemetry backend"
        self._push = push
        self._push_processes = push_processes
        self.node = node
        self.interval = interval
        self._provider = None  # _NvmlProvider or _AmdSmiProvider once initialized
        self._handles: list = []
        self._uuids: list[str] = []
        self._buffer: list[GpuSample] = []
        self._process_buffer: list[GpuProcessSample] = []
        self._buffer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._warner = RateLimitedWarner(logger)
        self.available = self._init_provider(nvml=nvml, amdsmi=amdsmi)

    def _init_provider(self, *, nvml, amdsmi) -> bool:
        # [FUNCTION DOC] HELPER: Candidate providers: the injected NVML module, else the injected AMD
        # module, else auto-detect (NVML then AMD SMI, importing each library by name). For each
        # candidate, tries to build the provider and initialize it; any exception (import error, driver
        # error, no devices) is recorded, logged at debug level, and the next candidate is tried. On the
        # first success stores the provider (handles and UUIDs are stored by the attempt) and returns
        # true. If all fail logs exactly one warning (naming the single failure, or all failures
        # joined) and returns false.
        if nvml is not None:
            candidates = [(_NvmlProvider, nvml)]
        elif amdsmi is not None:
            candidates = [(_AmdSmiProvider, amdsmi)]
        else:  # auto-detect: try NVML first, then AMD SMI
            candidates = [(_NvmlProvider, None), (_AmdSmiProvider, None)]

        failures = []
        for cls, api in candidates:
            try:
                provider = cls(api if api is not None else importlib.import_module(cls.module))
                self._handles, self._uuids = provider.initialize()
            except Exception as error:
                failures.append((cls.name, str(error)))
                logger.debug("%s unavailable on %s", cls.name, self.node, exc_info=True)
                continue
            self._provider = provider
            return True

        if len(failures) == 1:
            name, error = failures[0]
            logger.warning("%s unavailable on %s (%s); GPU utilization will not be collected", name, self.node, error)
        else:
            detail = "; ".join(f"{name}: {error}" for name, error in failures)
            logger.warning(
                "GPU telemetry unavailable on %s (%s); GPU utilization will not be collected", self.node, detail
            )
        return False

    # ------------------------------ lifecycle -------------------------------

    def gpu_uuids(self) -> list[str]:
        # [FUNCTION DOC] Copy of the device UUID list in device order (empty if unavailable).
        return list(self._uuids)

    def start(self) -> bool:
        # [FUNCTION DOC] Returns false without starting if unavailable. Otherwise asserts it has not
        # started, starts the daemon thread "dashboard-gpu-sampler" and returns true.
        """Begin sampling; returns False (and stays inert) when no SMI backend is usable."""
        if not self.available:
            return False
        assert self._thread is None, "sampler already started"
        self._thread = threading.Thread(target=self._run, name="dashboard-gpu-sampler", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        # [FUNCTION DOC] Signals the thread to stop, joins it with a timeout of the sample interval
        # plus the flush interval, then flushes buffered records synchronously.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + self.FLUSH_INTERVAL_SECONDS)
        self.flush()

    def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Thread body (monotonic deadlines for flush and process sampling,
        # both 5 s ahead). Loop until stopped: sample all devices with the wall-clock time; if a
        # process push callable exists and its deadline passed, sample processes and reschedule; if
        # the flush deadline passed, flush and reschedule; then wait one interval (or the stop signal).
        # Exceptions from the push callables propagate and end the thread.
        next_flush = time.monotonic() + self.FLUSH_INTERVAL_SECONDS
        next_process_sample = time.monotonic() + self.PROCESS_SAMPLE_INTERVAL_SECONDS
        while not self._stop_event.is_set():
            self.sample_once(time.time())
            if self._push_processes is not None and time.monotonic() >= next_process_sample:
                self.sample_processes_once(time.time())
                next_process_sample = time.monotonic() + self.PROCESS_SAMPLE_INTERVAL_SECONDS
            if time.monotonic() >= next_flush:
                self.flush()
                next_flush = time.monotonic() + self.FLUSH_INTERVAL_SECONDS
            self._stop_event.wait(self.interval)

    # -------------------------------- sampling ------------------------------

    def sample_once(self, ts: float) -> int:
        # [FUNCTION DOC] Returns 0 if unavailable. For each device by index, reads utilization, memory
        # and power; a failure logs a rate-limited warning and skips that device for this tick. Each
        # success appends a GPU sample (timestamp, node, gpu index, util, MiB, watts) to the buffer
        # under the lock. Returns the number of samples taken.
        """Sample every device once into the buffer. Returns the sample count."""
        if not self.available:
            return 0
        count = 0
        for gpu, handle in enumerate(self._handles):
            try:
                util, mem_mb, power_w = self._provider.read_device(handle)
            except Exception:
                self._warner.warn(
                    f"{self._provider.name} read failed for gpu {gpu} on {self.node}; skipping this tick"
                )
                continue
            with self._buffer_lock:
                self._buffer.append(
                    GpuSample(ts=ts, node=self.node, gpu=gpu, util=util, mem_mb=mem_mb, power_w=power_w)
                )
            count += 1
        return count

    def sample_processes_once(self, ts: float) -> int:
        # [FUNCTION DOC] Returns 0 if unavailable. For each device reads its process list (failure:
        # rate-limited warning, skip device) and appends one process record (timestamp, node, gpu,
        # pid, name, MiB) per process to the process buffer under the lock. Returns the number of
        # records added.
        """Per-process VRAM breakdown once per GPU: who is actually holding
        the memory, not just the per-GPU aggregate ``sample_once`` reports."""
        if not self.available:
            return 0
        count = 0
        for gpu, handle in enumerate(self._handles):
            try:
                processes = self._provider.read_processes(handle)
            except Exception:
                self._warner.warn(
                    f"{self._provider.name} process query failed for gpu {gpu} on {self.node}; skipping this tick"
                )
                continue
            for pid, name, mem_mb in processes:
                with self._buffer_lock:
                    self._process_buffer.append(
                        GpuProcessSample(ts=ts, node=self.node, gpu=gpu, pid=pid, name=name, mem_mb=mem_mb)
                    )
                count += 1
        return count

    def flush(self) -> None:
        # [FUNCTION DOC] Atomically takes both buffers under the lock, then (outside it) pushes the
        # sample batch if non-empty and the process batch if non-empty and a process push callable
        # exists. Without a process push callable the taken process records are discarded.
        with self._buffer_lock:
            batch, self._buffer = self._buffer, []
            process_batch, self._process_buffer = self._process_buffer, []
        if batch:
            self._push(self.node, batch)
        if process_batch and self._push_processes is not None:
            self._push_processes(self.node, process_batch)
