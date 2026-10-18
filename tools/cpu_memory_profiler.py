# CPU Memory Profiler - standalone tool for profiling system memory during training.
#
# Profiler usage (e.g. in train.py):
#   from tools.cpu_memory_profiler import CPUMemoryProfiler
#
#   # beginning of training loop
#   profiler = CPUMemoryProfiler(interval=0.5, output_path="cpu_memory_profile.csv")
#   profiler.start()
#
#   # at somewhere you want to mark with a label
#   profiler.mark("step_0/generate_start")
#
#   # end of training loop
#   profiler.stop()
#
# [MODULE DOC] Standalone library (importable as a tools module) with a background-thread
# sampler of machine-wide memory (used, available, percent) that writes a CSV timeline with
# phase labels attached to the first sample at or after each marker time.
import csv
import logging
import threading
import time

import psutil

logger = logging.getLogger(__name__)


class CPUMemoryProfiler:
    # [CLASS DOC] Records are tuples of elapsed seconds, used bytes, available bytes and percent;
    # markers are tuples of elapsed seconds and label. A lock guards both lists. Lifecycle: start,
    # optional marks, stop (writes the CSV). Not restartable cleanly; the CSV is only written
    # at stop.
    """System-level CPU memory profiler for Ray multi-process training.

    Samples total machine memory usage at regular intervals and supports
    phase markers to align with training stages (generate, offload, train, etc.).
    Uses psutil.virtual_memory() which naturally handles shared memory deduplication
    across Ray workers.
    """

    def __init__(self, interval=0.5, output_path="cpu_memory_profile.csv"):
        # [FUNCTION DOC] Stores the sampling interval in seconds (default 0.5) and CSV path
        # (default cpu_memory_profile.csv), and initializes empty records and markers, a lock, a
        # not-running state and no start time.
        self.interval = interval
        self.output_path = output_path
        self._running = False
        self._records = []  # [(elapsed_s, used_bytes, available_bytes, percent)]
        self._markers = []  # [(elapsed_s, label)]
        self._lock = threading.Lock()
        self._t0 = None

    def start(self):
        # [FUNCTION DOC] Records the start time, marks the profiler running, starts the daemon
        # sampling thread and logs a start message. Calling twice starts a second thread.
        self._t0 = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.info(f"CPU memory profiler started (interval={self.interval}s, output={self.output_path})")

    def _sample_loop(self):
        # [FUNCTION DOC] HELPER: Thread body. While running: reads machine memory and appends
        # (seconds since start, used, available, percent) under the lock, then sleeps for the
        # interval (so stop may wait up to one interval to join).
        while self._running:
            vm = psutil.virtual_memory()
            with self._lock:
                self._records.append(
                    (
                        time.time() - self._t0,
                        vm.used,
                        vm.available,
                        vm.percent,
                    )
                )
            time.sleep(self.interval)

    def mark(self, label: str):
        # [FUNCTION DOC] If running, appends (elapsed seconds, label) under the lock and logs the
        # label with the time, used gigabytes and percent from a fresh memory reading. Does
        # nothing when not started (or after stop).
        """Place a named marker on the timeline for phase alignment. No-op if not started."""
        if not self._running:
            return
        with self._lock:
            elapsed = time.time() - self._t0
            self._markers.append((elapsed, label))
        vm = psutil.virtual_memory()
        logger.info(f"[CPU profiler] {label} @ {elapsed:.1f}s, used={vm.used / 1e9:.2f}GB ({vm.percent}%)")

    def stop(self):
        # [FUNCTION DOC] Stops sampling, waits for the thread to finish, writes the CSV and logs
        # the peak used gigabytes and path. Raises an attribute error if never started.
        self._running = False
        self._thread.join()
        self._dump()
        logger.info(
            f"CPU memory profiler stopped. Peak used: {self.peak_used_gb:.2f}GB. " f"Saved to {self.output_path}"
        )

    def _dump(self):
        # [FUNCTION DOC] HELPER: Writes the CSV with columns time (seconds, two decimals), used
        # and available gigabytes (decimal, three decimals), percent (one decimal) and marker.
        # Markers are sorted by time; each is attached to the first sample whose time is at or
        # after the marker time (several joined by semicolons); markers later than the last
        # sample are dropped. Logs the path.
        markers_sorted = sorted(self._markers)
        with open(self.output_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "used_gb", "available_gb", "percent", "marker"])
            mi = iter(markers_sorted)
            next_marker = next(mi, None)
            for t, used, avail, pct in self._records:
                markers_in_interval = []
                while next_marker and next_marker[0] <= t:
                    markers_in_interval.append(next_marker[1])
                    next_marker = next(mi, None)
                w.writerow(
                    [
                        f"{t:.2f}",
                        f"{used / 1e9:.3f}",
                        f"{avail / 1e9:.3f}",
                        f"{pct:.1f}",
                        ";".join(markers_in_interval),
                    ]
                )
        logger.info(f"CPU memory profile saved to {self.output_path}")

    @property
    def peak_used_gb(self):
        # [FUNCTION DOC] Maximum used memory over all samples in decimal gigabytes (bytes divided
        # by one billion); zero if there are no samples.
        if not self._records:
            return 0.0
        return max(r[1] for r in self._records) / 1e9
