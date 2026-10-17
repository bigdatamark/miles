# [MODULE DOC] Command-line entry point that serves the dashboard web app over a dump directory.
# Parses command-line options, builds a metric store and a dump reader on the directory, wraps
# them in the HTTP application from the server module and runs it with uvicorn (blocking).
# Options: dump directory (required unless demo mode), host (default all interfaces), port
# (default 7788), follow mode, tensor LRU size (default 2), cache directory, demo mode, a flag
# forcing the fleet utilization overview, and the low-MFU warning threshold (default taken from
# the advisory module; zero disables the rule).
# In follow mode a daemon thread calls the store's follow method every FOLLOW_INTERVAL_SECONDS
# (2 seconds) to pick up appended telemetry lines.
"""Standalone dashboard server over a ``--dump-details`` directory.

    python -m miles.dashboard.serve --dump-details /path/to/dump_details \\
        [--host 0.0.0.0] [--port 7788] [--follow] [--tensor-lru 2] [--cache-dir DIR]

Works on any machine that can see the directory (login node, laptop over
NFS, the training node itself). ``--follow`` tails the JSONL telemetry
streams every few seconds for quasi-live viewing of a running job; dump
files are re-discovered per request either way. Typical remote usage is
behind an SSH port-forward.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import uvicorn

from miles.dashboard.advisory import DEFAULT_LOW_MFU
from miles.dashboard.dump_reader import DumpReader
from miles.dashboard.server import make_app
from miles.dashboard.store import MetricStore

FOLLOW_INTERVAL_SECONDS = 2.0


def make_demo_dir(target: Path) -> Path:
    # [FUNCTION DOC] Fills the target directory with synthetic data for demo mode and returns it.
    # Imports the test-suite dummy generators lazily (they only exist in a repo checkout); if the
    # import fails it exits the process with a message instead of raising an import error.
    # Writes 3 steps of dummy rollout/train dumps (8 prompts, 4 samples per prompt, max response
    # length 48) and then 3 steps of dummy telemetry (32 samples per step) into the directory.
    """Populate ``target`` with generated demo data (dumps + telemetry).

    Uses the dummy generators from the test suite, so it requires a miles
    repo checkout (they are deliberately not shipped in the wheel)."""
    try:
        from tests.fast.dashboard.dummy_dump import dump_dummy_run
        from tests.fast.dashboard.dummy_telemetry import dump_dummy_telemetry
    except ImportError as e:
        raise SystemExit("--demo needs the dummy generators under tests/; run from a miles repo checkout") from e
    dump_dummy_run(target, steps=3, num_prompts=8, n_samples_per_prompt=4, max_response_len=48)
    dump_dummy_telemetry(target, steps=3, samples_per_step=32)
    return target


def main(argv: list[str] | None = None) -> None:
    # [FUNCTION DOC] Parses arguments (from the given list, or the process arguments when None),
    # builds the app and serves it forever; returns only when the server stops.
    # Steps: in demo mode create a temporary directory and fill it with demo data, otherwise
    # require the dump directory option; assert the directory exists. Load the metric store from
    # the "dashboard" subdirectory, create the dump reader (cache directory and tensor LRU from
    # options), build the app (follow flag, utilization-overview flag, low-MFU threshold), start
    # the follow thread if requested, then run uvicorn on the host and port.
    # Raises an assertion error when no dump directory is given outside demo mode or when it is
    # not a directory. The demo temporary directory is never cleaned up.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump-details", default=None, help="the run's --dump-details directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7788)
    parser.add_argument("--follow", action="store_true", help="tail telemetry streams of a still-running job")
    parser.add_argument("--tensor-lru", type=int, default=2, help="rollout steps kept resident in tensor memory")
    parser.add_argument("--cache-dir", default=None, help="summary cache dir (default: <dump>/dashboard/cache)")
    parser.add_argument("--demo", action="store_true", help="serve generated demo data (needs a repo checkout)")
    parser.add_argument(
        "--use-utilization-overview",
        action="store_true",
        default=False,
        help="always show the fleet utilization overview instead of the per-rank carpet "
        "(auto-enabled above 64 lanes)",
    )
    parser.add_argument(
        "--low-mfu-threshold",
        type=float,
        default=DEFAULT_LOW_MFU,
        help="warn when mean actor-train MFU falls below this fraction; 0 disables the rule. "
        "What counts as low is model- and config-dependent, so tune it to the run "
        "(default: %(default)s)",
    )
    args = parser.parse_args(argv)

    if args.demo:
        import tempfile

        dump_dir = make_demo_dir(Path(tempfile.mkdtemp(prefix="miles_dashboard_demo_")))
    else:
        assert args.dump_details is not None, "--dump-details is required (or use --demo)"
        dump_dir = Path(args.dump_details)
    assert dump_dir.is_dir(), f"--dump-details directory not found: {dump_dir}"

    store = MetricStore.load(dump_dir / "dashboard")
    reader = DumpReader(dump_dir, cache_dir=args.cache_dir, tensor_lru=args.tensor_lru)
    app = make_app(
        store,
        reader,
        follow=args.follow,
        use_utilization_overview=args.use_utilization_overview,
        low_mfu=args.low_mfu_threshold,
    )

    if args.follow:
        # races the threadpool handlers: list appends are GIL-atomic, partition caches lock
        def _tail() -> None:
            # [FUNCTION DOC] HELPER: Background loop of the follow thread: sleep for the follow
            # interval, then ask the store to read newly appended lines; never returns and does
            # not catch exceptions, so a corrupt-line error ends the thread silently.
            while True:
                time.sleep(FOLLOW_INTERVAL_SECONDS)
                store.follow()

        threading.Thread(target=_tail, daemon=True, name="dashboard-follow").start()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
