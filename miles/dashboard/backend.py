# [MODULE DOC]
# Process-local state and lifecycle for the dashboard. Module globals hold the collector actor handle,
# whether this process is the primary (driver) one, a flag recording that resolving the named actor
# already failed, and a rate-limited warner. Functions: init (create or find the collector), resolve
# (look up the named actor with a timeout), non-blocking access, metric push, and shutdown. All remote
# calls are best effort so telemetry problems never crash training.
"""Dashboard tracking-backend internals and Ray lifecycle glue.

Call matrix of ``init_dashboard`` (design doc §6.1), reached through the
``MilesDashboardBackend`` adapter in ``tracking_utils``:

- ``train.py`` driver (``primary=True``): builds the config, creates the
  named ``DashboardCollector`` actor pinned to the driver node (same pattern
  as ``prometheus_utils``), starts its flush loop, and spawns one
  ``GpuSampler`` actor per GPU node.
- train actor main rank (``primary=False``): resolves the named actor so
  ``dashboard_log`` works there.
- rollout manager (``primary=False, router_addr=...``): additionally attaches
  a Timer phase sink for ``rollout`` / ``eval_rollout`` events. The router
  itself is registered later through ``hooks.register_router`` — at
  init_tracking time the router has not started yet.

All pushes are fire-and-forget; when the collector cannot be resolved the
module warns once and every later call is a no-op — the dashboard must never
take a training run down with it.
"""

from __future__ import annotations

import logging
import time

from miles.dashboard import hooks
from miles.dashboard.args import collector_config_from_args
from miles.dashboard.collector import COLLECTOR_ACTOR_NAME
from miles.dashboard.logging_utils import RateLimitedWarner
from miles.dashboard.store import MetricsRecord, Role

logger = logging.getLogger(__name__)

GET_ACTOR_TIMEOUT_SECONDS = 60.0
GET_ACTOR_INTERVAL_SECONDS = 2.0

_handle = None
_is_primary = False
_resolution_failed = False
_warner = RateLimitedWarner(logger)


def init_dashboard(args, *, primary: bool = True, router_addr: str | None = None, **kwargs) -> None:
    # [FUNCTION DOC] Primary (driver) path: marks the process primary; builds the collector
    # configuration (start timestamp = now); creates the named collector Ray actor hard-pinned (no
    # soft fallback) to the driver's node, passing a Prometheus handle factory only when forwarding
    # is enabled; blocks on a ping to ensure it is alive; fires the non-blocking start call; logs where
    # telemetry goes and how to view it. Non-primary path: resolves the named actor (waiting up to the
    # timeout); if unavailable returns quietly (telemetry disabled); if router_addr was given
    # (marking the rollout manager process) attaches the phase sink for the rollout manager role and
    # the trajectory sink. The router address value itself is not used because the router is not
    # started yet; it registers later via the hooks module. Extra keyword arguments are ignored.
    global _handle, _is_primary
    import ray

    if primary:
        _is_primary = True
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from miles.dashboard.collector import DashboardCollector

        config = collector_config_from_args(args, start_ts=time.time())
        _handle = (
            ray.remote(DashboardCollector)
            .options(
                name=COLLECTOR_ACTOR_NAME,
                # driver node, hard-pinned: the dump volume and the run's other
                # observability live there, and driver death ends the job anyway
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ray.get_runtime_context().get_node_id(), soft=False
                ),
            )
            .remote(
                config=config, prometheus_handle_factory=_prometheus_factory if config.forward_prometheus else None
            )
        )
        ray.get(_handle.ping.remote())
        _handle.start.remote()
        logger.info(
            "miles dashboard: telemetry -> %s | view live: python -m miles.dashboard.serve "
            "--dump-details %s --follow --port 7788",
            config.dashboard_dir,
            args.dump_details,
        )
        return

    if resolve_collector() is None:
        return
    if router_addr is not None:
        # the kwarg marks the rollout-manager process. It cannot be used for
        # scraping: init_tracking runs before start_rollout_servers, so the
        # address is still "http://None:None" here — the real registration
        # happens via hooks.register_router once the router is up.
        hooks.attach_phase_sink(_handle, Role.ROLLOUT_MANAGER)
        hooks.attach_trajectory_sink(_handle)


def resolve_collector():
    # [FUNCTION DOC] Returns the cached handle if present, or none if resolution already failed
    # (sticky). Otherwise polls the Ray named-actor lookup every 2 seconds until found (cached and
    # returned) or 60 seconds pass, in which case logs one warning, sets the failure flag and returns
    # none. Blocks (sleeps) while waiting.
    """This process's collector handle, resolving the named actor on first
    use (train ranks never run init_dashboard themselves). Returns None —
    permanently, after one warning — when the actor cannot be found."""
    global _handle, _resolution_failed
    if _handle is not None or _resolution_failed:
        return _handle
    import ray

    deadline = time.monotonic() + GET_ACTOR_TIMEOUT_SECONDS
    while True:
        try:
            _handle = ray.get_actor(COLLECTOR_ACTOR_NAME)
            return _handle
        except ValueError:
            if time.monotonic() >= deadline:
                logger.warning(
                    "dashboard collector actor not found after %.0fs; telemetry from this process is disabled",
                    GET_ACTOR_TIMEOUT_SECONDS,
                )
                _resolution_failed = True
                return None
            time.sleep(GET_ACTOR_INTERVAL_SECONDS)


def current_collector():
    # [FUNCTION DOC] Returns the cached handle or none; never waits or resolves.
    """The already-resolved handle, or None. Never blocks (hot paths)."""
    return _handle


def dashboard_log(metrics: dict, *, step: int | None = None, step_key: str | None = None) -> None:
    # [FUNCTION DOC] Fire-and-forget push of one metrics record (timestamp now, optional step key and
    # step, only scalar entries of the mapping) to the collector. Does nothing if there is no handle.
    # Any exception while sending is swallowed and reported through the rate-limited warner.
    if _handle is None:
        return
    try:
        _handle.push_metrics.remote(
            MetricsRecord(ts=time.time(), step_key=step_key, step=step, metrics=_scalars_only(metrics))
        )
    except Exception:
        _warner.warn("dashboard metric push failed; dropping this log call")


def finish_dashboard() -> None:
    # [FUNCTION DOC] Detaches the hooks and flushes them. If this is the primary process with a
    # handle: synchronously asks the collector to shut down (30 second timeout; failures are logged as
    # warnings with traceback) so the final flush lands, then kills the actor. Always resets the
    # handle, primary flag and resolution-failed flag.
    global _handle, _is_primary, _resolution_failed
    hooks.detach_and_flush()
    if _handle is not None and _is_primary:
        import ray

        try:
            # synchronous: the final flush must land before the driver exits
            ray.get(_handle.shutdown.remote(), timeout=30)
        except Exception:
            logger.warning("dashboard collector shutdown incomplete", exc_info=True)
        ray.kill(_handle)
    _handle = None
    _is_primary = False
    _resolution_failed = False


# ------------------------------ ray helpers ---------------------------------


def _prometheus_factory():
    # [FUNCTION DOC] HELPER: Runs inside the collector actor. Looks up the named Prometheus collector
    # actor and returns its handle, or none if it does not exist.
    """Resolves the miles prometheus collector from inside the dashboard
    collector actor (its module-global handle is per-process)."""
    import ray

    from miles.utils.tracking_utils.prometheus_utils import _COLLECTOR_ACTOR_NAME

    try:
        return ray.get_actor(_COLLECTOR_ACTOR_NAME)
    except ValueError:
        return None


def _scalars_only(metrics: dict) -> dict:
    # [FUNCTION DOC] HELPER: Keeps int, float, string and bool values as is; converts other values
    # whose type defines float conversion (numpy or torch scalars) to float; drops everything else.
    # Note booleans pass through the first branch. A multi-element tensor with a float conversion
    # method would raise on conversion (not caught here).
    """The JSONL store persists plain scalars; tracking payloads can carry
    tensors or numpy values, which are converted or dropped here."""
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)):
            out[key] = value
        elif hasattr(type(value), "__float__"):  # numpy/torch zero-dim scalars
            out[key] = float(value)
    return out
