# [MODULE DOC] Prometheus metric export through a named Ray actor. The driver creates the actor, pinned
# to the driver node, which starts the metrics HTTP server and keeps one labeled gauge per sanitized
# metric name; other processes look the actor up by name and push metric mappings to it through
# remote calls. Holds a module-global handle to the actor (none when unavailable).
import logging
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)

_METRIC_PREFIX = "miles_metric_"
_COLLECTOR_ACTOR_NAME = "miles_prometheus_collector"
_GET_ACTOR_TIMEOUT = 60
_GET_ACTOR_INTERVAL = 2

_collector_handle = None


def init_prometheus(args, start_server: bool = False):
    # [FUNCTION DOC] Sets the module-global collector handle. With start_server true: creates the named
    # actor with a hard node-affinity to the current node (so the HTTP server binds on the driver node),
    # passing the args, blocks on a ping to confirm it is up, and logs. With start_server false:
    # polls for the named actor every 2 seconds for up to 60 seconds; if not found by the deadline logs
    # a warning and returns leaving the handle unset (metrics are then silently not reported, though
    # the backend's log method asserts). Returns nothing.
    """Initialize the Prometheus metric collector.

    The driver process (``start_server=True``) creates a named Ray
    actor that holds the HTTP server and all gauges.  Actor processes
    (``start_server=False``) look up the existing named actor.  Ray
    remote calls transport metrics across nodes transparently.
    """
    global _collector_handle

    if start_server:
        current_node_id = ray.get_runtime_context().get_node_id()
        _collector_handle = (
            ray.remote(_PrometheusCollector)
            .options(
                name=_COLLECTOR_ACTOR_NAME,
                # Hard-pin: the HTTP server must bind on the driver node where
                # the Prometheus scraper expects it.  Driver node failure kills
                # the entire Ray job anyway, so soft fallback adds no value.
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=current_node_id,
                    soft=False,
                ),
            )
            .remote(args=args)
        )
        ray.get(_collector_handle.ping.remote())
        logger.info("Prometheus collector actor created")
    else:
        deadline = time.monotonic() + _GET_ACTOR_TIMEOUT
        while True:
            try:
                _collector_handle = ray.get_actor(_COLLECTOR_ACTOR_NAME)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    logger.warning(
                        "Prometheus collector actor not found "
                        f"after {_GET_ACTOR_TIMEOUT}s, "
                        "metrics will not be reported"
                    )
                    return
                time.sleep(_GET_ACTOR_INTERVAL)


def get_prometheus():
    # [FUNCTION DOC] Returns the module-global collector actor handle, or none if not initialized.
    """Return the collector actor handle, or ``None``."""
    return _collector_handle


class _PrometheusCollector:
    # [CLASS DOC] HELPER: Actor body holding the gauge registry (sanitized name to gauge) and the
    # constant label (run name). Wrapped into a Ray actor by the init function.
    """Ray actor that owns the Prometheus HTTP server and gauges.

    Runs on the driver node.  All processes push metrics here via
    ``handle.update.remote(metrics)`` — works across nodes because
    Ray handles the RPC transparently.
    """

    def __init__(self, *, args):
        # [FUNCTION DOC] HELPER: Imports the Prometheus client lazily, keeps the gauge class, picks the
        # run name (prometheus run name argument, else wandb group, else "miles_training"), builds the
        # single run_name label, starts the Prometheus HTTP server on the port argument (an OS error
        # propagates if the port is taken) and logs the bind address and run name.
        from prometheus_client import Gauge, start_http_server

        self._Gauge = Gauge
        self._gauges: dict = {}
        self._run_name = (
            getattr(args, "prometheus_run_name", None) or getattr(args, "wandb_group", None) or "miles_training"
        )
        self._label_keys = ["run_name"]
        self._label_vals = [self._run_name]

        port = args.prometheus_port
        start_http_server(port)

        bind_ip = get_current_node_ip()
        logger.info("Prometheus metrics server started on %s:%d, run_name=%s", bind_ip, port, self._run_name)

    def update(self, metrics: dict):
        # [FUNCTION DOC] HELPER: For each integer or float value (booleans pass since they are
        # integers; other types skipped), derives the gauge name as the metric prefix plus the key with
        # slashes and hyphens turned into underscores and at-signs turned into "_at_"; creates the
        # gauge (help text is the original key, one run_name label) on first use and sets its value for
        # the run label. Distinct keys that sanitize equally share one gauge.
        """Set gauge values for all numeric metrics."""
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            safe_key = _METRIC_PREFIX + (key.replace("/", "_").replace("-", "_").replace("@", "_at_"))
            if safe_key not in self._gauges:
                self._gauges[safe_key] = self._Gauge(
                    safe_key,
                    key,
                    self._label_keys,
                )
            self._gauges[safe_key].labels(*self._label_vals).set(value)

    def ping(self):
        # [FUNCTION DOC] HELPER: Liveness check used to wait for actor construction; returns true.
        return True
