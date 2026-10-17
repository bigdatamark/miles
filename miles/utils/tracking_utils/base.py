# [MODULE DOC] Backend interface and built-in adapters for experiment tracking, plus the manager that
# initializes the enabled backends and fans out log, step-key grouping and finish calls. Backends are
# selected by a registry mapping name to (backend class, argument flag name). Heavy third-party
# imports are done lazily inside methods.
"""
Shared tracking interface for experiment logging backends.

Each backend implements ``init / log / finish``, and :class:`TrackingManager` fans out
calls to every active backend.

To add a new backend:
--------------------
1. Subclass :class:`TrackingBackend`.
2. Add it to ``BACKEND_REGISTRY`` in ``tracking.py`` (not base --
   importing a backend module into base would be circular).
3. Add a corresponding ``--use-<name>`` CLI flag in ``arguments.py``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from miles.utils.workers.types import ClusterBackend

logger = logging.getLogger(__name__)


class TrackingBackend(ABC):
    # [CLASS DOC] Abstract interface every logging backend must satisfy: init, log and finish, plus an
    # optional support check and an optional step-key grouping hook.
    # Interface every logging backend must satisfy.

    @classmethod
    def is_supported(cls, args) -> bool:
        # [FUNCTION DOC] Whether the backend can run given the parsed arguments; the default says yes.
        return True

    @abstractmethod
    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] Abstract: prepare the backend. The primary flag distinguishes the main process
        # that creates the run from secondary processes that attach to it; extra keywords are backend
        # specific.
        ...

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        # [FUNCTION DOC] Abstract: record a mapping of metric name to value, with an optional step.
        # Callers also pass a step-key keyword naming the metric that carries the step.
        ...

    @abstractmethod
    def finish(self) -> None:
        # [FUNCTION DOC] Abstract: flush and close the backend.
        ...

    def define_step_key_metric_group(self, prefix: str, step_key: str) -> None:
        # [FUNCTION DOC] Optional hook; the default does nothing.
        """Declare that ``{prefix}/*`` metrics plot against ``step_key``; no-op for
        backends that take the step numerically on every log call."""
        return


# Thin adapters for backwards compatibility to keep wandb_utils and tensorboard_utils untouched.
class WandbBackend(TrackingBackend):
    # [CLASS DOC] Adapter over the Weights and Biases helper module. Remembers which (prefix, step key)
    # groups were already defined.
    # Delegates to the existing ``wandb_utils`` helpers.

    def __init__(self) -> None:
        # [FUNCTION DOC] Starts with no defined step-key groups.
        self._defined_step_key_groups: set[tuple[str, str]] = set()

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] Calls the primary or secondary wandb initializer (by the primary flag) with the
        # args and extra keywords.
        from . import wandb_utils

        if primary:
            wandb_utils.init_wandb_primary(args, **kwargs)
        else:
            wandb_utils.init_wandb_secondary(args, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        # [FUNCTION DOC] Logs the metrics mapping to the active wandb run. The step argument is ignored
        # (the step comes from a metric inside the mapping, defined via step-key grouping).
        import wandb

        wandb.log(metrics)

    def define_step_key_metric_group(self, prefix: str, step_key: str) -> None:
        # [FUNCTION DOC] Once per (prefix, step key) pair: defines the step key as a wandb metric and
        # makes every metric under the prefix plot against it. Repeated calls are no-ops.
        # Call from the primary tracking process: definitions from secondary shared-mode writers are lost.
        if (prefix, step_key) in self._defined_step_key_groups:
            return
        import wandb

        wandb.define_metric(step_key)
        wandb.define_metric(f"{prefix}/*", step_metric=step_key)
        self._defined_step_key_groups.add((prefix, step_key))

    def finish(self) -> None:
        # [FUNCTION DOC] Finishes the wandb run.
        import wandb

        wandb.finish()


class TensorboardBackend(TrackingBackend):
    # [CLASS DOC] Adapter over the TensorBoard writer wrapper; does nothing until initialized.
    def __init__(self) -> None:
        # [FUNCTION DOC] Starts with no adapter.
        self._adapter = None

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] Creates the TensorBoard adapter from the args (the primary flag is ignored, so
        # every process creates its own writer).
        from .tensorboard_utils import _TensorboardAdapter

        self._adapter = _TensorboardAdapter(args)

    def log(self, metrics: dict[str, Any], step: int | None = None, *, step_key: str | None = None, **kwargs) -> None:
        # [FUNCTION DOC] If initialized, drops the entry whose key exactly equals the step key and logs
        # the rest with the explicit step.
        if self._adapter is not None:
            # Strip the caller's exact step-key entry (e.g. "train/step",
            # "rollout/step") — tensorboard receives step as an explicit
            # argument instead. Matching by exact key rather than endswith
            # avoids dropping user metrics that happen to end in "/step".
            data = {k: v for k, v in metrics.items() if k != step_key}
            self._adapter.log(data=data, step=step)

    def finish(self) -> None:
        # [FUNCTION DOC] Finishes the adapter if one exists.
        if self._adapter is not None:
            self._adapter.finish()


class MlflowBackend(TrackingBackend):
    # [CLASS DOC] Adapter over the MLflow helper module.

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] Delegates to the MLflow initializer with the primary flag.
        from . import mlflow_utils

        mlflow_utils.init_mlflow(args, primary=primary, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        # [FUNCTION DOC] Delegates to the MLflow metric logger with the step.
        from . import mlflow_utils

        mlflow_utils.log_metrics(metrics, step=step)

    def finish(self) -> None:
        # [FUNCTION DOC] Ends the active MLflow run via the helper.
        from . import mlflow_utils

        mlflow_utils.finish()


class PrometheusBackend(TrackingBackend):
    # [CLASS DOC] Adapter over the Ray-actor-based Prometheus collector.
    # Wraps the existing Ray-actor based prometheus collector. The actor lifetime is
    # tied to the Ray job, so finish() is intentionally a no-op.

    @classmethod
    def is_supported(cls, args) -> bool:
        # [FUNCTION DOC] Supported on every cluster backend except Kubernetes.
        return ClusterBackend(args.cluster_backend) is not ClusterBackend.KUBERNETES

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] Initializes the collector, starting the metrics server only when primary.
        from .prometheus_utils import init_prometheus

        init_prometheus(args, start_server=primary)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        # [FUNCTION DOC] Looks up the collector actor handle, asserts it exists (assertion error with
        # guidance otherwise) and sends the metrics mapping to its update method as a fire-and-forget
        # remote call. Step is ignored.
        from .prometheus_utils import get_prometheus

        prom = get_prometheus()
        assert prom is not None, (
            "Prometheus collector is not initialized; ensure init_tracking(..., primary=...) ran on the "
            "driver and workers can resolve the miles_prometheus_collector Ray actor."
        )
        prom.update.remote(metrics)

    def finish(self) -> None:
        # [FUNCTION DOC] Does nothing (the actor lives as long as the Ray job).
        return


class MilesDashboardBackend(TrackingBackend):
    # [CLASS DOC] Adapter over the live dashboard package.
    # Live dashboard collection (see miles/dashboard). Lazy imports keep the
    # dashboard package out of processes that never enable it.

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] Delegates to the dashboard initializer with the primary flag.
        from miles.dashboard.backend import init_dashboard

        init_dashboard(args, primary=primary, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None, *, step_key: str | None = None, **kwargs) -> None:
        # [FUNCTION DOC] Delegates to the dashboard logger with step and step key.
        from miles.dashboard.backend import dashboard_log

        dashboard_log(metrics, step=step, step_key=step_key)

    def finish(self) -> None:
        # [FUNCTION DOC] Delegates to the dashboard finisher.
        from miles.dashboard.backend import finish_dashboard

        finish_dashboard()


class TrackingManager:
    # [CLASS DOC] Owns the list of active backends built from a registry mapping backend name to
    # (backend class, argument flag name). Not thread-safe.
    # Initializes and logs to every enabled backend; used internally by ``tracking_utils``.

    def __init__(self, registry: dict[str, tuple[type[TrackingBackend], str]]) -> None:
        # [FUNCTION DOC] Stores the registry and starts with no active backends.
        self._backends: list[TrackingBackend] = []
        self._registry = registry

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        # [FUNCTION DOC] For each registry entry in order: skips it if the args lack a truthy value for
        # its flag; skips it with a warning if the class reports unsupported; otherwise instantiates
        # the backend, initializes it with the primary flag and keywords, and appends it. Backend init
        # errors propagate, leaving earlier backends active. Calling twice adds duplicates.
        for name, (cls, flag) in self._registry.items():
            if not getattr(args, flag, False):
                continue
            if not cls.is_supported(args):
                logger.warning("Skipping tracking backend %s: unsupported on this cluster backend", name)
                continue
            logger.info("Initialising tracking backend: %s", name)
            backend = cls()
            backend.init(args, primary=primary, **kwargs)
            self._backends.append(backend)

    def log(self, metrics: dict[str, Any], step: int | None = None, step_key: str | None = None) -> None:
        # [FUNCTION DOC] Forwards the metrics with step and step key to each active backend in order;
        # an exception from one backend stops the rest.
        for backend in self._backends:
            backend.log(metrics, step=step, step_key=step_key)

    def define_step_key_metric_group(self, prefix: str, step_key: str) -> None:
        # [FUNCTION DOC] Forwards the grouping declaration to each active backend.
        for backend in self._backends:
            backend.define_step_key_metric_group(prefix, step_key)

    def finish(self) -> None:
        # [FUNCTION DOC] Finishes every active backend, logging (not raising) any exception per backend,
        # then clears the active list.
        for backend in self._backends:
            try:
                backend.finish()
            except Exception:
                logger.exception(
                    "Error finishing tracking backend %s",
                    type(backend).__name__,
                )
        self._backends.clear()
