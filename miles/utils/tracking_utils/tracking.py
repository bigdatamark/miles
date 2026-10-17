# [MODULE DOC] Public tracking entry points over a single process-wide manager. Defines the registry of
# all backends (name to backend class and the argument flag enabling it) and thin functions to
# initialize backends, declare step-key metric groups, log metrics (also mirroring them to the audit
# event logger when it is initialized) and finish. The registry lives here, not in the base module, to
# avoid a circular import.
import logging

import torch

from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import MetricEvent

from .base import (
    MilesDashboardBackend,
    MlflowBackend,
    PrometheusBackend,
    TensorboardBackend,
    TrackingBackend,
    TrackingManager,
    WandbBackend,
)
from .ci_history import CiHistoryBackend

# The full registry lives here, not base.py: base must never import a backend
# module (ci_history imports TrackingBackend from base -> circular). This
# module is the tracking entry point and the one place that imports every
# backend, so it owns the registry.
BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
    "prometheus": (PrometheusBackend, "use_prometheus"),
    "ci_history": (CiHistoryBackend, "ci_enable_metrics_capture"),
    "miles_dashboard": (MilesDashboardBackend, "use_miles_dashboard"),
}

logger = logging.getLogger(__name__)
_manager = TrackingManager(BACKEND_REGISTRY)


def init_tracking(args, primary: bool = True, **kwargs):
    # [FUNCTION DOC] Initializes every backend whose enabling flag is truthy on the args (see the
    # manager); extra keywords go to each backend's init.
    _manager.init(args, primary=primary, **kwargs)


def define_step_key_metric_group(prefix: str, step_key: str) -> None:
    # [FUNCTION DOC] Forwards the grouping declaration to all active backends.
    """Declare a metric group plotted against its own step key (e.g. ``{name}/*`` vs ``{name}/step``).
    Only wandb acts on this; must be called from the primary tracking process or definitions may be lost."""
    _manager.define_step_key_metric_group(prefix, step_key)


def log(args, metrics, step_key: str):
    # [FUNCTION DOC] Takes the step as the metrics mapping's value under the step key (none if absent)
    # and logs the mapping to all backends. If the audit event logger is initialized, also logs a
    # metric event holding the mapping with tensor values converted to Python numbers via their item
    # method (so multi-element tensors raise), without printing. The args parameter is unused.
    step = metrics.get(step_key)
    _manager.log(metrics, step=step, step_key=step_key)

    if is_event_logger_initialized():
        serializable_metrics = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in metrics.items()}
        get_event_logger().log(MetricEvent, {"metrics": serializable_metrics}, print_log=False)


def finish_tracking():
    # [FUNCTION DOC] Finishes all active backends (errors are logged, not raised) and clears them.
    _manager.finish()
