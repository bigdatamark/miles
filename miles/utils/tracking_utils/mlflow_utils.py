# [MODULE DOC] MLflow helpers used by the MLflow tracking backend: initialization of a primary run or
# attachment of a secondary process to the primary's run, metric logging with key sanitization and
# silent dropping of non-numeric values, and run finalization. Imports mlflow lazily. Mutates the
# args namespace (stores the run id) and global MLflow state.
"""
MLflow tracking backend for miles.

MLflow docs for future reference:
  - Tracking overview : https://mlflow.org/docs/latest/ml/tracking/
  - Python API        : https://mlflow.org/docs/latest/python_api/mlflow.html
  - Remote tracking   : https://mlflow.org/docs/latest/tracking/server.html
"""

from __future__ import annotations

import logging
import os
import re
from copy import deepcopy
from typing import Any

logger = logging.getLogger(__name__)


# Helpers/utils
def _sanitize_key(key: str) -> str:
    # [FUNCTION DOC] HELPER: Replaces every character other than ASCII letters, digits, underscore,
    # hyphen, period, slash and whitespace with an underscore.
    return re.sub(r"[^a-zA-Z0-9_\-./\s]", "_", key)


def _compute_config_for_logging(args) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Deep-copies the args namespace's attribute mapping, adds an environment
    # section holding only the SLURM job id variable when set, and returns the dotted-key flattening
    # with all values as strings.
    # Build a flat param dict from *args*, mirroring ``wandb_utils._compute_config_for_logging``.
    raw = deepcopy(args.__dict__)

    whitelist_env_vars = ["SLURM_JOB_ID"]
    raw["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return _flatten_dict(raw)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Recursively flattens nested mappings into one level, joining key paths with
    # the separator (default period); non-mapping values (including lists) are converted to strings.
    # Recursively flatten nested dicts into ``dotted.key`` → ``str(value)`` pairs.
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def init_mlflow(args, *, primary: bool = True, **kwargs) -> None:
    # [FUNCTION DOC] If the use-mlflow argument is falsy, sets the stored run id argument to none and
    # returns. Otherwise sets the tracking URI (argument, else the MLFLOW_TRACKING_URI environment
    # variable, else the library default), selects the experiment named by the argument (created if
    # missing), and starts the primary run or attaches the secondary one by the primary flag.
    if not args.use_mlflow:
        args.mlflow_run_id = None
        return

    import mlflow

    tracking_uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", tracking_uri)

    experiment_name = args.mlflow_experiment_name
    mlflow.set_experiment(experiment_name)

    if primary:
        _init_mlflow_primary(args, experiment_name)
    else:
        _init_mlflow_secondary(args)


def _init_mlflow_primary(args, experiment_name: str) -> None:
    # [FUNCTION DOC] HELPER: Starts a run named by the run-name argument, falling back to the wandb
    # group argument; tags it with the SLURM job id (when set) and the rank argument as a string; logs
    # the flattened args as parameters; stores the new run id on the args namespace and logs it.
    import mlflow

    run_name = args.mlflow_run_name or args.wandb_group

    tags = {}
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        tags["slurm_job_id"] = slurm_job_id
    tags["rank"] = str(args.rank)

    run = mlflow.start_run(run_name=run_name, tags=tags)
    mlflow.log_params(_compute_config_for_logging(args))

    args.mlflow_run_id = run.info.run_id
    logger.info("MLflow run started: %s (experiment=%s, name=%s)", run.info.run_id, experiment_name, run_name)


def _init_mlflow_secondary(args) -> None:
    # [FUNCTION DOC] HELPER: Takes the run id from the args namespace, else the MLFLOW_RUN_ID
    # environment variable; if neither exists returns without attaching; otherwise starts (resumes)
    # that run in this process.
    """Attach to an existing MLflow run created by the primary rank."""
    import mlflow

    run_id = args.mlflow_run_id or os.environ.get("MLFLOW_RUN_ID")
    if run_id is None:
        return

    mlflow.start_run(run_id=run_id)
    logger.info("MLflow secondary attached to run: %s", run_id)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    # [FUNCTION DOC] Does nothing without an active run. Otherwise sanitizes each key and converts each
    # value to a float, silently skipping values that cannot be converted; if any remain, logs them in
    # one call with the step converted to an integer (or none). Colliding sanitized keys overwrite.
    import mlflow

    if mlflow.active_run() is None:
        return

    sanitized: dict[str, float] = {}
    for k, v in metrics.items():
        try:
            sanitized[_sanitize_key(k)] = float(v)
        except (TypeError, ValueError):
            continue

    if sanitized:
        mlflow.log_metrics(sanitized, step=int(step) if step is not None else None)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def finish() -> None:
    # [FUNCTION DOC] If a run is active, ends it and logs its id; otherwise does nothing.
    import mlflow

    if mlflow.active_run() is None:
        return

    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()
    logger.info("MLflow run ended: %s", run_id)
