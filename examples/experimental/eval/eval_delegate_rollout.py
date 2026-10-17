# [MODULE DOC]
# Rollout function plug-in that adds external (delegated) evaluation to the evaluation phase.
# During evaluation it builds, and caches by evaluation-config file path and modification time, an
# aggregate delegate client from the evaluation config named in the launch args, calls the
# external servers, flattens their numeric metrics under an "eval/" key prefix and logs them, and
# then also runs the built-in sglang evaluation rollout, whose output is returned with its
# metrics replaced by the external ones. Training rollouts are not supported. A module-level
# cache maps the expanded config path to a pair of modification time and client (or none).
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from examples.experimental.eval.eval_delegate import EvalDelegateClient, _rebuild_delegate_config
from omegaconf import OmegaConf

from miles.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.sglang_rollout import generate_rollout as base_generate_rollout
from miles.utils.file_arg_utils import PSEUDO_FILE_PREFIX, resolve_file_arg

logger = logging.getLogger(__name__)

_DELEGATE_CACHE: dict[str, tuple[float | None, EvalDelegateClient | None]] = {}


def generate_rollout(
    args, rollout_id: int, data_buffer: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    # [FUNCTION DOC] Rollout entry point (launch args, rollout id, data buffer, evaluation flag).
    # Asserts the evaluation flag is true (assertion error for training). Gets the delegate client;
    # if one exists, runs its evaluation for the rollout id, logs the metrics and raw response and
    # keeps the flattened metrics. Then runs the standard sglang rollout function with the same
    # arguments and returns its result after overwriting the result's metrics attribute with the
    # flattened external metrics (an empty mapping if no client).
    # KNOWN ISSUE: any metrics produced by the standard evaluation rollout are discarded.
    assert evaluation, "Delegate rollout is only supported for evaluation"
    flattened_metrics: dict[str, float] = {}

    client = _get_delegate_client(args)
    if client is not None:
        metrics, raw_response = client.evaluate(args, rollout_id)
        flattened_metrics = _log_delegate_metrics(args, rollout_id, metrics, raw_response)

    result = base_generate_rollout(args, rollout_id, data_buffer, evaluation=evaluation)
    result.metrics = flattened_metrics
    return result


def _get_delegate_client(args) -> EvalDelegateClient | None:
    # [FUNCTION DOC] HELPER: Returns the delegate client for the eval config of the launch args, or
    # none if no config is set. A config value with the pseudo-file prefix (inline content) is
    # built every call without caching. Otherwise the path is user-expanded and a cached client is
    # reused while the file's modification time is unchanged (a missing file gives a none time);
    # else the config is read, a client built and cached together with the time. Mutates the
    # module cache.
    config_path = getattr(args, "eval_config", None)
    if not config_path:
        return None

    if config_path.startswith(PSEUDO_FILE_PREFIX):
        return _build_delegate_client(args, resolve_file_arg(config_path))

    config_path = str(Path(config_path).expanduser())
    cache_entry = _DELEGATE_CACHE.get(config_path)
    mtime = _safe_mtime(config_path)
    if cache_entry and cache_entry[0] == mtime:
        return cache_entry[1]

    client = _build_delegate_client(args, resolve_file_arg(config_path))
    _DELEGATE_CACHE[config_path] = (mtime, client)
    return client


def _build_delegate_client(args, config_text: str) -> EvalDelegateClient | None:
    # [FUNCTION DOC] HELPER: Parses YAML config text with OmegaConf into a plain container with
    # interpolations resolved. Returns none (with a logged warning) if the root is not a mapping or
    # its eval key (or, if absent, the root itself) is not a mapping. Defaults are the defaults
    # mapping and delegates the delegate list under that section (each missing or empty treated as
    # empty). Converts delegate entries to environment records; if none, logs an info message and
    # returns none; otherwise returns the aggregate client built from them.
    cfg = OmegaConf.create(config_text)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        logger.warning("--eval-config must contain a mapping at the root.")
        return None

    eval_cfg = cfg_dict.get("eval", cfg_dict)
    if not isinstance(eval_cfg, dict):
        logger.warning("--eval-config must define an `eval` mapping or be a mapping itself.")
        return None

    defaults = dict(eval_cfg.get("defaults") or {})
    delegate_entries = list(eval_cfg.get("delegate") or [])
    env_configs = _rebuild_delegate_config(args, delegate_entries, defaults)
    if not env_configs:
        logger.info("No delegate environments configured under `eval.delegate`; skipping external eval.")
        return None

    return EvalDelegateClient.maybe_create(args, env_configs=env_configs)


def _safe_mtime(path: str) -> float | None:
    # [FUNCTION DOC] HELPER: Returns the file's modification time, or none if it cannot be
    # obtained because of an operating system error (such as a missing file).
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _log_delegate_metrics(args, rollout_id: int, metrics: dict | None, raw_response: dict | None) -> dict:
    # [FUNCTION DOC] HELPER: Flattens the metrics, logs the raw response (if not none) and the
    # flattened metrics at info level, and returns the flattened mapping.
    # KNOWN ISSUE: the args parameter is unused.
    flattened = _flatten_metrics(metrics)
    if raw_response is not None:
        logger.info("External eval raw response for rollout %s: %s", rollout_id, raw_response)
    logger.info("eval %s (external): %s", rollout_id, flattened)
    return flattened


def _flatten_metrics(metric_source: dict | None) -> dict:
    # [FUNCTION DOC] HELPER: Converts a possibly nested metrics mapping into a flat mapping of
    # slash-joined keys to floats. Returns empty if the input is not a mapping. Entries whose value
    # is not an integer or float are dropped (booleans count as integers). Keys not already
    # starting with the eval prefix get that prefix prepended.
    flattened_metrics: dict[str, float] = {}
    if not isinstance(metric_source, dict):
        return flattened_metrics
    for key, value in _iter_metric_items(metric_source):
        if not isinstance(value, (int, float)):
            continue
        metric_key = key if key.startswith("eval/") else f"eval/{key}"
        flattened_metrics[metric_key] = float(value)
    return flattened_metrics


def _iter_metric_items(metrics: dict, prefix: str = ""):
    # [FUNCTION DOC] HELPER: Generator yielding (key path, value) pairs for every leaf of a nested
    # mapping, where the key path joins nested keys with slashes under the optional prefix.
    for key, value in metrics.items():
        metric_key = f"{key}" if not prefix else f"{prefix}/{key}"
        if isinstance(value, dict):
            yield from _iter_metric_items(value, metric_key)
        else:
            yield metric_key, value
