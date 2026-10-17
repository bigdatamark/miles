# [MODULE DOC]
# Generic layer for delegating evaluation to external evaluation servers. Defines structured
# configuration records (per-dataset generation parameters and per-environment connection
# settings) parsed from a configuration mapping through OmegaConf, a base class for
# environment-specific clients, and an aggregating client that creates the configured clients
# (currently only the "skills" environment), runs them in order and merges their metrics. The
# router address of the rollout inference router is derived from launch args. Only the skills
# environment is registered; any other delegate name is an error at configuration time.
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from omegaconf import OmegaConf
from miles.utils.eval_config import DATASET_RUNTIME_SPECS, _apply_dataset_field_overrides

logger = logging.getLogger(__name__)


@dataclass
class EvalEnvDatasetConfig:
    # [CLASS DOC]
    # Structured per-dataset record of generation parameters sent to a delegate: dataset name
    # (empty by default), samples per eval prompt, temperature, top-p, top-k and maximum response
    # length (all optional, none by default). The class-level name tuple and spec table (taken from
    # the shared dataset runtime specs) are plain class attributes, not dataclass fields, and drive
    # how launch-arg defaults are applied to dataset overrides.
    """Dataset-level generation parameters shared across delegate clients."""

    name: str = ""
    n_samples_per_eval_prompt: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_len: int | None = None

    FIELD_NAMES = ("n_samples_per_eval_prompt", "temperature", "top_p", "top_k", "max_response_len")
    FIELD_SPECS = {field: DATASET_RUNTIME_SPECS[field] for field in FIELD_NAMES}

    @classmethod
    def parse(cls, args, dataset_cfg: Mapping[str, Any], defaults: Mapping[str, Any]) -> "EvalEnvDatasetConfig":
        # [FUNCTION DOC] Class method building a dataset record from a raw dataset mapping.
        # Treats missing defaults as empty. The name (trimmed text) must be non-empty and must not
        # contain a colon, else a value error. Calls the shared dataset-override helper with the
        # launch args, the dataset mapping, the defaults and the field spec table, which resolves
        # missing fields from defaults or launch args (the helper may mutate the dataset mapping).
        # Then merges the mapping onto a structured default of this class with OmegaConf (type
        # coercion and unknown-key rejection come from OmegaConf) and converts back to an object,
        # rebuilding through the constructor if conversion returned a plain mapping. Returns the
        # record.
        """Merge dataset overrides with defaults/CLI settings and coerce types via OmegaConf."""
        defaults = defaults or {}
        name = str(dataset_cfg.get("name", "")).strip()
        if not name:
            raise ValueError("Each delegate dataset entry must include a non-empty `name`.")
        if ":" in name:
            raise ValueError(
                "Colon in dataset name is not allowed; use `n_samples_per_eval_prompt` to configure samples per prompt."
            )

        _apply_dataset_field_overrides(args, dataset_cfg, defaults, cls.FIELD_SPECS)

        cfg = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(dataset_cfg))
        obj = OmegaConf.to_object(cfg)
        if not isinstance(obj, cls):
            obj = cls(**obj)
        return obj

    def to_payload(self) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a mapping of every dataclass field that is not none to its value
        # (so unset parameters are omitted), suitable for JSON.
        """Return a JSON-serializable payload for this dataset configuration."""
        payload: dict[str, Any] = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if value is None:
                continue
            payload[field_info.name] = value
        return payload


@dataclass
class EvalEnvConfig:
    # [CLASS DOC]
    # Structured record describing one external evaluation environment: name (empty default), URL
    # of the evaluation server (optional), request timeout in seconds (default 3600), maximum
    # retries (default one), extra HTTP headers (mapping, empty default) and per-environment
    # default generation parameters (mapping, empty default).
    """Environment definition shared across delegate implementations."""

    name: str = ""
    url: str | None = None
    timeout_secs: int = 3600
    max_retries: int = 1
    headers: dict[str, Any] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any], defaults: Mapping[str, Any]) -> "EvalEnvConfig":
        # [FUNCTION DOC] Class method merging the raw mapping (none treated as empty) onto a
        # structured default of this class with OmegaConf and returning the resulting record
        # (rebuilt through the constructor if OmegaConf produced a plain mapping).
        # KNOWN ISSUE: the defaults parameter is never used.
        cfg = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(raw or {}))
        obj = OmegaConf.to_object(cfg)
        if not isinstance(obj, cls):
            obj = cls(**obj)

        return obj


def _rebuild_delegate_config(
    args, raw_delegate_config: Sequence[Mapping[str, Any]] | None, defaults: Mapping[str, Any] | None
) -> list[EvalEnvConfig]:
    # [FUNCTION DOC] HELPER: Turns the raw delegate entries (a sequence of mappings, or none) into
    # environment records. Each entry's name is lowercased and trimmed; an empty name logs a
    # warning and is skipped. The name "skills" lazily imports the skills configuration builder
    # and calls it with the launch args, the entry and the defaults; a non-none result is
    # appended. Any other name raises a value error. Returns the list in input order.
    envs: list[EvalEnvConfig] = []
    defaults = defaults or {}
    for env in raw_delegate_config or []:
        env_name = str(env.get("name", "")).strip().lower()
        if not env_name:
            logger.warning("Each delegate entry must include a non-empty `name`.")
            continue
        if env_name == "skills":
            from examples.experimental.eval.nemo_skills.skills_config import build_skills_eval_env_config

            env_cfg = build_skills_eval_env_config(args, env, defaults)
            if env_cfg is not None:
                envs.append(env_cfg)
        else:
            raise ValueError(f"Unknown delegate environment: {env_name}")
    return envs


class EvalDelegateError(RuntimeError):
    # [CLASS DOC]
    # Runtime error type for failures reported by an external evaluation server (for example an
    # error status or a malformed answer); raised by concrete delegate clients.
    """Raised when the external evaluation server returns an error."""


class EvalClient:
    # [CLASS DOC]
    # Abstract base class for environment-specific evaluation clients. Carries a name used as the
    # key of the client's raw response in aggregated results.
    name: str = ""

    def __init__(self, name: str):
        # [FUNCTION DOC] Stores the client name.
        self.name = name

    def evaluate(self, args, rollout_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        # [FUNCTION DOC] Abstract: subclasses run one evaluation for the given rollout id and
        # return a pair of a metrics mapping and the raw server response mapping. This base
        # version always raises a not-implemented error.
        raise NotImplementedError("Subclasses must implement this method")


def _flatten(result: dict[str, Any], prefix: str | None = None) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Recursively flattens a nested mapping (none treated as empty) into a
    # single-level mapping whose keys join the nested keys with slashes, optionally under a prefix.
    # Non-mapping values are kept as leaves.
    # KNOWN ISSUE: not used within this module.
    """Flatten nested metric dicts into slash separated keys."""
    flattened: dict[str, Any] = {}
    for key, value in (result or {}).items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten(value, full_key))
        else:
            flattened[full_key] = value
    return flattened


class EvalDelegateClient:
    # [CLASS DOC]
    # Aggregates a list of environment-specific evaluation clients behind one evaluate call. Built
    # through the maybe_create factory, which returns none when nothing is configured.
    """Aggregate multiple environment-specific delegate clients."""

    def __init__(self, delegates: Sequence[EvalClient]):
        # [FUNCTION DOC] Stores a copy of the client list in the given order.
        self._delegates = list(delegates)

    @classmethod
    def maybe_create(cls, args, env_configs: Sequence[EvalEnvConfig] | None = None) -> Optional["EvalDelegateClient"]:
        # [FUNCTION DOC] Factory. Uses the given environment records, or, when none are passed,
        # the eval delegate config attribute of the launch args (may be missing). Returns none if
        # there are no records. Computes the router address as http host:port from the launch args'
        # router ip and port, creates one client per record (records with no registered client are
        # skipped with a warning) and returns an aggregate of them, or none if no client resulted.
        env_configs = list(env_configs) if env_configs is not None else getattr(args, "eval_delegate_config", None)
        if not env_configs:
            return None

        router_addr = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        delegates: list[EvalClient] = []
        for env_cfg in env_configs:
            delegate = cls._create_delegate(env_cfg, router_addr)
            if delegate is not None:
                delegates.append(delegate)
        if not delegates:
            return None
        return cls(delegates)

    @staticmethod
    def _create_delegate(env_cfg: EvalEnvConfig, router_addr: str):
        # [FUNCTION DOC] HELPER: Static method mapping an environment record to a client. For the
        # environment named skills, lazily imports the skills client class and builds it from the
        # record and router address. Otherwise logs a warning and returns none.
        env_name = env_cfg.name
        if env_name == "skills":
            from examples.experimental.eval.nemo_skills.skills_client import SkillsEvalClient

            return SkillsEvalClient.from_config(env_cfg, router_addr)
        logger.warning("No delegate client registered for environment: %s", env_name)
        return None

    def evaluate(self, args, rollout_id: int) -> tuple[dict[str, Any], dict[str, Any]]:
        # [FUNCTION DOC] Runs every client in order for the rollout id. Returns a pair: metrics
        # merged into one mapping (later clients overwrite equal keys; empty results skipped) and a
        # mapping from client name to its raw response. Exceptions from a client propagate and
        # abort the remaining clients.
        aggregated_metrics: dict[str, Any] = {}
        raw_responses: dict[str, Any] = {}
        for delegate in self._delegates:
            metrics, response = delegate.evaluate(args, rollout_id)
            if metrics:
                aggregated_metrics.update(metrics)
            raw_responses[delegate.name] = response
        return aggregated_metrics, raw_responses
