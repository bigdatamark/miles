# [MODULE DOC] Configuration of evaluation datasets. Each dataset entry may set its own sampling
# parameters and column keys; unset ones fall back first to a shared defaults mapping and then to
# launch arguments (eval-specific argument first, then the rollout/training one). Two spec tables
# list, per field, the keys to look up in the dataset entry, in the defaults, and the argument
# attribute names in fallback order. A private sentinel distinguishes "key absent" from an
# explicit none value.
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_MISSING = object()

# TODO: This is ugly, temporarily leave this. We should unify all the config name for dataset, default, and args. (advice from Tom.)
DATASET_RUNTIME_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "n_samples_per_eval_prompt": {
        "dataset_keys": ("n_samples_per_eval_prompt",),
        "default_keys": ("n_samples_per_eval_prompt",),
        "arg_attrs": ("n_samples_per_eval_prompt", "n_samples_per_prompt"),
    },
    "temperature": {
        "dataset_keys": ("temperature",),
        "default_keys": ("temperature",),
        "arg_attrs": ("eval_temperature", "rollout_temperature"),
    },
    "top_p": {
        "dataset_keys": ("top_p",),
        "default_keys": ("top_p",),
        "arg_attrs": ("eval_top_p", "rollout_top_p"),
    },
    "top_k": {
        "dataset_keys": ("top_k",),
        "default_keys": ("top_k",),
        "arg_attrs": ("eval_top_k", "rollout_top_k"),
    },
    "max_response_len": {
        "dataset_keys": ("max_response_len",),
        "default_keys": ("max_response_len",),
        "arg_attrs": ("eval_max_response_len", "rollout_max_response_len"),
    },
}

DATASET_SAMPLE_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "input_key": {
        "dataset_keys": ("input_key",),
        "default_keys": ("input_key",),
        "arg_attrs": ("eval_input_key", "input_key"),
    },
    "label_key": {
        "dataset_keys": ("label_key",),
        "default_keys": ("label_key",),
        "arg_attrs": ("eval_label_key", "label_key"),
    },
    "tool_key": {
        "dataset_keys": ("tool_key",),
        "default_keys": ("tool_key",),
        "arg_attrs": ("eval_tool_key", "tool_key"),
    },
    "metadata_key": {
        "dataset_keys": ("metadata_key",),
        "default_keys": ("metadata_key",),
        "arg_attrs": ("metadata_key",),
    },
}


def _first_not_missing(*values: Any) -> Any:
    # [FUNCTION DOC] HELPER: Returns the first argument that is not the "missing" sentinel (an
    # explicit none counts as present), or the sentinel if all are missing.
    for value in values:
        if value is not _MISSING:
            return value
    return _MISSING


def _pick_from_mapping(data: dict[str, Any], key_names: tuple[str, ...] | None) -> Any:
    # [FUNCTION DOC] HELPER: Returns the value of the first listed key present in the mapping
    # (even if its value is none); the "missing" sentinel if no key names are given or none is
    # present.
    if key_names is None:
        return _MISSING
    for key_name in key_names:
        if key_name in data:
            return data[key_name]
    return _MISSING


def pick_from_args(args: Any, attrs: tuple[str, ...]) -> Any:
    # [FUNCTION DOC] Returns the first listed attribute of the arguments object that exists and is
    # not none; none if there is no such attribute.
    for attr in attrs:
        value = getattr(args, attr, None)
        if value is not None:
            return value
    return None


def _ensure_metadata_overrides(value: Any) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: None becomes an empty mapping; a mapping is returned as is (same
    # object); anything else raises a type error.
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("metadata_overrides must be a mapping.")
    return value


@dataclass
class EvalDatasetConfig:
    # [CLASS DOC] Mutable record for one eval dataset. Required: name and path. Optional (default
    # none): reward model type; column keys (input, label, tool, metadata); samples per eval
    # prompt; sampling settings (temperature, top-p, top-k, max response length, stop strings, stop
    # token ids, minimum new tokens); path of a custom generate function; and a mapping of metadata
    # overrides (default empty) validated on construction.
    """Configuration for a single evaluation dataset."""

    name: str
    path: str
    rm_type: str | None = None

    # Dataset-specific overrides
    input_key: str | None = None
    label_key: str | None = None
    tool_key: str | None = None
    metadata_key: str | None = None

    n_samples_per_eval_prompt: int | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_response_len: int | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    min_new_tokens: int | None = None

    # per-dataset custom generate function (e.g., for tool calling)
    custom_generate_function_path: str | None = None

    metadata_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] Normalizes the metadata overrides (none becomes empty; non-mapping raises
        # a type error).
        self.metadata_overrides = _ensure_metadata_overrides(self.metadata_overrides)

    @property
    def cache_key(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Property: tuple of name, path and the four column keys, used to cache the
        # loaded dataset. Sampling settings and metadata overrides are deliberately not part of it.
        """Return a tuple uniquely identifying dataset config for caching."""
        return (
            self.name,
            self.path,
            self.input_key,
            self.label_key,
            self.tool_key,
            self.metadata_key,
        )

    def inject_metadata(self, sample_metadata: Any) -> dict[str, Any]:
        # [FUNCTION DOC] Returns a new mapping: a copy of the sample metadata (empty if it is not a
        # mapping), then the reward model type stored under rm_type if configured, then every
        # metadata override applied last (so overrides win, including over rm_type). Input is not
        # modified.
        """Return updated metadata merging overrides."""
        if not isinstance(sample_metadata, dict):
            metadata = {}
        else:
            metadata = dict(sample_metadata)

        if self.rm_type is not None:
            metadata["rm_type"] = self.rm_type

        for key, value in self.metadata_overrides.items():
            metadata[key] = value

        return metadata


def ensure_dataset_list(config: Any) -> list[dict[str, Any]]:
    # [FUNCTION DOC] Returns an empty list for none. A mapping (name to settings, null settings
    # allowed) becomes a list of copied dicts with the name filled in if absent. A list or tuple
    # becomes a list of copied dicts, each of which must contain a name (value error otherwise).
    # Any other type raises a type error. Inputs are copied, not mutated.
    """
    Normalize OmegaConf containers into a list of dicts.
    Accepts either a list or dictionary keyed by dataset name.
    """
    if config is None:
        return []

    if isinstance(config, dict):
        datasets = []
        for name, cfg in config.items():
            dataset = dict(cfg or {})
            dataset.setdefault("name", name)
            datasets.append(dataset)
        return datasets

    if isinstance(config, (list, tuple)):
        datasets = []
        for item in config:
            dataset = dict(item or {})
            if "name" not in dataset:
                raise ValueError("Each evaluation dataset entry must include a `name` field.")
            datasets.append(dataset)
        return datasets

    raise TypeError("eval.datasets must be either a list or a mapping.")


def _apply_dataset_field_overrides(
    args: Any, dataset_cfg: dict[str, Any], defaults: dict[str, Any], spec_names: dict[str, Any]
) -> None:
    # [FUNCTION DOC] HELPER: Mutates the dataset mapping. For each field in the spec table: the
    # value is the dataset's own (first matching key), else the shared default (first matching
    # key), else the launch-argument fallback (first attribute that is not none, or none). An
    # explicitly present value, even none, stops the fallback. The resolved value is always
    # stored under the field name.
    for field_name, spec in spec_names.items():
        dataset_value = _pick_from_mapping(dataset_cfg, spec["dataset_keys"])
        default_value = _pick_from_mapping(defaults, spec["default_keys"])
        resolved_value = _first_not_missing(dataset_value, default_value)
        if resolved_value is not _MISSING:
            dataset_cfg[field_name] = resolved_value
            continue
        dataset_cfg[field_name] = pick_from_args(args, spec["arg_attrs"])


def build_eval_dataset_configs(
    args: Any,
    raw_config: Iterable[dict[str, Any]],
    defaults: dict[str, Any],
) -> list[EvalDatasetConfig]:
    # [FUNCTION DOC] For each raw dataset mapping (null treated as empty, copied), resolves the
    # runtime fields (samples per eval prompt, temperature, top-p, top-k, max response length) and
    # sample-key fields (input, label, tool, metadata keys) from the dataset, the defaults mapping
    # (null treated as empty) and the launch arguments, then builds a dataset config from the
    # result. Returns the configs in order. Unknown keys or missing name/path raise a type error
    # from the dataclass constructor.
    defaults = defaults or {}
    datasets: list[EvalDatasetConfig] = []
    for cfg in raw_config:
        cfg_dict = dict(cfg or {})
        combined_specs = {**DATASET_RUNTIME_SPECS, **DATASET_SAMPLE_SPECS}
        _apply_dataset_field_overrides(args, cfg_dict, defaults, combined_specs)
        dataset = EvalDatasetConfig(**cfg_dict)
        datasets.append(dataset)
    return datasets
