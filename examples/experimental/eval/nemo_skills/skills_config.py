# [MODULE DOC]
# Structured configuration records for the NeMo Skills delegate, shared by the client and the
# server. The dataset record extends the generic delegate dataset record with validation of the
# dataset name and a derived "runtime name" that encodes the number of samples per prompt; the
# environment record extends the generic environment record with a list of dataset records and
# builds them from the raw configuration mapping.
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from examples.experimental.eval.eval_delegate import EvalEnvConfig, EvalEnvDatasetConfig


@dataclass
class SkillsEvalEnvDatasetConfig(EvalEnvDatasetConfig):
    # [CLASS DOC]
    # Dataset record for the skills environment; same fields as the generic dataset record, with
    # name validation applied at construction time.
    """Dataset configuration shared by the Skills client/server."""

    def __post_init__(self):
        # [FUNCTION DOC] Validation run after construction: trims whitespace from the name (none
        # treated as empty) and stores it back; raises a value error if it is empty or contains a
        # colon (the colon is reserved for the sample-count suffix of the runtime name).
        name = (self.name or "").strip()
        self.name = name
        if not name:
            raise ValueError("Each Skills dataset entry must include a non-empty `name`.")
        if ":" in name:
            raise ValueError(
                "Colon in dataset name is not allowed; use `n_samples_per_eval_prompt` to configure samples per prompt."
            )

    @property
    def runtime_name(self) -> str:
        # [FUNCTION DOC] Property: the benchmark identifier passed to NeMo Skills. Just the name if
        # samples per prompt is not set, otherwise the name, a colon and the sample count.
        if self.n_samples_per_eval_prompt is None:
            return self.name
        return f"{self.name}:{self.n_samples_per_eval_prompt}"

    @classmethod
    def parse(cls, args, dataset_cfg: Mapping[str, Any], defaults: Mapping[str, Any]):
        # [FUNCTION DOC] Class method delegating entirely to the parent's parser (which builds an
        # instance of this subclass); present only to give the subclass its own entry point.
        return super().parse(args, dataset_cfg, defaults)


@dataclass
class SkillsEvalEnvConfig(EvalEnvConfig):
    # [CLASS DOC]
    # Environment record for the skills environment: the generic environment fields plus a list of
    # dataset records (empty by default).
    """Environment configuration shared by the Skills client/server."""

    datasets: list[SkillsEvalEnvDatasetConfig] = field(default_factory=list)

    @classmethod
    def parse(cls, args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]) -> SkillsEvalEnvConfig:
        # [FUNCTION DOC] Class method building the environment record. Parses the raw mapping with
        # the parent's parser, then replaces the datasets list by parsing each raw dataset entry
        # (from the raw mapping's datasets key, none treated as empty) with the dataset parser using
        # the launch args and the record's own defaults mapping. Returns the record.
        # KNOWN ISSUE: the defaults argument is not used for the datasets (the record's own
        # defaults are used instead), and the parent's merge already converts the raw dataset
        # entries once before they are re-parsed here.
        base_cfg: SkillsEvalEnvConfig = super().parse(raw_env_config, defaults)
        datasets = raw_env_config.get("datasets") or []
        base_cfg.datasets = [
            SkillsEvalEnvDatasetConfig.parse(args, dataset_cfg, base_cfg.defaults) for dataset_cfg in datasets
        ]
        return base_cfg


def build_skills_eval_env_config(args, raw_env_config: Mapping[str, Any], defaults: Mapping[str, Any]):
    # [FUNCTION DOC] Public builder used by the delegate configuration code: returns the parsed
    # skills environment record for the raw environment mapping.
    return SkillsEvalEnvConfig.parse(args, raw_env_config, defaults)
