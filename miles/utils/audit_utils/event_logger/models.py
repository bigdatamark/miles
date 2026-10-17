# [MODULE DOC] Typed, immutable and strict (no extra fields) records for audit events and the
# environment report they may embed. Every concrete event extends the event base (timestamp and
# source identity) and has a literal "type" tag; the union alias Event is discriminated on that
# tag, so a JSON line can be parsed back to the right class. Naming rule, verified when this
# module is imported: the snake-case form of each event class name must equal its type tag plus
# "_event"; a violation raises an assertion error at import time.

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Discriminator

from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils.audit_utils.process_identity import ProcessIdentity
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class EnvReportEditablePackageInfo(FrozenStrictBaseModel):
    # [CLASS DOC] An editable (development-mode) installed package: name, version, location.
    name: str
    version: str
    location: str


class EnvReportGitRepoInfo(FrozenStrictBaseModel):
    # [CLASS DOC] State of a git checkout backing a package: package name, location, commit,
    # dirty flag, diff statistics text, an optional hash of uncommitted changes, the untracked
    # paths (flag says whether the list was truncated) and untracked paths whose content was not
    # hashed.
    package_name: str
    location: str
    commit: str
    dirty: bool
    diff_stat: str
    uncommitted_hash: str | None
    untracked_paths: list[str]
    untracked_paths_truncated: bool
    untracked_unhashed_paths: list[str]


class EnvReportArgsDump(FrozenStrictBaseModel):
    # [CLASS DOC] Dump of the parsed run arguments: a mapping of argument name to value and the
    # list of names that were skipped.
    values: dict[str, Any]
    skipped_names: list[str]


class EnvReportProcessFacts(FrozenStrictBaseModel):
    # [CLASS DOC] Facts about the reporting process: host name, command-line tokens, arguments
    # dump, environment variables (name to value strings) and an optional environment report
    # produced by the launcher.
    hostname: str
    argv: list[str]
    args: EnvReportArgsDump
    env_vars: dict[str, str]
    launcher_env_report: dict[str, Any] | None


class EnvReport(FrozenStrictBaseModel):
    # [CLASS DOC] Full software environment report: process facts, a mapping of key package names
    # to versions, editable packages, git repositories, the full list of installed packages (each
    # a string mapping) and a flag saying whether packages were actually probed.
    process: EnvReportProcessFacts
    key_versions: dict[str, str]
    editable_packages: list[EnvReportEditablePackageInfo]
    git_repos: list[EnvReportGitRepoInfo]
    full_pip_list: list[dict[str, str]]
    packages_probed: bool


class EventBase(FrozenStrictBaseModel):
    # [CLASS DOC] Base of all events: timestamp and the identity of the emitting process.
    timestamp: datetime
    source: ProcessIdentity


class _ActorTrainEventBase(EventBase):
    # [CLASS DOC] HELPER: Base for events emitted per training step: rollout id and attempt
    # number (default 0, incremented on retries).
    rollout_id: int
    attempt: int = 0


class OptimizerStateInfo(FrozenStrictBaseModel):
    # [CLASS DOC] State of one sub-optimizer in hashed form: a mapping from parameter index to
    # name and a nested state mapping.
    """Snapshot of one sub-optimizer's state with tensors replaced by hashes."""

    param_names: dict[int, str]
    state_dict: dict[str, Any]


class TrainEngineLocalWeightChecksumState(FrozenStrictBaseModel):
    # [CLASS DOC] Hashes of one training rank's local state: parameter hashes and buffer hashes
    # (name to hash string) and optimizer state hashes (one record per sub-optimizer).
    param_hashes: dict[str, str]
    buffer_hashes: dict[str, str]
    # May be skipped in non-debug mode if too expensive
    optimizer_hashes: list[OptimizerStateInfo]


class TrainEngineLocalWeightChecksumEvent(_ActorTrainEventBase):
    # [CLASS DOC] Event: local weight checksums of a training rank for a rollout step; carries
    # the checksum state record.
    type: Literal["train_engine_local_weight_checksum"] = "train_engine_local_weight_checksum"
    state: TrainEngineLocalWeightChecksumState


class WitnessSnapshotParamEvent(_ActorTrainEventBase):
    # [CLASS DOC] Event: snapshot of one witness module's parameter at the end of a train step.
    # Fields: the witness instance id (names pipeline stage, chunk and head or tail), the ids
    # whose witness row is nonzero, and the stale ids that the allocator marked as recycled.
    type: Literal["witness_snapshot_param"] = "witness_snapshot_param"
    instance_id: str
    # TODO: may shrink a contiguous range of numbers into a pair, if this is too large/slow
    nonzero_witness_ids: list[int]
    stale_ids: list[int]


class WitnessAllocateIdEvent(EventBase):
    # [CLASS DOC] Event: witness ids allocated to samples for a rollout attempt. Carries the
    # mapping from witness id to sample index and the allocator counter after the allocation (a
    # resumed run recovers the counter from the maximum of these). Unlike the train events, the
    # attempt has no default.
    type: Literal["witness_allocate_id"] = "witness_allocate_id"
    rollout_id: int
    attempt: int
    witness_id_to_sample_index: dict[int, int]
    # Allocator counter after this allocation; a resumed run recovers the allocator from it.
    counter_after: int


class TrainGroupStepEndEvent(EventBase):
    # [CLASS DOC] Event: end of a train step across cells. The outcome mapping goes from cell
    # index to either the marker string "error" or the list of per-rank step outcomes.
    type: Literal["train_group_step_end"] = "train_group_step_end"
    rollout_id: int
    cell_outcomes: dict[int, Literal["error"] | list[TrainStepOutcome]]


class CellReconfigureEvent(EventBase):
    # [CLASS DOC] Event: the set of alive cells was reconfigured. Fields: rollout id, quorum id,
    # the optional index of the source cell used for healing, the indices healed (healing
    # happened only if non-empty) and the indices alive afterwards.
    type: Literal["cell_reconfigure"] = "cell_reconfigure"
    rollout_id: int
    quorum_id: int
    src_cell_index: int | None
    # healing happened iff non-empty
    healed_cell_indices: list[int]
    alive_cell_indices_after: list[int]


class InferenceEngineWeightChecksumEvent(EventBase):
    # [CLASS DOC] Event: weight checksums reported by each rollout engine after a weight push.
    # The rollout id is minus one for a fresh run's initial sync. The optional trainer model id
    # names the pushed policy. The checksum field is a list with one mapping from tensor name to
    # hash per engine.
    type: Literal["inference_engine_weight_checksum"] = "inference_engine_weight_checksum"
    # The out-of-loop startup sync stamps start_rollout_id - 1, so -1 is a fresh run's initial sync.
    rollout_id: int
    # The policy whose weights were pushed, or None for a run that trains one unnamed policy.
    trainer_model_id: str | None = None
    # One {tensor -> hash} dict per rollout engine; a TP>1 engine's ranks merge with a rank{r}/ prefix.
    engine_checksums: list[dict[str, str]]


class TrainAdvantageComputationEvent(_ActorTrainEventBase):
    # [CLASS DOC] Event: per-sample advantages (a list of per-token floats per sample) and the
    # matching per-token witness ids (one list per sample, the same length as the advantages
    # list).
    type: Literal["train_advantage_computation"] = "train_advantage_computation"
    advantages: list[list[float]]
    witness_ids: list[list[int]]


class EnvReportEvent(EventBase):
    # [CLASS DOC] Event: carries one process's full environment report.
    type: Literal["env_report"] = "env_report"
    report: EnvReport


class EngineEnvReportEvent(EventBase):
    # [CLASS DOC] Event: environment report of an inference engine server: cell id, server URL
    # and the server's free-form info mapping.
    type: Literal["engine_env_report"] = "engine_env_report"
    cell_id: str
    server_url: str
    server_info: dict[str, Any]


class MetricEvent(EventBase):
    # [CLASS DOC] Event: a free-form metrics mapping, optionally tied to a rollout id and attempt
    # (both default to none).
    type: Literal["metric"] = "metric"
    rollout_id: int | None = None
    attempt: int | None = None
    metrics: dict[str, Any]


Event = Annotated[
    TrainEngineLocalWeightChecksumEvent
    | WitnessSnapshotParamEvent
    | WitnessAllocateIdEvent
    | TrainGroupStepEndEvent
    | CellReconfigureEvent
    | InferenceEngineWeightChecksumEvent
    | TrainAdvantageComputationEvent
    | EnvReportEvent
    | EngineEnvReportEvent
    | MetricEvent,
    Discriminator("type"),
]


def _to_snake_case(name: str) -> str:
    # [FUNCTION DOC] HELPER: Converts a camel-case name to lower snake case by inserting an
    # underscore before each uppercase letter that follows a lowercase letter or digit, then
    # lowercasing. Consecutive capitals are not split (an acronym stays together).
    import re

    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def _check_event_naming() -> None:
    # [FUNCTION DOC] HELPER: Import-time sanity check. For each class in the event union, takes
    # its default type tag and asserts that the snake-case form of the class name equals the tag
    # plus "_event"; the assertion message names the class and both forms. Called once at the
    # bottom of the module.
    import typing

    event_types = typing.get_args(typing.get_args(Event)[0])
    for cls in event_types:
        type_value = cls.model_fields["type"].default
        expected_snake = type_value + "_event"
        actual_snake = _to_snake_case(cls.__name__)
        assert actual_snake == expected_snake, (
            f"Event class {cls.__name__} (snake: {actual_snake}) does not match "
            f"type '{type_value}' (expected snake: {expected_snake})"
        )


_check_event_naming()
