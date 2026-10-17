# [MODULE DOC]
# Frozen, strict pydantic models describing the values files of the miles run and miles workbench
# helm charts. Field names are snake case in python and camel case when dumped (by alias). The
# models carry validation (dns label and subdomain patterns, port ranges, absolute or relative path
# patterns, quantity format, forbidden environment variable names) and json schema extras so the
# schema generated from them can validate values files. Module constants prefixed with an underscore
# are shared patterns and annotated types.
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel, to_snake

from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR
from miles.utils.external_utils.colocate_pairing.config import PairingConfig
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.env_vars import (
    BASE_GPU_ID_ENV_VAR,
    CELL_INDEX_ENV_VAR,
    NAMESPACE_ENV_VAR,
    POD_INDEX_ENV_VAR,
    RELEASE_ENV_VAR,
)
from miles.utils.workers.naming import (
    DNS_LABEL_PATTERN,
    DNS_SUBDOMAIN_PATTERN,
    POOL_NAME_MAX_LENGTH,
    PORT_NAME_MAX_LENGTH,
    PORT_NAME_PATTERN,
)

_DNS_LABEL = rf"^{DNS_LABEL_PATTERN}$"
_OPTIONAL_DNS_LABEL = rf"^({DNS_LABEL_PATTERN})?$"
_DNS_SUBDOMAIN = rf"^{DNS_SUBDOMAIN_PATTERN}$"
_OPTIONAL_DNS_SUBDOMAIN = rf"^({DNS_SUBDOMAIN_PATTERN})?$"

_NO_PARENT_TRAVERSAL = {"not": {"pattern": r"(^|/)\.\.(/|$)"}}
_PLATFORM_OWNED_ENV_VARS = [
    "PYTHONPATH",
    CELL_INDEX_ENV_VAR,
    POD_INDEX_ENV_VAR,
    BASE_GPU_ID_ENV_VAR,
    NAMESPACE_ENV_VAR,
    RELEASE_ENV_VAR,
]
_ENV_KEYS = {
    "propertyNames": {
        "pattern": "^[ -<>-~]+$",
        "not": {"enum": [LAUNCHER_REPORT_ENV_VAR, *_PLATFORM_OWNED_ENV_VARS]},
    }
}

_VOLUME_SOURCES = ("hostPath", "persistentVolumeClaim", "emptyDir")
_DEV_SHM_SOURCES = ("hostPath", "emptyDir")
_QUANTITY = r"^[0-9]+(\.[0-9]+)?([EPTGMk]i?|m)?$"
_OBJECT_NAME_MAX = 63
_KUBERNETES_NAME_MAX = 253
WORKBENCH_OBJECT_NAME_MAX = 52

_PoolName = Annotated[str, Field(min_length=1, max_length=POOL_NAME_MAX_LENGTH, pattern=_DNS_LABEL)]
_ObjectName = Annotated[str, Field(min_length=1, max_length=_OBJECT_NAME_MAX, pattern=_DNS_LABEL)]
_Port = Annotated[int, Field(ge=1, le=65535)]
_AbsolutePath = Annotated[str, Field(pattern="^/", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_OptionalAbsolutePath = Annotated[str, Field(pattern="^(/.*)?$", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_RelativePath = Annotated[str, Field(pattern="^([^/].*)?$", json_schema_extra=_NO_PARENT_TRAVERSAL)]
_EnvVars = Annotated[dict[str, str], Field(json_schema_extra=_ENV_KEYS)]

_Resources = dict[str, Any]


class ValuesModel(FrozenStrictBaseModel):
    # [CLASS DOC] Base of all values models: forbids extra fields, is frozen, uses camel case
    # aliases and still accepts population by python field name.
    model_config = ConfigDict(extra="forbid", frozen=True, alias_generator=to_camel, populate_by_name=True)

    def as_values(self) -> dict[str, Any]:
        # [FUNCTION DOC] Dumps the model to a plain nested mapping using camel case aliases and
        # omitting fields that are none; this is what is written to the helm values file.
        return self.model_dump(by_alias=True, exclude_none=True)


class PortEntry(ValuesModel):
    # [CLASS DOC] A named container port: name (validated port name pattern and max length) and
    # port number 1 to 65535.
    name: Annotated[str, Field(min_length=1, max_length=PORT_NAME_MAX_LENGTH, pattern=PORT_NAME_PATTERN)]
    port: _Port


class PoolEntry(ValuesModel):
    # [CLASS DOC] One pool as rendered by the chart: pool name and kubernetes object name (dns
    # labels), optional pool id, a non-empty container command, ports, environment, free-form string
    # metadata, replica count and pod size (each at least 1), raw resources mapping, an optional
    # restart stamp (non-empty) that rolls pods when changed, and an optional service account name.
    name: _PoolName
    object_name: _ObjectName
    pool_id: str | None = None
    command: Annotated[list[str], Field(min_length=1)]
    ports: list[PortEntry] | None = None
    env: _EnvVars | None = None
    meta: dict[str, str] | None = None
    replicas: Annotated[int, Field(ge=1)] | None = None
    size: Annotated[int, Field(ge=1)] | None = None
    resources: _Resources | None = None
    restart_at: Annotated[str, Field(min_length=1)] | None = None
    service_account_name: _ObjectName | None = None


class ObjectNames(ValuesModel):
    # [CLASS DOC] Names (dns labels, at most 63 characters) of the release-level kubernetes objects
    # that the chart creates and other components must refer to.
    orchestrator: _ObjectName
    platform_read: _ObjectName
    platform_read_delete: _ObjectName
    mooncake_master: _ObjectName
    colocate_pairing: _ObjectName
    uninstall: _ObjectName
    uninstall_manifest: _ObjectName


class AutoUninstallSection(ValuesModel):
    # [CLASS DOC] Switch for the chart's automatic uninstall job.
    enabled: bool


class OrchestratorSection(ValuesModel):
    # [CLASS DOC] Orchestrator settings: container command, raw resources and restart stamp.
    command: list[str] | None = None
    resources: _Resources | None = None
    restart_at: Annotated[str, Field(min_length=1)] | None = None


class MooncakeSection(ValuesModel):
    # [CLASS DOC] Mooncake store settings: enabled flag, rpc and metrics ports, raw resources.
    enabled: bool | None = None
    rpc_port: _Port | None = None
    metrics_port: _Port | None = None
    resources: _Resources | None = None


class RunValues(ValuesModel):
    # [CLASS DOC] The run section of the run chart: run id (dns label, limited length), absolute
    # paths of the state file and launch record, object names, orchestrator, the three pool lists
    # (static workers, inference engines, trainer engines), shared environment, mooncake, colocate
    # pairing config and auto-uninstall switch.
    id: Annotated[str, Field(max_length=RUN_ID_MAX_LENGTH, pattern=_DNS_LABEL)]
    state_file: Annotated[str, Field(min_length=1, pattern="^/")] | None = None
    launch_record: Annotated[str, Field(min_length=1, pattern="^/")] | None = None
    object_names: ObjectNames
    orchestrator: OrchestratorSection | None = None
    static_workers: list[PoolEntry] | None = None
    inference_engines: list[PoolEntry] | None = None
    trainer_engines: list[PoolEntry] | None = None
    env: _EnvVars | None = None
    mooncake: MooncakeSection | None = None
    colocate: PairingConfig | None = None
    auto_uninstall: AutoUninstallSection | None = None


class CommandJobValues(ValuesModel):
    # [CLASS DOC] Values of the one-shot command job. The json schema extra requires name, object
    # name and a non-empty command whenever the job is enabled. Other fields: completions (at least
    # 1), gpus per pod (at least 0), active deadline (at least 1 second) and time-to-live after
    # finish (at least 0 seconds).
    # KNOWN ISSUE: this class overrides the model config without repeating the base settings; in
    # pydantic the configs are merged, so this is harmless, but it looks like it replaces them.
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"enabled": {"const": True}}, "required": ["enabled"]},
                    "then": {
                        "required": ["name", "objectName", "command"],
                        "properties": {
                            "name": {"minLength": 1},
                            "objectName": {"minLength": 1},
                            "command": {"minItems": 1},
                        },
                    },
                }
            ]
        }
    )

    enabled: bool | None = None
    name: Annotated[str, Field(max_length=POOL_NAME_MAX_LENGTH, pattern=_OPTIONAL_DNS_LABEL)] | None = None
    object_name: Annotated[str, Field(max_length=_OBJECT_NAME_MAX, pattern=_OPTIONAL_DNS_LABEL)] | None = None
    command: list[str] | None = None
    completions: Annotated[int, Field(ge=1)] | None = None
    gpus_per_pod: Annotated[int, Field(ge=0)] | None = None
    active_deadline_seconds: Annotated[int, Field(ge=1)] | None = None
    ttl_seconds_after_finished: Annotated[int, Field(ge=0)] | None = None


class Image(ValuesModel):
    # [CLASS DOC] Container image: repository and tag (non-empty), optional pull policy (always,
    # if not present, or never) and optional list of pull secret names (dns subdomains).
    repository: Annotated[str, Field(min_length=1)]
    tag: Annotated[str, Field(min_length=1)]
    pull_policy: Literal["Always", "IfNotPresent", "Never"] | None = None
    pull_secrets: (
        list[Annotated[str, Field(min_length=1, max_length=_KUBERNETES_NAME_MAX, pattern=_DNS_SUBDOMAIN)]] | None
    ) = None


def _assert_one_source(volume: ValuesModel, sources: tuple[str, ...]) -> None:
    # [FUNCTION DOC] HELPER: Asserts that exactly one of the given source fields (given in camel
    # case, looked up by their snake case attribute) is set on the model. Raises an assertion error
    # listing the declared ones otherwise.
    declared = [source for source in sources if getattr(volume, to_snake(source)) is not None]
    assert len(declared) == 1, (
        f"a volume declares exactly one of {list(sources)}, but this one declares {declared}: a volume with none "
        f"is a mount kubernetes cannot satisfy, and one with several is a values file whose reader has to guess"
    )


class HostPathSource(ValuesModel):
    # [CLASS DOC] Host directory volume source: absolute path without parent traversal, and an
    # optional type (directory must exist, or directory created if missing).
    path: _AbsolutePath
    type: Literal["Directory", "DirectoryOrCreate"] | None = None


class PersistentVolumeClaimSource(ValuesModel):
    # [CLASS DOC] Persistent volume claim volume source, identified by a dns subdomain claim name.
    claim_name: Annotated[str, Field(min_length=1, max_length=_KUBERNETES_NAME_MAX, pattern=_DNS_SUBDOMAIN)]


class EmptyDirSource(ValuesModel):
    # [CLASS DOC] Empty directory volume source: medium (default or memory-backed) and optional size
    # limit written as a kubernetes quantity.
    medium: Literal["", "Memory"] | None = None
    size_limit: Annotated[str, Field(pattern=_QUANTITY)] | None = None


class VolumeMount(ValuesModel):
    # [CLASS DOC] A mount of a volume: absolute mount path, optional relative sub path (not
    # starting with a slash) and optional read-only flag; no parent traversal allowed in paths.
    mount_path: _AbsolutePath
    sub_path: _RelativePath | None = None
    read_only: bool | None = None


class VolumeEntry(ValuesModel):
    # [CLASS DOC] A named volume with exactly one source (host path, persistent volume claim, or
    # empty dir) and at least one mount. Exactly-one is enforced both by the json schema (one-of
    # required) and by an after-validation assertion.
    model_config = ConfigDict(json_schema_extra={"oneOf": [{"required": [key]} for key in _VOLUME_SOURCES]})

    name: _ObjectName
    host_path: HostPathSource | None = None
    persistent_volume_claim: PersistentVolumeClaimSource | None = None
    empty_dir: EmptyDirSource | None = None
    mounts: Annotated[list[VolumeMount], Field(min_length=1)]

    @model_validator(mode="after")
    def _declares_exactly_one_source(self) -> VolumeEntry:
        # [FUNCTION DOC] HELPER: After-validator asserting exactly one volume source is set; returns
        # the model unchanged.
        _assert_one_source(self, _VOLUME_SOURCES)
        return self


class DevShm(ValuesModel):
    # [CLASS DOC] Shared memory mount: absolute mount path plus exactly one source out of host path
    # or empty dir (enforced by schema and after-validation).
    model_config = ConfigDict(json_schema_extra={"oneOf": [{"required": [key]} for key in _DEV_SHM_SOURCES]})

    mount_path: _AbsolutePath
    host_path: HostPathSource | None = None
    empty_dir: EmptyDirSource | None = None

    @model_validator(mode="after")
    def _declares_exactly_one_source(self) -> DevShm:
        # [FUNCTION DOC] HELPER: After-validator asserting exactly one shm source is set; returns
        # the model unchanged.
        _assert_one_source(self, _DEV_SHM_SOURCES)
        return self


class Paths(ValuesModel):
    # [CLASS DOC] Filesystem locations: optional absolute root directory under which runs live.
    runs_root: _AbsolutePath | None = None


class Scheduling(ValuesModel):
    # [CLASS DOC] Pod placement settings passed through unvalidated: node selector, tolerations and
    # affinity.
    node_selector: dict[str, str] | None = None
    tolerations: list[dict[str, Any]] | None = None
    affinity: dict[str, Any] | None = None


class InfraValues(ValuesModel):
    # [CLASS DOC] Infrastructure section shared by charts: image, volumes (required), paths, dev shm,
    # scheduling and a shared environment (validated to exclude platform-owned and launcher report
    # variable names).
    image: Image
    volumes: list[VolumeEntry]
    paths: Paths | None = None
    dev_shm: DevShm | None = None
    scheduling: Scheduling | None = None
    env: _EnvVars | None = None


class RbacSection(ValuesModel):
    # [CLASS DOC] Role-based access control switches of the workbench chart: whether to create the
    # rbac objects and whether to include rules for leader worker sets.
    create: bool | None = None
    leader_worker_sets: bool | None = None


class ServiceAccountSection(ValuesModel):
    # [CLASS DOC] Service account selection: optional name (empty or a dns subdomain).
    name: Annotated[str, Field(max_length=_KUBERNETES_NAME_MAX, pattern=_OPTIONAL_DNS_SUBDOMAIN)] | None = None


class WorkbenchResources(ValuesModel):
    # [CLASS DOC] Resource requests and limits for the workbench pod: mappings from resource name to
    # string or number.
    requests: dict[str, str | float] | None = None
    limits: dict[str, str | float] | None = None


class MilesRunChartValues(ValuesModel):
    # [CLASS DOC] Top-level values of the miles run chart: infra, run, extra raw manifests (list of
    # strings) and the command job; every part is optional.
    infra: InfraValues | None = None
    run: RunValues | None = None
    extra_manifests: list[str] | None = None
    command_job: CommandJobValues | None = None


class MilesWorkbenchChartValues(ValuesModel):
    # [CLASS DOC] Top-level values of the miles workbench chart: required infra, optional object name
    # (empty or a dns label of at most 52 characters), resources, rbac and service account.
    infra: InfraValues
    object_name: Annotated[str, Field(max_length=WORKBENCH_OBJECT_NAME_MAX, pattern=_OPTIONAL_DNS_LABEL)] | None = None
    resources: WorkbenchResources | None = None
    rbac: RbacSection | None = None
    service_account: ServiceAccountSection | None = None
