# [MODULE DOC] Typed, immutable models of rendered Kubernetes manifests, parsed from yaml. Pod
# workload objects (stateful sets, leader-worker sets, deployments, daemon sets, replica sets,
# jobs) expose replicas, containers, a restart-at annotation of the pod template and per-container
# lookups; any other object is kept as a free-form mapping. The manifest groups objects by identity
# (api group, kind, namespace, name) and answers questions used by the launcher: lookup by object
# key, restart stamp of a stateful set, the last value of a flag in a container's command, and the
# state file path named by that command. Lookups assert uniqueness rather than guessing.
from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Literal, NamedTuple, TypeVar

import yaml
from pydantic import Field, field_validator

from miles.utils.external_utils.command_utils.common import ArgvManipulator
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import STATE_FILE_FLAG
from miles.utils.pydantic_utils import FrozenOpenBaseModel, FrozenStrictBaseModel

RESTART_AT_ANNOTATION = "miles.radixark.io/restart-at"

PodWorkloadKind = Literal["StatefulSet", "LeaderWorkerSet", "Deployment", "DaemonSet", "ReplicaSet", "Job"]
STATEFUL_SET_KIND: PodWorkloadKind = "StatefulSet"
LEADER_WORKER_SET_KIND: PodWorkloadKind = "LeaderWorkerSet"
_STATEFUL_SET_API_GROUP = "apps"

SpecT = TypeVar("SpecT")


class ManifestObjectKey(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable (hashable) identification of an object by kind and name, ignoring api
    # group and namespace.
    kind: str
    name: str


class EnvEntry(FrozenOpenBaseModel):
    # [CLASS DOC] One container environment variable: name and literal value (default empty; value
    # sources such as secret references are kept as extra fields, not as the value). Unknown fields
    # are allowed.
    name: str
    value: str = ""


class Container(FrozenOpenBaseModel):
    # [CLASS DOC] A container: name, command list (default empty) and environment entries
    # (default empty); unknown fields allowed.
    name: str
    command: list[str] = []
    env: list[EnvEntry] = []


class PodSpec(FrozenOpenBaseModel):
    # [CLASS DOC] Pod spec reduced to its containers (default empty); other fields kept as extras.
    containers: list[Container] = []


class PodTemplateMetadata(FrozenOpenBaseModel):
    # [CLASS DOC] Pod template metadata reduced to annotations (default empty mapping).
    annotations: dict[str, Any] = {}

    @field_validator("annotations", mode="before")
    @classmethod
    def _read_null_as_empty(cls, value: Any) -> Any:
        # [FUNCTION DOC] HELPER: Pre-validator turning an explicit null annotations value (as
        # rendered templates may produce) into an empty mapping; other values pass through.
        return {} if value is None else value


class PodTemplate(FrozenOpenBaseModel):
    # [CLASS DOC] Pod template: optional metadata and optional spec.
    metadata: PodTemplateMetadata | None = None
    spec: PodSpec | None = None


class WorkloadSpec(FrozenOpenBaseModel):
    # [CLASS DOC] Workload spec reduced to the replica count (optional) and the pod template
    # (optional).
    replicas: int | None = None
    template: PodTemplate | None = None


class ObjectMetadata(FrozenOpenBaseModel):
    # [CLASS DOC] Object metadata: name and optional namespace (missing means the manifest's
    # namespace).
    name: str
    namespace: str | None = None


def compute_api_group(api_version: str) -> str:
    # [FUNCTION DOC] The part of an api version before its last slash ("apps/v1" gives "apps");
    # the empty string for versions without a slash (the core group).
    return api_version.rpartition("/")[0]


class ObjectIdentity(NamedTuple):
    # [CLASS DOC] Full identity of an object as a sortable, hashable tuple: api group, kind,
    # namespace, name.
    api_group: str
    kind: str
    namespace: str
    name: str

    def __str__(self) -> str:
        # [FUNCTION DOC] The non-empty parts joined with slashes (so the core group is omitted).
        return "/".join(part for part in self if part)

    @property
    def key(self) -> ManifestObjectKey:
        # [FUNCTION DOC] Property: the kind-and-name key of this identity.
        return ManifestObjectKey(kind=self.kind, name=self.name)


class _ManifestObjectBase(FrozenOpenBaseModel, Generic[SpecT]):
    # [CLASS DOC] HELPER: Common shape of a manifest object, generic over its spec type: api
    # version (read from the camel-case key, default empty), kind, metadata, optional spec.
    api_version: str = Field(default="", alias="apiVersion")
    kind: str
    metadata: ObjectMetadata
    spec: SpecT | None = None

    def identity(self, *, default_namespace: str) -> ObjectIdentity:
        # [FUNCTION DOC] Builds the identity from the api group of the version, the kind, the
        # object's own namespace (or the given default if missing) and the name.
        return ObjectIdentity(
            api_group=compute_api_group(self.api_version),
            kind=self.kind,
            namespace=self.metadata.namespace or default_namespace,
            name=self.metadata.name,
        )

    @property
    def body(self) -> dict[str, Any]:
        # [FUNCTION DOC] Property: the object dumped back to a plain mapping, using the original
        # key spelling and only the fields that were present in the source (so comparisons ignore
        # defaults the model added).
        return self.model_dump(exclude_unset=True, by_alias=True)


class PodWorkloadObject(_ManifestObjectBase[WorkloadSpec]):
    # [CLASS DOC] A manifest object whose kind is one of the pod-running workload kinds, with a
    # typed workload spec and helpers for replicas, restart stamp and containers.
    kind: PodWorkloadKind

    @property
    def replicas(self) -> int | None:
        # [FUNCTION DOC] Property: the spec's replica count, or none if there is no spec or no
        # count.
        return self.spec.replicas if self.spec is not None else None

    @property
    def restart_at(self) -> str | None:
        # [FUNCTION DOC] Property: the value of the restart-at annotation on the pod template, or
        # none if the template, its metadata or the annotation is missing.
        template = self._pod_template
        if template is None or template.metadata is None:
            return None
        return template.metadata.annotations.get(RESTART_AT_ANNOTATION)

    @property
    def containers(self) -> list[Container]:
        # [FUNCTION DOC] Property: a new list of the pod template's containers; empty if no
        # template or spec.
        pod = template.spec if (template := self._pod_template) is not None else None
        return list(pod.containers) if pod is not None else []

    def container_named(self, container: str) -> Container | None:
        # [FUNCTION DOC] The container with that name, none if absent; asserts there is at most one.
        found = [described for described in self.containers if described.name == container]
        assert (
            len(found) <= 1
        ), f"{self.kind}/{self.metadata.name} declares {len(found)} containers named {container!r}"
        return found[0] if found else None

    @property
    def _pod_template(self) -> PodTemplate | None:
        # [FUNCTION DOC] HELPER: Property: the spec's pod template, or none without a spec.
        return self.spec.template if self.spec is not None else None


class GeneralManifestObject(_ManifestObjectBase[dict[str, Any]]):
    # [CLASS DOC] Any other manifest object; its spec is kept as a free-form mapping.
    pass


ManifestObject = PodWorkloadObject | GeneralManifestObject


class Manifest(FrozenStrictBaseModel):
    # [CLASS DOC] A rendered release: the default namespace and the list of objects (each a pod
    # workload or a general object, tried in that order during validation).
    namespace: str
    objects: list[ManifestObject]

    @classmethod
    def parse(cls, rendered: str, *, namespace: str) -> Manifest:
        # [FUNCTION DOC] Class method. Parses a multi-document yaml text (safe loader), drops empty
        # documents, and validates the rest into objects. Invalid yaml or objects raise.
        return cls(namespace=namespace, objects=[document for document in yaml.safe_load_all(rendered) if document])

    @property
    def by_identity(self) -> dict[ObjectIdentity, ManifestObject]:
        # [FUNCTION DOC] Property, recomputed each access: maps identity to object in listing
        # order; asserts no two objects share an identity.
        identified: dict[ObjectIdentity, ManifestObject] = {}
        for described in self.objects:
            identity = described.identity(default_namespace=self.namespace)
            assert identity not in identified, (
                f"two objects of this release are both {identity}, so one of them would hide the other "
                f"from the check that decides whether a relaunch may restart pods"
            )
            identified[identity] = described
        return identified

    @property
    def pod_workloads(self) -> list[PodWorkloadObject]:
        # [FUNCTION DOC] Property: the objects that are pod workloads, in listing order.
        return [described for described in self.objects if isinstance(described, PodWorkloadObject)]

    def object_keyed(self, *, key: ManifestObjectKey) -> ManifestObject | None:
        # [FUNCTION DOC] Keyword-only. The object with that kind and name (any api group and
        # namespace), none if absent; asserts at most one matches.
        matched = [described for identity, described in self.by_identity.items() if identity.key == key]
        assert len(matched) <= 1, (
            f"this release holds {len(matched)} objects keyed {key.kind}/{key.name}, so looking one up would answer "
            f"for whichever rendered first"
        )
        return matched[0] if matched else None

    def pod_workload_keyed(self, *, key: ManifestObjectKey) -> PodWorkloadObject | None:
        # [FUNCTION DOC] Keyword-only. Like the keyed lookup, but none unless the object is a pod
        # workload.
        described = self.object_keyed(key=key)
        return described if isinstance(described, PodWorkloadObject) else None

    def restart_at(self, *, object_name: str) -> str | None:
        # [FUNCTION DOC] Keyword-only. The restart stamp of the stateful set with that name, none
        # if it does not exist or has none.
        described = self.pod_workload_keyed(key=ManifestObjectKey(kind=STATEFUL_SET_KIND, name=object_name))
        return described.restart_at if described is not None else None

    def flag_value(self, flag: str, *, stateful_set: str, container: str) -> str | None:
        # [FUNCTION DOC] Finds the container of the stateful set (empty command if either is
        # missing). Asserts the command does not end with the bare flag (its value would be
        # unknown). Returns the effective (last) value of the flag in the command, or none.
        found = self._container(stateful_set=stateful_set, container=container)
        command = list(found.command) if found is not None else []
        assert not (command and command[-1] == flag), (
            f"container {container!r} of {stateful_set} ends its command with {flag}, which takes a value, so this "
            f"launch cannot tell what the installed release was told"
        )
        values = ArgvManipulator.get(command, flag)
        return values[-1] if values else None

    def state_file(self, *, stateful_set: str, container: str) -> Path | None:
        # [FUNCTION DOC] The path given by the orchestrator state-file flag in that container's
        # command, as a path; none if not named.
        named = self.flag_value(STATE_FILE_FLAG, stateful_set=stateful_set, container=container)
        return Path(named) if named is not None else None

    def _container(self, *, stateful_set: str, container: str) -> Container | None:
        # [FUNCTION DOC] HELPER: Looks up the stateful set (api group apps, this manifest's
        # namespace) by identity and returns its named container; none if the set is absent or has
        # no such container.
        identity = ObjectIdentity(
            api_group=_STATEFUL_SET_API_GROUP,
            kind=STATEFUL_SET_KIND,
            namespace=self.namespace,
            name=stateful_set,
        )
        named = self.by_identity.get(identity)
        return named.container_named(container) if named is not None else None
