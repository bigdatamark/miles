# [MODULE DOC] Compares two rendered manifests (the installed release and a proposed upgrade) to
# decide whether an upgrade is acceptable. Objects are matched by identity (api group, kind,
# namespace, name); every differing path inside a shared object is a change; objects only in the
# proposal are additions and only in the installed one are removals. A change is allowed if it is a
# replica count change of a scalable kind (leader-worker sets) where both sides have a count
# ("scaling"), or if the object's key is on the caller's whitelist; additions and removals are never
# allowed. The result also answers whether a given object would be rebuilt (anything other than
# scaling).
from __future__ import annotations

from typing import Any, Literal

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    LEADER_WORKER_SET_KIND,
    Manifest,
    ManifestObject,
    ManifestObjectKey,
    ObjectIdentity,
    PodWorkloadObject,
)
from miles.utils.pydantic_utils import FrozenStrictBaseModel

_SCALABLE_API_KINDS = frozenset({("leaderworkerset.x-k8s.io", LEADER_WORKER_SET_KIND)})
_REPLICAS_PATH = ("spec", "replicas")


class ManifestChange(FrozenStrictBaseModel):
    # [CLASS DOC] One differing path in one object: the object identity, the path as a tuple of key
    # names (list positions written as bracketed numbers), why it is allowed (scaling, whitelist or
    # none when not allowed), and a human description.
    identity: ObjectIdentity
    path: tuple[str, ...]
    allowed_by: Literal["scaling", "whitelist"] | None
    description: str


class ManifestDiffs(FrozenStrictBaseModel):
    # [CLASS DOC] Result of comparing two manifests: the changes, the added object identities and
    # the removed object identities (all default empty).
    changes: list[ManifestChange] = []
    additions: list[ObjectIdentity] = []
    removals: list[ObjectIdentity] = []

    @property
    def disallowed_changed(self) -> list[str]:
        # [FUNCTION DOC] Property: descriptions of changes that are not allowed by anything.
        return [change.description for change in self.changes if change.allowed_by is None]

    @property
    def allowed_changed(self) -> list[str]:
        # [FUNCTION DOC] Property: descriptions of changes allowed by scaling or the whitelist.
        return [change.description for change in self.changes if change.allowed_by is not None]

    @property
    def is_allowed(self) -> bool:
        # [FUNCTION DOC] Property: true only if there is no disallowed change, no addition and no
        # removal.
        return not (self.disallowed_changed or self.additions or self.removals)

    def rebuilds(self, *, key: ManifestObjectKey) -> bool:
        # [FUNCTION DOC] Keyword-only. True if an object with that kind and name is added or
        # removed, or has any change that is not a pure scaling change (whitelisted or disallowed
        # changes count as rebuilding); false otherwise.
        if any(identity.key == key for identity in (*self.additions, *self.removals)):
            return True
        return any(change.identity.key == key and change.allowed_by != "scaling" for change in self.changes)

    def summarize_allowed_changes(self) -> str:
        # [FUNCTION DOC] One line per allowed change, indented by two spaces, or "(nothing to
        # change)" when there are none.
        return "\n".join(f"  {entry}" for entry in self.allowed_changed) or "  (nothing to change)"

    def describe(self) -> str:
        # [FUNCTION DOC] Lines "changed: ..." for disallowed changes, then "added: ..." and
        # "removed: ..." for object identities, each indented by two spaces and joined by
        # newlines; "(no difference)" if there are none. Allowed changes are not listed.
        lines = []
        for label, entries in (
            ("changed", self.disallowed_changed),
            ("added", [str(identity) for identity in self.additions]),
            ("removed", [str(identity) for identity in self.removals]),
        ):
            lines += [f"  {label}: {entry}" for entry in entries]
        return "\n".join(lines) or "  (no difference)"


def diff_manifests(
    *, before: Manifest, after: Manifest, allow_diff_object_keys: frozenset[ManifestObjectKey] = frozenset()
) -> ManifestDiffs:
    # [FUNCTION DOC] Keyword-only. First asserts, for both manifests, that every whitelisted key
    # resolves to at most one object (and that identities are unique). Then matches objects by
    # identity; for each shared identity (sorted) and each differing path of the object bodies,
    # records a change; additions are identities only in the proposal, removals only in the
    # installed one (each sorted by text). The whitelist defaults to empty.
    for manifest in (before, after):
        _assert_unambiguous_keys(manifest, allow_diff_object_keys=allow_diff_object_keys)

    old = before.by_identity
    new = after.by_identity
    shared = sorted(set(old) & set(new))

    return ManifestDiffs(
        changes=[
            _compute_change(
                old[identity],
                new[identity],
                identity=identity,
                path=path,
                allow_diff_object_keys=allow_diff_object_keys,
            )
            for identity in shared
            for path in _differing_paths(old[identity].body, new[identity].body, ())
        ],
        additions=sorted(set(new) - set(old), key=str),
        removals=sorted(set(old) - set(new), key=str),
    )


def _assert_unambiguous_keys(manifest: Manifest, *, allow_diff_object_keys: frozenset[ManifestObjectKey]) -> None:
    # [FUNCTION DOC] HELPER: Looks up each whitelisted key (sorted by kind, name) in the manifest
    # purely for the side effect of its assertion that no key matches more than one object.
    for key in sorted(allow_diff_object_keys, key=lambda entry: (entry.kind, entry.name)):
        manifest.object_keyed(key=key)


def _compute_change(
    old: ManifestObject,
    new: ManifestObject,
    *,
    identity: ObjectIdentity,
    path: tuple[str, ...],
    allow_diff_object_keys: frozenset[ManifestObjectKey],
) -> ManifestChange:
    # [FUNCTION DOC] HELPER: Decides why the path change is allowed (if at all). The description is
    # "<identity>: replicas <old> -> <new>" for an allowed change at the replicas path, otherwise
    # "<identity>: <dotted path>" ("(root)" for an empty path).
    allowed_by = _compute_allowed_by(
        old, new, identity=identity, path=path, allow_diff_object_keys=allow_diff_object_keys
    )
    if allowed_by is not None and path == _REPLICAS_PATH:
        description = f"{identity}: replicas {_replicas(old)} -> {_replicas(new)}"
    else:
        description = f"{identity}: {_describe_path(path)}"
    return ManifestChange(identity=identity, path=path, allowed_by=allowed_by, description=description)


def _compute_allowed_by(
    old: ManifestObject,
    new: ManifestObject,
    *,
    identity: ObjectIdentity,
    path: tuple[str, ...],
    allow_diff_object_keys: frozenset[ManifestObjectKey],
) -> Literal["scaling", "whitelist"] | None:
    # [FUNCTION DOC] HELPER: "scaling" if the change is a replica count change of a scalable kind;
    # else "whitelist" if the object's key is whitelisted; else none. Scaling takes precedence.
    if _is_scaling(old, new, identity=identity, path=path):
        return "scaling"
    if identity.key in allow_diff_object_keys:
        return "whitelist"
    return None


def _is_scaling(old: ManifestObject, new: ManifestObject, *, identity: ObjectIdentity, path: tuple[str, ...]) -> bool:
    # [FUNCTION DOC] HELPER: True only for the leader-worker-set kind (in its api group), at the
    # replicas path, when both old and new objects have a replica count.
    if (identity.api_group, identity.kind) not in _SCALABLE_API_KINDS or path != _REPLICAS_PATH:
        return False
    return _replicas(old) is not None and _replicas(new) is not None


def _replicas(described: ManifestObject) -> int | None:
    # [FUNCTION DOC] HELPER: The replica count for pod workload objects (none when unset); none
    # for other object types.
    return described.replicas if isinstance(described, PodWorkloadObject) else None


def _describe_path(path: tuple[str, ...]) -> str:
    # [FUNCTION DOC] HELPER: The path elements joined with dots, or "(root)" when empty.
    return ".".join(path) or "(root)"


def _differing_paths(old: Any, new: Any, path: tuple[str, ...]) -> list[tuple[str, ...]]:
    # [FUNCTION DOC] HELPER: Recursive structural diff returning the paths (tuples of keys) where
    # two json-like values differ. Equal values: none. Two mappings: for each key in the sorted
    # union, a key missing on either side yields that key's path, otherwise recurse. Two lists of
    # equal length: recurse per position, naming positions as bracketed indices. Anything else
    # (different types, lists of different length, differing scalars) yields the current path as one
    # difference, without looking inside.
    if old == new:
        return []

    if isinstance(old, dict) and isinstance(new, dict):
        differences = []
        for key in sorted(set(old) | set(new)):
            if key not in old or key not in new:
                differences.append((*path, key))
            else:
                differences += _differing_paths(old[key], new[key], (*path, key))
        return differences

    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [
            difference
            for index, (old_item, new_item) in enumerate(zip(old, new, strict=True))
            for difference in _differing_paths(old_item, new_item, (*path, f"[{index}]"))
        ]

    return [path]
