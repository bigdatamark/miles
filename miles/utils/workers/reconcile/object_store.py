# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# In-memory cache of the objects delivered by a source, indexed by object key and grouped under
# parent keys. Applies replace, upsert and delete events and reports which parent keys were
# affected, so only those need reconciling. A key mapping function decides each object's parent;
# without one the object key is its own parent key.
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from miles.utils.workers.reconcile.source_event import (
    DeleteEvent,
    ObjectKey,
    ParentKey,
    ReplaceEvent,
    SourceEvent,
    UpsertEvent,
)

logger = logging.getLogger(__name__)

KeyMapFn = Callable[[Any], ParentKey | None]


@dataclass(frozen=True)
class _CachedObject:
    # [CLASS DOC] HELPER: Immutable cache entry: the object and the parent key it was mapped to.
    obj: Any
    parent: ParentKey


class ObjectStore:
    # [CLASS DOC]
    # Cache mapping object key to the object and its parent key. Not thread-safe; used from the
    # event loop only. Objects the key mapping function maps to none, or fails on, are never stored.
    def __init__(self, *, key_map: KeyMapFn | None) -> None:
        # [FUNCTION DOC] Store the optional key mapping function (object to parent key or none)
        # and start with an empty cache.
        self._key_map = key_map
        self._cache: dict[ObjectKey, _CachedObject] = {}

    def get_by_parent(self, parent_key: ParentKey) -> list[Any]:
        # [FUNCTION DOC] Return the cached objects whose parent is the given key, ordered by sorted
        # object key. Empty if none.
        return [self._cache[key].obj for key in sorted(self._cache) if self._cache[key].parent == parent_key]

    def parent_keys(self) -> set[ParentKey]:
        # [FUNCTION DOC] Return the set of parent keys of all cached objects.
        return {entry.parent for entry in self._cache.values()}

    def __contains__(self, key: ObjectKey) -> bool:
        # [FUNCTION DOC] Membership test: true iff an object with this key is cached.
        return key in self._cache

    def handle_event(self, event: SourceEvent) -> set[ParentKey]:
        # [FUNCTION DOC] Apply a replace, upsert or delete event to the cache and return the set
        # of parent keys whose objects may have changed (including old parents of moved objects).
        # Any other event kind raises an assertion error.
        match event:
            case ReplaceEvent():
                return self._handle_replace(event)
            case UpsertEvent():
                return self._handle_upsert(event)
            case DeleteEvent():
                return self._handle_delete(event)
            case _:
                raise AssertionError(f"Unknown source event {event=}")

    def _handle_replace(self, event: ReplaceEvent) -> set[ParentKey]:
        # [FUNCTION DOC] HELPER: Make the cache equal to the event's whole world: compute each
        # object's parent; upsert every object that has a parent; delete every cached key that is
        # not among the mapped ones (including objects whose mapping now gives none). Returns the
        # union of all affected parents. Unchanged objects are still upserted and so still mark
        # their parent as affected.
        parents = {key: self._parent_key_or_none(key=key, obj=obj) for key, obj in event.objects.items()}
        mapped = {key: obj for key, obj in event.objects.items() if parents[key] is not None}

        affected_parents: set[ParentKey] = set()
        for key, obj in mapped.items():
            affected_parents |= self._apply_upsert(key=key, obj=obj, parent=parents[key])
        for key in [key for key in self._cache if key not in mapped]:
            affected_parents |= self._apply_delete(key=key, last_obj=None)
        return affected_parents

    def _handle_upsert(self, event: UpsertEvent) -> set[ParentKey]:
        # [FUNCTION DOC] HELPER: Map the object to its parent. If it has one, store it (see the
        # upsert helper). If it has none and the key is cached, remove the cached entry (the
        # object moved out of scope) and report its old parent; if not cached, nothing changes.
        parent = self._parent_key_or_none(key=event.key, obj=event.obj)
        if parent is not None:
            return self._apply_upsert(key=event.key, obj=event.obj, parent=parent)
        if event.key not in self._cache:
            return set()
        return self._apply_delete(key=event.key, last_obj=None)

    def _handle_delete(self, event: DeleteEvent) -> set[ParentKey]:
        # [FUNCTION DOC] HELPER: Delete the key, passing along the event's last known object so the
        # parent can still be derived when the key is not cached.
        return self._apply_delete(key=event.key, last_obj=event.last_obj)

    def _apply_upsert(self, *, key: ObjectKey, obj: Any, parent: ParentKey) -> set[ParentKey]:
        # [FUNCTION DOC] HELPER: Store the object under the key with the given parent, replacing any
        # previous entry. Returns the new parent, plus the previous parent if there was an entry
        # (it may be the same parent).
        previous = self._cache.get(key)
        self._cache[key] = _CachedObject(obj=obj, parent=parent)
        return {parent} if previous is None else {parent, previous.parent}

    def _apply_delete(self, *, key: ObjectKey, last_obj: Any) -> set[ParentKey]:
        # [FUNCTION DOC] HELPER: Remove the key from the cache. If it was cached, report its parent.
        # If not cached: with no last known object, log a warning and report nothing; otherwise
        # derive the parent from the last object (reporting nothing if it maps to none) and
        # report that parent, so a delete for an unknown key still triggers reconciliation.
        removed = self._cache.pop(key, None)
        parent = removed.parent if removed is not None else None
        if parent is None:
            if last_obj is None:
                logger.warning(f"ObjectStore dropping a delete it cannot attribute to a parent {key=}")
                return set()
            parent = self._parent_key_or_none(key=key, obj=last_obj)
            if parent is None:
                return set()
        return {parent}

    def _parent_key_or_none(self, *, key: ObjectKey, obj: Any) -> ParentKey | None:
        # [FUNCTION DOC] HELPER: The parent key for an object: the object's own key when no mapping
        # function is configured, else the function's result (which may be none). If the function
        # raises an ordinary exception, log an error and return none, so the object is dropped.
        try:
            return key if self._key_map is None else self._key_map(obj)
        except Exception:
            logger.error(f"ObjectStore dropping an object whose key_map failed {key=}", exc_info=True)
            return None
