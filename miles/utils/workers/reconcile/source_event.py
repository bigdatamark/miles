# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER:
# Event types exchanged between a source and the object store. Objects are identified by an
# object key (string) and grouped under a parent key (string). A source stream must always
# begin with a replace event holding the whole current world; afterwards it sends upserts and
# deletes. The source event union and the source watch function type (a no-argument callable
# returning an async stream of events) are defined here.
from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

ObjectKey = str
ParentKey = str


@dataclass(frozen=True)
class UpsertEvent:
    # [CLASS DOC] HELPER: Immutable event: the object with this key was added or changed and is
    # now the given object.
    key: ObjectKey
    obj: Any


@dataclass(frozen=True)
class DeleteEvent:
    # [CLASS DOC] HELPER: Immutable event: the object with this key was removed. Carries the last
    # known object (possibly none), used to work out the parent when the key is not cached.
    key: ObjectKey
    last_obj: Any


@dataclass(frozen=True)
class ReplaceEvent:
    # [CLASS DOC] HELPER: Immutable event holding the complete current world as a mapping from
    # object key to object; anything not in it is considered gone.
    objects: dict[ObjectKey, Any]


SourceEvent = UpsertEvent | DeleteEvent | ReplaceEvent

SourceWatchFn = Callable[[], AsyncGenerator[SourceEvent, None]]
