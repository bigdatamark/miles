# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# A first-in first-out queue of hashable keys that ignores a key already waiting, so a key that
# changes many times before being processed is processed once. Used by the reconcile loop.
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Hashable
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT", bound=Hashable)


class WorkQueue(Generic[KeyT]):
    # [CLASS DOC]
    # Deduplicating asynchronous queue meant for one consumer. A key currently being processed
    # (already taken) can be added again and will be handed out again. Shutdown is permanent.
    def __init__(self) -> None:
        # [FUNCTION DOC] Create an empty queue, an unset wakeup signal and the not-shut-down state.
        self._keys: deque[KeyT] = deque()
        self._wakeup = asyncio.Event()
        self._shutdown = False

    def add(self, key: KeyT) -> None:
        # [FUNCTION DOC] Ignored after shutdown. Otherwise appends the key unless an equal key is
        # already waiting (a linear scan), and wakes a waiting consumer.
        if self._shutdown:
            return
        if key not in self._keys:
            self._keys.append(key)
        self._wakeup.set()

    async def get(self) -> KeyT | None:
        # [FUNCTION DOC] Return the oldest waiting key, waiting for one to be added if the queue is
        # empty. Returns none once the queue has been shut down, even if keys are still waiting.
        # Designed for a single consumer: the wakeup signal is shared.
        while not self._shutdown:
            if self._keys:
                return self._keys.popleft()
            self._wakeup.clear()
            await self._wakeup.wait()
        return None

    def shutdown(self) -> None:
        # [FUNCTION DOC] Mark the queue shut down and wake the consumer so a pending get returns
        # none. Later adds are ignored.
        self._shutdown = True
        self._wakeup.set()
