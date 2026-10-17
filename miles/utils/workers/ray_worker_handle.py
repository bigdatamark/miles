# [MODULE DOC]
# Worker handle implementation for workers that are Ray actors: forwards method calls to the
# actor and translates Ray actor-death errors into the worker-unreachable error used by the
# generic worker handle contract.
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import ray

from miles.utils.workers.worker_handle import (
    _WAIT_DEAD_PROBE_INTERVAL_SECONDS,
    BaseWorkerHandle,
    WorkerUnreachableError,
)

logger = logging.getLogger(__name__)


class RayWorkerHandle(BaseWorkerHandle):
    # [CLASS DOC]
    # Handle to a worker running as a Ray actor. Any attribute not defined on the handle is
    # treated as the name of a remote actor method and returns an async callable that invokes it.
    def __init__(self, actor_handle: ray.actor.ActorHandle) -> None:
        # [FUNCTION DOC]
        # Store the Ray actor handle.
        self._actor_handle = actor_handle

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        # [FUNCTION DOC]
        # Dynamic method proxy, called only for attributes not found normally. Names that start and
        # end with double underscores raise an attribute error (so copying, pickling and protocol
        # lookups are not hijacked). Otherwise returns an async function that invokes the actor
        # method of that name remotely with the given positional and keyword arguments, awaits and
        # returns its result. A Ray actor error (the actor died) is re-raised as the worker
        # unreachable error naming the method; other errors propagate unchanged.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: The returned proxy call; see the enclosing method.
            try:
                return await getattr(self._actor_handle, name).remote(*args, **kwargs)
            except ray.exceptions.RayActorError as e:
                raise WorkerUnreachableError(f"Worker died or is unreachable when calling {name!r}: {e!r}") from e

        return call

    async def wait_ready(self, *, timeout: float, allow_server_uuid_change: bool = False) -> None:
        # [FUNCTION DOC]
        # Wait until the actor answers its built-in readiness call, for at most the timeout in
        # seconds. The server-uuid flag is accepted for interface compatibility and ignored. Raises
        # the worker-unreachable error if the actor died or the timeout elapsed.
        del allow_server_uuid_change

        try:
            await asyncio.wait_for(self._actor_handle.__ray_ready__.remote(), timeout=timeout)
        except ray.exceptions.RayActorError as e:
            raise WorkerUnreachableError(f"Worker died before becoming ready: {e!r}") from e
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise WorkerUnreachableError(f"Worker not ready within {timeout}s") from e

    async def wait_idle(self, *, timeout: float) -> None:
        # [FUNCTION DOC]
        # Not supported for Ray actors: always raises a not-implemented error, since Ray does not
        # track running calls. The timeout is unused.
        raise NotImplementedError(
            "a ray actor does not track the calls it is running, so nobody can wait for it to go idle; only the "
            "rpc communication backend answers this"
        )

    async def probe_is_dead(self) -> bool:
        # [FUNCTION DOC]
        # Probe the actor with its readiness call, bounded by the shared probe interval. Returns
        # true only when the actor is definitively dead (actor error or task error). A timeout or
        # a temporarily-unavailable actor is inconclusive (the latter is logged at info level) and
        # yields false, as does a successful answer.
        try:
            await asyncio.wait_for(
                self._actor_handle.__ray_ready__.remote(), timeout=_WAIT_DEAD_PROBE_INTERVAL_SECONDS
            )
        except ray.exceptions.ActorUnavailableError as e:
            logger.info("Worker death probe was inconclusive; the actor is temporarily unavailable: %r", e)
        except (ray.exceptions.RayActorError, ray.exceptions.RayTaskError):
            return True
        except (TimeoutError, asyncio.TimeoutError):
            pass
        return False
