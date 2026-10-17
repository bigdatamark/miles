# [MODULE DOC]
# Defines the backend capability abstraction: the single entry point through which the rest of the
# system obtains cluster-backend-specific objects (worker providers and cell operations) without
# knowing whether the run is deployed on Kubernetes or Ray. Also defines a lazy wrapper that
# defers building the real capability until first use. Concrete implementations live in the
# sibling kubernetes and ray modules; the factory module picks one.
from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miles.utils.workers.cell_operations.base import BaseCellOperations
    from miles.utils.workers.worker_provider.base import BaseWorkerProvider


class BackendCapability(abc.ABC):
    # [CLASS DOC]
    # Abstract factory of the backend-specific pieces a run needs: a provider for dynamically
    # managed pools, a provider for statically addressed pools, and the cell operations object.
    # Subclasses exist per cluster backend (Kubernetes, Ray).
    # [FUNCTION DOC]
    # Abstract (one-line body, so this block sits above the decorator). Build a worker provider
    # that tracks the lifecycle of workers belonging to the given pools (identified by pool id).
    @abc.abstractmethod
    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider: ...

    # [FUNCTION DOC]
    # Abstract (block sits above the decorator). Build a worker provider for one pool whose
    # workers are addressed statically (fixed addresses rather than discovered).
    @abc.abstractmethod
    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider: ...

    # [FUNCTION DOC]
    # Abstract (block sits above the decorator). Return the object used to list, suspend, resume
    # and fault-inject cells.
    @abc.abstractmethod
    def cell_operations(self) -> BaseCellOperations: ...


class DeferredBackendCapability(BackendCapability):
    # [CLASS DOC]
    # Lazy proxy for a backend capability. It stores a zero-argument creator and builds the real
    # capability on first use, then caches it; every operation is forwarded unchanged to it.
    # Not thread-safe: concurrent first calls could build the capability more than once.
    def __init__(self, *, create: Callable[[], BackendCapability]) -> None:
        # [FUNCTION DOC]
        # Remember the creator callable and start with no cached capability. Nothing is built yet.
        self._create = create
        self._inner: BackendCapability | None = None

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        # [FUNCTION DOC]
        # Resolve the real capability (building it if needed) and delegate with the same pool ids.
        return self._resolve().dynamic_worker_provider(pool_ids=pool_ids)

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        # [FUNCTION DOC]
        # Resolve the real capability (building it if needed) and delegate with the same pool id.
        return self._resolve().static_worker_provider(pool_id=pool_id)

    def cell_operations(self) -> BaseCellOperations:
        # [FUNCTION DOC]
        # Resolve the real capability (building it if needed) and return its cell operations.
        return self._resolve().cell_operations()

    def _resolve(self) -> BackendCapability:
        # [FUNCTION DOC] HELPER: Return the cached real capability, calling the creator once on
        # first use and caching the result. Exceptions from the creator propagate and nothing is
        # cached in that case.
        if self._inner is None:
            self._inner = self._create()
        return self._inner
