# [MODULE DOC]
# Test double for the backend capability interface, which hands out worker providers and cell
# operations. Provides one fake class whose providers and operations are injected by the test and
# which records the pool ids requested.
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


class FakeBackendCapability(BackendCapability):
    # [CLASS DOC] Fake backend capability. Returns whatever dynamic provider, static provider and cell
    # operations object it was constructed with (each optional), asserting presence on use, and records
    # every request: a list of pool id lists for dynamic providers and a list of pool ids for static
    # ones.
    def __init__(
        self,
        *,
        cells_provider: Any = None,
        static_provider: Any = None,
        cell_operations: Any = None,
    ) -> None:
        # [FUNCTION DOC] Stores the keyword-only arguments (cells provider, static provider, cell
        # operations; all default to none) and initializes both request logs as empty lists. The base
        # class initializer is not called.
        self.cells_provider = cells_provider
        self.static_provider = static_provider
        self.operations = cell_operations
        self.requested_pool_ids: list[list[str]] = []
        self.requested_static_pool_ids: list[str] = []

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        # [FUNCTION DOC] Appends a list copy of the pool ids to the request log, then returns the
        # configured cells provider; raises an assertion error if none was configured.
        self.requested_pool_ids.append(list(pool_ids))
        assert self.cells_provider is not None, "this capability was built without a cells provider"
        return self.cells_provider

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        # [FUNCTION DOC] Appends the pool id to the static request log, then returns the configured
        # static provider; raises an assertion error if none was configured.
        self.requested_static_pool_ids.append(pool_id)
        assert self.static_provider is not None, "this capability was built without a static provider"
        return self.static_provider

    def cell_operations(self) -> Any:
        # [FUNCTION DOC] Returns the configured cell operations object; raises an assertion error if
        # none was configured.
        assert self.operations is not None, "this capability was built without cell operations"
        return self.operations
