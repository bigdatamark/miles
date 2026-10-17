# [MODULE DOC]
# Kubernetes implementation of the backend capability, wiring a Kubernetes run description to
# the Kubernetes dynamic worker provider, the static worker provider, and cell operations.
from __future__ import annotations

from collections.abc import Sequence

from miles.utils.workers.backend_capability.base import BackendCapability
from miles.utils.workers.cell_operations.base import BaseCellOperations
from miles.utils.workers.reconcile.loop import DEFAULT_RESYNC_PERIOD
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo, KubernetesWorkerProvider
from miles.utils.workers.worker_provider.static import StaticWorkerProvider
from miles.utils.workers.worker_spec import BaseWorkerSpec


class KubernetesBackendCapability(BackendCapability):
    # [CLASS DOC]
    # Backend capability for a run deployed through a Kubernetes release. Holds the run info (all
    # deployed pool specs), the release name, the specs of pools addressed statically (keyed by
    # pool id) and a ready-made cell operations object.
    def __init__(
        self,
        *,
        run: KubernetesRunInfo,
        release: str,
        static_specs: dict[str, BaseWorkerSpec],
        cell_operations: BaseCellOperations,
    ) -> None:
        # [FUNCTION DOC]
        # Store the run info, release name, static pool specs and cell operations. No I/O.
        self._run = run
        self._release = release
        self._static_specs = static_specs
        self._cell_operations = cell_operations

    def dynamic_worker_provider(self, *, pool_ids: Sequence[str]) -> BaseWorkerProvider:
        # [FUNCTION DOC]
        # Validate that every requested pool id is deployed by this run (asserts otherwise, listing
        # the unknown ids and the deployed ones), then create a Kubernetes worker provider for
        # those pools using the default resync period of the reconcile loop.
        unknown = [name for name in pool_ids if name not in self._run.specs]
        assert not unknown, f"{unknown} are not pool_ids of this run, which deploys {sorted(self._run.specs)}"
        return KubernetesWorkerProvider(run=self._run, pool_ids=list(pool_ids), resync_period=DEFAULT_RESYNC_PERIOD)

    def static_worker_provider(self, *, pool_id: str) -> BaseWorkerProvider:
        # [FUNCTION DOC]
        # Look up the static spec for the pool id; asserts (naming the statically addressed pools)
        # if the pool is not static. Returns a static worker provider built from the release name
        # and the spec, with no cluster watching.
        spec = self._static_specs.get(pool_id)
        assert (
            spec is not None
        ), f"{pool_id} is not a static pool of this run, which addresses {sorted(self._static_specs)} statically"
        return StaticWorkerProvider.of_release(release=self._release, spec=spec)

    def cell_operations(self) -> BaseCellOperations:
        # [FUNCTION DOC]
        # Return the cell operations object supplied at construction.
        return self._cell_operations
