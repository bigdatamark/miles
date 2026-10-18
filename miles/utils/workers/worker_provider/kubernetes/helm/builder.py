# [MODULE DOC]
# Builds the Kubernetes backend capability for a run deployed by the helm chart, splitting the
# worker specs into dynamically observed pools and statically addressed pools.
from __future__ import annotations

from miles.utils.workers.backend_capability.kubernetes import KubernetesBackendCapability
from miles.utils.workers.cell_operations.kubernetes import KubernetesCellOperations
from miles.utils.workers.reconcile.loop import DEFAULT_RESYNC_PERIOD
from miles.utils.workers.worker_provider.kubernetes.core.provider import KubernetesRunInfo, KubernetesWorkerProvider
from miles.utils.workers.worker_provider.kubernetes.helm import env
from miles.utils.workers.worker_spec import BaseWorkerSpec


def compute_helm_backend_capability(*, specs: list[BaseWorkerSpec]) -> KubernetesBackendCapability:
    # [FUNCTION DOC]
    # Read the release and namespace from the environment (asserting both are available; see the env
    # module), build the run info with the chart's instance label selector, the default label keys and
    # the specs of dynamic pools by name, and return the capability. Specs that do not declare a
    # dynamic pool become the static specs. Cell operations are backed by a provider over all
    # dynamic pools (sorted) with the default resync period; no watch is started here.
    release = env.current_release()
    run = KubernetesRunInfo(
        namespace=env.current_namespace(),
        label_selector=f"{env.INSTANCE_LABEL}={release}",
        label_keys=env.DEFAULT_LABEL_KEYS,
        specs={spec.name: spec for spec in specs if _declares_dynamic_pool(spec)},
    )

    return KubernetesBackendCapability(
        run=run,
        release=release,
        static_specs={spec.name: spec for spec in specs if not _declares_dynamic_pool(spec)},
        cell_operations=KubernetesCellOperations(
            provider=KubernetesWorkerProvider(
                run=run, pool_ids=sorted(run.specs), resync_period=DEFAULT_RESYNC_PERIOD
            ),
            namespace=run.namespace,
        ),
    )


def _declares_dynamic_pool(spec: BaseWorkerSpec) -> bool:
    # [FUNCTION DOC] HELPER: A pool is dynamic (its pods are observed in the cluster) iff its cells
    # use any gpu slots; gpu-less pools are addressed statically by name.
    return spec.scheduling.gpus_per_cell() > 0
