# [MODULE DOC] HELPER: Shared constants and small builders for the hot-restart tests: the fixed
# namespace and release name of a demo run, the names of its orchestrator, rollout executor,
# trainer controller actor and engine pool workloads, and constructors for cluster snapshots, pod
# facts and workload facts with sensible defaults. Not a test module; pure data builders.
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    STATEFUL_SET_KIND,
    ClusterSnapshot,
    PodFact,
    WorkloadFact,
)

NAMESPACE: str = "rl"
RELEASE: str = "miles-run-demo-all"
ORCHESTRATOR: str = "miles-run-demo-all-orchestrator"
ROLLOUT_EXECUTOR: str = "miles-run-demo-all-rollout-executor"
TRAINER: str = "miles-run-demo-all-trainer-controller-actor"
ENGINE_POOL: str = "miles-run-demo-all-rollout-engine"


def cluster_snapshot(
    *,
    pods: list[PodFact],
    workloads: list[WorkloadFact],
    trainer_boot_uuid: str | None = "boot-a",
    reads_missing: tuple[str, ...] = (),
) -> ClusterSnapshot:
    # [FUNCTION DOC] HELPER: Builds an immutable cluster snapshot from the given pod and workload
    # lists (converted to tuples), trainer boot uuid (default "boot-a"; none means the trainer
    # could not be reached) and a tuple of resource kinds whose reads were missing (default none).
    return ClusterSnapshot(
        pods=tuple(pods),
        workloads=tuple(workloads),
        trainer_boot_uuid=trainer_boot_uuid,
        reads_missing=reads_missing,
    )


def pod_fact(name: str, *, uid: str, restart_count: int = 0) -> PodFact:
    # [FUNCTION DOC] HELPER: Builds a pod fact with the given name, uid and restart count
    # (default 0).
    return PodFact(name=name, uid=uid, restart_count=restart_count)


def workload_fact(
    name: str,
    *,
    kind: str = STATEFUL_SET_KIND,
    generation: int = 1,
    pod_template_fingerprint: str = "template-a",
    restart_at: str | None = None,
) -> WorkloadFact:
    # [FUNCTION DOC] HELPER: Builds a workload fact with the given name and keyword fields: kind
    # (default stateful set), generation (default 1), pod template fingerprint (default
    # "template-a") and restart-at stamp (default none, meaning never restarted).
    return WorkloadFact(
        kind=kind,
        name=name,
        generation=generation,
        pod_template_fingerprint=pod_template_fingerprint,
        restart_at=restart_at,
    )
