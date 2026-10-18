# [MODULE DOC] HELPER: Builders of restart-themed cluster snapshots and evidence for the
# hot-restart assertion tests: a snapshot from pod uids and workload restart stamps, an evidence
# object from snapshots and records, a quiet (never restarted) run snapshot, a restarted snapshot
# whose orchestrator and rollout executor pods are replaced while the trainer survives, and a
# three-snapshot history with two restarts. Not a test module.
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    STATEFUL_SET_KIND,
    ClusterSnapshot,
    PodFact,
    WorkloadFact,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ENGINE_POOL,
    ORCHESTRATOR,
    RELEASE,
    ROLLOUT_EXECUTOR,
    TRAINER,
)

ENGINE_POOL_POD: str = f"{ENGINE_POOL}-0-1"


def restart_snapshot(
    *, uid_of_pod: dict[str, str], stamp_of_workload: dict[str, str | None], boot_uuid: str | None = "boot-a"
) -> ClusterSnapshot:
    # [FUNCTION DOC] HELPER: Builds a snapshot from a mapping of pod name to uid (pods sorted by
    # name, restart count 0) and a mapping of stateful set name to restart stamp (sorted by name).
    # A workload with no stamp is generation 1 with fingerprint "template-base" and no restart
    # stamp; one with a stamp is generation 2 with fingerprint "template-" plus the stamp. The
    # trainer boot uuid defaults to "boot-a" (none means unreachable).
    return ClusterSnapshot(
        pods=tuple(PodFact(name=name, uid=uid, restart_count=0) for name, uid in sorted(uid_of_pod.items())),
        workloads=tuple(
            WorkloadFact(
                kind=STATEFUL_SET_KIND,
                name=name,
                generation=1 if stamp is None else 2,
                pod_template_fingerprint="template-base" if stamp is None else f"template-{stamp}",
                restart_at=stamp,
            )
            for name, stamp in sorted(stamp_of_workload.items())
        ),
        trainer_boot_uuid=boot_uuid,
    )


def evidence_of(
    *,
    snapshots: list[ClusterSnapshot],
    records: list[HotRestartRecord],
    observation_attempts: int | None = None,
    observation_failures: int = 0,
) -> HotRestartEvidence:
    # [FUNCTION DOC] HELPER: Builds hot-restart evidence from snapshots and records (as tuples)
    # for the demo release. Observation attempts default to the number of snapshots when not
    # given (so no failed attempts are implied); observation failures default to 0.
    return HotRestartEvidence(
        records=tuple(records),
        snapshots=tuple(snapshots),
        release=RELEASE,
        observation_attempts=len(snapshots) if observation_attempts is None else observation_attempts,
        observation_failures=observation_failures,
    )


def quiet_run_snapshot(*, uid: str = "uid-o-1") -> ClusterSnapshot:
    # [FUNCTION DOC] HELPER: Snapshot of a never-restarted run: orchestrator pod (uid default
    # "uid-o-1"), rollout executor pod ("uid-r-1") and trainer pod ("uid-t"), no workload stamps.
    return restart_snapshot(
        uid_of_pod={f"{ORCHESTRATOR}-0": uid, f"{ROLLOUT_EXECUTOR}-0": "uid-r-1", f"{TRAINER}-0": "uid-t"},
        stamp_of_workload={ORCHESTRATOR: None, ROLLOUT_EXECUTOR: None, TRAINER: None},
    )


def restarted_snapshot(*, stamp: str, uid_suffix: str, boot_uuid: str | None = "boot-a") -> ClusterSnapshot:
    # [FUNCTION DOC] HELPER: Snapshot after a take-over: orchestrator and rollout executor pods
    # have uids ending in the given suffix and their workloads carry the given restart stamp;
    # the trainer pod keeps uid "uid-t" and its workload has no stamp. Boot uuid defaults to
    # "boot-a".
    return restart_snapshot(
        uid_of_pod={
            f"{ORCHESTRATOR}-0": f"uid-o-{uid_suffix}",
            f"{ROLLOUT_EXECUTOR}-0": f"uid-r-{uid_suffix}",
            f"{TRAINER}-0": "uid-t",
        },
        stamp_of_workload={ORCHESTRATOR: stamp, ROLLOUT_EXECUTOR: stamp, TRAINER: None},
        boot_uuid=boot_uuid,
    )


def two_restarts() -> list[ClusterSnapshot]:
    # [FUNCTION DOC] HELPER: Returns a quiet snapshot followed by one restarted with stamp "t1"
    # (uid suffix 2) and one with stamp "t2" (suffix 3).
    return [
        quiet_run_snapshot(),
        restarted_snapshot(stamp="t1", uid_suffix="2"),
        restarted_snapshot(stamp="t2", uid_suffix="3"),
    ]
