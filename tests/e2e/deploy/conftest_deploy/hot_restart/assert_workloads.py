# [MODULE DOC]
# HELPER: Assertions about which Kubernetes workloads and pods a hot restart touches. Using the
# list of cluster snapshots in the hot-restart evidence, it verifies that only the workloads
# expected to be rolled by a hot restart (the orchestration script's workloads) lost pods,
# changed pod template and carried restart stamps, that each of them was stamped once per
# restart, and that no pod belongs to an unknown workload. Pod-to-workload attribution is by name:
# a pod belongs to the longest workload name that is a prefix of the pod name followed by a dash.
from collections.abc import Iterable, Sequence

from tests.e2e.deploy.conftest_deploy.hot_restart.assert_process import (
    assert_baseline_read_before_first_take_over,
    assert_run_watched_closely,
    assert_trainer_not_rebooted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    ClusterSnapshot,
    compute_hot_restart_workloads,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence


# ============================ what a take-over rolls ==========================


def assert_take_overs_replaced_only_script(
    evidence: HotRestartEvidence, *, num_restarts: int, minimum_restarts: int
) -> None:
    # [FUNCTION DOC] Top-level check that take-overs replaced only the orchestration script.
    # Asserts the number of restarts is at least the minimum, then runs in order: the
    # watched-closely check, the baseline-before-first-take-over check, the only-orchestration-
    # restarted check, and the trainer-not-rebooted check. Raises assertion errors.
    assert num_restarts >= minimum_restarts, (
        f"the run was taken over {num_restarts} time(s) of the {minimum_restarts} it needs, so it ran to the end "
        f"without its orchestration script ever being replaced"
    )

    assert_run_watched_closely(evidence)
    assert_baseline_read_before_first_take_over(evidence)
    assert_only_orchestration_restarted(evidence, num_restarts=num_restarts)
    assert_trainer_not_rebooted(evidence)


def assert_only_orchestration_restarted(evidence: HotRestartEvidence, *, num_restarts: int) -> None:
    # [FUNCTION DOC] Compute the expected set of rolled workloads from the evidence's release, then
    # assert: (1) no pod seen in a whole-release snapshot is unattributed to a workload; (2) the
    # workloads with replaced pods (recreated uid, container restart, or pod set changing between
    # snapshots) are exactly the expected set; (3) the workloads whose pod template fingerprint
    # or restart stamp changed are exactly the expected set; (4) each expected workload was
    # observed with exactly num_restarts distinct restart stamps; (5) no workload outside the
    # expected set carries any stamp. Raises assertion errors with explanatory messages.
    expected = compute_hot_restart_workloads(evidence.release)

    unattributed = _compute_unattributed_pod_names(evidence.snapshots)
    assert not unattributed, (
        f"the pods {sorted(unattributed)} belong to no workload this run listed, so nothing would notice them "
        f"being replaced; every pod has to be owned by a statefulset or leaderworkerset of the release"
    )

    replaced = _compute_workloads_with_replaced_pods(evidence.snapshots)
    assert set(replaced) == expected, (
        f"a hot restart replaces the pods of {sorted(expected)} and leaves every other pod running; these lost a "
        f"pod instead: {replaced}"
    )

    rolled = _compute_workloads_with_changed_template(evidence.snapshots)
    assert rolled == expected, (
        f"only {sorted(expected)} may be rolled by a hot restart, and the pod template of {sorted(rolled)} "
        f"changed: the relaunch rewrote the run's trainers or engines"
    )

    stamps_of_workload = _compute_restart_stamps_of_workload(evidence.snapshots)
    for workload in sorted(expected):
        assert len(stamps := stamps_of_workload[workload]) == num_restarts, (
            f"{workload} was observed carrying {sorted(stamps)}, and {num_restarts} hot restart(s) stamp one value "
            f"each, so a restart either never reached this workload or never landed"
        )
    unexpected = {
        name: sorted(stamps) for name, stamps in stamps_of_workload.items() if stamps and name not in expected
    }
    assert (
        not unexpected
    ), f"a hot restart stamps exactly the two pod templates it replaces, and these carry a stamp too: {unexpected}"


# ========================== what the snapshots say ============================


def _compute_unattributed_pod_names(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    # [FUNCTION DOC] HELPER: Return names of pods, in snapshots that describe the whole release,
    # that match none of that snapshot's workload names.
    return {
        pod.name
        for snapshot in snapshots
        if snapshot.describes_whole_release
        for pod in snapshot.pods
        if _compute_workload_of_pod(pod.name, workloads=snapshot.workload_names) is None
    }


def _compute_workloads_with_replaced_pods(snapshots: Sequence[ClusterSnapshot]) -> dict[str, list[str]]:
    # [FUNCTION DOC] HELPER: Return a mapping from workload name to a sorted list of human-readable
    # reasons it lost a pod, for workloads with any. Across all snapshots it records, per pod
    # name, the set of uids and of container restart counts, and per workload the set of pod name
    # sets seen. Reasons: a pod seen with more than one uid (recreated), a pod with more than one
    # restart count (container restarted), a workload whose set of pod names differs between
    # snapshots (pods came and went; note this also counts partial snapshots). Pods matching no
    # workload are ignored here. Result keys are sorted.
    workloads = sorted({name for snapshot in snapshots for name in snapshot.workload_names})
    uids_of_pod: dict[str, set[str]] = {}
    restart_counts_of_pod: dict[str, set[int]] = {}
    pod_names_of_workload: dict[str, set[frozenset[str]]] = {}

    for snapshot in snapshots:
        seen_of_workload: dict[str, set[str]] = {one: set() for one in workloads}
        for pod in snapshot.pods:
            if (workload := _compute_workload_of_pod(pod.name, workloads=workloads)) is None:
                continue
            seen_of_workload[workload].add(pod.name)
            uids_of_pod.setdefault(pod.name, set()).add(pod.uid)
            restart_counts_of_pod.setdefault(pod.name, set()).add(pod.restart_count)
        for workload, seen in seen_of_workload.items():
            pod_names_of_workload.setdefault(workload, set()).add(frozenset(seen))

    reasons_of_workload: dict[str, list[str]] = {}
    for pod_name in sorted(uids_of_pod):
        workload = _compute_workload_of_pod(pod_name, workloads=workloads)
        assert workload is not None
        if len(uids := uids_of_pod[pod_name]) > 1:
            reasons_of_workload.setdefault(workload, []).append(f"pod {pod_name} was recreated as {sorted(uids)}")
        if len(counts := restart_counts_of_pod[pod_name]) > 1:
            reasons_of_workload.setdefault(workload, []).append(
                f"pod {pod_name} restarted a container: restartCount went through {sorted(counts)}"
            )
    for workload in workloads:
        if len(name_sets := pod_names_of_workload.get(workload, set())) > 1:
            reasons_of_workload.setdefault(workload, []).append(
                f"the pods of {workload} came and went: {sorted(sorted(one) for one in name_sets)}"
            )

    return {workload: sorted(reasons) for workload, reasons in sorted(reasons_of_workload.items())}


def _compute_workloads_with_changed_template(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    # [FUNCTION DOC] HELPER: Return the names of workloads that were observed with more than one
    # pod template fingerprint, or with more than one restart stamp, over all snapshots.
    fingerprints_of_workload: dict[str, set[str]] = {}
    stamps_of_workload = _compute_restart_stamps_of_workload(snapshots)
    for snapshot in snapshots:
        for workload in snapshot.workloads:
            fingerprints_of_workload.setdefault(workload.name, set()).add(workload.pod_template_fingerprint)
    return {
        name
        for name, fingerprints in fingerprints_of_workload.items()
        if len(fingerprints) > 1 or len(stamps_of_workload.get(name, set())) > 1
    }


def _compute_restart_stamps_of_workload(snapshots: Sequence[ClusterSnapshot]) -> dict[str, set[str]]:
    # [FUNCTION DOC] HELPER: Return a mapping from every workload name seen in any snapshot to the
    # set of distinct non-missing restart stamps observed on it (possibly empty).
    stamps_of_workload: dict[str, set[str]] = {}
    for snapshot in snapshots:
        for workload in snapshot.workloads:
            stamps = stamps_of_workload.setdefault(workload.name, set())
            if workload.restart_at is not None:
                stamps.add(workload.restart_at)
    return stamps_of_workload


def _compute_workload_of_pod(pod_name: str, *, workloads: Iterable[str]) -> str | None:
    # [FUNCTION DOC] HELPER: Return the longest workload name for which the pod name starts with
    # that name followed by a dash, or nothing if there is none.
    candidates = [one for one in workloads if pod_name.startswith(f"{one}-")]
    return max(candidates, key=len) if candidates else None
