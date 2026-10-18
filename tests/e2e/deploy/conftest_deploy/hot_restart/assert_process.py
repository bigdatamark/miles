# [MODULE DOC]
# HELPER: Assertions about the quality of the observation of a hot-restart run (as opposed to what
# the run did): that the cluster was watched closely enough for the other assertions to mean
# anything, that a healthy baseline was seen before the first take-over, and that the trainer
# process was never rebooted. They operate on the list of cluster snapshots collected in the
# hot-restart evidence. Constants: at least 2 snapshots must describe the whole release, and at
# least half of all observation attempts must have produced such a snapshot.
from collections.abc import Sequence

from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterSnapshot
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence

MINIMUM_COMPLETE_SNAPSHOTS: int = 2
MINIMUM_OBSERVATION_SUCCESS_RATIO: float = 0.5


def assert_run_watched_closely(evidence: HotRestartEvidence) -> None:
    # [FUNCTION DOC] Assert the observation was dense enough. Counts the snapshots that describe
    # the whole release; requires at least the minimum of 2 (otherwise a pod that survived every
    # restart cannot be told apart from one never looked at twice). Additionally, when the
    # number of recorded observation attempts (polls taken while the release was up) is positive,
    # requires the complete snapshot count to be at least half of the attempts. Raises assertion
    # errors with explanatory messages.
    complete = [one for one in evidence.snapshots if one.describes_whole_release]

    assert len(complete) >= MINIMUM_COMPLETE_SNAPSHOTS, (
        f"the whole release was read {len(complete)} time(s) out of {len(evidence.snapshots)} observation(s), fewer "
        f"than the {MINIMUM_COMPLETE_SNAPSHOTS} it takes to tell a pod that survived every restart from one that "
        f"was never looked at twice"
    )

    if (attempts := evidence.observation_attempts) > 0:
        assert len(complete) >= MINIMUM_OBSERVATION_SUCCESS_RATIO * attempts, (
            f"the whole release was read {len(complete)} time(s) out of {attempts} attempt(s), under "
            f"{MINIMUM_OBSERVATION_SUCCESS_RATIO:.0%}; those attempts are the polls taken while the release was "
            f"up, and a cluster that answers half the time hides the very pod replacement these assertions are "
            f"here to catch"
        )


def assert_baseline_read_before_first_take_over(evidence: HotRestartEvidence) -> None:
    # [FUNCTION DOC] Assert that a usable baseline was observed before any take-over. Finds the
    # index of the earliest snapshot in which any workload carries a restart stamp (defaulting
    # to the total number of snapshots when no stamp was ever seen). Requires that some snapshot
    # before that index describes the whole release, lists at least one workload, and has reached
    # the trainer's rpc server (boot uuid present). If the first snapshot is already stamped,
    # the checked range is empty and the assertion fails.
    stamped_at = min(
        _compute_first_snapshot_index_of_stamp(evidence.snapshots).values(), default=len(evidence.snapshots)
    )

    assert any(
        one.describes_whole_release and one.workloads and one.trainer_boot_uuid is not None
        for one in evidence.snapshots[:stamped_at]
    ), (
        f"none of the {stamped_at} observation(s) before the first take-over read the whole release and reached "
        f"the trainer's rpc server"
    )


def assert_trainer_not_rebooted(evidence: HotRestartEvidence) -> None:
    # [FUNCTION DOC] Assert the trainer process kept running through the take-overs. Collects the
    # distinct trainer boot uuids over all snapshots; requires at least one (the rpc server was
    # reached) and exactly one (the same process throughout). Then, for every restart stamp seen
    # (in sorted order), requires that at least one snapshot at or after the first snapshot
    # showing that stamp reached the trainer's rpc server, so the single uuid also covers the
    # time after that take-over. Raises assertion errors with explanatory messages.
    boot_uuids = _compute_trainer_boot_uuids(evidence.snapshots)
    assert boot_uuids, (
        f"the trainer's rpc server was never reached across {len(evidence.snapshots)} observation(s), so nothing "
        f"here says whether the process a take-over reloaded a checkpoint into is the one that had been training"
    )
    assert len(boot_uuids) == 1, (
        f"the trainer's rpc server answered with the boot uuid(s) {sorted(boot_uuids)}, not one: the process a "
        f"take-over reloaded a checkpoint into is not the process that had been training"
    )

    for stamp, index in sorted(_compute_first_snapshot_index_of_stamp(evidence.snapshots).items()):
        answered = [one for one in evidence.snapshots[index:] if one.trainer_boot_uuid is not None]
        assert answered, (
            f"the take-over stamped {stamp} and the trainer's rpc server was never reached after it, so the one "
            f"uuid this run collected only proves the trainer was alive before its script was replaced"
        )


def _compute_trainer_boot_uuids(snapshots: Sequence[ClusterSnapshot]) -> set[str]:
    # [FUNCTION DOC] HELPER: Return the set of trainer boot uuids present (non-missing) over the
    # snapshots.
    return {one.trainer_boot_uuid for one in snapshots if one.trainer_boot_uuid is not None}


def _compute_first_snapshot_index_of_stamp(snapshots: Sequence[ClusterSnapshot]) -> dict[str, int]:
    # [FUNCTION DOC] HELPER: Return a mapping from each restart stamp (the restart-at value of a
    # workload, ignoring workloads without one) to the index of the first snapshot, in order,
    # in which any workload carried that stamp.
    first_index_of_stamp: dict[str, int] = {}
    for index, snapshot in enumerate(snapshots):
        for workload in snapshot.workloads:
            if workload.restart_at is not None:
                first_index_of_stamp.setdefault(workload.restart_at, index)
    return first_index_of_stamp
