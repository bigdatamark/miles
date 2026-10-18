# [MODULE DOC]
# Tests for the hot-restart process assertions: the trainer must keep a single boot uuid across
# all observations (and be reached after the last take-over), the run must have been observed
# closely enough (at least two whole-release observations and a sufficient success ratio, with a
# fallback for older evidence lacking attempt counts), a baseline snapshot with the trainer uuid
# must exist before the first take-over, and the boot uuid set computation ignores unreachable
# readings. Evidence is built from the shared restart fact builders.
import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_process import (
    _compute_trainer_boot_uuids,
    assert_baseline_read_before_first_take_over,
    assert_run_watched_closely,
    assert_trainer_not_rebooted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import POD_KIND
from tests.fast.e2e.deploy.hot_restart.cluster_facts import cluster_snapshot
from tests.fast.e2e.deploy.hot_restart.restart_facts import (
    evidence_of,
    quiet_run_snapshot,
    restart_snapshot,
    restarted_snapshot,
    two_restarts,
)


class TestAssertTheTrainerNeverRebooted:
    # [CLASS DOC] Tests of the trainer-not-rebooted assertion.
    def test_one_boot_uuid_across_every_script_passes(self):
        # [FUNCTION DOC] Evidence with the two-restart history (same boot uuid throughout) passes.
        """The trainer process outliving both scripts is what makes this a hot restart at all."""
        assert_trainer_not_rebooted(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_second_boot_uuid_fails(self):
        # [FUNCTION DOC] Appending a snapshot with boot uuid "boot-b" makes the assertion raise an
        # assertion error matching "boot uuid".
        """A restarted rpc server serves a new process, whatever its pod's uid says."""
        snapshots = [*two_restarts(), restart_snapshot(uid_of_pod={}, stamp_of_workload={}, boot_uuid="boot-b")]

        with pytest.raises(AssertionError, match="boot uuid"):
            assert_trainer_not_rebooted(evidence_of(snapshots=snapshots, records=[]))

    def test_a_trainer_never_reached_again_after_a_take_over_fails(self):
        # [FUNCTION DOC] If the last snapshot (after the last take-over) has no boot uuid, the
        # assertion raises matching "never reached after".
        """A uuid only ever read before the last take-over says nothing about what survived it."""
        snapshots = [
            quiet_run_snapshot(),
            restarted_snapshot(stamp="t1", uid_suffix="2"),
            restarted_snapshot(stamp="t2", uid_suffix="3", boot_uuid=None),
        ]

        with pytest.raises(AssertionError, match="never reached after"):
            assert_trainer_not_rebooted(evidence_of(snapshots=snapshots, records=[]))

    def test_a_trainer_the_rpc_server_never_answered_for_fails_as_such(self):
        # [FUNCTION DOC] Three snapshots all without a boot uuid make the assertion raise matching
        # "never reached across 3 observation", rather than a misleading reboot message.
        """One uuid out of zero readings used to fail as "the uuid(s) [] , not one", which reads as a reboot."""
        never_answered = [restart_snapshot(uid_of_pod={}, stamp_of_workload={}, boot_uuid=None) for _ in range(3)]

        with pytest.raises(AssertionError, match="never reached across 3 observation"):
            assert_trainer_not_rebooted(evidence_of(snapshots=never_answered, records=[]))


class TestAssertTheRunWasWatchedCloselyEnough:
    # [CLASS DOC] Tests of the observation-density assertion.
    def test_a_run_observed_throughout_passes(self):
        # [FUNCTION DOC] The two-restart history (three whole observations) passes.
        """Every verdict about pods is worth exactly what the observations behind it cost."""
        assert_run_watched_closely(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_run_nobody_looked_at_twice_fails(self):
        # [FUNCTION DOC] A single snapshot fails, matching "never looked at twice".
        """Every workload verdict is a comparison between observations, and one of them says nothing."""
        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_run_watched_closely(evidence_of(snapshots=[quiet_run_snapshot()], records=[]))

    def test_a_run_whose_observations_never_saw_a_whole_release_fails(self):
        # [FUNCTION DOC] If every snapshot reports pod reads missing, none counts as a whole
        # release and the assertion raises matching "never looked at twice".
        """A cluster nobody could read whole looks exactly like a cluster where nothing was replaced."""
        partial = [one.model_copy(update={"reads_missing": (POD_KIND,)}) for one in two_restarts()]

        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_run_watched_closely(evidence_of(snapshots=partial, records=[]))

    def test_a_run_nothing_ever_observed_fails(self):
        # [FUNCTION DOC] Evidence with no snapshots fails, matching "never looked at twice".
        """Evidence written before anything was collected would pass every count there is."""
        with pytest.raises(AssertionError, match="never looked at twice"):
            assert_run_watched_closely(evidence_of(snapshots=[], records=[]))

    def test_a_run_read_whole_only_half_as_often_as_it_was_tried_fails(self):
        # [FUNCTION DOC] With 40 attempts and 37 failures (3 successes against the 3 snapshots) the
        # assertion raises matching "attempt(s), under" (success ratio below the required floor).
        # KNOWN ISSUE: the name says half as often, but the data is 3 successes of 40 attempts.
        """A cluster answering half the time looks exactly like a cluster where nothing was replaced."""
        with pytest.raises(AssertionError, match=r"attempt\(s\), under"):
            assert_run_watched_closely(
                evidence_of(snapshots=two_restarts(), records=[], observation_attempts=40, observation_failures=37)
            )

    def test_evidence_written_before_the_counts_existed_still_faces_the_absolute_floor(self):
        # [FUNCTION DOC] Evidence with observation attempts of 0 (older dumps without counts)
        # passes with three snapshots: only the absolute minimum observation count applies.
        """Older dumps carry no attempt count, and must not become unverifiable because of it."""
        assert_run_watched_closely(evidence_of(snapshots=two_restarts(), records=[], observation_attempts=0))


class TestAssertABaselineWasReadBeforeTheFirstTakeOver:
    # [CLASS DOC] Tests of the baseline-before-first-take-over assertion.
    def test_an_unstamped_snapshot_carrying_the_trainer_uuid_passes(self):
        # [FUNCTION DOC] The two-restart history starts with an unstamped snapshot holding the
        # trainer uuid, so the assertion passes.
        """The identity a take-over must preserve is only known from before it landed."""
        assert_baseline_read_before_first_take_over(evidence_of(snapshots=two_restarts(), records=[]))

    def test_a_run_first_observed_after_a_take_over_fails(self):
        # [FUNCTION DOC] Snapshots that all carry restart stamps (no pre-take-over baseline) make
        # the assertion raise matching "before the first take-over".
        """Counting from the first take-over's snapshot makes the identity it installed the baseline."""
        snapshots = [restarted_snapshot(stamp="t1", uid_suffix="2"), restarted_snapshot(stamp="t2", uid_suffix="3")]

        with pytest.raises(AssertionError, match="before the first take-over"):
            assert_baseline_read_before_first_take_over(evidence_of(snapshots=snapshots, records=[]))

    def test_a_baseline_whose_trainer_was_never_reached_fails(self):
        # [FUNCTION DOC] The unstamped baseline has no trainer boot uuid: raises matching "before
        # the first take-over".
        """A baseline without the trainer uuid cannot say the trainer outlived the first script."""
        snapshots = [
            quiet_run_snapshot().model_copy(update={"trainer_boot_uuid": None}),
            restarted_snapshot(stamp="t1", uid_suffix="2"),
        ]

        with pytest.raises(AssertionError, match="before the first take-over"):
            assert_baseline_read_before_first_take_over(evidence_of(snapshots=snapshots, records=[]))

    def test_a_baseline_read_from_a_release_nothing_could_list_fails(self):
        # [FUNCTION DOC] The unstamped baseline reports pod reads missing (partial read): raises
        # matching "before the first take-over".
        """A read that failed looks exactly like a release whose workloads were not there yet."""
        snapshots = [
            quiet_run_snapshot().model_copy(update={"reads_missing": (POD_KIND,)}),
            restarted_snapshot(stamp="t1", uid_suffix="2"),
        ]

        with pytest.raises(AssertionError, match="before the first take-over"):
            assert_baseline_read_before_first_take_over(evidence_of(snapshots=snapshots, records=[]))


class TestComputeTrainerBootUuids:
    # [CLASS DOC] Tests of the boot uuid set computation.
    def test_a_trainer_that_outlived_every_script_answers_with_one_boot_uuid(self):
        # [FUNCTION DOC] Snapshots with uuids "boot-a", none, "boot-a" yield the set {"boot-a"}:
        # unreachable readings are ignored and duplicates collapse.
        """A second uuid means the process a hot restart promised to keep alive was replaced."""
        snapshots = [
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid=None),
            cluster_snapshot(pods=[], workloads=[], trainer_boot_uuid="boot-a"),
        ]

        assert _compute_trainer_boot_uuids(snapshots) == {"boot-a"}
