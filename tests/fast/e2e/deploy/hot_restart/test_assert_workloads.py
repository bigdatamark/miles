# [MODULE DOC]
# Tests for the hot-restart workload assertions and their pure helpers: the main assertion that
# only the orchestrator and rollout executor workloads are restarted (by pod replacement, template
# rewrite and restart stamps, with the expected number of restarts) while the trainer and other
# workloads stay untouched, the set of workloads a hot restart may touch, attributing a pod to the
# workload whose name it extends (longest match wins), detecting pods owned by no known workload
# (skipping when a workload kind could not be listed), detecting workloads whose pods were
# replaced (new uid, restart count increase, disappearance), detecting changed pod templates
# (fingerprint or restart stamp, but not mere generation changes), and collecting the distinct
# restart stamps per workload. Data comes from the shared cluster and restart fact builders.
import pytest
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import (
    _compute_restart_stamps_of_workload,
    _compute_unattributed_pod_names,
    _compute_workload_of_pod,
    _compute_workloads_with_changed_template,
    _compute_workloads_with_replaced_pods,
    assert_only_orchestration_restarted,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import (
    LEADER_WORKER_SET_KIND,
    compute_hot_restart_workloads,
)
from tests.fast.e2e.deploy.hot_restart.cluster_facts import (
    ENGINE_POOL,
    ORCHESTRATOR,
    RELEASE,
    ROLLOUT_EXECUTOR,
    TRAINER,
    cluster_snapshot,
    pod_fact,
    workload_fact,
)
from tests.fast.e2e.deploy.hot_restart.restart_facts import (
    ENGINE_POOL_POD,
    evidence_of,
    quiet_run_snapshot,
    restart_snapshot,
    restarted_snapshot,
    two_restarts,
)


class TestAssertOnlyTheOrchestrationSideRestarted:
    # [CLASS DOC] Tests of the main workload restart assertion.
    def test_a_run_whose_script_was_replaced_twice_passes(self):
        # [FUNCTION DOC] The two-restart history with 2 expected restarts passes.
        """This is what the whole feature promises: two new scripts, the same trainers underneath."""
        assert_only_orchestration_restarted(evidence_of(snapshots=two_restarts(), records=[]), num_restarts=2)

    def test_a_pod_belonging_to_no_listed_workload_fails(self):
        # [FUNCTION DOC] A final snapshot with an engine pod while only the orchestrator workload
        # is listed: raises matching "belong to no workload".
        """A pod nothing owns is a pod this verdict silently says nothing about."""
        snapshots = [
            *two_restarts(),
            restart_snapshot(
                uid_of_pod={f"{ORCHESTRATOR}-0": "uid-o-3", ENGINE_POOL_POD: "uid-e"},
                stamp_of_workload={ORCHESTRATOR: "t2"},
            ),
        ]

        with pytest.raises(AssertionError, match="belong to no workload"):
            assert_only_orchestration_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)

    def test_a_trainer_pod_that_was_replaced_fails(self):
        # [FUNCTION DOC] A final snapshot where the trainer pod has a new uid: raises matching
        # "lost a pod".
        """A trainer that restarted lost the weights the take-over claims it reloaded a checkpoint into."""
        snapshots = [
            *two_restarts(),
            restart_snapshot(
                uid_of_pod={
                    f"{ORCHESTRATOR}-0": "uid-o-3",
                    f"{ROLLOUT_EXECUTOR}-0": "uid-r-3",
                    f"{TRAINER}-0": "uid-t-2",
                },
                stamp_of_workload={ORCHESTRATOR: "t2", ROLLOUT_EXECUTOR: "t2", TRAINER: None},
            ),
        ]

        with pytest.raises(AssertionError, match="lost a pod"):
            assert_only_orchestration_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)

    def test_a_trainer_whose_pod_template_was_rewritten_fails(self):
        # [FUNCTION DOC] A final snapshot where the trainer workload carries a restart stamp (its
        # template was rewritten) though its pod uid is unchanged: raises an assertion error.
        """An ordinary relaunch has to render a zero diff for every object it does not replace."""
        snapshots = [
            *two_restarts(),
            restart_snapshot(
                uid_of_pod={
                    f"{ORCHESTRATOR}-0": "uid-o-3",
                    f"{ROLLOUT_EXECUTOR}-0": "uid-r-3",
                    f"{TRAINER}-0": "uid-t",
                },
                stamp_of_workload={ORCHESTRATOR: "t2", ROLLOUT_EXECUTOR: "t2", TRAINER: "t2"},
            ),
        ]

        with pytest.raises(AssertionError):
            assert_only_orchestration_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)

    def test_a_second_restart_that_never_landed_fails(self):
        # [FUNCTION DOC] Only one restart happened while two were expected: raises matching "hot
        # restart".
        """Testing one take-over would not show that a script can take over what a script already took over."""
        snapshots = [quiet_run_snapshot(), restarted_snapshot(stamp="t1", uid_suffix="2")]

        with pytest.raises(AssertionError, match="hot restart"):
            assert_only_orchestration_restarted(evidence_of(snapshots=snapshots, records=[]), num_restarts=2)


# ========================== what the snapshots say ============================


class TestComputeHotRestartWorkloads:
    # [CLASS DOC] Test of the allowed-restart workload set.
    def test_a_hot_restart_names_the_orchestrator_and_the_rollout_executor(self):
        # [FUNCTION DOC] For the demo release the set is exactly the orchestrator and rollout
        # executor workload names.
        """Every other object of the release has to come out of the relaunch untouched."""
        assert compute_hot_restart_workloads(RELEASE) == frozenset({ORCHESTRATOR, ROLLOUT_EXECUTOR})


class TestComputeWorkloadOfPod:
    # [CLASS DOC] Tests of pod-to-workload attribution by name prefix.
    def test_a_pod_belongs_to_the_workload_whose_name_it_extends(self):
        # [FUNCTION DOC] A trainer ordinal pod is attributed to the trainer workload.
        """Pods are named after their statefulset plus an ordinal, and nothing else links them."""
        assert _compute_workload_of_pod(f"{TRAINER}-0", workloads=[TRAINER, ORCHESTRATOR]) == TRAINER

    def test_the_longest_matching_workload_wins(self):
        # [FUNCTION DOC] When one workload name prefixes another, a pod of the longer one is
        # attributed to the longer name.
        """One workload's name can prefix another's, and the shorter one would then claim its pods."""
        assert (
            _compute_workload_of_pod(
                f"{RELEASE}-router-extra-0", workloads=[f"{RELEASE}-router", f"{RELEASE}-router-extra"]
            )
            == f"{RELEASE}-router-extra"
        )

    def test_a_worker_of_a_leaderworkerset_group_belongs_to_that_leaderworkerset(self):
        # [FUNCTION DOC] An engine pool pod named with a group index and worker index suffix is
        # attributed to the engine pool.
        """A group's workers are named after the set, the group index and the worker index."""
        assert _compute_workload_of_pod(f"{ENGINE_POOL}-1-2", workloads=[ENGINE_POOL]) == ENGINE_POOL

    def test_a_pod_of_no_known_workload_belongs_to_none(self):
        # [FUNCTION DOC] A pod named "unrelated-0" with only the trainer workload known gives none.
        """Only the pods of this release's workloads say anything about this run."""
        assert _compute_workload_of_pod("unrelated-0", workloads=[TRAINER]) is None


class TestComputeUnattributedPodNames:
    # [CLASS DOC] Tests of detection of pods owned by no listed workload.
    def test_a_run_whose_every_pod_belongs_to_a_known_workload_leaves_the_bucket_empty(self):
        # [FUNCTION DOC] Engine and trainer pods with their workloads listed (the engine pool as
        # a leader-worker set) produce an empty set.
        """A pod nothing owns is a pod no restart verdict covers."""
        snapshot = cluster_snapshot(
            pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-e"), pod_fact(f"{TRAINER}-0", uid="uid-t")],
            workloads=[workload_fact(ENGINE_POOL, kind=LEADER_WORKER_SET_KIND), workload_fact(TRAINER)],
        )

        assert _compute_unattributed_pod_names([snapshot]) == set()

    def test_a_pod_of_a_workload_kind_nobody_listed_is_reported(self):
        # [FUNCTION DOC] An engine pod with only the trainer workload listed (and all kinds read)
        # yields the set containing the engine pod's name.
        """The engines were invisible to the verdict for as long as only statefulsets were read."""
        snapshot = cluster_snapshot(
            pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-e")], workloads=[workload_fact(TRAINER)]
        )

        assert _compute_unattributed_pod_names([snapshot]) == {f"{ENGINE_POOL}-0-1"}

    def test_an_observation_that_could_not_list_every_workload_is_not_read_as_an_orphan_pod(self):
        # [FUNCTION DOC] Same data as before but the snapshot reports the leader-worker set kind
        # as missing from the reads; the result is empty (an unreadable listing proves nothing).
        """A failed kubectl call says nothing about which workload a pod belongs to."""
        snapshot = cluster_snapshot(
            pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-e")],
            workloads=[workload_fact(TRAINER)],
            reads_missing=(LEADER_WORKER_SET_KIND,),
        )

        assert _compute_unattributed_pod_names([snapshot]) == set()


class TestComputeWorkloadsWhosePodsWereReplaced:
    # [CLASS DOC] Tests of detection of workloads with replaced pods across snapshots; the result
    # is a mapping keyed by workload name.
    def test_a_run_nothing_disturbed_replaces_no_pod(self):
        # [FUNCTION DOC] Two identical snapshots give an empty mapping.
        """This is what every workload outside the hot restart has to look like."""
        snapshot = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=[workload_fact(TRAINER)])

        assert _compute_workloads_with_replaced_pods([snapshot, snapshot]) == {}

    def test_a_pod_that_came_back_under_a_new_uid_names_its_workload(self):
        # [FUNCTION DOC] An orchestrator pod with the same name but a new uid between snapshots
        # reports only the orchestrator (the trainer pod is unchanged).
        """A rolled statefulset recreates its pod under the same name, so only the uid shows it."""
        workloads = [workload_fact(ORCHESTRATOR), workload_fact(TRAINER)]
        before = cluster_snapshot(
            pods=[pod_fact(f"{ORCHESTRATOR}-0", uid="uid-1"), pod_fact(f"{TRAINER}-0", uid="uid-t")],
            workloads=workloads,
        )
        after = cluster_snapshot(
            pods=[pod_fact(f"{ORCHESTRATOR}-0", uid="uid-2"), pod_fact(f"{TRAINER}-0", uid="uid-t")],
            workloads=workloads,
        )

        assert list(_compute_workloads_with_replaced_pods([before, after])) == [ORCHESTRATOR]

    def test_an_engine_pod_that_came_back_names_the_leaderworkerset_it_belongs_to(self):
        # [FUNCTION DOC] An engine pod with a new uid reports the engine pool leader-worker set.
        """The engines only take part in the verdict once their leaderworkerset is known."""
        workloads = [workload_fact(ENGINE_POOL, kind=LEADER_WORKER_SET_KIND)]
        before = cluster_snapshot(pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-1")], workloads=workloads)
        after = cluster_snapshot(pods=[pod_fact(f"{ENGINE_POOL}-0-1", uid="uid-2")], workloads=workloads)

        assert list(_compute_workloads_with_replaced_pods([before, after])) == [ENGINE_POOL]

    def test_a_container_that_restarted_in_place_names_its_workload(self):
        # [FUNCTION DOC] A trainer pod with the same uid but restart count 1 reports the trainer.
        """A trainer whose container crashed and came back lost the state a take-over relies on."""
        workloads = [workload_fact(TRAINER)]
        before = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=workloads)
        after = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t", restart_count=1)], workloads=workloads)

        assert list(_compute_workloads_with_replaced_pods([before, after])) == [TRAINER]

    def test_a_pod_that_disappeared_names_its_workload(self):
        # [FUNCTION DOC] A trainer pod present in the first snapshot and absent in the second
        # reports the trainer.
        """A pod deleted and not yet recreated is a restart caught mid-flight, not a run left alone."""
        workloads = [workload_fact(TRAINER)]
        before = cluster_snapshot(pods=[pod_fact(f"{TRAINER}-0", uid="uid-t")], workloads=workloads)
        after = cluster_snapshot(pods=[], workloads=workloads)

        assert list(_compute_workloads_with_replaced_pods([before, after])) == [TRAINER]


class TestComputeWorkloadsWhoseTemplateChanged:
    # [CLASS DOC] Tests of detection of workloads whose pod template changed across snapshots;
    # the result is a set of workload names.
    def test_a_relaunch_that_rolled_nothing_changes_no_generation(self):
        # [FUNCTION DOC] Two identical snapshots give an empty set.
        """An ordinary relaunch has to render a zero diff, or it would restart the whole run."""
        snapshot = cluster_snapshot(pods=[], workloads=[workload_fact(TRAINER), workload_fact(ORCHESTRATOR)])

        assert _compute_workloads_with_changed_template([snapshot, snapshot]) == set()

    def test_only_the_workloads_whose_template_fingerprint_changed_are_reported(self):
        # [FUNCTION DOC] Only the orchestrator's fingerprint differs between snapshots; the
        # result is exactly the orchestrator.
        """A template change is caught even when a workload controller leaves generation unchanged."""
        before = cluster_snapshot(pods=[], workloads=[workload_fact(ORCHESTRATOR), workload_fact(TRAINER)])
        after = cluster_snapshot(
            pods=[],
            workloads=[
                workload_fact(ORCHESTRATOR, pod_template_fingerprint="template-b"),
                workload_fact(TRAINER),
            ],
        )

        assert _compute_workloads_with_changed_template([before, after]) == {ORCHESTRATOR}

    def test_a_generation_change_without_a_template_change_is_ignored(self):
        # [FUNCTION DOC] A generation increase with the same fingerprint and stamp gives an empty
        # set.
        """A LeaderWorkerSet controller may advance generation while preserving its pod templates."""
        before = cluster_snapshot(pods=[], workloads=[workload_fact(TRAINER)])
        after = cluster_snapshot(pods=[], workloads=[workload_fact(TRAINER, generation=2)])

        assert _compute_workloads_with_changed_template([before, after]) == set()

    def test_a_workload_restamped_between_two_observations_of_one_generation_is_reported(self):
        # [FUNCTION DOC] The engine pool has generation 2 in both snapshots but its restart stamp
        # goes from "t1" to "t2" (with an unchanged fingerprint in the helper): it is reported.
        """An observation may miss the generation moving, but the stamp it carries is written to survive."""
        before = cluster_snapshot(pods=[], workloads=[workload_fact(ENGINE_POOL, generation=2, restart_at="t1")])
        after = cluster_snapshot(pods=[], workloads=[workload_fact(ENGINE_POOL, generation=2, restart_at="t2")])

        assert _compute_workloads_with_changed_template([before, after]) == {ENGINE_POOL}


class TestComputeRestartStampsOfWorkload:
    # [CLASS DOC] Test of per-workload restart stamp collection.
    def test_every_distinct_stamp_a_workload_carried_is_collected(self):
        # [FUNCTION DOC] Three snapshots where the orchestrator has no stamp, then "t1", then "t2"
        # and the trainer never has one: the result maps the orchestrator to {"t1", "t2"} and the
        # trainer to an empty set (unstamped values are not collected).
        """Each hot restart writes one stamp, so the count of them is the count of take-overs."""
        snapshots = [
            cluster_snapshot(pods=[], workloads=[workload_fact(ORCHESTRATOR), workload_fact(TRAINER)]),
            cluster_snapshot(
                pods=[], workloads=[workload_fact(ORCHESTRATOR, restart_at="t1"), workload_fact(TRAINER)]
            ),
            cluster_snapshot(
                pods=[], workloads=[workload_fact(ORCHESTRATOR, restart_at="t2"), workload_fact(TRAINER)]
            ),
        ]

        assert _compute_restart_stamps_of_workload(snapshots) == {ORCHESTRATOR: {"t1", "t2"}, TRAINER: set()}
