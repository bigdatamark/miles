# [MODULE DOC] Unit tests for reading a worker's identity from the pod environment: the names of the
# index environment variables, the cell index (must exist and lie inside the pool's cell count),
# numeric validation of all three indices, worker numbering across the pods of a cell, gpu id
# assignment per worker, the single-worker-pod defaults, the constructor context built from an
# identity, and the in-pod index used to offset rpc ports. Environments are plain dictionaries passed
# in explicitly; scheduling specs are built by a local helper.
from __future__ import annotations

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability

from miles.utils.workers.serving.worker_identity import (
    CELL_INDEX_ENV_VAR,
    POD_INDEX_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
    read_worker_identity,
    read_worker_in_pod_index,
)
from miles.utils.workers.worker_spec import SchedulingSpec


def scheduling(
    *, workers_per_pod: int = 1, gpu_slots_per_worker: int = 0, pods_per_cell: int = 1, num_cells: int = 1
) -> SchedulingSpec:
    # [FUNCTION DOC] HELPER: Builds a scheduling spec from pod-level numbers. The number of workers
    # per cell is workers per pod times pods per cell; gpus per worker and gpu slots per worker both
    # equal the given slots; gpus per node is workers per pod times the slots. Defaults: one worker per
    # pod, no gpus, one pod per cell, one cell.
    return SchedulingSpec(
        num_cells=num_cells,
        num_workers_per_cell=workers_per_pod * pods_per_cell,
        num_gpus_per_worker=gpu_slots_per_worker,
        num_gpu_slots_per_worker=gpu_slots_per_worker,
        num_gpus_per_node=workers_per_pod * gpu_slots_per_worker,
    )


def _environ(*, cell_index: str = "0", pod_index: str | None = None, subprocess_index: str | None = None):
    # [FUNCTION DOC] HELPER: Builds an environment mapping holding the cell index (default "0") and,
    # only when given, the pod index and the subprocess index, all as strings.
    environ = {CELL_INDEX_ENV_VAR: cell_index}
    if pod_index is not None:
        environ[POD_INDEX_ENV_VAR] = pod_index
    if subprocess_index is not None:
        environ[SUBPROCESS_INDEX_ENV_VAR] = subprocess_index
    return environ


class TestEnvVarOwnership:
    # [CLASS DOC] Tests the names of the index environment variables.
    def test_every_index_a_pod_reads_is_named_by_miles(self):
        # [FUNCTION DOC] The cell and pod index variable names are MILES_CELL_INDEX and MILES_POD_INDEX.
        """A pod created by something other than our chart cannot be asked to set an upstream name."""
        assert (CELL_INDEX_ENV_VAR, POD_INDEX_ENV_VAR) == ("MILES_CELL_INDEX", "MILES_POD_INDEX")


class TestReadCellIndex:
    # [CLASS DOC] Tests the cell index part of the identity.
    def test_reads_the_cell_the_platform_stamped_onto_the_pod(self):
        # [FUNCTION DOC] With four cells and cell index "3" the identity's cell index is 3.
        """The driver numbers the pool's cells the same way, so both sides must agree on this index."""
        assert (
            read_worker_identity(scheduling=scheduling(num_cells=4), environ=_environ(cell_index="3")).cell_index == 3
        )

    def test_refuses_a_pod_that_was_told_nothing_about_its_cell(self):
        # [FUNCTION DOC] An empty environment raises an assertion error naming the cell index variable.
        """A pod that guessed would report itself as a member of whichever cell it invented."""
        with pytest.raises(AssertionError, match=CELL_INDEX_ENV_VAR):
            read_worker_identity(scheduling=scheduling(), environ={})

    def test_refuses_a_cell_the_pool_was_never_scheduled_to_have(self):
        # [FUNCTION DOC] Cell index 4 with four cells (valid range 0 to 3) raises an assertion error
        # naming the cell index variable.
        """No address enumerates such a cell, so nothing would ever reach the workers this pod reports."""
        with pytest.raises(AssertionError, match=CELL_INDEX_ENV_VAR):
            read_worker_identity(scheduling=scheduling(num_cells=4), environ=_environ(cell_index="4"))


class TestReadWorkerIdentity:
    # [CLASS DOC] Tests the full worker identity: indices, numbering, gpu ids and constructor context.
    @pytest.mark.parametrize(
        "environment_name",
        [CELL_INDEX_ENV_VAR, POD_INDEX_ENV_VAR, SUBPROCESS_INDEX_ENV_VAR],
    )
    def test_a_non_numeric_worker_identity_index_is_rejected(self, environment_name: str):
        # [FUNCTION DOC] Parametrized over the three index variables. All are set to "0" except the
        # chosen one set to a non-number; for a two-pod, two-worker pod scheduling, an assertion error
        # naming that variable is raised.
        """A malformed platform index must not be accepted as part of a worker identity."""
        environ = _environ(cell_index="0", pod_index="0", subprocess_index="0")
        environ[environment_name] = "not-a-number"

        with pytest.raises(AssertionError, match=environment_name):
            read_worker_identity(
                scheduling=scheduling(workers_per_pod=2, gpu_slots_per_worker=1, pods_per_cell=2),
                environ=environ,
            )

    def test_numbers_the_workers_of_a_multi_pod_cell_end_to_end(self):
        # [FUNCTION DOC] With eight workers per pod and two pods, pod index 1 and subprocess index 3
        # give worker-in-cell index 8 plus 3, which is 11.
        """A cell's workers are numbered across its pods, so each pod has to offset by its own index."""
        identity = read_worker_identity(
            scheduling=scheduling(workers_per_pod=8, gpu_slots_per_worker=1, pods_per_cell=2),
            environ=_environ(pod_index="1", subprocess_index="3"),
        )

        assert identity.worker_in_cell_index == 11

    def test_refuses_a_pod_index_the_cell_does_not_have(self):
        # [FUNCTION DOC] Pod index 2 with two pods per cell raises an assertion error naming the pod
        # index variable.
        """A pod numbered past the cell would report workers that belong to no cell of the pool."""
        with pytest.raises(AssertionError, match=POD_INDEX_ENV_VAR):
            read_worker_identity(
                scheduling=scheduling(workers_per_pod=8, gpu_slots_per_worker=1, pods_per_cell=2),
                environ=_environ(pod_index="2", subprocess_index="0"),
            )

    def test_gives_each_worker_of_a_pod_its_own_gpus(self):
        # [FUNCTION DOC] Four workers per pod with two gpu slots each: subprocess 0 gets gpu ids 0 and
        # 1, subprocess 1 gets 2 and 3.
        """Two workers that claimed the same gpu would each initialise a full model on it."""
        pod = scheduling(workers_per_pod=4, gpu_slots_per_worker=2)
        first = read_worker_identity(scheduling=pod, environ=_environ(subprocess_index="0"))
        second = read_worker_identity(scheduling=pod, environ=_environ(subprocess_index="1"))

        assert (first.gpu_ids, second.gpu_ids) == ([0, 1], [2, 3])

    def test_a_single_worker_pod_needs_nothing_beyond_its_cell_index(self):
        # [FUNCTION DOC] With the default scheduling and only the cell index set, the identity is cell
        # 0, worker 0, and no gpu ids.
        """A static worker runs one worker with no supervisor, and must not depend on variables nobody sets."""
        identity = read_worker_identity(scheduling=scheduling(), environ=_environ())

        assert (identity.cell_index, identity.worker_in_cell_index, identity.gpu_ids) == (0, 0, [])

    def test_refuses_a_multi_pod_cell_whose_pod_was_told_no_index(self):
        # [FUNCTION DOC] With two pods per cell and no pod index set, an assertion error naming the
        # pod index variable is raised.
        """Every pod of the cell would then claim the leader's workers, and the cell would have no others."""
        with pytest.raises(AssertionError, match=POD_INDEX_ENV_VAR):
            read_worker_identity(
                scheduling=scheduling(workers_per_pod=4, gpu_slots_per_worker=1, pods_per_cell=2),
                environ=_environ(subprocess_index="1"),
            )

    def test_the_ctor_context_carries_the_cell_the_pod_is_in(self):
        # [FUNCTION DOC] For cell 1 of two, worker 1 of two with one gpu slot each, the constructor
        # context built with a fake backend capability reports cell index 1, worker-in-cell index 1 and
        # gpu ids [1].
        """ctor kwargs are computed from this context, and a trainer keys its checkpoints off the cell."""
        identity = read_worker_identity(
            scheduling=scheduling(workers_per_pod=2, gpu_slots_per_worker=1, num_cells=2),
            environ=_environ(subprocess_index="1", cell_index="1"),
        )

        context = identity.ctor_context(capability=FakeBackendCapability())

        assert (context.cell_index, context.worker_in_cell_index, context.gpu_ids) == (1, 1, [1])

    def test_the_ctor_context_always_carries_a_provider_factory(self):
        # [FUNCTION DOC] Building the constructor context without a capability argument raises a type
        # error (the capability is a required argument).
        """A spec asking for its engines must never find the factory missing, so it cannot be omitted."""
        identity = read_worker_identity(scheduling=scheduling(), environ=_environ())

        with pytest.raises(TypeError):
            identity.ctor_context()

    def test_refuses_a_multi_worker_pod_whose_process_was_told_no_subprocess_index(self):
        # [FUNCTION DOC] Two workers per pod and no subprocess index raises an assertion error naming
        # the subprocess index variable.
        """A process started outside the supervisor would claim the leading worker and leave the rest unserved."""
        with pytest.raises(AssertionError, match=SUBPROCESS_INDEX_ENV_VAR):
            read_worker_identity(scheduling=scheduling(workers_per_pod=2, gpu_slots_per_worker=1), environ=_environ())

    def test_refuses_a_subprocess_index_the_pod_was_not_launched_for(self):
        # [FUNCTION DOC] Subprocess index 2 with two workers per pod raises an assertion error naming
        # the subprocess index variable.
        """A worker beyond the pod's share would collide with a worker of the next pod in the cell."""
        with pytest.raises(AssertionError, match=SUBPROCESS_INDEX_ENV_VAR):
            read_worker_identity(
                scheduling=scheduling(workers_per_pod=2, gpu_slots_per_worker=1),
                environ=_environ(subprocess_index="2"),
            )


class TestReadWorkerInPodIndex:
    # [CLASS DOC] Tests the standalone reader of the in-pod worker index.
    def test_reports_the_index_the_supervisor_gave_this_subprocess(self):
        # [FUNCTION DOC] An environment with subprocess index "3" gives 3.
        """The rpc port is offset by this number, and only the supervisor's own variable carries it."""
        assert read_worker_in_pod_index({SUBPROCESS_INDEX_ENV_VAR: "3"}) == 3

    def test_an_unsupervised_process_is_the_only_worker_of_its_pod(self):
        # [FUNCTION DOC] An empty environment gives 0.
        """A worker launched without the supervisor must still bind the port the address book predicts."""
        assert read_worker_in_pod_index({}) == 0
