# [MODULE DOC] Async tests for passing per-worker "external data" payloads (for example critic
# values handed to the actor) through the train call of a trainer cell and of the trainer
# controller. Cells are real alive cells from the shared conftest backed by fake Ray actor workers
# that record their calls; the controller is built without its constructor around those cells.
# An autouse fixture initializes the global object store with the Ray backend. A dummy rollout data
# pack with one sample index and a mooncake object reference is used as train input.
from types import SimpleNamespace

import pytest
import ray
from tests.fast.ray.train.conftest import get_raw_actor_handles, make_alive_cell

from miles.ray.train import group as group_module
from miles.ray.train.group import TrainerController
from miles.utils import object_store
from miles.utils.data import RolloutDataPack
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.retry_utils import NonRetryableError, retry


async def _noop_run_after_step(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Asynchronous no-op stand-in for the test-action executor's
    # run-after-step hook; accepts and ignores any keyword arguments and returns nothing.
    return None


pytestmark = pytest.mark.asyncio

_DUMMY_DATA_PACK = RolloutDataPack(sample_indices=[0], data_ref=_MooncakeStoreObjectRef(payload="data"))


@pytest.fixture(autouse=True)
def _object_store_of_inited_controller() -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture. Initializes the process-global object store
    # instance with the Ray backend for both object storage and worker communication, without
    # contributing a store segment, because the controllers here skip their real init that would
    # normally create it and the failure path frees references through it. Mutates global state;
    # nothing is restored afterwards.
    """init() mints the store these controllers are faked past, and the failure path frees refs through it."""
    object_store.init_instance(
        SimpleNamespace(object_store_backend="ray", worker_comm_backend="ray"), contribute_segment=False
    )


def _make_controller(cells: list) -> TrainerController:
    # [FUNCTION DOC] HELPER: Builds a trainer controller without its constructor, holding the given
    # cells keyed by cell id. The arguments namespace disables the event analyzer and debug event
    # saving; there is no witness allocator, the independent-DP quorum id is zero, and the test
    # action executor is a stub whose run-after-step hook does nothing. Returns the controller.
    group = object.__new__(TrainerController)
    group._cells_by_id = {cell.cell_id: cell for cell in cells}
    group.args = SimpleNamespace(enable_event_analyzer=False, save_debug_event_data=None)
    group._witness_allocator = None
    group._indep_dp_quorum_id = 0
    group._test_action_executor = SimpleNamespace(run_after_step=_noop_run_after_step)
    return group


def _record_retry_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    # [FUNCTION DOC] HELPER: Replaces the controller module's retry function with a wrapper that
    # calls the real retry using a fake sleep which only records the requested delay (no real
    # waiting). Returns the list that accumulates the recorded delays.
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        # [FUNCTION DOC] HELPER: Appends the delay to the recorded list without sleeping.
        sleeps.append(delay)

    async def retry_with_fake_clock(fn, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the real retry on the function with the recording fake sleep
        # injected and all other keyword options forwarded; returns the retry result.
        return await retry(fn, sleep_fn=fake_sleep, **kwargs)

    monkeypatch.setattr(group_module, "retry", retry_with_fake_clock)
    return sleeps


def _count_cell_train_calls(monkeypatch: pytest.MonkeyPatch, cell) -> list[int]:
    # [FUNCTION DOC] HELPER: Wraps the cell's train method (via monkeypatch) so each invocation
    # appends the zero-based attempt number to a list before delegating to the original method with
    # the same keyword arguments. Returns that list; its length is the number of train attempts.
    attempts: list[int] = []
    original_train = cell.train

    async def counting_train(**kwargs):
        # [FUNCTION DOC] HELPER: Records the attempt number and awaits and returns the original
        # train method's result.
        attempts.append(len(attempts))
        return await original_train(**kwargs)

    monkeypatch.setattr(cell, "train", counting_train)
    return attempts


def _train_calls_of(cell) -> list[tuple]:
    # [FUNCTION DOC] HELPER: For each worker actor of the cell (in worker order), fetches its
    # recorded calls from the fake actor and keeps only those whose method name is train. Each call
    # is a triple of method name, positional arguments and keyword arguments. Returns a list with
    # one list of train calls per worker (the annotation says list of tuples but the elements are
    # lists).
    return [
        [call for call in ray.get(handle.get_calls.remote()) if call[0] == "train"]
        for handle in get_raw_actor_handles(cell)
    ]


class TestCellDistributesExternalData:
    # [CLASS DOC] Tests of how a trainer cell's train method distributes external data to its
    # workers.
    async def test_each_worker_receives_its_own_payload(self):
        # [FUNCTION DOC] Trains an alive two-worker cell with one payload per worker and asserts
        # each worker's single train call received its own payload as the external data keyword,
        # no positional arguments, and the rollout id (5) and rollout data reference passed
        # through.
        """The critic payload of worker i must reach exactly worker i, like the v1 group did."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        payloads = [{"values": ["v0"]}, {"values": ["v1"]}]

        await cell.train(
            rollout_id=5,
            rollout_data_ref="rollout",
            witness_info=None,
            attempt=0,
            external_data=payloads,
        )

        calls_per_worker = _train_calls_of(cell)
        assert [call[2]["external_data"] for [call] in calls_per_worker] == payloads
        assert all(call[1] == () for [call] in calls_per_worker)
        assert all(
            (call[2]["rollout_id"], call[2]["rollout_data_ref"]) == (5, "rollout") for [call] in calls_per_worker
        )

    async def test_no_payload_omits_the_keyword_entirely(self):
        # [FUNCTION DOC] Trains a cell without external data and asserts every worker's train call
        # received exactly the keywords rollout id, rollout data reference, witness info and
        # attempt, with no external data keyword.
        """Backends without an external_data parameter (fsdp) must keep working."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        await cell.train(rollout_id=7, rollout_data_ref="rollout", witness_info=None, attempt=0)

        for [call] in _train_calls_of(cell):
            assert call[2] == {"rollout_id": 7, "rollout_data_ref": "rollout", "witness_info": None, "attempt": 0}

    async def test_a_payload_count_mismatch_is_rejected(self):
        # [FUNCTION DOC] Trains a two-worker cell with only one payload and asserts a
        # non-retryable error is raised whose message mentions "one payload per train worker".
        """A wrong payload count would silently misroute values, so it must raise."""
        cell = make_alive_cell(0, alive_cell_indices=[0])

        with pytest.raises(NonRetryableError, match="one payload per train worker"):
            await cell.train(
                rollout_id=5,
                rollout_data_ref="rollout",
                witness_info=None,
                attempt=0,
                external_data=[{"values": []}],
            )


class TestControllerExternalData:
    # [CLASS DOC] Tests of the trainer controller's train method with external data, including
    # its single-cell restriction and non-retryable failure handling.
    async def test_train_forwards_external_data_to_the_cell(self):
        # [FUNCTION DOC] Trains a one-cell controller with two payloads and asserts each worker's
        # train call received its own payload.
        """The group hands the driver-provided payloads down to its only cell."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_controller([cell])
        payloads = [{"values": ["v0"]}, {"values": ["v1"]}]

        await group.train(3, _DUMMY_DATA_PACK, external_data=payloads)

        calls_per_worker = _train_calls_of(cell)
        assert [call[2]["external_data"] for [call] in calls_per_worker] == payloads

    async def test_external_data_is_rejected_with_multiple_cells(self):
        # [FUNCTION DOC] Trains a two-cell controller with external data and asserts an assertion
        # error mentioning "single cell" is raised.
        """Independent DP has no defined mapping from payloads to cells yet."""
        cells = [make_alive_cell(index, alive_cell_indices=[0, 1]) for index in range(2)]
        group = _make_controller(cells)

        with pytest.raises(AssertionError, match="single cell"):
            await group.train(3, _DUMMY_DATA_PACK, external_data=[{"values": []}])

    async def test_a_payload_count_mismatch_fails_immediately_without_backoff(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Installs the sleep recorder and the cell train attempt counter, then
        # trains the controller with a wrong payload count. Asserts a non-retryable error is raised
        # whose underlying cause mentions "one payload per train worker", that the cell was
        # trained only once, and that no backoff sleep occurred.
        """A deterministic argument error must abort the first attempt instead of burning the retry budget."""
        cell = make_alive_cell(0, alive_cell_indices=[0])
        group = _make_controller([cell])
        sleeps = _record_retry_sleeps(monkeypatch)
        attempts = _count_cell_train_calls(monkeypatch, cell)

        with pytest.raises(NonRetryableError) as excinfo:
            await group.train(3, _DUMMY_DATA_PACK, external_data=[{"values": []}])

        assert "one payload per train worker" in str(excinfo.value.__cause__)
        assert len(attempts) == 1
        assert sleeps == []
