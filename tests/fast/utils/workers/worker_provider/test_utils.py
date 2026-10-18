# [MODULE DOC] Tests for the worker provider utilities: apply_cell_observation reconciles one
# observed cell against the known workers hash by calling add and remove callbacks (unknown
# observed cell: add; disappeared known cell: remove; disappeared unknown cell: nothing; changed
# hash: remove then add, aborting and propagating if remove fails; unchanged hash: nothing), and
# the rpc handle builders (a handle aimed at the http address of the worker's rpc port; an
# assertion error "has no 'rpc' port" when the address map lacks it). Async tests run with the
# project's asyncio marker applied module-wide.
import pytest

from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import CellInfo
from miles.utils.workers.worker_provider.utils import (
    apply_cell_observation,
    build_rpc_handle,
    build_rpc_handle_of_worker_info,
)
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.asyncio

_CELL_ID = "spec-0"


def _make_cell_info(workers_hash: str = "hash-1") -> CellInfo:
    # [FUNCTION DOC] HELPER: Builds an alive cell info for cell spec-0 (pool spec, one worker
    # spec-0-0, empty metadata) with the given workers hash (default hash-1).
    return CellInfo(
        cell_id=_CELL_ID,
        pool_id="spec",
        alive=True,
        worker_names=["spec-0-0"],
        workers_hash=workers_hash,
        meta={},
    )


class _Recorder:
    # [CLASS DOC] HELPER: Records the add and remove callbacks as ordered tuples.
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def add(self, cell_id: str, observed: CellInfo) -> None:
        # [FUNCTION DOC] HELPER: Records ("add", cell id, observed workers hash).
        self.calls.append(("add", cell_id, observed.workers_hash))

    async def remove(self, cell_id: str) -> None:
        # [FUNCTION DOC] HELPER: Records ("remove", cell id).
        self.calls.append(("remove", cell_id))


class TestApplyCellObservation:
    # [CLASS DOC] Tests of the observation reconciliation rules.
    async def test_an_unknown_observed_cell_is_added(self):
        # [FUNCTION DOC] Observed cell with no known hash: one add call with hash-1.
        """A newly reported cell must enter the bookkeeping."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(),
            actual_workers_hash=None,
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("add", _CELL_ID, "hash-1")]

    async def test_a_disappeared_known_cell_is_removed(self):
        # [FUNCTION DOC] No observation but a known hash: one remove call.
        """A cell the provider stops reporting must leave the bookkeeping."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID, observed=None, actual_workers_hash="hash-1", add=recorder.add, remove=recorder.remove
        )

        assert recorder.calls == [("remove", _CELL_ID)]

    async def test_a_disappeared_unknown_cell_is_ignored(self):
        # [FUNCTION DOC] No observation and no known hash: no calls.
        """Removing what was never added would raise in the callbacks."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID, observed=None, actual_workers_hash=None, add=recorder.add, remove=recorder.remove
        )

        assert recorder.calls == []

    async def test_a_changed_workers_hash_replaces_the_cell(self):
        # [FUNCTION DOC] Observed hash-2 against known hash-1: remove then add (with hash-2), in
        # that order.
        """A new worker generation must not be served through the old cell object."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info("hash-2"),
            actual_workers_hash="hash-1",
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == [("remove", _CELL_ID), ("add", _CELL_ID, "hash-2")]

    async def test_a_failed_remove_aborts_replacement_and_propagates(self):
        # [FUNCTION DOC] The remove callback records and then raises a runtime error; the error
        # reaches the caller and no add call is made.
        """The replacement must not be installed while the old cell is still registered."""
        recorder = _Recorder()

        async def failing_remove(cell_id: str) -> None:
            # [FUNCTION DOC] HELPER: Records the removal through the recorder, then raises a
            # runtime error "remove boom".
            await recorder.remove(cell_id)
            raise RuntimeError("remove boom")

        with pytest.raises(RuntimeError, match="remove boom"):
            await apply_cell_observation(
                cell_id=_CELL_ID,
                observed=_make_cell_info("hash-2"),
                actual_workers_hash="hash-1",
                add=recorder.add,
                remove=failing_remove,
            )

        assert recorder.calls == [("remove", _CELL_ID)]

    async def test_an_unchanged_workers_hash_keeps_the_cell(self):
        # [FUNCTION DOC] Observed and known hash both hash-1: no calls.
        """Recreating the cell would throw away its accumulated state."""
        recorder = _Recorder()

        await apply_cell_observation(
            cell_id=_CELL_ID,
            observed=_make_cell_info(),
            actual_workers_hash="hash-1",
            add=recorder.add,
            remove=recorder.remove,
        )

        assert recorder.calls == []


class _DemoWorker:
    # [CLASS DOC] HELPER: Placeholder worker class with one method, used as the class to build an
    # rpc handle for.
    def report(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the text "ok".
        return "ok"


def _info(*, generation: int, port: int = 15000) -> WorkerInfo:
    # [FUNCTION DOC] HELPER: Builds a worker info named trainer-engine-actor-0-0 with the given
    # generation, an rpc address 10.0.0.7 at the given port (default 15000), no gpus and the demo
    # worker class path of this module.
    return WorkerInfo(
        name="trainer-engine-actor-0-0",
        generation=generation,
        self_addrs={"rpc": HostAndPort(host="10.0.0.7", port=port)},
        gpu_ids=[],
        worker_class=f"{__name__}._DemoWorker",
    )


class TestBuildRpcHandleFor:
    # [CLASS DOC] Tests of building an rpc handle from a worker info.
    def test_the_handle_points_at_the_address_the_worker_serves_on(self):
        # [FUNCTION DOC] The handle's transport server url (private attribute) equals
        # "http://10.0.0.7:15001".
        """A handle aimed anywhere else silently drives another worker on the same node."""
        handle = build_rpc_handle_of_worker_info(_info(generation=1, port=15001))

        assert handle._transport._server_url == "http://10.0.0.7:15001"


class TestBuildRpcHandle:
    # [CLASS DOC] Tests of building an rpc handle from a class and an address map.
    def test_build_rpc_handle_refuses_addresses_without_an_rpc_port(self):
        # [FUNCTION DOC] An address map holding only a metrics port raises an assertion error
        # matching "has no 'rpc' port".
        """A named non-RPC address cannot be substituted for the required RPC endpoint."""
        addrs = {"metrics": HostAndPort(host="10.0.0.7", port=15002)}

        with pytest.raises(AssertionError, match="has no 'rpc' port"):
            build_rpc_handle(worker_class=_DemoWorker, addrs=addrs)
