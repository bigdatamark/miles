# [MODULE DOC]
# Tests for how a trainer cell configures the rendezvous (master) address of its workers before
# the init call: every worker receives the address that the manager allocated to the first worker
# (with IPv6 bracket characters removed), configuration strictly precedes init, and a failure to
# configure any worker aborts before init and tears the whole cell down (marked errored, workers
# killed and awaited, health checking never started). Uses dummy ray actors and a fake worker
# manager from the shared conftest.
import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import RecordingHealthChecker, get_raw_actor_handles, make_cell, make_indep_dp_info

from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.asyncio


def _calls_of(cell, method: str) -> list:
    # [FUNCTION DOC] HELPER: For each worker of the cell, fetches the actor's call log and returns
    # the list of recorded calls whose method name equals the given one; the result has one list
    # per worker, in worker order.
    return [
        [call for call in ray.get(handle.get_calls.remote()) if call[0] == method]
        for handle in get_raw_actor_handles(cell)
    ]


class TestMasterAddrConfiguration:
    # [CLASS DOC] Tests the master address configuration sent to workers during cell init.
    async def test_every_worker_is_told_the_master_address_before_init(self):
        # [FUNCTION DOC] After initing a 2-actor cell, each worker has exactly one master address
        # configuration call with address 10.0.0.1 and port 20000 (the fake manager's default).
        """Workers rendezvous on the address the worker manager allocated for the cell."""
        cell = make_cell(0, actor_count=2)

        await cell.init(indep_dp_info=make_indep_dp_info(quorum_id=0), indep_dp_store_addr=None)

        for [call] in _calls_of(cell, "configure_master_addr_and_port"):
            assert call[2] == {"master_addr": "10.0.0.1", "master_port": 20000}

    async def test_the_master_address_is_configured_before_the_process_group_is_built(self):
        # [FUNCTION DOC] In the first worker's call log, the address configuration appears before init.
        """A rank that runs init first would build the process group without the address."""
        cell = make_cell(0, actor_count=1)

        await cell.init(indep_dp_info=make_indep_dp_info(quorum_id=0), indep_dp_store_addr=None)

        methods = [call[0] for call in ray.get(get_raw_actor_handles(cell)[0].get_calls.remote())]
        assert methods.index("configure_master_addr_and_port") < methods.index("init")

    async def test_every_worker_rendezvous_on_the_first_workers_unbracketed_endpoint(self):
        # [FUNCTION DOC] The fake manager reports different bracketed IPv6 master addresses for the
        # two workers ([fe80::a] port 21001 and [fe80::b] port 21002). After init, both workers are
        # configured with the first worker's endpoint with brackets removed: address fe80::a, port 21001.
        """Ranks must share one endpoint, and torch.distributed rejects the URI brackets of an IPv6 host."""
        train_conftest.fake_worker_manager.master_addr_per_worker = [
            HostAndPort(host="[fe80::a]", port=21001),
            HostAndPort(host="[fe80::b]", port=21002),
        ]
        cell = make_cell(0, actor_count=2)

        await cell.init(indep_dp_info=make_indep_dp_info(quorum_id=0), indep_dp_store_addr=None)

        for [call] in _calls_of(cell, "configure_master_addr_and_port"):
            assert call[2] == {"master_addr": "fe80::a", "master_port": 21001}


class _FailingConfigureWorkerHandle:
    # [CLASS DOC] HELPER: Fake worker handle whose master address configuration always fails with
    # an unreachable-worker error. Records call names and counts kill and wait-dead requests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty call record and zero kill and wait counts.
        self.calls: list[str] = []
        self.kill_self_count: int = 0
        self.wait_dead_count: int = 0

    async def configure_master_addr_and_port(self, **_kwargs) -> None:
        # [FUNCTION DOC] HELPER: Records the call, then raises a worker-unreachable error saying
        # the worker died before the rendezvous.
        self.calls.append("configure_master_addr_and_port")
        raise WorkerUnreachableError("worker died before rendezvous")

    async def init(self, **_kwargs) -> None:
        # [FUNCTION DOC] HELPER: Records the call; returns none.
        self.calls.append("init")

    async def kill_self(self) -> None:
        # [FUNCTION DOC] HELPER: Counts the kill request.
        self.kill_self_count += 1

    async def wait_dead(self, *, timeout: float) -> None:
        # [FUNCTION DOC] HELPER: Counts the wait-dead request; the timeout is ignored.
        self.wait_dead_count += 1


class TestMasterAddrConfigurationFailure:
    # [CLASS DOC] Tests the failure path of the master address configuration.
    async def test_a_failed_rendezvous_aborts_before_init_and_tears_the_whole_cell_down(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] With the cell's handles replaced by two failing handles and a recording
        # health checker, initing the cell raises the worker-unreachable error. Each handle saw
        # only the configuration call (init never dispatched), each was asked to kill itself once,
        # the cell is errored and not alive, and the health checker was never started.
        """A rank that never learned the address would hang the others in the process-group build."""
        checker = RecordingHealthChecker()
        cell = make_cell(0, actor_count=2, health_checker=checker)
        handles = [_FailingConfigureWorkerHandle(), _FailingConfigureWorkerHandle()]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)

        with pytest.raises(WorkerUnreachableError):
            await cell.init(indep_dp_info=make_indep_dp_info(quorum_id=0), indep_dp_store_addr=None)

        assert [handle.calls for handle in handles] == [["configure_master_addr_and_port"]] * 2
        assert [handle.kill_self_count for handle in handles] == [1, 1]
        assert not cell.is_alive
        assert cell.is_errored
        assert checker.start_count == 0
