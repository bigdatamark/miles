# [MODULE DOC]
# Unit tests for health checking of rollout server cells: when a server cell's health checker is
# active (depending on cell state and on the global forced-pause activeness, which wins over the
# state), that adding a cell to a rollout server starts a simple health checker only when rollout
# fault tolerance is enabled (otherwise a no-op checker), that a cell added while probing is paused
# stays inactive, and that disposing a cell stops its checker task. Network activation and health
# probes are replaced by stubs; every cell created is registered with a fixture (module-wide
# marker) that disposes tracked cells after each test.
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.rollout_server import RolloutServer
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActiveAndEpoch, NoopHealthChecker, SimpleHealthChecker
from miles.utils.workers.launch_gate import GATE_PORT_NAME
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")

_ADDRS: NamedHostAndPorts = {
    "primary": HostAndPort(host="10.0.0.1", port=30000),
    GATE_PORT_NAME: HostAndPort(host="10.0.0.1", port=13000),
}


class _StubProvider:
    # [CLASS DOC] HELPER: Stub worker provider that answers every address lookup with the same
    # module-level addresses (primary 10.0.0.1:30000 and the launch gate 10.0.0.1:13000).
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Ignores the worker name and returns the shared address mapping.
        return _ADDRS


def _make_meta(cell_id: str = "cell-0", **overrides) -> ServerCellMetadata:
    # [FUNCTION DOC] HELPER: Builds cell metadata for model default, worker type regular, one GPU
    # per engine, offset 0, no api key, worker name equal to the cell id plus -0, no offload,
    # weight updates on and hash pseudo-hash-0; any keyword overrides replace those fields.
    return ServerCellMetadata(
        **{
            "model_id": "default",
            "worker_type": "regular",
            "cell_id": cell_id,
            "num_gpus_per_engine": 1,
            "gpu_offset": 0,
            "sglang_api_key": None,
            "worker_name": f"{cell_id}-0",
            "needs_offload": False,
            "update_weights": True,
            "workers_hash": "pseudo-hash-0",
            **overrides,
        }
    )


def _make_server(*, ft_components=("rollout",), **overrides) -> RolloutServer:
    # [FUNCTION DOC] HELPER: Builds a real rollout server with no cells, colocated test arguments
    # carrying the given fault tolerance components (default only rollout), a named context lock,
    # the stub provider and any extra constructor keyword overrides.
    return RolloutServer(
        server_cells={},
        args=make_args(colocate=True, ft_components=list(ft_components)),
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        **overrides,
    )


def _make_cell(*, global_activeness=None, router=None, **meta_overrides) -> ServerCell:
    # [FUNCTION DOC] HELPER: Builds a real server cell (arguments with rollout fault tolerance,
    # metadata from the overrides, the given router client or an empty one, the stub provider, and
    # the given global activeness callable or one that always reports active at epoch 0) and
    # registers it with the tracking helper so it is disposed after the test. Returns the cell.
    return track_server_cell(
        ServerCell(
            args=make_args(ft_components=["rollout"]),
            meta=_make_meta(**meta_overrides),
            router_api_client=router or SimpleNamespace(),
            provider=_StubProvider(),
            health_checker_activeness=global_activeness or (lambda: ActiveAndEpoch(active=True, epoch=0)),
        )
    )


def _stub_network(monkeypatch, *, ready: bool = True) -> None:
    # [FUNCTION DOC] HELPER: Patches the server cell module so that launch gate activation does
    # nothing and the server health probe returns the given readiness (default true).
    async def _activate(gate_url: str) -> None:
        # [FUNCTION DOC] HELPER: Does nothing.
        pass

    async def _probe(server_url: str, api_key, timeout: float = 5.0) -> bool:
        # [FUNCTION DOC] HELPER: Returns the captured readiness flag.
        return ready

    monkeypatch.setattr(server_cell_module, "activate_launch_gate", _activate)
    monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)


async def _noop_add_worker(**kwargs) -> None:
    # [FUNCTION DOC] HELPER: Stand-in for a router client's add-worker call; accepts and ignores
    # any keywords.
    pass


class TestHealthCheckerActiveness:
    # [CLASS DOC] Tests the cell's private active-and-epoch computation used by its health checker.
    async def test_a_gated_cell_is_not_probed(self):
        # [FUNCTION DOC] A fresh uninitialized cell is reported inactive.
        """A colocated cell waits gated for the next window; its port is not even listening."""
        cell = _make_cell()

        assert not cell._get_health_checker_active_and_epoch().active

    async def test_a_booting_cell_is_not_probed(self, monkeypatch):
        # [FUNCTION DOC] With the network stubbed to not ready, init leaves the cell booting and
        # inactive.
        """Its engine has not answered yet, so a probe would count a false failure."""
        _stub_network(monkeypatch, ready=False)
        cell = _make_cell()

        await cell.init()

        assert not cell._get_health_checker_active_and_epoch().active

    async def test_a_cell_holding_stale_weights_is_probed(self, monkeypatch):
        # [FUNCTION DOC] With the network stubbed ready, init then one tick moves the cell to
        # pending weights, where it is reported active.
        """It answers requests with stale weights, so a crash there is a real failure."""
        _stub_network(monkeypatch)
        cell = _make_cell()

        await cell.init()
        await cell.tick()

        assert cell._get_health_checker_active_and_epoch().active

    async def test_a_cell_that_skips_pending_weights_is_probed(self, monkeypatch):
        # [FUNCTION DOC] A cell with weight updates off and a router client that has a no-op
        # add-worker: init then tick takes it straight to serving, where it is reported active.
        """A frozen model goes straight to serving, and an unwatched serving cell is a hole in FT."""
        _stub_network(monkeypatch)
        cell = _make_cell(router=SimpleNamespace(add_worker=_noop_add_worker), update_weights=False)

        await cell.init()
        await cell.tick()

        assert cell._get_health_checker_active_and_epoch().active

    async def test_a_disposed_cell_is_not_probed(self, monkeypatch):
        # [FUNCTION DOC] After init, tick and dispose the cell is reported inactive.
        """Nothing is left to answer once the cell has been torn down."""
        _stub_network(monkeypatch)
        cell = _make_cell()

        await cell.init()
        await cell.tick()
        await cell.dispose()

        assert not cell._get_health_checker_active_and_epoch().active

    async def test_a_forced_pause_wins_over_the_cell_state(self, monkeypatch):
        # [FUNCTION DOC] Uses a global activeness callable backed by a mutable flag. After init and
        # tick the cell is active; flipping the flag to false makes it inactive although its own
        # state is unchanged.
        """Engines are unusable while offloaded or mid weight update, whatever state they are in."""
        _stub_network(monkeypatch)
        active = {"value": True}
        cell = _make_cell(global_activeness=lambda: ActiveAndEpoch(active=active["value"], epoch=0))

        await cell.init()
        await cell.tick()
        assert cell._get_health_checker_active_and_epoch().active

        active["value"] = False

        assert not cell._get_health_checker_active_and_epoch().active


class TestAddCellHealthChecker:
    # [CLASS DOC] Tests the health checker created when a rollout server adds a cell, under the
    # server's lock (the server is disposed at the end of each test).
    async def test_a_new_cell_starts_its_checker(self):
        # [FUNCTION DOC] Adds an offloaded cell to a server with rollout fault tolerance; the
        # cell's checker is a simple health checker whose background task is already running.
        """Activeness is pulled per loop, so starting unconditionally is safe and needs no replay."""
        srv = _make_server()

        async with srv.context_lock:
            await srv.add_cell(_make_meta(needs_offload=True))

            checker = srv.server_cells["cell-0"]._health_checker
            assert isinstance(checker, SimpleHealthChecker)
            assert checker._task is not None
            await srv.dispose()

    async def test_a_cell_added_mid_window_does_not_probe(self):
        # [FUNCTION DOC] Pauses the server's activeness tracker, then adds a cell; the new cell is
        # reported inactive.
        """The window releases the lock, so reconcile can add a cell while probing is paused."""
        srv = _make_server()
        srv.health_checker_activeness.bump_active(False)

        async with srv.context_lock:
            await srv.add_cell(_make_meta(needs_offload=True))

            assert not srv.server_cells["cell-0"]._get_health_checker_active_and_epoch().active
            await srv.dispose()

    async def test_no_checker_is_created_without_rollout_fault_tolerance(self):
        # [FUNCTION DOC] A server without any fault tolerance components gives an added cell a
        # no-op health checker.
        """Without rollout FT nothing consumes the health status, so nothing probes."""
        srv = _make_server(ft_components=())

        async with srv.context_lock:
            await srv.add_cell(_make_meta(needs_offload=True))

            assert isinstance(srv.server_cells["cell-0"]._health_checker, NoopHealthChecker)
            await srv.dispose()


class TestDispose:
    # [CLASS DOC] Tests that disposing a cell stops its health checker.
    async def test_dispose_stops_the_checker_task(self):
        # [FUNCTION DOC] A new cell has a running checker task; after disposing the cell the task
        # reference is cleared.
        """An inactive checker still polls its predicate forever unless the task is cancelled."""
        cell = _make_cell()
        assert cell._health_checker._task is not None

        await cell.dispose()

        assert cell._health_checker._task is None
