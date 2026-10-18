# [MODULE DOC]
# Unit tests for the state machine of a rollout server cell: init (opens the launch gate and moves
# to initializing, retryable on address or gate failure, skipped gate for external engines, rejected
# twice), the periodic tick (probe readiness, move to pending weights, offload memory for colocated
# cells, the optional weight-update checker snapshot and tensor reset, direct serving for frozen
# models or debug rollout-only, retry on failure, and the initializing deadline), the address
# accessor, marking weights ready (registering with the router and serving), dispose from any
# stage, and the memory and weight check operations forwarded to the engine. Everything the cell
# reaches over the network (launch gate, health probe, engine api client, address computation) is
# replaced by recording fakes through a fixture; routers are recording fakes. All tests are marked
# with a fixture that disposes the tracked cells.
from __future__ import annotations

import time

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout import server_cell as server_cell_module
from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")

_ADDR_INFO = CellAddrInfo(
    server_url="http://10.0.0.1:30000",
    bootstrap_port=None,
    gate_url="http://10.0.0.1:13000",
)

_ADDR_INFO_NO_GATE = CellAddrInfo(
    server_url="http://10.0.0.1:30000",
    bootstrap_port=None,
    gate_url=None,
)


def _make_meta(**overrides) -> ServerCellMetadata:
    # [FUNCTION DOC] HELPER: Builds cell metadata for model default, regular worker type, cell id
    # inference-engine-0-0-0, worker inference-engine-0-0-0-0, one GPU, offset 0, no api key, no
    # offload, weight updates on and hash pseudo-hash-0; keyword overrides replace those fields.
    return ServerCellMetadata(
        **{
            "model_id": "default",
            "worker_type": "regular",
            "cell_id": "inference-engine-0-0-0",
            "num_gpus_per_engine": 1,
            "gpu_offset": 0,
            "sglang_api_key": None,
            "worker_name": "inference-engine-0-0-0-0",
            "needs_offload": False,
            "update_weights": True,
            "workers_hash": "pseudo-hash-0",
            **overrides,
        }
    )


class _StubProvider:
    # [CLASS DOC] HELPER: Stub worker provider answering every lookup with primary 10.0.0.1:30000
    # and gate 10.0.0.1:13000.
    async def get_addrs(self, worker_name: str) -> dict[str, HostAndPort]:
        # [FUNCTION DOC] HELPER: Ignores the worker name and returns the fixed address mapping.
        return dict(
            primary=HostAndPort(host="10.0.0.1", port=30000),
            gate=HostAndPort(host="10.0.0.1", port=13000),
        )


class _RecordingRouterApiClient:
    # [CLASS DOC] HELPER: Fake router client recording add and remove worker calls (name plus
    # keywords) and optionally raising a configured error from add-worker (after recording).
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the empty call list and unsets the add-worker error.
        self.calls: list[tuple[str, dict]] = []
        self.add_worker_error: Exception | None = None

    async def add_worker(self, **kwargs):
        # [FUNCTION DOC] HELPER: Records the call, then raises the configured error if set.
        self.calls.append(("add_worker", kwargs))
        if self.add_worker_error is not None:
            raise self.add_worker_error

    async def remove_worker(self, **kwargs):
        # [FUNCTION DOC] HELPER: Records the call.
        self.calls.append(("remove_worker", kwargs))


class _RecordingApiClient:
    # [CLASS DOC] HELPER: Fake engine api client recording memory release, memory resume and weight
    # check calls in a class-level list shared by all instances (reset by the fixture).
    calls: list[tuple[str, dict]] = []

    def __init__(self, server_url: str, api_key: str | None = None):
        # [FUNCTION DOC] HELPER: Stores the server url and optional api key.
        self.server_url = server_url
        self.api_key = api_key

    async def release_memory_occupation(self, tags=None):
        # [FUNCTION DOC] HELPER: Records a release call with the tags.
        _RecordingApiClient.calls.append(("release", dict(tags=tags)))

    async def resume_memory_occupation(self, tags=None):
        # [FUNCTION DOC] HELPER: Records a resume call with the tags.
        _RecordingApiClient.calls.append(("resume", dict(tags=tags)))

    async def check_weights(self, **kwargs):
        # [FUNCTION DOC] HELPER: Records a check-weights call with its keywords.
        _RecordingApiClient.calls.append(("check_weights", kwargs))


class _ResultApiClient:
    # [CLASS DOC] HELPER: Fake engine api client with class-level state shared by all instances: a
    # call list, a mapping from operation name to the result to return, and a mapping from
    # operation name to an error to raise. Each operation records its call (name plus keywords),
    # raises the configured error if any, else returns the configured result (or nothing).
    calls: list[tuple[str, dict]] = []
    results: dict[str, dict] = {}
    errors: dict[str, Exception] = {}

    def __init__(self, server_url: str, api_key: str | None = None):
        # [FUNCTION DOC] HELPER: Stores the server url and optional api key.
        self.server_url = server_url
        self.api_key = api_key

    async def release_memory_occupation(self, tags=None):
        # [FUNCTION DOC] HELPER: Delegates to the recording helper under the name
        # release_memory_occupation, with the tags.
        return await self._record("release_memory_occupation", dict(tags=tags))

    async def resume_memory_occupation(self, tags=None):
        # [FUNCTION DOC] HELPER: Delegates to the recording helper under the name
        # resume_memory_occupation, with the tags.
        return await self._record("resume_memory_occupation", dict(tags=tags))

    async def check_weights(self, action, allow_quant_error=False, selector="all", skip_list=None):
        # [FUNCTION DOC] HELPER: Delegates to the recording helper under the name check_weights,
        # with the action, quantization error flag, selector and skip list as keywords.
        return await self._record(
            "check_weights",
            dict(action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list),
        )

    async def _record(self, name: str, kwargs: dict):
        # [FUNCTION DOC] HELPER: Appends the name and keywords to the shared call list, raises the
        # configured error for that name if there is one, otherwise returns the configured result
        # for that name (nothing if unset).
        _ResultApiClient.calls.append((name, kwargs))
        if (error := _ResultApiClient.errors.get(name)) is not None:
            raise error
        return _ResultApiClient.results.get(name)


@pytest.fixture
def cell_env(monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Resets the recording api client's shared call list, then
    # patches the server cell module and class: launch gate activation records the gate url, the
    # health probe returns a mutable ready flag (initially true), the engine api client is the
    # recording fake, and the cell's address computation returns the fixed address info with a gate.
    # Returns a mapping with the activated gate url list, the mutable health mapping (key ready)
    # and the recorded memory and weight check calls (the shared class-level list).
    """Stub out everything the cell reaches over the network, and record the calls."""
    _RecordingApiClient.calls = []
    activated: list[str] = []
    health: dict[str, bool] = {"ready": True}

    async def _activate(gate_url: str) -> None:
        # [FUNCTION DOC] HELPER: Records the gate url.
        activated.append(gate_url)

    async def _compute_addr_info(self) -> CellAddrInfo:
        # [FUNCTION DOC] HELPER: Returns the fixed address info with a gate.
        return _ADDR_INFO

    async def _probe(server_url: str, api_key, timeout: float = 5.0) -> bool:
        # [FUNCTION DOC] HELPER: Returns the current ready flag.
        return health["ready"]

    monkeypatch.setattr(server_cell_module, "activate_launch_gate", _activate)
    monkeypatch.setattr(server_cell_module, "probe_server_healthy", _probe)
    monkeypatch.setattr(server_cell_module, "SGLangApiClient", _RecordingApiClient)
    monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)

    return dict(activated=activated, health=health, memory_calls=_RecordingApiClient.calls)


@pytest.fixture
def result_api_client(cell_env, monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture. Builds on the cell environment, clears the result api client's
    # shared calls, results and errors, and makes the cell use that fake as its engine api client.
    # Returns the fake class so tests can configure results and errors.
    """Give the cell an api client whose endpoints return distinct results and can be made to fail."""
    _ResultApiClient.calls = []
    _ResultApiClient.results = {}
    _ResultApiClient.errors = {}
    monkeypatch.setattr(server_cell_module, "SGLangApiClient", _ResultApiClient)

    return _ResultApiClient


def _make_cell(
    *, router: _RecordingRouterApiClient | None = None, args_overrides=None, **meta_overrides
) -> ServerCell:
    # [FUNCTION DOC] HELPER: Builds a real server cell with test arguments (plus the argument
    # overrides), metadata from the meta overrides, the given router client or a new recording one,
    # and the stub provider; registers it with the tracking helper so it is disposed after the test.
    return track_server_cell(
        ServerCell(
            args=make_args(**(args_overrides or {})),
            meta=_make_meta(**meta_overrides),
            router_api_client=router or _RecordingRouterApiClient(),
            provider=_StubProvider(),
        )
    )


class TestInit:
    # [CLASS DOC] Tests the cell's init, which computes addresses, opens the launch gate (when the
    # cell has one) and moves from uninitialized to initializing without waiting for readiness.
    async def test_a_fresh_cell_has_not_been_initialized_yet(self):
        # [FUNCTION DOC] A newly built cell is in the uninitialized state.
        """A constructed cell is only bookkeeping until someone releases its engine."""
        assert isinstance(_make_cell()._state, StateUninitialized)

    async def test_init_releases_the_gate_and_moves_to_initializing(self, cell_env):
        # [FUNCTION DOC] After init the gate url http://10.0.0.1:13000 was activated once, the state
        # is initializing, and the cell's address info is the fixed one.
        """Initialization is exactly the moment the engine is allowed to claim gpu memory."""
        cell = _make_cell()

        await cell.init()

        assert cell_env["activated"] == ["http://10.0.0.1:13000"]
        assert isinstance(cell._state, StateInitializing)
        assert cell.addr_info == _ADDR_INFO

    async def test_init_does_not_wait_for_the_engine_to_become_ready(self, cell_env):
        # [FUNCTION DOC] With the health probe reporting not ready, init still returns and the cell
        # is initializing.
        """Blocking here would stall the caller for the minutes an engine takes to load."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()

        await cell.init()

        assert cell.is_initializing

    async def test_initializing_twice_is_rejected(self, cell_env):
        # [FUNCTION DOC] A second init raises an assertion error; the cell stays initializing and the
        # gate was activated only once.
        """A second release would race the first engine's startup."""
        cell = _make_cell()
        await cell.init()

        with pytest.raises(AssertionError):
            await cell.init()

        assert cell.is_initializing
        assert len(cell_env["activated"]) == 1

    async def test_an_address_lookup_failure_leaves_the_cell_uninitialized_and_retryable(self, cell_env, monkeypatch):
        # [FUNCTION DOC] The address computation fails on its first call with a runtime error
        # ("not published yet") and succeeds afterwards. The first init raises it, the cell remains
        # uninitialized and no gate was activated; a second init succeeds and the cell is
        # initializing.
        """Worker addresses appear late, so a lookup error must not consume the cell's only init."""
        attempts = {"count": 0}

        async def _flaky_compute_addr_info(self) -> CellAddrInfo:
            # [FUNCTION DOC] HELPER: Counts the call; raises a runtime error on the first call and
            # returns the fixed address info afterwards.
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("worker addresses are not published yet")
            return _ADDR_INFO

        monkeypatch.setattr(ServerCell, "_compute_addr_info", _flaky_compute_addr_info)
        cell = _make_cell()

        with pytest.raises(RuntimeError, match="not published yet"):
            await cell.init()
        assert cell.is_uninitialized
        assert cell_env["activated"] == []

        await cell.init()

        assert isinstance(cell._state, StateInitializing)

    async def test_a_launch_gate_failure_leaves_the_cell_uninitialized_and_retryable(self, cell_env, monkeypatch):
        # [FUNCTION DOC] The gate activation fails on its first call ("gate refused the activation")
        # and records the url afterwards. The first init raises, the cell stays uninitialized; the
        # second init succeeds, the cell is initializing and the gate url was recorded once.
        """Recording the transition before the gate opens would leave the engine waiting forever."""
        attempts = {"count": 0}

        async def _flaky_activate(gate_url: str) -> None:
            # [FUNCTION DOC] HELPER: Counts the call; raises a runtime error on the first call and
            # records the gate url in the environment's activated list afterwards.
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("gate refused the activation")
            cell_env["activated"].append(gate_url)

        monkeypatch.setattr(server_cell_module, "activate_launch_gate", _flaky_activate)
        cell = _make_cell()

        with pytest.raises(RuntimeError, match="gate refused"):
            await cell.init()
        assert cell.is_uninitialized

        await cell.init()

        assert isinstance(cell._state, StateInitializing)
        assert cell_env["activated"] == ["http://10.0.0.1:13000"]

    async def test_a_cell_without_a_gate_port_skips_activation(self, cell_env, monkeypatch):
        # [FUNCTION DOC] The address computation returns address info without a gate url; after init
        # no gate was activated and the cell is initializing.
        """External engines run no launch-gate wrapper, so init must not dial one."""

        async def _compute(self) -> CellAddrInfo:
            # [FUNCTION DOC] HELPER: Returns the fixed address info without a gate url.
            return _ADDR_INFO_NO_GATE

        monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute)
        cell = _make_cell()

        await cell.init()

        assert cell_env["activated"] == []
        assert cell.is_initializing


class TestGatelessCellLifecycle:
    # [CLASS DOC] Tests the lifecycle of a cell without a launch gate (an external engine).
    async def test_a_gateless_cell_still_waits_for_a_weight_push_before_serving(self, cell_env, monkeypatch):
        # [FUNCTION DOC] With address info lacking a gate, init and one tick move the cell to
        # pending weights with no router calls; marking weights ready then makes it serving with
        # exactly one add-worker call.
        """An external cell keeps on-policy semantics: only a weight update publishes it to the router."""

        async def _compute(self) -> CellAddrInfo:
            # [FUNCTION DOC] HELPER: Returns the fixed address info without a gate url.
            return _ADDR_INFO_NO_GATE

        monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute)
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)

        await cell.init()
        await cell.tick()
        assert cell.is_pending_weights
        assert router.calls == []

        await cell.mark_weights_ready()

        assert cell.is_serving
        assert [name for name, _kwargs in router.calls] == ["add_worker"]


class TestTickReportsTheEngineEnv:
    # [CLASS DOC] Tests that the tick offers the engine environment reporter a chance to report.
    async def test_a_cell_that_serves_nothing_yet_is_not_asked_for_its_env(self, cell_env, monkeypatch):
        # [FUNCTION DOC] With the probe reporting not ready, the reporter's report-if-due method is
        # replaced by a recorder; after init and a tick the recorder was never called.
        """An engine that has not answered a probe cannot answer /server_info either."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()
        asked: list[str] = []
        monkeypatch.setattr(cell._env_reporter, "report_if_due", _record_into(asked))

        await cell.init()
        await cell.tick()

        assert asked == []

    async def test_a_serving_cell_is_asked_for_its_env_on_every_tick(self, cell_env, monkeypatch):
        # [FUNCTION DOC] With a ready engine and the reporter replaced by a recorder, init plus two
        # ticks result in the recorder being called twice with the cell id.
        # KNOWN ISSUE: the cell here is pending weights (it needs a weight push to serve), not
        # serving as the name suggests.
        """The reporter decides how often to actually read; the tick just keeps offering it the chance."""
        cell = _make_cell()
        asked: list[str] = []
        monkeypatch.setattr(cell._env_reporter, "report_if_due", _record_into(asked))

        await cell.init()
        await cell.tick()
        await cell.tick()

        assert asked == [cell.meta.cell_id, cell.meta.cell_id]


def _record_into(asked: list[str]):
    # [FUNCTION DOC] HELPER: Returns a replacement for the env reporter's report-if-due coroutine
    # function that appends the cell id it is called with to the given list.
    async def _report_if_due(*, cell_id: str, server_url: str, api_client) -> None:
        # [FUNCTION DOC] HELPER: Records the cell id; ignores the url and client.
        asked.append(cell_id)

    return _report_if_due


class TestTick:
    # [CLASS DOC] Tests the cell's tick, which advances initializing cells: probe readiness, then
    # (for offloaded cells) snapshot weights if the checker is on, release and resume memory, reset
    # tensors, and finally move to pending weights, or directly to serving for frozen models and
    # debug rollout-only runs (registering with the router). Failures leave the cell initializing so
    # a later tick retries.
    async def test_an_uninitialized_cell_is_not_probed(self, cell_env):
        # [FUNCTION DOC] Ticking a new cell leaves it uninitialized.
        """It has no address yet, so probing it would be dialing nothing."""
        cell = _make_cell()

        await cell.tick()

        assert isinstance(cell._state, StateUninitialized)

    async def test_a_cell_whose_engine_is_not_ready_stays_initializing(self, cell_env):
        # [FUNCTION DOC] With the probe not ready, two ticks after init leave the cell initializing.
        """Readiness is polled, so a not-yet-ready engine must simply be retried later."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()
        await cell.init()

        await cell.tick()
        await cell.tick()

        assert cell.is_initializing

    async def test_a_ready_engine_moves_on_to_awaiting_its_weights(self, cell_env):
        # [FUNCTION DOC] With a ready engine, init then tick leaves the cell in pending weights.
        """A loaded engine still serves nothing until the trainer has pushed weights."""
        cell = _make_cell()
        await cell.init()

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_ticking_a_pending_cell_again_does_not_redo_its_startup(self, cell_env):
        # [FUNCTION DOC] An offloaded cell is ticked into pending weights; a second tick changes
        # neither the state nor the recorded memory calls.
        """The sweep runs every second, so a second release would thrash the engine's memory."""
        cell = _make_cell(needs_offload=True)
        await cell.init()
        await cell.tick()
        calls_after_first_tick = list(cell_env["memory_calls"])

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)
        assert cell_env["memory_calls"] == calls_after_first_tick

    async def test_an_engine_sharing_gpus_with_the_trainer_hands_its_memory_back(self, cell_env):
        # [FUNCTION DOC] An offloaded cell after init and tick: the recorded engine calls are
        # exactly a release with no tags followed by a resume with the weights tag only.
        """Under colocation the engine must give the gpus back until the next rollout."""
        cell = _make_cell(needs_offload=True)
        await cell.init()

        await cell.tick()

        assert cell_env["memory_calls"] == [("release", dict(tags=None)), ("resume", dict(tags=["weights"]))]

    async def test_the_weight_checker_snapshots_before_the_memory_is_handed_back(self, cell_env):
        # [FUNCTION DOC] With the weight update equality check enabled on an offloaded cell, the
        # first check-weights call is recorded before the release call.
        """Releasing the occupation discards the loaded weights, so a later snapshot records garbage."""
        cell = _make_cell(needs_offload=True, args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        await cell.tick()

        names = [name for name, _kwargs in cell_env["memory_calls"]]
        assert names.index("check_weights") < names.index("release")

    async def test_an_engine_on_its_own_gpus_keeps_its_memory(self, cell_env):
        # [FUNCTION DOC] A cell without offload: init and tick record no engine calls at all.
        """Without colocation there is nobody to hand the memory to."""
        cell = _make_cell(needs_offload=False)
        await cell.init()

        await cell.tick()

        assert cell_env["memory_calls"] == []

    async def test_a_frozen_model_starts_serving_without_a_weight_update(self, cell_env):
        # [FUNCTION DOC] A cell with weight updates off: init then tick makes it serving and the
        # router recorded exactly one add-worker call.
        """A model the trainer never updates would otherwise wait for a push that never comes."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router, update_weights=False)
        await cell.init()

        await cell.tick()

        assert isinstance(cell._state, StateServing)
        assert [name for name, _kwargs in router.calls] == ["add_worker"]

    async def test_a_frozen_cell_whose_router_registration_fails_is_retried_by_a_later_tick(self, cell_env):
        # [FUNCTION DOC] A frozen cell with a router whose first add-worker fails with a runtime
        # error. The first tick raises and the cell is still initializing; the second tick makes it
        # serving and the router recorded exactly one successful add-worker call.
        """Nothing else ever revisits a frozen cell, so a transient router error must not strand it."""

        class _FlakyRouter(_RecordingRouterApiClient):
            # [CLASS DOC] HELPER: Recording router that fails the first add-worker call (before
            # recording) and behaves normally afterwards.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the base and sets the remaining failures to 1.
                super().__init__()
                self.failures = 1

            async def add_worker(self, **kwargs):
                # [FUNCTION DOC] HELPER: Raises a runtime error while failures remain (decrementing),
                # otherwise records the call through the base implementation.
                if self.failures > 0:
                    self.failures -= 1
                    raise RuntimeError("router rejected the worker")
                await super().add_worker(**kwargs)

        router = _FlakyRouter()
        cell = _make_cell(router=router, update_weights=False)
        await cell.init()

        with pytest.raises(RuntimeError):
            await cell.tick()
        assert cell.is_initializing

        await cell.tick()

        assert isinstance(cell._state, StateServing)
        assert [name for name, _kwargs in router.calls] == ["add_worker"]

    @pytest.mark.parametrize(
        ("failing_operation", "error_message"),
        [
            ("release_memory_occupation", "engine refused to release its memory"),
            ("resume_memory_occupation", "engine refused to resume its weights"),
        ],
    )
    async def test_a_cell_whose_memory_reconfiguration_fails_is_retried_by_a_later_tick(
        self, result_api_client, failing_operation: str, error_message: str
    ):
        # [FUNCTION DOC] Parametrized over a failing release and a failing resume. The offloaded
        # cell's tick raises the configured runtime error and the cell stays initializing; after the
        # error is removed a second tick moves it to pending weights.
        """A failed release or resume must leave initialization retryable instead of stranding the cell as pending."""
        result_api_client.errors[failing_operation] = RuntimeError(error_message)
        cell = _make_cell(needs_offload=True)
        await cell.init()

        with pytest.raises(RuntimeError, match=error_message):
            await cell.tick()
        assert cell.is_initializing

        result_api_client.errors.pop(failing_operation)
        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_a_cell_whose_weight_snapshot_fails_is_retried_by_a_later_tick(self, cell_env, monkeypatch):
        # [FUNCTION DOC] With the weight equality check on and an engine client whose first
        # check-weights call fails ("could not snapshot"): the first tick raises and the cell stays
        # initializing; the second tick moves it to pending weights.
        """Without the baseline the later comparison is meaningless, so the cell must not move on regardless."""

        class _FlakyApiClient(_RecordingApiClient):
            # [CLASS DOC] HELPER: Recording engine client whose first check-weights call fails.
            failures = 1

            async def check_weights(self, **kwargs):
                # [FUNCTION DOC] HELPER: Raises a runtime error while the class-level failure count
                # is positive (decrementing it); otherwise records through the base implementation.
                if _FlakyApiClient.failures > 0:
                    _FlakyApiClient.failures -= 1
                    raise RuntimeError("engine could not snapshot its weights")
                await super().check_weights(**kwargs)

        monkeypatch.setattr(server_cell_module, "SGLangApiClient", _FlakyApiClient)
        cell = _make_cell(args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        with pytest.raises(RuntimeError, match="could not snapshot"):
            await cell.tick()
        assert cell.is_initializing

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_debug_rollout_only_starts_serving_without_a_weight_update(self, cell_env):
        # [FUNCTION DOC] With the debug rollout-only argument, init then tick makes the cell serving.
        """With no trainer running, the engine must serve the weights it loaded from disk."""
        cell = _make_cell(args_overrides=dict(debug_rollout_only=True))
        await cell.init()

        await cell.tick()

        assert isinstance(cell._state, StateServing)

    async def test_an_engine_that_becomes_ready_later_is_picked_up_by_a_later_tick(self, cell_env):
        # [FUNCTION DOC] Not ready at first tick; after the probe flips to ready, another tick moves
        # the cell to pending weights.
        """Polling is only worth anything if the transition actually happens once readiness flips."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()
        await cell.init()
        await cell.tick()

        cell_env["health"]["ready"] = True
        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_a_cell_awaiting_a_real_weight_update_is_not_published_by_the_sweep(self, cell_env):
        # [FUNCTION DOC] A normal updatable cell ticked twice stays pending weights and the router
        # recorded no calls.
        """Publishing before the trainer pushes weights would serve the checkpoint's stale weights."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)
        assert router.calls == []

    async def test_a_serving_cell_is_inert_under_the_sweep(self, cell_env):
        # [FUNCTION DOC] A frozen cell that is already serving after one tick stays serving after
        # another and has still only one add-worker call recorded.
        """Re-publishing a live engine would duplicate it in the router."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router, update_weights=False)
        await cell.init()
        await cell.tick()

        await cell.tick()

        assert isinstance(cell._state, StateServing)
        assert [name for name, _kwargs in router.calls] == ["add_worker"]

    async def test_the_weight_baseline_is_snapshotted_before_memory_is_released(self, cell_env):
        # [FUNCTION DOC] Offloaded cell with the equality check on: the recorded engine calls, with
        # weight checks labeled by action, are exactly snapshot, release, resume, reset tensors.
        """The checker's baseline must be the freshly loaded checkpoint, not the remapped weight storage."""
        cell = _make_cell(needs_offload=True, args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        await cell.tick()

        names = [
            f"check_weights:{kwargs['action']}" if name == "check_weights" else name
            for name, kwargs in cell_env["memory_calls"]
        ]
        assert names == ["check_weights:snapshot", "release", "resume", "check_weights:reset_tensors"]

    async def test_the_snapshot_is_taken_over_the_whole_model_without_a_skip_list(self, cell_env):
        # [FUNCTION DOC] With the equality check on and a skip list configured, the one recorded
        # snapshot call uses selector all, no quantization error allowance and no skip list.
        """The baseline must match what the controller-side reset and comparison later cover."""
        cell = _make_cell(
            args_overrides=dict(check_weight_update_equal=True, check_weight_update_skip_list=["lm_head"])
        )
        await cell.init()

        await cell.tick()

        snapshot_calls = [
            kwargs
            for name, kwargs in cell_env["memory_calls"]
            if name == "check_weights" and kwargs["action"] == "snapshot"
        ]
        assert snapshot_calls == [dict(action="snapshot", allow_quant_error=False, selector="all", skip_list=None)]

    async def test_a_cell_primes_the_weight_update_checker_before_its_first_update(self, cell_env):
        # [FUNCTION DOC] With the equality check on and skip list a: the weight check actions are a
        # snapshot then a tensor reset, and the reset uses the configured skip list.
        """The checker scrambles the engine tensors so a no-op weight update cannot pass unnoticed."""
        cell = _make_cell(args_overrides=dict(check_weight_update_equal=True, check_weight_update_skip_list=["a"]))
        await cell.init()

        await cell.tick()

        checker_calls = [kwargs for name, kwargs in cell_env["memory_calls"] if name == "check_weights"]
        assert [kwargs["action"] for kwargs in checker_calls] == ["snapshot", "reset_tensors"]
        assert checker_calls[1]["skip_list"] == ["a"]

    async def test_the_weight_update_checker_stays_out_of_the_way_when_it_is_disabled(self, cell_env):
        # [FUNCTION DOC] With the check disabled (default), no weight check is recorded.
        """The checker is a ci-only tool and must not touch engines in a normal run."""
        cell = _make_cell()
        await cell.init()

        await cell.tick()

        assert [name for name, _kwargs in cell_env["memory_calls"] if name == "check_weights"] == []

    async def test_a_frozen_cell_is_not_scrambled_because_nothing_will_rewrite_it(self, cell_env):
        # [FUNCTION DOC] A frozen cell with the check enabled records no weight check.
        """Resetting tensors nobody updates would leave the frozen model serving garbage."""
        cell = _make_cell(update_weights=False, args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        await cell.tick()

        assert [name for name, _kwargs in cell_env["memory_calls"] if name == "check_weights"] == []

    async def test_a_failing_weight_check_leaves_the_cell_initializing_for_a_retry(self, cell_env, monkeypatch):
        # [FUNCTION DOC] Replaces the cell's weight check method with one that fails the tensor
        # reset; with the check enabled, a tick raises a runtime error and the cell stays
        # initializing.
        """Publishing despite a failed reset would silently disable a check the user asked for."""

        async def _failing_check_weights(self, action, allow_quant_error, selector, skip_list):
            # [FUNCTION DOC] HELPER: Raises a runtime error for the reset tensors action and does
            # nothing for other actions.
            if action == "reset_tensors":
                raise RuntimeError("engine refused to reset its tensors")

        monkeypatch.setattr(ServerCell, "check_weights", _failing_check_weights)
        cell = _make_cell(args_overrides=dict(check_weight_update_equal=True))
        await cell.init()

        with pytest.raises(RuntimeError):
            await cell.tick()

        assert cell.is_initializing

    async def test_a_weight_check_that_recovers_lets_the_cell_move_on(self, cell_env, monkeypatch):
        # [FUNCTION DOC] The cell's weight check fails the tensor reset once and then delegates to
        # the original. The first tick raises; a second tick moves the cell to pending weights.
        """The retry is only worth anything if a later tick actually completes the transition."""
        failures = {"remaining": 1}
        original_check_weights = ServerCell.check_weights

        async def _flaky_check_weights(self, action, allow_quant_error, selector, skip_list):
            # [FUNCTION DOC] HELPER: For the reset tensors action while failures remain, decrements
            # the counter and raises a runtime error; otherwise calls the original method with the
            # same arguments and returns its result.
            if action == "reset_tensors" and failures["remaining"] > 0:
                failures["remaining"] -= 1
                raise RuntimeError("engine refused to reset its tensors")
            return await original_check_weights(
                self, action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
            )

        monkeypatch.setattr(ServerCell, "check_weights", _flaky_check_weights)
        cell = _make_cell(args_overrides=dict(check_weight_update_equal=True))
        await cell.init()
        with pytest.raises(RuntimeError):
            await cell.tick()

        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)


def _pretend_started_long_ago(cell: ServerCell) -> None:
    # [FUNCTION DOC] HELPER: Replaces the cell's private state with a copy whose start time is older
    # than the initializing timeout constant (plus one second), using the current monotonic clock.
    cell._state = cell._state.model_copy(
        update=dict(start_time=time.monotonic() - server_cell_module.INITIALIZING_TIMEOUT_SECONDS - 1.0)
    )


class TestInitializingDeadline:
    # [CLASS DOC] Tests the cell's initializing deadline flag.
    async def test_a_cell_that_never_becomes_ready_reports_itself_past_its_deadline(self, cell_env):
        # [FUNCTION DOC] An engine never ready, with the start time pushed past the deadline: after
        # a tick the cell reports initializing past deadline.
        """A replacement whose engine died during startup used to stay Initializing forever."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()
        await cell.init()
        _pretend_started_long_ago(cell)

        await cell.tick()

        assert cell.is_initializing_past_deadline

    async def test_a_cell_within_its_deadline_is_not_reported(self, cell_env):
        # [FUNCTION DOC] An engine not ready yet but recently started: after a tick the cell is
        # initializing and not past its deadline.
        """Engines take minutes to load, so a slow startup must not be torn down."""
        cell_env["health"]["ready"] = False
        cell = _make_cell()
        await cell.init()

        await cell.tick()

        assert cell.is_initializing
        assert not cell.is_initializing_past_deadline

    async def test_a_transient_failure_before_the_deadline_is_not_reported(self, cell_env):
        # [FUNCTION DOC] A frozen cell whose router always fails add-worker: the tick raises a
        # runtime error, and the cell is still initializing and not past its deadline.
        """Reporting on the first error would throw away the existing retry contract."""

        class _FlakyRouter(_RecordingRouterApiClient):
            # [CLASS DOC] HELPER: Router whose add-worker always fails.
            async def add_worker(self, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises a runtime error.
                raise RuntimeError("router rejected the worker")

        cell = _make_cell(router=_FlakyRouter(), update_weights=False)
        await cell.init()

        with pytest.raises(RuntimeError):
            await cell.tick()

        assert cell.is_initializing
        assert not cell.is_initializing_past_deadline

    async def test_a_cell_that_finished_initializing_is_never_reported(self, cell_env):
        # [FUNCTION DOC] A cell ticked into pending weights is not past its deadline.
        """The deadline only governs startup; a serving cell must not be reclaimed by it."""
        cell = _make_cell()
        await cell.init()
        await cell.tick()

        assert isinstance(cell._state, StatePendingWeights)
        assert not cell.is_initializing_past_deadline


class TestAddressing:
    # [CLASS DOC] Tests the cell's address accessor.
    async def test_an_uninitialized_cell_has_no_address_to_offer(self, cell_env):
        # [FUNCTION DOC] Reading the address info of an uninitialized cell raises an assertion error.
        """Reading an address before the gate opens is what silently dialed nothing before."""
        cell = _make_cell()

        with pytest.raises(AssertionError):
            _ = cell.addr_info


class TestMarkWeightsReady:
    # [CLASS DOC] Tests marking a pending-weights cell ready: register with the router, then serve.
    async def test_it_publishes_the_cell_to_the_router_and_starts_serving(self, cell_env):
        # [FUNCTION DOC] After init, tick and mark weights ready, the router recorded exactly one
        # add-worker call with worker url http://10.0.0.1:30000, worker type regular, the
        # non-legacy api and no bootstrap port; the cell is serving.
        """Registering earlier would route requests at an engine holding stale weights."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()

        await cell.mark_weights_ready()

        assert router.calls == [
            (
                "add_worker",
                dict(
                    worker_url="http://10.0.0.1:30000",
                    worker_type="regular",
                    use_legacy_api=False,
                    bootstrap_port=None,
                ),
            )
        ]
        assert isinstance(cell._state, StateServing)

    async def test_a_cell_stays_pending_weights_when_the_router_rejects_the_registration(self, cell_env):
        # [FUNCTION DOC] The router fails add-worker with "router returned 503": marking weights
        # ready raises that runtime error and the cell stays pending weights.
        """Marking it serving on a failed add_worker would strand the cell: never registered, never retried."""
        router = _RecordingRouterApiClient()
        router.add_worker_error = RuntimeError("router returned 503")
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()

        with pytest.raises(RuntimeError, match="router returned 503"):
            await cell.mark_weights_ready()

        assert isinstance(cell._state, StatePendingWeights)

    async def test_a_failed_registration_can_be_retried_by_a_later_end_update_weights(self, cell_env):
        # [FUNCTION DOC] First mark-ready fails at the router; after the error is cleared a second
        # mark-ready succeeds, two add-worker calls were recorded in total and the cell is serving.
        """Because the cell is still pending, the next weight window registers it instead of skipping it."""
        router = _RecordingRouterApiClient()
        router.add_worker_error = RuntimeError("router returned 503")
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()
        with pytest.raises(RuntimeError):
            await cell.mark_weights_ready()

        router.add_worker_error = None
        await cell.mark_weights_ready()

        assert len([name for name, _kwargs in router.calls if name == "add_worker"]) == 2
        assert isinstance(cell._state, StateServing)

    async def test_a_cell_that_is_still_initializing_cannot_be_marked_ready(self, cell_env):
        # [FUNCTION DOC] Marking an initializing (not yet ticked) cell ready raises an assertion error.
        """Its engine may not even be loaded, so publishing it would break routing."""
        cell = _make_cell()
        await cell.init()

        with pytest.raises(AssertionError):
            await cell.mark_weights_ready()

    async def test_a_prefill_cell_publishes_its_bootstrap_port(self, cell_env, monkeypatch):
        # [FUNCTION DOC] A prefill cell whose address info carries bootstrap port 8998: after init,
        # tick and mark ready, the recorded add-worker call has bootstrap port 8998.
        """PD disaggregation needs the decode side to dial this port."""

        async def _compute_addr_info(self) -> CellAddrInfo:
            # [FUNCTION DOC] HELPER: Returns address info with server url http://10.0.0.1:30000,
            # bootstrap port 8998 and gate url http://10.0.0.1:13000.
            return CellAddrInfo(
                server_url="http://10.0.0.1:30000", bootstrap_port=8998, gate_url="http://10.0.0.1:13000"
            )

        monkeypatch.setattr(ServerCell, "_compute_addr_info", _compute_addr_info)
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router, worker_type="prefill")
        await cell.init()
        await cell.tick()

        await cell.mark_weights_ready()

        assert router.calls[0][1]["bootstrap_port"] == 8998


class TestDispose:
    # [CLASS DOC] Tests disposal of cells in each stage of the state machine.
    async def test_a_serving_cell_is_withdrawn_from_the_router(self, cell_env):
        # [FUNCTION DOC] A serving cell, once disposed, shows router calls add-worker then
        # remove-worker, and its state is disposed.
        """Leaving it registered would route requests at an engine that is going away."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        await cell.init()
        await cell.tick()
        await cell.mark_weights_ready()

        await cell.dispose()

        assert [name for name, _kwargs in router.calls] == ["add_worker", "remove_worker"]
        assert isinstance(cell._state, StateDisposed)

    @pytest.mark.parametrize("stage", ["uninitialized", "initializing", "pending_weights"])
    async def test_a_cell_can_be_disposed_from_any_earlier_stage(self, cell_env, stage: str):
        # [FUNCTION DOC] Parametrized over uninitialized, initializing and pending weights. Brings
        # the cell to that stage, disposes it, and asserts it is disposed and the router saw no calls.
        """Reconcile removes cells at arbitrary moments, including mid-startup."""
        router = _RecordingRouterApiClient()
        cell = _make_cell(router=router)
        if stage != "uninitialized":
            await cell.init()
        if stage == "pending_weights":
            await cell.tick()

        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)
        assert router.calls == []

    async def test_disposing_twice_is_harmless(self, cell_env):
        # [FUNCTION DOC] Two consecutive disposes of a new cell raise nothing and leave it disposed.
        """Teardown paths overlap, so a second dispose must not raise."""
        cell = _make_cell()

        await cell.dispose()
        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)

    async def test_a_router_that_rejects_the_removal_still_disposes_the_cell(self, cell_env):
        # [FUNCTION DOC] A serving cell whose router fails remove-worker: dispose completes without
        # raising and the cell is disposed.
        """Teardown is how a wedged engine is reclaimed, so a router error must not abort it."""

        class _RejectingRouter(_RecordingRouterApiClient):
            # [CLASS DOC] HELPER: Router whose remove-worker always fails.
            async def remove_worker(self, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises a runtime error.
                raise RuntimeError("router rejected the removal")

        cell = _make_cell(router=_RejectingRouter())
        await cell.init()
        await cell.tick()
        await cell.mark_weights_ready()

        await cell.dispose()

        assert isinstance(cell._state, StateDisposed)

    async def test_a_disposed_cell_is_inert_under_the_tick_sweep(self, cell_env):
        # [FUNCTION DOC] After init and dispose, a tick does nothing and the cell stays disposed.
        """The sweep may still hold a reference to it for one iteration."""
        cell = _make_cell()
        await cell.init()
        await cell.dispose()

        await cell.tick()

        assert isinstance(cell._state, StateDisposed)


class TestMemoryOperations:
    # [CLASS DOC] Tests the cell's offload and onload methods, which forward to the engine client.
    async def test_offload_releases_exactly_the_requested_tags_and_returns_the_engine_result(self, result_api_client):
        # [FUNCTION DOC] Offloading with the kv cache tag calls the engine's release once with that
        # tag list, and the cell returns the engine's configured result.
        """Dropping the tags would release the weights too, which the trainer expects to stay resident."""
        result_api_client.results["release_memory_occupation"] = dict(released="kv_cache")
        cell = _make_cell()
        await cell.init()

        result = await cell.offload(tags=["kv_cache"])

        assert result_api_client.calls == [("release_memory_occupation", dict(tags=["kv_cache"]))]
        assert result == dict(released="kv_cache")

    async def test_onload_resumes_exactly_the_requested_tags_and_returns_the_engine_result(self, result_api_client):
        # [FUNCTION DOC] Onloading with the weights tag calls the engine's resume once with that tag
        # list, and the cell returns the engine's configured result.
        """Multi-stage resume is driven tag by tag, so the caller's choice must reach the engine unchanged."""
        result_api_client.results["resume_memory_occupation"] = dict(resumed="weights")
        cell = _make_cell()
        await cell.init()

        result = await cell.onload(tags=["weights"])

        assert result_api_client.calls == [("resume_memory_occupation", dict(tags=["weights"]))]
        assert result == dict(resumed="weights")

    async def test_an_engine_that_rejects_an_offload_surfaces_the_error_to_the_caller(self, result_api_client):
        # [FUNCTION DOC] A release error from the engine ("engine is busy") propagates out of
        # offload as a runtime error.
        """Swallowing it would let the trainer start on gpus the engine still holds."""
        result_api_client.errors["release_memory_occupation"] = RuntimeError("engine is busy")
        cell = _make_cell()
        await cell.init()

        with pytest.raises(RuntimeError, match="engine is busy"):
            await cell.offload(tags=["kv_cache"])

    async def test_an_engine_that_rejects_an_onload_surfaces_the_error_to_the_caller(self, result_api_client):
        # [FUNCTION DOC] A resume error from the engine ("weights are unavailable") propagates out of
        # onload as a runtime error.
        """Swallowing it would let rollout proceed while the requested engine memory remains unavailable."""
        result_api_client.errors["resume_memory_occupation"] = RuntimeError("weights are unavailable")
        cell = _make_cell()
        await cell.init()

        with pytest.raises(RuntimeError, match="weights are unavailable"):
            await cell.onload(tags=["weights"])


class TestCheckWeights:
    # [CLASS DOC] Tests the cell's check-weights method, which forwards to the engine client.
    async def test_check_weights_forwards_every_argument_and_returns_the_engine_result(self, result_api_client):
        # [FUNCTION DOC] A compare check with quantization error allowed, selector attn and skip
        # list lm_head reaches the engine once with exactly those four values; the cell returns the
        # engine's configured result.
        """The controller drives comparisons through this method, so nothing may be defaulted away."""
        result_api_client.results["check_weights"] = dict(equal=False)
        cell = _make_cell()
        await cell.init()

        result = await cell.check_weights(
            action="compare", allow_quant_error=True, selector="attn", skip_list=["lm_head"]
        )

        assert result_api_client.calls == [
            (
                "check_weights",
                dict(action="compare", allow_quant_error=True, selector="attn", skip_list=["lm_head"]),
            )
        ]
        assert result == dict(equal=False)

    async def test_an_engine_that_rejects_a_weight_check_surfaces_the_error_to_the_caller(self, result_api_client):
        # [FUNCTION DOC] A check-weights error from the engine ("weights checker is unavailable")
        # propagates as a runtime error.
        """A silently dropped failure would report a passing check the engine never performed."""
        result_api_client.errors["check_weights"] = RuntimeError("weights checker is unavailable")
        cell = _make_cell()
        await cell.init()

        with pytest.raises(RuntimeError, match="weights checker is unavailable"):
            await cell.check_weights(action="compare", allow_quant_error=False, selector="all", skip_list=None)
