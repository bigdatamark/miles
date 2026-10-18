# [MODULE DOC]
# Tests for the ordering of operations around a weight update. Two layers are covered. First,
# the inference controller's start-of-update-window method: it must pause health monitoring, make
# cells ready (initializing colocated cells inside the window) and only then snapshot the
# updatable engines. Second, the driver's weight update function: the trainer broadcast must run
# strictly between the controller's start and end of the update window, the snapshot returned by
# start must be passed unchanged to end, calls must be scoped to the published policy, and the
# post-update checksum logging must be skipped or emitted under the right conditions. The
# controller is a real object with stubbed internals; the driver side uses an order recording
# fake controller.
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.inference_controller import InferenceController
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActivenessTracker


class _OrderRecordingInferenceController:
    # [CLASS DOC] HELPER: Fake inference controller that answers any attribute as an async method.
    # Each call appends the method name to a shared order list, records (name, args, kwargs) in
    # the calls list, and returns a fresh mock that is also stored in the results mapping under
    # the method name (so a later call of the same name replaces it).
    def __init__(self, order: list[str]):
        # [FUNCTION DOC] HELPER: Stores the shared order list and starts with empty call and
        # result records.
        self._order = order
        self.calls: list[tuple[str, tuple, dict]] = []
        self.results: dict[str, MagicMock] = {}

    def __getattr__(self, name: str):
        # [FUNCTION DOC] HELPER: Called for any attribute not found normally; returns an async
        # function that records the call and returns a fresh mock result.
        recorder = self

        async def _method(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Appends the captured method name to the order list, records
            # the call, stores and returns a new mock as the result.
            recorder._order.append(name)
            recorder.calls.append((name, args, kwargs))
            result = MagicMock()
            recorder.results[name] = result
            return result

        return _method


def _assert_the_snapshot_is_handed_back_unchanged(controller: _OrderRecordingInferenceController) -> None:
    # [FUNCTION DOC] HELPER: Asserts that end update weights was called exactly once and that its
    # snapshot argument is the very same object (identity) as the snapshot attribute of the result
    # that start update weights returned.
    end_kwargs: list[dict] = [kwargs for name, _args, kwargs in controller.calls if name == "end_update_weights"]

    assert len(end_kwargs) == 1
    assert (
        end_kwargs[0]["snapshot_cell_id_to_hashes"]
        is controller.results["start_update_weights"].snapshot_cell_id_to_hashes
    )


class _ColocatedCellStub:
    # [CLASS DOC] HELPER: Stub of a colocated engine cell that counts initializations. Before
    # init it is uninitialized and not pending weights; after init it is ready.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with zero initializations and not ready.
        self.init_count = 0
        self.ready = False

    async def init(self) -> None:
        # [FUNCTION DOC] HELPER: Increments the init count and marks the cell ready.
        self.init_count += 1
        self.ready = True

    @property
    def is_uninitialized(self) -> bool:
        # [FUNCTION DOC] HELPER: Property; true while init has not run.
        return not self.ready

    @property
    def is_pending_weights_or_serving(self) -> bool:
        # [FUNCTION DOC] HELPER: Property; true once init has run.
        return self.ready


class _ServerStub:
    # [CLASS DOC] HELPER: Stub of an engine server holding a mapping of cell stubs and an active
    # health checker activeness tracker.
    def __init__(self, server_cells: dict[str, _ColocatedCellStub]) -> None:
        # [FUNCTION DOC] HELPER: Stores the cell mapping and creates an active activeness tracker.
        self.server_cells = server_cells
        self.health_checker_activeness = ActivenessTracker(active=True)


def _make_inference_controller(**arg_overrides: object) -> InferenceController:
    # [FUNCTION DOC] HELPER: Creates a real inference controller from test rollout args (with the
    # given overrides), with no engine provider and no router providers.
    return InferenceController(make_args(**arg_overrides), engine_provider=None, router_providers=[])


@pytest.mark.asyncio
async def test_controller_pauses_health_checks_before_snapshotting_the_engines():
    # [FUNCTION DOC] Replaces three internals of a real controller with order recorders (health
    # monitoring pause, ensure cells ready, get updatable server), gives it a context lock, a
    # non-colocated args object and an empty server mapping, and calls start update weights. The
    # recorded order must be: pause, cells ready, updatable server snapshot.
    """``start_update_weights`` pauses the health monitor, then readies the cells, then reads the engine set."""
    order: list[str] = []
    controller = _make_inference_controller()

    async def _record_pause(model_id: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Records the health monitoring pause step.
        order.append("health_monitoring_pause")

    async def _record_ensure_cells_ready(model_id: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Records the ensure-cells-ready step.
        order.append("ensure_cells_ready")

    def _record_snapshot(model_id: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Records the updatable server snapshot step and returns none.
        order.append("get_updatable_server")
        return None

    controller.context_lock = ContextLock("InferenceController")
    controller.args = Namespace(colocate=False)
    controller.servers = {}
    controller._health_monitoring_pause = _record_pause
    controller._ensure_cells_ready = _record_ensure_cells_ready
    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert order == ["health_monitoring_pause", "ensure_cells_ready", "get_updatable_server"]


@pytest.mark.asyncio
async def test_start_update_weights_initializes_colocated_cells_before_snapshotting_the_engines():
    # [FUNCTION DOC] A controller with colocation on and one server holding one uninitialized cell
    # stub has only its updatable server snapshot stubbed (recording the cell's init count at the
    # time of the snapshot). After start update weights, the cell was initialized exactly once, and
    # the init count seen at snapshot time was already 1.
    """A colocated cell is initialized inside the weight update window, before the engine snapshot is taken."""
    controller = _make_inference_controller(colocate=True)
    cell = _ColocatedCellStub()
    controller.servers = {"default": _ServerStub({"a": cell})}
    init_counts_at_snapshot: list[int] = []

    def _record_snapshot(model_id: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Records the cell's current init count and returns none.
        init_counts_at_snapshot.append(cell.init_count)
        return None

    controller._get_updatable_server = _record_snapshot

    await controller.start_update_weights()

    assert cell.init_count == 1
    assert init_counts_at_snapshot == [1]


def _orchestration_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds the minimal driver argument namespace: no debug modes, start
    # rollout id 0, no fault test actions, mini fault tolerance controller enabled with a 0.01
    # second poll interval; overrides replace entries.
    values = dict(
        debug_train_only=False,
        debug_rollout_only=False,
        start_rollout_id=0,
        ci_ft_test_actions=None,
        ci_ft_test_actions_path=None,
        mini_ft_controller_enable=True,
        mini_ft_controller_poll_interval=0.01,
    )
    values.update(overrides)
    return Namespace(**values)


def _actor_model(order: list[str]) -> MagicMock:
    # [FUNCTION DOC] HELPER: Returns a mock actor model whose async update weights appends a
    # trainer-update-weights entry to the shared order list and returns the weight version 11.
    async def _record_update_weights(*, info: object, rollout_id: int | None = None) -> int:
        # [FUNCTION DOC] HELPER: Records the trainer update step in the order list and returns 11.
        order.append("trainer_update_weights")
        return 11

    actor_model = MagicMock()
    actor_model.update_weights = AsyncMock(side_effect=_record_update_weights)
    return actor_model


@pytest.mark.asyncio
async def test_the_script_brackets_the_broadcast_with_start_and_end_update_weights():
    # [FUNCTION DOC] Runs the driver weight update with an order recording controller. The first
    # three recorded steps must be: start update weights, trainer update weights, end update weights.
    """The fault-tolerant trainer runs the actual update RPC strictly inside the update window."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _OrderRecordingInferenceController(order)

    await update_weights(
        _orchestration_args(), _actor_model(order), MagicMock(set_weight_version=AsyncMock()), inference_controller
    )

    assert order[:3] == ["start_update_weights", "trainer_update_weights", "end_update_weights"]


@pytest.mark.asyncio
async def test_the_script_hands_end_update_weights_the_snapshot_start_returned():
    # [FUNCTION DOC] Runs the driver weight update with an order recording controller and checks
    # that the end call received the identical snapshot object returned by the start call.
    """The snapshot start_update_weights returned is handed back to end_update_weights unchanged."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _OrderRecordingInferenceController(order)

    await update_weights(
        _orchestration_args(), _actor_model(order), MagicMock(set_weight_version=AsyncMock()), inference_controller
    )

    _assert_the_snapshot_is_handed_back_unchanged(inference_controller)


@pytest.mark.asyncio
async def test_the_window_is_scoped_to_the_policy_the_script_is_publishing():
    # [FUNCTION DOC] With the event logger reported as initialized (logger fetch and checksum
    # flattening stubbed, the latter returning an empty list), runs the driver weight update for
    # rollout 3 and trainer model id alpha. The start call's keyword arguments must equal only the
    # model id alpha, and the weight check call must have the checksum action and model id alpha.
    """Without the scope, one policy's trainer broadcasts its weights into another policy's engines."""
    from miles.ray.placement_group import update_weights

    order: list[str] = []
    inference_controller = _OrderRecordingInferenceController(order)

    with patch("miles.ray.placement_group.is_event_logger_initialized", return_value=True), patch(
        "miles.ray.placement_group.get_event_logger"
    ), patch("miles.ray.placement_group.flatten_inference_engine_checksums", return_value=[]):
        await update_weights(
            _orchestration_args(),
            _actor_model(order),
            MagicMock(set_weight_version=AsyncMock()),
            inference_controller,
            rollout_id=3,
            trainer_model_id="alpha",
        )

    calls = {name: kwargs for name, _args, kwargs in inference_controller.calls}
    assert calls["start_update_weights"] == dict(model_id="alpha")
    assert calls["check_weights"] == dict(action="checksum", model_id="alpha")


def _checksum_response(engine_checksums: list[dict[str, str]]) -> list:
    # [FUNCTION DOC] HELPER: Builds a fake per-engine weight check response: one successful entry
    # (message ok) per given checksum mapping, each holding a single rank record with the checksums
    # and parallelism info of a target role at rank 0.
    """Build a flat per-engine check_weights('checksum') response."""
    return [
        {
            "success": True,
            "message": "ok",
            "ranks": [{"checksums": cs, "parallelism_info": [{"role": "target", "rank": 0}]}],
        }
        for cs in engine_checksums
    ]


class TestTheScriptLogsTheChecksumsTheEnginesNowServe:
    # [CLASS DOC] Tests when the driver logs per-engine weight checksums after an update.
    @staticmethod
    async def _log(
        args: Namespace, *, response=None, initialized: bool = True, trainer_model_id: str | None = None
    ) -> tuple[MagicMock, MagicMock]:
        # [FUNCTION DOC] HELPER: Calls the driver's checksum logging helper for rollout 0 with a
        # mock controller (async weight check returning the given response when one is given,
        # otherwise a plain mock) and a mock event logger whose initialized state is the given
        # flag. Returns the controller mock and the event logger mock. Patches the placement group
        # module's logger accessors for the duration of the call.
        from miles.ray.placement_group import _maybe_log_inference_engine_weight_checksums

        inference_controller = MagicMock()
        inference_controller.check_weights = AsyncMock(return_value=response) if response is not None else MagicMock()
        event_logger = MagicMock()
        with patch("miles.ray.placement_group.is_event_logger_initialized", return_value=initialized), patch(
            "miles.ray.placement_group.get_event_logger", return_value=event_logger
        ):
            await _maybe_log_inference_engine_weight_checksums(
                args, inference_controller=inference_controller, rollout_id=0, trainer_model_id=trainer_model_id
            )
        return inference_controller, event_logger

    async def test_no_event_logger_does_not_call_check_weights(self):
        # [FUNCTION DOC] With an uninitialized event logger, the controller's weight check is not called.
        """Without an initialized event logger, no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(), initialized=False)

        inference_controller.check_weights.assert_not_called()

    async def test_debug_train_only_skips_collection(self):
        # [FUNCTION DOC] With train-only debug, the controller's weight check is not called.
        """Without real rollout engines (debug_train_only), no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(debug_train_only=True))

        inference_controller.check_weights.assert_not_called()

    async def test_debug_rollout_only_skips_collection(self):
        # [FUNCTION DOC] With rollout-only debug, the controller's weight check is not called.
        """Without real train engines pushing weights (debug_rollout_only), no check_weights request is issued."""
        inference_controller, _ = await self._log(_orchestration_args(debug_rollout_only=True))

        inference_controller.check_weights.assert_not_called()

    async def test_enabled_logs_one_event_per_rollout(self):
        # [FUNCTION DOC] With two engine responses (checksums w=e0 and w=e1) the controller check
        # is awaited once with the checksum action and no model id, and the event logger logs once,
        # with a payload of rollout id 0, no trainer model id and the flattened per-engine
        # checksum mappings whose keys are prefixed with rank0/.
        """With event logger on and real engines, one event holds every engine's checksums."""
        response = _checksum_response([{"w": "e0"}, {"w": "e1"}])

        inference_controller, event_logger = await self._log(_orchestration_args(), response=response)

        inference_controller.check_weights.assert_awaited_once_with(action="checksum", model_id=None)
        event_logger.log.assert_called_once()
        assert event_logger.log.call_args.args[1] == dict(
            rollout_id=0, trainer_model_id=None, engine_checksums=[{"rank0/w": "e0"}, {"rank0/w": "e1"}]
        )

    async def test_a_named_policy_stamps_its_own_id_on_the_event(self):
        # [FUNCTION DOC] With trainer model id solver, the controller check receives that model id
        # and the logged payload carries trainer model id solver.
        """A multi policy run's event names the policy, so the comparator can tell two policies apart."""
        response = _checksum_response([{"w": "e0"}])

        inference_controller, event_logger = await self._log(
            _orchestration_args(), response=response, trainer_model_id="solver"
        )

        inference_controller.check_weights.assert_awaited_once_with(action="checksum", model_id="solver")
        assert event_logger.log.call_args.args[1] == dict(
            rollout_id=0, trainer_model_id="solver", engine_checksums=[{"rank0/w": "e0"}]
        )
