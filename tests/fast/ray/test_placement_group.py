# [MODULE DOC]
# Unit tests for the driver side orchestration functions of the placement group module: creating
# the rollout components (inference controller and rollout executor handles), taking over
# surviving inference controllers and trainers, creating placement groups per role, publishing
# weight updates with their locking window and checksum logging, and creating training models
# (restored rollout ids, start rollout overrides, per-policy trainer handles). All remote workers
# are replaced by mocks whose call order is recorded; no ray cluster is used.
# KNOWN ISSUE: the module wide asyncio marker also applies to the synchronous tests of the
# placement group creation class, which are not coroutines.
from __future__ import annotations

import logging
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.fixtures.megatron_config_fixtures import write_megatron_config, write_megatron_config_trainers

from miles.ray import placement_group as placement_group_module
from miles.ray.placement_group import (
    create_rollout_components,
    create_training_model,
    create_training_models,
    take_over_trainers,
)
from miles.ray.rollout.eval_fleet import EvalFleetInfo
from miles.utils.init_once import InitState
from miles.utils.workers.types import DeployComponent, DeploymentIdentity
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.asyncio


def _make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds rollout related test arguments (8 rollout gpus, 2 epochs, no
    # explicit rollout count, ray cluster backend, no eval gpus, no session server and so on) over
    # the parser defaults, applying the given overrides on top of the defaults.
    defaults = dict(
        rollout_num_gpus=8,
        pin_rollout_manager_to_head=False,
        num_rollout=None,
        num_epoch=2,
        sglang_router_ip=None,
        sglang_router_port=None,
        cluster_backend="ray",
        eval_num_gpus=0,
        debug_train_only=False,
        use_session_server=False,
    )
    defaults.update(overrides)
    return Namespace(**{**parser_defaults(), **defaults})


@pytest.fixture
def fake_components():
    # [FUNCTION DOC] HELPER: Fixture that replaces the collaborators of the rollout component
    # creation with mocks and yields a namespace of controller handle, executor handle, capability
    # and an ordered event list. Controller: mock handle with async mocks (not initialized by
    # default, no eval fleet) whose init records an event. Router address resolution writes router
    # ip 10.0.0.1 and port 4321 into args. Session server waiting does nothing for a missing
    # provider, otherwise writes one address and instance id into args and records an event. The
    # executor handle: mock handle whose init state check records an event and reports not inited,
    # whose init records an event, and which reports 5 rollouts per epoch. The backend capability
    # lookup returns a fake capability. Patching is undone when the fixture closes.
    events: list[str] = []

    controller_handle = MagicMock(name="inference_controller")
    controller_handle.check_weights = AsyncMock()
    controller_handle.offload = AsyncMock()
    controller_handle.is_initialized = AsyncMock(return_value=False)
    controller_handle.init = AsyncMock(side_effect=lambda: events.append("controller_init"))
    controller_handle.get_eval_fleet_info = AsyncMock(return_value=None)

    async def resolve_router_addrs(args, *, router_providers) -> dict:
        # [FUNCTION DOC] HELPER: Fake router address resolution; writes router ip 10.0.0.1 and port
        # 4321 into args and returns an empty mapping.
        args.sglang_router_ip = "10.0.0.1"
        args.sglang_router_port = 4321
        return {}

    async def fake_wait_session_server_ready(args, *, provider):
        # [FUNCTION DOC] HELPER: Fake session server readiness wait. Returns at once if the
        # provider is none; otherwise writes one session server address (10.0.0.2:5000) and one
        # instance id (session-0) into args and appends a session-servers-ready event.
        # the real one returns before touching anything when the run asked for no session server
        if provider is None:
            return
        args.session_server_addrs = ["10.0.0.2:5000"]
        args.session_server_instance_ids = ["session-0"]
        events.append("session_servers_ready")

    async def fake_executor_get_init_state() -> str:
        # [FUNCTION DOC] HELPER: Fake executor init state query; appends an event and reports the
        # not-initialized state value.
        events.append("executor_init_state_checked")
        return InitState.NOT_INITED.value

    executor_handle = MagicMock(name="rollout_executor")
    executor_handle.wait_ready = AsyncMock(return_value=None)
    executor_handle.get_init_state = AsyncMock(side_effect=fake_executor_get_init_state)
    executor_handle.init = AsyncMock(side_effect=lambda: events.append("executor_init"))
    executor_handle.get_num_rollout_per_epoch = AsyncMock(return_value=5)
    executor_handle.set_eval_fleet_info = AsyncMock(return_value=None)

    capability = FakeBackendCapability(static_provider=object())

    with patch(
        "miles.ray.placement_group.create_inference_controller_handle", lambda *, capability: controller_handle
    ), patch("miles.ray.placement_group.resolve_router_addrs", resolve_router_addrs), patch(
        "miles.ray.placement_group.wait_session_server_ready", fake_wait_session_server_ready
    ), patch(
        "miles.ray.placement_group.create_rollout_executor_handle", lambda *, capability: executor_handle
    ), patch(
        "miles.ray.placement_group.get_backend_capability", lambda args: capability
    ):
        yield Namespace(
            controller_handle=controller_handle,
            executor_handle=executor_handle,
            capability=capability,
            events=events,
        )


class TestFrozenInferenceChecksums:
    # [CLASS DOC] Tests checksum logging when the inference fleet has no updatable engines.
    async def test_no_updatable_engines_skips_the_checksum_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With an initialized event logger (mock) and a controller whose weight
        # check returns an empty list, the checksum logging helper still queries the controller
        # once but logs no event.
        """Frozen inference fleets have no weight update checksums to flatten or record."""
        monkeypatch.setattr(placement_group_module, "is_event_logger_initialized", lambda: True)
        event_logger = MagicMock()
        monkeypatch.setattr(placement_group_module, "get_event_logger", lambda: event_logger)
        controller = SimpleNamespace(check_weights=AsyncMock(return_value=[]))

        await placement_group_module._maybe_log_inference_engine_weight_checksums(
            _make_args(), inference_controller=controller, rollout_id=None, trainer_model_id=None
        )

        controller.check_weights.assert_awaited_once()
        event_logger.log.assert_not_called()


class TestCreateRolloutComponents:
    # [CLASS DOC] Tests the creation of the rollout components (inference controller handle,
    # rollout executor handle and rollouts per epoch) against the mocked collaborators: event
    # ordering, address resolution, rollout count derivation and train-only behaviour.
    async def test_train_only_with_eval_resolves_session_addresses(self, fake_components) -> None:
        # [FUNCTION DOC] With train-only debug, one eval gpu and session servers enabled, the
        # creation still waits for session servers: args receive the fake address and instance id.
        """Snapshot evaluation still needs session routing when training uses injected rollouts."""
        args = _make_args(num_rollout=1, debug_train_only=True, eval_num_gpus=1, use_session_server=True)

        await create_rollout_components(args)

        assert args.session_server_addrs == ["10.0.0.2:5000"]
        assert args.session_server_instance_ids == ["session-0"]

    async def test_the_executor_is_inited_after_the_session_servers_are_known(self, fake_components):
        # [FUNCTION DOC] With session servers enabled, the recorded event order is: session servers
        # ready, executor init state checked, controller init, executor init. The session addresses
        # and instance ids are in args afterwards.
        """The executor reads the session contract off args, so it must be written before init() runs."""
        args = _make_args(num_rollout=1, use_session_server=True)

        await create_rollout_components(args)

        assert fake_components.events == [
            "session_servers_ready",
            "executor_init_state_checked",
            "controller_init",
            "executor_init",
        ]
        assert args.session_server_addrs == ["10.0.0.2:5000"]
        assert args.session_server_instance_ids == ["session-0"]

    async def test_the_executor_is_waited_out_before_anything_is_initialized(self, fake_components):
        # [FUNCTION DOC] Without session servers, the event order is: executor init state checked,
        # controller init, executor init.
        """A hot restart finds the previous script's executor up, and initializing anything against it is the bug."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args)

        assert fake_components.events == ["executor_init_state_checked", "controller_init", "executor_init"]

    async def test_returns_two_worker_handles(self, fake_components):
        # [FUNCTION DOC] The first two elements of the returned triple are the controller handle and
        # the executor handle.
        """Both halves of rollout are independent workers, so the driver only ever holds handles."""
        args = _make_args(num_rollout=1)

        controller, executor, _ = await create_rollout_components(args)

        assert controller is fake_components.controller_handle
        assert executor is fake_components.executor_handle

    async def test_the_router_addresses_are_resolved_before_the_workers_are_initialized(self, fake_components):
        # [FUNCTION DOC] After creation, args hold the router ip and port written by the fake
        # resolver (10.0.0.1 and 4321).
        # KNOWN ISSUE: the test name claims an ordering (resolved before initialization) but only the
        # final values in args are asserted, not the order relative to the initializations.
        """The driver's args copy must carry the contract before anything downstream reads it."""
        args = _make_args(num_rollout=1)

        await create_rollout_components(args)

        assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.1", 4321)

    async def test_num_rollout_derived_from_executor_epoch_length(self, fake_components):
        # [FUNCTION DOC] With no explicit rollout count and 2 epochs, the executor is asked exactly
        # once for the rollouts per epoch (5): the third returned value is 5 and args receive 10.
        """num_rollout comes from the dataset, which the executor owns."""
        args = _make_args(num_rollout=None, num_epoch=2)

        _, _, num_rollout_per_epoch = await create_rollout_components(args)

        fake_components.executor_handle.get_num_rollout_per_epoch.assert_awaited_once_with()
        assert num_rollout_per_epoch == 5
        assert args.num_rollout == 10

    async def test_num_rollout_left_alone_when_explicitly_set(self, fake_components):
        # [FUNCTION DOC] With an explicit rollout count of 3, the executor is never asked for the
        # epoch length, the third returned value is none, and args keep 3.
        """An explicit --num-rollout skips asking the executor for the epoch length."""
        args = _make_args(num_rollout=3)

        _, _, num_rollout_per_epoch = await create_rollout_components(args)

        fake_components.executor_handle.get_num_rollout_per_epoch.assert_not_awaited()
        assert num_rollout_per_epoch is None
        assert args.num_rollout == 3

    async def test_a_train_only_run_resolves_no_inference_addresses(self, fake_components):
        # [FUNCTION DOC] With train-only debug, no static pool provider is requested from the
        # capability and args keep router ip none.
        """--debug-train-only deploys no routers or session servers, so nothing can be waited on."""
        args = _make_args(num_rollout=1, debug_train_only=True)

        await create_rollout_components(args)

        assert fake_components.capability.requested_static_pool_ids == []
        assert args.sglang_router_ip is None


class TestTakeOverInference:
    # [CLASS DOC] Tests how the rollout component creation treats an already running inference
    # controller (take over) versus an uninitialized one, by recording the mock controller calls.
    @staticmethod
    async def _take_over(fake_components) -> None:
        # [FUNCTION DOC] HELPER: Runs the rollout component creation with an explicit rollout
        # count of 1 (the fixture is only used for its patching).
        await create_rollout_components(_make_args(num_rollout=1))

    async def test_a_cold_run_builds_the_controller_it_found_uninitialized(self, fake_components):
        # [FUNCTION DOC] With the controller reporting not initialized, the ordered controller
        # calls are: initialization query, init, eval fleet info fetch.
        """A cold start reaches plain init() and none of the reset, so the whole call sequence is pinned."""
        await self._take_over(fake_components)

        controller = fake_components.controller_handle
        assert [name for name, _args, _kwargs in controller.mock_calls] == [
            "is_initialized",
            "init",
            "get_eval_fleet_info",
        ]

    async def test_a_surviving_controller_is_reset_instead_of_built_again(self, fake_components):
        # [FUNCTION DOC] With the controller reporting initialized, the ordered calls are:
        # initialization query, wait idle, wait for the expected cell count, abort all, eval fleet
        # info fetch. Init is never awaited and abort all is awaited exactly once with no arguments.
        """Initializing it a second time would rebuild the fleet this run just adopted."""
        fake_components.controller_handle.is_initialized = AsyncMock(return_value=True)
        for name in ("wait_idle", "wait_expected_num_cells", "abort_all"):
            setattr(fake_components.controller_handle, name, AsyncMock(return_value=False))

        await self._take_over(fake_components)

        assert [name for name, _args, _kwargs in fake_components.controller_handle.mock_calls] == [
            "is_initialized",
            "wait_idle",
            "wait_expected_num_cells",
            "abort_all",
            "get_eval_fleet_info",
        ]

        fake_components.controller_handle.init.assert_not_awaited()
        fake_components.controller_handle.abort_all.assert_awaited_once_with()

    async def test_the_eval_fleet_reaches_the_executor_through_an_rpc_call(self, fake_components):
        # [FUNCTION DOC] When the controller reports an eval fleet (router 10.0.0.2:31000, 2 gpus,
        # 1 per engine), the executor's set eval fleet info is awaited once with that info.
        """The controller is a worker: its fleet is only knowable by calling it, never by reading it."""
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)
        fake_components.controller_handle.get_eval_fleet_info = AsyncMock(return_value=info)

        await self._take_over(fake_components)

        fake_components.executor_handle.set_eval_fleet_info.assert_awaited_once_with(info)

    async def test_a_run_without_an_eval_fleet_wires_nothing_up(self, fake_components):
        # [FUNCTION DOC] When the controller reports no eval fleet, the fleet info is fetched once
        # and the executor's set eval fleet info is never awaited.
        """The controller answers that it deploys no fleet, and the executor is left alone."""
        await self._take_over(fake_components)

        fake_components.controller_handle.get_eval_fleet_info.assert_awaited_once_with()
        fake_components.executor_handle.set_eval_fleet_info.assert_not_awaited()

    async def test_the_executor_is_handed_the_fleet_the_controller_just_built(self, fake_components):
        # [FUNCTION DOC] The controller only reports its eval fleet after init runs (init replaces
        # the fleet query mock). The executor must still receive that fleet exactly once, i.e. the
        # fleet is fetched after the controller was initialized.
        """Checkpoint eval pins snapshots to these engines, so publishing a pre-init fleet evaluates nothing."""
        info = EvalFleetInfo(router=HostAndPort(host="10.0.0.2", port=31000), num_gpus=2, num_gpus_per_engine=1)

        async def _publish_fleet_on_init():
            # [FUNCTION DOC] HELPER: Fake controller init; replaces the controller's eval fleet
            # query with one returning the prepared fleet info.
            fake_components.controller_handle.get_eval_fleet_info = AsyncMock(return_value=info)

        fake_components.controller_handle.init = AsyncMock(side_effect=_publish_fleet_on_init)

        await self._take_over(fake_components)

        fake_components.executor_handle.set_eval_fleet_info.assert_awaited_once_with(info)


class TestCreatePlacementGroups:
    # [CLASS DOC] Tests the per-role placement group views derived from one underlying placement
    # group, whose creation is replaced by a fake returning sentinel data.
    @staticmethod
    def _args(**overrides) -> Namespace:
        # [FUNCTION DOC] HELPER: Builds test arguments for a critic run: actor 1 node of 2 gpus,
        # critic 1 node of 1 gpu, 3 rollout gpus, no eval gpus, no colocation, no debug flags, the
        # all-in-one deploy component; parser defaults underneath, overrides on top.
        defaults = dict(
            debug_train_only=False,
            debug_rollout_only=False,
            rollout_external=False,
            colocate=False,
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=2,
            critic_num_nodes=1,
            critic_num_gpus_per_node=1,
            rollout_num_gpus=3,
            eval_num_gpus=0,
            megatron_config=None,
            critic_load=None,
            critic_save=None,
            critic_lr=None,
            critic_lr_warmup_iters=None,
            deploy_component="all",
        )
        defaults.update(overrides)
        return Namespace(**{**parser_defaults(), **defaults})

    @staticmethod
    def _patched(monkeypatch, requested: list[int]):
        # [FUNCTION DOC] HELPER: Replaces the underlying placement group creation with a fake that
        # records the requested gpu count in the given list and returns a placement group info with
        # a sentinel group, a scrambled bundle index list and gpu ids 100 and up.
        from miles.ray import placement_group as placement_group_module
        from miles.ray.placement_group import PlacementGroupInfo

        def _fake_create(num_gpus):
            # [FUNCTION DOC] HELPER: Records the gpu count and returns the sentinel placement group
            # info: group sentinel, bundle indices (index times 3 plus 1) modulo the count, and gpu
            # ids 100 plus the position.
            requested.append(num_gpus)
            return PlacementGroupInfo(
                pg="pg-sentinel",
                pg_reordered_bundle_indices=[(index * 3 + 1) % num_gpus for index in range(num_gpus)],
                pg_reordered_gpu_ids=[100 + index for index in range(num_gpus)],
            )

        monkeypatch.setattr(placement_group_module, "_create_placement_group", _fake_create)

    def test_each_role_views_the_shared_pg_from_its_own_offset(self, monkeypatch):
        # [FUNCTION DOC] With actor 2 gpus, critic 1 gpu (sharing the actor group) and 3 rollout
        # gpus, a single placement group of 5 gpus is created. Every role has the sentinel group;
        # actor sees all 5 gpu ids; critic equals actor; rollout sees the last 3 gpu ids (offset by
        # the actor's 2) and the matching tail of the bundle indices.
        """Roles share one placement group; the critic reuses the actor slice and rollout starts after it."""
        from miles.ray.placement_group import create_placement_groups

        requested: list[int] = []
        self._patched(monkeypatch, requested)

        pgs = create_placement_groups(self._args())

        assert requested == [5]
        assert {name: info.pg for name, info in pgs.items()} == {role: "pg-sentinel" for role in pgs}
        assert pgs["actor"].pg_reordered_gpu_ids == [100, 101, 102, 103, 104]
        assert pgs["critic"] == pgs["actor"]
        assert pgs["rollout"].pg_reordered_gpu_ids == [102, 103, 104]
        assert pgs["rollout"].pg_reordered_bundle_indices == pgs["actor"].pg_reordered_bundle_indices[2:]

    def test_a_disabled_critic_gets_no_entry_at_all(self, monkeypatch):
        # [FUNCTION DOC] Without a critic the role map has only actor and rollout, a 5 gpu group is
        # still requested, and rollout still sees the last 3 gpu ids.
        """Without a critic the role map simply omits it, so consumers never see a None placement group."""
        from miles.ray.placement_group import create_placement_groups

        requested: list[int] = []
        self._patched(monkeypatch, requested)

        pgs = create_placement_groups(self._args(use_critic=False))

        assert sorted(pgs) == ["actor", "rollout"]
        assert requested == [5]
        assert pgs["rollout"].pg_reordered_gpu_ids == [102, 103, 104]

    def test_a_trainer_deployment_hands_the_rollout_entry_no_bundle(self, monkeypatch):
        # [FUNCTION DOC] For the trainer deploy component, only a 2 gpu group is requested, actor
        # sees gpu ids 100 and 101, and the rollout entry has empty gpu id and bundle index lists.
        """Its release installs no engine, so a rollout entry over the trainer's bundles would double-book them."""
        from miles.ray.placement_group import create_placement_groups

        requested: list[int] = []
        self._patched(monkeypatch, requested)

        pgs = create_placement_groups(self._args(deploy_component="trainer"))

        assert requested == [2]
        assert pgs["actor"].pg_reordered_gpu_ids == [100, 101]
        assert pgs["rollout"].pg_reordered_gpu_ids == []
        assert pgs["rollout"].pg_reordered_bundle_indices == []


class TestUpdateWeights:
    # [CLASS DOC] Tests the driver weight update function: bracketing by the controller's update
    # window, publication of the weight version, abort on failure, startup validation of fault
    # actions and the rollout id stamped on checksum events. Everything is mocked.
    def _fakes(self, *, weight_version: int | None):
        # [FUNCTION DOC] HELPER: Returns a mock actor model whose async update weights returns the
        # given version, and a mock rollout executor with an async set weight version.
        actor_model = MagicMock()
        actor_model.update_weights = AsyncMock(return_value=weight_version)
        rollout_executor = MagicMock()
        rollout_executor.set_weight_version = AsyncMock()
        return actor_model, rollout_executor

    @staticmethod
    def _args():
        # [FUNCTION DOC] HELPER: Parser default arguments with train-only debug on and rollout-only off.
        return Namespace(**{**parser_defaults(), "debug_train_only": True, "debug_rollout_only": False})

    async def test_the_executor_is_told_which_version_the_engines_now_serve(self):
        # [FUNCTION DOC] With a trainer reporting version 7, the update window start receives a model
        # id of none, and the executor's set weight version is awaited once with 7 and no trainer
        # model id.
        """Without this the executor stamps every sample it collects with weight_version=None."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)

        inference_controller = MagicMock(start_update_weights=AsyncMock(), end_update_weights=AsyncMock())

        await update_weights(self._args(), actor_model, rollout_executor, inference_controller, rollout_id=3)

        info = inference_controller.start_update_weights.await_args.kwargs["model_id"]
        assert info is None
        rollout_executor.set_weight_version.assert_awaited_once_with(7, trainer_model_id=None)

    async def test_the_startup_sync_validates_fault_actions_before_updating_weights(self, monkeypatch):
        # [FUNCTION DOC] The factory building the fault test action executor from args is replaced
        # by a mock raising an assertion error. Calling the weight update without a rollout id (the
        # startup sync) must re-raise that error, call the factory once, and never reach the
        # trainer's update weights.
        """A parked-loop action must be rejected even when no in-loop weight sync will run."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)
        inference_controller = MagicMock(start_update_weights=AsyncMock(), end_update_weights=AsyncMock())
        factory = MagicMock(side_effect=AssertionError("loop cannot be parked"))
        monkeypatch.setattr(placement_group_module.FTTestActionOrchestrationExecutor, "from_args", factory)

        with pytest.raises(AssertionError, match="loop cannot be parked"):
            await update_weights(self._args(), actor_model, rollout_executor, inference_controller)

        factory.assert_called_once()
        actor_model.update_weights.assert_not_awaited()

    async def test_the_published_version_names_the_policy_it_belongs_to(self):
        # [FUNCTION DOC] When called with trainer model id alpha, the executor's set weight version
        # is awaited once with version 7 and that trainer model id.
        """A version published under the wrong policy judges another policy's samples against these weights."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)

        inference_controller = MagicMock(start_update_weights=AsyncMock(), end_update_weights=AsyncMock())

        await update_weights(
            self._args(), actor_model, rollout_executor, inference_controller, rollout_id=3, trainer_model_id="alpha"
        )

        rollout_executor.set_weight_version.assert_awaited_once_with(7, trainer_model_id="alpha")

    @staticmethod
    def _checksum_args(*, start_rollout_id: int = 0):
        # [FUNCTION DOC] HELPER: Minimal arguments (both debug flags off) with the given start rollout id.
        return Namespace(
            debug_train_only=False,
            debug_rollout_only=False,
            start_rollout_id=start_rollout_id,
        )

    def _record_checksum_events(self, monkeypatch) -> list[dict]:
        # [FUNCTION DOC] HELPER: Makes the event logger appear initialized and replaces it with one
        # whose log call appends the payload (second argument) to the returned list; makes the
        # engine checksum flattening return an empty list; replaces the fault test action executor
        # with a mock whose factory returns an object with an async run-after-step.
        logged: list[dict] = []
        monkeypatch.setattr(placement_group_module, "is_event_logger_initialized", lambda: True)
        monkeypatch.setattr(
            placement_group_module,
            "get_event_logger",
            lambda: SimpleNamespace(log=lambda _event_class, payload: logged.append(payload)),
        )
        monkeypatch.setattr(placement_group_module, "flatten_inference_engine_checksums", lambda _result: [])
        monkeypatch.setattr(
            placement_group_module,
            "FTTestActionOrchestrationExecutor",
            MagicMock(from_args=MagicMock(return_value=MagicMock(run_after_step=AsyncMock()))),
        )
        return logged

    async def test_a_fresh_run_stamps_the_startup_sync_before_the_first_rollout(self, monkeypatch):
        # [FUNCTION DOC] Startup sync (no rollout id) with start rollout id 0 logs exactly one
        # checksum event whose rollout id is -1.
        """A fresh run's startup push precedes rollout 0, so it is stamped -1 instead of being left unattributed."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)
        inference_controller = MagicMock(
            start_update_weights=AsyncMock(), end_update_weights=AsyncMock(), check_weights=AsyncMock()
        )
        logged = self._record_checksum_events(monkeypatch)

        await update_weights(
            self._checksum_args(start_rollout_id=0), actor_model, rollout_executor, inference_controller
        )

        assert [payload["rollout_id"] for payload in logged] == [-1]

    async def test_a_startup_sync_after_a_restore_stamps_the_restored_rollout_id(self, monkeypatch):
        # [FUNCTION DOC] Startup sync with start rollout id 5 logs one event with rollout id 4 (the
        # last restored rollout, one before the start).
        """The restored rollout never runs again, so only this push can record the checksum of its weights."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)
        inference_controller = MagicMock(
            start_update_weights=AsyncMock(), end_update_weights=AsyncMock(), check_weights=AsyncMock()
        )
        logged = self._record_checksum_events(monkeypatch)

        await update_weights(
            self._checksum_args(start_rollout_id=5), actor_model, rollout_executor, inference_controller
        )

        assert [payload["rollout_id"] for payload in logged] == [4]

    async def test_an_in_loop_sync_still_stamps_its_own_rollout_id(self, monkeypatch):
        # [FUNCTION DOC] A sync called with rollout id 6 while the start rollout id is 5 logs one
        # event with rollout id 6.
        """The in-loop push must keep naming the rollout it publishes, not the id the run started at."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=7)
        inference_controller = MagicMock(
            start_update_weights=AsyncMock(), end_update_weights=AsyncMock(), check_weights=AsyncMock()
        )
        logged = self._record_checksum_events(monkeypatch)

        await update_weights(
            self._checksum_args(start_rollout_id=5), actor_model, rollout_executor, inference_controller, rollout_id=6
        )

        assert [payload["rollout_id"] for payload in logged] == [6]

    async def test_a_trainer_that_skipped_the_broadcast_publishes_nothing(self):
        # [FUNCTION DOC] When the trainer's update weights returns none, the executor's set weight
        # version is never awaited.
        """--debug-skip-weight-update leaves the engines on their old weights, so the version must not move."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=None)

        inference_controller = MagicMock(start_update_weights=AsyncMock(), end_update_weights=AsyncMock())

        await update_weights(self._args(), actor_model, rollout_executor, inference_controller)

        rollout_executor.set_weight_version.assert_not_awaited()

    async def test_a_trainer_that_fails_mid_sync_still_closes_the_controllers_lock_window(self):
        # [FUNCTION DOC] When the trainer's update weights raises a runtime error, the same error
        # propagates; the controller's abort update weights is awaited once with no arguments, its
        # normal end update weights is never awaited, and no weight version is published.
        """Leaving the window open blocks every later controller call, so a failed sync turns into a hang."""
        from miles.ray.placement_group import update_weights

        actor_model, rollout_executor = self._fakes(weight_version=None)
        actor_model.update_weights = AsyncMock(side_effect=RuntimeError("weight sync failed"))
        inference_controller = MagicMock(
            start_update_weights=AsyncMock(), abort_update_weights=AsyncMock(), end_update_weights=AsyncMock()
        )

        with pytest.raises(RuntimeError, match="weight sync failed"):
            await update_weights(self._args(), actor_model, rollout_executor, inference_controller)

        inference_controller.abort_update_weights.assert_awaited_once_with()
        inference_controller.end_update_weights.assert_not_awaited()
        rollout_executor.set_weight_version.assert_not_awaited()


def _make_trainer_handle(
    *, initialized: bool = False, deployment_identity: DeploymentIdentity | None = None
) -> MagicMock:
    # [FUNCTION DOC] HELPER: Builds a mock trainer controller handle with async mocks: initialized
    # query returning the given flag, wait idle returning none, init and load state returning the
    # restored rollout list [0], deployment identity query returning the given identity, and train
    # parallel config returning none.
    handle = MagicMock()
    handle.is_initialized = AsyncMock(return_value=initialized)
    handle.wait_idle = AsyncMock(return_value=None)
    handle.init = AsyncMock(return_value=[0])
    handle.load_state = AsyncMock(return_value=[0])
    handle.get_deployment_identity = AsyncMock(return_value=deployment_identity)
    handle.get_train_parallel_config = AsyncMock(return_value=None)
    return handle


class TestCreateTrainingModels:
    # [CLASS DOC] Tests creation of the training models: which trainer ids are addressed, how many
    # handles are built, the order of readiness, identity and init calls, the executor load
    # position, and rejection of a config declaring a critic.
    @staticmethod
    def _patched(monkeypatch, requested: list[str], *, initialized: bool = False) -> list[MagicMock]:
        # [FUNCTION DOC] HELPER: Replaces the trainer controller handle factory with one recording
        # each requested trainer id into the given list and returning a fresh mock trainer handle
        # (initialized flag as given), and the backend capability lookup with one returning a plain
        # object. Returns the list that collects the created handles.
        handles: list[MagicMock] = []

        def _create_handle(args, *, capability, trainer_id: str):
            # [FUNCTION DOC] HELPER: Fake handle factory; records the trainer id, creates and
            # remembers a mock trainer handle and returns it.
            requested.append(trainer_id)
            handle = _make_trainer_handle(initialized=initialized)
            handles.append(handle)
            return handle

        monkeypatch.setattr(placement_group_module, "create_trainer_controller_handle", _create_handle)
        monkeypatch.setattr(placement_group_module, "get_backend_capability", lambda args: object())
        return handles

    @staticmethod
    def _rollout_executor() -> MagicMock:
        # [FUNCTION DOC] HELPER: Returns a mock rollout executor with async set train parallel
        # config and load.
        rollout_executor = MagicMock()
        rollout_executor.set_train_parallel_config = AsyncMock()
        rollout_executor.load = AsyncMock()
        return rollout_executor

    @staticmethod
    def _args(tmp_path, **overrides) -> Namespace:
        # [FUNCTION DOC] HELPER: Builds arguments with a megatron config file for one policy named
        # alpha (written under the temporary directory), no critic, no start rollout id and no
        # external controller addresses, over parser defaults, plus overrides.
        defaults = dict(
            megatron_config=write_megatron_config(tmp_path, "alpha"),
            use_critic=False,
            start_rollout_id=None,
            trainer_controller_addrs=None,
        )
        defaults.update(overrides)
        return Namespace(**{**parser_defaults(), **defaults})

    async def test_a_configured_policy_is_addressed_by_its_own_trainer_id(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a one-policy megatron config named alpha, the only trainer handle
        # requested is for trainer id alpha-actor.
        """A single entry --megatron-config names the pool '<model_id>-actor'; 'actor' addresses nothing."""
        requested: list[str] = []
        self._patched(monkeypatch, requested)

        await create_training_models(self._args(tmp_path), self._rollout_executor())

        assert requested == ["alpha-actor"]

    async def test_the_handles_it_drives_are_built_exactly_once(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Exactly one handle is created for the single trainer.
        """The take-over reads a trainer and then drives it, and a second handle would repeat the readiness handshake."""
        requested: list[str] = []
        handles = self._patched(monkeypatch, requested)

        await create_training_models(self._args(tmp_path), self._rollout_executor())

        assert len(handles) == 1

    async def test_every_trainer_is_asked_what_it_is_before_anything_drives_it(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] In the recorded handle calls, the initialization query comes before init.
        """Initializing a trainer a previous script already built would throw away the state it is holding."""
        handles = self._patched(monkeypatch, [])

        await create_training_models(self._args(tmp_path), self._rollout_executor())

        [handle] = handles
        called = [name for name, _args, _kwargs in handle.mock_calls]
        assert called.index("is_initialized") < called.index("init")

    async def test_the_executor_is_loaded_at_the_position_the_trainers_start_from(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With an uninitialized trainer that restores rollout 0, the executor's load
        # is awaited once with -1 (one before the start rollout 0).
        """The dataset has to stand where the trainers do, whether the run was built or taken over."""
        self._patched(monkeypatch, [], initialized=False)
        rollout_executor = self._rollout_executor()

        await create_training_models(self._args(tmp_path), rollout_executor)

        rollout_executor.load.assert_awaited_once_with(-1)

    async def test_an_external_trainer_is_identified_and_driven_through_one_handle(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] With an external controller address for alpha-actor (address wait and the
        # in-run identity check are stubbed), exactly one handle exists and its recorded calls
        # place the identity query before the initialization query before init.
        """A second handle would identify one connection and drive another, so the check would guard nothing."""
        handles = self._patched(monkeypatch, [])
        monkeypatch.setattr(placement_group_module, "wait_static_addrs_ready", AsyncMock(return_value=None))
        monkeypatch.setattr(placement_group_module, "_assert_external_trainer_in_run", lambda identity, **kwargs: None)
        args = self._args(tmp_path, trainer_controller_addrs=["alpha-actor=10.0.0.5:1234"])

        await create_training_models(args, self._rollout_executor())

        [handle] = handles
        called = [name for name, _args, _kwargs in handle.mock_calls]
        assert called.index("get_deployment_identity") < called.index("is_initialized") < called.index("init")

    async def test_a_run_without_a_megatron_config_still_addresses_the_actor_and_critic_pools(self, monkeypatch):
        # [FUNCTION DOC] With no megatron config and a critic, handles are requested for trainer
        # ids actor and critic in that order.
        """Every existing single policy deployment names its two pools 'actor' and 'critic'."""
        requested: list[str] = []
        self._patched(monkeypatch, requested)
        args = Namespace(
            megatron_config=None,
            use_critic=True,
            start_rollout_id=None,
            trainer_model_id=None,
            kl_coef=0,
            use_opd=False,
            disable_param_buffers_cpu_backup=False,
            load=None,
            save=None,
            lr=1e-6,
            lr_warmup_iters=None,
            critic_load=None,
            critic_save=None,
            critic_lr=None,
            critic_lr_warmup_iters=None,
            trainer_controller_addrs=None,
        )

        await create_training_models(args, self._rollout_executor())

        assert requested == ["actor", "critic"]

    async def test_a_config_declaring_a_critic_is_refused_while_reading_that_config(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] A megatron config listing a policy alpha plus a critic entry for alpha
        # makes training model creation raise an assertion error saying the config declares a critic.
        """The critic pool would be deployed and never inited, so the run would hang waiting for it."""
        self._patched(monkeypatch, [])
        args = Namespace(
            megatron_config=write_megatron_config_trainers(
                tmp_path, [{"model_id": "alpha"}, {"model_id": "alpha", "role": "critic"}]
            ),
            use_critic=False,
            start_rollout_id=None,
            trainer_controller_addrs=None,
        )

        with pytest.raises(AssertionError, match="declares a critic for"):
            await create_training_models(args, self._rollout_executor())


class TestTakeOverTrainers:
    # [CLASS DOC] Tests the take-over of already running trainers: address readiness wait,
    # identity validation for external trainers, and the event log discard rule on resume.
    @staticmethod
    def _args(**overrides) -> Namespace:
        # [FUNCTION DOC] HELPER: Builds arguments with an external controller address for
        # alpha-actor, no megatron config, run uuid run-a and the all-in-one deploy component, over
        # parser defaults, plus overrides.
        defaults = dict(
            trainer_controller_addrs=["alpha-actor=10.0.0.5:1234"],
            megatron_config=None,
            run_uuid="run-a",
            deploy_component="all",
        )
        defaults.update(overrides)
        return Namespace(**{**parser_defaults(), **defaults})

    @staticmethod
    def _identity(**overrides) -> DeploymentIdentity:
        # [FUNCTION DOC] HELPER: Builds a deployment identity for run run-a, trainer deploy
        # component, instance and trainer id alpha-actor, with the given fields replaced.
        defaults = dict(
            run_uuid="run-a",
            deploy_component=DeployComponent.TRAINER.value,
            deploy_instance_id="alpha-actor",
            trainer_id="alpha-actor",
        )
        defaults.update(overrides)
        return DeploymentIdentity(**defaults)

    @staticmethod
    def _patched(monkeypatch, *, events: list[str]) -> None:
        # [FUNCTION DOC] HELPER: Replaces the static address readiness wait with a fake that
        # appends an addresses-ready event to the given list.
        async def wait_static_addrs_ready(addrs) -> None:
            # [FUNCTION DOC] HELPER: Appends an addresses-ready event; ignores the addresses.
            events.append("addrs_ready")

        monkeypatch.setattr(placement_group_module, "wait_static_addrs_ready", wait_static_addrs_ready)

    async def test_the_addresses_are_waited_for_before_anything_reads_a_trainer(self, monkeypatch):
        # [FUNCTION DOC] The recorded order of events is: addresses ready, identity query,
        # initialization query; the take-over returns false because the trainer is not initialized.
        """Reading a trainer at an address nothing answers at yet fails a take-over on a pod that is merely starting."""
        events: list[str] = []
        self._patched(monkeypatch, events=events)
        handle = _make_trainer_handle()
        handle.get_deployment_identity = AsyncMock(
            side_effect=lambda: events.append("identity") or self._identity(),
        )
        handle.is_initialized = AsyncMock(side_effect=lambda: events.append("is_initialized") or False)

        assert await take_over_trainers(self._args(), handles={"alpha-actor": handle}) is False

        assert events == ["addrs_ready", "identity", "is_initialized"]

    async def test_an_external_trainer_of_another_run_is_refused(self, monkeypatch):
        # [FUNCTION DOC] A trainer reporting run uuid run-b while this launch drives run-a makes the
        # take-over raise an assertion error saying the launch drives a different run.
        """Driving another run's trainer would mix its weight updates into this run's samples."""
        self._patched(monkeypatch, events=[])
        handle = _make_trainer_handle()
        handle.get_deployment_identity = AsyncMock(return_value=self._identity(run_uuid="run-b"))

        with pytest.raises(AssertionError, match="but this launch drives run"):
            await take_over_trainers(self._args(), handles={"alpha-actor": handle})

    async def test_an_external_trainer_declared_under_another_id_is_refused(self, monkeypatch):
        # [FUNCTION DOC] A trainer reporting trainer id beta-actor reached under the key alpha-actor
        # raises an assertion error naming both ids.
        """A controller address keyed under the wrong trainer must not drive that policy's training."""
        self._patched(monkeypatch, events=[])
        handle = _make_trainer_handle()
        handle.get_deployment_identity = AsyncMock(return_value=self._identity(trainer_id="beta-actor"))

        with pytest.raises(AssertionError, match="keys trainer 'beta-actor' under 'alpha-actor'"):
            await take_over_trainers(self._args(), handles={"alpha-actor": handle})

    async def test_a_run_that_deploys_its_own_trainers_verifies_no_identity(self, monkeypatch):
        # [FUNCTION DOC] Without external controller addresses, the take-over never queries the
        # trainer's deployment identity.
        """Without --trainer-controller-addrs the trainers are this release's own pools, so there is nothing to check."""
        self._patched(monkeypatch, events=[])
        handle = _make_trainer_handle()

        await take_over_trainers(self._args(trainer_controller_addrs=None), handles={"alpha-actor": handle})

        handle.get_deployment_identity.assert_not_awaited()

    @staticmethod
    def _recorded_discards(monkeypatch) -> list[Namespace]:
        # [FUNCTION DOC] HELPER: Replaces the event logger checkpoint discard function with the
        # append method of a new list, and returns that list (it collects the args of each discard).
        discarded: list[Namespace] = []
        monkeypatch.setattr(placement_group_module.event_logger_checkpoint, "discard", discarded.append)
        return discarded

    async def test_discards_the_log_without_a_checkpoint(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] A take-over of an initialized trainer whose requested load directory does
        # not exist returns true and discards the event log checkpoint exactly once with the args.
        """Such a run trains its steps again, and one log holding each of them twice is not a run anyone can compare."""
        self._patched(monkeypatch, events=[])
        discarded = self._recorded_discards(monkeypatch)
        args = self._args(requested_load=str(tmp_path / "ckpt"))

        handle = _make_trainer_handle(initialized=True, deployment_identity=self._identity())

        assert await take_over_trainers(args, handles={"alpha-actor": handle}) is True

        assert discarded == [args]

    async def test_keeps_the_log_with_a_checkpoint(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] When the requested load directory holds the latest checkpointed iteration
        # marker file, the take-over of an initialized trainer returns true and nothing is discarded.
        """That run resumes from its checkpoint, and the snapshot beside it is what replaces the log."""
        self._patched(monkeypatch, events=[])
        discarded = self._recorded_discards(monkeypatch)
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        (ckpt / "latest_checkpointed_iteration.txt").write_text("3")

        assert (
            await take_over_trainers(
                self._args(requested_load=str(ckpt)),
                handles={"alpha-actor": _make_trainer_handle(initialized=True, deployment_identity=self._identity())},
            )
            is True
        )

        assert discarded == []

    async def test_keeps_the_log_of_a_first_launch(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] For a trainer that is not initialized (a first launch), the take-over
        # returns false and nothing is discarded even though the load directory is absent.
        """A launch that installed the trainers itself is the run's first, and its log is the one it just opened."""
        self._patched(monkeypatch, events=[])
        discarded = self._recorded_discards(monkeypatch)

        assert (
            await take_over_trainers(
                self._args(requested_load=str(tmp_path / "ckpt")),
                handles={"alpha-actor": _make_trainer_handle(deployment_identity=self._identity())},
            )
            is False
        )

        assert discarded == []


class TestCreateTrainingModel:
    # [CLASS DOC] Tests creation of a single training model: consistency of restored rollout ids
    # across cells, start rollout selection and override, logging of overrides, and reload versus
    # init for resumed trainers.
    @staticmethod
    def _handle(*, restored: list[int]) -> MagicMock:
        # [FUNCTION DOC] HELPER: Returns a mock trainer handle whose async init and load state both
        # return the given restored rollout id list (one per cell).
        handle = MagicMock()
        handle.init = AsyncMock(return_value=restored)
        handle.load_state = AsyncMock(return_value=restored)
        return handle

    async def test_a_trainer_whose_cells_restored_different_rollouts_is_refused(self):
        # [FUNCTION DOC] Restored ids [5, 4] from one trainer raise an assertion error whose message
        # names the trainer and the restored list.
        """Cells of one trainer hold one model, so disagreeing positions mean a corrupted checkpoint set."""
        with pytest.raises(AssertionError, match=r"trainer 'alpha-actor' restored \[5, 4\]"):
            await create_training_model(
                Namespace(start_rollout_id=None),
                handle=self._handle(restored=[5, 4]),
                trainer_id="alpha-actor",
                resumed=False,
            )

    async def test_a_trainer_starts_where_its_cells_restored(self):
        # [FUNCTION DOC] Restored ids [3, 3] with no start override give a start rollout id of 3.
        """The restored position is what makes a resume continue instead of retraining old rounds."""
        info = await create_training_model(
            Namespace(start_rollout_id=None),
            handle=self._handle(restored=[3, 3]),
            trainer_id="alpha-actor",
            resumed=False,
        )

        assert info.start_rollout_id == 3

    async def test_an_explicit_start_rollout_id_wins_over_the_restored_one_on_a_cold_run(self):
        # [FUNCTION DOC] An explicit start rollout id 9 with restored [3] on a non-resumed run gives
        # a start rollout id of 9.
        """--start-rollout-id is the manual override for replaying or skipping rounds of a run being built."""
        info = await create_training_model(
            Namespace(start_rollout_id=9), handle=self._handle(restored=[3]), trainer_id="alpha-actor", resumed=False
        )

        assert info.start_rollout_id == 9

    async def test_a_trainer_told_to_start_elsewhere_than_it_restored_says_so(self, caplog):
        # [FUNCTION DOC] With start override 9 and restored [3], an info level log from the
        # placement group logger mentions the trainer id and the text of the override flag with 9.
        """A trainer that silently starts somewhere other than where it restored gives the operator nothing to read."""
        with caplog.at_level(logging.INFO, logger="miles.ray.placement_group"):
            await create_training_model(
                Namespace(start_rollout_id=9),
                handle=self._handle(restored=[3]),
                trainer_id="alpha-actor",
                resumed=False,
            )

        assert "alpha-actor" in caplog.text and "--start-rollout-id 9" in caplog.text

    async def test_a_trainer_told_to_start_where_it_restored_says_nothing(self, caplog):
        # [FUNCTION DOC] With start override 3 equal to restored [3], no log mentions the override flag.
        """Logging every trainer that was told where it already stands is noise on every ordinary launch."""
        with caplog.at_level(logging.INFO, logger="miles.ray.placement_group"):
            await create_training_model(
                Namespace(start_rollout_id=3),
                handle=self._handle(restored=[3]),
                trainer_id="alpha-actor",
                resumed=False,
            )

        assert "--start-rollout-id" not in caplog.text

    async def test_a_trainer_left_to_its_restored_position_says_nothing(self, caplog):
        # [FUNCTION DOC] With no override and restored [3], no log mentions the override flag.
        """The ordinary resume names no rollout at all, and it must not be reported as an override."""
        with caplog.at_level(logging.INFO, logger="miles.ray.placement_group"):
            await create_training_model(
                Namespace(start_rollout_id=None),
                handle=self._handle(restored=[3]),
                trainer_id="alpha-actor",
                resumed=False,
            )

        assert "--start-rollout-id" not in caplog.text

    async def test_the_restored_position_is_kept_beside_the_overridden_start(self):
        # [FUNCTION DOC] With override 9 and restored [3], the result still reports a restored
        # rollout id of 3 next to the start id.
        """Cross trainer checks compare where checkpoints actually were, which an override must not rewrite."""
        info = await create_training_model(
            Namespace(start_rollout_id=9), handle=self._handle(restored=[3]), trainer_id="alpha-actor", resumed=False
        )

        assert info.restored_rollout_id == 3

    async def test_an_explicit_start_rollout_id_wins_over_the_reload_of_a_taken_over_run_too(self):
        # [FUNCTION DOC] For a resumed (taken over) run with override 9 and restored [3], the start
        # id is 9 and the restored id is 3.
        """A take-over reads the same argument a cold start does, so one flag cannot mean two things."""
        info = await create_training_model(
            Namespace(start_rollout_id=9), handle=self._handle(restored=[3]), trainer_id="alpha-actor", resumed=True
        )

        assert info.start_rollout_id == 9
        assert info.restored_rollout_id == 3

    async def test_a_trainer_a_previous_script_built_is_reloaded_instead_of_inited(self):
        # [FUNCTION DOC] For a resumed run with restored [4], the start id is 4, load state is
        # awaited once with no arguments, and init is never awaited.
        """A survivor already holds the model, so building it again would throw away the run's own progress."""
        handle = self._handle(restored=[4])

        info = await create_training_model(
            Namespace(start_rollout_id=None), handle=handle, trainer_id="alpha-actor", resumed=True
        )

        assert info.start_rollout_id == 4
        handle.load_state.assert_awaited_once_with()
        handle.init.assert_not_awaited()
