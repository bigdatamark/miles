# [MODULE DOC]
# Unit tests for the rollout server module: sglang config resolution (eval model derivation, YAML
# validation, GPU offset and Megatron GPU computations), wiring done by the rollout server factory
# (routers, activeness trackers, locks, update-weights flags), engine list ordering by GPU offset,
# adding cells with rollback on failed init, duplicate cell ids, removal ordering, and when a cell
# is initialized (immediately or deferred for colocated cells that must wait for the weight update
# window). Server cell init and dispose are monkeypatched; a stub provider refuses address lookups,
# so no processes, network or GPUs are touched.
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.sglang_utils.sglang_config import (
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
    resolve_sglang_config,
)
from miles.ray.rollout.cell_state import CellAddrInfo, StateServing
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts


class TestRolloutServerPureFunctions:
    # [CLASS DOC] Tests pure configuration functions: resolving the sglang config from arguments
    # (including the derived eval model) and computing rollout GPU offsets and Megatron GPU counts.
    def test_resolve_sglang_config_yaml_gpu_mismatch_asserts(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Writes a YAML config with one model of 4 GPUs but sets 8 rollout GPUs in
        # the arguments; resolving the config raises an assertion error mentioning total GPUs.
        """A YAML fleet whose GPU total disagrees with the CLI allocation is rejected."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: actor\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 4\n"
            "        num_gpus_per_engine: 1\n"
        )
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=8)
        with pytest.raises(AssertionError, match="total GPUs"):
            resolve_sglang_config(args)

    def test_eval_fleet_inherits_rollout_engine_settings(self):
        # [FUNCTION DOC] With 2 eval GPUs and 2 GPUs per eval engine, the resolved config has one
        # model named eval that does not update weights, with one group of 2 GPUs and 2 per engine.
        # The group's overrides force the routed-experts and indexer-topk return flags off, and the
        # group's model path is the arguments' hf checkpoint.
        """The eval model carries only what makes it an eval fleet; the rest is inherited."""
        args = make_args(eval_num_gpus=2, eval_num_gpus_per_engine=2)
        config = resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        assert eval_model.update_weights is False
        [group] = eval_model.server_groups
        assert (group.num_gpus, group.num_gpus_per_engine) == (2, 2)
        # Eval samples never feed training, so the replay side-channels are forced off.
        assert group.overrides["enable_return_routed_experts"] is False
        assert group.overrides["enable_return_indexer_topk"] is False

        # The fleet boots on --hf-checkpoint; every eval overwrites those weights anyway.
        assert group.model_path == args.hf_checkpoint

    def test_tp_coupled_sizes_are_not_inherited_across_a_different_eval_tp(self):
        # [FUNCTION DOC] Rollout engines use 8 GPUs but eval engines 1: the eval group's overrides
        # set data parallel, pipeline parallel, expert parallel and attention context parallel sizes
        # all to 1.
        """Inheriting these across a different eval tp gives an engine that will not boot."""
        args = make_args(rollout_num_gpus_per_engine=8, eval_num_gpus=1, eval_num_gpus_per_engine=1)
        [group] = [m for m in resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert {k: group.overrides[k] for k in ("dp_size", "pp_size", "ep_size", "attn_cp_size")} == dict.fromkeys(
            ("dp_size", "pp_size", "ep_size", "attn_cp_size"), 1
        )

    def test_tp_coupled_sizes_are_inherited_when_the_tp_matches(self):
        # [FUNCTION DOC] Rollout and eval engines both use 2 GPUs: the eval group's overrides contain
        # no expert parallel size (left to the shared fill-in).
        args = make_args(rollout_num_gpus_per_engine=2, eval_num_gpus=2, eval_num_gpus_per_engine=2)
        [group] = [m for m in resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert "ep_size" not in group.overrides  # left to the shared --sglang-* fill-in

    def test_eval_sglang_overrides_win_over_the_tp_coupled_defaults(self):
        # [FUNCTION DOC] With differing tensor parallel sizes and an explicit eval expert parallel
        # size of 2, the eval group's expert parallel override is 2 (explicit beats the default of 1).
        args = make_args(
            rollout_num_gpus_per_engine=8,
            eval_num_gpus=2,
            eval_num_gpus_per_engine=2,
            eval_sglang_ep_size=2,
        )
        [group] = [m for m in resolve_sglang_config(args).models if m.name == "eval"][0].server_groups

        assert group.overrides["ep_size"] == 2

    def test_eval_sglang_overrides_reach_the_eval_group_only(self):
        # [FUNCTION DOC] An eval memory fraction of 0.95 appears in the eval model's first group
        # overrides and not in the default model's first group.
        args = make_args(eval_num_gpus=2, eval_sglang_mem_fraction_static=0.95)
        config = resolve_sglang_config(args)

        by_name = {m.name: m for m in config.models}
        assert by_name["eval"].server_groups[0].overrides["mem_fraction_static"] == 0.95
        assert "mem_fraction_static" not in by_name["default"].server_groups[0].overrides

    def test_debug_train_only_builds_only_the_eval_model(self):
        # [FUNCTION DOC] With debug train-only and 8 eval GPUs (1 per engine), the resolved models
        # are exactly the eval model, with 8 GPUs across its groups.
        args = make_args(debug_train_only=True, eval_num_gpus=8, eval_num_gpus_per_engine=1)

        config = resolve_sglang_config(args)

        assert [model.name for model in config.models] == ["eval"]
        assert sum(group.num_gpus for model in config.models for group in model.server_groups) == 8

    def test_debug_train_only_without_eval_fleet_builds_no_model(self):
        # [FUNCTION DOC] With debug train-only and zero eval GPUs the resolved model list is empty.
        args = make_args(debug_train_only=True, eval_num_gpus=0)

        assert resolve_sglang_config(args).models == []

    def test_yaml_eval_model_is_filled_from_cli_without_clobbering(self, tmp_path):
        # [FUNCTION DOC] A YAML config defines default (8 GPUs) and eval (2 GPUs, no per-engine
        # size) models; the arguments give eval GPUs per engine 2 and an eval memory fraction 0.95.
        # The resolved eval model does not update weights, its group takes 2 GPUs per engine and
        # the 0.95 memory fraction override from the command-line arguments.
        """Anything the YAML leaves unset falls through to the eval CLI args."""
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n"
        )
        args = make_args(
            sglang_config=str(cfg_path),
            rollout_num_gpus=8,
            eval_num_gpus=2,
            eval_num_gpus_per_engine=2,
            eval_sglang_mem_fraction_static=0.95,
        )
        config = resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        # Auto-inference would give True here and put the fleet in the broadcast group.
        assert eval_model.update_weights is False
        [group] = eval_model.server_groups
        assert group.num_gpus_per_engine == 2
        assert group.overrides["mem_fraction_static"] == 0.95

    def test_yaml_group_overrides_win_over_eval_cli(self, tmp_path):
        # [FUNCTION DOC] The YAML eval group sets a memory fraction override of 0.5 while the
        # arguments say 0.95; the resolved group keeps 0.5.
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "  - name: eval\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 2\n"
            "        num_gpus_per_engine: 1\n"
            "        overrides:\n"
            "          mem_fraction_static: 0.5\n"
        )
        args = make_args(
            sglang_config=str(cfg_path), rollout_num_gpus=8, eval_num_gpus=2, eval_sglang_mem_fraction_static=0.95
        )
        config = resolve_sglang_config(args)

        [eval_model] = [m for m in config.models if m.name == "eval"]
        assert eval_model.server_groups[0].overrides["mem_fraction_static"] == 0.5

    def test_compute_rollout_offset_colocate_returns_zero(self):
        # [FUNCTION DOC] With colocation on, the rollout GPU offset is 0 regardless of the actor size.
        args = make_args(
            colocate=True,
            debug_train_only=False,
            debug_rollout_only=False,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=False,
        )
        assert _compute_rollout_offset(args) == 0

    def test_compute_rollout_offset_critic_train_only(self):
        # [FUNCTION DOC] Not colocated, critic-only training with 1 node of 4 GPUs: offset is 4.
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 4

    def test_compute_rollout_offset_shared_actor_critic(self):
        # [FUNCTION DOC] Not colocated, actor with 8 GPUs and a critic with 4 GPUs: offset is 8
        # (the critic shares the actor's GPUs, so only the actor's count is skipped).
        args = make_args(
            colocate=False,
            debug_train_only=False,
            debug_rollout_only=False,
            critic_train_only=False,
            use_critic=True,
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
        )
        assert _compute_rollout_offset(args) == 8

    def test_compute_megatron_num_gpus_for_actor_only(self):
        # [FUNCTION DOC] Actor on 2 nodes of 8 GPUs, no critic: 16 Megatron GPUs.
        args = make_args(
            actor_num_nodes=2,
            actor_num_gpus_per_node=8,
            use_critic=False,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 16

    def test_compute_megatron_num_gpus_with_shared_critic(self):
        # [FUNCTION DOC] Actor with 8 GPUs plus a critic with 4 GPUs: 8 Megatron GPUs (the critic
        # is not added on top).
        args = make_args(
            actor_num_nodes=1,
            actor_num_gpus_per_node=8,
            use_critic=True,
            critic_num_nodes=1,
            critic_num_gpus_per_node=4,
            debug_rollout_only=False,
            critic_train_only=False,
        )
        assert _compute_megatron_num_gpus(args) == 8

    def test_compute_megatron_num_gpus_zero_when_debug_rollout_only(self):
        # [FUNCTION DOC] With debug rollout-only the Megatron GPU count is 0.
        args = make_args(debug_rollout_only=True)
        assert _compute_megatron_num_gpus(args) == 0


class TestCreateRolloutServersWiring:
    # [CLASS DOC] Tests the server factory's wiring using a YAML config with models actor
    # (updatable, 8 GPUs, 4 per engine) and ref (frozen, 4 GPUs).
    _CONFIG_YAML = (
        "sglang:\n"
        "  - name: actor\n"
        "    server_groups:\n"
        "      - worker_type: regular\n"
        "        num_gpus: 8\n"
        "        num_gpus_per_engine: 4\n"
        "  - name: ref\n"
        "    model_path: /fake/ref-model\n"
        "    update_weights: false\n"
        "    server_groups:\n"
        "      - worker_type: regular\n"
        "        num_gpus: 4\n"
        "        num_gpus_per_engine: 4\n"
    )

    @pytest.mark.asyncio
    async def test_create_rollout_servers_wires_each_model_and_router_completely(self, tmp_path):
        # [FUNCTION DOC] Writes the YAML config, builds debug rollout-only arguments with 12 GPUs,
        # and calls the factory with a named lock, the stub provider and router addresses for actor
        # and ref. Asserts: each server has its own router address; server model names are actor
        # then ref; update-weights flags are true and false; every server uses the given lock
        # object; pausing the actor's activeness tracker leaves the ref's tracker active.
        """Every model gets its own router and its own activeness tracker, and the injected lock and update_weights survive."""

        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(self._CONFIG_YAML)
        args = make_args(sglang_config=str(cfg_path), rollout_num_gpus=12, debug_rollout_only=True)
        lock = ContextLock("InferenceControllerUnderTest")
        router_addrs = {
            name: HostAndPort(host=f"10.0.0.{model_idx + 1}", port=20000 + model_idx)
            for model_idx, name in enumerate(("actor", "ref"))
        }

        servers = await create_rollout_servers(
            args,
            context_lock=lock,
            engine_provider=_StubProvider(),
            router_addrs=router_addrs,
        )

        assert {name: (srv.router_ip, srv.router_port) for name, srv in servers.items()} == {
            "actor": ("10.0.0.1", 20000),
            "ref": ("10.0.0.2", 20001),
        }
        assert [srv.model_name for srv in servers.values()] == ["actor", "ref"]
        assert (servers["actor"].update_weights, servers["ref"].update_weights) == (True, False)
        for srv in servers.values():
            assert srv.context_lock is lock
        servers["actor"].health_checker_activeness.bump_active(False)
        assert not servers["actor"].health_checker_activeness.get().active
        assert servers["ref"].health_checker_activeness.get().active


class TestEngineListOrdering:
    # [CLASS DOC] Tests that the server's derived engine lists are ordered by GPU offset.
    def _server_with_cells(self, num_cells: int) -> RolloutServer:
        # [FUNCTION DOC] HELPER: Builds a server whose cell mapping is inserted in string-sorted
        # cell id order (so 10 and 11 come before 2). Cell i has GPUs per engine i+1, GPU offset i
        # and api client "client-i". Uses empty arguments, the shared lock helper and the stub
        # provider. Returns the server.
        cells = {}
        for index in sorted(range(num_cells), key=lambda i: f"inference-engine-0-0-{i}"):
            meta = SimpleNamespace(num_gpus_per_engine=index + 1, gpu_offset=index)
            cells[f"inference-engine-0-0-{index}"] = SimpleNamespace(meta=meta, api_client=f"client-{index}")
        return RolloutServer(
            server_cells=cells,
            args=SimpleNamespace(),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )

    @pytest.mark.asyncio
    async def test_engine_lists_are_ordered_by_gpu_offset_not_insertion(self):
        # [FUNCTION DOC] With 12 cells, under the lock asserts the GPU offsets list is 0..11, the
        # api clients are client-0..client-11 and the GPU counts are 1..12, i.e. ordered by offset.
        """With 12 cells inserted in string-sorted id order all three derived lists come out offset-ordered."""
        srv = self._server_with_cells(12)
        async with srv.context_lock:
            assert srv.engine_gpu_offsets == list(range(12))
            assert srv.api_clients == [f"client-{i}" for i in range(12)]
            assert srv.engine_gpu_counts == [i + 1 for i in range(12)]


class TestAddCellRollback:
    # [CLASS DOC] Tests adding cells to a real rollout server, including rollback of a cell whose
    # init fails (the cell is disposed and dropped, others untouched, ids reusable) and disposal.
    def _make_meta(
        self, *, needs_offload: bool = False, cell_id: str = "inference-engine-0-0-0"
    ) -> ServerCellMetadata:
        # [FUNCTION DOC] HELPER: Builds cell metadata for model default, worker type regular, one
        # GPU per engine, offset 0, no api key, weight updates on, workers hash pseudo-hash-0, with
        # the given offload flag and cell id (worker name is the cell id plus -0).
        return ServerCellMetadata(
            model_id="default",
            worker_type="regular",
            cell_id=cell_id,
            num_gpus_per_engine=1,
            gpu_offset=0,
            sglang_api_key=None,
            worker_name=f"{cell_id}-0",
            needs_offload=needs_offload,
            update_weights=True,
            workers_hash="pseudo-hash-0",
        )

    @pytest.mark.asyncio
    async def test_disposing_the_server_removes_every_cell_it_tracks(self, monkeypatch):
        # [FUNCTION DOC] Colocated server with cell init patched to a no-op; under the lock adds a
        # cell and disposes the server, then asserts the cell mapping is empty.
        """Controller teardown must reach each cell so its health checker task stops with it."""
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=True, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())
            await srv.dispose()

        assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_failed_add_drops_the_cell_after_disposing_it(self, monkeypatch):
        # [FUNCTION DOC] Non-colocated server; cell init raises a runtime error and cell dispose is
        # replaced by a recorder. Under the lock, adding a cell raises the init error; afterwards
        # the cell mapping is empty and exactly one cell was disposed.
        """A cell kept after a failed init still matches its own observation, so no later sweep ever retries it."""
        disposed: list[ServerCell] = []

        async def _record_dispose(cell: ServerCell) -> None:
            # [FUNCTION DOC] HELPER: Appends the cell to the disposed list without real teardown.
            disposed.append(cell)

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _raise_async)
        monkeypatch.setattr(ServerCell, "dispose", _record_dispose)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="injected init failure"):
                await srv.add_cell(self._make_meta())

            assert srv.server_cells == {}
            assert len(disposed) == 1

    @pytest.mark.asyncio
    async def test_a_failed_add_really_disposes_the_dropped_cell(self, monkeypatch):
        # [FUNCTION DOC] Server with the rollout fault tolerance component enabled; cell init
        # records the cell and raises. After the failed add, the recorded cell (real dispose ran)
        # reports status phase Suspended. Finally calls the cell's finalizer explicitly to clean up.
        """Forgetting the entry without disposing it leaves the cell's health checker task running for ever."""
        built: list[ServerCell] = []

        async def _record_then_raise(cell: ServerCell) -> None:
            # [FUNCTION DOC] HELPER: Records the cell then raises a runtime error "injected init
            # failure".
            built.append(cell)
            raise RuntimeError("injected init failure")

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=["rollout"]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _record_then_raise)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="injected init failure"):
                await srv.add_cell(self._make_meta())

        [cell] = built
        assert cell.cell_status().phase == "Suspended"
        cell.__del__()

    @pytest.mark.asyncio
    async def test_a_failed_add_leaves_the_cells_already_tracked_alone(self, monkeypatch):
        # [FUNCTION DOC] Adds one good cell, then makes init raise and adds a second cell, expecting
        # the init error. Asserts only the first cell remains tracked, then disposes the server.
        """Only the cell that failed may be dropped, or a single bad engine unmanages the healthy ones too."""
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())
            monkeypatch.setattr(ServerCell, "init", _raise_async)

            with pytest.raises(RuntimeError, match="injected init failure"):
                await srv.add_cell(self._make_meta(cell_id="inference-engine-0-0-1"))

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_a_cell_whose_dispose_also_fails_is_still_dropped(self, monkeypatch):
        # [FUNCTION DOC] Both cell init and cell dispose fail (dispose runs the real teardown, then
        # raises "injected dispose failure"). Adding a cell raises the dispose error (not the init
        # error), and the cell mapping is still empty afterwards.
        """Disposing before forgetting would re-strand the cell in the map whenever teardown itself errors."""
        real_dispose = ServerCell.dispose

        async def _dispose_then_fail(cell: ServerCell) -> None:
            # [FUNCTION DOC] HELPER: Runs the real cell dispose, then raises a runtime error
            # "injected dispose failure".
            await real_dispose(cell)
            raise RuntimeError("injected dispose failure")

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _raise_async)
        monkeypatch.setattr(ServerCell, "dispose", _dispose_then_fail)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="injected dispose failure"):
                await srv.add_cell(self._make_meta())

            assert srv.server_cells == {}

    @pytest.mark.asyncio
    async def test_a_dropped_cell_id_can_be_added_again(self, monkeypatch):
        # [FUNCTION DOC] Init raises on the first add (dispose is a no-op), then init is switched to
        # a no-op and the same cell id is added again; it is tracked afterwards and the server is
        # disposed.
        """Retrying is the whole point: the next observation of the same cell must be able to build it again."""
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _raise_async)
        monkeypatch.setattr(ServerCell, "dispose", _noop_async)

        async with srv.context_lock:
            with pytest.raises(RuntimeError, match="injected init failure"):
                await srv.add_cell(self._make_meta())

            monkeypatch.setattr(ServerCell, "init", _noop_async)
            await srv.add_cell(self._make_meta())

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_a_successful_add_commits_the_cell(self, monkeypatch):
        # [FUNCTION DOC] With init a no-op, adding a cell leaves it tracked; then disposes.
        # KNOWN ISSUE: the docstring talks about adding the same id after a failure, but the test
        # contains no failure and only checks a plain successful add.
        """After the failure is gone the same cell id can be added normally."""
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(self._make_meta())

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()


class TestDuplicateCellId:
    # [CLASS DOC] Tests the rollout server's handling of a repeated cell id.
    @pytest.mark.asyncio
    async def test_adding_a_duplicate_cell_id_preserves_the_original_cell(self, monkeypatch):
        # [FUNCTION DOC] Colocated server, init a no-op. Adds a cell, adds the same id again and
        # expects an assertion error; the tracked cell is still the identical original object.
        # Then disposes the server.
        """Overwriting the entry would drop the first cell's health checker task and router registration on the floor."""
        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=True, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())
            original = srv.server_cells["inference-engine-0-0-0"]

            with pytest.raises(AssertionError):
                await srv.add_cell(TestAddCellRollback()._make_meta())

            assert srv.server_cells["inference-engine-0-0-0"] is original
            await srv.dispose()


class TestRemoveCellDisposal:
    # [CLASS DOC] Tests the ordering of cell removal: dispose first, forget afterwards.
    @pytest.mark.asyncio
    async def test_remove_cell_disposes_it_before_forgetting_it(self, monkeypatch):
        # [FUNCTION DOC] Cell dispose is replaced by a version that blocks on an event. Under the
        # lock, adds a cell and starts removing it as a task; waits (one second limit) until the
        # dispose started, then asserts the cell is still tracked and the removal is not done.
        # Releases the dispose, awaits the removal and asserts the mapping is empty.
        """A forgotten-but-undisposed cell keeps its router registration and health checker task alive forever."""
        dispose_started = asyncio.Event()
        allow_dispose = asyncio.Event()
        real_dispose = ServerCell.dispose

        async def _blocking_dispose(cell: ServerCell) -> None:
            # [FUNCTION DOC] HELPER: Signals that dispose started, waits for the release event, then
            # runs the real cell dispose.
            dispose_started.set()
            await allow_dispose.wait()
            await real_dispose(cell)

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=True, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _noop_async)
        monkeypatch.setattr(ServerCell, "dispose", _blocking_dispose)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())
            remove_task = asyncio.create_task(srv.remove_cell("inference-engine-0-0-0"))
            await asyncio.wait_for(dispose_started.wait(), timeout=1)

            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            assert not remove_task.done()

            allow_dispose.set()
            await asyncio.wait_for(remove_task, timeout=1)

        assert srv.server_cells == {}


class TestAddCellInitTiming:
    # [CLASS DOC] Tests when add-cell runs the cell's init: immediately for disaggregated or
    # non-offloaded cells, deferred for offloaded cells in colocate mode.
    @pytest.mark.asyncio
    async def test_a_disaggregated_cell_is_initialized_as_soon_as_it_appears(self, monkeypatch):
        # [FUNCTION DOC] Non-colocated server with cell init replaced by a recorder; under the lock
        # adds an offloading-disabled default cell and disposes; asserts init ran once for that id.
        """Nothing competes for its gpus, so waiting would only delay it becoming servable."""
        initialized: list[str] = []

        async def _record(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement cell init that records the cell id.
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=False, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta())
            await srv.dispose()

        assert initialized == ["inference-engine-0-0-0"]

    @pytest.mark.asyncio
    async def test_a_colocated_cell_is_only_tracked_until_the_weight_update_window(self, monkeypatch):
        # [FUNCTION DOC] Colocated server with recording init; adds a cell flagged as needing
        # offload; asserts init did not run yet while the cell is tracked; then disposes.
        """Its engine may not claim gpu memory while the trainer still holds it."""
        initialized: list[str] = []

        async def _record(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement cell init that records the cell id.
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=True, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta(needs_offload=True))

            assert initialized == []
            assert list(srv.server_cells) == ["inference-engine-0-0-0"]
            await srv.dispose()

    @pytest.mark.asyncio
    async def test_a_non_offloaded_cell_is_initialized_immediately_even_under_colocate(self, monkeypatch):
        # [FUNCTION DOC] Colocated server with recording init. Adds an offloaded cell (init deferred,
        # nothing recorded), then a non-offloaded eval cell, and asserts init ran only for the eval
        # cell. Then disposes.
        """A dedicated eval cell does not share GPUs with the trainer, so it must not wait for the first weight update."""
        initialized: list[str] = []

        async def _record(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement cell init that records the cell id.
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=True, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _record)
        meta_builder = TestAddCellRollback()

        async with srv.context_lock:
            await srv.add_cell(meta_builder._make_meta(needs_offload=True))

            assert initialized == []

            await srv.add_cell(meta_builder._make_meta(needs_offload=False, cell_id="eval-engine-0-0-0"))

            assert initialized == ["eval-engine-0-0-0"]
            await srv.dispose()


class TestDeferredInitMatchesTheStartupBarrier:
    # [CLASS DOC] Tests that the cells whose init is deferred are exactly those counted as
    # startable-before-up by the startup barrier.
    @pytest.mark.parametrize(
        ("colocate", "needs_offload", "deferred"),
        [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
        ids=["disaggregated_plain", "disaggregated_offloaded", "colocated_eval", "colocated_shared"],
    )
    @pytest.mark.asyncio
    async def test_only_a_cell_sharing_the_trainer_gpus_is_deferred_and_counted_before_it_is_up(
        self, monkeypatch, colocate: bool, needs_offload: bool, deferred: bool
    ):
        # [FUNCTION DOC] Parametrized over (colocate, needs offload, deferred): only colocated plus
        # offloaded is deferred. Adds one cell with those settings (recording init) and asserts init
        # was skipped exactly when deferred is expected, and the server's startable cell count is 1
        # when deferred and 0 otherwise. Then disposes.
        """Whichever cells skip startup init are exactly the ones the barrier may pass without an address."""
        initialized: list[str] = []

        async def _record(self) -> None:
            # [FUNCTION DOC] HELPER: Replacement cell init that records the cell id.
            initialized.append(self.meta.cell_id)

        srv = RolloutServer(
            server_cells={},
            args=make_args(colocate=colocate, ft_components=[]),
            context_lock=_make_lock(),
            engine_provider=_StubProvider(),
        )
        monkeypatch.setattr(ServerCell, "init", _record)

        async with srv.context_lock:
            await srv.add_cell(TestAddCellRollback()._make_meta(needs_offload=needs_offload))

            assert (initialized == []) is deferred
            assert srv._count_startable_cells() == (1 if deferred else 0)
            await srv.dispose()


async def _make_serving_server(monkeypatch, *, num_cells: int) -> RolloutServer:
    # [FUNCTION DOC] HELPER: Builds a colocated server (no ft components, no miles router) whose
    # router api client property is patched to a recording client. Under the lock adds num_cells
    # offloaded cells named default-i (GPU offset i, hash pseudo-hash-i), registers each with the
    # router using an address of host 10.0.0.(i+1) and forces its state to serving. Returns the
    # server. Side effects: patches the server class property; init is not patched, so each add
    # relies on offloaded cells being deferred.
    # KNOWN ISSUE: not referenced by any test in this file (dead helper).
    """A server whose cells are all registered and serving, without dialling anything."""
    router = _RecordingRouterApiClient()
    monkeypatch.setattr(RolloutServer, "_router_api_client", property(lambda self: router))

    srv = RolloutServer(
        server_cells={},
        args=make_args(colocate=True, ft_components=[], use_miles_router=False),
        context_lock=_make_lock(),
        engine_provider=_StubProvider(),
    )
    async with srv.context_lock:
        for cell_index in range(num_cells):
            await srv.add_cell(
                ServerCellMetadata(
                    model_id="default",
                    worker_type="regular",
                    cell_id=f"default-{cell_index}",
                    num_gpus_per_engine=1,
                    gpu_offset=cell_index,
                    sglang_api_key=None,
                    worker_name=f"default-{cell_index}-0",
                    needs_offload=True,
                    update_weights=True,
                    workers_hash=f"pseudo-hash-{cell_index}",
                )
            )
            cell = srv.server_cells[f"default-{cell_index}"]
            addr_info = CellAddrInfo(
                server_url=f"http://10.0.0.{cell_index + 1}:3000{cell_index}", bootstrap_port=None, gate_url=None
            )
            await cell._register_with_router(addr_info=addr_info)
            cell._state = StateServing(addr_info=addr_info)
    return srv


class _RecordingRouterApiClient:
    # [CLASS DOC] HELPER: Fake router api client that records worker add and remove calls.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the empty list of recorded calls (name plus keywords).
        self.calls: list[tuple[str, dict]] = []

    async def add_worker(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Records an add_worker call with its keyword arguments.
        self.calls.append(("add_worker", kwargs))

    async def remove_worker(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Records a remove_worker call with its keyword arguments.
        self.calls.append(("remove_worker", kwargs))


class _StubProvider:
    # [CLASS DOC] HELPER: Stub worker provider that fails the test if asked for worker addresses
    # (cell startup is patched out) and reports no opinion on the expected number of cells.
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Always raises an assertion error naming the worker.
        raise AssertionError(f"cell startup is patched out in this module ({worker_name=})")

    def expected_num_cells(self, *, group_id: str) -> int | None:
        # [FUNCTION DOC] HELPER: Returns nothing (no opinion) for any group.
        return None


def _make_lock() -> ContextLock:
    # [FUNCTION DOC] HELPER: Returns a new context lock named after the inference controller.
    return ContextLock("InferenceController")


async def _raise_async(self):
    # [FUNCTION DOC] HELPER: Replacement for an asynchronous method; ignores the instance and
    # always raises a runtime error "injected init failure".
    raise RuntimeError("injected init failure")


async def _noop_async(self):
    # [FUNCTION DOC] HELPER: Replacement for an asynchronous method that does nothing and returns
    # nothing.
    return None
