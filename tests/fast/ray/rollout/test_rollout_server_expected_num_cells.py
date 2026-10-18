# [MODULE DOC]
# Unit tests for how the rollout server factory computes each server's expected number of cells
# (the startup barrier target): cross-checked against the engine specs for several config shapes
# (single group, placeholder group, prefill/decode split, multi model, multi-node engine), the
# per-model router pairing, precedence between an explicit expected-cells flag, a provider's own
# answer and the config derivation, pool naming, router flag validation at startup, and the
# registering (split deployment) case. Configs are rendered to a temporary YAML file; providers are
# stubs that refuse address and worker lookups, so no cells are created.
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.specs.inference import compute_engine_pool_id, specs_inference_engine
from miles.utils.context_lock import ContextLock
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_CONFIG_SINGLE_GROUP: list[dict] = [
    dict(
        name="actor",
        server_groups=[dict(worker_type="regular", num_gpus=8, num_gpus_per_engine=1)],
    ),
]

_CONFIG_WITH_PLACEHOLDER: list[dict] = [
    dict(
        name="actor",
        server_groups=[
            dict(worker_type="regular", num_gpus=8, num_gpus_per_engine=4),
            dict(worker_type="placeholder", num_gpus=8, num_gpus_per_engine=4),
        ],
    ),
]

_CONFIG_PD_DISAGGREGATED: list[dict] = [
    dict(
        name="actor",
        server_groups=[
            dict(worker_type="prefill", num_gpus=4, num_gpus_per_engine=2),
            dict(worker_type="decode", num_gpus=8, num_gpus_per_engine=4),
        ],
    ),
]

_CONFIG_MULTI_MODEL: list[dict] = [
    dict(
        name="actor",
        server_groups=[
            dict(worker_type="regular", num_gpus=8, num_gpus_per_engine=2),
            dict(worker_type="placeholder", num_gpus=2, num_gpus_per_engine=2),
        ],
    ),
    dict(
        name="ref",
        model_path="/fake/ref-model",
        update_weights=False,
        server_groups=[dict(worker_type="regular", num_gpus=4, num_gpus_per_engine=4)],
    ),
]

_CONFIG_MULTI_NODE_ENGINE: list[dict] = [
    dict(
        name="actor",
        server_groups=[dict(worker_type="regular", num_gpus=16, num_gpus_per_engine=16)],
    ),
]


def _render_config_yaml(models: list[dict]) -> str:
    # [FUNCTION DOC] HELPER: Renders a list of model dictionaries to sglang YAML text: a top-level
    # sglang list; per model a name line, optional model path and update-weights lines (boolean
    # lowercased) and a server groups list, each group giving worker type, GPU count and GPUs per
    # engine. Returns the text ending with a newline. Pure.
    lines: list[str] = ["sglang:"]
    for model in models:
        lines.append(f"  - name: {model['name']}")
        if model.get("model_path") is not None:
            lines.append(f"    model_path: {model['model_path']}")
        if model.get("update_weights") is not None:
            lines.append(f"    update_weights: {str(model['update_weights']).lower()}")
        lines.append("    server_groups:")
        for group in model["server_groups"]:
            lines.append(f"      - worker_type: {group['worker_type']}")
            lines.append(f"        num_gpus: {group['num_gpus']}")
            lines.append(f"        num_gpus_per_engine: {group['num_gpus_per_engine']}")
    return "\n".join(lines) + "\n"


def _make_args_with_config(models: list[dict], tmp_path: Path) -> Namespace:
    # [FUNCTION DOC] HELPER: Writes the rendered models as a YAML file in the temporary directory
    # and returns test arguments pointing at it, with rollout GPUs equal to the sum of all groups'
    # GPUs, 8 GPUs per node and debug rollout-only on. Side effect: writes the YAML file.
    config_path = tmp_path / "sglang_config.yaml"
    config_path.write_text(_render_config_yaml(models))
    total_num_gpus = sum(group["num_gpus"] for model in models for group in model["server_groups"])
    return make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=total_num_gpus,
        num_gpus_per_node=8,
        debug_rollout_only=True,
    )


def _expected_num_cells_from_specs(args: Namespace) -> dict[int, int]:
    # [FUNCTION DOC] HELPER: Computes the ground truth: gets the engine specs for the arguments,
    # parses the model index from the second-to-last dash separated part of each spec name, and
    # sums the specs' scheduling cell counts per model index. Returns a mapping from model index
    # to cell count. Pure.
    specs_by_name = {spec.name: spec for spec in specs_inference_engine(args)}
    counts: dict[int, int] = {}
    for name, spec in specs_by_name.items():
        model_idx = int(name.split("-")[-2])
        counts[model_idx] = counts.get(model_idx, 0) + spec.scheduling.num_cells
    return counts


class _StubProvider(BaseWorkerProvider):
    # [CLASS DOC] HELPER: Provider that fails the test on address or worker info lookups (no cell
    # is created here) and inherits the base class's default of having no opinion about the
    # expected number of cells.
    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        # [FUNCTION DOC] HELPER: Always raises an assertion error naming the worker.
        raise AssertionError(f"no cell is created in this module ({worker_name=})")

    def get_worker_infos(self, *, cell_ids: list[str]) -> list:
        # [FUNCTION DOC] HELPER: Always raises an assertion error naming the cell ids.
        raise AssertionError(f"no cell is created in this module ({cell_ids=})")


class _CountingProvider(_StubProvider):
    # [CLASS DOC] HELPER: Stub provider that answers a fixed expected cell count for any group.
    def __init__(self, count: int) -> None:
        # [FUNCTION DOC] HELPER: Stores the count to answer.
        self._count = count

    def expected_num_cells(self, *, group_id: str) -> int:
        # [FUNCTION DOC] HELPER: Returns the stored count regardless of the group id.
        return self._count


def _make_router_addrs(models: list[dict]) -> dict[str, HostAndPort]:
    # [FUNCTION DOC] HELPER: Gives each model name a router address on 127.0.0.1 with port 20000
    # plus the model's position. Returns the mapping.
    return {
        model["name"]: HostAndPort(host="127.0.0.1", port=20000 + model_idx) for model_idx, model in enumerate(models)
    }


async def _create_servers(args: Namespace, models: list[dict]) -> dict[str, RolloutServer]:
    # [FUNCTION DOC] HELPER: Calls the server factory with a named context lock, the stub provider
    # and router addresses for the given models. Returns the created servers by model name.
    return await create_rollout_servers(
        args,
        context_lock=ContextLock("InferenceController"),
        engine_provider=_StubProvider(),
        router_addrs=_make_router_addrs(models),
    )


class TestExpectedNumCellsMatchesTheEngineSpecs:
    # [CLASS DOC] Tests that the factory's per-server expected cell count (without a provider
    # opinion) equals the number of cells the engine specs launch, per model.
    @pytest.mark.parametrize(
        "models",
        [
            _CONFIG_SINGLE_GROUP,
            _CONFIG_WITH_PLACEHOLDER,
            _CONFIG_PD_DISAGGREGATED,
            _CONFIG_MULTI_MODEL,
            _CONFIG_MULTI_NODE_ENGINE,
        ],
        ids=["single_group", "with_placeholder", "pd_disaggregated", "multi_model", "multi_node_engine"],
    )
    async def test_the_startup_barrier_expects_exactly_the_cells_the_specs_launch(
        self, models: list[dict], tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Parametrized over five configs (single group, with placeholder, prefill
        # and decode split, multi model, multi-node engine). Builds arguments from the config,
        # creates the servers, and asserts each model's init expected cell count equals the count
        # derived from the engine specs for that model index.
        """The barrier target must equal the engine cells RayWorkerManager actually starts, or startup hangs until timeout."""
        args = _make_args_with_config(models=models, tmp_path=tmp_path)
        expected_per_model_idx = _expected_num_cells_from_specs(args)

        servers = await _create_servers(args, models)

        actual_per_model_idx = {
            model_idx: servers[model["name"]].init_expected_num_cells for model_idx, model in enumerate(models)
        }
        assert actual_per_model_idx == expected_per_model_idx

    async def test_a_placeholder_group_contributes_no_cell_to_the_barrier(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Config with a regular group (8 GPUs, 4 per engine) and an equal placeholder
        # group: the actor server expects 2 cells (the placeholder adds none).
        """Placeholder groups only reserve GPU slots, so counting them would make the barrier unreachable."""
        args = _make_args_with_config(models=_CONFIG_WITH_PLACEHOLDER, tmp_path=tmp_path)

        servers = await _create_servers(args, _CONFIG_WITH_PLACEHOLDER)

        assert servers["actor"].init_expected_num_cells == 2

    async def test_every_model_gets_its_own_barrier_target(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Multi-model config: the actor expects 4 cells (8 GPUs at 2 per engine; the
        # placeholder adds none) and ref expects 1.
        """Sharing one pool size across models would block the small model behind the big one."""
        args = _make_args_with_config(models=_CONFIG_MULTI_MODEL, tmp_path=tmp_path)

        servers = await _create_servers(args, _CONFIG_MULTI_MODEL)

        assert servers["actor"].init_expected_num_cells == 4
        assert servers["ref"].init_expected_num_cells == 1

    async def test_every_model_talks_to_its_own_router(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Multi-model config: each server's router host and port equal the address
        # that was supplied for that model's name.
        """A server paired with another model's router would advertise its cells to the wrong fleet."""
        args = _make_args_with_config(models=_CONFIG_MULTI_MODEL, tmp_path=tmp_path)

        servers = await _create_servers(args, _CONFIG_MULTI_MODEL)

        actual = {
            name: HostAndPort(host=server.router_ip, port=server.router_port) for name, server in servers.items()
        }
        assert actual == _make_router_addrs(_CONFIG_MULTI_MODEL)


class TestExpectedNumCellsAsksTheProvider:
    # [CLASS DOC] Tests that the factory asks the engine provider for an expected cell count and
    # falls back to the config derivation only when the provider has no opinion (none).
    async def test_a_provider_with_an_opinion_overrides_the_config_derivation(self) -> None:
        # [FUNCTION DOC] External rollout with two engine addresses and a provider answering 2:
        # the default server's expected count is 2.
        """External providers know their fleet, so their count must beat the gpu-count formula."""
        args = make_args(rollout_external=True, rollout_external_engine_addrs=["host1:8000", "host2:8000"])

        servers = await create_rollout_servers(
            args,
            context_lock=ContextLock("InferenceController"),
            engine_provider=_CountingProvider(2),
            router_addrs={"default": HostAndPort(host="127.0.0.1", port=20000)},
        )

        assert servers["default"].init_expected_num_cells == 2

    async def test_a_provider_that_starts_with_no_cells_is_not_treated_as_having_no_opinion(self) -> None:
        # [FUNCTION DOC] Arguments whose config would give 2 cells, external rollout, and a provider
        # answering 0: the expected count is 0 (zero is an answer, not "no opinion").
        """An elastic provider legitimately announces zero cells at startup; folding that into the
        config formula would park the run on a barrier waiting for cells nobody will create."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4, rollout_external=True)

        servers = await create_rollout_servers(
            args,
            context_lock=ContextLock("InferenceController"),
            engine_provider=_CountingProvider(0),
            router_addrs={"default": HostAndPort(host="127.0.0.1", port=20000)},
        )

        assert servers["default"].init_expected_num_cells == 0

    async def test_a_provider_without_an_opinion_falls_back_to_the_config_derivation(self) -> None:
        # [FUNCTION DOC] 8 rollout GPUs at 4 per engine with the stub provider (no opinion): the
        # expected count is 2.
        """Backend providers announce cells they are told to launch, so the config stays the source."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4)

        servers = await create_rollout_servers(
            args,
            context_lock=ContextLock("InferenceController"),
            engine_provider=_StubProvider(),
            router_addrs={"default": HostAndPort(host="127.0.0.1", port=20000)},
        )

        assert servers["default"].init_expected_num_cells == 2


class TestEngineSpecNamingUsedByTheCrossCheck:
    # [CLASS DOC] Pins the engine pool id format that the cross-check helper parses.
    def test_pool_names_carry_the_model_index_the_cross_check_parses(self) -> None:
        # [FUNCTION DOC] The pool id for default arguments, model index 3 and group index 7 equals
        # the text inference-engine-all-3-7.
        """The cross-check maps specs back to models by name, so that encoding must stay stable."""
        assert compute_engine_pool_id(make_args(), model_idx=3, group_index=7) == "inference-engine-all-3-7"


class TestRouterFlagsAtStartup:
    # [CLASS DOC] Tests router related argument handling at startup.
    async def test_a_pinned_router_port_is_accepted(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Sets a fixed router port (31000) on single-group arguments and asserts the
        # server factory still returns a non-empty server mapping.
        """Three examples pass --sglang-router-port so a firewall rule can name the port in
        advance, and the spec layer pins it; rejecting it here fails those launches outright."""
        args = _make_args_with_config(models=_CONFIG_SINGLE_GROUP, tmp_path=tmp_path)
        args.sglang_router_port = 31000

        assert await _create_servers(args, _CONFIG_SINGLE_GROUP)

    async def test_an_external_router_ip_is_still_rejected(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Sets a router ip (10.0.0.9) on the arguments; resolving router addresses
        # with a stub router provider raises an assertion error saying external router mode was
        # removed.
        """Attaching to a router miles did not start is not supported yet, and silently starting
        one anyway would put two routers in front of the same engines."""
        args = _make_args_with_config(models=_CONFIG_SINGLE_GROUP, tmp_path=tmp_path)
        args.sglang_router_ip = "10.0.0.9"

        with pytest.raises(AssertionError, match="external router mode was removed"):
            await resolve_router_addrs(args, router_providers=[_StubProvider()])


class TestInitExpectedNumCellsOfARegisteringRun:
    # [CLASS DOC] Tests the expected cell count for a run whose engines are deployed elsewhere and
    # register themselves (deploy component set to primary): precedence between the explicit
    # expected-cells flag, a provider's answer, and the fallback of one cell.
    async def test_a_run_whose_engines_register_waits_for_one_cell_unless_told_otherwise(self) -> None:
        # [FUNCTION DOC] Primary deploy component, no flag, stub provider: expected count is 1.
        """A split run reaches its first rollout on one engine, so the simplest split needs no flag."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4, deploy_component="primary")

        servers = await _servers_of(args, provider=_StubProvider())

        assert servers["default"].init_expected_num_cells == 1

    async def test_a_run_whose_engines_register_waits_for_the_number_it_was_told(self) -> None:
        # [FUNCTION DOC] Primary deploy component with the expected-cells flag set to 3 and the
        # stub provider: expected count is 3.
        """Its own config sizes engines it does not deploy, so only the flag names what to wait for."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4, deploy_component="primary")
        args.init_expected_num_cells = 3

        servers = await _servers_of(args, provider=_StubProvider())

        assert servers["default"].init_expected_num_cells == 3

    async def test_a_run_deploying_its_own_engines_derives_the_barrier_from_its_config(self) -> None:
        # [FUNCTION DOC] No deploy component, 8 GPUs at 4 per engine, stub provider: expected count
        # is 2 (derived from the config).
        """It launches every cell it waits for, and the flag is refused where the arguments are validated."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4)

        servers = await _servers_of(args, provider=_StubProvider())

        assert servers["default"].init_expected_num_cells == 2

    async def test_the_flag_beats_a_provider_that_answers_for_its_own_fleet(self) -> None:
        # [FUNCTION DOC] Primary deploy component, flag 3, provider answering 7: expected count is
        # 3 (the explicit flag wins).
        """Nothing here can tell a stale provider answer from a live one, so an explicit number wins."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4, deploy_component="primary")
        args.init_expected_num_cells = 3

        servers = await _servers_of(args, provider=_CountingProvider(7))

        assert servers["default"].init_expected_num_cells == 3

    async def test_a_provider_that_answers_beats_the_registration_fallback(self) -> None:
        # [FUNCTION DOC] Primary deploy component, no flag, provider answering 7: expected count is
        # 7 (the provider's answer beats the one-cell fallback).
        """An external provider knows its own fleet, and the fallback is a guess of one cell."""
        args = make_args(rollout_num_gpus=8, rollout_num_gpus_per_engine=4, deploy_component="primary")

        servers = await _servers_of(args, provider=_CountingProvider(7))

        assert servers["default"].init_expected_num_cells == 7


async def _servers_of(args: Namespace, *, provider: BaseWorkerProvider) -> dict[str, RolloutServer]:
    # [FUNCTION DOC] HELPER: Calls the server factory with a named context lock, the given engine
    # provider and a single default router address (127.0.0.1 port 20000). Returns the servers.
    return await create_rollout_servers(
        args,
        context_lock=ContextLock("InferenceController"),
        engine_provider=provider,
        router_addrs={"default": HostAndPort(host="127.0.0.1", port=20000)},
    )
