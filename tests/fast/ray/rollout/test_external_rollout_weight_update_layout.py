# [MODULE DOC]
# Tests of how the gpu counts and offsets discovered for an external engine fleet reach the weight
# update group layout. Discovery of engine descriptions is faked through a patched fetch function; a
# rollout server is assembled from the discovered cells, and the broadcast-protocol connect function
# is run against recording engines with ray, async utilities and process-group initialization
# patched out, to observe the rank and world size each engine is told. Covers a prefill/decode fleet,
# deterministic ordering under out-of-order discovery answers, and a single multi-rank regular
# engine.
import asyncio
from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.backends.training_utils.weight_update.protocols.broadcast import connect_rollout_engines_from_distributed
from miles.ray.rollout import external_engine_provider as external_engine_provider_module
from miles.ray.rollout.external_engine_provider import StaticInferenceEngineWorkerProvider
from miles.ray.rollout.inference_controller import _compute_server_cell_meta_from_info
from miles.ray.rollout.rollout_server import RolloutServer
from miles.utils.context_lock import ContextLock

_BROADCAST_MODULE = "miles.backends.training_utils.weight_update.protocols.broadcast"


def _payload(*, num_gpus: int, disaggregation_mode: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds a fake server info answer with one internal state reporting the
    # given world size, the given disaggregation mode, and bootstrap port 12090 for prefill mode only
    # (none otherwise).
    return dict(
        internal_states=[dict(world_size=num_gpus)],
        disaggregation_mode=disaggregation_mode,
        disaggregation_bootstrap_port=12090 if disaggregation_mode == "prefill" else None,
    )


class _RecordingEngine:
    # [CLASS DOC] HELPER: Fake rollout engine whose weight update group initialization records the rank
    # and world size it was given into a shared list.
    def __init__(self, calls: list[dict]) -> None:
        # [FUNCTION DOC] HELPER: Stores the shared call record list.
        self._calls = calls

    def init_weights_update_group(self, master_address, master_port, rank, world_size, group_name, backend):
        # [FUNCTION DOC] HELPER: Appends a record of rank and world size and returns nothing. The master
        # address, port, group name and backend are accepted and ignored.
        self._calls.append(dict(rank=rank, world_size=world_size))
        return None


async def _discovered_server(
    monkeypatch,
    *,
    payloads: dict[str, dict[str, Any]],
    urls: list[str],
    rollout_num_gpus_per_engine: int,
) -> RolloutServer:
    # [FUNCTION DOC] HELPER: Replaces the provider module's server info fetch with a lookup in the
    # given payload mapping, builds external-rollout args (the prefill/decode router flag is on iff any
    # payload has a non-null disaggregation mode; total gpus is the sum of the reported world sizes),
    # creates and initializes the static provider, and wraps each discovered cell in a simple object
    # holding its server cell meta and a fake api client string named for its gpu offset. Returns a
    # rollout server over those cells, the args, a fresh inference-controller-named context lock, and
    # the provider as engine provider.
    async def _fetch(url: str, *, api_key: str | None) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Returns the mapped payload for the url, ignoring the api key.
        return payloads[url]

    monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)
    args = make_args(
        rollout_external=True,
        rollout_external_engine_addrs=urls,
        rollout_external_router_pd=any(p["disaggregation_mode"] != "null" for p in payloads.values()),
        rollout_num_gpus=sum(p["internal_states"][0]["world_size"] for p in payloads.values()),
        rollout_num_gpus_per_engine=rollout_num_gpus_per_engine,
    )
    provider = StaticInferenceEngineWorkerProvider(args=args)
    await provider.init()

    cells = {}
    for info in provider.cell_infos:
        meta = _compute_server_cell_meta_from_info(info)
        cells[info.cell_id] = SimpleNamespace(meta=meta, api_client=f"client-{meta.gpu_offset}")
    return RolloutServer(
        server_cells=cells,
        args=args,
        context_lock=ContextLock("InferenceController"),
        engine_provider=provider,
    )


def _connect(*, engine_gpu_counts: list[int], rollout_num_gpus_per_engine: int) -> list[dict]:
    # [FUNCTION DOC] HELPER: Creates one recording engine per entry of the gpu counts, then runs the
    # broadcast protocol's connect function for a group named miles-pp_0 with args holding only the
    # per-engine gpu argument, the engines, and the gpu counts. Inside the call, ray, the async
    # utilities (submit returns its argument as is, waiting on futures does nothing) and process-group
    # initialization of the broadcast module are patched out. Returns the recorded rank and world size
    # of each engine in call order.
    calls: list[dict] = []
    engines = [_RecordingEngine(calls) for _ in engine_gpu_counts]
    async_utils = SimpleNamespace(submit=lambda coro: coro, wait_futures=lambda futures: None)

    with (
        patch(f"{_BROADCAST_MODULE}.ray"),
        patch(f"{_BROADCAST_MODULE}.async_utils", async_utils),
        patch(f"{_BROADCAST_MODULE}.init_process_group"),
    ):
        connect_rollout_engines_from_distributed(
            Namespace(rollout_num_gpus_per_engine=rollout_num_gpus_per_engine),
            "miles-pp_0",
            engines,
            engine_gpu_counts=engine_gpu_counts,
        )
    return calls


class TestExternalPdFleetWeightUpdateLayout:
    # [CLASS DOC] Checks the weight update layout for an external prefill/decode fleet.
    @pytest.mark.asyncio
    async def test_the_discovered_gpu_counts_reach_the_update_group_unchanged(self, monkeypatch):
        # [FUNCTION DOC] A prefill and a decode engine of two gpus each, with a per-engine argument of
        # two. Under the server's lock asserts the server reports gpu counts [2, 2], offsets [0, 2], and
        # api clients named for offsets 0 and 2. Then connects with those counts but a different
        # per-engine argument of one (a deliberately different fallback) and asserts the engines were
        # given ranks 1 and 3 (rank zero is the trainer side) and a world size of five for both.
        """Discovered engine sizes reach the group instead of its deliberately different fallback."""
        urls = ["prefill:8000", "decode:8000"]
        payloads = {
            "http://prefill:8000": _payload(num_gpus=2, disaggregation_mode="prefill"),
            "http://decode:8000": _payload(num_gpus=2, disaggregation_mode="decode"),
        }

        srv = await _discovered_server(
            monkeypatch,
            payloads=payloads,
            urls=urls,
            rollout_num_gpus_per_engine=2,
        )

        async with srv.context_lock:
            assert srv.engine_gpu_counts == [2, 2]
            assert srv.engine_gpu_offsets == [0, 2]
            assert srv.api_clients == ["client-0", "client-2"]
            counts = srv.engine_gpu_counts

        calls = _connect(engine_gpu_counts=counts, rollout_num_gpus_per_engine=1)

        assert [call["rank"] for call in calls] == [1, 3]
        assert {call["world_size"] for call in calls} == {5}

    @pytest.mark.asyncio
    async def test_the_slower_engine_does_not_take_the_earlier_rank_range(self, monkeypatch):
        # [FUNCTION DOC] The first listed engine (prefill, slow) only answers discovery after the
        # second (decode, fast) has answered, via a fake fetch that spins by yielding to the event loop
        # until some answer was recorded. After initialization asserts the answer order was fast then
        # slow, yet the cells follow the listed order: worker types prefill then decode, two gpus per
        # engine each, and gpu offsets 0 and 2.
        """Discovery runs concurrently, so if cells were built as answers arrive the rank layout would
        depend on network timing and drift silently between runs."""
        urls = ["slow:8000", "fast:8000"]
        payloads = {
            "http://slow:8000": _payload(num_gpus=2, disaggregation_mode="prefill"),
            "http://fast:8000": _payload(num_gpus=2, disaggregation_mode="decode"),
        }
        answered: list[str] = []

        async def _fetch(url: str, *, api_key: str | None) -> dict[str, Any]:
            # [FUNCTION DOC] HELPER: For the slow url, repeatedly yields to the event loop until any
            # answer has been recorded; then records the url as answered and returns its payload.
            while url == "http://slow:8000" and not answered:
                await asyncio.sleep(0)
            answered.append(url)
            return payloads[url]

        monkeypatch.setattr(external_engine_provider_module, "_fetch_server_info_with_retry", _fetch)
        args = make_args(
            rollout_external=True,
            rollout_external_engine_addrs=urls,
            rollout_external_router_pd=True,
            rollout_num_gpus=4,
            rollout_num_gpus_per_engine=2,
        )
        provider = StaticInferenceEngineWorkerProvider(args=args)
        await provider.init()

        assert answered == ["http://fast:8000", "http://slow:8000"]
        assert [info.meta["worker_type"] for info in provider.cell_infos] == ["prefill", "decode"]
        assert [info.meta["num_gpus_per_engine"] for info in provider.cell_infos] == [2, 2]
        assert [info.meta["gpu_offset"] for info in provider.cell_infos] == [0, 2]


class TestExternalRegularEngineWeightUpdateLayout:
    # [CLASS DOC] Checks the weight update layout for an external fleet of regular engines.
    @pytest.mark.asyncio
    async def test_one_multi_rank_engine_claims_every_rank_it_reported(self, monkeypatch):
        # [FUNCTION DOC] A single regular engine reporting two gpus. Under the server's lock asserts the
        # gpu counts are [2]; then connects with a per-engine argument of one and asserts a single engine
        # was given rank 1 and world size 3 (one trainer rank plus the engine's two ranks).
        """The deleted external e2e ran a single multi-gpu engine, which is the only shape where the
        per-engine count and the fleet total disagree without any heterogeneity."""
        urls = ["host1:8000"]
        payloads = {"http://host1:8000": _payload(num_gpus=2, disaggregation_mode="null")}

        srv = await _discovered_server(
            monkeypatch,
            payloads=payloads,
            urls=urls,
            rollout_num_gpus_per_engine=2,
        )

        async with srv.context_lock:
            counts = srv.engine_gpu_counts
        assert counts == [2]

        calls = _connect(engine_gpu_counts=counts, rollout_num_gpus_per_engine=1)

        assert [call["rank"] for call in calls] == [1]
        assert {call["world_size"] for call in calls} == {3}
