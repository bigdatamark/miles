# [MODULE DOC]
# Unit tests for how a server cell derives its addresses: the private address info computation
# (server url from the primary port of the cell's rank-zero worker, gate url from the optional gate
# port, bootstrap port for prefill cells only, bracketed ipv6 hosts), the helper computing how many
# nodes an engine spans, and the cell's engine api client carrying the api key. A stub provider
# returns canned addresses and records the requested worker names. All tests are marked with a
# fixture that disposes the tracked cells.
from __future__ import annotations

import pytest
from tests.fast.ray.rollout.conftest import make_args, track_server_cell

from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata, compute_nodes_per_engine
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.usefixtures("dispose_tracked_server_cells")


def _make_meta(**overrides) -> ServerCellMetadata:
    # [FUNCTION DOC] HELPER: Builds cell metadata for model default, regular worker type, cell id
    # inference-engine-0-0-0 with worker name inference-engine-0-0-0-0, one GPU per engine, offset
    # 0, no api key, no offload, weight updates on and hash pseudo-hash-0; keyword overrides
    # replace any of these fields.
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
    # [CLASS DOC] HELPER: Stub worker provider that returns a fixed address mapping for any worker
    # and records the worker names it was asked about.
    def __init__(self, addrs: dict[str, HostAndPort]):
        # [FUNCTION DOC] HELPER: Stores the address mapping and an empty list of requested names.
        self._addrs = addrs
        self.requested_worker_names: list[str] = []

    async def get_addrs(self, worker_name: str) -> dict[str, HostAndPort]:
        # [FUNCTION DOC] HELPER: Records the worker name and returns the stored address mapping.
        self.requested_worker_names.append(worker_name)
        return self._addrs


@pytest.fixture
def stub_provider():
    # [FUNCTION DOC] HELPER: Fixture returning a factory function that builds a stub provider from
    # an address mapping.
    def _install(addrs: dict[str, HostAndPort]) -> _StubProvider:
        # [FUNCTION DOC] HELPER: Builds a stub provider for the given addresses.
        return _StubProvider(addrs)

    return _install


def _make_cell(provider: _StubProvider, **meta_overrides) -> ServerCell:
    # [FUNCTION DOC] HELPER: Builds a real server cell with default test arguments, metadata from
    # the overrides, no router client and the given provider, and registers it with the tracking
    # helper so it is disposed after the test. Returns the cell.
    return track_server_cell(
        ServerCell(args=make_args(), meta=_make_meta(**meta_overrides), router_api_client=None, provider=provider)
    )


class TestComputeAddrInfo:
    # [CLASS DOC] Tests the cell's private address info computation from provider addresses.
    async def test_the_cell_addresses_are_read_from_its_rank_zero_worker(self, stub_provider):
        # [FUNCTION DOC] Provider with primary 10.0.0.1:30000 and gate 10.0.0.1:13000: the provider
        # is asked only for worker inference-engine-0-0-0-0 and the server url is
        # http://10.0.0.1:30000.
        """A cell is addressed through the worker holding its master-mode ports."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13000),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert provider.requested_worker_names == ["inference-engine-0-0-0-0"]
        assert addr_info.server_url == "http://10.0.0.1:30000"

    async def test_the_gate_url_points_at_the_out_of_band_control_port(self, stub_provider):
        # [FUNCTION DOC] Gate port 13007 yields the gate url http://10.0.0.1:13007.
        """Activation must target the gate port, never the engine's serving port."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13007),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert addr_info.gate_url == "http://10.0.0.1:13007"

    async def test_a_provider_without_a_gate_port_yields_no_gate_url(self, stub_provider):
        # [FUNCTION DOC] Only a primary port provided: the gate url is none.
        """External engines publish no gate, so the cell must not fabricate one."""
        provider = stub_provider(dict(primary=HostAndPort(host="10.0.0.1", port=30000)))

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert addr_info.gate_url is None

    async def test_a_prefill_cell_also_carries_its_disaggregation_bootstrap_port(self, stub_provider):
        # [FUNCTION DOC] A prefill cell whose provider also offers a disaggregation bootstrap port
        # 11000 reports bootstrap port 11000.
        """PD disaggregation needs this port published to the router alongside the url."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13000),
                disaggregation_bootstrap=HostAndPort(host="10.0.0.1", port=11000),
            )
        )

        addr_info = await _make_cell(provider, worker_type="prefill")._compute_addr_info()

        assert addr_info.bootstrap_port == 11000

    async def test_a_regular_cell_has_no_bootstrap_port(self, stub_provider):
        # [FUNCTION DOC] A regular cell with only primary and gate ports reports bootstrap port none.
        """Only prefill engines allocate a bootstrap port, so the rest must report None."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="10.0.0.1", port=30000),
                gate=HostAndPort(host="10.0.0.1", port=13000),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert addr_info.bootstrap_port is None

    async def test_ipv6_hosts_stay_bracketed_in_both_urls(self, stub_provider):
        # [FUNCTION DOC] Hosts given as the bracketed ipv6 literal [fd00::1] produce server url
        # http://[fd00::1]:30000 and gate url http://[fd00::1]:13000.
        """Unbracketed ipv6 literals would make both urls unparseable."""
        provider = stub_provider(
            dict(
                primary=HostAndPort(host="[fd00::1]", port=30000),
                gate=HostAndPort(host="[fd00::1]", port=13000),
            )
        )

        addr_info = await _make_cell(provider)._compute_addr_info()

        assert addr_info.server_url == "http://[fd00::1]:30000"
        assert addr_info.gate_url == "http://[fd00::1]:13000"


class TestComputeNodesPerEngine:
    # [CLASS DOC] Tests the helper computing the number of nodes an engine spans.
    def test_an_engine_that_fits_inside_one_node_still_occupies_a_whole_node(self):
        # [FUNCTION DOC] Engines of 1 and 8 GPUs on 8-GPU nodes both span 1 node.
        """Plain integer division would report zero nodes for every engine smaller than a node."""
        assert compute_nodes_per_engine(num_gpus_per_engine=1, num_gpus_per_node=8) == 1
        assert compute_nodes_per_engine(num_gpus_per_engine=8, num_gpus_per_node=8) == 1

    def test_an_engine_spanning_several_nodes_reports_how_many_it_spans(self):
        # [FUNCTION DOC] An engine of 32 GPUs on 8-GPU nodes spans 4 nodes.
        """A multi-node engine is launched once per node it covers, so the count must scale with it."""
        assert compute_nodes_per_engine(num_gpus_per_engine=32, num_gpus_per_node=8) == 4


class TestTheApiClientOfACell:
    # [CLASS DOC] Tests the engine api client a cell builds during init.
    async def test_carries_the_engine_api_key(self, stub_provider) -> None:
        # [FUNCTION DOC] A cell with api key a-key and a provider returning only a primary address
        # is initialized for real (no network stubs); its api client then holds that api key.
        """A protected /server_info answers 401 without it, so the engine env would never be recorded."""
        cell = _make_cell(
            stub_provider(dict(primary=HostAndPort(host="10.0.0.1", port=30000))), sglang_api_key="a-key"
        )
        await cell.init()

        assert cell.api_client.api_key == "a-key"
