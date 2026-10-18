# [MODULE DOC]
# Unit tests for the router manager: resolving each model's router address through its own worker
# provider (concurrently), recording the result on the arguments exactly once, rejecting an
# externally configured router, waiting for a router's TCP port after looking up its address,
# and the session server readiness wait (concurrent address lookup and probing, disabled and
# invalid configurations, published instance lists with external addresses, host consistency
# checks, failure propagation, and a longer budget than routers get). TCP waits are replaced by
# recording or refusing stubs and providers are small fakes defined inside each test.
from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args

import miles.ray.rollout.router_manager as router_manager
from miles.ray.rollout.router_manager import resolve_router_addrs, wait_router_ready, wait_session_server_ready
from miles.rollout.session.types import SessionServerInstance
from miles.utils.workers.worker_spec import HostAndPort, NamedHostAndPorts

_TWO_MODEL_CONFIG = """\
sglang:
  - name: actor
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 2
  - name: ref
    model_path: /fake/ref-model
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 4
"""


def _make_two_model_args(tmp_path: Path) -> Namespace:
    # [FUNCTION DOC] HELPER: Writes the two-model YAML config (actor with 8 GPUs at 2 per engine,
    # frozen ref with 4 GPUs) into the temporary directory and returns test arguments pointing at
    # it with 12 rollout GPUs, 8 GPUs per node and no router ip, port or per-model routers set.
    config_path = tmp_path / "sglang_config.yaml"
    config_path.write_text(_TWO_MODEL_CONFIG)
    return make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=12,
        num_gpus_per_node=8,
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_model_routers=None,
    )


_ROUTER_PROVIDERS = [object()]


class TestResolveRouterAddrs:
    # [CLASS DOC] Tests resolving router addresses for all models of a run, one provider per model.
    async def test_all_model_routers_are_waited_concurrently(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Two models, each with a fake provider whose address lookup signals that it
        # started and then blocks on a release event. Replaces the TCP wait with a recorder. Starts
        # resolution as a task, waits (one second) until both lookups started (proving
        # concurrency), then releases both and awaits the task. Asserts the TCP probes were
        # recorded for both routers in model order (10.0.0.9 on ports 30000 and 30001).
        """All model router waits start before either router becomes ready."""
        args = _make_two_model_args(tmp_path)
        started = [asyncio.Event(), asyncio.Event()]
        released = [asyncio.Event(), asyncio.Event()]
        probed: list[tuple[str, int]] = []

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider for one model; its address lookup signals started,
            # waits for the release event, and returns a primary address with port 30000 plus the
            # model index.
            def __init__(self, model_idx: int) -> None:
                # [FUNCTION DOC] HELPER: Stores the model index.
                self.model_idx = model_idx

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Signals started, waits for release and returns the address.
                started[self.model_idx].set()
                await released[self.model_idx].wait()
                return {"primary": HostAndPort(host="10.0.0.9", port=30000 + self.model_idx)}

        async def _probe(host: str, port: int, *, timeout: float) -> None:
            # [FUNCTION DOC] HELPER: Replacement TCP wait; records the host and port.
            probed.append((host, port))

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", _probe)
        task = asyncio.create_task(resolve_router_addrs(args, router_providers=[_FakeProvider(0), _FakeProvider(1)]))

        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), timeout=1.0)
        finally:
            for event in released:
                event.set()
            await task

        assert probed == [("10.0.0.9", 30000), ("10.0.0.9", 30001)]

    async def test_records_every_models_router_on_args(self, monkeypatch):
        # [FUNCTION DOC] Single default model with the router wait replaced by a fake returning
        # 10.0.0.9 port 30000 plus the model index. Resolution returns a mapping from default to
        # that address and writes the primary router ip and port and the per-model router mapping
        # onto the arguments.
        """The driver-visible router contract (primary ip/port, per-model map) is written exactly once, here."""
        args = make_args(sglang_router_ip=None, sglang_router_port=None, sglang_model_routers=None)

        async def _fake_wait_router_ready(*, model_idx: int, provider) -> HostAndPort:
            # [FUNCTION DOC] HELPER: Returns a router address with port 30000 plus the model index.
            return HostAndPort(host="10.0.0.9", port=30000 + model_idx)

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_router_ready", _fake_wait_router_ready)

        router_addrs = await resolve_router_addrs(args, router_providers=_ROUTER_PROVIDERS)

        assert router_addrs == {"default": HostAndPort(host="10.0.0.9", port=30000)}
        assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.9", 30000)
        assert args.sglang_model_routers == {"default": ("10.0.0.9", 30000)}

    async def test_resolving_again_in_the_same_process_answers_from_the_record(self, monkeypatch):
        # [FUNCTION DOC] Resolves twice with the same arguments and a recording router wait. The
        # second result equals the first and the wait ran only once, for model index 0.
        """The driver and an in-process controller may both resolve; the second call must not re-wait."""
        args = make_args(sglang_router_ip=None, sglang_router_port=None, sglang_model_routers=None)
        waited: list[int] = []

        async def _fake_wait_router_ready(*, model_idx: int, provider) -> HostAndPort:
            # [FUNCTION DOC] HELPER: Records the model index and returns a router address with port
            # 30000 plus the model index.
            waited.append(model_idx)
            return HostAndPort(host="10.0.0.9", port=30000 + model_idx)

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_router_ready", _fake_wait_router_ready)

        first = await resolve_router_addrs(args, router_providers=_ROUTER_PROVIDERS)
        second = await resolve_router_addrs(args, router_providers=_ROUTER_PROVIDERS)

        assert second == first
        assert waited == [0]

    async def test_every_model_gets_its_own_router_and_model_zero_is_the_primary(self, monkeypatch, tmp_path: Path):
        # [FUNCTION DOC] Two models with two distinct provider objects and a recording router wait.
        # Asserts the wait ran for model indexes 0 then 1 with the respective providers, the result
        # maps actor to port 30000 and ref to port 30001, the per-model router mapping on the
        # arguments matches, and the primary router ip and port are the first model's.
        """Each model has its own router, and the driver-wide ip/port must be the first model's, not the last."""
        args = _make_two_model_args(tmp_path)
        waited: list[int] = []

        providers: list[object] = []
        two_model_providers = [object(), object()]

        async def _fake_wait_router_ready(*, model_idx: int, provider) -> HostAndPort:
            # [FUNCTION DOC] HELPER: Records the model index and provider and returns a router
            # address with port 30000 plus the model index.
            waited.append(model_idx)
            providers.append(provider)
            return HostAndPort(host="10.0.0.9", port=30000 + model_idx)

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_router_ready", _fake_wait_router_ready)

        router_addrs = await resolve_router_addrs(args, router_providers=two_model_providers)

        assert waited == [0, 1]
        assert providers == two_model_providers
        assert router_addrs == {
            "actor": HostAndPort(host="10.0.0.9", port=30000),
            "ref": HostAndPort(host="10.0.0.9", port=30001),
        }
        assert args.sglang_model_routers == {"actor": ("10.0.0.9", 30000), "ref": ("10.0.0.9", 30001)}
        assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.9", 30000)

    async def test_an_externally_configured_router_is_rejected(self):
        # [FUNCTION DOC] Arguments with a preset router ip and port: resolution raises an assertion
        # error saying external router mode was removed.
        """External router mode was removed, so a pre-set router address means a misconfigured run."""
        args = make_args(sglang_router_ip="10.0.0.1", sglang_router_port=3000, sglang_model_routers=None)

        with pytest.raises(AssertionError, match="external router mode was removed"):
            await resolve_router_addrs(args, router_providers=_ROUTER_PROVIDERS)


class TestRouterProvidersPerModel:
    # [CLASS DOC] Tests the requirement of one router provider per model.
    async def test_a_multi_model_run_needs_one_provider_per_model(self, tmp_path: Path):
        # [FUNCTION DOC] Two-model arguments with only one provider: resolution raises an assertion
        # error saying each model needs its own provider.
        """One provider answers for exactly one pool, so reusing model zero's would look up the wrong router."""
        args = _make_two_model_args(tmp_path)

        with pytest.raises(AssertionError, match="its own provider"):
            await resolve_router_addrs(args, router_providers=_ROUTER_PROVIDERS)


def _recording_probe(waited: list[tuple[str, int]]) -> Callable[..., Coroutine[Any, Any, None]]:
    # [FUNCTION DOC] HELPER: Returns a replacement TCP wait coroutine function that appends each
    # (host, port) it is called with to the given list and otherwise returns immediately.
    async def _probe(host: str, port: int, timeout: float) -> None:
        # [FUNCTION DOC] HELPER: Records the host and port; ignores the timeout.
        waited.append((host, port))

    return _probe


class TestWaitRouterReady:
    # [CLASS DOC] Tests the wait-for-router-ready function: look up the router address from the
    # provider by the spec worker name, then wait on its TCP port.
    async def test_returns_the_provider_addr_after_the_tcp_wait(self, monkeypatch):
        # [FUNCTION DOC] Waits for model index 1 with a provider that records the requested worker
        # name and returns 10.0.0.9:12345, with a recording TCP wait. Asserts the requested worker
        # name is inference-router-1-00000-00000, the TCP wait saw 10.0.0.9:12345 and the returned
        # address equals it.
        """The router address is looked up from the worker manager by the spec worker name."""
        requested: list[str] = []

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider recording requested worker names.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Records the worker name and returns a fixed primary address.
                requested.append(worker_name)
                return {"primary": HostAndPort(host="10.0.0.9", port=12345)}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        addr = await wait_router_ready(model_idx=1, provider=_FakeProvider())

        assert requested == ["inference-router-1-00000-00000"]
        assert waited == [("10.0.0.9", 12345)]
        assert addr == HostAndPort(host="10.0.0.9", port=12345)

    async def test_an_unreachable_router_port_fails_instead_of_returning_an_address(self, monkeypatch):
        # [FUNCTION DOC] The TCP wait raises a runtime error (server not ready); waiting for the
        # router propagates that error, whose message names 10.0.0.9:12345.
        """A router whose port never opens must fail startup rather than be reported ready."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider returning a fixed primary address.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Returns primary 10.0.0.9:12345.
                return {"primary": HostAndPort(host="10.0.0.9", port=12345)}

        async def _refuse(host: str, port: int, *, timeout: float) -> None:
            # [FUNCTION DOC] HELPER: Replacement TCP wait that always raises a runtime error saying
            # the server at the host and port is not ready after the timeout.
            raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", _refuse)

        with pytest.raises(RuntimeError, match="10.0.0.9:12345 not ready"):
            await wait_router_ready(model_idx=1, provider=_FakeProvider())

    async def test_a_failed_router_addr_lookup_fails_before_any_tcp_wait(self, monkeypatch):
        # [FUNCTION DOC] The provider's lookup raises a runtime error "router worker is not
        # registered"; waiting for the router propagates it and no TCP wait was recorded.
        """A router the worker manager cannot resolve must abort startup, not be probed anyway."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider whose lookup always fails.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Always raises a runtime error.
                raise RuntimeError("router worker is not registered")

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        with pytest.raises(RuntimeError, match="not registered"):
            await wait_router_ready(model_idx=1, provider=_FakeProvider())
        assert waited == []


class TestWaitSessionServerReady:
    # [CLASS DOC] Tests the session server readiness wait, which looks up each session server
    # worker's address, probes its TCP port, and publishes the instance list on the arguments.
    async def test_session_server_address_resolution_and_readiness_are_concurrent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Two session server workers. Address lookups block until released and TCP
        # probes block until released. Starts the wait as a task, waits (one second) until both
        # lookups started, releases the lookups, waits until both probes started, then releases
        # everything and awaits the task (so lookups, and then probes, are shown to be concurrent).
        """All address lookups and then all readiness probes start concurrently."""
        lookup_started = [asyncio.Event(), asyncio.Event()]
        lookups_released = asyncio.Event()
        probe_started = [asyncio.Event(), asyncio.Event()]
        probes_released = asyncio.Event()

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider that derives the worker index from the third dash
            # separated part of the worker name, signals the lookup start, waits for release and
            # returns a primary address with port 5005 plus the index.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: See the class description.
                index = int(worker_name.split("-")[2])
                lookup_started[index].set()
                await lookups_released.wait()
                return {"primary": HostAndPort(host="10.0.0.9", port=5005 + index)}

        async def _probe(host: str, port: int, *, timeout: float) -> None:
            # [FUNCTION DOC] HELPER: Replacement TCP wait; derives the index from the port (minus
            # 5005), signals the probe start and waits for the release event.
            index = port - 5005
            probe_started[index].set()
            await probes_released.wait()

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", _probe)
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
            run_uuid="00112233445566aa",
        )
        task = asyncio.create_task(wait_session_server_ready(args, provider=_FakeProvider()))

        try:
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in lookup_started)), timeout=1.0)
            lookups_released.set()
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in probe_started)), timeout=1.0)
        finally:
            lookups_released.set()
            probes_released.set()
            await task

    async def test_disabled_session_server_does_not_create_a_provider_or_publish_addresses(self, monkeypatch):
        # [FUNCTION DOC] With the session server disabled, the wait returns without error using a
        # provider that fails the test if used, and the arguments get no session server instances
        # attribute.
        """Disabling the session server publishes no addr / instance-id fields and resolves no addrs."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider that fails the test if any address is requested.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Always raises an assertion error.
                raise AssertionError("the disabled branch must not resolve any addrs")

        args = make_args(use_session_server=False)
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert not hasattr(args, "session_server_instances")

    async def test_enabled_without_hf_checkpoint_raises(self):
        # [FUNCTION DOC] Session server enabled with no hf checkpoint (and no provider): the wait
        # raises a value error mentioning hf-checkpoint.
        """Enabling the session server without a tokenizer source fails fast."""
        args = make_args(use_session_server=True, hf_checkpoint=None)
        with pytest.raises(ValueError, match="hf-checkpoint"):
            await wait_session_server_ready(args, provider=None)

    @pytest.mark.parametrize("workers", [0, -1])
    async def test_a_non_positive_worker_count_is_rejected(self, workers):
        # [FUNCTION DOC] Parametrized over worker counts 0 and -1: the wait raises a value error
        # mentioning session-server-workers before any address lookup.
        """A zero count published an empty address list, so the run only failed once a session was requested."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider that fails the test if any address is requested.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Always raises an assertion error.
                raise AssertionError("a non-positive worker count must not resolve any addrs")

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", session_server_workers=workers)
        with pytest.raises(ValueError, match="session-server-workers"):
            await wait_session_server_ready(args, provider=_FakeProvider())

    async def test_publishes_the_manager_addrs_and_instance_ids(self, monkeypatch):
        # [FUNCTION DOC] Two session server workers on one host with consecutive ports and run uuid
        # 00112233445566aa. Asserts the requested worker names are session-server-00000-00000 and
        # session-server-00001-00000; the published instances are 10.0.0.9:5005 and :5006 with
        # instance ids the run uuid plus -0 and -1; and the TCP waits saw those two addresses in order.
        """The driver-side contract (ip, ports, instance ids) comes from the worker manager addrs."""
        requested: list[str] = []

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider recording worker names and handing out consecutive
            # ports starting at 5005 on one host.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Records the name; returns port 5004 plus the request count.
                requested.append(worker_name)
                return {"primary": HostAndPort(host="10.0.0.9", port=5004 + len(requested))}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
            run_uuid="00112233445566aa",
        )
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert requested == ["session-server-00000-00000", "session-server-00001-00000"]
        assert args.session_server_instances == [
            SessionServerInstance(addr="10.0.0.9:5005", instance_id="00112233445566aa-0"),
            SessionServerInstance(addr="10.0.0.9:5006", instance_id="00112233445566aa-1"),
        ]
        assert waited == [("10.0.0.9", 5005), ("10.0.0.9", 5006)]

    async def test_servers_on_different_hosts_are_each_addressed_in_full(self, monkeypatch):
        # [FUNCTION DOC] Two session servers on different hosts (10.0.0.1 and 10.0.0.2, same port
        # 5005). Asserts the published instances carry each full host and port with instance ids
        # -0 and -1, and the TCP waits probed each full address.
        """Placement may spread the servers across hosts, so no instance may be addressed by a
        port under a host borrowed from another one."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider giving each successive lookup a new host.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts a lookup counter at zero.
                self._counter = 0

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Increments the counter and returns host 10.0.0.N, port 5005.
                self._counter += 1
                return {"primary": HostAndPort(host=f"10.0.0.{self._counter}", port=5005)}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
            run_uuid="00112233445566aa",
        )
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert args.session_server_instances == [
            SessionServerInstance(addr="10.0.0.1:5005", instance_id="00112233445566aa-0"),
            SessionServerInstance(addr="10.0.0.2:5005", instance_id="00112233445566aa-1"),
        ]
        assert waited == [("10.0.0.1", 5005), ("10.0.0.2", 5005)]

    async def test_instances_carry_each_host_external_address(self, monkeypatch):
        # [FUNCTION DOC] Two session servers whose addresses carry distinct external hosts
        # (100.64.0.1 and 100.64.0.2). Asserts each published instance has its own external address
        # as well as its cluster address, and that readiness probing uses the cluster addresses.
        """Each instance keeps its own node's external address; a shared one would send agents to the wrong instance."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider giving each successive lookup a new host and a
            # matching distinct external host.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts a lookup counter at zero.
                self._counter = 0

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Increments the counter and returns host 10.0.0.N, port 5005
                # and external host 100.64.0.N.
                self._counter += 1
                return {
                    "primary": HostAndPort(
                        host=f"10.0.0.{self._counter}",
                        port=5005,
                        external_host=f"100.64.0.{self._counter}",
                    )
                }

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
            run_uuid="00112233445566aa",
        )
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert args.session_server_instances == [
            SessionServerInstance(
                addr="10.0.0.1:5005", external_addr="100.64.0.1:5005", instance_id="00112233445566aa-0"
            ),
            SessionServerInstance(
                addr="10.0.0.2:5005", external_addr="100.64.0.2:5005", instance_id="00112233445566aa-1"
            ),
        ]
        # Readiness is the driver's own probe, so it runs against the cluster addresses.
        assert waited == [("10.0.0.1", 5005), ("10.0.0.2", 5005)]

    async def test_a_shared_external_host_publishes_every_instance_there(self, monkeypatch):
        # [FUNCTION DOC] Two session servers on one host with ports 5005 and 5006 and the argument
        # for a shared external host 100.64.0.1. Asserts the published external addresses are
        # 100.64.0.1:5005 and 100.64.0.1:5006 and the readiness probes use the cluster host.
        """--session-server-external-host names every instance's external address; the driver keeps its own."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider returning one host with incrementing ports.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts the port counter at 5004.
                self._port = 5004

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Increments the port and returns host 10.0.0.9 with it.
                self._port += 1
                return {"primary": HostAndPort(host="10.0.0.9", port=self._port)}

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
            run_uuid="00112233445566aa",
            session_server_external_host="100.64.0.1",
        )
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert [instance.external_addr for instance in args.session_server_instances] == [
            "100.64.0.1:5005",
            "100.64.0.1:5006",
        ]
        assert waited == [("10.0.0.9", 5005), ("10.0.0.9", 5006)]

    async def test_hosts_sharing_one_external_host_fail_before_any_readiness_wait(self, monkeypatch):
        # [FUNCTION DOC] Two session servers on different cluster hosts but reporting the same
        # external host: the wait raises a value error telling to set the node external ip variable
        # on each node, and no TCP wait was recorded.
        """Instances on different hosts published at one external host would send agents to the wrong one."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider returning successive hosts that all claim the same
            # external host 100.64.0.1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts a lookup counter at zero.
                self._counter = 0

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Increments the counter; returns host 10.0.0.N, port
                # 5004 plus N, and the shared external host.
                self._counter += 1
                return {
                    "primary": HostAndPort(
                        host=f"10.0.0.{self._counter}", port=5004 + self._counter, external_host="100.64.0.1"
                    )
                }

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=2,
            run_uuid="00112233445566aa",
        )
        with pytest.raises(ValueError, match="MILES_NODE_EXTERNAL_IP on each node"):
            await wait_session_server_ready(args, provider=_FakeProvider())

        assert waited == []

    async def test_one_unreachable_instance_fails_the_whole_readiness_wait(self, monkeypatch):
        # [FUNCTION DOC] Two session servers on ports 5005 and 5006; the TCP wait raises a runtime
        # error only for port 5006. The wait propagates that error, whose message names
        # 10.0.0.9:5006.
        """A single session server whose port never opens fails startup even if its siblings are up."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider returning one host with incrementing ports.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Starts a lookup counter at zero.
                self._counter = 0

            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Increments the counter; returns host 10.0.0.9, port 5004
                # plus the counter.
                self._counter += 1
                return {"primary": HostAndPort(host="10.0.0.9", port=5004 + self._counter)}

        async def _refuse_one(host: str, port: int, *, timeout: float) -> None:
            # [FUNCTION DOC] HELPER: Replacement TCP wait that raises a runtime error (server not
            # ready after the timeout) only for port 5006 and succeeds otherwise.
            if port == 5006:
                raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", _refuse_one)

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", session_server_workers=2)
        with pytest.raises(RuntimeError, match="10.0.0.9:5006 not ready"):
            await wait_session_server_ready(args, provider=_FakeProvider())

    async def test_a_failed_instance_addr_lookup_fails_before_any_tcp_wait(self, monkeypatch):
        # [FUNCTION DOC] The provider's lookup raises "session-server worker is not registered"; the
        # wait propagates a runtime error with that text and no TCP wait was recorded.
        """A session server the worker manager cannot resolve aborts startup before any TCP probe."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider whose lookup always fails.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Always raises a runtime error.
                raise RuntimeError("session-server worker is not registered")

        waited: list[tuple[str, int]] = []
        monkeypatch.setattr(
            "miles.ray.rollout.router_manager.wait_tcp_ready_async",
            _recording_probe(waited),
        )

        args = make_args(use_session_server=True, hf_checkpoint="/fake/model", session_server_workers=2)
        with pytest.raises(RuntimeError, match="not registered"):
            await wait_session_server_ready(args, provider=_FakeProvider())
        assert waited == []


class TestTheWaitCoversWhatEachServerDoesBeforeItBinds:
    # [CLASS DOC] Tests the TCP wait budgets: session servers get a longer budget than routers.
    async def test_a_session_server_is_given_longer_than_a_router(self, monkeypatch):
        # [FUNCTION DOC] Records the timeout passed to the TCP wait. Waits for one router, then for
        # one session server (same fake provider returning 10.0.0.9:5004). Asserts the session
        # server wait used exactly the module's session server ready timeout constant and that it is
        # larger than the router's budget.
        """A router binds its port first thing. A session server imports transformers and loads the
        tokenizer and chat template first, and it is scheduled by the platform before any of that, so
        holding it to the router's budget reports a server that is merely still starting as unreachable."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider returning a fixed primary address.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Returns primary 10.0.0.9:5004.
                return {"primary": HostAndPort(host="10.0.0.9", port=5004)}

        budgets: list[float] = []

        async def record_budget(host, port, timeout) -> None:
            # [FUNCTION DOC] HELPER: Replacement TCP wait that records the timeout budget.
            budgets.append(timeout)

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", record_budget)

        await wait_router_ready(model_idx=0, provider=_FakeProvider())
        router_budget = budgets.pop()

        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=1,
            run_uuid="00112233445566aa",
        )
        await wait_session_server_ready(args, provider=_FakeProvider())

        assert budgets == [router_manager._SESSION_SERVER_READY_TIMEOUT_SECONDS]
        assert budgets[0] > router_budget

    async def test_a_session_server_that_never_binds_still_fails(self, monkeypatch):
        # [FUNCTION DOC] The TCP wait raises a runtime error (not ready after the timeout); the
        # session server wait propagates it.
        # KNOWN ISSUE: the replacement TCP wait is a plain synchronous function; it raises when
        # called, so the failure happens at call time instead of while awaiting, unlike the real
        # asynchronous function.
        """The budget is there to cover a slow start, not to wait out one that is never coming."""

        class _FakeProvider:
            # [CLASS DOC] HELPER: Fake provider returning a fixed primary address.
            async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
                # [FUNCTION DOC] HELPER: Returns primary 10.0.0.9:5004.
                return {"primary": HostAndPort(host="10.0.0.9", port=5004)}

        def _refuse(host: str, port: int, timeout: float) -> None:
            # [FUNCTION DOC] HELPER: Replacement TCP wait that always raises a runtime error saying
            # the server is not ready after the timeout.
            raise RuntimeError(f"Server at {host}:{port} not ready after {timeout}s")

        monkeypatch.setattr("miles.ray.rollout.router_manager.wait_tcp_ready_async", _refuse)
        args = make_args(
            use_session_server=True,
            hf_checkpoint="/fake/model",
            session_server_workers=1,
            run_uuid="00112233445566aa",
        )

        with pytest.raises(RuntimeError, match="not ready after"):
            await wait_session_server_ready(args, provider=_FakeProvider())
