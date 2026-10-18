# [MODULE DOC] Tests for the static worker provider (workers deployed independently of the
# launcher, addressed by computed or given hosts): construction from a helm release (refuses a
# cell spread across pods) or from explicit rpc addresses (refuses an empty list); address
# answers (host recomputed with the shared naming function, one host per cell index, all declared
# ports, assertion errors for another pool, a nonexistent worker or cell); release and component
# naming stability and truncation; handles (rpc handle bound to the served class, refusal for a
# command-only pool); enumeration unsupported; explicit addresses (per-cell entries, one worker
# per cell); readiness waiting over all addresses concurrently with error propagation; and
# host:port parsing (hostnames, bracketed ipv6, refusing missing port, urls, empty host, bare
# ipv6). Async code is driven with asyncio run or async tests; no network access occurs because
# dialing is patched.
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_provider import static
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.static import StaticWorkerProvider, parse_host_and_port
from miles.utils.workers.worker_spec import CommandWorkerSpec, PortInfo, SchedulingSpec, ServeWorkerSpec

_RELEASE = "miles-run-c0ffee"
_ADDR_POOL_ID = "trainer-controller-actor"


class FakeController:
    # [CLASS DOC] HELPER: Placeholder served worker class referenced by import path.
    def health(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 1.
        return 1


def _served_spec(*, num_cells: int = 2) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a serve spec named trainer-controller with ports primary 7000
    # and rpc 8000, the given number of cells (default 2) of one gpu-less worker, and the fake
    # controller class.
    return ServeWorkerSpec(
        name="trainer-controller",
        port_infos=[PortInfo(name="primary", static_port=7000), PortInfo(name="rpc", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=num_cells, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=f"{__name__}.FakeController",
        ctor_kwargs=lambda context: {},
    )


def _command_spec() -> CommandWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds a command spec named inference-router-0 with a primary port 8000
    # and a single gpu-less worker, launching a python router module.
    return CommandWorkerSpec(
        name="inference-router-0",
        port_infos=[PortInfo(name="primary", static_port=8000)],
        env_var=lambda context: {},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        launch_command=lambda context: "python -m router",
    )


def _provider(spec=None) -> StaticWorkerProvider:
    # [FUNCTION DOC] HELPER: Builds a static provider from the test release and the given spec
    # (default: the served spec).
    return StaticWorkerProvider.of_release(release=_RELEASE, spec=spec or _served_spec())


def _addr_provider(*addrs: str) -> StaticWorkerProvider:
    # [FUNCTION DOC] HELPER: Builds a static provider from explicit rpc addresses (each parsed from
    # host:port text; default one address 10.0.0.1:8000) for the trainer-controller-actor pool with
    # the fake controller class.
    return StaticWorkerProvider.of_rpc_addrs(
        pool_id=_ADDR_POOL_ID,
        addrs=[parse_host_and_port(addr) for addr in addrs or ("10.0.0.1:8000",)],
        worker_class=f"{__name__}.FakeController",
    )


class TestConstruction:
    # [CLASS DOC] Tests of the provider's two factory functions.
    def test_of_release_refuses_a_cell_spread_over_multiple_pods(self):
        # [FUNCTION DOC] A spec with one cell of 16 single-gpu workers on 8-gpu nodes makes the
        # release factory raise an assertion error matching "workers do not share one host".
        """Static cell addresses require every worker in the cell to share one host."""
        spec = _served_spec().model_copy(
            update={
                "scheduling": SchedulingSpec(
                    num_cells=1,
                    num_workers_per_cell=16,
                    num_gpus_per_worker=1,
                    num_gpu_slots_per_worker=1,
                    num_gpus_per_node=8,
                )
            }
        )

        with pytest.raises(AssertionError, match="workers do not share one host"):
            StaticWorkerProvider.of_release(release=_RELEASE, spec=spec)

    def test_of_rpc_addrs_refuses_an_empty_address_list(self):
        # [FUNCTION DOC] An empty address list makes the addresses factory raise an assertion error
        # matching "needs at least one cell".
        """An explicitly addressed pool must contain at least one reachable cell."""
        with pytest.raises(AssertionError, match="needs at least one cell"):
            StaticWorkerProvider.of_rpc_addrs(
                pool_id=_ADDR_POOL_ID,
                addrs=[],
                worker_class=f"{__name__}.FakeController",
            )


class TestAddresses:
    # [CLASS DOC] Tests of address answers for a release-based provider (run to completion with
    # asyncio run).
    def test_recomputes_the_host_the_chart_gave_the_cell(self):
        # [FUNCTION DOC] Worker trainer-controller-00000-00000 has its primary host equal to the
        # shared static-worker host function result for cell 0 and port 7000.
        """The launcher and the pod run the same naming function, so no address has to be shipped between them."""
        addrs = asyncio.run(_provider().get_addrs("trainer-controller-00000-00000"))

        assert addrs["primary"].host == naming.static_worker_host(_RELEASE, "trainer-controller", 0)
        assert addrs["primary"].port == 7000

    def test_each_cell_answers_on_its_own_host(self):
        # [FUNCTION DOC] Cell 1's primary host equals the static-worker host for cell index 1.
        """Every cell of a static pool is its own workload object, so cell index selects the hostname."""
        second = asyncio.run(_provider().get_addrs("trainer-controller-00001-00000"))

        assert second["primary"].host == naming.static_worker_host(_RELEASE, "trainer-controller", 1)

    def test_carries_every_port_the_spec_declares(self):
        # [FUNCTION DOC] The address map has exactly the port names primary and rpc.
        """A caller that asked for a port the spec declares must not have to know its number."""
        addrs = asyncio.run(_provider().get_addrs("trainer-controller-00000-00000"))

        assert sorted(addrs) == ["primary", "rpc"]

    def test_refuses_a_worker_of_another_pool(self):
        # [FUNCTION DOC] A worker name of the router pool raises an assertion error mentioning that
        # name.
        """One provider answers one pool, and guessing across pools would return a plausible wrong host."""
        with pytest.raises(AssertionError, match="inference-router-0-00000-00000"):
            asyncio.run(_provider().get_addrs("inference-router-0-00000-00000"))

    def test_refuses_a_worker_the_cell_never_holds(self):
        # [FUNCTION DOC] Worker index 3 in a one-worker cell raises an assertion error mentioning
        # the worker name.
        """A worker index beyond the cell size would be answered with cell zero's ports."""
        with pytest.raises(AssertionError, match="trainer-controller-00000-00003"):
            asyncio.run(_provider().get_addrs("trainer-controller-00000-00003"))

    def test_refuses_a_cell_the_run_never_deployed(self):
        # [FUNCTION DOC] Cell 7 of a 2-cell pool raises an assertion error mentioning the worker
        # name.
        """A cell beyond the replica count has no workload object, so its host would never resolve."""
        with pytest.raises(AssertionError, match="trainer-controller-00007-00000"):
            asyncio.run(_provider().get_addrs("trainer-controller-00007-00000"))


class TestReleaseNaming:
    # [CLASS DOC] Tests of the shared component naming function applied to release names.
    def test_two_runs_that_share_a_truncated_prefix_get_different_objects(self):
        # [FUNCTION DOC] Two long releases that differ only in their trailing run number produce
        # different component names, each within the component name budget.
        """The trailing digits are exactly what tells two runs of the same experiment apart."""
        long_prefix = "miles-run-a-rather-long-experiment-name-260810-123456"

        first = naming.component_name(f"{long_prefix}-001", "trainer-controller")
        second = naming.component_name(f"{long_prefix}-002", "trainer-controller")

        assert first != second
        assert len(first) <= naming.COMPONENT_NAME_BUDGET

    def test_the_same_release_always_names_the_same_object(self):
        # [FUNCTION DOC] Calling the naming function twice with the same long release and component
        # gives equal names (deterministic).
        """helm upgrade only adopts an object it can find again under the same name."""
        release = "miles-run-a-rather-long-experiment-name-260810-123456-001"

        assert naming.component_name(release, "trainer-controller") == naming.component_name(
            release, "trainer-controller"
        )

    def test_a_release_that_fits_is_left_alone(self):
        # [FUNCTION DOC] A short release gives "<release>-trainer-controller" unchanged.
        """A readable name is worth keeping, and nothing is lost when nothing is truncated."""
        assert naming.component_name(_RELEASE, "trainer-controller") == f"{_RELEASE}-trainer-controller"


class TestHandles:
    # [CLASS DOC] Tests of handle construction.
    def test_calls_a_served_worker_through_its_rpc_port(self):
        # [FUNCTION DOC] The handle is an rpc handle whose worker class name (private attribute) is
        # FakeController.
        """A controller is reached by rpc, and the class it serves is named by its own spec."""
        handle = _provider().get_handle("trainer-controller-00000-00000")

        assert isinstance(handle, RpcWorkerHandle)
        assert handle._worker_cls_name == "FakeController"

    def test_refuses_to_call_a_pool_that_is_only_launched(self):
        # [FUNCTION DOC] A provider over a command spec raises an assertion error matching
        # "launched as a command" when a handle is requested.
        """A command worker declares no worker class, so nothing describes its rpc methods."""
        with pytest.raises(AssertionError, match="launched as a command"):
            _provider(_command_spec()).get_handle("inference-router-0-00000-00000")


class TestEnumeration:
    # [CLASS DOC] Tests of the unsupported worker enumeration.
    def test_refuses_to_enumerate_workers(self):
        # [FUNCTION DOC] Asking for worker infos raises a not-implemented error matching "does not
        # enumerate workers".
        """Nothing observes a static pool, so a caller expecting live workers must fail loudly."""
        with pytest.raises(NotImplementedError, match="does not enumerate workers"):
            _provider().get_worker_infos(cell_ids=["trainer-controller-00000"])


class TestAddressesGivenExplicitly:
    # [CLASS DOC] Tests of a provider built from explicit rpc addresses.
    def test_answers_the_address_it_was_given(self):
        # [FUNCTION DOC] The rpc address of worker 0 of cell 0 is "http://10.0.0.1:8000".
        """The whole point of addressing another deployment is that nothing discovers its address at runtime."""
        addrs = asyncio.run(_addr_provider().get_addrs(f"{_ADDR_POOL_ID}-00000-00000"))

        assert addrs["rpc"].addr == "http://10.0.0.1:8000"

    def test_addresses_each_instance_by_its_own_entry(self):
        # [FUNCTION DOC] Two given addresses: cell 1 resolves to the second ("http://10.0.0.2:9000").
        """A composite controller is given several addresses, and cell one is not cell zero."""
        provider = _addr_provider("10.0.0.1:8000", "10.0.0.2:9000")

        assert (
            asyncio.run(provider.get_addrs(f"{_ADDR_POOL_ID}-00001-00000")).get("rpc").addr == "http://10.0.0.2:9000"
        )

    def test_refuses_a_worker_nobody_named(self):
        # [FUNCTION DOC] Cell 3 with only one address raises an assertion error mentioning the name.
        """Answering an address for an instance that was never given would invent a host out of thin air."""
        with pytest.raises(AssertionError, match=f"{_ADDR_POOL_ID}-00003-00000"):
            asyncio.run(_addr_provider().get_addrs(f"{_ADDR_POOL_ID}-00003-00000"))

    def test_refuses_a_second_worker_in_a_cell(self):
        # [FUNCTION DOC] Worker index 1 of a cell raises an assertion error mentioning the name.
        """A statically addressed controller is one process, so worker one of its cell does not exist."""
        with pytest.raises(AssertionError, match=f"{_ADDR_POOL_ID}-00000-00001"):
            asyncio.run(_addr_provider().get_addrs(f"{_ADDR_POOL_ID}-00000-00001"))

    def test_builds_an_rpc_handle_on_the_given_address(self):
        # [FUNCTION DOC] The handle for worker 0 of cell 0 is an rpc handle.
        """The orchestration script reaches an independently deployed controller only over rpc."""
        assert isinstance(_addr_provider().get_handle(f"{_ADDR_POOL_ID}-00000-00000"), RpcWorkerHandle)


class TestWaitStaticAddrsReady:
    # [CLASS DOC] Tests of waiting until static addresses accept connections.
    async def test_a_failed_address_readiness_check_is_propagated(self) -> None:
        # [FUNCTION DOC] Patches the asyncio connection opener with one that raises a runtime error
        # carrying host and port; waiting on "unavailable:1234" raises that error.
        """A failed address readiness check aborts startup with its dial error."""

        async def _dial(host: str, port: int) -> None:
            # [FUNCTION DOC] HELPER: Always raises a runtime error "Cannot dial <host>:<port>".
            raise RuntimeError(f"Cannot dial {host}:{port}")

        with patch.object(asyncio, "open_connection", _dial):
            with pytest.raises(RuntimeError, match="Cannot dial unavailable:1234"):
                await static.wait_static_addrs_ready([parse_host_and_port("unavailable:1234")])

    async def test_dials_every_address_before_the_run_calls_it(self):
        # [FUNCTION DOC] The module's tcp-ready wait is replaced by a recorder; waiting on a:1 and
        # b:2 dials exactly those two (host, port) pairs.
        """A deployment installed a moment earlier is not yet listening, and the first call would simply fail."""
        dialled: list[tuple[str, int]] = []

        async def _dial(host, port, *, timeout):
            # [FUNCTION DOC] HELPER: Records the host and port; ignores the timeout.
            dialled.append((host, port))

        with patch.object(static, "wait_tcp_ready_async", _dial):
            await static.wait_static_addrs_ready([parse_host_and_port("a:1"), parse_host_and_port("b:2")])

        assert sorted(dialled) == [("a", 1), ("b", 2)]

    async def test_a_slow_address_does_not_hold_up_the_others(self):
        # [FUNCTION DOC] A fake dial sleeping 20 ms for the host "slow" and not at all otherwise;
        # waiting on slow then fast completes with completion order fast, slow (concurrent dials).
        """Waiting serially would multiply one ready budget by the number of deployments a run reaches."""
        order: list[str] = []

        async def _dial(host, port, *, timeout):
            # [FUNCTION DOC] HELPER: Sleeps 0.02 seconds for the host "slow" (otherwise zero), then
            # appends the host to the completion order list.
            await asyncio.sleep(0.02 if host == "slow" else 0)
            order.append(host)

        with patch.object(static, "wait_tcp_ready_async", _dial):
            await static.wait_static_addrs_ready([parse_host_and_port("slow:1"), parse_host_and_port("fast:2")])

        assert order == ["fast", "slow"]


class TestParseHostAndPort:
    # [CLASS DOC] Tests of the host:port text parser.
    @pytest.mark.parametrize(
        ("addr", "expected"),
        [
            ("10.0.0.1:8000", ("10.0.0.1", 8000)),
            ("host.namespace.svc:8000", ("host.namespace.svc", 8000)),
            ("[2001:db8::1]:8000", ("[2001:db8::1]", 8000)),
        ],
    )
    def test_reads_the_forms_a_user_writes_on_the_command_line(self, addr, expected):
        # [FUNCTION DOC] Parametrized over an ipv4 address, a service host name and a bracketed
        # ipv6 address with a port; the parsed host (brackets kept) and integer port equal the
        # expected pair.
        """Every one of these is what somebody copies out of a log line into the next launch."""
        parsed = parse_host_and_port(addr)

        assert (parsed.host, parsed.port) == expected

    def test_an_ipv6_address_survives_being_rebuilt_into_a_url(self):
        # [FUNCTION DOC] The parsed bracketed ipv6 address yields the url "http://[2001:db8::1]:8000".
        """The rpc client dials `.addr`, and an unbracketed ipv6 host makes that url unparseable."""
        assert parse_host_and_port("[2001:db8::1]:8000").addr == "http://[2001:db8::1]:8000"

    def test_refuses_an_address_without_a_port(self):
        # [FUNCTION DOC] "10.0.0.1" raises an assertion error mentioning host:port.
        """A controller is reached at a port, and guessing one would fail much later and much less clearly."""
        with pytest.raises(AssertionError, match="host:port"):
            parse_host_and_port("10.0.0.1")

    @pytest.mark.parametrize("addr", ["http://10.0.0.1:8000", "http://host.namespace.svc:8000/"])
    def test_refuses_an_address_written_as_a_url(self, addr):
        # [FUNCTION DOC] Parametrized over two http urls; each raises an assertion error mentioning
        # host:port.
        """A scheme is not part of an address here, and silently dropping it would hide a wrong flag value."""
        with pytest.raises(AssertionError, match="host:port"):
            parse_host_and_port(addr)

    def test_refuses_an_address_that_names_no_host(self):
        # [FUNCTION DOC] ":8000" raises an assertion error mentioning "no host".
        """A port with no host parses cleanly and then dials whatever the empty host resolves to."""
        with pytest.raises(AssertionError, match="no host"):
            parse_host_and_port(":8000")

    @pytest.mark.parametrize("addr", ["2001:db8::1", "fd00::1"])
    def test_refuses_a_bare_ipv6_address_written_without_brackets(self, addr):
        # [FUNCTION DOC] Parametrized over two unbracketed ipv6 addresses; each raises an assertion
        # error mentioning ipv6.
        """Its own colons parse as the port separator, so it becomes a truncated host and a one-digit port."""
        with pytest.raises(AssertionError, match="ipv6"):
            parse_host_and_port(addr)
