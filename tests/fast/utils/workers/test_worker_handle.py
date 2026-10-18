# [MODULE DOC] Tests for worker handles: the abstract base handle contract (wait_ready and the
# death probe are abstract; wait_idle defaults to raising not-implemented), the unreachable-worker
# error, the ray handle (relays method calls to actor methods and maps actor death to the
# unreachable error; readiness probe; wait_idle unsupported; wait_dead probing loop with a
# per-probe budget, retry on inconclusive results, a deadline that logs an error and returns) and
# the rpc handle's wait_dead on an unreachable address. Ray remote methods are replaced by fake
# objects returning coroutines built from small factories; time and sleeping are monkeypatched
# in the deadline tests.
import asyncio
import logging
from types import SimpleNamespace

import pytest
import ray

from miles.utils.workers import worker_handle as worker_handle_module
from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError


class _ProbeableWorker:
    # [CLASS DOC] HELPER: Minimal worker class with one method, used to construct an rpc handle.
    def demo(self) -> int:
        # [FUNCTION DOC] HELPER: Returns 1.
        return 1


class TestBaseWorkerHandle:
    # [CLASS DOC] Tests of the abstract handle contract.
    def test_rpc_handle_implements_the_contract(self):
        # [FUNCTION DOC] The rpc handle class is a subclass of the base handle.
        """The rpc client is a worker handle, so callers can hold the base type."""
        assert issubclass(RpcWorkerHandle, BaseWorkerHandle)

    def test_ray_handle_implements_the_contract(self):
        # [FUNCTION DOC] The ray handle class is a subclass of the base handle.
        """The ray wrapper is a worker handle, so callers can hold the base type."""
        assert issubclass(RayWorkerHandle, BaseWorkerHandle)

    def test_a_handle_that_cannot_be_waited_out_says_so_rather_than_reporting_idle(self):
        # [FUNCTION DOC] Defines a subclass implementing only wait_ready and the death probe (not
        # the idle wait); running the default wait_idle with a 1 second timeout raises a
        # not-implemented error whose message mentions "running a call".
        """Callers read this answer off the base type, and a silent pass would report a busy worker idle."""

        class Minimal(BaseWorkerHandle):
            # [CLASS DOC] HELPER: Minimal concrete handle: no-op readiness wait and a death probe
            # that reports dead.
            async def wait_ready(self, *, timeout: float) -> None: ...

            async def probe_is_dead(self) -> bool:
                # [FUNCTION DOC] HELPER: Always reports the worker dead.
                return True

        with pytest.raises(NotImplementedError, match="running a call"):
            asyncio.run(Minimal().wait_idle(timeout=1.0))

    def test_incomplete_implementation_rejected(self):
        # [FUNCTION DOC] A subclass with no methods cannot be instantiated (type error).
        """A handle that implements neither wait_ready nor the death probe cannot be instantiated."""

        class Incomplete(BaseWorkerHandle):
            # [CLASS DOC] HELPER: Subclass with no implementations.
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_a_handle_implementing_only_wait_ready_is_rejected(self):
        # [FUNCTION DOC] A subclass with only the readiness wait cannot be instantiated (type
        # error) because the abstract death probe is missing.
        # KNOWN ISSUE: the docstring says wait_dead is required, but in the base class wait_dead is
        # concrete and the abstract method is the death probe.
        """wait_dead is required on its own, so a readiness-only handle cannot be instantiated."""

        class ReadyOnly(BaseWorkerHandle):
            # [CLASS DOC] HELPER: Subclass implementing only the readiness wait.
            async def wait_ready(self, *, timeout: float) -> None: ...

        with pytest.raises(TypeError):
            ReadyOnly()

    def test_a_handle_implementing_only_wait_dead_is_rejected(self):
        # [FUNCTION DOC] A subclass with only an overridden death wait cannot be instantiated (type
        # error): the abstract readiness wait and death probe are both missing.
        """wait_ready is required on its own, so a death-only handle cannot be instantiated."""

        class DeadOnly(BaseWorkerHandle):
            # [CLASS DOC] HELPER: Subclass implementing only the death wait.
            async def wait_dead(self, *, timeout: float) -> None: ...

        with pytest.raises(TypeError):
            DeadOnly()


class TestWorkerUnreachableError:
    # [CLASS DOC] Tests of the unreachable-worker error type.
    def test_is_plain_exception_without_submission_state(self) -> None:
        # [FUNCTION DOC] Creates the error with message "boom"; its text is "boom" and it has no
        # submitted attribute.
        """The error carries only its message and standard exception state."""
        error = WorkerUnreachableError("boom")

        assert str(error) == "boom"
        assert not hasattr(error, "submitted")


class _FakeRemoteMethod:
    # [CLASS DOC] HELPER: Fake ray remote method. Holds a list of factories producing awaitables;
    # each remote call records its arguments and uses the next factory (the last one repeats).
    def __init__(self, coro_factories):
        self._coro_factories = list(coro_factories)
        self.call_count = 0
        self.args_seen: list[tuple] = []
        self.kwargs_seen: list[dict] = []

    def remote(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Records the positional and keyword arguments, increments the call
        # count and returns the awaitable made by the factory for this call.
        self.args_seen.append(args)
        self.kwargs_seen.append(kwargs)
        self.call_count += 1
        index = min(self.call_count - 1, len(self._coro_factories) - 1)
        return self._coro_factories[index]()


def _return_factory(value):
    # [FUNCTION DOC] HELPER: Returns a factory of coroutines that resolve to the value.
    async def _coro():
        # [FUNCTION DOC] HELPER: Resolves to the captured value.
        return value

    return _coro


def _raise_factory(exc):
    # [FUNCTION DOC] HELPER: Returns a factory of coroutines that raise the given exception object.
    async def _coro():
        # [FUNCTION DOC] HELPER: Raises the captured exception.
        raise exc

    return _coro


def _hang_factory():
    # [FUNCTION DOC] HELPER: Returns an awaitable that sleeps one hour (a hung call).
    return asyncio.sleep(3600)


def _never_resolving_factory():
    # [FUNCTION DOC] HELPER: Returns a fresh unresolved future of the running loop; must be called
    # inside a running event loop.
    return asyncio.get_running_loop().create_future()


def _ray_actor_error():
    # [FUNCTION DOC] HELPER: Returns a new ray actor error (worker process gone).
    return ray.exceptions.RayActorError()


def _ray_task_error():
    # [FUNCTION DOC] HELPER: Returns a ray task error instance created without running its
    # constructor.
    return ray.exceptions.RayTaskError.__new__(ray.exceptions.RayTaskError)


def _ray_actor_unavailable_error():
    # [FUNCTION DOC] HELPER: Returns a ray actor-unavailable error (actor may still be alive) with
    # a connection-reset message.
    return ray.exceptions.ActorUnavailableError("gcs connection reset", None)


def _make_handle(**methods) -> tuple[RayWorkerHandle, SimpleNamespace]:
    # [FUNCTION DOC] HELPER: Builds a plain namespace whose attributes are the given fake methods
    # and wraps it in a ray worker handle. Returns the handle and the namespace.
    inner = SimpleNamespace(**methods)
    return RayWorkerHandle(inner), inner


def _make_monotonic(values):
    # [FUNCTION DOC] HELPER: Returns a fake monotonic clock that yields the given values one per
    # call and then keeps returning the last one.
    seq = list(values)
    state = {"i": 0}

    def _monotonic():
        # [FUNCTION DOC] HELPER: Returns the next scripted time value (the last repeats).
        i = state["i"]
        if i < len(seq):
            state["i"] += 1
            return seq[i]
        return seq[-1]

    return _monotonic


@pytest.mark.asyncio
class TestRayWorkerHandleDispatch:
    # [CLASS DOC] Tests of the ray handle's generic method relay.
    async def test_forwards_kwargs_and_returns_the_result(self):
        # [FUNCTION DOC] Calling echo with a keyword argument returns the fake result 7 and the
        # fake method saw exactly that keyword argument.
        """The handle relays a keyword-only call to the underlying actor method."""
        handle, inner = _make_handle(echo=_FakeRemoteMethod([_return_factory(7)]))

        result = await handle.echo(value=7)

        assert result == 7
        assert inner.echo.kwargs_seen == [{"value": 7}]

    async def test_forwards_positional_args_unchanged(self):
        # [FUNCTION DOC] A positional plus a keyword argument reach the fake method unchanged and
        # the result is returned.
        """Callers may pass positionally, exactly as they would on the in-process object."""
        handle, inner = _make_handle(echo=_FakeRemoteMethod([_return_factory(7)]))

        result = await handle.echo(7, flag=True)

        assert result == 7
        assert inner.echo.args_seen == [(7,)]
        assert inner.echo.kwargs_seen == [{"flag": True}]

    async def test_actor_death_is_reported_as_unreachable(self):
        # [FUNCTION DOC] A ray actor error from the call is converted to the unreachable-worker
        # error.
        """RayActorError means the worker process is gone, not that the call was bad."""
        handle, _inner = _make_handle(echo=_FakeRemoteMethod([_raise_factory(_ray_actor_error())]))

        with pytest.raises(WorkerUnreachableError):
            await handle.echo(value=1)

    async def test_application_errors_propagate_unchanged(self):
        # [FUNCTION DOC] A value error raised by the fake method reaches the caller as a value error
        # with the same message.
        """A failure inside the worker method must reach the caller as-is."""
        handle, _inner = _make_handle(train=_FakeRemoteMethod([_raise_factory(ValueError("boom"))]))

        with pytest.raises(ValueError, match="boom"):
            await handle.train(rollout_id=1)

    async def test_dunder_lookup_raises_attribute_error(self):
        # [FUNCTION DOC] Reading a dunder-style attribute from the handle raises an attribute error
        # instead of producing a remote call.
        """Serialization probes dunder attributes, which must not become remote calls."""
        handle, _inner = _make_handle()

        with pytest.raises(AttributeError):
            _ = handle.__custom_dunder__


@pytest.mark.asyncio
class TestRayWorkerHandleWaitReady:
    # [CLASS DOC] Tests of the ray handle's readiness wait, which calls the actor's ray-ready
    # method.
    async def test_returns_when_the_probe_succeeds(self):
        # [FUNCTION DOC] A succeeding probe makes wait_ready return after exactly one probe call.
        """A constructed actor answers the readiness probe immediately."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_return_factory(None)]))

        await handle.wait_ready(timeout=1.0)

        assert inner.__ray_ready__.call_count == 1

    async def test_actor_death_is_reported_as_unreachable(self):
        # [FUNCTION DOC] An actor error from the probe makes wait_ready raise the unreachable-worker
        # error.
        """A worker that died during startup can never become ready."""
        handle, _inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_actor_error())]))

        with pytest.raises(WorkerUnreachableError):
            await handle.wait_ready(timeout=1.0)

    async def test_a_hung_probe_times_out_as_unreachable(self):
        # [FUNCTION DOC] A probe that hangs with a 0.01 second timeout raises the unreachable-worker
        # error.
        """A worker that never answers within the deadline is unreachable."""
        handle, _inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_hang_factory]))

        with pytest.raises(WorkerUnreachableError):
            await handle.wait_ready(timeout=0.01)

    async def test_allowing_the_uuid_to_change_is_accepted_and_changes_nothing(self):
        # [FUNCTION DOC] Passing the allow-server-uuid-change flag is accepted and still results in
        # a single probe call.
        """Every run takes this path at startup, and refusing the flag aborted the ray backend there."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_return_factory(None)]))

        await handle.wait_ready(timeout=1.0, allow_server_uuid_change=True)

        assert inner.__ray_ready__.call_count == 1


@pytest.mark.asyncio
class TestRayWorkerHandleWaitIdle:
    # [CLASS DOC] Tests of the ray handle's idle wait.
    async def test_a_ray_actor_cannot_be_waited_out(self):
        # [FUNCTION DOC] wait_idle raises a not-implemented error whose message mentions the rpc
        # communication backend.
        """A ray actor tracks no calls, so this backend has to fail loudly rather than report a worker idle."""
        handle, _inner = _make_handle()

        with pytest.raises(NotImplementedError, match="rpc communication backend"):
            await handle.wait_idle(timeout=1.0)


@pytest.mark.asyncio
class TestRayWorkerHandleWaitDead:
    # [CLASS DOC] Tests of the ray handle's death wait: it repeatedly probes the ray-ready method;
    # an actor error or task error confirms death; an unavailable error or probe timeout is
    # inconclusive and is retried after a sleep; if the deadline passes it logs an error
    # ("Timed out after <n>s waiting for ...") and returns.
    async def test_returns_immediately_when_actor_error_on_first_probe(self):
        # [FUNCTION DOC] Actor error on the first probe: returns after one probe.
        """A dead actor whose first probe raises RayActorError is confirmed dead after one probe."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_actor_error())]))

        await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 1

    async def test_returns_immediately_when_task_error_on_first_probe(self):
        # [FUNCTION DOC] Task error on the first probe: returns after one probe.
        """RayTaskError on the first probe is also treated as confirmed actor death."""
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_task_error())]))

        await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 1

    async def test_a_temporarily_unavailable_actor_is_not_confirmed_dead(self, monkeypatch):
        # [FUNCTION DOC] Probe interval shortened to 0.01 seconds; first probe raises
        # actor-unavailable, second raises actor error. wait_dead (guarded by a 10 second outer
        # timeout) returns after exactly 2 probes.
        """ActorUnavailableError means the actor may still be alive, so the probe is inconclusive and retried."""
        monkeypatch.setattr(worker_handle_module, "_WAIT_DEAD_PROBE_INTERVAL_SECONDS", 0.01)
        handle, inner = _make_handle(
            __ray_ready__=_FakeRemoteMethod(
                [_raise_factory(_ray_actor_unavailable_error()), _raise_factory(_ray_actor_error())]
            )
        )

        await asyncio.wait_for(handle.wait_dead(timeout=120.0), timeout=10.0)

        assert inner.__ray_ready__.call_count == 2

    async def test_a_persistently_unavailable_actor_only_gives_up_at_the_deadline(self, monkeypatch, caplog):
        # [FUNCTION DOC] Sleep is a recorder and the clock yields 0, 1, 2, 200 seconds; the probe
        # always raises actor-unavailable. With a 120 second timeout, asserts 3 probes happened and
        # the error log contains "Timed out".
        """Unavailability alone never confirms death; the loop keeps probing until the deadline expires."""
        slept: list[float] = []

        async def _noop_sleep(seconds: float) -> None:
            # [FUNCTION DOC] HELPER: Records the requested sleep and returns immediately.
            slept.append(seconds)

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(
            worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 1.0, 2.0, 200.0]))
        )
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(_ray_actor_unavailable_error())]))

        with caplog.at_level(logging.ERROR, logger=worker_handle_module.__name__):
            await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 3
        assert "Timed out" in caplog.text

    async def test_a_hung_probe_is_timed_out_and_retried(self, monkeypatch):
        # [FUNCTION DOC] Probe interval 0.01 seconds; the first probe never resolves, the second
        # raises actor error. wait_dead (10 second outer guard) returns after 2 probes.
        """A probe that never answers is abandoned on the probe budget, and the next probe confirms death."""
        monkeypatch.setattr(worker_handle_module, "_WAIT_DEAD_PROBE_INTERVAL_SECONDS", 0.01)
        handle, inner = _make_handle(
            __ray_ready__=_FakeRemoteMethod([_never_resolving_factory, _raise_factory(_ray_actor_error())])
        )

        await asyncio.wait_for(handle.wait_dead(timeout=120.0), timeout=10.0)

        assert inner.__ray_ready__.call_count == 2

    async def test_retries_after_timeout_then_confirms_death(self, monkeypatch):
        # [FUNCTION DOC] First probe raises a timeout, second an actor error; clock 0 then 1. After
        # wait_dead, 2 probes happened and exactly one sleep of 1.0 second was requested.
        """A probe timeout is tolerated; the loop retries and confirms death on the next probe."""
        slept = []

        async def _noop_sleep(seconds):
            # [FUNCTION DOC] HELPER: Records the requested sleep and returns immediately.
            slept.append(seconds)

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 1.0])))
        handle, inner = _make_handle(
            __ray_ready__=_FakeRemoteMethod(
                [
                    _raise_factory(asyncio.TimeoutError()),
                    _raise_factory(_ray_actor_error()),
                ]
            )
        )

        await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 2
        assert slept == [1.0]

    async def test_a_live_worker_is_never_confirmed_dead_before_the_deadline(self, monkeypatch, caplog):
        # [FUNCTION DOC] The probe always succeeds; the clock yields 0, 60, 200 seconds with a 120
        # second timeout. Asserts 2 probes, one 1.0 second sleep and exactly one error record
        # containing "Timed out after 120s waiting for".
        """A worker whose readiness probe keeps succeeding is not reported dead until the deadline expires."""
        slept: list[float] = []

        async def _noop_sleep(seconds: float) -> None:
            # [FUNCTION DOC] HELPER: Records the requested sleep and returns immediately.
            slept.append(seconds)

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(
            worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 60.0, 200.0]))
        )
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_return_factory(None)]))

        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.ray_worker_handle"):
            await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 2
        assert slept == [1.0]
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert "Timed out after 120s waiting for" in error_records[0].getMessage()

    async def test_deadline_reached_returns_and_logs_error(self, monkeypatch, caplog):
        # [FUNCTION DOC] The probe raises a timeout and the clock jumps from 0 to 200 seconds with a
        # 120 second timeout. Asserts one probe and exactly one error record containing "Timed out
        # after 120s waiting for".
        """When the timeout deadline is exceeded after a hung probe, it returns and logs an ERROR."""

        async def _noop_sleep(seconds):
            # [FUNCTION DOC] HELPER: Returns immediately without sleeping.
            return None

        monkeypatch.setattr(worker_handle_module.asyncio, "sleep", _noop_sleep)
        monkeypatch.setattr(worker_handle_module, "time", SimpleNamespace(monotonic=_make_monotonic([0.0, 200.0])))
        handle, inner = _make_handle(__ray_ready__=_FakeRemoteMethod([_raise_factory(asyncio.TimeoutError())]))

        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.ray_worker_handle"):
            await handle.wait_dead(timeout=120.0)

        assert inner.__ray_ready__.call_count == 1
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(error_records) == 1
        assert "Timed out after 120s waiting for" in error_records[0].getMessage()


class TestRpcWorkerHandleWaitDead:
    # [CLASS DOC] Tests of the rpc handle's death wait.
    @pytest.mark.asyncio
    async def test_a_server_that_cannot_be_reached_is_confirmed_dead(self):
        # [FUNCTION DOC] An rpc handle pointed at an unused local port (127.0.0.1:1) returns from
        # wait_dead within a 1 second timeout without raising.
        """A cell heals once its ranks are gone, and nothing answering at their address is that proof."""
        handle = RpcWorkerHandle(_ProbeableWorker, server_url="http://127.0.0.1:1")

        await handle.wait_dead(timeout=1.0)
