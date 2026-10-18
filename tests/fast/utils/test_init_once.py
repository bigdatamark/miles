# [MODULE DOC]
# Tests for the init-once guard: a per-component state holder (not initialized, initializing,
# initialized, init failed) plus a decorator that wraps a component's init method (sync or async) so
# it can run at most once. Verified: state reporting before, during, after and after a failed init;
# a second init is refused with an assertion error mentioning a stale worker (and naming the state
# found, e.g. init-failed or initializing); an overlapping async init is refused while the first
# still completes; and cancelling an in-flight async init marks it failed and blocks reuse. The
# module marks all tests as asyncio-enabled and defines three small component classes as fixtures.
import asyncio

import pytest

from miles.utils.init_once import InitOnce, InitState, init_once

pytestmark = pytest.mark.asyncio


class _Component:
    # [CLASS DOC] HELPER: Synchronous component whose init method is guarded by the init-once
    # decorator. Counts how many times the init body ran.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create the guard (named after the class) and a zero run counter.
        self._init_once = InitOnce(type(self).__name__)
        self.runs = 0

    @init_once
    def init(self, *, fail: bool = False) -> str:
        # [FUNCTION DOC] HELPER: Guarded init. Increments the run counter, raises a value error
        # ("init blew up") if asked to fail, otherwise returns the text "built".
        self.runs += 1
        if fail:
            raise ValueError("init blew up")
        return "built"

    def is_initialized(self) -> bool:
        # [FUNCTION DOC] HELPER: Return whether the guard reports the initialized state.
        return self._init_once.is_initialized()


class _AsyncComponent:
    # [CLASS DOC] HELPER: Same as the synchronous component but with a guarded async init method.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create the guard (named after the class) and a zero run counter.
        self._init_once = InitOnce(type(self).__name__)
        self.runs = 0

    @init_once
    async def init(self, *, fail: bool = False) -> str:
        # [FUNCTION DOC] HELPER: Guarded async init. Increments the run counter, raises a value
        # error ("init blew up") if asked to fail, otherwise returns the text "built".
        self.runs += 1
        if fail:
            raise ValueError("init blew up")
        return "built"

    def is_initialized(self) -> bool:
        # [FUNCTION DOC] HELPER: Return whether the guard reports the initialized state.
        return self._init_once.is_initialized()


class _BlockingAsyncComponent:
    # [CLASS DOC] HELPER: Component whose guarded async init signals entry through one event and then
    # blocks until a second event is set, to let tests interleave other calls or cancellation.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Create the guard, the entered and release events, and a zero run
        # counter.
        self._init_once = InitOnce(type(self).__name__)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.runs = 0

    @init_once
    async def init(self) -> str:
        # [FUNCTION DOC] HELPER: Guarded async init. Count the run, set the entered event, wait for
        # the release event, then return the text "built".
        self.runs += 1
        self.entered.set()
        await self.release.wait()
        return "built"


class TestTheStateAComponentReports:
    # [CLASS DOC] Tests of the state exposed by the guard in the different phases of init.
    def test_a_component_that_never_ran_init_is_not_initialized(self):
        # [FUNCTION DOC] A freshly constructed component reports not initialized.
        """A component only answers yes once it finished building, never because it was constructed."""
        assert _Component().is_initialized() is False

    def test_a_component_that_finished_init_is_initialized(self):
        # [FUNCTION DOC] After a successful init the component reports initialized.
        """This is the answer a take-over reads to decide between building and resuming."""
        component = _Component()

        component.init()

        assert component.is_initialized() is True

    def test_a_component_whose_init_raised_is_not_initialized(self):
        # [FUNCTION DOC] A failing init propagates its value error; afterwards the component is not
        # initialized and the guard's state is init-failed.
        """A half-built component taken over as a built one would run with a fleet nobody finished wiring."""
        component = _Component()

        with pytest.raises(ValueError):
            component.init(fail=True)

        assert component.is_initialized() is False
        assert component._init_once.state is InitState.INIT_FAILED

    def test_a_component_inside_its_own_init_is_not_initialized(self):
        # [FUNCTION DOC] Inside the guard's context-manager scope, the guard reports not
        # initialized and its state is initializing.
        """The window between entering init and finishing it must never read as built."""
        guard = InitOnce("Component")

        with guard.guarding():
            assert guard.is_initialized() is False
            assert guard.state is InitState.INITIALIZING


class TestInitRunsExactlyOnce:
    # [CLASS DOC] Tests that a second init attempt is refused in synchronous components.
    def test_a_second_init_is_refused(self):
        # [FUNCTION DOC] After a successful init, a second init raises an assertion error mentioning
        # a stale worker, and the body ran only once.
        """Re-initializing a live component would throw away the state it is holding."""
        component = _Component()
        component.init()

        with pytest.raises(AssertionError, match="stale worker"):
            component.init()

        assert component.runs == 1

    def test_a_second_init_is_refused_after_the_first_one_raised(self):
        # [FUNCTION DOC] After a failing first init, a second init still raises the stale-worker
        # assertion error, and the body ran only once.
        """Nothing can rebuild a process that already ran half of an init over its own state."""
        component = _Component()
        with pytest.raises(ValueError):
            component.init(fail=True)

        with pytest.raises(AssertionError, match="stale worker"):
            component.init()

        assert component.runs == 1

    def test_the_refusal_names_the_state_it_found(self):
        # [FUNCTION DOC] After a failing init, the refusal of a second init has a message that
        # names the init-failed state.
        """An operator reading the crash has to be able to tell a finished init from a failed one."""
        component = _Component()
        with pytest.raises(ValueError):
            component.init(fail=True)

        with pytest.raises(AssertionError, match="init_failed"):
            component.init()


class TestTheGuardOnAsyncInit:
    # [CLASS DOC] Tests of the guard on async init methods, including overlap and cancellation.
    async def test_an_async_init_that_finished_reports_initialized(self):
        # [FUNCTION DOC] Awaiting the async init returns "built" and the component then reports
        # initialized.
        """Three of the four components this guards await their init."""
        component = _AsyncComponent()

        assert await component.init() == "built"
        assert component.is_initialized() is True

    async def test_an_async_init_that_raised_reports_uninitialized(self):
        # [FUNCTION DOC] A failing awaited init propagates the value error and leaves the component
        # not initialized.
        """The failure path has to hold for the awaited form too."""
        component = _AsyncComponent()

        with pytest.raises(ValueError):
            await component.init(fail=True)

        assert component.is_initialized() is False

    async def test_a_second_async_init_is_refused(self):
        # [FUNCTION DOC] After one completed async init, a second awaited init raises the
        # stale-worker assertion error.
        """The exactly-once refusal is what catches an operator driving a live fleet by hand."""
        component = _AsyncComponent()
        await component.init()

        with pytest.raises(AssertionError, match="stale worker"):
            await component.init()

    async def test_an_overlapping_async_init_is_refused_while_the_first_finishes(self) -> None:
        # [FUNCTION DOC] Start a blocking init as a task and wait until it has entered; release it
        # and immediately try a second init, which must raise an assertion error mentioning
        # initializing. Afterwards the body ran once, the first task returns "built" and the state
        # is initialized.
        """An overlapping init is refused while the original invocation can still finish."""
        component = _BlockingAsyncComponent()
        first_init: asyncio.Task[str] = asyncio.create_task(component.init())
        await component.entered.wait()
        component.release.set()

        with pytest.raises(AssertionError, match="initializing"):
            await component.init()

        assert component.runs == 1
        assert await first_init == "built"
        assert component._init_once.state is InitState.INITED

    async def test_cancelling_async_init_marks_it_failed_and_prevents_reuse(self) -> None:
        # [FUNCTION DOC] Start a blocking init as a task, wait until it entered, cancel it and await
        # it (expecting cancellation). Assert the guard state is init-failed, a later init is
        # refused with an assertion error mentioning init-failed, and the body ran only once.
        """Cancelling an in-flight init marks the component failed and permanently refuses reuse."""
        component = _BlockingAsyncComponent()
        init_task: asyncio.Task[str] = asyncio.create_task(component.init())
        await component.entered.wait()

        init_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await init_task

        assert component._init_once.state is InitState.INIT_FAILED
        with pytest.raises(AssertionError, match="init_failed"):
            await component.init()
        assert component.runs == 1
