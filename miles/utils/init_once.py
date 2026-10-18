# [MODULE DOC]
# One-shot initialization guard. A small state machine (not inited, initializing, inited, init
# failed) object and a decorator that wraps an init method (sync or async) so it can run at most
# once per object; a second call, even after failure, asserts.
import contextlib
import enum
import functools
import inspect
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

_T = TypeVar("_T", bound=Callable[..., Any])


class InitState(enum.Enum):
    # [CLASS DOC] States of the guard with text values not_inited, initializing, inited, init_failed.
    NOT_INITED = "not_inited"
    INITIALIZING = "initializing"
    INITED = "inited"
    INIT_FAILED = "init_failed"


class InitOnce:
    # [CLASS DOC] State holder for one-time initialization of an owner (not thread safe).
    def __init__(self, debug_owner_name: str) -> None:
        # [FUNCTION DOC] Stores the owner name for error messages; starts in the not-inited state.
        self._debug_owner_name = debug_owner_name
        self._state = InitState.NOT_INITED

    @property
    def state(self) -> InitState:
        # [FUNCTION DOC] Property: the current state.
        return self._state

    def is_initialized(self) -> bool:
        # [FUNCTION DOC] True only in the inited state (false while initializing or after failure).
        return self._state is InitState.INITED

    @contextlib.contextmanager
    def guarding(self) -> Iterator[None]:
        # [FUNCTION DOC] Context manager around the initialization body. Entering asserts the state is
        # not-inited and moves to initializing. A normal exit moves to inited; any exception (even
        # system exit or cancellation) moves to init-failed and is re-raised. Yields nothing.
        self._enter()
        try:
            yield
        except BaseException:
            self._leave(succeeded=False)
            raise
        self._leave(succeeded=True)

    def _enter(self) -> None:
        # [FUNCTION DOC] HELPER: Asserts not-inited (message names the owner and current state) and
        # sets initializing.
        assert self._state is InitState.NOT_INITED, (
            f"{self._debug_owner_name} already ran init in this process and is now {self._state.value}, so this is "
            f"a stale worker being reused as a fresh one"
        )
        self._state = InitState.INITIALIZING

    def _leave(self, *, succeeded: bool) -> None:
        # [FUNCTION DOC] HELPER: Asserts initializing; sets inited on success else init-failed.
        assert self._state is InitState.INITIALIZING
        self._state = InitState.INITED if succeeded else InitState.INIT_FAILED


def init_once(fn: _T) -> _T:
    # [FUNCTION DOC] Decorator for methods. Returns a wrapper (async for coroutine functions, sync
    # otherwise, preserving metadata) that runs the method inside the guard of the instance's
    # init-once attribute (the instance must have one) and returns its result.
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Async wrapper; awaits the method inside the guard.
            with self._init_once.guarding():
                return await fn(self, *args, **kwargs)

        return async_guarded  # type: ignore[return-value]

    @functools.wraps(fn)
    def guarded(self: Any, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Sync wrapper; calls the method inside the guard.
        with self._init_once.guarding():
            return fn(self, *args, **kwargs)

    return guarded  # type: ignore[return-value]
