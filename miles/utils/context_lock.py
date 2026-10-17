# [MODULE DOC] An asynchronous mutual-exclusion lock whose ownership is tied to the current
# asyncio task context (a context variable) rather than to a task object, plus decorators that
# enforce a locking discipline on methods of a class.
# Key ideas: each acquisition gets a generation number so stale grants are detected; only one
# context lock may be held per context at a time; a lock can be "detached" (held but owned by no
# context) by one method and "reattached" by another, allowing a lock to be held across calls.
# Decorators: with_lock (hold the lock for the whole call), acquires_lock (take the lock and leave
# it held, detached, after return), releases_lock (reattach and release at the end),
# requires_lock (assert the caller already holds it), lock_exempt (mark as intentionally
# unlocked). The class decorator enforce_lock_discipline asserts every method of a class carries
# one of these markers. Classes using the decorators must expose a lock object under the
# attribute name context_lock. Misuse raises assertion errors.
import asyncio
import contextlib
import contextvars
import functools
import inspect
import logging
import time
from collections.abc import AsyncIterator, Callable
from types import TracebackType
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

WAIT_LOG_INTERVAL_SECONDS: float = 5.0
LOCK_ATTRIBUTE_NAME: str = "context_lock"

_DISCIPLINE_MARKER_ATTRIBUTE_NAME: str = "_context_lock_discipline"
_DISCIPLINE_ENFORCED_ATTRIBUTE_NAME: str = "_context_lock_discipline_enforced"
_DISCIPLINE_OWNER_ATTRIBUTE_NAME: str = "_context_lock_discipline_owner"

# the annotation machinery (PEP 649) plants these in the class dict; they are not methods of the class
_ANNOTATION_MEMBER_NAMES: frozenset[str] = frozenset({"__annotate__", "__annotate_func__"})


class _LockGrant(NamedTuple):
    # [CLASS DOC] HELPER: Record stored in the context variable naming the lock a context holds
    # and the generation number of that acquisition.
    lock: "ContextLock"
    generation: int


_held_lock: contextvars.ContextVar["_LockGrant | None"] = contextvars.ContextVar("held_context_lock", default=None)


class ContextLock:
    # [CLASS DOC] Named async lock wrapping a standard asyncio lock. Ownership is recorded in a
    # module-level context variable as a grant (this lock plus a generation number); the lock is
    # "held in the current context" only if that grant matches the lock's active generation. A
    # context may hold at most one such lock at a time. Not reentrant. Usable as an async context
    # manager. Logs an info message every 5 seconds while waiting to acquire.
    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Creates the underlying lock, remembers the name (used in messages), and
        # starts with generation counter zero, no active generation, and not detached.
        self._name = name
        self._lock = asyncio.Lock()
        self._generation_counter: int = 0
        self._active_generation: int | None = None
        self._detached = False

    @property
    def name(self) -> str:
        # [FUNCTION DOC] Property: the lock's name given at construction.
        return self._name

    @property
    def locked(self) -> bool:
        # [FUNCTION DOC] Property: whether the underlying lock is currently locked by anyone
        # (including a detached holder).
        return self._lock.locked()

    @property
    def held_in_current_context(self) -> bool:
        # [FUNCTION DOC] Property: true only if the lock has an active generation and the current
        # context's grant names this lock with exactly that generation. False when unlocked,
        # detached, or held by another context.
        if self._active_generation is None:
            return False
        grant = _held_lock.get()
        return grant is not None and grant.lock is self and grant.generation == self._active_generation

    async def __aenter__(self) -> "ContextLock":
        # [FUNCTION DOC] Acquires the lock (see acquire) and returns the lock itself.
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Releases the lock on exit. On normal exit a strict release (asserts the
        # lock is held by this context). On exceptional exit a lenient release: releases if held,
        # otherwise logs an error and skips, so the original exception is not masked by an
        # assertion. Never suppresses the exception.
        if exc_type is None:
            self.release()
        else:
            self._release_if_held()

    async def acquire(self) -> None:
        # [FUNCTION DOC] Asserts the current context holds no live context lock (any, including
        # this one). Starts a background reminder task that logs progress while waiting, awaits the
        # underlying lock, and always cancels the reminder afterwards. On success sets the
        # context's grant to this lock with a freshly issued generation number (marking it
        # active). Cancellation while waiting leaves the lock unacquired.
        self._assert_no_live_grant("acquire")
        wait_reminder = asyncio.ensure_future(self._remind_while_waiting())
        try:
            await self._lock.acquire()
        finally:
            wait_reminder.cancel()
        _held_lock.set(_LockGrant(lock=self, generation=self._issue_generation()))

    def release(self) -> None:
        # [FUNCTION DOC] Asserts this context holds the lock, clears the active generation and the
        # context's grant, then releases the underlying lock.
        self._assert_held_in_current_context()
        self._active_generation = None
        _held_lock.set(None)
        self._lock.release()

    def _release_if_held(self) -> None:
        # [FUNCTION DOC] HELPER: Releases if this context holds the lock; otherwise logs an error
        # that the lock is not held by this context and does nothing (no exception).
        if self.held_in_current_context:
            self.release()
        else:
            logger.error(f"Lock {self._name!r} is not held by the current context on exception exit; skipping release")

    @contextlib.asynccontextmanager
    async def with_released(self) -> AsyncIterator[None]:
        # [FUNCTION DOC] Async context manager that temporarily gives up the lock. Releases it
        # (asserting it is held), yields, and on exit (even after an exception) waits to acquire it
        # again, so the body of the surrounding block resumes holding the lock with a new
        # generation. The reacquire may itself assert if the context now holds another lock.
        self.release()
        try:
            yield
        finally:
            await self.acquire()

    def detach(self) -> None:
        # [FUNCTION DOC] Hands the held lock off without releasing it. Asserts this context holds
        # it; marks the lock detached, clears the active generation and the context's grant. The
        # underlying lock stays locked until some context reattaches and releases it.
        self._assert_held_in_current_context()
        self._detached = True
        self._active_generation = None
        _held_lock.set(None)

    def reattach(self) -> None:
        # [FUNCTION DOC] Takes ownership of a detached lock in the current context. Asserts the
        # lock was detached and is still locked, and that the context holds no live context lock;
        # then clears the detached flag and sets the context grant with a new generation.
        assert self._detached, f"Cannot reattach lock {self._name!r}: it was not detached"
        assert self._lock.locked(), f"Cannot reattach lock {self._name!r}: it is no longer locked"
        self._assert_no_live_grant("reattach")
        self._detached = False
        _held_lock.set(_LockGrant(lock=self, generation=self._issue_generation()))

    def _issue_generation(self) -> int:
        # [FUNCTION DOC] HELPER: Increments the generation counter, records the new value as the
        # active generation and returns it.
        self._generation_counter += 1
        self._active_generation = self._generation_counter
        return self._generation_counter

    def _assert_no_live_grant(self, action: str) -> None:
        # [FUNCTION DOC] HELPER: Asserts the current context has no grant, or its grant's lock is
        # not currently held in this context (a stale grant is fine). The failure message names the
        # attempted action and this lock.
        grant = _held_lock.get()
        assert (
            grant is None or not grant.lock.held_in_current_context
        ), f"Cannot {action} lock {self._name!r}: a context lock is already held"

    def _assert_held_in_current_context(self) -> None:
        # [FUNCTION DOC] HELPER: Asserts this lock is held in the current context.
        assert self.held_in_current_context, f"Lock {self._name!r} must be held by the current context"

    async def _remind_while_waiting(self) -> None:
        # [FUNCTION DOC] HELPER: Runs forever until cancelled; every 5 seconds logs at info level
        # how many whole seconds the caller has waited for this lock. Meant to run as a background
        # task alongside acquire.
        wait_start_time = time.monotonic()
        while True:
            await asyncio.sleep(WAIT_LOG_INTERVAL_SECONDS)
            logger.info(f"Still waiting for lock {self._name!r} after {time.monotonic() - wait_start_time:.0f}s")


def enforce_lock_discipline(cls: type) -> type:
    # [FUNCTION DOC] Class decorator. Walks the class's own attributes (skipping the annotation
    # bookkeeping entries); for each function reachable from a plain function, static method,
    # class method or property accessor, asserts it carries a discipline marker (from with_lock,
    # acquires_lock, releases_lock, requires_lock or lock_exempt), naming the offending member
    # otherwise. Records the class as an owner on each function (and the functions it wraps) so
    # decorated wrappers can check the receiver later, then sets an enforced flag on the class.
    # Returns the same class. Inherited members are not checked.
    for member_name, member in vars(cls).items():
        if member_name in _ANNOTATION_MEMBER_NAMES:
            continue
        for fn in _extract_checkable_functions(member):
            assert getattr(fn, _DISCIPLINE_MARKER_ATTRIBUTE_NAME, None) is not None, (
                f"{cls.__name__}.{member_name} must be decorated with one of the context-lock decorators "
                f"(e.g. with_lock or lock_exempt)"
            )
            _propagate_discipline_owner(fn=fn, owner=cls)
    setattr(cls, _DISCIPLINE_ENFORCED_ATTRIBUTE_NAME, True)
    return cls


def with_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Method decorator for async methods (asserts at decoration time that the
    # function is a coroutine function). The returned wrapper holds the instance's lock for the
    # whole call and returns the method's result. Marked as with_lock.
    assert inspect.iscoroutinefunction(fn), f"{fn.__qualname__} must be async to use with_lock"

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Asserts the receiver belongs to a class enforcing discipline, then runs
        # the method inside an async context on the instance's lock (acquire, call, release; the
        # lock is looked up from the instance attribute context_lock and must be a context lock).
        _assert_owner_enforces_discipline(wrapper, self)
        async with _get_lock(self):
            return await fn(self, *args, **kwargs)

    return _mark(wrapper, "with_lock")


def acquires_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Method decorator for async methods that take the instance's lock and leave it
    # held after returning (asserts coroutine function at decoration time). Marked as
    # acquires_lock. Pair with a releases_lock method called later.
    assert inspect.iscoroutinefunction(fn), f"{fn.__qualname__} must be async to use acquires_lock"

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Checks the receiver, acquires the instance's lock, awaits the method. If
        # the method raises anything (including cancellation), releases the lock if still held and
        # re-raises. On success detaches the lock (still locked, owned by no context) and returns
        # the method's result.
        _assert_owner_enforces_discipline(wrapper, self)
        lock = _get_lock(self)
        await lock.acquire()
        try:
            result = await fn(self, *args, **kwargs)
        except BaseException:
            lock._release_if_held()
            raise
        lock.detach()
        return result

    return _mark(wrapper, "acquires_lock")


def releases_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Method decorator for async methods that finish a hold begun by an
    # acquires_lock method (asserts coroutine function at decoration time). Marked as
    # releases_lock.
    assert inspect.iscoroutinefunction(fn), f"{fn.__qualname__} must be async to use releases_lock"

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Checks the receiver, reattaches the detached lock to the current context
        # (asserts it was detached and still locked), runs the method with the lock held, and in
        # a final step releases the lock even if the method raised. Returns the method's result.
        _assert_owner_enforces_discipline(wrapper, self)
        lock = _get_lock(self)
        lock.reattach()
        try:
            return await fn(self, *args, **kwargs)
        finally:
            lock.release()

    return _mark(wrapper, "releases_lock")


def requires_lock(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Method decorator (sync or async, chosen by whether the function is a
    # coroutine function) that asserts at call time that the caller's context already holds the
    # instance's lock, then calls the method unchanged and returns its result. Never acquires the
    # lock. Marked as requires_lock.
    def assert_precondition(self: Any) -> None:
        # [FUNCTION DOC] HELPER: Checks the receiver belongs to a discipline-enforcing owner class
        # and that the current context holds the instance's lock, using the wrapper chosen below
        # for the owner lookup. Raises assertion errors otherwise.
        _assert_owner_enforces_discipline(wrapper, self)
        _assert_own_lock_held(fn, self)

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] Async variant: runs the precondition check, then awaits the method
            # and returns its result.
            assert_precondition(self)
            return await fn(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] Sync variant: runs the precondition check, then calls the method and
            # returns its result.
            assert_precondition(self)
            return fn(self, *args, **kwargs)

    return _mark(wrapper, "requires_lock")


def lock_exempt(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Decorator that only marks a function as deliberately exempt from locking
    # (satisfying the class discipline check) and returns the same function unwrapped.
    return _mark(fn, "lock_exempt")


def _extract_checkable_functions(member: Any) -> list[Callable[..., Any]]:
    # [FUNCTION DOC] HELPER: Returns the functions to check for a class member: the underlying
    # function of a static or class method; the getter, setter and deleter that exist for a
    # property; the member itself if a plain function; otherwise an empty list.
    if isinstance(member, (staticmethod, classmethod)):
        return [member.__func__]
    if isinstance(member, property):
        return [accessor for accessor in (member.fget, member.fset, member.fdel) if accessor is not None]
    if inspect.isfunction(member):
        return [member]
    return []


def _propagate_discipline_owner(fn: Callable[..., Any], owner: type) -> None:
    # [FUNCTION DOC] HELPER: Follows the chain of wrapped functions starting at fn (stopping on a
    # repeat or when there is none). For each plain function in the chain (unwrapping bound
    # methods), adds the owner class to a set stored on the function under a private attribute,
    # creating the set if absent. Mutates function attributes only.
    target: Any = fn
    seen: set[int] = set()
    while target is not None and id(target) not in seen:
        seen.add(id(target))
        function = target.__func__ if inspect.ismethod(target) else target
        if inspect.isfunction(function):
            owners = getattr(function, _DISCIPLINE_OWNER_ATTRIBUTE_NAME, None)
            if owners is None:
                setattr(function, _DISCIPLINE_OWNER_ATTRIBUTE_NAME, {owner})
            else:
                owners.add(owner)
        target = getattr(target, "__wrapped__", None)


def _get_lock(obj: Any) -> ContextLock:
    # [FUNCTION DOC] HELPER: Reads the context_lock attribute from the object (raises the usual
    # missing-attribute error if absent) and asserts it is a context lock; returns it.
    lock = getattr(obj, LOCK_ATTRIBUTE_NAME)
    assert isinstance(lock, ContextLock), f"{type(obj).__name__}.{LOCK_ATTRIBUTE_NAME} must be a ContextLock"
    return lock


def _mark(fn: Callable[..., Any], discipline: str) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Stores the discipline name on the function under the private marker
    # attribute and returns the function.
    setattr(fn, _DISCIPLINE_MARKER_ATTRIBUTE_NAME, discipline)
    return fn


def _assert_own_lock_held(fn: Callable[..., Any], obj: Any) -> None:
    # [FUNCTION DOC] HELPER: Asserts the object's context lock is held by the current context; the
    # failure message names the function and the lock.
    lock = _get_lock(obj)
    assert (
        lock.held_in_current_context
    ), f"{fn.__qualname__} must be called with the {lock.name!r} context lock held by the current context"


def _assert_owner_enforces_discipline(wrapper: Callable[..., Any], obj: Any) -> None:
    # [FUNCTION DOC] HELPER: Reads the owner-class set recorded on the wrapper. Asserts it exists
    # (the class used the enforcement decorator) and that the receiver belongs to at least one
    # recorded owner, with messages naming the wrapper and owners.
    owners = getattr(wrapper, _DISCIPLINE_OWNER_ATTRIBUTE_NAME, None)
    assert owners is not None, (
        f"{wrapper.__qualname__} uses a context-lock decorator, but its class is not decorated "
        f"with @enforce_lock_discipline"
    )
    assert any(_receiver_belongs_to(obj, owner) for owner in owners), (
        f"{wrapper.__qualname__} is guarded on behalf of {sorted(owner.__name__ for owner in owners)}, "
        f"but was called on a {type(obj).__name__}"
    )


def _receiver_belongs_to(obj: Any, owner: type) -> bool:
    # [FUNCTION DOC] HELPER: If the receiver is itself a class, true when it is a subclass of the
    # owner; otherwise true when it is an instance of the owner.
    if isinstance(obj, type):
        return issubclass(obj, owner)
    return isinstance(obj, owner)
