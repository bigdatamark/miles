# [MODULE DOC] Cross-process limits shared by a rollout process and the agent child processes it
# starts, built on advisory file locks (kernel releases a lock when its holder exits or is killed).
# One directory per rollout "scope" under the system temporary directory holds the lock files; the
# scope name is "pid<rollout pid>" unless inherited through an environment variable, which the
# parent sets for its children. Provides an asynchronous counting semaphore (N named slot files, a
# non-blocking exclusive lock on one slot) and a blocking exclusive lock. Lock file names are
# sanitized to letters, digits, dot, underscore and hyphen.
"""Limits shared by a rollout process and the agent processes it starts.

Under ``--custom-agent-function-mode subproc`` every agent-function call has a
process of its own, so a module-level lock or semaphore limits nothing. These
are ``flock`` locks on files in one directory per rollout process instead: the
kernel releases a slot when its holder exits, including when the holder is killed.
"""

import asyncio
import fcntl
import os
import random
import re
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

# An agent process inherits its rollout process's scope through this variable.
SCOPE_ENV_VAR = "MILES_ROLLOUT_LIMITS_SCOPE"

_POLL_S = 0.2


def current_scope() -> str:
    # [FUNCTION DOC] The scope name: the value of the scope environment variable if set and non-empty
    # (we are a child of a rollout process), else "pid" followed by this process id.
    """Name the rollout process whose limits apply here: this process, or the one that started it."""
    return os.environ.get(SCOPE_ENV_VAR) or f"pid{os.getpid()}"


@asynccontextmanager
async def rollout_semaphore(name: str, limit: int) -> AsyncIterator[None]:
    # [FUNCTION DOC] Async context manager holding one of "limit" named slots across all processes of
    # the scope. Raises value error for a limit below 1 before waiting. Waits (polling, no timeout)
    # until some slot is free, then yields; on exit (including errors or cancellation) closes the
    # descriptor, releasing the slot. Changing the limit between callers of the same name is not
    # coordinated: slots are just files numbered from 0.
    """Hold one of ``limit`` slots named ``name`` for the duration of the block."""
    if limit < 1:
        raise ValueError(f"rollout_semaphore {name!r} needs a positive limit, got {limit}")
    fd = await _acquire_any_slot(name, limit)
    try:
        yield
    finally:
        os.close(fd)  # closing the descriptor releases its lock


@contextmanager
def rollout_lock(name: str) -> Iterator[None]:
    # [FUNCTION DOC] Context manager taking an exclusive lock on the named file, blocking the calling
    # THREAD (not async-friendly) until available, yielding while held, and releasing on exit by
    # closing the descriptor. If acquisition raises, the descriptor is closed too.
    """Hold the lock named ``name``, blocking the calling thread until it is free."""
    fd = _open_lock_file(f"{name}.lock")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


async def _acquire_any_slot(name: str, limit: int) -> int:
    # [FUNCTION DOC] HELPER: Loop forever until a slot lock is won. Each round tries the slots in a
    # fresh random order (avoids everyone contending for slot 0) with a non-blocking exclusive lock; a
    # busy slot's descriptor is closed and skipped; any other error closes the descriptor and
    # propagates. When no slot is free, sleeps asynchronously for 0.2 s scaled by a random factor
    # between 0.5 and 1.5, then retries. Returns the open descriptor of the won slot (caller must
    # close it). Cancellation during the sleep leaves no descriptor open.
    while True:
        # a random order keeps waiters from all contending for slot 0
        for slot in random.sample(range(limit), limit):
            fd = _open_lock_file(f"{name}.{slot}.lock")
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)  # another holder has this slot
                continue
            except BaseException:
                os.close(fd)
                raise
            return fd
        await asyncio.sleep(_POLL_S * (0.5 + random.random()))


def _open_lock_file(file_name: str) -> int:
    # [FUNCTION DOC] HELPER: Open (creating if needed, read-write, owner-only permissions) the lock
    # file in the scope directory, after replacing every character outside letters, digits, dot,
    # underscore and hyphen in the name with an underscore. Returns the raw descriptor.
    return os.open(_lock_dir() / re.sub(r"[^A-Za-z0-9._-]", "_", file_name), os.O_CREAT | os.O_RDWR, 0o600)


def _lock_dir() -> Path:
    # [FUNCTION DOC] HELPER: The per-scope directory "miles-rollout-limits-<scope>" under the system
    # temporary directory, created if missing (single level, no error if present). Returns its path.
    lock_dir = Path(tempfile.gettempdir()) / f"miles-rollout-limits-{current_scope()}"
    lock_dir.mkdir(exist_ok=True)
    return lock_dir
