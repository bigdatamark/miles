# [MODULE DOC] HELPER: Process-global registry for a fake pod api used by kubernetes provider
# tests. Tests install a fake api object (module-level list), code under test obtains it through
# the installed async context manager (which records each use in the close-calls list when the
# block exits), and reset clears both lists. Not thread safe; state is shared across tests until
# reset is called.
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

_INSTALLED: list[Any] = []
CLOSE_CALLS: list[Any] = []


def install(api: object) -> object:
    # [FUNCTION DOC] HELPER: Appends the api to the installed list (the most recent wins) and
    # returns it.
    _INSTALLED.append(api)
    return api


def current() -> Any:
    # [FUNCTION DOC] HELPER: Returns the most recently installed api; fails an assertion if none
    # was installed.
    assert _INSTALLED, "no fake pod api was installed"
    return _INSTALLED[-1]


@asynccontextmanager
async def installed() -> AsyncIterator[Any]:
    # [FUNCTION DOC] HELPER: Async context manager yielding the most recently installed api (fails
    # an assertion if none, so a test cannot reach a real cluster). When the block exits, normally
    # or by exception, the api is appended to the close-calls list.
    assert _INSTALLED, "no fake pod api was installed, so this test would talk to a real cluster"

    api = _INSTALLED[-1]
    try:
        yield api
    finally:
        CLOSE_CALLS.append(api)


def reset() -> None:
    # [FUNCTION DOC] HELPER: Empties both the installed list and the close-calls list.
    _INSTALLED.clear()
    CLOSE_CALLS.clear()
