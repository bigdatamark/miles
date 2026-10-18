# [MODULE DOC] HELPER: Fake worker classes and small records shared by the rpc server tests: a worker
# with sync and async methods (including two methods in one named concurrency group), an async-only
# worker, an outcome recorder, and a pair type bundling an executor with its method specs.
from __future__ import annotations

import asyncio
from typing import NamedTuple

from miles.utils.workers.rpc.common.metadata import RpcMethodSpec, rpc
from miles.utils.workers.rpc.common.protocol import CallStatusResponse
from miles.utils.workers.rpc.server.executor import RpcCallExecutor


class SyncAndAsyncWorker:
    # [CLASS DOC] HELPER: Worker mixing sync methods (one plain, one raising, two sharing the "left"
    # concurrency group) and one async method.
    def demo_sync(self, value: int) -> int:
        # [FUNCTION DOC] HELPER: Returns value plus one.
        return value + 1

    def demo_sync_raises(self) -> None:
        # [FUNCTION DOC] HELPER: Always raises a runtime error with the message "boom".
        raise RuntimeError("boom")

    @rpc(concurrency_group="left")
    def demo_on_left(self) -> str:
        # [FUNCTION DOC] HELPER: Returns "left"; runs in the left concurrency group.
        return "left"

    @rpc(concurrency_group="left")
    def demo_also_on_left(self) -> str:
        # [FUNCTION DOC] HELPER: Returns "also-left"; runs in the left concurrency group too.
        return "also-left"

    async def demo_async(self) -> str:
        # [FUNCTION DOC] HELPER: Returns "async".
        return "async"


class AsyncOnlyWorker:
    # [CLASS DOC] HELPER: Worker with only async methods, one of which signals start and then blocks
    # forever; the signal is an asyncio event created at construction.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the started event (unset).
        self.started = asyncio.Event()

    async def demo_async(self) -> str:
        # [FUNCTION DOC] HELPER: Returns "async".
        return "async"

    async def demo_async_raises(self) -> None:
        # [FUNCTION DOC] HELPER: Always raises a runtime error with the message "boom".
        raise RuntimeError("boom")

    async def demo_async_blocks(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the started event, then waits forever on an event nobody sets
        # (ends only by cancellation).
        self.started.set()
        await asyncio.Event().wait()


class OutcomeRecorder:
    # [CLASS DOC] HELPER: Collects call outcomes handed to its finish method, in arrival order.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty outcome list.
        self.outcomes: list[CallStatusResponse] = []

    def finish(self, *, outcome: CallStatusResponse) -> None:
        # [FUNCTION DOC] HELPER: Appends the outcome to the list.
        self.outcomes.append(outcome)


class ExecutorUnderTest(NamedTuple):
    # [CLASS DOC] HELPER: Immutable pair of an rpc call executor and the mapping from method name to
    # method spec it was built from.
    executor: RpcCallExecutor
    specs: dict[str, RpcMethodSpec]
