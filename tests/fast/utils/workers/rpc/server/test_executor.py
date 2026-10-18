# [MODULE DOC] Unit tests for the rpc server call executor: which concurrency groups it reports
# (unique, sorted, sync methods only) and how it tracks started calls in its private background task
# set (a task is held while running and dropped after success, failure or cancellation, with the
# outcome recorded through a finish callback). Executors are built by the make-executor fixture.
import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from tests.fast.utils.workers.rpc.server.fake_workers import (
    AsyncOnlyWorker,
    ExecutorUnderTest,
    OutcomeRecorder,
    SyncAndAsyncWorker,
)


class TestConcurrencyGroups:
    # [CLASS DOC] Tests the executor's list of concurrency group names.
    def test_concurrency_groups_reports_unique_sorted_sync_groups(
        self, make_executor: Callable[[type], ExecutorUnderTest]
    ) -> None:
        # [FUNCTION DOC] For the mixed worker, the groups are exactly the default and left groups,
        # each once and sorted.
        """Each sync concurrency group is reported exactly once, in sorted order."""
        assert make_executor(SyncAndAsyncWorker).executor.concurrency_groups == ["default", "left"]

    def test_async_only_worker_gets_no_concurrency_group(
        self, make_executor: Callable[[type], ExecutorUnderTest]
    ) -> None:
        # [FUNCTION DOC] For the async-only worker, the list is empty.
        """A worker with only async methods needs no executor group at all."""
        assert make_executor(AsyncOnlyWorker).executor.concurrency_groups == []


class TestBackgroundTasks:
    # [CLASS DOC] Tests the lifetime of tasks the executor starts for calls.
    @pytest.mark.parametrize(
        ("method_name", "kwargs", "status"),
        [("demo_sync", {"value": 1}, "success"), ("demo_sync_raises", {}, "failed")],
    )
    async def test_finished_tasks_are_removed_from_background_task_set(
        self,
        make_executor: Callable[[type], ExecutorUnderTest],
        method_name: str,
        kwargs: dict[str, Any],
        status: str,
    ) -> None:
        # [FUNCTION DOC] Parametrized over a succeeding and a raising sync method. Starts the call with
        # an outcome recorder as finish callback; right after start exactly one task is held. After
        # awaiting it, the recorded outcome statuses are the expected single status and the held set is
        # empty.
        """A started call is held while it runs and dropped once it reaches a terminal outcome."""
        under_test = make_executor(SyncAndAsyncWorker)
        recorder = OutcomeRecorder()

        under_test.executor.start(
            spec=under_test.specs[method_name], kwargs=kwargs, call_id="c1", finish=recorder.finish
        )
        running = set(under_test.executor._background_tasks)
        assert len(running) == 1

        await asyncio.gather(*running)

        assert [outcome.status for outcome in recorder.outcomes] == [status]
        assert under_test.executor._background_tasks == set()

    async def test_cancelled_tasks_are_removed_from_background_task_set(
        self, make_executor: Callable[[type], ExecutorUnderTest]
    ) -> None:
        # [FUNCTION DOC] Starts the blocking async method, waits (up to 1 second) until the worker
        # reports it started, cancels the held task and awaits it expecting a cancellation error, then
        # yields once. Asserts one recorded outcome with status failed and an empty held set.
        """A task cancelled while its worker is running is dropped after cancellation finishes."""
        under_test = make_executor(AsyncOnlyWorker)
        worker = under_test.executor._worker
        assert isinstance(worker, AsyncOnlyWorker)
        recorder = OutcomeRecorder()

        under_test.executor.start(
            spec=under_test.specs["demo_async_blocks"], kwargs={}, call_id="c1", finish=recorder.finish
        )
        task = next(iter(under_test.executor._background_tasks))
        await asyncio.wait_for(worker.started.wait(), timeout=1.0)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)

        assert [outcome.status for outcome in recorder.outcomes] == ["failed"]
        assert under_test.executor._background_tasks == set()
