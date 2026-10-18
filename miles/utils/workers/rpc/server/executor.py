# [MODULE DOC]
# Runs rpc method invocations of a worker object in the background. Async methods run directly
# on the event loop. Sync methods run on one dedicated single-thread pool per concurrency group,
# so calls in the same group are serialized in submission order while different groups can run in
# parallel and the event loop is never blocked.
from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from miles.utils.tracking_utils.structured_log import log_structured
from miles.utils.workers.rpc.common.metadata import RpcMethodSpec
from miles.utils.workers.rpc.common.protocol import CallStatusResponse

logger = logging.getLogger(__name__)


class RpcCallExecutor:
    # [CLASS DOC]
    # Executes calls against one worker object. Holds strong references to background tasks until
    # they finish, so they are not garbage collected midway. The thread pools are never shut down.
    def __init__(self, *, worker: object, specs: dict[str, RpcMethodSpec]) -> None:
        # [FUNCTION DOC] Store the worker and create one single-thread pool for each distinct
        # concurrency group among the sync methods in the specs (async methods need no pool).
        self._worker = worker
        self._executors = {
            group: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"rpc-{group}")
            for group in sorted({spec.concurrency_group for spec in specs.values() if not spec.is_async})
        }
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def concurrency_groups(self) -> list[str]:
        # [FUNCTION DOC] Property: sorted names of the groups that have a thread pool.
        return sorted(self._executors)

    def start(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]) -> None:
        # [FUNCTION DOC] Schedule the call as a background task on the running event loop and
        # return immediately. The finish callback will be called exactly once with the outcome as a
        # keyword argument. The task reference is kept until it completes.
        task = asyncio.create_task(self._run(spec=spec, kwargs=kwargs, call_id=call_id, finish=finish))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _run(
        self, *, spec: RpcMethodSpec, kwargs: dict[str, Any], call_id: str, finish: Callable[..., None]
    ) -> None:
        # [FUNCTION DOC] HELPER: Run the worker method and report the outcome through finish. On
        # success the result is encoded with the method's serializer (an encoding failure counts
        # as a call failure) and reported as success. On an ordinary exception, the full traceback
        # text is reported as failed. On cancellation, a failed outcome with the cancellation text
        # is reported and the cancellation is re-raised. Logs start and end at debug level, and
        # failures at warning or error level.
        started_at = time.monotonic()
        log_fields = {"tag": "rpc", "op": "execute", "method": spec.name, "call": call_id}
        log_structured(logger.debug, phase="start", **log_fields, group=spec.concurrency_group)

        try:
            result = await self._call_worker(spec=spec, kwargs=kwargs)
            outcome = CallStatusResponse(status="success", result=spec.serializer.encode_result(result))
            log_structured(
                logger.debug, phase="end", ok=True, **log_fields, elapsed_s=round(time.monotonic() - started_at, 3)
            )
        except asyncio.CancelledError as e:
            log_structured(logger.warning, phase="end", ok=False, cancelled=True, **log_fields)
            finish(outcome=CallStatusResponse(status="failed", error=repr(e)))
            raise
        except Exception as e:
            log_structured(logger.error, phase="end", ok=False, **log_fields, exc_info=True)
            outcome = CallStatusResponse(status="failed", error="".join(traceback.format_exception(e)))

        finish(outcome=outcome)

    async def _call_worker(self, *, spec: RpcMethodSpec, kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Look up the method on the worker by name and call it with the
        # keyword arguments: awaited directly if async, otherwise run in the thread pool of its
        # concurrency group. Returns the raw method result.
        method = getattr(self._worker, spec.name)
        if spec.is_async:
            return await method(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            self._executors[spec.concurrency_group], functools.partial(method, **kwargs)
        )
