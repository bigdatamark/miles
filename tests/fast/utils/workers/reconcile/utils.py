# [MODULE DOC] HELPER:
# Shared test doubles and builders for the reconcile tests: a helper that lets pending event-loop
# work run, a scriptable fake event source (with open and close accounting), a recorder of parent
# keys, a collector that drains an event stream in a background task, builders for pods, wire
# pods and replace events, a scriptable fake pod api that records list and stream calls and
# stream closing, and a fake client api exception carrying a status.
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.reconcile.k8s_api import PodListPage, PodWatchEvent
from miles.utils.workers.reconcile.source_event import ParentKey, ReplaceEvent, SourceEvent


async def settle(iterations: int = 200) -> None:
    # [FUNCTION DOC] HELPER: Yields control to the event loop the given number of times (default
    # 200) so that all ready tasks make progress; does not advance real or fake time.
    for _ in range(iterations):
        await asyncio.sleep(0)


class StreamEnd:
    # [CLASS DOC] HELPER: Marker item; queued into a fake source to make its stream end normally.
    pass


class StreamError:
    # [CLASS DOC] HELPER: Marker item carrying an exception; queued into a fake source to make
    # its stream raise that exception.
    def __init__(self, error: BaseException) -> None:
        # [FUNCTION DOC] HELPER: Keeps the exception to raise.
        self.error = error


class FakeSource:
    # [CLASS DOC] HELPER: Callable factory of event streams, mimicking a source opener. Each call
    # opens a new stream backed by its own queue that the test fills through the emit method.
    # Counts how many times it was opened and how many streams were closed. Can be told to make
    # the first several opens fail inside the stream, or the first several calls fail outright.
    def __init__(self, *, fail_opens: int = 0, fail_calls: int = 0) -> None:
        # [FUNCTION DOC] HELPER: Sets counters to zero and stores how many initial opens should
        # fail inside the stream (fail_opens) and how many initial calls should raise
        # immediately (fail_calls).
        self.open_count = 0
        self.closed_count = 0
        self._fail_opens = fail_opens
        self._fail_calls = fail_calls
        self._queues: list[asyncio.Queue[Any]] = []

    def __call__(self) -> AsyncGenerator[SourceEvent, None]:
        # [FUNCTION DOC] HELPER: Counts the open. If the open number is within the failing-call
        # budget raises a runtime error immediately. Otherwise creates a queue (remembered for
        # emit) and returns the stream over it, which will itself fail if the open number is
        # within the failing-open budget. The fail-call budget is checked on the same counter,
        # so the two budgets overlap rather than add.
        self.open_count += 1
        if self.open_count <= self._fail_calls:
            raise RuntimeError("fake source factory failure")
        queue: asyncio.Queue[Any] = asyncio.Queue()
        self._queues.append(queue)
        return self._iterate(queue, fail=self.open_count <= self._fail_opens)

    async def _iterate(self, queue: asyncio.Queue[Any], *, fail: bool) -> AsyncGenerator[SourceEvent, None]:
        # [FUNCTION DOC] HELPER: The stream. If asked to fail raises a runtime error on first
        # iteration. Otherwise waits for queue items forever: an end marker ends the stream, an
        # error marker raises its exception, anything else is yielded. Increments the closed
        # count when the stream finishes for any reason (end, error, close, cancellation).
        try:
            if fail:
                raise RuntimeError("fake source open failure")
            while True:
                item = await queue.get()
                if isinstance(item, StreamEnd):
                    return
                if isinstance(item, StreamError):
                    raise item.error
                yield item
        finally:
            self.closed_count += 1

    def emit(self, *events: Any) -> None:
        # [FUNCTION DOC] HELPER: Puts the events, in order, on the queue of the most recently
        # opened stream without waiting. Fails with an index error if nothing was opened yet.
        for event in events:
            self._queues[-1].put_nowait(event)


class ParentKeyRecorder:
    # [CLASS DOC] HELPER: Async callback object that records the parent keys it is called with.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with an empty list of recorded keys.
        self.parent_keys: list[ParentKey] = []

    async def __call__(self, parent_key: ParentKey) -> None:
        # [FUNCTION DOC] HELPER: Appends the parent key to the recorded list.
        self.parent_keys.append(parent_key)


class EventCollector:
    # [CLASS DOC] HELPER: Drains an event stream in a background task, collecting events into a
    # list and remembering any non-cancellation error. Must be created inside a running event
    # loop.
    def __init__(self, stream: AsyncGenerator[SourceEvent, None]) -> None:
        # [FUNCTION DOC] HELPER: Stores the stream and immediately starts the draining task.
        self.events: list[SourceEvent] = []
        self.error: BaseException | None = None
        self._stream = stream
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        # [FUNCTION DOC] HELPER: Appends every streamed event to the list. Cancellation is
        # re-raised; any other exception (including base exceptions) is stored as the error and
        # ends the task quietly.
        try:
            async for event in self._stream:
                self.events.append(event)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self.error = error

    async def close(self) -> None:
        # [FUNCTION DOC] HELPER: Cancels the task and waits for it, closes the stream, then
        # asserts that no error was recorded (message shows the error).
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        await self._stream.aclose()
        assert self.error is None, f"stream failed with {self.error=}"


def make_pod(name: str, *, cell: str | None = "cell-a", resource_version: str = "1") -> Pod:
    # [FUNCTION DOC] HELPER: Builds a validated pod model from the wire-shaped fake with the
    # given name, optional cell label (default "cell-a"; none means no labels) and resource
    # version (default "1").
    return Pod.model_validate(wire_pod(name, cell=cell, resource_version=resource_version))


def wire_pod(name: str, *, cell: str | None = "cell-a", resource_version: str = "1") -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds an attribute-style object mimicking a deserialized client
    # pod: metadata with the name, a uid derived from the name, the resource version, labels
    # (empty when cell is none, otherwise a mapping from "cell" to the value) and no
    # annotations, plus no spec and no status.
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            uid=f"uid-{name}",
            resource_version=resource_version,
            labels={} if cell is None else {"cell": cell},
            annotations=None,
        ),
        spec=None,
        status=None,
    )


def replace_of(*pods: Pod) -> ReplaceEvent:
    # [FUNCTION DOC] HELPER: Builds a replace event whose objects are the pods keyed by name.
    return ReplaceEvent(objects={pod.metadata.name: pod for pod in pods})


def make_pod_list(pods: list[Pod], *, resource_version: str) -> PodListPage:
    # [FUNCTION DOC] HELPER: Builds a pod list page from the pods and the list's resource version.
    return PodListPage(pods=pods, resource_version=resource_version)


def pod_cell(pod: Pod) -> str:
    # [FUNCTION DOC] HELPER: Returns the pod's "cell" label; raises a key error if absent.
    return pod.metadata.labels["cell"]


class FakePodApi:
    # [CLASS DOC] HELPER: Scriptable fake of the pod api. List calls consume queued pages
    # (a page may be an exception to raise); stream calls consume queued scripts (a script is a
    # list of events or exceptions). All calls are recorded with their arguments, and stream
    # closing is counted.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with empty page and script queues, empty call records,
        # and zero closed and generator-closed stream counters.
        self.list_pages: deque[Any] = deque()
        self.stream_scripts: deque[list[Any]] = deque()
        self.list_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.closed_streams = 0
        self.aclosed_streams = 0

    async def list_pods(self, *, namespace: str, label_selector: str) -> Any:
        # [FUNCTION DOC] HELPER: Records the namespace and selector, pops the next scripted page
        # and returns it, or raises it if it is an exception. Fails an assertion if the queue is
        # empty.
        self.list_calls.append(dict(namespace=namespace, label_selector=label_selector))
        page = self.list_pages.popleft() if self.list_pages else None
        if isinstance(page, BaseException):
            raise page
        assert page is not None, "FakePodApi ran out of scripted list pages"
        return page

    async def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        # [FUNCTION DOC] HELPER: Records namespace, selector, resource version and timeout.
        # Pops the next script (none if the queue is empty). Yields the script items in order,
        # raising any item that is an exception. When there is no script at all, waits forever
        # after starting (a script that is an empty list ends immediately). If the consumer
        # closes the generator, counts it as generator-closed and re-raises. Always counts a
        # closed stream when finishing for any reason.
        self.stream_calls.append(
            dict(
                namespace=namespace,
                label_selector=label_selector,
                resource_version=resource_version,
                timeout_seconds=timeout_seconds,
            )
        )
        script = self.stream_scripts.popleft() if self.stream_scripts else None
        try:
            for item in script or []:
                if isinstance(item, BaseException):
                    raise item
                yield item
            if script is None:
                await asyncio.Event().wait()
        except GeneratorExit:
            self.aclosed_streams += 1
            raise
        finally:
            self.closed_streams += 1


class FakeApiException(Exception):
    # [CLASS DOC] HELPER: Stand-in for the client library's api exception, carrying a status that
    # may be a number or a string.
    def __init__(self, status: int | str) -> None:
        # [FUNCTION DOC] HELPER: Sets the message to "api exception" followed by the status and
        # stores the status.
        super().__init__(f"api exception {status}")
        self.status = status
