# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# Kubernetes reflector: converts the list-then-watch protocol of the pod API into the generic
# stream of source events consumed by the reconcile loop. Every (re)list yields a replace event
# carrying the whole world; watch frames yield upsert and delete events. It tracks a resume
# cursor (resource version), relists when the server rejects the cursor or a frame cannot be
# read, and otherwise retries after a fixed delay forever.
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass

from miles.utils.test_utils.clock import Clock, RealClock
from miles.utils.workers.reconcile.k8s_api import (
    EVENT_TYPE_BOOKMARK,
    EVENT_TYPE_DELETED,
    EVENT_TYPE_ERROR,
    POD_EVENT_TYPES,
    KubernetesPodApi,
    PodWatchEvent,
    exception_rejects_cursor,
)
from miles.utils.workers.reconcile.source_event import DeleteEvent, ReplaceEvent, SourceEvent, UpsertEvent

logger = logging.getLogger(__name__)


class KubernetesReflector:
    # [CLASS DOC]
    # Source of events for the pods of one namespace that match one label selector. The watch
    # method is a restartable infinite async stream; callers cancel or close it to stop.
    def __init__(
        self,
        *,
        kube_client: KubernetesPodApi,
        namespace: str,
        label_selector: str,
        watch_timeout_seconds: int = 300,
        retry_delay: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        # [FUNCTION DOC]
        # Store the configuration. The pod API client, namespace and label selector select what is
        # reflected; the watch timeout (default 300 seconds) bounds each watch request; the retry
        # delay (default 1 second) is the pause between attempts; the clock (default real time)
        # supplies sleeping so tests can fake time. Asserts that both durations are positive.
        assert retry_delay > 0, f"{retry_delay=} must be positive"
        assert watch_timeout_seconds > 0, f"{watch_timeout_seconds=} must be positive"

        self._kube_client = kube_client
        self._namespace = namespace
        self._label_selector = label_selector
        self._watch_timeout_seconds = watch_timeout_seconds
        self._retry_delay = retry_delay
        self._clock = clock or RealClock()

    async def watch(self) -> AsyncGenerator[SourceEvent, None]:
        # [FUNCTION DOC]
        # Async generator of source events that never ends by itself. Keeps one cursor. Loop:
        # run one list-or-watch cycle, forwarding its events; when the cycle ends normally, sleep
        # the retry delay and start another. If a cycle raises (ordinary exceptions only): when the
        # error shows the cursor was rejected, log a warning and clear the cursor so the next cycle
        # relists; when it is the unreadable-frame error, log an error with trace and also clear
        # the cursor; otherwise log an error with trace and keep the cursor. In all error cases
        # sleep the retry delay before the next cycle.
        cursor = _WatchCursor()
        while True:
            try:
                async with aclosing(self._watch_once(cursor)) as events:
                    async for event in events:
                        yield event
                await self._clock.sleep(self._retry_delay)
            except Exception as exception:
                if exception_rejects_cursor(exception):
                    logger.warning(f"KubernetesReflector cursor is no longer usable, relisting {cursor=}")
                    cursor.resource_version = None
                elif isinstance(exception, _UnreadableFrameError):
                    logger.error("KubernetesReflector could not read a frame, relisting", exc_info=True)
                    cursor.resource_version = None
                else:
                    logger.error("KubernetesReflector stream failed, retrying", exc_info=True)
                await self._clock.sleep(self._retry_delay)

    async def _watch_once(self, cursor: _WatchCursor) -> AsyncGenerator[SourceEvent, None]:
        # [FUNCTION DOC] HELPER: One cycle. If the cursor is empty, list the pods, yield a replace
        # event keyed by pod name containing all of them, and set the cursor to the list's
        # resource version. Then open a watch from the cursor, bounded by the watch timeout, and
        # for each frame: an error frame that does not reject the cursor raises a runtime error;
        # one that rejects it logs a warning, clears the cursor and ends the cycle (so the next one
        # relists); any other frame is converted to a source event (a conversion failure raises the
        # unreadable-frame error and the cursor does not advance), the cursor advances to the
        # frame's resource version when it has one, and the event, if any, is yielded. The watch
        # stream is always closed on exit. The cycle ends normally when the server closes or times
        # out the watch.
        if cursor.resource_version is None:
            page = await self._kube_client.list_pods(namespace=self._namespace, label_selector=self._label_selector)
            yield ReplaceEvent(objects={pod.metadata.name: pod for pod in page.pods})
            cursor.resource_version = page.resource_version

        async with aclosing(
            self._kube_client.stream_pods(
                namespace=self._namespace,
                label_selector=self._label_selector,
                resource_version=cursor.resource_version,
                timeout_seconds=self._watch_timeout_seconds,
            )
        ) as stream:
            async for raw_event in stream:
                if raw_event.type == EVENT_TYPE_ERROR:
                    if not raw_event.rejects_cursor:
                        raise RuntimeError(f"KubernetesReflector received error event {raw_event=}")
                    logger.warning(f"KubernetesReflector received a cursor error event, relisting {raw_event=}")
                    cursor.resource_version = None
                    return

                try:
                    event = _to_source_event(raw_event)
                except Exception as exception:
                    raise _UnreadableFrameError(f"a frame the cursor cannot advance past: {raw_event=}") from exception
                cursor.resource_version = raw_event.resource_version or cursor.resource_version
                if event is not None:
                    yield event


class _UnreadableFrameError(Exception):
    # [CLASS DOC] HELPER: Raised when a watch frame cannot be converted to a source event, meaning
    # the cursor cannot move past it and the reflector must relist.
    pass


@dataclass
class _WatchCursor:
    # [CLASS DOC] HELPER: Mutable holder of the resume resource version; empty means a full list
    # is needed before watching.
    resource_version: str | None = None


def _to_source_event(raw_event: PodWatchEvent) -> SourceEvent | None:
    # [FUNCTION DOC] HELPER: Convert a pod watch event to a source event keyed by pod name: a
    # deleted event becomes a delete event carrying the pod as the last known object; added and
    # modified become upsert events. Asserts that these types carry a pod. Bookmarks yield nothing
    # silently; any other type logs a warning and yields nothing.
    if raw_event.type in POD_EVENT_TYPES:
        assert raw_event.pod is not None, f"a {raw_event.type} frame carries no pod"
        key = raw_event.pod.metadata.name
        if raw_event.type == EVENT_TYPE_DELETED:
            return DeleteEvent(key=key, last_obj=raw_event.pod)
        return UpsertEvent(key=key, obj=raw_event.pod)
    if raw_event.type != EVENT_TYPE_BOOKMARK:
        logger.warning(f"KubernetesReflector ignoring unknown event {raw_event.type=}")
    return None
