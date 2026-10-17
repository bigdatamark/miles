# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC]
# Thin abstraction over the Kubernetes pod list and watch calls used by the reflector, with
# pydantic result types, plus the rules for recognizing that a watch cursor (resource version)
# was rejected by the server (status 410 gone or 504, or an expired or gone reason), in which
# case the caller must relist instead of resuming.
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any, Protocol

from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Pod, WatchFrame

logger = logging.getLogger(__name__)

_CURSOR_REJECTED_CODES = (410, 504)
_CURSOR_REJECTED_REASONS = ("Expired", "Gone")

EVENT_TYPE_ADDED = "ADDED"
EVENT_TYPE_MODIFIED = "MODIFIED"
EVENT_TYPE_DELETED = "DELETED"
EVENT_TYPE_BOOKMARK = "BOOKMARK"
EVENT_TYPE_ERROR = "ERROR"

POD_EVENT_TYPES = (EVENT_TYPE_ADDED, EVENT_TYPE_MODIFIED, EVENT_TYPE_DELETED)


class PodListPage(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Immutable result of listing pods: the pods and the list's resource version, from which a
    # watch can resume.
    pods: list[Pod]
    resource_version: str


class PodWatchEvent(FrozenStrictBaseModel):
    # [CLASS DOC]
    # Immutable single event from a pod watch: the event type string, the pod (only for added,
    # modified and deleted events, else absent), the resource version carried by the frame
    # (absent if none) and whether this is an error frame saying the cursor was rejected.
    type: str
    pod: Pod | None
    resource_version: str | None
    rejects_cursor: bool

    @classmethod
    def from_frame(cls, *, event_type: str, obj: Any) -> PodWatchEvent:
        # [FUNCTION DOC]
        # Build an event from a raw watch frame (event type and the raw object, as a mapping or
        # client object). Parses the object leniently as a watch frame to get the resource version
        # and error code and reason; additionally parses it as a pod when the type is added,
        # modified or deleted (validation errors propagate). The rejects-cursor flag is true only
        # for error-type frames whose code or reason signals a rejected cursor.
        frame = WatchFrame.model_validate(obj)
        return cls(
            type=event_type,
            pod=Pod.model_validate(obj) if event_type in POD_EVENT_TYPES else None,
            resource_version=frame.metadata.resource_version,
            rejects_cursor=event_type == EVENT_TYPE_ERROR and _frame_rejects_cursor(frame),
        )


class KubernetesPodApi(Protocol):
    # [CLASS DOC]
    # Structural interface for the pod calls the reflector needs: an async list and an async
    # streaming watch. Implemented by the asyncio client wrapper and by test fakes.
    # [FUNCTION DOC] (list_pods) Return one page holding all pods in the namespace that match the
    # label selector, together with the resource version of the list.
    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage: ...

    # [FUNCTION DOC] (stream_pods) Return an async stream of watch events for matching pods in the
    # namespace, resuming after the given resource version and ending by itself after the given
    # timeout in seconds. Implementations may also end it earlier or raise.
    def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]: ...


class KubernetesAsyncioPodApi:
    # [CLASS DOC]
    # Pod API backed by the core v1 API object of the async kubernetes client (passed in and used
    # as is, untyped).
    def __init__(self, *, core_v1_api: Any) -> None:
        # [FUNCTION DOC] Store the core v1 API client object.
        self._core_v1_api = core_v1_api

    async def list_pods(self, *, namespace: str, label_selector: str) -> PodListPage:
        # [FUNCTION DOC] List pods in the namespace with the label selector through the client and
        # return them with the list's resource version. Client errors propagate.
        pod_list = await self._core_v1_api.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        return PodListPage(pods=list(pod_list.items), resource_version=pod_list.metadata.resource_version)

    async def stream_pods(
        self, *, namespace: str, label_selector: str, resource_version: str, timeout_seconds: int
    ) -> AsyncGenerator[PodWatchEvent, None]:
        # [FUNCTION DOC]
        # Async generator that opens a watch on the pod list call with the given resume resource
        # version, timeout and with bookmarks enabled, and yields one pod watch event per received
        # frame (converted by the frame parser; a malformed frame raises out of the generator).
        # The watch object is always closed when the generator ends, is closed early or fails;
        # errors during closing are only logged. The kubernetes client import is local to defer
        # the optional dependency.
        from kubernetes_asyncio import watch as kubernetes_watch

        watcher = kubernetes_watch.Watch()
        try:
            async for event in watcher.stream(
                self._core_v1_api.list_namespaced_pod,
                namespace=namespace,
                label_selector=label_selector,
                resource_version=resource_version,
                timeout_seconds=timeout_seconds,
                allow_watch_bookmarks=True,
            ):
                yield PodWatchEvent.from_frame(event_type=event["type"], obj=event["object"])
        finally:
            await _close_quietly(watcher.close())


async def _close_quietly(closing: Any) -> None:
    # [FUNCTION DOC] HELPER: Await the given awaitable; any ordinary exception is logged as an
    # error and swallowed.
    try:
        await closing
    except Exception:
        logger.error("failed to close a Kubernetes watch stream", exc_info=True)


def exception_rejects_cursor(exception: BaseException) -> bool:
    # [FUNCTION DOC] True iff the exception carries an http status attribute equal to 410 or 504,
    # meaning the server no longer accepts the watch's resource version.
    return getattr(exception, "status", None) in _CURSOR_REJECTED_CODES


def _frame_rejects_cursor(frame: WatchFrame) -> bool:
    # [FUNCTION DOC] HELPER: True iff the frame's code is 410 or 504, or its reason is the text
    # Expired or Gone.
    return frame.code in _CURSOR_REJECTED_CODES or frame.reason in _CURSOR_REJECTED_REASONS
