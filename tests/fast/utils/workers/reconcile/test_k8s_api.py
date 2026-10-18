# [MODULE DOC]
# Unit tests for the Kubernetes pod api layer: closing of the watch stream (failures logged, not
# raised), parsing of watch frames (resource version in both attribute and camel-case mapping
# shapes, pod models, scheduling fields, non-pod frames), detection of frames and client
# exceptions that mean the watch cursor is dead, using small fake client objects.
from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from miles.utils.workers.k8s_types import ContainerStatus, Pod, PodMetadata, PodStatus
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi, PodWatchEvent, exception_rejects_cursor


def make_exception(**fields: Any) -> Exception:
    # [FUNCTION DOC] HELPER: Builds a plain exception with message "boom" and sets each given
    # keyword as an attribute on it (for example a status).
    exception = Exception("boom")
    for name, value in fields.items():
        setattr(exception, name, value)
    return exception


def make_wire_pod() -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds an attribute-style fake client pod: name "pod-0", uid
    # "uid-0", resource version "7", no labels or annotations, no spec, and a status with ip
    # 10.0.0.1, no conditions and one container status with restart count 2.
    return SimpleNamespace(
        metadata=SimpleNamespace(name="pod-0", uid="uid-0", resource_version="7", labels=None, annotations=None),
        spec=None,
        status=SimpleNamespace(
            pod_ip="10.0.0.1", conditions=None, container_statuses=[SimpleNamespace(restart_count=2)]
        ),
    )


class TestKubernetesAsyncioPodApi:
    # [CLASS DOC] Tests of the asyncio-client-backed pod api.
    async def test_stream_pods_swallows_and_logs_a_watch_close_failure(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # [FUNCTION DOC] Installs a fake watch module (restored by monkeypatch) whose watch
        # yields nothing and whose close raises a runtime error. Streams pods through the api and
        # asserts no events, no exception to the caller, exactly one error log message
        # "failed to close a Kubernetes watch stream", and that the log record carries exception
        # information.
        """An exhausted stream logs a watch close failure without exposing it to the caller."""

        class FakeWatch:
            # [CLASS DOC] HELPER: Fake watch with an empty stream and a failing close.
            async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
                # [FUNCTION DOC] HELPER: Empty async generator; ignores all arguments.
                if False:
                    yield None

            async def close(self) -> None:
                # [FUNCTION DOC] HELPER: Always raises a runtime error "watch close failed".
                raise RuntimeError("watch close failed")

        watch_module = ModuleType("kubernetes_asyncio.watch")
        watch_module.Watch = FakeWatch
        package = ModuleType("kubernetes_asyncio")
        package.watch = watch_module
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
        monkeypatch.setitem(sys.modules, "kubernetes_asyncio.watch", watch_module)
        api = KubernetesAsyncioPodApi(core_v1_api=SimpleNamespace(list_namespaced_pod=None))

        with caplog.at_level(logging.ERROR, logger="miles.utils.workers.reconcile.k8s_api"):
            events = [
                event
                async for event in api.stream_pods(
                    namespace="default", label_selector="app=trainer", resource_version="7", timeout_seconds=30
                )
            ]

        assert events == []
        assert caplog.messages == ["failed to close a Kubernetes watch stream"]
        assert caplog.records[0].exc_info is not None


class TestResourceVersionParsing:
    # [CLASS DOC] Reading the resource version out of a watch frame.
    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (SimpleNamespace(metadata=SimpleNamespace(resource_version="7")), "7"),
            (dict(metadata=dict(resourceVersion="7")), "7"),
        ],
    )
    def test_both_wire_shapes_are_read(self, obj: Any, expected: str) -> None:
        # [FUNCTION DOC] Parametrized over an attribute-style object and a camel-case mapping,
        # both with version "7"; asserts the parsed bookmark frame reports that version.
        """A deserialized model spells it as an attribute, a raw dict as a camelCase key."""
        assert PodWatchEvent.from_frame(event_type="BOOKMARK", obj=obj).resource_version == expected

    @pytest.mark.parametrize(
        "obj",
        [
            SimpleNamespace(),
            SimpleNamespace(metadata=None),
            SimpleNamespace(metadata=SimpleNamespace()),
            {},
            dict(metadata=None),
            dict(metadata={}),
        ],
    )
    def test_a_frame_without_a_readable_version_parses_to_none(self, obj: Any) -> None:
        # [FUNCTION DOC] Parametrized over six payloads lacking a readable version (missing,
        # empty or null metadata, in both shapes); asserts the parsed version is nothing.
        """A missing or malformed metadata block must parse to None, never raise: the caller keeps its cursor."""
        assert PodWatchEvent.from_frame(event_type="BOOKMARK", obj=obj).resource_version is None

    def test_a_payload_that_is_not_an_object_at_all_raises(self) -> None:
        # [FUNCTION DOC] Parsing a frame whose payload is a plain string raises a validation
        # error.
        """An unreadable envelope carries no cursor to keep, so it must raise and let the reflector relist."""
        with pytest.raises(ValidationError):
            PodWatchEvent.from_frame(event_type="BOOKMARK", obj="a payload that is not an object at all")


class TestPodParsing:
    # [CLASS DOC] Conversion of frame payloads into typed pod models.
    def test_a_pod_frame_carries_the_pod_the_wire_described(self) -> None:
        # [FUNCTION DOC] Parses an added frame with the fake wire pod; asserts the pod equals
        # the expected model (name, uid, version, pod ip and one container status).
        """Every consumer downstream reads typed fields, so the frame is validated once, here."""
        event = PodWatchEvent.from_frame(event_type="ADDED", obj=make_wire_pod())

        assert event.pod == Pod(
            metadata=PodMetadata(name="pod-0", uid="uid-0", resource_version="7"),
            status=PodStatus(pod_ip="10.0.0.1", container_statuses=[ContainerStatus(restart_count=2)]),
        )

    def test_a_raw_json_pod_is_read_through_its_camel_case_spelling(self) -> None:
        # [FUNCTION DOC] Parses a mapping-shaped pod with camel-case keys; asserts a pod exists
        # with node name "gpu-1", pod ip "10.0.0.1" and one container restart count 2.
        """A payload the client left as raw JSON spells every compound field differently."""
        obj = dict(
            metadata=dict(name="pod-0", uid="uid-0", resourceVersion="7"),
            spec=dict(nodeName="gpu-1"),
            status=dict(podIP="10.0.0.1", containerStatuses=[dict(restartCount=2)]),
        )

        event = PodWatchEvent.from_frame(event_type="ADDED", obj=obj)

        assert event.pod is not None
        assert (event.pod.spec.node_name, event.pod.status.pod_ip) == ("gpu-1", "10.0.0.1")
        assert [status.restart_count for status in event.pod.status.container_statuses] == [2]

    def test_the_scheduling_fields_a_gated_pod_is_paired_by_are_typed(self) -> None:
        # [FUNCTION DOC] Parses a pod with one scheduling gate and a node selector; asserts the
        # gate names and the selector mapping are preserved.
        """Colocate pairing reads the gates and the node selector off the pod, so PodSpec has to model them."""
        obj = dict(
            metadata=dict(name="pod-0", uid="uid-0"),
            spec=dict(
                nodeName=None,
                schedulingGates=[dict(name="miles.ai/awaiting-pair")],
                nodeSelector={"kubernetes.io/hostname": "gpu-1"},
            ),
        )

        event = PodWatchEvent.from_frame(event_type="ADDED", obj=obj)

        assert event.pod is not None
        assert [gate.name for gate in event.pod.spec.scheduling_gates] == ["miles.ai/awaiting-pair"]
        assert event.pod.spec.node_selector == {"kubernetes.io/hostname": "gpu-1"}

    def test_an_ungated_pod_reads_as_empty_rather_than_missing(self) -> None:
        # [FUNCTION DOC] Parses a pod with no spec given; asserts the scheduling gates are an
        # empty list and the node selector an empty mapping.
        """The client leaves both fields out entirely, and pairing must not have to guard every read."""
        event = PodWatchEvent.from_frame(event_type="ADDED", obj=make_wire_pod())

        assert event.pod is not None
        assert (event.pod.spec.scheduling_gates, event.pod.spec.node_selector) == ([], {})

    @pytest.mark.parametrize("event_type", ["BOOKMARK", "ERROR"])
    def test_a_frame_that_is_not_about_a_pod_carries_no_pod(self, event_type: str) -> None:
        # [FUNCTION DOC] Parametrized over bookmark and error frames with a status-like payload;
        # asserts the parsed frame has no pod.
        """BOOKMARK and ERROR frames carry a bare version and a Status, and parsing them as pods would fail."""
        assert PodWatchEvent.from_frame(event_type=event_type, obj=dict(code=410, reason="Expired")).pod is None

    def test_a_pod_frame_miles_cannot_read_is_refused(self) -> None:
        # [FUNCTION DOC] An added frame whose payload has no metadata raises a validation error.
        """A pod the apiserver sent that does not validate is a contract break, not a pod to skip quietly."""
        with pytest.raises(ValidationError):
            PodWatchEvent.from_frame(event_type="ADDED", obj=SimpleNamespace(metadata=None))


class TestCursorRejection:
    # [CLASS DOC] Detection of error frames that mean the watch cursor is expired or gone.
    @pytest.mark.parametrize(
        "obj",
        [
            SimpleNamespace(code=410, reason="Expired"),
            SimpleNamespace(code=504, reason="Timeout"),
            SimpleNamespace(code=None, reason="Expired"),
            dict(code=410),
            dict(reason="Gone"),
        ],
    )
    def test_an_error_frame_reporting_a_dead_cursor_is_flagged(self, obj: Any) -> None:
        # [FUNCTION DOC] Parametrized over error payloads with code 410 or 504, reason "Expired"
        # or "Gone", in both shapes; asserts the frame is flagged as rejecting the cursor.
        """Either the code or the reason is enough, in either wire shape."""
        assert PodWatchEvent.from_frame(event_type="ERROR", obj=obj).rejects_cursor

    @pytest.mark.parametrize(
        ("event_type", "obj"),
        [
            ("ERROR", SimpleNamespace(code=500, reason="InternalError")),
            ("ERROR", dict(code=500)),
            ("ERROR", SimpleNamespace()),
            ("BOOKMARK", dict(code=410)),
        ],
    )
    def test_anything_else_leaves_the_cursor_alone(self, event_type: str, obj: Any) -> None:
        # [FUNCTION DOC] Parametrized over error frames with other codes or no code, and a
        # bookmark with code 410; asserts none is flagged as rejecting the cursor.
        """Only an ERROR frame may invalidate a cursor, and only for a cursor-specific code."""
        assert not PodWatchEvent.from_frame(event_type=event_type, obj=obj).rejects_cursor

    def test_a_pod_frame_carrying_a_dead_cursor_code_leaves_the_cursor_alone(self) -> None:
        # [FUNCTION DOC] Adds code 410 and reason "Expired" attributes to a wire pod and parses
        # it as a modified frame; asserts it is not flagged and the pod name is still read.
        """A pod whose own fields happen to spell 410 must not be read as an expired-cursor error."""
        obj = make_wire_pod()
        obj.code = 410
        obj.reason = "Expired"

        event = PodWatchEvent.from_frame(event_type="MODIFIED", obj=obj)

        assert not event.rejects_cursor
        assert event.pod is not None and event.pod.metadata.name == "pod-0"


class TestExceptionRejection:
    # [CLASS DOC] Detection of client exceptions that mean the watch cursor is dead.
    @pytest.mark.parametrize(
        "exception",
        [make_exception(status=410), make_exception(status=504)],
    )
    def test_a_client_exception_reporting_a_dead_cursor_is_flagged(self, exception: BaseException) -> None:
        # [FUNCTION DOC] Parametrized over exceptions with numeric status 410 and 504; asserts
        # the classifier reports the cursor as rejected.
        """ApiException.status carries the HTTP code, and 410 and 504 both mean the cursor is gone."""
        assert exception_rejects_cursor(exception)

    @pytest.mark.parametrize(
        "exception",
        [make_exception(), make_exception(status=500), make_exception(status="410")],
    )
    def test_any_other_exception_is_a_plain_stream_failure(self, exception: BaseException) -> None:
        # [FUNCTION DOC] Parametrized over an exception without status, one with status 500 and
        # one with the string status "410"; asserts none is reported as rejecting the cursor
        # (only numeric statuses count).
        """A transient error, or a code in a shape the client never produces, keeps the cursor."""
        assert not exception_rejects_cursor(exception)
