# [MODULE DOC]
# Unit tests for the Kubernetes reflector, which turns a LIST-then-WATCH of pods into a stream of
# source events (one replace carrying the whole world, then upserts and deletes), driven by a
# scripted fake pod api and a fake clock. Covers cadence validation, the initial list, mapping of
# watch event types, cursor (resource version) tracking and reconnect rules, expiry handling that
# forces a relist, stream closing, lifecycle, the real api adapter against a fake client library,
# and the end-to-end feed into the reconcile loop.
from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from tests.fast.utils.workers.reconcile.utils import (
    EventCollector,
    FakeApiException,
    FakePodApi,
    ParentKeyRecorder,
    make_pod,
    make_pod_list,
    pod_cell,
    settle,
    wire_pod,
)

from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.k8s_api import KubernetesAsyncioPodApi, PodListPage, PodWatchEvent
from miles.utils.workers.reconcile.k8s_reflector import KubernetesReflector
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.reconcile.source_event import DeleteEvent, ReplaceEvent, UpsertEvent


def raw_event(event_type: str, obj: Any) -> PodWatchEvent:
    # [FUNCTION DOC] HELPER: Parses a watch frame from an event type string and a payload object.
    return PodWatchEvent.from_frame(event_type=event_type, obj=obj)


def unreadable_pod_frame(event_type: str) -> BaseException:
    # [FUNCTION DOC] HELPER: Asserts that parsing a frame with metadata lacking a name raises a
    # validation error, and returns that error (to be placed in a fake stream script so the
    # reflector sees an unreadable frame).
    with pytest.raises(ValidationError) as refusal:
        raw_event(event_type, SimpleNamespace(metadata=SimpleNamespace(resource_version="5")))
    return refusal.value


def make_status(*, code: int, reason: str = "Expired") -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds an attribute-style status payload with the code, reason
    # (default "Expired") and status text "Failure", as carried by watch error frames.
    return SimpleNamespace(code=code, reason=reason, status="Failure")


def _install_fake_kubernetes_asyncio(monkeypatch: Any) -> tuple[Any, dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Installs fake client library package and watch modules into the
    # module table (restored by monkeypatch). The fake watch records the function and keyword
    # arguments given to its stream call, yields one modified frame for a pod named
    # "pod-from-the-wire", and counts close calls. Returns the fake watch module and the shared
    # state mapping with keys func, kwargs and closed.
    state: dict[str, Any] = dict(func=None, kwargs=None, closed=0)

    class _FakeWatch:
        # [CLASS DOC] HELPER: Fake client watch object.
        def stream(self, func: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Records the function and keywords, returns itself as the
            # async iterable.
            state["func"] = func
            state["kwargs"] = kwargs
            return self

        def __aiter__(self) -> Any:
            # [FUNCTION DOC] HELPER: Returns an async iterator yielding exactly one modified
            # frame (mapping with type and object).
            async def _iterate() -> AsyncIterator[dict[str, Any]]:
                # [FUNCTION DOC] HELPER: Yields the single modified frame.
                yield dict(type="MODIFIED", object=wire_pod("pod-from-the-wire"))

            return _iterate()

        async def close(self) -> None:
            # [FUNCTION DOC] HELPER: Increments the shared closed counter.
            state["closed"] += 1

    watch_module = ModuleType("kubernetes_asyncio.watch")
    watch_module.Watch = _FakeWatch
    package = ModuleType("kubernetes_asyncio")
    package.watch = watch_module
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio", package)
    monkeypatch.setitem(sys.modules, "kubernetes_asyncio.watch", watch_module)
    return watch_module, state


def make_reflector(api: FakePodApi, *, clock: FakeClock | None = None, **kwargs: Any) -> KubernetesReflector:
    # [FUNCTION DOC] HELPER: Builds a reflector over the fake api for namespace "ns-test" and
    # selector "app=miles", with the given fake clock (a new one by default); extra keywords
    # (retry delay, watch timeout) are passed through.
    return KubernetesReflector(
        kube_client=api,
        namespace="ns-test",
        label_selector="app=miles",
        clock=clock or FakeClock(),
        **kwargs,
    )


class TestCadenceValidation:
    # [CLASS DOC] Construction-time validation of the reflector's timing options.
    @pytest.mark.parametrize("kwargs", [dict(retry_delay=0.0), dict(retry_delay=-1.0), dict(watch_timeout_seconds=0)])
    async def test_a_non_positive_cadence_is_rejected(self, kwargs):
        # [FUNCTION DOC] Parametrized over zero and negative retry delay and zero watch timeout;
        # asserts construction fails an assertion.
        """A zero retry delay or watch timeout would hammer the apiserver, so it never reaches construction."""
        with pytest.raises(AssertionError):
            make_reflector(FakePodApi(), **kwargs)


class TestInitialList:
    # [CLASS DOC] The first phase of the stream: the LIST.
    async def test_list_emits_one_replace_carrying_every_pod(self):
        # [FUNCTION DOC] Scripts a list of two pods; after letting the collector run, asserts a
        # single replace event keyed by the two pod names.
        """The initial LIST arrives as a single whole-world ReplaceEvent, keyed by pod name."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0"), make_pod("pod-1")], resource_version="100"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert sorted(collector.events[0].objects) == ["pod-0", "pod-1"]
        await collector.close()

    async def test_list_uses_namespace_and_label_selector(self):
        # [FUNCTION DOC] Asserts the single list call carried namespace "ns-test" and selector
        # "app=miles".
        """LIST is scoped to the configured namespace and selector."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert api.list_calls == [dict(namespace="ns-test", label_selector="app=miles")]
        await collector.close()

    async def test_watch_starts_from_the_list_resource_version(self):
        # [FUNCTION DOC] With list version "100" and watch timeout 42, asserts the one stream
        # call used that version and timeout with the same namespace and selector.
        """WATCH continues exactly where the LIST snapshot ended."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="100"))
        collector = EventCollector(make_reflector(api, watch_timeout_seconds=42).watch())
        await settle()

        assert api.stream_calls == [
            dict(namespace="ns-test", label_selector="app=miles", resource_version="100", timeout_seconds=42)
        ]
        await collector.close()

    async def test_the_default_watch_timeout_is_five_minutes(self):
        # [FUNCTION DOC] Asserts the first stream call used timeout 300 seconds when none was
        # configured.
        """A caller that configures nothing still gets a finite server-side watch timeout."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert api.stream_calls[0]["timeout_seconds"] == 300
        await collector.close()

    async def test_empty_list_still_emits_a_replace(self):
        # [FUNCTION DOC] An empty list still yields exactly one replace event.
        """An empty cluster must still lift the initial-sync barrier."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        await collector.close()

    async def test_list_failure_is_retried_after_the_delay(self):
        # [FUNCTION DOC] First list raises, second succeeds, retry delay 3 s on a fake clock.
        # Asserts no events before the delay, and one replace event after advancing 3 s.
        """A failing LIST is retried; no events escape before it succeeds."""
        api = FakePodApi()
        api.list_pages.append(RuntimeError("apiserver down"))
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="7"))
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=3.0).watch())
        await settle()
        assert collector.events == []

        await clock.elapse(3.0)
        await settle()
        assert [type(event) for event in collector.events] == [ReplaceEvent]
        await collector.close()


class TestWatchEvents:
    # [CLASS DOC] Mapping of watch frames to source events and the reconnect behavior of the
    # WATCH phase. Reconnect waits the retry delay on the fake clock.
    async def test_added_and_modified_map_to_upsert(self):
        # [FUNCTION DOC] Scripts an added and a modified frame of the same pod; asserts events
        # after the replace are two upserts keyed by the pod name and the last one carries
        # version "3".
        """ADDED and MODIFIED both become UpsertEvent."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                raw_event("ADDED", make_pod("pod-0", resource_version="2")),
                raw_event("MODIFIED", make_pod("pod-0", resource_version="3")),
            ]
        )
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert collector.events[1:] == [
            UpsertEvent(key="pod-0", obj=collector.events[1].obj),
            UpsertEvent(key="pod-0", obj=collector.events[2].obj),
        ]
        assert collector.events[2].obj.metadata.resource_version == "3"
        await collector.close()

    async def test_deleted_maps_to_delete_with_tombstone(self):
        # [FUNCTION DOC] A deleted frame yields a delete event with the pod's key and the pod
        # itself as the last known object.
        """DELETED carries the last known object so the consumer can attribute it."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        pod = make_pod("pod-0", cell="cell-x", resource_version="5")
        api.stream_scripts.append([raw_event("DELETED", pod)])
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert collector.events[1] == DeleteEvent(key="pod-0", last_obj=pod)
        await collector.close()

    async def test_bookmark_emits_nothing_but_advances_the_cursor(self):
        # [FUNCTION DOC] A bookmark frame with version "50", then a clean stream end; after the
        # retry delay asserts only the replace event exists, the reconnect used version "50", and
        # there was just one list call.
        """BOOKMARK only refreshes the resource version."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("BOOKMARK", make_pod("pod-ignored", resource_version="50"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert api.stream_calls[-1]["resource_version"] == "50"
        assert len(api.list_calls) == 1
        await collector.close()

    async def test_raw_dict_bookmark_advances_the_cursor(self):
        # [FUNCTION DOC] Same as the bookmark test but with a mapping payload using the
        # camel-case version key "77"; asserts the reconnect uses "77" and no relist happens.
        """kubernetes_asyncio leaves BOOKMARK payloads as raw dicts; the cursor must still move."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("BOOKMARK", dict(metadata=dict(resourceVersion="77")))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert api.stream_calls[-1]["resource_version"] == "77"
        assert len(api.list_calls) == 1
        await collector.close()

    async def test_unknown_event_type_is_ignored(self):
        # [FUNCTION DOC] A frame of unknown type followed by an added frame; asserts only the
        # replace and one upsert (for the added pod) are emitted.
        """An unrecognized event type does not break the stream."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [raw_event("WEIRD", make_pod("pod-0")), raw_event("ADDED", make_pod("pod-1", resource_version="2"))]
        )
        api.stream_scripts.append(None)
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, UpsertEvent]
        assert collector.events[1].key == "pod-1"
        await collector.close()

    @pytest.mark.parametrize("event_type", ["ADDED", "MODIFIED", "DELETED"])
    async def test_an_event_whose_key_cannot_be_read_stops_the_watch(self, event_type):
        # [FUNCTION DOC] Parametrized over added, modified, deleted. The stream script has an
        # unreadable frame followed by a valid pod; asserts only the replace event is emitted
        # and only one list and one stream call happened (no retry or relist follows), and the
        # collector closes without having recorded an error.
        """A pod frame miles cannot read is the apiserver breaking its contract, so the watch stops
        instead of delivering a store that is silently short of a pod."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                unreadable_pod_frame(event_type),
                raw_event("ADDED", make_pod("pod-0", resource_version="6")),
            ]
        )
        api.stream_scripts.append(None)
        collector = EventCollector(make_reflector(api).watch())
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert len(api.list_calls) == 1
        assert len(api.stream_calls) == 1
        await collector.close()

    async def test_a_malformed_event_does_not_advance_the_cursor_past_itself(self):
        # [FUNCTION DOC] A stream whose only item is an unreadable modified frame, then a quiet
        # stream. After the retry delay asserts both stream calls used version "1" and only one
        # list happened.
        """A frame that never parsed carries no progress, so reading a cursor out of it would skip
        whatever the apiserver sent between the last good frame and this one."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([unreadable_pod_frame("MODIFIED")])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [call["resource_version"] for call in api.stream_calls] == ["1", "1"]
        assert len(api.list_calls) == 1
        await collector.close()

    async def test_watch_end_resumes_without_relisting(self):
        # [FUNCTION DOC] A stream with one added frame (version "9") that then ends. Before the
        # retry delay only one stream call exists; after it, events are replace then upsert,
        # one list call, and stream versions "1" then "9".
        """A watch that ends cleanly is reopened from the latest cursor, with no second ReplaceEvent."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ADDED", make_pod("pod-0", resource_version="9"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        assert [call["resource_version"] for call in api.stream_calls] == ["1"]

        await clock.elapse(1.0)
        await settle()
        assert [type(event) for event in collector.events] == [ReplaceEvent, UpsertEvent]
        assert len(api.list_calls) == 1
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "9"]
        await collector.close()

    async def test_a_list_that_cannot_be_converted_does_not_advance_the_cursor(self):
        # [FUNCTION DOC] First list raises (conversion failure), second succeeds with version
        # "2". Asserts no events until the delay, then one replace, two list calls and the
        # stream started from version "2".
        """A LIST that cannot be converted is retried as a LIST, not silently downgraded to a WATCH."""
        api = FakePodApi()
        api.list_pages.append(RuntimeError("a listed pod could not be converted"))
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        assert collector.events == []

        await clock.elapse(1.0)
        await settle()
        assert [type(event) for event in collector.events] == [ReplaceEvent]
        assert len(api.list_calls) == 2
        assert api.stream_calls[-1]["resource_version"] == "2"
        await collector.close()

    async def test_transient_watch_error_retries_without_relisting(self):
        # [FUNCTION DOC] A stream with an added frame (version "4") then a runtime error; after
        # the 2 s delay asserts one list call and stream versions "1" then "4".
        """A non-410 stream error backs off and resumes from the last cursor."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [raw_event("ADDED", make_pod("pod-0", resource_version="4")), RuntimeError("connection reset")]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=2.0).watch())
        await settle()
        assert len(api.stream_calls) == 1

        await clock.elapse(2.0)
        await settle()
        assert len(api.list_calls) == 1
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "4"]
        await collector.close()

    async def test_the_reconnect_cursor_follows_event_order_not_version_comparison(self):
        # [FUNCTION DOC] Frames with versions "55" then "32"; asserts the reconnect uses "32"
        # (the last seen, not the numerically largest).
        """Resource versions are opaque strings: the newest event wins even when its value looks smaller."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                raw_event("ADDED", make_pod("pod-0", resource_version="55")),
                raw_event("MODIFIED", make_pod("pod-0", resource_version="32")),
            ]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()

        await clock.elapse(1.0)
        await settle()
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "32"]
        await collector.close()

    async def test_a_non_expiry_error_event_closes_the_stream_before_reconnecting(self):
        # [FUNCTION DOC] An error frame with code 500 followed by an added pod. Asserts the
        # stream was closed once, one list, only the replace emitted (the pod after the error
        # is discarded), and after the delay both stream calls used version "1".
        """A raw ERROR frame that is not an expiry tears its stream down instead of leaking it."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [
                raw_event("ERROR", make_status(code=500, reason="InternalError")),
                raw_event("ADDED", make_pod("pod-0", resource_version="9")),
            ]
        )
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=2.0).watch())
        await settle()

        assert api.closed_streams == 1
        assert len(api.list_calls) == 1
        assert [type(event) for event in collector.events] == [ReplaceEvent]

        await clock.elapse(2.0)
        await settle()
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "1"]
        await collector.close()

    async def test_streams_are_closed_on_reopen(self):
        # [FUNCTION DOC] A stream that ends immediately: asserts it was closed and one stream
        # call exists; after the delay a second stream call exists.
        """Every finished watch stream is closed before the next is opened."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()

        assert api.closed_streams == 1
        assert len(api.stream_calls) == 1

        await clock.elapse(1.0)
        await settle()

        assert len(api.stream_calls) == 2
        await collector.close()


class TestExpiry:
    # [CLASS DOC] Dead or expired cursors force a fresh list (a whole-world replace), paced by
    # the retry delay.
    async def test_expired_error_event_triggers_a_relist(self):
        # [FUNCTION DOC] An error frame with code 410 ends the first stream; after the 1 s
        # delay asserts two replace events (pod-0 then pod-1), two list calls and the second
        # stream starting at version "200".
        """A 410 ERROR event forces a fresh LIST and a new whole-world ReplaceEvent."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=410))])
        api.list_pages.append(make_pod_list([make_pod("pod-1")], resource_version="200"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert [sorted(event.objects) for event in collector.events] == [["pod-0"], ["pod-1"]]
        assert len(api.list_calls) == 2
        assert api.stream_calls[-1]["resource_version"] == "200"
        await collector.close()

    async def test_an_expired_error_event_waits_for_the_retry_delay_before_relisting(self):
        # [FUNCTION DOC] With a 3 s delay, asserts one list call right after the 410, still one
        # after 2 s, and two after 3 s total.
        """410 gets no free instant relist: it is paced like every other LIST/WATCH cycle."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=410))])
        api.list_pages.append(make_pod_list([], resource_version="200"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=3.0).watch())
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(2.0)
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(1.0)
        await settle()
        assert len(api.list_calls) == 2
        await collector.close()

    async def test_expired_error_event_as_dict_triggers_a_relist(self):
        # [FUNCTION DOC] Same as the expiry test but the error payload is a mapping; asserts two
        # replace events and two list calls.
        """The raw dict form of a 410 status is recognized too."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", dict(code=410, reason="Expired"))])
        api.list_pages.append(make_pod_list([], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert len(api.list_calls) == 2
        await collector.close()

    async def test_expired_api_exception_triggers_a_relist(self):
        # [FUNCTION DOC] The stream raises a fake api exception with status 410; after 5 s
        # asserts two replace events and the new stream starting at version "300".
        """An ApiException(410) raised by the client forces a relist."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([FakeApiException(410)])
        api.list_pages.append(make_pod_list([], resource_version="300"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=5.0).watch())
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(5.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert api.stream_calls[-1]["resource_version"] == "300"
        await collector.close()

    async def test_a_too_large_resource_version_exception_triggers_a_relist(self):
        # [FUNCTION DOC] List version "5000", stream raises status 504; asserts two replace
        # events and stream versions "5000" then "100".
        """A cursor from the future is unusable, so it is dropped for a fresh LIST instead of retried forever."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="5000"))
        api.stream_scripts.append([FakeApiException(504)])
        api.list_pages.append(make_pod_list([], resource_version="100"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        assert [call["resource_version"] for call in api.stream_calls] == ["5000", "100"]
        await collector.close()

    async def test_a_too_large_resource_version_error_event_triggers_a_relist(self):
        # [FUNCTION DOC] Same as above but via an error frame with code 504 reason "Timeout";
        # asserts two list calls and stream versions "5000" then "100".
        """The apiserver may report the same condition as an ERROR frame rather than a status exception."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="5000"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=504, reason="Timeout"))])
        api.list_pages.append(make_pod_list([], resource_version="100"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert len(api.list_calls) == 2
        assert [call["resource_version"] for call in api.stream_calls] == ["5000", "100"]
        await collector.close()

    async def test_non_expiry_error_event_is_a_stream_failure(self):
        # [FUNCTION DOC] An error frame with code 500; after the delay asserts no relist (one
        # list call) and both stream calls at version "1".
        """A non-410 ERROR event is treated as a failure, not as an expiry."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=500, reason="InternalError"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=2.0).watch())
        await settle()
        assert len(api.list_calls) == 1

        await clock.elapse(2.0)
        await settle()
        assert len(api.list_calls) == 1
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "1"]
        await collector.close()

    async def test_relist_replays_the_full_world(self):
        # [FUNCTION DOC] After an expiry, the second replace event holds exactly the pods of the
        # second list (pod-1 and pod-2).
        """After expiry the consumer receives the complete current state, not a delta."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0"), make_pod("pod-1")], resource_version="1"))
        api.stream_scripts.append([FakeApiException(410)])
        api.list_pages.append(make_pod_list([make_pod("pod-1"), make_pod("pod-2")], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert sorted(collector.events[1].objects) == ["pod-1", "pod-2"]
        await collector.close()


class TestCursor:
    # [CLASS DOC] How the reconnect cursor (last resource version) is maintained.
    @pytest.mark.parametrize("event_type", ["MODIFIED", "DELETED"])
    async def test_every_event_kind_advances_the_reconnect_cursor(self, event_type):
        # [FUNCTION DOC] Parametrized over modified and deleted frames at version "7"; after the
        # delay asserts stream versions "1" then "7".
        """The cursor follows the newest observed object, whatever the event type."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event(event_type, make_pod("pod-0", resource_version="7"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [call["resource_version"] for call in api.stream_calls] == ["1", "7"]
        await collector.close()

    async def test_event_without_resource_version_keeps_the_previous_cursor(self):
        # [FUNCTION DOC] A bookmark with no metadata at list version "11"; asserts both stream
        # calls used "11" and only one list happened.
        """An object with no metadata must not reset the cursor."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="11"))
        api.stream_scripts.append([raw_event("BOOKMARK", dict(other="thing"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [call["resource_version"] for call in api.stream_calls] == ["11", "11"]
        assert len(api.list_calls) == 1
        await collector.close()

    async def test_expiry_discards_the_rest_of_the_stale_stream(self):
        # [FUNCTION DOC] A 410 error frame followed by an added pod in the same stream; asserts
        # only two replace events appear (the stale pod is never delivered).
        """Events queued behind a 410 belong to a dead cursor and must not be delivered."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(
            [raw_event("ERROR", make_status(code=410)), raw_event("ADDED", make_pod("pod-stale"))]
        )
        api.list_pages.append(make_pod_list([], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert [type(event) for event in collector.events] == [ReplaceEvent, ReplaceEvent]
        await collector.close()

    async def test_watch_after_a_relist_reconnects_without_listing_again(self):
        # [FUNCTION DOC] Expiry via exception, relist at "100", then a stream with an added pod
        # at "101" that ends. After two delays asserts two list calls total and stream versions
        # "1", "100", "101".
        """The expired flag is cleared by the relist, so the next clean end is an ordinary reconnect."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([FakeApiException(410)])
        api.list_pages.append(make_pod_list([], resource_version="100"))
        api.stream_scripts.append([raw_event("ADDED", make_pod("pod-0", resource_version="101"))])
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()
        await clock.elapse(1.0)
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert len(api.list_calls) == 2
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "100", "101"]
        await collector.close()


class TestStreamClosing:
    # [CLASS DOC] Active closing of watch streams.
    async def test_an_expired_stream_is_closed_before_the_relist(self):
        # [FUNCTION DOC] After a 410 frame, asserts the stream generator was explicitly closed
        # exactly once.
        """Breaking out on 410 must actively close the watch, not wait for GC."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append([raw_event("ERROR", make_status(code=410)), raw_event("ADDED", make_pod("pod-0"))])
        api.list_pages.append(make_pod_list([], resource_version="2"))
        api.stream_scripts.append(None)
        clock = FakeClock()
        collector = EventCollector(make_reflector(api, clock=clock, retry_delay=1.0).watch())
        await settle()

        assert api.aclosed_streams == 1
        await collector.close()

    async def test_a_cancelled_watch_leaves_no_open_stream(self):
        # [FUNCTION DOC] With an open quiet stream, asserts nothing is closed yet, then closes
        # the collector and asserts the stream was closed once.
        """Cancelling the consumer must not leave the watch open."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        api.stream_scripts.append(None)
        collector = EventCollector(make_reflector(api).watch())
        await settle()
        assert api.closed_streams == 0

        await collector.close()
        await settle()
        assert api.closed_streams == 1


class TestLifecycle:
    # [CLASS DOC] Starting, cancelling and restarting the reflector's stream.
    async def test_cancellation_stops_the_generator(self):
        # [FUNCTION DOC] Closes the collector after the list; asserts no error was recorded.
        """Cancelling the consumer tears the reflector down without error."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([], resource_version="1"))
        collector = EventCollector(make_reflector(api).watch())
        await settle()
        await collector.close()

        assert collector.error is None

    async def test_each_watch_call_is_an_independent_stream(self):
        # [FUNCTION DOC] Calls watch twice on one reflector (the first collector closed before
        # the second starts); asserts the second yields a fresh replace, and the stream calls
        # used versions "1" then "2".
        """watch() can be called again to get a fresh LIST-then-WATCH stream."""
        api = FakePodApi()
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="1"))
        api.stream_scripts.append(None)
        api.list_pages.append(make_pod_list([make_pod("pod-0")], resource_version="2"))
        api.stream_scripts.append(None)
        reflector = make_reflector(api)

        first = EventCollector(reflector.watch())
        await settle()
        await first.close()
        second = EventCollector(reflector.watch())
        await settle()

        assert [type(event) for event in second.events] == [ReplaceEvent]
        assert [call["resource_version"] for call in api.stream_calls] == ["1", "2"]
        await second.close()


class TestKubernetesAsyncioPodApi:
    # [CLASS DOC] The real adapter over a fake core api and a fake watch library.
    async def test_list_pods_delegates_to_core_v1_api(self):
        # [FUNCTION DOC] With a fake core api returning one wire pod and version "100", asserts
        # the adapter returns the equal pod list page and called the namespaced list once with
        # namespace and label selector keywords.
        """The adapter forwards LIST to CoreV1Api.list_namespaced_pod."""
        calls: list[dict[str, Any]] = []

        class _CoreV1Api:
            # [CLASS DOC] HELPER: Fake core api recording list calls.
            async def list_namespaced_pod(self, **kwargs: Any) -> Any:
                # [FUNCTION DOC] HELPER: Records the keywords and returns one wire pod with list
                # version "100".
                calls.append(kwargs)
                return SimpleNamespace(items=[wire_pod("pod-0")], metadata=SimpleNamespace(resource_version="100"))

        api = KubernetesAsyncioPodApi(core_v1_api=_CoreV1Api())
        page = await api.list_pods(namespace="ns", label_selector="a=b")

        assert page == PodListPage(pods=[make_pod("pod-0")], resource_version="100")
        assert calls == [dict(namespace="ns", label_selector="a=b")]

    async def test_list_pods_refuses_an_item_that_is_not_a_pod(self):
        # [FUNCTION DOC] A fake core api returns a page whose only item is a string; asserts the
        # adapter's list raises a validation error.
        """A page miles cannot read must fail the LIST rather than reach the store half-converted."""

        class _CoreV1Api:
            # [CLASS DOC] HELPER: Fake core api returning an invalid page.
            async def list_namespaced_pod(self, **kwargs: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns a page whose item list holds a plain string.
                return SimpleNamespace(items=["not a pod"], metadata=SimpleNamespace(resource_version="100"))

        api = KubernetesAsyncioPodApi(core_v1_api=_CoreV1Api())

        with pytest.raises(ValidationError):
            await api.list_pods(namespace="ns", label_selector="a=b")

    async def test_stream_pods_forwards_watch_options_and_closes_the_watch(self, monkeypatch):
        # [FUNCTION DOC] Installs the fake client library, streams pods to completion and
        # asserts one modified event whose pod equals the wire pod and which does not reject the
        # cursor; the fake watch got the core api's list function and keywords namespace,
        # selector, resource version "42", timeout 7 and bookmarks allowed; the watch was
        # closed once.
        """The adapter drives kubernetes_asyncio's Watch and closes it, which has close() and no aclose()."""
        watch_module, state = _install_fake_kubernetes_asyncio(monkeypatch)
        _ = watch_module
        list_namespaced_pod = object()

        class _CoreV1Api:
            # [CLASS DOC] HELPER: Empty fake core api; the list function is attached afterwards.
            pass

        core_v1_api = _CoreV1Api()
        core_v1_api.list_namespaced_pod = list_namespaced_pod
        api = KubernetesAsyncioPodApi(core_v1_api=core_v1_api)

        events = []
        async for event in api.stream_pods(
            namespace="ns", label_selector="a=b", resource_version="42", timeout_seconds=7
        ):
            events.append(event)

        assert len(events) == 1
        assert events[0].type == "MODIFIED"
        assert events[0].pod == make_pod("pod-from-the-wire")
        assert events[0].rejects_cursor is False
        assert state["func"] is list_namespaced_pod
        assert state["kwargs"] == dict(
            namespace="ns",
            label_selector="a=b",
            resource_version="42",
            timeout_seconds=7,
            allow_watch_bookmarks=True,
        )
        assert state["closed"] == 1

    async def test_stream_pods_closes_the_watch_when_the_consumer_stops_early(self, monkeypatch):
        # [FUNCTION DOC] Reads one item from the stream then closes the generator; asserts the
        # fake watch was closed exactly once.
        """Abandoning the stream must not leak the underlying watch connection."""
        _, state = _install_fake_kubernetes_asyncio(monkeypatch)

        class _CoreV1Api:
            # [CLASS DOC] HELPER: Empty fake core api; the list function is attached afterwards.
            pass

        core_v1_api = _CoreV1Api()
        core_v1_api.list_namespaced_pod = object()
        api = KubernetesAsyncioPodApi(core_v1_api=core_v1_api)

        stream = api.stream_pods(namespace="ns", label_selector="a=b", resource_version="1", timeout_seconds=1)
        await stream.__anext__()
        await stream.aclose()

        assert state["closed"] == 1


class TestReflectorFeedsReconcileLoop:
    # [CLASS DOC] Integration of the reflector as the source of a reconcile loop.
    async def test_pod_stream_drives_cell_level_reconciles(self):
        # [FUNCTION DOC] Initial pods in cells a and b, a new pod in cell a, then an expiry whose
        # relist drops pod-1 and pod-2. Starts a loop with a recording reconcile keyed by cell.
        # Asserts the reconciled keys are exactly cell-a and cell-b, cell-b reconciled at least
        # twice (the missed delete is noticed), the store holds only pod-0 for cell-a and
        # nothing for cell-b.
        """End to end: pods from the reflector become cell reconciles, including a missed delete."""
        api = FakePodApi()
        api.list_pages.append(
            make_pod_list([make_pod("pod-0", cell="cell-a"), make_pod("pod-1", cell="cell-b")], resource_version="1")
        )
        api.stream_scripts.append(
            [raw_event("ADDED", make_pod("pod-2", cell="cell-a", resource_version="2")), FakeApiException(410)]
        )
        api.list_pages.append(make_pod_list([make_pod("pod-0", cell="cell-a")], resource_version="3"))
        api.stream_scripts.append(None)

        clock = FakeClock()
        reflector = make_reflector(api, clock=clock)
        recorder = ParentKeyRecorder()
        loop = ReconcileLoop(source=reflector.watch, reconcile=recorder, key_map=pod_cell, clock=clock)
        await loop.start()
        await settle()
        await clock.elapse(1.0)
        await settle()

        assert sorted(set(recorder.parent_keys)) == ["cell-a", "cell-b"]
        assert recorder.parent_keys.count("cell-b") >= 2
        assert [pod.metadata.name for pod in loop.get_by_parent("cell-a")] == ["pod-0"]
        assert loop.get_by_parent("cell-b") == []
        await loop.stop()

    async def test_initial_sync_barrier_holds_until_the_reflector_lists(self):
        # [FUNCTION DOC] The first list fails, the second succeeds, retry delay 5 s. Asserts the
        # loop's start does not finish and nothing is reconciled until 5 s pass; then start
        # completes with exactly one reconcile of cell-a.
        """ReconcileLoop.start() waits for the reflector's LIST to complete."""
        api = FakePodApi()
        api.list_pages.append(RuntimeError("apiserver down"))
        api.list_pages.append(make_pod_list([make_pod("pod-0", cell="cell-a")], resource_version="1"))
        api.stream_scripts.append(None)

        clock = FakeClock()
        reflector = make_reflector(api, clock=clock, retry_delay=5.0)
        recorder = ParentKeyRecorder()
        loop = ReconcileLoop(source=reflector.watch, reconcile=recorder, key_map=pod_cell, clock=clock)
        start_task = asyncio.create_task(loop.start())
        await settle()
        assert not start_task.done()
        assert recorder.parent_keys == []

        await clock.elapse(5.0)
        await settle()
        await start_task
        assert recorder.parent_keys == ["cell-a"]
        await loop.stop()
