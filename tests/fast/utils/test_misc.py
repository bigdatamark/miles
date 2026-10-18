# [MODULE DOC]
# Tests for the miscellaneous utilities module. Covered: a dictionary key projection helper (order
# follows requested keys, duplicates collapse, a missing key raises a key error and logs); GPU
# UUID lookup through a fake NVML module injected into the module table (order preserved, all-none
# on any failure); the node probe mixin (external IP from an environment variable, node IP with a
# host-IP override that is published verbatim and re-read on every call, free-port block scanning
# with wrap-around and exhaustion errors, port availability probe, and conversion of physical GPU
# ids to local ids under CUDA or ROCm visibility masks including edge cases); the periodic ticker
# (keeps calling, survives failing calls, dispose stops it, idempotent dispose, does not swallow
# caller cancellation); the cancel-and-await task helper (swallows its own cancellation, waits for
# unwinding, propagates caller cancellation and deadlines, reports teardown errors); a strict
# dictionary merge that rejects conflicting keys; and a mutable box value holder.
import asyncio
import logging
import socket
import sys
from contextlib import ExitStack
from dataclasses import dataclass

import pytest

from miles.utils import misc
from miles.utils.http_utils import MILES_HOST_IP_ENV
from miles.utils.misc import (
    MutableBox,
    NodeProbeMixin,
    SimpleTicker,
    cancel_and_await_task,
    filter_keys,
    get_current_node_ip,
    get_free_port,
    get_gpu_uuids,
    merge_asserting_consistency,
)


class TestFilterKeys:
    # [CLASS DOC] Tests of the key projection helper.
    def test_projects_dict_by_keys(self):
        # [FUNCTION DOC] Projecting keys a and c from a three-key mapping gives exactly those entries.
        """filter_keys returns only the requested keys with their values."""
        d = {"a": 1, "b": 2, "c": 3}
        assert filter_keys(d, ["a", "c"]) == {"a": 1, "c": 3}

    def test_empty_interest_keys_returns_empty_dict(self):
        # [FUNCTION DOC] An empty key list gives an empty mapping.
        """An empty interest list yields an empty dict regardless of input."""
        assert filter_keys({"a": 1, "b": 2}, []) == {}

    def test_preserves_interest_keys_order(self):
        # [FUNCTION DOC] Requesting keys c then a gives a result whose key order is c, a.
        """Result key order follows interest_keys, not the source dict order."""
        d = {"a": 1, "b": 2, "c": 3}
        assert list(filter_keys(d, ["c", "a"]).keys()) == ["c", "a"]

    def test_full_subset_returns_all_entries(self):
        # [FUNCTION DOC] Requesting every key returns an equal mapping.
        """Requesting every key returns the whole projection."""
        d = {"x": 10, "y": 20}
        assert filter_keys(d, ["x", "y"]) == {"x": 10, "y": 20}

    def test_duplicate_interest_key_collapses_to_single_entry(self):
        # [FUNCTION DOC] Requesting key a twice gives one entry for a.
        """A repeated interest key produces a single dict entry."""
        d = {"a": 1, "b": 2}
        assert filter_keys(d, ["a", "a"]) == {"a": 1}

    def test_missing_key_raises_key_error_and_logs(self, caplog):
        # [FUNCTION DOC] Requesting an absent key raises a key error, and at least one error-level
        # log record from the misc module logger mentions the helper's name.
        """A missing key raises KeyError and logs the error with context."""
        d = {"a": 1}
        with caplog.at_level(logging.ERROR, logger="miles.utils.misc"):
            with pytest.raises(KeyError):
                filter_keys(d, ["a", "missing"])
        assert any("filter_keys" in record.message for record in caplog.records)


@dataclass(frozen=True)
class _FakeGpuHandle:
    # [CLASS DOC] HELPER: Immutable fake GPU handle holding only the device index.
    index: int


@dataclass(frozen=True)
class _FakeNvmlUuid:
    # [CLASS DOC] HELPER: Immutable fake NVML UUID object whose string conversion yields its text.
    text: str

    def __str__(self) -> str:
        # [FUNCTION DOC] HELPER: Return the stored text.
        return self.text


class _FakeNvml:
    # [CLASS DOC] HELPER: Fake NVML library module exposing the three entry points the code under test
    # uses (init, handle by index, UUID by handle). Configurable with a mapping from index to UUID
    # text, an optional error raised by init, and a set of indices whose UUID lookup raises.
    def __init__(
        self,
        *,
        uuid_by_index: dict[int, str],
        init_error: Exception | None = None,
        uuid_error_indices: frozenset[int] = frozenset(),
    ) -> None:
        # [FUNCTION DOC] HELPER: Store the configuration.
        self._uuid_by_index = uuid_by_index
        self._init_error = init_error
        self._uuid_error_indices = uuid_error_indices

    def nvmlInit(self) -> None:
        # [FUNCTION DOC] HELPER: Raise the configured init error if any; otherwise do nothing.
        if self._init_error is not None:
            raise self._init_error

    def nvmlDeviceGetHandleByIndex(self, index: int) -> _FakeGpuHandle:
        # [FUNCTION DOC] HELPER: Return a fake handle for the index.
        return _FakeGpuHandle(index=index)

    def nvmlDeviceGetUUID(self, handle: _FakeGpuHandle) -> _FakeNvmlUuid:
        # [FUNCTION DOC] HELPER: Raise a runtime error if the handle's index is in the failing set;
        # otherwise return a fake UUID object with the configured text (a missing index raises a
        # key error).
        if handle.index in self._uuid_error_indices:
            raise RuntimeError(f"nvml uuid lookup failed for {handle.index}")
        return _FakeNvmlUuid(text=self._uuid_by_index[handle.index])


class TestGetGpuUuids:
    # [CLASS DOC] Tests of the module-level GPU UUID lookup, with the NVML module faked via the
    # module table.
    def test_get_gpu_uuids_returns_requested_nvml_uuids_in_order(self, monkeypatch) -> None:
        # [FUNCTION DOC] Look up indices 2 then 0 with a fake NVML; the result is the two UUID texts
        # in that request order, all plain strings.
        """Each requested gpu index is resolved through NVML, coerced to str, and answered in request order."""
        fake_nvml = _FakeNvml(uuid_by_index={0: "GPU-zero", 1: "GPU-one", 2: "GPU-two"})
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        uuids = get_gpu_uuids([2, 0])

        assert uuids == ["GPU-two", "GPU-zero"]
        assert all(isinstance(uuid, str) for uuid in uuids)

    def test_get_gpu_uuids_returns_none_per_gpu_when_nvml_fails(self, monkeypatch) -> None:
        # [FUNCTION DOC] With NVML init raising, looking up three indices returns a list of three
        # none values instead of raising.
        """A failing NVML init is swallowed and answered with exactly one None per requested gpu."""
        fake_nvml = _FakeNvml(uuid_by_index={}, init_error=RuntimeError("nvml unavailable"))
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        assert get_gpu_uuids([0, 1, 3]) == [None, None, None]

    def test_get_gpu_uuids_returns_all_none_when_one_lookup_fails(self, monkeypatch) -> None:
        # [FUNCTION DOC] When the lookup of index 1 raises, requesting indices 0 and 1 yields [none,
        # none] (the whole result is none, not just the failing entry).
        """A single failing uuid lookup yields all-None rather than a partial or short list."""
        fake_nvml = _FakeNvml(
            uuid_by_index={0: "GPU-zero", 1: "GPU-one"},
            uuid_error_indices=frozenset({1}),
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_nvml)

        assert get_gpu_uuids([0, 1]) == [None, None]


class TestNodeProbeMixin:
    # [CLASS DOC] Tests of the node probe mixin's static helpers (external IP, node IP, free ports,
    # port availability, GPU id mapping and UUIDs), mostly driven through environment variables.
    def test_get_node_external_ip_is_absent_until_the_node_sets_it(self, monkeypatch):
        # [FUNCTION DOC] The external IP is none when its variable is unset or empty, and equals the
        # value when set to an address.
        """A node without the env var reports None, which keeps its placed address in use."""
        monkeypatch.delenv("MILES_NODE_EXTERNAL_IP", raising=False)
        assert NodeProbeMixin._get_node_external_ip() is None

        monkeypatch.setenv("MILES_NODE_EXTERNAL_IP", "")
        assert NodeProbeMixin._get_node_external_ip() is None

        monkeypatch.setenv("MILES_NODE_EXTERNAL_IP", "100.64.0.7")
        assert NodeProbeMixin._get_node_external_ip() == "100.64.0.7"

    def test_get_node_ip_returns_nonempty_string(self):
        # [FUNCTION DOC] The node IP probe returns a non-empty string (real environment, no mocks).
        """The node ip probe answers with a usable address string."""
        node_ip = NodeProbeMixin._get_node_ip()
        assert isinstance(node_ip, str) and node_ip

    def test_get_free_port_block_returns_bindable_consecutive_ports(self) -> None:
        # [FUNCTION DOC] Find a free block start from port 15000 with ten consecutive ports, then
        # occupy the port at offset four with a listening socket; the block finder started there
        # for five ports must return a block of five that avoids the occupied port, and all five
        # ports can be bound at once (sockets held in an exit stack). Binds real sockets.
        """A block request returns five ports that can be bound simultaneously."""
        candidate_start: int = get_free_port(start_port=15000, consecutive=10)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_socket:
            occupied_socket.bind(("", candidate_start + 4))
            occupied_socket.listen()
            first_port: int = NodeProbeMixin._get_free_port_block(start_port=candidate_start, count=5)

            with ExitStack() as stack:
                for port in range(first_port, first_port + 5):
                    available_socket: socket.socket = stack.enter_context(
                        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    )
                    available_socket.bind(("", port))

    def test_the_scan_wraps_instead_of_walking_past_the_last_port(self, monkeypatch):
        # [FUNCTION DOC] With availability patched so only port 20005 is free, scanning for one port
        # from 65530 wraps around and returns 20005.
        """The allocator cursor only grows, so an unbounded scan spun forever and wedged its caller."""
        free_port = 20005
        monkeypatch.setattr(misc, "is_port_available", lambda port: port == free_port)

        assert get_free_port(start_port=65530, consecutive=1) == free_port

    def test_a_range_with_nothing_free_raises(self, monkeypatch):
        # [FUNCTION DOC] With no port available, scanning for four consecutive ports raises a runtime
        # error whose message mentions consecutive free ports.
        """Reporting exhaustion is the whole point: the old loop incremented past 65535 forever."""
        monkeypatch.setattr(misc, "is_port_available", lambda port: False)

        with pytest.raises(RuntimeError, match="consecutive free ports"):
            get_free_port(start_port=65000, consecutive=4)

    def test_a_block_that_cannot_fit_below_the_last_port_is_rejected(self, monkeypatch):
        # [FUNCTION DOC] With every port available, asking for two consecutive ports starting at
        # 65535 raises an assertion error (the block runs off the end).
        """Asking for a block that runs off the end is a caller bug, not something to scan for."""
        monkeypatch.setattr(misc, "is_port_available", lambda port: True)

        with pytest.raises(AssertionError):
            get_free_port(start_port=65535, consecutive=2)

    def test_is_port_available_separates_a_bound_port_from_a_free_one(self) -> None:
        # [FUNCTION DOC] Find two free consecutive ports from 15000; bind and listen on the first.
        # The availability probe reports false for the occupied port and true for the next one.
        """The probe answers False exactly for a port another process is already listening on."""
        free_port: int = get_free_port(start_port=15000, consecutive=2)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied_socket:
            occupied_socket.bind(("", free_port))
            occupied_socket.listen()

            assert NodeProbeMixin._is_port_available(port=free_port) is False
            assert NodeProbeMixin._is_port_available(port=free_port + 1) is True

    def test_get_node_ip_honours_the_host_ip_override(self, monkeypatch):
        # [FUNCTION DOC] With the host-IP environment variable set to an address, the node IP probe
        # returns that address.
        """Deployments where the reachable address is not the ray node ip pin it through this env var."""
        monkeypatch.setenv("MILES_HOST_IP", "10.20.30.40")

        assert NodeProbeMixin._get_node_ip() == "10.20.30.40"

    def test_get_node_ip_falls_back_to_the_ray_node_ip(self, monkeypatch):
        # [FUNCTION DOC] With the override unset, the node IP probe equals the current node IP
        # helper's answer.
        """Without an override the worker reports the address ray placed it on."""
        monkeypatch.delenv("MILES_HOST_IP", raising=False)

        assert NodeProbeMixin._get_node_ip() == get_current_node_ip()

    def test_get_node_ip_falls_back_when_the_override_is_set_but_empty(self, monkeypatch):
        # [FUNCTION DOC] With the override set to an empty string, the probe falls back to the
        # current node IP helper's answer.
        """An override exported as an empty string carries no address, so publishing it would advertise nothing."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "")

        assert NodeProbeMixin._get_node_ip() == get_current_node_ip()

    def test_get_node_ip_follows_an_override_that_changes_between_calls(self, monkeypatch):
        # [FUNCTION DOC] Changing the override value between two calls changes the probe's answer
        # (no caching).
        """The probe runs inside the worker at allocation time, so a cached first answer would outlive its address."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.20.30.40")
        assert NodeProbeMixin._get_node_ip() == "10.20.30.40"

        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.20.30.41")

        assert NodeProbeMixin._get_node_ip() == "10.20.30.41"

    def test_get_node_ip_publishes_a_non_numeric_override_verbatim(self, monkeypatch):
        # [FUNCTION DOC] An override holding a DNS name is returned unchanged.
        """Deployments pin a routable dns name here, so resolving or validating it would drop the reachable address."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "miles-worker-0.miles.svc.cluster.local")

        assert NodeProbeMixin._get_node_ip() == "miles-worker-0.miles.svc.cluster.local"

    def test_to_local_gpu_ids_passes_ids_through_without_a_visibility_mask(self, monkeypatch):
        # [FUNCTION DOC] With neither the CUDA nor the HIP visibility variable set, ids [4, 5] are
        # returned unchanged.
        """A worker that sees every gpu already got local ids, so remapping them would corrupt them."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[4, 5]) == [4, 5]

    def test_to_local_gpu_ids_remaps_physical_ids_under_a_visibility_mask(self, monkeypatch):
        # [FUNCTION DOC] With the CUDA mask "4,5,6,7", physical ids [4, 6] map to positions [0, 2]
        # within the mask.
        """The probe must run where the mask is, otherwise a masked worker is told to use a device it cannot see."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[4, 6]) == [0, 2]

    def test_to_local_gpu_ids_rejects_an_id_outside_the_visibility_mask(self, monkeypatch):
        # [FUNCTION DOC] With the CUDA mask "4,5", id 7 raises a runtime error saying it is not valid
        # under the CUDA visibility variable.
        """An id belonging to no visible device is a placement bug and must not be silently passed on."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

        with pytest.raises(RuntimeError, match="not valid under CUDA_VISIBLE_DEVICES"):
            NodeProbeMixin._to_local_gpu_ids(gpu_ids=[7])

    def test_to_local_gpu_ids_falls_back_to_the_rocm_visibility_mask(self, monkeypatch):
        # [FUNCTION DOC] With only the HIP mask "4,5,6,7" set, ids [5, 7] map to [1, 3].
        """On an AMD node the mask lives in HIP_VISIBLE_DEVICES, and ignoring it hands out a foreign id."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5,6,7")

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[5, 7]) == [1, 3]

    def test_to_local_gpu_ids_prefers_the_cuda_mask_when_both_masks_are_set(self, monkeypatch):
        # [FUNCTION DOC] With CUDA mask "6,7" and HIP mask "4,5", id 7 maps to [1] (the CUDA mask
        # wins).
        """Torch honours CUDA_VISIBLE_DEVICES first, so resolving against the rocm mask would disagree with it."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6,7")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[7]) == [1]

    def test_to_local_gpu_ids_accepts_ids_that_are_already_local_to_the_mask(self, monkeypatch):
        # [FUNCTION DOC] With the CUDA mask "4,5", ids [0, 1] (already local positions) are accepted
        # and returned unchanged.
        """A caller that resolved its ids elsewhere must not be rejected as out of range."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[0, 1]) == [0, 1]

    def test_to_local_gpu_ids_keeps_the_order_of_the_ids_it_was_given(self, monkeypatch):
        # [FUNCTION DOC] With mask "4,5,6,7", ids [7, 5] map to [3, 1], preserving input order.
        """The first entry becomes the engine base id, so reordering silently moves the engine to another device."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[7, 5]) == [3, 1]

    def test_to_local_gpu_ids_answers_a_worker_that_owns_no_gpu_with_an_empty_list(self, monkeypatch):
        # [FUNCTION DOC] With a mask set, an empty id list maps to an empty list.
        """Every worker is probed, including the gpu-less ones, so no gpu must not mean no answer."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[]) == []

    def test_to_local_gpu_ids_ignores_blank_entries_in_the_visibility_mask(self, monkeypatch):
        # [FUNCTION DOC] With mask "4,5," (trailing comma), id 5 maps to [1] without error.
        """A trailing comma is legal in the mask and must not crash the probe that parses it."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,")

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[5]) == [1]

    def test_to_local_gpu_ids_treats_an_empty_visibility_mask_as_no_mask(self, monkeypatch):
        # [FUNCTION DOC] With an empty CUDA mask and no HIP mask, id 3 passes through unchanged.
        """An empty mask spans no id space to map into, so the ids have to pass through untouched."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)

        assert NodeProbeMixin._to_local_gpu_ids(gpu_ids=[3]) == [3]

    def test_get_gpu_uuids_returns_one_entry_per_gpu(self):
        # [FUNCTION DOC] The mixin's UUID probe for three indices (real environment) returns three
        # entries, each either none or a string.
        """The uuid probe is best-effort: without NVML it still answers per gpu."""
        uuids = NodeProbeMixin._get_gpu_uuids([0, 1, 2])
        assert len(uuids) == 3
        assert all(uuid is None or isinstance(uuid, str) for uuid in uuids)


async def _append(calls: list[int]) -> None:
    # [FUNCTION DOC] HELPER: Async function that appends the number one to the given list.
    calls.append(1)


class TestSimpleTicker:
    # [CLASS DOC] Tests of the periodic async ticker.
    async def test_it_keeps_calling_its_function(self):
        # [FUNCTION DOC] A ticker with zero interval running for 20 ms calls its function more than
        # once before being disposed.
        """The ticked work only makes progress while the loop keeps coming back."""
        calls: list[int] = []

        ticker = SimpleTicker(lambda: _append(calls), interval_seconds=0.0)
        await asyncio.sleep(0.02)
        await ticker.dispose()

        assert len(calls) > 1

    async def test_it_survives_a_failing_call(self):
        # [FUNCTION DOC] A ticker whose function records a call then raises still gets called more
        # than once within 20 ms.
        """A raising sweep must not silently kill the loop for every later round."""
        calls: list[int] = []

        async def _boom() -> None:
            # [FUNCTION DOC] HELPER: Record a call, then raise a runtime error.
            calls.append(1)
            raise RuntimeError("tick exploded")

        ticker = SimpleTicker(_boom, interval_seconds=0.0)
        await asyncio.sleep(0.02)
        await ticker.dispose()

        assert len(calls) > 1

    async def test_dispose_stops_the_loop(self):
        # [FUNCTION DOC] After disposing the ticker and waiting another 20 ms, the call count has
        # not changed.
        """A surviving loop would keep working after its owner is gone."""
        calls: list[int] = []

        ticker = SimpleTicker(lambda: _append(calls), interval_seconds=0.0)
        await asyncio.sleep(0.02)
        await ticker.dispose()
        calls_after_dispose = len(calls)
        await asyncio.sleep(0.02)

        assert len(calls) == calls_after_dispose

    async def test_a_task_that_died_of_its_own_error_reports_it_on_dispose(self):
        # [FUNCTION DOC] A task that already failed with a runtime error ("ticker died") is passed to
        # the cancel-and-await helper, which re-raises that error.
        """Cancelling a task that already failed must surface the failure, not hide it behind the cancellation."""

        async def _explode() -> None:
            # [FUNCTION DOC] HELPER: Raise a runtime error ("ticker died").
            raise RuntimeError("ticker died")

        task = asyncio.create_task(_explode())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="ticker died"):
            await cancel_and_await_task(task)

    async def test_dispose_does_not_swallow_the_callers_cancellation(self):
        # [FUNCTION DOC] Run dispose inside a task and cancel that task right away; the task ends
        # up cancelled and its post-dispose marker was never recorded.
        """A teardown that eats its caller's cancellation lets the shutdown path run on regardless."""
        ticker = SimpleTicker(lambda: _append([]), interval_seconds=1000.0)
        finished: list[str] = []

        async def _dispose() -> None:
            # [FUNCTION DOC] HELPER: Dispose the ticker, then record that dispose returned.
            await ticker.dispose()
            finished.append("returned")

        task = asyncio.create_task(_dispose())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert task.cancelled() and finished == []

    async def test_disposing_twice_is_harmless(self):
        # [FUNCTION DOC] Disposing a ticker twice in a row does not raise.
        """Teardown paths overlap, so a second dispose must not raise."""
        ticker = SimpleTicker(lambda: _append([]), interval_seconds=0.0)

        await ticker.dispose()
        await ticker.dispose()


class TestCancelAndAwaitTask:
    # [CLASS DOC] Tests of the helper that cancels a task and waits for it to finish.
    async def test_it_swallows_the_cancellation_of_the_task_it_cancelled(self):
        # [FUNCTION DOC] Cancelling a task blocked on an event returns normally and the task ends
        # cancelled.
        """The cancellation the helper itself asked for is teardown noise, not something to raise at the caller."""
        task = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)

        await cancel_and_await_task(task)

        assert task.cancelled()

    async def test_it_returns_only_after_the_task_finished_unwinding(self):
        # [FUNCTION DOC] A task whose cleanup yields once before recording a marker: after the helper
        # returns, the task is done and the marker was recorded.
        """Returning while the cancelled task is still unwinding lets it act once more after its owner is gone."""
        unwound: list[str] = []

        async def _slow_teardown() -> None:
            # [FUNCTION DOC] HELPER: Block on an event; in the cleanup step yield once to the loop and
            # record the "cleaned" marker.
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0)
                unwound.append("cleaned")

        task = asyncio.create_task(_slow_teardown())
        await asyncio.sleep(0)

        await cancel_and_await_task(task)

        assert task.done() and unwound == ["cleaned"]

    async def test_it_requests_teardown_before_propagating_the_callers_cancellation(self):
        # [FUNCTION DOC] The target task absorbs its first cancellation and waits for a release
        # event. A caller task runs the helper; once the target saw the cancel, the caller itself is
        # cancelled. After releasing and awaiting the target: the caller ended cancelled and never
        # returned normally.
        """Being cancelled mid-teardown must still leave the task cancelled while the caller's cancellation travels on."""
        observed_cancel = asyncio.Event()
        release = asyncio.Event()
        returned: list[str] = []

        async def _absorbs_the_first_cancel() -> None:
            # [FUNCTION DOC] HELPER: Block on an event; on cancellation, signal that the cancel was
            # observed and then wait for the release event (swallowing the cancellation).
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                observed_cancel.set()
                await release.wait()

        task = asyncio.create_task(_absorbs_the_first_cancel())
        await asyncio.sleep(0)

        async def _caller() -> None:
            # [FUNCTION DOC] HELPER: Run the cancel-and-await helper on the target task, then record
            # that it returned.
            await cancel_and_await_task(task)
            returned.append("returned")

        caller = asyncio.create_task(_caller())
        await asyncio.wait_for(observed_cancel.wait(), timeout=2.0)
        caller.cancel()
        await asyncio.gather(caller, return_exceptions=True)
        release.set()
        await task

        assert caller.cancelled() and returned == []

    async def test_a_shutdown_deadline_still_fires_when_the_task_is_slow_to_unwind(self):
        # [FUNCTION DOC] The target absorbs its first cancel and waits for a release. Running the
        # helper under a 10 ms timeout scope raises a timeout error and the after-helper marker is
        # never recorded. The target is released and awaited at the end.
        """Eating the deadline's cancellation would let a timed-out shutdown continue as if it had made its deadline."""
        release = asyncio.Event()
        returned: list[str] = []

        async def _absorbs_the_first_cancel() -> None:
            # [FUNCTION DOC] HELPER: Block on an event; on cancellation, wait for the release event
            # (swallowing the cancellation).
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        task = asyncio.create_task(_absorbs_the_first_cancel())
        await asyncio.sleep(0)

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.01):
                await cancel_and_await_task(task)
                returned.append("returned")

        release.set()
        await task

        assert returned == []

    async def test_a_task_that_fails_while_unwinding_reports_its_error(self):
        # [FUNCTION DOC] A task whose cleanup step raises a runtime error ("teardown exploded") is
        # passed to the helper, which re-raises that error.
        """A teardown that raises its own error must not be mistaken for the cancellation the helper asked for."""

        async def _fails_on_teardown() -> None:
            # [FUNCTION DOC] HELPER: Block on an event; in the cleanup step always raise a runtime
            # error, replacing the cancellation.
            try:
                await asyncio.Event().wait()
            finally:
                raise RuntimeError("teardown exploded")

        task = asyncio.create_task(_fails_on_teardown())
        await asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="teardown exploded"):
            await cancel_and_await_task(task)


class TestMergeAssertingConsistency:
    # [CLASS DOC] Tests of the strict two-mapping merge.
    def test_disjoint_keys_are_merged(self):
        # [FUNCTION DOC] Merging mappings with different keys yields their union.
        """The common case: two views of the same cell describe different fields of it."""
        assert merge_asserting_consistency({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_a_key_both_sides_agree_on_is_kept_once(self):
        # [FUNCTION DOC] A key present in both mappings with equal values appears once in the merge.
        """Two pods of one cell repeat the cell-wide annotations, which is not a conflict."""
        assert merge_asserting_consistency({"a": 1, "b": 2}, {"b": 2, "c": 3}) == {"a": 1, "b": 2, "c": 3}

    def test_a_key_the_two_sides_disagree_on_is_rejected(self):
        # [FUNCTION DOC] A key with different values in the two mappings raises an assertion error
        # whose message mentions disagreement.
        """Silently picking a winner would hand the caller one pod's answer as the whole cell's."""
        with pytest.raises(AssertionError, match="disagree"):
            merge_asserting_consistency({"a": 1}, {"a": 2})


class TestMutableBox:
    # [CLASS DOC] Tests of the mutable value box.
    def test_a_closure_writes_through_the_box_its_caller_reads(self):
        # [FUNCTION DOC] A nested function increments the box's value by two; calling it twice makes
        # the caller see four.
        """A name rebound inside a nested function rebinds nothing its caller can see, which is the point."""
        box: MutableBox[int] = MutableBox(value=0)

        def advance() -> None:
            # [FUNCTION DOC] HELPER: Add two to the enclosing box's value.
            box.value += 2

        advance()
        advance()

        assert box.value == 4

    def test_a_box_holds_whatever_it_was_opened_with(self):
        # [FUNCTION DOC] A box created with none holds none; one created with a string holds it.
        """Callers open it on an optional handle as readily as on a counter."""
        assert MutableBox(value=None).value is None
        assert MutableBox(value="a").value == "a"

    def test_two_boxes_of_the_same_value_do_not_share_it(self):
        # [FUNCTION DOC] Two boxes created with zero; changing the first to seven leaves the second
        # at zero.
        """Each caller's box is its own; a shared default would let one run's cursor drive another's."""
        first: MutableBox[int] = MutableBox(value=0)
        second: MutableBox[int] = MutableBox(value=0)

        first.value = 7

        assert second.value == 0
