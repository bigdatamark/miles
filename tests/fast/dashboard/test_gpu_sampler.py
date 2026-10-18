# [MODULE DOC]
# Tests for the dashboard gpu sampler against fake NVIDIA (pynvml-like) and AMD (amdsmi-like)
# management libraries injected into the sampler, plus guarded tests that run against the real
# libraries when GPUs exist. Covers unit conversion, backend selection and fallback, device and
# process sampling, failure handling per device, buffering and flushing, the background thread
# lifecycle, and the keyword-only constructor contract.
import logging
import sys
import time

import pytest

from miles.dashboard import gpu_sampler as gpu_sampler_module
from miles.dashboard.gpu_sampler import GpuSampler
from miles.dashboard.store import GpuProcessSample, GpuSample


class FakeNvml:
    # [CLASS DOC] HELPER: Fake NVIDIA management library. Device handle equals device index.
    # Configurable device count, init failure, and a set of devices whose utilisation and process
    # queries raise a runtime error. Utilisation is 40 plus index percent; used memory is
    # (index+1) GiB in bytes; power is 600000 plus index milliwatts; each device has one compute
    # process with pid 1000+index using (index+1)*512 MiB; uuids and process names are bytes.
    """Just enough of pynvml for the sampler: handle == device index. UUIDs
    and process names come back as bytes, as the real pynvml returns them."""

    def __init__(self, count=2, fail_init=False, failing_devices=()):
        # [FUNCTION DOC] HELPER: Stores the device count, init-failure flag and failing set.
        self.count = count
        self.fail_init = fail_init
        self.failing_devices = set(failing_devices)

    def nvmlInit(self):
        # [FUNCTION DOC] HELPER: Raises a runtime error (driver/library mismatch) if init
        # failure was requested; otherwise does nothing.
        if self.fail_init:
            raise RuntimeError("driver/library version mismatch")

    def nvmlDeviceGetCount(self):
        # [FUNCTION DOC] HELPER: Returns the configured device count.
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        # [FUNCTION DOC] HELPER: Returns the index itself as the handle.
        return index

    def nvmlDeviceGetUUID(self, handle):
        # [FUNCTION DOC] HELPER: Returns the bytes of "GPU-fake-" followed by the handle.
        return f"GPU-fake-{handle}".encode()

    def nvmlDeviceGetUtilizationRates(self, handle):
        # [FUNCTION DOC] HELPER: Raises a runtime error for failing devices; otherwise returns
        # an object whose gpu attribute is 40 plus the handle.
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return type("Util", (), {"gpu": 40 + handle})()

    def nvmlDeviceGetMemoryInfo(self, handle):
        # [FUNCTION DOC] HELPER: Returns an object whose used attribute is (handle+1) GiB in bytes.
        return type("Mem", (), {"used": (handle + 1) * 1024 * 1024 * 1024})()  # GiB in bytes

    def nvmlDeviceGetPowerUsage(self, handle):
        # [FUNCTION DOC] HELPER: Returns 600000 plus the handle, in milliwatts.
        return 600_000 + handle  # milliwatts

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        # [FUNCTION DOC] HELPER: Raises a runtime error for failing devices; otherwise returns a
        # one-element list with an object holding pid 1000+handle and used gpu memory of
        # (handle+1)*512 MiB in bytes.
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return [type("Proc", (), {"pid": 1000 + handle, "usedGpuMemory": (handle + 1) * 512 * 1024 * 1024})()]

    def nvmlSystemGetProcessName(self, pid):
        # [FUNCTION DOC] HELPER: Returns the bytes of "proc-" followed by the pid.
        return f"proc-{pid}".encode()


class FakeAmdSmi:
    # [CLASS DOC] HELPER: Fake AMD management library using ROCm 6.1+ dictionary shapes. Handle
    # equals device index. Activity is 70 plus index; vram used is (index+1)*2048 MiB (reported
    # in MiB, not bytes); power info carries three distinct fields (500, 475, 250 plus index);
    # the process list has one named process (name only for device 0, otherwise "N/A") with
    # (index+1)*768 MiB in bytes, plus a bystander process with zero vram.
    """Just enough of the amdsmi API (ROCm >= 6.1 dict shapes); handle == device index."""

    def __init__(self, count=2, *, fail_init=False, failing_devices=()):
        # [FUNCTION DOC] HELPER: Stores the device count, init-failure flag and failing set.
        self.count = count
        self.fail_init = fail_init
        self.failing_devices = set(failing_devices)

    def amdsmi_init(self):
        # [FUNCTION DOC] HELPER: Raises a runtime error if init failure was requested.
        if self.fail_init:
            raise RuntimeError("AMD SMI initialization failed")

    def amdsmi_get_processor_handles(self):
        # [FUNCTION DOC] HELPER: Returns the list of handles 0 to count-1.
        return list(range(self.count))

    def amdsmi_get_gpu_device_uuid(self, handle):
        # [FUNCTION DOC] HELPER: Returns the text "GPU-amd-" followed by the handle.
        return f"GPU-amd-{handle}"

    def amdsmi_get_gpu_activity(self, handle):
        # [FUNCTION DOC] HELPER: Raises for failing devices; otherwise returns a mapping with
        # graphics activity 70+handle, memory controller activity 10 and multimedia activity 0.
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return {"gfx_activity": 70 + handle, "umc_activity": 10, "mm_activity": 0}

    def amdsmi_get_gpu_vram_usage(self, handle):
        # [FUNCTION DOC] HELPER: Returns total vram 192 GiB (in MiB) and used (handle+1)*2048 MiB.
        # AMD SMI reports vram_used in MiB, not bytes.
        return {"vram_total": 192 * 1024, "vram_used": (handle + 1) * 2048}

    def amdsmi_get_power_info(self, handle):
        # [FUNCTION DOC] HELPER: Returns a mapping with distinct socket, current socket and
        # average socket power values, so tests can detect selection of the wrong field.
        # Keep values distinct to catch selection of the wrong power field.
        return {
            "socket_power": 500 + handle,
            "current_socket_power": 475 + handle,
            "average_socket_power": 250 + handle,
        }

    def amdsmi_get_gpu_process_list(self, handle):
        # [FUNCTION DOC] HELPER: Raises for failing devices; otherwise returns two processes: a
        # real one (pid 2000+handle, name only for device 0, "N/A" otherwise, vram
        # (handle+1)*768 MiB in bytes) and a bystander (pid 9000+handle) with zero vram.
        if handle in self.failing_devices:
            raise RuntimeError("GPU is lost")
        return [
            {
                "pid": 2000 + handle,
                "name": f"amd-proc-{handle}" if handle == 0 else "N/A",
                "memory_usage": {"vram_mem": (handle + 1) * 768 * 1024 * 1024},
            },
            {"pid": 9000 + handle, "name": "kfd-bystander", "memory_usage": {"vram_mem": 0}},
        ]


class PushSpy:
    # [CLASS DOC] HELPER: Callable recording every (node, batch of gpu samples) push.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialises an empty call list.
        self.calls: list[tuple[str, list[GpuSample]]] = []

    def __call__(self, node, batch):
        # [FUNCTION DOC] HELPER: Records the node and batch pair.
        self.calls.append((node, batch))


class ProcessPushSpy:
    # [CLASS DOC] HELPER: Callable recording every (node, batch of gpu process samples) push.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialises an empty call list.
        self.calls: list[tuple[str, list[GpuProcessSample]]] = []

    def __call__(self, node, batch):
        # [FUNCTION DOC] HELPER: Records the node and batch pair.
        self.calls.append((node, batch))


def test_sample_once_converts_units():
    # [FUNCTION DOC] With a two-device fake NVIDIA library: the sampler is available and reports
    # both uuids as text. One sampling tick returns 2 and, after a flush, one push for the node
    # with samples util 40/41, memory 1024/2048 MB (bytes converted) and power 600 W (milliwatts
    # converted, rounded) for gpus 0 and 1.
    push = PushSpy()
    sampler = GpuSampler(push=push, node="10.0.0.1", nvml=FakeNvml(count=2))
    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-fake-0", "GPU-fake-1"]

    assert sampler.sample_once(ts=10.0) == 2
    sampler.flush()
    [(node, batch)] = push.calls
    assert node == "10.0.0.1"
    assert batch == [
        GpuSample(ts=10.0, node="10.0.0.1", gpu=0, util=40, mem_mb=1024, power_w=600),
        GpuSample(ts=10.0, node="10.0.0.1", gpu=1, util=41, mem_mb=2048, power_w=600),
    ]


def test_amd_sample_once_preserves_native_units_and_uuids():
    # [FUNCTION DOC] With a two-device fake AMD library: uuids are reported, a tick returns 2,
    # and the flushed samples use graphics activity as util (70/71), vram MiB as memory
    # (2048/4096, no conversion) and the unified socket power (500/501 W).
    push = PushSpy()
    sampler = GpuSampler(push=push, node="amd-node", amdsmi=FakeAmdSmi(count=2))
    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-amd-0", "GPU-amd-1"]

    assert sampler.sample_once(ts=11.0) == 2
    sampler.flush()
    [(node, batch)] = push.calls
    assert node == "amd-node"
    assert batch == [
        GpuSample(ts=11.0, node="amd-node", gpu=0, util=70, mem_mb=2048, power_w=500),
        GpuSample(ts=11.0, node="amd-node", gpu=1, util=71, mem_mb=4096, power_w=501),
    ]


@pytest.mark.parametrize(
    "power,expected",
    [
        # MI300+/ROCm 7.x: the unified field carries current power and wins.
        ({"socket_power": 500, "current_socket_power": 475, "average_socket_power": 250}, 500),
        # ROCm 7.x with the unified sensor unavailable ("N/A" per field).
        ({"socket_power": "N/A", "current_socket_power": 475, "average_socket_power": 250}, 475),
        # ROCm 6.0 numeric uint16 sentinel; no unified field on 6.x.
        ({"current_socket_power": 0xFFFF, "average_socket_power": 250}, 250),
        # Pre-MI300 6.x wrappers only expose the average.
        ({"average_socket_power": 250}, 250),
    ],
)
def test_amd_socket_power_prefers_current_then_falls_back(power, expected):
    # [FUNCTION DOC] Parametrised. The module's socket-power chooser picks, in order: the unified
    # socket field; else the current socket field; else the average field. Non-numeric "N/A" and
    # the 16-bit all-ones sentinel are treated as unavailable. Asserts the expected chosen value
    # for four power mappings (new ROCm, unified missing, old sentinel, average only).
    assert gpu_sampler_module._amd_socket_power(power) == expected


def test_amd_socket_power_raises_when_all_fields_unavailable():
    # [FUNCTION DOC] When every power field is unavailable or missing, the chooser raises a value
    # error whose message matches "socket power unavailable".
    with pytest.raises(ValueError, match="socket power unavailable"):
        gpu_sampler_module._amd_socket_power({"socket_power": "N/A", "current_socket_power": 0xFFFF})


def test_amd_unavailable_power_reports_zero_but_keeps_util_and_mem():
    # [FUNCTION DOC] With a one-device fake AMD library whose power info is all "N/A", one tick
    # still returns 1 sample with util 70, memory 2048 and power 0.
    push = PushSpy()
    amdsmi = FakeAmdSmi(count=1)
    amdsmi.amdsmi_get_power_info = lambda handle: {"socket_power": "N/A", "current_socket_power": "N/A"}
    sampler = GpuSampler(push=push, node="n", amdsmi=amdsmi)

    assert sampler.sample_once(ts=1.0) == 1
    sampler.flush()
    [sample] = push.calls[0][1]
    assert (sample.util, sample.mem_mb, sample.power_w) == (70, 2048, 0)


def test_flush_clears_buffer_and_skips_empty():
    # [FUNCTION DOC] Flushing an empty buffer makes no push. After one tick, two flushes produce
    # exactly one push (the buffer is cleared by the first).
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=1))
    sampler.flush()  # empty: no call
    assert push.calls == []

    sampler.sample_once(ts=1.0)
    sampler.flush()
    sampler.flush()  # cleared: no duplicate push
    assert len(push.calls) == 1


def test_nvml_init_failure_disables_sampler(caplog):
    # [FUNCTION DOC] If the injected NVIDIA library fails to init, construction logs a warning
    # containing "NVML unavailable"; the sampler is unavailable, start returns false, a tick
    # returns 0, and nothing is pushed.
    push = PushSpy()
    with caplog.at_level(logging.WARNING):
        sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(fail_init=True))
    assert not sampler.available
    assert sampler.start() is False
    assert sampler.sample_once(ts=1.0) == 0
    assert push.calls == []
    assert any("NVML unavailable" in r.message for r in caplog.records)


@pytest.mark.parametrize("backend", ["nvml", "amdsmi"])
def test_zero_devices_disable_sampler(backend):
    # [FUNCTION DOC] Parametrised over both backends. A backend reporting zero devices leaves the
    # sampler unavailable with an empty uuid list.
    fake = FakeNvml(count=0) if backend == "nvml" else FakeAmdSmi(count=0)
    sampler = GpuSampler(push=PushSpy(), node="n", **{backend: fake})
    assert not sampler.available
    assert sampler.gpu_uuids() == []


def test_production_auto_detection_falls_back_from_nvml_to_amdsmi(monkeypatch):
    # [FUNCTION DOC] Installs a failing fake NVIDIA module and a working one-device fake AMD
    # module into the import table, builds a sampler without injection, and asserts it is
    # available using the AMD backend (uuid "GPU-amd-0").
    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml(fail_init=True))
    monkeypatch.setitem(sys.modules, "amdsmi", FakeAmdSmi(count=1))

    sampler = GpuSampler(push=PushSpy(), node="n")

    assert sampler.available
    assert sampler.gpu_uuids() == ["GPU-amd-0"]


def test_explicit_injection_does_not_probe_the_other_backend(monkeypatch):
    # [FUNCTION DOC] Installs an AMD module whose init fails the test, then builds a sampler with
    # an explicitly injected failing NVIDIA fake. The sampler is unavailable and the AMD library
    # was never probed.
    class BoomAmdSmi:
        # [CLASS DOC] HELPER: AMD stand-in whose init fails the running test immediately.
        def amdsmi_init(self):
            # [FUNCTION DOC] HELPER: Fails the test with a message that AMD must not be probed.
            pytest.fail("explicit NVML injection must not probe AMD SMI")

    monkeypatch.setitem(sys.modules, "amdsmi", BoomAmdSmi())

    sampler = GpuSampler(push=PushSpy(), node="n", nvml=FakeNvml(fail_init=True))

    assert not sampler.available


def test_missing_optional_backends_disable_sampler(monkeypatch, caplog):
    # [FUNCTION DOC] Marks both optional libraries as not importable (null entries in the import
    # table). A sampler built without injection is unavailable, start returns false, and a
    # warning containing "GPU telemetry unavailable" was logged.
    # None in sys.modules makes `import pynvml` raise ImportError, as if absent.
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    with caplog.at_level(logging.WARNING):
        sampler = GpuSampler(push=PushSpy(), node="n")

    assert not sampler.available
    assert sampler.start() is False
    assert any("GPU telemetry unavailable" in record.message for record in caplog.records)


def test_only_one_backend_may_be_injected():
    # [FUNCTION DOC] Injecting both backends raises an assertion error matching "inject only one".
    with pytest.raises(AssertionError, match="inject only one"):
        GpuSampler(push=PushSpy(), node="n", nvml=FakeNvml(), amdsmi=FakeAmdSmi())


def test_failing_device_is_skipped_others_report(caplog):
    # [FUNCTION DOC] With 3 NVIDIA devices of which device 1 fails: a tick returns 2, the pushed
    # samples are for gpus 0 and 2, and a warning containing "skipping this tick" was logged.
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=3, failing_devices={1}))
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_amd_failing_device_is_skipped_while_others_report(caplog):
    # [FUNCTION DOC] Same as the NVIDIA case for AMD: device 1 of 3 fails, tick returns 2, gpus
    # 0 and 2 are pushed, and a warning "AMD SMI read failed for gpu 1" is logged.
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", amdsmi=FakeAmdSmi(count=3, failing_devices={1}))
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push.calls
    assert [sample.gpu for sample in batch] == [0, 2]
    assert any("AMD SMI read failed for gpu 1" in record.message for record in caplog.records)


@pytest.mark.parametrize(
    "method,payload",
    [
        ("amdsmi_get_gpu_activity", {"gfx_activity": "N/A", "umc_activity": 10, "mm_activity": 0}),
        ("amdsmi_get_gpu_vram_usage", {"vram_total": 192 * 1024, "vram_used": "N/A"}),
    ],
)
def test_amd_degraded_metric_value_skips_device_while_others_report(method, payload, caplog):
    # [FUNCTION DOC] Parametrised over the activity and vram queries. For device 0 only, the
    # named query returns a payload with an "N/A" value in util or memory. A tick returns 1,
    # only gpu 1 is pushed, and a warning "AMD SMI read failed for gpu 0" is logged.
    push = PushSpy()
    amdsmi = FakeAmdSmi(count=2)
    original = getattr(amdsmi, method)
    setattr(amdsmi, method, lambda handle: payload if handle == 0 else original(handle))
    sampler = GpuSampler(push=push, node="n", amdsmi=amdsmi)

    with caplog.at_level(logging.WARNING):
        assert sampler.sample_once(ts=1.0) == 1
    sampler.flush()
    assert [sample.gpu for sample in push.calls[0][1]] == [1]
    assert any("AMD SMI read failed for gpu 0" in record.message for record in caplog.records)


def test_amd_uses_smi_visible_order_without_refiltering_process_env(monkeypatch):
    # [FUNCTION DOC] Sets the HIP and ROCR visible-device environment variables to "2" (via the
    # test patcher), then samples a 3-device fake AMD library. All three devices are reported
    # with gpu ids 0, 1, 2: the sampler trusts the library's visible order and does not filter
    # again by the process environment.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2")
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", amdsmi=FakeAmdSmi(count=3))

    assert sampler.sample_once(ts=1.0) == 3
    sampler.flush()
    assert [sample.gpu for sample in push.calls[0][1]] == [0, 1, 2]


def test_thread_lifecycle_flushes_on_stop():
    # [FUNCTION DOC] Starts a sampler with a 10 ms interval (start returns true), sleeps 80 ms,
    # stops it, and asserts stop flushed buffered samples: at least one push and at least 3
    # samples in total (about 8 ticks expected, margin for scheduler jitter).
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", interval=0.01, nvml=FakeNvml(count=1))
    assert sampler.start() is True
    time.sleep(0.08)
    sampler.stop()
    assert push.calls, "stop() must flush buffered samples"
    total = sum(len(batch) for _, batch in push.calls)
    assert total >= 3  # ~8 ticks at 10ms; generous margin against scheduler jitter


def test_sample_processes_once_converts_units():
    # [FUNCTION DOC] With a 2-device NVIDIA fake and a process push target: process sampling
    # returns 2 and after flush pushes one batch with pid 1000/1001, names decoded from bytes
    # ("proc-1000", "proc-1001") and memory 512/1024 MB converted from bytes.
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=2), push_processes=push_processes)
    assert sampler.sample_processes_once(ts=5.0) == 2
    sampler.flush()
    [(node, batch)] = push_processes.calls
    assert node == "n"
    assert batch == [
        GpuProcessSample(ts=5.0, node="n", gpu=0, pid=1000, name="proc-1000", mem_mb=512),
        GpuProcessSample(ts=5.0, node="n", gpu=1, pid=1001, name="proc-1001", mem_mb=1024),
    ]


def test_amd_processes_convert_bytes_fall_back_on_name_and_drop_zero_vram():
    # [FUNCTION DOC] With a 2-device AMD fake: process sampling returns 2 (zero-vram bystander
    # processes are filtered out). Pushed samples convert bytes to MB (768, 1536), keep the
    # reported name for device 0, and fall back to "pid " plus the pid when the name is "N/A".
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push=PushSpy(), node="n", amdsmi=FakeAmdSmi(count=2), push_processes=push_processes)

    assert sampler.sample_processes_once(ts=5.0) == 2
    sampler.flush()
    [(node, batch)] = push_processes.calls
    assert node == "n"
    # the zero-VRAM kfd-bystander pids are filtered out
    assert batch == [
        GpuProcessSample(ts=5.0, node="n", gpu=0, pid=2000, name="amd-proc-0", mem_mb=768),
        GpuProcessSample(ts=5.0, node="n", gpu=1, pid=2001, name="pid 2001", mem_mb=1536),
    ]


def test_failing_device_skipped_for_process_sampling(caplog):
    # [FUNCTION DOC] NVIDIA, 3 devices with device 1 failing: process sampling returns 2, gpus 0
    # and 2 are pushed, and a "skipping this tick" warning is logged.
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(
        push=push, node="n", nvml=FakeNvml(count=3, failing_devices={1}), push_processes=push_processes
    )
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_processes_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push_processes.calls
    assert [s.gpu for s in batch] == [0, 2]
    assert any("skipping this tick" in r.message for r in caplog.records)


def test_amd_failing_device_is_skipped_for_process_sampling(caplog):
    # [FUNCTION DOC] AMD, 3 devices with device 1 failing: process sampling returns 2, gpus 0
    # and 2 are pushed, and a warning "AMD SMI process query failed for gpu 1" is logged.
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(
        push=PushSpy(),
        node="n",
        amdsmi=FakeAmdSmi(count=3, failing_devices={1}),
        push_processes=push_processes,
    )
    with caplog.at_level(logging.WARNING):
        assert sampler.sample_processes_once(ts=1.0) == 2
    sampler.flush()
    [(_, batch)] = push_processes.calls
    assert [sample.gpu for sample in batch] == [0, 2]
    assert any("AMD SMI process query failed for gpu 1" in record.message for record in caplog.records)


def test_process_batch_dropped_silently_without_push_processes():
    # [FUNCTION DOC] Without a process push target, process sampling still returns the count (1),
    # and after flush the regular push target received nothing.
    push = PushSpy()
    sampler = GpuSampler(push=push, node="n", nvml=FakeNvml(count=1))
    assert sampler.sample_processes_once(ts=1.0) == 1
    sampler.flush()
    assert push.calls == []


def test_interval_must_be_positive():
    # [FUNCTION DOC] Constructing a sampler with interval 0 raises an assertion error.
    with pytest.raises(AssertionError):
        GpuSampler(push=lambda n, b: None, node="n", interval=0, nvml=FakeNvml())


def test_real_nvml_when_gpus_present():
    # [FUNCTION DOC] Skipped if the NVIDIA library is missing or cannot init/shut down. Builds a
    # sampler on the real library and asserts: available; a tick yields at least one sample; a
    # process tick yields a non-negative count; flushed first sample has util in 0..100, non-
    # negative memory and power; first uuid starts with "GPU-"; any pushed process has a positive
    # pid, non-negative memory and a name. Guards the fake against drifting from the real API.
    # Guards the FakeNvml against drifting from the real pynvml API surface;
    # runs wherever a GPU + driver exist (devbox/CI-GPU), skips elsewhere.
    pynvml = pytest.importorskip("pynvml")
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except Exception:
        pytest.skip("no usable NVML device")

    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push=push, node="local", nvml=pynvml, push_processes=push_processes)
    assert sampler.available
    assert sampler.sample_once(ts=1.0) >= 1
    # idle test GPUs may have zero compute processes — asserting >= 0 just
    # guards that the real nvmlDeviceGetComputeRunningProcesses call doesn't raise
    assert sampler.sample_processes_once(ts=1.0) >= 0
    sampler.flush()
    [(_, batch)] = push.calls
    sample = batch[0]
    assert 0 <= sample.util <= 100
    assert sample.mem_mb >= 0 and sample.power_w >= 0
    assert sampler.gpu_uuids()[0].startswith("GPU-")
    if push_processes.calls:
        proc_sample = push_processes.calls[0][1][0]
        assert proc_sample.pid > 0 and proc_sample.mem_mb >= 0 and proc_sample.name


def test_real_amdsmi_when_gpus_present():
    # [FUNCTION DOC] Skipped if the AMD library is missing or the sampler is unavailable. Same
    # shape as the NVIDIA real-library test: one sample at least, valid util range, non-negative
    # memory and power, non-empty first uuid, and for any pushed process a positive pid, positive
    # memory and a name.
    amdsmi = pytest.importorskip("amdsmi")
    push = PushSpy()
    push_processes = ProcessPushSpy()
    sampler = GpuSampler(push=push, node="local", amdsmi=amdsmi, push_processes=push_processes)
    if not sampler.available:
        pytest.skip("no usable AMD SMI device")

    assert sampler.sample_once(ts=1.0) >= 1
    assert sampler.sample_processes_once(ts=1.0) >= 0
    sampler.flush()
    [(_, batch)] = push.calls
    sample = batch[0]
    assert 0 <= sample.util <= 100
    assert sample.mem_mb >= 0 and sample.power_w >= 0
    assert sampler.gpu_uuids()[0]
    if push_processes.calls:
        proc_sample = push_processes.calls[0][1][0]
        assert proc_sample.pid > 0 and proc_sample.mem_mb > 0 and proc_sample.name


class TestConstructorContract:
    # [CLASS DOC] Tests for how the sampler must be constructed.
    def test_a_positional_push_is_rejected(self):
        # [FUNCTION DOC] Passing the push target positionally raises a type error because the
        # constructor is keyword-only (it is created as a ray actor).
        """The sampler is built as a ray actor, so push must be bound by keyword only."""
        with pytest.raises(TypeError):
            GpuSampler(PushSpy(), node="n", nvml=FakeNvml())
