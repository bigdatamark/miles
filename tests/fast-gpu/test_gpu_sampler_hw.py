# [MODULE DOC] Hardware tests for the dashboard GPU telemetry sampler on a real multi-GPU CI
# runner: that automatic backend detection picks the vendor library matching the hardware (AMD SMI
# on ROCm builds, NVML otherwise) and that a sample pass reports every device's telemetry plus
# per-process samples with sane values. Uses a recording callable in place of the real telemetry
# sink. Registered for the 2-GPU CUDA suite and a ROCm suite.
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=60, suite="stage-b-2-gpu-h200", labels=["short"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=60, suite="stage-c-4-gpu-mi350", labels=["amd"])

import torch

from miles.dashboard.gpu_sampler import GpuSampler


class PushSpy:
    # [CLASS DOC] HELPER: Recording stand-in for the sampler's push callback; stores every
    # invocation as a pair (node name, batch of samples) in a list of calls.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts with an empty list of recorded calls.
        self.calls = []

    def __call__(self, node, batch):
        # [FUNCTION DOC] HELPER: Records the node and batch as a pair; returns nothing.
        self.calls.append((node, batch))


def test_auto_detection_picks_the_backend_matching_the_hardware():
    # [FUNCTION DOC] Creates a sampler with default backend detection, asserts it reports itself
    # available (message: no telemetry backend on a GPU runner), and that its provider name is "AMD
    # SMI" when torch is a HIP build, else "NVML".
    sampler = GpuSampler(push=PushSpy(), node="ci")
    assert sampler.available, "no GPU telemetry backend initialized on a GPU runner"
    expected = "AMD SMI" if torch.version.hip else "NVML"
    assert sampler._provider.name == expected


def test_every_device_reports_telemetry_and_processes():
    # [FUNCTION DOC] Creates a sampler with two recording sinks (device telemetry and processes).
    # Asserts it is available, that there are at least two devices with non-empty UUIDs, that one
    # device sample pass at timestamp 1.0 returns the device count and one process pass returns a
    # non-negative count. After flushing, exactly one telemetry push is recorded; its samples must be
    # for GPU indices 0..count-1 in order, with utilization between 0 and 100 and non-negative memory
    # and power. If any process batch was pushed, its first batch's samples must have positive pid,
    # non-negative memory and a non-empty name.
    push = PushSpy()
    push_processes = PushSpy()
    sampler = GpuSampler(push=push, node="ci", push_processes=push_processes)
    assert sampler.available

    uuids = sampler.gpu_uuids()
    count = len(uuids)
    assert count >= 2, f"expected a multi-GPU CI runner, saw {count} device(s)"
    assert all(uuids)

    assert sampler.sample_once(ts=1.0) == count
    assert sampler.sample_processes_once(ts=1.0) >= 0
    sampler.flush()
    [(_, batch)] = push.calls
    assert [sample.gpu for sample in batch] == list(range(count))
    for sample in batch:
        assert 0 <= sample.util <= 100
        assert sample.mem_mb >= 0 and sample.power_w >= 0
    if push_processes.calls:
        for process_sample in push_processes.calls[0][1]:
            assert process_sample.pid > 0 and process_sample.mem_mb >= 0 and process_sample.name


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
