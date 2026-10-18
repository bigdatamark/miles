# [MODULE DOC] GPU test for the NVMe-streamed optimizer state store (plugins optimizer package):
# that initializing the fp32 main parameters from the model parameters writes them bucket by bucket
# to the on-disk bucket file with exact bytes, reports the byte count, and frees the in-memory
# storage of the main parameter tensor. Uses hand-built stand-ins for the distributed optimizer and
# bypasses the store constructor. Registered for 2-GPU CUDA and ROCm nightly suites (miles-plugin label).
"""Focused GPU coverage for streamed optimizer main-param initialization."""

import os
from types import SimpleNamespace

import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles_plugins.optimizers.nvme_stream import NVMeOptimizerStateStore, _Bucket, _Entry, _resize, _Stager

register_cuda_ci(
    est_time=30,
    suite="stage-b-2-gpu-h200",
    labels=["miles-plugin"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(est_time=30, suite="nightly-stage-c-2-gpu-mi350", labels=["miles-plugin"])


def test_bucketwise_main_initialization_preserves_bytes_and_releases_cuda_storage(tmp_path):
    # [FUNCTION DOC] Builds a 600,000-element bfloat16 GPU model parameter (values from a float32
    # arange, so exactly representable up to rounding to bfloat16) and an empty float32 main parameter
    # whose storage is resized to zero. Creates one bucket file in the temp directory with a single
    # entry, a 1 MiB staging buffer and float32 segments for main/exp_avg/exp_avg_sq, plus a simple
    # namespace standing in for the distributed optimizer whose range map covers the whole parameter.
    # Creates the store without running its constructor and assigns its optimizer and bucket list.
    # Asserts: initialization returns the main parameter's byte count; the main parameter's storage is
    # empty afterwards; and the bytes read back from the file at offset 0 equal the model parameter
    # converted to float32 exactly. Closes the bucket file descriptor at the end (skipped if an
    # assertion fails earlier).
    model_param = torch.nn.Parameter(torch.arange(600_000, dtype=torch.float32, device="cuda").to(torch.bfloat16))
    main_param = torch.empty_like(model_param, dtype=torch.float32)
    _resize(main_param, 0)
    bucket = _Bucket(
        str(tmp_path / "bucket.bin"),
        entries=[_Entry(model_param, main_param, 0)],
        adam=None,
        stager=_Stager(1024 * 1024),
        dtypes={segment: torch.float32 for segment in ("main", "exp_avg", "exp_avg_sq")},
    )
    param_range = SimpleNamespace(start=0, end=model_param.numel(), size=model_param.numel())
    dist_opt = SimpleNamespace(
        model_fp32_groups=[],
        shard_fp32_groups=[],
        _get_model_param_range_map=lambda _param: {"param": param_range},
    )
    store = object.__new__(NVMeOptimizerStateStore)
    store.dist_opt = dist_opt
    store.buckets = [bucket]

    nbytes = main_param.numel() * main_param.element_size()
    assert store.initialize_main_from_model_params() == nbytes
    assert main_param.untyped_storage().nbytes() == 0
    restored = torch.frombuffer(bytearray(os.pread(bucket.fd, nbytes, 0)), dtype=torch.float32)
    torch.testing.assert_close(restored, model_param.float().cpu(), atol=0, rtol=0)
    os.close(bucket.fd)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
