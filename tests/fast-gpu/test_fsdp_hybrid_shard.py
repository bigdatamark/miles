# [MODULE DOC] Test that FSDP2 hybrid-sharded training produces the same gradients regardless of
# the replicate/shard topology on 4 GPUs: replicate size 1 (shard 4), 2 (shard 2) and 4 (shard 1).
# A separate worker script (sitting next to this file) is run under a distributed launcher once per
# topology as a subprocess; each run writes its gradients to a file, and this test compares the
# topologies against the replicate-size-1 reference. Registered for CUDA and ROCm 4-GPU suites with
# the fsdp label.
"""FSDP2 gradient parity across r1s4, r2s2, and r4s1."""

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=90, suite="stage-c-4-gpu-h200", labels=["fsdp"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=90, suite="nightly-stage-c-4-gpu-mi350", labels=["fsdp"])

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_WORKER = Path(__file__).with_name("_fsdp_hybrid_shard_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


def _run_worker(replicate_size: int, output: Path) -> None:
    # [FUNCTION DOC] HELPER: Launches the worker script in a subprocess through the torch
    # distributed launcher (standalone, one node, 4 processes per node) with the replicate size and
    # output path as arguments. The environment is the parent's plus unbuffered Python output and the
    # repository root prepended to the module search path. Captures text output with a 300 second
    # timeout (raises a timeout error if exceeded). Asserts exit status zero (message includes the
    # captured output) and that stdout contains the pass marker for the topology, formatted as
    # replicate size and shard size (4 divided by replicate size).
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=4",
            str(_WORKER),
            "--replicate-size",
            str(replicate_size),
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS r{replicate_size}s{4 // replicate_size}" in result.stdout


def test_fsdp_hybrid_shard_gradient_parity(tmp_path: Path) -> None:
    # [FUNCTION DOC] Runs the worker for replicate sizes 1, 2 and 4, loading each run's saved
    # gradient mapping (name to tensor) from a temporary directory with safe weights-only loading.
    # Asserts, for sizes 2 and 4 versus size 1, that the parameter names match and that each gradient
    # is close with relative tolerance 2e-5 and absolute tolerance 2e-6.
    topology_gradients = {}
    for replicate_size in (1, 2, 4):
        output = tmp_path / f"grad_r{replicate_size}.pt"
        _run_worker(replicate_size, output)
        topology_gradients[replicate_size] = torch.load(output, weights_only=True)

    reference = topology_gradients[1]
    for replicate_size in (2, 4):
        actual = topology_gradients[replicate_size]
        assert actual.keys() == reference.keys()
        for name, expected_gradient in reference.items():
            torch.testing.assert_close(actual[name], expected_gradient, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
