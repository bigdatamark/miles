# [MODULE DOC] GPU test of the torch-native training offload helper (moves model and optimizer
# state between GPU and pinned host memory) on a model sharded with FSDP2 (fully_shard) over a
# single-rank device mesh. Registered for 2-GPU Hopper CUDA and ROCm nightly suites (fsdp and
# torchtitan labels).
import torch
import torch.distributed as dist
import torch.nn as nn
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from miles.backends.training_utils.torch_native.offload import move_train_state

register_cuda_ci(est_time=30, suite="stage-b-2-gpu-h200", labels=["fsdp", "torchtitan"], hardware=["hopper"])
register_rocm_ci(est_time=60, suite="nightly-stage-c-2-gpu-mi350", labels=["fsdp", "torchtitan"])


def _local(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns the local shard of a distributed tensor, or the tensor itself if
    # it is an ordinary tensor.
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _train_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Collects the local tensors of all model parameters, followed by every
    # optimizer state value that is a tensor with at least one dimension (scalar step counters are
    # skipped), as a list in iteration order.
    tensors = [_local(param) for param in model.parameters()]
    for state in optimizer.state.values():
        tensors += [_local(value) for value in state.values() if isinstance(value, torch.Tensor) and value.dim() > 0]
    return tensors


def test_offload_round_trips_fsdp_state_through_pinned_host_memory(monkeypatch):
    # [FUNCTION DOC] Sets rendezvous variables (localhost, port 29541) and initializes a one-rank
    # NCCL group (destroyed in a finally block). Builds a two-linear-layer model on the GPU, shards it
    # with FSDP2 over a one-device mesh, takes one AdamW step on a random input (seed 0), and snapshots
    # all parameter and optimizer state tensors to the CPU. Offloads to the CPU via the move helper and
    # asserts every tensor is on the CPU and pinned and the parameters are still distributed tensors.
    # Moves back to the GPU and asserts all tensors are on CUDA and bit-equal to the snapshot. Finally
    # runs one more forward, backward and optimizer step to prove the restored state is usable.
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29541")
    dist.init_process_group("nccl", rank=0, world_size=1)
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(64, 64, bias=False), nn.Linear(64, 64, bias=False)).cuda()
        fully_shard(model, mesh=init_device_mesh("cuda", (1,)))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        inputs = torch.randn(4, 64, device="cuda")
        model(inputs).pow(2).mean().backward()
        optimizer.step()
        before = [tensor.detach().cpu().clone() for tensor in _train_state(model, optimizer)]

        move_train_state([model], [optimizer], "cpu")
        offloaded = _train_state(model, optimizer)
        assert all(tensor.device.type == "cpu" and tensor.is_pinned() for tensor in offloaded)
        assert all(isinstance(param, DTensor) for param in model.parameters())

        move_train_state([model], [optimizer], "cuda")
        restored = _train_state(model, optimizer)
        assert all(tensor.device.type == "cuda" for tensor in restored)
        assert all(torch.equal(a, b.cpu()) for a, b in zip(before, restored, strict=True))

        model(inputs).pow(2).mean().backward()
        optimizer.step()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
