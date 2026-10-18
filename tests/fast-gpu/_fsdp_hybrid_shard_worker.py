# [MODULE DOC] HELPER: Distributed worker script (launched with one process per GPU, typically four)
# that checks gradient parity of FSDP2 hybrid sharding. Each rank builds the same tiny model, runs
# a forward and backward pass on its own random batch, and compares the gradients of the
# fully-sharded model, built on a mesh with a given replicate size, against a reference computed
# without sharding (per-rank gradients averaged across ranks). It also checks that all ranks hold
# identical gradients and that the gathered parameters still equal the initial weights. Command
# line: required replicate size (the data-parallel replicate degree; the shard degree is world size
# divided by it) and required output path where rank 0 saves the gradients mapping and prints a
# pass line. Failures raise assertion errors on the failing rank. Contains a bias-free tiny model
# made of a few ReLU linear blocks plus an output layer.
"""Four-rank worker for FSDP2 hybrid-shard gradient parity."""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.fsdp_utils.parallel import build_fsdp_meshes


class _Block(nn.Module):
    # [CLASS DOC] HELPER: One bias-free square linear layer followed by a ReLU activation.
    def __init__(self, dim: int) -> None:
        # [FUNCTION DOC] HELPER: Creates the bias-free linear layer mapping dim to dim features.
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies the linear layer then ReLU; returns a tensor of the same
        # shape as the input.
        return torch.relu(self.linear(x))


class _TinyModel(nn.Module):
    # [CLASS DOC] HELPER: Small stack of blocks followed by a bias-free output linear layer; used as
    # the model under test.
    def __init__(self, dim: int = 32, depth: int = 3) -> None:
        # [FUNCTION DOC] HELPER: Builds depth blocks (default 3) of width dim (default 32) in a
        # module list, and an output linear layer of the same width without bias.
        super().__init__()
        self.blocks = nn.ModuleList(_Block(dim) for _ in range(depth))
        self.output = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Passes the input through every block in order, then the output
        # layer; returns the result.
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def _make_model() -> _TinyModel:
    # [FUNCTION DOC] HELPER: Seeds the global random generator with 1234 (so every rank and every
    # call yields identical weights) and returns a new tiny model moved to the current GPU.
    torch.manual_seed(1234)
    return _TinyModel().cuda()


def _materialize_gradient(param: nn.Parameter) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns the parameter's full gradient. Asserts a gradient exists. If it
    # is a distributed tensor, gathers it into a full tensor (a collective operation every rank must
    # call); otherwise returns it unchanged.
    grad = param.grad
    assert grad is not None
    return grad.full_tensor() if isinstance(grad, DTensor) else grad


def _reference_gradients(inputs: torch.Tensor, world_size: int) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC] HELPER: Computes the unsharded reference gradients. Builds a fresh identical
    # model, runs this rank's inputs through it and backpropagates the mean of the squared outputs.
    # For each named parameter, copies the gradient, sums it across all ranks with an all-reduce,
    # and divides by the world size, giving the cross-rank average. Returns a mapping from
    # parameter name to that averaged gradient. Collective: every rank must call it.
    model = _make_model()
    model(inputs).square().mean().backward()

    gradients = {}
    for name, param in model.named_parameters():
        assert param.grad is not None
        gradient = param.grad.detach().clone()
        dist.all_reduce(gradient)
        gradient /= world_size
        gradients[name] = gradient
    return gradients


def _fully_shard_model(model: _TinyModel, mesh) -> None:
    # [FUNCTION DOC] HELPER: Wraps each block with fully-sharded data parallelism on the given
    # device mesh, then wraps the whole model (covering the output layer) with the same mesh. Works
    # in place; returns nothing.
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)


def main() -> None:
    # [FUNCTION DOC] Worker entry. Parses the replicate size and output path; sets the CUDA device
    # from the local-rank environment variable (key error if missing) and initializes the NCCL
    # process group; computes the shard size as world size divided by replicate size. Builds the
    # data-parallel meshes via the project's helper and takes the "fsdp" mesh, asserting it has one
    # dimension when replicate size is 1 and two otherwise. Generates this rank's input batch of 8
    # by 32 values from a generator seeded with 9000 plus the rank, computes reference gradients,
    # builds the model, records its initial weights, shards it, runs forward and backward on the
    # same inputs. For every parameter: gathers the full gradient and asserts it matches the
    # reference within relative 2e-5 and absolute 2e-6; all-gathers it across ranks and asserts
    # every rank's copy is exactly equal to rank 0's; keeps a CPU copy; gathers the full parameter
    # and asserts it equals the initial weight (no update happened). Rank 0 then saves the
    # gradients mapping to the output path and prints a pass line with the replicate and shard
    # sizes; all ranks barrier and destroy the process group.
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicate-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_size = world_size // args.replicate_size

    meshes = build_fsdp_meshes(
        device_type="cuda",
        world_size=world_size,
        dp_replicate_size=args.replicate_size,
    )
    fsdp_mesh = meshes["fsdp"]
    assert fsdp_mesh.ndim == (1 if args.replicate_size == 1 else 2)

    generator = torch.Generator(device="cuda").manual_seed(9000 + rank)
    inputs = torch.randn(8, 32, generator=generator, device="cuda")
    expected_gradients = _reference_gradients(inputs, world_size)

    model = _make_model()
    initial_weights = {name: param.detach().clone() for name, param in model.named_parameters()}
    _fully_shard_model(model, fsdp_mesh)
    model(inputs).square().mean().backward()

    gradients = {}
    for name, param in model.named_parameters():
        actual = _materialize_gradient(param).detach()
        torch.testing.assert_close(actual, expected_gradients[name], rtol=2e-5, atol=2e-6)

        peers = [torch.empty_like(actual) for _ in range(world_size)]
        dist.all_gather(peers, actual)
        for peer in peers[1:]:
            torch.testing.assert_close(peers[0], peer, rtol=0, atol=0)
        gradients[name] = actual.cpu()

        full_param = gather_full_param(param)
        torch.testing.assert_close(full_param, initial_weights[name])

    if rank == 0:
        torch.save(gradients, args.output)
        print(f"PASS r{args.replicate_size}s{shard_size}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
