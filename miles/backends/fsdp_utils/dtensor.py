# [MODULE DOC] One helper that turns an FSDP2 parameter (a sharded distributed tensor, or an
# ordinary tensor) into a full tensor on the GPU, for exporting weights to the rollout engine.
"""DTensor materialization for FSDP2 weight export.

FSDP2 holds each parameter as a sharded ``DTensor``; the rollout engine needs the full tensor. Both
weight-sync paths gather it the same way -- move to CUDA first, then all-gather to ``Replicate``. Two
caveats: ``full_tensor()`` on a CPU DTensor picks the wrong collective backend (so move to CUDA first),
and ``redistribute`` on a 1-rank mesh trips an assert (so world_size==1 falls back to ``full_tensor()``).
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate


def gather_full_param(param: torch.Tensor, *, async_op: bool = False) -> torch.Tensor:
    # [FUNCTION DOC] Return the full local tensor for a parameter, on the current GPU. Steps:
    # move the parameter to CUDA (for a sharded tensor this moves the local shard); a plain tensor
    # is returned right there; with a single process the sharded tensor's own full-tensor method is
    # used (redistribute fails on a one-rank mesh; async flag ignored); otherwise redistribute to
    # replicated placement on every mesh dimension (all-gather), optionally asynchronously, and take
    # the local tensor, which in async mode is a lazy tensor the caller must wait on before use.
    # Collective: all ranks must call in the same order.
    """Materialize a (possibly FSDP2-sharded) param to a full local tensor on CUDA.

    Non-DTensor inputs are returned moved to CUDA unchanged. With ``async_op=True`` the all-gather is
    issued async and the returned tensor carries a ``.wait()`` the caller must drain before use.
    """
    full = param.cuda()
    if not isinstance(full, DTensor):
        return full
    if dist.get_world_size() == 1:
        # redistribute on a 1-rank mesh trips `assert compute_mesh is not None`
        return full.full_tensor()
    return full.redistribute(
        placements=[Replicate()] * full.device_mesh.ndim,
        async_op=async_op,
    ).to_local()
