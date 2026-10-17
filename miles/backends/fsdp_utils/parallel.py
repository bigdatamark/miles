# [MODULE DOC]
# Device-mesh and parallel-state construction for the FSDP backend. The backend is pure data
# parallel (optionally hybrid-sharded: replicate groups times shard groups), so this module builds
# the data-parallel mesh and the FSDP2 shard mesh and wraps them, plus single-rank groups for every
# other parallelism axis, in the shared ParallelState structure used by the training utilities.
import logging
from argparse import Namespace

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from miles.backends.training_utils.parallel import ParallelState
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.process_group_utils import GroupInfo

logger = logging.getLogger(__name__)


def build_fsdp_meshes(
    device_type: str,
    world_size: int,
    dp_replicate_size: int,
) -> dict[str, DeviceMesh]:
    # [FUNCTION DOC]
    # Creates a one-dimensional "dp" mesh over all ranks. The FSDP mesh is that same mesh unless
    # dp_replicate_size is above one, in which case it is a two-dimensional mesh with named axes
    # "dp_replicate" (size dp_replicate_size) and "dp_shard" (world size divided by it, integer
    # division; the caller must ensure divisibility). The two-dimensional mesh is obtained by
    # unflattening the dp mesh when the library supports that, else by initializing a fresh mesh.
    # Returns a mapping with keys "dp" and "fsdp". Side effect: device-mesh initialization (a
    # collective operation that creates process groups).
    """Build the data-parallel view and the FSDP2 shard mesh."""
    dp_mesh = init_device_mesh(
        device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("dp",),
    )
    fsdp_mesh = dp_mesh
    if dp_replicate_size > 1:
        if hasattr(dp_mesh, "_unflatten"):
            fsdp_mesh = dp_mesh._unflatten(
                0,
                (dp_replicate_size, world_size // dp_replicate_size),
                ("dp_replicate", "dp_shard"),
            )
        else:
            fsdp_mesh = init_device_mesh(
                device_type,
                mesh_shape=(dp_replicate_size, world_size // dp_replicate_size),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )

    return {
        "dp": dp_mesh,
        "fsdp": fsdp_mesh,
    }


def create_fsdp_parallel_state(args: Namespace) -> ParallelState:
    # [FUNCTION DOC]
    # Reads world size and rank from the initialized default process group, builds the meshes on
    # CUDA using args.dp_replicate_size, logs the mesh layout, and creates a one-rank process group
    # containing only this rank (a collective call: all ranks create theirs). Returns a parallel
    # state where: intra-dp uses the dp mesh's group with world-size ranks; intra-dp-with-cp uses the
    # world group plus the project's gloo group; context, tensor, pipeline, expert, expert-tensor and
    # independent-dp axes are all size one, rank zero, on the one-rank group; the meshes mapping is
    # attached. Raises whatever the distributed library raises if the process group is not initialized.
    """Create a ParallelState instance for FSDP configuration."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    meshes = build_fsdp_meshes(
        device_type="cuda",
        world_size=world_size,
        dp_replicate_size=args.dp_replicate_size,
    )
    dp_mesh = meshes["dp"]
    fsdp_mesh = meshes["fsdp"]

    logger.info(
        f"[Rank {rank}] FSDP mesh shape={fsdp_mesh.shape}, "
        f"dp_replicate_size={args.dp_replicate_size}, "
        f"dp_shard_size={world_size // args.dp_replicate_size}, "
        f"dp_rank={rank}"
    )

    # The FSDP backend is pure data parallel: every parallelism axis other than dp is
    # a single-rank group, so collectives issued on them are no-ops.
    self_group = dist.new_group([rank])

    parallel_state = ParallelState(
        intra_dp=GroupInfo(
            rank=rank,
            size=world_size,
            group=dp_mesh.get_group(),
        ),
        intra_dp_cp=GroupInfo(
            rank=rank,
            size=world_size,
            group=dist.group.WORLD,
            gloo_group=get_gloo_group(),
        ),
        cp=GroupInfo(
            rank=0,
            size=1,
            group=self_group,
        ),
        tp=GroupInfo(
            rank=0,
            size=1,
            group=self_group,
        ),
        pp=GroupInfo(rank=0, size=1, group=self_group),
        ep=GroupInfo(rank=0, size=1, group=self_group),
        etp=GroupInfo(rank=0, size=1, group=self_group),
        indep_dp=GroupInfo(rank=0, size=1, group=self_group),
        meshes=meshes,
    )

    return parallel_state
