# [MODULE DOC] Helpers for torch distributed: a lazily created global gloo (CPU) group, a
# context manager that serializes a code block across ranks, a copy of torch's process group
# initialization that permits several "main" groups in one process, and a whitening function that
# uses statistics aggregated over a process group. Module state: the gloo group handle.
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from packaging.version import parse
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)

from miles.utils.ft_utils.process_group_utils import GeneralPGUtil

GLOO_GROUP = None


def init_gloo_group():
    # [FUNCTION DOC] Creates, once, a new gloo-backend group over all ranks and stores it in the
    # module global (collective on first call); later calls return the stored group.
    """Initialize Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


def get_gloo_group():
    # [FUNCTION DOC] Returns the stored gloo group; raises a runtime error if it was never
    # initialized.
    # KNOWN ISSUE: the error message names a function "_init_gloo_group", which does not exist
    # (the real one is init_gloo_group).
    """Get the Gloo group for distributed communication."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call _init_gloo_group() first.")
    return GLOO_GROUP


@contextmanager
def one_rank_at_a_time():
    # [FUNCTION DOC] Context manager that makes ranks execute the managed block one after another
    # in rank order. Loops over ranks 0..world-1; the matching rank yields (runs its block), and
    # after each step all ranks meet at a barrier on the gloo group. So rank r runs its block only
    # after ranks below it finished. Requires the gloo group. If a block raises, that rank skips
    # its barrier and the other ranks will hang.
    for rank in range(dist.get_world_size()):
        if rank == dist.get_rank():
            yield
        dist.barrier(group=get_gloo_group())


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    # [FUNCTION DOC] Creates and returns a process group without registering it as the default
    # (copied from torch to allow several main groups). Exactly one of store or init_method may be
    # given (assertion otherwise). With a store: world size must be positive and rank non-negative.
    # With neither: init method defaults to environment-variable rendezvous. Backend defaults to
    # undefined, timeout to torch's default. Without a store, runs the rendezvous to get a store,
    # rank and world size, sets the store timeout and wraps the store in a prefix store keyed by the
    # group name. Passes the backend options under the keyword name that matches the torch version
    # (renamed in 2.6). Registers an identity mapping from group rank to global rank in torch's
    # private world table (side effect on torch internals).
    # KNOWN ISSUE: group_name defaults to none, but the prefix store requires a string, so the
    # rendezvous path fails unless a name is passed.
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = "backend_options" if parse(torch.__version__) >= parse("2.6") else "pg_options"
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
):
    # [FUNCTION DOC] Computes masked sum, sum of squares and mask count locally, sums these three
    # numbers across the given process group (a group-aware utility is chosen from the group) in
    # float32, and raises a value error if the global mask count is zero. Mean is sum over count;
    # variance is mean of squares minus squared mean, multiplied by count/(count-1) when the count
    # is at least 2. Returns (values minus mean) times the inverse square root of (variance plus
    # epsilon); if shift_mean is false the mean is added back (modifying the result in place).
    # The mask is not applied to the output.
    """
    Performs whitening on a tensor using global statistics from all participating GPUs.

    It calculates the global mean and variance across all ranks in the default
    process group (the WORLD) and uses these global statistics to normalize the
    local data on each rank.

    Args:
        values (torch.Tensor): The local tensor of values to whiten.
        mask (torch.Tensor): The local mask corresponding to the values.
        process_group: The process group for all_reduce.
                      If None, uses the default world group.
        shift_mean (bool): If True, the output is zero-mean. Defaults to True.
        epsilon (float): A small value for numerical stability.

    Returns:
        torch.Tensor: The locally whitened tensor using global statistics.
    """
    # Calculate local intermediate statistics
    local_sum = (values * mask).sum()
    local_sum_sq = ((values**2) * mask).sum()
    local_mask_sum = mask.sum()

    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )

    # Aggregate via all_reduce within the DP group
    GeneralPGUtil.create(process_group).all_reduce(stats_tensor, process_group, op=dist.ReduceOp.SUM)

    # Calculate global stats from aggregated results
    global_sum, global_sum_sq, global_mask_sum = stats_tensor

    if global_mask_sum.item() == 0:
        raise ValueError("The global mask sum across all participating GPUs is zero.")

    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean**2

    # Bessel's correction for unbiased estimate
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    # Whiten local data using global stats
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)

    if not shift_mean:
        whitened_values += global_mean

    return whitened_values
