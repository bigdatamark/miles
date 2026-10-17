# [MODULE DOC] Builds the context object that lets weights be rebuilt (rematerialized) from the
# optimizer's fp32 master weights instead of keeping a full pinned CPU copy. It collects the
# "extras" that have no master weight (expert-bias buffers, fp32 or frozen parameters), which still
# need a pinned backup; chooses how to cast master weights back into model parameters depending on
# the optimizer flavor (Megatron distributed optimizer, or the hybrid CPU/GPU optimizer when
# precision-aware); and asserts that every model parameter is covered by either the data-parallel
# buffers or the extras.
"""Wiring for --rematerialize-param-from-master-weight. Rebuilds the low-precision
weights from the optimizer's master weights instead of a pinned CPU copy."""

import logging
from argparse import Namespace
from collections.abc import Callable, Iterator, Sequence

import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix
from miles.utils.tensor_backper import MainCastContext

logger = logging.getLogger(__name__)


def _named_restore_extras(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Lazily yields (name, tensor) for tensors that cannot be rebuilt from
    # master weights. For each chunk (names prefixed with the chunk index as virtual stage and the
    # stripped parameter name): first every buffer whose name contains the expert-bias marker, then
    # every parameter that is fp32 or does not require gradients.
    """Tensors with no master weight to rebuild from, so they keep a pinned backup."""
    for vp_stage, model_module in enumerate(model):
        for name, buffer in model_module.named_buffers():
            if "expert_bias" in name:
                yield f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}", buffer
        for name, param in model_module.named_parameters():
            if param.dtype == torch.float32 or not param.requires_grad:
                yield f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}", param


def build_main_cast_context(args: Namespace, *, model: Sequence[torch.nn.Module], optimizer) -> MainCastContext:
    # [FUNCTION DOC] Creates the main-cast context for the given model chunks and optimizer
    # (a chained optimizer). Collects the extras list, logs (info) the count, size in MiB and first
    # twenty names, and returns a context holding: a function that casts master weights to model
    # params (precision-aware choice taken from arguments), the model chunks, a getter that
    # re-enumerates the extras on demand, the set of object ids of tensors restorable (from the
    # coverage assertion, which raises an assertion error if any parameter is uncovered), and the
    # check flag from arguments.
    extras = list(_named_restore_extras(model))
    extras_bytes = sum(t.numel() * t.element_size() for _, t in extras)
    logger.info(
        f"rematerialize-param-from-master-weight: {len(extras)} extra tensors "
        f"({extras_bytes / 2**20:.1f} MiB) kept in pinned backup: "
        f"{[name for name, _ in extras[:20]]}"
    )
    return MainCastContext(
        cast_main_to_params=_build_cast_main_to_params_fn(
            optimizer, precision_aware=args.use_precision_aware_optimizer
        ),
        model_chunks=model,
        extras_getter=lambda: _named_restore_extras(model),
        rematerializable_ids=_assert_rematerialize_coverage(model, extras),
        check=args.check_rematerialize_param_from_master_weight,
    )


def _build_cast_main_to_params_fn(optimizer, *, precision_aware: bool) -> Callable[[], None]:
    # [FUNCTION DOC] HELPER: Returns a no-argument function that copies master weights into model
    # params across the chained optimizers. Without precision-aware mode, the function asks each
    # distributed optimizer to copy its main params to model params. With precision-aware mode
    # (only valid with CPU optimizer offload), asserts every inner optimizer is the hybrid device
    # optimizer (imported lazily) and returns a function that replays its copy-back for each.
    dist_opts = optimizer.chained_optimizers
    if not precision_aware:

        def cast_mcore():
            # [FUNCTION DOC] HELPER: Calls the copy-main-to-model routine of every distributed
            # optimizer in order.
            for dist_opt in dist_opts:
                dist_opt._copy_main_params_to_model_params()

        return cast_mcore

    # Arg validation narrows precision-aware to --optimizer-cpu-offload, where
    # _copy_main_params_to_model_params is a no-op and HDO holds the masters instead.
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    inners = [dist_opt.optimizer for dist_opt in dist_opts]
    for inner in inners:
        assert isinstance(inner, HybridDeviceOptimizer), type(inner)

    def cast_hdo():
        # [FUNCTION DOC] HELPER: Replays the hybrid-optimizer copy-back for each inner optimizer.
        for inner in inners:
            _replay_hybrid_device_copy_back(inner)

    return cast_hdo


@torch.no_grad()
def _replay_hybrid_device_copy_back(hdo) -> None:
    # [FUNCTION DOC] HELPER: Under no-grad, copies the optimizer's CPU master data into each GPU
    # shard view in the GPU-to-CPU-copy map, then copies the fp32 master data into each shard view
    # of the param-to-fp32 map that was not already handled by the first map. Mutates the shard
    # views in place.
    """Replay HDO's two step post-hooks, which wrote this shard at step end."""
    for shard_view, cpu_master in hdo.gpu_params_map_cpu_copy.items():
        shard_view.data.copy_(cpu_master.data)
    # The CPU pass above already covered the params present in both maps.
    for shard_view, fp32_master in hdo.param_to_fp32_param.items():
        if shard_view not in hdo.gpu_params_map_cpu_copy:
            shard_view.data.copy_(fp32_master.data)


def _assert_rematerialize_coverage(model: Sequence[torch.nn.Module], extras: list[tuple[str, torch.Tensor]]) -> set:
    # [FUNCTION DOC] HELPER: Builds the set of object ids that can be restored: the extras plus every
    # param in each chunk's regular and expert-parallel data-parallel buffers. Then asserts every
    # named parameter of every chunk is in that set; on failure the assertion message gives the
    # count and the first ten uncovered names. Returns the set of ids.
    """Anything outside the DDP buffers and the extras backup would come back as garbage.

    DDP buffer membership is the right criterion: optimizer structures only cover this
    rank's shard under DP>1.
    """
    restorable = {id(t) for _, t in extras}
    for model_module in model:
        for buffer in model_module.buffers + model_module.expert_parallel_buffers:
            restorable.update(id(p) for p in buffer.params)
    uncovered = []
    for model_module in model:
        for name, param in model_module.named_parameters():
            if id(param) not in restorable:
                uncovered.append(name)
    assert not uncovered, (
        f"--rematerialize-param-from-master-weight cannot restore {len(uncovered)} params "
        f"(not in the DDP param buffers nor in the extras backup): {uncovered[:10]}"
    )
    return restorable
