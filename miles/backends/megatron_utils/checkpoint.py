# [MODULE DOC]
# Checkpoint loading and saving for the Megatron backend. Load dispatches on checkpoint format:
# a Megatron (torch distributed) checkpoint goes through Megatron's loader (after freshness checks
# for DeepSeek-V4 and linear-attention layouts), anything else is treated as a Hugging Face
# checkpoint loaded through Megatron-Bridge or, in raw conversion mode, the mbridge package. LoRA
# adapters are loaded or saved separately. Import-time side effect (inside a guarded import): the
# torch sharded-tensor machinery is monkey-patched to skip the expensive non-overlapping shard
# metadata validation, by making the sharding-spec post-init a no-op and replacing the sharded
# tensor constructor-from-local-shards-and-metadata with a version that does no cross-rank checks.
# If those torch internals are unavailable, the patch is silently skipped. Public names: load,
# save with LoRA, and the re-exported Megatron save.
import logging
import os
import re
import shutil
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist
from megatron.core.utils import unwrap_model

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import get_checkpoint_name
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils import megatron_bridge_utils
from miles_plugins.models.deepseek_v4.arguments import assert_checkpoint_is_current, is_dsv4_model

from .lora.utils import is_lora_enabled, is_lora_model, load_lora_adapter, save_lora_checkpoint
from .model_provider import LinearForLastLayer

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        # [FUNCTION DOC] HELPER: Replacement for the enumerable sharding spec's post-initialization
        # validation; deliberately does nothing, skipping the slow non-overlap check.
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        # [FUNCTION DOC] HELPER: Class-level replacement (bound as a class method) that builds a
        # sharded tensor from this rank's local shards and the global metadata with no validation of
        # shard overlap or cross-rank consistency. Steps: normalize the process group; take the
        # global rank; scan the metadata's shards and collect those placed on this rank (the
        # collected list is not used afterwards); infer the sharding spec from the shards metadata
        # if none is given; allocate an uninitialized sharded tensor object with the metadata's
        # size, dtype, layout, pinned-memory and requires-grad properties; assign the local shards;
        # prepare it for the process group (with RPC references flag) and run post-initialization.
        # Returns the sharded tensor. Mirrors torch's own method minus validation.
        # KNOWN ISSUE: the list of local shard metadatas is computed but never used, and the
        # local_shards argument is trusted without any check against it.
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint", "save_checkpoint_with_lora", "load_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # [FUNCTION DOC]
    # Loads a checkpoint from the load path in the global Megatron args. Returns a tuple: the
    # underlying loader's result (iteration, floating-point operations so far) followed by a
    # boolean saying whether the native optimizer state was restored through the LoRA adapter load.
    # Steps: unless an in-memory local checkpoint manager is present in the checkpointing context
    # (then disk validation is skipped), asserts the load path exists and is a non-empty directory.
    # If a local manager is present or the path looks like a Megatron checkpoint (tracker file
    # present, or directory named iter_ plus seven digits): for non-local loads of DeepSeek-V4
    # models asserts the checkpoint is current, and for models with linear attention buffers
    # asserts the linear-attention layout is current (raises value error if stale); then calls
    # Megatron's loader. Otherwise loads from Hugging Face format. Then, if LoRA is enabled and an
    # adapter path attribute is set, loads the adapter (optionally with optimizer state unless
    # no-load-optim): on success logs and, if the adapter yields an iteration, replaces the
    # iteration in the result; on failure warns that training starts with fresh adapter weights.
    # Side effects: reads disk, mutates model/optimizer/scheduler.
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()

    load_path = args.load

    has_local_checkpoint_manager = "local_checkpoint_manager" in (checkpointing_context or {})
    if has_local_checkpoint_manager:
        logger.info("Skipping disk path validation: using in-memory checkpoint via local_checkpoint_manager")
    else:
        assert Path(load_path).exists() and _is_dir_nonempty(
            load_path
        ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if has_local_checkpoint_manager or _is_megatron_checkpoint(load_path):
        if not has_local_checkpoint_manager and is_dsv4_model(args):
            assert_checkpoint_is_current(load_path)
        if not has_local_checkpoint_manager and _has_linear_attention(ddp_model):
            assert_linear_attn_checkpoint_is_current(load_path)
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        result = _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    # Load LoRA adapter weights if available
    native_optimizer_restored = False
    if is_lora_enabled(args):
        adapter_path = getattr(args, "lora_adapter_path", None)
        if adapter_path is not None:
            loaded, iteration, native_optimizer_restored = load_lora_adapter(
                ddp_model,
                adapter_path,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                load_optimizer=not args.no_load_optim,
            )
            if loaded:
                logger.info(f"Successfully loaded LoRA adapter from {adapter_path}")
                if iteration is not None:
                    result = (iteration, result[1])
            else:
                logger.warning(
                    f"LoRA is enabled and --lora-adapter-path={adapter_path} was specified, "
                    f"but adapter weights could not be loaded. "
                    f"Training will start with freshly initialized adapter weights."
                )

    return (*result, native_optimizer_restored)


def save_checkpoint_with_lora(
    iteration, model, optimizer, opt_param_scheduler, *, publisher: SnapshotPublisher | None = None
):
    # [FUNCTION DOC]
    # Saves a checkpoint for the iteration. For a LoRA model: asserts a snapshot publisher is
    # provided unless conversion mode is raw, then writes the adapter checkpoint (with publisher,
    # optimizer, scheduler, iteration) into the save directory under a seven-digit zero-padded
    # iteration folder named iter_NNNNNNN, subfolder adapter, and logs it. For a non-LoRA model
    # delegates to Megatron's save. Reads the Megatron global args for the save directory.
    """Extended save that handles LoRA adapters separately."""
    args = get_args()

    if is_lora_model(model):
        assert (
            publisher is not None or args.megatron_to_hf_mode == "raw"
        ), "Bridge LoRA checkpoint requires a snapshot publisher"
        save_dir = Path(args.save) / f"iter_{iteration:07d}" / "adapter"
        logger.info(f"Saving LoRA checkpoint to {save_dir}")
        save_lora_checkpoint(
            model,
            args,
            str(save_dir),
            publisher=publisher,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )
    else:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)


def publish_as_release(save_dir: str) -> None:
    # [FUNCTION DOC]
    # Computes the base directory of iteration 1 (non-release) and of the release slot under the
    # save directory using Megatron's naming, deletes any existing release directory (ignoring
    # errors), and moves iteration 1's directory to the release path. Side effects: file deletion
    # and rename. Raises a file-not-found-type error from the move if iteration 1 does not exist.
    """Rename this run's iteration directory to ``release``, replacing an earlier conversion.

    ``shutil.move`` into an existing directory nests rather than replaces, which would leave the
    previous conversion's weights in place behind a tracker that says the new one is there.
    """
    source_dir = get_checkpoint_name(save_dir, 1, False, return_base_dir=True)
    target_dir = get_checkpoint_name(save_dir, -1, True, return_base_dir=True)
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.move(source_dir, target_dir)


LINEAR_ATTN_LAYOUT_KEY = "weight_layout_version"


def _has_linear_attention(ddp_model) -> bool:
    # [FUNCTION DOC] HELPER: True if any buffer name in any model chunk contains the linear-attention
    # layout version key (which marks models using the shared linear-attention layer).
    return any(LINEAR_ATTN_LAYOUT_KEY in name for chunk in ddp_model for name, _ in chunk.named_buffers())


def assert_linear_attn_checkpoint_is_current(load_dir: str) -> None:
    # [FUNCTION DOC]
    # Validates a distributed (torch_dist) checkpoint directory. If a tracker file named
    # latest_checkpointed_iteration.txt exists, reads its text: either "release" or an integer
    # step, resolving the checkpoint subdirectory (release, or iter_ plus seven-digit step). Returns
    # silently if the resolved path is not a directory. Reads the checkpoint metadata and requires
    # at least one stored tensor key ending with the layout version key; otherwise raises a value
    # error telling the user to re-convert from the HF checkpoint. Reads disk only. Raises a
    # value error if the tracker contents are not an integer or "release".
    """Reject torch_dist checkpoints written before the shared linear-attention layer (group-major GDN
    q/k/v and conv rows, Kimi-K3 KDA under linear_attn.*). Megatron skips tensors a checkpoint lacks, so
    such a checkpoint would load scrambled or half-initialized instead of failing."""
    from torch.distributed.checkpoint import FileSystemReader

    path = Path(load_dir)
    step_file = path / "latest_checkpointed_iteration.txt"
    if step_file.is_file():
        step = step_file.read_text().strip()
        path = path / (step if step == "release" else f"iter_{int(step):07d}")
    if not path.is_dir():
        return
    if not any(
        key.endswith(LINEAR_ATTN_LAYOUT_KEY) for key in FileSystemReader(path).read_metadata().state_dict_metadata
    ):
        raise ValueError(
            f"{load_dir} was written before the shared linear-attention layer's weight layout and would load "
            "scrambled. Re-convert it from the HuggingFace checkpoint (tools/convert_hf_to_torch_dist.py)."
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    # [FUNCTION DOC] HELPER: True if the directory has the Megatron tracker file, or the directory's
    # own name is iter_ followed by exactly seven digits (a specific iteration directory).
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


@contextmanager
def _hide_critic_value_head_from_hf_load(ddp_model):
    # [FUNCTION DOC] HELPER: Context manager. On entry, finds in each unwrapped model chunk every
    # direct child module that is the critic value head (last-layer linear class) and removes it
    # from the chunk, so HF weight loading does not try to load or complain about it. On exit
    # (even on error) reattaches each head under its original name. Mutates the model temporarily.
    value_heads = [
        (chunk, name, head)
        for chunk in unwrap_model(ddp_model)
        for name, head in chunk.named_children()
        if isinstance(head, LinearForLastLayer)
    ]
    for chunk, name, _ in value_heads:
        delattr(chunk, name)
    try:
        yield
    finally:
        for chunk, name, head in value_heads:
            setattr(chunk, name, head)


def _load_hf_weights_with_mbridge(ddp_model, args, load_path: str) -> None:
    # [FUNCTION DOC] HELPER: Imports (lazily) the mbridge plugin registrations and AutoBridge. When
    # train offload is on, disables the memory saver while loading (staging copies must not be
    # tracked). Logs, then with the critic value head hidden, builds a bridge from the pretrained
    # path (trusting remote code) and loads the HF weights into the model in memory-efficient mode;
    # finally synchronizes CUDA. Mutates model weights; reads disk.
    """Raw-mode HF load through mbridge for models Megatron-Bridge lacks; staging copies bypass the memory saver."""
    from contextlib import nullcontext

    import torch

    import miles_plugins.mbridge  # noqa: F401
    from mbridge import AutoBridge

    if args.offload_train:
        from torch_memory_saver import torch_memory_saver

        bypass = torch_memory_saver.disable()
    else:
        bypass = nullcontext()
    logger.info(f"Load checkpoint from HuggingFace model into Megatron via mbridge (path={load_path})")
    with _hide_critic_value_head_from_hf_load(ddp_model), bypass:
        bridge = AutoBridge.from_pretrained(load_path, trust_remote_code=True)
        bridge.load_weights(ddp_model, load_path, memory_efficient=True)
    torch.cuda.synchronize()


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    # [FUNCTION DOC] HELPER: Loads HF-format weights into the model. If the conversion mode is not
    # "bridge", uses the mbridge path; otherwise uses Megatron-Bridge's AutoBridge from a
    # pretrained HF path (trusting remote code), with the model temporarily patched for the bridge
    # and the critic value head hidden. Then, when optimizer exists and fp16 or bf16 is on, asserts
    # main params are not loaded from checkpoint and asks the optimizer to refresh its main
    # parameters from the model parameters. Returns the tuple (iteration 0, floating-point
    # operation count 0), matching what Megatron reports for such loads.
    if args.megatron_to_hf_mode != "bridge":
        _load_hf_weights_with_mbridge(ddp_model, args, load_path)
    else:
        from megatron.bridge import AutoBridge

        logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

        with megatron_bridge_utils.patch_megatron_model(ddp_model), _hide_critic_value_head_from_hf_load(ddp_model):
            bridge = AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
            bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    # [FUNCTION DOC] HELPER: True if the directory has at least one entry (scans lazily).
    with os.scandir(path) as it:
        return any(it)
