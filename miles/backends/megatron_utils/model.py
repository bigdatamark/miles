# [MODULE DOC] Megatron-side model lifecycle for training: building the DDP-wrapped model chunks,
# the optimizer and the learning-rate scheduler, loading checkpoints, running forward-only
# inference (log-probs and the like), running one or many training steps over a rollout, and
# saving checkpoints.
# Several names are re-exported or imported only for side-effect-free compatibility (the LoRA
# bridge helpers are imported with a lint suppression), so callers may import them from here.
# Most functions read process-global Megatron state (parallel groups, global args) and require
# an initialized distributed environment and GPUs.
from __future__ import annotations

import dataclasses
import gc
import logging
import math
from argparse import Namespace
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model

from miles.backends.megatron_utils.ft.indep_dp import allreduce_grads_and_losses_across_replicas
from miles.backends.megatron_utils.local_weight_checksum import dump_local_weight_checksums
from miles.backends.megatron_utils.optimizer_state_reset import reset_optimizer_states
from miles.backends.training_utils.data.rollout import DataIterator, get_batch
from miles.backends.training_utils.data.sampling_mask import get_rollout_sampling_masks
from miles.backends.training_utils.loss.objective import loss_function
from miles.backends.training_utils.metrics.checks import check_grad_norm, check_kl
from miles.backends.training_utils.metrics.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_train_step,
)
from miles.backends.training_utils.types import TrainStepOutcome
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.audit_utils.witness.module import witness_dump_and_clear_stale
from miles.utils.dumper_utils import DumperMegatronUtil, DumperPhase
from miles.utils.lora.utils import is_multi_lora_enabled
from miles.utils.memory_utils import clear_memory
from miles.utils.test_utils.ft_test_actions import FTTestActionActorExecutor
from miles.utils.tracking_utils.structured_log import log_structured

from ...utils.misc import filter_keys
from ..training_utils.parallel import get_parallel_state
from .checkpoint import load_checkpoint, save_checkpoint, save_checkpoint_with_lora
from .ci_utils import (
    check_model_hashes,
    check_peak_gpu_memory_after_load,
    compute_model_hashes_by_layer,
    save_model_hashes,
)
from .initialize import is_first_replica_megatron_main_rank
from .lora.utils import is_lora_enabled, is_lora_model
from .model_provider import get_model_provider_func
from .parallel import get_packed_seq_params

logger = logging.getLogger(__name__)


def _has_loadable_ckpt(load_dir: str | None) -> bool:
    # [FUNCTION DOC] HELPER: Returns true only if the given load directory path is set (not empty or
    # missing), exists as a directory, and contains at least one entry. Pure filesystem read, no
    # side effects. Used to decide whether to load a checkpoint or start from freshly initialized
    # weights.
    """Whether ``--load`` holds anything; ``load_checkpoint`` dispatches dist vs HF itself."""
    return bool(load_dir) and Path(load_dir).is_dir() and any(Path(load_dir).iterdir())


from .fp32_param_utils import enforce_marked_param_dtypes
from .lora.bridge import _ensure_model_list, _setup_lora_model_via_bridge  # noqa: F401


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler | None:
    # [FUNCTION DOC] Builds Megatron's optimizer parameter scheduler (learning rate and weight decay
    # schedules) for the given optimizer. The scheduler works in units of samples, so every
    # iteration count is multiplied by the global batch size.
    # Side effect: mutates the arguments object. It sets the total training iterations to
    # (number of rollouts * rollout batch size * samples per prompt) integer-divided by the global
    # batch size, and, if the learning-rate decay iterations is unset, sets it to the total
    # training iterations.
    # Steps: decay steps = decay iterations * global batch size; weight-decay increment steps =
    # total training iterations * global batch size; the optional warmup-stable-decay steps =
    # its iterations * global batch size when given, else none. Warmup steps = warmup fraction *
    # decay steps when a warmup fraction is set (this can be a non-integer), otherwise warmup
    # iterations * global batch size. All other schedule knobs (initial/max/min learning rate,
    # decay style, start/end weight decay and its increment style, the checkpoint-scheduler
    # use/override flags, and the warmup-stable-decay style) are passed straight through from
    # the arguments. Returns the new scheduler; it never returns none despite the optional
    # return annotation.
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training.
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


# ---------------------------------------------------------------------------
# Model + Optimizer setup
# ---------------------------------------------------------------------------


def _is_muon_optimizer(optimizer: str | None) -> bool:
    # [FUNCTION DOC] HELPER: True when the optimizer name is set and contains the word "muon"
    # (case-insensitive), which also matches variants such as distributed or adaptive Muon.
    return optimizer is not None and "muon" in optimizer.lower()


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None]:
    # [FUNCTION DOC] Builds the model chunks, then the optimizer and scheduler. Raises assertion
    # errors if MoE upcycling is on, or if neither a load path nor a pretrained checkpoint is set.
    # Model construction has three paths: (1) multi-LoRA, or single LoRA for the actor role in
    # bridge conversion mode, goes through the LoRA bridge helper and then forces the dtypes of
    # parameters marked for it; (2) otherwise a model provider function is obtained for the role;
    # for the actor with LoRA enabled the provider is wrapped by a model-specific native LoRA
    # injector (Inkling when the custom provider path mentions it, Kimi K3 when the model name
    # mentions it, which also patches the parameter/grad buffer when train offload is on), and any
    # other model raises an assertion error telling the user to use bridge mode; Inkling also
    # requires the plain LoRA type; (3) the Megatron get-model helper then builds the DDP model
    # chunks from the provider.
    # If the debug flag to disable the optimizer is set, logs a warning (first replica main rank
    # only) and returns the model with no optimizer and no scheduler.
    # Optimizer config is built from the optimizer-config dataclass fields that also exist on the
    # arguments; timers are cleared; when streaming optimizer state to disk is on and the
    # optimizer is not Muon, deferred main-parameter initialization is switched on.
    # Muon branch: optionally sets up on-disk Muon state; for Inkling it forces off the split of
    # fused q/k/v (with an info log) so whole matrices are orthogonalized; builds the Muon
    # optimizer, with the layer-wise distributed mode on when the optimizer name contains "dist".
    # Multi-LoRA branch (checked after Muon): validates optimizer args and returns the model with
    # no optimizer and no scheduler (per-tenant optimizers are created later when a slot loads).
    # Default branch: builds the standard Megatron optimizer. Afterwards, for non-Muon with disk
    # streaming enabled, attaches optimizer state streaming. Finally builds the scheduler.
    # Returns (model chunks, optimizer or none, scheduler or none).
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance, or ``None`` when
              ``--debug-disable-optimizer`` is set.
            - The learning-rate/weight-decay scheduler tied to the optimizer, or
              ``None`` when ``--debug-disable-optimizer`` is set.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    # Multi-LoRA and single-LoRA (actor, bridge) both build via the bridge helper,
    # which picks the adapter type internally.
    if is_multi_lora_enabled(args) or (
        is_lora_enabled(args) and role == "actor" and args.megatron_to_hf_mode == "bridge"
    ):
        model = _setup_lora_model_via_bridge(args)
        enforce_marked_param_dtypes(model)
    else:
        provider_func = get_model_provider_func(args, role)
        if is_lora_enabled(args) and role == "actor":
            if "inkling" in (args.custom_model_provider_path or ""):
                assert args.lora_type == "lora", "Native Inkling does not implement --lora-type canonical_lora"
                from miles_plugins.models.inkling.lora import wrap_model_provider_with_inkling_lora

                provider_func = wrap_model_provider_with_inkling_lora(provider_func, args)
            # TODO: will rewrite in native lora refactor
            elif "kimi_k3" in (args.model_name or "").lower():
                from miles_plugins.models.kimi_k3.lora import wrap_model_provider_with_kimi_k3_lora

                from .lora.utils import patch_param_grad_buffer_for_colocate_mode_lora

                provider_func = wrap_model_provider_with_kimi_k3_lora(provider_func, args)
                if args.offload_train:
                    patch_param_grad_buffer_for_colocate_mode_lora()
            else:
                raise AssertionError(
                    "Native LoRA injection is only implemented for Inkling and Kimi K3; "
                    "use --megatron-to-hf-mode bridge"
                )
        model = get_model(provider_func, ModelType.encoder_or_decoder)

    if args.debug_disable_optimizer:
        if is_first_replica_megatron_main_rank():
            logger.warning(
                "Skipping Megatron optimizer and LR scheduler initialization "
                "because --debug-disable-optimizer is set."
            )
        return model, None, None

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    if args.stream_optimizer_state_to_disk and not _is_muon_optimizer(config.optimizer):
        config.defer_main_param_initialization = True
    config.timers = None

    if _is_muon_optimizer(config.optimizer):
        if args.stream_optimizer_state_to_disk:
            from miles_plugins.optimizers.nvme_stream import setup_muon_state_on_disk

            setup_muon_state_on_disk(args)
        if config.muon_split_qkv and "inkling" in (getattr(args, "custom_model_provider_path", None) or ""):
            if is_first_replica_megatron_main_rank():
                logger.info(
                    "Inkling fused qkvr detected: forcing muon_split_qkv=False " "(whole-matrix orthogonalization)."
                )
            config.muon_split_qkv = False
        optimizer = get_megatron_muon_optimizer(
            config=config,
            model_chunks=model,
            use_gloo_process_groups=args.use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in config.optimizer.lower(),
        )
    elif is_multi_lora_enabled(args):
        from miles.backends.megatron_utils.lora.optimizer import validate_multi_lora_optimizer_args

        # per-tenant SlotOptimizers are built at load_slot; there is no pool optimizer
        validate_multi_lora_optimizer_args(args)
        return model, None, None
    else:
        optimizer = get_megatron_optimizer(
            config=config,
            model_chunks=model,
            use_gloo_process_groups=args.use_gloo_process_groups,
        )

    if args.stream_optimizer_state_to_disk and not _is_muon_optimizer(config.optimizer):
        # Muon took the chunked-offloader route above; this store is DistOpt-only.
        from miles_plugins.optimizers.nvme_stream import setup_optimizer_state_streaming

        setup_optimizer_state_streaming(args, optimizer)

    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    return model, optimizer, opt_param_scheduler


# ---------------------------------------------------------------------------
# Forward pre-hook helpers
# ---------------------------------------------------------------------------


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    # [FUNCTION DOC] Calls enable-forward-pre-hook on every chunk. Asserts each chunk is a
    # Megatron DDP wrapper (assertion error otherwise). Mutates the chunks; returns nothing.
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    # [FUNCTION DOC] Calls disable-forward-pre-hook on every chunk, passing the param-sync flag
    # (when true, outstanding parameter all-gathers are completed during disabling). Asserts each
    # chunk is a Megatron DDP wrapper. Mutates the chunks; returns nothing.
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    # [FUNCTION DOC] True exactly when both the distributed optimizer and overlapped parameter
    # gather are enabled in the arguments; in that case the forward pre-hook must be turned off
    # around the first step and around saving.
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


# ---------------------------------------------------------------------------
# Forward-only inference
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    rollout_id: int,
    store_prefix: str = "",
    fp32_output: bool = True,
    use_rollout_sampling_mask: bool = False,
) -> dict[str, list[torch.Tensor]]:
    # [FUNCTION DOC] Runs inference-only pipeline passes over every step of a rollout and returns
    # the callback outputs (for example per-token log-probs) aggregated on the last pipeline stage.
    # Runs with gradients disabled (decorator). Inputs: the post-forward callback, arguments, model
    # chunks, one data iterator per chunk, a list with the microbatch count of each step, the
    # rollout id (used only for dump naming), an output key prefix, whether Megatron should upcast
    # the full output to fp32 (default true), and whether to score over the rollout's captured
    # sampling support (default false).
    # Steps: create a forward-only dumper helper; reset every data iterator; fetch the model
    # config; put all chunks in eval mode (disables dropout); if a custom before-log-prob hook path
    # is configured, load it and call it with the arguments, model and prefix; for each step run
    # the Megatron forward-backward function in forward-only mode with non-loss data collection,
    # appending its results to a list; restore train mode on all chunks; finalize the dumper.
    # Model timers in the config are cleared. On the last pipeline stage only, the collected
    # results are aggregated using the first data iterator and each key is stored under the
    # prefix plus key; other stages return an empty mapping. Returns that mapping from key to a
    # list of tensors.
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f: Post-forward callback used to compute and package outputs to collect.
        args: Runtime arguments.
        model: Sequence of DDP-wrapped model chunks.
        data_iterator: Iterable(s) yielding batches for inference.
        num_microbatches: Number of microbatches per rollout step.
        rollout_id: Rollout identifier (selects the per-rollout dump subdirectory).
        store_prefix: Prefix to prepend to stored output keys.
        fp32_output: Whether Megatron should upcast the complete model output to FP32.
        use_rollout_sampling_mask: Whether to score over each rollout token's
            captured sampling support.

    Returns:
        Aggregated outputs keyed by ``store_prefix + key``.
    """

    dumper_phase_util = DumperMegatronUtil(
        args, model, DumperPhase.FWD_ONLY, rollout_id=rollout_id, store_prefix=store_prefix
    )

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])

    @dumper_phase_util.wrap_forward_step
    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        # [FUNCTION DOC] HELPER: Forward step handed to Megatron's pipeline schedule (wrapped by the
        # dumper). Fetches a batch with the fixed forward-only keys (tokens, loss masks, multimodal
        # inputs, total and response lengths, max sequence lengths, witness ids, plus the sampling
        # mask ids and offsets when sampling-support scoring is on), using the configured padding
        # multiplier, query/key/value format and context-parallel allgather flag. Builds packed
        # sequence params; when the batch carries per-adapter token counts, sets the tokens per
        # adapter slot on the model (multi-LoRA). Runs the model with no labels or attention mask,
        # the input loss mask, optional witness ids (when witness is enabled), optional multimodal
        # inputs and the fp32 output flag. Returns the output tensor and a partially applied
        # callback that, given the output, computes the collected values using the unconcatenated
        # tokens, lengths, entropy flag, max sequence lengths and (optionally) the sampling mask.
        # The return_schedule_plan argument is accepted but unused.
        """Return the model output and its batch-bound loss callback."""

        forward_only_keys = [
            "tokens",
            "loss_masks",
            "multimodal_train_inputs",
            "total_lengths",
            "response_lengths",
            "max_seq_lens",
            "witness_ids",
        ]
        if use_rollout_sampling_mask:
            forward_only_keys.extend(["rollout_sampling_mask_ids", "rollout_sampling_mask_offsets"])
        batch = get_batch(
            data_iterator,
            forward_only_keys,
            args.data_pad_size_multiplier,
            args.qkv_format,
            allgather_cp=args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = get_packed_seq_params(batch, args)
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        rollout_sampling_mask = get_rollout_sampling_masks(batch) if use_rollout_sampling_mask else None

        if "adapter_token_counts" in batch:
            from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

            set_tokens_per_adapter_slot(model, batch["adapter_token_counts"])

        output_tensor = model(
            input_ids=tokens,
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=packed_seq_params,
            loss_mask=batch["input_loss_masks"],
            **(filter_keys(batch, ["witness_ids"]) if args.enable_witness else {}),
            **(batch["multimodal_train_inputs"] if batch["multimodal_train_inputs"] is not None else {}),
            fp32_output=fp32_output,
        )

        callback_kwargs = dict(
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=args.use_rollout_entropy,
            max_seq_lens=batch.get("max_seq_lens", None),
        )
        if use_rollout_sampling_mask:
            callback_kwargs["rollout_sampling_mask"] = rollout_sampling_mask

        return output_tensor, partial(f, **callback_kwargs)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from miles.utils.function_registry import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    for step_id in range(num_steps_per_rollout):
        # collect_non_loss_data
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            collect_non_loss_data=True,
        )

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    dumper_phase_util.finalize(model)

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        aggregated = aggregate_forward_results(forward_data_store, data_iterator[0], args, store_prefix="")
        for key, value in aggregated.items():
            rollout_data[f"{store_prefix}{key}"] = value
    return rollout_data


def _zero_grads(model: Sequence[DDP], optimizer: MegatronOptimizer | None, disable_optimizer: bool) -> None:
    # [FUNCTION DOC] HELPER: Zeroes the gradient buffer of every DDP chunk, and, unless the
    # optimizer is disabled, also zeroes the optimizer's gradients. Mutates in place.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    if not disable_optimizer:
        optimizer.zero_grad()


def run_forward_backward_pass(
    args, dumper_phase_util, data_iterator, model, num_microbatches, num_rollouts, forward_only=False
):
    # [FUNCTION DOC] Runs Megatron's forward-backward schedule once over the microbatches and
    # returns its per-microbatch loss outputs; it does not touch the optimizer. Inputs: arguments,
    # dumper helper, data iterators, model chunks, microbatch count, rollout count (for loss
    # scaling) and a forward-only flag (default false) that runs under no-grad and skips backward.
    # The sampling mask keys (mask ids and offsets) are requested only when sampling-support replay
    # is on and the loss type is policy loss or score centering.
    # KNOWN ISSUE: the parameter named forward-only shadows the module-level forward-only function;
    # the type annotations are absent.
    """One pipeline forward/backward pass over the microbatches; no optimizer interaction."""

    sampling_mask_keys = (
        ("rollout_sampling_mask_ids", "rollout_sampling_mask_offsets")
        if args.use_sampling_support_replay and args.loss_type in ("policy_loss", "score_centering")
        else ()
    )

    @dumper_phase_util.wrap_forward_step
    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict]],
    ]:
        # [FUNCTION DOC] HELPER: Training forward step (wrapped by the dumper). Fetches a batch
        # with a fixed key list: tokens, multimodal inputs, packed sequence params, lengths, loss
        # masks, log-probs, reference log-probs, values, advantages, returns, rollout log-probs,
        # rollout top-k ids/lengths/log-probs, max sequence lengths, witness ids, on-policy
        # distillation reverse KL, rollout mask sums, loss weights, target tokens, sample indices
        # and the optional sampling mask keys. Sets per-adapter token counts for multi-LoRA when
        # present. Temporarily sets the stage of every registered replay manager to the replay
        # forward stage and restores each previous stage after the model call (not restored if the
        # model raises).
        # Two model paths: when a schedule plan is requested (combined 1F1B), asserts that MTP
        # training and witness are off and builds a schedule plan from the tokens, with no
        # labels/attention mask and the input loss mask; otherwise calls the model with the same
        # inputs plus optional witness ids and multimodal inputs, upcasting output to fp32 unless
        # the loss type is policy loss, SFT loss or score centering.
        # Returns the output (tensor or plan) and the loss function partially applied with
        # arguments, batch, microbatch count, Megatron loss scaling on and the rollout count.
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                "rollout_topk_token_ids",
                "rollout_topk_lengths",
                "rollout_topk_log_probs",
                "max_seq_lens",
                "witness_ids",
                "opd_reverse_kl",
                "rollout_mask_sums",
                "loss_weights",
                "target_tokens",
                "sample_indices",
                *sampling_mask_keys,
            ],
            args.data_pad_size_multiplier,
            args.qkv_format,
            allgather_cp=args.allgather_cp,
        )

        if "adapter_token_counts" in batch:
            from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

            set_tokens_per_adapter_slot(model, batch["adapter_token_counts"])

        from miles.utils.replay_base import all_replay_managers

        old_stages = [m.stage for m in all_replay_managers]
        for m in all_replay_managers:
            m.stage = "replay_forward"

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            assert not args.enable_witness, "Witness is not supported with combined 1f1b (build_schedule_plan)"
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=None,
                attention_mask=None,
                labels=None,
                packed_seq_params=get_packed_seq_params(batch, args),
                loss_mask=batch["input_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": get_packed_seq_params(batch, args),
                # Aligned with input_ids. With labels=None, Megatron derives the MTP labels
                # from input_ids and shifts this mask itself; without MTP it never reads it.
                "loss_mask": batch["input_loss_masks"],
                **(filter_keys(batch, ["witness_ids"]) if args.enable_witness else {}),
            }

            if (x := batch["multimodal_train_inputs"]) is not None:
                forward_kwargs.update(x)

            output_tensor = model(
                **forward_kwargs, fp32_output=args.loss_type not in ("policy_loss", "sft_loss", "score_centering")
            )

        for m, old_stage in zip(all_replay_managers, old_stages, strict=True):
            m.stage = old_stage

        return output_tensor, partial(
            loss_function,
            args,
            batch,
            num_microbatches,
            apply_megatron_loss_scaling=True,
            num_rollouts=num_rollouts,
        )

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    with torch.no_grad() if forward_only else nullcontext():
        return forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=forward_only,
        )


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: OptimizerParamScheduler | None,
    num_microbatches: int,
    num_rollouts: int,
    witness_info: WitnessInfo | None,
    attempt: int,
    ft_test_action_executor: FTTestActionActorExecutor | None = None,
) -> tuple[dict[str, float], float, TrainStepOutcome]:
    # [FUNCTION DOC] Executes one optimizer step of a rollout. Returns (reduced loss mapping, grad
    # norm, outcome). The incoming arguments object is overwritten by the global Megatron
    # arguments, and the optimizer may be absent or disabled (then no stepping happens).
    # Steps: zero grads; run the optional custom before-train-step hook; run forward/backward.
    # If independent data parallelism has more than one replica (only step 0 is supported):
    # optionally trigger a fault-tolerance test crash, then all-reduce grads and losses across
    # replicas (loss scaled by rollouts unless per-token loss); a failed reduce marks the outcome
    # as discarded-should-retry and the step invalid.
    # If NaN/Inf checking in loss and grads is off (default on), prepares grads on the optimizer
    # and marks the step invalid when an overflow flag is found or the grad norm is NaN/Inf.
    # In CI with MTP training and short responses, checks that only MTP params have gradients.
    # When the outcome is normal, finalizes the dumper while grads are still attached. If the
    # step is valid and the optimizer is enabled: step the optimizer (asserting success), take the
    # new grad norm, and advance the scheduler by the rollout count. Then zero grads again, emit a
    # structured log line, and, for a normal outcome, dump local weight checksums and (with witness)
    # dump and clear stale witness data.
    # Loss is returned only on the last pipeline stage (virtual stages ignored) with a normal
    # outcome: the independent-DP reduced loss if used, else the aggregated train losses. All other
    # cases return an empty mapping.
    # KNOWN ISSUE: when the step is invalid because of NaN/Inf with a normal outcome the optimizer
    # step is skipped silently but the outcome is still normal; the optimizer-scheduler parameter
    # is otherwise used only for the scheduler step.
    """Run pipeline forward/backward, then step the optimizer and scheduler when gradients are valid."""
    args = get_args()
    parallel_state = get_parallel_state()
    dumper_phase_util = DumperMegatronUtil(args, model, DumperPhase.FWD_BWD, rollout_id=rollout_id)
    disable_optimizer = args.debug_disable_optimizer or optimizer is None
    _zero_grads(model, optimizer, disable_optimizer)

    if args.custom_megatron_before_train_step_hook_path:
        from miles.utils.function_registry import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    losses_reduced = run_forward_backward_pass(
        args, dumper_phase_util, data_iterator, model, num_microbatches, num_rollouts
    )

    outcome = TrainStepOutcome.NORMAL
    grad_norm = 0.0
    valid_step = True
    indep_dp_loss_reduced: dict[str, float] = {}

    if parallel_state.indep_dp.size > 1:
        assert step_id == 0, "indep-dp does not support multi step per train yet"

        if ft_test_action_executor is not None:
            ft_test_action_executor.maybe_crash(rollout_id=rollout_id, attempt=attempt)

        metric_num_rollouts = None if args.calculate_per_token_loss else num_rollouts
        ok, indep_dp_loss_reduced = allreduce_grads_and_losses_across_replicas(
            args, model, parallel_state, losses_reduced=losses_reduced, num_rollouts=metric_num_rollouts
        )
        if not ok:
            outcome = TrainStepOutcome.DISCARDED_SHOULD_RETRY
            valid_step = False

    if (not disable_optimizer) and (not getattr(args, "check_for_nan_in_loss_and_grad", True)):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training and args.rollout_max_response_len <= 128:
        # under response length <= 128, all outputs are truncated and loss mask is all zeros, so only MTP parameters have non-zero gradients
        from miles.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    # Dump backward tensors while gradients are still attached. The optimizer
    # step and subsequent zero_grad release them.
    if outcome == TrainStepOutcome.NORMAL:
        dumper_phase_util.finalize(model)

    if not disable_optimizer and valid_step:
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        assert update_successful
        opt_param_scheduler.step(increment=num_rollouts)

    _zero_grads(model, optimizer, disable_optimizer)

    log_structured(
        logger.info,
        tag="train",
        op="train_step",
        rollout=rollout_id,
        step=step_id,
        attempt=attempt,
        outcome=outcome.name,
        valid_step=valid_step,
    )

    if outcome == TrainStepOutcome.NORMAL:
        dump_local_weight_checksums(args=args, model=model, optimizer=optimizer)
        if args.enable_witness:
            witness_dump_and_clear_stale(model=model, witness_info=witness_info, optimizer=optimizer)

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            metric_num_rollouts = None if args.calculate_per_token_loss else num_rollouts
            loss_reduced = (
                indep_dp_loss_reduced
                if parallel_state.indep_dp.size > 1
                else aggregate_train_losses(losses_reduced, metric_num_rollouts)
            )
            return loss_reduced, grad_norm, outcome

    return {}, grad_norm, outcome


def finalize_model_grads_with_empty_cache(*args, **kwargs):
    # [FUNCTION DOC] Replacement for Megatron's finalize-model-grads hook. Checks free versus
    # total memory of the current GPU and, if less than ten percent is free, clears memory caches.
    # Then reduces the marked LoRA grads of the model chunks (first positional argument) and
    # delegates to Megatron's finalize-model-grads with all given arguments, returning its result.
    # Side effects: GPU memory query and cache clearing, in-place gradient reduction.
    # TODO: this is an ad-hoc method and we should figure out why the oom happens in the first place.
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    if free / total < 0.1:
        clear_memory()
    from .lora.utils import reduce_marked_lora_grads

    reduce_marked_lora_grads(args[0])
    return finalize_model_grads(*args, **kwargs)


def setup_train_iteration_config(args, model, optimizer, disable_optimizer):
    # [FUNCTION DOC] Configures the shared Megatron model config for training and returns it. Gets
    # the config from the first chunk; sets the gradient scaling function to none when the
    # optimizer is disabled, else the optimizer's loss scaler; clears timers. If the first chunk is
    # a DDP wrapper and overlapped grad reduce is on, sets the no-sync function (one function if
    # one chunk, else a list), and if grad-reduce alignment is on, the start-grad-sync function(s).
    # If overlapped param gather and its alignment are on, sets the start-param-sync function(s).
    # Finally installs the memory-aware finalize-grads function. Mutates the config.
    """Set the DDP gradient and parameter sync hooks for this training iteration."""
    config = get_model_config(model[0])
    config.grad_scale_func = None if disable_optimizer else optimizer.scale_loss
    config.timers = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        no_sync_funcs = [model_chunk.no_sync for model_chunk in model]
        config.no_sync_func = no_sync_funcs[0] if len(model) == 1 else no_sync_funcs
        if args.align_grad_reduce:
            grad_sync_funcs = [model_chunk.start_grad_sync for model_chunk in model]
            config.grad_sync_func = grad_sync_funcs[0] if len(model) == 1 else grad_sync_funcs
    if args.overlap_param_gather and args.align_param_gather:
        param_sync_funcs = [model_chunk.start_param_sync for model_chunk in model]
        config.param_sync_func = param_sync_funcs[0] if len(model) == 1 else param_sync_funcs
    config.finalize_model_grads_func = finalize_model_grads_with_empty_cache
    return config


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: OptimizerParamScheduler | None,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    num_rollouts: Sequence[int],
    witness_info: WitnessInfo | None,
    attempt: int,
    ft_test_action_executor: FTTestActionActorExecutor | None = None,
) -> TrainStepOutcome:
    # [FUNCTION DOC] Trains on one rollout: one optimizer step per entry of the microbatch list.
    # Reads the global Megatron arguments (not passed in). Asserts the microbatch and rollout
    # count lists have equal length, and for independent data parallelism that there is only one
    # step. Returns the outcome of the last step.
    # Setup: reset data iterators, set train mode, configure the shared model config, optionally
    # reset optimizer states (logged on first replica main rank), and, with manual garbage
    # collection, disable automatic GC and collect once (asserting a non-negative interval).
    # If the forward pre-hook must be blocked (distributed optimizer with overlapped param
    # gather), disables the hook without param sync and temporarily clears the param sync function
    # so the first step makes no param all-gather; both are restored after step 0.
    # Per step: call the single-step trainer. After step 0 restore the pre-hook and param sync
    # function. With MTP training, read the MTP loss from Megatron's logging tracker (single MTP
    # layer assumed, scaled by 1 or by the inverse of the step's microbatch count), clear the
    # tracker, and in CI check the loss bounds. On a normal outcome, on the first replica main
    # rank only: build extra metrics (MTP loss and per-param-group learning rates from the
    # scheduler), log the train step, in CI run the KL check (unless disabled) and the grad norm
    # check, and log a line prefixed by the role (the model's role attribute, default actor;
    # prefix omitted for actor). After all steps, disable the pre-hook if it was enabled.
    # KNOWN ISSUE: the docstring omits the witness info, attempt and fault-tolerance executor
    # parameters, and the return value. The variable that remembers whether the pre-hook is
    # enabled is set to false in the blocking branch redundantly. With zero steps the returned
    # outcome would be unbound.
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
        num_rollouts (Sequence[int]): Rollout count per step (total across DP).
    """
    parallel_state = get_parallel_state()
    args = get_args()
    disable_optimizer = args.debug_disable_optimizer or optimizer is None

    assert len(num_microbatches) == len(num_rollouts), (
        f"num_microbatches and num_rollouts must have the same length, "
        f"got {len(num_microbatches)} vs {len(num_rollouts)}"
    )

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    config = setup_train_iteration_config(args, model, optimizer, disable_optimizer)

    pre_hook_enabled = False

    if args.reset_optimizer_states and not disable_optimizer:
        if is_first_replica_megatron_main_rank():
            logger.info("Reset optimizer states")
        reset_optimizer_states(optimizer)

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    if parallel_state.indep_dp.size > 1:
        assert num_steps_per_rollout == 1, "indep_dp is incompatible with num_steps_per_rollout>1 currently"

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm, train_step_outcome = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            num_rollouts[step_id],
            witness_info=witness_info,
            attempt=attempt,
            ft_test_action_executor=ft_test_action_executor,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        mtp_losses = None
        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1.0 if args.calculate_per_token_loss else 1 / num_microbatches[step_id]
            MTPLossLoggingHelper.reduce_loss_in_tracker()
            tracker = MTPLossLoggingHelper.tracker
            # here we assume only one mtp layer
            if "values" in tracker:
                mtp_losses = (tracker["values"] * mtp_loss_scale).item()
            elif "loss_values" in tracker:
                mtp_losses = (tracker["loss_values"] * mtp_loss_scale).item()

            if mtp_losses is not None:
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from miles.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses)

        # per train step log.
        if (train_step_outcome == TrainStepOutcome.NORMAL) and is_first_replica_megatron_main_rank():
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"

            extra_metrics = {}
            if args.enable_mtp_training and mtp_losses is not None:
                extra_metrics["mtp_loss"] = mtp_losses

            if not disable_optimizer:
                for param_group_id, param_group in enumerate(optimizer.param_groups):
                    extra_metrics[f"lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            log_dict = log_train_step(
                args=args,
                loss_dict=loss_dict,
                grad_norm=grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                num_steps_per_rollout=num_steps_per_rollout,
                role=role,
                extra_metrics=extra_metrics,
                should_log=True,
            )

            if args.ci_test and not args.ci_disable_kl_checker:
                check_kl(args, log_dict, step_id, accumulated_step_id)

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_test:
                check_grad_norm(
                    args=args,
                    grad_norm=grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    role=role,
                    rank=parallel_state.effective_dp.rank,
                )

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    return train_step_outcome


def save(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: OptimizerParamScheduler | None,
    checkpointing_context: dict | None = None,
    non_persistent_ckpt: bool = False,
    *,
    snapshot_publisher: SnapshotPublisher | None = None,
) -> None:
    # [FUNCTION DOC] Saves a checkpoint for the given iteration. Reads global Megatron arguments.
    # In CI with model-hash saving on, first computes per-layer model hashes. If the forward
    # pre-hook must be blocked, disables it (with param sync) before saving and re-enables it
    # afterwards. LoRA models are saved through the LoRA saver, which also takes the optional
    # snapshot publisher (keyword-only); other models use Megatron's saver with zero floating
    # point operations, no data iterator, no common-state preprocessing, the checkpointing context
    # and the non-persistent flag. Afterwards writes the model hashes if computed.
    # KNOWN ISSUE: the checkpointing context and non-persistent flag are ignored for LoRA models,
    # and the hooks are not restored if saving raises.
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        checkpointing_context (dict | None): Context passed to Megatron's save_checkpoint
            (e.g. ``{'local_checkpoint_manager': manager}`` for in-memory checkpoints).
        non_persistent_ckpt (bool): If True, save a non-persistent (in-memory) checkpoint.
    """
    args = get_args()
    hashes = None
    if args.ci_test and args.ci_save_model_hash:
        hashes = compute_model_hashes_by_layer(model)
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)

    if is_lora_model(model):
        save_checkpoint_with_lora(iteration, model, optimizer, opt_param_scheduler, publisher=snapshot_publisher)
    else:
        save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far=0,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
            checkpointing_context=checkpointing_context,
            non_persistent_ckpt=non_persistent_ckpt,
        )

    if hashes is not None:
        save_model_hashes(args, model, iteration, hashes)
    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def initialize_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
    checkpointing_context=None,
) -> tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None, LoadCheckpointOutput]:
    # [FUNCTION DOC] Convenience composition: builds the model and optimizer for the role, then
    # loads checkpoint state into them. Returns (model chunks, optimizer, scheduler, load output).
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").
        checkpointing_context: pass-through checkpointing context

    Returns:
        tuple[list[DDP], MegatronOptimizer, OptimizerParamScheduler, LoadCheckpointOutput]:
            DDP-wrapped model chunks, optimizer, scheduler, and what the load answered.
    """
    model, optimizer, opt_param_scheduler = build_model_and_optimizer(args, role=role)

    load_output = load_model_state(
        args,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        role=role,
        checkpointing_context=checkpointing_context,
    )
    return model, optimizer, opt_param_scheduler, load_output


def build_model_and_optimizer(
    args: Namespace, *, role: str
) -> tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None]:
    # [FUNCTION DOC] Builds the model, optimizer and scheduler through the setup function, tags the
    # first model chunk with the role (read later for log prefixes), clears GPU/host memory caches,
    # and returns the three objects.
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    clear_memory()
    return model, optimizer, opt_param_scheduler


@dataclasses.dataclass(frozen=True)
class LoadCheckpointOutput:
    # [CLASS DOC] Immutable result of loading model state: the rollout id (iteration) found in
    # the loaded checkpoint (zero when nothing was loaded) and the rollout id training should
    # start from.
    loaded_rollout_id: int
    start_rollout_id: int


def load_model_state(
    args: Namespace,
    *,
    model: list[DDP],
    optimizer: MegatronOptimizer | None,
    opt_param_scheduler: OptimizerParamScheduler | None,
    role: str,
    checkpointing_context: dict | None,
) -> LoadCheckpointOutput:
    # [FUNCTION DOC] Loads checkpoint state into the model (and optimizer/scheduler) and works out
    # where training resumes. Keyword-only except the arguments.
    # With multi-LoRA, adapter parameters are hidden during the load so the bridge's conversion
    # walk does not see them. The load happens when the load path is unset (the pretrained
    # checkpoint then covers it) or the directory is non-empty; otherwise a warning is logged on
    # the first replica main rank and the iteration is zero.
    # Native Inkling LoRA (LoRA on, actor role, non-bridge mode, adapter path set, custom provider
    # mentions Inkling): if the adapter safetensors file exists and the native optimizer state was
    # not restored from the checkpoint, loads the adapter into the model and refreshes the
    # optimizer's fp32 master params so the first step does not overwrite the adapter.
    # Then checks peak GPU memory (CI), clears memory, and checks model hashes (CI). Steps the
    # scheduler by iteration times global batch size, unless the scheduler state came from the
    # checkpoint and iteration is positive (avoids double counting).
    # Start rollout: with finetune and no LoRA, asserts the loaded iteration is zero and starts at
    # zero; otherwise starts at iteration plus one. Returns the loaded and start ids.
    if is_multi_lora_enabled(args):
        # Hide adapter params so the bridge's conversion-task walk doesn't see them
        # while loading the base checkpoint.
        from megatron.bridge.peft.multi_lora_layers import hide_adapters

        load_ctx = hide_adapters(model)
    else:
        load_ctx = nullcontext()

    load_dir = getattr(args, "load", None)
    native_optimizer_restored = False
    # --load may be unset: setup_model_and_optimizer already asserted pretrained_checkpoint covers it.
    if load_dir is None or _has_loadable_ckpt(load_dir):
        with load_ctx:
            iteration, _, native_optimizer_restored = load_checkpoint(
                model,
                optimizer,
                opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=False,
            )
    else:
        if is_first_replica_megatron_main_rank():
            logger.warning("--load %r is empty; starting from model_provider-initialized weights", load_dir)
        iteration = 0

    if (
        is_lora_enabled(args)
        and role == "actor"
        and args.megatron_to_hf_mode != "bridge"
        and getattr(args, "lora_adapter_path", None)
        and "inkling" in (getattr(args, "custom_model_provider_path", None) or "")
    ):
        if (Path(args.lora_adapter_path) / "adapter_model.safetensors").exists() and not native_optimizer_restored:
            from miles_plugins.models.inkling.lora import load_inkling_lora_adapter

            load_inkling_lora_adapter(model, args.lora_adapter_path)
            if optimizer is not None:
                # refresh the fp32 masters, or the first step() restores the
                # pre-load init values over the adapter we just wrote
                optimizer.reload_model_params()

    check_peak_gpu_memory_after_load(args)
    clear_memory()

    check_model_hashes(args, model, iteration)

    # Megatron checkpoint loads can restore scheduler state directly. In that
    # case, stepping by the checkpoint iteration here would double-count.
    if opt_param_scheduler is not None and not (args.use_checkpoint_opt_param_scheduler and iteration > 0):
        opt_param_scheduler.step(increment=iteration * args.global_batch_size)

    if args.finetune and not is_lora_enabled(args):
        assert iteration == 0, (
            f"--finetune loaded {args.load} and found iteration {iteration}, so the checkpoint and the flag disagree "
            f"about where this run stands"
        )
        start_rollout_id = 0
    else:
        start_rollout_id = iteration + 1

    return LoadCheckpointOutput(loaded_rollout_id=iteration, start_rollout_id=start_rollout_id)
