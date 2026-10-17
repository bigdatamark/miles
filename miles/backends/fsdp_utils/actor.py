# [MODULE DOC] The Ray training actor of the FSDP backend and the function that shards a model
# with FSDP2. The actor builds a stock HuggingFace model (rank 0 loads real weights, other ranks
# start on the meta device unless embeddings are tied), applies the architecture adaptation hooks
# (class patches, packing, instance patches, post-load fixups, precision policy), shards it with
# FSDP2 over the "fsdp" device mesh, loads the full state dict by broadcast from rank 0, creates
# the optimizer, scheduler, optional reference model and weight updater, restores checkpoints and
# optionally offloads to CPU. Training steps themselves (loop, losses, step runner usage) live in
# the torch native base actor; this class supplies the model-specific forward, gradient zeroing,
# optimizer step and reference-model handling hooks.
import logging
import os
from contextlib import contextmanager, nullcontext
from functools import partial

import torch
import torch.distributed as dist

from miles.backends.fsdp_utils import checkpoint
from miles.backends.fsdp_utils.adaptations import routing_replay
from miles.backends.training_utils.metrics import train_dump
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native.step_runner import LinearStepRunner, StepMetrics
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.profile_utils import TrainProfiler
from miles.utils.replay_base import routing_replay_manager
from miles.utils.timer import Timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.workers.rpc.common.wire_types import Pickled

from .adaptations.class_patches import apply_class_patches, apply_model_instance_patches
from .adaptations.packing import apply_packing
from .adaptations.post_load_fixups import apply_post_load_fixups
from .adaptations.precision import apply_fp32_master, precision_forward_context, resolve_precision_policy
from .hf_weight_iterator import FSDPHfWeightIterator
from .lr_scheduler import get_lr_scheduler
from .parallel import create_fsdp_parallel_state
from .plugins.hf_kernels import HubKernels

logger = logging.getLogger(__name__)


class FSDPTrainRayActor(TorchNativeTrainRayActor):
    # [CLASS DOC] FSDP2 implementation of the torch native training actor, identified by the
    # backend name fsdp. Per-instance state set during init: model (FSDP2 sharded), optimizer,
    # learning rate scheduler, global and micro step counters, precision policy, hub kernels
    # handle, optional reference model with its runner, the CPU offload flag, parts lists for
    # models and optimizers, the weight updater and the max tokens per GPU. Unsupported here:
    # receiving a checkpoint from another rank, independent data-parallel quorums and their store.
    backend_name = "fsdp"

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Pickled,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
    ) -> int | None:  # type: ignore[override]
        # [FUNCTION DOC] Initialize the actor on one rank; a deferred hook starts the train-wait
        # timer when init finishes. Inputs: serialized run args, the role name, whether to build a
        # reference model, whether an on-policy distillation teacher is used (only forwarded to
        # the common init), the checkpoint-receive source rank and independent data-parallel info
        # and store address. Asserts that the unsupported options are unset (no receive source,
        # quorum id 0, no store address). Returns the rollout id to start from (an integer), or 0
        # immediately after the common setup, parallel state and seeding when the run is
        # rollout-only (no model is built then).
        # Steps in order: common base init; optional sglang dumper source patching; create and
        # set the FSDP parallel state; seed torch from args; (rollout-only exit); read the CPU
        # offload flag (default false) and, if both train offload and FSDP CPU offload are on,
        # turn train offload off in the shared args (CPU offload wins as more aggressive); rank 0
        # starts tracking in non-primary mode; default the start rollout id to 0 in args when
        # missing; create the profiler; load HF config and processor assets; resolve the precision
        # policy; enable routing replay; apply class patches and config-lifetime packing before
        # construction; prepare hub kernels (a collective, inert unless the hub backend is chosen);
        # enable true on policy batch-invariant ops; build the model on the right init device
        # (with the Triton attention bridge on ROCm) and log patched layers; apply instance patches,
        # bind hub kernels, install routing replay hooks, wrap in an fp32 master if the policy asks;
        # re-apply checkpoint weights over anything clobbered at load; apply post-load packing;
        # switch to train mode; copy out the full state dict; shard with FSDP2 using the policy
        # dtypes; load the full state dict by broadcast; store the model; enable gradient
        # checkpointing if configured; create AdamW (only the adam option is supported, else a
        # value error) from lr, betas, eps and weight decay; create the scheduler; zero the step
        # counters; load the training checkpoint payload; build the reference model and its
        # no-grad runner when requested; set the parts lists; build the weight updater with the
        # FSDP HF weight iterator; finalize checkpoint loading with the payload; store max tokens
        # per GPU; if train offload is on, put the actor to sleep; mark profiler init end.
        # Side effects: mutates args (offload flag, start rollout id), global parallel state,
        # collective communication, GPU memory.
        super()._init_common(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        # Unsupported
        assert recv_ckpt_src_rank is None
        assert indep_dp_info.quorum_id == 0
        assert indep_dp_store_addr is None

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        # Setup ParallelState for both CP and non-CP cases
        set_parallel_state(create_fsdp_parallel_state(args))

        torch.manual_seed(args.seed)

        if self.args.debug_rollout_only:
            return 0

        self.fsdp_cpu_offload = getattr(self.args, "fsdp_cpu_offload", False)
        # Offload train and fsdp cpu offload cannot be used together, fsdp_cpu_offload is more aggressive
        if self.args.offload_train and self.fsdp_cpu_offload:
            self.args.offload_train = False

        if dist.get_rank() == 0:
            init_tracking(args, primary=False)

        if getattr(self.args, "start_rollout_id", None) is None:
            self.args.start_rollout_id = 0

        self.prof = TrainProfiler(args)

        self.load_hf_assets(with_processor=True)

        self.precision_policy = resolve_precision_policy(self.hf_config, self.args)

        routing_replay.enable(args)

        # FSDP trains stock HF modeling: HF-compat patches + config-lifetime packing, before construction.
        apply_class_patches(self.hf_config, self.args)
        apply_packing(None, self.hf_config, "config")

        # Collective across all ranks; inert unless --kernel-backend hub.
        self.hub_kernels = HubKernels.prepare(self.args)

        # backend-level true-on-policy setup (batch-invariant ops)
        self._enable_true_on_policy_optimizations(args)

        init_context = self._get_init_weight_context_manager()

        model, n = self._build_model_with_attn_bridge(self.args.hf_checkpoint, init_context)
        if n > 0:
            logger.info(f"FSDPTrainRayActor applied triton attention patch to {n} layer(s)")

        apply_model_instance_patches(model, self.hf_config, self.args)
        self.hub_kernels.bind(model)
        routing_replay.install(model, self.hf_config)
        if self.precision_policy.keep_fp32_master:
            model = apply_fp32_master(model, self.precision_policy.sync_dtype_resolver)

        # re-assert the checkpoint over any param from_pretrained clobbered post-load (arch-gated, else no-op)
        apply_post_load_fixups(model, self.hf_config, self.args.hf_checkpoint)

        # post-load packing patches that need the instantiated model (NemotronH); no-op for archs that don't
        apply_packing(model, self.hf_config, "post_load")

        model.train()

        full_state = model.state_dict()

        model = apply_fsdp2(
            model,
            mesh=get_parallel_state().get_mesh("fsdp"),
            cpu_offload=self.fsdp_cpu_offload,
            args=self.args,
            param_dtype=self.precision_policy.param_dtype,
            reduce_dtype=self.precision_policy.reduce_dtype,
        )

        model = self._fsdp2_load_full_state_dict(
            model,
            full_state,
            get_parallel_state().get_mesh("fsdp"),
            cpu_offload=True if self.fsdp_cpu_offload else None,
        )

        self.model = model

        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if args.optimizer == "adam":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported options: 'adam'")

        # Initialize LR scheduler
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        self.global_step = 0
        self.micro_step = 0

        checkpoint_payload = checkpoint.load(self)

        # Create separate ref model if needed (kept in CPU until needed)
        self.ref_model = None
        if with_ref:
            self.ref_model = self._create_ref_model(args.ref_load)
            self.ref_runner = LinearStepRunner(partial(self._logprob_forward, self.ref_model))
        self.model_parts = [self.model]
        self.optimizers = [self.optimizer]

        self.weight_updater = self._build_weight_updater(self.model, FSDPHfWeightIterator.build)

        checkpoint.finalize_load(self, checkpoint_payload)

        self.max_tokens_per_gpu = args.max_tokens_per_gpu

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return int(getattr(self.args, "start_rollout_id", 0))

    def _has_image_text_to_text_impl(self) -> bool:
        # [FUNCTION DOC] HELPER: Whether the HF config describes a vision-language model that
        # should be built with the image-text-to-text auto class: true only if the config has a
        # vision sub-config and either has no custom auto-class mapping or that mapping lists the
        # image-text-to-text class.
        if not hasattr(self.hf_config, "vision_config"):
            return False
        auto_map = getattr(self.hf_config, "auto_map", None)
        return not auto_map or "AutoModelForImageTextToText" in auto_map

    def _get_model_cls(self):
        # [FUNCTION DOC] HELPER: Pick the HF class used to build the model. Vision-language configs
        # use the image-text-to-text auto class. Otherwise look up the config's model type in the
        # transformers table of native causal LM class names and return that concrete class from
        # the transformers module (avoiding the auto registries, which can be re-registered at
        # runtime by sglang's vendored configs and would silently mis-shape some models); if the type
        # is not natively known, return the causal LM auto class.
        if self._has_image_text_to_text_impl():
            from transformers import AutoModelForImageTextToText

            return AutoModelForImageTextToText
        else:
            import transformers
            from transformers import AutoModelForCausalLM
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

            # Resolve natively-supported archs by model_type string: AutoConfig/AutoModel registries can
            # be re-registered at runtime (sglang vendors a nemotron_h config whose hybrid_override_pattern
            # parsing mis-places the attention layers), which would silently train a mis-shaped model.
            native_cls_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(getattr(self.hf_config, "model_type", ""))
            if native_cls_name is not None:
                return getattr(transformers, native_cls_name)
            return AutoModelForCausalLM

    def _build_model_with_attn_bridge(self, checkpoint_path: str, init_context):
        # [FUNCTION DOC] HELPER: Instantiate the HF model from a checkpoint folder. The init
        # context argument is a factory that is called to get a context manager governing where
        # parameters are allocated. The Triton attention bridge is used only when the attention
        # implementation arg is triton AND the build is for ROCm; then the model is loaded with
        # eager attention and afterwards patched by the bridge; elsewhere the arg is passed to HF
        # unchanged (which rejects triton). Remote code is trusted. Returns a pair of the model and
        # the number of attention layers patched by the bridge (0 when not used).
        """Build HF model and optionally apply Triton attention bridge patch."""
        # ROCm-only: on other platforms "triton" falls through to from_pretrained, which rejects
        # it exactly as it did before this path existed.
        use_triton_bridge = self.args.attn_implementation == "triton" and torch.version.hip is not None
        effective_attn = "eager" if use_triton_bridge else self.args.attn_implementation

        with init_context():
            model = self._get_model_cls().from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                attn_implementation=effective_attn,
            )

        patched_layers = 0
        if use_triton_bridge:
            from .sglang_attn_bridge.hf_sglang_triton_patch import apply_sglang_triton_attention_patch

            patched_layers = apply_sglang_triton_attention_patch(model)
        return model, patched_layers

    def _enable_true_on_policy_optimizations(self, args):
        # [FUNCTION DOC] HELPER: When the true on policy mode arg is set, import sglang's batch
        # invariant ops and enable batch invariant mode with the batched matrix multiply variant
        # turned off (so the rotary embedding computation stays aligned with inference); logs an
        # info line. Does nothing otherwise. Global side effect: changes kernel dispatch process-wide.
        """Backend-level true-on-policy setup (batch-invariant ops), gated on the run mode."""
        if args.true_on_policy_mode:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            logger.info("FSDPTrainRayActor call enable_batch_invariant_mode for true-on-policy")
            enable_batch_invariant_mode(
                # In Qwen3, rope `inv_freq_expanded.float() @ position_ids_expanded.float()` uses bmm
                # and disabling it will make it aligned
                enable_bmm=False,
            )

    def _get_init_weight_context_manager(self):
        # [FUNCTION DOC] HELPER: Return a zero-argument factory (not a context manager itself) that
        # yields the context manager for model construction. If embeddings are not tied: rank 0
        # gets a factory for a CPU device context (real weights), all other ranks get the
        # empty-weights (meta device) factory from accelerate. If embeddings are tied: every rank
        # gets the CPU factory and an info line is logged, because meta tensors hang in that case.
        """Context manager for model init: meta device (no allocation) on non-rank-0, EXCEPT when
        tie_word_embeddings=True (meta tensors hang there) -- then full CPU load on all ranks.

        Ref: verl/utils/fsdp_utils.py::get_init_weight_context_manager
        """
        from accelerate import init_empty_weights

        use_meta_tensor = not self.hf_config.tie_word_embeddings

        def cpu_init_weights():
            # [FUNCTION DOC] HELPER: Factory returning a CPU device context for allocating real
            # weights on host memory.
            return torch.device("cpu")

        if use_meta_tensor:
            # Rank 0: CPU, others: meta device (memory efficient for large models)
            return init_empty_weights if dist.get_rank() != 0 else cpu_init_weights
        else:
            logger.info(f"[Rank {dist.get_rank()}] tie_word_embeddings=True, loading full model to CPU on all ranks")
            return cpu_init_weights

    def _fsdp2_load_full_state_dict(self, model, full_state, device_mesh, cpu_offload):
        # [FUNCTION DOC] HELPER: Fill a freshly sharded FSDP2 model from a full (unsharded) state
        # dict, returning the model. Rank 0 moves its real model to the current GPU; other ranks
        # allocate uninitialized storage on the GPU. Then loads with the distributed checkpoint API
        # in full-state-dict mode with broadcast from rank 0 (CPU offload option enabled when the
        # cpu_offload argument is not none; the callers pass either true or none). Buffers are not
        # covered by that API, so every buffer is broadcast from rank 0. With CPU offload the model
        # is then moved to host memory while buffers are moved back to the GPU. The device mesh
        # argument is unused.
        """Load the full state dict into the FSDP2 model, broadcasting rank-0 weights to all ranks
        (so only rank 0 reads from disk).

        Ref: verl/utils/fsdp_utils.py::fsdp2_load_full_state_dict
        """
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

        # Rank 0: move with weights, others: allocate empty tensors on device
        if dist.get_rank() == 0:
            model = model.to(device=torch.cuda.current_device(), non_blocking=True)
        else:
            # to_empty creates tensors on device without initializing memory
            model = model.to_empty(device=torch.cuda.current_device())

        is_cpu_offload = cpu_offload is not None
        options = StateDictOptions(full_state_dict=True, cpu_offload=is_cpu_offload, broadcast_from_rank0=True)

        set_model_state_dict(model, full_state, options=options)

        # set_model_state_dict will not broadcast buffers, so we need to broadcast them manually.
        for _name, buf in model.named_buffers():
            dist.broadcast(buf, src=0)

        if is_cpu_offload:
            model.to("cpu", non_blocking=True)
            for buf in model.buffers():
                buf.data = buf.data.to(torch.cuda.current_device())

        return model

    def _save_checkpoint(self, rollout_id: int) -> None:
        # [FUNCTION DOC] Hook called by the base actor to save training state after the given
        # rollout; delegates to the checkpoint module's save with this actor.
        checkpoint.save(self, rollout_id)

    @contextmanager
    def _ref_context(self):
        # [FUNCTION DOC] HELPER: Context manager bracketing computations with the reference model.
        # With no reference model it just yields. Otherwise, unless FSDP CPU offload is on, it first
        # moves the actor model to host memory, frees cached GPU memory and synchronizes ranks on the
        # CPU-side group; puts the reference model in eval mode; yields; and in a finally block frees
        # cache and synchronizes again, and (unless CPU offload) moves the actor back to the GPU and
        # synchronizes once more.
        if self.ref_model is None:
            yield
            return

        if not self.fsdp_cpu_offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            dist.barrier(group=get_gloo_group())
        self.ref_model.eval()
        try:
            yield
        finally:
            torch.cuda.empty_cache()
            dist.barrier(group=get_gloo_group())
            if not self.fsdp_cpu_offload:
                self.model.cuda()
                dist.barrier(group=get_gloo_group())

    def _logprob_forward(self, model: torch.nn.Module, batch: dict) -> torch.Tensor:
        # [FUNCTION DOC] Forward pass used to compute log probabilities for the given model
        # (actor or reference): builds model inputs from the batch, runs the model inside the
        # precision forward context and returns the raw logits in the model's native dtype (no
        # upcast). The function itself does not disable gradient tracking; the caller is expected to.
        """No-grad forward. Logits stay in native bf16; the loss path upcasts
        per-response chunks, which avoids a full-vocab fp32 tensor."""
        model_args = self._get_model_inputs_args(batch)
        with precision_forward_context(self.precision_policy):
            return model(**model_args).logits

    def _step_runner(self) -> LinearStepRunner:
        # [FUNCTION DOC] Create a new linear step runner wired to this actor's forward, gradient
        # zeroing and optimizer step methods (a new instance on every call).
        return LinearStepRunner(self._forward, self._zero_grad, self._apply_step)

    def _after_rollout(self, rollout_id: int, rollout_data) -> None:
        # [FUNCTION DOC] Hook run after each rollout's training. If a debug train data path is
        # configured, dumps the rollout data for that rollout id. If a reference update interval is
        # configured, a reference model exists and (rollout id plus one) is a multiple of the
        # interval, copies the actor's current state dict into the reference model (rank 0 logs it)
        # and moves the reference model back to host memory.
        if self.args.save_debug_train_data is not None:
            train_dump.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and self.ref_model is not None
        ):
            if dist.get_rank() == 0:
                logger.info(f"Updating ref model at rollout_id {rollout_id}")
            actor_state = self.model.state_dict()
            self.ref_model.load_state_dict(actor_state)
            self.ref_model.cpu()

    def _forward(self, batch: dict) -> torch.Tensor:
        # [FUNCTION DOC] Training forward pass for one batch: builds model inputs, and if routing
        # replay is currently in its backward stage, enters the replay forward stage for the
        # duration (so activation-checkpoint recomputation does not consume the backward cursor);
        # runs the actor model under the precision forward context and returns the logits.
        """The training pass brackets each real forward in ``replay_forward`` so
        activation-checkpoint recompute keeps the backward cursor to itself."""
        model_args = self._get_model_inputs_args(batch)
        replaying = routing_replay_manager.stage == routing_replay.REPLAY_BACKWARD
        replay_stage = routing_replay.stage(routing_replay.REPLAY_FORWARD) if replaying else nullcontext()
        with replay_stage, precision_forward_context(self.precision_policy):
            return self.model(**model_args).logits

    def _zero_grad(self) -> None:
        # [FUNCTION DOC] Clear gradients by setting them to none (not zero tensors).
        self.optimizer.zero_grad(set_to_none=True)

    def _apply_step(self) -> StepMetrics:
        # [FUNCTION DOC] Optimizer step: clip gradients by global norm to the clip arg (the norm is
        # a sharded tensor and is gathered to a full tensor), take the optimizer step, advance the
        # scheduler, and return step metrics holding the pre-clip gradient norm as a float and a
        # mapping from a per-parameter-group learning rate name to its value. No handling of a
        # non-finite norm: the step is taken anyway.
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad).full_tensor()
        self.optimizer.step()
        self.lr_scheduler.step()
        return StepMetrics(
            grad_norm=float(grad_norm.item()),
            extra_metrics={f"lr-pg_{i}": group["lr"] for i, group in enumerate(self.optimizer.param_groups)},
        )

    def _create_ref_model(self, ref_load_path: str | None):
        # [FUNCTION DOC] HELPER: Build the reference model from an HF checkpoint folder and return
        # it, sharded with FSDP2 with CPU offload always on. Raises a value error for a missing
        # path and a not-implemented error for a path that is not a directory. Steps: log; build
        # the model with the same init context logic (and Triton bridge when applicable, logging
        # the patched layer count); apply instance patches; bind hub kernels; wrap in an fp32
        # master only if the policy keeps one AND its parameter dtype is float32; copy the full
        # state dict; shard; load the state dict by broadcast with CPU offload.
        # Note: unlike the actor model, no routing replay install, post-load fixups or post-load
        # packing patches are applied to the reference model.
        """Create a separate FSDP2 ref model. ALWAYS uses CPUOffloadPolicy (regardless of the actor's
        offload setting) to save memory. Raises if ``ref_load_path`` is None or not a directory."""
        if ref_load_path is None:
            raise ValueError("ref_load_path must be provided when loading reference model")

        if os.path.isdir(ref_load_path):
            logger.info(f"[Rank {dist.get_rank()}] Creating separate ref model from {ref_load_path}")

            init_context = self._get_init_weight_context_manager()

            ref_model, ref_patch_n = self._build_model_with_attn_bridge(ref_load_path, init_context)
            if ref_patch_n > 0:
                logger.info(
                    f"[Rank {dist.get_rank()}] Applied triton attention patch to ref model ({ref_patch_n} layer(s))"
                )

            apply_model_instance_patches(ref_model, self.hf_config, self.args)
            self.hub_kernels.bind(ref_model)
            if self.precision_policy.keep_fp32_master and self.precision_policy.param_dtype is torch.float32:
                ref_model = apply_fp32_master(ref_model, self.precision_policy.sync_dtype_resolver)
            full_state = ref_model.state_dict()

            # Always use CPUOffloadPolicy for reference, let FSDP2 handle the offload. It is faster than model.cpu().
            ref_model = apply_fsdp2(
                ref_model,
                mesh=get_parallel_state().get_mesh("fsdp"),
                cpu_offload=True,
                args=self.args,
                param_dtype=self.precision_policy.param_dtype,
                reduce_dtype=self.precision_policy.reduce_dtype,
            )
            ref_model = self._fsdp2_load_full_state_dict(
                ref_model,
                full_state,
                get_parallel_state().get_mesh("fsdp"),
                cpu_offload=True,
            )

            logger.info(f"[Rank {dist.get_rank()}] Reference model created with FSDP2 CPUOffloadPolicy")
            return ref_model
        else:
            raise NotImplementedError(f"Loading from checkpoint file {ref_load_path} not yet implemented")

    def _get_model_inputs_args(self, batch: dict) -> dict:
        # [FUNCTION DOC] HELPER: Convert a batch into model keyword arguments: token ids and
        # position ids from the batch, no attention mask (packed layout relies on position ids), and
        # any multimodal training inputs merged in when present and non-empty.
        model_args = {
            "input_ids": batch["tokens"],
            "position_ids": batch["position_ids"],
            "attention_mask": None,
        }

        if batch.get("multimodal_train_inputs"):
            model_args.update(batch["multimodal_train_inputs"])

        return model_args


@torch.no_grad()
def apply_fsdp2(model, mesh=None, cpu_offload=False, args=None, param_dtype=None, reduce_dtype=None):
    # [FUNCTION DOC] Shard a model in place with FSDP2 and return it (runs with gradients
    # disabled). Optional CPU offload policy. The wrap units are all modules whose class name is in
    # the model's no-split module list, plus embedding modules when embeddings are not tied;
    # asserts the no-split list is non-empty. Mixed precision: param dtype defaults to float16 if
    # the args request fp16 else bfloat16 (so args is required in that case); reduce dtype defaults
    # to float32; logs the policy. Each unit is sharded first, then the root model, all with the
    # same mixed precision policy, offload policy and device mesh.
    """Apply FSDP2 (fully_shard) to the model.

    ``cpu_offload`` offloads params/grads/optimizer to CPU (the optimizer step runs on CPU).
    ``param_dtype``/``reduce_dtype`` are the MixedPrecisionPolicy dtypes; None falls back to the
    args-based default (bf16 / fp32, or fp16 param when args.fp16).

    Ref: https://github.com/volcengine/verl/blob/main/verl/utils/fsdp_utils.py
    """
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    offload_policy = CPUOffloadPolicy() if cpu_offload else None

    layer_cls_to_wrap = model._no_split_modules
    assert len(layer_cls_to_wrap) > 0 and next(iter(layer_cls_to_wrap)) is not None

    modules = [
        module
        for name, module in model.named_modules()
        if module.__class__.__name__ in layer_cls_to_wrap
        or (isinstance(module, torch.nn.Embedding) and not model.config.tie_word_embeddings)
    ]

    if param_dtype is None:
        param_dtype = torch.float16 if args.fp16 else torch.bfloat16
    if reduce_dtype is None:
        reduce_dtype = torch.float32

    logger.info(f"FSDP MixedPrecision Policy: param_dtype={param_dtype}, reduce_dtype={reduce_dtype}")

    fsdp_kwargs = {
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
        ),
        "offload_policy": offload_policy,
        "mesh": mesh,
    }

    # fully_shard each layer first, then the root model
    for module in modules:
        fully_shard(module, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    return model
