# [MODULE DOC]
# Ray actor that runs Megatron training for one rank. It owns the Megatron model chunks,
# optimizer and optimizer-parameter scheduler, optional host-side weight backups (for switching
# among actor / reference / teacher / old-actor / rollout-actor weights), the weight updater that
# pushes weights to rollout engines, the snapshot publisher for HF export, and offload (sleep /
# wake-up) of GPU memory through the memory saver. Public entry points called by the controller:
# init, load_state, sleep, wake_up, offload_grad_buffer, train, save_model, export_hf,
# update_weights, load_other_checkpoint, send_ckpt, reconfigure_indep_dp. Also defines a value spec
# for shipping critic values by object reference and two small module helpers.
import atexit
import logging
import os
import shutil
from contextlib import ExitStack, nullcontext

import torch
import torch.distributed as dist
from megatron.training.async_utils import maybe_finalize_async_save
from torch_memory_saver import torch_memory_saver

from miles.backends.megatron_utils.hf_export import save_hf_model
from miles.backends.megatron_utils.lora.utils import is_lora_enabled, lora_rollout_enabled
from miles.backends.megatron_utils.rematerialize_utils import build_main_cast_context
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import get_hf_weight_iterator
from miles.backends.training_utils.checkpoint.tracker import read_checkpoint_tracker_iteration
from miles.backends.training_utils.data.rollout import (
    DataIterator,
    get_data_iterator,
    get_num_rollouts,
    get_rollout_data,
)
from miles.backends.training_utils.loss.objective import (
    compute_advantages_and_returns,
    get_log_probs_and_entropy,
    get_values,
    log_train_advantage_computation_event,
)
from miles.backends.training_utils.metrics import train_dump
from miles.backends.training_utils.metrics.log_utils import log_cpu_memory, log_perf_data, log_rollout_data
from miles.backends.training_utils.replay.data import fill_replay_data, register_replay_list_sequential
from miles.backends.training_utils.types import TrainStepOutput
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.specs.train import compute_trainer_pool_id
from miles.ray.train_actor import TrainRayActor
from miles.utils import object_store
from miles.utils.argparse_utils import inplace_modify_args
from miles.utils.audit_utils.event_logger.logger import event_logger_context
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.lora.utils import build_lora_config, is_multi_lora_enabled
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.object_store import StoreObjectRef, ValueSpec
from miles.utils.reloadable_process_group import destroy_process_groups, monkey_patch_torch_dist, reload_process_groups
from miles.utils.replay_base import all_replay_managers, routing_replay_manager
from miles.utils.test_utils.ft_test_actions import FTTestActionActorExecutor
from miles.utils.timer import Timer, inverse_timer, timer
from miles.utils.tracking_utils.structured_log import with_logs
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.types import RolloutBatch
from miles.utils.workers.naming import compute_cell_id
from miles.utils.workers.rpc.common.wire_types import Pickled

from ...utils.profile_utils import TrainProfiler
from ...utils.tensor_backper import TensorBackuper
from ..training_utils.parallel import get_parallel_state
from .checkpoint import load_checkpoint
from .ft.checkpoint_transfer import recv_ckpt
from .ft.checkpoint_transfer import send_ckpt as _send_ckpt
from .ft.in_memory_checkpoint import InMemoryCheckpointManager
from .ft.indep_dp import reconfigure_indep_dp_group
from .initialize import RandomState, init, is_first_replica_megatron_main_rank
from .model import (
    LoadCheckpointOutput,
    TrainStepOutcome,
    build_model_and_optimizer,
    forward_only,
    load_model_state,
    save,
    train,
)
from .named_weights import named_params_and_buffers
from .optimizer_utils import reset_optimizer_state
from .parallel import verify_megatron_parallel_state
from .replay_utils import register_replay_list_moe

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

CRITIC_VALUES_VALUE_SPEC: dict[str, ValueSpec] = {"values": ValueSpec(codec="typed_ragged")}


def _setup_disk_offload_reclaim(disk_dir: str) -> None:
    # [FUNCTION DOC] HELPER: Does nothing when the directory path is empty or none. Otherwise
    # removes the directory tree (ignoring errors), recreates it empty, registers an exit handler
    # that removes it again at interpreter exit (ignoring errors), and logs. Side effects: file
    # system deletion and creation, exit hook registration.
    # KNOWN ISSUE: the caller passes the environment lookup of the disk backup directory variable,
    # which can be none; that case is silently skipped here even though disk offload was requested.
    """Wipe this rank's train disk-offload dir on startup and re-arm the atexit wipe.

    torch_memory_saver unlinks each backup file as its allocation is freed on a
    graceful teardown, but a SIGKILL'd run leaves stale files behind. The dir is
    per-rank (see actor_factory), so clearing it wholesale touches nobody else.
    """
    if not disk_dir:
        return
    shutil.rmtree(disk_dir, ignore_errors=True)
    os.makedirs(disk_dir, exist_ok=True)
    atexit.register(shutil.rmtree, disk_dir, ignore_errors=True)
    logger.info(f"Train disk-offload reclaim armed for {disk_dir} (startup wipe + atexit)")


class MegatronTrainRayActor(TrainRayActor):
    # [CLASS DOC]
    # Per-rank Megatron trainer. Key state: model (list of model chunks), optimizer, optimizer
    # parameter scheduler, a weights backuper (host copies keyed by tag: "actor", and optionally
    # "ref", "teacher", "old_actor", "rollout_actor"), the currently active model tag, an asleep
    # flag (GPU memory offloaded) and a grad-buffer-paused flag, the weight updater and snapshot
    # publisher (either may be none), the last rollout id trained, a captured random state taken
    # right after initialization (used to restore on hot restart), and an optional user hook that
    # post-processes rollout data. Replay managers (routing and other) are driven through stages:
    # "fallthrough" for ref/teacher passes, "record" or "replay_forward" for the actor log-prob
    # pass, "replay_backward" for training.
    @with_logs
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
    ) -> int | None:
        # [FUNCTION DOC]
        # Initializes the actor and returns the rollout id to start from (the loaded checkpoint's
        # start id), or zero in rollout-only debug mode. The decorators log the call and start the
        # "train_wait" timer when init returns/exits. Steps in order: clear updater/publisher;
        # patch the torch distributed module for reloadable process groups; run the base-class
        # common init (args, role, reference and distillation-teacher flags); register replay-list
        # functions (sequential for generic replay managers, MoE-specific for routing replay); run
        # the Megatron init with the independent-DP store address and info; build the fault-tolerance
        # test-action executor from the trainer pool and cell ids; apply dumper source patches if
        # enabled; on the first replica's main rank start tracking and register with the dashboard;
        # reject profile targets unsupported on Megatron (raises not-implemented only when the
        # PyTorch profiler is also on); create the profiler; load HF assets; record the parallel
        # config (empty with independent DP, else dp, cp, vpp sizes and microbatch group size);
        # barrier on the gloo group; when train offload is on, set the memory-saver margin from
        # args if positive and, for the disk target, arm disk reclaim on the backup directory
        # environment variable. Return zero now if rollout-only debug. For non-critic roles enable
        # each replay manager per its use-replay flag (and result checking in CI). Build the
        # in-memory/received checkpointing context (received from a source rank when healing, which
        # also overrides loading to take optimizer/rng and not finetune; or an in-memory local
        # manager when the non-persistent type is local), build model/optimizer/scheduler, capture
        # random state, set up hybrid context-parallel attention when cp size exceeds one, verify
        # the parallel state, mark awake. Critic role: load state, sleep if offloading, return start
        # id. Actor role: optionally build a main-weight cast context (rematerialize), create the
        # weights backuper over the actor weight getter, set active tag "actor", default vocab size
        # from the tokenizer, load state, create updater/publisher, load the optional rollout-data
        # post-process function, sleep if offloading, notify profiler, and return the start id.
        self.weight_updater: WeightUpdater | None = None
        self.snapshot_publisher: SnapshotPublisher | None = None
        monkey_patch_torch_dist()

        self._last_rollout_id: int | None = None
        super()._init_common(args, role, with_ref, with_opd_teacher=with_opd_teacher)

        for m in all_replay_managers:
            m.register_replay_list_func = register_replay_list_sequential
        routing_replay_manager.register_replay_list_func = register_replay_list_moe

        init(
            args,
            indep_dp_store_addr=indep_dp_store_addr,
            indep_dp_info=indep_dp_info,
        )

        trainer_pool_id = compute_trainer_pool_id(args.trainer_id)
        self._ft_test_action_executor = FTTestActionActorExecutor.from_args(
            args,
            cell_id=compute_cell_id(pool_id=trainer_pool_id, cell_index=indep_dp_info.cell_index),
            rank=self._rank,
        )

        if args.dumper_enable:
            from sglang.srt.debug_utils.dumper import dumper

            dumper.apply_source_patches()

        self._is_first_replica_megatron_main_rank = is_first_replica_megatron_main_rank()

        if self._is_first_replica_megatron_main_rank:
            init_tracking(args, primary=False)

        dashboard_hooks.register_train_actor(args)

        unsupported = {"train_actor", "train_log_probs"} & set(args.profile_target)
        if unsupported and args.use_pytorch_profiler:
            raise NotImplementedError(
                f"--profile-target {' '.join(sorted(unsupported))} is not supported for Megatron backend"
            )
        self.prof = TrainProfiler(args)

        self.load_hf_assets()

        self.train_parallel_config = (
            {}
            if args.indep_dp
            else {
                "dp_size": get_parallel_state().intra_dp.size,
                "cp_size": get_parallel_state().cp.size,
                "vpp_size": get_parallel_state().vpp_size,
                "microbatch_group_size_per_vp_stage": get_parallel_state().microbatch_group_size_per_vp_stage,
            }
        )
        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                # --train-memory-margin-bytes can tune this
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x
            if args.offload_train_target == "disk":
                _setup_disk_offload_reclaim(os.environ.get("TMS_DISK_BACKUP_DIR"))

        if self.args.debug_rollout_only:
            return 0

        if role != "critic":
            for m in all_replay_managers:
                m.enabled = getattr(self.args, f"use_{m.name}_replay", False)
                m.enable_check_replay_result = m.enabled and self.args.ci_test

        checkpointing_context = None
        if recv_ckpt_src_rank is not None:
            ckpt_manager = recv_ckpt(
                indep_dp=get_parallel_state().indep_dp,
                src_rank=recv_ckpt_src_rank,
            )
            checkpointing_context = {"local_checkpoint_manager": ckpt_manager}
        elif args.non_persistent_ckpt_type == "local":
            checkpointing_context = {"local_checkpoint_manager": InMemoryCheckpointManager()}

        heal_load_overrides: dict[str, object] = (
            dict(no_load_optim=False, no_load_rng=False, finetune=False) if recv_ckpt_src_rank is not None else {}
        )
        self.model, self.optimizer, self.opt_param_scheduler = build_model_and_optimizer(args, role=role)
        self._post_init_random_state = RandomState.capture()

        parallel_state = get_parallel_state()
        if parallel_state.cp.size > 1:
            from miles_plugins.models.hf_attention import detect_and_setup_hybrid_cp

            for model_chunk in self.model:
                detect_and_setup_hybrid_cp(
                    model_chunk, parallel_state.cp.group, parallel_state.cp.rank, parallel_state.cp.size
                )

        verify_megatron_parallel_state(self.model)

        self._asleep = False
        self._grad_buffer_paused = False

        if role == "critic":
            load_output = self._load_state_core(
                checkpointing_context=checkpointing_context, overrider_for_loading=heal_load_overrides
            )
            if self.args.offload_train:
                self.sleep()
            return load_output.start_rollout_id

        main_cast_ctx = None
        if args.rematerialize_param_from_master_weight:
            main_cast_ctx = build_main_cast_context(args, model=self.model, optimizer=self.optimizer)

        self.weights_backuper = TensorBackuper.create(
            source_getter=self._named_actor_weights,
            main_cast_ctx=main_cast_ctx,
        )
        self._active_model_tag: str | None = "actor"

        if self.args.vocab_size is None:
            self.args.vocab_size = self.tokenizer.vocab_size

        load_output = self._load_state_core(
            checkpointing_context=checkpointing_context, overrider_for_loading=heal_load_overrides
        )

        self._init_training_state()

        self.rollout_data_postprocess = None
        if (x := self.args.rollout_data_postprocess_path) is not None:
            from miles.utils.function_registry import load_function

            self.rollout_data_postprocess = load_function(x)

        if self.args.offload_train:
            self.sleep()

        self.prof.on_init_end()

        return load_output.start_rollout_id

    def _init_training_state(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the weight updater and snapshot publisher. Weight updates
        # are enabled unless debug-train-only is set. Snapshots are published when HF saving is
        # configured, or eval uses snapshots and an eval HF directory is set, or LoRA is enabled
        # with a save path and the conversion mode is not raw.
        args = self.args
        self._init_weight_updater_and_publisher(
            update_weights=not args.debug_train_only,
            publish_snapshots=(
                args.save_hf is not None
                or (args.eval_uses_snapshots and args.eval_hf_dir is not None)
                or (is_lora_enabled(args) and args.save is not None and args.megatron_to_hf_mode != "raw")
            ),
        )

    def _init_weight_updater_and_publisher(self, *, update_weights: bool, publish_snapshots: bool) -> None:
        # [FUNCTION DOC] HELPER: Resets updater and publisher to none, then optionally creates them.
        # Model name is args.model_name or else the lowercased class name of the HF config; the
        # quantization config comes from the HF config if present. If update_weights: determines
        # whether LoRA rollout is on; with LoRA on a non-colocated setup, asserts the conversion
        # mode is bridge (assertion error otherwise); creates the weight updater with the model,
        # the actor-weights getter, the HF weight iterator factory, parallel state, the LoRA flag and
        # (for LoRA) a LoRA config built from the adapter target modules. If publish_snapshots:
        # builds an HF weight iterator requiring pipeline-parallel gathering (quantization config
        # dropped for LoRA) and wraps it, plus the LoRA config when LoRA is enabled, in a snapshot
        # publisher.
        args = self.args
        model_name = type(self.hf_config).__name__.lower() if args.model_name is None else args.model_name
        quantization_config = getattr(self.hf_config, "quantization_config", None)
        self.weight_updater = None
        self.snapshot_publisher = None

        if update_weights:
            is_lora = lora_rollout_enabled(args)
            if is_lora and not args.colocate:
                assert args.megatron_to_hf_mode == "bridge", (
                    "LoRA weight sync over distributed engines requires "
                    f"--megatron-to-hf-mode bridge (got {args.megatron_to_hf_mode!r})."
                )
            self.weight_updater = WeightUpdater(
                args,
                self.model,
                weights_getter=self._get_actor_weights,
                model_name=model_name,
                quantization_config=quantization_config,
                iterator_factory=get_hf_weight_iterator,
                parallel_state=get_parallel_state(),
                is_lora=is_lora,
                lora_sync_config=(
                    build_lora_config(args, target_modules=args.lora_adapter_targets) if is_lora else None
                ),
            )

        if publish_snapshots:
            is_lora = is_lora_enabled(args)
            iterator = get_hf_weight_iterator(
                args,
                self.model,
                required_placement=WeightUpdatePlacement(gather_pp=True),
                model_name=model_name,
                quantization_config=None if is_lora else quantization_config,
            )
            self.snapshot_publisher = SnapshotPublisher(
                iterator, build_lora_config(args, target_modules=args.lora_adapter_targets) if is_lora else None
            )

    def _clear_quantized_weight_workspaces(self) -> None:
        # [FUNCTION DOC] HELPER: Only when the clear-on-offload option is set, the transformer
        # implementation is transformer_engine, and CUDA graphs are off (a captured graph bakes in
        # the workspace address): imports the engine's base module class lazily and empties the
        # FP8 workspace cache of every such module in every model chunk. Otherwise returns at once.
        if not (
            self.args.clear_quantized_weight_workspaces_on_offload
            and self.args.transformer_impl == "transformer_engine"
            # A captured CUDA graph replays with the workspace address baked in.
            and self.args.cuda_graph_impl == "none"
        ):
            return
        from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

        for model_chunk in self.model:
            for module in model_chunk.modules():
                if isinstance(module, TransformerEngineBaseModule):
                    module._fp8_workspaces.clear()

    @with_logs
    def load_state(self) -> int:
        # [FUNCTION DOC]
        # Hot restart: rolls the trainer back to the state of a checkpoint and returns the rollout
        # id to resume from. Preconditions are asserted (assertion errors): initialized; no
        # rollout-only debug, LoRA, multi-LoRA, colocate, rematerialization from master weights,
        # local non-persistent checkpoints, train offload, profiler, memory-history recording,
        # keep-old-actor, or reference-update interval; and a requested load path must exist.
        # Finishes any pending async save (blocking). If the load path has a checkpoint tracker
        # iteration, loads that checkpoint with overrides (load path, no step, no finetune, load
        # optimizer and rng). Otherwise asserts a finetune-from-scratch configuration (no fp16,
        # precision-aware optimizer, optimizer CPU offload or optimizer-state offload, not bridge
        # mode, finetune, no-load optimizer and rng, equal checkpoint and reference-checkpoint
        # steps), restores the random state captured after init, and resets optimizer state. The
        # scheduler step counter is reset to zero beforehand. Then performs the core load, clears the
        # last-rollout id, logs, and returns the start id.
        assert self.is_initialized()

        # reloading does not support things like these
        assert not self.args.debug_rollout_only
        assert not is_lora_enabled(self.args)
        assert not is_multi_lora_enabled(self.args)
        assert not self.args.colocate
        assert not self.args.rematerialize_param_from_master_weight
        assert self.args.non_persistent_ckpt_type != "local"
        assert not self.args.offload_train
        assert not self.args.use_pytorch_profiler
        assert not self.args.record_memory_history
        assert not self.args.keep_old_actor, (
            "--keep-old-actor holds a second copy of the actor this reload does not roll back, so the run would "
            "compare the reloaded actor against weights of a rollout it no longer stands at"
        )
        assert not (
            self.with_ref and self.args.ref_update_interval is not None
        ), "--ref-update-interval keeps the reference in memory only, and no checkpoint holds it"
        assert (requested_load := self.args.requested_load) is not None, "a hot restart needs --load"

        self._finalize_pending_async_save()

        resume_from_ckpt = read_checkpoint_tracker_iteration(requested_load) is not None
        if not resume_from_ckpt:
            assert not self.args.fp16
            assert not self.args.use_precision_aware_optimizer
            assert not self.args.optimizer_cpu_offload
            assert not self.args.offload_optimizer_states
            assert self.args.megatron_to_hf_mode != "bridge", "bridge mode unsupported"
            assert self.args.finetune
            assert self.args.no_load_optim
            assert self.args.no_load_rng
            assert self.args.ckpt_step == self.args.ref_ckpt_step

        if self.opt_param_scheduler is not None:
            self.opt_param_scheduler.num_steps = 0

        if resume_from_ckpt:
            overrider_for_loading: dict[str, object] = dict(
                load=requested_load, ckpt_step=None, finetune=False, no_load_optim=False, no_load_rng=False
            )
        else:
            logger.info(
                f"load_state found no checkpoint under --load {requested_load!r}; loading the state the run "
                f"started from"
            )
            overrider_for_loading = {}
            self._post_init_random_state.restore()
            if self.optimizer is not None:
                reset_optimizer_state(
                    self.optimizer,
                    stream_optimizer_state_to_disk=self.args.stream_optimizer_state_to_disk,
                    chunked_optimizer_state_offload=self.args.chunked_optimizer_state_offload,
                )

        load_output = self._load_state_core(
            checkpointing_context=None,
            overrider_for_loading=overrider_for_loading,
        )
        self._last_rollout_id = None

        logger.info(f"load_state rolled this trainer back to checkpoint iteration {load_output.loaded_rollout_id}")
        return load_output.start_rollout_id

    def _load_state_core(
        self, *, checkpointing_context: dict | None, overrider_for_loading: dict[str, object]
    ) -> LoadCheckpointOutput:
        # [FUNCTION DOC] HELPER: Loads model/optimizer/scheduler state through the model-state loader
        # while the given argument overrides are temporarily applied to the args object (restored
        # afterwards). For non-critic roles then loads the auxiliary checkpoints and switches to the
        # actor weights. Clears GPU memory caches and returns the loader's output (which carries
        # the start and loaded rollout ids).
        with inplace_modify_args(self.args, overrider_for_loading):
            load_output = load_model_state(
                self.args,
                model=self.model,
                optimizer=self.optimizer,
                opt_param_scheduler=self.opt_param_scheduler,
                role=self.role,
                checkpointing_context=checkpointing_context,
            )

        if self.role != "critic":
            self._load_auxiliary_checkpoints()
            self._switch_model("actor")

        # empty cache after initialization
        clear_memory()

        return load_output

    def _load_auxiliary_checkpoints(self) -> None:
        # [FUNCTION DOC] HELPER: Backs up the freshly loaded actor weights to host if weight backup
        # is enabled. Then, as configured, loads the reference model (from the ref load path), the
        # distillation teacher (from the teacher load path), and when keeping an old actor loads
        # the old actor from the main load path and, if the weight-update interval is one, also
        # backs up the current actor as the rollout actor. Each load overwrites the live model and
        # backs it up under its tag; the caller switches back to actor weights afterwards.
        if self._enable_weight_backup:
            self.weights_backuper.backup("actor")

        if self.with_ref:
            self.load_other_checkpoint("ref", self.args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if self.with_opd_teacher:
            self.load_other_checkpoint("teacher", self.args.opd_teacher_load)

        if self.args.keep_old_actor:
            # Load old_actor checkpoint
            self.load_other_checkpoint("old_actor", self.args.load)
            # Create rollout_actor as a copy of current actor
            if self.args.update_weights_interval == 1:
                self.weights_backuper.backup("rollout_actor")

    def _finalize_pending_async_save(self) -> None:
        # [FUNCTION DOC] HELPER: If asynchronous checkpoint saving is enabled, blocks until any
        # in-flight async save has been finalized; otherwise does nothing.
        if not self.args.async_save:
            return

        maybe_finalize_async_save(blocking=True)

    @with_logs
    @timer
    def sleep(self) -> None:
        # [FUNCTION DOC]
        # Offloads training GPU memory (requires train offload; asserts). No-op with a log if
        # already asleep. Steps: clear quantized weight workspaces; clear GPU and host memory; note
        # whether to log CPU memory (first-replica main rank, after at least one train step); destroy
        # process groups; pause memory-saver regions: with master-weight rematerialization on the
        # actor role, pause the grad-buffer and default tags (params stay resident for the weight
        # update, which pauses them later); with LoRA rollout, pause grad buffers once (tracking a
        # flag) and the default tag; otherwise pause everything. Marks asleep and logs memory;
        # logs CPU memory after offload when applicable. Decorated for logging and timing.
        assert self.args.offload_train
        if self._asleep:
            logger.info("sleep() called while already offloaded; skipping")
            return

        self._clear_quantized_weight_workspaces()
        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        should_log_cpu_memory = is_first_replica_megatron_main_rank() and self._last_rollout_id is not None

        destroy_process_groups()

        if self.args.rematerialize_param_from_master_weight and self.role == "actor":
            # Params stay resident for update_weights, which pauses them afterwards.
            torch_memory_saver.pause(tag="grad_buffer")
            torch_memory_saver.pause(tag="default")
        elif lora_rollout_enabled(self.args):
            # adapter params keep their host backup in "default"; the grad buffers have none
            if not self._grad_buffer_paused:
                torch_memory_saver.pause(tag="grad_buffer")
                self._grad_buffer_paused = True
            torch_memory_saver.pause(tag="default")
        else:
            torch_memory_saver.pause(tag=None)

        self._asleep = True
        print_memory("after offload model")

        if should_log_cpu_memory:
            log_cpu_memory(self._last_rollout_id, self.args, "after_offload_train")

    @with_logs
    @timer
    def offload_grad_buffer(self) -> None:
        # [FUNCTION DOC]
        # Asserts train offload and LoRA rollout are on. Returns without effect if already asleep
        # or the grad buffer is already paused. Otherwise pauses the memory-saver grad-buffer tag,
        # sets the paused flag, and logs memory before and after. Decorated for logging and timing.
        """Free the LoRA grad buffers ahead of sleep(), so the engine can resume its weights first."""
        assert self.args.offload_train
        assert lora_rollout_enabled(self.args), "only LoRA keeps its grad buffers in a no-backup region"
        if self._asleep or self._grad_buffer_paused:
            return
        print_memory("before offload grad buffer")
        torch_memory_saver.pause(tag="grad_buffer")
        self._grad_buffer_paused = True
        print_memory("after offload grad buffer")

    @with_logs
    @timer
    def wake_up(self) -> None:
        # [FUNCTION DOC]
        # Reverse of sleep (asserts train offload). If not asleep, only ensures process groups are
        # reloaded and returns. Otherwise resumes memory-saver regions (with LoRA rollout: default
        # and grad-buffer tags, clearing the grad-buffer-paused flag; else all), clears GPU memory,
        # reloads process groups, marks awake, and logs memory. Decorated for logging and timing.
        assert self.args.offload_train
        if not self._asleep:
            logger.info("wake_up() called while already resident; ensuring process groups only")
            reload_process_groups()
            return
        print_memory("before wake_up model")

        if lora_rollout_enabled(self.args):
            torch_memory_saver.resume(tag="default")
            torch_memory_saver.resume(tag="grad_buffer")
            self._grad_buffer_paused = False
        else:
            torch_memory_saver.resume(tag=None)

        clear_memory()
        reload_process_groups()
        self._asleep = False
        print_memory("after wake_up model")

    @property
    def _weight_sync_reads_tms_backup(self) -> bool:
        # [FUNCTION DOC] HELPER: Property. True only when all hold: colocated mode, LoRA enabled,
        # offload target is CPU, and no reference model, teacher or old actor is in use. In that case
        # weight sync reads the memory saver's own host backup instead of a separate actor backup.
        """Under colocated LoRA the frozen base already has a memory-saver host backup; a
        pinned "actor" copy of it would duplicate the whole base per rank. Model switching
        still needs the real backups, and a disk offload target leaves no host backup to read."""
        return (
            self.args.colocate
            and is_lora_enabled(self.args)
            and self.args.offload_train_target == "cpu"
            and not (self.with_ref or self.with_opd_teacher or self.args.keep_old_actor)
        )

    @property
    def _enable_weight_backup(self) -> bool:
        # [FUNCTION DOC] HELPER: Property. False when weight sync reads the memory saver backup;
        # otherwise true if any of: reference model, teacher, old actor, colocate, train offload.
        """Keep host weights for model switching and weight sync while offloaded."""
        if self._weight_sync_reads_tms_backup:
            return False
        return (
            self.with_ref
            or self.with_opd_teacher
            or self.args.keep_old_actor
            or self.args.colocate
            or self.args.offload_train
        )

    def _switch_model(self, target_tag: str) -> None:
        # [FUNCTION DOC] HELPER: Loads the host backup stored under the tag into the live model and
        # records it as active. No-op when weight backup is disabled. Raises a value error for a tag
        # with no backup.
        if not self._enable_weight_backup:
            return
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _set_replay_stage(self, stage: str) -> None:
        # [FUNCTION DOC] HELPER: Sets the stage string on every registered replay manager.
        for m in all_replay_managers:
            m.stage = stage

    @with_logs
    def _compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        rollout_id: int,
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        # [FUNCTION DOC] HELPER: Runs a forward-only pass over the data iterator computing log
        # probabilities and entropy, inside a timer named with the prefix plus "log_probs". Output
        # keys are prefixed with store_prefix (empty for actor, "ref_", "teacher_"); not forced to
        # float32. The rollout sampling mask is used only for the unprefixed (actor) pass and when
        # sampling-support replay is on. Returns a mapping from key to a list of tensors.

        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
                store_prefix=store_prefix,
                fp32_output=False,
                use_rollout_sampling_mask=store_prefix == "" and self.args.use_sampling_support_replay,
            )

    @with_logs
    @event_logger_context(
        lambda _self, rollout_id, rollout_data_ref, witness_info=None, attempt=0, external_data=None: dict(
            rollout_id=rollout_id, attempt=attempt
        )
    )
    def train(
        self,
        rollout_id: int,
        rollout_data_ref: StoreObjectRef | list[StoreObjectRef],
        witness_info: WitnessInfo | None = None,
        attempt: int = 0,
        external_data: TrainStepOutput | None = None,
    ) -> TrainStepOutput:
        # [FUNCTION DOC]
        # Runs one training step for a rollout and returns a train-step output. Decorated to log
        # and to record an event-log context with the rollout id and attempt. Bumps the heartbeat,
        # remembers the rollout id, wakes up if offloaded. Fetches the rollout data from the object
        # store reference(s) (within a timer "data_preprocess"; the fetch result is kept open until
        # the end via an exit stack); in rollout-only debug mode logs the data and returns a normal
        # outcome without training. Critic role: trains the critic under timer "critic_train".
        # Otherwise trains the actor, passing external data (e.g. critic values), witness info and
        # attempt.
        self._heartbeat.bump()
        self._last_rollout_id = rollout_id
        if self.args.offload_train and self._asleep:
            self.wake_up()

        with ExitStack() as stack:
            with timer("data_preprocess"):
                rollout_data, store_get_result = get_rollout_data(
                    self.args, rollout_data_ref, witness_info=witness_info
                )
                stack.enter_context(store_get_result)
                if self.args.debug_rollout_only:
                    log_rollout_data(rollout_id, self.args, rollout_data)
                    return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

            if self.role == "critic":
                with timer("critic_train"):
                    result = self._train_critic(rollout_id, rollout_data)
            else:
                result = self._train_actor(
                    rollout_id,
                    rollout_data,
                    external_data=external_data,
                    witness_info=witness_info,
                    attempt=attempt,
                )

            return result

    @with_logs
    def _train_critic(self, rollout_id: int, rollout_data: RolloutBatch) -> TrainStepOutput:
        # [FUNCTION DOC] HELPER: Builds the data iterator and microbatch counts; computes values by a
        # forward-only pass and merges them into the rollout data; computes advantages and returns;
        # sets the loss type on args to value loss (mutates args); runs the training loop with no
        # witness info and attempt zero. Bumps heartbeat. If this rank is the last pipeline stage
        # and values exist, copies them to CPU and puts them in the object store under the critic
        # value spec, getting a reference. Returns a train-step output with the outcome and the
        # values reference (none on other ranks).
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        rollout_data.update(
            forward_only(
                get_values,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                rollout_id=rollout_id,
            )
        )

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train_step_outcome: TrainStepOutcome = train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            get_num_rollouts(self.args, rollout_data, len(num_microbatches)),
            witness_info=None,
            attempt=0,
        )

        self._heartbeat.bump()
        values = None
        if get_parallel_state().is_pp_last_stage and "values" in rollout_data:
            # Ship by object reference
            values = object_store.get_instance().put(
                value={"values": [value.detach().cpu() for value in rollout_data["values"]]},
                value_spec=CRITIC_VALUES_VALUE_SPEC,
            )
        return TrainStepOutput(outcome=train_step_outcome, values=values)

    def _use_rollout_replay(self, m) -> bool:
        # [FUNCTION DOC] HELPER: Whether the args flag named "use_rollout_" plus the manager's name
        # plus "_replay" is set (default false); i.e. replay data comes from the rollout.
        return getattr(self.args, f"use_rollout_{m.name}_replay", False)

    @with_logs
    def _train_actor(
        self,
        rollout_id: int,
        rollout_data: RolloutBatch,
        external_data=None,
        *,
        witness_info: WitnessInfo | None,
        attempt: int,
    ) -> TrainStepOutput:
        # [FUNCTION DOC] HELPER:
        # Actor training for one rollout. Builds the data iterator and microbatch counts (one entry
        # per optimizer step). If skip-actor-forward-only is set, asserts exactly one optimizer step
        # and no precomputed actor log probs. For each replay manager using rollout replay, fills
        # its replay data from the rollout data. Then, inside the "train" timer (and out of the
        # "train_wait" timer): if advantages are computed here: for ref and teacher backups, set
        # replay stage "fallthrough", switch weights and merge their forward-only log probs under
        # "ref_"/"teacher_" prefixes; switch to old-actor (when kept) or actor weights; unless
        # skipped, or rollout log probs are used without mismatch metrics, set enabled replay
        # managers to "replay_forward" (rollout replay) or "record", compute actor log probs, then
        # clear forward replay data; with a critic and external data on the last pipeline stage,
        # fetch shipped values from the object store (asserting they exist) and materialize them on
        # the current GPU; switch back to actor weights if needed; compute advantages and returns
        # over the whole rollout and log the event. Then run the optional rollout-data post-process
        # function, log rollout data, set replay stage "replay_backward", train under the
        # "actor_train" timer with witness info, attempt and the fault-tolerance test executor, step
        # the profiler. After: save debug train data, clear enabled replay managers. If the outcome
        # is normal: refresh the host actor backup (or synchronize CUDA when backups are disabled)
        # and, when the reference-update interval divides rollout id plus one and a ref backup
        # exists, re-back-up ref from the current weights (under timer "ref_model_update"). Logs perf
        # data with weight updater metrics. Bumps heartbeat and returns the outcome in an output.
        # Create data iterator for log_probs and train.
        data_iterator, num_microbatches = get_data_iterator(self.args, self.model, rollout_data)
        num_optimizer_steps = len(num_microbatches)
        skip_actor_forward_only = self.args.skip_actor_forward_only
        if skip_actor_forward_only:
            option = "--skip-actor-forward-only"
            assert num_optimizer_steps == 1, f"{option} requires 1 optimizer step, got {num_optimizer_steps}"
            assert rollout_data.get("log_probs") is None, f"{option} requires rollout data without actor log probs"

        for m in all_replay_managers:
            if self._use_rollout_replay(m):
                fill_replay_data(
                    args=self.args,
                    models=self.model,
                    data_iterator=data_iterator,
                    num_microbatches=num_microbatches,
                    rollout_data=rollout_data,
                    data_key=m.data_key,
                    replay_list=m.replays,
                    register_replay_list_func=m.register_replay_list_func,
                    if_sp_region=m.if_sp_region,
                    indices_are_token_positions=m.replay_indices_are_token_positions,
                )

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("ref")
                    rollout_data.update(
                        self._compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            rollout_id=rollout_id,
                            store_prefix="ref_",
                        )
                    )
                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    self._set_replay_stage("fallthrough")
                    self._switch_model("teacher")
                    rollout_data.update(
                        self._compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            rollout_id=rollout_id,
                            store_prefix="teacher_",
                        )
                    )
                self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                if not skip_actor_forward_only and (
                    not self.args.use_rollout_logprobs or self.args.get_mismatch_metrics
                ):
                    for m in all_replay_managers:
                        if m.enabled:
                            if self._use_rollout_replay(m):
                                m.stage = "replay_forward"
                            else:
                                m.stage = "record"
                    rollout_data.update(
                        self._compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            rollout_id=rollout_id,
                            store_prefix="",
                        )
                    )
                    for m in all_replay_managers:
                        if self._use_rollout_replay(m):
                            m.clear_all_forward()

                if self.args.use_critic:
                    if external_data is not None and get_parallel_state().is_pp_last_stage:
                        values_ref = external_data.values
                        assert values_ref is not None, (
                            "actor and critic share the same parallel topology, so the critic rank "
                            "paired with a pp-last-stage actor rank must have shipped 'values'"
                        )
                        with object_store.get_instance().get(values_ref) as shipped:
                            rollout_data["values"] = _materialize_critic_values(
                                values=shipped["values"], device=torch.cuda.current_device()
                            )
                if self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)
                log_train_advantage_computation_event(rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args)

            log_rollout_data(rollout_id, self.args, rollout_data)

            # Train
            num_rollouts = get_num_rollouts(self.args, rollout_data, num_optimizer_steps)
            self._set_replay_stage("replay_backward")
            with timer("actor_train"):
                train_step_outcome = train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    num_rollouts,
                    witness_info=witness_info,
                    attempt=attempt,
                    ft_test_action_executor=self._ft_test_action_executor,
                )

            self.prof.step(rollout_id=rollout_id)

        train_dump.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        for m in all_replay_managers:
            if m.enabled:
                m.clear_all()

        if train_step_outcome == TrainStepOutcome.NORMAL:
            # update the cpu actor weight to the latest model
            if self._enable_weight_backup:
                self.weights_backuper.backup("actor")
            else:
                torch.cuda.synchronize()

            # Update ref model if needed
            if (
                self.args.ref_update_interval is not None
                and (rollout_id + 1) % self.args.ref_update_interval == 0
                and "ref" in self.weights_backuper.backup_tags
            ):
                with timer("ref_model_update"):
                    if is_first_replica_megatron_main_rank():
                        logger.info(f"Updating ref model at rollout_id {rollout_id}")
                    self.weights_backuper.backup("ref")

        log_perf_data(
            rollout_id,
            self.args,
            extra_metrics=self.weight_updater.pop_metrics() if self.weight_updater is not None else {},
        )

        self._heartbeat.bump()
        return TrainStepOutput(outcome=train_step_outcome)

    @timer
    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # [FUNCTION DOC]
        # Saves a Megatron checkpoint for the rollout id (timed). Returns immediately in
        # rollout-only debug mode. Steps: finalize any pending async save; save model, optimizer
        # and scheduler (passing the snapshot publisher); on the actor role with HF saving
        # configured, asserts a publisher exists and exports an HF checkpoint; if force_sync,
        # finalizes the async save again. If a custom post-save hook path is set and this is global
        # rank zero: finalize async save, compute the Megatron checkpoint directory and the
        # HF checkpoint directory (the HF path template formatted with the rollout id, actor only,
        # else none), load the hook function lazily and call it with args, rollout id and both
        # directories.
        self._heartbeat.bump()
        if self.args.debug_rollout_only:
            return

        self._finalize_pending_async_save()

        save(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            snapshot_publisher=self.snapshot_publisher,
        )

        if self.args.save_hf is not None and self.role == "actor":
            assert self.snapshot_publisher is not None, "HF export requires a snapshot publisher"
            save_hf_model(self.args, rollout_id, self.model, publisher=self.snapshot_publisher)

        if force_sync:
            self._finalize_pending_async_save()

        if self.args.custom_megatron_post_save_hook_path is not None and dist.get_rank() == 0:
            self._finalize_pending_async_save()

            from megatron.training.checkpointing import get_checkpoint_name

            from miles.utils.function_registry import load_function

            checkpoint_dir = get_checkpoint_name(self.args.save, rollout_id, return_base_dir=True)
            hf_checkpoint_dir = (
                self.args.save_hf.format(rollout_id=rollout_id)
                if self.args.save_hf is not None and self.role == "actor"
                else None
            )
            post_save_hook = load_function(self.args.custom_megatron_post_save_hook_path)
            post_save_hook(self.args, rollout_id, checkpoint_dir, hf_checkpoint_dir)

    @with_logs
    @timer
    def export_hf(self, rollout_id: int, path: str) -> None:
        # [FUNCTION DOC]
        # Collective HF export to the given directory path. Bumps heartbeat, asserts a snapshot
        # publisher exists, and calls the HF save routine with the publisher, the explicit path
        # and error raising enabled (so failures propagate). Decorated for logging and timing.
        """Export current weights as an HF checkpoint to ``path`` (collective).

        Uses the direct megatron->HF converters (the weight updater's machinery), so
        export coverage matches weight-sync coverage. Unlike the periodic --save-hf
        path inside save_model, failures propagate to the caller so an eval snapshot
        that failed to export can be skipped loudly.
        """
        self._heartbeat.bump()
        assert self.snapshot_publisher is not None, "HF export requires a snapshot publisher"
        save_hf_model(
            self.args, rollout_id, self.model, publisher=self.snapshot_publisher, path=path, raise_on_error=True
        )

    def _named_actor_weights(self, *, translate_gpu_to_cpu: bool = False):
        # [FUNCTION DOC] HELPER: Returns an iterator of (name, tensor) pairs of the model's
        # parameters and buffers; names are converted to global names only in raw conversion mode;
        # optionally translates GPU tensors to their CPU backup. Used as the backuper's source getter.
        return named_params_and_buffers(
            self.args,
            self.model,
            convert_to_global_name=self.args.megatron_to_hf_mode == "raw",
            translate_gpu_to_cpu=translate_gpu_to_cpu,
        )

    def _get_actor_weights(self):
        # [FUNCTION DOC] HELPER: Weights getter for the weight updater. Returns a mapping from name
        # to tensor: from the memory saver CPU backup when that mode applies; the host actor backup
        # when colocated, asleep, or the active tag is not actor; otherwise the live GPU weights.
        if self._weight_sync_reads_tms_backup:
            return dict(self._named_actor_weights(translate_gpu_to_cpu=True))
        # use cpu backup only when weight is not live on gpu
        if self.args.colocate or self._asleep or self._active_model_tag != "actor":
            return self.weights_backuper.get("actor")
        return dict(self._named_actor_weights())

    @with_logs
    @timer
    def update_weights(self, info: UpdatableEngines) -> int | None:
        # [FUNCTION DOC]
        # Pushes the actor weights to rollout engines; returns the new weight version, or none when
        # skipped. Returns none at once in train-only or rollout-only debug mode; asserts a weight
        # updater exists. If offloaded and asleep, reloads process groups (temporary) and destroys
        # them again before returning. Reconnects the updater when its connection status says so
        # for the snapshot cell hashes (with the memory saver disabled when offload is on, since
        # connection setup allocates CUDA tensors). In skip-weight-update debug mode: warns on rank
        # zero, pauses the param-buffer tag if rematerializing, tears down temporary groups, returns
        # none. Otherwise, with the memory saver disabled under offload: logs memory, runs the
        # update, in CI without LoRA verifies engine versions, and for keep-old-actor with interval
        # one rotates backups (rollout_actor to old_actor, then actor to rollout_actor) or with
        # other intervals re-backs-up old_actor. Afterwards pauses param buffer if rematerializing
        # and destroys temporary groups. Returns the updater's weight version.
        # KNOWN ISSUE: the keep-old-actor flag is read with a default here but accessed directly
        # elsewhere in this class.
        self._heartbeat.bump()
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return None

        assert self.weight_updater is not None, "weight update requires a weight updater"
        rollout_engines = info.rollout_engines

        process_groups_are_temporary = self.args.offload_train and self._asleep
        if process_groups_are_temporary:
            reload_process_groups()

        needs_reconnect = self.weight_updater.conn_status.needs_reconnect(info.snapshot_cell_id_to_hashes)
        if needs_reconnect:
            # Connection setup also allocates CUDA tensors (e.g. NCCL object
            # collectives). Do not reuse unmapped, offloaded allocator blocks.
            with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
                self.weight_updater.reconnect(info)

        if self.args.debug_skip_weight_update:
            if dist.get_rank() == 0:
                logger.warning("Skipping actor-to-rollout weight update because " "--debug-skip-weight-update is set.")
            if self.args.rematerialize_param_from_master_weight:
                torch_memory_saver.pause(tag="param_buffer")
            if process_groups_are_temporary:
                destroy_process_groups()
            return None

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and not is_lora_enabled(self.args):
                self.weight_updater.verify_engine_version(rollout_engines)

            if getattr(self.args, "keep_old_actor", False):
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if self.args.rematerialize_param_from_master_weight:
            torch_memory_saver.pause(tag="param_buffer")
        if process_groups_are_temporary:
            destroy_process_groups()

        return self.weight_updater.weight_version

    @with_logs
    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        # [FUNCTION DOC]
        # Loads a model-only checkpoint from the path into the live model and backs it up under
        # the tag, which becomes active. Temporarily mutates args: load path, no-load optimizer,
        # no-load rng, finetune true (the old values are restored after the load). For tag "ref"
        # with a reference checkpoint step, or tag "teacher" with a teacher checkpoint step, the
        # args checkpoint step is temporarily replaced and restored. Calls the checkpoint loader
        # with no optimizer or scheduler and an empty checkpointing context.
        # KNOWN ISSUE: args are not restored if loading raises (no try/finally).
        old_args = self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        # load_checkpoint reads self.args.ckpt_step to pick which iteration to load.
        # Temporarily override it for ref/teacher loads, then restore after the load below.
        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.ref_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            old_ckpt_step = self.args.ckpt_step
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
        )
        self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old_args

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        if model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            self.args.ckpt_step = old_ckpt_step

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    @with_logs
    def send_ckpt(self, dst_rank: int) -> None:
        # [FUNCTION DOC]
        # Fault-tolerance healing, sender side: sends the current model, optimizer, scheduler
        # state and the last trained rollout id as the iteration to the destination rank within the
        # independent-DP group. Asserts that keep-old-actor is off and at least one step was trained.
        # These states are not handled
        assert not self.args.keep_old_actor
        assert self._last_rollout_id is not None, "healing before the first train step is unsupported"

        _send_ckpt(
            indep_dp=get_parallel_state().indep_dp,
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.opt_param_scheduler,
            iteration=self._last_rollout_id,
            dst_rank=dst_rank,
        )

    @with_logs
    def reconfigure_indep_dp(self, indep_dp_info: IndepDPInfo, indep_dp_store_addr: str | None) -> None:
        # [FUNCTION DOC]
        # Rebuilds the independent data-parallel group for a changed membership using the new store
        # address and info and this rank's global rank and world size, then (if a weight updater
        # exists) marks its trainer connection as stale so the next update reconnects.
        reconfigure_indep_dp_group(
            parallel_state=get_parallel_state(),
            store_addr=indep_dp_store_addr,
            indep_dp_info=indep_dp_info,
            megatron_rank=dist.get_rank(),
            megatron_world_size=dist.get_world_size(),
        )
        if self.weight_updater is not None:
            self.weight_updater.conn_status.mark_trainer_stale()


def _materialize_critic_values(
    values: list[list[float] | torch.Tensor], device: torch.device | int
) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: Converts each entry (a list of floats or a tensor) into a float32
    # tensor on the given device (device object or index); uses a clone so results own their memory.
    # Returns a list of tensors in input order.
    return [torch.as_tensor(value, dtype=torch.float32, device=device).clone() for value in values]
