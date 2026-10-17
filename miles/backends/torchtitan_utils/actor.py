# [MODULE DOC] Ray training actor for the torchtitan backend. Specializes the torch-native actor
# base: builds a torchtitan trainer from the parsed arguments, publishes the matching parallel state,
# installs routing replay, loads the initial or resumed checkpoint, optionally builds a reference
# model that sits on the host between uses, wires the weight updater to rollout engines through the
# Hugging Face weight iterator, and saves checkpoints through the trainer's checkpointer.
import logging
from argparse import Namespace
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.backends.torchtitan_utils import compat
from miles.backends.torchtitan_utils.config import build_trainer_config
from miles.backends.torchtitan_utils.hf_weight_iterator import TitanHfWeightIterator
from miles.backends.torchtitan_utils.parallel import create_titan_parallel_state, parallel_dims_from_config
from miles.backends.torchtitan_utils.routing_replay import install as install_routing_replay
from miles.backends.torchtitan_utils.trainer import TitanTrainer
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.replay.routing_replay import enable as enable_routing_replay
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.utils.context_utils import with_defer
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.memory_utils import clear_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.workers.rpc.common.wire_types import Pickled

logger = logging.getLogger(__name__)


def _steps_per_rollout(args: Namespace) -> int:
    # [FUNCTION DOC] HELPER: Optimizer steps per rollout: rollout batch size times samples per
    # prompt, integer-divided by the global batch size, but never less than one.
    return max(args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size, 1)


class TorchtitanTrainRayActor(TorchNativeTrainRayActor):
    # [CLASS DOC] Training actor whose backend name is "torchtitan". Instance state set during init:
    # profiler, trainer wrapper, the trainer's model parts and optimizers, the token-alignment side
    # channel flag, an optional reference-model step runner (with the reference model parts kept on
    # the host), and the weight updater. Checkpoint healing, on-policy distillation teachers and
    # independent data parallelism with more than one quorum are unsupported.
    backend_name = "torchtitan"

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
        # [FUNCTION DOC] Initializes the actor and returns the rollout id to start from. A defer
        # decorator starts the "train wait" timer when init finishes (including on error). Keyword
        # options: whether a reference model is needed, whether an on-policy-distillation teacher is
        # needed (unsupported), the rank to receive a checkpoint from (must be none), independent
        # data parallel info (quorum id must be zero) and its store address (must be none);
        # violations raise assertion errors.
        # Steps: install the compatibility patches; run the common base initialization; build the
        # trainer config from the arguments (assets path = HF checkpoint, learning-rate total steps
        # = rollouts times steps per rollout, dump subdirectory "actor"); enable routing replay.
        # Debug rollout-only mode: publishes a parallel state built from the config and returns
        # zero without building a trainer. Otherwise: creates the train profiler, loads Hugging Face
        # assets, builds the trainer and records its model parts, optimizers and token-alignment
        # side channel; configures loss reduction; publishes the parallel state (marking whether
        # this rank holds the last pipeline stage); installs routing replay on the model parts;
        # initializes metric tracking (non-primary) only on context-parallel rank zero that is
        # also a metrics rank; loads the checkpoint through the trainer's checkpointer; computes the
        # resume rollout id as trainer step divided by steps per rollout; builds the reference
        # runner if requested; builds the weight updater with the Hugging Face iterator factory;
        # clears memory; puts training to sleep (offloads) if train offload is enabled; marks the
        # profiler init finished. Returns the explicit start rollout id from arguments if set, else
        # the computed resume id.
        compat.install()
        super()._init_common(args, role, with_ref, with_opd_teacher=with_opd_teacher)
        assert indep_dp_info.quorum_id == 0
        assert indep_dp_store_addr is None

        assert recv_ckpt_src_rank is None, "torchtitan backend does not support checkpoint healing"
        assert not with_opd_teacher, "torchtitan backend does not support on-policy distillation yet"

        config = build_trainer_config(
            args,
            hf_assets_path=args.hf_checkpoint,
            lr_total_steps=args.num_rollout * _steps_per_rollout(args),
            dump_subdir="actor",
        )

        enable_routing_replay(args)

        if args.debug_rollout_only:
            set_parallel_state(create_titan_parallel_state(parallel_dims_from_config(config.parallelism)))
            return 0

        self.prof = TrainProfiler(args)
        self.load_hf_assets()

        self.trainer = TitanTrainer(config)
        self.model_parts = self.trainer.model_parts
        self.optimizers = self.trainer.optimizers.optimizers
        self.align_token_side_channel = self.trainer.align_token_side_channel
        self.trainer.configure_loss_reduction()
        set_parallel_state(
            create_titan_parallel_state(self.trainer.parallel_dims, is_pp_last_stage=self.trainer.has_last_stage())
        )
        install_routing_replay(self.trainer.model_parts)

        cp_mesh = self.trainer.parallel_dims.get_optional_mesh("cp")
        cp_rank0 = cp_mesh is None or dist.get_rank(cp_mesh.get_group()) == 0
        if cp_rank0 and get_parallel_state().is_metrics_rank:
            init_tracking(args, primary=False)

        self.trainer.checkpointer.load()
        start_rollout_id = self.trainer.step // _steps_per_rollout(args)

        if with_ref:
            self.ref_runner = self._build_ref_runner(args)

        self.weight_updater = self._build_weight_updater(self.trainer, TitanHfWeightIterator.build)

        clear_memory()
        if args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return args.start_rollout_id if args.start_rollout_id is not None else start_rollout_id

    def _step_runner(self):
        # [FUNCTION DOC] HELPER: Returns the trainer's step runner used by the base class to run
        # training steps.
        return self.trainer.step_runner()

    def _build_ref_runner(self, args: Namespace):
        # [FUNCTION DOC] HELPER: Builds the reference model and returns its step runner. Raises a
        # value error when no reference load path is set. Builds a trainer config from the reference
        # path (single learning-rate step, dump subdirectory "ref"), a second trainer, configures
        # loss reduction, loads its checkpoint, then for every model part sets eval mode, disables
        # gradients and moves it to the CPU; empties the CUDA cache; stores the parts for later
        # device moves; logs an info message. The reference model stays on the host between passes.
        if not args.ref_load:
            raise ValueError("--ref-load is required to build a torchtitan reference model")
        ref_config = build_trainer_config(args, hf_assets_path=args.ref_load, lr_total_steps=1, dump_subdir="ref")
        ref_trainer = TitanTrainer(ref_config)
        ref_trainer.configure_loss_reduction()
        ref_trainer.checkpointer.load()
        for part in ref_trainer.model_parts:
            part.eval()
            part.requires_grad_(False)
            part.cpu()
        torch.cuda.empty_cache()
        self._ref_parts = ref_trainer.model_parts
        logger.info(f"Built a torchtitan reference trainer from {args.ref_load}; it lives on the host between passes")
        return ref_trainer.step_runner()

    @contextmanager
    def _ref_context(self):
        # [FUNCTION DOC] HELPER: Context manager that moves the reference model parts to the GPU for
        # the duration of the block and always moves them back to the host and empties the CUDA cache
        # afterwards, even on error.
        for part in self._ref_parts:
            part.cuda()
        try:
            yield
        finally:
            for part in self._ref_parts:
                part.cpu()
            torch.cuda.empty_cache()

    def _save_checkpoint(self, rollout_id: int) -> None:
        # [FUNCTION DOC] HELPER: Saves a checkpoint at the trainer's current step, flagged as the
        # last step. The rollout id argument is ignored.
        self.trainer.checkpointer.save(self.trainer.step, last_step=True)
