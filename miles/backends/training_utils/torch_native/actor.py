# [MODULE DOC]
# Base Ray training actor for PyTorch-native backends. One rollout step: fetch rollout data, build
# micro-batch iterators, optionally fill routing-replay buffers, compute reference and policy log
# probs (and entropy) with forward-only passes, compute advantages/returns and log rollout metrics,
# then run one optimizer step per global batch with a backend-supplied step runner, logging train
# metrics. Also provides model checkpoint hooks, offload to CPU (sleep) and back (wake up), and the
# weight update to inference engines. Subclasses must provide the step runner, checkpoint saving and
# set the backend name, model parts, optimizers, weight updater, profiler, HF config and tokenizer.
# Module constants list the batch keys fetched for forward-only passes, training steps and the
# sampling-support replay.
import functools
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import ClassVar

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from miles.backends.training_utils.data.rollout import DataIterator, get_batch, get_data_iterator, get_rollout_data
from miles.backends.training_utils.data.sampling_mask import get_rollout_sampling_masks
from miles.backends.training_utils.loss.objective import (
    compute_advantages_and_returns,
    get_log_probs_and_entropy,
    loss_function,
)
from miles.backends.training_utils.metrics import perf
from miles.backends.training_utils.metrics.checks import check_grad_norm
from miles.backends.training_utils.metrics.log_utils import (
    aggregate_forward_results,
    aggregate_train_losses,
    log_rollout_data,
    log_train_step,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.replay import routing_replay
from miles.backends.training_utils.torch_native.offload import move_train_state
from miles.backends.training_utils.torch_native.step_runner import StepRunner
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.backends.training_utils.weight_update.updater import WeightUpdater
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.ray.train_actor import TrainRayActor
from miles.utils.audit_utils.witness.allocator import WitnessInfo
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.flops_utils import flops_args_from_hf_config, fwd_tflops_per_gpu
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.object_store import StoreObjectRef
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import inverse_timer, timer

logger = logging.getLogger(__name__)

FORWARD_ONLY_KEYS = [
    "tokens",
    "loss_masks",
    "multimodal_train_inputs",
    "total_lengths",
    "response_lengths",
    "max_seq_lens",
]
TRAIN_KEYS = FORWARD_ONLY_KEYS + [
    "log_probs",
    "advantages",
    "returns",
    "ref_log_probs",
    "rollout_log_probs",
    "rollout_topk_token_ids",
    "rollout_topk_lengths",
    "rollout_topk_log_probs",
]
SAMPLING_MASK_KEYS = ["rollout_sampling_mask_ids", "rollout_sampling_mask_offsets"]


class TorchNativeTrainRayActor(TrainRayActor):
    # [CLASS DOC] Abstract training actor. Subclass-provided attributes: backend name (class level),
    # model parts, optimizers, weight updater, profiler, HF config, tokenizer. Optional: a reference
    # step runner (default none; when set, reference log probs are computed with it) and a callable
    # that aligns token-side-channel tensors (such as routing-replay indices) to the token layout
    # (default none). Hooks to override: the step runner factory, reference context manager, the
    # after-rollout hook and checkpoint saving.
    backend_name: ClassVar[str]
    model_parts: Sequence[torch.nn.Module]
    optimizers: Sequence[torch.optim.Optimizer]
    weight_updater: WeightUpdater
    prof: TrainProfiler
    hf_config: PretrainedConfig
    tokenizer: PreTrainedTokenizerBase
    ref_runner: StepRunner | None = None
    align_token_side_channel: Callable[[torch.Tensor, int], torch.Tensor] | None = None

    def _step_runner(self) -> StepRunner:
        # [FUNCTION DOC] Abstract hook: return the runner used for policy forward/backward and
        # optimizer steps. Raises not-implemented in the base class.
        raise NotImplementedError

    def _ref_context(self) -> AbstractContextManager:
        # [FUNCTION DOC] Hook: context manager entered around the reference log-prob pass (e.g. to
        # swap in reference weights). Default is a do-nothing context.
        return nullcontext()

    def _after_rollout(self, rollout_id: int, rollout_data: dict) -> None:
        # [FUNCTION DOC] Hook called at the end of each rollout's training with the rollout id and
        # the (mutated) rollout data. Default does nothing.
        pass

    @property
    def train_parallel_config(self) -> dict:
        # [FUNCTION DOC] Property: mapping with the data-parallel size (intra-DP group size from the
        # global parallel state) under the key for dp size.
        return {"dp_size": get_parallel_state().intra_dp.size}

    def _build_weight_updater(self, model, iterator_factory: Callable) -> WeightUpdater:
        # [FUNCTION DOC] HELPER: Construct the weight updater for the model. Model name = args model
        # name, or the lowercased HF config class name if unset. Weights getter returns nothing
        # (weights come from the iterator factory); quantization config is read from the HF config if
        # present; uses the global parallel state; LoRA is disabled.
        model_name = self.args.model_name
        if model_name is None:
            model_name = type(self.hf_config).__name__.lower()
        return WeightUpdater(
            self.args,
            model,
            weights_getter=lambda: None,
            model_name=model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
            iterator_factory=iterator_factory,
            parallel_state=get_parallel_state(),
            is_lora=False,
        )

    @functools.cached_property
    def _fwd_tflops(self) -> Callable[[list[int]], float] | None:
        # [FUNCTION DOC] HELPER: Cached property. Derives model sizing from the HF config; on any
        # error logs a warning (MFU will not be reported) and returns none. Otherwise returns a
        # function from a list of sequence lengths to forward TFLOPs per GPU, using the world size.
        try:
            flops_args = flops_args_from_hf_config(self.hf_config)
        except Exception as e:
            logger.warning(f"MFU will not be reported, {type(self.hf_config).__name__} could not be sized: {e}")
            return None
        return lambda seq_lens: fwd_tflops_per_gpu(seq_lens, flops_args, dist.get_world_size())

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # [FUNCTION DOC] Save a checkpoint through the subclass hook unless debug rollout-only mode
        # is on or no save path is configured. Asserts async saving is off. KNOWN ISSUE: the
        # force_sync parameter is unused.
        if self.args.debug_rollout_only or self.args.save is None:
            return
        assert not self.args.async_save, f"{type(self).__name__} does not support async_save yet."
        self._save_checkpoint(rollout_id)

    def _save_checkpoint(self, rollout_id: int) -> None:
        # [FUNCTION DOC] Abstract hook: write a checkpoint for the rollout id. Raises not-implemented.
        raise NotImplementedError

    @timer
    def sleep(self) -> None:
        # [FUNCTION DOC] Timed. If the offload-train option is on, move model and optimizer state to
        # CPU to free GPU memory.
        if self.args.offload_train:
            self._move_to("cpu")

    @timer
    def wake_up(self) -> None:
        # [FUNCTION DOC] Timed. If the offload-train option is on, move model and optimizer state
        # back to the GPU.
        if self.args.offload_train:
            self._move_to("cuda")

    def _move_to(self, device: str) -> None:
        # [FUNCTION DOC] HELPER: Print memory, move model parts and optimizer state to the device,
        # clear memory caches, synchronize all ranks on the gloo group barrier, print memory again.
        print_memory(f"before moving the model to {device}")
        move_train_state(self.model_parts, self.optimizers, device)
        clear_memory()
        dist.barrier(group=get_gloo_group())
        print_memory(f"after moving the model to {device}")

    def train(
        self,
        rollout_id: int,
        rollout_data_ref: StoreObjectRef | list[StoreObjectRef],
        witness_info: WitnessInfo | None = None,
        attempt: int = 0,
        external_data: TrainStepOutput | None = None,
    ) -> TrainStepOutput:
        # [FUNCTION DOC] Train on one rollout. Asserts no witness info, attempt 0 and no external
        # data (no critic in this backend). Bumps the heartbeat, wakes up if offloading, then within
        # a wait-inverse timer and a train timer loads the rollout data from the object store
        # reference(s) (holding the store result handle open) and, unless debug rollout-only is on
        # (returns a normal outcome immediately, skipping perf logging), runs the core training.
        # Afterwards logs perf data (primary rank = distributed rank 0, using the forward-flops
        # function), bumps the heartbeat again and returns a normal outcome.
        assert witness_info is None and attempt == 0
        assert (
            external_data is None
        ), f"the {self.backend_name} backend trains no critic, so it is never handed critic values"
        self._heartbeat.bump()
        if self.args.offload_train:
            self.wake_up()

        with inverse_timer("train_wait"), timer("train"):
            rollout_data, store_get_result = get_rollout_data(self.args, rollout_data_ref, witness_info=None)
            with store_get_result:
                if self.args.debug_rollout_only:
                    return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
                self._train_core(rollout_id=rollout_id, rollout_data=rollout_data)

        perf.log_perf_data_raw(
            rollout_id=rollout_id,
            args=self.args,
            is_primary_rank=dist.get_rank() == 0,
            compute_total_fwd_flops=self._fwd_tflops,
        )
        self._heartbeat.bump()
        return TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

    def _train_core(self, rollout_id: int, rollout_data: dict) -> None:
        # [FUNCTION DOC] HELPER: Core of one rollout's training, mutating the rollout data. Steps:
        # build micro-batch iterators and per-step microbatch counts (asserts non-empty); fill
        # routing replay buffers (aligned with the side-channel aligner); with the first iterator
        # and the runner from the hook: if a reference runner exists, compute reference log probs
        # (stored with a "ref_" prefix) under the fallthrough replay stage and the reference
        # context; compute policy log probs and entropy under the log-prob replay stage; rewind the
        # replay; compute advantages and returns; log rollout data; run the optimizer steps under
        # the replay-backward stage within an "actor_train" timer; reset replay; advance the profiler
        # and call the after-rollout hook.
        data_iterators, num_microbatches = get_data_iterator(self.args, self.model_parts, rollout_data)
        assert num_microbatches, f"empty microbatch schedule for micro_batch_size={self.args.micro_batch_size}"
        routing_replay.fill(
            self.args,
            self.model_parts,
            data_iterators,
            num_microbatches,
            rollout_data,
            align=self.align_token_side_channel,
        )
        data_iterator = data_iterators[0]
        runner = self._step_runner()

        if self.ref_runner is not None:
            with routing_replay.stage(routing_replay.FALLTHROUGH), self._ref_context():
                rollout_data.update(self._log_probs(self.ref_runner, data_iterator, num_microbatches, "ref_"))
        with routing_replay.stage(routing_replay.log_prob_stage(self.args)):
            rollout_data.update(self._log_probs(runner, data_iterator, num_microbatches))
        routing_replay.rewind()

        compute_advantages_and_returns(self.args, rollout_data)
        log_rollout_data(rollout_id, self.args, rollout_data)

        with routing_replay.stage(routing_replay.REPLAY_BACKWARD), timer("actor_train"):
            self._optimizer_steps(runner, data_iterator, num_microbatches, rollout_id)
        routing_replay.reset()

        self.prof.step(rollout_id=rollout_id)
        self._after_rollout(rollout_id, rollout_data)

    @torch.no_grad()
    def _log_probs(
        self, runner: StepRunner, data_iterator: DataIterator, num_microbatches: list[int], store_prefix: str = ""
    ) -> dict[str, list[torch.Tensor]]:
        # [FUNCTION DOC] HELPER: No-grad forward-only pass over all micro-batches. Resets the
        # iterator; fetches the forward-only keys (plus sampling-mask keys when this is the policy
        # pass, i.e. empty prefix, and sampling-support replay is on). For each step's micro-batch
        # count, iterates a progress bar (shown on rank 0 only) wrapped by the profiler, fetches
        # batches and runs the runner's forward-only step with a compute function that returns log
        # probs under "<prefix>log_probs" and, if produced (only for the policy pass), entropy.
        # Everything is timed under "<prefix>log_probs". Returns the aggregated mapping from key to
        # list of per-sample tensors (restoring original order under dynamic batching).
        args = self.args
        forward_store: list[dict] = []
        data_iterator.reset()
        use_rollout_sampling_mask = store_prefix == "" and args.use_sampling_support_replay
        keys = FORWARD_ONLY_KEYS + SAMPLING_MASK_KEYS if use_rollout_sampling_mask else FORWARD_ONLY_KEYS

        def compute(logits: torch.Tensor, batch: dict) -> dict:
            # [FUNCTION DOC] HELPER: For one batch's logits compute per-sample log probs (and
            # entropy only for the policy pass, not the reference pass), with the rollout sampling
            # mask when sampling-support replay is used. Returns a mapping with the prefixed log
            # probs and, when present, entropy.
            result = get_log_probs_and_entropy(
                logits=logits,
                args=args,
                unconcat_tokens=batch["unconcat_tokens"],
                total_lengths=batch["total_lengths"],
                response_lengths=batch["response_lengths"],
                with_entropy=(store_prefix == ""),
                max_seq_lens=batch.get("max_seq_lens"),
                rollout_sampling_mask=get_rollout_sampling_masks(batch) if use_rollout_sampling_mask else None,
            )
            entry = {f"{store_prefix}log_probs": result["log_probs"]}
            if "entropy" in result:
                entry["entropy"] = result["entropy"]
            return entry

        with timer(f"{store_prefix}log_probs"):
            for microbatches in num_microbatches:
                progress = tqdm(range(microbatches), desc=f"{store_prefix}log_probs", disable=dist.get_rank() != 0)
                batches = self._fetch_batches(self.prof.iterate_train_log_probs(progress), data_iterator, keys)
                forward_store.extend(runner.forward_only_step(batches, compute))

        return aggregate_forward_results(forward_store, data_iterator, args, store_prefix)

    def _optimizer_steps(
        self, runner: StepRunner, data_iterator: DataIterator, num_microbatches: list[int], rollout_id: int
    ) -> None:
        # [FUNCTION DOC] HELPER: Run one optimizer step per entry of the microbatch schedule. Resets
        # the iterator; keys are the train keys (plus sampling-mask keys with sampling-support
        # replay). For each step: zero grads; fetch that step's micro-batches (progress bar on rank
        # 0, profiler-wrapped); run forward-backward with the loss closure bound to args and the
        # micro-batch count; apply the optimizer step to get metrics (grad norm, extras). In CI test
        # mode checks the grad norm (rank = intra-DP x CP rank). Logs the step with aggregated
        # train losses, the role "actor", the total step count per rollout, and logging only on the
        # metrics rank.
        args = self.args
        data_iterator.reset()
        state = get_parallel_state()
        keys = TRAIN_KEYS + SAMPLING_MASK_KEYS if args.use_sampling_support_replay else TRAIN_KEYS

        for step_id, microbatches in enumerate(num_microbatches):
            runner.zero_grad()
            progress = tqdm(range(microbatches), desc="actor_train", disable=dist.get_rank() != 0)
            batches = self._fetch_batches(self.prof.iterate_train_actor(progress), data_iterator, keys)
            losses_reduced = runner.forward_backward_step(batches, partial(_step_loss, args, microbatches))
            metrics = runner.apply_step()

            if args.ci_test:
                check_grad_norm(
                    args=args,
                    grad_norm=metrics.grad_norm,
                    rollout_id=rollout_id,
                    step_id=step_id,
                    role="actor",
                    rank=state.intra_dp_cp.rank,
                )
            log_train_step(
                args=args,
                loss_dict=aggregate_train_losses(losses_reduced),
                grad_norm=metrics.grad_norm,
                rollout_id=rollout_id,
                step_id=step_id,
                num_steps_per_rollout=len(num_microbatches),
                role="actor",
                extra_metrics=metrics.extra_metrics,
                should_log=state.is_metrics_rank,
            )

    def _fetch_batches(self, progress, data_iterator: DataIterator, keys: list[str]) -> Iterator[dict]:
        # [FUNCTION DOC] HELPER: Generator. For each tick of the progress iterable, yields the next
        # batch from the data iterator for the given keys, padded per the pad multiplier, in the
        # configured qkv format, with position ids included. Lazy: nothing is fetched until
        # consumed.
        for _ in progress:
            yield get_batch(
                data_iterator,
                keys,
                self.args.data_pad_size_multiplier,
                self.args.qkv_format,
                get_position_ids=True,
            )

    @timer
    def update_weights(self, info: UpdatableEngines) -> int | None:  # type: ignore[override]
        # [FUNCTION DOC] Timed. Push current weights to the rollout engines. Returns none (no
        # update) in debug train-only or rollout-only mode. Otherwise reconnects the weight updater
        # to the engines if needed, updates weights (with memory printouts around it), in CI test
        # mode verifies the engines' weight version, clears memory, and returns the updater's weight
        # version.
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return None
        self.weight_updater.reconnect_if_needed(info)
        print_memory("before update_weights")
        self.weight_updater.update_weights()
        print_memory("after update_weights")
        if self.args.ci_test:
            self.weight_updater.verify_engine_version(info.rollout_engines)
        clear_memory()
        return self.weight_updater.weight_version


def _step_loss(args, num_microbatches: int, logits: torch.Tensor, batch: dict) -> tuple[torch.Tensor, dict]:
    # [FUNCTION DOC] HELPER: Loss closure for the step runner (bound to args and micro-batch count by
    # partial application). Calls the generic loss function without Megatron loss scaling and
    # returns (loss, logging mapping), discarding the normalizer.
    loss, _normalizer, log_dict = loss_function(
        args=args,
        batch=batch,
        num_microbatches=num_microbatches,
        logits=logits,
        apply_megatron_loss_scaling=False,
    )
    return loss, log_dict
