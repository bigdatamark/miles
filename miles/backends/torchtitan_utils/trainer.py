# [MODULE DOC] Torchtitan trainer subclass used for RL training, plus a thin step-runner facade.
# The trainer takes lists of already-built microbatches (token and position tensors) instead of
# reading a dataset, pads them to shapes required by pipeline or context parallelism, runs forward
# (and backward) through torchtitan's step functions with the RL loss adapter capturing per-
# microbatch outputs, applies the optimizer step with gradient clipping, and configures loss
# scaling and context-parallel handling. Constants: flex-attention block size 128 tokens and a
# context-parallel length bucket of 1024 tokens.
import functools
import inspect
from collections.abc import Callable

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import PipelineScheduleSingle
from torch.distributed.tensor import DTensor
from torchtitan.distributed import utils as titan_dist_utils
from torchtitan.distributed.context_parallel import cp_shard
from torchtitan.trainer import Trainer

from miles.backends.torchtitan_utils import routing_replay
from miles.backends.training_utils.torch_native.step_runner import StepMetrics
from miles.utils.hf_utils.config import load_hf_config

_FLEX_BLOCK = 128
_CP_LENGTH_BUCKET = 1024


class TitanTrainer(Trainer):
    # [CLASS DOC] Trainer specialized for RL. Inherits model parts, optimizers, learning-rate
    # schedulers, checkpointer, loss function (here the RL loss adapter), parallel dimensions,
    # device and pipeline schedule from torchtitan's trainer; adds padding rules, microbatch
    # preparation, forward-only and forward-backward passes over lists of microbatches, an optimizer
    # step returning metrics, loss-scale and context-parallel setup, and a step-runner factory.
    @functools.cached_property
    def _family_forward_kwargs(self) -> dict:
        # [FUNCTION DOC] HELPER: Cached extra keyword arguments for the model's forward. Empty unless
        # the first model part's forward accepts a special-tokens parameter (vision-language
        # families); then a mapping with the image and video placeholder token ids read from the
        # Hugging Face config at the assets path (defaults -1 and -2 when absent).
        kwargs = {}
        if "special_tokens" in inspect.signature(self.model_parts[0].forward).parameters:
            hf_config = load_hf_config(self.config.hf_assets_path)
            kwargs["special_tokens"] = {
                "image_id": getattr(hf_config, "image_token_id", -1),
                "video_id": getattr(hf_config, "video_token_id", -2),
            }
        return kwargs

    def padded_length(self, n_tokens: int) -> int:
        # [FUNCTION DOC] Returns the length a packed microbatch of the given token count must be
        # padded to. With pipeline parallelism: the fixed training sequence length (value error if
        # the count exceeds it, since pipeline stages exchange a fixed shape). With context
        # parallelism: the count rounded up to a multiple of the larger of (context-parallel size
        # times the 128-token flex block) and 1024. Otherwise: unchanged.
        if self.parallel_dims.pp_enabled:
            target = self.config.training.seq_len
            if n_tokens > target:
                raise ValueError(
                    f"packed microbatch of {n_tokens} tokens exceeds --seq-length "
                    f"{target}, which is the fixed shape PP stages exchange"
                )
            return target
        if self.parallel_dims.cp_enabled:
            align = max(self.parallel_dims.cp * _FLEX_BLOCK, _CP_LENGTH_BUCKET)
            return n_tokens + (align - n_tokens % align) % align
        return n_tokens

    def align_token_side_channel(self, tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        # [FUNCTION DOC] Makes a per-token tensor (first dimension is tokens) line up with the
        # model's local tokens. Pads at the end with the given value up to the padded length; with
        # context parallelism, shards it with the same sharding and load balancer as the model input
        # (adding a batch dimension, moving to the trainer device, and returning to the original
        # device); with tensor parallelism (sequence-parallel layout), requires the token count to
        # divide the tensor-parallel size (value error otherwise) and keeps this rank's contiguous
        # chunk. Returns the local tensor.
        missing = self.padded_length(tensor.shape[0]) - tensor.shape[0]
        if missing:
            pad = torch.full((missing, *tensor.shape[1:]), pad_value, dtype=tensor.dtype, device=tensor.device)
            tensor = torch.cat([tensor, pad], dim=0)
        if self.parallel_dims.cp_enabled:
            (local,), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (tensor.unsqueeze(0).to(self.device),),
                None,
                self.config.parallelism.context_parallel_load_balancer,
                input_seq_dim=1,
            )
            tensor = local.squeeze(0).to(tensor.device)
        if self.parallel_dims.tp_enabled:
            mesh = self.parallel_dims.get_mesh("tp")
            tp = mesh.size()
            if tensor.shape[0] % tp:
                raise ValueError(
                    f"a {tensor.shape[0]}-token side channel does not divide across {tp} tensor-parallel ranks"
                )
            tensor = tensor.chunk(tp, dim=0)[dist.get_rank(mesh.get_group())]
        return tensor

    def _microbatch_inputs(self, batches: list) -> tuple[list[dict], list[torch.Tensor]]:
        # [FUNCTION DOC] HELPER: Converts microbatch dictionaries (with token and position tensors,
        # batch dimension first) into model input dictionaries and label tensors. With pipeline
        # parallelism, raises a value error unless the number of microbatches equals the number the
        # schedule was built for. Each microbatch's tokens are padded on the right with zeros to the
        # padded length, and positions are extended with consecutive numbers starting at zero for the
        # padding. The model input dictionary has input tokens, positions, and the family-specific
        # extra arguments. Labels are tensors shaped like the padded tokens filled with the
        # microbatch's index (as long integers), which the loss adapter decodes. Returns the list of
        # input dictionaries and the list of labels.
        # KNOWN ISSUE: padded positions restart from zero instead of continuing after the last
        # real position.
        if self.parallel_dims.pp_enabled:
            expected = self.num_pipeline_parallel_microbatches
            if len(batches) != expected:
                raise ValueError(
                    f"the PP schedule was built for {expected} microbatches per optimizer step "
                    f"but this step has {len(batches)}; global_batch_size / dp / "
                    "micro_batch_size must be constant (no dynamic batch sizing with PP)"
                )

        def _model_inputs(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Returns the microbatch's (tokens, positions) padded to the
            # padded length as described by the enclosing function.
            tokens, positions = batch["tokens"], batch["position_ids"]
            pad = self.padded_length(tokens.shape[1]) - tokens.shape[1]
            if pad:
                tokens = torch.nn.functional.pad(tokens, (0, pad), value=0)
                extra = torch.arange(pad, device=positions.device, dtype=positions.dtype)
                positions = torch.cat([positions, extra.unsqueeze(0)], dim=1)
            return tokens, positions

        input_dicts = []
        for batch in batches:
            tokens, positions = _model_inputs(batch)
            input_dicts.append({"input": tokens, "positions": positions, **self._family_forward_kwargs})
        labels = [torch.full_like(input_dicts[i]["input"], i, dtype=torch.long) for i in range(len(batches))]
        return input_dicts, labels

    def _bypass_schedule_probe(self, *, has_backward: bool) -> None:
        # [FUNCTION DOC] HELPER: If the pipeline schedule is about to run its metadata-inference
        # forward, tells routing replay to bypass that probe so it does not consume replay data.
        if self._pipeline_will_infer_metadata(has_backward=has_backward):
            routing_replay.bypass_schedule_initialization(self.model_parts)

    def _pipeline_will_infer_metadata(self, *, has_backward: bool) -> bool:
        # [FUNCTION DOC] HELPER: True when the pipeline schedule has not yet initialized forward
        # metadata, or when its backward-initialized state differs from the requested backward flag.
        # Reads the single-stage or multi-stage private flags depending on the schedule class.
        schedule = self.pp_schedule
        if isinstance(schedule, PipelineScheduleSingle):
            forward_initialized = schedule._stage_forward_initialized
            backward_initialized = schedule._stage_backward_initialized
        else:
            forward_initialized = schedule._stages_forward_initialized
            backward_initialized = schedule._stages_backward_initialized
        return not forward_initialized or has_backward != backward_initialized

    def run_forward_backward(self, batches, loss_closure: Callable) -> list[dict]:
        # [FUNCTION DOC] Runs training forward and backward over the microbatches, accumulating
        # gradients (no optimizer step). Arms the loss adapter in training mode with the batches and
        # closure, prepares inputs and labels, and uses a scalar one as the valid-token count
        # (scaling is done by the adapter). Inside the replay consumption guard: with pipeline
        # parallelism, bypasses the schedule probe if needed and sends all microbatches in one
        # forward-backward step; otherwise runs one forward-backward step per microbatch in order.
        # Returns the loss adapter's per-microbatch log dictionaries on the stage that holds the
        # loss, else an empty list.
        batches = list(batches)
        self.loss_fn.arm(batches, loss_closure, is_training=True)
        input_dicts, labels = self._microbatch_inputs(batches)
        ones = torch.ones((), device=self.device)
        with routing_replay.consumption_guard(self.model_parts, len(batches)):
            if self.parallel_dims.pp_enabled:
                self._bypass_schedule_probe(has_backward=True)
                self.forward_backward_step(input_dict=input_dicts, labels=labels, global_valid_tokens=ones)
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    self.forward_backward_step(input_dict=input_dict, labels=label, global_valid_tokens=ones)
        return self.loss_fn.collect() if self.has_last_stage() else []

    def run_forward(self, batches, compute: Callable) -> list:
        # [FUNCTION DOC] Runs inference-only forward over the microbatches with the compute function
        # (maps logits and microbatch to a result). Arms the loss adapter in evaluation mode and
        # prepares inputs and labels. Inside the replay consumption guard: with pipeline
        # parallelism, post-processes every microbatch through the trainer's data post-processing
        # to build per-microbatch input, extra-keyword and target lists, bypasses the schedule probe
        # if needed, and runs the schedule's evaluation (inputs given only on the first stage,
        # targets and a loss list only on the last stage, outputs not returned); otherwise, for each
        # microbatch, post-processes, runs the first model part and passes the output and label to
        # the loss adapter. Returns the adapter's per-microbatch results on the stage that has the
        # loss, else an empty list.
        batches = list(batches)
        self.loss_fn.arm(batches, compute, is_training=False)
        input_dicts, labels = self._microbatch_inputs(batches)
        with routing_replay.consumption_guard(self.model_parts, len(batches)):
            if self.parallel_dims.pp_enabled:
                arg_mbs, kwarg_mbs, target_mbs = [], [], []
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    arg_mbs.append((inputs,))
                    kwarg_mbs.append(extra)
                    target_mbs.append(label)
                losses = [] if self.pp_has_last_stage else None
                self._bypass_schedule_probe(has_backward=False)
                self.pp_schedule.eval(
                    arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                    kwarg_mbs=kwarg_mbs,
                    target_mbs=target_mbs if self.pp_has_last_stage else None,
                    losses=losses,
                    return_outputs=False,
                )
            else:
                for input_dict, label in zip(input_dicts, labels, strict=True):
                    inputs, label, extra = self.post_dataloading_process(input_dict, label)
                    pred = self.model_parts[0](inputs, **extra)
                    self.loss_fn(pred, label)
        return self.loss_fn.collect() if self.has_last_stage() else []

    def apply_optimizer_step(self) -> StepMetrics:
        # [FUNCTION DOC] Clips gradients of all model parts to the configured maximum norm (using the
        # pipeline mesh and whether expert parallelism is on), waits for any checkpoint staging,
        # steps the optimizers and learning-rate schedulers, and increments the step counter. A
        # distributed gradient norm is converted to a full tensor. Returns step metrics with the
        # norm as a float and the schedulers' metrics as extra metrics.
        grad_norm = titan_dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()
        self.step += 1
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        return StepMetrics(grad_norm=float(grad_norm.item()), extra_metrics=self.lr_schedulers.get_metrics())

    def configure_loss_reduction(self) -> None:
        # [FUNCTION DOC] Sets the loss adapter's gradient scale to one over (data-parallel replicate
        # size times shard size times context-parallel size). If context parallelism is on: raises
        # the dynamo recompile limit to at least twice (sequence length divided by 1024, plus one);
        # raises a value error unless the load balancer is head-tail; and gives the adapter the
        # context-parallel mesh and balancer. Global side effect on torch dynamo configuration.
        dims = self.parallel_dims
        self.loss_fn.set_gradient_scale(1.0 / (dims.dp_replicate * dims.dp_shard * dims.cp))
        if not dims.cp_enabled:
            return
        torch._dynamo.config.recompile_limit = max(
            torch._dynamo.config.recompile_limit, 2 * (self.config.training.seq_len // _CP_LENGTH_BUCKET + 1)
        )
        balancer_type = self.config.parallelism.context_parallel_load_balancer
        if balancer_type != "headtail":
            raise ValueError(
                f"context_parallel_load_balancer={balancer_type!r} is not supported yet: only a "
                "data-independent sharding can be undone from the sequence length"
            )
        self.loss_fn.set_context_parallel(self.parallel_dims.get_mesh("cp"), balancer_type)

    def has_last_stage(self) -> bool:
        # [FUNCTION DOC] True when pipeline parallelism is off, or this rank holds the last pipeline
        # stage (the rank that sees logits and the loss).
        return (not self.parallel_dims.pp_enabled) or self.pp_has_last_stage

    def step_runner(self) -> "TrainerStepRunner":
        # [FUNCTION DOC] Returns a new step-runner facade bound to this trainer.
        return TrainerStepRunner(self)


class TrainerStepRunner:
    # [CLASS DOC] Facade exposing the trainer through the step-runner interface the torch-native
    # actor expects: forward-only, forward-backward, zero gradients and apply step.
    def __init__(self, trainer: TitanTrainer):
        # [FUNCTION DOC] Stores the trainer.
        self.trainer = trainer

    def forward_only_step(self, batches, compute: Callable) -> list:
        # [FUNCTION DOC] Delegates to the trainer's forward-only pass and returns its results.
        return self.trainer.run_forward(batches, compute)

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        # [FUNCTION DOC] Delegates to the trainer's forward-backward pass and returns its results.
        return self.trainer.run_forward_backward(batches, loss_closure)

    def zero_grad(self) -> None:
        # [FUNCTION DOC] Zeroes optimizer gradients by setting them to none (frees memory).
        self.trainer.optimizers.zero_grad(set_to_none=True)

    def apply_step(self) -> StepMetrics:
        # [FUNCTION DOC] Delegates to the trainer's optimizer step and returns its metrics.
        return self.trainer.apply_optimizer_step()
