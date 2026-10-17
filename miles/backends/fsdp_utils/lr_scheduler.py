# [MODULE DOC]
# Learning-rate scheduling for the FSDP training backend. Provides a scheduler class derived from
# the framework's base scheduler that supports warmup followed by constant, linear, cosine,
# inverse-square-root or warmup-stable-decay (WSD) decay, plus a factory that derives the schedule
# lengths from the run arguments (rollout count, batch sizes, lr_* flags). The decay formulas mirror
# the Megatron-style scheduler so FSDP and Megatron runs follow the same curves.

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Learning rate scheduler for FSDP training."""

import logging
import math

import torch
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override

logger = logging.getLogger(__name__)


class FSDPLRScheduler(LRScheduler):
    # [CLASS DOC]
    # Stepwise learning-rate scheduler whose step counter is the base class's last_epoch (the number
    # of scheduler steps taken). The learning rate for each optimizer parameter group is computed
    # from the step as: linear warmup from init_lr to max_lr over lr_warmup_steps, then (after
    # warmup) the chosen decay style down to min_lr at lr_decay_steps, and min_lr forever after.
    # A parameter group may override the bounds through optional "max_lr" and "min_lr" entries.
    # Supported decay styles: "constant", "inverse-square-root", "linear", "cosine", and "WSD"
    # (hold at max until the last wsd_decay_steps steps, then anneal using lr_wsd_decay_style, one of
    # linear, cosine, exponential, minus_sqrt).
    """Learning rate scheduler for FSDP training.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        init_lr (float): Initial learning rate.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        lr_warmup_steps (int): Number of warmup steps.
        lr_decay_steps (int): Number of decay steps.
        lr_decay_style (str): Decay style for learning rate.
        use_checkpoint_lr_scheduler (bool, optional): Whether to use the checkpoint values
            for the lr scheduler.
        override_lr_scheduler (bool, optional): Whether to override the lr scheduler values
            with the class values.
        wsd_decay_steps (int, optional): Number of weight decay decay steps.
        lr_wsd_decay_style (str, optional): Decay style for learning rate during weight decay decay
            steps.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: int,
        lr_decay_steps: int,
        lr_decay_style: str,
        use_checkpoint_lr_scheduler: bool | None = True,
        override_lr_scheduler: bool | None = False,
        wsd_decay_steps: int | None = None,
        lr_wsd_decay_style: str | None = None,
        last_epoch: int = -1,
    ) -> None:
        # [FUNCTION DOC]
        # Validates and stores the schedule parameters, then initializes the base scheduler (which
        # takes the initial step and so sets the optimizer's starting learning rates) and logs the
        # decay style. Inputs are the optimizer plus the lr bounds, step counts and style names
        # described in the class docstring; max_lr is coerced to a float.
        # Raises an assertion error when: min_lr is negative, max_lr is below min_lr, init_lr is
        # above max_lr, lr_decay_steps is not positive, lr_warmup_steps is not strictly below
        # lr_decay_steps, the style is WSD without wsd_decay_steps, or both override and
        # use-checkpoint flags are set.
        # KNOWN ISSUE: use_checkpoint_lr_scheduler and override_lr_scheduler are only stored and
        # cross-checked here; nothing in this class reads them afterwards (no state-dict logic).
        # Store our custom parameters
        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style

        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler

        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, "both override and use-checkpoint are set."

        # Initialize parent class
        super().__init__(optimizer, last_epoch)

        logger.info(f"> learning rate decay style: {self.lr_decay_style}")

    def _get_lr_for_group(self, param_group: dict) -> float:
        # [FUNCTION DOC] HELPER: Computes the learning rate for one parameter group at the current
        # step (last_epoch). Bounds come from the group's optional max_lr / min_lr entries, falling
        # back to the scheduler's values. Order of evaluation:
        #   1. If warmup is enabled and step <= warmup steps: linear interpolation from init_lr to
        #      max_lr (so the step equal to warmup steps yields exactly max_lr).
        #   2. Constant style: max_lr.
        #   3. Step beyond decay steps: min_lr.
        #   4. Inverse-square-root: max_lr scaled by sqrt(max(warmup,1)/max(step,1)), floored at min_lr.
        #   5. Otherwise decay_ratio = (step - warmup) / (decay_steps - warmup) in [0, 1], and the
        #      result is min_lr + coeff * (max_lr - min_lr), with coeff = 1 - ratio (linear),
        #      half-cosine from 1 to 0 (cosine), or for WSD: 1 until step passes
        #      (decay_steps - wsd_decay_steps), then annealed over the wsd_decay_steps with linear
        #      (1 - r), cosine, exponential (2 * 0.5^r - 1) or minus_sqrt (1 - sqrt(r)).
        # Raises a generic exception for an unknown decay style; an assertion error when the ratio is
        # outside [0, 1] or when the WSD annealing style is unrecognized (coefficient left unset).
        # Pure: reads state only, returns a float.
        """Compute learning rate for a specific parameter group.

        Args:
            param_group (dict): parameter group from the optimizer.

        Returns:
            float: learning rate for this parameter group.
        """
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.last_epoch <= self.lr_warmup_steps:
            return self.init_lr + ((max_lr - self.init_lr) * float(self.last_epoch) / float(self.lr_warmup_steps))

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == "constant":
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.last_epoch > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.last_epoch, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.last_epoch - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0

        delta_lr = max_lr - min_lr
        coeff = None

        if self.lr_decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == "WSD":
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.last_epoch <= wsd_anneal_start_:
                coeff = 1.0
            else:
                wsd_steps = self.last_epoch - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                if self.lr_wsd_decay_style == "linear":
                    coeff = 1.0 - wsd_decay_ratio
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
                elif self.lr_wsd_decay_style == "minus_sqrt":
                    coeff = 1.0 - math.sqrt(wsd_decay_ratio)
        else:
            raise Exception(f"{self.lr_decay_style} decay style is not supported.")

        assert coeff is not None
        return min_lr + coeff * delta_lr

    @override
    def get_lr(self) -> list[float]:
        # [FUNCTION DOC]
        # Override of the base-class hook invoked on each scheduler step: returns one learning rate
        # per optimizer parameter group, in group order, computed by the per-group helper.
        """Compute the learning rates for each parameter group.

        Returns:
            list[float]: A list of learning rates, one for each parameter group.
        """
        return [self._get_lr_for_group(group) for group in self.optimizer.param_groups]


def get_lr_scheduler(args, optimizer: torch.optim.Optimizer) -> FSDPLRScheduler:
    # [FUNCTION DOC]
    # Factory building the scheduler from the run arguments. Side effects: sets train_iters on the
    # args object (num_rollout * rollout_batch_size * n_samples_per_prompt // global_batch_size) and,
    # when lr_decay_iters is unset, sets it to train_iters. Decay steps = lr_decay_iters; WSD decay
    # steps = lr_wsd_decay_iters when given (else none); warmup steps = lr_warmup_fraction times the
    # decay steps when the fraction is given (may be a non-integer value), otherwise lr_warmup_iters.
    # Other fields map directly: lr_warmup_init -> init_lr, lr -> max_lr, min_lr, lr_decay_style,
    # use_checkpoint_lr_scheduler, override_lr_scheduler, lr_wsd_decay_style. Returns the scheduler.
    """Create and configure the learning-rate scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args: Training/runtime arguments (namespace).
        optimizer (torch.optim.Optimizer): Optimizer bound to the model.

    Returns:
        FSDPLRScheduler: Initialized scheduler bound to ``optimizer``.
    """
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters
    lr_scheduler = FSDPLRScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return lr_scheduler
