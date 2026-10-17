# [MODULE DOC]
# Step runner abstraction for the torch-native actor: a protocol with four operations (forward-only
# over batches, forward+backward over batches with a loss closure, zero gradients, apply optimizer
# step returning metrics), a metrics record, and a simple implementation for models that run one
# forward per batch (no pipeline schedule).
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import torch


@dataclass
class StepMetrics:
    # [CLASS DOC] Result of an optimizer step: the global gradient norm (before clipping) and a
    # mapping of additional float metrics (default empty), such as learning rates.
    grad_norm: float
    extra_metrics: dict[str, float] = field(default_factory=dict)


class StepRunner(Protocol):
    # [CLASS DOC] Structural interface implemented by backend runners; method contracts below.
    def forward_only_step(self, batches: Iterator[dict], compute: Callable[[torch.Tensor, dict], dict]) -> list:
        # [FUNCTION DOC] Contract: run the model forward on each batch (no gradient required),
        # apply compute(logits, batch) and return the list of its result mappings in batch order.
        ...

    def forward_backward_step(
        self, batches: Iterator[dict], loss_closure: Callable[[torch.Tensor, dict], tuple[torch.Tensor, dict]]
    ) -> list[dict]:
        # [FUNCTION DOC] Contract: for each batch run forward, call loss_closure(logits, batch) to
        # get (loss, log mapping), run backward on the loss (accumulating gradients) and return the
        # list of log mappings.
        ...

    def zero_grad(self) -> None:
        # [FUNCTION DOC] Contract: clear accumulated gradients.
        ...

    def apply_step(self) -> StepMetrics:
        # [FUNCTION DOC] Contract: perform the optimizer step (including clipping and scheduler as
        # the backend defines) and return the step metrics.
        ...


def _forward_only() -> None:
    # [FUNCTION DOC] HELPER: Default for the zero-grad and step callables of a runner built for
    # forward passes only; always raises a runtime error saying so.
    raise RuntimeError("this runner was built for forward passes only")


class LinearStepRunner:
    # [CLASS DOC] Step runner for non-pipelined models: one forward callable maps a batch to logits;
    # gradient zeroing and the optimizer step are injected callables. Gradients accumulate across
    # batches via repeated backward calls.
    def __init__(
        self,
        forward_fn: Callable[[dict], torch.Tensor],
        zero_grad_fn: Callable[[], None] = _forward_only,
        step_fn: Callable[[], StepMetrics] = _forward_only,
    ):
        # [FUNCTION DOC] Store the three callables. The defaults for zero-grad and step raise a
        # runtime error when called (for forward-only runners, e.g. reference models).
        self._forward = forward_fn
        self._zero_grad = zero_grad_fn
        self._step = step_fn

    def forward_only_step(self, batches, compute: Callable) -> list:
        # [FUNCTION DOC] For each batch: forward, then compute(logits, batch); returns the list of
        # results. Does not itself disable autograd (the caller is expected to).
        return [compute(self._forward(batch), batch) for batch in batches]

    def forward_backward_step(self, batches, loss_closure: Callable) -> list[dict]:
        # [FUNCTION DOC] For each batch: forward, evaluate the loss closure to get (loss, log
        # mapping), run backward on the loss (accumulating gradients) and collect the log mapping.
        # Returns the list of log mappings in batch order.
        logs = []
        for batch in batches:
            loss, log_dict = loss_closure(self._forward(batch), batch)
            loss.backward()
            logs.append(log_dict)
        return logs

    def zero_grad(self) -> None:
        # [FUNCTION DOC] Calls the injected zero-grad callable.
        self._zero_grad()

    def apply_step(self) -> StepMetrics:
        # [FUNCTION DOC] Calls the injected step callable and returns its metrics.
        return self._step()
