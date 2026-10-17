# [MODULE DOC] Extension point for evaluating on a user-owned backend from HF checkpoint snapshots.
# Defines: a helper that copies run args retargeted to another router and GPU sizing (used for the
# dedicated eval fleet), an exception that skips one eval point with an attributable reason, an
# abstract rollout-function base whose call validates the input and delegates to an abstract
# per-checkpoint evaluation, and a predicate to detect such a function from its configured path.
"""Run eval on a backend of your own.

Subclass ``CheckpointEvalFn`` and point ``--eval-function-path`` at it. Every eval
point hands you a finished HF checkpoint directory and takes back results; what
happens in between is yours — an sglang server you launched, a remote scoring
service, a black box with an HTTP API.

The trainer exports the snapshot, dispatches it without blocking the training loop,
logs the result at the snapshot's step, and reclaims the directory afterwards. Raise
``EvalSkip(reason)`` to skip a point with attribution instead of counting as a crash.

Requires ``train_async.py`` and a snapshot source (``--eval-hf-dir`` or ``--save-hf``).
``examples/infra_features/fully_async/external_eval_fn.py`` is a working implementation.
"""

import abc
import copy
import inspect
import logging
from argparse import Namespace

from miles.rollout.base_types import BaseRolloutFn, RolloutFnEvalInput, RolloutFnEvalOutput, RolloutFnInput
from miles.utils.function_registry import load_function

__all__ = [
    "retarget_args",
    "EvalSkip",
    "CheckpointEvalFn",
    "is_checkpoint_eval_fn",
]

logger = logging.getLogger(__name__)


def retarget_args(args: Namespace, router_ip, router_port, num_gpus: int, num_gpus_per_engine: int) -> Namespace:
    # [FUNCTION DOC] Return a shallow copy of args (the original is not modified; nested objects are
    # shared) with four fields replaced: the router IP and port, the total rollout GPU count, and the
    # rollout GPUs per engine.
    """Shallow-copy ``args`` with the router address and GPU sizing swapped for eval.

    Generate functions read the router from ``args`` and ``GenerateState`` sizes its
    semaphore off the GPU counts, so a retargeted copy runs the standard eval path
    against a different set of engines unchanged.
    """
    eval_args = copy.copy(args)
    eval_args.sglang_router_ip = router_ip
    eval_args.sglang_router_port = router_port
    eval_args.rollout_num_gpus = num_gpus
    eval_args.rollout_num_gpus_per_engine = num_gpus_per_engine
    return eval_args


class EvalSkip(Exception):
    # [CLASS DOC] Exception signalling that this eval point should be skipped, carrying a short reason
    # string that becomes the skip metric name suffix.
    """Raise from a ``CheckpointEvalFn`` to skip this eval point with an attributable
    reason (logged as ``eval/skipped_{reason}``) instead of counting as a crash."""

    def __init__(self, reason: str):
        # [FUNCTION DOC] Store the reason as the exception message and as the reason attribute.
        super().__init__(reason)
        self.reason = reason


class CheckpointEvalFn(BaseRolloutFn, abc.ABC):
    # [CLASS DOC] Abstract eval-only rollout function. Subclasses implement the per-checkpoint
    # evaluation; the base call checks preconditions and delegates; dispose is an optional no-op hook.
    """Contract for eval backends that consume HF checkpoint snapshots.

    ``__init__`` takes a ``RolloutFnConstructorInput`` and prepares the backend —
    launch a server, attach to one, open a client. ``evaluate_checkpoint`` then runs
    one eval point against the snapshot at ``checkpoint_dir``.
    """

    @abc.abstractmethod
    async def evaluate_checkpoint(self, checkpoint_dir: str, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        # [FUNCTION DOC] Abstract: evaluate the finished HF snapshot in the directory using the eval
        # input (rollout id, weight version, optional generation state) and return the evaluation
        # output (per-dataset data plus optional metrics). May raise the skip exception to skip.
        ...

    async def __call__(self, input: RolloutFnInput) -> RolloutFnEvalOutput:
        # [FUNCTION DOC] Entry called by the rollout executor. Asserts the input is an evaluation input
        # (the train function must stay on its own path) and that a snapshot directory was dispatched
        # (assertion errors explain the snapshot requirements); then awaits and returns the
        # per-checkpoint evaluation for the directory.
        assert input.evaluation, "CheckpointEvalFn only serves eval; keep the train fn on --rollout-function-path"
        assert input.hf_dir is not None, (
            "no snapshot was dispatched — checkpoint eval fns require train_async.py "
            "and a snapshot source (--eval-hf-dir or --save-hf)"
        )
        return await self.evaluate_checkpoint(input.hf_dir, input)

    async def dispose(self) -> None:  # noqa: B027 — optional hook, deliberately a no-op default
        # [FUNCTION DOC] Optional hook to release resources created in the constructor (servers,
        # clients). Default does nothing.
        """Tear down anything launched in ``__init__``. Called by RolloutManager.dispose()."""


def is_checkpoint_eval_fn(eval_function_path: str | None) -> bool:
    # [FUNCTION DOC] Load the object at the configured eval function path (side effect: imports it) and
    # return true iff it is a class deriving from the checkpoint eval base. Plain functions and
    # instances give false. A missing path (none) is passed straight to the loader, whose behavior for
    # none is not handled here.
    """Whether ``--eval-function-path`` points at a black-box checkpoint backend."""
    eval_fn = load_function(eval_function_path)
    return inspect.isclass(eval_fn) and issubclass(eval_fn, CheckpointEvalFn)
