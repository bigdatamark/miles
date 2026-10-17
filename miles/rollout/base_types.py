# [MODULE DOC] Shared types and conventions for rollout functions: immutable input records (for
# construction, training and evaluation), mutable output records (training samples as groups per
# prompt, evaluation data per dataset), the abstract base class of new-style rollout functions
# (callable with save/load/dispose hooks), the input/output records of per-sample generate functions,
# and the legacy call adapter that normalizes old-style return values.
from __future__ import annotations

import abc
from argparse import Namespace
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miles.rollout.data_source import DataSource
from miles.utils.types import Sample

if TYPE_CHECKING:
    from miles.rollout.inference_rollout.inference_rollout_common import GenerateState


@dataclass(frozen=True)
class RolloutFnConstructorInput:
    # [CLASS DOC] Frozen record given to a rollout function's constructor: the launch args and the data
    # source from which prompts are drawn (and, for buffering rollouts, to which samples are returned).
    args: Namespace
    # TODO may refactor DataSource API
    data_source: DataSource


@dataclass(frozen=True)
class RolloutFnBaseInput:
    # [CLASS DOC] Frozen base of per-call inputs: the integer id of the rollout step being produced.
    rollout_id: int

    @property
    def evaluation(self):
        # [FUNCTION DOC] Property: whether this call is an evaluation. Abstract here (raises
        # not-implemented error); subclasses return a constant.
        raise NotImplementedError


# subclassing for different data in the future
@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    # [CLASS DOC] Input of a training rollout call: adds the integer weight version the engines serve
    # (none before the first weight update) and the id of the requesting trainer model (none for a
    # single policy).
    # engine weight version, None before the first weight update
    weight_version: int | None = None
    # which policy model asked for this rollout; None when the run trains one policy
    trainer_model_id: str | None = None

    @property
    def evaluation(self):
        # [FUNCTION DOC] Property: always false for training input.
        return False


@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    # [CLASS DOC] Input of an evaluation call: optionally a generation state (engine client, semaphore,
    # sizing) already pointed at the engines to evaluate; the weight version as text (snapshot
    # evaluation); and the directory of the Hugging Face snapshot being evaluated (none for evaluating
    # the live training engines).
    generate_state: GenerateState | None = None
    weight_version: str | None = None
    hf_dir: str | None = None

    @property
    def evaluation(self):
        # [FUNCTION DOC] Property: always true for evaluation input.
        return True


# TODO make it frozen
@dataclass
class RolloutFnTrainOutput:
    # [CLASS DOC] Result of a training rollout: a list of sample groups (one list of samples per
    # prompt, possibly nested deeper for compact rollouts) and an optional mapping of extra metrics to
    # log. KNOWN ISSUE: the metrics annotation says mapping but the default is none.
    samples: list[list[Sample]]
    metrics: dict[str, Any] = None


# TODO make it frozen
@dataclass
class RolloutFnEvalOutput:
    # [CLASS DOC] Result of an evaluation: a mapping from dataset name to a mapping with that dataset's
    # results (rewards list, optional samples and truncated flags) and an optional mapping of extra
    # metrics. KNOWN ISSUE: the metrics annotation says mapping but the default is none.
    data: dict[str, dict[str, Any]]
    metrics: dict[str, Any] = None


RolloutFnInput = RolloutFnTrainInput | RolloutFnEvalInput
RolloutFnOutput = RolloutFnTrainOutput | RolloutFnEvalOutput


class BaseRolloutFn(abc.ABC):
    # [CLASS DOC] Abstract base of new-style rollout functions (train or eval). Keeps the constructor
    # input. Subclasses implement the call; checkpoint, restore and teardown hooks are optional no-ops.
    # The call may be synchronous or (as the checkpoint-eval subclass does) asynchronous; callers
    # use a compatibility wrapper.

    def __init__(self, input: RolloutFnConstructorInput) -> None:
        # [FUNCTION DOC] Store the constructor input as the constructor input attribute.
        self.constructor_input = input

    @abc.abstractmethod
    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        # [FUNCTION DOC] Abstract: produce the output for a training or evaluation input. Base raises
        # not-implemented error.
        raise NotImplementedError

    def save(self, rollout_id: int) -> None:
        # [FUNCTION DOC] Hook: persist any internal state for the rollout id. Default does nothing.
        return None

    def load(self, rollout_id: int | None) -> None:
        # [FUNCTION DOC] Hook: restore internal state for the rollout id (none = default/latest).
        # Default does nothing.
        return None

    async def dispose(self) -> None:
        # [FUNCTION DOC] Hook: release resources; asynchronous. Default does nothing.
        return None


@dataclass(frozen=True)
class GenerateFnInput:
    # [CLASS DOC] Frozen input of a per-sample generate function: the shared generation state, the
    # sample to fill in, the sampling parameters mapping for this call, and whether it is an
    # evaluation.
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

    @property
    def args(self) -> Namespace:
        # [FUNCTION DOC] Property: the launch args held by the generation state.
        return self.state.args


@dataclass(frozen=True)
class GenerateFnOutput:
    # [CLASS DOC] Frozen output of a per-sample generate function: a single sample or a list of samples
    # (one generate can yield several).
    # One generate may lead to multiple samples, such as multi-agent, tree-like exploration, or
    # multi-turn with removing thinking tokens.
    samples: Sample | list[Sample]


def call_rollout_fn(fn, *args, evaluation: bool, **kwargs):
    # [FUNCTION DOC] Call a legacy rollout function with the given positional arguments, keyword
    # arguments and an added evaluation keyword. If it returns something that is not already an output
    # record, wraps it: as an evaluation output (the value is the per-dataset data mapping) when
    # evaluating, else as a training output (the value is the sample groups). Returns the output record.
    """Legacy rollout function call interface. Used when MILES_USE_LEGACY_ROLLOUT_V1 is enabled."""
    output = fn(*args, **kwargs, evaluation=evaluation)

    # compatibility for legacy version
    if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
        output = RolloutFnEvalOutput(data=output) if evaluation else RolloutFnTrainOutput(samples=output)

    return output
