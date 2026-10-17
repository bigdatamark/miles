# [MODULE DOC]
# Compatibility layer between legacy function-style rollout / generate callables and the newer
# class-based interfaces. Provides adapters that wrap legacy callables and loaders that resolve a
# dotted path into either a class-based implementation or an adapted legacy function.
import inspect
from collections.abc import Callable

from miles.rollout.base_types import (
    BaseRolloutFn,
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainOutput,
)
from miles.utils.async_utils import run
from miles.utils.function_registry import load_function


class LegacyRolloutFnAdapter(BaseRolloutFn):
    # [CLASS DOC]
    # Wraps a legacy rollout function (taking args, rollout id, data source, and an evaluation
    # keyword) as a class-based rollout function, normalizing its return value into the typed
    # train or eval output objects.
    def __init__(self, input: RolloutFnConstructorInput, fn: Callable):
        # [FUNCTION DOC] Initialize the base class with the constructor input, then store the
        # launch args and data source from it and the wrapped legacy callable.
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.fn = fn

    def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        # [FUNCTION DOC]
        # Call the legacy function with launch args, the rollout id, the data source and the
        # evaluation flag as a keyword. If the result is already a typed train or eval output it
        # is returned as is (it may be a coroutine if the legacy function is async; no awaiting
        # happens here). Otherwise the raw result is wrapped: as an eval output (data field) when
        # evaluating, else as a train output (samples field).
        output = self.fn(self.args, input.rollout_id, self.data_source, evaluation=input.evaluation)

        # compatibility for legacy version
        if not isinstance(output, (RolloutFnTrainOutput, RolloutFnEvalOutput)):
            output = RolloutFnEvalOutput(data=output) if input.evaluation else RolloutFnTrainOutput(samples=output)

        return output


def load_rollout_function(input: RolloutFnConstructorInput, path: str) -> BaseRolloutFn:
    # [FUNCTION DOC]
    # Resolve a dotted path to an object via the function registry. If it is a class it must be a
    # subclass of the base rollout function class (else a type error naming the path) and is
    # instantiated with the constructor input. Otherwise it is treated as a legacy function and
    # wrapped in the legacy adapter. Returns the rollout function object.
    fn = load_function(path)

    if inspect.isclass(fn):
        if not issubclass(fn, BaseRolloutFn):
            raise TypeError(
                f"Class-based rollout function {path} must subclass "
                f"miles.rollout.base_types.BaseRolloutFn, got {fn.__name__}."
            )
        return fn(input)
    else:
        return LegacyRolloutFnAdapter(input, fn)


def call_rollout_function(fn, input: RolloutFnInput) -> RolloutFnOutput:
    # [FUNCTION DOC]
    # Call a rollout function with the input; if the result is a coroutine, run it to completion
    # synchronously with the shared async runner. Returns the rollout output.
    output = fn(input)

    if inspect.iscoroutine(output):
        output = run(output)

    return output


class LegacyGenerateFnAdapter:
    # [CLASS DOC]
    # Adapts a legacy async generate function (args, sample, sampling params, optional evaluation
    # keyword) to the single-input generate interface returning a typed generate output.
    def __init__(self, fn: Callable):
        # [FUNCTION DOC] Store the callable and record once whether its signature has a parameter
        # named evaluation.
        self.fn = fn
        self._has_evaluation_param = "evaluation" in inspect.signature(fn).parameters

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        # [FUNCTION DOC]
        # Await the legacy function with args, sample and sampling params taken from the input,
        # passing the evaluation flag as a keyword only if the function accepts it. A result that
        # is not already a typed generate output is wrapped as the samples of one.
        if self._has_evaluation_param:
            output = await self.fn(input.args, input.sample, input.sampling_params, evaluation=input.evaluation)
        else:
            output = await self.fn(input.args, input.sample, input.sampling_params)

        if not isinstance(output, GenerateFnOutput):
            output = GenerateFnOutput(samples=output)

        return output


def load_generate_function(path: str):
    # [FUNCTION DOC]
    # Resolve a dotted path to a generate function via the registry. Returns None if nothing was
    # loaded; a class is instantiated with no arguments; a legacy-style function is wrapped in the
    # legacy adapter; any other callable is returned unchanged.
    fn = load_function(path)
    if fn is None:
        return None

    if inspect.isclass(fn):
        return fn()
    elif _is_legacy_generate_fn(fn):
        return LegacyGenerateFnAdapter(fn)
    else:
        return fn


def _is_legacy_generate_fn(fn: Callable) -> bool:
    # [FUNCTION DOC] HELPER: Heuristic: a generate function is legacy if it has at least three
    # parameters and its first parameter is not named input.
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    return len(params) >= 3 and params[0] != "input"
