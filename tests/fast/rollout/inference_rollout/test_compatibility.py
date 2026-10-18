# [MODULE DOC] Tests of the compatibility layer that loads rollout functions and generate
# functions by registry name or path, whatever their format. Rollout formats: a legacy plain
# function (with raw or typed output, wrapped by an adapter), a synchronous class and an
# asynchronous class (both subclassing the base rollout class). Generate formats: a legacy
# asynchronous function with or without an evaluation parameter (wrapped by an adapter), a new
# asynchronous function taking a typed input, and a new class with an asynchronous call method.
# Functions are registered temporarily in the function registry under "test:" names. Also covered:
# the base class's default no-op save/load, custom save/load, independent instances per load,
# rejection of classes that are abstract or do not subclass the base, and that the shipped rollout
# classes pass the loader's class check (their constructors are patched out).
import asyncio
import inspect
from unittest.mock import MagicMock, patch

import pytest

from miles.rollout.base_types import (
    BaseRolloutFn,
    GenerateFnInput,
    GenerateFnOutput,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.inference_rollout.compatibility import (
    LegacyGenerateFnAdapter,
    LegacyRolloutFnAdapter,
    call_rollout_function,
    load_generate_function,
    load_rollout_function,
)
from miles.utils.async_utils import run
from miles.utils.function_registry import function_registry, load_function


@pytest.fixture
def constructor_input():
    # [FUNCTION DOC] HELPER: Fixture returning a rollout constructor input whose arguments and
    # data source are placeholder strings.
    return RolloutFnConstructorInput(args="dummy_args", data_source="dummy_data_source")


@pytest.fixture
def make_generate_fn_input():
    # [FUNCTION DOC] HELPER: Fixture returning a factory function for generate inputs.
    def _make(evaluation: bool = False):
        # [FUNCTION DOC] HELPER: Builds a generate input with a mock state (with mock arguments),
        # a plain mapping as the sample, sampling parameters with temperature 0.7, and the given
        # evaluation flag (default false).
        state = MagicMock()
        state.args = MagicMock()

        return GenerateFnInput(
            state=state,
            sample={"text": "test prompt"},
            sampling_params={"temperature": 0.7},
            evaluation=evaluation,
        )

    return _make


class TestSupportedRolloutFormats:
    # [CLASS DOC] Documentation-style tests showing each supported rollout function format and
    # how it is adapted when loaded and called with a train or eval input.
    """
    Documentation test to show various supported rollout function formats
    """

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_1_legacy_function_raw_output(self, constructor_input, evaluation):
        # [FUNCTION DOC] Registers a legacy function that returns a raw mapping of metrics in
        # eval mode and a raw nested list of samples in train mode. Loads it, calls it with an eval
        # or train input (rollout id 1), and asserts the loaded object is the legacy adapter and
        # the result is wrapped into the eval output with the mapping as its data, or the train
        # output with the nested list as its samples.
        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            # [FUNCTION DOC] HELPER: Legacy rollout function; returns a metrics mapping when the
            # evaluation flag is true, otherwise a nested list holding one sample mapping.
            if evaluation:
                return {"metric": {"accuracy": 0.9}}
            return [[{"text": "sample"}]]

        with function_registry.temporary("test:legacy_rollout", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_rollout")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, LegacyRolloutFnAdapter)
            if evaluation:
                assert isinstance(result, RolloutFnEvalOutput)
                assert result.data == {"metric": {"accuracy": 0.9}}
            else:
                assert isinstance(result, RolloutFnTrainOutput)
                assert result.samples == [[{"text": "sample"}]]

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_2_legacy_function_typed_output(self, constructor_input, evaluation):
        # [FUNCTION DOC] Same as the raw format test but the legacy function already returns the
        # typed eval or train output; asserts the typed result passes through with its data or
        # samples intact.
        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            # [FUNCTION DOC] HELPER: Legacy rollout function returning a typed eval output in eval
            # mode and a typed train output otherwise.
            if evaluation:
                return RolloutFnEvalOutput(data={"ds": {"acc": 0.95}})
            return RolloutFnTrainOutput(samples=[[{"text": "typed"}]])

        with function_registry.temporary("test:legacy_typed", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_typed")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            if evaluation:
                assert isinstance(result, RolloutFnEvalOutput)
                assert result.data == {"ds": {"acc": 0.95}}
            else:
                assert isinstance(result, RolloutFnTrainOutput)
                assert result.samples == [[{"text": "typed"}]]

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_3_sync_class(self, constructor_input, evaluation):
        # [FUNCTION DOC] Registers a synchronous class subclassing the base rollout class, loads
        # it, calls it with an eval or train input, and asserts the loaded object is an instance of
        # that class and the result type matches the mode (eval or train output).
        class SyncRolloutFn(BaseRolloutFn):
            # [CLASS DOC] HELPER: Synchronous rollout class returning a typed output per mode.
            def __init__(self, input: RolloutFnConstructorInput):
                # [FUNCTION DOC] HELPER: Accepts the constructor input and does nothing.
                pass

            def __call__(self, input):
                # [FUNCTION DOC] HELPER: Returns a typed eval output when the input is an
                # evaluation, otherwise a typed train output.
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"test": {"score": 1}})
                return RolloutFnTrainOutput(samples=[[{"text": "sync"}]])

        with function_registry.temporary("test:sync_class", SyncRolloutFn):
            fn = load_rollout_function(constructor_input, "test:sync_class")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, SyncRolloutFn)
            expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
            assert isinstance(result, expected_type)

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_async_class(self, constructor_input, evaluation):
        # [FUNCTION DOC] Same as the synchronous class test for a class whose call method is
        # asynchronous; the call helper resolves the coroutine, so the result is the typed output.
        class AsyncRolloutFn(BaseRolloutFn):
            # [CLASS DOC] HELPER: Asynchronous rollout class returning a typed output per mode.
            def __init__(self, input: RolloutFnConstructorInput):
                # [FUNCTION DOC] HELPER: Accepts the constructor input and does nothing.
                pass

            async def __call__(self, input):
                # [FUNCTION DOC] HELPER: Sleeps 1 millisecond, then returns a typed eval output
                # for an evaluation input, otherwise a typed train output.
                await asyncio.sleep(0.001)
                if input.evaluation:
                    return RolloutFnEvalOutput(data={"benchmark": {"accuracy": 0.98}})
                return RolloutFnTrainOutput(samples=[[{"text": "async"}]])

        with function_registry.temporary("test:async_class", AsyncRolloutFn):
            fn = load_rollout_function(constructor_input, "test:async_class")

            input_cls = RolloutFnEvalInput if evaluation else RolloutFnTrainInput
            result = call_rollout_function(fn, input_cls(rollout_id=1))

            assert isinstance(fn, AsyncRolloutFn)
            expected_type = RolloutFnEvalOutput if evaluation else RolloutFnTrainOutput
            assert isinstance(result, expected_type)


class TestRolloutFnCheckpointing:
    # [CLASS DOC] Tests of the save/load checkpoint hooks of rollout functions and of the loader's
    # class validation.
    def test_base_save_and_load_default_to_no_ops(self, constructor_input):
        # [FUNCTION DOC] Loads a stateless subclass defining only the call method and asserts its
        # inherited save and load return none.
        """A rollout function that keeps no state inherits save/load and needs no code."""

        class StatelessRolloutFn(BaseRolloutFn):
            # [CLASS DOC] HELPER: Rollout class with only a call method.
            def __call__(self, input):
                # [FUNCTION DOC] HELPER: Returns an empty train output.
                return RolloutFnTrainOutput(samples=[])

        with function_registry.temporary("test:stateless", StatelessRolloutFn):
            fn = load_rollout_function(constructor_input, "test:stateless")

            assert fn.save(1) is None
            assert fn.load(1) is None

    def test_legacy_function_gets_no_op_save_and_load(self, constructor_input):
        # [FUNCTION DOC] Loads a legacy function and asserts the adapter's save and load both
        # return none (called with a rollout id and with none).
        """Plain slime-style rollout functions stay usable: the adapter supplies the new methods."""

        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            # [FUNCTION DOC] HELPER: Legacy rollout function returning a nested list with one
            # sample mapping.
            return [[{"text": "sample"}]]

        with function_registry.temporary("test:legacy_checkpoint", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_checkpoint")

            assert isinstance(fn, LegacyRolloutFnAdapter)
            assert fn.save(1) is None
            assert fn.load(None) is None

    def test_legacy_adapter_exposes_the_constructor_input(self, constructor_input):
        # [FUNCTION DOC] Loads a legacy function and asserts the adapter's constructor input
        # attribute is the very object passed to the loader.
        """The base class promises constructor_input, so the framework's own subclasses must set it too."""

        def legacy_rollout_fn(args, rollout_id, data_source, evaluation=False):
            # [FUNCTION DOC] HELPER: Legacy rollout function returning a nested list with one
            # sample mapping.
            return [[{"text": "sample"}]]

        with function_registry.temporary("test:legacy_constructor_input", legacy_rollout_fn):
            fn = load_rollout_function(constructor_input, "test:legacy_constructor_input")

            assert fn.constructor_input is constructor_input

    def test_custom_save_and_load_are_used(self, constructor_input):
        # [FUNCTION DOC] Loads a stateful class that records its save and load calls, invokes both
        # with rollout id 4, and asserts the recorded calls are save then load with that id.
        """A stateful rollout function checkpoints through its own save/load."""

        class StatefulRolloutFn(BaseRolloutFn):
            # [CLASS DOC] HELPER: Rollout class recording save and load invocations in a list.
            def __init__(self, input: RolloutFnConstructorInput):
                # [FUNCTION DOC] HELPER: Starts with an empty call record.
                self.calls = []

            def __call__(self, input):
                # [FUNCTION DOC] HELPER: Returns an empty train output.
                return RolloutFnTrainOutput(samples=[])

            def save(self, rollout_id):
                # [FUNCTION DOC] HELPER: Records a save entry with the rollout id.
                self.calls.append(("save", rollout_id))

            def load(self, rollout_id):
                # [FUNCTION DOC] HELPER: Records a load entry with the rollout id.
                self.calls.append(("load", rollout_id))

        with function_registry.temporary("test:stateful", StatefulRolloutFn):
            fn = load_rollout_function(constructor_input, "test:stateful")
            fn.save(4)
            fn.load(4)

            assert fn.calls == [("save", 4), ("load", 4)]

    def test_same_path_loaded_twice_yields_independent_instances(self, constructor_input):
        # [FUNCTION DOC] Loads the same registered class twice, saves on the first with rollout id
        # 2, and asserts the two objects are different and the second's record is still empty.
        """--eval-function-path defaults to --rollout-function-path, and each load builds its own object."""

        class StatefulRolloutFn(BaseRolloutFn):
            # [CLASS DOC] HELPER: Rollout class recording rollout ids passed to save.
            def __init__(self, input: RolloutFnConstructorInput):
                # [FUNCTION DOC] HELPER: Starts with an empty call record.
                self.calls = []

            def __call__(self, input):
                # [FUNCTION DOC] HELPER: Returns an empty train output.
                return RolloutFnTrainOutput(samples=[])

            def save(self, rollout_id):
                # [FUNCTION DOC] HELPER: Records the rollout id.
                self.calls.append(rollout_id)

        with function_registry.temporary("test:same_path", StatefulRolloutFn):
            train_fn = load_rollout_function(constructor_input, "test:same_path")
            eval_fn = load_rollout_function(constructor_input, "test:same_path")

            train_fn.save(2)

            assert train_fn is not eval_fn
            assert eval_fn.calls == []

    def test_subclass_without_call_cannot_be_instantiated(self, constructor_input):
        # [FUNCTION DOC] Loading a subclass that does not implement the abstract call method
        # raises a type error mentioning "abstract".
        """__call__ is abstract, so a missing implementation fails at load, not mid-run."""

        class MissingCall(BaseRolloutFn):
            # [CLASS DOC] HELPER: Subclass with no call implementation (abstract).
            pass

        with function_registry.temporary("test:missing_call", MissingCall):
            with pytest.raises(TypeError, match="abstract"):
                load_rollout_function(constructor_input, "test:missing_call")

    def test_class_not_subclassing_base_is_rejected(self, constructor_input):
        # [FUNCTION DOC] Loading a class with a matching constructor and call method that does
        # not subclass the base rollout class raises a type error saying it must subclass.
        """Rejected at load time, not at the first checkpoint hours into a run."""

        class NotARolloutFn:
            # [CLASS DOC] HELPER: Look-alike class outside the base class hierarchy.
            def __init__(self, input: RolloutFnConstructorInput):
                # [FUNCTION DOC] HELPER: Accepts the constructor input and does nothing.
                pass

            def __call__(self, input):
                # [FUNCTION DOC] HELPER: Returns an empty train output.
                return RolloutFnTrainOutput(samples=[])

        with function_registry.temporary("test:not_a_rollout_fn", NotARolloutFn):
            with pytest.raises(TypeError, match="must subclass"):
                load_rollout_function(constructor_input, "test:not_a_rollout_fn")


class TestSupportedGenerateFormats:
    # [CLASS DOC] Documentation-style tests showing each supported generate function format and
    # how it is adapted when loaded and called with a generate input.
    """
    Documentation test similar to TestSupportedRolloutFormats
    """

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_1_legacy_function_with_evaluation_param(self, make_generate_fn_input, evaluation):
        # [FUNCTION DOC] Registers a legacy asynchronous function that accepts an evaluation
        # parameter and returns a string, loads it, runs the loaded callable on a generate input,
        # and asserts the loaded object is the legacy generate adapter and the result is a generate
        # output whose samples is that string.
        async def legacy_generate_fn(args, sample, sampling_params, evaluation=False):
            # [FUNCTION DOC] HELPER: Legacy generate function returning the string "my_sample".
            return "my_sample"

        with function_registry.temporary("test:legacy_gen_eval", legacy_generate_fn):
            fn = load_generate_function("test:legacy_gen_eval")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, LegacyGenerateFnAdapter)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_2_legacy_function_without_evaluation_param(self, make_generate_fn_input, evaluation):
        # [FUNCTION DOC] Same as the previous format for a legacy function without the evaluation
        # parameter; the adapter must not pass it. Asserts the adapter type and the generate
        # output wrapping the string.
        async def legacy_generate_fn(args, sample, sampling_params):
            # [FUNCTION DOC] HELPER: Legacy generate function returning the string "my_sample".
            return "my_sample"

        with function_registry.temporary("test:legacy_gen", legacy_generate_fn):
            fn = load_generate_function("test:legacy_gen")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, LegacyGenerateFnAdapter)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_3_new_async_function_api(self, make_generate_fn_input, evaluation):
        # [FUNCTION DOC] Registers a new-style asynchronous function taking a generate input and
        # returning a generate output; asserts running the loaded callable returns a generate
        # output with the same samples.
        async def generate(input: GenerateFnInput) -> GenerateFnOutput:
            # [FUNCTION DOC] HELPER: Returns a generate output whose samples is "my_sample".
            return GenerateFnOutput(samples="my_sample")

        with function_registry.temporary("test:new_async", generate):
            fn = load_generate_function("test:new_async")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"

    @pytest.mark.parametrize("evaluation", [False, True])
    def test_format_4_new_class_api(self, make_generate_fn_input, evaluation):
        # [FUNCTION DOC] Registers a new-style class with an asynchronous call method; asserts the
        # loaded object is an instance of the class and running it returns a generate output with
        # the same samples.
        class MyGenerateFn:
            # [CLASS DOC] HELPER: Generate class with an asynchronous call method.
            async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
                # [FUNCTION DOC] HELPER: Returns a generate output whose samples is "my_sample".
                return GenerateFnOutput(samples="my_sample")

        with function_registry.temporary("test:new_class", MyGenerateFn):
            fn = load_generate_function("test:new_class")

            result = run(fn(make_generate_fn_input(evaluation)))

            assert isinstance(fn, MyGenerateFn)
            assert isinstance(result, GenerateFnOutput)
            assert result.samples == "my_sample"


class TestShippedRolloutFunctions:
    # [CLASS DOC] Tests that rollout classes shipped in the repository satisfy the loader's check.
    @pytest.mark.parametrize(
        "path",
        [
            "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn",
            "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn",
            "examples.infra_features.fully_async.external_eval_fn.ExternalSglangEvalFn",
        ],
    )
    def test_a_shipped_rollout_function_passes_the_loader_class_check(
        self, path: str, constructor_input: RolloutFnConstructorInput
    ) -> None:
        # [FUNCTION DOC] Parametrized over the shipped inference rollout class, the fully
        # asynchronous rollout class and an example external evaluation class. Imports the class
        # by path and asserts it is a class; then loads it through the loader with its constructor
        # patched to do nothing (so no heavy setup runs) and asserts the loaded object is an
        # instance of the base rollout class.
        """load_rollout_function rejects a class outside the hierarchy, and it runs inside the
        rollout actor, so a miles-shipped path failing it kills the run at startup."""
        loaded = load_function(path)
        assert inspect.isclass(loaded)

        with patch.object(loaded, "__init__", return_value=None):
            fn = load_rollout_function(constructor_input, path)

        assert isinstance(fn, BaseRolloutFn)
