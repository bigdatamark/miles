# [MODULE DOC] Unit tests for rpc method metadata collection (collecting rpc method specs from a
# worker class, the rpc decorator, per-method serializers, positional parameter names, argument
# canonicalization and handling of the trace context parameter ray injects). Worker classes are
# defined inline per test; the tests check which members are collected, how concurrency groups are
# discovered through decorator chains, query/result encoding and validation, and the explicit
# type errors raised at collection time for malformed method signatures.
import functools
import inspect
from collections.abc import Callable
from typing import Annotated, Any

import pytest
from pydantic import Field, ValidationError

from tests.fast.utils.workers.rpc.common.postponed_annotation_worker import LatePayload, PostponedWorker

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import (
    DEFAULT_CONCURRENCY_GROUP,
    RpcMethodSpec,
    canonicalize_method_arguments,
    collect_rpc_method_specs,
    rpc,
)


class _Payload(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict pydantic model with a required text field and an integer count
    # defaulting to one; used as parameter and return type of test worker methods.
    text: str
    count: int = 1


def _passthrough(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Decorator that wraps a function in a forwarding wrapper which copies the
    # wrapped function's metadata (so the wrapped function stays discoverable through the wrapped
    # attribute and copied attributes). Returns the wrapper.
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Calls the wrapped function with all arguments and returns its result.
        return fn(*args, **kwargs)

    return wrapper


def _inject_ray_trace_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Decorator that imitates ray's rewrite of actor method signatures: sets an
    # explicit signature on the same function object, equal to the original plus an extra keyword-only
    # parameter named as ray's trace context with default nothing and no annotation. Mutates and
    # returns the same function.
    signature = inspect.signature(fn)
    fn.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter("_ray_trace_ctx", inspect.Parameter.KEYWORD_ONLY, default=None),
        ]
    )
    return fn


def _opaque_passthrough(fn: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] HELPER: Decorator that wraps a function in a forwarding wrapper which copies
    # nothing from it (no name or attribute copying) but sets only the wrapped link to the original
    # function. Used to check that markers are found by walking the wrapped link. Returns the wrapper.
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] HELPER: Calls the wrapped function with all arguments and returns its result.
        return fn(*args, **kwargs)

    wrapper.__wrapped__ = fn
    return wrapper


class _GoodWorker:
    # [CLASS DOC] HELPER: Worker class with a mix of members: three collectible public instance methods
    # (a sync one with a default argument, an async one taking and returning the payload model, and a
    # sync one in the "heavy" concurrency group returning nothing), plus members that must be skipped:
    # a class attribute, a classmethod, a staticmethod, a property and an underscore-prefixed method.
    demo_class_attribute = 3

    def demo_default_arg(self, a: int, b: int = 10) -> int:
        # [FUNCTION DOC] HELPER: Returns the sum of a and b.
        return a + b

    async def demo_async_model(self, payload: _Payload) -> _Payload:
        # [FUNCTION DOC] HELPER: Returns the given payload unchanged.
        return payload

    @rpc(concurrency_group="heavy")
    def demo_grouped(self, step: int) -> None:
        # [FUNCTION DOC] HELPER: Does nothing; exists to carry the heavy concurrency group.
        pass

    @classmethod
    def demo_classmethod(cls, x: int) -> int:
        # [FUNCTION DOC] HELPER: Returns x; a class method that must not be collected.
        return x

    @staticmethod
    def demo_staticmethod(x: int) -> int:
        # [FUNCTION DOC] HELPER: Returns x; a static method that must not be collected.
        return x

    @property
    def demo_property(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the constant one; a property that must not be collected.
        return 1

    def _demo_private(self, x):
        # [FUNCTION DOC] HELPER: Does nothing; an underscore-prefixed method (also unannotated) that
        # must be skipped without error.
        pass


class TestCollectSpecs:
    # [CLASS DOC] Tests which members of the good worker are collected and the basic per-spec flags.
    def test_mutating_collected_specs_does_not_poison_the_cache(self) -> None:
        # [FUNCTION DOC] Collects specs, clears the returned mapping, collects again and asserts the
        # second mapping still holds the three public methods (callers get an independent copy).
        """Mutating one collected mapping leaves later collections complete."""
        specs = collect_rpc_method_specs(_GoodWorker)
        specs.clear()

        collected_again = collect_rpc_method_specs(_GoodWorker)
        assert set(collected_again) == {"demo_default_arg", "demo_async_model", "demo_grouped"}

    def test_collects_public_methods_only(self):
        # [FUNCTION DOC] Asserts the collected names are exactly the three public instance methods.
        """Public methods are collected; underscore-prefixed ones are skipped."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert set(specs) == {"demo_default_arg", "demo_async_model", "demo_grouped"}

    def test_non_instance_method_members_are_skipped(self):
        # [FUNCTION DOC] Asserts that the class method, static method, property and class attribute
        # names are absent from the collected specs.
        """Classmethods, staticmethods and properties are skipped like plain attributes."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert {"demo_classmethod", "demo_staticmethod", "demo_property", "demo_class_attribute"}.isdisjoint(specs)

    def test_default_concurrency_group(self):
        # [FUNCTION DOC] Asserts an undecorated method's spec has the default concurrency group.
        """Undecorated methods fall into the default concurrency group."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].concurrency_group == DEFAULT_CONCURRENCY_GROUP

    def test_decorated_concurrency_group(self):
        # [FUNCTION DOC] Asserts the decorated method's spec has the group name heavy.
        """@rpc(concurrency_group=...) is picked up by introspection."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_grouped"].concurrency_group == "heavy"

    def test_is_async_flag(self):
        # [FUNCTION DOC] Asserts the async method's spec is flagged async and the sync one is not.
        """Coroutine methods are flagged async, plain ones are not."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_async_model"].is_async and not specs["demo_default_arg"].is_async


class TestDecoratorChainConcurrencyGroup:
    # [CLASS DOC] Tests that the concurrency group marker is found wherever it sits in a decorator
    # chain, and that the outermost marker wins. Each test defines a worker inline, collects specs and
    # asserts the group of the single method.
    def test_marker_above_wrapper_is_found(self):
        # [FUNCTION DOC] The rpc marker is applied outside a metadata-copying wrapper; expects heavy.
        """@rpc applied outside a functools.wraps wrapper still declares its concurrency group."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with one method decorated marker-outside.
            @rpc(concurrency_group="heavy")
            @_passthrough
            def demo_marker_outside(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_outside"].concurrency_group == "heavy"

    def test_marker_below_wrapper_is_found(self):
        # [FUNCTION DOC] The rpc marker is applied inside (below) a metadata-copying wrapper; expects
        # heavy.
        """@rpc applied inside a functools.wraps wrapper still declares its concurrency group."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with one method decorated marker-inside.
            @_passthrough
            @rpc(concurrency_group="heavy")
            def demo_marker_inside(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_inside"].concurrency_group == "heavy"

    def test_marker_between_two_wrapper_layers_is_found(self):
        # [FUNCTION DOC] The marker sits between two wrapper layers; expects heavy.
        """@rpc sandwiched between two wrapper layers is still found by walking the decorator chain."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with one method whose marker is sandwiched.
            @_passthrough
            @rpc(concurrency_group="heavy")
            @_passthrough
            def demo_marker_nested(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_nested"].concurrency_group == "heavy"

    def test_marker_hidden_by_a_wrapper_that_copies_nothing_is_found(self):
        # [FUNCTION DOC] The marker is under a wrapper that copies no attributes and only links the
        # wrapped function; expects heavy, proving the chain is walked through the wrapped link.
        """A wrapper that does not copy the wrapped function's attributes cannot hide the marker."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker whose method is wrapped by an opaque wrapper.
            @_opaque_passthrough
            @rpc(concurrency_group="heavy")
            def demo_marker_hidden(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_hidden"].concurrency_group == "heavy"

    def test_outermost_marker_wins_over_an_inner_one(self):
        # [FUNCTION DOC] Two markers (outer and inner) are separated by an opaque wrapper; expects the
        # outer group name.
        """When two decorator layers each declare a group, the outermost declaration decides."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a method carrying conflicting markers.
            @rpc(concurrency_group="outer")
            @_opaque_passthrough
            @rpc(concurrency_group="inner")
            def demo_marker_conflict(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_marker_conflict"].concurrency_group == "outer"


class TestQueryModel:
    # [CLASS DOC] Tests decoding of query dictionaries through a method's serializer, which builds a
    # validated keyword argument mapping from the method signature.
    def test_decode_query_applies_defaults(self):
        # [FUNCTION DOC] Decoding a query with only a yields a mapping with a and the default b of 10.
        """Omitted parameters with defaults resolve to their default values."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_default_arg"].serializer.decode_query({"a": 5}) == {"a": 5, "b": 10}

    def test_decode_query_parses_nested_model(self):
        # [FUNCTION DOC] Decoding a query whose payload is a plain dictionary yields a real payload
        # model instance.
        """Nested pydantic payloads are revived into real model instances."""
        specs = collect_rpc_method_specs(_GoodWorker)
        kwargs = specs["demo_async_model"].serializer.decode_query({"payload": {"text": "hi"}})
        assert kwargs["payload"] == _Payload(text="hi")

    def test_missing_required_param_rejected(self):
        # [FUNCTION DOC] An empty query raises a validation error because a is required.
        """Missing required parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({})

    def test_unknown_param_rejected(self):
        # [FUNCTION DOC] A query with an extra unknown key raises a validation error.
        """Extra unknown parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": 1, "unknown": 2})

    def test_wrong_type_rejected(self):
        # [FUNCTION DOC] A string for an integer parameter raises a validation error (no coercion).
        """Type-mismatched parameters raise a validation error."""
        specs = collect_rpc_method_specs(_GoodWorker)
        with pytest.raises(ValidationError):
            specs["demo_default_arg"].serializer.decode_query({"a": "not-an-int"})


class TestParameterKinds:
    # [CLASS DOC] Tests accepted parameter kinds: keyword-only parameters and a positional-only receiver.
    def test_keyword_only_parameters_are_supported(self):
        # [FUNCTION DOC] A method with a required and an optional keyword-only parameter decodes a
        # query with only the required one into both values (default filled), and raises a validation
        # error when the required one is missing.
        """Keyword-only parameters are accepted, land in the query model and decode like normal ones."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with one keyword-only method.
            def demo_keyword_only(self, *, required: int, optional: str = "fallback") -> int:
                # [FUNCTION DOC] HELPER: Returns the required value.
                return required

        specs = collect_rpc_method_specs(Worker)
        serializer = specs["demo_keyword_only"].serializer
        assert serializer.decode_query({"required": 5}) == {"required": 5, "optional": "fallback"}
        with pytest.raises(ValidationError):
            serializer.decode_query({"optional": "only"})

    def test_positional_only_receiver_is_supported(self):
        # [FUNCTION DOC] A method whose receiver is positional-only is collected, and decoding a query
        # with the remaining parameter works.
        """A positional-only self is accepted because the receiver never reaches the wire."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker whose method declares a positional-only receiver.
            def demo_positional_receiver(self, /, value: int) -> int:
                # [FUNCTION DOC] HELPER: Returns the value.
                return value

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_positional_receiver"].serializer.decode_query({"value": 7}) == {"value": 7}


class TestAnnotatedParameters:
    # [CLASS DOC] Tests that annotated-metadata constraints on parameters are kept.
    def test_annotated_parameter_constraints_are_preserved(self):
        # [FUNCTION DOC] A parameter constrained to be at least one decodes the value three and raises
        # a validation error for zero.
        """Constraints carried by Annotated metadata survive hint resolution and are enforced."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a constrained integer parameter.
            def demo_constrained(self, value: Annotated[int, Field(ge=1)]) -> int:
                # [FUNCTION DOC] HELPER: Returns the value.
                return value

        serializer = collect_rpc_method_specs(Worker)["demo_constrained"].serializer
        assert serializer.decode_query({"value": 3}) == {"value": 3}
        with pytest.raises(ValidationError):
            serializer.decode_query({"value": 0})


class TestPostponedAnnotations:
    # [CLASS DOC] Tests resolution of string annotations using a worker defined in a module with
    # postponed annotation evaluation.
    def test_string_annotations_resolved_in_worker_module(self):
        # [FUNCTION DOC] Decodes a query for the postponed worker and asserts the payload becomes the
        # late payload model instance.
        """A worker module using postponed annotations still builds real typed models."""
        specs = collect_rpc_method_specs(PostponedWorker)
        kwargs = specs["demo_transform"].serializer.decode_query({"payload": {"text": "hi"}})
        assert kwargs["payload"] == LatePayload(text="hi")

    def test_string_return_annotation_resolved(self):
        # [FUNCTION DOC] Decodes a plain dictionary result with the same method's serializer and
        # asserts it becomes the late payload model instance.
        """A postponed return annotation resolves into a working result adapter."""
        specs = collect_rpc_method_specs(PostponedWorker)
        assert specs["demo_transform"].serializer.decode_result({"text": "hi"}) == LatePayload(text="hi")


class TestInheritance:
    # [CLASS DOC] Tests collection over class inheritance.
    def test_inherited_methods_collected(self):
        # [FUNCTION DOC] A subclass of the good worker with one more method has the three inherited
        # methods plus its own in the collected specs.
        """Methods inherited from a base worker class are exposed too."""

        class Child(_GoodWorker):
            # [CLASS DOC] HELPER: Subclass adding one extra method.
            def demo_child_only(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Child)
        assert {"demo_default_arg", "demo_async_model", "demo_grouped", "demo_child_only"} <= set(specs)


class TestResultAdapter:
    # [CLASS DOC] Tests encoding and decoding of method results by the serializer.
    def test_result_roundtrip(self):
        # [FUNCTION DOC] Encodes a payload model result, asserts it is plain data (not a model) equal
        # to a dictionary with text and the default count, and that decoding it gives back the
        # original model.
        """Return values are encoded as plain json data and decode back into the model."""
        specs = collect_rpc_method_specs(_GoodWorker)
        serializer = specs["demo_async_model"].serializer
        dumped = serializer.encode_result(_Payload(text="hi"))
        assert not isinstance(dumped, _Payload)
        assert dumped == {"text": "hi", "count": 1}
        assert serializer.decode_result(dumped) == _Payload(text="hi")

    def test_none_return_annotation(self):
        # [FUNCTION DOC] For the method annotated to return nothing, decoding a null result yields
        # nothing.
        """Methods annotated -> None get a NoneType result adapter."""
        specs = collect_rpc_method_specs(_GoodWorker)
        assert specs["demo_grouped"].serializer.decode_result(None) is None


class TestFailLoud:
    # [CLASS DOC] Tests that malformed worker classes make collection raise a type error with a
    # recognizable message. Each test defines the offending worker inline.
    def test_async_method_with_non_default_concurrency_group_rejected(self):
        # [FUNCTION DOC] An async method in the train group raises a type error mentioning the
        # concurrency group name.
        """An async method with a non-default concurrency group fails at collection time."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with an async method in a non-default group.
            @rpc(concurrency_group="train")
            async def demo_async_grouped(self) -> int:
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        with pytest.raises(TypeError, match="concurrency_group"):
            collect_rpc_method_specs(Worker)

    def test_missing_param_annotation_rejected(self):
        # [FUNCTION DOC] An unannotated parameter raises a type error saying it must be type-annotated.
        """A parameter without a type annotation fails at collection time."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with an unannotated parameter.
            def demo_unannotated_arg(self, x) -> int:
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        with pytest.raises(TypeError, match="must be type-annotated"):
            collect_rpc_method_specs(Worker)

    def test_missing_return_annotation_rejected(self):
        # [FUNCTION DOC] A missing return annotation raises a type error mentioning the return type
        # annotation.
        """A method without a return annotation fails at collection time."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with an unannotated return.
            def demo_unannotated_return(self, x: int):
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        with pytest.raises(TypeError, match="return type annotation"):
            collect_rpc_method_specs(Worker)

    def test_var_positional_rejected(self):
        # [FUNCTION DOC] A variadic positional parameter raises a type error mentioning args.
        """*args signatures fail at collection time."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a variadic positional method.
            def demo_var_positional(self, *x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        with pytest.raises(TypeError, match="args"):
            collect_rpc_method_specs(Worker)

    def test_var_keyword_rejected(self):
        # [FUNCTION DOC] A variadic keyword parameter raises a type error mentioning kwargs.
        """**kwargs signatures fail at collection time."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a variadic keyword method.
            def demo_var_keyword(self, **x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        with pytest.raises(TypeError, match="kwargs"):
            collect_rpc_method_specs(Worker)

    def test_positional_only_rejected(self):
        # [FUNCTION DOC] A positional-only non-receiver parameter raises a type error mentioning
        # positional-only.
        """Positional-only parameters fail at collection time since calls pass kwargs."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a positional-only parameter.
            def demo_positional_only(self, x: int, /) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        with pytest.raises(TypeError, match="positional-only"):
            collect_rpc_method_specs(Worker)

    def test_non_self_receiver_rejected(self):
        # [FUNCTION DOC] A receiver not named self raises a type error mentioning the receiver
        # parameter self.
        """An unconventionally named receiver is refused rather than silently dropped."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker whose receiver is named this.
            def demo_odd_receiver(this, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        with pytest.raises(TypeError, match="receiver parameter 'self'"):
            collect_rpc_method_specs(Worker)

    def test_forgotten_self_is_rejected_instead_of_eating_the_first_argument(self):
        # [FUNCTION DOC] A method with no receiver, whose first parameter is a real one, raises a type
        # error mentioning the receiver parameter self.
        """A method that forgets self would otherwise lose its first parameter off the wire."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker whose method forgot the receiver.
            def demo_forgot_self(a: int, b: int) -> int:
                # [FUNCTION DOC] HELPER: Returns the sum of a and b.
                return a + b

        with pytest.raises(TypeError, match="receiver parameter 'self'"):
            collect_rpc_method_specs(Worker)

    def test_keyword_only_receiver_rejected(self):
        # [FUNCTION DOC] A keyword-only receiver raises a type error saying the receiver must be taken
        # positionally.
        """A keyword-only self is refused at collection time instead of blowing up on call."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a keyword-only receiver.
            def demo_keyword_only_receiver(*, self) -> int:
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        with pytest.raises(TypeError, match="receiver parameter positionally"):
            collect_rpc_method_specs(Worker)

    def test_method_without_any_parameter_rejected(self):
        # [FUNCTION DOC] A method with no parameters at all raises a type error saying it must take a
        # receiver parameter.
        """A method taking no parameters at all is refused for lacking a receiver."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a parameterless method.
            def demo_no_parameters() -> int:
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        with pytest.raises(TypeError, match="must take a receiver parameter"):
            collect_rpc_method_specs(Worker)

    def test_public_nested_model_class_rejected_as_non_method(self):
        # [FUNCTION DOC] A public nested model class raises a type error whose message says it is not
        # a method and does not talk about a receiver.
        """A public nested model class is refused as not being a method at all."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a nested public model class and one valid method.
            class Config(StrictBaseModel):
                # [CLASS DOC] HELPER: Nested strict model with one text field.
                text: str

            def demo_ok(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        with pytest.raises(TypeError) as excinfo:
            collect_rpc_method_specs(Worker)
        assert "not a method" in str(excinfo.value)
        assert "receiver" not in str(excinfo.value)

    def test_public_class_alias_attribute_rejected_as_non_method(self):
        # [FUNCTION DOC] A public attribute aliasing a class raises a type error that says it is not a
        # method and does not talk about a receiver.
        """A public class alias attribute is refused as not being a method at all."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a class alias attribute and one valid method.
            demo_alias = _Payload

            def demo_ok(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        with pytest.raises(TypeError) as excinfo:
            collect_rpc_method_specs(Worker)
        assert "not a method" in str(excinfo.value)
        assert "receiver" not in str(excinfo.value)

    def test_wrapped_async_method_stays_async(self):
        # [FUNCTION DOC] A method wrapped by a metadata-copying async wrapper is still flagged async,
        # and its serializer decodes a query for x.
        """A functools.wraps-decorated async method is still detected as async."""

        def passthrough(fn):
            # [FUNCTION DOC] HELPER: Decorator returning a metadata-copying async wrapper that awaits
            # the wrapped coroutine function and returns its result.
            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Awaits the wrapped function with all arguments.
                return await fn(*args, **kwargs)

            return wrapper

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a wrapped async method.
            @passthrough
            async def demo_wrapped_async(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_wrapped_async"].is_async
        assert specs["demo_wrapped_async"].serializer.decode_query({"x": 1}) == {"x": 1}

    def test_no_public_methods_collects_an_empty_surface(self):
        # [FUNCTION DOC] A worker with only an underscore-prefixed method collects to an empty mapping
        # without raising.
        """A worker whose whole value is a lifecycle side effect answers no call, and still has to be servable."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with only a private method.
            def _demo_hidden(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        assert collect_rpc_method_specs(Worker) == {}

    def test_any_annotation_allowed(self):
        # [FUNCTION DOC] A method with parameter and return annotated as the any type is accepted and
        # its serializer passes a list value through unchanged.
        """Any-annotated parameters are accepted and passed through."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with any-typed parameter and return.
            def demo_any(self, x: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_any"].serializer.decode_query({"x": [1, "a"]}) == {"x": [1, "a"]}

    def test_only_positional_or_keyword_parameters_may_be_filled_positionally(self):
        # [FUNCTION DOC] For a method with two ordinary parameters and one keyword-only one, the spec's
        # positional parameter names tuple holds only the two ordinary ones, in order.
        """A caller's positional arguments are named in declaration order, and keyword-only names are not in it."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with mixed parameter kinds.
            def demo_mixed(self, a: int, b: int, *, c: int) -> int:
                # [FUNCTION DOC] HELPER: Returns a plus b plus c.
                return a + b + c

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_mixed"].positional_parameter_names == ("a", "b")


def _spec_of(method: Callable[..., Any]) -> RpcMethodSpec:
    # [FUNCTION DOC] HELPER: Creates a fresh class named Worker holding only the given function under
    # its own name, collects its specs and returns the spec of that method.
    worker_cls = type("Worker", (), {method.__name__: method})
    return collect_rpc_method_specs(worker_cls)[method.__name__]


class TestPositionalParameterNames:
    # [CLASS DOC] Tests the ordered tuple of names that positional arguments bind to. Each test
    # defines a function with a receiver and checks the tuple through the spec-of helper.
    def test_positional_parameter_names_keep_declaration_order(self):
        # [FUNCTION DOC] Names are in declaration order (zeta before alpha), not alphabetical.
        """Parameter names are recorded in declaration order, not the alphabetical order methods are collected in."""

        def demo_pair(self, zeta: int, alpha: int) -> int:
            # [FUNCTION DOC] HELPER: Returns zeta minus alpha.
            return zeta - alpha

        assert _spec_of(demo_pair).positional_parameter_names == ("zeta", "alpha")

    def test_the_receiver_is_not_a_positional_parameter_name(self):
        # [FUNCTION DOC] The receiver is excluded: the tuple holds only the value parameter.
        """The receiver never reaches the wire, so a caller's first positional binds to the first real parameter."""

        def demo_one(self, value: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the value.
            return value

        assert _spec_of(demo_one).positional_parameter_names == ("value",)

    def test_a_positional_only_receiver_does_not_shift_the_parameter_names(self):
        # [FUNCTION DOC] A positional-only receiver is still excluded: the tuple holds only value.
        """A method declaring its receiver positional-only still binds callers to its own first parameter."""

        def demo_positional_receiver(self, /, value: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the value.
            return value

        assert _spec_of(demo_positional_receiver).positional_parameter_names == ("value",)

    def test_a_method_without_parameters_has_no_positional_parameter_names(self):
        # [FUNCTION DOC] A method with only a receiver has an empty tuple.
        """A method taking only the receiver accepts no positional arguments at all."""

        def demo_none(self) -> int:
            # [FUNCTION DOC] HELPER: Returns zero.
            return 0

        assert _spec_of(demo_none).positional_parameter_names == ()

    def test_keyword_only_parameters_are_absent_from_the_positional_names(self):
        # [FUNCTION DOC] A method with only keyword-only parameters has an empty tuple.
        """Keyword-only parameters cannot be filled positionally, so they are not part of the positional order."""

        def demo_keyword_only(self, *, a: int, b: int) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        assert _spec_of(demo_keyword_only).positional_parameter_names == ()


class TestCanonicalizeMethodArguments:
    # [CLASS DOC] Tests the function that merges positional and keyword arguments into one keyword
    # mapping keyed by parameter name, and its local errors. It takes the method spec, a tuple of
    # positionals and a mapping of keywords.
    def test_positional_arguments_are_named_in_declaration_order(self):
        # [FUNCTION DOC] Two positionals bind to zeta and alpha in order.
        """Positionals are zipped onto parameter names in order, so the wire carries each value under its own name."""

        def demo_pair(self, zeta: int, alpha: int) -> int:
            # [FUNCTION DOC] HELPER: Returns zeta minus alpha.
            return zeta - alpha

        canonical = canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(1, 2), kwargs={})
        assert canonical == {"zeta": 1, "alpha": 2}

    def test_unfilled_trailing_parameters_are_omitted_rather_than_padded(self):
        # [FUNCTION DOC] One positional for a two-parameter method yields only the first key, so the
        # default of the second is applied later.
        """A short positional call leaves later parameters absent so the server still applies their defaults."""

        def demo_pair(self, a: int, b: int = 10) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        assert canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(1,), kwargs={}) == {"a": 1}

    def test_keyword_arguments_alone_are_passed_through_unchanged(self):
        # [FUNCTION DOC] A call with only keywords returns them unchanged.
        """A purely keyword call keeps working exactly as before positional support existed."""

        def demo_pair(self, a: int, b: int = 10) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        canonical = canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(), kwargs={"a": 1, "b": 2})
        assert canonical == {"a": 1, "b": 2}

    def test_positional_and_keyword_arguments_are_merged(self):
        # [FUNCTION DOC] One positional for a and one keyword for b merge into both keys.
        """Positionals and keywords naming distinct parameters combine into one keyword-shaped payload."""

        def demo_pair(self, a: int, b: int = 10) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        canonical = canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(1,), kwargs={"b": 2})
        assert canonical == {"a": 1, "b": 2}

    def test_a_keyword_only_parameter_is_still_reachable_by_keyword(self):
        # [FUNCTION DOC] A positional for a plus a keyword for the keyword-only c merge into both keys.
        """Positionals fill the ordinary parameters while a keyword-only parameter arrives by name."""

        def demo_mixed(self, a: int, *, c: int) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus c.
            return a + c

        canonical = canonicalize_method_arguments(spec=_spec_of(demo_mixed), args=(1,), kwargs={"c": 3})
        assert canonical == {"a": 1, "c": 3}

    def test_filling_every_parameter_positionally_is_allowed(self):
        # [FUNCTION DOC] Exactly as many positionals as positional names is the boundary case and is
        # accepted.
        """Supplying exactly as many positionals as parameters is the boundary case and must not be rejected."""

        def demo_pair(self, a: int, b: int = 10) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        canonical = canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(1, 2), kwargs={})
        assert canonical == {"a": 1, "b": 2}

    def test_one_positional_too_many_is_rejected(self):
        # [FUNCTION DOC] Three positionals for a two-parameter method raise a type error whose message
        # reads that the method takes at most 2 positional arguments, got 3.
        """An excess positional cannot be named, so it fails locally with the TypeError an in-process call raises."""

        def demo_pair(self, a: int, b: int = 10) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        with pytest.raises(TypeError) as excinfo:
            canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(1, 2, 3), kwargs={})
        assert "demo_pair() takes at most 2 positional arguments, got 3" in str(excinfo.value)

    def test_a_keyword_only_parameter_cannot_be_filled_positionally(self):
        # [FUNCTION DOC] A second positional for a method with one ordinary and one keyword-only
        # parameter raises a type error saying at most 1 positional arguments.
        """A keyword-only parameter is not part of the positional order, so a positional for it is an error."""

        def demo_mixed(self, a: int, *, c: int) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus c.
            return a + c

        with pytest.raises(TypeError, match="at most 1 positional arguments"):
            canonicalize_method_arguments(spec=_spec_of(demo_mixed), args=(1, 3), kwargs={})

    def test_a_parameterless_method_rejects_any_positional_argument(self):
        # [FUNCTION DOC] One positional for a receiver-only method raises a type error saying at most 0
        # positional arguments.
        """A method taking only the receiver has no name to bind a positional to."""

        def demo_none(self) -> int:
            # [FUNCTION DOC] HELPER: Returns zero.
            return 0

        with pytest.raises(TypeError, match="at most 0 positional arguments"):
            canonicalize_method_arguments(spec=_spec_of(demo_none), args=(1,), kwargs={})

    def test_a_parameter_given_positionally_and_by_keyword_is_rejected(self):
        # [FUNCTION DOC] A given both positionally and as a keyword raises a type error whose message
        # says the method got multiple values for a, listing the name.
        """A duplicated parameter must raise instead of one value silently overwriting the other."""

        def demo_pair(self, a: int, b: int = 10) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        with pytest.raises(TypeError) as excinfo:
            canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(1,), kwargs={"a": 2})
        assert "demo_pair() got multiple values for ['a']" in str(excinfo.value)

    def test_every_duplicated_parameter_is_reported_in_sorted_order(self):
        # [FUNCTION DOC] Two duplicated names (zeta and alpha) are all reported, sorted alphabetically,
        # in the error message.
        """All conflicting names are listed, sorted, so the message does not depend on dict iteration order."""

        def demo_pair(self, zeta: int, alpha: int) -> int:
            # [FUNCTION DOC] HELPER: Returns zeta minus alpha.
            return zeta - alpha

        with pytest.raises(TypeError) as excinfo:
            canonicalize_method_arguments(spec=_spec_of(demo_pair), args=(1, 2), kwargs={"zeta": 3, "alpha": 4})
        assert "['alpha', 'zeta']" in str(excinfo.value)

    def test_an_unknown_keyword_is_left_for_the_server_to_reject(self):
        # [FUNCTION DOC] An unknown keyword is kept in the merged mapping (not rejected by
        # canonicalization), and decoding that mapping through the serializer raises a validation error.
        """Client-side canonicalization only binds positionals; unknown names stay for query validation to refuse."""

        def demo_pair(self, a: int, b: int = 10) -> int:
            # [FUNCTION DOC] HELPER: Returns a plus b.
            return a + b

        spec = _spec_of(demo_pair)
        assert canonicalize_method_arguments(spec=spec, args=(1,), kwargs={"nope": 2}) == {"a": 1, "nope": 2}
        with pytest.raises(ValidationError):
            spec.serializer.decode_query(canonicalize_method_arguments(spec=spec, args=(1,), kwargs={"nope": 2}))


class TestRayInjectedTraceContext:
    # [CLASS DOC] Tests tolerance of the keyword-only trace context parameter that ray injects into
    # actor method signatures, and that look-alike parameters are still validated.
    def test_a_method_carrying_rays_injected_trace_context_is_still_collected(self):
        # [FUNCTION DOC] With the injected parameter, positional names are still a and b, a query for
        # them decodes, and a query that includes the trace context key raises a validation error (the
        # injected parameter is not on the wire).
        """Ray rewrites actor method signatures with a keyword-only _ray_trace_ctx, which must not reach the wire."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker whose method got the injected parameter.
            @_inject_ray_trace_context
            def demo_traced(self, a: int, b: int) -> int:
                # [FUNCTION DOC] HELPER: Returns a plus b.
                return a + b

        specs = collect_rpc_method_specs(Worker)
        assert specs["demo_traced"].positional_parameter_names == ("a", "b")
        assert specs["demo_traced"].serializer.decode_query({"a": 1, "b": 2}) == {"a": 1, "b": 2}
        with pytest.raises(ValidationError):
            specs["demo_traced"].serializer.decode_query({"a": 1, "b": 2, "_ray_trace_ctx": {}})

    def test_an_unannotated_parameter_beside_the_injected_one_is_still_rejected(self):
        # [FUNCTION DOC] An unannotated parameter x next to the injected one raises a type error naming
        # parameter x as needing an annotation.
        """Skipping ray's parameter must not excuse a genuinely unannotated parameter on the same method."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with an injected parameter and an unannotated one.
            @_inject_ray_trace_context
            def demo_traced(self, x) -> int:
                # [FUNCTION DOC] HELPER: Returns x.
                return x

        with pytest.raises(TypeError, match="parameter 'x' must be type-annotated"):
            collect_rpc_method_specs(Worker)

    def test_an_ordinary_parameter_borrowing_rays_name_is_still_rejected(self):
        # [FUNCTION DOC] A normal (positional-or-keyword) unannotated parameter named like the trace
        # context raises a type error naming it.
        """Only ray's keyword-only injection is skipped, not any parameter that happens to share its name."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with an ordinary parameter borrowing ray's name.
            def demo_shadowed(self, _ray_trace_ctx) -> int:
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        with pytest.raises(TypeError, match="parameter '_ray_trace_ctx' must be type-annotated"):
            collect_rpc_method_specs(Worker)


class TestAWorkersOwnParameterNamedLikeRaysInjection:
    # [CLASS DOC] Tests that only the exact shape ray injects (keyword-only, default nothing,
    # unannotated) is skipped; a worker's own parameter of the same name is validated normally.
    def test_an_unannotated_keyword_only_parameter_of_that_name_with_another_default_is_refused(self):
        # [FUNCTION DOC] An unannotated keyword-only parameter with default zero raises a type error
        # naming it as needing an annotation.
        """Ray's injection defaults to None, so this one is the worker's own and takes the annotation check."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker declaring its own parameter with the ray name.
            def demo_declared(self, *, _ray_trace_ctx=0) -> int:
                # [FUNCTION DOC] HELPER: Returns the parameter.
                return _ray_trace_ctx

        with pytest.raises(TypeError, match="parameter '_ray_trace_ctx' must be type-annotated"):
            collect_rpc_method_specs(Worker)

    def test_an_unannotated_required_keyword_only_parameter_of_that_name_is_refused(self):
        # [FUNCTION DOC] A required unannotated keyword-only parameter of that name raises a type error
        # naming it as needing an annotation.
        """Skipping it joined no request model and left the method uncallable without ever saying so."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with a required keyword-only parameter of that name.
            def demo_declared(self, *, _ray_trace_ctx) -> int:
                # [FUNCTION DOC] HELPER: Returns the parameter.
                return _ray_trace_ctx

        with pytest.raises(TypeError, match="parameter '_ray_trace_ctx' must be type-annotated"):
            collect_rpc_method_specs(Worker)

    def test_an_annotated_keyword_only_parameter_of_that_name_satisfies_the_annotation_check(self):
        # [FUNCTION DOC] An annotated keyword-only parameter of that name with default three is
        # accepted; positional names hold only a.
        """The check the skip used to excuse is the only thing asked of a parameter the worker declares."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with an annotated parameter of the ray name.
            def demo_declared(self, a: int, *, _ray_trace_ctx: int = 3) -> int:
                # [FUNCTION DOC] HELPER: Returns a plus the parameter.
                return a + _ray_trace_ctx

        assert collect_rpc_method_specs(Worker)["demo_declared"].positional_parameter_names == ("a",)

    def test_the_shape_ray_actually_injects_is_still_skipped(self):
        # [FUNCTION DOC] With the exact injected shape, collection succeeds, positional names hold only
        # a, and a query for a decodes.
        """Every part of the match has to keep admitting the shape ray rewrites actor methods with."""

        class Worker:
            # [CLASS DOC] HELPER: Inline worker with the exact injected shape.
            @_inject_ray_trace_context
            def demo_traced(self, a: int) -> int:
                # [FUNCTION DOC] HELPER: Returns a.
                return a

        specs = collect_rpc_method_specs(Worker)

        assert specs["demo_traced"].positional_parameter_names == ("a",)
        assert specs["demo_traced"].serializer.decode_query({"a": 1}) == {"a": 1}
