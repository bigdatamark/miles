# [MODULE DOC] Unit tests for the pickled escape hatch wire type of the rpc layer: a parameter
# annotated with the pickled type is encoded as a tagged base64 string in json and decoded back to
# the original object, while in-process values and untagged dictionaries pass through untouched,
# malformed tagged payloads are rejected, and parameters beside it stay strictly typed. Also checks
# how rpc method spec collection treats annotated, unannotated and hatch parameters.
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from pydantic.errors import PydanticSchemaGenerationError

from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.common.wire_types import PICKLED_TAG, Pickled


class _PickledCall(BaseModel):
    # [CLASS DOC] HELPER: Pydantic model with a single field typed with the pickled escape hatch.
    args: Pickled


class _MixedCall(BaseModel):
    # [CLASS DOC] HELPER: Pydantic model with a pickled field and a strictly typed integer field.
    args: Pickled
    rollout_id: int


class TestThePickledEscapeHatch:
    # [CLASS DOC] Tests encoding, decoding and validation behavior of the pickled field type.
    def test_a_namespace_survives_a_json_round_trip_unchanged(self):
        # [FUNCTION DOC] Dumps a model holding an argument namespace to json text and validates it back;
        # asserts the restored field is a namespace with the same two attribute values.
        """The trainer is built from the whole args, and a lossy round trip would change its run."""
        call = _PickledCall(args=Namespace(num_rollout=7, save=None))

        restored = _PickledCall.model_validate_json(call.model_dump_json())

        assert isinstance(restored.args, Namespace)
        assert (restored.args.num_rollout, restored.args.save) == (7, None)

    def test_a_value_no_wire_type_reproduces_survives(self):
        # [FUNCTION DOC] A namespace holding a filesystem path and a set round-trips through json and
        # both attributes compare equal to the originals.
        """This is the whole reason for the hatch: args holds objects json cannot describe."""
        call = _PickledCall(args=Namespace(path=Path("/models/qwen"), tags={"a", "b"}))

        restored = _PickledCall.model_validate_json(call.model_dump_json())

        assert restored.args.path == Path("/models/qwen")
        assert restored.args.tags == {"a", "b"}

    def test_the_encoded_form_is_a_tagged_string(self):
        # [FUNCTION DOC] Dumping in json mode produces a mapping whose pickled tag key holds a string.
        """A reader of the wire can tell the hatch apart from an ordinary payload."""
        encoded = _PickledCall(args=Namespace(a=1)).model_dump(mode="json")["args"]

        assert isinstance(encoded[PICKLED_TAG], str)

    def test_an_in_process_value_is_taken_as_is(self):
        # [FUNCTION DOC] Constructing the model with an object keeps the very same object (identity).
        """A caller in the same process passes the object itself, and a copy would drop mutations."""
        args = Namespace(a=1)

        assert _PickledCall(args=args).args is args

    def test_an_untagged_dict_is_taken_as_is(self) -> None:
        # [FUNCTION DOC] A plain dictionary without the tag key is kept as the same object.
        """An ordinary dictionary is data rather than an encoded pickle envelope."""
        args = {"rollout_id": 7}

        assert _PickledCall(args=args).args is args

    def test_a_malformed_tagged_payload_is_rejected(self) -> None:
        # [FUNCTION DOC] A mapping with the tag key and an invalid base64 string raises a validation error.
        """A marked payload with invalid base64 is rejected instead of accepted as data."""
        with pytest.raises(ValidationError):
            _PickledCall(args={PICKLED_TAG: "a"})

    def test_the_hatch_does_not_loosen_its_neighbours(self):
        # [FUNCTION DOC] A string given for the integer field beside a pickled field raises a
        # validation error.
        """The hatch is per parameter: everything beside it stays strictly wire-typed."""
        with pytest.raises(ValidationError):
            _MixedCall(args=Namespace(a=1), rollout_id="seven")


class TestTheStrictnessThePickledHatchDoesNotWeaken:
    # [CLASS DOC] Tests that spec collection stays strict for parameters not using the hatch.
    def test_an_unannotated_parameter_is_still_refused(self):
        # [FUNCTION DOC] Collecting specs for a worker with an unannotated parameter raises a type
        # error saying it must be type-annotated.
        """Without an annotation there is no wire type at all, and the hatch is opt-in per parameter."""

        class _Worker:
            # [CLASS DOC] HELPER: Inline worker with an unannotated parameter.
            def act(self, value) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        with pytest.raises(TypeError, match="must be type-annotated"):
            collect_rpc_method_specs(_Worker)

    def test_a_parameter_without_the_hatch_keeps_its_wire_type(self):
        # [FUNCTION DOC] A keyword-only parameter typed as an argument namespace (no hatch) makes spec
        # collection raise a pydantic schema generation error.
        """A plain object reaches a strictly typed parameter only if the wire type accepts it."""

        class _Worker:
            # [CLASS DOC] HELPER: Inline worker with a namespace-typed parameter and no hatch.
            def act(self, *, args: Namespace) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        with pytest.raises(PydanticSchemaGenerationError):
            collect_rpc_method_specs(_Worker)

    def test_the_hatch_makes_that_same_parameter_acceptable(self):
        # [FUNCTION DOC] With the parameter typed via the hatch, collection succeeds and encoding a
        # query with a namespace and an integer keeps the integer value 3.
        """The hatch is what lets the argparse Namespace cross while the rest stays typed."""

        class _Worker:
            # [CLASS DOC] HELPER: Inline worker with a pickled parameter and an integer parameter.
            def act(self, *, args: Pickled, rollout_id: int) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        specs = collect_rpc_method_specs(_Worker)

        assert specs["act"].serializer.encode_query(dict(args=Namespace(a=1), rollout_id=3))["rollout_id"] == 3
