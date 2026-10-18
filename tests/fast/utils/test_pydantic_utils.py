# [MODULE DOC]
# Tests for the shared pydantic base models (strict, frozen strict, frozen partial, frozen open).
# Four small model classes each with one float field are defined, and every base is checked to
# preserve non-finite floats (not-a-number, positive and negative infinity) across a JSON dump and
# re-validation round trip.
import math

import pytest
from pydantic import BaseModel

from miles.utils.pydantic_utils import (
    FrozenOpenBaseModel,
    FrozenPartialBaseModel,
    FrozenStrictBaseModel,
    StrictBaseModel,
)


class _StrictFloatModel(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict base model with a single float field.
    value: float


class _FrozenStrictFloatModel(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: Frozen strict base model with a single float field.
    value: float


class _FrozenPartialFloatModel(FrozenPartialBaseModel):
    # [CLASS DOC] HELPER: Frozen partial base model with a single float field.
    value: float


class _FrozenOpenFloatModel(FrozenOpenBaseModel):
    # [CLASS DOC] HELPER: Frozen open base model with a single float field.
    value: float


class TestNonFiniteFloatJsonRoundTrip:
    # [CLASS DOC] JSON round-trip tests of non-finite floats on the shared bases.
    @pytest.mark.parametrize(
        "model_class",
        [_StrictFloatModel, _FrozenStrictFloatModel, _FrozenPartialFloatModel, _FrozenOpenFloatModel],
    )
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_every_shared_model_base_preserves_non_finite_floats_across_json(
        self,
        model_class: type[BaseModel],
        value: float,
    ) -> None:
        # [FUNCTION DOC] Parametrized over the four model classes and the values not-a-number, plus
        # infinity and minus infinity. Dump an instance to JSON and validate it back; a restored
        # not-a-number is checked with the nan test (since it never equals itself), infinities by
        # equality.
        """Every shared model base preserves non-finite floats across a JSON round trip."""
        restored_value = model_class.model_validate_json(model_class(value=value).model_dump_json()).value

        if math.isnan(value):
            assert math.isnan(restored_value)
        else:
            assert restored_value == value
