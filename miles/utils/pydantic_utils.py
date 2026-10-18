# [MODULE DOC]
# HELPER: Shared pydantic base models. All serialize infinite and nan floats as json constants
# instead of failing. They differ in strictness about unknown fields and mutability.
from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    # [CLASS DOC] HELPER: Mutable model that rejects unknown fields.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


class FrozenStrictBaseModel(BaseModel):
    # [CLASS DOC] HELPER: Immutable (hashable if fields are) model that rejects unknown fields.
    model_config = ConfigDict(extra="forbid", frozen=True, ser_json_inf_nan="constants")


class FrozenPartialBaseModel(BaseModel):
    # [CLASS DOC] HELPER: Immutable model that ignores unknown fields, can be built from arbitrary
    # objects' attributes, and accepts field names even when aliases exist; for reading a subset of
    # a larger structure.
    model_config = ConfigDict(
        extra="ignore", frozen=True, from_attributes=True, populate_by_name=True, ser_json_inf_nan="constants"
    )


class FrozenOpenBaseModel(BaseModel):
    # [CLASS DOC] HELPER: Immutable model that keeps unknown fields as extra attributes.
    model_config = ConfigDict(extra="allow", frozen=True, ser_json_inf_nan="constants")
