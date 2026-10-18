# [MODULE DOC]
# Json-compatible encoding of rpc arguments and results using pydantic models built from method
# annotations. Plain json cannot carry nan or infinities, so non-finite floats are replaced by a
# tagged single-key mapping on encode and restored on decode; a payload that itself contains the
# reserved tag key is rejected on encode.
from __future__ import annotations

import dataclasses
import math
from typing import Any

import typing_extensions
from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model

NON_FINITE_FLOAT_TAG = "__miles_non_finite_float__"

_WIRE_CONFIG = ConfigDict(ser_json_inf_nan="constants", ser_json_bytes="base64", val_json_bytes="base64")

_TOKEN_BY_FLOAT = {math.inf: "inf", -math.inf: "-inf"}
_FLOAT_BY_TOKEN = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


@dataclasses.dataclass(frozen=True)
class RpcSerializer:
    # [CLASS DOC]
    # Immutable pair for one method: a generated model validating the arguments (unknown fields
    # forbidden) and a type adapter for the result type. All encode functions produce
    # json-compatible values; decode functions validate and rebuild typed values.
    query_model: type[BaseModel]
    result_adapter: TypeAdapter[Any]

    @classmethod
    def create(cls, *, query_model_name: str, query_fields: dict[str, Any], result_annotation: Any) -> RpcSerializer:
        # [FUNCTION DOC]
        # Build a serializer. Creates the argument model with the given name and fields (mapping
        # from name to a pair of type and default), forbidding extra fields and configuring wire
        # settings (nan and infinity as constants, bytes as base64). The result adapter uses the
        # same wire settings unless the result type is a model, dataclass or typed dict, which
        # keep their own configuration.
        query_model = create_model(
            query_model_name, __config__=ConfigDict(extra="forbid", **_WIRE_CONFIG), **query_fields
        )
        result_config = None if _carries_own_config(result_annotation) else _WIRE_CONFIG
        return cls(query_model=query_model, result_adapter=TypeAdapter(result_annotation, config=result_config))

    def encode_query(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] Validate the keyword arguments with the argument model (raises a pydantic
        # validation error on missing, unknown or ill-typed values), dump to json-compatible form
        # and tag non-finite floats.
        return _NonFiniteFloatCodec.encode(self.query_model(**kwargs).model_dump(mode="json"))

    def decode_query(self, query: dict[str, Any]) -> dict[str, Any]:
        # [FUNCTION DOC] Restore non-finite floats, validate with the argument model and return a
        # mapping from parameter name to typed value (shallow conversion of the model).
        return dict(self.query_model(**_NonFiniteFloatCodec.decode(query)))

    def encode_result(self, result: Any) -> Any:
        # [FUNCTION DOC] Dump the result to json-compatible form via the result adapter (type
        # mismatch warnings are turned into errors), then tag non-finite floats.
        return _NonFiniteFloatCodec.encode(self.result_adapter.dump_python(result, mode="json", warnings="error"))

    def decode_result(self, payload: Any) -> Any:
        # [FUNCTION DOC] Restore non-finite floats and validate the payload against the result type.
        return self.result_adapter.validate_python(_NonFiniteFloatCodec.decode(payload))


class _NonFiniteFloatCodec:
    # [CLASS DOC] HELPER: Stateless recursive codec over dictionaries and lists that swaps
    # non-finite floats with a tagged single-entry dictionary and back. Tuples and other containers
    # are not traversed.
    @classmethod
    def encode(cls, value: Any) -> Any:
        # [FUNCTION DOC] HELPER: Replace each nan or infinity with a dictionary holding the tag key
        # and the token nan, inf or minus inf. Recurses through dictionaries and lists; raises a
        # value error if a dictionary already contains the reserved tag key.
        if isinstance(value, float) and not math.isfinite(value):
            return {NON_FINITE_FLOAT_TAG: _TOKEN_BY_FLOAT.get(value, "nan")}
        if isinstance(value, dict):
            if NON_FINITE_FLOAT_TAG in value:
                raise ValueError(f"rpc payloads must not contain the reserved key {NON_FINITE_FLOAT_TAG!r}")
            return {key: cls.encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.encode(item) for item in value]
        return value

    @classmethod
    def decode(cls, value: Any) -> Any:
        # [FUNCTION DOC] HELPER: Inverse of encode: a dictionary with exactly the tag key and a
        # known token becomes the matching float; other dictionaries and lists are rebuilt
        # recursively; everything else is returned as is.
        if isinstance(value, dict):
            token = value.get(NON_FINITE_FLOAT_TAG)
            if len(value) == 1 and token in _FLOAT_BY_TOKEN:
                return _FLOAT_BY_TOKEN[token]
            return {key: cls.decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls.decode(item) for item in value]
        return value


def _carries_own_config(annotation: Any) -> bool:
    # [FUNCTION DOC] HELPER: True iff the annotation is a class that is a pydantic model, a
    # dataclass or a typed dictionary (types for which a separate adapter configuration is not
    # allowed or meaningful).
    if not isinstance(annotation, type):
        return False
    return (
        issubclass(annotation, BaseModel)
        or dataclasses.is_dataclass(annotation)
        or typing_extensions.is_typeddict(annotation)
    )
