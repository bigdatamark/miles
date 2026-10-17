# [MODULE DOC]
# JSON-side translation between the Tinker SDK wire format and the gateway's internal payloads:
# validating requests against the SDK's own request models, decoding commands (forward-backward,
# optimizer step, save/load state, save sampler weights) and sampling requests into internal
# mappings, building internal datums, decoding tensor data (dense or 1-D sparse), and rendering
# results back to JSON. Content errors in training commands are not raised but returned as a
# validation error inside the payload so they fail in sequence order. Module constants: default
# Adam parameters taken from the SDK, and a mapping from operation name to the SDK request model
# used to validate its JSON body (forward-backward travels as protobuf or is checked in datum
# building instead).
"""Translate SDK JSON requests and results.

Datums encode input x and explicit target labels t as x + [t[-1]],
so each output scores logprob(t[i] | x[0..i])."""

import math

import pydantic

from miles.tinker.core.input_validation import validate_save_options
from miles.tinker.core.types import LOSS_INPUT_KEYS, UserInputError
from tinker import types as tinker_types
from tinker.types.sample_response import MASK_LOGPROB

# materialized at the boundary so core and the executor can require every key
ADAM_PARAM_DEFAULTS = tinker_types.AdamParams().model_dump()

# JSON-wire request models; forward_backward travels as protobuf and its datums
# are checked in build_datum instead
REQUEST_TYPES = {
    "optim_step": tinker_types.OptimStepRequest,
    "save_state": tinker_types.SaveWeightsRequest,
    "load_state": tinker_types.LoadWeightsRequest,
    "save_weights_for_sampler": tinker_types.SaveWeightsForSamplerRequest,
}


def validate_against_sdk(request_type, payload: dict) -> None:
    # [FUNCTION DOC] Validate the payload with the given pydantic request model; on failure raise a
    # user input error carrying only the first validation problem as "location: message" with the
    # dotted field location (or the word request if empty), hiding the original error chain.
    """The pinned SDK's request models are the wire contract; reject what they reject."""
    try:
        request_type.model_validate(payload)
    except pydantic.ValidationError as error:
        first = error.errors()[0]
        location = ".".join(str(part) for part in first["loc"])
        raise UserInputError(f"{location or 'request'}: {first['msg']}") from None


def validate_create_model(payload: dict) -> None:
    # [FUNCTION DOC] Validate a create-model body with the SDK's model (user input error on failure).
    validate_against_sdk(tinker_types.CreateModelRequest, payload)


def validate_create_sampling_session(payload: dict) -> None:
    # [FUNCTION DOC] Validate a create-sampling-session body with the SDK's model (user input error
    # on failure).
    validate_against_sdk(tinker_types.CreateSamplingSessionRequest, payload)


def decode_command(op: str, payload: dict) -> tuple[str, dict]:
    # [FUNCTION DOC]
    # Decode a JSON command body into (operation, internal payload). The body must contain the
    # model id and sequence id (else a user input error is raised, since the failure could not be
    # attributed to a queue position). Any user input, key, type, value or index error while
    # decoding the rest is converted into a payload that holds only the model id, sequence id and a
    # validation error text, so the command fails in order without running; a forward-backward
    # command flagged forward-only is reported as the forward-only operation even in that case.
    """Decode content errors into ordered failures when the envelope is identifiable."""
    try:
        envelope = {"model_id": payload["model_id"], "seq_id": payload["seq_id"]}
    except KeyError as error:
        raise UserInputError(f"missing command envelope field {error.args[0]!r}") from None
    try:
        return _decode_command(op, payload, envelope)
    except (UserInputError, KeyError, TypeError, ValueError, IndexError) as error:
        if op == "forward_backward" and payload.get("forward_only"):
            op = "forward_only"
        return op, envelope | {"validation_error": str(error)}


def _decode_command(op: str, payload: dict, decoded: dict) -> tuple[str, dict]:
    # [FUNCTION DOC]
    # HELPER: Decode by operation, extending the given envelope mapping in place and returning
    # (operation, payload). Operations with an SDK model are validated against it first.
    # Forward-backward: each datum's model input chunks are flattened into tokens and its loss
    # inputs into lists; datums are built; the loss function name and config (empty if missing)
    # are added; the operation becomes forward-only if flagged. Optimizer step: Adam parameters are
    # the SDK defaults overlaid by the given ones. Save state: save options validated; checkpoint
    # name from the path (may be none) and an overwrite flag (default false). Load state: path,
    # optimizer flag, optional access token. Save sampler weights: options validated; sampler path
    # from the path. Unknown operation raises a user input error.
    if (request_type := REQUEST_TYPES.get(op)) is not None:
        validate_against_sdk(request_type, payload)
    if op == "forward_backward":
        fb_input = payload["forward_backward_input"]
        datums = [
            (model_input_tokens(datum["model_input"]), _decode_inputs(datum["loss_fn_inputs"]))
            for datum in fb_input["data"]
        ]
        decoded |= {
            "datums": [build_datum(tokens, inputs, i) for i, (tokens, inputs) in enumerate(datums)],
            "loss_fn": fb_input["loss_fn"],
            "loss_fn_config": fb_input.get("loss_fn_config") or {},
        }
        return ("forward_only" if payload.get("forward_only") else op), decoded
    if op == "optim_step":
        return op, decoded | {"adam_params": {**ADAM_PARAM_DEFAULTS, **payload["adam_params"]}}
    if op == "save_state":
        validate_save_options(payload)
        return op, decoded | {"name": payload.get("path"), "overwrite": bool(payload.get("overwrite", False))}
    if op == "load_state":
        return op, decoded | {
            "path": payload["path"],
            "optimizer": payload["optimizer"],
            "weights_access_token": payload.get("weights_access_token"),
        }
    if op == "save_weights_for_sampler":
        validate_save_options(payload)
        return op, decoded | {"sampler_path": payload.get("path")}
    raise UserInputError(f"unknown command op {op!r}")


def model_input_tokens(model_input: dict) -> list[int]:
    # [FUNCTION DOC] Concatenate the token lists of the model input's chunks in order. Any chunk
    # type other than encoded text raises a user input error.
    tokens: list[int] = []
    for chunk in model_input["chunks"]:
        if chunk.get("type") != "encoded_text":
            raise UserInputError(f"unsupported model_input chunk type: {chunk.get('type')}")
        tokens.extend(chunk["tokens"])
    return tokens


def build_datum(input_tokens: list[int], inputs: dict[str, list], index: int) -> dict:
    # [FUNCTION DOC]
    # Build the internal datum from input token ids and a mapping of loss input names to value
    # lists (index is only for messages). Raises user input errors for unknown input names (only
    # the known wire loss inputs and target tokens are allowed), for nested (non-1-D) values, and
    # when the target token count differs from the input token count. The target tokens input is
    # required (a missing one raises a key error, which the command decoder converts). Result:
    # tokens are the inputs plus the last target token (so position i scores target i given inputs
    # up to i), the target length, the target tokens, and each present known loss input converted
    # to a list of floats under its internal key.
    """One decoded datum (token list + loss_fn_inputs lists) -> internal datum."""
    unknown = set(inputs) - set(LOSS_INPUT_KEYS) - {"target_tokens"}
    if unknown:
        raise UserInputError(f"datum {index}: unknown loss_fn_inputs {sorted(unknown)}")
    for name, values in inputs.items():
        if any(isinstance(value, (list, tuple)) for value in values):
            raise UserInputError(
                f"datum {index}: loss_fn_inputs[{name!r}] must be 1-D; multi-target inputs are not supported"
            )
    targets = list(inputs["target_tokens"])
    if len(targets) != len(input_tokens):
        raise UserInputError(
            f"datum {index}: target_tokens length {len(targets)} != model_input length {len(input_tokens)}"
        )
    datum = {"tokens": input_tokens + targets[-1:], "target_len": len(targets), "target_tokens": targets}
    for wire_key, datum_key in LOSS_INPUT_KEYS.items():
        if wire_key in inputs:
            datum[datum_key] = [float(value) for value in inputs[wire_key]]
    return datum


def _decode_inputs(loss_fn_inputs: dict) -> dict[str, list]:
    # [FUNCTION DOC] HELPER: Convert every tensor-data entry of the loss inputs mapping to a plain
    # list, keeping names.
    return {name: tensor_data_to_list(value) for name, value in loss_fn_inputs.items()}


def tensor_data_to_list(tensor_data) -> list:
    # [FUNCTION DOC] Convert SDK tensor data to a Python list. A list passes through unchanged. A
    # non-mapping raises a user input error. A shape with more than one dimension is rejected.
    # Sparse compressed-row data is expanded to a dense list; otherwise the data field is copied to
    # a list (missing data raises a user input error).
    if isinstance(tensor_data, list):
        return tensor_data
    if not isinstance(tensor_data, dict):
        raise UserInputError(f"expected TensorData, got {type(tensor_data).__name__}")
    if len(tensor_data.get("shape") or []) > 1:
        raise UserInputError("multi-target inputs are not supported; loss_fn_inputs must be 1-D")
    if tensor_data.get("sparse_crow_indices") is not None:
        return _dense_from_csr(tensor_data)
    data = tensor_data.get("data")
    if data is None:
        raise UserInputError("TensorData without data")
    return list(data)


def _dense_from_csr(tensor_data: dict) -> list:
    # [FUNCTION DOC] HELPER: Expand 1-D sparse compressed-row data to a dense list of the shape's
    # length filled with zeros, placing each value at its column index (strictly paired, so a
    # length mismatch raises a value error). The row pointer list must have exactly two entries
    # (user input error). Its content is not otherwise checked; an out-of-range column raises an
    # index error. A shape that is not one-dimensional raises a value error.
    (length,) = tensor_data["shape"]
    if len(tensor_data["sparse_crow_indices"]) != 2:
        raise UserInputError("1-D CSR expected")
    dense = [0] * length
    for col, value in zip(tensor_data["sparse_col_indices"], tensor_data["data"], strict=True):
        dense[col] = value
    return dense


def decode_sample_request(payload: dict) -> dict:
    # [FUNCTION DOC] Validate a sampling body with the SDK model (user input error on failure) and
    # return the internal payload: model path, base model, sampling session id, sequence id, number
    # of samples (default 1), prompt token ids flattened from the prompt, sampling parameters
    # (empty if missing), prompt log-probs flag as boolean, and the top-k prompt log-prob count
    # (default 0; null counts as 0). Unsupported prompt chunk types raise a user input error.
    validate_against_sdk(tinker_types.SampleRequest, payload)
    return {
        "model_path": payload.get("model_path"),
        "base_model": payload.get("base_model"),
        "sampling_session_id": payload.get("sampling_session_id"),
        "seq_id": payload.get("seq_id"),
        "num_samples": payload.get("num_samples", 1),
        "prompt_tokens": model_input_tokens(payload["prompt"]),
        "sampling_params": payload.get("sampling_params") or {},
        "prompt_logprobs": bool(payload.get("prompt_logprobs")),
        "topk_prompt_logprobs": payload.get("topk_prompt_logprobs", 0) or 0,
    }


# -------- result rendering (JSON; proto_codec renders the binary forms) --------


def render_result(result: dict) -> dict:
    # [FUNCTION DOC]
    # Render a settled internal result mapping as the SDK's JSON response, by its operation.
    # Forward-backward and forward-only: type forward_backward, output type array record, one entry
    # per datum with a one-element loss tensor under the loss-sum key and a log-prob tensor, and
    # total metrics with the summed loss (float). Sample: sequences as is; prompt log-probs with
    # not-a-number replaced by null; top-k prompt log-probs as per-position lists of (token id,
    # log-prob) pairs dropping the padding sentinel (token 0 with the mask log-prob), a position
    # with no pairs becoming null. Create model: model id. Save state: type save_weights with the
    # path. Save sampler weights: path plus the sampling session id if present. Load state: type
    # only. Optimizer step: metrics. Unknown operation raises an assertion error.
    op = result["op"]
    if op in ("forward_backward", "forward_only"):
        outputs = result["outputs"]
        return {
            "type": "forward_backward",
            "loss_fn_output_type": "ArrayRecord",
            "loss_fn_outputs": [
                {"loss:sum": _tensor_json([output["loss"]]), "logprobs": _tensor_json(output["logprobs"])}
                for output in outputs
            ],
            "metrics": {"loss:sum": float(sum(output["loss"] for output in outputs))},
        }
    if op == "sample":
        rendered = {"type": "sample", "sequences": result["sequences"]}
        if result.get("prompt_logprobs") is not None:
            rendered["prompt_logprobs"] = [
                None if math.isnan(logprob) else logprob for logprob in result["prompt_logprobs"]
            ]
        if result.get("topk_prompt_logprobs") is not None:
            topk = result["topk_prompt_logprobs"]
            rendered["topk_prompt_logprobs"] = [
                [
                    (token_id, logprob)
                    for token_id, logprob in zip(ids, probs, strict=True)
                    if (token_id, logprob) != (0, MASK_LOGPROB)
                ]
                or None
                for ids, probs in zip(topk["token_ids"], topk["logprobs"], strict=True)
            ]
        return rendered
    if op == "create_model":
        return {"type": "create_model", "model_id": result["model_id"]}
    if op == "save_state":
        return {"type": "save_weights", "path": result["path"]}
    if op == "save_weights_for_sampler":
        rendered = {"type": "save_weights_for_sampler", "path": result["path"]}
        if "sampling_session_id" in result:
            rendered["sampling_session_id"] = result["sampling_session_id"]
        return rendered
    if op == "load_state":
        return {"type": "load_weights"}
    if op == "optim_step":
        return {"type": "optim_step", "metrics": result["metrics"]}
    raise AssertionError(f"unrenderable result op {op!r}")


def _tensor_json(values: list[float]) -> dict:
    # [FUNCTION DOC] HELPER: Wire tensor description of a flat float list: 32-bit float dtype, a
    # one-dimensional shape equal to the length, and the data list.
    return {"dtype": "float32", "shape": [len(values)], "data": values}
