# [MODULE DOC]
# Protobuf-side translation for the Tinker SDK wire format: decoding (optionally zstd-compressed)
# forward-backward requests with binary token and tensor encodings into the gateway's internal
# payload, and encoding forward-backward and sampling results as protobuf messages with packed
# little-endian arrays. Module constants: the protobuf content type string, a mapping from protobuf
# dtypes (32-bit float, 64-bit int, 32-bit int) to numpy dtypes, a mapping from stop reason names
# to protobuf enum values, and a mapping from result operation names to their encoder functions.
"""Translate Tinker SDK protobuf requests and results, including compressed requests."""

import numpy as np

from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import build_datum
from tinker.proto import tinker_public_pb2 as public_pb

PROTO_CONTENT_TYPE = "application/x-protobuf"

_PROTO_DTYPE_TO_NUMPY = {
    public_pb.DTYPE_FLOAT32: np.dtype(np.float32),
    public_pb.DTYPE_INT64: np.dtype(np.int64),
    public_pb.DTYPE_INT32: np.dtype(np.int32),
}
_STOP_REASON_TO_PROTO = {
    "stop": public_pb.STOP_REASON_STOP,
    "length": public_pb.STOP_REASON_LENGTH,
}


def maybe_decompress(body: bytes, content_encoding: str | None) -> bytes:
    # [FUNCTION DOC] If the content encoding is exactly zstd, decompress the body (the zstandard
    # library is imported lazily; library errors propagate); otherwise return the body unchanged.
    if content_encoding == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(body)
    return body


def decode_forward_backward_request(body: bytes) -> tuple[str, dict]:
    # [FUNCTION DOC] Parse the serialized request message (a malformed message raises the protobuf
    # decode error). The operation is forward-only if the message flag is set, else forward-
    # backward. Content errors (user input, key, type, value, index errors) during datum decoding
    # are converted to a payload with the model id, sequence id and a validation error text, so the
    # command fails in order; the operation is still returned.
    """ForwardBackwardRequest proto -> (op, internal payload)."""
    message = public_pb.ForwardBackwardRequest()
    message.ParseFromString(body)
    op = "forward_only" if message.forward_only else "forward_backward"
    try:
        return op, _decode_forward_backward(message)
    except (UserInputError, KeyError, TypeError, ValueError, IndexError) as error:
        return op, {"model_id": message.model_id, "seq_id": message.seq_id, "validation_error": str(error)}


def _decode_forward_backward(message) -> dict:
    # [FUNCTION DOC] HELPER: Build the internal payload from a parsed message. Per datum: each
    # model input chunk must be encoded text (else a user input error); its tokens are raw
    # little-endian 32-bit integers; loss inputs are decoded to lists by name; the datum is built
    # with its position. The loss config is the numeric config map, replaced entirely by the
    # version two map when that one is non-empty; each version two entry holds either a number or
    # text (neither set raises a user input error). Returns a mapping with model id, sequence id,
    # datums, loss function name and loss config.
    datums = []
    for index, datum in enumerate(message.data):
        tokens: list[int] = []
        for chunk in datum.model_input:
            if chunk.WhichOneof("chunk") != "encoded_text":
                raise UserInputError(f"unsupported model_input chunk type: {chunk.WhichOneof('chunk')}")
            tokens.extend(np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist())
        inputs = {name: _decode_tensor(name, tensor) for name, tensor in datum.loss_fn_inputs.items()}
        datums.append(build_datum(tokens, inputs, index))

    loss_fn_config = dict(message.loss_fn_config)
    # Tinker SDK's v2 protobuf config supports both numeric and string values.
    if message.loss_fn_config_v2:
        loss_fn_config = {}
        for name, value in message.loss_fn_config_v2.items():
            kind = value.WhichOneof("value")
            if kind is None:
                raise UserInputError(f"loss_fn_config[{name!r}]: missing number or text value")
            loss_fn_config[name] = getattr(value, kind)

    decoded = {
        "model_id": message.model_id,
        "seq_id": message.seq_id,
        "datums": datums,
        "loss_fn": message.loss_fn,
        "loss_fn_config": loss_fn_config,
    }
    return decoded


def _decode_tensor(name: str, tensor) -> list:
    # [FUNCTION DOC] HELPER: Decode a protobuf tensor to a Python list (name is for messages).
    # Unsupported dtype or more than one dimension raise user input errors. Dense encoding is read
    # as a raw array of the dtype. Sparse row encoding creates a zero array of the single shape
    # dimension and sets values at the 64-bit column indices (out-of-range indices raise an index
    # error; a non-1-D shape raises a value error). No encoding set raises a user input error.
    np_dtype = _PROTO_DTYPE_TO_NUMPY.get(tensor.dtype)
    if np_dtype is None:
        raise UserInputError(f"loss_fn_inputs[{name!r}]: unsupported tensor dtype {tensor.dtype}")
    if len(tensor.shape) > 1:
        raise UserInputError(f"loss_fn_inputs[{name!r}]: shape {list(tensor.shape)} is not supported (1-D only)")
    encoding = tensor.WhichOneof("encoding")
    if encoding == "dense":
        return np.frombuffer(tensor.dense, dtype=np_dtype).tolist()
    if encoding == "sparse_csr":
        (length,) = tensor.shape
        dense = np.zeros(length, dtype=np_dtype)
        cols = np.frombuffer(tensor.sparse_csr.col_indices, dtype=np.int64)
        dense[cols] = np.frombuffer(tensor.sparse_csr.values, dtype=np_dtype)
        return dense.tolist()
    raise UserInputError(f"loss_fn_inputs[{name!r}]: tensor without data")


# -------- result encoding --------


def encode_forward_backward_output(result: dict) -> bytes:
    # [FUNCTION DOC]
    # Serialize a forward-backward or forward-only result to a protobuf message. Sets the output
    # type name to array record and the total loss metric (sum of per-datum losses). A single record
    # holds the datum count and two batched 32-bit float fields: loss (one value per datum) and
    # log-probs (one flat array per datum). Each batched field stores its arrays concatenated as
    # bytes plus 64-bit cumulative byte offsets (count plus one entries starting at zero); no
    # trailing shape means each datum is a flat vector. Returns the serialized bytes.
    message = public_pb.ForwardBackwardOutput()
    message.loss_fn_output_type = "ArrayRecord"
    outputs = result["outputs"]
    message.metrics["loss:sum"] = float(sum(output["loss"] for output in outputs))

    record = message.loss_fn_outputs.add()
    record.num_datums = len(outputs)
    for field, arrays in (
        ("loss:sum", [np.asarray([output["loss"]], dtype=np.float32) for output in outputs]),
        ("logprobs", [np.asarray(output["logprobs"], dtype=np.float32) for output in outputs]),
    ):
        batched = record.fields[field]
        batched.dtype = public_pb.DTYPE_FLOAT32
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([array.nbytes for array in arrays], out=offsets[1:])
        batched.offsets = offsets.tobytes()
        batched.data = b"".join(array.tobytes() for array in arrays)
        # flat per-datum arrays: empty trailing_shape decodes as shape [n]
    return message.SerializeToString()


def encode_sample_response(result: dict) -> bytes:
    # [FUNCTION DOC] Serialize a sampling result to a protobuf message. Each sequence gets a stop
    # reason enum (unknown names raise a key error), its tokens as raw 32-bit integers and, if
    # present, log-probs as raw 32-bit floats. Prompt log-probs, if present, are raw 32-bit floats.
    # Top-k prompt log-probs, if present, set the prompt length and k from the token id array
    # shape (a two-dimensional array of 32-bit integers) and store token ids and log-probs as raw
    # bytes. Returns the serialized bytes.
    message = public_pb.SampleResponse()
    for sequence in result["sequences"]:
        out = message.sequences.add()
        out.stop_reason = _STOP_REASON_TO_PROTO[sequence["stop_reason"]]
        out.tokens = np.asarray(sequence["tokens"], dtype=np.int32).tobytes()
        if sequence.get("logprobs") is not None:
            out.logprobs = np.asarray(sequence["logprobs"], dtype=np.float32).tobytes()
    if result.get("prompt_logprobs") is not None:
        message.prompt_logprobs = np.asarray(result["prompt_logprobs"], dtype=np.float32).tobytes()
    if result.get("topk_prompt_logprobs") is not None:
        topk = result["topk_prompt_logprobs"]
        token_ids = np.asarray(topk["token_ids"], dtype=np.int32)
        message.topk_prompt_logprobs.prompt_length, message.topk_prompt_logprobs.k = token_ids.shape
        message.topk_prompt_logprobs.token_ids = token_ids.tobytes()
        message.topk_prompt_logprobs.logprobs = np.asarray(topk["logprobs"], dtype=np.float32).tobytes()
    return message.SerializeToString()


PROTO_ENCODERS = {
    "forward_backward": encode_forward_backward_output,
    "forward_only": encode_forward_backward_output,
    "sample": encode_sample_response,
}
