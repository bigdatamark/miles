# [MODULE DOC] Tests (module skipped when the tinker SDK is not installed) that the server's
# protobuf codec round-trips with the real SDK's protobuf converters and agrees with the JSON
# path: an SDK-serialized forward/backward request decodes into the same internal datums as the
# equivalent JSON request, sparse tensors decode dense, compressed bodies decompress, the SDK can
# parse the server's forward/backward output and sample response encodings, and content errors
# (missing targets, bad shape) keep the model-queue envelope (operation, model id, sequence id)
# while carrying a validation error. Constant: the token list 3, 1, 4, 1, 5 used for requests.
"""Protobuf and JSON preserve the same datums and SDK result shapes."""

import pytest

tinker = pytest.importorskip("tinker")

from miles.tinker.server.encoding import decode_command  # noqa: E402
from miles.tinker.server.proto_codec import (  # noqa: E402
    decode_forward_backward_request,
    encode_forward_backward_output,
    encode_sample_response,
    maybe_decompress,
)
from tinker import types  # noqa: E402
from tinker.proto.request_conv import forward_backward_request_to_proto  # noqa: E402
from tinker.proto.response_conv import deserialize_forward_backward_output, deserialize_sample_response  # noqa: E402

TOKENS = [3, 1, 4, 1, 5]


def _sdk_request(loss_fn_inputs: dict) -> bytes:
    # [FUNCTION DOC] HELPER: Builds, with the SDK's own types, a forward/backward request for model
    # model-x, sequence 7, cross-entropy loss with config beta 0.5 and one datum (input tokens
    # all but the last, target tokens all but the first plus the extra inputs), converts it to
    # protobuf and returns the serialized bytes.
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(TOKENS[:-1]),
        loss_fn_inputs={"target_tokens": TOKENS[1:], **loss_fn_inputs},
    )
    request = types.ForwardBackwardRequest(
        model_id="model-x",
        seq_id=7,
        forward_backward_input=types.ForwardBackwardInput(
            data=[datum], loss_fn="cross_entropy", loss_fn_config={"beta": 0.5}
        ),
    )
    return forward_backward_request_to_proto(request).SerializeToString()


def test_an_sdk_proto_request_decodes_to_internal_datums():
    # [FUNCTION DOC] Decoding the SDK request with weights gives operation forward_backward, model
    # id model-x, sequence 7, loss config beta 0.5 and one internal datum (all five tokens, target
    # length 4, target tokens, weights).
    op, decoded = decode_forward_backward_request(_sdk_request({"weights": [1.0, 0.0, 1.0, 1.0]}))

    assert (op, decoded["model_id"], decoded["seq_id"]) == ("forward_backward", "model-x", 7)
    assert decoded["loss_fn_config"] == {"beta": 0.5}
    assert decoded["datums"] == [
        {"tokens": TOKENS, "target_len": 4, "target_tokens": TOKENS[1:], "weights": [1.0, 0.0, 1.0, 1.0]}
    ]


def test_both_wire_halves_produce_the_same_datums():
    # [FUNCTION DOC] The datums decoded from the protobuf request equal those decoded from an
    # equivalent JSON forward/backward payload.
    proto_datums = decode_forward_backward_request(_sdk_request({"weights": [1.0, 1.0, 1.0, 1.0]}))[1]["datums"]
    _, json_decoded = decode_command(
        "forward_backward",
        {
            "model_id": "model-x",
            "seq_id": 7,
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [{"type": "encoded_text", "tokens": TOKENS[:-1]}]},
                        "loss_fn_inputs": {"target_tokens": TOKENS[1:], "weights": [1.0, 1.0, 1.0, 1.0]},
                    }
                ],
                "loss_fn": "cross_entropy",
            },
        },
    )
    assert proto_datums == json_decoded["datums"]


def test_a_sparse_tensor_decodes_dense():
    # [FUNCTION DOC] A sparse float32 weights tensor of shape 4 with one stored value 2.0 at
    # column 2 decodes to the dense list 0.0, 0.0, 2.0, 0.0.
    sparse = types.TensorData(
        dtype="float32", shape=[4], data=[2.0], sparse_crow_indices=[0, 1], sparse_col_indices=[2]
    )
    _, decoded = decode_forward_backward_request(_sdk_request({"weights": sparse}))
    assert decoded["datums"][0]["weights"] == [0.0, 0.0, 2.0, 0.0]


def test_a_zstd_body_decompresses():
    # [FUNCTION DOC] Skipped without the zstandard package. A zstd-compressed request body
    # decompresses to the original bytes when the encoding is zstd; with no encoding the body is
    # returned unchanged.
    zstandard = pytest.importorskip("zstandard")
    body = _sdk_request({"weights": [1.0, 1.0, 1.0, 1.0]})
    compressed = zstandard.ZstdCompressor().compress(body)

    assert maybe_decompress(compressed, "zstd") == body
    assert maybe_decompress(body, None) == body


def test_the_sdk_parses_our_forward_backward_output():
    # [FUNCTION DOC] Encodes a forward/backward result with two outputs (losses 2.5 and 1.0) and
    # parses it with the SDK's deserializer: the summed-loss metric is 3.5, each per-datum loss
    # record holds its own loss, and the first datum's logprobs match.
    encoded = encode_forward_backward_output(
        {
            "op": "forward_backward",
            "outputs": [{"loss": 2.5, "logprobs": [-0.1, -0.2]}, {"loss": 1.0, "logprobs": [-0.3]}],
        }
    )
    parsed = deserialize_forward_backward_output(encoded)

    assert parsed.metrics["loss:sum"] == pytest.approx(3.5)
    assert [list(record["loss:sum"].data) for record in parsed.loss_fn_outputs] == [[2.5], [1.0]]
    assert list(parsed.loss_fn_outputs[0]["logprobs"].data) == pytest.approx([-0.1, -0.2])


def test_the_sdk_parses_our_sample_response():
    # [FUNCTION DOC] Encodes a sample result (one sequence with tokens 5, 6, logprobs and stop
    # reason length, prompt logprobs, and top-k prompt logprobs) and parses it with the SDK's
    # deserializer: sequence tokens, stop reason, logprobs and prompt logprobs match the input.
    encoded = encode_sample_response(
        {
            "op": "sample",
            "sequences": [{"sequence_id": "s", "tokens": [5, 6], "logprobs": [-0.5, -0.6], "stop_reason": "length"}],
            "prompt_logprobs": [-1.0, -2.0],
            "topk_prompt_logprobs": {"token_ids": [[1, 2]], "logprobs": [[-0.1, -0.2]]},
        }
    )
    parsed = deserialize_sample_response(encoded)

    sequence = parsed.sequences[0]
    assert (list(sequence.tokens), sequence.stop_reason) == ([5, 6], "length")
    assert list(sequence.logprobs) == pytest.approx([-0.5, -0.6])
    assert list(parsed.prompt_logprobs) == pytest.approx([-1.0, -2.0])


@pytest.mark.parametrize("forward_only", [False, True])
@pytest.mark.parametrize("invalid", ["missing_targets", "bad_shape"])
def test_content_errors_preserve_the_model_queue_envelope(forward_only, invalid):
    # [FUNCTION DOC] Parametrized over forward-only on/off and two kinds of invalid content. Parses
    # the SDK request into the protobuf message, sets the forward-only flag, then either deletes
    # the target tokens input or appends a bogus dimension to the weights shape, re-serializes and
    # decodes. Asserts the decoded operation matches the flag, the model id and sequence id are
    # still present, and a non-empty validation error is reported.
    from tinker.proto import tinker_public_pb2 as public_pb

    request = public_pb.ForwardBackwardRequest()
    request.ParseFromString(_sdk_request({"weights": [1.0] * 4}))
    request.forward_only = forward_only
    if invalid == "missing_targets":
        del request.data[0].loss_fn_inputs["target_tokens"]
    else:
        request.data[0].loss_fn_inputs["weights"].shape.append(2)
    op, decoded = decode_forward_backward_request(request.SerializeToString())
    assert op == ("forward_only" if forward_only else "forward_backward")
    assert decoded["model_id"] == "model-x" and decoded["seq_id"] == 7
    assert decoded["validation_error"]
