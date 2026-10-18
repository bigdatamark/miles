# [MODULE DOC] Unit tests of the tinker server's JSON encoding layer: decoding forward/backward
# requests (shifted datum becomes tokens plus target tokens, forward-only selects a different
# operation, length mismatches are reported as a validation error entry rather than raised,
# explicit targets pass through), mapping wire input names to internal datum fields, optimizer
# parameter defaults and unknown-key validation, rejection of unknown fields on other commands,
# dense and compressed-sparse-row tensor data conversion, sample request defaults, and rendering
# of results into the SDK's shapes.
"""JSON decoding validates datum inputs and renders SDK result shapes."""

from miles.tinker.server.encoding import (
    ADAM_PARAM_DEFAULTS,
    build_datum,
    decode_command,
    decode_sample_request,
    render_result,
    tensor_data_to_list,
)


def _datum(tokens: list[int], **extra_inputs) -> dict:
    # [FUNCTION DOC] HELPER: Builds a wire-format datum from a token list: the model input is one
    # encoded-text chunk with all tokens except the last, and the loss function inputs hold the
    # target tokens (all tokens except the first) plus the extra inputs.
    return {
        "model_input": {"chunks": [{"type": "encoded_text", "tokens": tokens[:-1]}]},
        "loss_fn_inputs": {"target_tokens": tokens[1:], **extra_inputs},
    }


class TestDecodeForwardBackward:
    # [CLASS DOC] Tests of decoding forward/backward wire requests into internal commands.
    def _decode(self, datum, forward_only=False):
        # [FUNCTION DOC] HELPER: Wraps the one datum in a forward/backward payload for model
        # "model", sequence 1, cross-entropy loss and the forward-only flag, decodes it as a
        # forward/backward command and returns the (operation, decoded payload) pair.
        payload = {
            "model_id": "model",
            "seq_id": 1,
            "forward_only": forward_only,
            "forward_backward_input": {"data": [datum], "loss_fn": "cross_entropy"},
        }
        return decode_command("forward_backward", payload)

    def test_a_shifted_datum_becomes_one_datum(self):
        # [FUNCTION DOC] A wire datum from tokens 1-4 with three unit weights decodes to operation
        # forward_backward and exactly one internal datum with the full four tokens, target length
        # 3, target tokens 2, 3, 4 and the weights.
        op, decoded = self._decode(_datum([1, 2, 3, 4], weights=[1.0, 1.0, 1.0]))
        assert op == "forward_backward"
        assert decoded["datums"] == [
            {"tokens": [1, 2, 3, 4], "target_len": 3, "target_tokens": [2, 3, 4], "weights": [1.0, 1.0, 1.0]}
        ]

    def test_forward_only_selects_the_op(self):
        # [FUNCTION DOC] The forward-only flag makes the decoded operation forward_only.
        op, _ = self._decode(_datum([1, 2, 3]), forward_only=True)
        assert op == "forward_only"

    def test_a_length_mismatch_is_rejected(self):
        # [FUNCTION DOC] With one target token for a two-token input, decoding does not raise but
        # the decoded payload carries a validation error text that mentions length.
        datum = _datum([1, 2, 3])
        datum["loss_fn_inputs"]["target_tokens"] = [2]
        _, decoded = self._decode(datum)
        assert "length" in decoded["validation_error"]

    def test_explicit_targets_pass_through(self):
        # [FUNCTION DOC] Explicit target tokens 0 and 9 for a three-token wire datum yield internal
        # tokens 1, 2, 9 (the last input token replaced by the final target to complete the
        # sequence) and target tokens 0, 9.
        """rl_loop pads the prompt region with dummy zero targets; labels need not be the shifted sequence."""
        datum = _datum([1, 2, 3])
        datum["loss_fn_inputs"]["target_tokens"] = [0, 9]
        _, decoded = self._decode(datum)
        assert decoded["datums"][0]["tokens"] == [1, 2, 9]
        assert decoded["datums"][0]["target_tokens"] == [0, 9]


def test_build_datum_maps_the_wire_input_names():
    # [FUNCTION DOC] Building a datum from tokens 1, 2 with wire inputs named logprobs and
    # advantages maps the logprobs input to the internal sampling logprobs field and keeps the
    # advantages.
    datum = build_datum([1, 2], {"target_tokens": [2, 3], "logprobs": [0.5, 0.5], "advantages": [1.0, -1.0]}, 0)
    assert datum["sampling_logprobs"] == [0.5, 0.5]
    assert datum["advantages"] == [1.0, -1.0]


class TestAdamParams:
    # [CLASS DOC] Tests of optimizer parameter decoding.
    def test_defaults_fill_missing_keys(self):
        # [FUNCTION DOC] A partial optimizer parameter mapping (only learning rate 3e-4) decodes
        # keeping that value and filling in the full set of default keys.
        _, decoded = decode_command(
            "optim_step", {"model_id": "m", "seq_id": 1, "adam_params": {"learning_rate": 3e-4}}
        )
        assert decoded["adam_params"]["learning_rate"] == 3e-4
        assert set(decoded["adam_params"]) == set(ADAM_PARAM_DEFAULTS)

    def test_unknown_keys_are_rejected(self):
        # [FUNCTION DOC] An unknown optimizer key (momentum) yields a validation error text that
        # names the key.
        """The SDK's AdamParams model is the contract; a key it lacks must not be silently dropped."""
        _, decoded = decode_command("optim_step", {"model_id": "m", "seq_id": 1, "adam_params": {"momentum": 0.9}})
        assert "momentum" in decoded["validation_error"]


def test_an_off_model_key_is_rejected():
    # [FUNCTION DOC] A save-state payload with an unexpected key (bogus) yields a validation error
    # text naming that key.
    _, decoded = decode_command("save_state", {"model_id": "m", "seq_id": 1, "bogus": 1})
    assert "bogus" in decoded["validation_error"]


class TestTensorData:
    # [CLASS DOC] Tests of converting wire tensor data to plain lists.
    def test_dense_tensor_data(self):
        # [FUNCTION DOC] Dense tensor data with values 1.0, 2.0 converts to that list.
        assert tensor_data_to_list({"data": [1.0, 2.0], "shape": [2]}) == [1.0, 2.0]

    def test_csr_expands_to_dense(self):
        # [FUNCTION DOC] Compressed-sparse-row data of shape 4 with row pointer [0, 2], column
        # indices [1, 3] and values [5.0, 7.0] expands to [0, 5.0, 0, 7.0].
        sparse = {"shape": [4], "sparse_crow_indices": [0, 2], "sparse_col_indices": [1, 3], "data": [5.0, 7.0]}
        assert tensor_data_to_list(sparse) == [0, 5.0, 0, 7.0]


def test_decode_sample_request_defaults():
    # [FUNCTION DOC] A minimal sample request (one encoded-text prompt chunk, max tokens 1)
    # decodes with 1 sample, prompt tokens 1, 2 and zero top-k prompt logprobs.
    decoded = decode_sample_request(
        {"prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2]}]}, "sampling_params": {"max_tokens": 1}}
    )
    assert decoded["num_samples"] == 1
    assert decoded["prompt_tokens"] == [1, 2]
    assert decoded["topk_prompt_logprobs"] == 0


class TestRenderResult:
    # [CLASS DOC] Tests of rendering internal results into SDK response shapes.
    def test_forward_backward_renders_per_datum_records(self):
        # [FUNCTION DOC] A forward/backward result with one output (loss 2.0, logprobs 0.1, 0.2)
        # renders output type ArrayRecord, metrics with the summed loss 2.0, and per-datum
        # logprobs as a float32 array record of shape 2 with those values.
        rendered = render_result({"op": "forward_backward", "outputs": [{"loss": 2.0, "logprobs": [0.1, 0.2]}]})
        assert rendered["loss_fn_output_type"] == "ArrayRecord"
        assert rendered["metrics"] == {"loss:sum": 2.0}
        assert rendered["loss_fn_outputs"][0]["logprobs"] == {"dtype": "float32", "shape": [2], "data": [0.1, 0.2]}
