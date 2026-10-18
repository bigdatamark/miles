# [MODULE DOC] Unit tests of the tinker runtime (the single translator between the gateway's
# internal language and the miles trainer / SGLang engines), with the trainer call replaced by
# fakes. Covers: building trainer train data from slot-tagged datums (rollout batch keys, optional
# keys, dynamic global batch size); merging replicated worker reports ordered by sample index;
# building SGLang generate requests from sampling parameters (required max tokens, parameter name
# mapping, stop ids versus strings, empty stop disabling EOS, top-k prompt logprobs); parsing
# engine responses (sequence tokens and stop reason, NaN-padded prompt logprobs, ragged top-k
# padding verified through the SDK protobuf round trip, aborted samples reported as errors);
# forward-only loss selection; per-sample seed offsets; padding batches to a data-parallel
# multiple; and passing actor error verdicts through. Requires the tinker SDK to be installed
# (imported without a skip guard) and torch.
"""Runtime translation preserves datum order, sampling parameters, and token logprobs."""

import pytest
import torch

from miles.tinker.core.types import UserInputError
from miles.tinker.runtime import (
    MilesBackend,
    _build_train_data,
    _pad_to_dp_multiple,
    _prompt_logprobs,
    _slot_failure,
    _to_sequence,
    _topk_prompt_logprobs,
)
from miles.tinker.server.proto_codec import encode_sample_response
from tinker.proto.response_conv import deserialize_sample_response


def _datum(tokens: list[int], **extra) -> dict:
    # [FUNCTION DOC] HELPER: Builds an internal datum mapping: the tokens, a target length of one
    # less than the token count, target tokens equal to all but the first token, plus the extra
    # fields.
    return {"tokens": tokens, "target_len": len(tokens) - 1, "target_tokens": tokens[1:], **extra}


class TestBuildTrainData:
    # [CLASS DOC] Tests of converting slot-tagged datums into the trainer's rollout batch mapping.
    def test_datums_become_rollout_batch_keys(self):
        # [FUNCTION DOC] Two datums for slots 3 and 5 (3 and 2 tokens) produce per-datum token
        # lists, target tokens, response lengths 2 and 1, total lengths 3 and 2, all-ones loss
        # masks, adapter slots 3 and 5, sample indices 0 and 1 and a dynamic global batch size of 2.
        train_data = _build_train_data([(3, _datum([1, 2, 3])), (5, _datum([4, 5]))])
        assert train_data["tokens"] == [[1, 2, 3], [4, 5]]
        assert train_data["target_tokens"] == [[2, 3], [5]]
        assert train_data["response_lengths"] == [2, 1]
        assert train_data["total_lengths"] == [3, 2]
        assert train_data["loss_masks"] == [[1, 1], [1]]
        assert train_data["adapter_slots"] == [3, 5]
        assert train_data["sample_indices"] == [0, 1]
        assert train_data["dynamic_global_batch_size"] == 2

    def test_optional_datum_keys_map_to_batch_names(self):
        # [FUNCTION DOC] A datum with weights, advantages and sampling logprobs maps them to the
        # batch keys loss weights, advantages and rollout log probs (each a list of one list).
        train_data = _build_train_data(
            [(0, _datum([1, 2], weights=[1.0], advantages=[2.0], sampling_logprobs=[-0.5]))]
        )
        assert train_data["loss_weights"] == [[1.0]]
        assert train_data["advantages"] == [[2.0]]
        assert train_data["rollout_log_probs"] == [[-0.5]]


async def test_forward_backward_merges_worker_replicas():
    # [FUNCTION DOC] With the backend's trainer call replaced by a fake returning two identical
    # rank reports (per-datum records given out of order by sample index), forward/backward for two
    # datums returns outputs ordered by sample index (loss 1.0 with logprob -0.1, then loss 2.0
    # with logprob -0.2), each logprob as a plain list.
    backend = MilesBackend(trainer=None, router_url="http://router")
    per_datum = [
        {"sample_index": 1, "loss": 2.0, "logprobs": torch.tensor([-0.2])},
        {"sample_index": 0, "loss": 1.0, "logprobs": torch.tensor([-0.1])},
    ]

    async def fake_call_trainer(method, batch_id, train_data):
        # [FUNCTION DOC] HELPER: Async fake trainer call; ignores its arguments and returns two
        # worker reports that both carry the same per-datum records.
        return [{"per_datum": per_datum}, {"per_datum": per_datum}]  # two ranks report the same datums

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2])), (0, _datum([3, 4]))], "cross_entropy", {})
    assert outputs == [
        {"loss": 1.0, "logprobs": [pytest.approx(-0.1)]},
        {"loss": 2.0, "logprobs": [pytest.approx(-0.2)]},
    ]


class TestGenerateRequest:
    # [CLASS DOC] Tests of building engine generate requests from sampling payloads.
    def _request(self, params: dict, **payload_extra) -> dict:
        # [FUNCTION DOC] HELPER: Builds a sample payload (prompt tokens 1, 2, the sampling params,
        # one sample, no prompt logprobs, zero top-k, plus extra fields) and returns the engine
        # generate request the backend builds for adapter name m@1.
        payload = {
            "prompt_tokens": [1, 2],
            "sampling_params": params,
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
            **payload_extra,
        }
        return MilesBackend(None, "http://router")._generate_request(payload, lora_name="m@1")

    def test_max_tokens_is_required(self):
        # [FUNCTION DOC] Sampling params without max tokens raise a user input error mentioning
        # max_tokens.
        with pytest.raises(UserInputError, match="max_tokens"):
            self._request({})

    def test_params_map_to_sglang_names(self):
        # [FUNCTION DOC] Max tokens 8 becomes max new tokens 8, seed 7 becomes the sampling seed,
        # the lora path is the adapter name, and return-logprob is true.
        request = self._request({"max_tokens": 8, "temperature": 0.0, "seed": 7})
        assert request["sampling_params"]["max_new_tokens"] == 8
        assert request["sampling_params"]["sampling_seed"] == 7
        assert request["lora_path"] == "m@1"
        assert request["return_logprob"] is True

    def test_stop_splits_token_ids_from_strings(self):
        # [FUNCTION DOC] A stop list of integers becomes stop token ids; a stop list of strings
        # becomes stop strings.
        by_ids = self._request({"max_tokens": 1, "stop": [7, 8]})
        by_text = self._request({"max_tokens": 1, "stop": ["\n"]})
        assert by_ids["sampling_params"]["stop_token_ids"] == [7, 8]
        assert by_text["sampling_params"]["stop"] == ["\n"]

    def test_an_empty_stop_list_disables_eos(self):
        # [FUNCTION DOC] An explicitly empty stop list sets ignore-eos and includes neither stop
        # strings nor stop token ids; an absent stop keeps default EOS behavior (no ignore-eos).
        request = self._request({"max_tokens": 1, "stop": []})
        assert request["sampling_params"]["ignore_eos"] is True
        assert "stop" not in request["sampling_params"] and "stop_token_ids" not in request["sampling_params"]
        default = self._request({"max_tokens": 1})
        assert "ignore_eos" not in default["sampling_params"], "stop=None keeps the default EOS behavior"

    def test_topk_requests_prompt_logprobs(self):
        # [FUNCTION DOC] Requesting 3 top-k prompt logprobs sets the logprob start length to 0 and
        # the top-logprobs count to 3.
        request = self._request({"max_tokens": 1}, topk_prompt_logprobs=3)
        assert request["logprob_start_len"] == 0
        assert request["top_logprobs_num"] == 3


class TestEngineResponseParsing:
    # [CLASS DOC] Tests of parsing engine responses into gateway results.
    def test_to_sequence_reads_tokens_and_stop_reason(self):
        # [FUNCTION DOC] A response with two output token logprob pairs and finish type length
        # yields tokens 11, 12, logprobs -0.1, -0.2 and stop reason length.
        response = {
            "meta_info": {
                "output_token_logprobs": [(-0.1, 11), (-0.2, 12)],
                "finish_reason": {"type": "length"},
            }
        }
        sequence = _to_sequence(response)
        assert (sequence["tokens"], sequence["logprobs"], sequence["stop_reason"]) == (
            [11, 12],
            [-0.1, -0.2],
            "length",
        )

    def test_prompt_logprobs_pad_the_unscored_first_token(self):
        # [FUNCTION DOC] Input token logprobs with an unscored (none) first position give a NaN
        # first value (checked by self-inequality) and -0.5 for the second.
        response = {"meta_info": {"input_token_logprobs": [(None, 1), (-0.5, 2)]}}
        first, second = _prompt_logprobs(response)
        assert first != first, "the unscored position must be NaN"
        assert second == -0.5

    def test_topk_pads_ragged_positions(self):
        # [FUNCTION DOC] Input top logprobs with an unscored first position and one entry for the
        # second, requested with k=2, give token ids [[0, 0], [5, 0]] (padded with zeros) and a
        # logprob -0.1 at the second position's first slot. The padded result is then encoded as a
        # sample response and decoded with the SDK: the first position decodes to none and the
        # second has exactly one (token 5, logprob about -0.1) pair, so padding is dropped.
        response = {"meta_info": {"input_top_logprobs": [None, [(-0.1, 5)]]}}
        topk = _topk_prompt_logprobs(response, k=2)
        assert topk["token_ids"] == [[0, 0], [5, 0]]
        assert topk["logprobs"][1][0] == -0.1
        result = {"sequences": [], "topk_prompt_logprobs": topk}
        decoded = deserialize_sample_response(encode_sample_response(result)).topk_prompt_logprobs
        assert decoded[0] is None
        assert len(decoded[1]) == 1
        token_id, logprob = decoded[1][0]
        assert token_id == 5
        assert logprob == pytest.approx(-0.1)


async def test_forward_only_runs_the_requested_loss():
    # [FUNCTION DOC] With a fake trainer call recording the method name and the loss function in
    # the train data, forward-only with the importance sampling loss calls the forward_only method
    # with that loss and returns one output with loss 3.0 and logprob about -0.3.
    backend = MilesBackend(trainer=None, router_url="http://router")
    captured = {}

    async def fake_call_trainer(method, batch_id, train_data):
        # [FUNCTION DOC] HELPER: Async fake trainer call; records the method and loss function and
        # returns one worker report with one per-datum record (sample index 0, loss 3.0).
        captured["method"] = method
        captured["loss_fn"] = train_data["loss_fn"]
        return [{"per_datum": [{"sample_index": 0, "loss": 3.0, "logprobs": torch.tensor([-0.3])}]}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_only(1, [(0, _datum([1, 2]))], "importance_sampling", {})
    assert captured == {"method": "forward_only", "loss_fn": "importance_sampling"}
    assert outputs == [{"loss": 3.0, "logprobs": [pytest.approx(-0.3)]}]


def test_a_pinned_seed_still_gets_one_model_queue_per_sample():
    # [FUNCTION DOC] Deriving per-sample requests from one with sampling seed 7 yields seeds 7, 8,
    # 9 for samples 0-2; the original request's seed stays 7 (not mutated); and a request without a
    # seed gets none.
    # KNOWN ISSUE: the test name speaks of a model queue per sample, but the test checks only the
    # per-sample seed offsets.
    from miles.tinker.runtime import _with_sample_seed

    request = {"sampling_params": {"sampling_seed": 7, "temperature": 0.0}}
    assert [_with_sample_seed(request, i)["sampling_params"]["sampling_seed"] for i in range(3)] == [7, 8, 9]
    assert request["sampling_params"]["sampling_seed"] == 7
    assert "sampling_seed" not in _with_sample_seed({"sampling_params": {}}, 2)["sampling_params"]


class TestPadToDpMultiple:
    # [CLASS DOC] Tests of padding a datum batch to a multiple of the data-parallel size.
    def test_padding_replicates_the_last_datum_with_a_padding_marker(self):
        # [FUNCTION DOC] Padding one datum to a multiple of 4 gives 4 entries; each filler has the
        # same slot and tokens as the last datum and a truthy padding marker, and the original datum
        # is not mutated (it has no padding key).
        slot_datums = [(0, _datum([1, 2, 3], weights=[1.0, 1.0], advantages=[2.0, 2.0]))]
        padded = _pad_to_dp_multiple(slot_datums, 4)
        assert len(padded) == 4
        for slot, filler in padded[1:]:
            assert slot == 0 and filler["tokens"] == [1, 2, 3] and filler["padding"]
        assert "padding" not in slot_datums[0][1], "the original datum must not be mutated"


async def test_forward_backward_pads_the_batch_and_drops_padding_outputs():
    # [FUNCTION DOC] With data-parallel size 2 and a fake trainer call that records the train data
    # and returns one record per padded datum, a single-datum forward/backward is sent as a batch
    # of 2 (dynamic global batch size 2) with loss masks [[1], [0]] (padding masked out), and only
    # the one real output is returned.
    backend = MilesBackend(trainer=None, router_url="http://router", dp_size=2)
    seen = {}

    async def fake_call_trainer(method, batch_id, train_data):
        # [FUNCTION DOC] HELPER: Async fake trainer call; copies the train data into the shared
        # mapping and returns one worker report with a record per datum (sample index and loss
        # equal to the position, a fixed logprob).
        seen.update(train_data)
        per_datum = [
            {"sample_index": index, "loss": float(index), "logprobs": torch.tensor([-0.1])}
            for index in range(len(train_data["tokens"]))
        ]
        return [{"per_datum": per_datum}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2], weights=[1.0]))], "cross_entropy", {})
    assert seen["dynamic_global_batch_size"] == 2, "a singleton batch must be padded to the DP size"
    assert seen["loss_masks"] == [[1], [0]], "the zero mask removes padding from every loss term"
    assert len(outputs) == 1, "padding outputs are dropped"


def test_an_actor_error_verdict_survives_runtime_translation():
    # [FUNCTION DOC] Given worker reports [none, error mapping], the slot-failure helper returns
    # the very same error mapping object.
    failure = {"error": "bad shard"}
    assert _slot_failure([None, failure]) is failure


def test_an_aborted_sample_fails_instead_of_passing_as_a_stop():
    # [FUNCTION DOC] A response whose finish type is abort converts to a sequence result carrying
    # an error text that mentions abort.
    """A truncated sequence fed to RL as a completed sample corrupts training data silently."""
    response = {"meta_info": {"output_token_logprobs": [(-0.1, 11)], "finish_reason": {"type": "abort"}}}
    assert "abort" in _to_sequence(response)["error"]
