# [MODULE DOC] Tests of how a session server turns a client's chat request arguments into the
# request sent to the inference backend ("wire" body) plus the prompt rendering arguments.
# Three areas: (1) resolution by server configuration: the backend request always asks for log
# probabilities and meta info, never trims stop strings, mirrors the launch replay flags, records
# the client's stream and log probability intent separately, rejects client-supplied control fields
# (input ids, start lengths, adapter path) and malformed chat template kwargs, and handles the
# LoRA adapter path and model name suffix; (2) full request preparation with a token-in-token-out
# tokenizer: template arguments (tools stay top-level, client tools override launch tools,
# history compatibility checks), sampling defaults and sampling support replay, evaluation
# requests where replay features are forced off, and a model hook that may modify a deep copy of
# the arguments without mutating the client's input; (3) the template argument projection and the
# turn-argument filter. Tokenizers are real token-in-token-out wrappers around mock tokenizers.
"""Server constraints, full model argument resolution, and the renderer projection."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.errors import MessageValidationError
from miles.rollout.session.request_args import (
    ClientResponseIntent,
    apply_session_sampling_defaults,
    filter_turn_args,
    prepare_chat_request,
    resolve_request_args_by_config,
)
from miles.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, extract_template_args
from miles.utils.lora.utils import LORA_ADAPTER_NAME


class TestResolveRequestArgsByConfig:
    # [CLASS DOC] Tests of the config-driven request argument resolution, which returns the wire
    # mapping (the same mapping object, modified in place) and the client's response intent.
    def test_default_config_body_and_key_order(self):
        # [FUNCTION DOC] With the default config, resolves a request with model, temperature, an
        # unknown nested field and messages. Asserts the wire body keeps those entries first in
        # their original order, then adds log probabilities on, meta info on, no stop trim off and
        # both replay flags off; and that the returned mapping is the very input mapping.
        request_args = {"model": "m", "temperature": 0.7, "unknown": {"x": 1}, "messages": []}

        wire, _ = resolve_request_args_by_config(request_args, make_session_server_config())

        assert wire == {
            "model": "m",
            "temperature": 0.7,
            "unknown": {"x": 1},
            "messages": [],
            "logprobs": True,
            "return_meta_info": True,
            "no_stop_trim": False,
            "return_routed_experts": False,
            "return_indexer_topk": False,
        }
        assert list(wire)[:4] == ["model", "temperature", "unknown", "messages"]
        assert wire is request_args

    @pytest.mark.parametrize(
        "client_args, expected",
        [
            ({}, ClientResponseIntent(stream=False, logprobs=False)),
            ({"logprobs": True, "stream": True}, ClientResponseIntent(stream=True, logprobs=True)),
            ({"logprobs": False}, ClientResponseIntent(stream=False, logprobs=False)),
        ],
        ids=["omitted", "both-asked", "explicit-false"],
    )
    def test_response_intent_keeps_the_client_values_the_backend_request_overrides(self, client_args, expected):
        # [FUNCTION DOC] Parametrized over an omitted, fully asked, and explicitly false client
        # log probability/stream request. Asserts the wire body always has log probabilities and
        # meta info on and no stream entry, while the returned intent records the client's own
        # stream and log probability values (default false each).
        wire, intent = resolve_request_args_by_config(dict(client_args), make_session_server_config())

        assert wire["logprobs"] is True and wire["return_meta_info"] is True and "stream" not in wire
        assert intent == expected

    def test_replay_flags_follow_the_launch_flags(self):
        # [FUNCTION DOC] With routing and indexer replay enabled in the config, a client asking
        # for routed experts off still gets both the routed experts and indexer top-k flags on.
        config = make_session_server_config(use_rollout_routing_replay=True, use_rollout_indexer_replay=True)
        wire, _ = resolve_request_args_by_config({"return_routed_experts": False}, config)
        assert wire["return_routed_experts"] is True
        assert wire["return_indexer_topk"] is True

    @pytest.mark.parametrize("field", ["input_ids", "routed_experts_start_len", "logprob_start_len", "lora_path"])
    def test_client_tito_control_fields_are_rejected(self, field):
        # [FUNCTION DOC] Parametrized over input ids, routed experts start length, log probability
        # start length and adapter path. A client supplying any of them with a non-null value
        # gets a message validation error naming the field.
        with pytest.raises(MessageValidationError, match=f"{field}="):
            resolve_request_args_by_config({field: 1}, make_session_server_config())

    def test_lora_path_follows_lora_rollout_enabled(self):
        # [FUNCTION DOC] With LoRA rank 8 the wire body gets the adapter path set to the fixed
        # adapter name; with LoRA rank 8 but train-only LoRA, or without LoRA, there is no
        # adapter path.
        wire, _ = resolve_request_args_by_config({}, make_session_server_config(lora_rank=8))
        assert wire["lora_path"] == LORA_ADAPTER_NAME
        wire, _ = resolve_request_args_by_config({}, make_session_server_config(lora_rank=8, lora_train_only=True))
        assert "lora_path" not in wire
        wire, _ = resolve_request_args_by_config({}, make_session_server_config())
        assert "lora_path" not in wire

    def test_model_adapter_suffix_is_rejected_only_with_lora_rollout(self):
        # [FUNCTION DOC] A model name with an adapter suffix ("base:adapter") raises a message
        # validation error mentioning the LoRA adapter when LoRA rollout is on, and is passed
        # through unchanged when it is off.
        with pytest.raises(MessageValidationError, match="LoRA adapter"):
            resolve_request_args_by_config({"model": "base:adapter"}, make_session_server_config(lora_rank=8))
        wire, _ = resolve_request_args_by_config({"model": "base:adapter"}, make_session_server_config())
        assert wire["model"] == "base:adapter"

    @pytest.mark.parametrize("kwargs", ["oops", [], False, 1])
    def test_malformed_kwargs_are_refused(self, kwargs):
        # [FUNCTION DOC] Parametrized over a string, an empty list, false and an integer as the
        # chat template kwargs; each raises a message validation error saying they must be an
        # object.
        with pytest.raises(MessageValidationError, match="chat_template_kwargs must be an object"):
            resolve_request_args_by_config({"chat_template_kwargs": kwargs}, make_session_server_config())

    def test_tools_must_be_top_level(self):
        # [FUNCTION DOC] Tools placed inside the chat template kwargs raise a message validation
        # error saying tools belong at the top level.
        with pytest.raises(MessageValidationError, match="tools belongs at the top level"):
            resolve_request_args_by_config({"chat_template_kwargs": {"tools": []}}, make_session_server_config())

    def test_control_field_errors_precede_template_shape_errors(self):
        # [FUNCTION DOC] A request with both a forbidden input ids field and malformed template
        # kwargs reports the input ids error (control field checks run first).
        with pytest.raises(MessageValidationError, match="input_ids="):
            resolve_request_args_by_config(
                {"input_ids": [1], "chat_template_kwargs": "oops"}, make_session_server_config()
            )

    @pytest.mark.parametrize("field", ["input_ids", "routed_experts_start_len", "logprob_start_len", "lora_path"])
    def test_null_control_fields_are_removed(self, field):
        # [FUNCTION DOC] Parametrized over the four control fields. A field set to none is
        # accepted and removed from the wire body.
        wire, _ = resolve_request_args_by_config({field: None}, make_session_server_config())
        assert field not in wire

    def test_selected_lora_path_is_accepted(self):
        # [FUNCTION DOC] With LoRA rank 8, a client adapter path equal to the fixed adapter name
        # is accepted and stays in the wire body.
        wire, _ = resolve_request_args_by_config(
            {"lora_path": LORA_ADAPTER_NAME}, make_session_server_config(lora_rank=8)
        )
        assert wire["lora_path"] == LORA_ADAPTER_NAME


class TestPrepareChatRequest:
    # [CLASS DOC] Tests of full request preparation, which returns an object with the wire body,
    # the prompt rendering (template) arguments and the response intent. Class constants: launch
    # template kwargs (thinking disabled) and a one-function tool list.
    LAUNCH = {"enable_thinking": False}
    TOOLS = [{"type": "function", "function": {"name": "get_weather"}}]

    @staticmethod
    def _tito(**kwargs) -> TITOTokenizer:
        # [FUNCTION DOC] HELPER: Static method building a token-in-token-out tokenizer around a
        # mock tokenizer, forwarding extra keyword options (such as chat template kwargs).
        return TITOTokenizer(MagicMock(), **kwargs)

    def test_wire_carries_the_template_args_the_prompt_is_rendered_with(self):
        # [FUNCTION DOC] A request with tools and its own thinking template kwarg (true, overriding
        # the launch value false) is prepared. Asserts the template arguments are the client's
        # thinking value plus the tools, the wire body has the tools and the client's template
        # kwargs only, and log probabilities are on (config-driven resolution ran on the same
        # body).
        client_args = {"messages": [], "tools": self.TOOLS, "chat_template_kwargs": {"enable_thinking": True}}

        prepared = prepare_chat_request(
            client_args,
            self._tito(chat_template_kwargs=self.LAUNCH),
            config=make_session_server_config(),
            turn_args=None,
        )

        assert prepared.template_args == {"enable_thinking": True, "tools": self.TOOLS}
        assert prepared.body["tools"] == self.TOOLS
        assert prepared.body["chat_template_kwargs"] == {"enable_thinking": True}
        assert prepared.body["logprobs"] is True  # resolve_request_args_by_config ran on the same body

    def test_sampling_replay_uses_session_defaults_and_keeps_request_filters(self):
        # [FUNCTION DOC] With sampling defaults (temperature 0.7, top-p 0.95, top-k 32) and
        # sampling support replay on, a request giving only top-p 0.8 and custom params is
        # prepared. Asserts the sampling mask is requested, temperature and top-k come from the
        # defaults, top-p keeps the request's 0.8, and the custom params are untouched.
        prepared = prepare_chat_request(
            {"top_p": 0.8, "custom_params": {"caller_option": "kept"}},
            self._tito(),
            config=make_session_server_config(),
            turn_args=None,
            sampling_defaults={"temperature": 0.7, "top_p": 0.95, "top_k": 32},
            sampling_support_replay=True,
        )

        assert prepared.body["return_sampling_mask"] is True
        assert prepared.body["temperature"] == 0.7
        assert prepared.body["top_p"] == 0.8
        assert prepared.body["top_k"] == 32
        assert prepared.body["custom_params"] == {"caller_option": "kept"}

    def test_sampling_replay_cannot_be_disabled_by_a_request(self):
        # [FUNCTION DOC] A request asking for the sampling mask off still gets it on when sampling
        # support replay is enabled for the session.
        prepared = prepare_chat_request(
            {"return_sampling_mask": False},
            self._tito(),
            config=make_session_server_config(),
            turn_args=None,
            sampling_defaults={"temperature": 0.7, "top_p": 0.95, "top_k": 32},
            sampling_support_replay=True,
        )

        assert prepared.body["return_sampling_mask"] is True

    def test_bounded_request_requires_session_sampling_replay(self):
        # [FUNCTION DOC] A request with top-k 32 while the session has no sampling replay raises a
        # message validation error about bounded training-request sampling.
        with pytest.raises(MessageValidationError, match="bounded training-request sampling"):
            prepare_chat_request(
                {"temperature": 1.0, "top_p": 1.0, "top_k": 32},
                self._tito(),
                config=make_session_server_config(),
                turn_args=None,
            )

    @pytest.mark.parametrize("client_tools", [None, [], [{"function": {"name": "override"}}]])
    def test_launch_tools_stay_top_level_and_client_tools_override_them(self, client_tools):
        # [FUNCTION DOC] Parametrized over client tools of none (omitted), an empty list and a
        # one-tool list. Launch template kwargs include the standard tools. Prepares a first turn
        # and asserts the effective tools are the client's tools when non-empty, otherwise the
        # launch tools; the wire template kwargs hold only the launch non-tool options; and the
        # template arguments are those options plus the tools. A continuation turn with no
        # client arguments and the first turn's body as history reproduces the same tools and
        # template arguments. The launch kwargs mapping is never mutated.
        launch_kwargs = {**self.LAUNCH, "tools": self.TOOLS}
        original = deepcopy(launch_kwargs)
        client_args = {} if client_tools is None else {"tools": client_tools}
        tito = self._tito(chat_template_kwargs=launch_kwargs)
        prepared = prepare_chat_request(client_args, tito, config=make_session_server_config(), turn_args=None)
        expected_tools = client_tools or self.TOOLS
        assert prepared.body["tools"] == expected_tools
        assert prepared.body["chat_template_kwargs"] == self.LAUNCH
        assert prepared.template_args == {**self.LAUNCH, "tools": expected_tools}
        continued = prepare_chat_request({}, tito, config=make_session_server_config(), turn_args=prepared.body)
        assert continued.body["tools"] == expected_tools
        assert continued.template_args == prepared.template_args
        assert launch_kwargs == original

    def test_inherited_tools_reach_the_wire_and_empty_args_are_retained(self):
        # [FUNCTION DOC] First scenario: recorded history carries the launch template kwargs and
        # tools while the new request passes an empty tool list; asserts the wire body still
        # carries the historical tools and template kwargs. Second scenario: a request with an
        # empty tool list and empty template kwargs, no history and no launch kwargs; asserts the
        # template arguments are empty, the wire tools are none and the wire template kwargs stay
        # an empty mapping.
        recorded = {"chat_template_kwargs": self.LAUNCH, "tools": self.TOOLS}
        prepared = prepare_chat_request(
            {"messages": [], "tools": []},
            self._tito(chat_template_kwargs=self.LAUNCH),
            config=make_session_server_config(),
            turn_args=recorded,
        )
        assert prepared.body["tools"] == self.TOOLS
        assert prepared.body["chat_template_kwargs"] == self.LAUNCH

        prepared = prepare_chat_request(
            {"messages": [], "tools": [], "chat_template_kwargs": {}},
            self._tito(),
            config=make_session_server_config(),
            turn_args=None,
        )
        assert prepared.template_args == {}
        assert prepared.body["tools"] is None
        assert prepared.body["chat_template_kwargs"] == {}

    def test_a_refused_request_is_a_400(self):
        # [FUNCTION DOC] A request adding tools when the history had none raises a message
        # validation error saying tools changed, whose status code is 400.
        with pytest.raises(MessageValidationError, match="tools changed") as excinfo:
            prepare_chat_request(
                {"messages": [], "tools": self.TOOLS},
                self._tito(chat_template_kwargs=self.LAUNCH),
                config=make_session_server_config(),
                turn_args={"chat_template_kwargs": self.LAUNCH},
            )
        assert excinfo.value.status_code == 400

    def test_model_receives_and_updates_full_args_without_mutating_client_input(self):
        # [FUNCTION DOC] Uses a tokenizer subclass whose request resolution hook asserts it
        # receives the full backend arguments (log probabilities on, no stream key), calls the base
        # hook (which returns the same mapping), then changes the temperature, appends to a nested
        # unknown field and adds a template kwarg. Asserts the client's original mapping is
        # unchanged (deep-copied), the wire body holds the model's changes, the template arguments
        # include the added kwarg, and the response intent records the client's stream request.
        class Model(TITOTokenizer):
            # [CLASS DOC] HELPER: Tokenizer subclass with a modifying request resolution hook.
            def resolve_request_args(self, request_args, *, turn_args):
                # [FUNCTION DOC] HELPER: Checks the incoming arguments, delegates to the base
                # hook, then mutates and returns the same mapping (temperature 0.3, extra nested
                # value 2, a model option template kwarg set).
                assert request_args["logprobs"] is True
                assert "stream" not in request_args
                assert super().resolve_request_args(request_args, turn_args=turn_args) is request_args
                request_args["temperature"] = 0.3
                request_args["unknown"]["values"].append(2)
                request_args["chat_template_kwargs"]["model_option"] = True
                return request_args

        client_args = {"temperature": 0.8, "unknown": {"values": [1]}, "stream": True}
        original = deepcopy(client_args)
        prepared = prepare_chat_request(
            client_args,
            Model(MagicMock(), chat_template_kwargs={"model_option": False}),
            config=make_session_server_config(),
            turn_args=None,
        )
        assert client_args == original
        assert prepared.body["temperature"] == 0.3
        assert prepared.body["unknown"] == {"values": [1, 2]}
        assert prepared.template_args == {"model_option": True}
        assert prepared.response_intent.stream is True


def test_template_projection_only_selects_render_fields():
    # [FUNCTION DOC] Projects a request with model, temperature, input ids, tools and template
    # kwargs to its template arguments and asserts only the template kwargs and tools remain; a
    # request with just a temperature projects to an empty mapping.
    request_args = {
        "model": "m",
        "temperature": 0.7,
        "input_ids": [1],
        "tools": [{"name": "f"}],
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert extract_template_args(request_args) == {"enable_thinking": True, "tools": [{"name": "f"}]}
    assert extract_template_args({"temperature": 0.7}) == {}


@pytest.mark.parametrize("field", ["input_ids", "messages"])
def test_filter_turn_args_drops_payload_before_copying_and_isolates_metadata(field):
    # [FUNCTION DOC] Parametrized over the input ids and messages fields. The payload under test
    # is a mock whose deep copy raises an assertion error. Filtering the turn arguments must drop
    # the payload before copying (so no error), return the remaining entries, and return a deep
    # copy: mutating the nested template kwargs of the result leaves the original untouched.
    # The original still holds the same payload object.
    payload = MagicMock()
    payload.__deepcopy__ = MagicMock(side_effect=AssertionError("excluded payload must not be copied"))
    turn_args = {field: payload, "temperature": 0.7, "chat_template_kwargs": {"nested": [1]}}

    metadata = filter_turn_args(turn_args)

    assert metadata == {"temperature": 0.7, "chat_template_kwargs": {"nested": [1]}}
    metadata["chat_template_kwargs"]["nested"].append(2)
    assert turn_args["chat_template_kwargs"] == {"nested": [1]}
    assert turn_args[field] is payload


def test_filter_turn_args_accepts_an_explicit_drop_list():
    # [FUNCTION DOC] With an explicit drop list of only input ids, the filter keeps messages and
    # the seed; with an empty drop list it keeps everything (the result equals the input).
    turn_args = {"input_ids": [1], "messages": [{"role": "user", "content": "hi"}], "seed": 42}
    assert filter_turn_args(turn_args, drop_keys=("input_ids",)) == {"messages": turn_args["messages"], "seed": 42}
    assert filter_turn_args(turn_args, drop_keys=()) == turn_args


@pytest.mark.parametrize("sampling", [{}, {"temperature": 0.0}, {"temperature": 1.2, "top_p": 0.8, "top_k": 16}])
def test_eval_keeps_sampling_resolution_and_overrides_model_replay(sampling):
    # [FUNCTION DOC] Parametrized over client sampling choices (none, temperature 0, and
    # temperature 1.2 with top-p 0.8 and top-k 16). The model hook fills missing temperature
    # 0.7, top-p 0.9 and top-k 32, forces the sampling mask, routed experts and indexer top-k
    # flags on, and sets a routed experts start length of 7. The client also sends the mask flag
    # and a start length of 99. The config enables routing and indexer replay and the call is an
    # evaluation with sampling support replay on. Asserts the sampling values equal the model
    # defaults overridden by the client's own choices; all three replay flags are false (forced
    # off for evaluation); the start length is absent; and neither the client arguments nor the
    # history mapping were mutated.
    class Model(TITOTokenizer):
        # [CLASS DOC] HELPER: Tokenizer subclass whose hook fills sampling defaults and forces
        # replay-related fields that evaluation must override.
        def resolve_request_args(self, request_args, *, turn_args):
            # [FUNCTION DOC] HELPER: Delegates to the base hook, sets default temperature, top-p
            # and top-k when missing, forces the three replay flags to true and the routed experts
            # start length to 7, and returns the mapping.
            request_args = super().resolve_request_args(request_args, turn_args=turn_args)
            request_args.setdefault("temperature", 0.7)
            request_args.setdefault("top_p", 0.9)
            request_args.setdefault("top_k", 32)
            request_args.update(return_sampling_mask=True, return_routed_experts=True, return_indexer_topk=True)
            request_args["routed_experts_start_len"] = 7
            return request_args

    config = make_session_server_config(use_rollout_routing_replay=True, use_rollout_indexer_replay=True)
    client_args = {**sampling, "return_sampling_mask": True, "routed_experts_start_len": 99}
    history = {"temperature": 0.3, "chat_template_kwargs": {}}
    original, original_history = deepcopy(client_args), deepcopy(history)
    prepared = prepare_chat_request(
        client_args,
        Model(MagicMock()),
        config=config,
        turn_args=history,
        evaluation=True,
        sampling_support_replay=True,
    )
    assert {key: prepared.body[key] for key in ("temperature", "top_p", "top_k")} == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 32,
        **sampling,
    }
    assert all(
        prepared.body[key] is False for key in ("return_sampling_mask", "return_routed_experts", "return_indexer_topk")
    )
    assert "routed_experts_start_len" not in prepared.body
    assert client_args == original and history == original_history


@pytest.mark.parametrize("evaluation", [False, True])
def test_session_sampling_defaults_fill_only_omitted_fields(evaluation):
    # [FUNCTION DOC] Parametrized over evaluation. The client gives a temperature (0.1 for
    # evaluation, else 0.6), a none top-p and a top-k (-1 for evaluation, else 10). With session
    # defaults (0.6, 0.9, 20) and sampling replay on for training only, asserts the body keeps the
    # client's temperature and top-k, fills the none top-p with the default 0.9, and the client
    # mapping is not mutated.
    temperature = 0.1 if evaluation else 0.6
    top_k = -1 if evaluation else 10
    client_args = {"temperature": temperature, "top_p": None, "top_k": top_k}
    original = deepcopy(client_args)
    prepared = prepare_chat_request(
        client_args,
        TITOTokenizer(MagicMock()),
        config=make_session_server_config(),
        turn_args=None,
        evaluation=evaluation,
        sampling_defaults={"temperature": 0.6, "top_p": 0.9, "top_k": 20},
        sampling_support_replay=not evaluation,
    )
    assert {key: prepared.body[key] for key in ("temperature", "top_p", "top_k")} == {
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": top_k,
    }
    assert client_args == original


def test_without_session_sampling_defaults_omitted_fields_stay_unset():
    # [FUNCTION DOC] Preparing an empty request without session sampling defaults leaves
    # temperature, top-p and top-k out of the body.
    prepared = prepare_chat_request(
        {}, TITOTokenizer(MagicMock()), config=make_session_server_config(), turn_args=None
    )
    assert not {"temperature", "top_p", "top_k"} & set(prepared.body)


def test_apply_session_sampling_defaults_treats_none_as_unset():
    # [FUNCTION DOC] Applies defaults (0.6, 0.9, 20) in place to a mapping with a none
    # temperature and top-k -1. Asserts the none temperature is replaced by 0.6, the explicit -1
    # is kept, and the missing top-p is added as 0.9.
    request_args = {"temperature": None, "top_k": -1}
    apply_session_sampling_defaults(request_args, {"temperature": 0.6, "top_p": 0.9, "top_k": 20})
    assert request_args == {"temperature": 0.6, "top_k": -1, "top_p": 0.9}


@pytest.mark.parametrize("evaluation", [False, True])
def test_template_compatibility_applies_to_train_and_eval(evaluation):
    # [FUNCTION DOC] Parametrized over evaluation. A request adding a tool when the history has
    # none raises a message validation error saying tools changed, in both training and
    # evaluation.
    with pytest.raises(MessageValidationError, match="tools changed"):
        prepare_chat_request(
            {"tools": [{"type": "function", "function": {"name": "new_tool"}}]},
            TITOTokenizer(MagicMock()),
            config=make_session_server_config(),
            turn_args={"chat_template_kwargs": {}},
            evaluation=evaluation,
        )
