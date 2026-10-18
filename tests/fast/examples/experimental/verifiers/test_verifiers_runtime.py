# [MODULE DOC]
# Tests of the verifiers example's sglang transport against the real published verifiers library
# (skipped on Python older than 3.11 and when verifiers 0.2.0 or newer, or renderers 0.1.8 or
# newer, is not installed). Covers: a minimal environment config with only a taskset id being
# accepted as a version-one (non-legacy) environment config whose environment builds the harbor
# taskset; the library's train client producing a chat response (message, prompt and completion
# token ids, completion log probs, valid raw chat response) when run through the transport against
# a faked http post and a fake renderer; and the train client rejecting non-chat dialects.
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Verifiers requires Python 3.11+", allow_module_level=True)

pytest.importorskip("verifiers", minversion="0.2.0")
pytest.importorskip("renderers", minversion="0.1.8")

from examples.experimental.verifiers.verifiers_rollout import MilesSGLangTransport
from verifiers.v1.clients.train import TrainClient
from verifiers.v1.dialects import ChatDialect, ResponsesDialect
from verifiers.v1.env import EnvConfig, Environment
from verifiers.v1.types import SamplingConfig


def _args(**overrides):
    # [FUNCTION DOC] HELPER: Builds an argument namespace for the transport with defaults (no
    # lora, max context 64, max response 8, skip special tokens, no stop settings, no model
    # routers, router at 127.0.0.1 port 30000 with round robin policy, no tokenizer path), updated
    # by the keyword overrides.
    values = {
        "lora_adapter_path": None,
        "lora_rank": 0,
        "rollout_max_context_len": 64,
        "rollout_max_prompt_len": None,
        "rollout_max_response_len": 8,
        "rollout_skip_special_tokens": True,
        "rollout_stop": None,
        "rollout_stop_token_ids": None,
        "sglang_model_routers": None,
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_policy": "round_robin",
        "sglang_router_port": 30000,
        "sglang_tokenizer_path": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_minimal_env_config_uses_the_v1_environment_contract():
    # [FUNCTION DOC] Validates a config with only a taskset id of harbor; builds the environment;
    # asserts the config is not legacy, its environment id is harbor, and the environment's taskset
    # type is named HarborTaskset.
    config = EnvConfig.model_validate({"taskset": {"id": "harbor"}})

    environment = Environment(config)

    assert config.is_legacy is False
    assert config.env_id == "harbor"
    assert type(environment.taskset).__name__ == "HarborTaskset"


class _Rendered:
    # [CLASS DOC] HELPER: Fake rendered prompt with token ids [10, 11], no multimodal data, both
    # tokens flagged as content, and a single message span covering both tokens.
    token_ids = [10, 11]
    multi_modal_data = None
    is_content = [True, True]

    @staticmethod
    def message_token_spans():
        # [FUNCTION DOC] HELPER: Returns a list with one span (0, 2).
        return [(0, 2)]


class _Renderer:
    # [CLASS DOC] HELPER: Fake renderer that supports tools and returns canned results after
    # asserting its inputs.
    supports_tools = True

    def render(self, messages, *, tools, add_generation_prompt):
        # [FUNCTION DOC] HELPER: Asserts the messages are a single user message with content
        # question, no tools, and generation prompt enabled; returns the fake rendered prompt.
        assert messages == [{"role": "user", "content": "question"}]
        assert tools is None
        assert add_generation_prompt is True
        return _Rendered()

    @staticmethod
    def get_stop_token_ids():
        # [FUNCTION DOC] HELPER: Returns the list [99].
        return [99]

    @staticmethod
    def parse_response(token_ids, *, tools):
        # [FUNCTION DOC] HELPER: Asserts the token ids are [20, 21] and no tools; returns a
        # message namespace with content answer, no reasoning content and no tool calls.
        assert token_ids == [20, 21]
        assert tools is None
        return SimpleNamespace(content="answer", reasoning_content=None, tool_calls=[])


@pytest.mark.asyncio
async def test_published_train_client_runs_through_miles_transport(monkeypatch):
    # [FUNCTION DOC] Fakes the http post (asserts no headers are passed; returns an sglang-shaped
    # response with 2 completion tokens 20 and 21, finish reason stop and their log probs -0.1 and
    # -0.2). Builds the library train client over the transport with a fake renderer as its pool
    # and asks for a chat response for one user message question with temperature 0.2, max tokens 2
    # and session id trace-id. Asserts the response message content is answer, prompt ids are
    # [10, 11], completion ids [20, 21], completion log probs [-0.1, -0.2], and the raw response
    # validates against the chat dialect.
    async def fake_post(_url, _payload, headers=None):
        # [FUNCTION DOC] HELPER: Asserts no headers were passed; returns the canned sglang-shaped response mapping.
        assert headers is None
        return {
            "request_id": "request-id",
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 20], [-0.2, 21]],
            },
        }

    monkeypatch.setattr("miles.utils.http_utils.post", fake_post)
    client = TrainClient(MilesSGLangTransport(_args()), renderer_model_name="test/model")
    client._pool = _Renderer()

    response = await client.get_response(
        ChatDialect(),
        {"messages": [{"role": "user", "content": "question"}]},
        "test/model",
        SamplingConfig(temperature=0.2, max_tokens=2),
        session_id="trace-id",
    )

    assert response.message.content == "answer"
    assert response.tokens.prompt_ids == [10, 11]
    assert response.tokens.completion_ids == [20, 21]
    assert response.tokens.completion_logprobs == [-0.1, -0.2]
    ChatDialect().validate_response(response.raw)


@pytest.mark.asyncio
async def test_published_train_client_rejects_non_chat_dialects():
    # [FUNCTION DOC] Asking the library train client for a response with the responses dialect
    # raises a not-implemented error mentioning the chat-completions dialect.
    client = TrainClient(MilesSGLangTransport(_args()), renderer_model_name="test/model")

    with pytest.raises(NotImplementedError, match="chat-completions dialect"):
        await client.get_response(
            ResponsesDialect(),
            {"input": "question"},
            "test/model",
            SamplingConfig(max_tokens=2),
        )
