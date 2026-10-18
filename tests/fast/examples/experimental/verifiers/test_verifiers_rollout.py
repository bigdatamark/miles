# [MODULE DOC]
# Tests for the verifiers rollout adapter module of the verifiers example (imported under its
# package path; the whole module is skipped on Python older than 3.11). Covers: validation of the
# argument namespace before any episode runs; the config path read from an environment variable
# and the toml-only config loader; library version compatibility rejections (installed versions
# are patched); the transport's finish reason handling; renderer identity inference from standard
# checkpoint paths; the train client wrapper (local tokenizer, renderer pool, and mapping of
# unsupported tool, dialect and relay requests to configuration errors with status 400); conversion
# of traces into training samples (tokens, loss mask from the sampled mask, log probs, reward,
# routing key, status, structured prompts, named rewards, error status taking precedence over
# truncation, graph branches rejected); group conversion and the standard post-processing and
# dynamic filter on converted groups; group reward sample count validation; episode ownership of
# group reward computation; sampling config and eval args construction; the sglang transport
# (request translation to the generate endpoint, multimodal rejection, bounded seen-session
# cache, lazy router address resolution, default model router, eval transport using live router
# args); evaluation rejecting group reward models and extracting structured rewards; and the
# rollout function contract with the base class. Argument namespaces and trace objects are built
# by helpers using plain namespaces; adapters are created without running their constructors.
import asyncio
import sys
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

if sys.version_info < (3, 11):
    pytest.skip("Verifiers requires Python 3.11+", allow_module_level=True)

import examples.experimental.verifiers.verifiers_rollout as verifiers_rollout
from examples.experimental.verifiers.verifiers_rollout import (
    MilesSGLangTransport,
    VerifiersRolloutFn,
    _config_path,
    _finish_reason,
    _import_verifiers,
    _load_config_data,
    _make_eval_args,
    _raise_for_unsupported_trace_errors,
    _renderer_identity,
    _trace_eval_reward,
    _train_client,
    _validate_args,
    _validate_group_reward_sample_counts,
    trace_to_sample,
    trace_to_samples,
)

from miles.rollout.base_types import BaseRolloutFn, RolloutFnConstructorInput
from miles.utils.types import Sample


def _args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds an argument namespace with defaults for the rollout adapter
    # (no lora, no reward key, max context 64, max response 8, skip special tokens, no stop
    # settings, no model routers, router at 127.0.0.1 port 30000 with round robin policy, no
    # tokenizer path), updated by the keyword overrides.
    values = {
        "lora_adapter_path": None,
        "lora_rank": 0,
        "reward_key": None,
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


def _branch(*, index=0, token_ids=None, sampled_mask=None, logprobs=None):
    # [FUNCTION DOC] HELPER: Builds a fake trace branch with an index, token ids (default five
    # ids: two prompt, three completion), a sampled mask (default false, false, true, false, true)
    # and log probs (default 0.0, 0.0, -0.1, 0.0, -0.2).
    return SimpleNamespace(
        index=index,
        token_ids=[10, 11, 20, 21, 22] if token_ids is None else token_ids,
        sampled_mask=[False, False, True, False, True] if sampled_mask is None else sampled_mask,
        logprobs=[0.0, 0.0, -0.1, 0.0, -0.2] if logprobs is None else logprobs,
    )


def _trace(**overrides):
    # [FUNCTION DOC] HELPER: Builds a fake trace with defaults: id trace-1, one default branch, a
    # task whose data has prompt solve this and idx task-1, rewards score 1.25 and bonus 0.75,
    # metrics of turns 2.0, stop condition done, no error, not truncated, total reward 2.0, last
    # reply answer, 2 turns; updated by the keyword overrides.
    values = {
        "id": "trace-1",
        "branches": [_branch()],
        "task": SimpleNamespace(data=SimpleNamespace(prompt="solve this", idx="task-1")),
        "rewards": {"score": 1.25, "bonus": 0.75},
        "metrics": {"turns": 2.0},
        "stop_condition": "done",
        "error": None,
        "has_error": False,
        "is_truncated": False,
        "reward": 2.0,
        "last_reply": "answer",
        "num_turns": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _wiring_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds an argument namespace for wiring validation with all
    # unsupported features off (global dataset, partial rollout, multimodal keys none, chat
    # template path none, opd, routing replay, indexer replay), updated by the keyword overrides.
    values = {
        "rollout_global_dataset": False,
        "partial_rollout": False,
        "multimodal_keys": None,
        "chat_template_path": None,
        "use_opd": False,
        "use_rollout_routing_replay": False,
        "use_rollout_indexer_replay": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_supported_wiring_passes_validation():
    # [FUNCTION DOC] The default wiring arguments validate without raising.
    _validate_args(_wiring_args())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rollout_global_dataset": True}, "--disable-rollout-global-dataset"),
        ({"partial_rollout": True}, "cannot be resumed"),
        ({"multimodal_keys": {"image": "image"}}, "text-only renderer inputs"),
        ({"chat_template_path": "/tmp/custom.jinja"}, "custom\n?\s*Jinja template"),
        ({"use_opd": True}, "--use-opd"),
        ({"use_rollout_routing_replay": True}, "--use-rollout-routing-replay"),
        ({"use_rollout_indexer_replay": True}, "--use-rollout-indexer-replay"),
    ],
)
def test_unsupported_wiring_fails_before_any_episode_runs(overrides, message):
    # [FUNCTION DOC] Parametrized: each unsupported option (global dataset, partial rollout,
    # multimodal keys, custom chat template, opd, routing replay, indexer replay) makes validation
    # raise a value error whose message matches the given text.
    # KNOWN ISSUE: the custom chat template pattern is a plain (non-raw) string containing a
    # backslash-s escape, which is an invalid string escape sequence and warns on newer Python.
    with pytest.raises(ValueError, match=message):
        _validate_args(_wiring_args(**overrides))


def test_config_path_comes_from_the_environment(monkeypatch):
    # [FUNCTION DOC] With the verifiers config variable set to a path, the config path function returns that path.
    monkeypatch.setenv("VERIFIERS_CONFIG", "/tmp/vf.toml")

    assert _config_path() == "/tmp/vf.toml"


def test_missing_config_env_var_names_the_variable(monkeypatch):
    # [FUNCTION DOC] With the variable unset, the config path function raises a value error naming the variable.
    monkeypatch.delenv("VERIFIERS_CONFIG", raising=False)

    with pytest.raises(ValueError, match="VERIFIERS_CONFIG"):
        _config_path()


def test_config_loader_uses_verifiers_toml_format(tmp_path):
    # [FUNCTION DOC] Writes a toml file with a taskset id and asserts the loader returns the
    # nested mapping.
    path = tmp_path / "verifiers.toml"
    path.write_text('[taskset]\nid = "gsm8k-v1"\n')

    assert _load_config_data(path) == {"taskset": {"id": "gsm8k-v1"}}


def test_config_loader_rejects_non_toml_formats(tmp_path):
    # [FUNCTION DOC] A yaml-named file makes the loader raise a value error mentioning Verifiers TOML.
    path = tmp_path / "verifiers.yaml"
    path.write_text("taskset: gsm8k-v1\n")

    with pytest.raises(ValueError, match="Verifiers TOML"):
        _load_config_data(path)


def test_verifiers_0_2_1_is_rejected_with_compatibility_reason(monkeypatch):
    # [FUNCTION DOC] Patches the installed-version lookup so verifiers reports 0.2.1; importing
    # verifiers through the adapter raises a runtime error explaining that sglang pins the openai
    # package.
    monkeypatch.setattr(
        "examples.experimental.verifiers.verifiers_rollout._installed_version",
        lambda package: {"verifiers": "0.2.1"}[package],
    )

    with pytest.raises(RuntimeError, match="SGLang 0.5.15 pins OpenAI"):
        _import_verifiers()


def test_renderers_0_1_11_is_rejected_with_compatibility_reason(monkeypatch):
    # [FUNCTION DOC] Patches installed versions to verifiers 0.2.0 and renderers 0.1.11;
    # importing raises a runtime error naming the supported renderers range (at least 0.1.8, below
    # 0.1.11) and that the newer version removes the renderer pool apis.
    versions = {"verifiers": "0.2.0", "renderers": "0.1.11"}
    monkeypatch.setattr(
        "examples.experimental.verifiers.verifiers_rollout._installed_version",
        versions.__getitem__,
    )

    with pytest.raises(
        RuntimeError,
        match=r"renderers>=0\.1\.8,<0\.1\.11.*removes the renderer pool APIs",
    ):
        _import_verifiers()


def test_transport_does_not_treat_aborted_generation_as_complete():
    # [FUNCTION DOC] A response whose meta info finish reason type is abort makes the finish
    # reason function raise a runtime error mentioning aborted.
    with pytest.raises(RuntimeError, match="aborted"):
        _finish_reason({"meta_info": {"finish_reason": {"type": "abort"}}})


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("/models/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507"),
        (
            "/cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/revision",
            "Qwen/Qwen3-4B-Instruct-2507",
        ),
        ("/models/private-finetune", None),
    ],
)
def test_renderer_identity_is_inferred_from_standard_checkpoint_paths(checkpoint, expected):
    # [FUNCTION DOC] Skipped unless renderers (at least 0.1.8) is installed. Parametrized: a plain
    # model folder path and a huggingface cache snapshot path both map to the Qwen organization
    # model id; an unknown fine-tune path maps to none.
    pytest.importorskip("renderers", minversion="0.1.8")

    assert _renderer_identity(checkpoint) == expected


def test_train_client_uses_local_tokenizer_with_inferred_renderer_identity(monkeypatch):
    # [FUNCTION DOC] Skipped unless renderers is installed. Replaces the renderer pool class,
    # renderer factory and tokenizer loader with recording fakes, and uses a fake base train client
    # class. Builds the train client for a huggingface cache snapshot checkpoint with a custom
    # tokenizer path and pool size 3, then asks it for its renderer pool with chat template kwargs
    # enable_thinking false. Asserts the pool is the fake pool class and the recorded values:
    # pool size 3; tokenizer loaded from the custom tokenizer path; renderer identity derived from
    # the checkpoint path (Qwen model id); config none; the template kwargs; renderer created.
    renderers = pytest.importorskip("renderers", minversion="0.1.8")
    checkpoint = "/cache/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/revision"
    seen = {}

    class BaseTrainClient:
        # [CLASS DOC] HELPER: Fake base train client storing its constructor arguments and an empty renderer pool slot.
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            # [FUNCTION DOC] HELPER: Stores the client, pool size, config and renderer model name; the pool starts as none.
            self.openai = openai
            self.pool_size = pool_size
            self.config = config
            self.renderer_model_name = renderer_model_name
            self._pool = None

    class RendererPool:
        # [CLASS DOC] HELPER: Fake renderer pool that records its size and one renderer made by its factory.
        def __init__(self, factory, size):
            # [FUNCTION DOC] HELPER: Records the size and the result of calling the factory once.
            seen["size"] = size
            seen["renderer"] = factory()

    def load_tokenizer(source):
        # [FUNCTION DOC] HELPER: Records the source and returns a namespace whose name-or-path is the source.
        seen["source"] = source
        return SimpleNamespace(name_or_path=source)

    def create_renderer(tokenizer, config, *, chat_template_kwargs=None):
        # [FUNCTION DOC] HELPER: Records the tokenizer's name-or-path as identity, the config and
        # the template kwargs; returns the string renderer.
        seen["identity"] = tokenizer.name_or_path
        seen["config"] = config
        seen["kwargs"] = chat_template_kwargs
        return "renderer"

    monkeypatch.setattr(renderers, "RendererPool", RendererPool)
    monkeypatch.setattr(renderers, "create_renderer", create_renderer)
    monkeypatch.setattr("renderers.base.load_tokenizer", load_tokenizer)
    runtime = SimpleNamespace(TrainClient=BaseTrainClient)

    args = _args(sglang_tokenizer_path="/models/custom-tokenizer")
    client = _train_client(runtime, args, checkpoint, pool_size=3)
    pool = client._renderer_pool(checkpoint, chat_template_kwargs={"enable_thinking": False})

    assert isinstance(pool, RendererPool)
    assert seen == {
        "size": 3,
        "source": "/models/custom-tokenizer",
        "identity": "Qwen/Qwen3-4B-Instruct-2507",
        "config": None,
        "kwargs": {"enable_thinking": False},
        "renderer": "renderer",
    }


@pytest.mark.asyncio
async def test_train_client_reports_unsupported_tool_renderer_as_configuration_error():
    # [FUNCTION DOC] Skipped unless renderers is installed. The fake base client's get-response
    # raises a value error saying the renderer pool does not support tools; asserts the wrapped
    # client raises a provider error mentioning the sglang tokenizer path option, with status code 400.
    pytest.importorskip("renderers", minversion="0.1.8")

    class ProviderError(Exception):
        # [CLASS DOC] HELPER: Fake provider error carrying a status code keyword argument.
        def __init__(self, message, *, status_code):
            # [FUNCTION DOC] HELPER: Passes the message to the base exception and stores the status code.
            super().__init__(message)
            self.status_code = status_code

    class BaseTrainClient:
        # [CLASS DOC] HELPER: Fake base train client whose get-response always raises a value error about tools.
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            # [FUNCTION DOC] HELPER: Ignores all arguments.
            pass

        async def get_response(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises a value error saying the renderer pool does not support tools.
            raise ValueError("RendererPool does not support tools.")

    runtime = SimpleNamespace(ProviderError=ProviderError, TrainClient=BaseTrainClient)
    client = _train_client(runtime, _args(), "/models/private-finetune", pool_size=1)

    with pytest.raises(ProviderError, match="--sglang-tokenizer-path") as error:
        await client.get_response()

    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_train_client_reports_unsupported_dialect_as_configuration_error():
    # [FUNCTION DOC] Skipped unless renderers is installed. The fake base client's get-response
    # raises a not-implemented error about the chat-completions dialect; asserts the wrapped
    # client raises a provider error saying the request is not supported, with status code 400.
    pytest.importorskip("renderers", minversion="0.1.8")

    class ProviderError(Exception):
        # [CLASS DOC] HELPER: Fake provider error carrying a status code keyword argument.
        def __init__(self, message, *, status_code):
            # [FUNCTION DOC] HELPER: Passes the message to the base exception and stores the status code.
            super().__init__(message)
            self.status_code = status_code

    class BaseTrainClient:
        # [CLASS DOC] HELPER: Fake base train client whose get-response always raises a not-implemented error.
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            # [FUNCTION DOC] HELPER: Ignores all arguments.
            pass

        async def get_response(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises a not-implemented error saying only the chat-completions dialect is supported.
            raise NotImplementedError("only the chat-completions dialect is supported")

    runtime = SimpleNamespace(ProviderError=ProviderError, TrainClient=BaseTrainClient)
    client = _train_client(runtime, _args(), "/models/test", pool_size=1)

    with pytest.raises(ProviderError, match="does not support this request") as error:
        await client.get_response()

    assert error.value.status_code == 400


@pytest.mark.parametrize(("method", "message"), [("relay", "streaming"), ("relay_aux", "auxiliary")])
@pytest.mark.asyncio
async def test_train_client_reports_unsupported_relay_paths_as_configuration_errors(method, message):
    # [FUNCTION DOC] Skipped unless renderers is installed. Parametrized over the relay method
    # (message streaming) and the auxiliary relay method (message auxiliary): calling it on the
    # wrapped client raises a provider error mentioning the message, with status code 400.
    pytest.importorskip("renderers", minversion="0.1.8")

    class ProviderError(Exception):
        # [CLASS DOC] HELPER: Fake provider error carrying a status code keyword argument.
        def __init__(self, text, *, status_code):
            # [FUNCTION DOC] HELPER: Passes the text to the base exception and stores the status code.
            super().__init__(text)
            self.status_code = status_code

    class BaseTrainClient:
        # [CLASS DOC] HELPER: Fake base train client with no behavior beyond construction.
        def __init__(self, openai, pool_size, config=None, renderer_model_name=None):
            # [FUNCTION DOC] HELPER: Ignores all arguments.
            pass

    runtime = SimpleNamespace(ProviderError=ProviderError, TrainClient=BaseTrainClient)
    client = _train_client(runtime, _args(), "/models/test", pool_size=1)

    with pytest.raises(ProviderError, match=message) as error:
        await getattr(client, method)()

    assert error.value.status_code == 400


def test_trace_to_sample_preserves_training_fields_and_verifiers_reward():
    # [FUNCTION DOC] Converts the default trace with group index 3 and index 9 and asserts: group
    # and sample indices; prompt text; tokens equal to the branch ids; response text equal to the
    # last reply; response length 3; loss mask [1,0,1] (sampled mask over the completion part);
    # rollout log probs [-0.1, 0.0, -0.2]; reward 2.0; routing key equal to the trace id; status
    # completed; metadata task index; and that the sample passes its own validation.
    sample = trace_to_sample(_args(), _trace(), group_index=3, index=9)

    assert sample.group_index == 3
    assert sample.index == 9
    assert sample.prompt == "solve this"
    assert sample.tokens == [10, 11, 20, 21, 22]
    assert sample.response == "answer"
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 0, 1]
    assert sample.rollout_log_probs == [-0.1, 0.0, -0.2]
    assert sample.reward == 2.0
    assert sample.routing_key == "trace-1"
    assert sample.status == Sample.Status.COMPLETED
    assert sample.metadata["verifiers"]["task_index"] == "task-1"
    sample.validate()


def test_trace_to_sample_preserves_named_rewards_when_reward_key_is_set():
    # [FUNCTION DOC] With reward key score, asserts the sample reward is a mapping of the named
    # rewards plus a reward entry with the total (2.0), and the sample's reward value for those
    # args is the score 1.25.
    args = _args(reward_key="score")

    sample = trace_to_sample(args, _trace(), group_index=0, index=0)

    assert sample.reward == {"score": 1.25, "bonus": 0.75, "reward": 2.0}
    assert sample.get_reward_value(args) == 1.25


def test_trace_to_sample_serializes_structured_prompt_messages():
    # [FUNCTION DOC] A prompt made of message objects with a dump method (asserted to be called
    # with json mode and none-excluded options) converts to a list of plain role/content mappings.
    class Message:
        # [CLASS DOC] HELPER: Fake message whose dump method checks its keyword arguments and returns a user message mapping.
        def model_dump(self, **kwargs):
            # [FUNCTION DOC] HELPER: Asserts the keyword arguments are mode json and exclude none true; returns the user message mapping.
            assert kwargs == {"mode": "json", "exclude_none": True}
            return {"role": "user", "content": "solve this"}

    trace = _trace(task=SimpleNamespace(data=SimpleNamespace(prompt=[Message()], idx="task-1")))

    sample = trace_to_sample(_args(), trace, group_index=0, index=0)

    assert sample.prompt == [{"role": "user", "content": "solve this"}]


def test_trace_to_sample_marks_error_before_truncation():
    # [FUNCTION DOC] A trace that both has an error and is truncated converts to a failed sample
    # (error wins over truncation) whose metadata holds the dumped error mapping.
    error = SimpleNamespace(model_dump=lambda **_kwargs: {"type": "ProviderError"})

    sample = trace_to_sample(
        _args(),
        _trace(error=error, has_error=True, is_truncated=True),
        group_index=0,
        index=0,
    )

    assert sample.status == Sample.Status.FAILED
    assert sample.metadata["verifiers"]["error"] == {"type": "ProviderError"}


def test_failed_eval_trace_with_missing_named_reward_returns_none():
    # [FUNCTION DOC] An errored trace without any named rewards yields none for the reward key score.
    trace = _trace(has_error=True, rewards={})

    assert _trace_eval_reward(trace, "score") is None


def test_successful_eval_trace_requires_configured_named_reward():
    # [FUNCTION DOC] A successful trace without the named reward raises a key error mentioning score.
    with pytest.raises(KeyError, match="score"):
        _trace_eval_reward(_trace(rewards={}), "score")


def test_unsupported_trace_error_is_not_resampled_forever():
    # [FUNCTION DOC] A trace whose error message says the adapter does not support the request
    # (responses dialect) makes the unsupported-error check raise a runtime error mentioning that
    # dialect, instead of being retried.
    error = SimpleNamespace(message="Miles' Verifiers adapter does not support this request: ResponsesDialect")

    with pytest.raises(RuntimeError, match="ResponsesDialect"):
        _raise_for_unsupported_trace_errors([_trace(error=error, has_error=True)])


def test_graph_branches_fail_before_miles_can_corrupt_trace_groups():
    # [FUNCTION DOC] A trace with two branches makes the multi-sample conversion raise a
    # not-implemented error mentioning multiple graph branches.
    trace = _trace(branches=[_branch(index=0), _branch(index=1)])

    with pytest.raises(NotImplementedError, match="multiple graph branches"):
        trace_to_samples(_args(), trace, group_index=4, index_start=10)


def test_convert_group_uses_standard_miles_group_shape():
    # [FUNCTION DOC] Builds an adapter without its constructor (args and a zero next-sample
    # index) and converts two traces into a group. Asserts the group has two sample objects, then
    # runs the standard rollout data post-processing (trim disabled, global batch size 1, static
    # batch size, data parallel size 1) on the group and asserts the flattened samples keep the
    # routing keys first and second in order.
    from miles.ray.rollout.rollout_data_conversion import postprocess_rollout_data

    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args()
    adapter._next_sample_index = 0

    group = adapter._convert_group(
        [_trace(id="first"), _trace(id="second")],
        group_index=2,
    )

    assert len(group) == 2
    assert all(isinstance(sample, Sample) for sample in group)
    args = SimpleNamespace(
        disable_rollout_trim_samples=True,
        global_batch_size=1,
        use_dynamic_global_batch_size=False,
    )
    flattened, _ = postprocess_rollout_data(args, [group], train_parallel_config={"dp_size": 1})
    assert [sample.routing_key for sample in flattened] == ["first", "second"]


def test_standard_dynamic_filter_accepts_converted_branch_group():
    # [FUNCTION DOC] Converts two traces with rewards 0.0 and 1.0 into a group and asserts the
    # standard nonzero-reward-std dynamic sampling filter decides to keep it.
    from miles.rollout.filter_hub.dynamic_sampling_filters import check_reward_nonzero_std

    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args()
    adapter._next_sample_index = 0
    group = adapter._convert_group(
        [_trace(id="low", reward=0.0), _trace(id="high", reward=1.0)],
        group_index=0,
    )

    assert check_reward_nonzero_std(adapter.args, group).keep


@pytest.mark.parametrize(
    ("overrides", "option"),
    [
        (
            {"eval_interval": None, "n_samples_per_prompt": 1, "n_samples_per_eval_prompt": 1},
            "--n-samples-per-prompt",
        ),
        (
            {"eval_interval": 1, "n_samples_per_prompt": 2, "n_samples_per_eval_prompt": 1},
            "--n-samples-per-eval-prompt",
        ),
    ],
)
def test_group_reward_tasks_require_multiple_rollouts(overrides, option):
    # [FUNCTION DOC] Parametrized: with one sample per training prompt and eval off, or with eval
    # on and one sample per eval prompt, validating group reward sample counts for one task whose
    # sample factory returns one item raises a value error naming the offending option.
    args = _args(**overrides)
    tasks = [object()]

    with pytest.raises(ValueError, match=option):
        _validate_group_reward_sample_counts(args, tasks, lambda _task, _kind: [object()])


def test_group_reward_eval_count_is_ignored_when_eval_is_disabled():
    # [FUNCTION DOC] With eval off, a single sample per eval prompt does not matter; validation
    # does not raise.
    args = _args(
        eval_interval=None,
        n_samples_per_prompt=2,
        n_samples_per_eval_prompt=1,
    )

    _validate_group_reward_sample_counts(args, [object()], lambda _task, _kind: [object()])


def test_group_reward_train_count_is_ignored_for_eval_only_runs():
    # [FUNCTION DOC] With zero training rollouts (eval only), a single sample per training prompt
    # does not matter; validation does not raise.
    args = _args(
        num_rollout=0,
        eval_interval=1,
        n_samples_per_prompt=1,
        n_samples_per_eval_prompt=2,
    )

    _validate_group_reward_sample_counts(args, [object()], lambda _task, _kind: [object()])


@pytest.mark.asyncio
async def test_verifiers_episode_owns_group_reward_computation():
    # [FUNCTION DOC] Skipped unless verifiers (at least 0.2.0) and renderers are installed. A fake
    # environment returns a fake episode for task, context and group size 2 (asserted); running the
    # episode asserts a semaphore is given and then sets the two traces' rewards to -1.0 and 1.0.
    # Runs the adapter's task group runner on an adapter built without its constructor and asserts
    # the returned traces carry the rewards set by the episode (the episode, not the adapter,
    # computes the group rewards).
    pytest.importorskip("verifiers", minversion="0.2.0")
    pytest.importorskip("renderers", minversion="0.1.8")
    traces = [_trace(id="a", reward=0.0), _trace(id="b", reward=0.0)]

    class Episode:
        # [CLASS DOC] HELPER: Fake episode returning the two traces after assigning group rewards.
        rollouts = []

        async def run(self, semaphore):
            # [FUNCTION DOC] HELPER: Asserts a semaphore was given, sets the traces' rewards to -1.0 and 1.0, returns the traces.
            assert semaphore is not None
            traces[0].reward = -1.0
            traces[1].reward = 1.0
            return traces

    class Environment:
        # [CLASS DOC] HELPER: Fake environment building the fake episode.
        def episode(self, task, ctx, n):
            # [FUNCTION DOC] HELPER: Asserts the task, context and group size are the expected values; returns an episode.
            assert task == "task"
            assert ctx == "ctx"
            assert n == 2
            return Episode()

    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args(sglang_enable_deterministic_inference=False)
    adapter.env = Environment()
    adapter.ctx = "ctx"

    result = await adapter._run_task_group("task", 2, asyncio.Semaphore(2), seed_base=0)

    assert [trace.reward for trace in result] == [-1.0, 1.0]


def test_sampling_config_preserves_miles_minimum_tokens():
    # [FUNCTION DOC] Builds a sampling config through the adapter's static helper with a fake
    # config class (validation returns its input) and args with min new tokens 3; asserts the
    # resulting mapping has min tokens 3.
    class SamplingConfig:
        # [CLASS DOC] HELPER: Fake sampling config class whose validation returns the data unchanged.
        @staticmethod
        def model_validate(data):
            # [FUNCTION DOC] HELPER: Returns the data unchanged.
            return data

    config = VerifiersRolloutFn._sampling_config(
        SamplingConfig,
        _args(
            apply_chat_template_kwargs={},
            rollout_min_new_tokens=3,
            rollout_temperature=0.7,
            rollout_top_k=20,
            rollout_top_p=0.9,
        ),
    )

    assert config["min_tokens"] == 3


def test_eval_args_clear_training_prompt_cap_and_preserve_other_fallbacks():
    # [FUNCTION DOC] Eval args made from args with eval max context 128 and unset other eval
    # values, reward key score, and training caps of context 64, prompt 32, response 8. Asserts the
    # eval args use context 128, clear the prompt cap to none (the training prompt cap is not
    # inherited), fall back to the training response cap 8, and fall back to reward key score.
    args = _args(
        eval_max_context_len=128,
        eval_max_prompt_len=None,
        eval_max_response_len=None,
        eval_min_new_tokens=None,
        eval_reward_key=None,
        eval_temperature=None,
        eval_top_k=None,
        eval_top_p=None,
        reward_key="score",
        rollout_max_context_len=64,
        rollout_max_prompt_len=32,
        rollout_max_response_len=8,
    )

    eval_args = _make_eval_args(args)

    assert eval_args.rollout_max_context_len == 128
    assert eval_args.rollout_max_prompt_len is None
    assert eval_args.rollout_max_response_len == 8
    assert eval_args.reward_key == "score"


@pytest.mark.asyncio
async def test_transport_translates_renderer_request_to_sglang(monkeypatch):
    # [FUNCTION DOC] Replaces the module's http post with a fake recording url, payload and
    # headers and returning an sglang-shaped response (request id; meta info with 2 completion
    # tokens, finish reason stop and output token log probabilities for token ids 20 and 21). The
    # transport (router policy manual) posts a renderer-style request to the inference generate
    # url with model, token ids [10, 11], sampling params (temperature 0.2, max tokens 2, stop
    # token ids [99], logprobs 1) and a session id header. Asserts the response json's first choice
    # has token ids [20, 21], per-token log probs with token_id:N names, and finish reason stop;
    # and the recorded request went to the plain generate url with input ids, sampling params
    # translated (max new tokens, skip special tokens true, no stop trim true, spaces between
    # special tokens false, n 1), return logprob true, and the routing key header set from the
    # session id.
    requests = []

    async def fake_post(url, payload, headers=None):
        # [FUNCTION DOC] HELPER: Records the url, payload and headers; returns the canned sglang-shaped response mapping.
        requests.append((url, payload, headers))
        return {
            "request_id": "request-id",
            "meta_info": {
                "completion_tokens": 2,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 20], [-0.2, 21]],
            },
        }

    monkeypatch.setattr("miles.utils.http_utils.post", fake_post)
    transport = MilesSGLangTransport(_args(sglang_router_policy="manual"))

    response = await transport.post(
        "http://127.0.0.1:30000/inference/v1/generate",
        body={
            "model": "test/model",
            "token_ids": [10, 11],
            "sampling_params": {"temperature": 0.2, "max_tokens": 2, "stop_token_ids": [99], "logprobs": 1},
        },
        options={"headers": {"X-Session-ID": "trace-id"}},
    )

    assert response.json()["choices"][0] == {
        "token_ids": [20, 21],
        "logprobs": {
            "content": [
                {"token": "token_id:20", "logprob": -0.1},
                {"token": "token_id:21", "logprob": -0.2},
            ]
        },
        "finish_reason": "stop",
    }
    assert requests == [
        (
            "http://127.0.0.1:30000/generate",
            {
                "input_ids": [10, 11],
                "sampling_params": {
                    "temperature": 0.2,
                    "stop_token_ids": [99],
                    "max_new_tokens": 2,
                    "skip_special_tokens": True,
                    "no_stop_trim": True,
                    "spaces_between_special_tokens": False,
                    "n": 1,
                },
                "return_logprob": True,
            },
            {"X-SMG-Routing-Key": "trace-id"},
        )
    ]


@pytest.mark.asyncio
async def test_transport_rejects_multimodal_features():
    # [FUNCTION DOC] Posting a request body that carries a features entry makes the transport
    # raise a not-implemented error mentioning multimodal.
    transport = MilesSGLangTransport(_args())

    with pytest.raises(NotImplementedError, match="multimodal"):
        await transport.post(
            "http://127.0.0.1:30000/inference/v1/generate",
            body={"token_ids": [1], "sampling_params": {}, "features": {}},
        )


@pytest.mark.asyncio
async def test_transport_bounds_seen_sessions(monkeypatch):
    # [FUNCTION DOC] With a fake http post returning a one-token stop response and the
    # transport's session cache size set to 2, posts three requests with session ids one, two and
    # three. Asserts the transport's seen sessions, in order, are only two and three (oldest evicted).
    async def fake_post(_url, _payload, headers=None):
        # [FUNCTION DOC] HELPER: Returns a canned sglang-shaped response with one completion token (id 20) and finish reason stop.
        return {
            "meta_info": {
                "completion_tokens": 1,
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 20]],
            }
        }

    monkeypatch.setattr("miles.utils.http_utils.post", fake_post)
    transport = MilesSGLangTransport(_args())
    transport._session_cache_size = 2
    body = {"token_ids": [10], "sampling_params": {}}

    for session_id in ("one", "two", "three"):
        await transport.post(
            "http://127.0.0.1:30000/inference/v1/generate",
            body=body,
            options={"headers": {"X-Session-ID": session_id}},
        )

    assert list(transport._seen_sessions) == ["two", "three"]


def test_transport_resolves_router_address_lazily():
    # [FUNCTION DOC] Creates the transport while the router ip and port are unset, then sets them
    # on the same args; asserts the base url uses the newly set address.
    args = _args(sglang_router_ip=None, sglang_router_port=None)
    transport = MilesSGLangTransport(args)

    args.sglang_router_ip = "10.0.0.4"
    args.sglang_router_port = 3210

    assert transport.base_url == "http://10.0.0.4:3210/v1"


def test_transport_uses_default_model_router():
    # [FUNCTION DOC] With a model routers mapping holding a default and a ref entry, asserts the
    # base url uses the default router's address.
    args = _args(
        sglang_model_routers={
            "default": ("10.0.0.5", 3211),
            "ref": ("10.0.0.6", 3212),
        }
    )

    assert MilesSGLangTransport(args).base_url == "http://10.0.0.5:3211/v1"


def test_eval_transport_keeps_live_router_args():
    # [FUNCTION DOC] An eval transport given separate rollout router args; the model routers are
    # set on the rollout args after construction; asserts the base url follows the rollout args' default router.
    rollout_args = _args(sglang_router_ip=None, sglang_router_port=None)
    eval_args = _args(sglang_router_ip=None, sglang_router_port=None)
    transport = MilesSGLangTransport(eval_args, router_args=rollout_args)

    rollout_args.sglang_model_routers = {"default": ("10.0.0.7", 3213)}

    assert transport.base_url == "http://10.0.0.7:3213/v1"


@pytest.mark.asyncio
async def test_eval_rejects_miles_group_reward_model():
    # [FUNCTION DOC] An adapter (built without constructor) whose args enable the group reward
    # model; running eval raises an assertion error saying group reward model is not supported for eval rollout.
    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args(group_rm=True)

    with pytest.raises(AssertionError, match="Group RM is not supported for eval rollout"):
        await adapter._call_eval(SimpleNamespace(rollout_id=0))


@pytest.mark.asyncio
async def test_eval_extracts_structured_miles_reward(monkeypatch):
    # [FUNCTION DOC] Sets up an adapter without constructor: custom reward module path, eval
    # reward key score, no group rm, one sample per eval prompt, one task, a serving context
    # manager that does nothing, a task group runner returning one trace with only a bonus reward,
    # and a reward applier that sets the group's first reward to a mapping with score 0.75 and
    # details. Stubs the dumper utility module (sglang configuration no-op) in the module table and
    # the miles utils package. Runs eval and asserts the output data for the environment id
    # holds the rewards list [0.75] (the configured key extracted from the structured reward).
    import miles.utils as miles_utils

    @asynccontextmanager
    async def serving():
        # [FUNCTION DOC] HELPER: Async context manager that yields immediately and does nothing else.
        yield

    async def configure_sglang(_args):
        # [FUNCTION DOC] HELPER: Does nothing and returns none.
        return None

    async def run_group(*_args, **_kwargs):
        # [FUNCTION DOC] HELPER: Returns a list with one trace whose rewards hold only a bonus of 0.25.
        return [_trace(rewards={"bonus": 0.25})]

    async def apply_reward(group):
        # [FUNCTION DOC] HELPER: Sets the first element's reward to a mapping with score 0.75 and details ok.
        group[0].reward = {"score": 0.75, "details": "ok"}

    dumper_utils = SimpleNamespace(configure_sglang=configure_sglang)
    monkeypatch.setitem(sys.modules, "miles.utils.dumper_utils", dumper_utils)
    monkeypatch.setattr(miles_utils, "dumper_utils", dumper_utils, raising=False)
    adapter = object.__new__(VerifiersRolloutFn)
    adapter.args = _args(
        custom_rm_path="tests.fake_reward",
        eval_reward_key="score",
        group_rm=False,
        n_samples_per_eval_prompt=1,
        rm_type=None,
        rollout_batch_size=1,
        rollout_seed=1,
    )
    adapter.eval_args = _args(reward_key="score")
    adapter.config = SimpleNamespace(env_id="test-env")
    adapter.env = SimpleNamespace(serving=serving)
    adapter.eval_ctx = object()
    adapter.max_concurrent = 1
    adapter.model = "test/model"
    adapter._tasks = [object()]
    adapter._next_sample_index = 0
    adapter._run_task_group = run_group
    adapter._apply_miles_rewards = apply_reward

    output = await adapter._call_eval(SimpleNamespace(rollout_id=0))

    assert output.data["test-env"]["rewards"] == [0.75]


class TestRolloutFnContract:
    # [CLASS DOC] Tests that the adapter class fits the rollout function loader's contract.
    def test_it_is_a_rollout_fn_the_loader_accepts(self):
        # [FUNCTION DOC] Asserts the adapter class is a subclass of the base rollout function class.
        """The loader rejects a rollout function that forgets the base, however complete it is."""
        assert issubclass(VerifiersRolloutFn, BaseRolloutFn)

    def test_the_constructor_input_reaches_the_base(self, monkeypatch):
        # [FUNCTION DOC] Patches the module's verifiers import to raise a sentinel exception, calls
        # the adapter's constructor on a bare instance with a constructor input holding args and a
        # data source object, expects the sentinel to propagate, and asserts the instance's
        # constructor input attribute already holds that data source (the base constructor ran first).
        """Skipping super().__init__ leaves constructor_input missing on every path that reads it."""

        class _ImportVerifiersSentinel(Exception):
            # [CLASS DOC] HELPER: Marker exception raised by the patched verifiers import.
            pass

        def raise_sentinel():
            # [FUNCTION DOC] HELPER: Always raises the sentinel exception.
            raise _ImportVerifiersSentinel

        monkeypatch.setattr(verifiers_rollout, "_import_verifiers", raise_sentinel)
        data_source = object()
        constructor_input = RolloutFnConstructorInput(args=_args(), data_source=data_source)
        adapter = object.__new__(VerifiersRolloutFn)

        with pytest.raises(_ImportVerifiersSentinel):
            VerifiersRolloutFn.__init__(adapter, constructor_input)

        assert adapter.constructor_input.data_source is data_source
