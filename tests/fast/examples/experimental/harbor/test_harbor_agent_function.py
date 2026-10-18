# [MODULE DOC]
# Offline tests for the in-process Harbor agent function module (imported by bare name). A fake
# harbor package is installed in the module table by an autouse fixture, with an environment type
# enum, recording config constructors and a fake trial class (records created trials, returns a
# scripted result or raises it, optionally delays). Sections: building the trial config from
# environment variables, a sample's metadata and the session url (environment type, sandbox
# reclaim timers, resource overrides, enforcement policies, per-agent bindings), mapping a trial
# result to reward/exit status/metrics, and the async entry point (verdict, timeout, provider key
# resolution, failure handling).
"""Offline tests for the in-process Harbor agent function (no Harbor, no sandbox).

The ``harbor`` package is faked in sys.modules (module-local autouse fixture),
so these tests run where Harbor is not installed. The contract against the
REAL package lives in test_harbor_contract.py.
"""

import asyncio
import enum
import os
import sys
import types
from datetime import datetime, timedelta
from types import SimpleNamespace

import harbor_agent_function as haf
import pytest


class _EnvironmentType(str, enum.Enum):
    # [CLASS DOC] HELPER: Fake of the sandbox environment type string enumeration with docker,
    # daytona, e2b and modal members.
    DOCKER = "docker"
    DAYTONA = "daytona"
    E2B = "e2b"
    MODAL = "modal"


def _record(name):
    # [FUNCTION DOC] HELPER: Returns a constructor that builds a simple namespace holding the
    # keyword arguments plus a kind field set to the given name (a stand-in for a config model).
    def ctor(**kwargs):
        # [FUNCTION DOC] HELPER: Builds the namespace described above.
        return SimpleNamespace(_kind=name, **kwargs)

    return ctor


class FakeTrial:
    # [CLASS DOC] HELPER: Fake harbor trial with class-level state shared by tests: the list of
    # created trials, the scripted result (a value returned, or an exception raised, by run), and
    # an artificial run delay in seconds. Reset by the fake harbor fixture for every test.
    """Records the config it was created with; ``run`` returns a scripted result."""

    created: list = []
    result = None
    run_delay_s = 0.0

    def __init__(self, config):
        # [FUNCTION DOC] HELPER: Stores the config and a paths namespace whose trial dir is a
        # fixed temp folder named after the task path's final component.
        self.config = config
        self.paths = SimpleNamespace(trial_dir=f"/tmp/harbor_trials/{config.task.path.name}")

    @classmethod
    async def create(cls, config):
        # [FUNCTION DOC] HELPER: Async factory: builds a trial from the config, appends it to the
        # class-level created list, and returns it.
        trial = cls(config)
        cls.created.append(trial)
        return trial

    async def run(self):
        # [FUNCTION DOC] HELPER: Sleeps for the class-level delay if it is nonzero, then raises the
        # scripted result if it is an exception, else returns it.
        import asyncio

        if self.run_delay_s:
            await asyncio.sleep(self.run_delay_s)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture(autouse=True)
def fake_harbor(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture installing fake harbor modules in the module table
    # (harbor, models, environment type, trial models and config with recording constructors for
    # agent, task, trial, verifier and environment configs, and the trial module exposing the fake
    # trial class). Resets the fake trial's created list, result and delay, then yields the class.
    # Patches are undone by the monkeypatch fixture.
    harbor = types.ModuleType("harbor")
    models = types.ModuleType("harbor.models")
    env_type = types.ModuleType("harbor.models.environment_type")
    env_type.EnvironmentType = _EnvironmentType
    trial_models = types.ModuleType("harbor.models.trial")
    config = types.ModuleType("harbor.models.trial.config")
    for name in ("AgentConfig", "TaskConfig", "TrialConfig", "VerifierConfig", "EnvironmentConfig"):
        setattr(config, name, _record(name))
    trial_pkg = types.ModuleType("harbor.trial")
    trial_mod = types.ModuleType("harbor.trial.trial")
    trial_mod.Trial = FakeTrial
    for mod in (harbor, models, env_type, trial_models, config, trial_pkg, trial_mod):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    FakeTrial.created = []
    FakeTrial.result = None
    FakeTrial.run_delay_s = 0.0
    yield FakeTrial


def run_async(coro):
    # [FUNCTION DOC] HELPER: Runs the coroutine to completion in a fresh event loop and returns its result.
    return asyncio.run(coro)


@pytest.fixture
def tasks_dir(tmp_path, monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture creating a task-1 folder in the temp path and setting the
    # environment: tasks dir to the temp path, environment type e2b, a dummy e2b api key, and
    # removing env kwargs and both resource enforcement policy variables. Returns the temp path.
    (tmp_path / "task-1").mkdir()
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    # so tests never read a real key file from the developer's machine
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    monkeypatch.delenv("HARBOR_ENV_KWARGS", raising=False)
    monkeypatch.delenv("HARBOR_CPU_ENFORCEMENT_POLICY", raising=False)
    monkeypatch.delenv("HARBOR_MEMORY_ENFORCEMENT_POLICY", raising=False)
    return tmp_path


def _verdict(reward=1.0, **agent_fields):
    # [FUNCTION DOC] HELPER: Builds a fake successful trial result: no exception, a verifier result
    # with the given reward, an agent result (1000 input tokens, 200 output tokens, no cost, 7
    # steps, 5 tool calls in metadata, plus extra fields), a 90 second overall duration from a
    # fixed start, environment setup 0-10s, no agent setup, agent execution 10-80s, verifier 80-90s.
    t0 = datetime(2026, 1, 1, 0, 0, 0)
    return SimpleNamespace(
        exception_info=None,
        verifier_result=SimpleNamespace(rewards={"reward": reward}),
        agent_result=SimpleNamespace(
            n_input_tokens=1000,
            n_output_tokens=200,
            cost_usd=None,
            n_steps=7,
            metadata={"tool_calls": 5},
            **agent_fields,
        ),
        started_at=t0,
        finished_at=t0 + timedelta(seconds=90),
        environment_setup=SimpleNamespace(started_at=t0, finished_at=t0 + timedelta(seconds=10)),
        agent_setup=None,
        agent_execution=SimpleNamespace(started_at=t0 + timedelta(seconds=10), finished_at=t0 + timedelta(seconds=80)),
        verifier=SimpleNamespace(started_at=t0 + timedelta(seconds=80), finished_at=t0 + timedelta(seconds=90)),
    )


# --- trial config ----------------------------------------------------------


def test_environment_type_is_passed_straight_through(tasks_dir, monkeypatch):
    # [FUNCTION DOC] With environment type modal and a json env kwargs variable, asserts the built
    # config carries type modal, the kwargs unchanged, and delete set to true.
    monkeypatch.setenv("HARBOR_ENV_TYPE", "modal")
    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"passthrough_probe": 1}')
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.type.value == "modal"
    assert cfg.environment.kwargs == {"passthrough_probe": 1}
    assert cfg.environment.delete is True


def test_daytona_reclaim_timer_outlasts_the_trial_cap(tasks_dir, monkeypatch):
    # [FUNCTION DOC] Three steps for daytona with a 1200 second trial timeout: defaults give auto
    # stop 50 minutes and auto delete 1440 minutes; a caller auto-stop of 45 minutes (safe) is
    # kept and extra kwargs pass through; a caller auto-stop of 20 minutes (not longer than the
    # trial) raises a value error mentioning the mid-trial stop.
    """Harbor's Daytona defaults never reclaim a sandbox a killed worker left
    behind; ours must, without ever stopping a live trial."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "daytona")
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "1200")  # 20 minutes
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.kwargs == {"auto_stop_interval_mins": 50, "auto_delete_interval_mins": 1440}

    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"auto_stop_interval_mins": 45, "auto_snapshot": true}')
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.kwargs == {
        "auto_stop_interval_mins": 45,  # the caller's value wins when it is safe
        "auto_delete_interval_mins": 1440,
        "auto_snapshot": True,
    }

    monkeypatch.setenv("HARBOR_ENV_KWARGS", '{"auto_stop_interval_mins": 20}')
    with pytest.raises(ValueError, match="mid-trial"):
        haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})


def test_reclaim_timers_are_daytona_only(tasks_dir, monkeypatch):
    # [FUNCTION DOC] With environment type e2b, asserts the environment kwargs are empty.
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert cfg.environment.kwargs == {}


@pytest.mark.parametrize(
    "var, field, value",
    [
        ("HARBOR_OVERRIDE_CPUS", "override_cpus", 2),
        ("HARBOR_OVERRIDE_MEMORY_MB", "override_memory_mb", 20480),
        ("HARBOR_OVERRIDE_STORAGE_MB", "override_storage_mb", 10240),
    ],
)
def test_resource_overrides_reach_harbor_and_reject_nonpositive(tasks_dir, monkeypatch, var, field, value):
    # [FUNCTION DOC] Parametrized over cpu, memory and storage override variables. Setting the
    # variable makes the matching environment config field equal the number; setting it to zero
    # raises a value error naming the variable.
    monkeypatch.setenv(var, str(value))
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})
    assert getattr(cfg.environment, field) == value

    monkeypatch.setenv(var, "0")
    with pytest.raises(ValueError, match=var):
        haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})


@pytest.mark.parametrize(
    "var, field",
    [
        ("HARBOR_CPU_ENFORCEMENT_POLICY", "cpu_enforcement_policy"),
        ("HARBOR_MEMORY_ENFORCEMENT_POLICY", "memory_enforcement_policy"),
    ],
)
def test_resource_enforcement_policies_reach_harbor(tasks_dir, monkeypatch, var, field):
    # [FUNCTION DOC] Parametrized over the cpu and memory enforcement variables. Setting one to
    # request makes the matching environment config field equal that string.
    monkeypatch.setenv(var, "request")

    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "mini-swe-agent"}, "http://s/v1", {})

    assert getattr(cfg.environment, field) == "request"


def test_unknown_environment_type_is_an_error_not_docker(tasks_dir, monkeypatch):
    # [FUNCTION DOC] An unrecognised environment type string raises a value error.
    """The agent server silently fell back to docker on an unknown HARBOR_ENV_TYPE; this path refuses."""
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2bb")
    with pytest.raises(ValueError):
        haf.build_trial_config({"instance_id": "task-1"}, "http://s/v1", {})


def test_environment_type_is_required(tasks_dir, monkeypatch):
    # [FUNCTION DOC] With the environment type variable removed, building raises a value error naming the variable.
    monkeypatch.delenv("HARBOR_ENV_TYPE")
    with pytest.raises(ValueError, match="HARBOR_ENV_TYPE"):
        haf.build_trial_config({"instance_id": "task-1"}, "http://s/v1", {})


def test_mini_swe_agent_binding_hands_the_session_url_through_openai_env(tasks_dir, monkeypatch):
    # [FUNCTION DOC] With model name glm and sample max sequence length 4096, asserts for the
    # mini-swe-agent: agent name, model name prefixed with the openai provider, the openai base url
    # env equal to the session url, cost tracking env set to ignore errors, the sequence length
    # passed as a kwarg, model info max output tokens 8192 (default), and the task path's last
    # component equal to the instance id.
    monkeypatch.setenv("AGENT_MODEL_NAME", "glm")
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "mini-swe-agent", "max_seq_len": 4096},
        "http://s/v1",
        {"temperature": 0.8},
    )
    assert cfg.agent.name == "mini-swe-agent"
    assert cfg.agent.model_name == "openai/glm"
    assert cfg.agent.env["OPENAI_API_BASE"] == "http://s/v1"
    assert cfg.agent.env["MSWEA_COST_TRACKING"] == "ignore_errors"
    assert cfg.agent.kwargs["max_seq_len"] == 4096
    assert cfg.agent.kwargs["model_info"]["max_output_tokens"] == 8192
    assert cfg.task.path.name == "task-1"


def test_terminus_2_binding_aborts_on_truncation_and_carries_sampling_params(tasks_dir, monkeypatch):
    # [FUNCTION DOC] For terminus-2 with max_tokens 512 sampling params and no length policy
    # variable, asserts: response-length-exceeded policy abort, llm call kwargs equal to the
    # sampling params, api base equal to the session url, llm kwargs holding a dummy api key (and
    # no top-level api key kwarg), and env holding dummy openai key and the base url.
    monkeypatch.delenv("HARBOR_RESPONSE_LENGTH_POLICY", raising=False)
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "terminus-2"}, "http://s/v1", {"max_tokens": 512}
    )
    assert cfg.agent.kwargs["response_length_exceeded_policy"] == "abort"
    assert cfg.agent.kwargs["llm_call_kwargs"] == {"max_tokens": 512}
    assert cfg.agent.kwargs["api_base"] == "http://s/v1"
    # no top-level api_key parameter on Terminus2; litellm gets it per call via llm_kwargs
    assert cfg.agent.kwargs["llm_kwargs"] == {"api_key": "dummy"}
    assert "api_key" not in cfg.agent.kwargs
    assert cfg.agent.env == {"OPENAI_API_KEY": "dummy", "OPENAI_API_BASE": "http://s/v1"}


def test_claude_code_binding_uses_anthropic_env(tasks_dir, monkeypatch):
    # [FUNCTION DOC] With max output tokens 4096 and session url ending in /v1, asserts: anthropic
    # base url is the session root (version suffix stripped), tool search disabled, max output
    # tokens env 4096, and web search and web fetch disallowed as a comma string.
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "4096")
    cfg = haf.build_trial_config({"instance_id": "task-1", "agent_name": "claude-code"}, "http://s/sess/v1", {})
    # the session root: the SDK appends /v1/messages, the server route is
    # /sessions/{id}/v1/messages -- a /v1-suffixed base would 404 on /v1/v1/messages
    assert cfg.agent.env["ANTHROPIC_BASE_URL"] == "http://s/sess"
    assert cfg.agent.env["ENABLE_TOOL_SEARCH"] == "false"
    assert cfg.agent.env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "4096"
    assert cfg.agent.kwargs["disallowed_tools"] == "WebSearch,WebFetch"


def test_opencode_binding_remaps_the_openai_provider(tasks_dir, monkeypatch):
    # [FUNCTION DOC] For opencode with model glm, sequence length 65536 and max_tokens 512,
    # asserts: model name uses the openai-compatible provider; the provider config entry uses the
    # ai-sdk openai-compatible package and the session base url; the model entry has context limit
    # 65536 and output limit 512; both the openai base-url and api-base env variables hold the
    # session url; no max turns kwarg.
    """OpenCode resolves provider id "openai" to the Responses API, which the session server rejects."""
    monkeypatch.setenv("AGENT_MODEL_NAME", "glm")
    monkeypatch.delenv("AGENT_MAX_OUTPUT_TOKENS", raising=False)
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "opencode", "max_seq_len": 65536}, "http://s/v1", {"max_tokens": 512}
    )
    assert cfg.agent.model_name == "openai-compatible/glm"
    provider = cfg.agent.kwargs["opencode_config"]["provider"]
    entry = provider["openai-compatible"]
    assert entry["npm"] == "@ai-sdk/openai-compatible"
    assert entry["options"]["baseURL"] == "http://s/v1"
    # limits: without them OpenCode never auto-compacts a custom-provider model
    assert entry["models"]["glm"] == {"limit": {"context": 65536, "output": 512}}
    # both URL vars: OpenCode consults OPENAI_BASE_URL, litellm-style agents OPENAI_API_BASE
    assert cfg.agent.env["OPENAI_BASE_URL"] == "http://s/v1"
    assert cfg.agent.env["OPENAI_API_BASE"] == "http://s/v1"
    # no turn cap: OpenCode has no flag for it, a kwarg would be silently discarded
    assert "max_turns" not in cfg.agent.kwargs


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/glm", ("openai-compatible", "glm")),
        ("bare-model", ("openai-compatible", "bare-model")),
        ("anthropic/claude", ("anthropic", "claude")),
    ],
)
def test_opencode_provider_split(model, expected):
    # [FUNCTION DOC] Parametrized: the provider/model splitter maps an openai-prefixed name to the
    # openai-compatible provider with the bare model, a bare name to openai-compatible, and an
    # anthropic-prefixed name to anthropic with the bare model.
    assert haf._opencode_provider_model(model) == expected


@pytest.mark.parametrize("bad", ["", "../etc", "no-such-task"])
def test_instance_id_must_name_a_task_dir(tasks_dir, bad):
    # [FUNCTION DOC] Parametrized over an empty id, a path-traversal id and an unknown id; each
    # must raise a value error or file-not-found error.
    with pytest.raises((ValueError, FileNotFoundError)):
        haf.build_trial_config({"instance_id": bad}, "http://s/v1", {})


# --- result mapping --------------------------------------------------------


def test_verdict_maps_reward_metrics_and_timings():
    # [FUNCTION DOC] Maps a successful fake result and asserts: reward 1.0, exit status Submitted,
    # eval report equal to the verifier rewards, metrics with turns 7, tool calls 5, input tokens
    # 1000, total time 90, environment setup time 10, eval time 10, and no agent setup time (it
    # was absent).
    out = haf.trial_result_to_metadata(_verdict(reward=1.0))
    assert out["reward"] == 1.0
    assert out["exit_status"] == "Submitted"
    assert out["eval_report"] == {"reward": 1.0}
    m = out["agent_metrics"]
    assert m["turns"] == 7 and m["tool_calls"] == 5 and m["n_input_tokens"] == 1000
    assert m["total_time"] == 90.0 and m["env_setup_time"] == 10.0 and m["eval_time"] == 10.0
    assert "agent_setup_time" not in m


@pytest.mark.parametrize(
    ("exc_type", "exit_status"),
    [
        ("AgentTimeoutError", "TimeLimitExceeded"),
        ("EnvironmentStartTimeoutError", "TimeLimitExceeded"),
        ("SingleTurnMaxSeqLenExceededError", "SequenceLengthLimitExceeded"),
        ("RuntimeError", "AgentError"),
    ],
)
def test_harbor_exceptions_map_to_the_exit_status_vocabulary(exc_type, exit_status):
    # [FUNCTION DOC] Parametrized: a result with an exception of the given type name and no
    # verifier or agent results maps to reward 0.0 and the expected exit status (agent and
    # environment-start timeouts to time limit exceeded; single-turn max-length error to sequence
    # length limit exceeded; any other to agent error).
    result = SimpleNamespace(
        exception_info=SimpleNamespace(exception_type=exc_type), verifier_result=None, agent_result=None
    )
    out = haf.trial_result_to_metadata(result)
    assert out["reward"] == 0.0
    assert out["exit_status"] == exit_status


# --- entry -----------------------------------------------------------------


def test_run_returns_the_verdict_and_trial_dir(tasks_dir, fake_harbor):
    # [FUNCTION DOC] Runs the async entry with a scripted successful trial for task-1 and a
    # session url with a host and port. Asserts reward 1.0, exit status Submitted, a trial dir
    # ending with the task name, exactly one created trial, and that its agent env's openai api
    # base is the given session url with /v1 appended.
    fake_harbor.result = _verdict(reward=1.0)

    out = run_async(
        haf.run(
            "http://trainer.tailnet:30000/sessions/s1",
            [],
            {"temperature": 0.8},
            {"instance_id": "task-1", "agent_name": "mini-swe-agent"},
        )
    )

    assert out["reward"] == 1.0 and out["exit_status"] == "Submitted"
    assert out["trial_dir"].endswith("task-1")
    (trial,) = fake_harbor.created
    # in-sandbox agents call the model from inside the sandbox: base_url already names the
    # instance the way the sandbox reaches it, so it goes through with only /v1 appended
    assert trial.config.agent.env["OPENAI_API_BASE"] == "http://trainer.tailnet:30000/sessions/s1/v1"


def test_run_scores_a_timeout_zero(tasks_dir, fake_harbor, monkeypatch):
    # [FUNCTION DOC] Sets the trial timeout to zero and the fake trial to run for ten seconds.
    # Asserts the entry returns exactly reward 0.0, exit status time limit exceeded and empty eval
    # report and agent metrics.
    """The trial may still be running (the policy may be what stalls it): a negative sample, not a discard."""
    fake_harbor.result = _verdict()
    fake_harbor.run_delay_s = 10
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "0")

    out = run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert out == {"reward": 0.0, "exit_status": "TimeLimitExceeded", "eval_report": {}, "agent_metrics": {}}


def test_run_resolves_the_provider_key_file_into_the_sdk_env_var(tasks_dir, fake_harbor, tmp_path, monkeypatch):
    # [FUNCTION DOC] Writes a key (with trailing newline) to a file, removes the direct e2b key
    # variable and points the key-file variable at the file. After running the entry, asserts the
    # e2b key environment variable holds the key stripped of the newline. Side effect: the
    # process environment retains the variable set by the code under test.
    """The launcher forwards the key by PATH; the SDK reads its own env var, so
    the worker must resolve the file -- the gap the first live e2e run hit."""
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_test_key\n")
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", str(key_file))
    fake_harbor.result = _verdict()

    run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert os.environ["E2B_API_KEY"] == "e2b_test_key"


def test_run_raises_when_no_provider_key_is_resolvable(tasks_dir, fake_harbor, monkeypatch):
    # [FUNCTION DOC] With no direct key and a key-file variable naming a missing file, asserts the
    # entry raises a runtime error saying there is no API key and that no trial was created.
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("E2B_API_KEY_FILE", "/nonexistent/api_key")
    with pytest.raises(RuntimeError, match="no API key"):
        run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert fake_harbor.created == []


def test_run_scores_a_trial_exception_zero(tasks_dir, fake_harbor):
    # [FUNCTION DOC] The fake trial raises a runtime error; asserts the entry returns reward 0.0 and exit status agent error.
    fake_harbor.result = RuntimeError("sandbox exploded")
    out = run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "task-1"}))
    assert out["reward"] == 0.0 and out["exit_status"] == "AgentError"


def test_run_raises_on_a_missing_task_instead_of_scoring_zero(tasks_dir, fake_harbor):
    # [FUNCTION DOC] With an instance id naming no task folder, asserts the entry raises a
    # file-not-found error and no trial was created.
    """A config error fails every sample; raising beats training on silent all-zero rewards."""
    with pytest.raises(FileNotFoundError):
        run_async(haf.run("http://s/sessions/s1", [], {}, {"instance_id": "missing"}))
    assert fake_harbor.created == []
