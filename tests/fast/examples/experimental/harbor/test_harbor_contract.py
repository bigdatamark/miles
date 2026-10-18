# [MODULE DOC]
# Contract tests against the real harbor package, skipped entirely when harbor is not importable.
# They check that what the agent function module builds is accepted by harbor's own models: the
# pass-through environment type names exist, every harness binding name is a real harbor agent
# name, the built config validates as a real trial config for every binding plus an unknown
# agent, enforcement policy values convert to harbor's enum values, the trial class exposes
# create and run, and the real trial result model still has the fields the result mapping reads.
"""Contract tests against the REAL harbor package: what the agent function
builds must be something Harbor's own models accept.

Skipped wherever harbor is not installed (CPU CI today); they run in any
environment provisioned per the example README. They catch the drift the
faked-package unit tests cannot: an EnvironmentType value or AgentName that
stopped existing, a TrialConfig/AgentConfig field that moved. (Pydantic's
default extra="ignore" means a RENAMED optional field can still slip through;
required-field and enum drift is caught.)
"""

import pytest

harbor = pytest.importorskip("harbor")

import harbor_agent_function as haf  # noqa: E402
from harbor.models.agent.name import AgentName  # noqa: E402
from harbor.models.environment_type import EnvironmentType  # noqa: E402
from harbor.models.trial.config import TrialConfig  # noqa: E402
from harbor.trial.trial import Trial  # noqa: E402


@pytest.fixture
def tasks_dir(tmp_path, monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture creating a task-1 folder in the temp path and setting the
    # tasks dir variable to it and the environment type variable to e2b. Returns the temp path.
    (tmp_path / "task-1").mkdir()
    monkeypatch.setenv("HARBOR_TASKS_DIR", str(tmp_path))
    monkeypatch.setenv("HARBOR_ENV_TYPE", "e2b")
    return tmp_path


@pytest.mark.parametrize("env_type", ["docker", "daytona", "e2b", "modal"])
def test_the_pass_through_environment_types_exist(env_type):
    # [FUNCTION DOC] For docker, daytona, e2b and modal, asserts constructing the real environment
    # type enum from the string gives a member with that value.
    assert EnvironmentType(env_type).value == env_type


@pytest.mark.parametrize("agent_name", sorted(haf.HARNESS_BINDINGS))
def test_every_binding_names_a_real_harbor_agent(agent_name):
    # [FUNCTION DOC] For each key of the harness bindings, asserts the real agent name enum has a member with that value.
    assert AgentName(agent_name).value == agent_name


@pytest.mark.parametrize("agent_name", [*sorted(haf.HARNESS_BINDINGS), "some-other-openai-chat-agent"])
def test_the_built_config_is_a_valid_trial_config(tasks_dir, agent_name):
    # [FUNCTION DOC] For each binding and an unbound agent name, builds a config (sequence length
    # 4096, a session url, temperature and max tokens sampling params) and asserts it is an instance
    # of the real trial config and its environment type is e2b.
    """Real pydantic validation of everything build_trial_config assembles."""
    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": agent_name, "max_seq_len": 4096},
        "http://trainer:30000/sessions/s1/v1",
        {"temperature": 0.8, "max_tokens": 512},
    )
    assert isinstance(cfg, TrialConfig)
    assert cfg.environment.type == EnvironmentType.E2B


def test_resource_enforcement_policy_uses_harbor_model(tasks_dir, monkeypatch):
    # [FUNCTION DOC] Sets cpu policy request and memory policy limit; asserts the built config's
    # environment fields are enum members whose values are request and limit.
    monkeypatch.setenv("HARBOR_CPU_ENFORCEMENT_POLICY", "request")
    monkeypatch.setenv("HARBOR_MEMORY_ENFORCEMENT_POLICY", "limit")

    cfg = haf.build_trial_config(
        {"instance_id": "task-1", "agent_name": "mini-swe-agent"},
        "http://trainer:30000/sessions/s1/v1",
        {},
    )

    assert cfg.environment.cpu_enforcement_policy.value == "request"
    assert cfg.environment.memory_enforcement_policy.value == "limit"


def test_the_trial_entrypoints_exist():
    # [FUNCTION DOC] Asserts the real trial class has callable create and run attributes.
    assert callable(Trial.create) and callable(Trial.run)


def test_the_result_fields_the_mapping_reads_still_exist():
    # [FUNCTION DOC] Asserts the real trial result model's field names include start and finish
    # times, environment setup, agent setup, agent execution, verifier, exception info, verifier
    # result and agent result.
    """trial_result_to_metadata reads these via getattr(default None): a rename
    would silently drop metrics/timings, so lock the names here."""
    from harbor.models.trial.result import TrialResult

    fields = set(TrialResult.model_fields)
    assert {
        "started_at",
        "finished_at",
        "environment_setup",
        "agent_setup",
        "agent_execution",
        "verifier",
        "exception_info",
        "verifier_result",
        "agent_result",
    } <= fields
