# [MODULE DOC]
# Offline tests for the harbor launcher helper module (imported by bare name): the function that
# turns parsed launcher arguments into the environment variables forwarded to the harbor workers
# (environment type, tasks dir, provider key file paths and urls, optional knobs read from the
# process environment) with preflight checks, and the function building the agentic training
# argument string. Harbor and the provider sdks are replaced by empty modules so they only need to
# be importable.
"""Offline tests for the launcher's Harbor-side wiring (no harbor, no keys)."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import launch_common
import pytest


def _args(**overrides):
    # [FUNCTION DOC] HELPER: Builds a simple namespace of launcher args with defaults (environment
    # type e2b, empty env kwargs, tasks dir /tasks, trials dir /trials, model name, agent timeout
    # 5400, empty key file and modal config paths), updated by the keyword overrides.
    defaults = dict(
        harbor_env_type="e2b",
        harbor_env_kwargs="",
        harbor_tasks_dir="/tasks",
        harbor_trials_dir="/trials",
        agent_model_name="model",
        agent_timeout=5400,
        daytona_api_key_file="",
        e2b_api_key_file="",
        modal_config_file="",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def fake_harbor_and_sdk(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture installing empty placeholder modules for harbor, e2b,
    # daytona and modal in the module table so the preflight import succeeds.
    """harbor and the provider SDKs only need to be importable for the preflight."""
    for name in ("harbor", "e2b", "daytona", "modal"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_env_type_is_required(monkeypatch):
    # [FUNCTION DOC] An empty environment type raises a value error naming the environment type variable.
    with pytest.raises(ValueError, match="HARBOR_ENV_TYPE"):
        launch_common.harbor_env_vars(_args(harbor_env_type=""))


def test_docker_is_refused_with_a_pointer_to_the_agent_server():
    # [FUNCTION DOC] Environment type docker raises a value error pointing to the docker agent
    # server recipe; the same with surrounding spaces and mixed case raises a value error about
    # the Docker daemon (the guard normalizes the value).
    with pytest.raises(ValueError, match="swe-agent-harbor-docker"):
        launch_common.harbor_env_vars(_args(harbor_env_type="docker"))
    # the worker normalizes HARBOR_ENV_TYPE, so the guard must see the same value
    with pytest.raises(ValueError, match="Docker daemon"):
        launch_common.harbor_env_vars(_args(harbor_env_type=" Docker "))


def test_missing_harbor_names_the_install(monkeypatch):
    # [FUNCTION DOC] With harbor made unimportable (module table entry none), asserts a runtime
    # error mentioning the README install line.
    monkeypatch.setitem(sys.modules, "harbor", None)
    with pytest.raises(RuntimeError, match="README install line"):
        launch_common.harbor_env_vars(_args())


def test_known_provider_is_provisioned_by_key_path(monkeypatch, tmp_path):
    # [FUNCTION DOC] With an e2b key file and an e2b api url in the process environment, asserts the
    # returned env maps: environment type e2b, tasks dir, key file path (the path, not the secret),
    # and the api url; and that the secret text appears nowhere in the mapping.
    key_file = tmp_path / "api_key"
    key_file.write_text("e2b_secret\n")
    monkeypatch.setenv("E2B_API_URL", "http://agentenv.internal:8000")

    env = launch_common.harbor_env_vars(_args(e2b_api_key_file=str(key_file)))

    assert env["HARBOR_ENV_TYPE"] == "e2b"
    assert env["HARBOR_TASKS_DIR"] == "/tasks"
    assert env["E2B_API_KEY_FILE"] == str(key_file)
    assert env["E2B_API_URL"] == "http://agentenv.internal:8000"
    assert "e2b_secret" not in str(env)


def test_unknown_provider_passes_through_with_a_notice(monkeypatch, capsys):
    # [FUNCTION DOC] For an unknown provider (runloop), asserts the env keeps the type, the
    # printed output contains a notice that no credential wiring is known for it, and no env
    # variable name contains KEY.
    """The backend still reaches Harbor untouched; the operator is told no
    credential wiring exists for it."""
    env = launch_common.harbor_env_vars(_args(harbor_env_type="runloop"))
    assert env["HARBOR_ENV_TYPE"] == "runloop"
    assert "no credential wiring known for 'runloop'" in capsys.readouterr().out
    assert not any("KEY" in var for var in env)


def test_env_kwargs_and_server_knobs_are_forwarded_when_set(monkeypatch):
    # [FUNCTION DOC] Sets an e2b key, response length policy abort, cpu override 2 and cpu
    # enforcement request in the process environment (and removes the max sequence length), with
    # env kwargs json in the args. Asserts those four values and the kwargs are forwarded and the
    # unset max sequence length is not.
    monkeypatch.setenv("E2B_API_KEY", "e2b_x")  # worker-env key supply
    monkeypatch.setenv("HARBOR_RESPONSE_LENGTH_POLICY", "abort")
    monkeypatch.setenv("HARBOR_OVERRIDE_CPUS", "2")
    monkeypatch.setenv("HARBOR_CPU_ENFORCEMENT_POLICY", "request")
    monkeypatch.delenv("HARBOR_MAX_SEQ_LEN", raising=False)

    env = launch_common.harbor_env_vars(_args(harbor_env_kwargs='{"auto_snapshot": true}'))

    assert env["HARBOR_ENV_KWARGS"] == '{"auto_snapshot": true}'
    assert env["HARBOR_RESPONSE_LENGTH_POLICY"] == "abort"
    assert env["HARBOR_OVERRIDE_CPUS"] == "2"
    assert env["HARBOR_CPU_ENFORCEMENT_POLICY"] == "request"
    assert "HARBOR_MAX_SEQ_LEN" not in env


def test_agentic_wiring_is_the_shipped_wiring():
    # [FUNCTION DOC] Builds agentic train args for model family qwen3 and four session server
    # workers and asserts they contain the custom agent function path, the agentic tool call
    # generate function path, the no-aborted dynamic sampling filter path, the tito model, the
    # worker count, and the default session server port 30000. Imports the two miles-side
    # functions to prove those paths resolve, and asserts every directory returned by the
    # pythonpath helper exists.
    """One shared copy: the flags the GPU e2e exercises are the flags the recipes ship."""
    args = launch_common.agentic_train_args(tito_model="qwen3", session_server_workers=4)
    assert "--custom-agent-function-path harbor_agent_function.run" in args
    assert "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate" in args
    assert "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted" in args
    assert "--tito-model qwen3" in args and "--session-server-workers 4" in args
    assert "--session-server-port 30000" in args  # the default; multi-launcher hosts pass their own
    # the miles-side paths must resolve, or every launcher fails at load_function time
    from miles.rollout.filter_hub.dynamic_sampling_filters import check_no_aborted  # noqa: F401
    from miles.rollout.generate_hub.agentic_tool_call import generate  # noqa: F401

    for d in launch_common.agentic_pythonpath_dirs():
        assert Path(d).is_dir(), d
