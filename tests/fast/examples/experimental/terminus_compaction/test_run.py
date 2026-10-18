# [MODULE DOC]
# Tests for the launch script of the terminus compaction example (examples/experimental/terminus-compaction/run.py,
# loaded at import time through the shared launch-script import helper). Builds the script's
# argument object with fixed run id, model and output dirs, session server host values and an agent
# server url, then checks the training argument string (split shell-style into tokens) and the
# extra environment variables the script derives. Tests remove the wandb key from the environment
# where the training args are built.
import shlex

import pytest

from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

_EXAMPLE_DIR = REPO_ROOT / "examples" / "experimental" / "terminus-compaction"
run = import_launch_script(_EXAMPLE_DIR / "run.py")


def _value(argv: list[str], option: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the token right after the first occurrence of the option
    # token. Raises a value error if the option is absent.
    return argv[argv.index(option) + 1]


@pytest.fixture
def args():
    # [FUNCTION DOC] HELPER: Fixture building the script's argument object with run id
    # 260101-example, model dir /models, output dir /output, session server ip 0.0.0.0, external
    # host trainer.example and agent server url http://agent.example:11000; all other fields keep script defaults.
    return run.ScriptArgs(
        run_id="260101-example",
        model_dir="/models",
        output_dir="/output",
        session_server_ip="0.0.0.0",
        session_server_external_host="trainer.example",
        agent_server_url="http://agent.example:11000",
    )


def test_recipe_enables_compaction_aware_session_training(args, monkeypatch):
    # [FUNCTION DOC] Without a wandb key, builds the train args and asserts: session server mode
    # v2, session server ip 0.0.0.0, tito model glm47, rollout batch size 4, 8 samples per prompt,
    # global batch size 32, 100 rollouts, max response length 8192, and prompt data path
    # /root/tb2_train_89.jsonl.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    argv = shlex.split(run._build_train_args(args))

    assert _value(argv, "--use-session-server") == "v2"
    assert _value(argv, "--session-server-ip") == "0.0.0.0"
    assert _value(argv, "--tito-model") == "glm47"
    assert _value(argv, "--rollout-batch-size") == "4"
    assert _value(argv, "--n-samples-per-prompt") == "8"
    assert _value(argv, "--global-batch-size") == "32"
    assert _value(argv, "--num-rollout") == "100"
    assert _value(argv, "--rollout-max-response-len") == "8192"
    assert _value(argv, "--prompt-data") == "/root/tb2_train_89.jsonl"


def test_recipe_records_dashboard_traces_and_honors_gpu_count(args, monkeypatch):
    # [FUNCTION DOC] Without a wandb key, builds the train args and asserts: dump details dir is
    # output dir / run id / details, 8 gpus per node, the dashboard, rollout entropy, training
    # entropy and multi-turn logging flags are present, and no separate rollout gpu count option.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    argv = shlex.split(run._build_train_args(args))

    assert _value(argv, "--dump-details") == "/output/260101-example/details"
    assert _value(argv, "--num-gpus-per-node") == "8"
    assert "--use-miles-dashboard" in argv
    assert "--use-rollout-entropy" in argv
    assert "--observe-training-entropy" in argv
    assert "--log-multi-turn" in argv
    assert "--rollout-num-gpus" not in argv


def test_session_bind_override_is_optional(args, monkeypatch):
    # [FUNCTION DOC] With an empty session server ip, asserts the train args omit the session server ip option.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    args.session_server_ip = ""

    assert "--session-server-ip" not in shlex.split(run._build_train_args(args))


def test_agent_runtime_environment_reuses_harbor_adapters(args):
    # [FUNCTION DOC] Asserts the extra environment variables are exactly: python path pointing at
    # the swe-agent harbor docker example directory under the repository root, the agent server
    # url, the agent model name equal to the model dir basename-style value model, and the agent
    # trial timeout 7200.
    # KNOWN ISSUE: the model dir given in the args fixture is /models but the expected agent model
    # name is model; the derivation is not visible from this test (it depends on the script).
    env = run._extra_env_vars(args)

    assert env == {
        "PYTHONPATH": str(REPO_ROOT / "examples" / "swe-agent-harbor-docker"),
        "AGENT_SERVER_URL": "http://agent.example:11000",
        "AGENT_MODEL_NAME": "model",
        "AGENT_TRIAL_TIMEOUT": "7200",
    }


def test_agent_runtime_environment_omits_optional_hosts(args):
    # [FUNCTION DOC] With an empty host ip attribute on the args, asserts the extra environment
    # variables do not include the host ip variable.
    args.miles_host_ip = ""

    env = run._extra_env_vars(args)

    assert "MILES_HOST_IP" not in env
