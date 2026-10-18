# [MODULE DOC]
# Tests for the verifiers example launch script's execute function. Shell commands are recorded
# instead of run, the Ray backend's nvlink check is patched to report none, and external-Ray and
# submit environment flags are set so the final recorded command is the ray job submission. The
# tests assert the legacy rollout flag (an environment variable) is consistent between the
# adapter's choice of rollout function (new class-based function versus legacy generate function)
# and the runtime environment forwarded to the Ray job, for combinations of the ambient process
# value and an extra environment variable assignment given through the script arguments.
import json
import shlex

import pytest
from examples.experimental.verifiers import run
from tests.fast.utils.command_recorder import patch_helper, record_commands

from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend

LEGACY_ROLLOUT_ENV = "MILES_USE_LEGACY_ROLLOUT_V1"


def _rollout_config(submit_command: str) -> tuple[str, dict[str, str]]:
    # [FUNCTION DOC] HELPER: Splits the submit command shell-style, returns a pair: the value after
    # the rollout function path option, and the environment variables mapping parsed from the json
    # of the runtime-env argument (the part after the equals sign). Raises if either is missing.
    argv = shlex.split(submit_command)
    rollout_fn = argv[argv.index("--rollout-function-path") + 1]
    runtime_env_arg = next(arg for arg in argv if arg.startswith("--runtime-env-json="))
    runtime_env = json.loads(runtime_env_arg.split("=", 1)[1])["env_vars"]
    return rollout_fn, runtime_env


@pytest.mark.parametrize(
    ("ambient_value", "extra_env_vars", "expected_rollout_fn", "expected_runtime_value"),
    [
        (None, "", "verifiers_rollout.VerifiersRolloutFn", None),
        ("1", "", "verifiers_rollout.generate_rollout", "1"),
        ("0", f"{LEGACY_ROLLOUT_ENV}=1", "verifiers_rollout.generate_rollout", "1"),
        ("1", f"{LEGACY_ROLLOUT_ENV}=0", "verifiers_rollout.VerifiersRolloutFn", "0"),
    ],
)
def test_adapter_and_ray_runtime_use_the_same_legacy_flag(
    monkeypatch,
    tmp_path,
    ambient_value,
    extra_env_vars,
    expected_rollout_fn,
    expected_runtime_value,
):
    # [FUNCTION DOC] Parametrized over four cases: (ambient unset, no extra) gives the class-based
    # rollout function and no runtime flag; (ambient 1, no extra) gives the legacy function with
    # runtime flag 1; (ambient 0, extra sets flag 1) gives the legacy function with flag 1 (the
    # extra assignment wins over the ambient value); (ambient 1, extra sets flag 0) gives the
    # class-based function with flag 0. Setup: record commands, patch the nvlink check to false, set
    # external ray, submit enabled and master address variables, remove ray address and nvls
    # variables, and set or remove the ambient flag. Runs execute with a config path in the temp dir
    # and the extra env vars string; reads the last recorded command and asserts the rollout
    # function and the runtime-env flag (absent when the expected value is none).
    commands = record_commands(monkeypatch)
    patch_helper(monkeypatch, "_check_has_nvlink", lambda self: False, backend_class=RayCommandBackend)
    monkeypatch.setenv("MILES_SCRIPT_EXTERNAL_RAY", "1")
    monkeypatch.setenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("NCCL_NVLS_ENABLE", raising=False)
    if ambient_value is None:
        monkeypatch.delenv(LEGACY_ROLLOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(LEGACY_ROLLOUT_ENV, ambient_value)

    run.execute(
        run.ScriptArgs(
            verifiers_config=str(tmp_path / "verifiers.toml"),
            extra_env_vars=extra_env_vars,
        )
    )

    rollout_fn, runtime_env = _rollout_config(commands[-1])
    assert rollout_fn == expected_rollout_fn
    if expected_runtime_value is None:
        assert LEGACY_ROLLOUT_ENV not in runtime_env
    else:
        assert runtime_env[LEGACY_ROLLOUT_ENV] == expected_runtime_value
