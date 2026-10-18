# [MODULE DOC]
# Tests for the peer-to-peer weight transfer example's run script (loaded from the examples folder
# as a module through its file path, once per test module). The script keeps a table of run
# configurations per model, some pinning a rotary base value. Tests check that the model-argument
# environment is empty without a pinned rotary base and carries the model arguments rotary base
# variable (as a string) with one, that a pinned base survives into the real bash command's
# expanded model argument array, and that every listed profile expands to a non-empty argument list
# starting with a flag. Expansion runs a real bash subprocess.
import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
RUN_SCRIPT = REPO_ROOT / "examples" / "infra_features" / "p2p_weight_transfer" / "run.py"

_PROFILES_PINNING_A_ROTARY_BASE = ["Qwen3-235B-A22B-Instruct-2507", "Qwen3-30B-A3B", "GLM-4.5-Air"]


@pytest.fixture(scope="module")
def run_module() -> ModuleType:
    # [FUNCTION DOC] HELPER: Module-scoped fixture that loads the run script as a module named
    # p2p_weight_transfer_run from its file path, executing its top-level code, and returns it.
    spec = importlib.util.spec_from_file_location("p2p_weight_transfer_run", RUN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand_model_args(run_module: ModuleType, model_name: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Builds the script's bash command for the named run configuration,
    # appends a statement printing each element of the model argument array on its own line, runs
    # it with a real bash subprocess (shell mode, captured text output, raising on a nonzero exit),
    # and returns the stdout lines (the expanded model arguments).
    """Run the very snippet run.py hands to bash, so this keeps testing the real command."""
    command = run_module.build_model_args_command(run_module.RUN_CONFIGS[model_name])
    result = subprocess.run(
        f'{command} && printf "%s\\n" "${{MODEL_ARGS[@]}}"',
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_model_args_env_is_empty_when_the_profile_pins_no_rotary_base(run_module: ModuleType) -> None:
    # [FUNCTION DOC] For the Qwen3-4B profile, asserts its rotary base is none and the built model
    # argument environment is an empty mapping.
    """A profile without rotary_base must not inject any MODEL_ARGS_* override."""
    cfg = run_module.RUN_CONFIGS["Qwen3-4B"]
    assert cfg.rotary_base is None
    assert run_module.build_model_args_env(cfg) == {}


def test_model_args_env_carries_the_rotary_base_a_profile_pins(run_module: ModuleType) -> None:
    # [FUNCTION DOC] For the Qwen3-235B profile, asserts the rotary base is 5000000 and the model
    # argument environment is exactly one variable, the model args rotary base, set to the string 5000000.
    """A profile pinning rotary_base must surface it as the MODEL_ARGS_* name the model definition reads."""
    cfg = run_module.RUN_CONFIGS["Qwen3-235B-A22B-Instruct-2507"]
    assert cfg.rotary_base == 5000000
    assert run_module.build_model_args_env(cfg) == {"MODEL_ARGS_ROTARY_BASE": "5000000"}


@pytest.mark.parametrize("model_name", _PROFILES_PINNING_A_ROTARY_BASE)
def test_a_pinned_rotary_base_reaches_the_expanded_model_args(run_module: ModuleType, model_name: str) -> None:
    # [FUNCTION DOC] Parametrized over the three profiles pinning a rotary base. Expands the model
    # args in bash and asserts the token after the rotary-base flag equals the profile's pinned value as text.
    """The knob has to survive into the shell that expands MODEL_ARGS, not only into ray's runtime env."""
    tokens = expand_model_args(run_module, model_name)

    assert str(run_module.RUN_CONFIGS[model_name].rotary_base) == tokens[tokens.index("--rotary-base") + 1]


@pytest.mark.parametrize("model_name", sorted({"Qwen3-4B", "GLM-4.7-Flash", *_PROFILES_PINNING_A_ROTARY_BASE}))
def test_every_profile_expands_to_a_usable_argv(run_module: ModuleType, model_name: str) -> None:
    # [FUNCTION DOC] Parametrized over the sorted union of Qwen3-4B, GLM-4.7-Flash and the three
    # rotary-pinning profiles. Asserts the expansion is non-empty and its first token starts with two dashes.
    """A profile naming a model that no longer exists would submit a job with no architecture flags."""
    tokens = expand_model_args(run_module, model_name)

    assert tokens
    assert tokens[0].startswith("--")
