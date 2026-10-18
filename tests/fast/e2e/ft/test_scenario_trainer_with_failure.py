# [MODULE DOC]
# Tests for the argument builders of the trainer-with-failure ft scenario's second phase (phase B)
# for the baseline and target sides. The built argument strings are split with shell-style
# splitting into tokens and inspected: deterministic-collective and gradient clipping options for
# real-rollout versus fake-rollout modes, and the test-action schedule and data-injection start
# rollout for the target side. Module constants name one real-rollout mode and one fake-rollout mode.
import json
import shlex

from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_trainer_with_failure import (
    NUM_PHASE_A_STEPS,
    _build_baseline_args,
    _build_target_args,
)

_REAL_ROLLOUT_MODE = "kill_train__dp2_cp2"
_FAKE_ROLLOUT_MODE = "kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer"


def _option_value(tokens: list[str], option: str) -> str:
    # [FUNCTION DOC] HELPER: Returns the token right after the first occurrence of the option
    # token. Raises a value error if the option is absent; an index error if it is the last token.
    return tokens[tokens.index(option) + 1]


def _phase_b_tokens(mode_name: str, side: str) -> list[str]:
    # [FUNCTION DOC] HELPER: Builds the phase B arguments for the named mode with the baseline
    # builder when side is baseline, otherwise the target builder, using a per-side temp dump path
    # and the dumper disabled, then splits the string into shell tokens.
    build = _build_baseline_args if side == "baseline" else _build_target_args
    return shlex.split(build(MODES[mode_name], f"/tmp/{side}/phase_b", enable_dumper=False))


def test_real_rollout_sides_share_deterministic_collective_and_inactive_clipping() -> None:
    # [FUNCTION DOC] For both sides of the real-rollout mode, asserts the debug deterministic
    # collective flag is present and the clip-grad option value is 10.0.
    """Both sides of the live-rollout comparison must fold reductions identically with clipping inactive."""
    for side in ("baseline", "target"):
        tokens = _phase_b_tokens(_REAL_ROLLOUT_MODE, side)
        assert "--debug-deterministic-collective" in tokens
        assert _option_value(tokens, "--clip-grad") == "10.0"


def test_fake_rollout_modes_keep_the_production_collective() -> None:
    # [FUNCTION DOC] For both sides of the fake-rollout mode, asserts neither the deterministic
    # collective flag nor the clip-grad option is present.
    """Modes without live rollout keep the production reduction path and default clipping."""
    for side in ("baseline", "target"):
        tokens = _phase_b_tokens(_FAKE_ROLLOUT_MODE, side)
        assert "--debug-deterministic-collective" not in tokens
        assert "--clip-grad" not in tokens


def test_injection_starts_after_the_fault_rollout() -> None:
    # [FUNCTION DOC] Parses the target side's CI ft test actions option as JSON and reads the data
    # injection start rollout id. Asserts every action fires at the rollout one past the phase A
    # step count, the injection starts one rollout later, and the baseline side has no injection
    # data path option.
    """Only post-fault rollouts may inject baseline data; the fault rollout trains live samples."""
    target_tokens = _phase_b_tokens(_REAL_ROLLOUT_MODE, "target")
    actions = json.loads(_option_value(target_tokens, "--ci-ft-test-actions"))
    fault_rollout_ids = {action["at_rollout"] for action in actions}
    start = int(_option_value(target_tokens, "--ci-inject-rollout-data-start-rollout-id"))

    assert fault_rollout_ids == {NUM_PHASE_A_STEPS + 1}
    assert start == NUM_PHASE_A_STEPS + 2
    assert "--ci-inject-rollout-data-path" not in _phase_b_tokens(_REAL_ROLLOUT_MODE, "baseline")
