# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_rollout_deterministic.py.

# [MODULE DOC] CI entry running the rollout-engine deterministic comparison scenario on the
# colocated rollout-kill mode with 4 cells. Registers a CUDA test (estimated 3600 seconds,
# 8-GPU H200 stage, long-FT label, hopper and blackwell) that is currently disabled (soak tests
# pending CI infra; its disable reason notes two uncalibrated constants, a loss-decrease
# constant and the time estimate). When executed as a script it runs the scenario for that mode;
# importing only registers.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_rollout_deterministic import run_ci

register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long"],
    hardware=["hopper", "blackwell"],
    disabled=(
        "FT soak tests pending CI infra support: every ft-long entry is disabled for the same reason, and the "
        "specific infra gap is not recorded anywhere in the repo. Unblock condition: an ft-long capable "
        "stage-c-8-gpu-h200 lane, then drop this argument. Two of this entry's constants are also uncalibrated "
        "until it has run once: MIN_LOSS_DECREASE and est_time. See tests/e2e/ft/README.md."
    ),
)

_MODE: str = "kill_rollout__dp4__colocate"

if __name__ == "__main__":
    run_ci(_MODE)
