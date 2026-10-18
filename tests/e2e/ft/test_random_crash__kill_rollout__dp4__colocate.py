# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_random_crash.py.

# [MODULE DOC] CI entry running the random-crash soak scenario on the colocated rollout-kill mode
# with 4 data-parallel cells. Registers a CUDA test (estimated 4800 seconds, 8-GPU H200 stage,
# long-FT label, hopper and blackwell) that is currently disabled with a recorded reason
# (soak tests pending CI infra; a fast-layer test stands in). When executed as a script it runs
# the scenario for that mode through the shared run function; importing only registers.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_random_crash import run_ci

register_cuda_ci(
    est_time=4800,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long"],
    hardware=["hopper", "blackwell"],
    disabled=(
        "FT soak tests pending CI infra support: every ft-long entry is disabled for the same reason, so "
        "run-ci-ft-long currently schedules nothing. Unblock condition: an ft-long capable "
        "stage-c-8-gpu-h200 lane, then drop this argument -- nothing in the test itself is known broken. "
        "Until then tests/fast/e2e/ft/test_rollout_gated_recovery.py is the fast-layer stand-in for this entry "
        "(suspend -> gated relaunch -> recovery, no GPU). See tests/e2e/ft/README.md."
    ),
)

_MODE: str = "kill_rollout__dp4__colocate"

if __name__ == "__main__":
    run_ci(_MODE)
