# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_trainer_deterministic.py.

# [MODULE DOC] CI entry running the deterministic trainer-healing comparison scenario on the
# train-kill mode with 2 cells and context parallel 2 on the 5-layer MoE model (real rollout).
# Registers a CUDA test (estimated 3200 seconds, 8-GPU H200 stage, short-FT label, hopper and
# blackwell) that is currently disabled (to be enabled in a future FT delivery). When executed
# as a script it runs the scenario for that mode; importing only registers.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_trainer_deterministic import run_ci

register_cuda_ci(
    est_time=3200,
    suite="stage-c-8-gpu-h200",
    labels=["ft-short"],
    hardware=["hopper", "blackwell"],
    disabled="will enable in future FT delivery",
)

_MODE: str = "kill_train__dp2_cp2__moe_5layer"

if __name__ == "__main__":
    run_ci(_MODE)
