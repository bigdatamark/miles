# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_trainer_no_failure.py.

# [MODULE DOC] CI entry running the trainer no-failure comparison scenario on the train-kill mode
# with 2 cells and context parallel 2 on the 5-layer MoE model (real rollout). Registers an
# enabled CUDA test (estimated 1200 seconds, 8-GPU H100 stage, short-FT label, hopper and
# blackwell). When executed as a script it runs the scenario for that mode; importing only
# registers.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_trainer_no_failure import run_ci

register_cuda_ci(
    est_time=1200,
    suite="stage-c-8-gpu-h100",
    labels=["ft-short"],
    hardware=["hopper", "blackwell"],
)

_MODE: str = "kill_train__dp2_cp2__moe_5layer"

if __name__ == "__main__":
    run_ci(_MODE)
