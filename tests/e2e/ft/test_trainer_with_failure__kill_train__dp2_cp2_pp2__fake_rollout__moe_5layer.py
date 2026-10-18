# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_trainer_with_failure.py.

# [MODULE DOC] CI entry running the trainer with-failure comparison scenario on the train-kill
# mode with 2 cells, context and pipeline parallel 2, replayed (fake) rollout, 5-layer MoE
# model. Registers a CUDA test (estimated 2100 seconds, 8-GPU H200 stage, short-FT label,
# hopper and blackwell) that is currently disabled (to be enabled in a future FT delivery). When
# executed as a script it runs the scenario for that mode; importing only registers.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_trainer_with_failure import run_ci

register_cuda_ci(
    est_time=2100,
    suite="stage-c-8-gpu-h200",
    labels=["ft-short"],
    hardware=["hopper", "blackwell"],
    disabled="will enable in future FT delivery",
)

_MODE: str = "kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer"

if __name__ == "__main__":
    run_ci(_MODE)
