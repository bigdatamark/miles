# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin per-mode CI entry: registers the test and runs ONE mode via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_trainer_no_failure.py.

# [MODULE DOC] CI entry running the trainer no-failure comparison scenario on the train-kill mode
# with 4 cells, context parallel 2, replayed (fake) rollout, 5-layer MoE model. Registers an
# enabled CUDA test (estimated 900 seconds, 8-GPU H100 stage, short-FT label, hopper and
# blackwell) and an AMD ROCm nightly test (estimated 1100 seconds, 8-GPU MI350 nightly suite,
# short-FT label). When executed as a script it runs the scenario for that mode; importing only
# registers.

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.ft.conftest_ft.scenario_trainer_no_failure import run_ci

register_cuda_ci(
    est_time=900,
    suite="stage-c-8-gpu-h100",
    labels=["ft-short"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(est_time=1100, suite="nightly-stage-c-8-gpu-mi350", labels=["ft-short"])

_MODE: str = "kill_train__dp4_cp2__fake_rollout__moe_5layer"

if __name__ == "__main__":
    run_ci(_MODE)
