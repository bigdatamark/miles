# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# Thin CI entry: registers the test and runs it via bare `python3 <file>`
# (the CUDA CI runner's execution model). Scenario logic lives in
# tests/e2e/ft/conftest_ft/scenario_realistic_gsm8k_fully_async.py.

# [MODULE DOC] CI entry running the fully-async variant of the realistic gsm8k fault-recovery
# scenario. Takes no mode. Registers a CUDA test (estimated 9000 seconds, 8-GPU H200 stage,
# long-FT label, hopper and blackwell) that is currently disabled (soak tests pending CI
# infra). When executed as a script it runs the scenario with defaults; importing only
# registers.

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.ft.conftest_ft.scenario_realistic_gsm8k_fully_async import run_ci

register_cuda_ci(
    est_time=9000,
    suite="stage-c-8-gpu-h200",
    labels=["ft-long"],
    hardware=["hopper", "blackwell"],
    disabled="FT soak tests pending CI infra support",
)

if __name__ == "__main__":
    run_ci()
