# [MODULE DOC]
# Entry point of the realistic GSM8K hot-restart soak deploy test (disabled in CI: it needs a
# Kubernetes cluster backend and the fault-tolerance soak tests wait on CI infrastructure).
# Registered as a 12000-second 8-GPU H200 CUDA test (hopper and blackwell; deploy and ft-long
# labels). Run as a script it executes the realistic soak scenario with default options: random
# hot restarts of a deployed GSM8K training run, checking only the script is replaced and that
# each take-over loses a bounded number of steps.
# NOTE: You MUST read tests/e2e/deploy/README.md as source-of-truth and documentations
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_realistic_gsm8k import run_ci

register_cuda_ci(
    est_time=12000,
    suite="stage-c-8-gpu-h200",
    labels=["deploy", "ft-long"],
    hardware=["hopper", "blackwell"],
    disabled="needs a Kubernetes cluster backend; FT soak tests pending CI infra support",
)

if __name__ == "__main__":
    run_ci()
