# [MODULE DOC]
# Entry point of the checkpointed hot-restart deploy test (disabled in CI: it needs a Kubernetes
# cluster backend). Registered as a 5400-second 8-GPU H200 CUDA test (hopper and blackwell,
# deploy label). Run as a script it executes the "checkpointed" scenario: a deployed run saving
# every 2 rollouts is frozen after steps 2 and 4 and taken over by hot restarts, and the test
# verifies that only the steps after the latest checkpoint were redone and that deterministic
# metrics match an untouched baseline run.
# NOTE: You MUST read tests/e2e/deploy/README.md as source-of-truth and documentations
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import CHECKPOINTED, run_ci

register_cuda_ci(
    est_time=5400,
    suite="stage-c-8-gpu-h200",
    labels=["deploy"],
    hardware=["hopper", "blackwell"],
    disabled="needs a Kubernetes cluster backend",
)

if __name__ == "__main__":
    run_ci(CHECKPOINTED)
