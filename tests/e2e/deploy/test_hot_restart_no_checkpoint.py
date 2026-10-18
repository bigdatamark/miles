# [MODULE DOC]
# Entry point of the no-checkpoint hot-restart deploy test (disabled in CI: it needs a Kubernetes
# cluster backend). Registered as a 6000-second 8-GPU H200 CUDA test (hopper and blackwell,
# deploy label). Run as a script it executes the "no_checkpoint" scenario: a deployed run saving
# every 4 rollouts is frozen after step 1, before its first save, and taken over by a hot restart;
# the test verifies the run restarted from scratch (every earlier step retrained), saved after
# the restart, and that deterministic metrics match an untouched baseline run.
# NOTE: You MUST read tests/e2e/deploy/README.md as source-of-truth and documentations
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.hot_restart.scenario_hot_restart_deterministic import NO_CHECKPOINT, run_ci

register_cuda_ci(
    est_time=6000,
    suite="stage-c-8-gpu-h200",
    labels=["deploy"],
    hardware=["hopper", "blackwell"],
    disabled="needs a Kubernetes cluster backend",
)

if __name__ == "__main__":
    run_ci(NO_CHECKPOINT)
