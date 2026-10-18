# [MODULE DOC]
# Entry point of the multi-policy split-deployment deploy test (disabled in CI: it needs a
# Kubernetes cluster backend). Registered as a 2400-second 4-GPU H200 CUDA test (hopper and
# blackwell; deploy, multi-policy and fully-async labels). Run as a script it executes the
# multi-policy split scenario: solver and verifier policies on GSM8K deployed as separate
# releases, then verifies training arguments, rollouts, finite metrics and train/rollout
# log-probability agreement.
# NOTE: You MUST read tests/e2e/deploy/README.md as source-of-truth and documentations
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.split.scenario_split_multi_policy import run_ci

register_cuda_ci(
    est_time=2400,
    suite="stage-c-4-gpu-h200",
    labels=["deploy", "multi-policy", "fully-async"],
    hardware=["hopper", "blackwell"],
    disabled="needs a Kubernetes cluster backend",
)

if __name__ == "__main__":
    run_ci()
