# [MODULE DOC]
# Entry point of the deterministic split-deployment deploy test (disabled in CI: it needs a
# Kubernetes cluster backend). Registered as a 2600-second 8-GPU H200 CUDA test (hopper and
# blackwell, deploy label). Run as a script it executes the split-deterministic scenario:
# a baseline unsplit run versus the same run deployed as separate releases, comparing their
# deterministic metrics and engine counts.
# NOTE: You MUST read tests/e2e/deploy/README.md as source-of-truth and documentations
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.deploy.conftest_deploy.split.scenario_split_deterministic import run_ci

register_cuda_ci(
    est_time=2600,
    suite="stage-c-8-gpu-h200",
    labels=["deploy"],
    hardware=["hopper", "blackwell"],
    disabled="needs a Kubernetes cluster backend",
)

if __name__ == "__main__":
    run_ci()
