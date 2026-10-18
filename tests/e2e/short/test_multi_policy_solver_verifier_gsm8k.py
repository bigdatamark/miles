# [MODULE DOC] End-to-end test of multi-policy training with a solver and a verifier policy on
# GSM8K, fully asynchronous, driven by the example launcher script (its argument class and prepare
# step) and a shared multi-policy execute helper from the e2e conftest module. There are no
# test functions or constants beyond the rollout count (from an environment variable, default 5)
# and a save interval of 2. When run as a script it builds the arguments from the default config
# with those two overrides, prepares, strips proxy variables, then executes with the default
# experiment-tracking arguments for this file. Registered for CUDA 4-GPU H200 (est. 3000 s) and
# ROCm 4-GPU MI350 nightly (est. 1700 s), labels short, multi-policy and fully-async.
# KNOWN ISSUE: the CUDA estimate of 3000 s exceeds the 2400 s cap for registrations that are
# neither nightly nor labeled long.
import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import ScriptArgs, prepare
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.conftest_multi_policy import execute

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=3000,
    suite="stage-c-4-gpu-h200",
    labels=["short", "multi-policy", "fully-async"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(est_time=1700, suite="nightly-stage-c-4-gpu-mi350", labels=["short", "multi-policy", "fully-async"])

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "5"))
SAVE_INTERVAL = 2


if __name__ == "__main__":
    args = dataclasses.replace(
        command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT, save_interval=SAVE_INTERVAL
    )
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args, wandb_args=command_utils.get_default_wandb_args(__file__))
