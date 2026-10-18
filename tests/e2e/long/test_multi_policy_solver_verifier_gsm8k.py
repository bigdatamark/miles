# [MODULE DOC] Long end-to-end test of the multi-policy solver/verifier gsm8k example recipe
# (CUDA: 4-GPU H200 stage, estimated 21600 seconds, long label, hopper and blackwell; ROCm:
# nightly 4-GPU MI350 suite, estimated 24000 seconds). Run as a script: takes the example's
# default script arguments from the default config, overrides the rollout count with an
# environment variable (default 250), runs the example's prepare step, removes all proxy
# environment variables (both cases) from the process, and runs the shared multi-policy execute
# helper with default wandb args and per-model evaluation score bounds. The bounds (calibrated
# from a full 250-rollout run, set at about a third of the observed growth) require, for the
# solver, an initial score at most 0.52, a peak at least 0.53 and growth at least 0.03, and for
# the verifier an initial score at most 0.65, a peak at least 0.70 and growth at least 0.08.
# Importing only registers the test and defines the constants.

import dataclasses
import os

from examples.multi_policy.run_solver_verifier_gsm8k import SOLVER_MODEL_ID, VERIFIER_MODEL_ID, ScriptArgs, prepare
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.conftest_multi_policy import EvalScoreBounds, execute

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=21600, suite="stage-c-4-gpu-h200", labels=["long"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=24000, suite="nightly-stage-c-4-gpu-mi350", labels=["long"])

NUM_ROLLOUT = int(os.environ.get("MILES_TEST_NUM_ROLLOUT", "250"))

# Calibrated against a full 250-rollout run of this recipe: eval/gsm8k/solver
# rose .473 -> .566 (first -> best point) and eval/gsm8k/verifier .569 -> .821;
# thresholds sit at roughly one third of the observed growth.
EVAL_SCORE_BOUNDS = {
    SOLVER_MODEL_ID: EvalScoreBounds(initial_max=0.52, peak_min=0.53, min_growth=0.03),
    VERIFIER_MODEL_ID: EvalScoreBounds(initial_max=0.65, peak_min=0.70, min_growth=0.08),
}


if __name__ == "__main__":
    args = dataclasses.replace(command_utils.default_config(ScriptArgs), num_rollout=NUM_ROLLOUT)
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        args,
        wandb_args=command_utils.get_default_wandb_args(__file__),
        eval_score_bounds=EVAL_SCORE_BOUNDS,
    )
