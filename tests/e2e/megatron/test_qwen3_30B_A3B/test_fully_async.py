# [MODULE DOC] Single-case end-to-end test of fully-asynchronous rollout for Qwen3-30B-A3B on a
# disaggregated layout (fully async cannot colocate): 3 training GPUs with pipeline parallel 3 and
# an uneven layer split (first stage 17 layers, last stage 14, middle 17) so that layer offsets
# differ from the even split on both non-first stages, plus 1 rollout GPU with one engine. Uses
# broadcast weight transfer and runs 3 rollouts to cover a cold start, a drain from a warm queue,
# and a drain across a weight update (which pauses generation and makes the persistent worker
# recycle aborted groups). Registered for CUDA 4-GPU H200 (est. 1500 s) and ROCm 4-GPU MI350
# nightly (est. 800 s), both currently disabled, with five CI metric gates (grad norm, PPO KL,
# train/rollout logprob absolute difference, train/rollout KL, raw reward). Run as a script:
# strips proxy variables, then prepare and execute.
# KNOWN ISSUE: both registrations are disabled with a reason that cites no issue or PR.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "weight-update", "fully-async"],
    hardware=["hopper", "blackwell"],
    disabled="Outdated and simple.",
)
register_rocm_ci(
    est_time=800,
    suite="nightly-stage-c-4-gpu-mi350",
    labels=["megatron", "weight-update", "fully-async"],
    disabled="Outdated and simple.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

# Fully-async rollout on the disaggregated topology (it cannot colocate). Three
# rollouts exercise the states that only exist with a persistent worker: a cold
# start, a drain from an already-warm queue, and a drain across a weight update
# (which pauses generation and makes the worker recycle aborted groups).
CASE = CaseConfig(
    use_deepep=False,
    use_fp8_rollout=False,
    use_int4_rollout=False,
    use_bridge=False,
    use_r3=False,
    # 3 train + 1 rollout with an uneven PP3 split (17/17/14): layer offsets 0/17/34 differ
    # from the even split on both non-first stages.
    num_gpus_per_node=3,
    cp_size=1,
    pp_size=3,
    tp_size=1,
    ep_size=1,
    colocate=False,
    rollout_num_gpus=1,
    rollout_num_gpus_per_engine=1,
    update_weight_transfer_mode="broadcast",
    num_rollout=3,
    fully_async=True,
    extra_args="--decoder-first-pipeline-num-layers 17 --decoder-last-pipeline-num-layers 14 ",
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
