# [MODULE DOC] Single-case end-to-end test of rollout routing replay (R3) for Qwen3-30B-A3B on 4
# GPUs, colocated, BF16 rollout, no DeepEP: Megatron context parallel 2, pipeline 1, tensor 2,
# expert 4; one 4-GPU SGLang engine. This is the topology reused by the DeepEP R3 variants.
# Registered for CUDA 4-GPU H200 (est. 1200 s) and ROCm 4-GPU MI350 nightly (est. 1000 s), both
# currently disabled, with five CI metric gates (grad norm, PPO KL, train/rollout logprob
# absolute difference, train/rollout KL, raw reward). Run as a script: strips proxy variables,
# then prepare and execute.
# KNOWN ISSUE: both registrations are disabled with a reason that cites no issue or PR.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1200,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "replay"],
    hardware=["hopper", "blackwell"],
    disabled="Outdated and simple.",
)
register_rocm_ci(
    est_time=1000, suite="nightly-stage-c-4-gpu-mi350", labels=["megatron", "replay"], disabled="Outdated and simple."
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=False,
    use_fp8_rollout=False,
    use_int4_rollout=False,
    use_bridge=False,
    use_r3=True,
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=4,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
