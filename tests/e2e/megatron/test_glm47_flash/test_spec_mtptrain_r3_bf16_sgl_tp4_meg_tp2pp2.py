# [MODULE DOC] GLM-4.7-Flash case: colocated on 4 GPUs with Megatron tensor parallel 2, pipeline
# parallel 2, context parallel 1, expert parallel 2, and one SGLang engine of 4 GPUs (tensor
# parallel 4); speculative decoding with MTP training and rollout routing replay on (case
# defaults), bf16. Same shape as the AMD standalone variant. Registers a CUDA test (4-GPU H200
# stage, estimated 1200 seconds, label megatron, hopper and blackwell) and an AMD ROCm nightly
# test (4-GPU MI350, estimated 1500 seconds), plus metric history gates on grad norm, PPO KL,
# train/rollout logprob absolute difference, train/rollout KL and raw reward. Run as a script:
# strip all proxy environment variables (both cases), prepare, execute.

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1200, suite="stage-c-4-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=1500, suite="nightly-stage-c-4-gpu-mi350", labels=["megatron"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=False,
    # tp2/pp2/cp1/ep2 on 4 GPUs, the same shape as test_amd_spec_mtptrain_r3_bf16_sgl_tp4_meg_tp2pp2.
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=2,
    tp_size=2,
    ep_size=2,
    # GLM-4.7-Flash has 20 attention heads; non-EP SGLang TP must divide it.
    rollout_num_gpus_per_engine=4,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
