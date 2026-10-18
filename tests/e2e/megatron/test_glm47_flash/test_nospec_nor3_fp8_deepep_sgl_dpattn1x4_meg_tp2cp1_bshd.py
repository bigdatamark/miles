# [MODULE DOC] GLM-4.7-Flash case: fixed micro-batch (BSHD) training with DeepEP on both training
# and rollout sides, FP8 rollout, without speculative decoding or routing replay; colocated on 4
# GPUs with Megatron tensor parallel 2 and expert parallel 4 (context and pipeline parallel 1),
# micro batch size 2, and one SGLang engine of 4 GPUs using data-parallel attention (DP size 4)
# with expert parallel 4. Registers a CUDA test (4-GPU H200 stage, estimated 1400 seconds, label
# megatron, hopper only; 4 GPUs because 8-GPU runners cannot bring NVSHMEM up over IB, which
# DeepEP low-latency mode needs) and metric history gates on grad norm, PPO KL, train/rollout
# logprob absolute difference, train/rollout KL and raw reward. Run as a script: strip all proxy
# environment variables (both cases), prepare, execute.

"""GLM-4.7-Flash in the BSHD layout with DeepEP on both sides and an FP8 rollout, without spec or R3.

Colocated on 4 GPUs: Megatron trains fixed micro-batches of 2 samples, each padded to the longest
sample on its data-parallel rank, at TP2 (DP2) with EP4 over DeepEP. One SGLang engine serves
RadixArk/glm47-flash-blockwise-fp8 with DP attention (attention TP1 x DP4), EP4 and DeepEP auto; every
weight update re-quantizes, so the equality check allows quantization error.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

# 4x H200: the 8x H200 runners cannot bring NVSHMEM up over IB, which DeepEP low-latency mode needs.
register_cuda_ci(est_time=1400, suite="stage-c-4-gpu-h200", labels=["megatron"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=True,
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    qkv_format="bshd",
    # Each of the 2 data-parallel ranks gets 16 samples per step: 8 micro-batches of 2.
    micro_batch_size=2,
    rollout_num_gpus_per_engine=4,
    # Attention TP1 x DP4: the FP8 checkpoint loads only at attention TP <= 2 (see _common).
    sglang_dp_size=4,
    sglang_enable_dp_attention=True,
    # SGLang DeepEP forces EP to the engine TP.
    sglang_ep_size=4,
    use_fp8_rollout=True,
    use_spec=False,
    use_r3=False,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
