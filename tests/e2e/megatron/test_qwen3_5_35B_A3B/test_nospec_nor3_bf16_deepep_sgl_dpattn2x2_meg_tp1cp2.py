# [MODULE DOC] Single-case end-to-end test for Qwen3.5-35B-A3B on 4 colocated GPUs: no speculative
# decoding, no MTP training, no rollout routing replay, BF16 rollout with DeepEP on both sides.
# Megatron: tensor 1, context 2, expert 4 (default flex dispatcher), max tokens per GPU 4096
# because the gated delta-net backward context-parallel kernel runs out of memory during autotune
# at 8192. SGLang: one 4-GPU engine with data-parallel attention (data parallel 2), expert
# parallel 4, DeepEP dispatching in BF16 (the automatic dtype would dispatch FP8). Weight check
# uses the target-only selector and skips vision weights. Registered for CUDA 4-GPU H200 (est.
# 1500 s, Hopper only) with five CI metric gates. Run as a script: strips proxy variables, then
# prepare and execute.
"""Qwen3.5-35B-A3B without speculative decoding or R3: the BF16 DeepEP case.

Megatron runs TP1 x CP2 with EP4 through flex (DeepEP). The rollout runs DP attention
(attention TP2 x DP2) with EP4 through SGLang DeepEP on the DeepGEMM runner with BF16
dispatch, serving the BF16 checkpoint.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_5_35B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1500, suite="stage-c-4-gpu-h200", labels=["megatron", "qwen35"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    # tp1/cp2/ep4 on 4x H200: dense DP2 with CP2 and no TP.
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
    tp_size=1,
    ep_size=4,
    # 4096 as in the other CP=2 case: the GatedDeltaNet backward's fla CP kernel OOMs in Triton
    # autotune at 8192.
    max_tokens_per_gpu=4096,
    # One TP4 engine: attention TP2 x DP2, EP4.
    rollout_num_gpus_per_engine=4,
    sglang_dp_size=2,
    sglang_enable_dp_attention=True,
    sglang_ep_size=4,
    use_deepep=True,
    # "auto" would dispatch FP8 for this BF16 model on the DeepGEMM runner.
    sglang_deepep_dispatcher_output_dtype="bf16",
    use_spec=False,
    enable_mtp_training=False,
    use_r3=False,
    check_weight_update_selector="target",
    # miles has no VLM/vision implementation on the training side, so vision weights are
    # never synced; exclude them from the weight-equality check.
    check_weight_update_skip_list=("visual",),
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
