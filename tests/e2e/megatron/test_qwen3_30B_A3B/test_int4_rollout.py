# [MODULE DOC] Single-case end-to-end test: Qwen3-30B-A3B on 4 GPUs, colocated, with INT4 rollout
# (fake quantization-aware training on the training side, an INT4-converted checkpoint for
# SGLang). Megatron side: context parallel 2, pipeline 1, tensor 2, expert 4; SGLang side: four
# single-GPU engines. The shared preparation step runs the HF-to-INT4 conversion and the shared
# argument builder adds the INT4 environment and quantization-tolerant weight-update check.
# Registered for CUDA 4-GPU H200 (est. 1700 s, Hopper and Blackwell) with five CI metric gates
# (grad norm, PPO KL, train/rollout logprob absolute difference, train/rollout KL, raw reward).
# Run as a script: strips proxy variables, then prepare and execute.
import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(est_time=1700, suite="stage-c-4-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=False,
    use_fp8_rollout=False,
    use_int4_rollout=True,
    use_bridge=False,
    use_r3=False,
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=1,
    tp_size=2,
    ep_size=4,
    rollout_num_gpus_per_engine=1,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
