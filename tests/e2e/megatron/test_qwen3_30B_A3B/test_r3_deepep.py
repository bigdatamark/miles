# [MODULE DOC] Single-case end-to-end test of rollout routing replay (R3) with DeepEP for
# Qwen3-30B-A3B on 4 GPUs, colocated, BF16 rollout. It is the BF16 DeepEP control for the
# baseline R3 topology (Megatron context 2, pipeline 1, tensor 2, expert 4; one 4-GPU SGLang
# engine with expert parallel 4) so that a failure isolates DeepEP from FP8 rollout in the
# disabled FP8 variant. SGLang max running requests is lowered to 256 because 512 over tensor
# parallel 4 would sit exactly at the 128-token DeepEP low-latency cap. Registered for CUDA 4-GPU
# H200 (est. 1400 s) with five CI metric gates (grad norm, PPO KL, train/rollout logprob absolute
# difference, train/rollout KL, raw reward). Run as a script: strips proxy variables, then
# prepare and execute.
import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

# BF16 DeepEP control: test_r3_baseline's topology with DeepEP on both sides, so a failure
# here isolates DeepEP from the FP8 rollout in the disabled test_r3_deepep_fp8.
register_cuda_ci(
    est_time=1400, suite="stage-c-4-gpu-h200", labels=["megatron", "replay"], hardware=["hopper", "blackwell"]
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=True,
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
    sglang_ep_size=4,
    # 512 running requests / engine TP4 would sit exactly at the 128-token DeepEP low-latency cap.
    sglang_max_running_requests=256,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
