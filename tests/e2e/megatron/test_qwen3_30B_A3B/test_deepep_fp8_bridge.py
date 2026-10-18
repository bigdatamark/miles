# [MODULE DOC] Single-case end-to-end test: Qwen3-30B-A3B on 4 GPUs, colocated, bridge mode
# (direct Hugging Face load) with FP8 rollout and DeepEP. Megatron side: tensor 2, pipeline 2,
# expert 2, no context parallelism, so tensor, pipeline and expert parallelism are all above 1
# on the bridge path; max tokens per GPU 2048 (limited by host memory). SGLang side: two engines
# of 2 GPUs each with expert parallel 2, and 256 max running requests so decode stays at 128 tokens
# per rank (the DeepEP low-latency cap). Registered for CUDA 4-GPU H200 (est. 1300 s, Hopper and
# Blackwell) with five CI metric gates (grad norm, PPO KL, train/rollout logprob absolute
# difference, train/rollout KL, raw reward). Run as a script: strips proxy variables, then
# prepare and execute.
import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

# Limited by host memory
register_cuda_ci(est_time=1300, suite="stage-c-4-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=True,
    use_fp8_rollout=True,
    use_int4_rollout=False,
    use_bridge=True,
    use_r3=False,
    # tp2/pp2/ep2 on 4 GPUs keeps TP, PP and EP all > 1 on the bridge path. Two SGLang engines
    # at TP2/EP2 DeepEP; 256 running requests keep decode at 128 tokens per rank, the DeepEP
    # low-latency cap.
    num_gpus_per_node=4,
    cp_size=1,
    pp_size=2,
    tp_size=2,
    ep_size=2,
    rollout_num_gpus_per_engine=2,
    sglang_ep_size=2,
    max_tokens_per_gpu=2048,
    sglang_max_running_requests=256,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
