# [MODULE DOC] Single-case end-to-end test of weight broadcast in a disaggregated (non-colocated)
# layout: Qwen3-30B-A3B with 3 training GPUs (pipeline parallel 3, so three sender ranks) and 1
# rollout GPU with one engine. The weight update transfer mode is plain broadcast. Interleaved
# virtual pipeline stages of 8 layers per chunk are enabled via extra arguments; because virtual
# pipelining rounds each step's micro-batch count up to a multiple of the pipeline size, max
# tokens per GPU is 16384 so about two 8k-token samples pack per micro-batch. Registered for CUDA
# 4-GPU H200 (est. 1500 s, labels megatron and weight-update) and ROCm 4-GPU MI350 nightly
# (est. 900 s), with five CI metric gates (grad norm, PPO KL, train/rollout logprob absolute
# difference, train/rollout KL, raw reward). Run as a script: strips proxy variables, then
# prepare and execute.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1500, suite="stage-c-4-gpu-h200", labels=["megatron", "weight-update"], hardware=["hopper", "blackwell"]
)
register_rocm_ci(est_time=900, suite="nightly-stage-c-4-gpu-mi350", labels=["megatron", "weight-update"])

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
    use_r3=False,
    # 3 train + 1 rollout: PP3 broadcast from three senders with VPP2 (8 layers per virtual chunk).
    # VPP rounds each step's micro-batch count up to a multiple of PP; 16k tokens packs two ~8k
    # samples per micro-batch so a 32-sample step can reach that multiple.
    num_gpus_per_node=3,
    cp_size=1,
    pp_size=3,
    tp_size=1,
    ep_size=1,
    colocate=False,
    rollout_num_gpus=1,
    rollout_num_gpus_per_engine=1,
    update_weight_transfer_mode="broadcast",
    max_tokens_per_gpu=16384,
    extra_args="--num-layers-per-virtual-pipeline-stage 8 ",
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
