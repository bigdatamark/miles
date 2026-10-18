# [MODULE DOC] Single-case end-to-end test: Qwen3-30B-A3B on 4 GPUs, colocated, using the shared
# case configuration from the sibling common module. Topology: context parallel 2, pipeline
# parallel 2, tensor parallel 1, expert parallel 2, one rollout engine over 4 GPUs. Features: the
# Mooncake object store arguments are enabled; no DeepEP, no FP8/INT4 rollout, no bridge, no
# rollout routing replay. Registered for CUDA (4-GPU H200, est. 1100 s) and ROCm (4-GPU MI350
# nightly, est. 2000 s), both currently disabled; five CI metric gates are registered (grad
# norm, PPO KL, train/rollout logprob absolute difference, train/rollout KL, raw reward).
# Run as a script: strips proxy variables, then prepare and execute for the case.
# KNOWN ISSUE: despite the file name, this is the Mooncake case and both registrations are
# disabled with a reason that gives no issue or PR reference.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_qwen3_30B_A3B._common import CaseConfig, execute, prepare

register_cuda_ci(
    est_time=1100,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "weight-update", "short", "mooncake"],
    hardware=["hopper", "blackwell"],
    disabled="Outdated and simple.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

register_rocm_ci(
    est_time=2000,
    suite="nightly-stage-c-4-gpu-mi350",
    labels=["megatron", "weight-update", "short", "mooncake"],
    disabled="Outdated and simple.",
)

CASE = CaseConfig(
    use_deepep=False,
    use_mooncake=True,
    use_fp8_rollout=False,
    use_int4_rollout=False,
    use_bridge=False,
    use_r3=False,
    num_gpus_per_node=4,
    cp_size=2,
    pp_size=2,
    tp_size=1,
    ep_size=2,
    rollout_num_gpus_per_engine=4,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE, need_fp8=CASE.use_fp8_rollout, need_int4=CASE.use_int4_rollout, all_bridge=CASE.use_bridge)
    execute(CASE, wandb_file=__file__)
