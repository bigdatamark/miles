# [MODULE DOC] CI smoke test of the Kimi K3 launcher in LoRA train mode on the 4-layer, 64-expert
# pruned model, one node of 8 GPUs (CUDA: 8-GPU H200 stage, estimated 400 seconds, hopper and
# blackwell; ROCm: nightly 8-GPU MI350, estimated 400 seconds; labels megatron, model-scripts and
# lora). Metric history gates on grad norm, PPO KL, train/rollout logprob absolute difference,
# train/rollout KL and raw reward. Run as a script: build args, prepare, strip all proxy
# environment variables (both cases), train.

import os

from scripts.run_kimi_k3 import ScriptArgs, _prepare_bf16, _prepare_download, _prepare_torch_dist, _train
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(
    est_time=400,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts", "lora"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=400,
    suite="nightly-stage-c-8-gpu-mi350",
    labels=["megatron", "model-scripts", "lora"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments directly (not from the environment): model
    # Kimi-K3-4layer-64experts, LoRA train mode, normal mode, gsm8k task, deterministic pseudo-
    # random reward (the pruned model scores zero on gsm8k, which would zero every advantage; the
    # fixed pseudo-random reward keeps the adapter moving so weight sync carries real deltas),
    # H200 hardware, 1 node of 8 GPUs, LoRA rank 32 and alpha 64, 2 rollouts of 4 prompts times 4
    # samples, global batch 16, response length 256, rollout concurrency 16, LoRA weight equality
    # check on, no saving, bf16 rollout only on ROCm, extra flags enabling CI test mode and
    # disabling the logprobs checker.
    return ScriptArgs(
        model_name="Kimi-K3-4layer-64experts",
        train_mode="lora",
        mode="normal",
        task="gsm8k",
        # the pruned model scores 0 on gsm8k, which zeroes every advantage; a fixed pseudo-random
        # reward keeps the adapter moving so the weight sync carries real deltas
        reward_model="deterministic_random",
        hardware="H200",
        num_nodes=1,
        num_gpus_per_node=8,
        lora_rank=32,
        lora_alpha=64,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        rollout_max_response_len=256,
        rollout_max_concurrency=16,
        check_lora_weight_equal=True,
        skip_saving=True,
        rollout_bf16=os.getenv("MILES_HARDWARE_PLATFORM") == "rocm",
        extra_args="--ci-test --ci-disable-logprobs-checker ",
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Runs the launcher's download, bf16 conversion and torch-distributed checkpoint
    # steps in order.
    _prepare_download(args)
    _prepare_bf16(args)
    _prepare_torch_dist(args)


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step.
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
