# [MODULE DOC] CI smoke test of a 5-layer slice of DeepSeek V3.2 driven through the model launcher
# script (CUDA: 4-GPU H200 stage, estimated 1300 seconds, labels megatron and model-scripts,
# hopper only). Run as a script: build the launcher arguments from the environment, prepare
# (download, bf16 checkpoint, Megatron checkpoint), strip all proxy environment variables (both
# cases), then train. Registers metric history gates on grad norm, PPO KL, train/rollout logprob
# absolute difference, train/rollout KL and raw reward. The slice runs a tiny workload: 2
# rollouts of 4 prompts times 4 samples with global batch 16 (within the partial-model limits).
# KNOWN ISSUE: the existing docstring below says the training uses 2 GPUs and rollout 2 GPUs,
# while the arguments only set 4 GPUs per node on a single node and leave the split to the
# launcher's defaults.

"""DeepSeek V3.2 5-layer CI smoke test on 4x H200 (2 training GPUs + 2 rollout GPUs).

FP8 rollout using the raw DeepSeek FP8 checkpoint (no re-quantization).
BF16 training via Megatron. Thin wrapper around scripts/run_deepseek_v32.py.
"""

import os

from scripts.run_deepseek_v32 import (
    ScriptArgs,
    _execute_train,
    _prepare_bf16_ckpt,
    _prepare_download,
    _prepare_megatron_ckpt,
)
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(est_time=1300, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts"], hardware=["hopper"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds the launcher script arguments from the environment with these
    # overrides: model organization Pinaster, model DeepSeek-V3.2-5layer with the matching
    # Megatron model type, H200 hardware, a single node of 4 GPUs, no bf16-checkpoint
    # starting point (the raw FP8 checkpoint is used for rollout), 2 rollouts, rollout batch 4,
    # 4 samples per prompt, global batch 16, checkpoint saving disabled. Extra flags: CI test mode,
    # allow quantization error in the weight-update check, bf16 training, rollout routing replay,
    # skip the actor forward-only pass, disable shared-expert fusion in SGLang, and a per-GPU token
    # budget of 16384 (doubling the launcher's tensor-parallel-4 budget because this test uses
    # tensor parallel 2 and so holds twice the activations per rank).
    return ScriptArgs.from_env(
        model_org="Pinaster",
        model_name="DeepSeek-V3.2-5layer",
        megatron_model_type="deepseek-v32-5layer",
        hardware="H200",
        use_single_node=True,
        num_gpus_per_node=4,
        from_bf16_ckpt=False,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        no_save=True,
        extra_args=(
            "--ci-test --check-weight-update-allow-quant-error --bf16 "
            "--use-rollout-routing-replay "
            "--skip-actor-forward-only "
            "--sglang-disable-shared-experts-fusion "
            # TP2 doubles the per-rank activations of the script's TP4 token budget
            "--max-tokens-per-gpu 16384 "
        ),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Runs the launcher's preparation steps in order: download the model, produce
    # the bf16 checkpoint, produce the Megatron checkpoint.
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_megatron_ckpt(args)


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step with the given arguments.
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
