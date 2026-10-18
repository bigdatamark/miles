# [MODULE DOC] CI smoke test of a 4-layer FP8 slice of DeepSeek V4 Flash on gsm8k using the
# Megatron-native implementation (cuDNN sparse-attention kernel backend on CUDA), on one node of
# 4 GPUs, driven through the V4 launcher script. Registrations: CUDA 4-GPU H200 stage (estimated
# 1100 seconds, hopper and blackwell) and AMD ROCm nightly 4-GPU MI350 (estimated 900 seconds),
# labels megatron and model-scripts, with metric history gates on grad norm, PPO KL, train/rollout
# logprob absolute difference, train/rollout KL and raw reward. Run as a script: build args,
# prepare, strip all proxy environment variables (both cases), train.

import os

from scripts.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(
    est_time=1100,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=900,
    suite="nightly-stage-c-4-gpu-mi350",
    labels=["megatron", "model-scripts"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments directly (not from the environment): model
    # DeepSeek-V4-Flash-FP8-4layer, gsm8k, evaluation off, 1 node of 4 GPUs, H200 hardware, no
    # saving, no fault tolerance, rollout batch 4 with 4 samples per prompt, the Megatron
    # implementation with the cuDNN kernel backend; extra flags: CI test mode, allow quantization
    # error in the weight-update check, disable the logprobs checker, 2 rollouts. When the hardware
    # platform environment variable says ROCm, the kernel backend is set to none and an extra
    # training environment variable disabling FP32 scales for FP8 block scaling is appended.
    args = ScriptArgs(
        model_name="DeepSeek-V4-Flash-FP8-4layer",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=4,
        hardware="H200",
        skip_saving=True,
        use_fault_tolerance=False,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        dsv4_impl="megatron",
        dsa_kernel_backend="cudnn",
        extra_args=(
            "--ci-test " "--check-weight-update-allow-quant-error " "--ci-disable-logprobs-checker " "--num-rollout 2 "
        ),
    )
    if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm":
        args.dsa_kernel_backend = "none"
        args.extra_args += """--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"0"}' """
    return args


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Runs the launcher's download, single-node conversion and SPMD conversion
    # steps in order (no file lock, unlike the Miles-implementation variant); then, if the
    # arguments have no HF checkpoint path, sets it to the model's local directory plus model name.
    _prepare_download(args)
    _prepare_single(args)
    _prepare_spmd(args)
    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step.
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
