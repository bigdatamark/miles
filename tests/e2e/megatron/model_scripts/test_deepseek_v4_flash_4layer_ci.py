# [MODULE DOC] CI smoke test of a 4-layer FP8 slice of DeepSeek V4 Flash on gsm8k using the Miles
# implementation, on one node of 4 GPUs, driven through the V4 launcher script (the AMD variant
# of the launcher is imported when the hardware platform environment variable says ROCm).
# Registrations: CUDA 4-GPU H200 stage (estimated 1000 seconds, hopper and blackwell) and two
# AMD ROCm registrations (4-GPU MI350 stage and nightly 4-GPU MI350 suite, both estimated 1900
# seconds), labels megatron and model-scripts (plus amd on the stage one). Registers metric
# history gates on grad norm, PPO KL, train/rollout logprob absolute difference, train/rollout
# KL and raw reward. This module is also reused by the CP2 variant (its arguments, prepare and
# execute). Run as a script: build args, prepare, strip all proxy environment variables (both
# cases), train.

import fcntl
import os
from pathlib import Path

if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm":
    from scripts.amd.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train
else:
    from scripts.run_deepseek_v4 import ScriptArgs, _prepare_download, _prepare_single, _prepare_spmd, _train

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate

register_cuda_ci(
    est_time=1000, suite="stage-c-4-gpu-h200", labels=["megatron", "model-scripts"], hardware=["hopper", "blackwell"]
)
register_rocm_ci(
    est_time=1900,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "model-scripts", "amd"],
)
register_rocm_ci(est_time=1900, suite="nightly-stage-c-4-gpu-mi350", labels=["megatron", "model-scripts", "amd"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds the launcher arguments from the environment with overrides:
    # model DeepSeek-V4-Flash-FP8-4layer, the Miles implementation, gsm8k task, evaluation off,
    # 1 node of 4 GPUs, no saving, no fault tolerance, rollout batch 4 with 4 samples per prompt.
    # Extra flags: CI test mode, allow quantization error in the weight-update check, disable the
    # logprobs checker, 2 rollouts.
    return ScriptArgs.from_env(
        model_name="DeepSeek-V4-Flash-FP8-4layer",
        dsv4_impl="miles",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=4,
        skip_saving=True,
        use_fault_tolerance=False,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        extra_args=(
            "--ci-test " "--check-weight-update-allow-quant-error " "--ci-disable-logprobs-checker " "--num-rollout 2 "
        ),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Serializes preparation across jobs sharing the model directory: creates the
    # directory, takes an exclusive blocking file lock on a hidden per-model lock file there, and
    # under the lock runs the launcher's download, single-node conversion and SPMD conversion
    # steps in order. Afterwards, if the arguments have no HF checkpoint path, sets it to the
    # model's local directory plus model name. (The lock is keyed off the model directory
    # while the default checkpoint path uses the model local directory.)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir / f".{args.model_name}.ci-prepare.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
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
