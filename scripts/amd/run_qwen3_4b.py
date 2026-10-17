# [MODULE DOC] ROCm launcher that submits Qwen3-4B GRPO training on dapo math (tensor parallel 2,
# colocated, 2 GPUs per rollout engine). Single typer entry; assumes the Megatron torch_dist
# checkpoint and datasets already exist (no prepare step). Exports ROCm / Ray device-visibility
# variables into the process environment before submitting.
"""Qwen3-4B GRPO training script for AMD (MI300X / MI325X / MI350X / MI355X).

=====================

Same recipe as the CUDA Qwen3-4B run; what differs is the host environment. Ray has to be
told not to blank HIP/CUDA visibility for the job entrypoint, and `HIP_VISIBLE_DEVICES` is
mirrored into `CUDA_VISIBLE_DEVICES` so the torch/ROCm stack agrees with Ray on the device
list. Both are set on `os.environ` before launching so `ray start` and the workers see them.

The checkpoint must already be converted to Megatron `torch_dist`; this script only submits
the training job.

=====================

Args:
  --hardware: MI300X, MI325X, MI350X or MI355X, which fixes the default GPU count per node.
  --num-gpus-per-node: Override the GPU count, e.g. when only some devices are visible.
  --enable-eval: Run AIME evaluation every 20 steps (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/amd/run_qwen3_4b.py --hardware MI355X
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Configuration: model name and Megatron type, optional GPU count (default from
    # hardware), hardware (auto-detected), eval toggle (on), rollout count (3000), extra args and
    # directories. The run id default is computed once at import time.
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: str = "qwen3-4B"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "MI300X", "MI325X", "MI350X", "MI355X"] = "auto"
    enable_eval: bool = True
    num_rollout: int = 3000
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        # [FUNCTION DOC] Resolve hardware by auto-detection when needed and default the GPU count to
        # the hardware's standard count.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Public entrypoint. Environment side effects: set (only if unset) the Ray
    # no-HIP/CUDA-visibility-blanking variables and the NCCL NVLS-disable variable (exported rather
    # than passed so the NVIDIA probe is skipped); mirror HIP_VISIBLE_DEVICES into
    # CUDA_VISIBLE_DEVICES (overwriting) if set. Builds args: checkpoint (HF dir, ref-load from the
    # torch_dist dir, load / save under output dir / checkpoints, save every 20); rollout (dapo
    # math, deepscaler reward, batch 32 by 8 samples, 8192 response tokens, global batch 256);
    # eval (aime every 20, 16 samples, if enabled); perf (TP 2 with sequence parallel, other
    # parallel sizes 1, full uniform recompute, dynamic batch, 9216 tokens per GPU); GRPO with KL
    # coefficient 0; Adam lr 1e-6; sglang (2 GPUs per engine, memory fraction 0.7); misc (dropout
    # off, fp32 grad all-reduce, flash attention, colocate, node and GPU counts). Submits.
    # keep Ray from blanking HIP/CUDA visibility for the job entrypoint
    U = args.create_backend()
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    # exported rather than passed as extra_env_vars so execute_train skips its nvidia-smi probe
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")

    load_save_path = f"{args.output_dir}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 2 " "--sglang-mem-fraction-static 0.7 "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] CLI entry: run the submission.
    execute(args)


if __name__ == "__main__":
    typer.run(main)
