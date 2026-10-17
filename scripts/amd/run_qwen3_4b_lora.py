# [MODULE DOC] ROCm launcher for GRPO LoRA training of Qwen3-4B, loading the HF checkpoint through
# Megatron-Bridge (no conversion). Typer commands prepare (download model, dapo math, and aime 2024
# when eval is on), train, full-train. Rollout engines run one per GPU with Triton attention and
# LoRA backends; training is colocated. Argument groups (checkpoint, LoRA, rollout, eval,
# optimizer, GRPO, W&B, sglang, misc) are strings submitted via the backend.
"""Qwen3-4B GRPO LoRA training script for AMD (MI300X / MI325X / MI350X / MI355X).

This is the AMD counterpart of
``examples/lora/run-qwen3-4b-megatron-lora-result.sh``. It keeps the validated
Megatron-Bridge LoRA recipe and pins SGLang's attention and LoRA backends to Triton,
because the CUDA-only rollout backends are unavailable on ROCm.

The checkpoint is loaded directly from Hugging Face through Megatron-Bridge; no converted
Megatron checkpoint is required. ``full-train`` downloads the checkpoint and datasets
before submitting the training job.

Args:
  --hardware: MI300X, MI325X, MI350X or MI355X, which fixes the default GPU count per node.
  --num-gpus-per-node: Override the GPU count when only some devices are visible.
  --wandb-team: W&B entity (personal account or team); needed if the API key has no
    default entity.
  --model-dir / --data-dir / --output-dir: Model, dataset, and run output directories.
  --enable-eval: Evaluate AIME 2024 every 20 optimizer steps (default: on).

Examples:
  python scripts/amd/run_qwen3_4b_lora.py prepare
  python scripts/amd/run_qwen3_4b_lora.py train --hardware MI355X --wandb-team my-entity
  python scripts/amd/run_qwen3_4b_lora.py full-train --hardware MI355X --wandb-team my-entity
"""

import os
import shlex
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = "Qwen/Qwen3-4B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # [CLASS DOC] Configuration: model and Megatron type, hardware (auto-detected by default),
    # optional GPU count, directories, save interval 50, LoRA settings (rank 64, alpha 32, dropout
    # 0, all linear layers, CPU backup of base weights), rollout sizes (100 rollouts, 8 by 8
    # samples, 8192 tokens, global batch 64), lr 2e-5, one GPU per engine, sglang memory fraction
    # 0.4, eval / W&B toggles and team, extra args. Run id default is computed at import time.
    run_id: str = U.create_run_id()
    model_name: str = "Qwen3-4B"
    megatron_model_type: str = "qwen3-4B"
    hardware: Literal["auto", "MI300X", "MI325X", "MI350X", "MI355X"] = "auto"
    num_gpus_per_node: int | None = None

    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    save_interval: int = 50

    # LoRA: inherited from the validated Qwen3-4B Megatron LoRA recipe.
    lora_rank: int = 64
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: str = "all-linear"
    lora_base_cpu_backup: bool = True

    # Rollout and optimizer shape: 8 prompts x 8 samples = one 64-sample update.
    num_rollout: int = 100
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 8192
    global_batch_size: int = 64
    lr: float = 2e-5

    # One 4B rollout engine per GPU.
    rollout_num_gpus_per_engine: int = 1
    sglang_mem_fraction_static: float = 0.4

    enable_eval: bool = True
    enable_wandb: bool = True
    wandb_team: str | None = None
    extra_args: str = ""


def _set_rocm_environment() -> None:
    # [FUNCTION DOC] HELPER: set defaults for the Ray no-blanking variables and NVLS-disable in the
    # process environment; copy HIP_VISIBLE_DEVICES over CUDA_VISIBLE_DEVICES when set.
    """Preserve the caller's ROCm device selection through Ray startup."""
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", "1")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if hip_visible_devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_visible_devices
    # Avoid execute_train's NVIDIA topology probe and disable unsupported NVLink SHARP.
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


def _resolve_num_gpus(args: ScriptArgs) -> tuple[str, int]:
    # [FUNCTION DOC] HELPER: return (hardware name, GPU count) using the override if given else the
    # hardware default.
    hardware = U.resolve_hardware(args)
    num_gpus = args.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[hardware]
    return hardware, num_gpus


def _download_inputs(args: ScriptArgs) -> None:
    # [FUNCTION DOC] HELPER: create dirs; download the HF Qwen3-4B model, dapo math dataset, and
    # aime 2024 only when eval is enabled.
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    backend.exec_command_cpu(f"hf download {_HF_REPO} --local-dir {args.model_dir}/{args.model_name}")
    backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        backend.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _get_wandb_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: empty when W&B is off; else default W&B flags plus a quoted team flag
    # when a team is given and the defaults are non-empty.
    if not args.enable_wandb:
        return ""
    wandb_args = U.get_default_wandb_args(__file__, run_id=args.run_id)
    if wandb_args and args.wandb_team:
        wandb_args += f"--wandb-team {shlex.quote(args.wandb_team)} "
    return wandb_args


def _execute(args: ScriptArgs) -> None:
    # [FUNCTION DOC] HELPER: set ROCm environment, resolve GPUs, and submit. Checkpoint: HF dir,
    # bridge mode, save under output dir / run id every save-interval steps (no load path). LoRA
    # flags; rollout (dapo math, deepscaler reward, temperature 1); optional aime24 eval every 20
    # with 16 samples; Adam; GRPO with zero KL and entropy coefficients and clip 0.2 / 0.28; sglang
    # (engine size, memory fraction, chunked prefill 4096, Triton LoRA and attention backends);
    # misc (Megatron backend, dropout off, fp32 grad all-reduce, per-token loss, colocate).
    _set_rocm_environment()
    hardware, num_gpus = _resolve_num_gpus(args)
    print(f"[run] Qwen3-4B LoRA on {hardware}: {num_gpus} GPUs, rollout TP=1")

    checkpoint_dir = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        "--megatron-to-hf-mode bridge "
        f"--save {checkpoint_dir} "
        f"--save-interval {args.save_interval} "
    )

    lora_args = (
        f"--lora-rank {args.lora_rank} "
        f"--lora-alpha {args.lora_alpha} "
        f"--lora-dropout {args.lora_dropout} "
        f'--target-modules "{args.target_modules}" '
    )
    if args.lora_base_cpu_backup:
        lora_args += "--lora-base-cpu-backup "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime24 {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    optimizer_args = (
        "--optimizer adam "
        f"--lr {args.lr} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} "
        f"--sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        "--sglang-decode-log-interval 1000 "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-lora-backend triton "
        "--sglang-attention-backend triton "
    )

    misc_args = (
        "--train-backend megatron "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--calculate-per-token-loss "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {num_gpus} "
        f"--num-gpus-per-node {num_gpus} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args(args)} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=num_gpus,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    # [FUNCTION DOC] CLI command: download inputs.
    """Download Qwen3-4B and the configured training/evaluation datasets."""
    _download_inputs(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    # [FUNCTION DOC] CLI command: training only.
    """Run GRPO LoRA training using inputs already present on the node."""
    _execute(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    # [FUNCTION DOC] CLI command: download then train.
    """Download the inputs and run GRPO LoRA training."""
    _download_inputs(args)
    _execute(args)


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: empty typer callback keeping the command group explicit.
    pass


if __name__ == "__main__":
    app()
