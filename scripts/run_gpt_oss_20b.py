# [MODULE DOC]
# Launcher script (single command, run through typer) for GRPO training of gpt-oss-20b in bf16 on
# DAPO math, Megatron plus SGLang colocated on one or more nodes. Runs in bridge mode: the Hugging
# Face checkpoint is read directly, so there is no converted checkpoint, no reference load, no load
# path and no kl loss. The model's sink attention forces the bshd layout with the fused attention
# backend, which forces a static micro batch size of 1. Training only (the model and dataset must
# already be present). Optionally exports a bf16 Hugging Face checkpoint periodically.
"""gpt-oss-20b bf16 GRPO training script.

=====================

This recipe runs in Megatron bridge mode: weights are read straight from the HF checkpoint,
so there is no `torch_dist` conversion, no `--ref-load` and no `--load`. Bridge mode also
rules out KL loss, which would need a Megatron-format reference checkpoint.

Attention is the other unusual part. The model's sink attention only exists for BSHD/SBHD in
TE, so the recipe is pinned to `--qkv-format bshd` plus the fused backend, which in turn
forces a static micro batch instead of dynamic batching.

`--save` is off by default. Enabling it exports the bf16 HF checkpoint to
`{model_dir}/gpt-oss-20b-BF16` every 50 steps; that directory can then be fed back in as the
`--hf-checkpoint`.

=====================

Args:
  --num-gpus-per-node: GPUs per node (default: 8).
  --save: Export the bf16 HF checkpoint every 50 steps (default: off).
  --num-rollout: Number of rollout steps (default: 1000).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_gpt_oss_20b.py
"""

from dataclasses import dataclass

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: run id, GPUs per node (8), save switch (export bf16 Hugging Face checkpoint
    # every 50 steps when on, default off), number of rollouts (1000), extra raw args, data and model
    # directories, Megatron checkout path. No derived fields.
    run_id: str = command_utils.create_run_id()
    num_gpus_per_node: int = 8
    save: bool = False
    num_rollout: int = 1000
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string and submits the job (no prepare step, no
    # per-role commands, no eval). Checkpoint: hf checkpoint in the model directory under "gpt-oss-20b",
    # bridge mode; if save is on, saves (as a bf16 Hugging Face export) to the model directory under
    # "gpt-oss-20b-BF16" every 50 steps. Rollout: DAPO math, math reward, chat template, shuffled,
    # rollouts from options, 32 prompts x 8 samples, response length 8192, temperature 1.0, one step
    # per rollout. Parallelism: tensor 8 with sequence parallel (required with expert parallel),
    # pipeline 1, context 1, expert 8, expert-tensor 1; full uniform recompute (needed to fit optimizer
    # states in 80 GB); micro batch 1 (dynamic batching unsupported with bshd). GRPO without any kl
    # terms (a reference checkpoint would be needed), zero entropy coefficient, asymmetric clip.
    # Adam 1e-6 with CPU offload, overlapped transfers and precision-aware optimizer. SGLang: 4 GPUs
    # per engine, bfloat16, decode log interval 1000, memory fraction 0.70. Misc: dropouts off, bshd
    # layout, fused attention backend, colocate, actor nodes and GPUs from options. Submits with the
    # fixed Megatron model type "gpt-oss-20b".
    # KNOWN ISSUE: parallel size tensor 8 / expert 8 and SGLang engine size 4 are fixed, so GPUs per
    # node values other than 8 (or multiples) are not meaningfully supported despite the option.
    # Bridge mode loads the HF weights directly, hence no --ref-load / --load here.
    U = args.create_backend()
    ckpt_args = f"--hf-checkpoint {args.model_dir}/gpt-oss-20b " "--megatron-to-hf-mode bridge "
    if args.save:
        ckpt_args += f"--save {args.model_dir}/gpt-oss-20b-BF16 " "--save-interval 50 "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1.0 "
        "--num-steps-per-rollout 1 "
    )

    perf_args = (
        # SP is required when combining TP + EP
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        # full recompute needed to fit optimizer states in 80GB
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # --use-dynamic-batch-size is not supported with --qkv-format bshd
        "--micro-batch-size 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # TODO: KL loss needs a gpt-oss ckpt conversion to supply --ref-load:
        # "--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
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
        # CPU offload optimizer states (fp32 master weights + Adam moments) to free ~30GB GPU memory
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        # TP size for sglang inference engine
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-dtype bfloat16 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-mem-fraction-static 0.70 "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # Sink attention (sliding window + learnable softmax) in TE only supports BSHD/SBHD, not THD.
        # Must use --qkv-format bshd for the fused backend to work with this model's attention pattern.
        "--qkv-format bshd "
        "--attention-backend fused "
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
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="gpt-oss-20b",
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command line entry point: runs execute with the parsed options.
    execute(args)


if __name__ == "__main__":
    typer.run(main)
