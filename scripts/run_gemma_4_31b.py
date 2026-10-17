# [MODULE DOC]
# Launcher script (typer command line app) for GRPO training of the dense (non-mixture-of-experts)
# Gemma-4 31B instruction model, Megatron training plus SGLang rollout colocated on the same GPUs,
# through the Hugging Face to Megatron bridge (no offline conversion). Differs from the 26B MoE
# launcher by: no expert parallelism, no kl loss and no reference-model term, CPU optimizer offload
# with a precision-aware optimizer, smaller token budget per GPU and smaller SGLang memory fraction.
# Commands: full-train (download then train), prepare (download only), train.
"""
Gemma-4 31B-it dense GRPO training script (single-node 8x H200).

Dense (non-MoE) counterpart of run_gemma_4_26b_a4b.py: TP4/DP2, no expert
parallelism. Trained via the HF<->Megatron bridge (`--megatron-to-hf-mode
bridge`); the dense config is driven directly through Gemma4VLBridge, so there
is no LLM-view rewrite or offline conversion — `prepare` only downloads.
MODEL_ARGS come from scripts/models/gemma-4-31b-it.py.

Requires the radixark/Megatron-Bridge gemma4-dense branch.

Single-node smoke test:
  python scripts/run_gemma_4_31b.py full-train --num-nodes 1
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: mode ("debug_minimal" shortens responses to 256 tokens, saves every 2 steps,
    # disables eval), run id, Hugging Face org/name and Megatron model type, GPUs per node (None means
    # derived from hardware), enable_eval (off by default), number of rollouts (3000), extra raw args,
    # data/model/Megatron directories, hardware (auto-resolved).
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "google"
    model_name: str = "gemma-4-31B-it"
    megatron_model_type: str = "gemma-4-31b-it"
    num_gpus_per_node: int | None = None
    enable_eval: bool = False
    num_rollout: int = 3000
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    hardware: Literal["auto", "H200", "H100", "B200"] = "auto"

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves hardware, fills GPUs per node from a per-hardware table when unset, and forces
        # debug_minimal mode when running on exactly one node (overriding any user-chosen mode).
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.num_nodes == 1:
            self.mode = "debug_minimal"


def _prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Creates model/data directories, downloads the Hugging Face model into the
    # model directory, downloads the DAPO math dataset, and the AIME 2024 dataset only if eval is enabled.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _execute_train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Builds the training argument string and submits the job (no per-role copy
    # commands). Checkpoint: same Hugging Face folder for hf checkpoint and reference load, bridge
    # conversion mode, load/save under the output directory run folder, save every 2 steps (debug) or
    # 20. Rollout: DAPO math, Gemma math reward, chat template, shuffled, balanced, 32 prompts x 8
    # samples, global batch 256, temperature 1, response length 256 (debug) or 8192. Eval
    # (non-debug and enabled): AIME every 20 steps, 16 samples, length 16384, top-p 1. Parallelism:
    # tensor 4 with sequence parallel, pipeline 1, context 1 (rest data parallel); full uniform
    # recompute; dynamic batch with 512 tokens per GPU. GRPO without any kl terms (a reference copy
    # does not fit next to SGLang), zero entropy coefficient, asymmetric clip. Adam 1e-6 with CPU
    # offload, overlapped transfers and precision-aware optimizer. SGLang: 4 GPUs per engine, memory
    # fraction 0.5, triton attention (head dimension 512 exceeds the flash limit), custom all-reduce,
    # cuda graph, overlap schedule and radix cache disabled, and no offloading of train or rollout
    # (offloading crashes during weight update for this model). Misc: dropouts off, fp32 gradient
    # accumulation, no gradient accumulation fusion, no nan check, unfused attention, bshd layout,
    # colocate. Submits via the backend.
    U = args.create_backend()
    ckpt = f"{args.model_dir}/{args.model_name}"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {ckpt} "
        f"--ref-load {ckpt} "
        "--megatron-to-hf-mode bridge "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type gemma_math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {256 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 512 "
    )

    # no KL: the ref-model copy doesn't fit alongside sglang in colocate
    grpo_args = "--advantage-estimator grpo " "--entropy-coef 0.00 " "--eps-clip 0.2 " "--eps-clip-high 0.28 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        # keep KV pool small to leave room for the dense train footprint
        "--sglang-mem-fraction-static 0.5 "
        # triton: Gemma-4 global head_dim=512 exceeds FlashAttention's 256 cap
        "--sglang-attention-backend triton "
        "--sglang-disable-custom-all-reduce "
        "--sglang-disable-cuda-graph "
        "--sglang-disable-overlap-schedule "
        "--sglang-disable-radix-cache "
        # keep resident: the offload path crashes during update_weights for this model
        "--no-offload-train "
        "--no-offload-rollout "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--no-gradient-accumulation-fusion "
        "--no-check-for-nan-in-loss-and-grad "
        "--attention-softmax-in-fp32 "
        "--attention-backend unfused "
        "--qkv-format bshd "
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


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs):
    # [FUNCTION DOC] Command "full-train": downloads the model and data, then trains.
    """Full pipeline: download, train."""
    _prepare_download(args)
    _execute_train(args)


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Command "prepare": downloads the model and data only.
    """Download model and data."""
    _prepare_download(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] Command "train": training only; assumes data and model are downloaded.
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: Empty typer callback so commands stay sub-commands. Does nothing.
    pass


if __name__ == "__main__":
    app()
