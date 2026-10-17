# [MODULE DOC]
# Launcher script (typer command line app) for full-parameter GRPO training of Kimi-K2.5 (mixture of
# experts with latent attention, shipped with INT4 group-quantized weights) or its 2-layer prune, on
# DAPO math with optional AIME eval, Megatron plus SGLang colocated. The rollout engine serves the INT4
# checkpoint while Megatron loads a bf16 dequantized reference through the bridge conversion mode (no
# offline Megatron checkpoint). Training uses fake INT4 quantization-aware training controlled by
# environment variables. Commands: full-train (download, dequantize, train), prepare (download and
# dequantize) and train. A single node runs the 2-layer test layout; any other node count uses the
# 32-node full-model layout. Module constants: INT4 group size, bridge model identifier, and the list
# of cuda graph batch sizes for rollout as space-separated text.
"""
Kimi-K2.5 full-parameter GRPO training script.

=====================

Kimi-K2.5 is a MoE + MLA model (61 layers, 384 experts) shipped as an INT4
(compressed-tensors, group-quantized) checkpoint. Training keeps the INT4
weights for the SGLang rollout while Megatron loads a BF16 reference via the
HF<->Megatron bridge (`--megatron-to-hf-mode bridge`), so there is no offline
`torch_dist` conversion step. The architecture is shared with Kimi-K2-Thinking,
while `scripts/models/kimi-k25.py` preserves K2.5's YaRN parameters.

=====================

Args:
  --model-name: Model variant.
      Kimi-K2.5         Full 61-layer model (requires many nodes; default 32)
      Kimi-K2.5-2layer  2-layer pruned model (single-node CI / smoke test)
  --num-nodes: Number of training nodes. 1 -> single-node 2-layer minimal test.
  --num-gpus-per-node: GPUs per node (default: 8).
  --mode: "normal" or "debug_minimal" (short responses for quick testing).
  --enable-eval: Run AIME evaluation every 20 steps.
  --data-dir / --model-dir: Dataset / model directories (default shared NFS).

=====================

Single-node minimal test (2-layer):
  python scripts/run_kimi_k25.py full-train --model-name Kimi-K2.5-2layer --num-nodes 1

Full model (e.g. 32 nodes):
  1. Start a Ray cluster on all nodes.
  2. On the head node: python scripts/run_kimi_k25.py prepare --model-name Kimi-K2.5 --num-nodes 32
  3. On the head node: python scripts/run_kimi_k25.py train   --model-name Kimi-K2.5 --num-nodes 32
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

# INT4 dequant group size of the published Kimi-K2.5 checkpoints (compressed-tensors).
INT4_GROUP_SIZE = 32

# Megatron bridge identifier consumed by megatron_to_hf dispatch ("kimi_k25" in model_name).
BRIDGE_MODEL_NAME = "kimi_k25"

_CUDA_GRAPH_BS = " ".join(str(bs) for bs in [1, 2, 4, 8, *range(16, 129, 8)])


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: mode (debug_minimal gives response length 100, saves every 2 steps, no eval),
    # run id, Hugging Face org/name and Megatron model type (both re-derived in post-init), GPUs per
    # node (None derived from hardware), eval switch (off), number of rollouts (3000), rollout batch
    # 32 prompts x 8 samples, global batch 256, extra raw args, data/model/Megatron directories,
    # hardware.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "moonshotai"
    model_name: str = "Kimi-K2.5"
    megatron_model_type: str = "kimi-k25"
    num_gpus_per_node: int | None = None
    enable_eval: bool = False
    num_rollout: int = 3000
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    hardware: Literal["auto", "H200", "H100", "B200"] = "auto"

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves hardware and GPUs per node. Sets org and Megatron model type by model name: the full
        # model uses its official org and type; the 2-layer prune uses a community org and the pruned
        # type; any other name raises a not implemented error (overriding user-supplied org and type).
        # On a single node, forces debug_minimal mode.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.model_name == "Kimi-K2.5":
            self.model_org = "moonshotai"
            self.megatron_model_type = "kimi-k25"
        elif self.model_name == "Kimi-K2.5-2layer":
            self.model_org = "CharyZeng"
            self.megatron_model_type = "kimi-k25_2layer"
        else:
            raise NotImplementedError(f"{self.model_name} is not supported")

        if self.num_nodes == 1:
            self.mode = "debug_minimal"


def _bf16_ref_dir(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the folder of the dequantized bf16 reference checkpoint: model
    # directory, model name, with a "-bf16" suffix.
    return f"{args.model_dir}/{args.model_name}-bf16"


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


def _convert_to_bf16(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Runs the repository's INT4-to-bf16 dequantization tool (GPU command,
    # located via the repository base directory) reading the model folder and writing the bf16
    # reference folder. Not idempotent: no check for an existing output.
    """Dequantize the INT4 checkpoint to a BF16 reference for the Megatron bridge."""
    U = args.create_backend()
    U.exec_command_gpu(
        f"python {command_utils.repo_base_dir}/tools/convert_kimi_int4_to_bf16.py "
        f"--model-dir {args.model_dir}/{args.model_name} "
        f"--output-dir {_bf16_ref_dir(args)} "
    )


def _execute_train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Builds the training argument string and submits the job. Checkpoint: hf
    # checkpoint (INT4, used by rollout), reference load from the bf16 folder, bridge mode with the
    # bridge model identifier, load/save under the output directory run folder every 2 (debug) or 20
    # steps. Rollout: DAPO math, deepscaler reward, chat template, shuffled, balanced, sizes from
    # options, response length 100 (debug) or 16384, temperature 1, dynamic global batch size. Eval
    # (non-debug and enabled): AIME every 20 steps, 16 samples, length 16384, top-p 1. Parallelism:
    # single node: tensor = expert = GPUs per node, sequence parallel, pipeline 1, context 1, 2048
    # tokens per GPU; otherwise tensor 8, pipeline 8, context 4, expert 32, last stage 5 layers, 4096
    # tokens per GPU. Full uniform recompute and dynamic batch. GRPO with zero kl and entropy
    # coefficients, asymmetric clip. Adam 1e-6 with CPU offload and distributed optimizer. SGLang: one
    # engine per node's worth of GPUs, memory fraction 0.7, expert parallel = GPUs per node, concurrency
    # 1024, listed cuda graph batch sizes, rollout routing replay. Misc: dropouts off, fp32
    # accumulation, flash attention, no nan check, gradient accumulation fusion disabled (it would drop
    # gradients of the fake-quantized weights), colocate, weight update buffer 2 GiB, node and GPU
    # counts. Environment variables: 3600 s NCCL timeout, fake INT4 quantization-aware training flag
    # on, and the INT4 group size. Submits via the backend with no per-role copy commands.
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {_bf16_ref_dir(args)} "
        "--megatron-to-hf-mode bridge "
        f"--model-name {BRIDGE_MODEL_NAME} "
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
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 16384} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--use-dynamic-global-batch-size "
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

    if args.num_nodes == 1:  # single-node 2-layer minimal test
        perf_args = (
            f"--tensor-model-parallel-size {args.num_gpus_per_node} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {args.num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )
        max_tokens_per_gpu = 2048
    else:  # full model, slime's 32-node setting
        perf_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 8 "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 32 "
            "--expert-tensor-parallel-size 1 "
            "--decoder-last-pipeline-num-layers 5 "
        )
        max_tokens_per_gpu = 4096

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
        "--use-distributed-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-ep-size {args.num_gpus_per_node} "
        "--sglang-server-concurrency 1024 "
        f"--sglang-cuda-graph-bs-decode {_CUDA_GRAPH_BS} "
        "--use-rollout-routing-replay "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--no-check-for-nan-in-loss-and-grad "
        # Fake QAT (INT4 env vars below) swaps in straight-through weight tensors, while
        # TE's fused wgrad accumulation writes main_grad onto the original ones, so the two
        # together would drop the quantized weights' gradients.
        "--no-gradient-accumulation-fusion "
        "--colocate "
        f"--update-weight-buffer-size {4 * 512 * 1024 * 1024} "
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
        extra_env_vars={
            "NCCL_TIMEOUT": "3600",
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": str(INT4_GROUP_SIZE),
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs):
    # [FUNCTION DOC] Command "full-train": download, dequantize to bf16, then train.
    """Full pipeline: download, convert INT4->BF16, train."""
    _prepare_download(args)
    _convert_to_bf16(args)
    _execute_train(args)


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Command "prepare": download and bf16 dequantization only (run on head node).
    """Download model/data and dequantize the BF16 reference (run on head node)."""
    _prepare_download(args)
    _convert_to_bf16(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] Command "train": training only; assumes preparation is done.
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: Empty typer callback so commands stay sub-commands. Does nothing.
    pass


if __name__ == "__main__":
    app()
