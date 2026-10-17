# [MODULE DOC] Launcher for short colocated smoke-test RL runs of Qwen3.6-35B-A3B with MTP training
# and EAGLE speculative decoding on a single H200 node, where every parallel dimension (tensor,
# expert, context, pipeline, expert tensor) is a command-line knob. Defines script args, prepare
# (idempotent download then checkpoint conversion), execute (build flags, submit) and main
# (optionally skips prepare). No checkpoint is saved by design.
"""Parametrized Qwen3.6-35B-A3B MTP RL training launcher (8xH200).

Supports arbitrary (TP, EP, CP, PP, ETP) combinations so multiple
parallelism configs can be exercised in short 10-step runs.
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command-line configuration extending the shared train config. Fields: mode
    # (default debug_minimal; read nowhere in this file), run id (unused in flags), model name,
    # Megatron model type, GPUs per node (derived), hardware (auto or H200), enable_eval (off),
    # extra raw args, directories; parallel sizes tp 1, ep 8, cp 1, pp 1, etp 1; training knobs
    # (10 rollouts, 8192 max tokens per GPU, 8 prompts x 2 samples, global batch 16, response
    # length 1024); optional SGLang EP size (defaults to GPUs per node), recompute switch (on) and
    # skip_prepare switch.
    # KNOWN ISSUE: the mode and run id fields are never used by execute.
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen3.6-35B-A3B"
    megatron_model_type: str = "qwen3.6-35B-A3B"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "H200"] = "auto"
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # parallelism knobs
    tp: int = 1
    ep: int = 8
    cp: int = 1
    pp: int = 1
    etp: int = 1

    # training knobs
    num_rollout: int = 10
    max_tokens_per_gpu: int = 8192
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 2
    global_batch_size: int = 16
    rollout_max_response_len: int = 1024

    # extra perf knobs
    sglang_ep_size: int | None = None  # defaults to num_gpus_per_node
    recompute: bool = True
    skip_prepare: bool = False

    def __post_init__(self):
        # [FUNCTION DOC] Resolves hardware (auto detects) and defaults GPUs per node from the
        # hardware table when not given. Mutates self.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates model and data directories, then downloads the HF model, the
    # dapo-math-17k and the aime-2024 datasets only if their target paths do not already exist
    # (shell test-or-download commands, so symlinked cluster copies are reused), and finally
    # converts the HF checkpoint to Megatron torch_dist format in the model directory.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    # model path is a symlink to /cluster_public; skip download if already present
    U.exec_command_cpu(
        f"test -e {args.model_dir}/{args.model_name} || "
        f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    # datasets are symlinked; skip if present
    U.exec_command_cpu(
        f"test -e {args.data_dir}/dapo-math-17k || "
        f"hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir {args.data_dir}/dapo-math-17k"
    )
    U.exec_command_cpu(
        f"test -e {args.data_dir}/aime-2024 || "
        f"hf download --repo-type dataset zhuzilin/aime-2024 --local-dir {args.data_dir}/aime-2024"
    )

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string and submits the colocated job with the
    # speculative-decoding v2 environment variable set to one.
    # Checkpoints: only HF dir and reference load (torch_dist) are passed; no save or load flags,
    # deliberately, so no checkpoint is written. Rollout: dapo-math-17k, deepscaler reward, sizes
    # from args. Eval only if enabled: AIME with one sample, interval 1000, max 4096 tokens. SGLang
    # EP size is the arg or the GPUs per node. Recompute flags (full, uniform, 1 layer) only when
    # recompute is on. Parallel flags come straight from the args, with sequence parallel always
    # on; dynamic batch with configured tokens per GPU. Algorithm: grpo with kl loss (zero
    # coefficient). Optimizer: adam lr 1e-6 with CPU offload. SGLang: engine size equals GPUs per
    # node, memory fraction 0.7, graph batch sizes up to 128, EAGLE speculative decoding (2 steps,
    # top-k 1, 3 draft tokens), max running requests 256, extra-buffer mamba radix cache. MTP
    # training: one layer, loss scale 0.2. Misc: dropout off, fp32 allreduce, flash attention, flex
    # token dispatcher, node/GPU counts, colocate. No wandb arguments.
    # KNOWN ISSUE: sequence parallel is always passed even when tensor parallel is 1.
    U = args.create_backend()
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"

    # Smoke runs: no checkpoint save (Megatron's final save is forced on the
    # last rollout whenever --save-interval is set — miles/utils/misc.py:192 —
    # so we omit both --save and --save-interval to keep ~464G per-config off
    # disk). The ref-load is still required to initialise model weights.
    ckpt_args = f"--hf-checkpoint {args.model_dir}/{args.model_name} " f"--ref-load {ref_load_path} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args += (
            "--eval-interval 1000 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 4096 "
            "--eval-top-p 1 "
        )

    sglang_ep = args.sglang_ep_size if args.sglang_ep_size is not None else args.num_gpus_per_node
    recompute = (
        ("--recompute-granularity full " "--recompute-method uniform " "--recompute-num-layers 1 ")
        if args.recompute
        else ""
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.tp} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.pp} "
        f"--context-parallel-size {args.cp} "
        f"--expert-model-parallel-size {args.ep} "
        f"--expert-tensor-parallel-size {args.etp} "
        f"{recompute}"
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-ep-size {sglang_ep} "
        "--sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 "
        # mtp speculative decoding
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        "--sglang-max-running-requests 256 "
        "--sglang-mamba-radix-cache-strategy extra_buffer "
    )

    mtp_args = "--enable-mtp-training " "--mtp-num-layers 1 " "--mtp-loss-scaling-factor 0.2 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--moe-token-dispatcher-type flex "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{mtp_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            "SGLANG_ENABLE_SPEC_V2": "1",
        },
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command-line entry point: runs prepare unless skip_prepare is set, then
    # execute.
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
