# [MODULE DOC] Launcher for colocated GRPO training of Qwen3.5-35B-A3B with multi-token-prediction
# (MTP) training and EAGLE-style speculative decoding in SGLang, on H200 or B300 nodes with the
# Megatron backend. Defines the script args, prepare (download, convert to Megatron torch_dist),
# execute (build grouped flags and submit) and main (prepare then execute).
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command-line configuration extending the shared train config. Fields: mode
    # (normal or debug_minimal with tiny sizes), run id, model name, Megatron model type, GPUs per
    # node (derived from hardware if unset), hardware (auto, H200, B300), enable_eval (off),
    # extra raw args, directories, enable_spec (speculative decoding, on), enable_spec_v2 (on),
    # enable_r3 (rollout routing replay, off), parallelism preset (tensor 2 x context 2, or tensor
    # 1; both with expert parallel 8), max response length (16384), and optional SGLang expert
    # parallel size and max running requests (unset omits the flag).
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen3.5-35B-A3B"
    megatron_model_type: str = "qwen3.5-35B-A3B"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "H200", "B300"] = "auto"
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    enable_spec: bool = True
    enable_spec_v2: bool = True
    enable_r3: bool = False
    parallelism: Literal["tp2-cp2-ep8", "tp1-ep8"] = "tp2-cp2-ep8"
    rollout_max_response_len: int = 16384
    # rollout-engine knobs, orthogonal to `parallelism`; None omits the flag and keeps
    # sglang's own default (ep_size = engine tp size, max_running_requests from memory)
    sglang_ep_size: int | None = None
    sglang_max_running_requests: int | None = None

    def __post_init__(self):
        # [FUNCTION DOC] Resolves hardware (auto detects) and defaults GPUs per node from a
        # per-hardware table when not given. Mutates self.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates model and data directories, downloads the HF model and the
    # dapo-math-17k and aime-2024 datasets, then converts the HF checkpoint to Megatron torch_dist
    # format in the model directory (backend shell commands; network, disk and GPU side effects).
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string and submits the colocated job. Sets the
    # environment variable enabling speculative decoding v2 to one or zero according to the flag.
    # Checkpoints: HF dir, reference load from the torch_dist dir, load/save in a per-run folder,
    # save interval 2 in debug else 20. Rollout: dapo-math-17k, deepscaler reward; debug uses 64
    # rollouts, 8 prompts x 2 samples, response 100, global batch 16; normal uses 3000 rollouts,
    # 32 x 8, configured response length, global batch 256. Eval (AIME, 16 samples, every 20) only
    # outside debug and when enabled. Parallelism: tensor and context sizes from the preset
    # (tp2-cp2 or tp1 with context 1), sequence parallel, expert parallel always 8, full recompute,
    # dynamic batch with 8192 tokens per GPU, log-prob chunk size 4096. Algorithm: grpo with kl loss
    # (zero coefficient), clip 0.2/0.28. Optimizer: adam lr 1e-6 with CPU offload. SGLang: 8 GPUs
    # per engine, memory fraction 0.7, fixed CUDA graph batch list up to 256, optional EP size and
    # max running requests, optional routing replay, on B300 the cutlass moe runner and trtllm
    # attention, and with speculative decoding an EAGLE setup (2 steps, top-k 1, 3 draft tokens)
    # plus an extra-buffer mamba radix cache strategy when v2 is on. MTP training: one layer, loss
    # scale 0.2. Misc: dropout off, fp32 allreduce, attention backend auto on B300 else flash, flex
    # token dispatcher, node/GPU counts, colocate.
    # KNOWN ISSUE: the B300 comment says triton backends are needed, but the code selects
    # flashinfer cutlass and trtllm attention; the expert-parallel size is fixed at 8 regardless of
    # GPUs per node; the parallelism preset name tp2-cp2-ep8 claims EP8 for tp1-ep8 too.
    U = args.create_backend()
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {ref_load_path} "
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
        "--rm-type deepscaler "
        f"--num-rollout {64 if args.mode == 'debug_minimal' else 3000} "
        f"--rollout-batch-size {8 if args.mode == 'debug_minimal' else 32} "
        f"--n-samples-per-prompt {2 if args.mode == 'debug_minimal' else 8} "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {16 if args.mode == 'debug_minimal' else 256} "
        "--balance-data "
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

    tp_size, cp_size = (1, 1) if args.parallelism == "tp1-ep8" else (2, 2)

    perf_args = (
        f"--tensor-model-parallel-size {tp_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {cp_size} "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "  # 32768 on H200
        "--log-probs-chunk-size 4096 "  # chunk logits when computing log probs to avoid OOM
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
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 "
    )

    if args.sglang_ep_size is not None:
        sglang_args += f"--sglang-ep-size {args.sglang_ep_size} "
    if args.sglang_max_running_requests is not None:
        sglang_args += f"--sglang-max-running-requests {args.sglang_max_running_requests} "

    if args.enable_r3:
        sglang_args += "--use-rollout-routing-replay "

    if args.hardware == "B300":
        # B300 (sm103) needs triton backends — flashinfer MoE/attention not yet supported
        # has weight update reshape issue.
        sglang_args += "--sglang-moe-runner-backend flashinfer_cutlass " "--sglang-attention-backend trtllm_mha "
    if args.enable_spec:
        sglang_args += (
            # mtp speculative decoding
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 2 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 3 "
        )
        if args.enable_spec_v2:
            sglang_args += "--sglang-mamba-radix-cache-strategy extra_buffer "

    mtp_args = "--enable-mtp-training " "--mtp-num-layers 1 " "--mtp-loss-scaling-factor 0.2 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # B300: auto selects FA2 via TE (patched for sm103); other GPUs: flash (FA2 direct)
        f"--attention-backend {'auto' if args.hardware == 'B300' else 'flash'} "
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
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
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
            "SGLANG_ENABLE_SPEC_V2": "1" if args.enable_spec_v2 else "0",
        },
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command-line entry point: runs prepare, then execute.
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
