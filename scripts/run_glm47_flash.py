# [MODULE DOC]
# Launcher script (single command, run through typer) for GRPO training of GLM-4.7-Flash with
# Megatron training and SGLang rollout colocated on the same GPUs, on DAPO math with optional AIME
# eval. The single command first prepares (download model and datasets, convert checkpoint to a
# Megatron distributed one) and then trains. Rollout uses EAGLE speculative decoding and rollout
# routing replay; the engine size defaults depend on hardware because the model has 20 attention
# heads.
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: mode (debug_minimal saves every 2 steps, response length 100, no eval), run
    # id, Hugging Face org/name, Megatron model type, GPUs per node (None means derived from
    # hardware), hardware (auto, H200, B200), rollout GPUs per engine (None means derived: 2 on B200
    # else 1), optional SGLang attention backend name, eval switch (on), extra raw args, and
    # data/model/Megatron directories.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-4.7-Flash"
    megatron_model_type: str = "glm4.7-flash"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "H200", "B200"] = "auto"
    rollout_num_gpus_per_engine: int | None = None  # None => derive from hardware
    sglang_attention_backend: str | None = None
    enable_eval: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves hardware and fills GPUs per node from a per-hardware table when unset.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates model/data directories, downloads the Hugging Face model, downloads the
    # DAPO math and AIME 2024 datasets, then converts the model to a Megatron distributed checkpoint
    # in the model directory (single-node conversion with default parallelism, using all GPUs per
    # node). Side effects: shell commands and downloads through the backend.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} " f"--local-dir {args.model_dir}/{args.model_name}"
    )
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
    # [FUNCTION DOC] Builds the training arguments and submits the job. Checkpoint: hf checkpoint from
    # the model directory, reference load from the distributed folder, load/save under the output
    # directory run folder, every 2 steps (debug) or 20. Rollout: DAPO math, deepscaler reward, chat
    # template, shuffled, 3000 rollouts, 32 prompts x 8 samples, global batch 256, temperature 1,
    # response length 100 (debug) or 8192. Eval (non-debug and enabled): AIME every 20 steps, 16
    # samples, length 16384, temperature 0.6, top-p 0.95. Parallelism: tensor 4 with sequence
    # parallel, pipeline 1, context 1, expert 8, expert-tensor 1; full uniform recompute; dynamic
    # batch 32768 tokens per GPU. GRPO with kl loss flag on but zero coefficients, asymmetric clip.
    # Adam 1e-6 with CPU offload and precision-aware optimizer. SGLang: engine GPUs from the option or
    # 2 on B200 else 1 (rollout tensor size must divide the 20 attention heads), memory fraction 0.7,
    # EAGLE speculative decoding (2 steps, top-k 1, 3 draft tokens), rollout routing replay; the
    # attention backend flag is added unless unset or "default"; on B200 with an unset, default or
    # flashinfer backend the flashinfer MLA ragged option is disabled. Misc: dropouts off, fp32 grad
    # accumulation, flash attention, colocate, fault tolerance. Submits via the backend.
    U = args.create_backend()
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += (
            "--eval-interval 20 "
            f"--eval-prompt-data aime24 {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-temperature 0.6 "
            "--eval-top-p 0.95 "
        )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
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

    # GLM-4.7-Flash has 20 attention heads, so rollout TP must divide 20.
    rollout_num_gpus_per_engine = (
        args.rollout_num_gpus_per_engine
        if args.rollout_num_gpus_per_engine is not None
        else (2 if args.hardware == "B200" else 1)
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.7 "
        # EAGLE speculative decoding (MTP)
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        # rollout routing replay
        "--use-rollout-routing-replay "
    )

    if args.sglang_attention_backend not in (None, "default"):
        sglang_args += f"--sglang-attention-backend {args.sglang_attention_backend} "

    if args.hardware == "B200" and args.sglang_attention_backend in (None, "default", "flashinfer"):
        sglang_args += "--sglang-flashinfer-mla-disable-ragged "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        # "--ci-test "
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
    # [FUNCTION DOC] Command line entry point: runs prepare, then execute (training).
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
