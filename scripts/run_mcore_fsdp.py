# [MODULE DOC]
# Launcher script (single command, run through typer) for GRPO training of Qwen3-4B variants on DAPO
# math, with a choice of training backend (FSDP or Megatron), optional reference model with kl
# loss, colocated or split (half training, half rollout) GPU layout, optional multi-dataset eval
# (AIME, GPQA, IFBench), optional deterministic "true on-policy" mode and optional dynamic sampling.
# The single command downloads models and datasets (converting the checkpoint for Megatron) and then
# trains. Used for comparing the two backends.
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: train backend (fsdp default or megatron), use_ref (reference model and kl
    # loss; off), colocate (on), model name (default Qwen3-4B-Instruct-2507), GPUs per node (None
    # derived), hardware (auto, H100, GB300), mode, run id, multi_eval (three eval datasets), true
    # on-policy switch, dynamic sampling switch, eval switch (on), Megatron model type (None; derived
    # for the Megatron backend), extra raw args, directories.
    train_backend: Literal["fsdp", "megatron"] = "fsdp"
    use_ref: bool = False
    colocate: bool = True
    model_name: str = "Qwen3-4B-Instruct-2507"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "H100", "GB300"] = "auto"
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    multi_eval: bool = False
    true_on_policy: bool = False
    dynamic_sampling: bool = False
    enable_eval: bool = True
    megatron_model_type: str | None = None
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves hardware and GPUs per node. For the Megatron backend, sets the Megatron model type
        # from a fixed table (Qwen3-4B, its instruct 2507 variant, and base); any other model name
        # raises a key error.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.train_backend == "megatron":
            self.megatron_model_type = {
                "Qwen3-4B": "qwen3-4B",
                "Qwen3-4B-Instruct-2507": "qwen3-4B-Instruct-2507",
                "Qwen3-4B-Base": "qwen3-4B",
            }[self.model_name]


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates model/data directories, downloads the Qwen organization model and the
    # DAPO math, AIME 2024, GPQA diamond and IFBench datasets; for the Megatron backend also converts
    # the checkpoint to a Megatron distributed checkpoint in the model directory.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} " f"--local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
    U.hf_download_dataset("zyzshishui0627/gpqa_diamond", data_dir=args.data_dir)
    U.hf_download_dataset("zyzshishui0627/IFBench", data_dir=args.data_dir)
    if args.train_backend == "megatron":
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}",
            megatron_path=args.megatron_path,
        )


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string and submits the job. Checkpoint: hf
    # checkpoint; load/save under the output directory run folder every 2 (debug) or 20 steps;
    # reference load only when use_ref (converted folder for Megatron, plain folder for FSDP).
    # Rollout: DAPO math, math reward, chat template, shuffled, balanced; debug: 10 rollouts, 8 prompts
    # x 8 samples, length 100, global batch 64; normal: 3000 rollouts, 64 x 16, length 32768, global
    # batch 1024; optional over-sampling 128 with nonzero-reward-std filter (non-debug). Eval
    # (non-debug and enabled): every 20 steps; either a single AIME set (16 samples, top-p 1) or, with
    # multi_eval, a generated config file passed as a pseudo-file with AIME, GPQA and IFBench (top-p
    # 0.7). Perf: dynamic batch, 9216 tokens per GPU (Megatron) or 32768. GRPO with zero coefficients
    # and kl loss flag only when a reference is used. Adam 1e-6. SGLang: 1 GPU per engine, memory
    # fraction 0.7 (Megatron) or 0.75, chunked prefill 4096. FSDP backend: buffer 512 MiB, gradient
    # checkpointing, flash attention 2 (3 with true on-policy plus fa3 SGLang backend). Megatron
    # backend: tensor 2 and context 4 on 8-GPU nodes else 1, sequence parallel, full uniform recompute,
    # dropout off, fp32 accumulation, flash attention. Layout: colocate, or split with half the GPUs
    # per node for training and half for rollout. FSDP adds an expandable-segments environment for
    # trainers. Fault tolerance and dump details. True on-policy adds deterministic inference flags
    # and environment variables. Wandb flags with a hard-coded project and a group built from backend,
    # reference and layout. For Megatron, runs a shell probe of nvidia-smi at call time to detect
    # NVLink and sets python path, max connections and NVLS accordingly.
    # KNOWN ISSUE: the wandb project name is hard-coded and wandb is always enabled.
    # KNOWN ISSUE: the group name always says Qwen3-4B regardless of the model name.
    # KNOWN ISSUE: the os module is imported inside the function instead of at the module top.
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 20} "
        f"--save-retain-interval {2 if args.mode == 'debug_minimal' else 20} "
    )

    if args.train_backend == "megatron":
        # Megatron uses torch_dist checkpoint format for ref model
        if args.use_ref:
            ckpt_args += f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
    else:
        # FSDP now supports ref-load with HF checkpoint
        if args.use_ref:
            ckpt_args += f"--ref-load {args.model_dir}/{args.model_name} "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type math "
        f"--num-rollout {10 if args.mode == 'debug_minimal' else 3000} "
        f"--rollout-batch-size {8 if args.mode == 'debug_minimal' else 64} "
        f"--n-samples-per-prompt {8 if args.mode == 'debug_minimal' else 16} "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
        "--rollout-temperature 1 "
        f"--global-batch-size {64 if args.mode == 'debug_minimal' else 1024} "
    )

    if args.dynamic_sampling and (args.mode != "debug_minimal"):
        rollout_args += (
            "--over-sampling-batch-size 128 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_max_response_len = 32768
        eval_args += "--eval-interval 20 "
        if args.multi_eval:
            eval_config_text = f"""
eval:
  defaults:
    max_response_len: {eval_max_response_len}
    top_p: 0.7
  datasets:
    - name: aime
      path: {args.data_dir}/aime-2024/aime-2024.jsonl
      rm_type: math
      n_samples_per_eval_prompt: 16
    - name: gpqa
      path: {args.data_dir}/gpqa_diamond/gpqa_eval.jsonl
      rm_type: gpqa
      n_samples_per_eval_prompt: 2
    - name: ifbench
      path: {args.data_dir}/IFBench/IFBench_eval.jsonl
      rm_type: ifbench
      n_samples_per_eval_prompt: 1
""".strip()
            eval_args += f"--eval-config {command_utils.encode_pseudo_file(eval_config_text)} "
        else:
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 16 "
                f"--eval-max-response-len {eval_max_response_len} "
                "--eval-top-p 1 "
            )

    perf_args = (
        "--use-dynamic-batch-size " f"--max-tokens-per-gpu {9216 if args.train_backend == 'megatron' else 32768} "
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
    if args.use_ref:
        grpo_args += "--use-kl-loss "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static {0.7 if args.train_backend == 'megatron' else 0.75} "
        "--sglang-decode-log-interval 1000 "
        "--sglang-chunked-prefill-size 4096 "
    )

    if args.train_backend == "fsdp":
        train_backend_args = (
            "--train-backend fsdp "
            f"--update-weight-buffer-size {512 * 1024 * 1024} "  # 512MB
            "--gradient-checkpointing "
            "--attn-implementation flash_attention_2 "
        )
        if args.true_on_policy:
            train_backend_args += "--sglang-attention-backend fa3 " "--attn-implementation flash_attention_3 "
    else:
        tp_size = 2 if args.num_gpus_per_node == 8 else 1
        cp_size = 4 if args.num_gpus_per_node == 8 else 1
        train_backend_args = (
            f"--tensor-model-parallel-size {tp_size} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            f"--context-parallel-size {cp_size} "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
        )

    if args.colocate:
        misc_args = (
            f"--actor-num-nodes {args.num_nodes} " f"--actor-num-gpus-per-node {args.num_gpus_per_node} " "--colocate "
        )
    else:
        actor_gpus = args.num_gpus_per_node // 2
        rollout_gpus = args.num_gpus_per_node // 2
        misc_args = (
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node {actor_gpus} "
            f"--rollout-num-gpus {rollout_gpus} "
        )

    if args.train_backend == "fsdp":
        misc_args += """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """

    misc_args += "--use-fault-tolerance " f"--dump-details {args.output_dir}/{args.run_id}/dump_details "

    misc_env_vars = {}

    true_on_policy_args = ""
    true_on_policy_envs = {}
    if args.true_on_policy:
        true_on_policy_args = (
            "--sglang-enable-deterministic-inference "
            "--sglang-rl-on-policy-target fsdp "
            "--deterministic-mode "
            "--true-on-policy-mode "
        )
        true_on_policy_envs = {
            "NCCL_ALGO": "allreduce:tree",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    backend_name = "fsdp" if args.train_backend == "fsdp" else "megatron"
    ref_name = "ref" if args.use_ref else "noref"
    colocate_name = "" if args.colocate else "-dist"
    wandb_group = f"qwen3-4B-{backend_name}-{ref_name}{colocate_name}"

    wandb_args = f"--use-wandb " f"--wandb-project miles-dev-megatron-fsdp " f"--wandb-group {wandb_group} "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{eval_args} "
        f"{perf_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{train_backend_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
        f"{wandb_args} "
        f"{args.extra_args} "
    )

    extra_env_vars = misc_env_vars
    if args.train_backend == "megatron":
        import os

        nvlink_count = os.popen('nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l').read().strip()
        has_nvlink = "1" if int(nvlink_count or "0") > 0 else "0"

        extra_env_vars |= {
            "PYTHONPATH": args.megatron_path,
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": has_nvlink,
        }

    extra_env_vars |= true_on_policy_envs

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type if args.train_backend == "megatron" else None,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command line entry point: runs prepare, then execute.
    """Main entry point for unified MCORE/FSDP training script.

    Examples:
        # FSDP colocated without ref
        python run_mcore_fsdp.py --train-backend fsdp --colocate --no-use-ref

        # FSDP distributed with ref
        python run_mcore_fsdp.py --train-backend fsdp --no-colocate --use-ref

        # Megatron colocated without ref
        python run_mcore_fsdp.py --train-backend megatron --colocate --no-use-ref

        # Megatron distributed with ref
        python run_mcore_fsdp.py --train-backend megatron --no-colocate --use-ref
    """
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
