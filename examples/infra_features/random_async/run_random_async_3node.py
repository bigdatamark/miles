# [MODULE DOC] Launcher for the synthetic random-async stress test of Qwen3.5-35B-A3B on three
# 8-GPU nodes: two nodes for training (context parallel 8, expert parallel 8, FP8 blockwise) and
# rollout GPUs served by a prefill/decode disaggregated sglang layout described by a YAML file
# next to this script (one prefill one decode engine group). The rollout function is the random
# one from this directory, so no dataset is needed; weights are loaded from a FP8 HF checkpoint
# and the learning rate is zero. Command line program; prepare stages the BF16 and FP8 models
# and the converted checkpoint (skippable); execute submits the async training job.
import os
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command line options extending the shared train config: mode (normal, or
    # debug_minimal with max response length 100), run id (generated at import), model name and
    # Megatron type, GPUs per node (8), actor node count (2), total rollout GPUs (8), number of
    # rollouts (4), model and Megatron directories, pause generation mode (in place default or
    # retract), weight transfer mode (broadcast default, packed broadcast or p2p), a switch to
    # skip the prepare step and free-form extra arguments.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen3.5-35B-A3B"
    megatron_model_type: str = "qwen3.5-35B-A3B"
    num_gpus_per_node: int = 8
    actor_num_nodes: int = 2
    rollout_num_gpus: int = 8
    num_rollout: int = 4
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    pause_generation_mode: Literal["in_place", "retract"] = "in_place"
    update_weight_transfer_mode: Literal["broadcast", "broadcast_packed", "p2p"] = "broadcast"
    skip_prepare: bool = False
    extra_args: str = ""


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Stage models: create the model dir; download the BF16 HF model unless either
    # a converted checkpoint already marked as release exists or the HF directory exists; download
    # the FP8 variant unless its directory exists; then convert the BF16 HF checkpoint to a
    # distributed checkpoint with one process per GPU. Shell conditionals make the downloads
    # idempotent. No return.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir}")
    U.exec_command_cpu(
        f'test "$(cat {args.model_dir}/{args.model_name}_torch_dist/latest_checkpointed_iteration.txt 2>/dev/null)" = release || '
        f"test -e {args.model_dir}/{args.model_name} || "
        f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.exec_command_cpu(
        f"test -e {args.model_dir}/{args.model_name}-FP8 || "
        f"hf download Qwen/{args.model_name}-FP8 --local-dir {args.model_dir}/{args.model_name}-FP8"
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
    # [FUNCTION DOC] Build the training arguments and submit the async training job.
    # Raises a value error for in-place pause with p2p transfer. Uses the engine layout YAML in
    # this script's directory. Argument groups: checkpoints (HF checkpoint is the FP8 variant,
    # reference load is the converted BF16 checkpoint, load path under the run folder of the
    # output dir); rollout (the random rollout function, no global dataset, rollout count from
    # options, 32 prompts x 16 samples, max response 8192 or 100 in debug mode, global batch
    # 512, data balancing, pause mode); perf (context parallel 8, expert parallel 8, full uniform
    # recompute, dynamic batch of 8000 tokens per GPU, log-prob chunk 1024, flex DeepEP token
    # dispatcher, Transformer Engine FP8 e4m3 blockwise); runtime (manual router prefill/decode
    # policies, min-load assignment, expandable-segments allocator env for training); GRPO
    # (zero-coefficient low variance KL loss, clip 0.2/0.28, importance sampling correction);
    # Adam with learning rate and weight decay zero; sglang (4 GPUs per engine, the YAML config,
    # memory fraction 0.85, FA3 attention, data-parallel attention 4, expert parallel 4, DP LM
    # head, DeepEP all-to-all, context length 80000, metrics on, server concurrency 384, plus
    # the p2p seed flag when p2p); misc (flash attention, transfer mode, actor node count and
    # GPUs, rollout GPU count). The job environment disables the flashinfer version check, pins
    # the random rollout limits (max context 60000, concurrency 64 per GPU), enables forced
    # prefill DP rank query with a 900 second disaggregation wait, and extends the python path
    # with Megatron and this directory so the rollout module is importable.
    # Note the 3 node total comes from 2 actor nodes plus the 8 rollout GPUs.
    U = args.create_backend()
    if args.pause_generation_mode == "in_place" and args.update_weight_transfer_mode == "p2p":
        raise ValueError(
            "in_place + p2p is not supported: P2P transfer engine conflicts with "
            "active NCCL inference. Use broadcast with in_place, or retract with p2p."
        )

    example_dir = os.path.dirname(os.path.abspath(__file__))
    sglang_config_path = os.path.join(example_dir, "sglang_config_qwen3_5_35b_1p1d.yaml")
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name}-FP8/ "
        f"--ref-load {ref_load_path} "
        f"--load {load_save_path} "
    )

    rollout_args = (
        "--rollout-function-path random_async_rollout.generate_rollout_random_async "
        "--disable-rollout-global-dataset "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 16 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        "--global-batch-size 512 "
        "--balance-data "
        f"--pause-generation-mode {args.pause_generation_mode} "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 8 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8000 "
        "--log-probs-chunk-size 1024 "
        "--moe-token-dispatcher-type flex "
        "--moe-flex-dispatcher-backend deepep "
        "--transformer-impl transformer_engine "
        "--fp8-format e4m3 "
        "--fp8-recipe blockwise "
    )

    runtime_args = (
        "--router-prefill-policy manual "
        "--router-decode-policy manual "
        "--router-assignment-mode min_load "
        '--train-env-vars \'{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}\' '
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 0.0 "
        "--lr-decay-style constant "
        "--weight-decay 0.0 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_extra = ""
    if args.update_weight_transfer_mode == "p2p":
        sglang_extra = "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        f"--sglang-config {sglang_config_path} "
        f"--sglang-mem-fraction-static 0.85 {sglang_extra}"
        "--sglang-attention-backend fa3 "
        "--sglang-enable-dp-attention "
        "--sglang-data-parallel-size 4 "
        "--sglang-expert-parallel-size 4 "
        "--sglang-enable-dp-lm-head "
        "--sglang-moe-a2a-backend deepep "
        "--sglang-context-length 80000 "
        "--sglang-enable-metrics "
        "--sglang-server-concurrency 384 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        f"--attention-backend flash --update-weight-transfer-mode {args.update_weight_transfer_mode} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{runtime_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        train_script="train_async.py",
        megatron_path=args.megatron_path,
        extra_env_vars={
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "RANDOM_ASYNC_MAX_CONTEXT_TOKENS": "60000",
            "RANDOM_ASYNC_CONCURRENCY_PER_GPU": "64",
            "SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK": "1",
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "900",
            "PYTHONPATH": f"{args.megatron_path}:{example_dir}",
        },
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command line entry: run prepare unless the skip switch is set, then execute.
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
