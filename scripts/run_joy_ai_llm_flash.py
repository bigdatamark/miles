# [MODULE DOC]
# Launcher script (single command, run through typer) for GRPO training of JoyAI-LLM-Flash on DAPO
# math with optional AIME eval, Megatron training on a few GPUs and SGLang rollout on others in the
# same node (training and rollout GPUs are listed separately, so no colocation flag is used).
# Blackwell hardware only. The single command downloads the model and data, optionally converts the
# model to MXFP8 for rollout, converts it to a Megatron checkpoint, and then trains. Supports MXFP8
# rollout and MXFP8 training with a few layers kept in bf16, mismatch correction (MIS), and a CI mode.
# Module-level constants: layer-name fragments kept in high precision for MXFP8 (Hugging Face names
# and Megatron names, as space-separated text) and the transformer engine precision config text that
# keeps latent attention up-projections and shared expert projections in bf16.
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


MXFP8_HIGH_PRECISION_LAYERS_HF = ".kv_b_proj. .shared_experts."
MXFP8_HIGH_PRECISION_LAYERS_MEGATRON = (
    ".linear_kv_up_proj .linear_k_up_proj .linear_v_up_proj " ".shared_experts.linear_fc1 .shared_experts.linear_fc2"
)
MXFP8_TE_PRECISION_CONFIG = """
configs:
  bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe: {}
matchers:
  mla_kv_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_kv_up_proj"
    config: "bf16"
  absorbed_k_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_k_up_proj"
    config: "bf16"
  absorbed_v_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_v_up_proj"
    config: "bf16"
  shared_fc1:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc1"
    config: "bf16"
  shared_fc2:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc2"
    config: "bf16"
""".strip()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: mode (debug_minimal gives short responses of 100 tokens, saves every 2 steps,
    # no eval), run id, Hugging Face org/name, Megatron model type, GPUs per node (None derived from
    # hardware), actor GPUs (default 4) and rollout GPUs (default 4), hardware (Blackwell types only),
    # eval switch, extra raw args, directories, rollout_mxfp8 and train_mxfp8 (training requires
    # rollout), MIS switch with rejection-sampling flag, CI test flag, checkpoint saving switch (on),
    # number of rollouts, global batch size, optional padding multiple and log-prob chunk size, and
    # the number of layers kept in bf16 at the start (1) and end (6) when using MXFP8.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "jdopensource"
    model_name: str = "JoyAI-LLM-Flash"
    megatron_model_type: str = "joyai-llm-flash"
    num_gpus_per_node: int | None = None
    actor_num_gpus_per_node: int | None = 4
    rollout_num_gpus: int | None = 4
    hardware: Literal["auto", "B200", "B300", "GB200", "GB300"] = "auto"
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    rollout_mxfp8: bool = False
    train_mxfp8: bool = False
    enable_mis: bool = False
    tis_use_rs: bool = True
    ci_test: bool = False
    save_checkpoints: bool = True
    num_rollout: int = 3000
    global_batch_size: int = 256
    data_pad_size_multiplier: int | None = None
    log_probs_chunk_size: int | None = None
    mxfp8_num_layers_at_start_in_bf16: int = 1
    mxfp8_num_layers_at_end_in_bf16: int = 6

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves hardware and GPUs per node from a per-hardware table when unset. Asserts that MXFP8
        # training is only requested together with MXFP8 rollout.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.train_mxfp8:
            assert self.rollout_mxfp8, "train_mxfp8 requires rollout_mxfp8"


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates model/data directories, downloads the Hugging Face model, the DAPO math
    # and AIME 2024 datasets. If rollout MXFP8 is on, runs the Hugging Face to MXFP8 conversion tool
    # (GPU command) into a "-MXFP8" folder, passing the bf16 layer counts at start and end, the high
    # precision layer-name list, and the user's extra args. Then converts the original checkpoint to a
    # Megatron distributed checkpoint in the model directory using the actor GPUs.
    # KNOWN ISSUE: the user's extra raw args are appended to the MXFP8 conversion command, which is
    # meant for the training command.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)

    if args.rollout_mxfp8:
        U.exec_command_gpu(
            f"python tools/convert_hf_to_mxfp8.py --model-dir {args.model_dir}/{args.model_name} "
            f"--save-dir {args.model_dir}/{args.model_name}-MXFP8 "
            f"--num-layers-at-start-in-bf16 {args.mxfp8_num_layers_at_start_in_bf16} "
            f"--num-layers-at-end-in-bf16 {args.mxfp8_num_layers_at_end_in_bf16} "
            f"--extra-high-precision-layers-hf {MXFP8_HIGH_PRECISION_LAYERS_HF} "
            f"{args.extra_args} "
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        # To support multi-node training, for simplicity, we put model into shared folder
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs, *, wandb_file: str = __file__):
    # [FUNCTION DOC] Builds the training argument string and submits the job. The wandb_file argument
    # selects the file used to derive default logging names (defaults to this file). Checkpoint: hf
    # checkpoint is the MXFP8 folder when MXFP8 training is on, else the plain folder; reference load
    # from the distributed folder; load/save under the output directory run folder (every 2 steps debug
    # else 20) only if saving is on. Rollout: DAPO math, math reward, chat template, shuffled, rollouts
    # and global batch from options, 32 prompts x 8 samples, temperature 1, response length 100 (debug)
    # or 8192, balanced data. Eval (non-debug and enabled): AIME every 20 steps, 16 samples, length
    # 16384, top-p 1. Perf: full uniform recompute, dynamic batch 32768 tokens per GPU, optional padding
    # multiple and log-prob chunk. GRPO with kl loss on at zero coefficient, zero entropy, asymmetric
    # clip. Adam 1e-6. Misc: dropout off, fp32 accumulation, auto attention backend, one actor node with
    # the configured actor GPUs, separate rollout GPU count, fault tolerance. MXFP8 training (Blackwell
    # only): transformer engine mxfp8 recipe with first and last layers in bf16. Hardware (only
    # Blackwell types; others raise not implemented): tensor and expert parallel equal to actor GPUs,
    # sequence parallel, pipeline 1, context 1; SGLang memory fraction 0.7 and trtllm latent-attention
    # backend; MXFP8 rollout uses 2-GPU engines with data-parallel attention, flashinfer gemm and moe
    # backends, cuda graph 256, routing replay, shared-experts fusion disabled, high-precision layer
    # lists, optimizer CPU offload group, and a precision config pseudo-file; otherwise 1-GPU engines
    # with cuda graph 256. Optional MIS config (token-level truncated importance sampling) and custom
    # weight function. Optional CI test flag. Submits via the backend with no extra environment.
    # KNOWN ISSUE: the data-parallel attention flag is added twice in the MXFP8 rollout engine flags.
    # KNOWN ISSUE: MXFP8 rollout without MXFP8 training loads the bf16 hf checkpoint into rollout
    # (the MXFP8 folder is only chosen when training MXFP8 is on).
    U = args.create_backend()
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    if args.train_mxfp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-MXFP8"
    else:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    ckpt_args = f"--hf-checkpoint {hf_checkpoint}/ " f"--ref-load {ref_load_path} "
    if args.save_checkpoints:
        ckpt_args += (
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
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
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

    perf_args = (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
    )
    if args.data_pad_size_multiplier is not None:
        perf_args += f"--data-pad-size-multiplier {args.data_pad_size_multiplier} "
    if args.log_probs_chunk_size is not None:
        perf_args += f"--log-probs-chunk-size {args.log_probs_chunk_size} "

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

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend auto "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        "--use-fault-tolerance "
        # f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {}

    if args.train_mxfp8:
        match args.hardware:
            case "B200" | "B300" | "GB200" | "GB300":
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe mxfp8 "
                    "--first-last-layers-bf16 "
                    f"--num-layers-at-start-in-bf16 {args.mxfp8_num_layers_at_start_in_bf16} "
                    f"--num-layers-at-end-in-bf16 {args.mxfp8_num_layers_at_end_in_bf16} "
                    # "--fp8-param-gather "
                    # "--reuse-grad-buf-for-mxfp8-param-ag "
                    # --moe-router-padding-for-quantization
                )

    match args.hardware:
        case "B200" | "B300" | "GB200" | "GB300":
            perf_args += (
                f"--tensor-model-parallel-size {args.actor_num_gpus_per_node} "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                f"--expert-model-parallel-size {args.actor_num_gpus_per_node} "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = "--sglang-mem-fraction-static 0.7 " "--sglang-attention-backend trtllm_mla "
            if args.rollout_mxfp8:
                sglang_world_size = 2
                sglang_decode_max_bs = 256
                sglang_args += (
                    "--sglang-enable-dp-attention "
                    f"--rollout-num-gpus-per-engine {sglang_world_size} "
                    "--sglang-fp8-gemm-backend flashinfer_trtllm "
                    "--sglang-moe-runner-backend flashinfer_trtllm_routed "
                    f"--sglang-tp-size {sglang_world_size} "
                    f"--sglang-dp-size {sglang_world_size} "
                    "--sglang-enable-dp-attention "
                    f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
                    # f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    # f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                )
                misc_args += (
                    "--use-rollout-routing-replay "
                    "--sglang-disable-shared-experts-fusion "
                    f"--extra-high-precision-layers-hf {MXFP8_HIGH_PRECISION_LAYERS_HF} "
                    f"--extra-high-precision-layers-megatron {MXFP8_HIGH_PRECISION_LAYERS_MEGATRON} "
                )
                optimizer_args += (
                    "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
                )
                misc_args += (
                    f"--te-precision-config-file {command_utils.encode_pseudo_file(MXFP8_TE_PRECISION_CONFIG)} "
                )
            else:
                sglang_args += "--rollout-num-gpus-per-engine 1 " "--sglang-cuda-graph-max-bs-decode 256 "
        case _:
            raise NotImplementedError

    if args.enable_mis:
        config_text = f"""
use_tis: true
use_rs: {"true" if args.tis_use_rs else "false"}
tis_level: "token"
rs_level: "token"
tis_mode: "truncate"
tis_lower_bound: 0.5
tis_upper_bound: 2.0
rs_lower_bound: null
rs_upper_bound: null
rs_veto_threshold: 1.0e-4
tis_batch_normalize: true
""".strip()
        misc_args += (
            f"--custom-config-path {command_utils.encode_pseudo_file(config_text)} "
            "--custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(wandb_file, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{'--ci-test ' if args.ci_test else ''}"
        f"{eval_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**misc_env_vars},
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command line entry point: runs prepare, then execute (training).
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
