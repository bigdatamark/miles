# [MODULE DOC]
# Launcher script (typer command line app) for GRPO training of DeepSeek-V3.2 (sparse attention
# model) with Megatron for training and SGLang for rollout, on DAPO math data with AIME eval.
# Supports optional precision variants: block-quant FP8 rollout, MXFP8 rollout (Blackwell
# hardware), MXFP8 training, and optional train/inference mismatch correction (MIS) config.
# Supports a single-node layout (half the GPUs train, half serve rollout) or a multi-node layout.
# Commands: full-train, prepare, prepare-megatron-ckpt, train. The module-level constants list
# layer-name fragments kept in high precision when converting to MXFP8 (one set for Hugging Face
# weight names, one for Megatron module names). Importing has no side effects besides creating the
# app and a default run id.
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_HF = (".kv_b_proj.",)
DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_MEGATRON = (
    ".linear_kv_up_proj",
    ".linear_k_up_proj",
    ".linear_v_up_proj",
)


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (node count, output dir,
    # backend creation). Key fields: mode (debug_minimal shortens responses to 100 tokens, saves
    # every 2 steps and disables eval); from_bf16_ckpt (download an already-bf16 checkpoint into the
    # "-bf16" directory instead of casting); use_single_node (split one node into trainer and
    # rollout halves); num_gpus_per_node / actor_num_nodes / actor_num_gpus_per_node /
    # rollout_num_gpus (None means derived or unset); hardware ("auto" is resolved to a concrete
    # GPU type); directory fields; num_rollout, rollout_batch_size, n_samples_per_prompt,
    # global_batch_size; no_save (skip checkpoint saving); rollout_mxfp8 / rollout_fp8 /
    # train_mxfp8 / fp8_param_gather (precision options; fp8_param_gather is not implemented);
    # enable_mis and tis_use_rs (train/inference mismatch correction and its rejection-sampling flag).
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "deepseek-ai"
    model_name: str = "DeepSeek-V3.2"
    megatron_model_type: str = "deepseek-v32"
    from_bf16_ckpt: bool = False
    use_single_node: bool = False
    num_gpus_per_node: int | None = None
    actor_num_nodes: int | None = None
    actor_num_gpus_per_node: int | None = 8
    rollout_num_gpus: int | None = None
    hardware: Literal["auto", "B200", "B300", "GB200", "GB300", "H100", "H200"] = "auto"
    enable_eval: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    num_rollout: int = 3000
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 256
    no_save: bool = False
    rollout_mxfp8: bool = False
    rollout_fp8: bool = False
    train_mxfp8: bool = False
    fp8_param_gather: bool = False
    enable_mis: bool = False
    tis_use_rs: bool = True

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves the hardware type (auto-detected when "auto") and fills GPUs per node from a
        # per-hardware table when unset. With use_single_node, forces one actor node and gives half
        # of the node's GPUs to training and half to rollout (integer division).
        # Raises assertion errors if both fp8 and mxfp8 rollout are requested, or if MXFP8 rollout or
        # training is requested on H100/H200 hardware (no native MXFP8).
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.use_single_node:
            # Half of the node trains and the other half serves rollout.
            self.actor_num_nodes = 1
            self.actor_num_gpus_per_node = self.num_gpus_per_node // 2
            self.rollout_num_gpus = self.num_gpus_per_node // 2
        assert not (self.rollout_fp8 and self.rollout_mxfp8), "rollout_fp8 and rollout_mxfp8 are mutually exclusive"
        if self.hardware in ("H100", "H200"):
            assert not self.rollout_mxfp8, "MXFP8 rollout is not supported on H100/H200 (no native MXFP8)"
            assert not self.train_mxfp8, "MXFP8 training is not supported on H100/H200 (no native MXFP8)"


def _prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Creates model and data directories, downloads the Hugging Face model
    # (into the model directory, using a "-bf16" suffixed folder when from_bf16_ckpt is set, else
    # the plain model name), then downloads the DAPO math and AIME 2024 datasets. Runs shell commands.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    if args.from_bf16_ckpt:
        U.exec_command_cpu(
            f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}-bf16"
        )
    else:
        U.exec_command_cpu(
            f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
        )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _prepare_bf16_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Unless the checkpoint was downloaded as bf16, casts the fp8 model to
    # bf16 into a "-bf16" folder under the model directory. No-op when from_bf16_ckpt is set.
    U = args.create_backend()
    if not args.from_bf16_ckpt:
        U.fp8_cast_bf16(
            path_src=f"{args.model_dir}/{args.model_name}",
            path_dst=f"{args.model_dir}/{args.model_name}-bf16/",
        )


def _prepare_mxfp8_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Only when rollout_mxfp8 is set, runs the Hugging Face to MXFP8 conversion
    # tool on the bf16 checkpoint, saving to a "-MXFP8" folder. If the user extra args do not already
    # name the high-precision layers option, appends the default list of Hugging Face layer-name
    # fragments to keep in high precision. Runs the tool as a GPU command.
    U = args.create_backend()
    if args.rollout_mxfp8:
        extra_args = args.extra_args
        if "--extra-high-precision-layers-hf" not in extra_args:
            extra_args += (
                f" --extra-high-precision-layers-hf {' '.join(DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_HF)} "
            )
        U.exec_command_gpu(
            f"python tools/convert_hf_to_mxfp8.py --model-dir {args.model_dir}/{args.model_name}-bf16 "
            f"--save-dir {args.model_dir}/{args.model_name}-MXFP8 "
            f"{extra_args} "
        )


def _prepare_fp8_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Only when rollout_fp8 is set, runs the Hugging Face to FP8 conversion tool
    # on the bf16 checkpoint with block quantization (block size 128 by 128), saving to a "-FP8"
    # folder under the model directory. Runs as a GPU command.
    """Convert BF16 checkpoint to block-quant FP8 (for sglang rollout, no MXFP8)."""
    U = args.create_backend()
    if args.rollout_fp8:
        U.exec_command_gpu(
            f"python tools/convert_hf_to_fp8.py "
            f"--model-dir {args.model_dir}/{args.model_name}-bf16 "
            f"--save-dir {args.model_dir}/{args.model_name}-FP8 "
            f"--strategy block --block-size 128 128"
        )


def _prepare_megatron_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Converts the bf16 checkpoint to a Megatron distributed checkpoint in the
    # model directory (shared folder so multi-node training can read it). Single-node mode converts
    # on the actor GPUs with default parallelism. Otherwise runs a multi-node conversion on the actor
    # nodes with tensor parallel 4, expert parallel 16, pipeline 6, last pipeline stage 13 layers,
    # expert-tensor parallel 1.

    U = args.create_backend()
    if args.use_single_node:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}-bf16",
            megatron_path=args.megatron_path,
        )

    else:
        extra_args = (
            "--tensor-model-parallel-size 4 "
            "--expert-model-parallel-size 16 "
            "--pipeline-model-parallel-size 6 "
            "--decoder-last-pipeline-num-layers 13 "
            "--expert-tensor-parallel-size 1 "
        )
        multinode = True
        num_nodes = args.actor_num_nodes

        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.actor_num_gpus_per_node,
            multinode=multinode,
            num_nodes=num_nodes,
            extra_args=extra_args,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}-bf16",
            megatron_path=args.megatron_path,
        )


def _prepare_cmd(args: ScriptArgs) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns the per-role pre-launch commands as a mapping. Empty in single-node
    # mode. Otherwise the "trainer" entry joins two rsync copies (Megatron distributed checkpoint and
    # the rollout Hugging Face checkpoint, whose name gets "-MXFP8" or "-FP8" suffix according to the
    # rollout precision) from the model directory to the local model directory, run in sequence.
    # KNOWN ISSUE: the rollout precision suffix is chosen from rollout flags only, so train_mxfp8
    # without rollout_mxfp8 copies the plain name while the training function points at the MXFP8 one.
    if args.use_single_node:
        return {}

    if args.rollout_mxfp8:
        hf_suffix = "-MXFP8"
    elif args.rollout_fp8:
        hf_suffix = "-FP8"
    else:
        hf_suffix = ""
    hf_name = f"{args.model_name}{hf_suffix}"

    copies = [
        command_utils.rsync_cmd(
            f"{args.model_dir}/{args.model_name}_torch_dist",
            f"{args.model_local_dir}/{args.model_name}_torch_dist",
        ),
        command_utils.rsync_cmd(f"{args.model_dir}/{hf_name}", f"{args.model_local_dir}/{hf_name}"),
    ]
    return {"trainer": " && ".join(copies)}


def _execute_train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Builds the training argument string and submits the job.
    # Checkpoint: hf checkpoint is the MXFP8 folder if either rollout or training MXFP8 is on, else
    # FP8 folder if rollout fp8, else plain; reference load is the Megatron distributed folder; load
    # path under the output directory; saving (every 20 steps, or 2 in debug mode) unless no_save.
    # Rollout: DAPO math, deepscaler reward, shuffled, chat template, response length 100 (debug) or
    # 8192, temperature 1, balanced data. Eval (only non-debug and enabled): AIME every 20 steps, 16
    # samples, length 16384, top-p 1. Perf: full uniform recompute, dynamic batch size, 32768 tokens
    # per GPU, padding multiple 4096, log-prob chunk 1024. GRPO with zero kl/entropy coefficients.
    # Adam optimizer at 1e-6. Optimizer CPU offload with precision-aware optimizer is used except for
    # MXFP8 training with fp8 param gather. MXFP8 training (Blackwell GPUs only) adds transformer-engine
    # mxfp8 flags; with H100/H200 etc. the training MXFP8 block is skipped silently (no else).
    # Parallelism by hardware (Blackwell and Hopper types accepted; anything else raises not
    # implemented): single node uses tensor and expert parallel equal to actor GPUs per node;
    # multi node uses tensor 2, pipeline 4, last stage 13 layers, expert 16. SGLang uses sparse
    # attention backend, flashmla sparse decode and prefill, bf16 kv cache, page size 64, memory
    # fraction 0.8, engine size 2 (single node) or 8 with data-parallel attention; MXFP8 rollout adds
    # flashinfer backends and a precision config pseudo-file keeping some projections in bf16; the
    # non-MXFP8 path picks the triton moe runner on H100/H200 else flashinfer. With enable_mis, adds
    # a pseudo-file config with truncated token-level importance sampling and a custom weight function.
    # Environment variables set force MLA for sparse attention, disable prefill dense threshold and
    # disable NVSHMEM over NCCL. Submits via the backend with the per-role copy commands.
    # Raises not implemented error for fp8 param gather and for unsupported hardware.
    # KNOWN ISSUE: the lines after the fp8 param gather raise (extra flags) are unreachable dead code.
    U = args.create_backend()
    ref_load_path = f"{args.model_dir}/{args.model_name}_torch_dist"
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    if args.rollout_mxfp8 or args.train_mxfp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-MXFP8"
    elif args.rollout_fp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-FP8"
    else:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    ckpt_args = f"--hf-checkpoint {hf_checkpoint}/ " f"--ref-load {ref_load_path} " f"--load {load_save_path} "
    if not args.no_save:
        ckpt_args += (
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
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
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
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
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
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--allgather-cp "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {args.rollout_num_gpus} "
        "--use-fault-tolerance "
        # f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    misc_env_vars = {
        "SGLANG_NSA_FORCE_MLA": "1",
        # "INDEXER_ROPE_NEOX_STYLE": "1",  # v3.2 uses NeoX (non-interleaved) style; GLM-5 uses interleaved.
        "SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
        "NVSHMEM_DISABLE_NCCL": "1",
    }

    if args.train_mxfp8:
        match args.hardware:
            case "B200" | "B300" | "GB200" | "GB300":
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe mxfp8 "
                    # --moe-router-padding-for-quantization
                )
                if args.fp8_param_gather:
                    raise NotImplementedError("FP8 param gather is not supported yet.")
                    misc_args += (
                        "--fp8-param-gather "
                        "--reuse-grad-buf-for-mxfp8-param-ag "
                        "--overlap-param-gather "
                        "--overlap-grad-reduce "
                    )
                    optimizer_args += (
                        "--use-precision-aware-optimizer "
                        # "--offload-optimizer-states "
                    )
                else:
                    optimizer_args += (
                        "--optimizer-cpu-offload "
                        "--overlap-cpu-optimizer-d2h-h2d "
                        "--use-precision-aware-optimizer "
                    )
    else:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    match args.hardware:
        case "B200" | "B300" | "GB200" | "GB300" | "H100" | "H200":
            if args.use_single_node:
                perf_args += (
                    f"--tensor-model-parallel-size {args.actor_num_gpus_per_node} "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 1 "
                    "--context-parallel-size 1 "
                    f"--expert-model-parallel-size {args.actor_num_gpus_per_node} "
                    "--expert-tensor-parallel-size 1 "
                )
            else:
                perf_args += (
                    "--tensor-model-parallel-size 2 "
                    "--sequence-parallel "
                    "--pipeline-model-parallel-size 4 "
                    "--decoder-last-pipeline-num-layers 13 "
                    "--context-parallel-size 1 "
                    "--expert-model-parallel-size 16 "
                    "--expert-tensor-parallel-size 1 "
                )

            sglang_args = (
                "--sglang-mem-fraction-static 0.8 "
                "--sglang-attention-backend nsa "
                "--sglang-dsa-decode-backend flashmla_sparse "
                "--sglang-dsa-prefill-backend flashmla_sparse "
                "--sglang-kv-cache-dtype bf16 "
                # NSA KV cache requires page_size=64 on CUDA.
                "--sglang-page-size 64 "
            )

            if args.rollout_mxfp8:
                if args.use_single_node:
                    sglang_world_size = 2
                else:
                    sglang_world_size = 8
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {sglang_world_size} "
                    "--sglang-fp8-gemm-backend flashinfer_trtllm "
                    "--sglang-moe-runner-backend flashinfer_trtllm_routed "
                    f"--sglang-tp-size {sglang_world_size} "
                    f"--sglang-dp-size {sglang_world_size} "
                    "--sglang-enable-dp-attention "
                    "--sglang-enable-dp-lm-head "
                    # f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    # f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
                    # "--sglang-moe-dense-tp-size 1 "
                )

                if "--extra-high-precision-layers-hf" not in args.extra_args:
                    misc_args += (
                        f"--extra-high-precision-layers-hf {' '.join(DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_HF)} "
                        f"--extra-high-precision-layers-megatron {' '.join(DEFAULT_MXFP8_EXTRA_HIGH_PRECISION_LAYERS_MEGATRON)} "
                    )

                te_precision_config_text = """
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
""".strip()
                if "--te-precision-config-file" not in args.extra_args:
                    misc_args += (
                        f"--te-precision-config-file {command_utils.encode_pseudo_file(te_precision_config_text)} "
                    )
            else:
                if args.use_single_node:
                    sglang_world_size = 2
                else:
                    sglang_world_size = 8
                sglang_decode_max_bs = 256
                if args.hardware in ("H100", "H200"):
                    sglang_moe_runner_backend = "triton"
                else:
                    sglang_moe_runner_backend = "flashinfer_trtllm_routed"
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {sglang_world_size} "
                    f"--sglang-moe-runner-backend {sglang_moe_runner_backend} "
                    f"--sglang-tp-size {sglang_world_size} "
                    f"--sglang-dp-size {sglang_world_size} "
                    "--sglang-enable-dp-attention "
                    "--sglang-enable-dp-lm-head "
                    # f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    # f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
                    # "--sglang-moe-dense-tp-size 1 "
                )
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
        extra_env_vars={**misc_env_vars},
        megatron_path=args.megatron_path,
        prepare_cmd=_prepare_cmd(args),
    )


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs):
    # [FUNCTION DOC] Command "full-train": download, bf16 cast, MXFP8 conversion, FP8 conversion,
    # Megatron conversion, then training, in order (conversions that are not requested are no-ops).
    """Full pipeline: download, cast, convert, copy, train."""
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_mxfp8_ckpt(args)
    _prepare_fp8_ckpt(args)
    _prepare_megatron_ckpt(args)
    _execute_train(args)


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Command "prepare": same preparation steps as full-train but without training.
    """Download model/data and convert to Megatron checkpoints (run on head node)."""
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_mxfp8_ckpt(args)
    _prepare_fp8_ckpt(args)
    _prepare_megatron_ckpt(args)


@app.command()
@command_utils.dataclass_cli
def prepare_megatron_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] Command "prepare-megatron-ckpt": only the bf16-to-Megatron checkpoint conversion.
    _prepare_megatron_ckpt(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] Command "train": runs training only; assumes all preparation has been done.
    """Run training only (assumes prepare is done)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: Empty typer callback so commands stay sub-commands. Does nothing.
    pass


if __name__ == "__main__":
    app()
