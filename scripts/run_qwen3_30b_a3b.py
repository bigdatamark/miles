# [MODULE DOC] Launcher for GRPO training of Qwen3-30B-A3B (MoE) with Megatron and SGLang, covering
# several hardware generations (H100, Blackwell parts, Rubin) and optional low-precision modes
# (FP8, MXFP8, INT4, NVFP4 rollout; FP8, MXFP8, NVFP4 training), Megatron bridge mode, and
# train/infer mismatch correction. Structure: a script-args dataclass with derived defaults,
# prepare (download models and datasets, convert to low precision and to Megatron torch_dist),
# execute (assemble grouped flag strings and submit the job via command_utils), and main (runs
# prepare then execute). Relies on command_utils for shell commands, hardware resolution and the
# pseudo-file encoding of inline config text.
import os
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command-line configuration, extending the shared train config. Key fields: mode
    # (normal or a minimal debug mode with short responses and frequent saves), run id, model name,
    # Megatron model type, GPU counts (per node, actor per node, rollout total; all optional and
    # derived), no_colocate (separate actor and rollout GPUs), hardware (auto-detected by default),
    # enable_eval, extra raw args, directories, boolean precision switches for rollout (fp8, mxfp8,
    # int4, nvfp4, attention fp8) and training (fp8, mxfp8, nvfp4), megatron bridge switch, mis
    # (train-infer mismatch correction) switch, and tis_use_rs (whether the mis config also uses
    # rejection sampling, default on). Validation happens in the post-init step.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_name: str = "Qwen3-30B-A3B"
    megatron_model_type: str = "qwen3-30B-A3B"
    num_gpus_per_node: int | None = None
    actor_num_gpus_per_node: int | None = None
    rollout_num_gpus: int | None = None
    no_colocate: bool = False
    hardware: Literal["auto", "H100", "B200", "B300", "GB200", "GB300", "Rubin"] = "auto"
    enable_eval: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    rollout_fp8: bool = False
    rollout_mxfp8: bool = False
    rollout_int4: bool = False
    rollout_nvfp4: bool = False
    rollout_attn_fp8: bool = False
    train_fp8: bool = False
    train_mxfp8: bool = False
    train_nvfp4: bool = False
    enable_megatron_bridge: bool = False
    enable_mis: bool = False
    # TODO improve, should be able to override more easily
    tis_use_rs: bool = True

    def __post_init__(self):
        # [FUNCTION DOC] Resolves derived defaults and validates the configuration, mutating self.
        # Steps: resolve hardware (auto detects), default GPUs per node from a per-hardware table,
        # force no_colocate on when rollout nvfp4 is chosen. In non-colocated mode the actor gets
        # half the node's GPUs (integer division) and rollout the remainder; in colocated mode
        # both default to the full node. Explicitly given values are kept. Raises assertion errors
        # if more than one rollout precision mode or more than one training precision mode is set,
        # or if mxfp8/nvfp4 (rollout or train) is requested on non-Blackwell hardware (only B200,
        # B300, GB200, GB300 allowed, so Rubin is rejected).
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        self.no_colocate = self.no_colocate or self.rollout_nvfp4
        if self.no_colocate:
            self.actor_num_gpus_per_node = self.actor_num_gpus_per_node or self.num_gpus_per_node // 2
            self.rollout_num_gpus = self.rollout_num_gpus or self.num_gpus_per_node - self.actor_num_gpus_per_node
        else:
            self.actor_num_gpus_per_node = self.actor_num_gpus_per_node or self.num_gpus_per_node
            self.rollout_num_gpus = self.rollout_num_gpus or self.num_gpus_per_node

        assert (
            sum((self.rollout_fp8, self.rollout_mxfp8, self.rollout_int4, self.rollout_nvfp4)) <= 1
        ), "only one rollout precision mode can be enabled"
        assert (
            sum((self.train_fp8, self.train_mxfp8, self.train_nvfp4)) <= 1
        ), "only one train precision mode can be enabled"
        if any((self.rollout_mxfp8, self.rollout_nvfp4, self.train_mxfp8, self.train_nvfp4)):
            assert self.hardware in ("B200", "B300", "GB200", "GB300"), "mxfp8 and nvfp4 only support Blackwell GPUs"


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Downloads and converts everything needed before training. Side effects (all
    # via the backend's shell runners): create model and data directories; download the HF model
    # and the dapo-math-17k and aime-2024 datasets; if rollout fp8, download the prebuilt FP8
    # model; if rollout mxfp8, nvfp4 or int4, run the matching conversion tool on GPU into a
    # sibling directory with the precision suffix (extra args are passed to the mxfp8 and nvfp4
    # converters only). The nvfp4 conversion is prefixed with environment assignments that disable
    # fast math, merged with any NVTE, FLASHINFER or TRTLLM fast-math variables already in the
    # caller's environment (the caller's values win). Unless megatron bridge is enabled, converts
    # the HF checkpoint to Megatron torch_dist format into the shared model directory.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download Qwen/{args.model_name} --local-dir {args.model_dir}/{args.model_name}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)

    if args.rollout_fp8:
        U.exec_command_cpu(
            f"hf download Qwen/{args.model_name}-FP8 --local-dir {args.model_dir}/{args.model_name}-FP8"
        )

    if args.rollout_mxfp8:
        U.exec_command_gpu(
            f"python tools/convert_hf_to_mxfp8.py --model-dir {args.model_dir}/{args.model_name} "
            f"--save-dir {args.model_dir}/{args.model_name}-MXFP8 "
            f"{args.extra_args} "
        )

    if args.rollout_nvfp4:
        nvfp4_env_vars = {
            "NVTE_USE_FAST_MATH": "0",
            "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
            "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
            **{
                key: value
                for key, value in os.environ.items()
                if "NVTE" in key or "FLASHINFER" in key or key == "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH"
            },
        }
        nvfp4_env_prefix = " ".join(f"{key}={value}" for key, value in nvfp4_env_vars.items()) + " "
        U.exec_command_gpu(
            f"{nvfp4_env_prefix}"
            f"python tools/convert_hf_to_nvfp4.py --model-dir {args.model_dir}/{args.model_name} "
            f"--save-dir {args.model_dir}/{args.model_name}-NVFP4 "
            f"{args.extra_args} "
        )

    if args.rollout_int4:
        U.exec_command_gpu(
            f"python tools/convert_hf_to_int4_direct.py --model-dir {args.model_dir}/{args.model_name} --save-dir {args.model_dir}/{args.model_name}-INT4"
        )

    if not args.enable_megatron_bridge:
        U.convert_checkpoint(
            model_name=args.model_name,
            megatron_model_type=args.megatron_model_type,
            num_gpus_per_node=args.num_gpus_per_node,
            # To support multi-node training, for simplicity, we put model into shared folder
            dir_dst=args.model_dir,
            hf_checkpoint=f"{args.model_dir}/{args.model_name}",
            megatron_path=args.megatron_path,
        )


# TODO improve layering: split algorithm vs infra
def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string and extra environment variables, then
    # submits the job through the backend. Raises not-implemented error for unsupported
    # (hardware, node count) combinations: supported are H100 with 1 node, Blackwell parts with 1, 2
    # or 4 nodes, and Rubin with 1 node.
    # Checkpoint selection (first match): rollout fp8 gives the FP8 dir, train mxfp8 gives the
    # MXFP8 dir, rollout nvfp4 the NVFP4 dir, rollout int4 the INT4 dir, else the plain HF dir.
    # Reference load is the HF dir in bridge mode, else the torch_dist dir; load and save use a
    # per-run checkpoints folder. Save interval is 2 in debug mode else 20.
    # Rollout: dapo-math-17k, deepscaler reward, 3000 rollouts, 32 prompts x 8 samples, response
    # max 100 (debug) or 8192, global batch 256. Eval (AIME, 16 samples, 16384 tokens, top-p 1,
    # every 20) only when not debug and enable_eval. Algorithm: grpo with kl loss (coef zero),
    # clip 0.2/0.28. Perf: full recompute, dynamic batch, 32768 tokens per GPU.
    # Misc: fault tolerance, dump details into run dir; colocated runs add colocate flag, others
    # pass actor and rollout GPU counts. Int4 rollout sets fake-QAT environment variables (group
    # size 128) and disables gradient accumulation fusion. Train fp8/mxfp8 uses transformer engine
    # bf16 with the mxfp8 recipe on Blackwell/Rubin or the blockwise recipe (with an fp32 scale
    # variable) on H100/H200. Train nvfp4 adds fp4 flags, several NVTE environment variables, CPU
    # offloaded precision-aware optimizer, and a YAML precision config passed as a pseudo-file
    # (routed expert linear layers in nvfp4, everything else bf16). Bridge mode adds the bridge
    # flag. Mismatch correction (mis) passes a pseudo-file config (token-level truncated TIS with
    # bounds 0.5 to 2.0, optional rejection sampling, veto threshold 1e-4, batch normalize) plus
    # the custom weights function path.
    # Parallelism and sglang by case: H100 1 node uses TP4, EP8, CPU offloaded optimizer, engine
    # size 2 (fp8), 1 (int4) or 8. Blackwell/Rubin: TP is min(4, actor GPUs per node), EP equals
    # actor GPUs when not colocated, else the node GPU count for train mxfp8 or 4; trtllm_mha
    # attention; Rubin uses the triton moe runner; rollout fp8, mxfp8, nvfp4 each set engine size,
    # backends, max running requests and CUDA graph batch size (256); nvfp4 also sets extra
    # environment variables and copies NVTE/FLASHINFER variables from the caller; otherwise engine
    # size 4 and graph batch 512. Attention fp8 rollout adds an fp8 KV cache.
    # KNOWN ISSUE: the H200 case in the train fp8 hardware match is unreachable because the
    # hardware field does not allow H200. Also, the checkpoint chosen for train mxfp8 is the MXFP8
    # directory, which prepare only creates when rollout mxfp8 is set.
    U = args.create_backend()
    ref_load_path = (
        f"{args.model_dir}/{args.model_name}/"
        if args.enable_megatron_bridge
        else f"{args.model_dir}/{args.model_name}_torch_dist"
    )
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"

    if args.rollout_fp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-FP8"
    elif args.train_mxfp8:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-MXFP8"
    elif args.rollout_nvfp4:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-NVFP4"
    elif args.rollout_int4:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}-INT4"
    else:
        hf_checkpoint = f"{args.model_dir}/{args.model_name}"
    ckpt_args = (
        f"--hf-checkpoint {hf_checkpoint}/ "
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
        # need to comment this when using model with MLA
        "--attention-backend flash "
        f"--actor-num-nodes {args.num_nodes} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    if args.no_colocate:
        misc_args += (
            f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} " f"--rollout-num-gpus {args.rollout_num_gpus} "
        )
    else:
        misc_args += f"--actor-num-gpus-per-node {args.num_gpus_per_node} " "--colocate "
    misc_env_vars = {}

    if args.rollout_int4:
        misc_env_vars |= {
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
        }
        # Fake QAT swaps in straight-through weight tensors, while TE's fused wgrad
        # accumulation writes main_grad onto the original ones, so the two together
        # would drop the quantized weights' gradients.
        misc_args += "--no-gradient-accumulation-fusion "

    if args.train_fp8 or args.train_mxfp8:
        match args.hardware:
            case "B200" | "B300" | "GB200" | "GB300" | "Rubin":
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe mxfp8 "
                    # "--fp8-param-gather "
                    # "--reuse-grad-buf-for-mxfp8-param-ag "
                    # --moe-router-padding-for-quantization
                )
            case "H100" | "H200":
                # ref: fp8 blog
                misc_args += (
                    "--transformer-impl transformer_engine "
                    "--bf16 "
                    "--fp8-format e4m3 "
                    "--fp8-recipe blockwise "
                    # "--fp8-param-gather "
                )
                misc_env_vars |= {
                    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
                }
    elif args.train_nvfp4:
        match args.hardware:
            case "B200" | "B300" | "GB200" | "GB300":
                misc_args += (
                    "--transformer-impl transformer_engine " "--bf16 " "--fp4-format e2m1 " "--fp4-recipe nvfp4 "
                )
        misc_env_vars |= {
            "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
            "NVTE_NVFP4_DISABLE_RHT": "1",
            "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
            "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
            "NVTE_BACKWARD_OVERRIDE": "high_precision",
            "NVTE_USE_FAST_MATH": "0",
        }
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
        te_precision_config_text = """
configs:
    nvfp4:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe:
            fp4_quantization_recipe: "nvfp4"
    bf16:
        transformer_engine_config_type: "TEQuantizationParams"
        training_recipe: {}
matchers:
    routed_experts_fc1_nvfp4:
        type: "glob"
        enabled: true
        pattern: "*.mlp.experts.linear_fc1"
        config: "nvfp4"
    routed_experts_fc2_nvfp4:
        type: "glob"
        enabled: true
        pattern: "*.mlp.experts.linear_fc2"
        config: "nvfp4"
    default_bf16:
        type: "glob"
        enabled: true
        pattern: "*"
        config: "bf16"
""".strip()
        misc_args += f"--te-precision-config-file {command_utils.encode_pseudo_file(te_precision_config_text)} "

    if args.enable_megatron_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

    match (args.hardware, args.num_nodes):
        case ("H100", 1):
            perf_args += (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
            sglang_args = (
                f"--rollout-num-gpus-per-engine {2 if args.rollout_fp8 else 1 if args.rollout_int4 else 8} "
                "--sglang-mem-fraction-static 0.7 "
                "--sglang-cuda-graph-max-bs-decode 512 "
            )
            optimizer_args += (
                "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
            )
        case ("B200" | "B300" | "GB200" | "GB300", 1 | 2 | 4) | ("Rubin", 1):
            perf_args += (
                f"--tensor-model-parallel-size {min(4, args.actor_num_gpus_per_node)} "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 1 "
                "--context-parallel-size 1 "
                "--expert-tensor-parallel-size 1 "
            )
            if args.no_colocate:
                perf_args += f"--expert-model-parallel-size {args.actor_num_gpus_per_node} "
            else:
                perf_args += f"--expert-model-parallel-size {args.num_gpus_per_node if args.train_mxfp8 else 4} "
            sglang_args = "--sglang-mem-fraction-static 0.7 " "--sglang-attention-backend trtllm_mha "
            if args.hardware == "Rubin":
                sglang_args += "--sglang-moe-runner-backend triton "
            if args.rollout_fp8:
                sglang_world_size = 2
                sglang_attn_tp_size = 2
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--rollout-num-gpus-per-engine 2 "
                    f"--sglang-ep-size {sglang_world_size} "
                    "--sglang-moe-runner-backend deep_gemm "
                    "--sglang-moe-a2a-backend deepep "
                    f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
                )
            elif args.rollout_mxfp8:
                sglang_world_size = 1
                sglang_attn_tp_size = 1
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--rollout-num-gpus-per-engine 1 "
                    "--sglang-fp8-gemm-backend triton "
                    # Currently, only cutlass moe runner is supported in sglang for mxfp8, which does not support ep
                    # f"--sglang-ep-size {sglang_world_size} "
                    "--sglang-moe-runner-backend cutlass "
                    # TODO: mxfp8 deepep and deepgemm is not supported in sglang yet
                    # "--sglang-moe-a2a-backend deepep "
                    f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
                    f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
                    f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
                )
            elif args.rollout_nvfp4:
                sglang_world_size = 2
                sglang_decode_max_bs = 256
                sglang_args += (
                    f"--rollout-num-gpus-per-engine {sglang_world_size} "
                    "--sglang-moe-runner-backend flashinfer_trtllm_routed "
                    f"--sglang-tp-size {sglang_world_size} "
                    f"--sglang-ep-size {sglang_world_size} "
                    f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
                    "--sglang-kv-cache-dtype bf16 "
                )
                misc_env_vars |= {
                    "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
                    "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
                    "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
                }
                misc_env_vars |= {
                    key: value
                    for key, value in os.environ.items()
                    if "NVTE" in key or "FLASHINFER" in key or key == "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH"
                }
            else:
                sglang_args += "--rollout-num-gpus-per-engine 4 " "--sglang-cuda-graph-max-bs-decode 512 "
        case _:
            raise NotImplementedError

    if args.rollout_attn_fp8:
        sglang_args += "--sglang-kv-cache-dtype fp8_e4m3 "

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
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command-line entry point: runs prepare, then execute, with the parsed args.
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
