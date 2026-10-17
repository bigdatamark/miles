# [MODULE DOC]
# Launcher script (typer command line app) for GRPO training of GLM-5.2 (744B total / 40B active
# parameters; mixture of experts with latent attention and sparse attention using cross-layer index
# sharing) or its 5-layer prune, with Megatron training and SGLang rollout colocated on all GPUs, on
# DAPO math. Commands: full-train (download, validate, optional FP8 cast, Megatron conversion, train),
# prepare (same without training) and train. Supports FP8 block-quantized rollout checkpoint, DeepEP,
# EAGLE speculative decoding, prefill/decode disaggregated rollout, two SGLang engine recipes, optional
# optimizer offload, and a hardware-dependent parallel layout (single-node test, 16+ nodes of 4 GPUs,
# or 16+ nodes otherwise). Important model-specific constraint: with sparse-attention index sharing,
# every pipeline stage must start on a layer that computes its own index, which dictates the
# first/last pipeline stage layer counts used throughout.
"""
GLM-5.2 744B-A40B Training Script

=====================

Tested on H200, B200, GB300

Please use the `radixark/miles:dev` docker image.

GLM-5.2 shares GLM-5's glm_moe_dsa architecture (DSA + cross-layer index sharing);
this script differs from run_glm5_744b_a40b.py only in the checkpoint it loads, the
glm5.2 megatron model args (rotary-base 8000000), and the GLM-5.2 rollout recipe
(FP8 KV cache, flashmla_kv decode, 4-step EAGLE). Config follows the upstream slime
`run-glm5.2-744B-A40B.sh` recipe.

=====================

Args:
  --model-name: Model variant to use.
      GLM-5.2        Full 744B model (requires >=16 nodes)
      GLM-5.2_5layer 5-layer pruned model (single-node testing)
  --num-nodes: Number of nodes for training. Determines parallelism config:
      1   -> for GLM-5.2_5layer minimal test
      16+ -> for full GLM-5.2 model
  --num-gpus-per-node: GPUs per node (default: 8)
  --mode: "normal" or "debug_minimal" (shorter response length for quick testing)
  --fp8-rollout: Enable FP8 rollout (converts HF checkpoint to FP8 block quant for sglang; megatron still uses bf16)
  --enable-eval: Enable evaluation every 20 steps
  --enable-mtp: Enable multi-token prediction (EAGLE speculative decoding; full model only, MTP layer is pruned away)
  --enable-optimizer-offload: Offload optimizer to CPU
  --data-dir: Directory for datasets (default: /root/datasets, shared NFS)
  --model-dir: Directory for model weights and converted checkpoints (default: /root/models, shared NFS)
  --model-local-dir: Node-local directory for model copies (default: /root/local_data, local disk)

=====================

I. Usage for single node minimal test:
  `python scripts/run_glm5_2_744b_a40b.py full-train --model-name GLM-5.2_5layer --num-nodes 1`

=====================

II. Usage for full model (>=16 nodes):

  1. Setup containers on all nodes

  2. Start Ray cluster on all nodes

  3. Download model/data + validate checkpoint + convert to megatron (run on head node).
       `python scripts/run_glm5_2_744b_a40b.py prepare --model-name GLM-5.2 --num-nodes 32 --fp8-rollout`

  4. Run training; it first copies the model from shared NFS to each node's local disk.
     Execute on head node; uses Ray internally for distributed training.
       python scripts/run_glm5_2_744b_a40b.py train --model-name GLM-5.2 --num-nodes 32 --fp8-rollout --enable-mtp
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: mode (debug_minimal gives response length 100 and no eval), run id, Hugging
    # Face org/name (name is "GLM-5.2" or "GLM-5.2_5layer"; org derived), Megatron model type (derived
    # for pruned names), GPUs per node (None derived from hardware), fp8_rollout, use_deepep (SGLang)
    # and megatron_use_deepep, eval switch, MTP switch, enable_pd (prefill/decode disaggregation, on by
    # default), optimizer offload, sglang_config ("low-latency" or "balanced", see inline comment),
    # rollout sizes (3000 rollouts, 8 prompts x 8 samples, global batch 64), extra raw args,
    # directories (model_local_dir is the destination of the trainer copy step), hardware.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "zai-org"
    model_name: str = "GLM-5.2"
    megatron_model_type: str = "glm5.2-744B-A40B"
    num_gpus_per_node: int | None = None
    fp8_rollout: bool = False
    use_deepep: bool = True
    megatron_use_deepep: bool = True
    enable_eval: bool = False
    enable_mtp: bool = False
    enable_pd: bool = True
    enable_optimizer_offload: bool = False
    # Engine recipe. "low-latency": one TP8 engine per node pair, EAGLE 5/1/6.
    # "balanced": the GLM-5.2 cookbook serving shape -- one 4-GPU engine per
    # node with dp-attention + deepep and EAGLE 1/1/2.
    sglang_config: Literal["low-latency", "balanced"] = "low-latency"
    num_rollout: int = 3000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"
    hardware: Literal["auto", "H200", "B200", "GB300"] = "auto"

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves hardware and GPUs per node. On GB300, asserts Megatron DeepEP is off (known failure).
        # If SGLang DeepEP is off, forces Megatron DeepEP off. On a single node, disables prefill/decode
        # disaggregation and forces debug_minimal mode. For the "balanced" recipe, asserts no
        # disaggregation and that GPUs per node is divisible by 4. If the model name has a layer-count
        # suffix, sets the Megatron model type to the pruned variant. Sets the org to zai-org for the
        # full model or Pinaster for the 5-layer model; any other name raises a not implemented error.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if self.hardware == "GB300":
            assert not self.megatron_use_deepep, (
                "Known issue: Megatron's DeepEP fail on GB300. " "Please specify --no-megatron-use-deepep."
            )
        if not self.use_deepep:
            self.megatron_use_deepep = False
        if self.num_nodes == 1:
            self.enable_pd = False
            self.mode = "debug_minimal"
        if self.sglang_config == "balanced":
            assert not self.enable_pd, "balanced is a single-engine shape; PD disaggregation needs low-latency"
            assert self.num_gpus_per_node % 4 == 0, "balanced places one 4-GPU engine per node"

        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.megatron_model_type = f"glm5.2-744B-A40B_{m.group(1)}layer"

        if self.model_name == "GLM-5.2":
            self.model_org = "zai-org"
        elif self.model_name == "GLM-5.2_5layer":
            self.model_org = "Pinaster"
        else:
            raise NotImplementedError(f"{self.model_name} is not supported")


def _is_pruned(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: True when the model name contains a number followed by "layer".
    return re.search(r"(\d+)layer", args.model_name) is not None


def _validate_glm_checkpoint(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Reads the downloaded checkpoint's config file in the model directory.
    # Raises file-not-found if missing. Expected layer count is 78 for the full model or the number in
    # the name for pruned ones (other names raise not implemented). Raises a runtime error unless the
    # config has the native sparse-attention MoE model type, architecture list with the single causal
    # LM class, and the expected layer count; also raises if the config has a custom class mapping entry
    # (meaning the checkpoint is a non-native variant to be updated).
    """Validate the basic native GLM-5.2 config fields."""
    config_path = Path(args.model_dir) / args.model_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path) as f:
        config = json.load(f)

    if args.model_name == "GLM-5.2":
        expected_num_layers = 78
    elif (m := re.search(r"(\d+)layer", args.model_name)) is not None:
        expected_num_layers = int(m.group(1))
    else:
        raise NotImplementedError(f"{args.model_name} is not supported")

    if (
        config.get("model_type") != "glm_moe_dsa"
        or config.get("architectures") != ["GlmMoeDsaForCausalLM"]
        or config.get("num_hidden_layers") != expected_num_layers
    ):
        raise RuntimeError(
            f"{config_path} must use native GLM-5.2 config with "
            f"model_type=glm_moe_dsa, architectures=[GlmMoeDsaForCausalLM], "
            f"and num_hidden_layers={expected_num_layers}"
        )
    if "auto_map" in config:
        raise RuntimeError(f"{config_path} must not contain auto_map. Try update your checkpoint.")


def _convert_to_fp8(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Idempotent FP8 cast. Skips (printing a notice) if the "_fp8" output folder
    # already contains the safetensors index file. Otherwise runs the Hugging Face to FP8 tool as a GPU
    # command with block quantization 128 by 128 and 16 workers, from the model folder to the output
    # folder. The policy side stays bf16.
    """Convert HF checkpoint to FP8 (block quantization). Megatron still uses bf16."""
    U = args.create_backend()
    src = f"{args.model_dir}/{args.model_name}"
    dst = f"{args.model_dir}/{args.model_name}_fp8"
    sentinel = Path(dst) / "model.safetensors.index.json"
    if sentinel.exists():
        print(f"_convert_to_fp8 skip {dst} since {sentinel} exists")
        return
    U.exec_command_gpu(
        f"python tools/convert_hf_to_fp8.py "
        f"--model-dir {src} --save-dir {dst} "
        f"--strategy block --block-size 128 128 "
        f"--max-workers 16"
    )


def _prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Creates model/data directories, downloads the Hugging Face model into the
    # model directory and the DAPO math dataset into the data directory.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_megatron_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Converts the Hugging Face checkpoint to a Megatron distributed checkpoint
    # in the model directory. Tensor and expert-tensor parallel always 1. For pruned models with at most
    # 5 layers: single GPU, single node, pipeline 1, expert 1 (avoids the converter's automatic
    # pipeline bump that would split onto a layer that skips index computation). Otherwise a multi-node
    # job across all nodes with pipeline 4, expert = total GPUs / 4, and 18 / 20 layers on the first /
    # last pipeline stages. The distributed format is parallelism-agnostic on load.
    U = args.create_backend()
    extra_args = "--tensor-model-parallel-size 1 " "--expert-tensor-parallel-size 1 "
    num_gpus_per_node = args.num_gpus_per_node
    multinode = True
    num_nodes = None

    num_layers_match = re.search(r"(\d+)layer", args.model_name)
    if num_layers_match and int(num_layers_match.group(1)) <= 5:
        # Pruned model converts on a single GPU: EP=1 holds all experts, and PP=1 keeps
        # the one pipeline stage starting on a computing layer (DSA cross-layer index
        # sharing forbids a stage starting on a skip layer). nproc=1 also avoids the
        # convert tool's PP auto-bump (which would split onto a skip layer).
        extra_args += "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        num_gpus_per_node = 1
        multinode = False
    else:
        # torch_dist is parallelism-agnostic on load, so EP can follow the
        # hardware rather than the training config.
        world_size = args.num_nodes * args.num_gpus_per_node
        extra_args += (
            "--pipeline-model-parallel-size 4 "
            f"--expert-model-parallel-size {world_size // 4} "
            "--decoder-first-pipeline-num-layers 18 "
            "--decoder-last-pipeline-num-layers 20 "
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def _prepare_cmd(args: ScriptArgs) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns a mapping with a "trainer" entry joining, in sequence, rsync copies
    # of the Megatron distributed checkpoint and the rollout Hugging Face folder ("_fp8" suffixed when
    # FP8 rollout) from the shared model directory to the local model directory.
    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
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
    # Checkpoint: hf checkpoint from the local model directory (FP8 folder when FP8 rollout), reference
    # load from local distributed folder, load/save under the output directory every 20 steps.
    # Rollout: DAPO math, deepscaler reward, chat template, shuffled, response length 100 (debug) or
    # 32768, temperature 1, sizes from options. Eval (non-debug and enabled): every 20 steps, top-p 1
    # (no eval dataset is passed here). Parallelism by cluster: one node (test) tensor 4, sequence
    # parallel, expert = GPUs; 16+ nodes of 4 GPUs: tensor 8, pipeline 4 (18/20 edge layers), expert 16;
    # 16+ nodes otherwise: tensor 4, pipeline 8 (14/16 edge layers), context 8, expert 32; anything else
    # raises not implemented. Then full uniform recompute, dynamic batch (2048 tokens per GPU for
    # pruned models else 8192), padding multiple 1024, log-prob chunk 16384. GRPO with zero kl/entropy,
    # asymmetric clip, truncated importance sampling bounds 0.5 to 2.0. Adam 1e-6, CPU offload group if
    # enabled. SGLang engine size: with disaggregation, decode batch 8 and 16 GPUs (under 16 nodes) or
    # 64; without, decode batch 32 and 4 GPUs (balanced) or min(8, GPUs per node). Memory fraction 0.70
    # on a single node else 0.85 (values justified by inline measurements), expert parallel = engine
    # size, consistent-hashing router; disaggregation adds data-parallel attention, dense TP 1, DP LM
    # head and one prefill server; the balanced recipe adds data-parallel attention; DeepEP all-to-all
    # for balanced or FP8+DeepEP (with auto mode for FP8); MTP adds EAGLE (balanced 1 step/2 draft
    # tokens, low-latency 5/6) with the sparse attention draft backend; FP8 KV cache, flashmla decode and
    # sparse prefill backends, page size 64, cuda graph and running-request limits, 3600 s watchdog; on
    # B200/GB300 for non-balanced non-FP8-DeepEP runs, the moe backend is flashinfer routed (FP8) or
    # triton (bf16). Environment variables: DeepEP dispatch token limit (64 with disaggregation else
    # 256), forced MLA, indexer rope style 0, NVSHMEM over NCCL disabled. Misc: dropouts off, fp32
    # accumulation, flash attention, all-gather context-parallel split, 2 GiB weight buffer, colocate,
    # rematerialize parameters from master weights, Megatron DeepEP with flex dispatcher or all-to-all
    # dispatcher. Submits via the backend with the per-role copy commands.
    # KNOWN ISSUE: the code comment "need to comment this when using model with MLA" sits beside a
    # flash attention flag for an MLA model (stale comment).
    # KNOWN ISSUE: eval is enabled via interval/top-p only; no eval prompt data flag is provided, so
    # eval relies on extra args or defaults.
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    hf_name = f"{args.model_name}_fp8" if args.fp8_rollout else args.model_name
    ckpt_args = (
        f"--hf-checkpoint {args.model_local_dir}/{hf_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
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
        f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
    )

    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 1 "

    if args.num_nodes == 1:  # minimal test for the 5-layer model
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {args.num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes >= 16 and args.num_gpus_per_node == 4:  # GB300 64-GPU config
        # TP=8 * PP=4 * DP=2 = 64 GPUs. EP=16 is what fits the fp32 optimizer
        # states on 276GB GPUs; 18/20 puts the DSA stage starts (see below) on
        # computing layers 1,19,39,59.
        perf_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 18 "
            "--decoder-last-pipeline-num-layers 20 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes >= 16:  # slime's setting for the full model
        # TP=4 * PP=8 * CP=8 = 256 GPUs (32 nodes) for one training group; EP=32.
        # DSA cross-layer index sharing needs every pipeline stage to START on a
        # computing layer (index_topk_freq=4, index_skip_topk_offset=3 -> computing
        # layers 1,2,3,7,11,...,75). first=14, last=16 leaves 6 middle stages of 8;
        # stage starts land on global layers 1,15,23,31,39,47,55,63 -- all computing.
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 8 "
            "--decoder-first-pipeline-num-layers 14 "
            "--decoder-last-pipeline-num-layers 16 "
            "--context-parallel-size 8 "
            "--expert-model-parallel-size 32 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        raise NotImplementedError

    perf_args += (
        # ------------
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if _is_pruned(args) else 8192} "
        "--data-pad-size-multiplier 1024 "
        "--log-probs-chunk-size 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        # GLM-5.2 recipe uses truncated importance sampling
        "--use-tis "
        "--tis-clip-low 0.5 "
        "--tis-clip 2.0 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if args.enable_optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    balanced = args.sglang_config == "balanced"
    if args.enable_pd:
        sglang_decode_max_bs = 8
        if args.num_nodes < 16:
            sglang_world_size = 16
        else:
            sglang_world_size = 64
    else:
        sglang_decode_max_bs = 32
        sglang_world_size = 4 if balanced else min(8, args.num_gpus_per_node)

    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        # 0.85: measured on the full 744B, 64x GB300 (276GB). 0.75 there leaves
        # ~59GB per GPU unused and the KV pool at 26k tokens, which caps
        # trajectories and starves concurrency; 0.85 gives 553k tokens and still
        # ends steady state with 33GB spare.
        # 0.70: the value the 5-layer smoke test passes with on 4x H200 (140GB).
        # Pruned weights make 0.85 nearly all KV cache there, leaving the
        # weight-checker snapshot nowhere to allocate.
        f"--sglang-mem-fraction-static {0.70 if args.num_nodes == 1 else 0.85} "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-router-policy consistent_hashing "
    )
    if args.enable_pd:
        # slime native config
        sglang_args += (
            "--sglang-enable-dp-attention "
            f"--sglang-dp-size {sglang_world_size} "
            "--sglang-moe-dense-tp-size 1 "
            "--sglang-enable-dp-lm-head "
        )
    elif balanced:
        # dp-attention implies dp-aware routing (set in sglang_utils.arguments):
        # min_load must see the dp ranks, or requests pile onto whichever rank
        # sglang picks internally while the others sit idle.
        sglang_args += "--sglang-enable-dp-attention " f"--sglang-dp-size {sglang_world_size} "
    if balanced or (args.fp8_rollout and args.use_deepep):
        sglang_args += "--sglang-moe-a2a-backend deepep "
    if args.fp8_rollout and args.use_deepep:
        sglang_args += "--sglang-deepep-mode auto "
    if args.enable_mtp:
        # balanced follows the cookbook's serving depth; low-latency drafts
        # deeper, which pays off only when the engine is not already busy.
        steps, draft_tokens = (1, 2) if balanced else (5, 6)
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            f"--sglang-speculative-num-steps {steps} "
            "--sglang-speculative-eagle-topk 1 "
            f"--sglang-speculative-num-draft-tokens {draft_tokens} "
            "--sglang-speculative-draft-attention-backend nsa "
        )
    if args.enable_pd:
        sglang_args += "--prefill-num-servers 1 "
    sglang_args += (
        "--sglang-kv-cache-dtype fp8_e4m3 "
        # flashmla_kv decode / flashmla_sparse prefill (GLM-5.2 recipe)
        "--sglang-dsa-decode-backend flashmla_kv "
        "--sglang-dsa-prefill-backend flashmla_sparse "
        "--sglang-attention-backend nsa "
        "--sglang-page-size 64 "
        f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
        # concurrency
        f"--sglang-max-running-requests {256 if balanced else 512} "
        f"--sglang-chunked-prefill-size {32768 if balanced else 2048 * sglang_world_size} "
        "--sglang-watchdog-timeout 3600 "
    )
    if args.hardware in ("B200", "GB300") and not balanced and not (args.fp8_rollout and args.use_deepep):
        # TODO: fix bf16 trtllm weight update
        if args.fp8_rollout:
            sglang_args += "--sglang-moe-runner-backend flashinfer_trtllm_routed "
        else:
            sglang_args += "--sglang-moe-runner-backend triton "
    sglang_extra_env_vars = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": f"{64 if args.enable_pd else 256}",
        "SGLANG_NSA_FORCE_MLA": "1",
    }

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        # DSA + context parallel uses the sequential allgather-CP layout; the
        # index-share provider gathers index_k/kv across the CP group to match.
        "--allgather-cp "
        # ------------
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--rematerialize-param-from-master-weight "
    )

    if args.megatron_use_deepep:
        misc_args += "--moe-enable-deepep " "--moe-token-dispatcher-type flex "
    else:
        misc_args += "--moe-token-dispatcher-type alltoall "

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
            **sglang_extra_env_vars,
            "INDEXER_ROPE_NEOX_STYLE": "0",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
        megatron_path=args.megatron_path,
        prepare_cmd=_prepare_cmd(args),
    )


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs):
    # [FUNCTION DOC] Command "full-train": download, validate checkpoint, optional FP8 cast, Megatron
    # conversion, then training.
    # KNOWN ISSUE: the module docstring says the model-local-dir default is the local data folder, but
    # the actual default equals the shared model directory.
    """Full pipeline: download, convert, copy, train."""
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)
    _execute_train(args)


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Command "prepare": same as full-train but without training (run on head node).
    """Download model/data and convert to megatron checkpoint (run on head node)."""
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)


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
