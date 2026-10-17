# [MODULE DOC] ROCm launcher for GLM-5.2 pruned slices (default 5 layers: 3 dense plus 2 MoE) on
# a single node, with typer commands full-train, prepare (download, validate, optional FP8
# conversion, Megatron conversion) and train. Training args are grouped strings submitted through
# the shared command utilities; sglang uses the NSA / tilelang DSA attention backends.
"""
GLM-5.2 training script for ROCm (GLM-5.2 5-layer slice).

Supports:
  - GLM-5.2_5layer  5-layer prune of GLM-5.2 (3 dense + 2 MoE) for the 4-GPU CI lane.

Usage patterns:

  1. One-shot pipeline (download + convert + train):
       python scripts/amd/run_glm5_2_744b_a40b.py full-train \
           --model-name GLM-5.2_5layer --num-nodes 1 --num-gpus-per-node 4

  2. Individual steps (download/convert -> train):
       python scripts/amd/run_glm5_2_744b_a40b.py prepare --model-name GLM-5.2_5layer
       python scripts/amd/run_glm5_2_744b_a40b.py train \
           --model-name GLM-5.2_5layer --num-nodes 1 --num-gpus-per-node 4
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # [CLASS DOC] Configuration: mode, run id, HF org (default Pinaster) and model name (must
    # contain the layer count, e.g. a 5layer name), Megatron model type (overwritten from the name),
    # GPUs per node (default 4), FP8 rollout and DeepEP options, optional optimizer CPU offload,
    # rollout sizes, global batch size, extra args and directories.
    # KNOWN ISSUE: the megatron_model_type field default is always overridden in the post-init, so
    # passing it on the command line has no effect; run_id default is computed at import time.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = U.create_run_id()
    model_org: str = "Pinaster"
    model_name: str = "GLM-5.2_5layer"
    megatron_model_type: str = "glm5.2-744B-A40B_5layer"
    num_gpus_per_node: int = 4
    fp8_rollout: bool = False
    use_deepep: bool = True
    enable_optimizer_offload: bool = False
    num_rollout: int = 3000
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        # [FUNCTION DOC] Forces debug_minimal mode on a single node. Extracts the layer count from
        # the model name (digits before the word layer) and sets the Megatron model type to the
        # 5.2 744B pattern with that count; raises not-implemented error if the name has none.
        if self.num_nodes == 1:
            self.mode = "debug_minimal"

        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.megatron_model_type = f"glm5.2-744B-A40B_{m.group(1)}layer"
        else:
            raise NotImplementedError(f"{self.model_name} is not supported")


def _validate_glm_checkpoint(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: read the model's config.json (file-not-found error if missing) and
    # require model type glm_moe_dsa, the single architecture GlmMoeDsaForCausalLM and a hidden
    # layer count equal to the number in the model name (not-implemented error if the name has no
    # count); also forbids an auto_map entry. Raises runtime errors with explanatory messages.
    """Validate the basic native GLM-5.2 config fields."""
    config_path = Path(args.model_dir) / args.model_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path) as f:
        config = json.load(f)

    if (m := re.search(r"(\d+)layer", args.model_name)) is not None:
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
    # [FUNCTION DOC] HELPER: skip (printing a notice) if the destination (model dir / name plus
    # _fp8) already has a safetensors index; otherwise run the repo's conversion tool on a GPU with
    # block strategy, 128 by 128 blocks and 16 workers.
    """Convert HF checkpoint to FP8 (block quantization). Megatron still uses bf16."""
    src = f"{args.model_dir}/{args.model_name}"
    dst = f"{args.model_dir}/{args.model_name}_fp8"
    sentinel = Path(dst) / "model.safetensors.index.json"
    if sentinel.exists():
        print(f"_convert_to_fp8 skip {dst} since {sentinel} exists")
        return
    args.create_backend().exec_command_gpu(
        f"python tools/convert_hf_to_fp8.py "
        f"--model-dir {src} --save-dir {dst} "
        f"--strategy block --block-size 128 128 "
        f"--max-workers 16"
    )


def _prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: create model and data dirs, download the HF model from the org and
    # download the dapo math 17k dataset.
    backend = args.create_backend()
    backend.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    backend.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    backend.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)


def _prepare_megatron_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: convert the HF checkpoint to Megatron format on one GPU, single node,
    # with TP, EP and PP all 1 (EP 1 holds all experts; PP 1 keeps the one stage starting on a
    # computing layer because cross-layer index sharing forbids starting on a skip layer).
    # Pruned model converts on a single GPU: EP=1 holds all experts, and PP=1 keeps
    # the one pipeline stage starting on a computing layer (DSA cross-layer index
    # sharing forbids a stage starting on a skip layer). nproc=1 also avoids the
    # convert tool's PP auto-bump (which would split onto a skip layer).
    extra_args = (
        "--tensor-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--expert-model-parallel-size 1 "
    )

    args.create_backend().convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=1,
        multinode=False,
        num_nodes=None,
        extra_args=extra_args,
        dir_dst=args.model_dir,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}",
        megatron_path=args.megatron_path,
    )


def _execute_train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: assemble and submit the training args. Checkpoint: HF path (the fp8
    # copy when fp8 rollout), ref-load from the torch_dist dir, load / save under output dir / run
    # id, save interval 20. Rollout: dapo math, deepscaler reward, response length 100 in debug mode
    # else 32768, temperature 1. Perf: TP 4 with sequence parallel, PP 1, CP 1, EP = GPUs per node,
    # expert TP 1, full uniform recompute, dynamic batch size at 2048 tokens per GPU. GRPO with
    # truncated importance sampling (clip 0.5 to 2.0). Adam lr 1e-6, optional CPU offload. Sglang:
    # engine size min(8, GPUs), memory fraction 0.70, EP equal to engine size, consistent-hashing
    # router, optional mori all-to-all with fp8 rollout and DeepEP, fp8 KV cache, tilelang DSA
    # backends, NSA attention, page size 64, decode CUDA-graph batch 32, 512 running requests.
    # Misc: dropout off, fp32 grad all-reduce, flash attention backend, allgather-cp, colocate,
    # alltoall MoE dispatcher. Environment: forces MLA in the NSA path and non-neox indexer rope.
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

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {args.num_gpus_per_node} "
        "--expert-tensor-parallel-size 1 "
        # ------------
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
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

    sglang_decode_max_bs = 32
    sglang_world_size = min(8, args.num_gpus_per_node)

    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        # 0.70: the value the 5-layer smoke test passes with on 4x H200 (140GB).
        # Pruned weights make 0.85 nearly all KV cache there, leaving the
        # weight-checker snapshot nowhere to allocate.
        "--sglang-mem-fraction-static 0.70 "
        f"--sglang-ep-size {sglang_world_size} "
        "--sglang-router-policy consistent_hashing "
    )
    if args.fp8_rollout and args.use_deepep:
        sglang_args += "--sglang-moe-a2a-backend mori " "--sglang-deepep-mode auto "
    sglang_args += (
        "--sglang-kv-cache-dtype fp8_e4m3 "
        "--sglang-dsa-decode-backend tilelang "
        "--sglang-dsa-prefill-backend tilelang "
        "--sglang-attention-backend nsa "
        "--sglang-page-size 64 "
        f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
        # concurrency
        "--sglang-max-running-requests 512 "
        f"--sglang-chunked-prefill-size {2048 * sglang_world_size} "
        "--sglang-watchdog-timeout 3600 "
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
        "--moe-token-dispatcher-type alltoall "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            "SGLANG_NSA_FORCE_MLA": "1",
            "INDEXER_ROPE_NEOX_STYLE": "0",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    # [FUNCTION DOC] CLI command: download, validate the config, FP8-convert if fp8 rollout,
    # convert to Megatron, train.
    """Full pipeline: download, convert, train."""
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)
    _execute_train(args)


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    # [FUNCTION DOC] CLI command: the preparation half of full-train (no training).
    """Download model/data and convert to megatron checkpoint."""
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] CLI command: only the training submission.
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: empty typer callback; its presence keeps the command group explicit
    # (subcommand names required) even though only a few commands exist.
    pass


if __name__ == "__main__":
    app()
