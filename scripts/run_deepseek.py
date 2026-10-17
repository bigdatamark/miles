# [MODULE DOC]
# Launcher script (typer command line app) for GRPO reinforcement-learning training of the
# DeepSeek-V3 family (full model, or the 5-layer / 20-layer pruned variants selected by a
# "<N>layer" substring in the model name) on a colocated Megatron + SGLang setup.
# Three commands are exposed: full-train (download, cast to bf16, convert to Megatron
# checkpoint, then train), prepare (the first three steps only) and train (training only).
# All work is delegated to the command utilities backend; this module only assembles shell
# command lines and the long training argument string. Nothing is executed at import time
# other than creating the typer app and one run id (a default field value).
"""
This file is in preview, and will be further refined and optimized.
"""

import re
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options for this launcher, extending the shared train configuration (which
    # supplies node count, output directory, backend creation, etc.).
    # mode: "normal" or "debug_minimal" (debug mode shrinks response length to 100 and drops
    # dynamic sampling and eval). run_id: default is generated once at import time.
    # model_org / model_name / megatron_model_type: Hugging Face org, repo name and Megatron
    # model type key. num_gpus_per_node defaults to 4. enable_eval toggles periodic evaluation.
    # task: "dapo_aime" (DAPO math train data, AIME eval) or "gsm8k". extra_args: raw string
    # appended to the training argument line. data_dir / model_dir: shared storage for datasets
    # and checkpoints; model_local_dir: node-local copy target; megatron_path: Megatron checkout.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "deepseek-ai"
    model_name: str = "DeepSeek-V3"
    megatron_model_type: str = "deepseek-v3"
    num_gpus_per_node: int = 4
    enable_eval: bool = True
    extra_args: str = ""
    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    model_local_dir: str = "/root/local_data"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        # [FUNCTION DOC]
        # If the model name contains a number followed by "layer", switch the model org to the
        # pruned-model publisher "fzyzcjy" and set the Megatron model type to the deepseek-v3
        # type with that layer count suffix. Otherwise leaves all fields unchanged.
        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.model_org = "fzyzcjy"
            self.megatron_model_type = f"deepseek-v3-{m.group(1)}layer"


def _prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Download step. Creates the model and data directories, downloads the
    # Hugging Face model repo into the model directory under the model name, then downloads the
    # datasets for the task (DAPO math plus AIME 2024 for "dapo_aime", GSM8K for "gsm8k").
    # Side effects: runs CPU shell commands and dataset downloads through the backend.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(
        f"hf download {args.model_org}/{args.model_name} --local-dir {args.model_dir}/{args.model_name}"
    )
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _prepare_bf16_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Casts the downloaded fp8 checkpoint to bf16, writing it to a sibling
    # directory named with a "-bf16" suffix under the model directory. Runs via the backend.
    U = args.create_backend()
    U.fp8_cast_bf16(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_dir}/{args.model_name}-bf16/",
    )


def _prepare_megatron_ckpt(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Converts the bf16 Hugging Face checkpoint into a Megatron distributed
    # checkpoint written into the model directory. Parallelism flags depend on the model name:
    # the 5-layer prune uses pipeline 1, expert 1, a single node and at most 4 GPUs; the 20-layer
    # prune uses expert parallel 4 on 2 nodes (pipeline size chosen by the converter); every other
    # model uses pipeline 8, expert 4 with 7 layers in the first and 6 in the last pipeline stage
    # on multiple nodes (node count left unset). Tensor and expert-tensor parallel are always 1.
    # Side effect: runs the conversion job through the backend.
    # TODO unify 5layer w/ 20layer, also maybe unify the whole script
    U = args.create_backend()
    extra_args = "--tensor-model-parallel-size 1 " "--expert-tensor-parallel-size 1 "
    num_gpus_per_node = args.num_gpus_per_node
    multinode = True
    num_nodes = None
    if args.model_name == "DeepSeek-V3-0324-5layer":
        extra_args += "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        num_gpus_per_node = min(4, num_gpus_per_node)
        multinode = False
    elif args.model_name == "DeepSeek-V3-0324-20layer":
        extra_args += (
            "--expert-model-parallel-size 4 "
            # PP info will be auto determined by converter script
        )
        num_nodes = 2
    else:
        extra_args += (
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
        )

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.model_name}-bf16",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_per_node,
        multinode=multinode,
        num_nodes=num_nodes,
        extra_args=extra_args,
        dir_dst=args.model_dir,
        megatron_path=args.megatron_path,
    )


def _prepare_cmd(args: ScriptArgs) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Builds the per-role pre-launch commands. Returns a mapping with one
    # entry, "trainer", whose value is two rsync copy commands (Megatron distributed checkpoint
    # and the Hugging Face model, from the shared model directory to the node-local directory)
    # joined so that the second runs only if the first succeeds. No side effects.
    copies = [
        command_utils.rsync_cmd(
            f"{args.model_dir}/{args.model_name}_torch_dist",
            f"{args.model_local_dir}/{args.model_name}_torch_dist",
        ),
        command_utils.rsync_cmd(f"{args.model_dir}/{args.model_name}", f"{args.model_local_dir}/{args.model_name}"),
    ]
    return {"trainer": " && ".join(copies)}


def _execute_train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Assembles the full training argument string and submits the job.
    # Groups: checkpoint (hf checkpoint and reference load from the local model directory, load and
    # save under the output directory run folder, save every 20 steps), rollout (math reward, 3000
    # rollouts, batch 128 prompts, 8 samples each, temperature 1; in non-debug mode over-sampling
    # of 256 with a nonzero-reward-std dynamic filter; response length 100 in debug mode, 32768 for
    # dapo_aime, 256 for gsm8k), eval (only if not debug and enabled; every 20 steps, top-p 1),
    # parallelism (chosen by node count: up to 2 nodes, up to 4 nodes, or more; the largest tier
    # uses pipeline 1 for the 5-layer model else 4, and adds a last-stage layer count of 13 unless
    # the model name has a layer-count suffix; two tiers are marked as temporary configurations in
    # the code), recompute full-uniform, dynamic batch size with a
    # deliberately tiny 2048 tokens per GPU, GRPO with zero kl and entropy coefficients and
    # asymmetric clipping, Adam at 1e-6, SGLang engine settings (world size 4 up to 4 nodes else
    # 64; data-parallel attention size 1 or 8; expert parallel equal to world size; DeepEP low
    # latency all-to-all; concurrency derived from a decode batch size of 256), and misc flags
    # (dropout off, fp32 gradient accumulation, colocate, fault tolerance, dump details).
    # Submits through the backend with an environment variable for the DeepEP dispatch token limit
    # and the per-role copy commands.
    U = args.create_backend()
    load_save_path = f"{args.output_dir}/{args.run_id}/checkpoints"
    ckpt_args = (
        f"--hf-checkpoint {args.model_local_dir}/{args.model_name} "
        f"--ref-load {args.model_local_dir}/{args.model_name}_torch_dist "
        f"--load {load_save_path} "
        f"--save {load_save_path} "
        "--save-interval 20 "
        "--save-retain-interval 20 "
    )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 1 "
        # ------------
        "--num-steps-per-rollout 4 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 256 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    # sometimes disable eval to speed up debugging
    eval_args = ""
    if (args.mode != "debug_minimal") and args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 1 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len {100 if args.mode == 'debug_minimal' else 32768} "
            )
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 32768 "
            )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                # Deliberately make it very short for this easy task
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    if args.num_nodes <= 2:
        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )
    elif args.num_nodes <= 4:
        # TODO remove this temp cfg
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 4 "
            "--expert-tensor-parallel-size 1 "
        )
    else:
        # TODO choose a good config (currently randomly change to suit 64gpu)
        perf_args = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            f"--pipeline-model-parallel-size {1 if args.model_name == 'DeepSeek-V3-0324-5layer' else 4} "
            "--context-parallel-size 4 "
            "--expert-model-parallel-size 16 "
            "--expert-tensor-parallel-size 1 "
        )
        if re.search(r"(\d+)layer", args.model_name) is None:
            perf_args += "--decoder-last-pipeline-num-layers 13 "
    perf_args += (
        # ------------
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        # ------------
        "--use-dynamic-batch-size "
        # TODO temp use tiny value
        "--max-tokens-per-gpu 2048 "
        # "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # coef is 0 anyway, so the reference model is not loaded at all
        # "--use-kl-loss "
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
        # ------------
        # "--optimizer-cpu-offload "
        # "--overlap-cpu-optimizer-d2h-h2d "
        # "--use-precision-aware-optimizer "
    )

    sglang_decode_max_bs = 256
    sglang_world_size = 4 if args.num_nodes <= 4 else 64
    sglang_attn_dp_size = 1 if args.num_nodes <= 4 else 8
    sglang_attn_tp_size = sglang_world_size // sglang_attn_dp_size
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-tp-size {sglang_world_size} "
        f"--sglang-ep-size {sglang_world_size} "
        # dp attention
        "--sglang-enable-dp-attention "
        f"--sglang-dp-size {sglang_attn_dp_size} "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        # enable deepep for sglang
        "--sglang-moe-a2a-backend deepep "
        "--sglang-deepep-mode low_latency "
        # make every dp rank has 128 concurrency
        "--sglang-server-concurrency 1024 "
        f"--sglang-max-running-requests {sglang_world_size * sglang_decode_max_bs // sglang_attn_tp_size} "
        f"--sglang-chunked-prefill-size {sglang_world_size * sglang_decode_max_bs} "
        f"--sglang-cuda-graph-max-bs-decode {sglang_decode_max_bs} "
        # For quick experiments
        # """--sglang-json-model-override-args '{"num_hidden_layers": 5}' """
    )
    sglang_extra_env_vars = {
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": f"{sglang_decode_max_bs}",
    }

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        # "--attention-backend flash "
        f"--update-weight-buffer-size {4 * 1024 ** 3} "
        # TODO maybe enable it
        # use deepep for megatron
        # "--moe-enable-deepep "
        # "--moe-token-dispatcher-type flex "
        # ------------
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--use-fault-tolerance "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
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
        # TODO may get it from `config`
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**sglang_extra_env_vars},
        megatron_path=args.megatron_path,
        prepare_cmd=_prepare_cmd(args),
    )


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Command "full-train": runs download, bf16 cast, Megatron conversion, then
    # training, in that order. Inputs come from the command line fields.
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_megatron_ckpt(args)
    _execute_train(args)


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Command "prepare": download, bf16 cast and Megatron conversion only; no training.
    _prepare_download(args)
    _prepare_bf16_ckpt(args)
    _prepare_megatron_ckpt(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs) -> None:
    # [FUNCTION DOC] Command "train": runs training only; assumes prepare has already been done.
    _execute_train(args)


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: Empty typer callback; exists so the app always treats commands as
    # sub-commands even when only one is defined. Does nothing.
    pass


if __name__ == "__main__":
    app()
