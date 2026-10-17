# [MODULE DOC]
# Launcher script (single command, run through typer) for full-parameter GRPO training of Kimi-K2
# (Instruct or Thinking), Megatron plus SGLang colocated on a 32-node by 8-GPU cluster, on DAPO math
# with optional AIME eval. Training only: the Megatron distributed checkpoint and Hugging Face
# checkpoint must already exist, and the ray cluster is expected to be already joined (the launcher
# is meant to be run in the external-ray mode). Both variants share one recipe, expressed through a
# frozen per-model table of differences (checkpoint folder names, response length, eval samples,
# truncated importance sampling, Megatron DeepEP).
"""Kimi-K2 full-parameter GRPO training script.

=====================

One recipe covers both released Kimi-K2 checkpoints. Instruct and Thinking share the
32-node TP8/PP8/CP4/EP32 topology, the dp-attention rollout layout and the whole GRPO /
optimizer schedule; they differ only in the checkpoint they load, the response-length
budget, the eval dataset path, whether truncated importance sampling is enabled, and
whether DeepEP drives the Megatron MoE dispatch.

The checkpoint must already be converted to Megatron `torch_dist`; this script only
submits the training job. It also assumes a Ray cluster that is already joined
across all 32 nodes, so run it with `MILES_SCRIPT_EXTERNAL_RAY=1`: the launcher then
skips `ray start` and submits straight to the running cluster.

=====================

Args:
  --model-name: Model variant, one of Kimi-K2-Instruct / Kimi-K2-Thinking.
  --num-nodes: Number of training nodes (default: 32, the only tested topology).
  --num-gpus-per-node: GPUs per node (default: 8).
  --enable-eval: Run AIME evaluation every 20 steps (default: on).
  --save: Write checkpoints to --output-dir (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_kimi_k2.py --model-name Kimi-K2-Thinking
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

_MODEL_NAMES = Literal["Kimi-K2-Instruct", "Kimi-K2-Thinking"]


@dataclass(frozen=True)
class _Recipe:
    # [CLASS DOC] HELPER: Immutable record of the per-variant differences. Fields: Megatron model type
    # name, Hugging Face checkpoint folder name, converted-checkpoint folder prefix (suffix
    # "_torch_dist" is added by the launcher), maximum response length for rollout and eval, number
    # of samples per eval prompt, whether truncated importance sampling is on, and whether Megatron
    # uses DeepEP for MoE dispatch.
    megatron_model_type: str
    hf_checkpoint: str
    torch_dist_name: str
    max_response_len: int
    n_samples_per_eval_prompt: int
    use_tis: bool
    megatron_deepep: bool


_RECIPES: dict[str, _Recipe] = {
    "Kimi-K2-Instruct": _Recipe(
        megatron_model_type="kimi-k2",
        hf_checkpoint="Kimi-K2-Instruct",
        torch_dist_name="Kimi-K2",
        max_response_len=32768,
        n_samples_per_eval_prompt=8,
        use_tis=False,
        megatron_deepep=True,
    ),
    "Kimi-K2-Thinking": _Recipe(
        megatron_model_type="kimi-k2-thinking",
        hf_checkpoint="Kimi-K2-Thinking-fp8",
        torch_dist_name="Kimi-K2-Thinking",
        max_response_len=16384,
        n_samples_per_eval_prompt=16,
        use_tis=True,
        megatron_deepep=False,
    ),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (output dir, backend creation).
    # Fields: run id, model name (Instruct or Thinking), node count (overridden default 32), GPUs per
    # node (8), eval switch (on), save switch (on), number of rollouts (100), extra raw args, data and
    # model directories, Megatron checkout path. The recipe property gives the per-model table entry.
    run_id: str = command_utils.create_run_id()
    model_name: _MODEL_NAMES = "Kimi-K2-Instruct"
    num_nodes: int = 32
    num_gpus_per_node: int = 8
    enable_eval: bool = True
    save: bool = True
    num_rollout: int = 100
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def recipe(self) -> _Recipe:
        # [FUNCTION DOC] Property: the per-model recipe record looked up by the model name.
        return _RECIPES[self.model_name]


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string and submits the job. Checkpoint: hf checkpoint
    # and reference load (converted folder) from the model directory per the recipe; the load path is
    # the output directory's checkpoints folder (always given, even when not saving); saving to the same
    # folder every 20 steps only if save is on. Rollout: DAPO math, math reward, chat template,
    # shuffled, 128 prompts x 8 samples with 256 prompts over-sampled and filtered to those with
    # nonzero reward standard deviation, response length from the recipe, temperature 1, 4 optimizer
    # steps per rollout, balanced data. Eval (if enabled): AIME 2024 every 20 steps with the recipe's
    # sample count and response length, top-p 1. Parallelism: tensor 8 with sequence parallel,
    # pipeline 8, context 4, expert 32, expert-tensor 1, last pipeline stage 5 layers; full uniform
    # recompute, dynamic batch 16384 tokens per GPU. GRPO with kl loss on at zero coefficient, zero
    # entropy, asymmetric clip, and truncated importance sampling per recipe. Adam 1e-6 with CPU offload
    # group. SGLang: 16-GPU engines, memory fraction 0.7, data-parallel attention size 8, dense TP 1,
    # DP LM head, expert parallel 16, server concurrency 1024. Misc: dropouts off, fp32 accumulation,
    # flash attention, optional Megatron DeepEP with flex dispatcher per recipe (overrides all-to-all),
    # colocate, 2 GiB weight update buffer, actor nodes and GPUs from options. Submits via the backend
    # with the recipe's Megatron model type and no extra environment variables or copy commands.
    # KNOWN ISSUE: the "need to comment this when using model with MLA" note sits beside a flash
    # attention flag although the model uses latent attention (stale comment).
    U = args.create_backend()
    recipe = args.recipe

    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{recipe.hf_checkpoint} "
        f"--ref-load {args.model_dir}/{recipe.torch_dist_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints "
    )
    if args.save:
        ckpt_args += f"--save {args.output_dir}/checkpoints " "--save-interval 20 "

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 128 "
        "--n-samples-per-prompt 8 "
        f"--rollout-max-response-len {recipe.max_response_len} "
        "--rollout-temperature 1 "
        # over-sample, then drop the prompts whose rewards carry no gradient signal
        "--over-sampling-batch-size 256 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--num-steps-per-rollout 4 "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            f"--n-samples-per-eval-prompt {recipe.n_samples_per_eval_prompt} "
            f"--eval-max-response-len {recipe.max_response_len} "
            "--eval-top-p 1 "
        )

    perf_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 8 "
        "--context-parallel-size 4 "
        "--expert-model-parallel-size 32 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 5 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
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
    if recipe.use_tis:
        grpo_args += "--use-tis "

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
        "--rollout-num-gpus-per-engine 16 "
        "--sglang-mem-fraction-static 0.7 "
        # dp attention
        "--sglang-enable-dp-attention "
        "--sglang-dp-size 8 "
        "--sglang-moe-dense-tp-size 1 "
        "--sglang-enable-dp-lm-head "
        "--sglang-ep-size 16 "
        # make every dp rank has 128 concurrency
        "--sglang-server-concurrency 1024 "
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
    )
    if recipe.megatron_deepep:
        # use deepep for megatron; the dispatcher overrides the alltoall from the model args
        misc_args += "--moe-enable-deepep " "--moe-token-dispatcher-type flex "
    misc_args += (
        "--colocate "
        f"--update-weight-buffer-size {4 * 512 * 1024 * 1024} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
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
        megatron_model_type=recipe.megatron_model_type,
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command line entry point: runs execute with the parsed options.
    execute(args)


if __name__ == "__main__":
    typer.run(main)
