# [MODULE DOC] Launcher for GSPO training of Qwen3-Next-80B-A3B in one of two topologies: a 4-node
# colocated production layout or a single-node layout with 2 rollout GPUs and 6 training GPUs.
# Topology differences are captured in a frozen recipe table (module-level lookup keyed by
# topology name). No prepare step: the converted torch_dist checkpoint and datasets must exist.
# Entry points are execute and the typer-wrapped main.
"""Qwen3-Next-80B-A3B GSPO training script.

=====================

One recipe covers both topologies of the run. `4node` is the production layout: 4 nodes x
8 GPUs with the rollout engines colocated on the training GPUs, TP2/PP4/CP2/EP8, MTP
speculative decoding and DeepEP token dispatch. `single-node` fits the same model onto a
single 8-GPU node by dedicating 2 GPUs to rollout and 6 to training (PP6), which leaves
room for neither expert parallelism nor a draft model, so every batch dimension shrinks
along with it. Everything else -- dataset, GSPO constants, optimizer schedule, eval -- is
shared.

`4node` joins the remaining hosts to the local ray head over ssh, reading MASTER_ADDR and
MLP_WORKER_0_HOST from the environment; pass --no-join-ray-workers when the ray cluster
is already complete.

The checkpoint must already be converted to Megatron `torch_dist`; this script only
submits the training job.

=====================

Args:
  --topology: Cluster layout, one of 4node / single-node.
  --num-gpus-per-node: Physical GPUs per node (default: 8).
  --join-ray-workers: Join every host of /root/mpi_rack_hostfile to the ray cluster,
    multi-node topologies only (default: on).
  --model-dir / --data-dir: Checkpoint / dataset directories.

=====================

  python scripts/run_qwen3_next_80b_a3b.py --topology single-node
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

_TOPOLOGIES = Literal["4node", "single-node"]


@dataclass(frozen=True)
class _Recipe:
    # [CLASS DOC] HELPER: Immutable per-topology recipe: actor nodes and GPUs per node, rollout GPU
    # count (none means rollout shares the training GPUs, i.e. colocated), rollout count, rollout
    # batch size, samples per prompt, temperature, global batch size, tensor/pipeline/context/
    # expert parallel sizes, dynamic-batch token budget, eval samples per prompt and top-p, GPUs
    # per rollout engine, SGLang expert parallel size, whether to enable speculative decoding,
    # the MoE token dispatcher type and whether to enable DeepEP.
    actor_num_nodes: int
    actor_num_gpus_per_node: int
    # None means the rollout engines share the training GPUs
    rollout_num_gpus: int | None
    num_rollout: int
    rollout_batch_size: int
    n_samples_per_prompt: int
    rollout_temperature: float
    global_batch_size: int
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    context_parallel_size: int
    expert_model_parallel_size: int
    max_tokens_per_gpu: int
    n_samples_per_eval_prompt: int
    eval_top_p: float
    rollout_num_gpus_per_engine: int
    sglang_ep_size: int
    enable_spec: bool
    moe_token_dispatcher_type: str
    moe_enable_deepep: bool


_RECIPES: dict[str, _Recipe] = {
    "4node": _Recipe(
        actor_num_nodes=4,
        actor_num_gpus_per_node=8,
        rollout_num_gpus=None,
        num_rollout=3000,
        rollout_batch_size=32,
        n_samples_per_prompt=8,
        rollout_temperature=1,
        global_batch_size=256,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=4,
        context_parallel_size=2,
        expert_model_parallel_size=8,
        max_tokens_per_gpu=8192,
        n_samples_per_eval_prompt=16,
        eval_top_p=1,
        rollout_num_gpus_per_engine=8,
        sglang_ep_size=8,
        enable_spec=True,
        moe_token_dispatcher_type="flex",
        moe_enable_deepep=True,
    ),
    "single-node": _Recipe(
        actor_num_nodes=1,
        actor_num_gpus_per_node=6,
        rollout_num_gpus=2,
        num_rollout=300,
        rollout_batch_size=16,
        n_samples_per_prompt=4,
        rollout_temperature=0.8,
        global_batch_size=64,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=6,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        max_tokens_per_gpu=2048,
        n_samples_per_eval_prompt=2,
        eval_top_p=0.7,
        rollout_num_gpus_per_engine=2,
        sglang_ep_size=1,
        enable_spec=False,
        moe_token_dispatcher_type="alltoall",
        moe_enable_deepep=False,
    ),
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command-line configuration extending the shared train config. Fields: run id,
    # topology (4node or single-node), model name and Megatron model type, physical GPUs per node
    # (8), join_ray_workers (on; only applied for multi-node recipes), extra raw args, and data,
    # model and Megatron directories. The selected recipe is exposed by a read-only property.
    run_id: str = command_utils.create_run_id()
    topology: _TOPOLOGIES = "4node"
    model_name: str = "Qwen3-Next-80B-A3B-Thinking"
    megatron_model_type: str = "qwen3-next-80B-A3B"
    num_gpus_per_node: int = 8
    join_ray_workers: bool = True
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    @property
    def recipe(self) -> _Recipe:
        # [FUNCTION DOC] Property: looks up the recipe for the chosen topology.
        return _RECIPES[self.topology]


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Builds the training argument string from the topology recipe and submits the
    # job. Checkpoints: HF and torch_dist reference under the model dir, load and save in the
    # checkpoints folder under the output dir every 20 steps. Rollout: dapo-math-17k, deepscaler
    # reward, max response 8192, other sizes and temperature from the recipe. Eval: AIME every 20,
    # max 16384, sample count and top-p from the recipe. Parallel sizes from the recipe with
    # sequence parallel, expert tensor parallel 1, full recompute, dynamic batch. Algorithm: gspo
    # with zero kl/entropy coefficients, clip 4e-4. Optimizer: adam lr 1e-6 constant with CPU
    # offload. SGLang: engine size and EP size from the recipe, memory fraction 0.8, graph batch
    # sizes up to 128; with speculative decoding also an EAGLE setup (2 steps, top-k 1, 3 draft
    # tokens, draft weights CPU backup, max running requests 512). Misc: dropout off, fp32
    # allreduce, flash attention, dispatcher type, node and GPU counts, DeepEP flag if set, and
    # either colocate (no rollout GPU count) or an explicit rollout GPU count.
    # Ray workers are joined over ssh (using the MASTER_ADDR environment variable, which must then
    # be set or a key error is raised when the callback runs, and the optional MLP_WORKER_0_HOST)
    # only if join_ray_workers is on and the recipe has more than one actor node.
    U = args.create_backend()
    ckpt_args = (
        f"--hf-checkpoint {args.model_dir}/{args.model_name} "
        f"--ref-load {args.model_dir}/{args.model_name}_torch_dist "
        f"--load {args.output_dir}/checkpoints "
        f"--save {args.output_dir}/checkpoints "
        "--save-interval 20 "
    )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.recipe.num_rollout} "
        f"--rollout-batch-size {args.recipe.rollout_batch_size} "
        f"--n-samples-per-prompt {args.recipe.n_samples_per_prompt} "
        "--rollout-max-response-len 8192 "
        f"--rollout-temperature {args.recipe.rollout_temperature} "
        f"--global-batch-size {args.recipe.global_batch_size} "
        "--balance-data "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
        f"--n-samples-per-eval-prompt {args.recipe.n_samples_per_eval_prompt} "
        "--eval-max-response-len 16384 "
        f"--eval-top-p {args.recipe.eval_top_p} "
    )

    perf_args = (
        f"--tensor-model-parallel-size {args.recipe.tensor_model_parallel_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {args.recipe.pipeline_model_parallel_size} "
        f"--context-parallel-size {args.recipe.context_parallel_size} "
        f"--expert-model-parallel-size {args.recipe.expert_model_parallel_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.recipe.max_tokens_per_gpu} "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
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

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.recipe.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.8 "
        f"--sglang-ep-size {args.recipe.sglang_ep_size} "
        "--sglang-cuda-graph-bs-decode 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 "
    )
    if args.recipe.enable_spec:
        sglang_args += (
            # mtp
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 2 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 3 "
            "--sglang-enable-draft-weights-cpu-backup "
            "--sglang-max-running-requests 512 "
        )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to drop this when using model with MLA
        "--attention-backend flash "
        f"--moe-token-dispatcher-type {args.recipe.moe_token_dispatcher_type} "
        f"--actor-num-nodes {args.recipe.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.recipe.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
    )
    if args.recipe.moe_enable_deepep:
        misc_args += "--moe-enable-deepep "
    if args.recipe.rollout_num_gpus is None:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {args.recipe.rollout_num_gpus} "

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

    join_workers = args.join_ray_workers and args.recipe.actor_num_nodes > 1
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        before_ray_job_submit=(
            (
                lambda: U.ssh_start_ray_workers(
                    master_addr=os.environ["MASTER_ADDR"],
                    num_gpus_per_node=args.num_gpus_per_node,
                    head_host=os.environ.get("MLP_WORKER_0_HOST"),
                )
            )
            if join_workers
            else None
        ),
        megatron_path=args.megatron_path,
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs):
    # [FUNCTION DOC] Command-line entry point; just calls execute (no prepare step).
    execute(args)


if __name__ == "__main__":
    typer.run(main)
