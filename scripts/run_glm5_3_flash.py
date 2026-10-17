# [MODULE DOC]
# Launcher script (typer command line app with a single "train" command) for GRPO training of
# GLM-5.3-Flash or its 4-layer prune on DAPO math, with Megatron training and SGLang rollout
# colocated, intended for an already running ray cluster. Training only: the Megatron distributed
# checkpoint and Hugging Face checkpoint must already exist. Cluster shapes are validated: the full
# model on 32 or 64 GPUs, the 4-layer model on exactly 8 GPUs. Offloads the training state to disk
# during rollout. The module table maps model names to Megatron model types.
"""GLM-5.3-Flash DAPO training on an already-running ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1).

python scripts/run_glm5_3_flash.py train --model-name GLM-5.3-Flash --num-nodes 8 --num-gpus-per-node 4
python scripts/run_glm5_3_flash.py train --num-nodes 1 --num-gpus-per-node 8
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MODEL_REGISTRY = {
    "GLM-5.3-Flash": "glm5.3-flash",
    "GLM-5.3-Flash-4layer": "glm5.3-flash-4layer",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration, overriding the node count default
    # (2 nodes) and fixing GPUs per node (4). Fields: model name (default 4-layer prune), run id,
    # optional explicit hf checkpoint (derived in post-init), directories (models, converted Megatron
    # checkpoints, datasets, run outputs, Megatron checkout, training-state disk offload), number of
    # rollouts (5), response length (4096), weight-update equality check switch (on), routing replay
    # switch (off), skip saving (on), extra raw args.
    model_name: Literal["GLM-5.3-Flash", "GLM-5.3-Flash-4layer"] = "GLM-5.3-Flash-4layer"
    num_nodes: int = 2
    num_gpus_per_node: int = 4
    run_id: str = U.create_run_id()
    hf_checkpoint: str | None = None
    model_dir: str = "/root/models"
    ckpt_dir: str = "/root/ckpt"
    data_dir: str = "/root/datasets"
    save_dir: str = "/root/shared_data"
    megatron_path: str = "/root/Megatron-LM"
    train_offload_dir: str = "/root/train_offload"
    num_rollout: int = 5
    rollout_max_response_len: int = 4096
    check_weight_update_equal: bool = True
    enable_r3: bool = False
    skip_saving: bool = True
    extra_args: str = ""

    def __post_init__(self):
        # [FUNCTION DOC]
        # If no hf checkpoint was given, sets it to the model directory plus the model name.
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"


def _train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Builds the training argument string and submits the job. Looks up the
    # Megatron model type. Checkpoint: hf checkpoint; reference load from the converted-checkpoint
    # directory under "<model type>_torch_dist"; unless saving is skipped (default), load/save under
    # the save directory run folder every 10 steps without optimizer or RNG state. Rollout: math
    # reward, chat template, shuffled, rollouts and response length from options, 4 prompts x 8
    # samples, temperature 0.8, one step per rollout, balanced data, DAPO data. Parallelism: with the
    # full model, asserts total GPUs is 32 or 64 and uses tensor 8, sequence parallel, pipeline 4
    # (11 first / 12 last layers), context 1, expert = GPUs / 4, 8-GPU engines; with the 4-layer model,
    # asserts exactly 8 GPUs and uses tensor 2, pipeline 2, expert 2, 4-GPU engines. Full uniform
    # recompute, micro batch 1, 8192 tokens per GPU. GRPO zero coefficients. Adam 1e-6. SGLang: tensor
    # and expert size = engine size, triton moe (so routing replay has materialized top-k), chunked
    # prefill 8192, radix cache off, tilelang prefill/decode backends, bf16 KV cache, router health
    # settings. Misc: memory margin, training-state offload to disk, memory fraction 0.7, colocate,
    # model name flag for loading, packed layout, long distributed timeout, health checks, optional
    # weight-update equality check skipping the visual parameters, optional routing replay. Environment
    # variables: health check timeout, the caller's python path (copied at call time), single-thread
    # compile pool (avoids a deadlock), and the triton and inductor cache directories if set in the
    # caller's environment (because ray replaces the worker environment). Submits via the backend.
    # Raises an assertion error on unsupported GPU counts.
    # KNOWN ISSUE: the assertion on GPU count compares the total across nodes, so the node count and
    # GPUs per node options are only valid in combinations giving those totals.
    megatron_model_type = _MODEL_REGISTRY[args.model_name]

    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} --ref-load {args.ckpt_dir}/{megatron_model_type}_torch_dist "
    if not args.skip_saving:
        load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
        ckpt_args += (
            f"--load {load_save_path} --save {load_save_path} --save-interval 10 "
            "--no-save-optim --no-save-rng --no-load-optim --no-load-rng "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
    )

    num_gpus = args.num_nodes * args.num_gpus_per_node
    if args.model_name == "GLM-5.3-Flash":
        assert num_gpus in (32, 64), f"the full-model layout is validated on 32 or 64 GPUs, got {num_gpus}"
        parallel_args = (
            "--tensor-model-parallel-size 8 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 11 "
            "--decoder-last-pipeline-num-layers 12 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {num_gpus // 4} "
            "--expert-tensor-parallel-size 1 "
        )
        engine_gpus = 8
    else:
        assert num_gpus == 8, f"the 4-layer layout is validated on 8 GPUs, got {num_gpus}"
        parallel_args = (
            "--tensor-model-parallel-size 2 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 2 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 2 "
            "--expert-tensor-parallel-size 1 "
        )
        engine_gpus = 4
    engine_args = (
        f"--rollout-num-gpus-per-engine {engine_gpus} --sglang-tp-size {engine_gpus} --sglang-ep-size {engine_gpus} "
    )

    perf_args = (
        f"{parallel_args}"
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 8192 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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

    sglang_args = (
        f"{engine_args}"
        # routing replay needs materialized topk ids; the SM100 default resolves to a fused runner
        "--sglang-moe-runner-backend triton "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-disable-radix-cache "
        "--sglang-dsa-prefill-backend tilelang "
        "--sglang-dsa-decode-backend tilelang "
        "--sglang-kv-cache-dtype bfloat16 "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        "--accumulate-allreduce-grads-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024**3} "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--offload-train-target disk "
        f"--offload-train-disk-dir {args.train_offload_dir} "
        "--sglang-mem-fraction-static 0.7 "
        "--colocate "
        "--model-name glm5_next "
        "--qkv-format thd "
        "--rollout-health-check-interval 300 "
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-timeout 300 "
    )
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal --check-weight-update-skip-list visual. "
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {sglang_args} {misc_args} {args.extra_args} "
    )

    extra_env_vars = {
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        # inductor's subprocess compile pool deadlocks under the torch_memory_saver LD_PRELOAD
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    }
    # ray's runtime_env replaces the actor env; without the JIT cache dirs the kernels recompile every run
    for cache_var in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        cache_dir = os.environ.get(cache_var)
        if cache_dir:
            extra_env_vars[cache_var] = cache_dir

    args.create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        extra_env_vars=extra_env_vars,
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] Command "train": runs training; assumes checkpoints and datasets exist.
    _train(args)


if __name__ == "__main__":
    app()
