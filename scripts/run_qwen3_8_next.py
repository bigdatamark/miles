# [MODULE DOC] Launcher for short GRPO smoke/validation runs of the Qwen3.8-Flash-Next hybrid model
# (full model on 32 GPUs, or a 4-layer reduced variant on 4 or 8 GPUs) with the Megatron backend,
# colocated SGLang rollout and training state offloaded to disk. Exposes a typer app with a single
# train command; there is no prepare step, so the HF checkpoint and the converted torch_dist
# reference checkpoint must exist. Imports the command utilities under the short alias U.
"""Qwen3.8-Flash-Next DAPO training on an already-running ray cluster (MILES_SCRIPT_EXTERNAL_RAY=1).

python scripts/run_qwen3_8_next.py train --num-nodes 8 --num-gpus-per-node 4
python scripts/run_qwen3_8_next.py train --model-name Qwen3.8-Flash-Next-4layer --num-nodes 1 --num-gpus-per-node 8
"""

import os
from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_MODEL_REGISTRY = {
    "Qwen3.8-Flash-Next": "qwen3.8-flash-next",
    "Qwen3.8-Flash-Next-4layer": "qwen3.8-flash-next-4layer",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # [CLASS DOC] Command-line configuration extending the shared train config. Fields: model name
    # (full or 4-layer variant), nodes (8) and GPUs per node (4), run id, optional HF checkpoint
    # path, directories (models, converted checkpoints, datasets, saves, train offload, Megatron
    # checkout), rollout count (5) and max response length (4096), check_weight_update_equal
    # (on), enable_r3 (routing replay, off), skip_saving (on; no checkpoint load/save flags) and
    # extra raw args.
    model_name: Literal["Qwen3.8-Flash-Next", "Qwen3.8-Flash-Next-4layer"] = "Qwen3.8-Flash-Next"
    num_nodes: int = 8
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
        # [FUNCTION DOC] Defaults the HF checkpoint path to the model directory plus model name.
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"


def _train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Builds the training argument string and environment variables and
    # submits the job through the backend. Raises an assertion error when the total GPU count
    # (nodes x GPUs per node) does not match the validated layout: exactly 32 for the full model
    # (pipeline parallel 8, engine of 8 GPUs) or 4 or 8 for the 4-layer model (pipeline parallel 2,
    # engine of 4 GPUs). Tensor parallel is 2, context parallel 1, expert parallel is total GPUs
    # divided by pipeline size, sglang tp and ep both equal the engine size.
    # Checkpoints: HF checkpoint and reference load from the converted dir keyed by the Megatron
    # model type; unless skip_saving, also load/save a per-run folder every 10 steps without
    # optimizer or rng state. Rollout: dapo-math-17k, math reward, thinking-mode chat template
    # kwargs, 4 prompts x 8 samples, temperature 0.8, one step per rollout. Algorithm: grpo with
    # zero kl/entropy coefficients, clip 0.2/0.28; adam lr 1e-6. Perf: full recompute, micro batch
    # 1, 8192 tokens per GPU. SGLang: flashinfer linear attention prefill, triton moe runner,
    # chunked prefill 8192, radix cache off, router health check thresholds. Misc: 1 GiB weight
    # update buffer, offload training state to disk under the offload dir, memory fraction 0.7,
    # colocate, a fixed internal model name, thd qkv format, flashqla linear attention, a custom
    # model provider path, long health-check and distributed timeouts; optional weight-update
    # equality check (skipping visual and ple embedding modules) and routing replay.
    # Environment for the job: health-check timeout, multimem all-gather disabled, the caller's
    # python path, a single inductor compile thread (works around a deadlock with the memory
    # saver preload), and the triton and inductor cache dirs when set in the caller's environment
    # (needed because the ray runtime environment replaces the actor environment).
    # KNOWN ISSUE: the module docstring says the script targets an external ray cluster, but this
    # function does nothing to enforce or configure that.
    megatron_model_type = _MODEL_REGISTRY[args.model_name]

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} " f"--ref-load {args.ckpt_dir}/{megatron_model_type}_torch_dist "
    )
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
        '--apply-chat-template-kwargs \'{"thinking_mode":"thinking"}\' '
    )

    num_gpus = args.num_nodes * args.num_gpus_per_node
    if args.model_name == "Qwen3.8-Flash-Next":
        assert num_gpus == 32, f"the full-model layout is validated on 32 GPUs, got {num_gpus}"
        pipeline_parallel_size, engine_gpus = 8, 8
    else:
        assert num_gpus in (4, 8), f"the 4-layer layout is validated on 4 or 8 GPUs, got {num_gpus}"
        pipeline_parallel_size, engine_gpus = 2, 4
    parallel_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {pipeline_parallel_size} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {num_gpus // pipeline_parallel_size} "
        "--expert-tensor-parallel-size 1 "
    )
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
        "--sglang-linear-attn-prefill-backend flashinfer "
        "--sglang-moe-runner-backend triton "
        "--sglang-chunked-prefill-size 8192 "
        "--sglang-disable-radix-cache "
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
        "--model-name qwen4_exp "
        "--qkv-format thd "
        "--linear-attention-backend flashqla "
        "--custom-model-provider-path "
        "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider "
        "--rollout-health-check-interval 300 "
        "--distributed-timeout-minutes 60 "
        "--rollout-health-check-timeout 300 "
    )
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal " "--check-weight-update-skip-list visual. ple_embedding. "
    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    train_args = (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} {sglang_args} {misc_args} {args.extra_args} "
    )

    extra_env_vars = {
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "SGLANG_DISABLE_MULTIMEM_AG": "1",
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
    # [FUNCTION DOC] The only command: submits the training job via the train helper.
    _train(args)


if __name__ == "__main__":
    app()
