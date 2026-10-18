# [MODULE DOC] CI smoke test of the rollout indexer top-k replay path on a 4-layer slice of GLM-5
# (every layer has an indexer), on one node of 2 GPUs with tensor and expert parallel 2 and
# optimizer offload (CUDA: 2-GPU H200 stage, estimated 1100 seconds, labels megatron,
# model-scripts and replay, hopper and blackwell). Enabling rollout indexer replay also forces
# training-side indexer replay, so training consumes the per-layer top-k emitted by SGLang.
# Metric history gates are registered on grad norm, PPO KL, train/rollout logprob absolute
# difference, train/rollout KL and raw reward. Run as a script: build args, prepare, strip all
# proxy environment variables (both cases), train.

import os

from scripts.run_glm5_744b_a40b import (
    ScriptArgs,
    _execute_train,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate


# Basic smoke test that exercises the rollout indexer-topk replay path on
# GLM-5 (every layer has an indexer). Enabling --use-rollout-indexer-replay
# also forces --use-indexer-replay via miles_validate_args, so the training
# side consumes the per-layer topk emitted by SGLang.

register_cuda_ci(
    est_time=1100,
    suite="stage-c-2-gpu-h200",
    labels=["megatron", "model-scripts", "replay"],
    hardware=["hopper", "blackwell"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments from the environment with overrides: H200
    # hardware, model GLM-5_4layer, 1 node of 2 GPUs, 2 rollouts of 4 prompts times 4 samples,
    # global batch 16, optimizer offload on. Extra flags: CI test mode, logprobs checker disabled,
    # rollout indexer replay on, skip the actor forward-only pass, response length 4096, a cap of
    # 1900000 total tokens in SGLang (to avoid CPU out-of-memory), and tensor parallel 2, pipeline
    # parallel 1, expert parallel 2.
    return ScriptArgs.from_env(
        hardware="H200",
        model_name="GLM-5_4layer",
        num_nodes=1,
        num_gpus_per_node=2,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        enable_optimizer_offload=True,
        extra_args=(
            "--ci-test "
            "--ci-disable-logprobs-checker "
            "--use-rollout-indexer-replay "
            "--skip-actor-forward-only "
            "--rollout-max-response-len 4096 "
            # preserve to avoid CPU OOM
            "--sglang-max-total-tokens 1900000 "
            # exercise indexer replay with two-way tensor and expert parallelism
            "--tensor-model-parallel-size 2 "
            "--pipeline-model-parallel-size 1 "
            "--expert-model-parallel-size 2 "
        ),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates the output directory through the arguments' backend; runs the
    # launcher's download step and checkpoint validation, then produces the Megatron checkpoint
    # (no FP8 conversion, unlike the sibling CI test).
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    _prepare_megatron_ckpt(args)


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step.
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
