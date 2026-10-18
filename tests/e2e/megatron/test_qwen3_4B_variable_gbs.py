# [MODULE DOC] End-to-end test of a global batch size that does not divide the collected rollout
# size, for Qwen3-4B on 4 colocated GPUs in bridge mode (direct Hugging Face weights conversion).
# Each rollout collects 8 prompts times 4 samples = 32 samples; global batch 24 with partial
# train steps allowed and sample trimming disabled yields a full step of 24 and a final partial
# step of 8. Context parallel 2 (tensor 1, pipeline 1), sequence parallel, dynamic batching at
# 8192 tokens per GPU, 4096 response tokens, 2 rollouts. SGLang engines of 2 GPUs. Registered for
# CUDA 4-GPU H200 (est. 500 s) and ROCm 4-GPU MI350 nightly (est. 500 s). Run as a script: prepare,
# strip proxy variables, then execute.
"""E2E for variable global batch sizes: global_batch_size does not divide the
collected rollout count, trim is disabled, and --allow-partial-train-step
trains the trailing rollouts as a smaller final step (e.g. steps of [24, 8]).
Runs with cp2 so the rollout-side schedule's token cap and PP-free mbs
alignment are exercised on a mid-size model."""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=500, suite="stage-c-4-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=500, suite="nightly-stage-c-4-gpu-mi350", labels=["megatron"])

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 4


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories, downloads Qwen3-4B and the dapo-math-17k
    # dataset, and converts the model to a Megatron distributed checkpoint on 4 GPUs. Side effects:
    # shell commands, network, filesystem, GPU.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def execute():
    # [FUNCTION DOC] Builds the argument string and launches training with 4 GPUs. Rollout: 2
    # rollouts of 8 prompts times 4 samples, 4096 response tokens, temperature 0.8, global batch
    # 24 with the allow-partial-train-step and disable-trim-samples flags. Parallelism: context 2,
    # sequence parallel, full recompute, dynamic batching at 8192 tokens per GPU. GRPO with KL loss
    # at zero coefficient, Adam at 1e-6. SGLang: 2 GPUs per engine, mem fraction 0.7. The CI test
    # flag is appended inline; colocated, with bridge conversion mode. Returns nothing.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 0.8 "
        # 32 rollouts per batch with gbs=24: one full step + a partial step of 8.
        "--global-batch-size 24 "
        "--allow-partial-train-step "
        "--disable-rollout-trim-samples "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
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

    sglang_args = "--rollout-num-gpus-per-engine 2 " "--sglang-mem-fraction-static 0.7 "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        "--ci-test "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
