# [MODULE DOC] End-to-end LoRA training smoke test: a 2-rollout GRPO run on gsm8k with
# Qwen2.5-0.5B-Instruct using LoRA (rank 32, alpha 32, no dropout, all linear layers) through the
# Megatron-to-HF bridge on 4 colocated GPUs (CUDA: 4-GPU H200 stage, estimated 400 seconds, lora
# label, hopper and blackwell; ROCm: nightly 4-GPU MI350 suite, estimated 500 seconds). Run as a
# script: prepare, strip all proxy environment variables (both cases) from the process, execute.
# The only pass criterion is that training finishes (the generic CI test flag is set; no metric
# threshold). Evaluation every 2 rollouts is on unless the environment variable for enabling
# eval is set to 0. Checkpoints are saved every 2 rollouts to a fixed root-level directory.
# KNOWN ISSUE: the existing docstring below says 8 GPUs are required, but the test uses 4 GPUs
# and is registered on a 4-GPU stage.
"""E2E test for LoRA training with Qwen2.5-0.5B on GSM8K.

Uses the Megatron backend with bridge mode.  Runs a short GRPO training loop
with LoRA enabled (rank=32, all-linear) to validate:
  - LoRA model setup via Bridge
  - LoRA weight sync to SGLang rollout engines
  - LoRA checkpoint save (native + HF PEFT format)
  - Training completes without errors

Requires: 8 GPUs, Qwen2.5-0.5B-Instruct model, GSM8K dataset.
Triggered by label: run-ci-lora
"""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=400, suite="stage-c-4-gpu-h200", labels=["lora"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=500, suite="nightly-stage-c-4-gpu-mi350", labels=["lora"])


ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4


def prepare():
    # [FUNCTION DOC] Creates fixed root-level model and dataset directories, downloads the
    # Qwen2.5-0.5B-Instruct model, and downloads the gsm8k dataset repository (as a dataset repo)
    # into the dataset directory, through the default config's backend.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu("hf download --repo-type dataset zhuzilin/gsm8k --local-dir /root/datasets/gsm8k")


def execute():
    # [FUNCTION DOC] Builds the training arguments and launches training on 4 GPUs. Groups:
    # checkpoint (HF directory with the bridge conversion mode); LoRA (rank 32, alpha 32, dropout
    # 0, all-linear target modules); rollout (gsm8k train data, chat template, shuffling, math
    # reward, 2 rollouts, 8 prompts times 8 samples, response length 1024, temperature 1.0, global
    # batch 32); eval (gsm8k test set, 1 sample, length 1024, top-k 1, plus an eval interval of 2
    # when eval is enabled); parallelism all 1 with sequence parallel, dynamic batch size, 4096
    # tokens per GPU; GRPO with zero KL and entropy coefficients and clip 0.2/0.28; Adam with
    # learning rate 1e-5 constant, weight decay 0.1; SGLang one GPU per engine with static memory
    # fraction 0.4; the CI test flag; saving every 2 rollouts to a fixed checkpoint directory;
    # misc (zero dropout, fp32 gradient accumulation and softmax, flash attention, per-token loss,
    # the Miles router, one actor node of 4 GPUs, colocated). Wandb args are named after this file.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " "--megatron-to-hf-mode bridge "

    lora_args = "--lora-rank 32 " "--lora-alpha 32 " "--lora-dropout 0.0 " '--target-modules "all-linear" '

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 32 "
    )

    eval_args = (
        f"{'--eval-interval 2 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-5 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.4 "

    ci_args = "--ci-test "

    save_args = "--save-interval 2 " "--save /root/checkpoints/lora-qwen2.5-0.5B-ci "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--calculate-per-token-loss "
        "--use-miles-router "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{lora_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{save_args} "
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
