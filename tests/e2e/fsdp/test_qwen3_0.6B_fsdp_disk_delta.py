# [MODULE DOC]
# End-to-end script testing FSDP weight updates through the "disk-delta" transfer protocol on
# Qwen3-0.6B with 4 GPUs (2 for training, 2 for rollout, not colocated, asynchronous training
# script). Registered as a 900-second 4-GPU H200 CUDA test (hopper only, weight-update label) and a
# 500-second nightly 4-GPU MI350 ROCm test. Uses the legacy module-level command helper API (not
# the backend object). Flow as a script: download the model and GSM8K, strip proxy variables, run
# 3 rollouts of 32 prompts with 8 samples each (oversampling 64 with a dynamic filter requiring
# non-zero reward spread, global batch 256) with weight updates published to a disk directory and
# reconciled against a local checkpoint directory under the root filesystem. KL checker disabled.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

import miles.utils.external_utils.command_utils.legacy as U

register_cuda_ci(
    est_time=900,
    suite="stage-c-4-gpu-h200",
    labels=["weight-update"],
    hardware=["hopper"],
)
register_rocm_ci(est_time=500, suite="nightly-stage-c-4-gpu-mi350", labels=["weight-update"])

# The FSDP backend streams through the shared WeightUpdater, so every transfer
# protocol is open to it. disk-delta is the one that reconciles the stream
# against the checkpoint's own bytes, which is what makes it worth a case.

MODEL_NAME = "Qwen3-0.6B"
NUM_GPUS = 4


def prepare():
    # [FUNCTION DOC] Create the models and datasets directories, download the model from the hub
    # into the models directory, and download the GSM8K dataset, all through the legacy helper
    # functions. Side effects: network downloads, filesystem.
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    # [FUNCTION DOC] Launch the training job through the legacy execute-train helper with no
    # Megatron model type and the asynchronous training script. Arguments: the HF checkpoint; GSM8K
    # rollouts (math reward, 3 rollouts, batch 32, 8 samples, up to 1024 tokens, temperature 1,
    # oversampling 64 with a non-zero-reward-spread filter, global batch 256); Adam learning rate
    # 1e-6 with weight decay 0.1; GRPO with zero KL-loss, KL and entropy coefficients and clipping
    # 0.2 / 0.28 (KL loss left disabled); one rollout GPU per engine with SGLang metrics; a node
    # with half the GPUs (2) for training and half (2) for rollout (not colocated), the FSDP
    # backend, disk-delta weight transfer with a published directory and a local checkpoint
    # directory; CI test mode with the KL checker disabled.
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        # "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-enable-metrics "

    misc_args = (
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS // 2} "
        f"--rollout-num-gpus {NUM_GPUS // 2} "
        "--train-backend fsdp "
        "--update-weight-transfer-mode disk-delta "
        "--update-weight-disk-dir /root/fsdp_delta/published "
        "--update-weight-local-checkpoint-dir /root/fsdp_delta/local "
    )

    ci_args = "--ci-test --ci-disable-kl-checker "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
