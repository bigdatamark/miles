# [MODULE DOC]
# End-to-end script for FSDP hybrid-shard training of Qwen3-4B on 4 GPUs (colocated) with 2-way
# data-parallel replication, i.e. 2 replicas of 2-way sharding (r2s2). Registered as a 600-second
# 4-GPU H200 CUDA test (hopper and blackwell, fsdp label) and a 600-second 4-GPU MI350 ROCm test
# in the non-nightly stage (fsdp and amd labels). As a script: download model and math dataset,
# strip proxy variables, train 2 rollouts of 8 prompts with 8 samples (math reward, up to 4096
# tokens, global batch 32) with GRPO and CI test mode.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=600,
    suite="stage-c-4-gpu-h200",
    labels=["fsdp"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=600,
    suite="stage-c-4-gpu-mi350",
    labels=["fsdp", "amd"],
)

NUM_GPUS = 4
DP_REPLICATE_SIZE = 2

MODEL_NAME = "Qwen3-4B"


def prepare():
    # [FUNCTION DOC] Through the command backend: create the models and datasets directories,
    # download the Qwen3-4B model into the models directory and download the math training
    # dataset. Side effects: network downloads, filesystem.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")


def execute():
    # [FUNCTION DOC] Launch the training job through the backend's execute-train (no Megatron
    # model type). Arguments: the HF checkpoint; math dataset rollouts (math reward, 2 rollouts of
    # 8 prompts with 8 samples, up to 4096 tokens, temperature 1, global batch 32); Adam learning
    # rate 1e-6 with weight decay 0.1; GRPO with zero KL-loss, KL and entropy coefficients and
    # clipping 0.2 / 0.28; FSDP backend with data-parallel replicate size 2 and a 512 MiB update
    # buffer; one GPU per rollout engine with SGLang metrics; CI test mode; one node, 4 GPUs,
    # colocated; default tracking arguments.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
    )

    fsdp_args = (
        "--train-backend fsdp " f"--dp-replicate-size {DP_REPLICATE_SIZE} " "--update-weight-buffer-size 536870912 "
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
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-decode-log-interval 1000 " "--sglang-enable-metrics "

    ci_args = "--ci-test "

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{fsdp_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
