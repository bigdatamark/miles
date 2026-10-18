# [MODULE DOC]
# End-to-end script for FSDP true-on-policy training of Qwen3-4B on 2 GPUs (colocated) with
# deterministic inference. Registered as an 800-second 2-GPU H200 CUDA test (hopper only, fsdp
# label). Module settings: an evaluation switch from an environment variable (default on) and
# 2 GPUs. As a script: download the model, the math training dataset and AIME 2024, strip proxy
# variables, train 2 rollouts of 8 prompts with 8 samples (up to 4096 tokens, global batch 32) with
# GRPO, CI test mode and the true-on-policy settings.
import os

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=800,
    suite="stage-c-2-gpu-h200",
    labels=["fsdp"],
    hardware=["hopper"],
)

ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))
NUM_GPUS = 2

MODEL_NAME = "Qwen3-4B"


def prepare():
    # [FUNCTION DOC] Through the command backend: create the models and datasets directories,
    # download the Qwen3-4B model into the models directory and download the math training dataset
    # and the AIME 2024 dataset. Side effects: network downloads, filesystem.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")


def execute():
    # [FUNCTION DOC] Launch the training job through the backend's execute-train (no Megatron
    # model type) with extra environment variables (tree allreduce algorithm, non-deterministic
    # algorithms disallowed, fixed cuBLAS workspace configuration, single CUDA device connection).
    # Arguments: the HF checkpoint; math rollouts (2 rollouts of 8 prompts with 8 samples, up to
    # 4096 tokens, temperature 1, global batch 32); Adam learning rate 1e-6 with weight decay 0.1;
    # GRPO with zero KL-loss, KL and entropy coefficients and clipping 0.2 / 0.28; FSDP backend with
    # a 512 MiB update buffer; evaluation arguments on the AIME 2024 dataset (1 sample, 4096 tokens,
    # top-p 0.7); SGLang with one GPU per engine, metrics, deterministic inference, the dense
    # Qwen3 true-on-policy contract, fa3 attention backends and deterministic plus true-on-policy
    # modes; CI test mode; one node, 2 GPUs, colocated.
    # KNOWN ISSUE: the evaluation switch only controls the evaluation interval flag; when it is off
    # the evaluation dataset arguments are still passed, so evaluation arguments are never fully
    # removed.
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

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data aime /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 4096 "
        "--eval-top-p 0.7 "
    )

    fsdp_args = "--train-backend fsdp " "--update-weight-buffer-size 536870912 "

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

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-enable-deterministic-inference "
        "--sglang-true-on-policy-contract qwen3_dense_true_on_policy_v1 "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
        "--deterministic-mode "
        "--true-on-policy-mode "
    )

    ci_args = "--ci-test "

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{fsdp_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    extra_env_vars = {
        "NCCL_ALGO": "allreduce:tree",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    }

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars=extra_env_vars,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
