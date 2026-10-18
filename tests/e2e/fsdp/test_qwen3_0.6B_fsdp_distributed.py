# [MODULE DOC]
# End-to-end script for a long FSDP distributed training run of Qwen3-0.6B on GSM8K using 4 GPUs
# (2 for training, 2 for rollout, not colocated, asynchronous training script) with periodic
# evaluation and a metric threshold check. Registered as a 2200-second 4-GPU H200 CUDA test
# (hopper and blackwell, long label) and a 2400-second nightly 4-GPU MI350 ROCm test. As a script
# it downloads the model and dataset, strips proxy variables, then trains: 65 rollouts (3000 when
# the infinite-run environment switch is on; the count is chosen not to be an exact multiple of the
# evaluation interval because the asynchronous mode causes offsets), evaluating every 20 rollouts
# on GSM8K test with greedy decoding, and requires the GSM8K evaluation score to reach a loose 0.71
# threshold.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=2200,
    suite="stage-c-4-gpu-h200",
    labels=["long"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=2400,
    suite="nightly-stage-c-4-gpu-mi350",
    labels=["long"],
)

MODEL_NAME = "Qwen3-0.6B"
NUM_GPUS = 4


def prepare():
    # [FUNCTION DOC] Through the command backend: create the models and datasets directories,
    # download the model into the models directory and download the GSM8K dataset. Side effects:
    # network downloads, filesystem.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    # [FUNCTION DOC] Launch the training job through the backend's execute-train with no Megatron
    # model type and the asynchronous training script. Arguments: the HF checkpoint; GSM8K
    # rollouts (math reward, 65 rollouts or 3000 in infinite-run mode, batch 32, 8 samples, up to
    # 1024 tokens, temperature 1, oversampling 64 with a non-zero-reward-spread filter, global
    # batch 256); evaluation every 20 rollouts on GSM8K test (1 sample, 1024 tokens, top-k 1);
    # Adam learning rate 1e-6 with weight decay 0.1; GRPO with zero KL-loss, KL and entropy
    # coefficients and clipping 0.2 / 0.28; one rollout GPU per engine with SGLang metrics; half the
    # GPUs (2) for training and half for rollout, FSDP backend; CI test mode with the KL checker
    # disabled and a metric check that "eval/gsm8k" reaches 0.71.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        # NOTE cannot be exactly multiple of eval-interval, since async causes some offsets
        f"--num-rollout {3000 if command_utils.get_env_enable_infinite_run() else 65} "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 64 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 256 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
        "--eval-top-k 1 "
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
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{eval_args} "
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
