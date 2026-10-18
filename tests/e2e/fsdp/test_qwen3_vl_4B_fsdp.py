# [MODULE DOC]
# End-to-end script for FSDP training of the vision-language model Qwen3-VL-4B-Instruct on the
# geometry (geo3k) image-URL dataset with 2 GPUs (colocated). Registered as an 800-second 2-GPU
# H200 CUDA test (hopper only, fsdp label). Module settings: an evaluation switch from an
# environment variable (default on), 2 GPUs, model and dataset names. As a script: download model
# and dataset, strip proxy variables one by one, then train 2 rollouts of 8 prompts with 8 samples
# (up to 4096 tokens, global batch 32) with GRPO, gradient checkpointing, multimodal image keys
# and CI test mode.
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

MODEL_NAME = "Qwen3-VL-4B-Instruct"
DATASET_NAME = "chenhegu/geo3k_imgurl"


def prepare():
    # [FUNCTION DOC] Through the command backend: create the models and datasets directories,
    # download the vision-language model into the models directory and download the geo3k
    # dataset. Side effects: network downloads, filesystem.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset(DATASET_NAME)


def execute():
    # [FUNCTION DOC] Launch the training job through the backend's execute-train (no Megatron
    # model type) with the extra environment variable setting a single CUDA device connection.
    # Arguments: the HF checkpoint; geo3k rollouts (problem key, answer label, math reward, 2
    # rollouts of 8 prompts with 8 samples, up to 4096 tokens, temperature 1, global batch 32); the
    # multimodal keys mapping "image" to the "images" column; Adam learning rate 1e-6 with weight
    # decay 0.1; GRPO with zero KL-loss, KL and entropy coefficients and clipping 0.2 / 0.28; FSDP
    # backend with gradient checkpointing and a 512 MiB update buffer; evaluation arguments on the
    # geo3k test set (1 sample, 4096 tokens), with the interval flag (every 20 rollouts) only when
    # the evaluation switch is on; SGLang with one GPU per engine, memory fraction 0.6, metrics,
    # fa3 attention backends; CI test mode; one node, 2 GPUs, colocated.
    # KNOWN ISSUE: when the evaluation switch is off the evaluation dataset arguments are still
    # passed; only the interval flag is dropped.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/geo3k_imgurl/train.parquet "
        "--input-key problem "
        "--label-key answer "
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

    # multimodal keys required for vlm datasets
    multimodal_args = '--multimodal-keys \'{"image": "images"}\' '

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        "--eval-prompt-data geo3k /root/datasets/geo3k_imgurl/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 4096 "
    )

    fsdp_args = "--train-backend fsdp " "--gradient-checkpointing " "--update-weight-buffer-size 536870912 "

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
        "--sglang-mem-fraction-static 0.6 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
    )

    ci_args = "--ci-test "

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{multimodal_args} "
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
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    execute()
