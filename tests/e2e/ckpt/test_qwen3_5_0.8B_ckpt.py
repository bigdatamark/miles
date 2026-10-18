# [MODULE DOC]
# End-to-end checkpoint save/load script for Qwen3.5-0.8B on 2 GPUs (Megatron backend).
# Registered as a 1400-second 2-GPU H200 CUDA test (hopper and blackwell, ckpt label) and a
# 1000-second nightly 2-GPU MI350 ROCm test. Same flow as the Qwen3-0.6B checkpoint script:
# prepare the model, strip proxy variables, then run synchronous save, load, asynchronous save
# and load again, with the model hash saved and later checked. Differences: one rollout engine
# per GPU, and the weight-update check skips parameters named "visual" because the training side
# has no Qwen3.5 vision implementation, so vision weights are never synced. Module settings:
# evaluation switch from an environment variable (default on), model name and type, 2 GPUs,
# container-local save directory under the checkpoints root.
# KNOWN ISSUE: the evaluation switch is defined but never used, so no evaluation arguments are
# ever passed (the evaluation dataset is still downloaded in the prepare step).
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=1400, suite="stage-c-2-gpu-h200", labels=["ckpt"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=1000, suite="nightly-stage-c-2-gpu-mi350", labels=["ckpt"])

ENABLE_EVAL = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen3.5-0.8B"
MODEL_TYPE = "qwen3.5-0.8B"
NUM_GPUS = 2
# Container-local: /root/models is a host directory shared by every runner on the host.
SAVE_DIR = f"/root/checkpoints/{MODEL_NAME}_miles"


def _get_latest_checkpointed_iteration() -> int:
    # [FUNCTION DOC] HELPER: Read the latest-checkpointed-iteration marker file in the save
    # directory, strip whitespace, and return the iteration number as an integer. Raises a value
    # error if the text is not all digits; a missing file raises the usual file-not-found error.
    latest_path = f"{SAVE_DIR}/latest_checkpointed_iteration.txt"
    with open(latest_path, encoding="utf-8") as f:
        latest_text = f.read().strip()
    if not latest_text.isdigit():
        raise ValueError(f"Invalid latest checkpoint value: {latest_text}")
    return int(latest_text)


def prepare():
    # [FUNCTION DOC] Prepare the Qwen3.5-0.8B run. Through the command backend: create the models
    # and datasets directories, download the model into the models directory, delete the save
    # directory, download the math training dataset and the AIME 2024 dataset, then convert the
    # Hugging Face checkpoint to the Megatron distributed format across 2 GPUs into the models
    # directory. Side effects: network downloads, filesystem changes, GPU job.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command_cpu(f"rm -rf {SAVE_DIR}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute(mode: str = "", ckpt_step: int | None = None):
    # [FUNCTION DOC] Run one training job in the given checkpoint mode. Mode "save": save every
    # step to the save directory. Mode "async_save": same plus asynchronous save with a persistent
    # checkpoint worker. Mode "load": load from the save directory at the given checkpoint step
    # (required; interpolated as-is) with low-memory resume. Any other mode (default empty
    # string): only the reference checkpoint is loaded and nothing is saved. Fixed settings: 2
    # rollouts of batch 4 with 2 samples per prompt on the math dataset (deepscaler reward,
    # response up to 1024 tokens, temperature 0.8), global batch 8, tensor parallel 2 with
    # sequence parallel, full recompute, dynamic batch with 16384 tokens per GPU, GRPO with zero
    # KL and entropy coefficients and clip 0.2, Adam learning rate 1e-6, rollout engines of 1 GPU
    # each with 0.7 memory fraction and a list of decode CUDA graph batch sizes, dropout off,
    # fp32 gradient accumulation and softmax, flash attention backend, colocated on 1 node. CI
    # test mode is on with a weight-update skip list of "visual"; save modes also save the model
    # hash and load mode checks it. Launched through the backend's execute-train.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    if mode == "save":
        ckpt_args += f"--save {SAVE_DIR} "
        ckpt_args += "--save-interval 1 "
    elif mode == "async_save":
        ckpt_args += f"--save {SAVE_DIR} "
        ckpt_args += "--save-interval 1 "
        ckpt_args += "--async-save "
        ckpt_args += "--use-persistent-ckpt-worker "
    elif mode == "load":
        ckpt_args += f"--load {SAVE_DIR} "
        ckpt_args += f"--ckpt-step {ckpt_step} "
        ckpt_args += "--low-memory-resume "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
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
        "--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.7 --sglang-cuda-graph-bs-decode 1 2 4 8 16 "
    )

    # miles has no Qwen3.5 vision implementation on the training side, so vision weights are never synced.
    ci_args = "--ci-test --check-weight-update-skip-list visual "
    if mode in {"save", "async_save"}:
        ci_args += "--ci-save-model-hash "
    if mode == "load":
        ci_args += "--ci-check-model-hash "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
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
    execute("save")
    execute("load", ckpt_step=_get_latest_checkpointed_iteration())
    execute("async_save")
    execute("load", ckpt_step=_get_latest_checkpointed_iteration())
