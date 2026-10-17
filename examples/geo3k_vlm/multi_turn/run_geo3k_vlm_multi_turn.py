# [MODULE DOC] Self-executing launcher for GRPO training of a Qwen3-VL model on the geo3k image
# dataset with the multi-turn scoring-tool rollout of this package, colocated on the local GPUs.
# Configuration is read from environment variables at import time: MILES_SCRIPT_MODEL_NAME
# (default Qwen3-VL-2B-Instruct; must be one of the 2B, 4B, 8B Instruct or Thinking models),
# MILES_SCRIPT_NUM_GPUS (default 4), MILES_SCRIPT_EXTERNAL_RAY (default 0) and
# MILES_SCRIPT_TRAIN_BACKEND (megatron by default, or fsdp); invalid model or backend names fail an
# assertion at import. Paths are fixed under /root. Running the script downloads the model and
# dataset (prepare) and then submits the training job (execute).
# KNOWN ISSUE: the external-Ray setting is read but never used.
import os

from miles.utils.external_utils import command_utils

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-VL-2B-Instruct")
assert MODEL_NAME in {
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Thinking",
}

NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "4"))
EXTERNAL_RAY = int(os.environ.get("MILES_SCRIPT_EXTERNAL_RAY", "0"))
TRAIN_BACKEND = os.environ.get("MILES_SCRIPT_TRAIN_BACKEND", "megatron").lower()
assert TRAIN_BACKEND in {"fsdp", "megatron"}

DATASET_NAME = "VeraIsHere/geo3k_imgurl_processed"
DATA_ROOT = "/root/datasets/geo3k_imgurl_processed"
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, "train.parquet")


def get_megatron_model_type(model_name: str) -> str:
    # [FUNCTION DOC] Maps a vision-language model name to the Megatron text model type name: removes
    # the "-Instruct" and "-Thinking" suffixes, turns the "Qwen3-VL-" prefix into "qwen3-" and the
    # size "-2B" into "-1.7B" (the 2B vision model has a 1.7B text backbone). Example: the 4B
    # Instruct model gives "qwen3-4B". Pure string function.
    model_type = model_name.replace("-Instruct", "").replace("-Thinking", "")
    model_type = model_type.replace("Qwen3-VL-", "qwen3-")
    return model_type.replace("-2B", "-1.7B")


def prepare():
    # [FUNCTION DOC] Creates /root/models and /root/datasets, downloads the model from the Qwen hub
    # organization into /root/models/<model name>, downloads the dataset into its data root if the
    # train parquet file does not exist, and raises a file-not-found error if the train file is still
    # missing afterwards. Side effects: shell commands, network, disk.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    data_missing = not os.path.exists(TRAIN_DATA_PATH)
    if data_missing:
        U.exec_command_cpu(f"hf download --repo-type dataset {DATASET_NAME} --local-dir {DATA_ROOT}")
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Dataset not found. Expected local dataset at {TRAIN_DATA_PATH}; ")


def execute():
    # [FUNCTION DOC] Builds the training arguments and submits the job. Blocks: checkpoint (HF
    # checkpoint at /root/models/<model name>); W&B flags only if WANDB_API_KEY is set (project
    # miles-dev, group geo3k_vlm_multi_turn, key placed on the command line); rollout (train parquet,
    # input key problem, label key answer, image column mapping, math reward type, chat template,
    # this package's multi-turn generate function and its config file, shuffle, 3000 rollouts, batch
    # 64, 8 samples per prompt, response length 4096, temperature 1, global batch 512); GRPO with zero
    # KL and entropy coefficients and clip 0.2 / 0.28 (the KL-loss flag is not passed); Adam optimizer
    # (1e-6 constant, weight decay 0.1, betas 0.9/0.98); SGLang (1 GPU per engine, memory fraction
    # 0.6, a CUDA graph batch size list of 1, 2, 4, 8 and 16 to 256 in steps of 8); backend flags:
    # Megatron (tensor parallel 4 with sequence parallel, other parallel sizes 1, full uniform
    # recompute, dynamic batch 4096 tokens per GPU, flash attention, bridge conversion mode, load from
    # the model dir; also sets the environment variable MODEL_ARGS_ROTARY_BASE to 5000000 in this
    # process and derives the Megatron model type from the model name) or FSDP (gradient
    # checkpointing, fa3 attention, 512 MiB weight update buffer, no Megatron model type); misc (one
    # actor node, GPUs from the environment setting, rollout GPUs equal to the same count, colocated).
    # Passes the W&B key through the job environment if set. Submits via the default backend's
    # execute-train. Evaluation arguments exist only as commented-out code.
    # KNOWN ISSUE: the rollout GPU count is passed together with the colocate flag, although it is
    # ignored when colocated; the tensor parallel size is fixed at 4 regardless of the GPU count.
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    wandb_args = (
        (
            "--use-wandb "
            "--wandb-project miles-dev "
            "--wandb-group geo3k_vlm_multi_turn "
            f"--wandb-key '{wandb_api_key}' "
        )
        if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
        else ""
    )

    rollout_args = (
        f"--prompt-data {TRAIN_DATA_PATH} "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--rm-type math "
        "--apply-chat-template "
        "--custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate "
        "--custom-config-path examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml "
        "--rollout-shuffle "
        "--num-rollout 3000 "
        "--rollout-batch-size 64 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 512 "
    )

    # eval_args = (
    #     "--eval-interval 20 "
    #     f"--eval-prompt-data geo3k_eval {TRAIN_DATA_PATH}@[0:64] "
    #     "--n-samples-per-eval-prompt 1 "
    #     "--eval-max-response-len 4096 "
    #     "--eval-top-k 1 "
    # )

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
        f"--sglang-cuda-graph-bs-decode {' '.join(map(str, [1, 2, 4, 8] + list(range(16, 257, 8))))} "
    )

    fsdp_args = (
        "--train-backend fsdp "
        "--gradient-checkpointing "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
        "--update-weight-buffer-size 536870912 "
    )

    megatron_args = (
        "--train-backend megatron "
        f"--load /root/models/{MODEL_NAME} "
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--megatron-to-hf-mode bridge "
    )

    misc_args = (
        "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " f"--rollout-num-gpus {NUM_GPUS} " "--colocate "
    )

    if TRAIN_BACKEND == "megatron":
        backend_args = megatron_args
        megatron_model_type = get_megatron_model_type(MODEL_NAME)
        os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    else:
        backend_args = fsdp_args
        megatron_model_type = None

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{backend_args} "
        f"{misc_args} "
        f"{wandb_args} "
        # f"{get_default_wandb_args(__file__)} "
    )

    command_utils.default_config().create_backend().execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=megatron_model_type,
        extra_env_vars=({"WANDB_API_KEY": os.environ["WANDB_API_KEY"]} if os.environ.get("WANDB_API_KEY") else {}),
    )


if __name__ == "__main__":
    prepare()
    execute()
