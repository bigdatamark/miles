# [MODULE DOC]
# End-to-end script checking that the FSDP and Megatron training backends compute the same
# gradient norm on identical data, for Qwen3-0.6B on 2 GPUs (colocated). Registered as a
# 900-second 2-GPU H200 CUDA test (hopper and blackwell, fsdp label) and an 800-second nightly
# 2-GPU MI350 ROCm test. Module constants: model name and Megatron type, 2 GPUs, context parallel
# 1, Megatron tensor and pipeline parallel 1. As a script: prepare (download model and math
# dataset, convert to Megatron format), strip proxy variables, then execute three training jobs:
# (1) FSDP generates and saves the rollout debug data; (2) FSDP trains only on that saved data
# and saves its gradient norm; (3) Megatron trains only on the same data and checks its gradient
# norm against the saved one. Temporary files (gradient norm file, debug data for rollout 0) are
# removed afterwards even on failure.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=900,
    suite="stage-c-2-gpu-h200",
    labels=["fsdp"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=800,
    suite="nightly-stage-c-2-gpu-mi350",
    labels=["fsdp"],
)


MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 2
CP_SIZE = 1
MEGATRON_TP_SIZE = 1
MEGATRON_PP_SIZE = 1


def prepare():
    # [FUNCTION DOC] Through the command backend: create the models and datasets directories,
    # download the model and the math training dataset, and convert the Hugging Face checkpoint to
    # the Megatron distributed format across 2 GPUs into the models directory. Side effects:
    # network downloads, filesystem, GPU job.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
    )


def execute():
    # [FUNCTION DOC] Run three training jobs sharing a common argument string (HF checkpoint, one
    # rollout of 8 prompts with 8 samples on the math dataset with deepscaler reward and up to 8192
    # tokens, global batch 64, dynamic batch with 8192 tokens per GPU, GRPO with zero coefficients
    # and clip 4e-4, Adam learning rate 1e-6, one rollout GPU per engine with chunked prefill 4096
    # and memory fraction 0.75, CI test mode, colocated on 2 GPUs). Job 1 (FSDP backend with
    # flash_attention_2, gradient checkpointing, context parallel 1, 512 MiB update buffer,
    # expandable-segments allocator in the training process) also saves the rollout debug data to a
    # templated file. Job 2 (same FSDP settings) loads that data, saves its gradient norm to a
    # file and runs train-only. Job 3 (Megatron backend with tensor, pipeline, context parallel 1,
    # sequence parallel, full recompute, a 3 GiB training memory margin, flash attention, fp32
    # accumulation, dropout off) loads the same data, checks against the saved gradient norm and
    # skips the actor forward-only pass, train-only. In a finally block, deletes the gradient norm
    # file and the rollout-0 debug data file if they exist. Side effects: three GPU jobs, files in
    # the working directory.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/"

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
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
        "--rollout-num-gpus-per-engine 1 " "--sglang-chunked-prefill-size 4096 " "--sglang-mem-fraction-static 0.75 "
    )

    ci_args = "--ci-test "

    misc_args = "--actor-num-nodes 1 " "--colocate " f"--actor-num-gpus-per-node {NUM_GPUS} "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    debug_data_path = "test_rollout_data_megatron_fsdp_align_{rollout_id}.pt"
    grad_norm_path = "grad_norm_fsdp.pt"

    fsdp_args = (
        "--train-backend fsdp "
        "--attn-implementation flash_attention_2 "
        "--gradient-checkpointing "
        f"--context-parallel-size {CP_SIZE} "
        f"--update-weight-buffer-size {512 * 1024 * 1024} "
        """--train-env-vars '{"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True"}' """
    )

    try:
        U.execute_train(
            train_args=train_args + (f"{fsdp_args}" f"--save-debug-rollout-data {debug_data_path} "),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=None,
        )

        U.execute_train(
            train_args=train_args
            + (
                f"{fsdp_args}"
                f"--load-debug-rollout-data {debug_data_path} "
                f"--ci-save-grad-norm {grad_norm_path} "
                "--debug-train-only "
            ),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=None,
        )

        U.execute_train(
            train_args=train_args
            + (
                f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
                f"--tensor-model-parallel-size {MEGATRON_TP_SIZE} "
                "--sequence-parallel "
                f"--pipeline-model-parallel-size {MEGATRON_PP_SIZE} "
                f"--context-parallel-size {CP_SIZE} "
                "--expert-model-parallel-size 1 "
                "--expert-tensor-parallel-size 1 "
                "--recompute-granularity full "
                "--recompute-method uniform "
                "--recompute-num-layers 1 "
                "--train-memory-margin-bytes 3221225472 "
                f"--load-debug-rollout-data {debug_data_path} "
                f"--ci-load-grad-norm {grad_norm_path} "
                "--skip-actor-forward-only "
                "--attention-dropout 0.0 "
                "--hidden-dropout 0.0 "
                "--accumulate-allreduce-grads-in-fp32 "
                "--attention-softmax-in-fp32 "
                "--attention-backend flash "
                "--debug-train-only "
            ),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )

    finally:
        if os.path.exists(grad_norm_path):
            os.remove(grad_norm_path)
        debug_data_file = debug_data_path.format(rollout_id=0)
        if os.path.exists(debug_data_file):
            os.remove(debug_data_file)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
