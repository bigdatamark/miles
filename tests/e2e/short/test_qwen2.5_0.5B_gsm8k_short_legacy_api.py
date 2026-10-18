# [MODULE DOC] End-to-end compatibility test of the legacy module-level command utilities API (used
# through a module alias instead of a created backend object) on Qwen2.5-0.5B-Instruct with GSM8K,
# 8 GPUs colocated (4 when a few-GPU environment flag is set), bridge mode, Mooncake object store
# arguments with the Mooncake master started before the Ray job is submitted, and fault tolerance.
# Rollout: 3 rollouts of 8 prompts times 4 samples, oversampling 16, non-zero reward std filter,
# 1024 tokens, global batch 32; eval flags use interval 20 (never reached in 3 rollouts).
# Registered for CUDA 8-GPU H100 (est. 400 s) and ROCm 8-GPU MI350 nightly (est. 300 s), labels
# short and mooncake. A todo notes it should run on 4 GPUs. Run as a script: prepare, strip proxy
# variables, execute.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

import miles.utils.external_utils.command_utils.legacy as U

# TODO: Run this compatibility test on 4 GPUs instead of 8.
register_cuda_ci(
    est_time=400,
    suite="stage-c-8-gpu-h100",
    labels=["short", "mooncake"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(est_time=300, suite="nightly-stage-c-8-gpu-mi350", labels=["short", "mooncake"])

FEW_GPU = U.get_bool_env_var("MILES_TEST_FEW_GPU", "0")

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4 if FEW_GPU else 8


def prepare():
    # [FUNCTION DOC] Using the legacy helpers: creates model/dataset directories and downloads the
    # Qwen2.5-0.5B-Instruct model and GSM8K. Side effects: shell commands, network, filesystem.
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    # [FUNCTION DOC] Builds the argument string and launches training through the legacy API with
    # the module-level GPU count (4 or 8), colocated. Mooncake object store arguments come right
    # after the checkpoint arguments. Settings: GRPO with KL loss at zero coefficient, Adam at
    # 1e-6, all parallel sizes 1 with sequence parallel, dynamic batching at 9216 tokens per GPU,
    # SGLang one GPU per engine with mem fraction 0.7 and metrics, CI test flag, fault tolerance
    # flags, bridge mode. A hook starts the Mooncake master before the Ray job is submitted.
    # Returns nothing.
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-interval 20 "
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
        "--max-tokens-per-gpu 9216 "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 " "--sglang-enable-metrics "

    ci_args = "--ci-test "

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
    )

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
        f"{U.get_mooncake_object_store_args()} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{fault_tolerance_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        before_ray_job_submit=U.start_mooncake_master,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
