# [MODULE DOC] End-to-end test of by-rollout accounting on Qwen2.5-0.5B-Instruct with 2 colocated
# GPUs in bridge mode: a custom "compact split" generate function (from the manual tests package)
# splits each rollout execution into two training samples sharing a rollout id, and training must
# keep siblings in one step, count steps in rollouts and weight the loss per rollout (see the
# existing docstring). Rollout: GSM8K, math reward, 2 rollouts of 8 prompts times 4 samples,
# 1024 tokens, temperature 0.8, global batch 32; eval flags with interval 20 (never reached in 2
# rollouts); all parallel sizes 1; SGLang one GPU per engine, mem fraction 0.7; CI test flag.
# Registered nightly-only for CUDA 2-GPU H200 (est. 400 s) and ROCm 2-GPU MI350 nightly (est.
# 400 s). Run as a script: prepare, strip proxy variables, execute.
"""E2E for by-rollout accounting: a compact generate fn splits every rollout
execution into two training samples sharing ``rollout_id``; the rollout-side
schedule must keep siblings in one step, count steps in rollouts, and train
with per-rollout loss weighting."""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=400, suite="stage-c-2-gpu-h200", labels=["short"], hardware=["hopper", "blackwell"], nightly=True
)
register_rocm_ci(est_time=400, suite="nightly-stage-c-2-gpu-mi350", labels=["short"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories and downloads the Qwen2.5-0.5B-Instruct
    # model and the GSM8K dataset. No checkpoint conversion (bridge mode). Side effects: shell
    # commands, network, filesystem.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    # [FUNCTION DOC] Builds the argument string and launches training with 2 GPUs. Uses the
    # Hugging Face directory as both checkpoint and reference load; the custom generate function
    # path points at the compact split generator; GRPO with KL loss at zero coefficient, Adam at
    # 1e-6, dynamic batching at 9216 tokens per GPU, colocated, bridge mode. The CI test flag is
    # appended inline. Returns nothing.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--custom-generate-function-path tests.manual.compact_split_generate.generate "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 "

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
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        "--ci-test "
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
