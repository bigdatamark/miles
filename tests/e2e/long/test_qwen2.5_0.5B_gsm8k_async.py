# [MODULE DOC] Long end-to-end GRPO training test on gsm8k with Qwen2.5-0.5B-Instruct using the
# asynchronous train script with separate (non-colocated) training and rollout GPUs on 2 GPUs
# (CUDA: 2-GPU H200 stage, estimated 5400 seconds, long label, hopper and blackwell; ROCm:
# nightly 2-GPU MI350 suite, estimated 5400 seconds). Run as a script: prepare downloads model
# and data into fixed root-level directories, strips all proxy environment variables (both
# cases), then execute launches training. The run passes only if the built-in CI metric checker
# finds eval accuracy on gsm8k at least 0.55.

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=5400, suite="stage-c-2-gpu-h200", labels=["long"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=5400, suite="nightly-stage-c-2-gpu-mi350", labels=["long"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2


def prepare():
    # [FUNCTION DOC] Creates fixed model and dataset directories under the filesystem root,
    # downloads the Qwen2.5-0.5B-Instruct model into the model directory, and downloads the gsm8k
    # dataset (default location) through the default config's backend.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    # [FUNCTION DOC] Builds the training arguments and launches the asynchronous train script on
    # 2 GPUs. Same recipe as the synchronous gsm8k test (math reward, 250 rollouts or 3000 under
    # the infinite-run switch, batch 32 times 8, response length 1024, temperature 1, GRPO with
    # zero-coefficient KL loss, Adam learning rate 1e-6, eval every 20 rollouts, gsm8k accuracy
    # threshold 0.55, SGLang one GPU per engine, bridge mode) except: model and data paths are
    # fixed root-level directories, there is no colocation, and GPUs are split: the actor gets
    # one GPU (the larger of 1 and a quarter of the GPUs) and rollout gets one GPU (the larger of
    # 1 and three quarters of the GPUs, rounded down).
    # KNOWN ISSUE: with 2 GPUs the formulas give 1 actor plus 1 rollout GPU; the "quarter/three
    # quarters" intent only holds for larger GPU counts.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {3000 if command_utils.get_env_enable_infinite_run() else 250} "
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

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        # "--micro-batch-size 1 "
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

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.55 "  # loose threshold at 250 step
    )

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
        f"--actor-num-gpus-per-node {max(1, NUM_GPUS // 4)} "
        f"--rollout-num-gpus {max(1, NUM_GPUS * 3 // 4)} "
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
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
