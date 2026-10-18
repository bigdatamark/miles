# [MODULE DOC] End-to-end test of training offload (to CPU) in a non-colocated layout for
# Qwen3-4B: 4 training GPUs (tensor parallel 2) and 4 separate rollout GPUs as single-GPU engines,
# 8 GPUs in total. The CI flag requiring engines to verify received weights equal the trained
# weights is on. The only assertions are those inside training. Registered for CUDA 8-GPU H100
# (est. 400 s) and ROCm 8-GPU MI350 nightly (est. 300 s) with two CI metric gates (grad norm and
# raw reward). Run as a script: prepare then execute.
"""E2E test for --offload-train with the rollout engines on separate GPUs.

Four training GPUs and four rollout GPUs, no --colocate. The actor is asleep at the
end of init, so the very first weight sync runs against an offloaded trainer: the
connection setup must not allocate through paused memory-saver blocks, and the
weights must come from the host backup rather than the unmapped param buffers.
--check-weight-update-equal makes the engines verify what they received.
"""

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
TRAIN_GPUS = 4
ROLLOUT_GPUS = 4

register_cuda_ci(
    est_time=400,
    suite="stage-c-8-gpu-h100",
    labels=["megatron", "weight-update"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(est_time=300, suite="nightly-stage-c-8-gpu-mi350", labels=["megatron", "weight-update"])
register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="rollout/raw_reward")


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories, downloads Qwen3-4B and the dapo-math-17k
    # dataset, and converts the model to a Megatron distributed checkpoint using the training GPU
    # count (4). Side effects: shell commands, network, filesystem, GPU.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=TRAIN_GPUS)


def execute():
    # [FUNCTION DOC] Builds the argument string and launches training. Rollout: 2 rollouts of 4
    # prompts times 2 samples, 256 response tokens, temperature 0.8, global batch 8. Tensor
    # parallel 2 with sequence parallel, full recompute, dynamic batching at 2048 tokens per GPU.
    # GRPO with KL loss at zero coefficient, plain Adam. Training offload targets CPU. SGLang: 4
    # rollout GPUs, one per engine, mem fraction 0.6. CI flags: CI test and check-weight-update-
    # equal. No colocation flag, so actor and rollout are disjoint. The actor is asleep at the end
    # of init, so the first weight sync happens against an offloaded trainer. Launches with
    # training plus rollout GPUs (8) per node.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist "
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
        "--rollout-max-response-len 256 "
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
        "--max-tokens-per-gpu 2048 "
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
    offload_args = "--offload-train " "--offload-train-target cpu "
    sglang_args = (
        f"--rollout-num-gpus {ROLLOUT_GPUS} " "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.6 "
    )
    ci_args = "--ci-test " "--check-weight-update-equal "
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {TRAIN_GPUS} "
    )
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{offload_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )
    U.execute_train(train_args=train_args, num_gpus_per_node=TRAIN_GPUS + ROLLOUT_GPUS, megatron_model_type=MODEL_TYPE)


if __name__ == "__main__":
    prepare()
    execute()
