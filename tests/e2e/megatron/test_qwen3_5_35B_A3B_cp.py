# [MODULE DOC] End-to-end test of context parallelism for Qwen3.5-35B-A3B on 8 colocated GPUs
# (tensor 1, pipeline 1, expert parallel = 8 divided by context size, sequence parallel). One
# training run is launched with a context-parallel size read from an environment variable
# (default 2) when run as a script; two thin wrappers exist for sizes 2 and 4. The run uses
# MTP training (one layer, loss scale 0.2), EAGLE speculative decoding with spec v2 (needs the
# extra-buffer mamba radix cache strategy), one 8-GPU SGLang engine with expert parallel 8, Adam
# with CPU offload, 2 rollouts of 8 prompts times 8 samples (8192 tokens), and AIME-2024 eval
# flags without an interval. Registered for CUDA 8-GPU H100 (est. 900 s) but currently disabled as
# flaky; there is no registered CI metric gate. Module constants: model name/type and 8 GPUs.
# Run as a script: prepare, strip proxy variables, then run with the chosen context size.
# KNOWN ISSUE: a fixme comment above the registration marks it as unfixed, and the disabled reason
# cites no issue or PR. The two wrapper functions are not called from the script entry point.
"""E2E test for Qwen3.5-35B-A3B with Context Parallel (CP=2 and CP=4).

Validates that GDN layers use real fla native CP (state passing) instead of
duplicated all-gather computation.  See: https://github.com/radixark/miles/issues/878
"""

import os

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils

# FIXME: fix this
register_cuda_ci(
    est_time=900,
    suite="stage-c-8-gpu-h100",
    labels=["megatron"],
    hardware=["hopper", "blackwell"],
    disabled="Flaky; temporarily disabled to validate PR correctness",
)

MODEL_NAME = "Qwen3.5-35B-A3B"
MODEL_TYPE = "qwen3.5-35B-A3B"
NUM_GPUS = 8


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories, downloads the Qwen3.5-35B-A3B model and the
    # dapo-math-17k and aime-2024 datasets, and converts the model to a Megatron distributed
    # checkpoint on 8 GPUs. Side effects: shell commands, network, filesystem, GPU.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def _execute_with_cp(cp_size: int):
    # [FUNCTION DOC] HELPER: Launches one training run with the given context-parallel size.
    # Asserts (assertion error) that 8 GPUs divide evenly by the size; expert parallel size is the
    # quotient. Argument groups: rollout (2 rollouts, 8 prompts times 8 samples, 8192 tokens,
    # temperature 1, global batch 32, balanced); eval flags for AIME-2024 (no interval, so eval is
    # not scheduled); performance (tensor 1, pipeline 1, full recompute, dynamic batching at 8192
    # tokens per GPU); GRPO with zero-coefficient KL loss; Adam with CPU offload; SGLang (one
    # 8-GPU engine, mem fraction 0.7, expert parallel 8, 512 max running requests, EAGLE with 2
    # steps, top-k 1, 3 draft tokens, extra-buffer mamba radix cache); MTP training; CI test flag;
    # colocate, flex token dispatcher, parameter rematerialization. Sets the spec v2 environment
    # variable for the launch, which uses 8 GPUs.
    """Run a short training loop with the given context-parallel size."""
    U = command_utils.default_config().create_backend()
    assert NUM_GPUS % cp_size == 0
    ep_size = NUM_GPUS // cp_size

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {cp_size} "
        f"--expert-model-parallel-size {ep_size} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-ep-size {NUM_GPUS} "
        "--sglang-max-running-requests 512 "
        "--sglang-speculative-algorithm EAGLE "
        "--sglang-speculative-num-steps 2 "
        "--sglang-speculative-eagle-topk 1 "
        "--sglang-speculative-num-draft-tokens 3 "
        # SGLang requires extra_buffer + SGLANG_ENABLE_SPEC_V2=1 to combine
        # speculative decoding with radix cache on Qwen3.5MoE; the prod
        # script run_qwen3_5_35b_a3b_mtp.py already pairs these two.
        "--sglang-mamba-radix-cache-strategy extra_buffer "
    )

    mtp_args = "--enable-mtp-training " "--mtp-num-layers 1 " "--mtp-loss-scaling-factor 0.2 "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--moe-token-dispatcher-type flex "
        "--rematerialize-param-from-master-weight "
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
        f"{mtp_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={"SGLANG_ENABLE_SPEC_V2": "1"},
    )


def execute_cp2():
    # [FUNCTION DOC] Runs the shared launcher with context-parallel size 2 (expert parallel 4).
    """Qwen3.5-35B-A3B with CP=2."""
    _execute_with_cp(cp_size=2)


def execute_cp4():
    # [FUNCTION DOC] Runs the shared launcher with context-parallel size 4 (expert parallel 2).
    """Qwen3.5-35B-A3B with CP=4."""
    _execute_with_cp(cp_size=4)


if __name__ == "__main__":
    cp_size = int(os.environ.get("CP_SIZE", "2"))
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    _execute_with_cp(cp_size)
