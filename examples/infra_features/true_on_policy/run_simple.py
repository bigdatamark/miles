# [MODULE DOC] Script (no command line parser; configured by environment variables read at
# import time) that trains Qwen3-0.6B or Qwen3-4B on GSM8K with the FSDP backend colocated on
# the GPUs, in "true on policy" mode: deterministic inference and training kernels (FA3
# attention on both sides, deterministic NCCL and cuBLAS settings) so training log probabilities
# match inference exactly. Environment: model name (Qwen3-0.6B default, or Qwen3-4B), mode
# (normal, debug_minimal, debug_one_sample; asserted at import) and GPU count (default 1). Run
# directly: prepare downloads the model and dataset, execute submits the job. Modes: normal runs
# 3000 rollouts of 32 prompts x 8 samples with periodic eval; debug_one_sample runs two rollouts
# of one sample with response length 2, disables CUDA graphs and turns on the SGLang dumper and
# debug prints; debug_minimal differs from normal only by having no eval.
import os


from miles.utils.external_utils import command_utils

MODEL_NAME = os.environ.get("MILES_SCRIPT_MODEL_NAME", "Qwen3-0.6B")
assert MODEL_NAME in {"Qwen3-0.6B", "Qwen3-4B"}

MODE = os.environ.get("MILES_SCRIPT_MODE", "normal")
assert MODE in {"normal", "debug_minimal", "debug_one_sample"}

NUM_GPUS = int(os.environ.get("MILES_SCRIPT_NUM_GPUS", "1"))


def prepare():
    # [FUNCTION DOC] Create the models and datasets dirs under the root dir, download the chosen
    # Qwen model into the models dir and the GSM8K dataset (default data dir) via the default
    # backend. No return.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    # [FUNCTION DOC] Build the training arguments and submit with no Megatron model type (FSDP).
    # Rollout: GSM8K train parquet with the math reward, chat template, shuffle; sizes depend on
    # mode (see module doc), response length 1024, temperature 1, global batch 256 (1 in one-sample
    # mode). Eval (normal mode only): every 20 rollouts on GSM8K test, one sample, response 1024,
    # top-k 1. GRPO with zero KL and entropy coefficients (KL loss itself disabled), clip
    # 0.2/0.28. Adam 1e-6 constant. Sglang: one GPU per engine, metrics on, static memory
    # fraction 0.2 for the 4B model and 0.4 otherwise, CUDA graphs off in one-sample mode.
    # FSDP: weight update bucket of 512 MiB. CI flags: CI test on, KL checker off, and a metric
    # check that the gsm8k eval metric reaches 0.71. Misc: one actor node with the GPU count,
    # colocated, FSDP backend; the 4B model additionally uses dynamic batch with 2048 tokens per
    # GPU. True on policy: deterministic sglang inference with the FSDP RL target, FA3 attention in
    # both sglang and the trainer, deterministic and true-on-policy modes; environment pins the
    # NCCL algorithm to tree all-reduce, disables nondeterministic Transformer Engine algorithms
    # and sets the cuBLAS workspace config; sglang dumper and debug print are enabled only in
    # one-sample mode.
    # KNOWN ISSUE: debug modes without eval (debug_minimal, debug_one_sample) still enable the
    # CI metric check on the gsm8k eval metric, which would have no value to check.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {2 if MODE == 'debug_one_sample' else 3000} "
        f"--rollout-batch-size {1 if MODE == 'debug_one_sample' else 32} "
        f"--n-samples-per-prompt {1 if MODE == 'debug_one_sample' else 8} "
        f"--rollout-max-response-len {2 if MODE == 'debug_one_sample' else 1024} "
        "--rollout-temperature 1 "
        # temp remove this to make test easier
        # "--over-sampling-batch-size 64 "
        # "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        f"--global-batch-size {1 if MODE == 'debug_one_sample' else 256} "
    )

    eval_args = ""
    if MODE == "normal":
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

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--sglang-decode-log-interval 1000 "
        "--sglang-enable-metrics "
        f"--sglang-mem-fraction-static {0.2 if MODEL_NAME == 'Qwen3-4B' else 0.4} "
        f"{'--sglang-disable-cuda-graph ' if MODE == 'debug_one_sample' else ''}"
    )

    fsdp_args = (
        # Set to true for FULL_STATE_DICT mode, false for SHARDED_STATE_DICT mode (default)
        # "--fsdp-full-params "  # Uncomment this line to enable full params mode
        # Set the bucket size for weight update
        "--update-weight-buffer-size 536870912 "  # 512MB
    )

    ci_args = (
        "--ci-test "
        "--ci-disable-kl-checker "
        "--ci-metric-checker-key eval/gsm8k "
        "--ci-metric-checker-threshold 0.71 "  # loose threshold at 60 step
    )

    misc_args = "--actor-num-nodes 1 " f"--actor-num-gpus-per-node {NUM_GPUS} " "--colocate " "--train-backend fsdp "

    if MODEL_NAME == "Qwen3-4B":
        misc_args += (
            "--use-dynamic-batch-size "
            # TODO pick a good value
            "--max-tokens-per-gpu 2048 "
        )

    true_on_policy_args = (
        "--sglang-enable-deterministic-inference "
        "--sglang-rl-on-policy-target fsdp "
        "--sglang-attention-backend fa3 "
        "--attn-implementation flash_attention_3 "
        "--deterministic-mode "
        "--true-on-policy-mode "
    )
    true_on_policy_envs = {
        # TODO note: "Ring" in original RL PR, "allreduce:tree" in SGLang
        # "NCCL_ALGO": "Ring",
        "NCCL_ALGO": "allreduce:tree",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{eval_args} "
        f"{fsdp_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{true_on_policy_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=None,
        extra_env_vars={
            **true_on_policy_envs,
            "SGLANG_DUMPER_ENABLE": "1" if MODE == "debug_one_sample" else "0",
            "SGLANG_TEMP_UTILS_ENABLE_DEBUG_PRINT": "1" if MODE == "debug_one_sample" else "0",
        },
    )


if __name__ == "__main__":
    prepare()
    execute()
