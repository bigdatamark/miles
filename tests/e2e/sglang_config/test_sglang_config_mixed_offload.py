# [MODULE DOC] End-to-end test of the inline SGLang config with two model deployments sharing 4
# colocated GPUs on Qwen2.5-0.5B (bridge mode): an "actor" entry (2 GPUs, weight updates on) and a
# frozen "ref" entry (2 GPUs, weight updates off), both overlapping with the training actor and
# therefore offloaded. The YAML is passed as a pseudo file. The test runs 2 short GRPO rollouts on
# GSM8K and passes if the offload and onload cycle completes (see the existing docstring for the
# behaviors covered). Registered nightly-only for CUDA 4-GPU H200 (est. 400 s) and for ROCm 4-GPU
# MI350 nightly (est. 300 s). Run as a script: prepare, strip proxy variables, execute.
# KNOWN ISSUE: the eval interval (20) exceeds the number of rollouts (2), so evaluation never runs.
"""E2E test: mixed offload with updatable + frozen models.

Deploys two models via --sglang-config in colocate mode:
  - "actor": update_weights=true, 2 GPUs -> overlaps with megatron, gets offloaded
    and weights updated from training.
  - "ref":   update_weights=false, 2 GPUs -> overlaps with megatron, gets offloaded
    and weights restored from disk (update_weights_from_disk).

Key coverage:
  - Per-group needs_offload (both overlap with megatron in colocate mode)
  - update_weights_from_disk for frozen model
  - Selective flush_cache (only for offloaded / updatable engines)
  - Offload/onload cycle completes without crash
"""

import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=400, suite="stage-c-4-gpu-h200", labels=["short"], hardware=["hopper", "blackwell"], nightly=True
)
register_rocm_ci(est_time=300, suite="nightly-stage-c-4-gpu-mi350", labels=["short"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4

# Two models on 4 GPUs (colocate): actor gets weight updates, ref is frozen.
SGLANG_CONFIG_YAML = """\
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 2
        num_gpus_per_engine: 1
  - name: ref
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 2
        num_gpus_per_engine: 1
"""


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories, downloads the Qwen2.5-0.5B-Instruct model
    # and the GSM8K dataset. No checkpoint conversion (bridge mode). Side effects: shell commands,
    # network, filesystem.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    # [FUNCTION DOC] Writes the two-deployment YAML config to a pseudo file and launches training
    # with 4 GPUs. Rollout: GSM8K parquet, math reward, 2 rollouts of 8 prompts times 4 samples,
    # 512 response tokens, temperature 0.8, global batch 32. Eval flags on GSM8K (interval 20,
    # 512 tokens, top-k 1). All parallel sizes 1 with sequence parallel, dynamic batching at 4096
    # tokens per GPU. GRPO with KL loss at zero coefficient and a lower clip only, Adam at 1e-6.
    # SGLang: 1 GPU per engine, mem fraction 0.7, decode CUDA-graph max batch 32, config pseudo
    # file. CI test flag; colocated; bridge conversion mode. Returns nothing.
    U = command_utils.default_config().create_backend()
    config_arg = command_utils.encode_pseudo_file(SGLANG_CONFIG_YAML)

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 512 "
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
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
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
        "--rollout-num-gpus-per-engine 1 "
        f"--sglang-mem-fraction-static 0.7 "
        "--sglang-cuda-graph-max-bs-decode 32 "
        f"--sglang-config {config_arg} "
    )

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
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
