# [MODULE DOC]
# Minimal demonstration script (all work happens at import time, no functions) that launches
# single-round formal-math GRPO training of Qwen3-8B on one node with eight GPUs by submitting a
# job to a Ray cluster already listening at the local dashboard address (port 8265). It builds the
# training argument string in groups (checkpoint paths under the root directory, rollout with the
# custom Lean reward function and the minimal demo dataset, evaluation on miniF2F every 20 steps,
# performance, GRPO, optimiser, inference engine, miscellaneous with colocation), adds wandb
# options only if the WANDB_API_KEY environment variable is set, composes the model architecture
# arguments from the model definition helper, and runs a shell that exports unbuffered output and
# submits the job with a runtime environment (Megatron path, single CUDA connection, NCCL NVLS
# enabled) running the training entry script. The assembled command is printed before it runs.
# Requires the converted reference checkpoint and the prepared minimal demo dataset to exist.
# KNOWN ISSUE: the printed command contains the wandb API key in clear text when it is set, and the
# exit status of the submission is ignored. The computed repository base directory is unused.
"""
Example to demonstrate how to launch training.
You can also do the same using a .sh or others, and we use Python here just for simplicity.
"""

import json
import os
import subprocess
from pathlib import Path

from miles.utils.external_utils.model_args_utils import shell_safe_model_args

repo_base_dir = Path(os.path.abspath(__file__)).resolve().parents[4]

MODEL_NAME, MODEL_TYPE = "Qwen3-8B", "qwen3-8B"

NUM_GPUS = 8

ckpt_args = (
    f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
    f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    "--save-interval 20 "
    f"--load /root/{MODEL_NAME}_miles "
    f"--save /root/{MODEL_NAME}_miles "
)

rollout_args = (
    "--prompt-data /root/datasets/formal_math_single_round/minimal_demo/flc_train.jsonl "
    "--input-key prompt "
    "--apply-chat-template "
    "--rollout-shuffle "
    "--custom-rm-path examples.experimental.formal_math.single_round.reward_fn.reward_fn "
    "--reward-key reward_value "
    "--log-reward-category reward_cat "
    "--rollout-batch-size 32 "
    "--n-samples-per-prompt 8 "
    "--rollout-max-response-len 8192 "
    "--rollout-temperature 1 "
    "--global-batch-size 256 "
    "--balance-data "
    "--num-rollout 3000 "
)

eval_args = (
    "--eval-interval 20 "
    "--n-samples-per-eval-prompt 1 "
    "--eval-max-response-len 16384 "
    "--eval-top-p 1 "
    "--eval-prompt-data "
    "minif2f /root/datasets/formal_math_single_round/minimal_demo/minif2f_test.jsonl "
)

perf_args = (
    "--tensor-model-parallel-size 2 "
    "--sequence-parallel "
    "--pipeline-model-parallel-size 1 "
    "--context-parallel-size 1 "
    "--expert-model-parallel-size 1 "
    "--expert-tensor-parallel-size 1 "
    "--recompute-granularity full "
    "--recompute-method uniform "
    "--recompute-num-layers 1 "
    "--use-dynamic-batch-size "
    "--max-tokens-per-gpu 6144 "
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

sglang_args = "--rollout-num-gpus-per-engine 8 " "--sglang-mem-fraction-static 0.7 "

misc_args = (
    "--attention-dropout 0.0 "
    "--hidden-dropout 0.0 "
    "--accumulate-allreduce-grads-in-fp32 "
    "--attention-softmax-in-fp32 "
    "--attention-backend flash "
    "--actor-num-nodes 1 "
    "--actor-num-gpus-per-node 8 "
    "--colocate "
    "--log-passrate "
)

wandb_args = (
    (
        "--use-wandb "
        "--wandb-project miles-formal-math-run-minimal "
        "--wandb-group demo "
        f"--wandb-key '{wandb_api_key}' "
    )
    if (wandb_api_key := os.environ.get("WANDB_API_KEY"))
    else ""
)

train_args = (
    f"{ckpt_args} "
    f"{rollout_args} "
    f"{optimizer_args} "
    f"{grpo_args} "
    f"{wandb_args} "
    f"{perf_args} "
    f"{eval_args} "
    f"{sglang_args} "
    f"{misc_args} "
)

runtime_env_json = json.dumps(
    {
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root/Megatron-LM/",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "NCCL_NVLS_ENABLE": "1",
        }
    }
)

cmd = (
    f"export PYTHONUNBUFFERED=1 && "
    f'ray job submit --address="http://127.0.0.1:8265" '
    f"--runtime-env-json='{runtime_env_json}' "
    f"-- python3 train.py "
    f"{shell_safe_model_args(MODEL_TYPE)} "
    f"{train_args}"
)

print(f"EXEC: {cmd}", flush=True)
subprocess.run(["bash", "-c", cmd], shell=False)
