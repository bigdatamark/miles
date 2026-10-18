# [MODULE DOC] CI smoke test of the GLM-5 launcher script on a 4-layer slice, on one node of 2 GPUs
# with tensor parallel 2 and optimizer offload (CUDA: 2-GPU H200 stage, estimated 900 seconds,
# labels megatron and model-scripts, hopper and blackwell). Verifies the script works for the
# sparse-attention model code path shared with DeepSeek V3.2, not accuracy. Metric history gates
# are registered on grad norm, PPO KL, train/rollout logprob absolute difference, train/rollout
# KL and raw reward. Run as a script: build args, prepare, strip all proxy environment variables
# (both cases), train.

import os

from scripts.run_glm5_744b_a40b import (
    ScriptArgs,
    _convert_to_fp8,
    _execute_train,
    _prepare_download,
    _prepare_megatron_ckpt,
    _validate_glm_checkpoint,
)
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate


# This CI test is an example smoke test for the DSA model code path used by DeepSeek V3.2 and GLM-5. It only verifies that the training script is functional, not model accuracy.


register_cuda_ci(
    est_time=900, suite="stage-c-2-gpu-h200", labels=["megatron", "model-scripts"], hardware=["hopper", "blackwell"]
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments from the environment with overrides: H200
    # hardware, model GLM-5_4layer, 1 node of 2 GPUs, 2 rollouts of 4 prompts times 4 samples,
    # global batch 16, optimizer offload on, extra flags enabling CI test mode, disabling the
    # logprobs checker and setting tensor parallel size 2.
    return ScriptArgs.from_env(
        hardware="H200",
        model_name="GLM-5_4layer",
        num_nodes=1,
        num_gpus_per_node=2,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        enable_optimizer_offload=True,
        extra_args=("--ci-test " "--ci-disable-logprobs-checker " "--tensor-model-parallel-size 2 "),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates the output directory through the arguments' backend; runs the
    # launcher's download step and checkpoint validation; converts to FP8 only if the arguments
    # request FP8 rollout; then produces the Megatron checkpoint.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)
    _validate_glm_checkpoint(args)
    if args.fp8_rollout:
        _convert_to_fp8(args)
    _prepare_megatron_ckpt(args)


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step.
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
