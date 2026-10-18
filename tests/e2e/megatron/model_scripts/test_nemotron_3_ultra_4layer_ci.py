# [MODULE DOC] CI smoke test of the Nemotron 3 Ultra launcher (hybrid Mamba2, attention and
# latent mixture-of-experts) on a 4-layer slice, on one node of 8 H200 GPUs (CUDA: 8-GPU H200
# stage, estimated 900 seconds, labels megatron and model-scripts, hopper and blackwell). The
# slice keeps source layers 0, 1, 7, 8 giving the block pattern Mamba, MoE, attention, MoE, with
# the MoE config untouched. It needs 8 GPUs because SGLang data-parallel attention requires the
# attention tensor-parallel size to divide the Mamba group count, and the CI test flag turns on
# the Megatron-to-SGLang weight equality check, the main thing guarded. Metric history gates on
# grad norm, PPO KL, train/rollout logprob absolute difference, train/rollout KL and raw reward.
# Run as a script: build args, prepare, strip all proxy environment variables (both cases),
# train.

import os

from scripts.run_nemotron_3_ultra_550b_a55b import ScriptArgs, _execute_train, _prepare_download
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

# Smoke test for the Nemotron-3-Ultra (nemotron_h: hybrid Mamba2 + Attention + latent-MoE)
# training script. It runs a 4-layer slice on a single 8-GPU H200 node and only verifies that
# the training script is functional, not model accuracy.
#
# The slice keeps source layers 0,1,7,8 renumbered to 0..3, so its block pattern is
# mamba, moe, attention, moe ("ME*E") -- every block type the full 108-layer model has, with
# the 512-expert / top-22 / moe_latent_size=2048 MoE config untouched. 8 GPUs (not 4) because
# SGLang DP-attention needs attn_tp to divide Mamba n_groups=8, and --ci-test turns on the
# Megatron -> SGLang weight equality check, which is the main thing this test guards.


register_cuda_ci(
    est_time=900,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper", "blackwell"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments from the environment with overrides: H200
    # hardware, model organization CharyZeng, the bf16 4-layer Nemotron 3 Ultra model, debug-
    # minimal mode, 1 node of 8 GPUs, 2 rollouts of 8 prompts times 2 samples, global batch 16, no
    # saving, extra flags enabling CI test mode and disabling the logprobs checker.
    return ScriptArgs.from_env(
        hardware="H200",
        model_org="CharyZeng",
        model_name="NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer",
        mode="debug_minimal",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=2,
        rollout_batch_size=8,
        n_samples_per_prompt=2,
        global_batch_size=16,
        skip_saving=True,
        extra_args=("--ci-test " "--ci-disable-logprobs-checker "),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates the output directory through the arguments' backend and runs the
    # launcher's download step.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.output_dir}")
    _prepare_download(args)


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step.
    _execute_train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
