# [MODULE DOC] CI smoke test of a 6-layer slice of Inkling Small with full-parameter training
# on dapo-math, on one node of 4 GPUs (one rollout engine across all 4). The AMD variant of the
# launcher is imported when the hardware platform environment variable says ROCm. Registrations:
# CUDA 4-GPU H200 stage (estimated 900 seconds, hopper and blackwell) and AMD ROCm 4-GPU MI350
# stage (estimated 1800 seconds); labels megatron and model-scripts (plus amd on ROCm). Metric
# history gates on grad norm, PPO KL, train/rollout logprob absolute difference, train/rollout
# KL and raw reward. Run as a script: build args, prepare, strip all proxy environment variables
# (both cases), train.

import os

if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm":
    from scripts.amd.run_inkling import _MODEL_REGISTRY, ScriptArgs, _train
else:
    from scripts.run_inkling import _MODEL_REGISTRY, ScriptArgs, _train

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate


register_cuda_ci(
    est_time=900,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=1800,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "model-scripts", "amd"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

_MODEL_ORG = "pb09204048"


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments from the environment with overrides: model
    # Inkling-Small-6layer, full train mode, dapo-math task, 1 node of 4 GPUs with 4 GPUs per
    # rollout engine, 2 rollouts of 4 prompts times 4 samples, global batch 16, response length
    # 512, SGLang context length 1024. Extra flags: CI test mode, KL and logprobs checkers
    # disabled, skip visual and audio parameters in the weight-update check, offload the train
    # target to CPU.
    return ScriptArgs.from_env(
        model_name="Inkling-Small-6layer",
        train_mode="full",
        task="dapo_math",
        num_nodes=1,
        num_gpus_per_node=4,
        rollout_num_gpus_per_engine=4,
        num_rollout=2,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        rollout_max_response_len=512,
        sglang_context_length=1024,
        extra_args=(
            "--ci-test "
            "--ci-disable-kl-checker "
            "--check-weight-update-skip-list visual. audio. "
            "--ci-disable-logprobs-checker "
            "--offload-train-target cpu "
        ),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates the model and data directories; downloads the model (from the fixed
    # model organization) into the arguments' HF checkpoint path; downloads the dapo-math-17k
    # dataset into the data directory; converts the checkpoint to Megatron format on the node's
    # GPUs, looking the Megatron model type up in the launcher's model registry by model name.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=_MODEL_REGISTRY[args.model_name],
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    # [FUNCTION DOC] Launches training through the launcher's train step.
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
