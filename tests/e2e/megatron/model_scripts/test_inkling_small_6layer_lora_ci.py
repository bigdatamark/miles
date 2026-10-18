# [MODULE DOC] CI smoke test of the Inkling launcher in LoRA train mode on the 6-layer slice:
# shared-outer grouped-expert LoRA served through SGLang's virtual-experts path with one 4-GPU
# engine, with the adapter sync verified by checksum; one node of 4 GPUs. Registrations: CUDA
# 4-GPU H200 stage (estimated 1100 seconds, hopper and blackwell) and AMD ROCm nightly 4-GPU
# MI350 (estimated 500 seconds); labels megatron, model-scripts and lora. Metric history gates on
# grad norm, PPO KL, train/rollout logprob absolute difference, train/rollout KL and raw reward.
# Run as a script: build args, prepare, strip all proxy environment variables (both cases), train.
# KNOWN ISSUE: the launcher is imported only from the non-AMD location, although the arguments
# add a ROCm-specific attention backend and the test is registered on ROCm, whereas the
# full-parameter sibling switches to the AMD launcher on ROCm.

import os

from scripts.run_inkling import _MODEL_REGISTRY, ScriptArgs, _train
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.ci.metric_history import register_ci_gate


# Smoke test for scripts/run_inkling.py --train-mode lora on the 6-layer slice:
# shared-outer grouped-expert LoRA served through SGLang's virtual-experts path, one
# 4-GPU engine, adapter sync verified by checksum. Functionality, not accuracy.


register_cuda_ci(
    est_time=1100,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts", "lora"],
    hardware=["hopper", "blackwell"],
)
register_rocm_ci(
    est_time=500,
    suite="nightly-stage-c-4-gpu-mi350",
    labels=["megatron", "model-scripts", "lora"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

_MODEL_ORG = "pb09204048"


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments from the environment with overrides: model
    # Inkling-Small-6layer, LoRA train mode, dapo-math task, 1 node of 4 GPUs with 4 GPUs per
    # rollout engine, 2 rollouts of 4 prompts times 4 samples, global batch 16, response length
    # 512, SGLang context length 1024. Extra flags: CI test mode, KL and logprobs checkers
    # disabled, skip visual and audio parameters and two engine-derived adapter buffer names in
    # the weight-update check (frozen towers and derived buffers never match the snapshot), check
    # LoRA weight equality; on ROCm also the Triton attention backend in SGLang.
    return ScriptArgs.from_env(
        model_name="Inkling-Small-6layer",
        train_mode="lora",
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
            # frozen towers and the engine-derived adapter buffers never match the snapshot
            "--check-weight-update-skip-list visual. audio. ._w1_delta ._a_cat "
            "--ci-disable-logprobs-checker "
            "--check-lora-weight-equal "
        )
        + ("--sglang-attention-backend triton " if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm" else ""),
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Creates the model and data directories; downloads the model (from the fixed
    # model organization) into the arguments' HF checkpoint path; downloads the dapo-math-17k
    # dataset; converts the checkpoint to Megatron format on the node's GPUs, looking the Megatron
    # model type up in the launcher's model registry by model name.
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
