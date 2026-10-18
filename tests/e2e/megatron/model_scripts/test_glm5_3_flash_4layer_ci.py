# [MODULE DOC] CI smoke test of a 4-layer slice of GLM-5.3 Flash on one node of 8 GPUs (CUDA:
# 8-GPU H200 stage, estimated 1800 seconds, labels megatron and model-scripts, hopper and
# blackwell). The registration is currently disabled pending a CI image with the model's SGLang
# support. Metric history gates are registered on grad norm, PPO KL, train/rollout logprob
# absolute difference, train/rollout KL and raw reward. Unlike sibling tests, preparation is done
# inline here rather than by the launcher. Run as a script: build args, prepare, strip all proxy
# environment variables (both cases), train.

import os

from scripts.run_glm5_3_flash import _MODEL_REGISTRY, ScriptArgs, _train

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate


register_cuda_ci(
    est_time=1800,
    suite="stage-c-8-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper", "blackwell"],
    disabled="needs a CI image with GLM-5.3-Flash sglang support (branch sglang-miles-glm53); "
    "re-enable once it lands.",
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

_MODEL_ORG = "CharyZeng"


def _args() -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Builds launcher arguments from the environment with overrides: model
    # GLM-5.3-Flash-4layer, 1 node of 8 GPUs, 5 rollouts, rollout response length 512, rollout
    # routing replay enabled, no saving, and extra flags enabling CI test mode, disabling the KL
    # and logprobs checkers, and offloading the train target to CPU.
    return ScriptArgs.from_env(
        model_name="GLM-5.3-Flash-4layer",
        num_nodes=1,
        num_gpus_per_node=8,
        num_rollout=5,
        rollout_max_response_len=512,
        enable_r3=True,
        skip_saving=True,
        extra_args="--ci-test --ci-disable-kl-checker --ci-disable-logprobs-checker --offload-train-target cpu ",
    )


def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Prepares inputs. Sets, in this process, the environment variable asking the
    # converter to keep pipeline parallel size 1 and the one limiting CUDA device connections to 1.
    # Creates the model, checkpoint and data directories; downloads the model (from the fixed
    # model organization) into the arguments' HF checkpoint path; downloads the dapo-math-17k
    # dataset; converts the checkpoint to Megatron format with tensor parallel 2 and pipeline
    # parallel 1, looking the model type up in the launcher's model registry by model name and
    # using it as both the model name and Megatron model type.
    U = args.create_backend()
    os.environ["CONVERT_KEEP_PP1"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.ckpt_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=_MODEL_REGISTRY[args.model_name],
        megatron_model_type=_MODEL_REGISTRY[args.model_name],
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.ckpt_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args="--tensor-model-parallel-size 2 --pipeline-model-parallel-size 1",
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
