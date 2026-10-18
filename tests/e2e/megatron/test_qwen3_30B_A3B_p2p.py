# [MODULE DOC] End-to-end test of the peer-to-peer weight update mode for Qwen3-30B-A3B on 8
# GPUs, non-colocated: a 6-GPU training actor pool (pipeline parallel 3, context parallel 2,
# tensor 1, expert 2, so data parallel is 1) and a 2-GPU rollout pool with a single engine,
# trained through the asynchronous train script. p2p weight update cannot be combined with
# colocation, hence disjoint pools. The actor uses CPU optimizer offload because fp32 main
# params and Adam moments do not fit with only a data/context shard of 2. The rollout engine
# seeds weights through the transfer engine. Registered for CUDA 8-GPU H100 (est. 800 s) with
# five CI metric gates. Module constants: model name/type, total GPUs 8, rollout GPUs 2, actor
# GPUs 6. Run as a script: prepare then execute (no proxy stripping).
from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8
ROLLOUT_NUM_GPUS = 2  # inference engine pool
ACTOR_NUM_GPUS = NUM_GPUS - ROLLOUT_NUM_GPUS  # 6: training actor pool (p2p keeps it disjoint from rollout)

register_cuda_ci(
    est_time=800,
    suite="stage-c-8-gpu-h100",
    labels=["megatron", "weight-update"],
    hardware=["hopper", "blackwell"],
)

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")


def prepare():
    # [FUNCTION DOC] Creates model/dataset directories, downloads the Qwen3-30B-A3B Hugging Face
    # model and the dapo-math-17k dataset, and converts the model to a Megatron distributed
    # checkpoint using all 8 GPUs. Side effects: shell commands, network, filesystem, GPU.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def execute():
    # [FUNCTION DOC] Builds the argument string and launches asynchronous training. Rollout: 2
    # rollouts of 4 prompts times 2 samples, 100 response tokens, temperature 0.8, global batch 8,
    # balanced data. Parallelism: pipeline 3, context 2, expert 2, sequence parallel, full
    # recompute, dynamic batching with 2048 tokens per GPU. GRPO with zero KL/entropy
    # coefficients and the actor forward-only pass skipped. Adam with CPU offload. SGLang: one
    # engine of 2 GPUs, 2 rollout GPUs, mem fraction 0.7, remote-instance weight loader seeded via
    # the transfer engine. Weight update: p2p mode with a 1 GiB buffer; alltoall token dispatcher.
    # The launch sets an environment variable disabling the nvidia peer-memory module because the
    # CI host lacks it (GPU memory is registered via dmabuf instead). Launches with 8 GPUs per node.
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
        "--rollout-max-response-len 100 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 8 "
        "--balance-data "
    )

    # p2p weight update is incompatible with --colocate, so actor and rollout use
    # disjoint pools (6 actor + 2 rollout = 8). tp1 * cp2 * pp3 = 6 fills the actor
    # pool (dp1); experts span ep = 6 / pp3 = 2 ranks. (ep8 would need all 8 GPUs
    # on the actor, impossible while rollout needs its own.)
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 3 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 2 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--skip-actor-forward-only "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        # fp32 main params + adam m/v don't fit on 6 GPUs (dp-cp shard is only 2)
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 2 "
        f"--rollout-num-gpus {ROLLOUT_NUM_GPUS} "
        "--sglang-mem-fraction-static 0.7 "
        "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {ACTOR_NUM_GPUS} "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        "--update-weight-transfer-mode p2p "
        "--moe-token-dispatcher-type alltoall "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
        # fallback: CI host lacks nvidia_peermem; register GPU memory via dmabuf instead
        extra_env_vars={"WITH_NVIDIA_PEERMEM": "0"},
    )


if __name__ == "__main__":
    prepare()
    execute()
