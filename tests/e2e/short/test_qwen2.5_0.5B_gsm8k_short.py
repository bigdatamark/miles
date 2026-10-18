# [MODULE DOC] End-to-end test of short GRPO training on Qwen2.5-0.5B-Instruct with GSM8K, 2
# colocated GPUs, bridge mode, fault tolerance (health check every 5 s, 10 s timeout, no initial
# wait). Besides serving as its own test, the module exposes a parameterized entrypoint (worker
# communication backend, object store backend defaulting to Mooncake, and the calling test file used
# for experiment-tracking names) that sibling tests for the Ray RPC communication backend with
# Mooncake or Ray stores reuse. Model and data directories come from the shared test directory
# helpers. Rollout: 2 rollouts of 8 prompts times 4 samples, oversampling 16, non-zero reward std
# filter, 1024 tokens, global batch 32; eval flags use interval 20 (never reached in 2 rollouts).
# Registered for CUDA 2-GPU H200 (est. 400 s) and ROCm 2-GPU MI350 nightly (est. 300 s), labels
# short and mooncake. Run as a script: entrypoint with the Ray communication backend.
import os

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.common_dirs import get_test_data_dir, get_test_model_dir

from miles.utils.external_utils import command_utils
from miles.utils.object_store import ObjectStoreBackend
from miles.utils.workers.types import WorkerCommBackend

register_cuda_ci(
    est_time=400, suite="stage-c-2-gpu-h200", labels=["short", "mooncake"], hardware=["hopper", "blackwell"]
)
register_rocm_ci(est_time=300, suite="nightly-stage-c-2-gpu-mi350", labels=["short", "mooncake"])

MODEL_DIR = get_test_model_dir()
DATA_DIR = get_test_data_dir()
MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 2


def entrypoint(
    *,
    comm_backend: WorkerCommBackend,
    object_store_backend: ObjectStoreBackend = ObjectStoreBackend.MOONCAKE,
    test_file: str,
) -> None:
    # [FUNCTION DOC] Full flow for a caller: prepare assets, remove proxy variables from the
    # environment, then run the training with the given worker communication backend, object store
    # backend and test file path (used to name the tracking run). Returns nothing.
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(comm_backend=comm_backend, object_store_backend=object_store_backend, test_file=test_file)


def prepare():
    # [FUNCTION DOC] Creates the model and data directories (from the shared helpers), downloads
    # Qwen2.5-0.5B-Instruct into the model directory and GSM8K into the data directory. Side
    # effects: shell commands, network, filesystem.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k", data_dir=DATA_DIR)


def execute(
    *,
    comm_backend: WorkerCommBackend,
    object_store_backend: ObjectStoreBackend = ObjectStoreBackend.MOONCAKE,
    test_file: str,
) -> None:
    # [FUNCTION DOC] Builds the argument string and launches training with 2 GPUs, colocated.
    # Keyword-only inputs: worker communication backend; object store backend (default Mooncake);
    # test file path for tracking names. The communication argument is omitted for the Ray backend
    # and otherwise names the backend's value. The object store arguments are the Ray backend flag
    # for the Ray store, else the Mooncake argument helper's text; they are placed right after the
    # checkpoint arguments, and the communication argument is last. Other settings: GRPO with KL
    # loss at zero coefficient, Adam at 1e-6, all parallel sizes 1 with sequence parallel, dynamic
    # batching at 9216 tokens per GPU, SGLang one GPU per engine with mem fraction 0.7 and metrics,
    # CI test flag, fault tolerance flags, bridge mode. Returns nothing.
    U = command_utils.default_config().create_backend()
    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}/ " f"--ref-load {MODEL_DIR}/{MODEL_NAME}/ "

    rollout_args = (
        f"--prompt-data {DATA_DIR}/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
    )

    eval_args = (
        "--eval-interval 20 "
        f"--eval-prompt-data gsm8k {DATA_DIR}/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 1024 "
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
        "--max-tokens-per-gpu 9216 "
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

    sglang_args = "--rollout-num-gpus-per-engine 1 " "--sglang-mem-fraction-static 0.7 " "--sglang-enable-metrics "

    ci_args = "--ci-test "

    fault_tolerance_args = (
        "--use-fault-tolerance "
        "--rollout-health-check-interval 5 "
        "--rollout-health-check-timeout 10 "
        "--rollout-health-check-first-wait 0 "
    )

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

    worker_comm_args = "" if comm_backend is WorkerCommBackend.RAY else f"--worker-comm-backend {comm_backend.value} "

    object_store_args = (
        f"--object-store-backend {ObjectStoreBackend.RAY.value} "
        if object_store_backend is ObjectStoreBackend.RAY
        else command_utils.get_mooncake_object_store_args()
    )

    train_args = (
        f"{ckpt_args} "
        f"{object_store_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(test_file)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{fault_tolerance_args} "
        f"{misc_args} "
        f"{worker_comm_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    entrypoint(comm_backend=WorkerCommBackend.RAY, test_file=__file__)
