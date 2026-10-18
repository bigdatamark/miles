# [MODULE DOC] HELPER: Shared building blocks for the Qwen3-30B-A3B Megatron end-to-end tests in
# this directory (one test file per case). Provides a case-configuration dataclass describing
# topology and feature toggles, a prepare function (downloads and conversions), a function that
# builds the full training argument string, and an execute function that launches training.
# The argument builder is separate from execute so a CPU-only golden test can inspect the string.
# Module-level state: the model name and Megatron model type, and a tight-host-memory flag read
# from an environment variable (default on) that selects fp16 optimizer state for Adam.
import os
from dataclasses import dataclass, field
from typing import Literal

from miles.utils.external_utils import command_utils

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"

TIGHT_HOST_MEMORY = bool(int(os.environ.get("MILES_TEST_TIGHT_HOST_MEMORY", "1")))


@dataclass
class CaseConfig:
    # [CLASS DOC] HELPER: Plain data holder describing one test case. Topology fields: GPUs per
    # node, context/pipeline/tensor/expert parallel sizes, optional expert tensor parallel (default
    # 1), rollout GPUs per engine, optional SGLang expert- and data-parallel sizes, data-parallel
    # attention flag, SGLang max running requests (default 512). Feature toggles (all default off):
    # DeepEP, FP8 rollout, INT4 rollout, bridge mode (direct HF load), rollout routing replay (R3),
    # Mooncake object store. Other fields: max tokens per GPU (default 8192), colocate (default
    # true), total rollout GPUs (required when not colocated), optional weight-update transfer mode,
    # number of rollouts (default 2), fully-async flag, optimizer choice (adam or dist_muon), extra
    # argument text appended verbatim, and extra environment variables for the launch.
    num_gpus_per_node: int
    cp_size: int
    pp_size: int
    tp_size: int
    ep_size: int
    rollout_num_gpus_per_engine: int
    etp_size: int = 1
    sglang_ep_size: int = None
    sglang_dp_size: int = None
    sglang_enable_dp_attention: bool = False
    sglang_max_running_requests: int = 512
    use_deepep: bool = False
    use_fp8_rollout: bool = False
    use_int4_rollout: bool = False
    use_bridge: bool = False
    use_r3: bool = False
    use_mooncake: bool = False
    max_tokens_per_gpu: int = 8192
    colocate: bool = True
    rollout_num_gpus: int = None
    update_weight_transfer_mode: str = None
    num_rollout: int = 2
    fully_async: bool = False
    optimizer: Literal["adam", "dist_muon"] = "adam"
    extra_args: str = ""
    extra_env_vars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # [FUNCTION DOC] HELPER: Validation only; nothing is inferred or mutated. Raises a value
        # error if: the optimizer is not one of the two supported names; fully-async is combined
        # with colocation (the async trainer rejects colocation); GPUs per node is not divisible
        # by context size times pipeline size; non-colocated without a rollout GPU count; or the
        # rollout pool (GPUs per node if colocated, else rollout GPUs) is not divisible by GPUs
        # per engine. If a weight-update transfer mode is given it must be broadcast or
        # broadcast-packed, enforced with an assertion (assertion error, not value error).
        if self.optimizer not in ("adam", "dist_muon"):
            raise ValueError(f"unsupported optimizer: {self.optimizer}")
        # Validation only — topology values are passed explicitly, not inferred.
        if self.fully_async and self.colocate:
            raise ValueError("fully_async requires colocate=False: train_async.py rejects colocation")
        if self.num_gpus_per_node % (self.cp_size * self.pp_size) != 0:
            raise ValueError(
                "num_gpus_per_node must be divisible by cp_size * pp_size: "
                f"{self.num_gpus_per_node=} {self.cp_size=} {self.pp_size=}"
            )
        if not self.colocate and self.rollout_num_gpus is None:
            raise ValueError("rollout_num_gpus must be set when colocate is False")
        rollout_pool = self.num_gpus_per_node if self.colocate else self.rollout_num_gpus
        if rollout_pool % self.rollout_num_gpus_per_engine != 0:
            raise ValueError(
                "rollout pool must be divisible by rollout_num_gpus_per_engine: "
                f"{rollout_pool=} {self.rollout_num_gpus_per_engine=}"
            )
        if self.update_weight_transfer_mode is not None:
            assert self.update_weight_transfer_mode in ("broadcast", "broadcast_packed")


def prepare(case: CaseConfig, *, need_fp8: bool, need_int4: bool, all_bridge: bool) -> None:
    # [FUNCTION DOC] Downloads and converts assets for a case. Creates model and dataset
    # directories; downloads the BF16 Hugging Face model; if FP8 is needed, downloads the FP8
    # model; if INT4 is needed, runs the direct HF-to-INT4 conversion tool on the GPU into a
    # sibling INT4 directory; downloads the dapo-math-17k and aime-2024 datasets; and unless
    # all_bridge is true, converts the model to a Megatron distributed checkpoint using the case's
    # GPUs per node (bridge mode reads the HF checkpoint directly, so no conversion). Side
    # effects: shell commands, network, filesystem, GPU. Returns nothing.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu("hf download Qwen/Qwen3-30B-A3B --local-dir /root/models/Qwen3-30B-A3B")
    if need_fp8:
        U.exec_command_cpu("hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/models/Qwen3-30B-A3B-FP8")
    if need_int4:
        U.exec_command_gpu(
            f"python tools/convert_hf_to_int4_direct.py "
            f"--model-dir /root/models/{MODEL_NAME} "
            f"--save-dir /root/models/{MODEL_NAME}-INT4"
        )
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    # Bridge mode reads the HF checkpoint directly; non-bridge variants need
    # the torch_dist conversion. With one case per file, "all_bridge" reduces
    # to the single case being a bridge case.
    if not all_bridge:
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=case.num_gpus_per_node,
        )


def build_train_args(case: CaseConfig, *, wandb_file: str) -> str:
    # [FUNCTION DOC] Assembles the single space-separated training argument string for a case;
    # pure apart from reading environment variables and a call to the wandb-args helper (keyed on
    # the given test file path). Raises a value error if INT4 and FP8 rollout are both set.
    # Sections: checkpoint (HF checkpoint is the INT4, FP8 or BF16 dir; reference load is the HF
    # dir in bridge mode else the Megatron distributed dir; INT4 adds the flag disabling gradient
    # accumulation fusion because fake QAT weights are straight-through); rollout (dapo-math
    # data, 8 prompts times 8 samples, 8192 response tokens, global batch 32, balanced data);
    # eval (aime24 prompts, 16384 tokens, top-k 1; the eval interval of 20 is added only when the
    # eval environment variable is set, otherwise the eval data flags remain without an
    # interval); performance (parallel sizes from the case, full recompute, dynamic batching;
    # when tight host memory is on and the optimizer is Adam, fp16 for first/second moments and
    # main params); GRPO (GSPO estimator, KL loss omitted in bridge mode, TIS, and the routing
    # replay flag: rollout-routing replay for R3 cases, plain routing replay otherwise);
    # optimizer (Adam gets CPU offload and precision-aware optimizer; dist_muon gets chunked
    # state offload of fraction 1.0 and 1024 MB chunks); SGLang (per-engine GPUs, mem fraction
    # 0.7, metrics, DeepEP a2a backend plus deep_gemm runner for BF16 only, optional EP/DP
    # sizes and DP attention); CI flags (CI test, plus allow-quant-error for FP8/INT4); misc
    # (dropouts zero, fp32 grad allreduce and softmax, flash attention, colocate with
    # rematerialization for Adam or explicit rollout GPU count, transfer mode, bridge mode,
    # fully-async, Mooncake args, DeepEP flex dispatcher or alltoall). Returns the string with the
    # case's extra args appended.
    # KNOWN ISSUE: the default-off eval flags are always emitted even when evaluation is
    # disabled, so only the interval flag is conditional.
    """Build the train_args string for `case`.

    Split out from `execute()` so the CPU-only golden test can inspect the
    string without monkeypatching `execute_train`.
    """
    if case.use_int4_rollout and case.use_fp8_rollout:
        raise ValueError("use_int4_rollout and use_fp8_rollout are mutually exclusive")

    enable_eval = bool(int(os.environ.get("MILES_TEST_ENABLE_EVAL", "0")))

    ref_load = f"/root/models/{MODEL_NAME}" if case.use_bridge else f"/root/{MODEL_NAME}_torch_dist"
    if case.use_int4_rollout:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}-INT4/ " f"--ref-load {ref_load} "
        # Fake QAT swaps in straight-through weight tensors, while TE's fused wgrad
        # accumulation writes main_grad onto the original ones, so the two together
        # would drop the quantized weights' gradients.
        ckpt_args += "--no-gradient-accumulation-fusion "
    elif case.use_fp8_rollout:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}-FP8 " f"--ref-load {ref_load} "
    else:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load {ref_load} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {case.num_rollout} "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if enable_eval else ''}"
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {case.tp_size} "
        "--sequence-parallel "
        f"--pipeline-model-parallel-size {case.pp_size} "
        f"--context-parallel-size {case.cp_size} "
        f"--expert-model-parallel-size {case.ep_size} "
        f"--expert-tensor-parallel-size {case.etp_size} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {case.max_tokens_per_gpu} "
    )

    if TIGHT_HOST_MEMORY and case.optimizer == "adam":
        perf_args += "--exp-avg-dtype fp16 "
        perf_args += "--exp-avg-sq-dtype fp16 "
        perf_args += "--main-params-dtype fp16 "

    # r3 path uses --use-rollout-routing-replay; non-r3 uses --use-routing-replay.
    routing_flag = "--use-rollout-routing-replay" if case.use_r3 else "--use-routing-replay"
    grpo_args = (
        "--advantage-estimator gspo "
        f"{'' if case.use_bridge else '--use-kl-loss '}"
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
        "--use-tis "
        f"{routing_flag} "
    )

    optimizer_args = (
        f"--optimizer {case.optimizer} "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if case.optimizer == "dist_muon":
        # Muon offloads through LayerWise; HybridDeviceOptimizer is Adam-only.
        optimizer_args += (
            "--chunked-optimizer-state-offload "
            "--optimizer-state-offload-fraction 1.0 "
            "--optimizer-state-offload-chunk-size-mb 1024 "
        )
    else:
        optimizer_args += (
            "--optimizer-cpu-offload " "--overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
        )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {case.rollout_num_gpus_per_engine} "
        "--sglang-mem-fraction-static 0.7 "
        f"--sglang-max-running-requests {case.sglang_max_running_requests} "
        "--sglang-enable-metrics "
    )

    if case.use_deepep:
        sglang_args += "--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto "
        if not (case.use_fp8_rollout or case.use_int4_rollout):
            # SGLang has DeepEP MoE kernels for BF16 experts only on the DeepGEMM runner
            # (FP8 already resolves auto to DeepGEMM).
            sglang_args += "--sglang-moe-runner-backend deep_gemm "
    if case.sglang_ep_size is not None:
        sglang_args += f"--sglang-expert-parallel-size {case.sglang_ep_size} "
    if case.sglang_dp_size is not None:
        sglang_args += f"--sglang-data-parallel-size {case.sglang_dp_size} "
    if case.sglang_enable_dp_attention:
        sglang_args += "--sglang-enable-dp-attention "

    ci_args = "--ci-test "
    if case.use_fp8_rollout or case.use_int4_rollout:
        ci_args += "--check-weight-update-allow-quant-error "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {case.num_gpus_per_node} "
    )
    if case.colocate:
        misc_args += "--colocate "
        if case.optimizer == "adam":
            misc_args += "--rematerialize-param-from-master-weight "
    else:
        misc_args += f"--rollout-num-gpus {case.rollout_num_gpus} "

    if case.update_weight_transfer_mode is not None:
        misc_args += f"--update-weight-transfer-mode {case.update_weight_transfer_mode} "

    if case.use_bridge:
        misc_args += "--megatron-to-hf-mode bridge "

    if case.fully_async:
        misc_args += "--fully-async "

    if case.use_mooncake:
        misc_args += command_utils.get_mooncake_object_store_args()

    if case.use_deepep:
        misc_args += "--moe-token-dispatcher-type flex --moe-enable-deepep "
    else:
        misc_args += "--moe-token-dispatcher-type alltoall "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(wandb_file)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{case.extra_args} "
    )
    return train_args


def execute(case: CaseConfig, *, wandb_file: str) -> None:
    # [FUNCTION DOC] Launches training for a case. Builds the argument string, then builds the
    # extra environment: for INT4 rollout sets the fake-QAT flag and group size 128, then merges
    # the case's own extra variables (these win on conflict). Calls the backend train launcher
    # with GPU count equal to GPUs per node plus rollout GPUs when not colocated, the model type,
    # and the async train script when fully-async else the standard one. Returns nothing.
    U = command_utils.default_config().create_backend()
    train_args = build_train_args(case, wandb_file=wandb_file)

    extra_env_vars = {}
    if case.use_int4_rollout:
        extra_env_vars |= {
            "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
            "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
        }
    extra_env_vars |= case.extra_env_vars

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=case.num_gpus_per_node + (0 if case.colocate else case.rollout_num_gpus),
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py" if case.fully_async else "train.py",
        extra_env_vars=extra_env_vars,
    )
