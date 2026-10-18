# [MODULE DOC] Self-contained CI test of a 5-layer slice of DeepSeek V3.2 with MXFP8 (block-scaled
# 8-bit float) training and rollout on 8 GPUs (4 actor GPUs, 4 rollout GPUs in engines of 2), for
# Blackwell hardware (CUDA: 8-GPU B200 stage, estimated 3600 seconds, labels megatron and
# model-scripts). The registration is currently disabled. Defines the model and path constants,
# a transformer-engine precision config text keeping selected attention, shared expert and
# indexer projections in bf16, a prepare step (download, convert to bf16, quantize to MXFP8
# keeping the first and last layer and selected projections in bf16, convert to Megatron) and an
# execute step. Run as a script: prepare, strip all proxy environment variables (both cases),
# execute.
# KNOWN ISSUE: the disabled reason refers to a test named after the V3.2 5-layer CI on H100, but
# the sibling V3.2 5-layer test in this folder is the fp8 one registered on H200.
# KNOWN ISSUE: the run id constant is created but never used.

import os

from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils

register_cuda_ci(
    est_time=3600,
    suite="stage-c-8-gpu-b200",
    labels=["megatron", "model-scripts"],
    hardware=["blackwell"],
    disabled="Temporarily disabled; superseded by test_deepseek_v32_5layer_ci on H100.",
)

MODEL_ORG = "Pinaster"
MODEL_NAME = "DeepSeek-V3.2-5layer"
MODEL_TYPE = "deepseek-v32-5layer"
NUM_GPUS = 8
ACTOR_NUM_GPUS = 4
ROLLOUT_NUM_GPUS = 4
ROLLOUT_GPUS_PER_ENGINE = 2
NUM_LAYERS_AT_START_IN_BF16 = 1
NUM_LAYERS_AT_END_IN_BF16 = 1
RUN_ID = command_utils.create_run_id()

MODEL_DIR = "/root/models"
DATA_DIR = "/root/datasets"
MEGATRON_PATH = "/root/Megatron-LM"

TE_PRECISION_CONFIG = """
configs:
  bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe: {}
matchers:
  mla_kv_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_kv_up_proj"
    config: "bf16"
  absorbed_k_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_k_up_proj"
    config: "bf16"
  absorbed_v_up_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.linear_v_up_proj"
    config: "bf16"
  shared_fc1:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc1"
    config: "bf16"
  shared_fc2:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc2"
    config: "bf16"
  dsa_indexer_wq_b_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.wq_b"
    config: "bf16"
  dsa_indexer_wk_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.wk"
    config: "bf16"
  dsa_indexer_weights_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.weights_proj"
    config: "bf16"
""".strip()


def prepare():
    # [FUNCTION DOC] Prepares model and data through the default backend. Steps: create the model
    # and dataset directories; download the Pinaster DeepSeek-V3.2-5layer checkpoint and the
    # dapo-math-17k dataset; cast the FP8 checkpoint to bf16 into a sibling directory with a bf16
    # suffix; run the MXFP8 conversion tool on GPUs from the bf16 copy into a directory with an
    # MXFP8 suffix, keeping the first and last layer in bf16 and additionally keeping the kv up
    # projection, shared experts, and the indexer query-b, key and weights projections in higher
    # precision; convert the bf16 checkpoint to a Megatron checkpoint using the 4 actor GPUs.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu(f"mkdir -p {MODEL_DIR} {DATA_DIR}")
    U.exec_command_cpu(f"hf download {MODEL_ORG}/{MODEL_NAME} --local-dir {MODEL_DIR}/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=DATA_DIR)

    U.fp8_cast_bf16(
        path_src=f"{MODEL_DIR}/{MODEL_NAME}",
        path_dst=f"{MODEL_DIR}/{MODEL_NAME}-bf16/",
    )

    U.exec_command_gpu(
        f"python tools/convert_hf_to_mxfp8.py "
        f"--model-dir {MODEL_DIR}/{MODEL_NAME}-bf16 "
        f"--save-dir {MODEL_DIR}/{MODEL_NAME}-MXFP8 "
        f"--num-layers-at-start-in-bf16 {NUM_LAYERS_AT_START_IN_BF16} "
        f"--num-layers-at-end-in-bf16 {NUM_LAYERS_AT_END_IN_BF16} "
        "--extra-high-precision-layers-hf "
        ".kv_b_proj. "
        ".shared_experts. "
        ".wq_b. "
        ".wk. "
        ".weights_proj. "
    )

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=ACTOR_NUM_GPUS,
        dir_dst=MODEL_DIR,
        hf_checkpoint=f"{MODEL_DIR}/{MODEL_NAME}-bf16",
        megatron_path=MEGATRON_PATH,
    )


def execute():
    # [FUNCTION DOC] Launches training on 8 GPUs. Sets the ray temporary directory environment
    # variable to a default if unset and registers the precision config text as a pseudo-file.
    # Arguments: HF checkpoint is the MXFP8 directory, reference load is the Megatron checkpoint;
    # rollout (dapo-math-17k, deepscaler reward, 2 rollouts of 4 prompts times 4 samples, response
    # length 8192, temperature 1, global batch 16, data balancing); parallelism (tensor and expert
    # parallel 4, sequence parallel, full uniform recompute of 1 layer, dynamic batch size with
    # 32768 tokens per GPU, padding multiple 4096, log-prob chunk 1024); GRPO with KL loss
    # coefficient 0; Adam learning rate 1e-6 with CPU optimizer offload and precision-aware
    # optimizer; SGLang with the sparse-attention backend and flash MLA sparse decode/prefill,
    # flashinfer top-k, bf16 KV cache, page size 64, engines of 2 GPUs with data-parallel
    # attention, flashinfer TRT-LLM FP8 GEMM and routed MoE runner, CUDA graph batch 256; mixed
    # precision (MXFP8 recipe, e4m3 format, first/last layer bf16, high-precision layer lists for
    # HF and Megatron names, the TE precision config file); routing replay, no shared expert
    # fusion, zero dropout, flash attention, all-gather context parallel, update-weight buffer of
    # 2 GiB, 4 actor GPUs and 4 rollout GPUs, fault tolerance enabled. Extra environment variables
    # control the sparse-attention top-k fusion and tie-breaking and disable NCCL use by NVSHMEM.
    U = command_utils.default_config().create_backend()
    os.environ.setdefault("RAY_TMPDIR", "/tmp/ray")
    te_precision_config_path = command_utils.encode_pseudo_file(TE_PRECISION_CONFIG)

    ckpt_args = f"--hf-checkpoint {MODEL_DIR}/{MODEL_NAME}-MXFP8/ " f"--ref-load {MODEL_DIR}/{MODEL_NAME}_torch_dist "

    rollout_args = (
        f"--prompt-data {DATA_DIR}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    perf_args = (
        f"--tensor-model-parallel-size {ACTOR_NUM_GPUS} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ACTOR_NUM_GPUS} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 32768 "
        "--data-pad-size-multiplier 4096 "
        "--log-probs-chunk-size 1024 "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    sglang_args = (
        "--sglang-mem-fraction-static 0.8 "
        "--sglang-attention-backend dsa "
        "--sglang-dsa-decode-backend flashmla_sparse "
        "--sglang-dsa-prefill-backend flashmla_sparse "
        "--sglang-dsa-topk-backend flashinfer "
        "--sglang-kv-cache-dtype bf16 "
        "--sglang-page-size 64 "
        f"--rollout-num-gpus-per-engine {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-fp8-gemm-backend flashinfer_trtllm "
        "--sglang-moe-runner-backend flashinfer_trtllm_routed "
        f"--sglang-tp-size {ROLLOUT_GPUS_PER_ENGINE} "
        f"--sglang-dp-size {ROLLOUT_GPUS_PER_ENGINE} "
        "--sglang-enable-dp-attention "
        "--sglang-enable-dp-lm-head "
        "--sglang-cuda-graph-max-bs-decode 256 "
    )

    ci_args = "--ci-test --check-weight-update-allow-quant-error "

    mixed_precision_args = (
        "--transformer-impl transformer_engine "
        "--bf16 "
        "--fp8-format e4m3 "
        "--fp8-recipe mxfp8 "
        "--first-last-layers-bf16 "
        f"--num-layers-at-start-in-bf16 {NUM_LAYERS_AT_START_IN_BF16} "
        f"--num-layers-at-end-in-bf16 {NUM_LAYERS_AT_END_IN_BF16} "
        "--extra-high-precision-layers-hf "
        ".kv_b_proj. "
        ".shared_experts. "
        ".wq_b. "
        ".wk. "
        ".weights_proj. "
        "--extra-high-precision-layers-megatron "
        ".linear_kv_up_proj "
        ".linear_k_up_proj "
        ".linear_v_up_proj "
        ".shared_experts.linear_fc1 "
        ".shared_experts.linear_fc2 "
        ".wq_b "
        ".wk "
        ".weights_proj "
        f"--te-precision-config-file {te_precision_config_path} "
    )

    misc_args = (
        "--use-rollout-routing-replay "
        "--sglang-disable-shared-experts-fusion "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--allgather-cp "
        "--miles-dsa-topk-backend flashinfer "
        f"--update-weight-buffer-size {2 * 1024 ** 3} "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {ACTOR_NUM_GPUS} "
        f"--num-gpus-per-node {NUM_GPUS} "
        f"--rollout-num-gpus {ROLLOUT_NUM_GPUS} "
        "--use-fault-tolerance "
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
        f"{mixed_precision_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        megatron_path=MEGATRON_PATH,
        extra_env_vars={
            "SGLANG_DSA_FUSE_TOPK": "1",
            "SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD": "0",
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
            "NVSHMEM_DISABLE_NCCL": "1",
        },
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
