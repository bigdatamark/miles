# [MODULE DOC] ROCm (AMD MI355X) launcher for DeepSeek-V4-Flash GRPO training with a typer command
# line. Commands: prepare-download (fetch HF checkpoint and dataset), prepare-single (cast FP8
# checkpoint to BF16), prepare-spmd (convert BF16 to a Megatron torch_dist checkpoint), train, and
# full-train (runs the missing preparation steps, skipping those whose sentinel file exists, then
# trains). All shell work goes through the command utilities backend. Supports the full 43-layer
# Flash model and a 4-layer prune for smoke tests. Training args are assembled as grouped strings
# (checkpoint, rollout, perf, GRPO, optimizer, sglang, misc) and submitted via the backend.
# KNOWN ISSUE: the usage examples in the docstring below say scripts/run_deepseek_v4.py but the
# file lives under scripts/amd/.
"""
DeepSeek V4 training script.

Supports:
  - DeepSeek-V4-Flash-FP8         Public FP8 repackage of deepseek-ai/DeepSeek-V4-Flash
                                  (sgl-project/DeepSeek-V4-Flash-FP8, 291B, 43 layers).
                                  Verified full-model profile: 4 nodes x 8 GPUs on MI355X (gfx950).
  - DeepSeek-V4-Flash-FP8-4layer  4-layer prune of the above for single-node
                                  smoke testing. **Cannot generate meaningful output -
                                  pipeline-only sanity check.**

Usage patterns:

  1. One-shot full pipeline (download + convert + train):
       python scripts/run_deepseek_v4.py full-train \
           --model-name DeepSeek-V4-Flash-FP8-4layer \
           --num-nodes 1 --num-gpus-per-node 8

  2. Individual steps (download -> FP8->BF16 -> BF16->torch_dist -> train):
       python scripts/run_deepseek_v4.py prepare-download --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-single   --model-name DeepSeek-V4-Flash-FP8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-spmd     --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 1 --num-gpus-per-node 8
       python scripts/run_deepseek_v4.py train            --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 4 --num-gpus-per-node 8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

_DEFAULT_MODEL_ORG = {
    "DeepSeek-V4-Flash-FP8": "sgl-project",
    # 4-layer prune of sgl-project/DeepSeek-V4-Flash-FP8.
    "DeepSeek-V4-Flash-FP8-4layer": "Pinaster",
}

_MEGATRON_MODEL_TYPE = {
    "DeepSeek-V4-Flash-FP8": "deepseek-v4-flash",
    "DeepSeek-V4-Flash-FP8-4layer": "deepseek-v4-flash-4layer",
}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC] Command-line configuration (extends the shared train config with node count etc.).
    # Key fields: mode (normal or debug_minimal, default debug_minimal; non-debug adds oversampling
    # with a nonzero-reward-std dynamic filter), model_name (full Flash FP8 or its 4-layer prune),
    # model_org (HF org, derived from model_name when empty), task (dapo_aime or gsm8k), rollout
    # sizes, optional cp_size, checkpoint and data directories, model_local_dir (defaults to
    # model_dir; if different, model dirs are rsynced to it before training), optimizer CPU offload,
    # fault tolerance, debug dump / replay options, precision options (R3 routing replay,
    # deterministic mode, FP8 blockwise training default on, MIS) and free-form extra args.
    # Derived (not constructor) fields: colocate, actor node / GPU counts, rollout GPU count.
    # KNOWN ISSUE: run_id default is computed once at import time; debug_train_rollout_id is typed
    # as an optional string but is assigned an integer default inside the train routine.
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = command_utils.create_run_id()
    model_org: str = ""
    model_name: Literal[
        "DeepSeek-V4-Flash-FP8",
        "DeepSeek-V4-Flash-FP8-4layer",
    ] = "DeepSeek-V4-Flash-FP8"

    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    enable_eval: bool = True
    enable_mtp: bool = False
    dsv4_impl: Literal["miles"] = "miles"
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    cp_size: int | None = None

    hf_checkpoint: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    # Defaults to model_dir. Set explicitly when shared NFS -> per-node local NVMe copy is needed.
    model_local_dir: str | None = None
    save_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # performance configs
    num_gpus_per_node: int = 8
    # use colocate by default. will switch to disaggregated mode when 0 < rollout_num_nodes < num_nodes
    rollout_num_nodes: int = 0
    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)
    rollout_num_gpus: int = field(init=False)
    optimizer_offload: bool = True
    use_fault_tolerance: bool = True

    # debug configs
    dump_details: bool = False
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    debug_data_root: str = "/root/shared_data"
    skip_saving: bool = False

    # precision configs
    enable_r3: bool = True
    train_deterministic: bool = True
    # Megatron-side training precision: blockwise FP8 128x128 GEMMs (fp32 scales) when True,
    # BF16 when False. Rollout always serves the source FP8 checkpoint either way.
    fp8_training: bool = True
    enable_mis: bool = False

    # pass any extra sglang/miles/megatron args through `--extra-args '--your-arg'`
    extra_args: str = ""

    def __post_init__(self):
        # [FUNCTION DOC] Fill derived fields. Model org from the per-model table if empty; local model
        # dir defaults to model dir. Asserts rollout node count is non-negative and below the total
        # node count. Colocated when rollout nodes is zero (rollout GPUs = all GPUs); otherwise
        # disaggregated: actor nodes = total minus rollout nodes, rollout GPUs = rollout nodes times
        # GPUs per node. Actor GPUs per node equals GPUs per node.
        if not self.model_org:
            self.model_org = _DEFAULT_MODEL_ORG[self.model_name]
        if self.model_local_dir is None:
            self.model_local_dir = self.model_dir
        assert self.rollout_num_nodes >= 0
        assert self.rollout_num_nodes < self.num_nodes
        self.colocate = self.rollout_num_nodes == 0
        self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node
        if self.colocate:
            self.rollout_num_gpus = self.num_nodes * self.num_gpus_per_node
        else:
            self.rollout_num_gpus = self.rollout_num_nodes * self.num_gpus_per_node

    @property
    def megatron_model_type(self):
        # [FUNCTION DOC] Property: Megatron model-type name looked up from the model name (key error
        # for unknown names).
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def torch_dist_name(self):
        # [FUNCTION DOC] Property: directory name of the converted checkpoint, model name plus the
        # suffix _torch_dist.
        return f"{self.model_name}_torch_dist"

    @property
    def bf16_name(self):
        # [FUNCTION DOC] Property: directory name of the BF16-cast checkpoint, model name plus -bf16.
        return f"{self.model_name}-bf16"


def _download_dataset(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: download the dataset(s) for the task from HF into the data dir: dapo
    # math 17k and aime 2024 for dapo_aime, gsm8k for gsm8k.
    """Download the task-specific dataset(s)."""
    U = args.create_backend()
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _hf_checkpoint_path(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: the explicit hf_checkpoint if given, else model dir / model name.
    """Resolve hf_checkpoint path: explicit override wins, else {model_dir}/{model_name}."""
    return args.hf_checkpoint or f"{args.model_dir}/{args.model_name}"


def _ensure_4layer_model_type(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: only for the 4-layer model; if its config.json exists and contains the
    # legacy model type string deepseek_ref, rewrite the file in place with deepseek_v4 and print a
    # patch notice. Otherwise no-op.
    """Undo the old deepseek_ref workaround for local 4-layer prunes."""
    if args.model_name != "DeepSeek-V4-Flash-FP8-4layer":
        return
    cfg = Path(_hf_checkpoint_path(args)) / "config.json"
    if not cfg.exists():
        return
    text = cfg.read_text()
    if '"model_type": "deepseek_ref"' in text:
        cfg.write_text(text.replace('"model_type": "deepseek_ref"', '"model_type": "deepseek_v4"'))
        print(f"[patch] {cfg}: model_type deepseek_ref -> deepseek_v4")


def _prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: create model and data dirs; download the HF checkpoint into model dir /
    # model name unless an explicit hf_checkpoint was supplied; patch the 4-layer config; download
    # the dataset. Idempotent (HF skips existing blobs).
    """Download HF checkpoint + task dataset. Idempotent: hf skips existing blobs."""
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    # Only download if the user has NOT supplied a pre-existing checkpoint dir.
    # (prepare_single / train with --hf-checkpoint bypass this.)
    if args.hf_checkpoint is None:
        dest = f"{args.model_dir}/{args.model_name}"
        U.exec_command_cpu(f"hf download {args.model_org}/{args.model_name} " f"--local-dir {dest}")
    _ensure_4layer_model_type(args)
    _download_dataset(args)


@app.command()
@command_utils.dataclass_cli
def prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] CLI command wrapper around the private download helper.
    """Download HF checkpoint + dataset from HuggingFace. Run on one node (shared NFS)."""
    _prepare_download(args)


def _prepare_single(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: download the dataset, then cast the FP8 source checkpoint to BF16 into
    # model dir / bf16 name.
    U = args.create_backend()
    _download_dataset(args)

    src = _hf_checkpoint_path(args)
    U.fp8_cast_bf16(
        path_src=src,
        path_dst=f"{args.model_dir}/{args.bf16_name}/",
    )


@app.command()
@command_utils.dataclass_cli
def prepare_single(args: ScriptArgs):
    # [FUNCTION DOC] CLI command wrapper around the private FP8-to-BF16 helper.
    """FP8 -> BF16 cast for Megatron. Needs --hf-checkpoint (or pre-downloaded). One node."""
    _prepare_single(args)


def _prepare_spmd(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: convert the BF16 checkpoint to a Megatron torch_dist checkpoint through
    # the backend. Extra args: the implementation flag, expert TP 1, context parallel 1, plus a
    # parallel layout: single actor node with the 4-layer model uses TP/PP/EP all 1; single node
    # with the full Flash model uses TP 1, PP 1, EP 8; any other combination raises not-implemented
    # error. GPUs used: actor GPUs per node, or 1 for the 4-layer model. Multi-node conversion when
    # more than one actor node.
    U = args.create_backend()
    is_4layer = args.model_name == "DeepSeek-V4-Flash-FP8-4layer"
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    extra_args = f"--dsv4-impl {args.dsv4_impl} --expert-tensor-parallel-size 1 --context-parallel-size 1 "
    if actor_num_nodes == 1 and is_4layer:
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        )
    elif actor_num_nodes == 1 and args.model_name == "DeepSeek-V4-Flash-FP8":
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 8 "
        )
    else:
        raise NotImplementedError(
            f"No verified SPMD conversion config for {args.model_name} "
            f"({actor_num_nodes} actor nodes x {actor_num_gpus_per_node} GPUs/node). "
            f"Please specify your conversion parallel config in `run_deepseek_v4.py`."
        )

    num_gpus_for_convert = actor_num_gpus_per_node
    if is_4layer:
        num_gpus_for_convert = 1

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.bf16_name}",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_for_convert,
        multinode=True if actor_num_nodes > 1 else False,
        num_nodes=actor_num_nodes,
        extra_args=extra_args,
        dir_dst=f"{args.model_dir}",
        megatron_path=args.megatron_path,
    )


@app.command()
@command_utils.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    # [FUNCTION DOC] CLI command wrapper around the private conversion helper.
    _prepare_spmd(args)


def _prepare_cmd(args: ScriptArgs) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: pre-train commands per role. Empty mapping when local and shared model
    # dirs are the same; otherwise a mapping for the trainer role holding rsync commands (joined
    # with shell and-and) that copy the torch_dist checkpoint and the HF model dir to the local dir.
    if args.model_local_dir == args.model_dir:
        return {}

    copies = [
        command_utils.rsync_cmd(
            f"{args.model_dir}/{args.torch_dist_name}", f"{args.model_local_dir}/{args.torch_dist_name}"
        ),
        command_utils.rsync_cmd(f"{args.model_dir}/{args.model_name}", f"{args.model_local_dir}/{args.model_name}"),
    ]
    return {"trainer": " && ".join(copies)}


def _get_parallel_config(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: parallelism flags string. One actor node: TP = GPUs per node divided by
    # CP size (cp_size default 1), sequence parallel, PP 1, CP as given (allgather-cp flag added when
    # CP > 1), EP = GPUs per node, expert TP 1. Eight GPUs per node and 32 total (4 nodes): TP 4,
    # sequence parallel, PP 4 with 11 layers first stage and 10 last (43 layers), CP 1, EP 8,
    # expert TP 1. Anything else raises not-implemented error.
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    total_gpus = actor_num_nodes * actor_num_gpus_per_node

    # Single-node smoke-test configs
    if actor_num_nodes == 1:
        cp_size = args.cp_size or 1
        return (
            f"--tensor-model-parallel-size {actor_num_gpus_per_node // cp_size} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            f"--context-parallel-size {cp_size} "
            f"{'--allgather-cp ' if cp_size > 1 else ''}"
            f"--expert-model-parallel-size {actor_num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )

    if actor_num_gpus_per_node == 8:
        if total_gpus == 32:  # 4 nodes x 8 GPUs (MI355X, full Flash): TP4/PP4/EP8, 43 layers = 11+11+11+10
            return (
                "--tensor-model-parallel-size 4 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 4 "
                "--decoder-first-pipeline-num-layers 11 "
                "--decoder-last-pipeline-num-layers 10 "
                "--context-parallel-size 1 "
                # Raising context parallelism also needs --allgather-cp: DeepSeek V4 has no
                # zigzag CP path, and arguments.py asserts on the flag rather than setting it.
                # "--allgather-cp "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} GPUs. "
        f"Please specify your parallel config in `run_deepseek_v4._get_parallel_config`."
    )


def _train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: build the whole training argument string and submit it. Prints the
    # precision and topology; patches the 4-layer config. Blocks: checkpoint (HF path, ref-load from
    # local torch_dist; load / save under save dir / run id unless skip_saving); rollout (math reward,
    # shuffled, 3000 rollouts, temperature 0.8; non-debug adds oversampling and dynamic filter);
    # eval (interval 20 if enabled); task-specific data and lengths (dapo_aime: 8192 response tokens,
    # thinking mode, aime eval with 8 samples; gsm8k: 256 tokens); performance (parallel config plus
    # full uniform recompute, micro batch 1, 2048 tokens per GPU); GRPO and optimizer (Adam, lr 1e-6,
    # optional CPU offload, and on 4 actor nodes a 0.75 offload fraction without offloading train
    # weights); sglang (4-GPU engines, TP 4, EP 4, router health settings); misc (dropout off,
    # weight buffer 1 GiB, memory margin, colocate or rollout GPU count, optional dump / replay /
    # MIS / fault tolerance / routing replay / deterministic mode with NCCL ring / FP8 blockwise
    # training with BF16 / MTP speculative decoding with EAGLE). Environment variables for sglang
    # and aiter are set. Finally calls the backend execute with the model type and the prepare
    # command mapping. Mutates args when a debug train run id is set without a rollout id (sets 1).
    U = args.create_backend()
    print(f"[precision] fp8_training={args.fp8_training}")
    print(
        f"running on {args.num_nodes} nodes "
        f"({args.actor_num_nodes} actor nodes x {args.actor_num_gpus_per_node} GPUs/node, "
        f"{args.rollout_num_gpus} rollout GPUs, colocate={args.colocate})"
    )
    _ensure_4layer_model_type(args)

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} " f"--ref-load {args.model_local_dir}/{args.torch_dist_name} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} " f"--save {load_save_path} " "--save-interval 20 " "--save-retain-interval 20 "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len 8192 "
                """--apply-chat-template-kwargs '{"thinking_mode":"thinking"}' """
            )
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 4096 "
            )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    perf_args = _get_parallel_config(args)

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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
    if args.optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--use-precision-aware-optimizer " "--overlap-cpu-optimizer-d2h-h2d "
        )
        if args.actor_num_nodes == 4:
            # 4-node PP4 memory balance: partial optimizer offload (keep ~25% on GPU) + keep train
            # weights on GPU; pair with --sglang-mem-fraction-static 0.5.
            optimizer_args += "--optimizer-offload-fraction 0.75 " "--no-offload-train "

    sglang_world_size = 4
    sglang_tp_size = 4
    sglang_dp_size = 1
    sglang_ep_size = 4
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        f"--sglang-tp-size {sglang_tp_size} "
        f"--sglang-dp-size {sglang_dp_size} "
        f"--sglang-ep-size {sglang_ep_size} "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "  # TODO improve
    )
    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "0",
        "SGLANG_OPT_FP8_WO_A_GEMM": "0",
        "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
        # unified_kv lives in compressor_v2 only; on HIP the v1 path leaves
        # compress_kv_pool unset and the memory pool asserts on it.
        "SGLANG_OPT_USE_COMPRESSOR_V2": "true",
        "SGLANG_OPT_USE_TILELANG_INDEXER": "true",
        "SGLANG_OPT_USE_JIT_NORM": "true",
        "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "AITER_BF16_FP8_MOE_BOUND": "0",
    }

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--sglang-mem-fraction-static 0.5 "
        "--sglang-watchdog-timeout 1800 "  # ROCm: slow aiter gemm tune under colocate; avoid watchdog SIGQUIT
        "--accumulate-allreduce-grads-in-fp32 "
        f"--dsv4-impl {args.dsv4_impl} "  # ROCm has no cudnn/flash_mla path for the megatron impl
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format thd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    if args.colocate:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {args.rollout_num_gpus} "

    if args.dump_details:
        misc_args += f"--dump-details {args.debug_data_root}/{args.run_id}/dump_details "

    if args.enable_mis:
        misc_args += (
            "--use-tis "
            "--custom-config-path examples/infra_features/train_infer_mismatch_helper/mis.yaml "
            "--custom-tis-function-path examples.infra_features.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "

    if args.debug_train_run_id is not None:
        if args.debug_train_rollout_id is None:
            args.debug_train_rollout_id = 1
        misc_args += (
            f"--load-debug-rollout-data "
            f"{args.debug_data_root}/{args.debug_train_run_id}/dump_details/rollout_data/{args.debug_train_rollout_id}.pt "
        )
        misc_args += "--debug-train-only "

    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "
        # Skip indexer-replay for now
        # misc_args += "--use-rollout-indexer-replay "

    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    if args.fp8_training:
        misc_args += "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " "--fp8-recipe blockwise "
        misc_args += """--train-env-vars '{"NVTE_FP8_BLOCK_SCALING_FP32_SCALES":"0"}' """
        # ROCm TE MoE FP8 lacks fused wgrad accumulation; disable the fusion.
        misc_args += "--no-gradient-accumulation-fusion "

    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )
        # gfx950: use RCCL all-gather for speculative decoding; aiter can deadlock.
        extra_env_vars |= {"SGLANG_USE_AITER_AG": "false"}

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**extra_env_vars},
        megatron_path=args.megatron_path,
        prepare_cmd=_prepare_cmd(args),
    )


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] CLI command: run training only via the private train helper.
    """Run training. Assumes data/model/torch_dist are already prepared on {model_local_dir}."""
    _train(args)


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs):
    # [FUNCTION DOC] CLI command: download; cast to BF16 unless the BF16 dir already has its
    # safetensors index file; convert to torch_dist unless the latest-iteration marker exists;
    # default hf_checkpoint to local model dir / model name (mutating args); then train.
    _prepare_download(args)

    bf16_dir = Path(f"{args.model_dir}/{args.bf16_name}")
    bf16_sentinel = bf16_dir / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        _prepare_single(args)
    else:
        print(f"[full_train] Skipping FP8->BF16 cast: {bf16_sentinel} already exists.")

    torch_dist_dir = Path(f"{args.model_dir}/{args.torch_dist_name}")
    torch_dist_sentinel = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"[full_train] Skipping BF16->torch_dist conversion: {torch_dist_sentinel} already exists.")

    if args.hf_checkpoint is None:
        args.hf_checkpoint = f"{args.model_local_dir}/{args.model_name}"

    _train(args)


if __name__ == "__main__":
    app()
