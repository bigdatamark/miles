# [MODULE DOC]
# Launcher script (typer command line app) for GRPO training of NVIDIA Nemotron-3-Ultra-550B-A55B
# (108-layer hybrid state-space/attention latent mixture of experts) or a 4-layer slice, Megatron
# bridge mode (the Hugging Face checkpoint is loaded directly) with colocated SGLang, on DAPO math
# with optional AIME eval. Commands: full-train (download, train), prepare (download only), train.
# Parallel layouts: pruned models give all GPUs to expert parallel with tensor 1; the full model is
# only provided for 128 GPUs. Rollout engines use data-parallel attention so attention tensor size
# divides the state-space group count of 8. Module constant: the full model name.
"""
NVIDIA Nemotron-3-Ultra-550B-A55B GRPO RL training script.

=====================

nemotron_h is a hybrid Mamba2 + Attention + latent-MoE architecture (108 layers,
512 experts top-22, moe_latent_size=2048). It loads through NVIDIA
``megatron.bridge`` (``--megatron-to-hf-mode bridge``) plus the miles
NemotronHBridge MoE/latent shim in ``miles_plugins/megatron_bridge/nemotron_h.py``.

Tested on H200.

Please use the `radixark/miles:dev` docker image.

=====================

Args:
  --model-name: Model variant to use.
      NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16          Full 108-layer model (16 nodes)
      NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer   4-layer slice (single node)
  --num-nodes: Number of nodes. Determines the parallelism config (see _parallelism).
  --mode: "normal" or "debug_minimal" (shorter response length for quick testing)
  --check-weight-update-equal: Assert the Megatron -> SGLang weight sync restores
      every tensor exactly (poisons SGLang's weights first).

Weights load straight from the HF checkpoint: miles' load_checkpoint dispatches on
what --load points at, and an HF directory routes to _load_checkpoint_hf, i.e. the
same megatron.bridge path this model is mapped through. No offline Megatron dist
conversion step is needed.

Mamba n_groups=8 caps attention/mamba tensor-parallel at 8 (n_groups % tp == 0).
The 550B (~1.1TB bf16) does not fit one 8-GPU SGLang engine, so rollout uses
32-GPU engines with EP=32 + DP-attention (dp=4) so attention/mamba run at
attn_tp = 32/4 = 8.

Rollout routing-replay (--use-rollout-routing-replay) is NOT enabled for the
108-layer Ultra yet (the routing capturer shape needs a fix for per-layer top-22
under DP-attention); train/rollout logprob diff is ~0.01 without it.

=====================

I. Usage for single node 4-layer smoke test:
  `python scripts/run_nemotron_3_ultra_550b_a55b.py full-train \
      --model-name NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16-4layer --num-nodes 1`

=====================

II. Usage for the full model (16 nodes):

  1. Setup containers on all nodes

  2. Start Ray cluster on all nodes

  3. Download model/data. Run on **head node**.
       `python scripts/run_nemotron_3_ultra_550b_a55b.py prepare --num-nodes 16`

  4. Run training. Execute on head node; uses Ray internally.
       `python scripts/run_nemotron_3_ultra_550b_a55b.py train --num-nodes 16`
"""

import re
from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

app = typer.Typer()

FULL_MODEL_NAME = "NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    # [CLASS DOC]
    # Command line options extending the shared train configuration (num_nodes, output dir, backend
    # creation). Fields: mode (debug_minimal gives response length 256 instead of 8192), run id, org,
    # model name (full by default), Megatron model type (re-derived for slices), GPUs per node (None
    # derived), hardware (auto, H200, B200, GB300), eval switch (off), optimizer offload (on),
    # weight-update equality check (off), rollouts (30), 32 prompts x 8 samples, global batch 128,
    # tokens per GPU 1024, save interval 50, skip-saving switch, extra raw args, directories.
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = command_utils.create_run_id()
    model_org: str = "nvidia"
    model_name: str = FULL_MODEL_NAME
    megatron_model_type: str = "nemotron-3-ultra-550b-a55b"
    num_gpus_per_node: int | None = None
    hardware: Literal["auto", "H200", "B200", "GB300"] = "auto"
    enable_eval: bool = False
    enable_optimizer_offload: bool = True
    check_weight_update_equal: bool = False
    num_rollout: int = 30
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    global_batch_size: int = 128
    max_tokens_per_gpu: int = 1024
    save_interval: int = 50
    skip_saving: bool = False
    extra_args: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    def __post_init__(self):
        # [FUNCTION DOC]
        # Resolves hardware and GPUs per node. If the model name has a layer-count suffix, sets the
        # Megatron model type to the slice variant; otherwise, any name other than the full model name
        # raises a not implemented error.
        self.hardware = command_utils.resolve_hardware(self)
        self.num_gpus_per_node = self.num_gpus_per_node or command_utils.NUM_GPUS_OF_HARDWARE[self.hardware]
        if (m := re.search(r"(\d+)layer", self.model_name)) is not None:
            self.megatron_model_type = f"nemotron-3-ultra-550b-a55b-{m.group(1)}layer"
        elif self.model_name != FULL_MODEL_NAME:
            raise NotImplementedError(f"{self.model_name} is not supported")


def _is_pruned(args: ScriptArgs) -> bool:
    # [FUNCTION DOC] HELPER: True when the model name contains a number followed by "layer".
    return re.search(r"(\d+)layer", args.model_name) is not None


def _hf_checkpoint(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Returns the model folder path: model directory plus model name.
    return f"{args.model_dir}/{args.model_name}"


def _parallelism(args: ScriptArgs) -> tuple[int, int, int, int]:
    # [FUNCTION DOC] HELPER: Returns (tensor, pipeline, expert, expert-tensor) sizes. Pruned models:
    # (1, 1, total GPUs, 1). Full model: (8, 4, 32, 1) only when total GPUs is exactly 128; any other
    # total raises a not implemented error naming the GPU count and model.
    """(tp, pp, ep, etp). Mamba n_groups=8 caps attention/mamba tp at 8."""
    total_gpus = args.num_nodes * args.num_gpus_per_node
    if _is_pruned(args):
        # A few layers fit on one node; give every rank to expert parallel
        # (512 experts / EP8 = 64 per rank) and keep attention/mamba at tp 1.
        return 1, 1, total_gpus, 1
    if total_gpus == 128:
        return 8, 4, 32, 1
    raise NotImplementedError(f"No parallelism config for {total_gpus} GPUs with {args.model_name}")


def _sglang_args(args: ScriptArgs) -> str:
    # [FUNCTION DOC] HELPER: Builds the rollout engine flag string. Pruned models: engine size = GPUs per
    # node, data-parallel size 2, memory fraction 0.6. Full model: engine size 32, data-parallel size 4,
    # fraction 0.7. Asserts total GPUs is a multiple of the engine size, and that the attention tensor
    # size (engine / data-parallel) is integral and divides 8 (the state-space group count); raises
    # assertion errors otherwise. Emits engine size, expert size = engine size, data-parallel size,
    # data-parallel attention, memory fraction.
    total_gpus = args.num_nodes * args.num_gpus_per_node
    if _is_pruned(args):
        gpus_per_engine, dp_size, mem_fraction = args.num_gpus_per_node, 2, 0.6
    else:
        # The 550B does not fit one 8-GPU engine.
        gpus_per_engine, dp_size, mem_fraction = 32, 4, 0.7
    assert total_gpus % gpus_per_engine == 0, f"{total_gpus=} must be a multiple of {gpus_per_engine=}"
    attn_tp = gpus_per_engine // dp_size
    assert (
        gpus_per_engine % dp_size == 0 and 8 % attn_tp == 0
    ), f"attn_tp = {gpus_per_engine}/{dp_size} = {attn_tp} must divide Mamba n_groups=8"
    return (
        f"--rollout-num-gpus-per-engine {gpus_per_engine} "
        f"--sglang-ep-size {gpus_per_engine} "
        f"--sglang-dp-size {dp_size} "
        "--sglang-enable-dp-attention "
        f"--sglang-mem-fraction-static {mem_fraction} "
    )


def _prepare_download(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Creates model/data directories, downloads the Hugging Face model into the
    # model folder, the DAPO math dataset, and AIME 2024 only if eval is enabled.
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {args.model_org}/{args.model_name} --local-dir {_hf_checkpoint(args)}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    if args.enable_eval:
        U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)


def _execute_train(args: ScriptArgs):
    # [FUNCTION DOC] HELPER: Builds the training argument string and submits the job. Checkpoint: the
    # Hugging Face folder as hf checkpoint and reference load (routed through the bridge), bridge mode;
    # unless skipping, saves weights only (no optimizer) under the run folder at the save interval.
    # Rollout: DAPO math, deepscaler reward, chat template, shuffled, sizes from options, response
    # length 256 (debug) or 8192, temperature 1, balanced. Optional eval: AIME every 20 steps, 16
    # samples, length 16384, top-p 1. Parallelism from the helper (sequence parallel when tensor > 1),
    # context 1, full uniform recompute, dynamic batch with token budget, log-prob chunk 128. GRPO with
    # zero kl and entropy coefficients (no kl loss flag), asymmetric clip. Adam 1e-6 with optional CPU
    # offload group. SGLang flags from the helper. Misc: dropout off, fp32 accumulation and softmax,
    # auto attention backend, colocate, actor nodes and GPUs, rollout GPU total, dump details, optional
    # weight-update equality check. Environment variable skipping the SGLang kernel version guard.
    # Submits via the backend.
    # KNOWN ISSUE: the per-node GPU count flag (needed under colocate on smaller nodes) is not passed,
    # and the rollout GPU count flag is passed together with colocate, where it is ignored.
    # KNOWN ISSUE: the module docstring says routing replay is not enabled for the full model, and it
    # is not enabled for slices either (no replay flag anywhere).
    U = args.create_backend()
    tp, pp, ep, etp = _parallelism(args)
    total_gpus = args.num_nodes * args.num_gpus_per_node

    ckpt_args = (
        # --ref-load at an HF directory routes load_checkpoint to _load_checkpoint_hf,
        # i.e. megatron.bridge; --load defaults to it when there is nothing to resume.
        f"--hf-checkpoint {_hf_checkpoint(args)} "  # tokenizer + SGLang rollout
        f"--ref-load {_hf_checkpoint(args)} "
        "--megatron-to-hf-mode bridge "
    )
    if not args.skip_saving:
        ckpt_args += (
            f"--save {args.output_dir}/{args.run_id}/checkpoints "
            f"--save-interval {args.save_interval} "
            "--no-save-optim "  # weights-only
        )

    rollout_args = (
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {256 if args.mode == 'debug_minimal' else 8192} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )

    eval_args = ""
    if args.enable_eval:
        eval_args = (
            "--eval-interval 20 "
            f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
            "--n-samples-per-eval-prompt 16 "
            "--eval-max-response-len 16384 "
            "--eval-top-p 1 "
        )

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        f"--pipeline-model-parallel-size {pp} "
        "--context-parallel-size 1 "
        f"--expert-model-parallel-size {ep} "
        f"--expert-tensor-parallel-size {etp} "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
        "--log-probs-chunk-size 128 "
    )
    if tp > 1:
        perf_args += "--sequence-parallel "

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
    if args.enable_optimizer_offload:
        optimizer_args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend auto "
        "--colocate "
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--rollout-num-gpus {total_gpus} "
        f"--dump-details {args.output_dir}/{args.run_id}/dump_details "
    )
    if args.check_weight_update_equal:
        misc_args += "--check-weight-update-equal "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{_sglang_args(args)} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={
            # The nemotron DP-attention path uses existing kernels; skip the
            # blanket sgl-kernel version guard.
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1",
        },
        megatron_path=args.megatron_path,
    )


@app.command()
@command_utils.dataclass_cli
def full_train(args: ScriptArgs):
    # [FUNCTION DOC] Command "full-train": download, then train.
    """Full pipeline: download, train."""
    _prepare_download(args)
    _execute_train(args)


@app.command()
@command_utils.dataclass_cli
def prepare(args: ScriptArgs):
    # [FUNCTION DOC] Command "prepare": download only (run on head node).
    """Download model/data (run on head node)."""
    _prepare_download(args)


@app.command()
@command_utils.dataclass_cli
def train(args: ScriptArgs):
    # [FUNCTION DOC] Command "train": training only; assumes data and model are prepared.
    """Run training only (assumes data is prepared)."""
    _execute_train(args)


@app.callback()
def _callback() -> None:
    # [FUNCTION DOC] HELPER: Empty typer callback so commands stay sub-commands. Does nothing.
    pass


if __name__ == "__main__":
    app()
