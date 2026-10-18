# [MODULE DOC] End-to-end test of the tensor dumper: runs a one-sample GRPO step on Qwen3-30B-A3B
# (8 colocated GPUs; training with tensor 2, pipeline 2, context 2, expert 2, expert-tensor 2;
# rollout as one 4-GPU engine with data-parallel attention 4 to avoid a flashinfer all-reduce
# fusion crash) with the dumper enabled for inference, forward-only and forward-backward phases,
# source patchers for the training and inference sides written to temporary YAML files, and
# routing replay on. After training it verifies the dump directories (expected fields per phase)
# and runs the external comparator between the inference dumps (baseline) and the forward-backward
# dumps (target) with a relaxed difference threshold of 0.0085 (bf16 drift over 24+ layers) and an
# allow-skipped pattern for the auxiliary input tensors. Two modes exist: packed token layout
# (thd) and batch-sequence layout (bshd). Interface: a typer command-line app with commands run
# (full pipeline for one mode) and compare (re-run only verification on an existing dump
# directory); invoked without a command, it runs every mode in order. Module state: a temporary run
# directory created at import time (never removed) holding the patcher configs and dumps, the model
# name/type, 8 GPUs, mode-to-arguments mapping, expected dump fields, and the repository root which
# is prepended to the import path at import time. Registered for CUDA 8-GPU H100 (est. 900 s) and
# ROCm 8-GPU MI350 nightly (est. 1800 s).
# KNOWN ISSUE: the expected-fields mapping is keyed by "engine_*" while the verified patterns list
# uses "engines/engine_*", so the lookup for the engine pattern finds nothing and engine dumps are
# checked without expected fields.
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.
# The comparator must report all-passed with zero failures — no exceptions.

# Usage: This is a typer CLI with 2 commands:
#   python test_dumper.py run --mode <mode>        Full: prepare + execute + verify + comparator
#   python test_dumper.py compare --mode <mode> --dump-dir <path>
#                                                    Re-run comparator on existing dumps
#
# After running miles once (the expensive execute step), you can re-run the
# comparator many times via "compare" to investigate issues without re-running training.

import sys
import tempfile
from pathlib import Path
from typing import Annotated

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
if str(_MILES_ROOT) not in sys.path:
    sys.path.insert(0, str(_MILES_ROOT))

import typer
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci
from tests.e2e.conftest_dumper import (
    MEGATRON_PATCHER_YAMLS,
    SGLANG_SOURCE_PATCHER_CONFIG_YAML,
    SOURCE_PATCHED_FIELDS,
    check_dump_dir,
    clear_proxy_env,
    run_and_verify_comparator,
)

from miles.utils.external_utils import command_utils

register_cuda_ci(est_time=900, suite="stage-c-8-gpu-h100", labels=["short"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=1800, suite="nightly-stage-c-8-gpu-mi350", labels=["short"])


app: typer.Typer = typer.Typer()

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8

_RUN_DIR: Path = Path(tempfile.mkdtemp(prefix="test_miles_dumper_"))
MEGATRON_SOURCE_PATCHER_CONFIG_PATH: str = str(_RUN_DIR / "megatron_source_patcher.yaml")
SGLANG_SOURCE_PATCHER_CONFIG_PATH: str = str(_RUN_DIR / "sglang_source_patcher.yaml")

EXP_PATTERNS: list[str] = ["engines/engine_*", "fwd_only", "fwd_bwd"]

_MEGATRON_AUX_THD: list[str] = ["input_ids", "cu_seqlens_q", "cu_seqlens_kv", "qkv_format"]
_MEGATRON_AUX_BSHD: list[str] = ["input_ids"]

EXPECTED_FIELDS: dict[str, dict[str, list[str]]] = {
    "tp2_pp2_cp2_ep2_etp2": {
        "engine_*": ["input_ids", "positions"] + SOURCE_PATCHED_FIELDS,
        "fwd_only": _MEGATRON_AUX_THD + SOURCE_PATCHED_FIELDS,
        "fwd_bwd": _MEGATRON_AUX_THD + SOURCE_PATCHED_FIELDS,
    },
    "tp2_pp2_cp2_ep2_etp2_bshd": {
        "engine_*": ["input_ids", "positions"] + SOURCE_PATCHED_FIELDS,
        "fwd_only": _MEGATRON_AUX_BSHD + SOURCE_PATCHED_FIELDS,
        "fwd_bwd": _MEGATRON_AUX_BSHD + SOURCE_PATCHED_FIELDS,
    },
}

CONFIGS: dict[str, str] = {
    "tp2_pp2_cp2_ep2_etp2": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 2 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    ),
    "tp2_pp2_cp2_ep2_etp2_bshd": (
        "--tensor-model-parallel-size 2 --sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 2 --expert-tensor-parallel-size 2 "
        "--qkv-format bshd --micro-batch-size 1 "
    ),
}


def _resolve_mode(mode: str) -> tuple[str, str]:
    # [FUNCTION DOC] HELPER: Validates the mode name against the known configurations. Raises a
    # typer bad-parameter error listing valid modes if unknown. Returns the mode and its
    # performance-argument string.
    if mode not in CONFIGS:
        raise typer.BadParameter(f"Unknown mode {mode!r}, valid: {list(CONFIGS.keys())}")
    return mode, CONFIGS[mode]


def prepare(dump_dir: str, mode: str) -> None:
    # [FUNCTION DOC] Creates model/dataset directories, downloads Qwen3-30B-A3B and GSM8K,
    # converts the model to a Megatron distributed checkpoint on 8 GPUs, and force-removes the dump
    # directory. Then writes the training-side source patcher YAML (the bshd variant when the mode
    # name ends with the bshd suffix, else the thd variant) and the inference-side source patcher
    # YAML to the temporary run directory. Side effects: shell commands, network, filesystem, GPU.
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)
    U.exec_command_cpu(f"rm -rf {dump_dir}")

    megatron_yaml: str = MEGATRON_PATCHER_YAMLS["bshd" if mode.endswith("_bshd") else "thd"]
    Path(MEGATRON_SOURCE_PATCHER_CONFIG_PATH).write_text(megatron_yaml)
    Path(SGLANG_SOURCE_PATCHER_CONFIG_PATH).write_text(SGLANG_SOURCE_PATCHER_CONFIG_YAML)


def _execute(perf_args: str, dump_subdir: str, dump_dir: str) -> None:
    # [FUNCTION DOC] HELPER: Launches the one-step training with dumping into the dump directory's
    # subdirectory for the mode. Rollout: GSM8K parquet, math reward, shuffle, 1 rollout of 1 prompt
    # with 1 sample, global batch 1, only 3 response tokens, temperature 0.8, SGLang CUDA graphs
    # disabled (required for dumping). Optimizer: Adam with CPU offload and precision-aware
    # optimizer. GRPO with clip 0.2. The caller's performance arguments are included. SGLang: engine
    # of 4 GPUs, mem fraction 0.6, data parallel 4 with DP attention. Dumper arguments enable
    # dumping for inference, forward-only and forward-backward phases (no model value or gradient
    # dumps in the training phases) with a layer filter keeping non-layer tensors and layers below
    # 3 and layer 24, and point both source patchers at the temporary YAML files. Misc: dropouts
    # off, flash attention, colocated on 8 GPUs, alltoall dispatcher, routing replay from rollout.
    # Launches training with 8 GPUs. Returns nothing.
    U = command_utils.default_config().create_backend()
    full_dump_dir: str = f"{dump_dir}/{dump_subdir}"

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type math "
        "--rollout-max-response-len 3 --rollout-temperature 0.8 "
        # NOTE: Only generate 1 training sample
        "--num-rollout 1 --rollout-batch-size 1 --n-samples-per-prompt 1 --global-batch-size 1 "
        # NOTE: Must disable cuda graph to allow dumping
        "--sglang-disable-cuda-graph "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--optimizer-cpu-offload --use-precision-aware-optimizer "
    )

    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "

    sglang_args = (
        "--rollout-num-gpus-per-engine 4 "
        "--sglang-mem-fraction-static 0.6 "
        # Workaround: enable DP attention to avoid flashinfer allreduce fusion,
        # which crashes with base_gpu_id != 0 in colocate mode.
        # See https://github.com/flashinfer-ai/flashinfer/pull/2662
        "--sglang-dp-size 4 --sglang-enable-dp-attention "
    )

    dumper_filter: str = "'filter=layer_id is None or layer_id < 3 or layer_id == 24'"
    dumper_args = (
        f"--dumper-enable --dumper-dir {full_dump_dir} "
        f"--dumper-inference {dumper_filter} "
        f"--dumper-fwd-only enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-fwd-bwd enable_model_value=0 enable_model_grad=0 {dumper_filter} "
        f"--dumper-source-patcher-config-train {MEGATRON_SOURCE_PATCHER_CONFIG_PATH} "
        f"--dumper-source-patcher-config-inference {SGLANG_SOURCE_PATCHER_CONFIG_PATH} "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {NUM_GPUS} --colocate "
        "--moe-token-dispatcher-type alltoall "
        "--use-rollout-routing-replay "
    )

    train_args = " ".join(
        [
            ckpt_args,
            rollout_args,
            optimizer_args,
            grpo_args,
            perf_args,
            sglang_args,
            dumper_args,
            misc_args,
            command_utils.get_default_wandb_args(__file__),
        ]
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


def _verify_dumps(config_name: str, dump_subdir: str, dump_dir: str) -> None:
    # [FUNCTION DOC] HELPER: For each of the three dump patterns (engines, forward-only,
    # forward-backward) calls the dump-directory checker on the mode's dump subdirectory with the
    # expected fields for that pattern from the mode's table (none if the pattern key is missing),
    # then prints a success line. Raises whatever the checker raises, and a key error for an
    # unknown config name.
    base: Path = Path(dump_dir) / dump_subdir
    config_expected: dict[str, list[str]] = EXPECTED_FIELDS[config_name]
    for pattern in EXP_PATTERNS:
        check_dump_dir(base, pattern, expected_fields=config_expected.get(pattern))
    print(f"All dump verifications passed for {dump_subdir}!")


def _verify_comparator(dump_subdir: str, dump_dir: str) -> None:
    # [FUNCTION DOC] HELPER: Runs the comparator with the inference (engines) dumps as baseline and
    # the forward-backward dumps as target, using a diff threshold of 0.0085 (relaxed from the
    # default 0.001 because deep layers accumulate bf16 drift) and an allow-skipped pattern that
    # tolerates the input id, position, cumulative sequence length and qkv format entries being
    # skipped. The shared helper asserts the comparator reports all passed with no failures.
    baseline_dir: Path = Path(f"{dump_dir}/{dump_subdir}/engines")
    target_dir: Path = Path(f"{dump_dir}/{dump_subdir}/fwd_bwd")
    # Relax threshold: deep layers (e.g. layer 24 in PP stage 1)
    # accumulate bf16 numerical drift across 24+ transformer layers,
    # reaching ~0.008 rel_diff which exceeds the default 0.001 threshold.
    run_and_verify_comparator(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        extra_args=[
            "--diff-threshold",
            "0.0085",
            "--allow-skipped-pattern",
            "input_ids|positions|cu_seqlens_q|cu_seqlens_kv|qkv_format",
        ],
    )


@app.command()
def run(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
) -> None:
    # [FUNCTION DOC] Command-line command "run": the full pipeline for one mode. Resolves the mode
    # (bad-parameter error if unknown), puts dumps under a "dumps" folder in the temporary run
    # directory and prints the run directory, then prepares (downloads, conversion, patcher files),
    # clears proxy variables from the environment, runs training with dumping, verifies the dump
    # directories and verifies the comparator result.
    """Full pipeline: prepare + execute + verify + comparator."""
    config_name, perf_args = _resolve_mode(mode)
    dump_dir: str = str(_RUN_DIR / "dumps")
    print(f"Run directory: {_RUN_DIR}")

    prepare(dump_dir=dump_dir, mode=config_name)
    clear_proxy_env()
    _execute(perf_args=perf_args, dump_subdir=config_name, dump_dir=dump_dir)
    _verify_dumps(config_name=config_name, dump_subdir=config_name, dump_dir=dump_dir)
    _verify_comparator(dump_subdir=config_name, dump_dir=dump_dir)


@app.command()
def compare(
    mode: Annotated[str, typer.Option(help="Config mode: " + ", ".join(CONFIGS.keys()))],
    dump_dir: Annotated[str, typer.Option(help="Path to existing dump base directory")],
) -> None:
    # [FUNCTION DOC] Command-line command "compare": for an existing dump base directory and mode,
    # runs only the dump verification and the comparator verification (no preparation or training).
    # Raises a bad-parameter error for an unknown mode.
    """Re-run comparator on existing dumps (no training)."""
    config_name, _ = _resolve_mode(mode)

    _verify_dumps(config_name=config_name, dump_subdir=config_name, dump_dir=dump_dir)
    _verify_comparator(dump_subdir=config_name, dump_dir=dump_dir)


@app.callback(invoke_without_command=True)
def _ci_default(ctx: typer.Context) -> None:
    # [FUNCTION DOC] HELPER: Callback of the command-line app, invoked even without a subcommand
    # (the CI mode). If a subcommand was given it does nothing. Otherwise runs the full pipeline
    # sequentially for every configured mode in definition order (thd then bshd).
    if ctx.invoked_subcommand is not None:
        return
    for mode in CONFIGS:
        run(mode=mode)


if __name__ == "__main__":
    app()
