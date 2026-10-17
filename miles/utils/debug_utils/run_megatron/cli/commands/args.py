# [MODULE DOC] Dataclasses describing the options of the run_megatron commands. Every field is
# created through a helper that attaches a help string in the field metadata, which the
# dataclass-to-command-line converter uses for option help. Classes: CommonRunArgs (options shared
# by run and run-and-compare), RunArgs (adds output directory and parallelism sizes),
# RunAndCompareArgs (baseline/target layouts and thresholds) and CompareArgs (dump directories and
# comparison options).
from __future__ import annotations

import dataclasses
from pathlib import Path


def _field(
    default: object = dataclasses.MISSING,
    *,
    help: str,
) -> dataclasses.Field:  # type: ignore[type-arg]
    # [FUNCTION DOC] HELPER: Creates a dataclass field with the given default (no default, i.e.
    # required, when omitted) and a metadata entry named help holding the help text.
    """Shorthand to create a dataclass field with typer metadata."""
    return dataclasses.field(default=default, metadata={"help": help})  # type: ignore[arg-type]


@dataclasses.dataclass
class CommonRunArgs:
    # [CLASS DOC] Options shared by every run. Required: model_type (name of a model definition
    # script under scripts/models) and hf_checkpoint (HuggingFace checkpoint path). Optional with
    # defaults: ref_load (Megatron checkpoint path, none), sp (sequence parallelism, off),
    # run_backward (off), prompt_mode (math; one of math / file / text), prompt_text, prompt_file,
    # seq_length (137), batch_size (micro batch, 1), apply_chat_template (off), role (actor; actor
    # or critic), source_patcher_config (YAML path), top_k (print top-k predictions per position;
    # 0 disables), dumper_filter (expression, empty), megatron_path (Megatron-LM location) and
    # extra_args (string of extra worker arguments, empty).
    model_type: str = _field(help="Model type matching scripts/models/{model_type}.py")
    hf_checkpoint: Path = _field(help="HuggingFace checkpoint path")
    ref_load: Path | None = _field(default=None, help="Megatron checkpoint path")
    sp: bool = _field(default=False, help="Enable sequence parallelism")
    run_backward: bool = _field(default=False, help="Run backward pass")
    prompt_mode: str = _field(default="math", help="Prompt mode: math / file / text")
    prompt_text: str | None = _field(default=None, help="Prompt text (for text mode)")
    prompt_file: Path | None = _field(default=None, help="Prompt file (for file mode)")
    # odd + somewhat large; also the fine-structure constant
    seq_length: int = _field(default=137, help="Sequence length")
    batch_size: int = _field(default=1, help="Micro batch size")
    apply_chat_template: bool = _field(default=False, help="Apply chat template")
    role: str = _field(default="actor", help="Model role: actor / critic")
    source_patcher_config: Path | None = _field(default=None, help="Source patcher YAML config path")
    top_k: int = _field(default=0, help="Print top-k predictions per position (0=disabled)")
    dumper_filter: str = _field(default="", help="Dumper filter expression")
    megatron_path: Path | None = _field(default=None, help="Path to Megatron-LM")
    extra_args: str = _field(default="", help="Extra args passed to worker")


@dataclasses.dataclass
class RunArgs(CommonRunArgs):
    # [CLASS DOC] Options of a single run. Adds: output_dir (default /tmp/run_megatron_dump), the
    # parallel sizes tp, pp, cp (default 1 each), ep (expert parallel, none meaning the tensor
    # parallel size per its help), etp (expert tensor parallel, 1), routing replay dump and load
    # paths, and logprob_output (directory for per-token logprob json files).
    output_dir: Path = _field(default=Path("/tmp/run_megatron_dump"), help="Dump output directory")
    tp: int = _field(default=1, help="Tensor parallel size")
    pp: int = _field(default=1, help="Pipeline parallel size")
    cp: int = _field(default=1, help="Context parallel size")
    ep: int | None = _field(default=None, help="Expert parallel size (default=tp)")
    etp: int = _field(default=1, help="Expert tensor parallel size")
    routing_replay_dump_path: Path | None = _field(default=None, help="Routing replay dump path")
    routing_replay_load_path: Path | None = _field(default=None, help="Routing replay load path")
    logprob_output: Path | None = _field(default=None, help="Directory to save per-token logprob JSON files")


@dataclasses.dataclass(kw_only=True)
class RunAndCompareArgs(CommonRunArgs):
    # [CLASS DOC] Options of a baseline-versus-target run (keyword-only fields). Required:
    # output_base_dir, baseline and target (parallel layouts written as command-line flag strings
    # such as tensor parallel 1 / context parallel 2). Optional: routing_replay (record on the
    # baseline, replay on the target; off), compare_logprobs (on), baseline_extra_args and
    # target_extra_args (appended to the common extra arguments for that run only), diff_threshold
    # (activation pass/fail threshold) and logprob_threshold (max absolute logprob difference).
    output_base_dir: Path = _field(help="Base output directory for dumps")
    baseline: str = _field(help='Baseline parallel config, e.g. "--tp 1 --cp 1"')
    target: str = _field(help='Target parallel config, e.g. "--tp 2 --cp 2"')
    routing_replay: bool = _field(
        default=False,
        help="Enable routing replay (record on baseline, replay on target)",
    )
    compare_logprobs: bool = _field(
        default=True,
        help="Compute and compare per-token logprobs between baseline and target",
    )
    baseline_extra_args: str = _field(default="", help="Extra megatron args for baseline run only")
    target_extra_args: str = _field(default="", help="Extra megatron args for target run only")
    diff_threshold: float | None = _field(default=None, help="Activation diff pass/fail threshold")
    logprob_threshold: float | None = _field(default=None, help="Logprob max abs diff threshold")


@dataclasses.dataclass
class CompareArgs:
    # [CLASS DOC] Options of a comparison of two existing dumps. Required: baseline_dir and
    # target_dir. Optional: output_format (text; text or json), override dimension strings for
    # baseline and target, patch_config YAML path, diff_threshold, baseline and target logprob
    # directories (both needed for the logprob comparison) and logprob_threshold.
    baseline_dir: Path = _field(help="Baseline dump directory")
    target_dir: Path = _field(help="Target dump directory")
    output_format: str = _field(default="text", help="Output format: text / json")
    override_baseline_dims: str | None = _field(default=None, help="Override baseline dims")
    override_target_dims: str | None = _field(default=None, help="Override target dims")
    patch_config: Path | None = _field(default=None, help="Patch config YAML path")
    diff_threshold: float | None = _field(default=None, help="Pass/fail threshold")
    baseline_logprob_dir: Path | None = _field(default=None, help="Baseline logprob JSON directory")
    target_logprob_dir: Path | None = _field(default=None, help="Target logprob JSON directory")
    logprob_threshold: float | None = _field(default=None, help="Logprob max abs diff threshold")
