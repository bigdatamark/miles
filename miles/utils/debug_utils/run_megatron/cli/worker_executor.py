# [MODULE DOC] Builds the strings used to launch the Megatron worker: the full torchrun shell
# command, the worker's argument string (Megatron flags plus script flags plus user extras), and
# the dumper environment variables. All outputs are plain strings or mappings; nothing is run
# here and values are not shell-quoted.
"""Build torchrun command and worker arguments."""

from pathlib import Path

from miles.utils.debug_utils.run_megatron.cli.parallel_utils import ParallelConfig
from miles.utils.debug_utils.run_megatron.worker.script_args import WORKER_SCRIPT_ARGS_BRIDGE, WorkerScriptArgs
from miles.utils.external_utils.model_args_utils import load_model_args


def build_torchrun_cmd(
    *,
    model_type: str,
    megatron_path: Path,
    nproc: int,
    worker_args: str,
) -> str:
    # [FUNCTION DOC] Returns one shell command string: sets the Python path to the Megatron path
    # in front of the existing one, sets the device max connections variable to 1, runs torchrun
    # with the given processes per node on the worker module, followed by the model arguments
    # loaded for the model type (may raise from the loader), dropout flags set to zero for hidden
    # and attention dropout, and the given worker argument string.
    """Build the full shell command to launch the worker via torchrun."""
    worker_module: str = "miles.utils.debug_utils.run_megatron.worker.main"

    cmd: str = (
        f"PYTHONPATH={megatron_path}:$PYTHONPATH "
        f"CUDA_DEVICE_MAX_CONNECTIONS=1 "
        f"torchrun --nproc-per-node {nproc} "
        f"-m {worker_module} "
        f"{load_model_args(model_type)} "
        f"--hidden-dropout 0 --attention-dropout 0 "
        f"{worker_args}"
    )
    return cmd


def build_worker_args(
    *,
    parallel: ParallelConfig,
    sp: bool,
    seq_length: int,
    batch_size: int,
    script_args: WorkerScriptArgs,
    extra_args: str,
) -> str:
    # [FUNCTION DOC] Routing replay is considered in use when the script args have either a dump
    # or a load path. Returns a space-joined string of: Megatron flags (see the flags helper), the
    # script flags produced by the bridge from the script args, and the user's extra args when
    # non-empty (appended last, so they come last on the command line).
    """Build the worker argument string.

    Megatron-native flags come from declarative tables in
    ``_build_megatron_flags``; ``--script-*`` flags come from the bridge.
    """
    use_routing_replay: bool = (
        script_args.routing_replay_dump_path is not None or script_args.routing_replay_load_path is not None
    )
    parts: list[str] = [
        _build_megatron_flags(
            parallel=parallel,
            sp=sp,
            seq_length=seq_length,
            batch_size=batch_size,
            use_routing_replay=use_routing_replay,
        ),
        WORKER_SCRIPT_ARGS_BRIDGE.to_cli_args(script_args),
    ]
    if extra_args:
        parts.append(extra_args)

    return " ".join(parts)


def build_dumper_env(
    *,
    output_dir: Path,
    run_backward: bool,
    dumper_filter: str,
) -> dict[str, str]:
    # [FUNCTION DOC] Returns a mapping of dumper environment variables: enable set to "1", the
    # dump directory set to the output directory, experiment name "standalone" (this is why
    # comparisons read the "standalone" subdirectory). Adds the filter variable only if the filter
    # is non-empty, and the model-gradient dump flag "1" only when running backward.
    """Build DUMPER_* environment variables for the worker."""
    env: dict[str, str] = {
        "DUMPER_ENABLE": "1",
        "DUMPER_DIR": str(output_dir),
        "DUMPER_EXP_NAME": "standalone",
    }
    if dumper_filter:
        env["DUMPER_FILTER"] = dumper_filter
    if run_backward:
        env["DUMPER_ENABLE_MODEL_GRAD"] = "1"
    return env


def _build_megatron_flags(
    *,
    parallel: ParallelConfig,
    sp: bool,
    seq_length: int,
    batch_size: int,
    use_routing_replay: bool,
) -> str:
    # [FUNCTION DOC] HELPER: Returns space-joined flags. Valued flags (always emitted unless the
    # value is none): tensor, pipeline, context, expert (effective) and expert-tensor parallel
    # sizes, sequence length, micro batch size and global batch size (both set to the batch size,
    # so one micro batch per step). Boolean flags emitted when true: sequence parallel (per
    # argument), bf16 (always), no gradient accumulation fusion (always), routing replay (per
    # argument).
    """Build Megatron-native CLI flags from declarative tables."""
    key_value_args: list[tuple[str, object | None]] = [
        ("--tensor-model-parallel-size", parallel.tp),
        ("--pipeline-model-parallel-size", parallel.pp),
        ("--context-parallel-size", parallel.cp),
        ("--expert-model-parallel-size", parallel.effective_ep),
        ("--expert-tensor-parallel-size", parallel.etp),
        ("--seq-length", seq_length),
        ("--micro-batch-size", batch_size),
        ("--global-batch-size", batch_size),
    ]

    bool_flags: list[tuple[str, bool]] = [
        ("--sequence-parallel", sp),
        ("--bf16", True),
        ("--no-gradient-accumulation-fusion", True),
        ("--use-routing-replay", use_routing_replay),
    ]

    parts: list[str] = []
    for flag, value in key_value_args:
        if value is not None:
            parts.append(f"{flag} {value}")
    for flag, condition in bool_flags:
        if condition:
            parts.append(flag)

    return " ".join(parts)
