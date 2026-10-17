# [MODULE DOC] The compare command: compares activation dumps of a baseline and a target run by
# invoking sglang's comparator module in a subprocess (preset sglang_megatron), and optionally
# compares per-token logprobs. Exits the process with status 1 if either comparison fails.
"""``compare`` CLI command."""

import subprocess
import sys

import typer

from miles.utils.debug_utils.run_megatron.cli.commands.args import CompareArgs
from miles.utils.debug_utils.run_megatron.logprob_comparator import compare_logprobs
from miles.utils.external_utils.command_utils.common import run_shell_command
from miles.utils.typer_utils import dataclass_cli


def register(app: typer.Typer) -> None:
    # [FUNCTION DOC] HELPER: Adds the compare function as a command of the given typer app.
    """Register the ``compare`` command on *app*."""
    app.command()(compare)


def compare_impl(args: CompareArgs) -> None:
    # [FUNCTION DOC] Runs the activation comparison, then, only if both logprob directories are
    # given, the logprob comparison with the given threshold (default 0.001 when none). If either
    # failed, prints a line naming the failed comparisons and exits the process with status 1
    # (raises system exit); otherwise prints a completion line. Both comparisons always run, even
    # if the first fails.
    """Core compare logic, called by both ``compare`` command and ``run_and_compare``."""
    activation_passed = _run_activation_comparison(args)

    logprob_passed = True
    if args.baseline_logprob_dir is not None and args.target_logprob_dir is not None:
        logprob_passed = compare_logprobs(
            baseline_dir=args.baseline_logprob_dir,
            target_dir=args.target_logprob_dir,
            threshold=args.logprob_threshold if args.logprob_threshold is not None else 1e-3,
        )

    if not activation_passed or not logprob_passed:
        failures: list[str] = []
        if not activation_passed:
            failures.append("activation comparison")
        if not logprob_passed:
            failures.append("logprob comparison")
        print(f"[cli] FAILED: {', '.join(failures)}", flush=True)
        sys.exit(1)

    print("[cli] Compare completed.", flush=True)


def _run_activation_comparison(args: CompareArgs) -> bool:
    # [FUNCTION DOC] HELPER: Builds a command line that runs the sglang comparator module with the
    # current Python interpreter, passing baseline path, target path, output format and the
    # sglang_megatron preset, plus each optional flag (override baseline dims, override target
    # dims, patch config, diff threshold) only when set. Joins the parts with spaces (no quoting)
    # and runs it through the shared shell-command runner. Returns true on success; returns false
    # (after printing a message) if the process exits with a failure status.
    cmd_parts: list[str] = [
        sys.executable,
        "-m",
        "sglang.srt.debug_utils.comparator",
        "--baseline-path",
        str(args.baseline_dir),
        "--target-path",
        str(args.target_dir),
        "--output-format",
        args.output_format,
        "--preset",
        "sglang_megatron",
    ]

    optional_args: dict[str, object | None] = {
        "--override-baseline-dims": args.override_baseline_dims,
        "--override-target-dims": args.override_target_dims,
        "--patch-config": args.patch_config,
        "--diff-threshold": args.diff_threshold,
    }
    for flag, value in optional_args.items():
        if value is not None:
            cmd_parts.extend([flag, str(value)])

    try:
        run_shell_command(" ".join(cmd_parts))
        return True
    except subprocess.CalledProcessError:
        print("[cli] Activation comparison failed", flush=True)
        return False


@dataclass_cli(env_var_prefix="")
def compare(args: CompareArgs) -> None:
    # [FUNCTION DOC] The compare command: options come from the dataclass with no environment
    # variable prefix; delegates to the shared comparison routine.
    """Run comparator on existing dump directories."""
    compare_impl(args)
