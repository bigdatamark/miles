# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations

# [MODULE DOC] Shared typer option type aliases (annotated types carrying help text) reused by the
# FT test command-line apps. Each alias is an annotated type; none has a default here, defaults
# come from the commands using them. Aliases: mode (required string), optional mode (string or
# none; a suite whose scenario fixes its topology takes none), random seed (int), phase name
# (string, for multi-phase tests), dump base directory (optional string), enable-dumper (bool),
# number of train calls (int), number of rollouts (int), eval gsm8k accuracy threshold (float),
# fully-async flag (bool, trains through the async train script), mean seconds between trainer
# cell fault injections (float), and mean seconds between rollout engine injections (float).

from typing import Annotated

import typer

ModeOption = Annotated[str, typer.Option(help="Test mode variant")]
OptionalModeOption = Annotated[
    str | None, typer.Option(help="Test mode variant; a suite whose scenario fixes its topology takes none")
]
SeedOption = Annotated[int, typer.Option(help="Random seed for fault injection")]
PhaseOption = Annotated[str, typer.Option(help="Phase name (multi-phase tests)")]
DumpDirOption = Annotated[str | None, typer.Option(help="Dump base directory")]
EnableDumperOption = Annotated[bool, typer.Option(help="Enable dumper output")]
NumStepsOption = Annotated[int, typer.Option(help="Number of train() calls")]
NumRolloutOption = Annotated[int, typer.Option(help="Number of rollouts")]
MetricThresholdOption = Annotated[float, typer.Option(help="eval/gsm8k accuracy threshold")]
FullyAsyncOption = Annotated[bool, typer.Option(help="Train through train_async.py with --fully-async")]
TrainerCrashIntervalSecondsOption = Annotated[float, typer.Option(help="Mean seconds between trainer cell injections")]
RolloutCrashIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between rollout engine injections")
]
