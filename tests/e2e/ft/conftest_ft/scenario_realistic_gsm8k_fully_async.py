# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.


# [MODULE DOC] Fully-async variant of the realistic gsm8k fault-recovery scenario: a thin typer app
# whose run command takes the same options as the base scenario (same defaults, minus the
# fully-async option) and delegates to it with the fully-async flag forced on.

import typer
from tests.e2e.ft.conftest_ft import scenario_realistic_gsm8k
from tests.e2e.ft.conftest_ft.cli_options import (
    MetricThresholdOption,
    NumRolloutOption,
    RolloutCrashIntervalSecondsOption,
    SeedOption,
    TrainerCrashIntervalSecondsOption,
)

app: typer.Typer = typer.Typer()


@app.command(name="run")
def run_ci(
    seed: SeedOption = scenario_realistic_gsm8k.DEFAULT_SEED,
    num_rollout: NumRolloutOption = scenario_realistic_gsm8k.DEFAULT_NUM_ROLLOUT,
    trainer_crash_interval_seconds: TrainerCrashIntervalSecondsOption = scenario_realistic_gsm8k.DEFAULT_TRAINER_CRASH_INTERVAL_SECONDS,
    rollout_crash_interval_seconds: RolloutCrashIntervalSecondsOption = scenario_realistic_gsm8k.DEFAULT_ROLLOUT_CRASH_INTERVAL_SECONDS,
    metric_threshold: MetricThresholdOption = scenario_realistic_gsm8k.DEFAULT_METRIC_THRESHOLD,
) -> None:
    # [FUNCTION DOC] Command entry: calls the base realistic gsm8k run with all given options and
    # fully-async forced to true.
    scenario_realistic_gsm8k.run_ci(
        seed=seed,
        num_rollout=num_rollout,
        trainer_crash_interval_seconds=trainer_crash_interval_seconds,
        rollout_crash_interval_seconds=rollout_crash_interval_seconds,
        metric_threshold=metric_threshold,
        fully_async=True,
    )


if __name__ == "__main__":
    app()
