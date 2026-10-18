# [MODULE DOC]
# HELPER: Scenario for the realistic GSM8K hot-restart deploy test: a long randomized soak of a
# deployed training run on GSM8K during which the orchestration script is taken over (hot
# restarted) at random times by the fault-injection framework, using a single virtual
# "hot-restart" cell type with two fake healthy cells, which exist only while the run is within
# the injection window (not in the last 15 rollouts). Checkpoints are saved every 3 rollouts into
# the run's checkpoint directory (loaded from there too) and fault tolerance is disabled.
# After the run: relaunch threads are joined, installs must have been clean and no injection may
# have failed; the evidence is written and checked (only the script was replaced, at least one
# restart; each take-over lost at most save interval plus one steps; the logs rolled aside by
# take-overs show the resume points are consistent). Provides a command-line app with a "run"
# command taking seed, rollout count, metric threshold and mean seconds between take-overs
# (default 600). Constants: test name, save interval 3, minimum restarts 1, the quiescence
# length, the virtual cell type and names.
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.common.utils import assert_cluster_can_deploy_runs
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_checkpoint import (
    read_discarded_event_dirs,
    read_step_events,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import assert_take_overs_replaced_only_script
from tests.e2e.deploy.conftest_deploy.hot_restart.cluster_observer import ClusterObserver, observing_cluster
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import compute_checkpoint_dir, compute_release_of_config
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import HotRestartEvidence, HotRestartRecord
from tests.e2e.deploy.conftest_deploy.hot_restart.fault_form import HOT_RESTART_FORM_NAME, HotRestartFaultForm
from tests.e2e.ft.conftest_ft.cli_options import MetricThresholdOption, NumRolloutOption, SeedOption
from tests.e2e.ft.conftest_ft.fault_injection.fault_forms import CellFaultForms
from tests.e2e.ft.conftest_ft.fault_injection.state import Event, InjectionEvent
from tests.e2e.ft.conftest_ft.fault_injection.views import compute_num_successful_injections_of_form
from tests.e2e.ft.conftest_ft.scenario_realistic_gsm8k import (
    DEFAULT_METRIC_THRESHOLD,
    DEFAULT_NUM_ROLLOUT,
    DEFAULT_SEED,
    Gsm8kRun,
    run_realistic_gsm8k,
)

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.misc import MutableBox

app: typer.Typer = typer.Typer()

TEST_NAME: str = "hot_restart_realistic_gsm8k"
SAVE_INTERVAL: int = 3
MIN_HOT_RESTARTS: int = 1
MAX_REDONE_STEPS_PER_TAKE_OVER: int = SAVE_INTERVAL + 1
DEFAULT_HOT_RESTART_INTERVAL_SECONDS: float = 600.0
TERMINAL_QUIESCENCE_ROLLOUTS: int = 15
_HOT_RESTART_CELL_TYPE: str = "hot-restart-virtual-cell"
_VIRTUAL_CELL_NAMES: tuple[str, str] = ("hot-restart-virtual-cell-0", "hot-restart-virtual-cell-1")

HotRestartIntervalSecondsOption = Annotated[
    float, typer.Option(help="Mean seconds between take-overs of the orchestration script")
]


@app.command(name="run")
def run_ci(
    seed: SeedOption = DEFAULT_SEED,
    num_rollout: NumRolloutOption = DEFAULT_NUM_ROLLOUT,
    metric_threshold: MetricThresholdOption = DEFAULT_METRIC_THRESHOLD,
    hot_restart_interval_seconds: HotRestartIntervalSecondsOption = DEFAULT_HOT_RESTART_INTERVAL_SECONDS,
) -> None:
    # [FUNCTION DOC] Command-line entry "run". Asserts the cluster can deploy runs; creates a
    # cluster observer for the run's release; computes the largest rollout id still allowed for
    # injection (number of rollouts minus the 15 quiescence rollouts minus one). Runs the
    # realistic GSM8K soak inside the observer's polling context with: the hot-restart cell type
    # mean interval, a form-creating callback (builds the fault forms once, asserting not built
    # twice, and keeps the single hot-restart form), a virtual cell provider, extra training
    # arguments (checkpoint save/load flags, the tracking run id equal to the run id, and the
    # weight-update checker disabled), not fully asynchronous, fault tolerance disabled. Afterwards
    # asserts a form was built; joins its relaunch threads; asserts clean installs and that no
    # take-over injection event failed; builds and writes the evidence from the form's records
    # and the observer's data under the run's dump directory; asserts only the script was
    # replaced (restart count = successful injections of the form, minimum 1); asserts each
    # take-over lost at most 4 steps and resumed within the save interval according to the
    # rolled-aside logs; prints a pass line with seed and rollouts.
    config = command_utils.default_config()
    assert_cluster_can_deploy_runs(config)

    observer = ClusterObserver(
        release=compute_release_of_config(config), namespace=config.namespace, trainer_id=DEFAULT_TRAINER_ID
    )
    hot_restart_form: MutableBox[HotRestartFaultForm | None] = MutableBox(value=None)
    max_allowed_rollout_id = num_rollout - TERMINAL_QUIESCENCE_ROLLOUTS - 1

    def create_forms(run: Gsm8kRun) -> CellFaultForms:
        # [FUNCTION DOC] HELPER: Callback building the fault forms for a run: create the hot-restart
        # forms with the maximum allowed rollout id, assert the stored form box is still empty (the
        # forms must be built once), store the single form of the hot-restart cell type in the box,
        # and return the forms mapping.
        forms = create_hot_restart_forms(run, max_allowed_rollout_id=max_allowed_rollout_id)
        assert hot_restart_form.value is None, (
            "the run's fault forms were built twice, so the form this soak reads at the end is not the one the "
            "second run was injected with"
        )
        [hot_restart_form.value] = forms[_HOT_RESTART_CELL_TYPE]
        return forms

    with observing_cluster(observer):
        outcome = run_realistic_gsm8k(
            config=config,
            test_name=TEST_NAME,
            seed=seed,
            num_rollout=num_rollout,
            metric_threshold=metric_threshold,
            fully_async=False,
            mean_interval_seconds_of_cell_type={_HOT_RESTART_CELL_TYPE: hot_restart_interval_seconds},
            create_forms=create_forms,
            get_virtual_cells=lambda: _create_virtual_cells_before(hot_restart_form.value),
            build_extra_train_args=lambda dump_dir: _build_train_args(dump_dir, wandb_run_id=config.run_id),
            enable_fault_tolerance=False,
        )

    form = hot_restart_form.value
    assert form is not None, "no fault form was ever built for this run, so nothing here was ever taken over"

    form.join_relaunches()
    form.assert_take_overs_installed_cleanly()
    assert_no_take_over_attempt_failed(outcome.injector.event_log.events)

    evidence = HotRestartEvidence(
        records=form.records,
        snapshots=tuple(observer.snapshots),
        release=observer.release,
        observation_attempts=observer.attempts,
        observation_failures=observer.failures,
    )
    evidence.write(dump_dir=outcome.run.dump_dir)
    assert_take_overs_replaced_only_script(
        evidence,
        num_restarts=compute_num_successful_injections_of_form(
            outcome.injector.event_log.events, form_name=HOT_RESTART_FORM_NAME
        ),
        minimum_restarts=MIN_HOT_RESTARTS,
    )
    assert_take_over_loss_within_save_interval(evidence.records)
    assert_take_overs_resumed_within_save_interval(outcome.run.dump_dir, records=evidence.records)

    print(f"Hot restart realistic gsm8k test PASSED (seed={seed}, rollouts={num_rollout})")


def _build_train_args(dump_dir: str, *, wandb_run_id: str) -> str:
    # [FUNCTION DOC] HELPER: Return the extra training arguments: the checkpoint save/load/interval
    # flags for the dump directory, the tracking run id flag set to the given id (so relaunches
    # continue one tracking run), and the flag disabling the weight-update checker (a hot restart
    # restarts the script, so the checker would not be meaningful).
    return build_checkpoint_args(dump_dir) + f"--wandb-run-id {wandb_run_id} " + "--ci-disable-weight-update-checker "


def assert_no_take_over_attempt_failed(events: list[Event]) -> None:
    # [FUNCTION DOC] Assert that no injection event of the hot-restart form is marked unsuccessful
    # (every draw of this form fires, so a failure means the cluster refused the relaunch or it
    # never reached the run). Failure message lists the failed events.
    failed = [
        one
        for one in events
        if isinstance(one, InjectionEvent) and one.form_name == HOT_RESTART_FORM_NAME and not one.succeeded
    ]

    assert not failed, (
        f"{len(failed)} take-over attempt(s) failed: {failed}. Every draw of this form fires, so a failure here is "
        f"a relaunch the cluster refused or one that never reached the run, not a draw that was declined"
    )


def assert_take_over_loss_within_save_interval(records: Sequence[HotRestartRecord]) -> None:
    # [FUNCTION DOC] For each record, compute the resume point (the saved iteration at trigger, or
    # minus one if none) and the number of redone steps as frozen rollout id (the progress at
    # trigger) minus the resume point; assert it is between 0 and the save interval plus one
    # (4) inclusive.
    for record in records:
        resumed_from = -1 if record.saved_iteration_at_trigger is None else record.saved_iteration_at_trigger
        redone = record.frozen_rollout_id - resumed_from

        assert 0 <= redone <= MAX_REDONE_STEPS_PER_TAKE_OVER, (
            f"take-over {record.index} was drawn against a run standing at step {record.frozen_rollout_id} holding "
            f"iteration {record.saved_iteration_at_trigger}, so it threw away {redone} step(s); a take-over resumes "
            f"from the last checkpoint and a run saving every {SAVE_INTERVAL} step(s) cannot owe more than "
            f"{MAX_REDONE_STEPS_PER_TAKE_OVER}"
        )


def assert_take_overs_resumed_within_save_interval(dump_dir: str, *, records: Sequence[HotRestartRecord]) -> None:
    # [FUNCTION DOC] Verify resume points from the event logs. Reads the rolled-aside logs (one
    # per record) plus the surviving log. For each record with its log and the following log:
    # the frozen step is the log's highest step (minus one if empty); the steps whose entries
    # are unchanged in the next log must form a contiguous prefix from zero (the carried steps),
    # whose end is the resume point; the redone count (frozen minus resume) must be between 0 and
    # save interval plus one. Raises assertion errors with explanatory messages.
    logs = _read_replaced_logs(dump_dir, num_take_overs=len(records))

    for record, log, later_log in zip(records, logs[:-1], logs[1:], strict=True):
        frozen_rollout_id = max(log, default=-1)
        survived = sorted(rollout_id for rollout_id, event in log.items() if later_log.get(rollout_id) == event)
        resumed_from = max(survived, default=-1)

        assert survived == list(
            range(resumed_from + 1)
        ), f"take-over {record.index} carried the steps {survived} over, expected {list(range(resumed_from + 1))}"
        redone = frozen_rollout_id - resumed_from
        assert 0 <= redone <= MAX_REDONE_STEPS_PER_TAKE_OVER, (
            f"take-over {record.index} replaced a log that had reached step {frozen_rollout_id} and resumed at "
            f"step {resumed_from}, so it redid {redone} step(s), more than {MAX_REDONE_STEPS_PER_TAKE_OVER}"
        )


def _read_replaced_logs(dump_dir: str, *, num_take_overs: int) -> list[dict[int, str]]:
    # [FUNCTION DOC] HELPER: Return the per-step event entries of every log in chronological order:
    # each rolled-aside log, then the surviving live log. Asserts the number of rolled-aside
    # directories equals the number of take-overs and that their roll-aside timestamps (second
    # resolution) are distinct and in sorted order (name order equals time order).
    discarded_dirs = read_discarded_event_dirs(dump_dir)
    assert len(discarded_dirs) == num_take_overs, (
        f"every take-over rolls the log it replaced aside, but {num_take_overs} of them left "
        f"{[one.name for one in discarded_dirs]} under {dump_dir}"
    )

    rolled_aside_at = [_read_log_rollaside_times(one) for one in discarded_dirs]
    assert (
        sorted(set(rolled_aside_at)) == rolled_aside_at
    ), f"the take-overs under {dump_dir} rolled their logs aside at {rolled_aside_at}, two in the same second"

    replaced = [_read_finished_steps_of_log(one) for one in discarded_dirs]
    return [*replaced, _read_finished_steps_of_log(Path(dump_dir) / EVENTS_DIRNAME)]


def _read_log_rollaside_times(events_dir: Path) -> str:
    # [FUNCTION DOC] HELPER: Parse the directory name against the pattern ".trash_" followed by
    # an 8-digit date, underscore, 6-digit time, underscore and hexadecimal suffix; assert it
    # matches and return the date-time part (as text).
    matched = re.fullmatch(r"\.trash_(\d{8}_\d{6})_[0-9a-f]+", events_dir.name)
    assert matched is not None, f"{events_dir.name} does not name the moment the log was rolled aside"
    return matched.group(1)


def _read_finished_steps_of_log(events_dir: Path) -> dict[int, str]:
    # [FUNCTION DOC] HELPER: Read training-step events from the directory, assert each rollout id
    # has exactly one entry, and return a mapping from rollout id to that entry.
    logged = read_step_events(events_dir)
    repeated = {rollout_id: len(events) for rollout_id, events in logged.items() if len(events) != 1}
    assert not repeated, f"{events_dir} describes the step(s) {repeated} more than once"
    return {rollout_id: events[0] for rollout_id, events in logged.items()}


def create_hot_restart_forms(run: Gsm8kRun, *, max_allowed_rollout_id: int) -> CellFaultForms:
    # [FUNCTION DOC] Create the single hot-restart fault form for a run (using the run's launch
    # function, configuration, checkpoint directory under its dump directory, events directory and
    # the maximum allowed rollout id). Returns a mapping from the hot-restart virtual cell type
    # to a one-element list holding the form.
    form = HotRestartFaultForm(
        launch=run.launch,
        config=run.config,
        checkpoint_dir=compute_checkpoint_dir(run.dump_dir),
        events_dir=run.events_dir,
        max_allowed_rollout_id=max_allowed_rollout_id,
    )
    return {_HOT_RESTART_CELL_TYPE: [form]}


def _create_virtual_cells() -> list[dict]:
    # [FUNCTION DOC] HELPER: Build two fake cluster cell objects (names "hot-restart-virtual-cell-0"
    # and "-1"), each with the cell-type label of the hot-restart type, phase Running and a
    # Healthy condition true. They exist only so the injection framework has cells to draw.
    return [
        {
            "metadata": {"name": name, "labels": {"miles.io/cell-type": _HOT_RESTART_CELL_TYPE}},
            "status": {"phase": "Running", "conditions": [{"type": "Healthy", "status": "True"}]},
        }
        for name in _VIRTUAL_CELL_NAMES
    ]


def _create_virtual_cells_before(form: HotRestartFaultForm | None) -> list[dict]:
    # [FUNCTION DOC] HELPER: Return the fake cells while the form exists and says the run is within
    # its injection window; otherwise an empty list (no cells to draw, so no further injection).
    if form is None or not form.is_within_injection_window():
        return []
    return _create_virtual_cells()


def build_checkpoint_args(dump_dir: str) -> str:
    # [FUNCTION DOC] Return the argument string "--save <dir> --load <dir> --save-interval 3 " where
    # the directory is the run's checkpoint directory under the dump directory (trailing space
    # included).
    checkpoint_dir = compute_checkpoint_dir(dump_dir)
    return f"--save {checkpoint_dir} --load {checkpoint_dir} --save-interval {SAVE_INTERVAL} "


if __name__ == "__main__":
    app()
