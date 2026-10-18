# [MODULE DOC]
# HELPER: Scenario for the deterministic hot-restart deploy tests. Compares two deployed runs of
# the Qwen3-0.6B hot-restart example (a "baseline" side that trains straight through and a
# "target" side that is frozen at scheduled steps and taken over by hot restarts) on a Kubernetes
# cluster. Both train 6 rollouts with deterministic settings, a shared checkpoint directory
# per side, weight decay zero and mooncake object store arguments; the target's first freeze is
# armed through a freeze plan file and later freezes are written by the relaunch closure. Two
# modes: "checkpointed" (save every 2 steps; freeze after step 2 holding iteration 1, and after
# step 4 holding iteration 3; verdict: only post-checkpoint steps redone) and "no_checkpoint"
# (save every 4; freeze after step 1 with no checkpoint; verdict: redone from scratch). After the
# runs it checks the evidence (take-overs replaced only the script, redo accounting) and that the
# deterministic metrics and engine count of both sides match (at least 4 trained rollouts;
# weight-version metrics excluded). Module state: a mapping from dump directory to the training
# arguments installed there, so relaunches repeat exactly the installed arguments. Exposes a
# command-line app with one sub-command per mode.
import dataclasses
import hashlib
import shlex
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import typer
from examples.infra_features.hot_restart.run_qwen3_0_6b_hot_restart import ScriptArgs, build_train_args
from examples.infra_features.split_deployment.address_book import DEFAULT_TRAINER_ID
from tests.e2e.deploy.conftest_deploy.common.example_args import (
    assert_example_parallelism_matches,
    build_deterministic_test_args,
    build_script_args,
    with_replaced_value,
    without_weight_decay,
)
from tests.e2e.deploy.conftest_deploy.common.utils import compare_deterministic_sides, run_on_cluster
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_checkpoint import (
    assert_only_post_checkpoint_steps_redone,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_redone_from_scratch import (
    assert_unsaved_run_redone_from_scratch,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.assert_workloads import assert_take_overs_replaced_only_script
from tests.e2e.deploy.conftest_deploy.hot_restart.driver import (
    HotRestartDriver,
    ScheduledFreeze,
    compute_checkpoint_dir,
    compute_release_of_config,
    driving_hot_restarts,
    relaunch_with_hot_restart,
)
from tests.e2e.deploy.conftest_deploy.hot_restart.evidence import TRAIN_STEP_METRIC_KEY, HotRestartEvidence
from tests.e2e.deploy.conftest_deploy.hot_restart.freeze_plan import (
    arm_first_freeze,
    compute_freeze_plan_path,
    write_freeze_plan,
)
from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.execution import DATA_DIR, MODEL_DIR
from tests.e2e.ft.conftest_ft.modes import DENSE_MODEL_HF_REPO, DENSE_MODEL_NAME, DENSE_MODEL_TYPE, FTTestMode

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import ArgvManipulator, get_mooncake_object_store_args
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH

# ========================== constants and mode table ==========================


NUM_ROLLOUTS: int = 6
MIN_TRAINED_ROLLOUTS: int = 4
SAVE_FLAG: str = "--save"
LOAD_FLAG: str = "--load"
WANDB_GROUP_FLAG: str = "--wandb-group"
WANDB_RUN_ID_FLAG: str = "--wandb-run-id"
GLOBAL_BATCH_SIZE_FLAG: str = "--global-batch-size"
ROLLOUT_BATCH_SIZE_FLAG: str = "--rollout-batch-size"
SAMPLES_PER_PROMPT_FLAG: str = "--n-samples-per-prompt"
ASYNC_SAVE_FLAG: str = "--async-save"
_WEIGHT_VERSION_METRIC_KEYS: tuple[str, ...] = tuple(
    f"rollout/weight_version/{statistic}" for statistic in ("mean", "median", "max", "min")
)

_MODE: FTTestMode = FTTestMode(
    model_name=DENSE_MODEL_NAME,
    model_hf_repo=DENSE_MODEL_HF_REPO,
    megatron_model_type=DENSE_MODEL_TYPE,
    num_cells=2,
    train_gpus_per_node=4,
    rollout_num_engines=2,
    rollout_gpus_per_engine=1,
    parallel_args="--context-parallel-size 2",
)

_TRAIN_ARGS_OF_DUMP_DIR: dict[str, str] = {}


AssertRedoneFn = Callable[..., object]


@dataclass(frozen=True)
class HotRestartMode:
    # [CLASS DOC] HELPER: Immutable description of one hot-restart scenario: its name, the
    # checkpoint save interval in rollouts, the tuple of scheduled freezes, and the verdict function
    # (called with keyword arguments: dump directory, checkpoint directory, records, number of
    # rollouts and schedule) that checks which steps were redone.
    name: str
    save_interval: int
    schedule: tuple[ScheduledFreeze, ...]
    assert_redone: AssertRedoneFn

    @property
    def test_name(self) -> str:
        # [FUNCTION DOC] HELPER: The test name: "hot_restart_" followed by the mode name.
        return f"hot_restart_{self.name}"

    @property
    def num_restarts(self) -> int:
        # [FUNCTION DOC] HELPER: Number of scheduled take-overs.
        return len(self.schedule)

    @property
    def frozen_rollout_ids(self) -> tuple[int, ...]:
        # [FUNCTION DOC] HELPER: Tuple of the frozen rollout ids of the schedule, in order.
        return tuple(one.frozen_rollout_id for one in self.schedule)


CHECKPOINTED: HotRestartMode = HotRestartMode(
    name="checkpointed",
    save_interval=2,
    schedule=(
        ScheduledFreeze(frozen_rollout_id=2, saved_iteration=1),
        ScheduledFreeze(frozen_rollout_id=4, saved_iteration=3),
    ),
    assert_redone=assert_only_post_checkpoint_steps_redone,
)
NO_CHECKPOINT: HotRestartMode = HotRestartMode(
    name="no_checkpoint",
    save_interval=4,
    schedule=(ScheduledFreeze(frozen_rollout_id=1, saved_iteration=None),),
    assert_redone=assert_unsaved_run_redone_from_scratch,
)
MODES: tuple[HotRestartMode, ...] = (CHECKPOINTED, NO_CHECKPOINT)


# ================================ entry points ================================


def create_app_and_run_ci(restart_mode: HotRestartMode) -> tuple[typer.Typer, Callable[[], None]]:
    # [FUNCTION DOC] Build the comparison command-line app and CI entry point for a mode, via the
    # fault-tolerance comparison framework: baseline arguments plain, target arguments with the
    # first freeze armed, comparison function for the mode, a target-side context manager that
    # drives take-overs, a per-side configuration function that suffixes the run id with the side
    # name, and a mode resolver that always returns the single dense 4-GPU context-parallel mode.
    # Returns the app and the CI callable wrapped so it first asserts the cluster can deploy runs.
    app, run_ci = create_comparison_app_and_run_ci(
        test_name=restart_mode.test_name,
        build_baseline_args=partial(_build_args, restart_mode),
        build_target_args=partial(_build_frozen_args, restart_mode),
        compare_fn=partial(_compare, restart_mode),
        target_side_context=partial(_driving_take_overs_of, restart_mode),
        config_for_side=_config_for_comparison_side,
        resolve_mode_fn=lambda _name: _MODE,
    )
    return app, run_on_cluster(run_ci)


def run_ci(restart_mode: HotRestartMode) -> None:
    # [FUNCTION DOC] Build the CI entry point for the mode and call it immediately (running the
    # whole baseline-versus-target comparison on the cluster). Side effects: deploys and runs
    # training.
    create_app_and_run_ci(restart_mode)[1]()


def read_installed_args(dump_dir: str) -> str:
    # [FUNCTION DOC] Return the training arguments last installed for the dump directory in this
    # process; asserts they exist (a relaunch must repeat the installed arguments, since rebuilding
    # them would render a different pod template).
    assert (args := _TRAIN_ARGS_OF_DUMP_DIR.get(dump_dir)) is not None, (
        f"nothing installed a run under {dump_dir} in this process; a relaunch that rebuilds its arguments instead "
        f"of repeating the installed ones renders a pod template of its own"
    )
    return args


def _config_for_comparison_side(
    side: str, config: command_utils.ExecuteTrainConfig
) -> command_utils.ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Return a copy of the configuration whose run id gets the suffix "-"
    # plus the side name, giving each side its own release. Asserts the side is the baseline or
    # target name, and that the longer run id still fits the maximum run id length.
    assert side in (BASELINE_SIDE, TARGET_SIDE), f"unknown comparison side {side!r}"
    suffix = f"-{side}"
    assert len(config.run_id) + len(suffix) <= RUN_ID_MAX_LENGTH, "run id too long for a side suffix"
    return dataclasses.replace(config, run_id=f"{config.run_id}{suffix}")


# ========================== train argument building ===========================


def _build_args(restart_mode: HotRestartMode, mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    # [FUNCTION DOC] HELPER: Build the training argument string for one side. Builds the example's
    # script arguments, builds the example's training arguments and sets weight decay to zero;
    # points both the save and load flags at the side's checkpoint directory; if a tracking group
    # flag is present, sets the tracking group and run id flags to a group name derived from the
    # test name and a hash of the dump directory (so a relaunch continues the same tracking run);
    # appends the mooncake object store arguments; then asserts the mode's declared parallelism
    # matches, that one training event is logged per rollout, and that asynchronous save is not
    # used. Records the arguments in the module mapping for the dump directory and returns them.
    checkpoint_dir = str(compute_checkpoint_dir(dump_dir))
    script_args = _build_script_args(restart_mode, mode=mode, dump_dir=dump_dir, enable_dumper=enable_dumper)

    args = without_weight_decay(build_train_args(script_args))
    for flag in (SAVE_FLAG, LOAD_FLAG):
        args = with_replaced_value(args, flag=flag, value=checkpoint_dir)
    if ArgvManipulator.is_defined(shlex.split(args), WANDB_GROUP_FLAG):
        wandb_run_id = _compute_wandb_group(test_name=restart_mode.test_name, dump_dir=dump_dir)
        for flag in (WANDB_GROUP_FLAG, WANDB_RUN_ID_FLAG):
            args = with_replaced_value(args, flag=flag, value=wandb_run_id)
    args += get_mooncake_object_store_args()

    assert_example_parallelism_matches(mode, train_args=args)
    _assert_one_train_event_per_step(args)
    _assert_run_saves_before_step_report(args)
    _TRAIN_ARGS_OF_DUMP_DIR[dump_dir] = args
    return args


def _assert_one_train_event_per_step(train_args: str) -> None:
    # [FUNCTION DOC] HELPER: Parse the global batch size, rollout batch size and samples-per-prompt
    # values (each must be defined exactly once, else unpacking fails) and assert the global batch
    # size equals rollout batch size times samples per prompt, so each rollout takes exactly one
    # optimizer step and logs one training-step event (the redo accounting counts one per
    # rollout).
    argv = shlex.split(train_args)
    [global_batch_size] = ArgvManipulator.get(argv, GLOBAL_BATCH_SIZE_FLAG)
    [rollout_batch_size] = ArgvManipulator.get(argv, ROLLOUT_BATCH_SIZE_FLAG)
    [samples_per_prompt] = ArgvManipulator.get(argv, SAMPLES_PER_PROMPT_FLAG)

    assert int(global_batch_size) == int(rollout_batch_size) * int(samples_per_prompt), (
        f"the redo accounting of this scenario counts one {TRAIN_STEP_METRIC_KEY} event per rollout, and a run "
        f"whose global batch {global_batch_size} is not {rollout_batch_size} x {samples_per_prompt} takes several "
        f"optimizer steps per rollout and logs one event for each, so every attempt count would be a multiple of "
        f"what the schedule reasons about"
    )


def _assert_run_saves_before_step_report(train_args: str) -> None:
    # [FUNCTION DOC] HELPER: Assert the asynchronous-save flag is not present in the arguments
    # (an async checkpoint could land after the step that triggered it, breaking the pinned
    # checkpoint assumption of the take-overs).
    assert not ArgvManipulator.is_defined(shlex.split(train_args), ASYNC_SAVE_FLAG), (
        f"{ASYNC_SAVE_FLAG} lets a checkpoint land after the step that triggered it, and every take-over here is "
        f"pinned to the checkpoint the frozen run is already holding"
    )


def _build_frozen_args(
    restart_mode: HotRestartMode, mode: FTTestMode, dump_dir: str, enable_dumper: bool = True
) -> str:
    # [FUNCTION DOC] HELPER: Build the target side's arguments: the plain arguments for the side,
    # then arm the first scheduled freeze (deleting any stale sentinel, writing the freeze plan for
    # the first frozen rollout id, appending the plan flag). Stores the final arguments in the
    # module mapping for the dump directory (overwriting the unarmed ones) and returns them.
    # TODO ad hoc hack: revert after the args refactor
    args = arm_first_freeze(
        _build_args(restart_mode, mode, dump_dir, enable_dumper),
        side_dump_dir=dump_dir,
        frozen_rollout_id=restart_mode.frozen_rollout_ids[0],
    )
    _TRAIN_ARGS_OF_DUMP_DIR[dump_dir] = args
    return args


def _build_script_args(
    restart_mode: HotRestartMode, *, mode: FTTestMode, dump_dir: str, enable_dumper: bool
) -> ScriptArgs:
    # [FUNCTION DOC] HELPER: Build the example's script arguments. Asserts the mode has real
    # rollout engines and is not colocated (trainers and engines must stay up on separate GPUs
    # while the script is replaced), and that the freeze schedule leaves a redo window. Then
    # builds the arguments from the default execute-train configuration with: the mode's model name
    # and Megatron type, 6 rollouts, the mode's save interval, actor GPUs, engine count and GPUs per
    # engine, model and data directories, and extra deterministic-run arguments for the dump
    # directory and dumper switch.
    assert mode.has_real_rollout, (
        f"{restart_mode.test_name} replaces the rollout executor of a live run, and mode {mode.model_name} has no "
        f"engines for it to drive"
    )
    assert not mode.colocate, (
        f"{restart_mode.test_name} keeps a run's trainers and engines up while their script is replaced, and mode "
        f"{mode.model_name} colocates them on shared gpus"
    )

    assert_freeze_schedule_leaves_redo_window(restart_mode)

    return build_script_args(
        command_utils.default_config(),
        script_args_class=ScriptArgs,
        model_name=mode.model_name,
        megatron_model_type=mode.megatron_model_type,
        num_rollout=NUM_ROLLOUTS,
        save_interval=restart_mode.save_interval,
        actor_num_gpus=mode.train_gpus_per_node,
        num_engines=mode.rollout_num_engines,
        gpus_per_engine=mode.rollout_gpus_per_engine,
        model_dir=MODEL_DIR,
        data_dir=DATA_DIR,
        extra_args=build_deterministic_test_args(mode=mode, dump_dir=dump_dir, enable_dumper=enable_dumper),
    )


def _compute_wandb_group(*, test_name: str, dump_dir: str) -> str:
    # [FUNCTION DOC] HELPER: Return the test name, an underscore, and the first 12 hex characters of
    # the SHA-256 of the dump directory string.
    return f"{test_name}_{hashlib.sha256(dump_dir.encode()).hexdigest()[:12]}"


# ============================= take-over driving ==============================


@contextmanager
def _driving_take_overs_of(
    restart_mode: HotRestartMode, mode: FTTestMode, dump_dir: str, config: command_utils.ExecuteTrainConfig
) -> Iterator[None]:
    # [FUNCTION DOC] HELPER: Context manager used around the target side's launch. Computes the
    # release and freeze plan path for the side, deletes the side's dump directory (ignoring
    # errors), defines a relaunch closure (writes the freeze plan for the next frozen rollout id,
    # or an empty plan after the last, then relaunches with hot restart using the installed
    # arguments for the dump directory), builds a driver (event log directory, namespace, default
    # trainer id, schedule of the mode) and runs the body under the driver's context manager
    # (which writes the evidence file). After the body completes, asserts all scheduled restarts
    # happened.
    release = compute_release_of_config(config)
    # TODO ad hoc hack: revert after the args refactor
    plan_path = compute_freeze_plan_path(dump_dir)

    shutil.rmtree(dump_dir, ignore_errors=True)

    def relaunch(frozen_rollout_id: int | None) -> None:
        # [FUNCTION DOC] HELPER: Write the freeze plan for the given rollout id (empty plan if
        # none), then relaunch the installed release with the hot-restart configuration using the
        # installed arguments of the dump directory.
        write_freeze_plan(plan_path, frozen_rollout_id=frozen_rollout_id)
        relaunch_with_hot_restart(
            train_args=read_installed_args(dump_dir), mode=mode, config=config, installed_release=release
        )

    driver = HotRestartDriver(
        relaunch=relaunch,
        checkpoint_dir=compute_checkpoint_dir(dump_dir),
        events_dir=Path(dump_dir) / EVENTS_DIRNAME,
        release=release,
        namespace=config.namespace,
        trainer_id=DEFAULT_TRAINER_ID,
        freeze_plan_path=plan_path,
        schedule=restart_mode.schedule,
    )

    with driving_hot_restarts(driver, dump_dir=dump_dir):
        yield

    driver.assert_all_restarts_happened()


# ========================= comparison and assertions ==========================


def assert_freeze_schedule_leaves_redo_window(restart_mode: HotRestartMode) -> None:
    # [FUNCTION DOC] Validate a mode's schedule against its save interval. Asserts: the save
    # interval is below the number of rollouts; the largest frozen rollout id is below the last
    # rollout id (leaving a step after the last take-over that is not a redone one); frozen rollout
    # ids are all distinct (the driver tells freezes apart by the step in the sentinel). With the
    # set of rollout ids that save a checkpoint, asserts for each scheduled freeze: its frozen step
    # is not itself a save step; and its pinned saved iteration equals the latest save step below
    # the frozen step (or none when there is none).
    assert restart_mode.save_interval < NUM_ROLLOUTS, (
        f"{restart_mode.test_name} watches the run save after a take-over, and a run of {NUM_ROLLOUTS} step(s) "
        f"saving every {restart_mode.save_interval} only ever saves the last step it trains"
    )
    assert max(restart_mode.frozen_rollout_ids) < NUM_ROLLOUTS - 1, (
        f"{restart_mode.test_name} freezes the run after step {max(restart_mode.frozen_rollout_ids)} of "
        f"{NUM_ROLLOUTS}, leaving no step past the last take-over that is not a redone one"
    )
    frozen = restart_mode.frozen_rollout_ids
    assert len(set(frozen)) == len(frozen), (
        f"{restart_mode.test_name} freezes the run twice after the same step ({list(frozen)}), and the driver "
        f"tells one freeze from the one before it by the step the sentinel names: the sentinel a previous freeze "
        f"left would be read as this one, and the take-over would go before the run had parked"
    )

    saved = compute_saved_rollout_ids(save_interval=restart_mode.save_interval)
    for scheduled in restart_mode.schedule:
        assert scheduled.frozen_rollout_id not in saved, (
            f"{restart_mode.test_name} freezes the run after step {scheduled.frozen_rollout_id}, which a run "
            f"saving every {restart_mode.save_interval} step(s) checkpoints, so the take-over would redo nothing"
        )
        pinned = max((one for one in saved if one < scheduled.frozen_rollout_id), default=None)
        assert scheduled.saved_iteration == pinned, (
            f"{restart_mode.test_name} pins the take-over after step {scheduled.frozen_rollout_id} to iteration "
            f"{scheduled.saved_iteration}, and a run saving every {restart_mode.save_interval} step(s) holds "
            f"{pinned} by then: the take-over resumes from somewhere this scenario never reasoned about"
        )


def compute_saved_rollout_ids(*, save_interval: int) -> frozenset[int]:
    # [FUNCTION DOC] Return the rollout ids (zero-based, below the number of rollouts) after which
    # a checkpoint is saved: those whose one-based step number is a multiple of the save interval,
    # plus the final rollout.
    return frozenset(one for one in range(NUM_ROLLOUTS) if (one + 1) % save_interval == 0 or one == NUM_ROLLOUTS - 1)


def _compare(restart_mode: HotRestartMode, dump_dir: str, mode: FTTestMode) -> None:
    # [FUNCTION DOC] HELPER: Final comparison of both sides, then print a pass message. Loads the
    # target side's evidence; asserts take-overs replaced only the script (restart count taken from
    # the evidence records, minimum from the mode's schedule); calls the mode's redone verdict with
    # the target's dump and checkpoint directories, the records, 6 rollouts and the schedule; then
    # compares the deterministic metrics of baseline and target sides (at least 4 trained rollouts,
    # weight-version statistics excluded) and checks the engine count equals the mode's.
    baseline_dir: str = f"{dump_dir}/{BASELINE_SIDE}"
    target_dir: str = f"{dump_dir}/{TARGET_SIDE}"

    evidence = HotRestartEvidence.load(dump_dir=target_dir)
    assert_take_overs_replaced_only_script(
        evidence, num_restarts=len(evidence.records), minimum_restarts=restart_mode.num_restarts
    )
    restart_mode.assert_redone(
        dump_dir=target_dir,
        checkpoint_dir=str(compute_checkpoint_dir(target_dir)),
        records=evidence.records,
        num_rollouts=NUM_ROLLOUTS,
        schedule=restart_mode.schedule,
    )

    compare_deterministic_sides(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        expected_engine_count=mode.rollout_num_engines,
        min_trained_rollouts=MIN_TRAINED_ROLLOUTS,
        exclude_keys=list(_WEIGHT_VERSION_METRIC_KEYS),
    )

    print(f"Hot restart {restart_mode.name} comparison test PASSED")


# ================================= app wiring =================================


app: typer.Typer = typer.Typer()
for one in MODES:
    app.add_typer(create_app_and_run_ci(one)[0], name=one.name)

if __name__ == "__main__":
    app()
