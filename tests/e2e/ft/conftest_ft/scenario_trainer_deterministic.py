# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

# [MODULE DOC] Deterministic trainer-healing scenario (comparison test, two phases). Each side
# trains 3 rollouts per phase (6 total; phase_a ids 0-2, phase_b ids 3-5), saving a checkpoint
# at the end of phase_a that phase_b loads. The target side has FT enabled and, in each phase,
# scripted actions stop and restart the last trainer cell at the second rollout of the phase so
# that it heals from another replica at the third. Everything is deterministic, so the
# comparison is bitwise for all metrics (except grad norm, which gets a tight relative bound
# because its summation order depends on distributed-optimizer shard count) and for all dumped
# tensors; this proves state copied during healing is reconstructed bit for bit. Exposes the
# comparison app and CI runner.

from pathlib import Path

from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE, create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args, get_train_env_vars_arg
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.ray.specs.train import compute_trainer_pool_id
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.test_utils.comparisons.dumps import (
    INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    INPUT_TENSORS_SKIP_PATTERN,
    compare_dumps,
)
from miles.utils.test_utils.comparisons.inference_engine_checksums import compare_inference_engine_checksums
from miles.utils.test_utils.comparisons.metrics import compare_metrics
from miles.utils.test_utils.ft_test_actions import compute_ft_test_actions_arg
from miles.utils.test_utils.reconfigure_assertions import ReconfigureInfo, assert_reconfigure_events
from miles.utils.workers.naming import compute_cell_id

# --num-rollout is the exclusive global end id (TOTAL_NUM_ROLLOUTS); --debug-exit-after-rollout counts rollouts within the current run.
NUM_ROLLOUTS_PER_PHASE: int = 3
TOTAL_NUM_ROLLOUTS: int = 2 * NUM_ROLLOUTS_PER_PHASE
PHASE_START_ROLLOUT_IDS: dict[str, int] = {"phase_a": 0, "phase_b": NUM_ROLLOUTS_PER_PHASE}


def _build_actions(*, phase_start_rollout_id: int, num_cells: int) -> list[dict]:
    # [FUNCTION DOC] HELPER: Builds the scripted FT actions for one phase: at the phase's second
    # rollout (start id plus one), at its end, stop the last trainer cell (index cell count minus
    # one, in the actor trainer pool) and then start it again. Returns two action mappings in
    # that order.
    heal_trigger_rollout_id: int = phase_start_rollout_id + 1
    target_cell_id: str = compute_cell_id(pool_id=compute_trainer_pool_id("actor"), cell_index=num_cells - 1)
    return [
        {"at_rollout": heal_trigger_rollout_id, "action": "stop_cell_at_end", "cell_id": target_cell_id},
        {"at_rollout": heal_trigger_rollout_id, "action": "start_cell_at_end", "cell_id": target_cell_id},
    ]


def _expected_reconfigures(*, is_target: bool, phase: str, num_cells: int) -> list[ReconfigureInfo]:
    # [FUNCTION DOC] HELPER: The reconfigure events expected for a side and phase. The baseline
    # expects none. The target expects one heal in phase_a, and in phase_b the phase_a heal event
    # again followed by phase_b's (so the phase_b events are expected to include the earlier
    # phase's heal). Each heal event is at the phase start id plus two, copies
    # state from cell index 0 into the last cell, and leaves all cells alive. Raises a value
    # error for an unknown phase name (when the side is target).
    def heal_at(phase_name: str) -> ReconfigureInfo:
        # [FUNCTION DOC] HELPER: Builds the expected heal event of the named phase.
        return ReconfigureInfo(
            rollout_id=PHASE_START_ROLLOUT_IDS[phase_name] + 2,
            src_cell_index=0,
            healed_cell_indices=[num_cells - 1],
            alive_cell_indices_after=list(range(num_cells)),
        )

    if not is_target:
        return []
    if phase == "phase_a":
        return [heal_at("phase_a")]
    if phase == "phase_b":
        return [heal_at("phase_a"), heal_at("phase_b")]
    raise ValueError(f"unknown phase: {phase!r}")


def _build_phase_args(mode: FTTestMode, dump_dir: str, *, is_target: bool, enable_dumper: bool = True) -> str:
    # [FUNCTION DOC] HELPER: Train arguments for one side and phase. The phase is the last path
    # component of the dump directory (assertion error if it is not a known phase name). Starts
    # from the common args for 6 total rollouts, adds deterministic mode with deterministic env
    # vars and deterministic collectives; the target also gets FT args. Saves a checkpoint under
    # the dump dir with interval 3 and exits after 3 rollouts of the run; phases other than
    # phase_a load the phase_a checkpoint (found by swapping the phase in the path). The target
    # also gets the scripted FT actions.
    phase_name: str = dump_dir.rsplit("/", 1)[-1]
    assert phase_name in PHASE_START_ROLLOUT_IDS, (
        f"dump dir {dump_dir!r} does not end in a phase name; this multi-phase scenario "
        f"requires --phase ({'|'.join(PHASE_START_ROLLOUT_IDS)})"
    )
    phase_start_rollout_id: int = PHASE_START_ROLLOUT_IDS[phase_name]

    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=TOTAL_NUM_ROLLOUTS, enable_dumper=enable_dumper)
    base += "--deterministic-mode " + get_train_env_vars_arg(mode, deterministic=True)
    base += "--debug-deterministic-collective "

    if is_target:
        base += get_ft_args(mode)

    base += f"--save {dump_dir}/ckpt --save-interval {NUM_ROLLOUTS_PER_PHASE} "
    base += f"--debug-exit-after-rollout {NUM_ROLLOUTS_PER_PHASE} "
    if phase_name != "phase_a":
        phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
        base += f"--load {phase_a_dir}/ckpt "

    if is_target:
        actions = _build_actions(phase_start_rollout_id=phase_start_rollout_id, num_cells=mode.num_cells)
        base += compute_ft_test_actions_arg(actions)

    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    # [FUNCTION DOC] HELPER: Phase arguments for the baseline (no FT).
    return _build_phase_args(mode, dump_dir, is_target=False, enable_dumper=enable_dumper)


def _build_target_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    # [FUNCTION DOC] HELPER: Phase arguments for the target (FT enabled, scripted heal actions).
    return _build_phase_args(mode, dump_dir, is_target=True, enable_dumper=enable_dumper)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    # [FUNCTION DOC] HELPER: Comparison hook; for each phase: (1) both sides' reconfigure events
    # must equal the expected ones; (2) train metrics must be exactly equal except grad norm;
    # (3) grad norm must match within relative tolerance 1e-6; (4) the set of dump leaf folders of
    # the baseline must equal exactly the phase's three rollouts (so a post-healing rollout is
    # present); (5) all dumped tensors must match with zero relative difference, allowing the
    # shared skip/failed patterns for input tensors; (6) when the mode has real rollout, the
    # inference engine checksums must match. Prints a pass message at the end.
    # Bitwise (zero-tolerance) comparison. The deterministic healing test exists to
    # prove that state pulled from another replica during healing is reconstructed
    # *bit-for-bit*: a state-copy bug is trivial to introduce and an approximate
    # ("looks close") check would silently miss it. So every assertion is exact --
    # all metrics must be equal (rtol=atol=0) and every dumped tensor must match
    # bitwise (predicate "rel <= 0" for every tensor, no near-zero tolerance).
    # Sole exception: train/grad_norm sums squared shard fragments, so its
    # bracketing depends on the distributed-optimizer shard count (8 in the flat
    # baseline vs 2 per FT cell) -- a few fp32 ulps of drift are inherent to
    # comparing different shardings, and the grads themselves are still compared
    # bitwise by compare_dumps below. It gets a tight non-zero gate instead.
    #
    # This requires the run to be fully deterministic on both sides.
    # Any divergence is a real bug and must be fixed at the source, never hidden by
    # loosening these thresholds.
    grad_norm_key = "train/grad_norm"
    for phase, phase_start_rollout_id in PHASE_START_ROLLOUT_IDS.items():
        baseline_dir = f"{dump_dir}/{BASELINE_SIDE}/{phase}"
        target_dir = f"{dump_dir}/{TARGET_SIDE}/{phase}"
        for side, side_dir in ((BASELINE_SIDE, baseline_dir), (TARGET_SIDE, target_dir)):
            assert_reconfigure_events(
                Path(f"{side_dir}/{EVENTS_DIRNAME}"),
                expected=_expected_reconfigures(is_target=side == TARGET_SIDE, phase=phase, num_cells=mode.num_cells),
            )

        compare_metrics(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            rtol=0.0,
            atol=0.0,
            key_prefixes=["train/"],
            exclude_keys=[grad_norm_key],
        )
        compare_metrics(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            rtol=1e-6,
            atol=0.0,
            key_prefixes=[grad_norm_key],
            exclude_keys=[],
        )

        phase_rollout_ids = range(phase_start_rollout_id, phase_start_rollout_id + NUM_ROLLOUTS_PER_PHASE)
        expected_leaves = {f"fwd_bwd/rollout_{rollout_id}" for rollout_id in phase_rollout_ids}
        actual_leaves = {
            str(p.parent.relative_to(Path(f"{baseline_dir}/dumps")))
            for p in Path(f"{baseline_dir}/dumps").rglob("*.pt")
        }
        assert actual_leaves == expected_leaves, (
            f"{phase}: dump leaves {actual_leaves} do not match the expected rollouts {expected_leaves}; "
            f"the post-healing rollout must be present or healing was never exercised"
        )

        compare_dumps(
            baseline_dir=baseline_dir,
            target_dir=target_dir,
            diff_thresholds=[(".*", "rel <= 0")],
            allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
            allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
        )

        if mode.has_real_rollout:
            compare_inference_engine_checksums(
                baseline_dir=baseline_dir,
                target_dir=target_dir,
            )
    print("Deterministic healing comparison test PASSED")


TEST_NAME: str = "trainer_deterministic"
PHASES: list[str] = list(PHASE_START_ROLLOUT_IDS)


app, run_ci = create_comparison_app_and_run_ci(
    test_name=TEST_NAME,
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
    phases=PHASES,
)

if __name__ == "__main__":
    app()
