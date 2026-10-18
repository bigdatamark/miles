# [MODULE DOC]
# Tests for the rollout-deterministic ft scenario: the training argument string it builds (shared
# deterministic inference recipe, no true-on-policy flags), the helper that maps injection
# timestamps to the rollout windows they fall into (a window index is the number of rollouts
# already completed when the injection happened), and the predicate that stops admitting new
# faults once the latest completed rollout id reaches the fault-free terminal tail.
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from tests.e2e.ft.conftest_ft.modes import MODES
from tests.e2e.ft.conftest_ft.scenario_rollout_deterministic import (
    NUM_ROLLOUTS,
    TERMINAL_FAULT_FREE_ROLLOUTS,
    _build_args,
    _compute_crashed_rollouts,
    _rollout_fault_injection_enabled,
)

_BASE = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def test_rollout_deterministic_uses_the_shared_deterministic_recipe_without_true_on_policy(tmp_path: Path) -> None:
    # [FUNCTION DOC] Builds args for the colocated rollout-kill mode with a temp dump dir and
    # asserts present: deterministic inference flag, flashinfer attention backend, the batch
    # invariant matmul fallback env setting set to false, health check interval 1.0, the
    # deterministic mode flag. Asserts absent: context parallel size, every true-on-policy flag
    # variant, the fa3 attention backend, and recompute-logprobs-via-prefill.
    """The rollout-FT comparison must use pure deterministic inference rather than true-on-policy."""
    args = _build_args(MODES["kill_rollout__dp4__colocate"], dump_dir=str(tmp_path))

    assert "--sglang-enable-deterministic-inference " in args
    assert "--sglang-attention-backend flashinfer " in args
    assert '"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "false"' in args
    assert "--rollout-health-check-interval 1.0 " in args
    assert "--deterministic-mode " in args
    assert "--context-parallel-size " not in args
    assert "--true-on-policy-mode" not in args
    assert "--sglang-true-on-policy-contract" not in args
    assert "--true-on-policy-contract" not in args
    assert "--sglang-attention-backend fa3" not in args
    assert "--recompute-logprobs-via-prefill" not in args


class TestComputeCrashedRollouts:
    # [CLASS DOC] Tests for mapping injection times and rollout completion times to a set of crashed rollout windows.
    def test_two_crashes_before_any_rollout_finished_share_one_window(self) -> None:
        # [FUNCTION DOC] Two injections ten seconds apart, both before the only rollout completion
        # (an hour later); asserts the set of windows is just window zero.
        """Both landing before the first generation is the vacuous run this scenario has to reject."""
        crashed = _compute_crashed_rollouts(
            injected_at=[_BASE, _BASE + timedelta(seconds=10)], rollout_completions=[(0, _BASE + timedelta(hours=1))]
        )

        assert crashed == {0}

    def test_crashes_on_either_side_of_a_finished_rollout_are_two_windows(self) -> None:
        # [FUNCTION DOC] One injection before and one after the completion of rollout zero;
        # asserts windows zero and one.
        """This is the run the scenario is meant to produce: crashes spread across the loss curve."""
        crashed = _compute_crashed_rollouts(
            injected_at=[_BASE, _BASE + timedelta(seconds=20)],
            rollout_completions=[(0, _BASE + timedelta(seconds=10))],
        )

        assert crashed == {0, 1}

    def test_a_run_with_no_crashes_has_no_windows(self) -> None:
        # [FUNCTION DOC] No injections; asserts the result is the empty set.
        """An empty result must not read as coverage; the caller's floor is what rejects it."""
        assert _compute_crashed_rollouts(injected_at=[], rollout_completions=[(0, _BASE)]) == set()

    def test_repeated_metrics_from_one_rollout_count_as_one_completed_rollout(self) -> None:
        # [FUNCTION DOC] Two completion records for the same rollout id before a later injection;
        # asserts the injection falls in window one (the rollout counts once).
        """Repeated metric events must not advance an injection by multiple rollout windows."""
        crashed = _compute_crashed_rollouts(
            injected_at=[_BASE + timedelta(seconds=20)],
            rollout_completions=[
                (0, _BASE + timedelta(seconds=10)),
                (0, _BASE + timedelta(seconds=15)),
            ],
        )

        assert crashed == {1}


def test_rollout_fault_window_closes_before_the_final_rollouts(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Fakes the completion-time reader to report rollouts with ids from zero up to
    # the total minus the terminal fault-free count (exclusive); asserts fault injection is not
    # enabled for a dump dir.
    """The deterministic recovery tail admits no new rollout fault."""
    completed_rollouts = [(rollout_id, _BASE) for rollout_id in range(NUM_ROLLOUTS - TERMINAL_FAULT_FREE_ROLLOUTS)]

    def read_completed_rollouts(dump_dir: str) -> list[tuple[int, datetime]]:
        # [FUNCTION DOC] HELPER: Ignores the dump dir and returns the prepared completion list.
        return completed_rollouts

    monkeypatch.setattr(
        "tests.e2e.ft.conftest_ft.scenario_rollout_deterministic.read_rollout_completion_times",
        read_completed_rollouts,
    )

    assert not _rollout_fault_injection_enabled("/dump")


def test_rollout_fault_window_uses_the_latest_completed_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] Fakes the reader to report two metric events for the same rollout id (the
    # last id before the tail starts); asserts fault injection is disabled, showing the decision uses the
    # highest completed rollout id rather than the event count.
    """The rollout ID watermark, not the number of metric events, starts the recovery tail."""
    completed_rollouts = [
        (NUM_ROLLOUTS - TERMINAL_FAULT_FREE_ROLLOUTS - 1, _BASE),
        (NUM_ROLLOUTS - TERMINAL_FAULT_FREE_ROLLOUTS - 1, _BASE + timedelta(seconds=1)),
    ]

    def read_completed_rollouts(dump_dir: str) -> list[tuple[int, datetime]]:
        # [FUNCTION DOC] HELPER: Ignores the dump dir and returns the prepared completion list.
        return completed_rollouts

    monkeypatch.setattr(
        "tests.e2e.ft.conftest_ft.scenario_rollout_deterministic.read_rollout_completion_times",
        read_completed_rollouts,
    )

    assert not _rollout_fault_injection_enabled("/dump")
