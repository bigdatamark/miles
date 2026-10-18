# [MODULE DOC] Unit tests for the metric comparison helpers of the test utilities: reading metric
# events and series from event logs, keeping only the final attempt per rollout, per-metric and
# per-step comparison with relative and absolute tolerances, checking that two sides line up,
# reading rollout completion times, classifying metric namespaces, and asserting metrics are
# finite and nonzero. Module constants: the gradient norm metric key, and a fixed timestamp and
# source used to construct metric events. Event logs are written with the real event logger into
# temporary directories.
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest

from miles.utils.audit_utils.event_logger.logger import EventLogger
from miles.utils.audit_utils.event_logger.models import MetricEvent, WitnessAllocateIdEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.comparisons.metrics import (
    _check_events_line_up,
    _check_single_metric,
    _check_step_metrics,
    _keep_only_final_attempt,
    assert_gradients_nonzero,
    assert_metric_finite_and_nonzero,
    assert_metrics_classified,
    read_metric_events,
    read_metric_series,
    read_rollout_completion_times,
)

_KEY: str = "train/grad_norm"

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXED_SOURCE = SimpleProcessIdentity(component="main")


def _metric_event(
    *, rollout_id: int | None, attempt: int | None, metrics: dict[str, Any] | None = None
) -> MetricEvent:
    # [FUNCTION DOC] HELPER: Builds a metric event with the fixed timestamp and source, the given
    # rollout id and attempt (either may be none), and the given metrics mapping (empty when none
    # is given).
    return MetricEvent(
        timestamp=_FIXED_TS,
        source=_FIXED_SOURCE,
        rollout_id=rollout_id,
        attempt=attempt,
        metrics=metrics if metrics is not None else {},
    )


class TestReadMetricEvents:
    # [CLASS DOC] Tests of reading metric events from an events directory.
    def test_only_metric_events_are_returned_in_emission_order(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Logs metric, witness allocate, metric events and asserts only the two
        # metric events come back, in the order they were logged.
        """Mixed event streams yield only metric events in their original emission order."""
        event_logger = EventLogger(log_dir=tmp_path, source=_FIXED_SOURCE, file_name="main.jsonl")
        event_logger.log(MetricEvent, dict(metrics={"sequence": 1}), print_log=False)
        event_logger.log(
            WitnessAllocateIdEvent,
            dict(rollout_id=0, attempt=0, witness_id_to_sample_index={10: 0}, counter_after=11),
            print_log=False,
        )
        event_logger.log(MetricEvent, dict(metrics={"sequence": 2}), print_log=False)
        event_logger.close()

        events = read_metric_events(tmp_path)

        assert all(isinstance(event, MetricEvent) for event in events)
        assert [event.metrics for event in events] == [{"sequence": 1}, {"sequence": 2}]

    def test_absent_events_directory_returns_empty(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A directory that does not exist gives an empty list, with no error.
        """An events directory that does not exist yields an empty metric stream."""
        assert read_metric_events(tmp_path / "absent") == []


class TestKeepOnlyFinalAttempt:
    # [CLASS DOC] Tests of the private filter that keeps, per rollout id, only the events with
    # the highest attempt number (none counts as 0).
    def test_keeps_highest_attempt_for_single_rollout(self) -> None:
        # [FUNCTION DOC] Attempts 0, 1, 2 of one rollout leave only the attempt 2 event.
        """Among attempts 0,1,2 for one rollout_id, only the attempt=2 event survives."""
        events = [
            _metric_event(rollout_id=1, attempt=0),
            _metric_event(rollout_id=1, attempt=1),
            _metric_event(rollout_id=1, attempt=2),
        ]
        kept = _keep_only_final_attempt(events)
        assert [e.attempt for e in kept] == [2]

    def test_highest_attempt_resolved_independently_per_rollout(self) -> None:
        # [FUNCTION DOC] Rollout 1 with attempts 0 and 1, rollout 2 with attempt 0: result holds
        # the pairs (1, 1) and (2, 0).
        """Each rollout_id keeps its own max attempt; different maxima coexist."""
        events = [
            _metric_event(rollout_id=1, attempt=0),
            _metric_event(rollout_id=1, attempt=1),
            _metric_event(rollout_id=2, attempt=0),
        ]
        kept = _keep_only_final_attempt(events)
        assert {(e.rollout_id, e.attempt) for e in kept} == {(1, 1), (2, 0)}

    def test_none_attempt_normalized_to_zero_and_dropped_when_mixed(self) -> None:
        # [FUNCTION DOC] A none attempt and an attempt 1 for the same rollout: only the attempt 1
        # event is kept.
        """attempt=None normalizes to 0, so it is dropped when an attempt=1 event shares the rollout_id."""
        events = [
            _metric_event(rollout_id=1, attempt=None),
            _metric_event(rollout_id=1, attempt=1),
        ]
        kept = _keep_only_final_attempt(events)
        assert [e.attempt for e in kept] == [1]

    def test_empty_input_returns_empty(self) -> None:
        # [FUNCTION DOC] An empty list gives an empty list.
        """An empty event list yields an empty result."""
        assert _keep_only_final_attempt([]) == []

    def test_ties_on_max_attempt_all_kept(self) -> None:
        # [FUNCTION DOC] Two attempt 2 events of one rollout are both kept, in input order.
        """Multiple events tied at the max attempt for a rollout_id are all retained."""
        events = [
            _metric_event(rollout_id=1, attempt=2, metrics={"a": 1}),
            _metric_event(rollout_id=1, attempt=2, metrics={"b": 2}),
        ]
        kept = _keep_only_final_attempt(events)
        assert len(kept) == 2
        assert [e.metrics for e in kept] == [{"a": 1}, {"b": 2}]


class TestCheckSingleMetric:
    # [CLASS DOC] Tests of the private single metric comparison, called with a step index, key,
    # baseline value, target value and relative and absolute tolerances; it returns a list of
    # issue strings (empty when the values are acceptable).
    def test_equal_values_no_issue(self) -> None:
        # [FUNCTION DOC] Equal numbers give no issue.
        """Exactly equal numeric values produce no issue."""
        assert _check_single_metric(0, "k", 1.5, 1.5, rtol=0.01, atol=0.0) == []

    def test_within_atol_no_issue(self) -> None:
        # [FUNCTION DOC] A 1e-9 difference with zero relative tolerance and absolute tolerance
        # 1e-6 gives no issue.
        """A difference within atol is accepted even if relative difference would exceed rtol."""
        assert _check_single_metric(0, "k", 1.0, 1.0 + 1e-9, rtol=0.0, atol=1e-6) == []

    def test_relative_difference_above_rtol_reports_issue(self) -> None:
        # [FUNCTION DOC] Values 1.0 and 2.0 with relative tolerance 0.1 give exactly one issue
        # mentioning the metric key and the relative difference.
        """A relative difference above rtol (and above atol) yields exactly one issue."""
        issues = _check_single_metric(3, "train/loss", 1.0, 2.0, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "train/loss" in issues[0]
        assert "rel_diff" in issues[0]

    def test_nan_detected(self) -> None:
        # [FUNCTION DOC] A not-a-number baseline gives exactly one issue saying NaN was detected.
        """A NaN on either side produces a 'NaN detected' issue."""
        issues = _check_single_metric(0, "k", float("nan"), 1.0, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "NaN detected" in issues[0]

    def test_matching_inf_no_issue(self) -> None:
        # [FUNCTION DOC] Two positive infinities compare equal and give no issue.
        """inf == inf compares equal and produces no issue."""
        assert _check_single_metric(0, "k", float("inf"), float("inf"), rtol=0.1, atol=0.0) == []

    def test_inf_vs_finite_reports_mismatch(self) -> None:
        # [FUNCTION DOC] Infinity versus 1.0 gives exactly one issue saying infinity mismatch.
        """inf versus a finite value produces an 'inf mismatch' issue."""
        issues = _check_single_metric(0, "k", float("inf"), 1.0, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "inf mismatch" in issues[0]

    def test_both_zero_no_issue(self) -> None:
        # [FUNCTION DOC] Two zeros with zero tolerances give no issue.
        """Two exact zeros short-circuit to no issue."""
        assert _check_single_metric(0, "k", 0.0, 0.0, rtol=0.0, atol=0.0) == []

    def test_non_numeric_skipped(self) -> None:
        # [FUNCTION DOC] A string baseline value is skipped and gives no issue.
        """A non-numeric value on either side is skipped (no issue)."""
        assert _check_single_metric(0, "k", "abc", 1.0, rtol=0.0, atol=0.0) == []

    def test_tiny_baseline_uses_relative_floor(self) -> None:
        # [FUNCTION DOC] Baseline 0 versus target 5e-13 with relative tolerance 0.1 and zero
        # absolute tolerance gives one issue mentioning the relative difference, because the
        # denominator is floored at 1e-12 and so the relative difference is large.
        """A near-zero baseline uses the 1e-12 denominator floor, making a tiny abs diff a large rel diff."""
        issues = _check_single_metric(0, "k", 0.0, 5e-13, rtol=0.1, atol=0.0)
        assert len(issues) == 1
        assert "rel_diff" in issues[0]


class TestCheckStepMetrics:
    # [CLASS DOC] Tests of the private per-step comparison, called with a step index, baseline
    # event, target event, a list of key prefixes to compare, a relative tolerance, an absolute
    # tolerance and optionally excluded keys; it returns issue strings.
    def test_matching_selected_keys_report_nothing(self) -> None:
        # [FUNCTION DOC] Both events hold the same train loss; no issues.
        """Both sides carrying the same selected keys with the same values is the passing case."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        assert _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0) == []

    def test_a_key_only_the_baseline_has_is_reported(self) -> None:
        # [FUNCTION DOC] Target lacks a selected key present in baseline; one issue saying it is
        # present in baseline but missing in target.
        """A metric the target stopped producing must not vanish from the comparison."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})
        target = _metric_event(rollout_id=0, attempt=0, metrics={})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0)

        assert len(issues) == 1
        assert "present in baseline but missing in target" in issues[0]

    def test_a_key_only_the_target_has_is_reported(self) -> None:
        # [FUNCTION DOC] Baseline lacks a selected key present in target; one issue saying it is
        # present in target but missing in baseline (regression guard).
        """Regression: a metric the target alone produces used to pass unnoticed."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0)

        assert len(issues) == 1
        assert "present in target but missing in baseline" in issues[0]

    def test_a_target_only_key_outside_the_prefixes_is_ignored(self) -> None:
        # [FUNCTION DOC] A target-only perf key is outside the compared train prefix; no issues.
        """Only the namespaces this comparison claims to cover are compared."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"perf/step_time": 1.0})

        assert _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0) == []

    def test_an_excluded_target_only_key_is_ignored(self) -> None:
        # [FUNCTION DOC] A target-only train loss key that is in the exclude list gives no issue.
        """An explicitly excluded key stays excluded whichever side carries it."""
        baseline = _metric_event(rollout_id=0, attempt=0, metrics={})
        target = _metric_event(rollout_id=0, attempt=0, metrics={"train/loss": 1.0})

        issues = _check_step_metrics(0, baseline, target, ["train/"], 0.1, atol=0.0, exclude_keys=["train/loss"])

        assert issues == []


class TestCheckEventsLineUp:
    # [CLASS DOC] Tests of the private check that baseline and target event lists describe the
    # same rollout ids position by position; returns issue strings.
    def test_two_sides_describing_different_rollouts_are_reported(self) -> None:
        # [FUNCTION DOC] Baseline rollouts 0 and 1 versus target rollouts 1 and 2 give two issues,
        # one per position.
        """Comparing by read order passes silently when the sides are offset but the numbers happen to agree."""
        baseline = [_metric_event(rollout_id=0, attempt=0), _metric_event(rollout_id=1, attempt=0)]
        target = [_metric_event(rollout_id=1, attempt=0), _metric_event(rollout_id=2, attempt=0)]

        issues = _check_events_line_up(baseline, target)

        assert len(issues) == 2

    def test_a_retried_rollout_still_lines_up_with_its_baseline(self) -> None:
        # [FUNCTION DOC] Same rollout id with attempt 0 in baseline and attempt 1 in target gives
        # no issues (attempt numbers are not compared).
        """The target retries a crashed rollout, and comparing its winning attempt is the point of the run."""
        baseline = [_metric_event(rollout_id=3, attempt=0)]
        target = [_metric_event(rollout_id=3, attempt=1)]

        assert _check_events_line_up(baseline, target) == []


class TestReadRolloutCompletionTimes:
    # [CLASS DOC] Tests of reading (rollout id, timestamp) pairs from the rollout executor's
    # metric events under the events subdirectory of a dump directory.
    def test_only_rollout_executor_events_are_completion_times(self, dump_dir: str) -> None:
        # [FUNCTION DOC] Writes one metric event from a rollout executor writer (rollout 3) and
        # one from the trainer-like main writer (rollout 4) into separate files; asserts only
        # rollout 3 is reported.
        """Trainer metrics sharing a rollout id must not be mistaken for rollout completion."""
        events_dir = Path(dump_dir) / "events"
        rollout_logger = EventLogger(
            log_dir=events_dir,
            source=SimpleProcessIdentity(component="rollout_executor"),
            file_name="rollout.jsonl",
        )
        trainer_logger = EventLogger(log_dir=events_dir, source=_FIXED_SOURCE, file_name="trainer.jsonl")
        rollout_logger.log(MetricEvent, dict(rollout_id=3, attempt=0, metrics={}), print_log=False)
        trainer_logger.log(MetricEvent, dict(rollout_id=4, attempt=0, metrics={}), print_log=False)
        rollout_logger.close()
        trainer_logger.close()

        completion_times = read_rollout_completion_times(dump_dir)

        assert [rollout_id for rollout_id, _timestamp in completion_times] == [3]

    def test_identified_rollout_executor_events_are_sorted_by_completion_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Replaces the datetime class used by the event logger module with a
        # subclass whose current-time method returns, in order, four preset timestamps (seconds 4,
        # 1, 3, 2 of a fixed minute). Logs four executor events: rollouts 20, none, 10 and 20
        # (second attempt). Asserts the result excludes the event without a rollout id and lists
        # the remaining three as pairs sorted by timestamp (20 at second 2, 10 at 3, 20 at 4).
        """Completion times contain every identified executor event ordered by its timestamp."""
        timestamps = iter(
            [
                datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            ]
        )

        class _SequencedDatetime(datetime):
            # [CLASS DOC] HELPER: Datetime subclass whose current-time method ignores the time
            # zone and returns the next preset timestamp from the enclosing iterator (raises a
            # stop iteration error when exhausted).
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                # [FUNCTION DOC] HELPER: Returns the next preset timestamp.
                return next(timestamps)

        monkeypatch.setattr("miles.utils.audit_utils.event_logger.logger.datetime", _SequencedDatetime)
        dump_dir = tmp_path / "run"
        event_logger = EventLogger(
            log_dir=dump_dir / "events",
            source=SimpleProcessIdentity(component="rollout_executor"),
            file_name="main.jsonl",
        )
        for rollout_id, attempt in [(20, 0), (None, 0), (10, 0), (20, 1)]:
            event_logger.log(
                MetricEvent,
                {"rollout_id": rollout_id, "attempt": attempt, "metrics": {}},
                print_log=False,
            )
        event_logger.close()

        assert read_rollout_completion_times(str(dump_dir)) == [
            (20, datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc)),
            (10, datetime(2026, 1, 1, 0, 0, 3, tzinfo=timezone.utc)),
            (20, datetime(2026, 1, 1, 0, 0, 4, tzinfo=timezone.utc)),
        ]

    def test_final_identified_rollouts_are_sorted_by_completion_time(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Patches the event logger's datetime to return timestamps at seconds 2
        # then 1, logs executor events for rollouts 2 then 1 (without attempt), and asserts the
        # returned rollout ids are [1, 2], i.e. sorted by timestamp rather than emission order.
        """Identified executor completions are returned in timestamp order, not emission order."""
        timestamps = iter(
            [
                datetime(2026, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
            ]
        )

        class _SequencedDatetime(datetime):
            # [CLASS DOC] HELPER: Datetime subclass whose current-time method ignores the time
            # zone and returns the next preset timestamp from the enclosing iterator.
            @classmethod
            def now(cls, tz: tzinfo | None = None) -> datetime:
                # [FUNCTION DOC] HELPER: Returns the next preset timestamp.
                return next(timestamps)

        monkeypatch.setattr("miles.utils.audit_utils.event_logger.logger.datetime", _SequencedDatetime)
        dump_dir = tmp_path / "run"
        event_logger = EventLogger(
            log_dir=dump_dir / "events",
            source=SimpleProcessIdentity(component="rollout_executor"),
            file_name="main.jsonl",
        )
        for rollout_id in (2, 1):
            event_logger.log(MetricEvent, {"rollout_id": rollout_id, "metrics": {}}, print_log=False)
        event_logger.close()

        assert [rollout_id for rollout_id, _ in read_rollout_completion_times(str(dump_dir))] == [1, 2]


class TestReadMetricSeries:
    # [CLASS DOC] Tests of reading a (rollout id, value) series for one metric key.
    def test_only_numeric_values_from_the_final_attempt_are_returned(self, dump_dir: str) -> None:
        # [FUNCTION DOC] Events include a superseded attempt, an integer and a float on the final
        # attempt of rollout 0, a boolean, a different key and a string value. Asserts the series
        # holds only the two numeric final-attempt values of rollout 0, as floats (1.0 and 2.5).
        """Only numeric final-attempt values survive, with integers normalized to floats."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={_KEY: 99.0}),
                _metric_event(rollout_id=0, attempt=1, metrics={_KEY: 1}),
                _metric_event(rollout_id=0, attempt=1, metrics={_KEY: 2.5}),
                _metric_event(rollout_id=1, attempt=0, metrics={_KEY: True}),
                _metric_event(rollout_id=2, attempt=0, metrics={"train/loss": 3.0}),
                _metric_event(rollout_id=3, attempt=0, metrics={_KEY: "4.0"}),
            ],
        )

        assert read_metric_series(dump_dir, key=_KEY) == [(0, 1.0), (0, 2.5)]

    def test_only_numeric_values_from_final_attempts_form_the_series(self, dump_dir: str) -> None:
        # [FUNCTION DOC] Superseded, boolean and string values are excluded; only rollout 4's
        # final attempt integer 3 remains as 3.0.
        """Superseded, Boolean, missing, and textual metric values do not enter the final series."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=4, attempt=0, metrics={_KEY: 9.0}),
                _metric_event(rollout_id=4, attempt=1, metrics={_KEY: 3}),
                _metric_event(rollout_id=5, attempt=0, metrics={_KEY: False}),
                _metric_event(rollout_id=6, attempt=0, metrics={_KEY: "4.0"}),
            ],
        )

        assert read_metric_series(dump_dir, key=_KEY) == [(4, 3.0)]


class TestAssertMetricsClassified:
    # [CLASS DOC] Tests of the classification assertion that every final-attempt metric key
    # falls under a compared or ignored prefix.
    def test_an_unclassified_namespace_fails_closed(self, dump_dir: str) -> None:
        # [FUNCTION DOC] A metric key outside both prefix lists makes the assertion raise an
        # assertion error mentioning that key.
        """A final-attempt metric outside every declared namespace makes classification fail closed."""
        _write_metric_events(
            dump_dir,
            [_metric_event(rollout_id=0, attempt=0, metrics={"unexpected/value": 1.0})],
        )

        with pytest.raises(AssertionError, match="unexpected/value"):
            assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))


class TestAssertEveryMetricIsClassified:
    # [CLASS DOC] More tests of the classification assertion, covering the reported key list and
    # the accepted case.
    def test_unknown_final_attempt_namespaces_are_reported(self, dump_dir: str) -> None:
        # [FUNCTION DOC] A stale unknown key on a superseded attempt is ignored; the final
        # attempt has two unknown keys. Asserts the error message lists exactly those two keys in
        # sorted order.
        """Sorted unknown final-attempt keys are reported while superseded unknown keys are ignored."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={"stale/unknown": 1.0}),
                _metric_event(
                    rollout_id=0,
                    attempt=1,
                    metrics={"train/loss": 1.0, "perf/throughput": 2.0, "zeta/value": 3.0, "alpha/value": 4.0},
                ),
            ],
        )

        with pytest.raises(AssertionError, match=r"metrics \['alpha/value', 'zeta/value'\]"):
            assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))

    def test_compared_and_ignored_final_attempt_namespaces_are_accepted(self, dump_dir: str) -> None:
        # [FUNCTION DOC] Final-attempt keys under the compared and ignored prefixes pass, while
        # an unknown key on a superseded attempt is disregarded.
        """Final-attempt keys covered by either compared or ignored prefixes are accepted."""
        _write_metric_events(
            dump_dir,
            [
                _metric_event(rollout_id=0, attempt=0, metrics={"stale/unknown": 1.0}),
                _metric_event(
                    rollout_id=0,
                    attempt=1,
                    metrics={"train/loss": 1.0, "perf/throughput": 2.0},
                ),
            ],
        )

        assert_metrics_classified(dump_dir, compared=("train/",), ignored=("perf/",))


class TestAssertMetricWasFiniteAndNonzero:
    # [CLASS DOC] Tests of the finite and nonzero metric assertion, which counts distinct
    # rollouts having at least one usable (finite, nonzero) value for a key and requires a
    # minimum count; also covers the gradient norm wrapper.
    def test_a_run_that_trained_as_many_rollouts_as_asked_passes(self, dump_dir) -> None:
        # [FUNCTION DOC] Two rollouts with usable values and a minimum of two; passes.
        """The happy path has to stay reachable, or the refusals below prove nothing."""
        _write_metrics(dump_dir, [(0, 0.5), (1, 0.4)])

        assert_metric_finite_and_nonzero(side="target", dump_dir=dump_dir, key=_KEY, min_rollouts=2)

    def test_one_rollout_reported_over_several_steps_is_not_several_rollouts(self, dump_dir) -> None:
        # [FUNCTION DOC] Three values all for rollout 0 with a minimum of two raise an assertion
        # error saying only 1 of 1 rollout qualified.
        """Several optimizer steps of one rollout say that one rollout trained, however many events they write."""
        _write_metrics(dump_dir, [(0, 0.5), (0, 0.4), (0, 0.3)])

        with pytest.raises(AssertionError, match="in only 1 of 1 rollout"):
            assert_metric_finite_and_nonzero(side="target", dump_dir=dump_dir, key=_KEY, min_rollouts=2)

    def test_a_rollout_whose_only_usable_step_is_zero_does_not_count(self, dump_dir) -> None:
        # [FUNCTION DOC] Rollout 1 only has a zero value; with a minimum of two the assertion
        # error says only 1 of 2 rollouts qualified.
        """A gradient of zero moved no weights, which is what this assertion exists to catch."""
        _write_metrics(dump_dir, [(0, 0.5), (1, 0.0)])

        with pytest.raises(AssertionError, match="in only 1 of 2 rollout"):
            assert_metric_finite_and_nonzero(side="target", dump_dir=dump_dir, key=_KEY, min_rollouts=2)

    @pytest.mark.parametrize("unusable_value", [float("nan"), float("inf"), float("-inf"), 0.0])
    def test_nonfinite_and_zero_values_do_not_count_as_trained_rollouts(
        self, dump_dir: str, unusable_value: float
    ) -> None:
        # [FUNCTION DOC] Parametrized over not-a-number, positive and negative infinity and zero
        # for rollout 1 (rollout 0 is fine). The gradient wrapper with a minimum of two trained
        # rollouts raises an assertion error stating the target gradient norm is finite and
        # nonzero in only 1 rollout.
        """The gradient wrapper rejects rollouts whose train/grad_norm is non-finite or zero."""
        _write_metrics(dump_dir, [(0, 0.5), (1, unusable_value)])

        with pytest.raises(AssertionError, match="target: train/grad_norm is finite and non-zero in only 1"):
            assert_gradients_nonzero(side="target", dump_dir=dump_dir, min_trained_rollouts=2)


@pytest.fixture
def dump_dir(tmp_path) -> str:
    # [FUNCTION DOC] HELPER: Fixture returning the path string of a not-yet-created run
    # directory inside the temporary path.
    return str(tmp_path / "run")


def _write_metric_events(dump_dir: str, events: list[MetricEvent]) -> None:
    # [FUNCTION DOC] HELPER: Writes the given metric events through the real event logger into
    # the events subdirectory of the dump directory (file main.jsonl). Each event is dumped
    # without its timestamp, source and type fields so the logger fills those in (so timestamps
    # are the current time). Logger is closed afterwards; nothing is printed.
    event_logger = EventLogger(log_dir=Path(dump_dir) / "events", source=_FIXED_SOURCE, file_name="main.jsonl")
    for event in events:
        partial = event.model_dump(exclude={"timestamp", "source", "type"})
        event_logger.log(MetricEvent, partial, print_log=False)
    event_logger.close()


def _write_metrics(dump_dir: str, points: list[tuple[int, float]]) -> None:
    # [FUNCTION DOC] HELPER: For each (rollout id, value) pair, logs a metric event at attempt 0
    # carrying the gradient norm key with that value into the events subdirectory of the dump
    # directory (file main.jsonl), then closes the logger.
    event_logger = EventLogger(log_dir=Path(dump_dir) / "events", source=_FIXED_SOURCE, file_name="main.jsonl")
    for rollout_id, value in points:
        event_logger.log(MetricEvent, dict(rollout_id=rollout_id, attempt=0, metrics={_KEY: value}), print_log=False)
    event_logger.close()
