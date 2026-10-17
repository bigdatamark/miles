# [MODULE DOC] HELPER: Utilities for reading and comparing logged training metric events (from the
# audit event logger) between a baseline run and a target run, plus single-run assertions on metric
# series. Events from superseded fault-tolerance retry attempts are discarded before use. The
# comparison collects human-readable issue strings and asserts at the end; it also prints a
# per-step table of metric differences.
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import polars as pl
from sglang.srt.debug_utils.comparator.display import _render_polars_as_text

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import MetricEvent

logger = logging.getLogger(__name__)

_REQUIRED_METRIC_KEYS: list[str] = ["train/grad_norm", "train/loss"]


def compare_metrics(
    baseline_dir: str,
    target_dir: str,
    *,
    rtol: float,
    atol: float,
    key_prefixes: list[str],
    exclude_keys: list[str],
) -> None:
    # [FUNCTION DOC] HELPER: Asserts the target run's metrics match the baseline's within
    # tolerance. Reads metric events from both dump directories, keeps only the final attempt per
    # rollout, then collects issues: event-count problems (either side empty, or counts differing),
    # and per-position rollout id misalignment. Only when none of those exist, iterates over the
    # paired events: prints a comparison table and checks each step's metrics selected by the key
    # prefixes minus the excluded keys (relative tolerance rtol, absolute tolerance atol). Always
    # also checks that the required metrics (training grad norm and loss) appear in the baseline.
    # Asserts the issue list is empty (message counts and lists them) and prints the number of
    # steps compared.
    baseline_events = _read_metric_events(Path(baseline_dir))
    target_events = _read_metric_events(Path(target_dir))

    # FT retries (healing path) leave events from earlier failed attempts. Only
    # the highest-attempt events per rollout_id reflect the successful run.
    baseline_events = _keep_only_final_attempt(baseline_events)
    target_events = _keep_only_final_attempt(target_events)

    issues: list[str] = []
    issues += _check_event_counts(baseline_events, target_events, baseline_dir, target_dir)
    issues += _check_events_line_up(baseline_events, target_events)

    if not issues:
        for step_idx, (b_event, t_event) in enumerate(zip(baseline_events, target_events, strict=True)):
            _print_step_comparison_table(step_idx, b_event, t_event, key_prefixes, exclude_keys=exclude_keys)
            issues += _check_step_metrics(
                step_idx, b_event, t_event, key_prefixes, rtol, atol=atol, exclude_keys=exclude_keys
            )

    issues += _check_required_keys_exist(baseline_events)

    assert not issues, f"MetricEvent comparison found {len(issues)} issue(s):\n" + "\n".join(
        f"  - {i}" for i in issues
    )
    print(f"MetricEvent comparison passed: {len(baseline_events)} steps compared")


def read_metric_series(dump_dir: str, *, key: str) -> list[tuple[int, float]]:
    # [FUNCTION DOC] HELPER: Returns, in event order, (rollout id, value as float) for each
    # final-attempt event whose metrics contain the key with a numeric (integer or float, not
    # boolean) value. Events lacking the key or with other value types are skipped.
    events = _keep_only_final_attempt(_read_metric_events(Path(dump_dir)))
    return [
        (event.rollout_id, float(value))
        for event in events
        if isinstance(value := event.metrics.get(key), (int, float)) and not isinstance(value, bool)
    ]


def assert_gradients_nonzero(*, side: str, dump_dir: str, min_trained_rollouts: int) -> None:
    # [FUNCTION DOC] HELPER: Convenience wrapper asserting the training gradient-norm metric is
    # finite and nonzero in at least the given number of distinct rollouts.
    assert_metric_finite_and_nonzero(
        side=side, dump_dir=dump_dir, key="train/grad_norm", min_rollouts=min_trained_rollouts
    )


def assert_metric_finite_and_nonzero(*, side: str, dump_dir: str, key: str, min_rollouts: int) -> None:
    # [FUNCTION DOC] HELPER: Reads the metric series for the key and counts distinct rollout ids
    # having at least one value that is finite and not zero; asserts that count is at least
    # min_rollouts (message includes side, key, counts and the full series).
    series = read_metric_series(dump_dir, key=key)
    usable = [(rollout_id, value) for rollout_id, value in series if math.isfinite(value) and value != 0.0]
    usable_rollouts = {rollout_id for rollout_id, _ in usable}

    assert len(usable_rollouts) >= min_rollouts, (
        f"{side}: {key} is finite and non-zero in only {len(usable_rollouts)} of "
        f"{len({rollout_id for rollout_id, _ in series})} rollout(s) ({series}), so this run's weights may have "
        f"moved on nothing training produced"
    )


def assert_metrics_classified(dump_dir: str, *, compared: tuple[str, ...], ignored: tuple[str, ...]) -> None:
    # [FUNCTION DOC] HELPER: Collects all metric keys across final-attempt events and asserts every
    # key starts with one of the compared or ignored prefixes; otherwise the assertion message
    # lists the sorted unclassified keys. Guards against metrics silently escaping a comparison.
    keys = {key for event in _keep_only_final_attempt(_read_metric_events(Path(dump_dir))) for key in event.metrics}
    unclassified: list[str] = sorted(key for key in keys if not key.startswith(compared + ignored))

    assert not unclassified, (
        f"metrics {unclassified} belong to no namespace this comparison has classified, so they would be dropped "
        f"from one that claims to cover everything; add them to the compared prefixes {list(compared)}, or to the "
        f"ignored ones {list(ignored)} with a reason they cannot match"
    )


def read_rollout_completion_times(dump_dir: str) -> list[tuple[int, datetime]]:
    # [FUNCTION DOC] HELPER: Returns (rollout id, timestamp) pairs for all metric events (all
    # attempts, not filtered) that mark rollout completion, sorted by timestamp ascending.
    events = _read_metric_events(Path(dump_dir))
    return sorted(
        ((event.rollout_id, event.timestamp) for event in events if _is_rollout_completion(event)),
        key=lambda one: one[1],
    )


def _is_rollout_completion(event: MetricEvent) -> bool:
    # [FUNCTION DOC] HELPER: True when the event has a rollout id and was emitted by the rollout
    # executor component.
    return event.rollout_id is not None and event.source.component == "rollout_executor"


def _check_events_line_up(baseline_events: list[MetricEvent], target_events: list[MetricEvent]) -> list[str]:
    # [FUNCTION DOC] HELPER: Pairs events by position (stopping at the shorter list) and returns
    # one issue string per position where the rollout ids differ.
    return [
        f"step {index}: baseline is rollout {b.rollout_id} while target is rollout {t.rollout_id}, so the two "
        f"sides are not describing the same step"
        for index, (b, t) in enumerate(zip(baseline_events, target_events, strict=False))
        if b.rollout_id != t.rollout_id
    ]


def _keep_only_final_attempt(events: list[MetricEvent]) -> list[MetricEvent]:
    # [FUNCTION DOC] HELPER: Computes the highest attempt number per rollout id (missing attempt
    # counts as zero) and returns, preserving order, only events whose attempt equals that
    # maximum for their rollout id. Several events of the same rollout and final attempt are all
    # kept.
    """Keep only events from the highest-attempt for each rollout_id.

    During FT healing, a crashed rollout is retried at attempt+1; events from
    the failed attempt are partial and should be discarded for comparison.

    Rollout-side metrics (e.g. RolloutExecutor log_rollout_metrics) have
    attempt=None — they are not part of the FT retry stream, so we treat them
    as a single attempt (normalized to 0).
    """

    def _attempt(e: MetricEvent) -> int:
        # [FUNCTION DOC] HELPER: The event's attempt number, with a missing attempt read as zero.
        return e.attempt if e.attempt is not None else 0

    max_attempt_by_rollout: dict[int, int] = defaultdict(int)
    for e in events:
        max_attempt_by_rollout[e.rollout_id] = max(max_attempt_by_rollout[e.rollout_id], _attempt(e))
    return [e for e in events if _attempt(e) == max_attempt_by_rollout[e.rollout_id]]


def _check_event_counts(
    baseline: list[MetricEvent],
    target: list[MetricEvent],
    baseline_dir: str,
    target_dir: str,
) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns issue strings: one for each empty side (naming its
    # directory), and one when both are non-empty but have different lengths.
    issues: list[str] = []
    if len(baseline) == 0:
        issues.append(f"No MetricEvents found in baseline dir: {baseline_dir}")
    if len(target) == 0:
        issues.append(f"No MetricEvents found in target dir: {target_dir}")
    if len(baseline) > 0 and len(target) > 0 and len(baseline) != len(target):
        issues.append(f"MetricEvent count mismatch: baseline={len(baseline)}, target={len(target)}")
    return issues


def _check_step_metrics(
    step_idx: int,
    baseline_event: MetricEvent,
    target_event: MetricEvent,
    key_prefixes: list[str],
    rtol: float,
    *,
    atol: float,
    exclude_keys: list[str] | None = None,
) -> list[str]:
    # [FUNCTION DOC] HELPER: For one paired step, selects metric keys by prefix minus exclusions on
    # each side; returns issues for keys present on only one side (baseline-only first, then
    # target-only, each sorted), followed by the per-key value comparison issues for keys present
    # on both sides, in sorted key order.
    baseline_keys = _select_keys(baseline_event, key_prefixes, exclude_keys=exclude_keys)
    target_keys = _select_keys(target_event, key_prefixes, exclude_keys=exclude_keys)

    issues: list[str] = [
        f"Step {step_idx}: metric '{key}' present in baseline but missing in target"
        for key in sorted(baseline_keys - target_keys)
    ]
    issues += [
        f"Step {step_idx}: metric '{key}' present in target but missing in baseline"
        for key in sorted(target_keys - baseline_keys)
    ]
    for key in sorted(baseline_keys & target_keys):
        issues += _check_single_metric(
            step_idx, key, baseline_event.metrics[key], target_event.metrics[key], rtol, atol=atol
        )
    return issues


def _select_keys(event: MetricEvent, key_prefixes: list[str], *, exclude_keys: list[str] | None) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of the event's metric keys that start with any of the
    # prefixes and are not in the exclusion list.
    prefixes: tuple[str, ...] = tuple(key_prefixes)
    excluded: set[str] = set(exclude_keys or [])
    return {key for key in event.metrics if key.startswith(prefixes) and key not in excluded}


def _check_single_metric(
    step_idx: int,
    key: str,
    baseline_val: object,
    target_val: object,
    rtol: float,
    atol: float,
) -> list[str]:
    # [FUNCTION DOC] HELPER: Compares one metric's baseline and target values and returns a list
    # of zero or one issue strings. Non-numeric values on either side: no issue (ignored). Either
    # value NaN: issue. Either infinite: issue unless both are equal infinities. Both exactly
    # zero: no issue. Absolute difference within atol: no issue. Otherwise the relative
    # difference is the absolute difference divided by the larger magnitude (floored at 1e-12);
    # an issue is returned if it exceeds rtol. Note an absolute difference within atol wins even
    # if the relative difference is large (so the check passes when either tolerance is met).
    if not isinstance(baseline_val, (int, float)) or not isinstance(target_val, (int, float)):
        return []

    if math.isnan(baseline_val) or math.isnan(target_val):
        return [f"Step {step_idx}, metric '{key}': NaN detected (baseline={baseline_val}, target={target_val})"]
    if math.isinf(baseline_val) or math.isinf(target_val):
        if baseline_val != target_val:
            return [f"Step {step_idx}, metric '{key}': inf mismatch (baseline={baseline_val}, target={target_val})"]
        return []

    if baseline_val == 0.0 and target_val == 0.0:
        return []

    abs_diff = abs(baseline_val - target_val)
    if abs_diff <= atol:
        return []

    rel_diff = abs_diff / max(abs(baseline_val), abs(target_val), 1e-12)
    if rel_diff > rtol:
        return [
            f"Step {step_idx}, metric '{key}': baseline={baseline_val}, target={target_val}, "
            f"rel_diff={rel_diff:.6f} > rtol={rtol}"
        ]
    return []


def _print_step_comparison_table(
    step_idx: int,
    baseline_event: MetricEvent,
    target_event: MetricEvent,
    key_prefixes: list[str],
    *,
    exclude_keys: list[str] | None = None,
) -> None:
    # [FUNCTION DOC] HELPER: Prints a text table for one step to standard output, with one row per
    # baseline metric (sorted by key) that matches a prefix and has numeric values on both sides
    # (metrics missing in the target or non-numeric are skipped). Columns: metric name, baseline
    # and target in scientific notation, absolute difference, and relative difference as a percent
    # (relative to the larger magnitude, floored at 1e-12) with an "(excluded)" suffix for
    # excluded keys. Prints nothing when there are no rows. Uses the sglang comparator's polars
    # renderer.
    rows: list[dict[str, str]] = []
    for key in sorted(baseline_event.metrics):
        if not any(key.startswith(p) for p in key_prefixes):
            continue
        b_val = baseline_event.metrics[key]
        t_val = target_event.metrics.get(key)
        if not isinstance(b_val, (int, float)) or t_val is None or not isinstance(t_val, (int, float)):
            continue
        excluded = "(excluded)" if exclude_keys and key in exclude_keys else ""
        abs_diff = abs(b_val - t_val)
        denom = max(abs(b_val), abs(t_val), 1e-12)
        rel_diff = abs_diff / denom
        rows.append(
            {
                "metric": key,
                "baseline": f"{b_val:.6e}",
                "target": f"{t_val:.6e}",
                "abs_diff": f"{abs_diff:.2e}",
                "rel_diff": f"{rel_diff:.4%}{excluded}",
            }
        )

    if not rows:
        return
    df = pl.DataFrame(rows)
    print(_render_polars_as_text(df, title=f"Step {step_idx} metric comparison"))


def _check_required_keys_exist(events: list[MetricEvent]) -> list[str]:
    # [FUNCTION DOC] HELPER: Unions the metric keys of all events and returns one issue string for
    # each required key (grad norm, loss) absent, listing the available keys.
    all_keys: set[str] = set()
    for event in events:
        all_keys.update(event.metrics.keys())

    issues: list[str] = []
    for required in _REQUIRED_METRIC_KEYS:
        if required not in all_keys:
            issues.append(
                f"Required metric '{required}' not found in any baseline MetricEvent. "
                f"Available keys: {sorted(all_keys)}"
            )
    return issues


def read_metric_events(events_dir: Path) -> list[MetricEvent]:
    # [FUNCTION DOC] HELPER: Reads all audit events in the given events directory and returns only
    # metric events, in read order; returns an empty list if the directory does not exist.
    """Read all MetricEvents written into one events directory."""
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, MetricEvent)]


def _read_metric_events(dump_dir: Path) -> list[MetricEvent]:
    # [FUNCTION DOC] HELPER: Reads metric events from the standard events subdirectory of a run's
    # dump directory.
    return read_metric_events(dump_dir / EVENTS_DIRNAME)
