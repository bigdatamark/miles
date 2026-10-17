# [MODULE DOC] HELPER: Assertions over recorded inference-engine weight-checksum events (written by
# the audit event logger in each run's dump directory): baseline-vs-target equality, engine
# count, and proof that weights actually changed between rollouts. All functions raise assertion
# errors on failure and print a one-line success summary.
from pathlib import Path

from miles.utils.audit_utils.event_analyzer.rules import inference_engine_weight_checksum_consistency
from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import ChecksumMismatchIssue, compare_flat_dicts
from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME, read_events
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent


def compare_inference_engine_checksums(baseline_dir: str, target_dir: str) -> None:
    # [FUNCTION DOC] HELPER: Asserts the engine weight checksums of the target run equal those of
    # the baseline run. Reads checksum events from both dump directories (asserting each has at
    # least one); asserts, using the production consistency rule, that engines within each side
    # agree with each other; reduces each side to one checksum mapping per (trainer model id,
    # rollout id) taking the first engine; asserts both sides cover exactly the same keys; then
    # compares the flat checksum dictionaries per key in sorted order and asserts there are no
    # mismatches (message lists each labeled pair, key and two values). Prints how many rollouts
    # were compared.
    baseline = _read_inference_engine_checksum_events(Path(baseline_dir))
    target = _read_inference_engine_checksum_events(Path(target_dir))
    assert baseline, f"No InferenceEngineWeightChecksumEvents found in baseline dir: {baseline_dir}"
    assert target, f"No InferenceEngineWeightChecksumEvents found in target dir: {target_dir}"

    # Each side's engines must already agree internally (same invariant as the production rule), so
    # one representative engine per rollout then proves baseline == target regardless of engine count.
    assert not inference_engine_weight_checksum_consistency.check(
        baseline
    ), "Baseline engines disagree with each other"
    assert not inference_engine_weight_checksum_consistency.check(target), "Target engines disagree with each other"

    baseline_by_model_and_rollout = _checksums_by_model_and_rollout_id(baseline)
    target_by_model_and_rollout = _checksums_by_model_and_rollout_id(target)
    assert baseline_by_model_and_rollout.keys() == target_by_model_and_rollout.keys(), (
        f"Engine checksum (model_id, rollout_id) sets differ: "
        f"baseline={sorted(baseline_by_model_and_rollout)} "
        f"vs target={sorted(target_by_model_and_rollout)}"
    )

    mismatches: list[ChecksumMismatchIssue] = []
    for key in sorted(baseline_by_model_and_rollout):
        model_id, rollout_id = key
        mismatches += list(
            compare_flat_dicts(
                a=baseline_by_model_and_rollout[key],
                b=target_by_model_and_rollout[key],
                label_a=f"baseline/{model_id}/rollout_{rollout_id}",
                label_b=f"target/{model_id}/rollout_{rollout_id}",
            )
        )
    assert not mismatches, "Engine weight checksum baseline-vs-target mismatch:\n" + "\n".join(
        f"  - {m.label_a} vs {m.label_b} key {m.key}: {m.value_a} != {m.value_b}" for m in mismatches
    )
    print(f"Engine weight checksum comparison passed: {len(baseline_by_model_and_rollout)} rollout(s) compared")


def assert_engine_count(*, side: str, dump_dir: str, expected: int) -> None:
    # [FUNCTION DOC] HELPER: Asserts that every weight-update checksum event in the dump directory
    # lists exactly the expected number of engines. Fails if there are no events, or if the set of
    # distinct per-event engine counts is anything other than a single value equal to expected
    # (the side name is used in messages). Prints a confirmation.
    events = _read_inference_engine_checksum_events(Path(dump_dir))
    assert events, f"{side}: no InferenceEngineWeightChecksumEvents in {dump_dir}, so no engine ever took weights"

    counted = sorted({len(event.engine_checksums) for event in events})
    assert counted == [expected], (
        f"{side}: weights were pushed to {counted} engine(s), not {expected}; a run served by fewer engines still "
        f"trains, so nothing else would notice engines that never joined"
    )

    print(f"{side}: every weight update covered {expected} engine(s)")


def assert_engine_weights_moved(*, side: str, dump_dir: str) -> None:
    # [FUNCTION DOC] HELPER: Asserts the engine weights changed over the run. Reduces events to one
    # checksum mapping per (model id, rollout id); asserts there are at least two such entries
    # (otherwise nothing to compare), and that the entries are not all identical (compared as
    # sorted key/value tuples), i.e. at least two distinct checksum sets exist. Prints the number
    # of rollouts and distinct checksum sets. Raises assertion errors, including from the
    # reduction helper on duplicates.
    by_model_and_rollout = _checksums_by_model_and_rollout_id(_read_inference_engine_checksum_events(Path(dump_dir)))
    assert len(by_model_and_rollout) > 1, (
        f"{side}: engine weight checksums cover {sorted(by_model_and_rollout)}, so there is no pair to compare "
        f"and nothing proves the run pushed an update at all"
    )

    distinct: set[tuple[tuple[str, str], ...]] = {tuple(sorted(one.items())) for one in by_model_and_rollout.values()}
    assert len(distinct) > 1, (
        f"{side}: every one of {len(by_model_and_rollout)} rollouts pushed byte-identical engine weights, so the "
        f"optimizer moved nothing and a bitwise comparison against another such run would prove nothing"
    )

    print(
        f"{side}: engine weights moved across {len(by_model_and_rollout)} rollout(s), "
        f"{len(distinct)} distinct checksum(s)"
    )


def _checksums_by_model_and_rollout_id(
    events: list[InferenceEngineWeightChecksumEvent],
) -> dict[tuple[str | None, int], dict[str, str]]:
    # [FUNCTION DOC] HELPER: Indexes events by (trainer model id, rollout id), keeping the first
    # engine's checksum mapping (name to checksum string) of each. Asserts each key occurs only
    # once and that each event has at least one engine checksum mapping.
    by_model_and_rollout: dict[tuple[str | None, int], dict[str, str]] = {}
    for event in events:
        key = (event.trainer_model_id, event.rollout_id)
        assert key not in by_model_and_rollout, f"Duplicate InferenceEngineWeightChecksumEvent for {key}"
        assert event.engine_checksums, f"No engine checksums for {key}"
        by_model_and_rollout[key] = event.engine_checksums[0]
    return by_model_and_rollout


def _read_inference_engine_checksum_events(dump_dir: Path) -> list[InferenceEngineWeightChecksumEvent]:
    # [FUNCTION DOC] HELPER: Reads all audit events from the events subdirectory of a dump
    # directory and keeps only engine weight checksum events. Returns an empty list if the events
    # directory does not exist.
    """Read all InferenceEngineWeightChecksumEvents from the events directory."""
    events_dir: Path = dump_dir / EVENTS_DIRNAME
    if not events_dir.exists():
        return []
    all_events = read_events(events_dir)
    return [e for e in all_events if isinstance(e, InferenceEngineWeightChecksumEvent)]
