# [MODULE DOC] Comparison routine shared by FT tests that expect baseline and target runs to be
# bit-identical. Defines the metric key prefixes that are compared (train and rollout) and those
# deliberately ignored (perf).

from pathlib import Path

from tests.e2e.ft.conftest_ft.app import BASELINE_SIDE, TARGET_SIDE

from miles.utils.audit_utils.event_logger.logger import EVENTS_DIRNAME
from miles.utils.test_utils.comparisons.dumps import (
    INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    INPUT_TENSORS_SKIP_PATTERN,
    compare_dumps,
)
from miles.utils.test_utils.comparisons.inference_engine_checksums import (
    assert_engine_weights_moved,
    compare_inference_engine_checksums,
)
from miles.utils.test_utils.comparisons.metrics import (
    assert_gradients_nonzero,
    assert_metrics_classified,
    compare_metrics,
)
from miles.utils.test_utils.reconfigure_assertions import assert_reconfigure_events

COMPARED_METRIC_PREFIXES: tuple[str, ...] = ("train/", "rollout/")
UNCOMPARED_METRIC_PREFIXES: tuple[str, ...] = ("perf/",)


def compare_deterministic_sides(
    *,
    baseline_dir: str,
    target_dir: str,
    min_trained_rollouts: int,
    exclude_keys: list[str] | None = None,
) -> None:
    # [FUNCTION DOC] Asserts the baseline and target dump directories are exactly equal. Inputs:
    # the two side directories, the minimum number of trained rollouts required for the gradient
    # check, and optional metric keys to exclude from comparison. Steps in order: (1) both sides
    # must have recorded no reconfigure events in their events sub-directory; (2) every metric in
    # each side must be classified as compared or ignored by prefix; (3) metrics with the compared
    # prefixes are compared with zero relative and absolute tolerance, minus excluded keys;
    # (4) tensor dumps are compared with a threshold requiring zero relative difference for every
    # name, tolerating skipped or failed input tensors matching the shared allow patterns;
    # (5) inference engine weight checksums of the two sides are compared; (6) for each side,
    # engine weights must have changed during training and gradients must be nonzero for at
    # least the minimum number of trained rollouts. Raises assertion errors from any check.
    for side_dir in (baseline_dir, target_dir):
        assert_reconfigure_events(Path(side_dir) / EVENTS_DIRNAME, expected=[])

    for side_dir in (baseline_dir, target_dir):
        assert_metrics_classified(side_dir, compared=COMPARED_METRIC_PREFIXES, ignored=UNCOMPARED_METRIC_PREFIXES)
    compare_metrics(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        rtol=0.0,
        atol=0.0,
        key_prefixes=list(COMPARED_METRIC_PREFIXES),
        exclude_keys=list(exclude_keys or []),
    )

    compare_dumps(
        baseline_dir=baseline_dir,
        target_dir=target_dir,
        diff_thresholds=[(".*", "rel <= 0")],
        allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
        allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    )

    compare_inference_engine_checksums(baseline_dir=baseline_dir, target_dir=target_dir)

    for side, side_dir in ((BASELINE_SIDE, baseline_dir), (TARGET_SIDE, target_dir)):
        assert_engine_weights_moved(side=side, dump_dir=side_dir)
        assert_gradients_nonzero(side=side, dump_dir=side_dir, min_trained_rollouts=min_trained_rollouts)
