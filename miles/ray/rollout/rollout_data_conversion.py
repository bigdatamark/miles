# [MODULE DOC] Post-processing of freshly collected rollout data before it is converted for
# training: validates sibling samples of compact rollouts, flattens nested lists, and trims the
# sample list to a multiple of the global batch size (optionally a dynamic one chosen to yield
# exactly one training step per rollout).
import itertools
import logging

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def postprocess_rollout_data(args, data, train_parallel_config):
    # [FUNCTION DOC] Turn raw rollout output (a possibly nested list of samples) into a flat list ready
    # for batching. Returns (data, metadata mapping). Steps: validate compact rollout ids; flatten
    # nested lists until the first element is not a list; a rollout is "compact" if any sample has a
    # rollout id, and compact rollouts are never trimmed by sample count. Unless trimming is
    # disabled or the rollout is compact: pick the global batch size from args, or, with dynamic
    # global batch size enabled, compute it from the data-parallel size and sample count and record
    # it in metadata under dynamic_global_batch_size; if the sample count is not a multiple of the
    # batch size, drop trailing samples down to the largest multiple (raises value error when fewer
    # samples than one batch). Logs counts. The input must be non-empty (indexing the first element).
    metadata = {}

    validate_compact_rollout_ids(data)

    # flatten the data if it is a list of lists
    while isinstance(data[0], list):
        data = list(itertools.chain.from_iterable(data))

    # Compact rollouts must not be trimmed by sample count; the schedule drops
    # whole trailing rollouts instead.
    is_compact = any(s.rollout_id is not None for s in data)

    if not args.disable_rollout_trim_samples and not is_compact:
        global_batch_size = args.global_batch_size
        if args.use_dynamic_global_batch_size:
            logger.info(f"Collected {len(data)} samples from rollout to train with dynamic global batch size")
            dynamic_global_batch_size = _compute_dynamic_global_batch_size(
                args, train_parallel_config=train_parallel_config, num_samples=len(data)
            )
            metadata["dynamic_global_batch_size"] = dynamic_global_batch_size
            global_batch_size = dynamic_global_batch_size

        if len(data) % global_batch_size != 0:
            trim_len = (len(data) // global_batch_size) * global_batch_size
            if trim_len == 0:
                raise ValueError(f"Not enough samples {len(data)} for global_batch_size {global_batch_size}")
            origin_data_length = len(data)
            data = data[:trim_len]
            logger.info(f"trim number of samples from {origin_data_length} to {trim_len}")
        logger.info(f"Final collected {len(data)} samples from rollout to train")

    return data, metadata


def validate_compact_rollout_ids(node, depth=0):
    # [FUNCTION DOC] Recursively validate a nested rollout output node (a sample, or a list of nodes).
    # A lone sample is fine. A non-list, non-sample node raises assertion error. For a list whose
    # first element is a sample (a leaf group): only if depth is at least 2 and it has more than one
    # sample, asserts that no sample lacks a rollout id (message lists missing positions) and that
    # all share the same rollout id. Other lists recurse into each item with depth plus one.
    # Shallow leaves (ordinary groups at depth below 2, or single-sample leaves) are not checked.
    """Require compact leaves (``list[Sample]`` at depth >= 2, >1 sibling) to
    share a non-None ``rollout_id``; default rollout shapes skip validation."""
    if isinstance(node, Sample):
        return
    assert isinstance(node, list), f"unexpected rollout output node type: {type(node).__name__}"
    if node and isinstance(node[0], Sample):
        if depth >= 2 and len(node) > 1:
            rids = [s.rollout_id for s in node]
            missing = [i for i, r in enumerate(rids) if r is None]
            assert not missing, (
                f"Compact rollout returned {len(node)} samples but rollout_id is unset on "
                f"positions {missing}. Set Sample.rollout_id on every sibling so the loss "
                "reducer can aggregate them as one rollout instead of N."
            )
            assert len(set(rids)) == 1, f"Sibling samples from one compact rollout must share rollout_id; got {rids}."
        return
    for item in node:
        validate_compact_rollout_ids(item, depth + 1)


def _first_sample(group):
    # [FUNCTION DOC] HELPER: The first sample found by repeatedly taking the first element of nested
    # lists. KNOWN ISSUE: not used anywhere in this module (possibly dead code).
    return _first_sample(group[0]) if isinstance(group[0], list) else group[0]


def _nested_sample_count(group) -> int:
    # [FUNCTION DOC] HELPER: Total number of non-list leaves in a nested list structure (a non-list
    # counts as one). KNOWN ISSUE: not used anywhere in this module (possibly dead code).
    if not isinstance(group, list):
        return 1
    return sum(_nested_sample_count(item) for item in group)


def _compute_dynamic_global_batch_size(args, train_parallel_config, num_samples: int) -> int:
    # [FUNCTION DOC] HELPER: Choose a batch size that gives exactly one training step: the sample count
    # rounded down to a multiple of the data-parallel size (read from the parallel config mapping).
    # If that is zero (fewer samples than the dp size) use the dp size and warn; in that case the
    # later trimming raises since samples are fewer than a batch. Logs when the result differs from
    # the configured batch size or samples are wasted. Returns the integer batch size.
    """Calculate dynamic global_batch_size to ensure only one training step.

    Strategy: global_batch_size = num_samples rounded down to a multiple of dp_size
    This ensures num_steps_per_rollout = num_samples // global_batch_size = 1
    """
    dp_size = train_parallel_config["dp_size"]
    original_gbs = args.global_batch_size

    # Round down to a multiple of dp_size to ensure only one training step
    dynamic_gbs = (num_samples // dp_size) * dp_size

    if dynamic_gbs == 0:
        # Too few samples, use at least dp_size
        dynamic_gbs = dp_size
        logger.warning(f"num_samples={num_samples} < dp_size={dp_size}, using dp_size as global_batch_size")

    # Calculate how many samples will be discarded
    wasted = num_samples - dynamic_gbs

    if dynamic_gbs != original_gbs or wasted > 0:
        logger.info(
            f"Dynamic global_batch_size: {original_gbs} -> {dynamic_gbs} "
            f"(num_samples={num_samples}, dp_size={dp_size}, "
            f"num_steps=1, wasted={wasted})"
        )

    return dynamic_gbs
