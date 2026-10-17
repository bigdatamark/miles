# [MODULE DOC] Debug dumps and loads of rollout data. Provides: a JSON-lines trajectory sidecar of raw
# conversations, a parquet "dashboard columns" mirror of per-token columns for the dashboard, a
# torch-serialized rollout dump (save and load, with optional subsample on load), path template
# and file stem helpers, and a CI utility that injects recorded rollout data in place of generated
# data and checks the generated batch matches it. Dump file stems are
# optional "eval_" prefix + optional trainer model id prefix + rollout id.
import json
import logging
from collections import defaultdict
from pathlib import Path
from string import Formatter

import torch

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def trajectory_rows(samples: list[Sample]) -> list[dict]:
    # [FUNCTION DOC] Build trajectory rows, one per sample whose metadata holds a raw conversation
    # (messages list); samples without metadata or messages are skipped. Each row is a mapping with
    # sample index, sample occurrence, group index, status (the status enum's string value), reward
    # (kept only if numeric, else nothing), prompt, and messages. The occurrence is the running
    # count of earlier samples with the same index over the full input list (including skipped
    # ones), so numbering matches the dashboard. Pure function; returns the list of rows.
    """One row per sample that recorded a raw conversation
    (``metadata["messages"]``, attached by the session / multi_turn paths).

    ``sample_occurrence`` counts over the FULL sample list -- the numbering the
    dashboard uses everywhere -- not over the recorded subset, so a sample
    stays addressable even when an earlier leaf of the same index recorded no
    conversation."""
    rows = []
    occurrences: defaultdict[int, int] = defaultdict(int)
    for sample in samples:
        occurrence = occurrences[sample.index]
        occurrences[sample.index] += 1
        messages = sample.metadata.get("messages") if sample.metadata else None
        if messages is None:
            continue
        rows.append(
            dict(
                sample_index=sample.index,
                sample_occurrence=occurrence,
                group_index=sample.group_index,
                status=sample.status.value,
                reward=sample.reward if isinstance(sample.reward, (int, float)) else None,
                prompt=sample.prompt,
                messages=messages,
            )
        )
    return rows


_warned_no_polars = False


def save_dashboard_columns(samples: list[Sample], path: Path) -> None:
    # [FUNCTION DOC] Write a parquet file with one row per sample holding sample index, response
    # length, total length (token count), tokens (32-bit ints), loss mask (8-bit ints) and rollout
    # log probs (32-bit floats), with small row groups (8 rows) for point reads. The polars library
    # is optional: if missing, a warning is logged once per process (module-level flag) and nothing
    # is written. Writes atomically: parents are created, data goes to a temporary file with a tmp
    # suffix next to the target, then replaces the target. Loss mask and log probs may be nothing
    # per sample, depending on schema nullability. Returns nothing.
    """Point-read mirror of the per-token rollout columns (parquet, one row
    per sample) so the dashboard token view never full-loads the .pt."""
    global _warned_no_polars
    try:
        import polars as pl
    except ImportError:
        if not _warned_no_polars:
            logger.warning("polars not installed; skipping dashboard_columns dump (pip install miles[dashboard])")
            _warned_no_polars = True
        return
    schema = dict(
        sample_index=pl.Int32,
        response_length=pl.Int32,
        total_length=pl.Int32,
        tokens=pl.List(pl.Int32),
        loss_mask=pl.List(pl.Int8),
        rollout_log_probs=pl.List(pl.Float32),
    )
    frame = pl.DataFrame(
        [
            dict(
                sample_index=sample.index,
                response_length=sample.response_length,
                total_length=len(sample.tokens),
                tokens=list(sample.tokens),
                loss_mask=sample.loss_mask,
                rollout_log_probs=sample.rollout_log_probs,
            )
            for sample in samples
        ],
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.write_parquet(tmp, row_group_size=8)
    tmp.replace(path)


def save_debug_trajectory_data(
    args, samples: list[Sample], rollout_id, evaluation: bool, trainer_model_id: str | None = None
):
    # [FUNCTION DOC] Dump raw conversations to a JSON-lines file if the save-trajectory option in
    # args is set (otherwise return). Builds trajectory rows; if none exist, writes no file (the
    # dashboard keys off the file's presence). Otherwise formats the path from the template and the
    # dump stem, creates parent directories, writes one compact JSON object per line, and then
    # removes the messages entry from every sample's metadata (mutation) so the later torch dump
    # stays small. Returns nothing.
    if (path_template := args.save_debug_trajectory_data) is None:
        return
    rows = trajectory_rows(samples)
    if not rows:
        return  # no conversations: no file (the dashboard keys off its presence)
    stem = _compute_dump_stem(rollout_id, evaluation=evaluation, trainer_model_id=trainer_model_id)
    path = _format_dump_path(path_template, stem=stem)
    logger.info(f"Save trajectory dump to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    for sample in samples:
        if sample.metadata:
            sample.metadata.pop("messages", None)  # the sidecar is their home; keep the .pt lean


def load_debug_rollout_data(args, rollout_id: int) -> tuple[list[Sample], dict]:
    # [FUNCTION DOC] Load the debug rollout dump whose path is the load template in args with the
    # rollout id substituted. Returns (samples, metadata mapping). If a subsample ratio is set,
    # keeps roughly that fraction: the first half of the target row count from the front and the
    # same count from the end, and logs the change. KNOWN ISSUE: the tail slice is written with a
    # negated half count after floor division of the target, i.e. negative floor division, so for an
    # odd target count the tail is one longer than the head, and a ratio giving zero rows makes the
    # tail slice start at the very beginning (empty-negative edge) and return all rows.
    data, metadata = _load_rollout_data_file(Path(args.load_debug_rollout_data.format(rollout_id=rollout_id)))
    if (ratio := args.load_debug_rollout_data_subsample) is not None:
        original_num_rows = len(data)
        rough_subsample_num_rows = int(original_num_rows * ratio)
        data = data[: rough_subsample_num_rows // 2] + data[-rough_subsample_num_rows // 2 :]
        logger.info(
            f"Subsample loaded debug rollout data using {ratio=} and change num rows {original_num_rows} -> {len(data)}"
        )
    return data, metadata


def save_debug_rollout_data(
    args, data, rollout_id, evaluation: bool, metadata: dict | None = None, trainer_model_id: str | None = None
) -> None:
    # [FUNCTION DOC] Save a rollout (training or evaluation) to disk if the save option in args is
    # set. Input data is a list of samples for training, or for evaluation a mapping whose values
    # each hold a samples list (flattened in order). Steps: compute stem and path; create parent
    # directory; flatten samples; call trajectory dump (which strips conversation messages from
    # sample metadata); write dashboard columns parquet to a sibling-of-parent directory named
    # dashboard_columns, named by the rollout stem; finally torch-serialize a mapping with rollout
    # id, metadata (empty if none) and the list of sample dictionaries to the path.
    # TODO to be refactored (originally Buffer._set_data)
    if (path_template := args.save_debug_rollout_data) is not None:
        stem = _compute_dump_stem(rollout_id, evaluation=evaluation, trainer_model_id=trainer_model_id)
        path = _format_dump_path(path_template, stem=stem)
        logger.info(f"Save debug rollout data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        samples = [sample for info in data.values() for sample in info["samples"]] if evaluation else list(data)
        save_debug_trajectory_data(args, samples, rollout_id, evaluation, trainer_model_id=trainer_model_id)
        save_dashboard_columns(samples, path.parent.parent / "dashboard_columns" / f"rollout_{stem}.parquet")

        # TODO may improve the format
        dump_data = dict(samples=[sample.to_dict() for sample in samples])
        torch.save(dict(rollout_id=rollout_id, metadata=metadata or {}, **dump_data), path)


def _format_dump_path(path_template: str, *, stem: str) -> Path:
    # [FUNCTION DOC] HELPER: Substitute the stem into the template's rollout id placeholder and
    # return a path. Raises assertion error if the template has no such placeholder field.
    assert any(
        field_name == "rollout_id" for _, field_name, _, _ in Formatter().parse(path_template)
    ), f"Debug dump path template {path_template!r} must contain the {{rollout_id}} placeholder"
    return Path(path_template.format(rollout_id=stem))


def _compute_dump_stem(rollout_id, *, evaluation: bool, trainer_model_id: str | None) -> str:
    # [FUNCTION DOC] HELPER: Build the file stem: the rollout id as text; if a trainer model id is
    # given, prefix it joined by an underscore; if this is an evaluation, prefix "eval_" outermost.
    ans = str(rollout_id)
    if (x := trainer_model_id) is not None:
        ans = f"{x}_{ans}"
    if evaluation:
        ans = f"eval_{ans}"
    return ans


class RolloutDataInjectionUtil:
    # [CLASS DOC] HELPER: Stateless namespace (class methods only) for the CI feature that replaces
    # freshly generated rollout data with a previously recorded dump and verifies the two agree
    # closely enough. Configured through the inject path, start rollout id and minimum match ratio
    # options in args.

    @classmethod
    def should_inject(cls, args, rollout_id: int) -> bool:
        # [FUNCTION DOC] True when an inject path is configured and the rollout id is at or after
        # the configured start rollout id; false if no path is configured.
        if args.ci_inject_rollout_data_path is None:
            return False
        return rollout_id >= args.ci_inject_rollout_data_start_rollout_id

    @classmethod
    def load(cls, args, rollout_id: int) -> tuple[list[Sample], dict]:
        # [FUNCTION DOC] Load the recorded dump for the rollout id from the inject path template.
        # Raises assertion error if the file does not exist. Logs the replacement. Returns
        # (samples, metadata) from the shared file loader.
        path = Path(args.ci_inject_rollout_data_path.format(rollout_id=rollout_id))
        assert path.is_file(), f"Recorded rollout data to inject is missing: {path}"
        logger.info(f"CI rollout-data injection: replacing generated data of rollout {rollout_id} with {path}")
        return _load_rollout_data_file(path)

    @classmethod
    def assert_matches_generated(
        cls, args, *, generated: list[Sample], injected: list[Sample], rollout_id: int
    ) -> None:
        # [FUNCTION DOC] Verify the generated batch pairs with the injected recording. Raises
        # assertion error if the lengths differ, if any pair has different prompt tokens (prompt =
        # tokens before the response), or if the mean over samples of the response token match
        # ratio is not strictly greater than the minimum match ratio from args. Logs the mean, the
        # minimum and the threshold. An empty batch with equal (zero) lengths would divide by zero
        # when averaging.
        min_match_ratio: float = args.ci_inject_rollout_data_min_match_ratio
        assert len(generated) == len(
            injected
        ), f"rollout {rollout_id}: sample count mismatch, generated {len(generated)} vs injected {len(injected)}"

        ratios: list[float] = []
        for index, (generated_sample, injected_sample) in enumerate(zip(generated, injected, strict=True)):
            assert cls._prompt_tokens(generated_sample) == cls._prompt_tokens(injected_sample), (
                f"rollout {rollout_id}: prompt tokens mismatch at sample {index}; "
                "injected recording does not pair with the generated batch"
            )
            ratios.append(cls._response_token_match_ratio(generated_sample, injected_sample))

        mean_ratio = sum(ratios) / len(ratios)
        logger.info(
            f"CI rollout-data injection match for rollout {rollout_id}: "
            f"mean response token match {mean_ratio:.4f} (min {min(ratios):.4f}, "
            f"threshold {min_match_ratio}) over {len(ratios)} samples"
        )
        assert mean_ratio > min_match_ratio, (
            f"rollout {rollout_id}: generated responses match the injected recording at only "
            f"{mean_ratio:.4f} (threshold {min_match_ratio}); the engine "
            "weights likely diverged from the baseline beyond ulp-level drift"
        )

    @classmethod
    def _response_token_match_ratio(cls, a: Sample, b: Sample) -> float:
        # [FUNCTION DOC] HELPER: Fraction of positionally equal response tokens between two samples,
        # divided by the longer response length; 1.0 when both responses are empty.
        response_a = cls._response_tokens(a)
        response_b = cls._response_tokens(b)
        denominator = max(len(response_a), len(response_b))
        if denominator == 0:
            return 1.0
        matched = sum(token_a == token_b for token_a, token_b in zip(response_a, response_b, strict=False))
        return matched / denominator

    @classmethod
    def _prompt_tokens(cls, sample: Sample) -> list[int]:
        # [FUNCTION DOC] HELPER: The leading tokens before the trailing response_length tokens.
        return sample.tokens[: len(sample.tokens) - sample.response_length]

    @classmethod
    def _response_tokens(cls, sample: Sample) -> list[int]:
        # [FUNCTION DOC] HELPER: The trailing response_length tokens of the sample.
        return sample.tokens[len(sample.tokens) - sample.response_length :]


def _load_rollout_data_file(path: Path) -> tuple[list[Sample], dict]:
    # [FUNCTION DOC] HELPER: Deserialize a torch dump (with arbitrary-object loading enabled, so only
    # trusted files should be loaded), convert each stored dictionary back into a sample, and
    # return (samples, metadata) where metadata is the stored mapping or empty if missing/falsy.
    payload = torch.load(path, weights_only=False)
    data = [Sample.from_dict(sample) for sample in payload["samples"]]
    metadata = payload.get("metadata") or {}
    return data, metadata
