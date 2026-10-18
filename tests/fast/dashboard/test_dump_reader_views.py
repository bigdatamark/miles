# [MODULE DOC]
# Tests for the derived views of the dump reader: per-step sample summary (with its parquet
# cache and versioning), per-group statistics, cross-step aggregates, token level payloads,
# trajectory lookup by occurrence, and the weight-version / turn summary helper. Uses synthetic
# dumps from the dummy dump writer, and one optional real-data test gated by an environment
# variable naming a dump directory.
import json
import os
import shutil
import statistics
import time

import polars as pl
import pytest
from tests.fast.dashboard.dummy_dump import blank_samples, dump_dummy_run

from miles.dashboard.dump_reader import DumpReader, _weight_version_summary
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

REMOVED = (3,)  # within-step positions marked remove_sample=True by the fixture


@pytest.fixture
def reader(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture. Writes a 2-step dummy dump (data-parallel size 2, tensor
    # parallel duplication 2, with eval) whose within-step position 3 is marked as removed, and
    # returns a reader over it.
    dump_dummy_run(tmp_path, steps=2, dp_size=2, tp_dup=2, with_eval=True, remove_sample_indices=REMOVED)
    return DumpReader(tmp_path)


def _df_row(df, sample_index):
    # [FUNCTION DOC] HELPER: Returns the single row of the frame whose sample index column equals
    # the given index as a mapping from column name to value; asserts exactly one row matches.
    matches = df.filter(df["sample_index"] == sample_index)
    assert matches.height == 1
    return matches.row(0, named=True)


def test_summary_matches_hand_computed(reader):
    # [FUNCTION DOC] For step 0, asserts the summary has one row per sample and each row's group,
    # response length, raw reward, normalised reward and dumped rank match the joined data. For
    # samples with any unmasked response position, recomputes (within 1e-5 relative) the mean and
    # max absolute difference between train and rollout log probs, the mean importance ratio
    # (exponential of the difference) and mean entropy over the masked positions only.
    joined = reader.load_joined(0)
    df = reader.summary(0)
    assert df.height == len(joined.samples)

    for sample in joined.samples:
        entry = _df_row(df, sample.index)
        row = joined.train_rows[(sample.index, 0)]
        assert entry["group_index"] == sample.group_index
        assert entry["response_length"] == sample.response_length
        assert entry["raw_reward"] == sample.reward
        assert entry["normalized_reward"] == pytest.approx(row.reward)
        assert entry["dumped_rank"] == row.rank

        mask = row.loss_mask > 0
        if mask.any():
            expected = (row.log_probs - row.rollout_log_probs).abs()[mask]
            assert entry["mean_abs_lp_diff"] == pytest.approx(float(expected.mean()), rel=1e-5)
            assert entry["max_abs_lp_diff"] == pytest.approx(float(expected.max()), rel=1e-5)
            assert entry["mean_imp_ratio"] == pytest.approx(
                float((row.log_probs - row.rollout_log_probs).exp()[mask].mean()), rel=1e-5
            )
            assert entry["mean_entropy"] == pytest.approx(float(row.entropy[mask].mean()), rel=1e-5)


def test_summary_uses_speculative_counters(reader):
    # [FUNCTION DOC] Builds a sample with 1 correct of 4 proposed speculative drafts, calls the
    # reader's private summary row builder without a train row, and asserts the acceptance rate
    # is 0.25.
    sample = Sample(spec_info=Sample.SpecInfo(spec_num_correct_drafts=1, spec_num_proposed_drafts=4))

    entry = reader._summary_row(sample, None, rollout_id=0, sample_occurrence=0)

    assert entry["spec_accept_rate"] == 0.25


def test_summary_removed_sample_has_no_masked_stats(reader):
    # [FUNCTION DOC] The summary row of the removed sample (step 0, position 3) has the removal
    # flag true and none for entropy, log prob difference, importance ratio, advantage and
    # return means.
    df = reader.summary(0)
    entry = _df_row(df, REMOVED[0])  # step 0: Sample.index == within-step position
    assert entry["remove_sample"] is True
    for column in ("mean_entropy", "mean_abs_lp_diff", "mean_imp_ratio", "adv_mean", "return_mean"):
        assert entry[column] is None, column


def test_summary_parquet_cache_hit_and_invalidation(reader, monkeypatch):
    # [FUNCTION DOC] Computes the step 0 summary and asserts its parquet cache file exists. Then
    # makes every dump load raise and clears the in-memory joined cache: the summary is still
    # served equal from parquet. After bumping a train file's modification time 5 seconds into the
    # future, the summary must recompute (hitting the failing load) and raise the runtime error.
    df = reader.summary(0)
    assert (reader.cache_dir / "rollout_0.parquet").exists()

    def boom(self, path):
        # [FUNCTION DOC] HELPER: Replacement for the reader's torch load; always raises a
        # runtime error saying recompute was attempted.
        raise RuntimeError("recompute attempted")

    monkeypatch.setattr(DumpReader, "_torch_load", boom)
    reader._joined_cache.clear()
    assert reader.summary(0).equals(df)  # served from parquet, no dump loads

    stamp = time.time() + 5
    os.utime(next(reader.train_dir.glob("0_*.pt")), (stamp, stamp))
    with pytest.raises(RuntimeError, match="recompute attempted"):
        reader.summary(0)  # source changed -> cache invalidated -> recompute


def test_groups(reader):
    # [FUNCTION DOC] Asserts the groups frame has one row per sample group, and each row's count,
    # mean reward, sample standard deviation and zero-std flag (std at most 1e-12) match values
    # recomputed from the raw sample rewards.
    joined = reader.load_joined(0)
    groups_df = reader.groups(0)
    rewards_by_group = {}
    for sample in joined.samples:
        rewards_by_group.setdefault(sample.group_index, []).append(sample.reward)
    assert groups_df.height == len(rewards_by_group)

    for entry in groups_df.iter_rows(named=True):
        rewards = rewards_by_group[entry["group_index"]]
        assert entry["n"] == len(rewards)
        assert entry["reward_mean"] == pytest.approx(statistics.mean(rewards))
        assert entry["reward_std"] == pytest.approx(statistics.stdev(rewards))
        assert entry["zero_std"] == (statistics.stdev(rewards) <= 1e-12)


def test_step_aggregates(reader):
    # [FUNCTION DOC] Asserts step aggregates list rollout ids [0, 1] and per step the sample
    # count and mean reward equal the summary's, with a non-null mean log prob difference.
    aggregates = reader.step_aggregates()
    assert aggregates["rollout_id"].to_list() == [0, 1]
    for entry in aggregates.iter_rows(named=True):
        df = reader.summary(entry["rollout_id"])
        assert entry["n_samples"] == df.height
        assert entry["reward_mean"] == pytest.approx(df["raw_reward"].mean())
        assert entry["mean_abs_lp_diff"] is not None


def test_tokens_full_range(reader):
    # [FUNCTION DOC] Requests the whole token payload for the first sample of step 0 and asserts:
    # total length, prompt length equals response offset, token ids, train log prob and importance
    # ratio list lengths equal the response length, rollout log probs match, and the first
    # log-prob difference equals train minus rollout log prob from the joined row.
    joined = reader.load_joined(0)
    sample = joined.samples[0]
    payload = reader.tokens(0, sample.index)
    prompt_len = len(sample.tokens) - sample.response_length

    assert payload["total_len"] == len(sample.tokens)
    assert payload["prompt_len"] == payload["response_offset"] == prompt_len
    assert payload["token_ids"] == sample.tokens
    assert len(payload["train_log_probs"]) == sample.response_length
    assert len(payload["imp_ratio"]) == sample.response_length
    assert payload["rollout_log_probs"] == pytest.approx(sample.rollout_log_probs)
    row = joined.train_rows[(sample.index, 0)]
    assert payload["lp_diff"][0] == pytest.approx(float(row.log_probs[0] - row.rollout_log_probs[0]))


def test_tokens_null_the_stats_where_the_loss_is_masked(reader):
    # [FUNCTION DOC] For the removed sample (loss mask all zero), every per-token statistic list
    # (train and rollout log probs, difference, ratio, reference log probs, advantages) is a list
    # of nulls of mask length. For an ordinary sample the mask is all ones and train log probs
    # equal the joined row's values.
    """mask=0 positions hold placeholders the engine never scored: they serialize
    as null, never as numbers a chart would mistake for data."""
    masked = reader.tokens(0, REMOVED[0])
    assert masked["loss_mask"] and set(masked["loss_mask"]) == {0}
    for key in ("train_log_probs", "rollout_log_probs", "lp_diff", "imp_ratio", "ref_log_probs", "advantages"):
        assert masked[key] == [None] * len(masked["loss_mask"]), key

    row = reader.load_joined(0).train_rows[(0, 0)]
    unmasked = reader.tokens(0, 0)
    assert set(unmasked["loss_mask"]) == {1}
    assert unmasked["train_log_probs"] == pytest.approx([float(v) for v in row.log_probs])


def test_tokens_window_straddles_response_boundary(reader):
    # [FUNCTION DOC] Exercises token windows: a window from two before the prompt end to three
    # after yields 5 tokens, response offset 2 and 3 stat values; a prompt-only window has empty
    # stats and offset 2; an end far beyond the sequence is clamped to total length; an empty
    # range raises a value error mentioning "empty token range"; an unknown sample index raises a
    # key error mentioning "unknown sample_index".
    sample = reader.load_joined(0).samples[0]
    prompt_len = len(sample.tokens) - sample.response_length

    payload = reader.tokens(0, sample.index, start=prompt_len - 2, end=prompt_len + 3)
    assert len(payload["token_ids"]) == 5
    assert payload["response_offset"] == 2
    assert len(payload["train_log_probs"]) == 3  # stats only cover the response overlap

    prompt_only = reader.tokens(0, sample.index, start=0, end=2)
    assert prompt_only["train_log_probs"] == []
    assert prompt_only["response_offset"] == 2

    clamped = reader.tokens(0, sample.index, start=0, end=10**6)
    assert clamped["end"] == len(sample.tokens)

    with pytest.raises(ValueError, match="empty token range"):
        reader.tokens(0, sample.index, start=5, end=5)

    with pytest.raises(KeyError, match="unknown sample_index"):
        reader.tokens(0, 10**6)


def test_tokens_decode_via_dumped_tokenizer(reader):
    # [FUNCTION DOC] Asserts the token text list is the dummy tokenizer's rendering "t" followed
    # by each token id.
    payload = reader.tokens(0, 0)
    assert payload["token_text"] == [f"t{tid}" for tid in payload["token_ids"]]


def test_tokens_eval_sample_has_rollout_side_only(reader):
    # [FUNCTION DOC] For an eval sample, rollout log probs are present while train log probs,
    # difference, ratio, entropy, advantages and loss mask are all none.
    payload = reader.tokens(0, 0, evaluation=True)
    assert payload["rollout_log_probs"] is not None
    for key in ("train_log_probs", "lp_diff", "imp_ratio", "entropy", "advantages", "loss_mask"):
        assert payload[key] is None, key


def test_tokens_without_entropy_or_tokenizer(tmp_path):
    # [FUNCTION DOC] Dumps one step with no entropy, no eval and no tokenizer. Asserts entropy and
    # reference entropy are none, train log probs are present, token text is none and the summary
    # mean entropy for sample 0 is none.
    dump_dummy_run(tmp_path, steps=1, with_entropy=False, with_eval=False, with_tokenizer=False)
    reader = DumpReader(tmp_path)
    payload = reader.tokens(0, 0)
    assert payload["entropy"] is None and payload["ref_entropy"] is None
    assert payload["train_log_probs"] is not None
    assert payload["token_text"] is None
    assert _df_row(reader.summary(0), 0)["mean_entropy"] is None


def test_joined_lru_eviction(tmp_path):
    # [FUNCTION DOC] With a tensor cache capacity of 1: the second joined call for step 0 returns
    # the same object (cache hit); loading step 1 evicts step 0 so the cache keys are only
    # (1, not-evaluation).
    dump_dummy_run(tmp_path, steps=2, with_eval=False)
    reader = DumpReader(tmp_path, tensor_lru=1)
    first = reader.joined(0)
    assert reader.joined(0) is first  # cache hit
    reader.joined(1)
    assert list(reader._joined_cache) == [(1, False)]  # 0 evicted


@pytest.mark.skipif(
    "MILES_DASHBOARD_REALDATA_DIR" not in os.environ,
    reason="set MILES_DASHBOARD_REALDATA_DIR to a real --dump-details dir",
)
def test_realdata_views(tmp_path):
    # [FUNCTION DOC] Skipped unless the real-data directory environment variable is set. Against
    # one specific real dump (cache in the temporary path) asserts hard-coded measured values:
    # step 0 has 256 samples of which 112 truncated, no null log-prob-difference or entropy
    # means, 32 groups (8 samples per prompt), a 64-token window with text, and 5 aggregate steps.
    # KNOWN ISSUE: the numbers are tied to one particular recorded run, so it is not portable.
    reader = DumpReader(os.environ["MILES_DASHBOARD_REALDATA_DIR"], cache_dir=tmp_path)
    df = reader.summary(0)
    assert df.height == 256
    assert df["truncated"].cast(int).sum() == 112  # measured on qwen30b-dash step 0
    assert df["mean_abs_lp_diff"].null_count() == 0
    assert df["mean_entropy"].null_count() == 0

    groups_df = reader.groups(0)
    assert groups_df.height == 32  # 256 samples / 8 per prompt

    sample_index = int(df["sample_index"][0])
    payload = reader.tokens(0, sample_index, start=0, end=64)
    assert len(payload["token_ids"]) == 64
    assert payload["token_text"] is not None

    aggregates = reader.step_aggregates()
    assert aggregates.height == 5


def test_summary_and_tokens_survive_dump_without_log_probs(tmp_path):
    # [FUNCTION DOC] Dumps a default run and removes the log probs column from every step 0 train
    # shard (asserting it was present). The summary then has all-null log-prob-difference and
    # importance-ratio means, reward not entirely null, and step aggregates still produce rows.
    # KNOWN ISSUE: despite the name, no token payload is requested, and the function re-imports
    # names already imported at module level.
    """A 744B-scale run may not dump train log_probs at all: everything
    derived from them degrades to None instead of KeyError -> HTTP 404
    (disagg report 2026-07-14)."""
    import torch
    from tests.fast.dashboard.dummy_dump import dump_dummy_run

    from miles.dashboard.dump_reader import DumpReader

    dump_dummy_run(tmp_path)
    for shard in (tmp_path / "train_data").glob("0_*.pt"):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        assert payload["rollout_data"].pop("log_probs") is not None  # must actually remove it
        torch.save(payload, shard)

    reader = DumpReader(tmp_path)
    df = reader.summary(0)
    assert df["mean_abs_lp_diff"].is_null().all()
    assert df["mean_imp_ratio"].is_null().all()
    assert df["reward"].null_count() < df.height  # the rest of the summary is intact
    assert reader.step_aggregates().height >= 1  # the metrics page path


def test_staleness_and_agentic_columns(reader):
    # [FUNCTION DOC] In the dummy dump every third sample is a two-turn agentic sample with mixed
    # weight versions and one tool message. Asserts for those: mixed-version flag true, minimum
    # turns 2, minimum tool calls 1, and weight version minus minimum weight version equals 1.
    # For the others: not mixed, at most 1 turn, tool calls all null (string prompts). The
    # aggregate mixed-version fraction for step 0 is strictly between 0 and 1.
    import polars as pl

    df = reader.summary(0)
    agentic = df.filter(pl.col("sample_index") % 3 == 0)
    plain = df.filter(pl.col("sample_index") % 3 != 0)
    # dummy dump: every third sample is two-turn with mixed versions + one tool message
    assert agentic["mixed_version"].all() and agentic["turns"].min() == 2
    assert agentic["tool_calls"].min() == 1
    assert not plain["mixed_version"].any() and plain["turns"].max() == 1
    assert plain["tool_calls"].is_null().all()  # string prompts: nothing to count
    assert (agentic["weight_version"].cast(pl.Int64) - agentic["weight_version_min"] == 1).all()

    aggregates = reader.step_aggregates()
    assert 0 < aggregates["mixed_version_frac"][0] < 1


def test_groups_null_rewards_are_not_zero_std(tmp_path):
    # [FUNCTION DOC] Replaces the reader's summary with a stub frame of two groups whose raw
    # rewards are all null, and asserts no group is flagged zero-std (absent data is not a
    # degenerate group).
    # missing train dumps leave raw_reward null for every sample; that is
    # absent data, not a degenerate group
    reader = DumpReader(tmp_path)
    reader.summary = lambda rollout_id, evaluation=False: pl.DataFrame(
        {
            "group_index": [0, 0, 1, 1],
            "raw_reward": pl.Series([None] * 4, dtype=pl.Float64),
            "response_length": [1, 2, 3, 4],
            "truncated": [False] * 4,
        }
    )
    assert not reader.groups(0)["zero_std"].any()


def test_summary_columns_declaration_matches_reality(reader):
    # [FUNCTION DOC] Asserts the reader's declared summary column list equals the columns of the
    # real train summary and of the eval summary, in order.
    # SUMMARY_COLUMNS only shapes the no-sample case, so nothing else would
    # notice it drifting: pin it against the columns a real summary produces
    assert list(DumpReader.SUMMARY_COLUMNS) == reader.summary(0).columns
    assert list(DumpReader.SUMMARY_COLUMNS) == reader.summary(0, evaluation=True).columns


def test_step_with_no_samples_reads_as_empty_not_broken(tmp_path):
    # [FUNCTION DOC] Dumps 3 steps and blanks the samples of the middle step. Asserts step 1's
    # summary has zero rows but the full declared columns and an empty sample index list; groups
    # is empty with the zero-std column; aggregates still list steps [0, 1, 2], the blank step has
    # sample count 0 and a none mean reward, and step 2 is measured normally.
    """A step can be dumped with no samples at all. Its summary must still
    carry the full schema, or every view below it fails on a missing column
    instead of reporting an empty step -- and because step_aggregates() walks
    every step, one such step would take the whole metrics page down with it."""
    dump_dummy_run(tmp_path, steps=3, with_eval=False)
    blank_samples(tmp_path, 1)  # a MIDDLE step, so step_aggregates must walk past it
    reader = DumpReader(tmp_path)

    empty = reader.summary(1)
    assert empty.height == 0
    assert empty.columns == list(DumpReader.SUMMARY_COLUMNS)
    assert empty["sample_index"].to_list() == []  # the trajectories endpoint indexes this

    assert reader.groups(1).height == 0
    assert "zero_std" in reader.groups(1).columns

    aggregates = reader.step_aggregates()
    assert aggregates["rollout_id"].to_list() == [0, 1, 2]
    blank = aggregates.filter(pl.col("rollout_id") == 1).row(0, named=True)
    assert blank["n_samples"] == 0
    assert blank["reward_mean"] is None
    # the steps on either side are still measured normally
    assert aggregates.filter(pl.col("rollout_id") == 2).row(0, named=True)["n_samples"] == reader.summary(2).height


def test_empty_step_survives_the_parquet_cache(tmp_path):
    # [FUNCTION DOC] Dumps 2 steps, blanks step 1, summarises it with a custom cache directory
    # (writes parquet), then a fresh reader over the same cache must serve it from parquet with
    # zero rows and the full declared columns.
    # the all-null frame is written to parquet and read back on the next call
    dump_dummy_run(tmp_path, steps=2, with_eval=False)
    blank_samples(tmp_path, 1)
    assert DumpReader(tmp_path, cache_dir=tmp_path / "c").summary(1).columns == list(DumpReader.SUMMARY_COLUMNS)
    reloaded = DumpReader(tmp_path, cache_dir=tmp_path / "c").summary(1)  # served from the parquet written above
    assert reloaded.height == 0
    assert reloaded.columns == list(DumpReader.SUMMARY_COLUMNS)


def test_trajectory_occurrences_follow_the_full_sample_order(tmp_path):
    # [FUNCTION DOC] Dumps a step where sample index 0 is duplicated. Reads the trajectory sidecar
    # lines and asserts the two index-0 rows carry persisted occurrence numbers [0, 1]. Removes
    # the occurrence-0 row from the sidecar and rewrites it; then trajectory lookup for occurrence
    # 1 finds index 0, and occurrence 0 raises a key error (no conversation recorded).
    """The trajectory sidecar only holds samples that recorded a conversation,
    so file positions cannot number occurrences: when the first of two TITO
    leaves sharing an index has no conversation, the recorded one is
    occurrence 1 in summary/tokens and must be occurrence 1 here too. The
    writer persists that numbering in each row."""
    dump_dummy_run(tmp_path, steps=1, duplicate_first_sample_index=True)
    sidecar = tmp_path / "trajectory" / "0.jsonl"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    index0 = [row for row in rows if int(row["sample_index"]) == 0]
    assert [row["sample_occurrence"] for row in index0] == [0, 1]  # full-list numbering, persisted
    # leaf 0 recorded no conversation: only the occurrence-1 row remains
    rows.remove(index0[0])
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))

    reader = DumpReader(tmp_path)
    assert int(reader.trajectory_messages(0, 0, sample_occurrence=1)["sample_index"]) == 0
    with pytest.raises(KeyError):
        reader.trajectory_messages(0, 0, sample_occurrence=0)  # the conversationless leaf


def test_trajectory_legacy_sidecar_falls_back_to_file_positions(tmp_path):
    # [FUNCTION DOC] Strips the occurrence field from every sidecar row, rewrites the file, and
    # asserts lookup of occurrence 1 for sample index 0 still works (numbered by file position).
    """Sidecars written before the occurrence column keep working, numbered by
    file position (the best available for them)."""
    dump_dummy_run(tmp_path, steps=1, duplicate_first_sample_index=True)
    sidecar = tmp_path / "trajectory" / "0.jsonl"
    rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
    for row in rows:
        row.pop("sample_occurrence")
    sidecar.write_text("".join(json.dumps(row) + "\n" for row in rows))

    reader = DumpReader(tmp_path)
    assert int(reader.trajectory_messages(0, 0, sample_occurrence=1)["sample_index"]) == 0


def test_pre_fix_columnless_cache_is_not_served(tmp_path):
    # [FUNCTION DOC] Plants a columnless parquet for blank step 1 with matching source stamps but
    # a stored summary version of 4. Asserts the summary recomputes with the full declared columns
    # and that the reader's current summary version constant is greater than 4.
    """A cache dir populated before the no-sample schema fix holds a columnless
    parquet for the aborted step, and its mtime stamps still match the dump.
    Only SUMMARY_VERSION tells it apart from a valid cache, so the schema fix
    had to ship with a bump past 4 -- the version that wrote those caches."""
    dump_dummy_run(tmp_path, steps=2, with_eval=False)
    blank_samples(tmp_path, 1)
    reader = DumpReader(tmp_path, cache_dir=tmp_path / "c")
    reader.cache_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([], strict=False).write_parquet(reader.cache_dir / "rollout_1.parquet")
    stale = {**reader._source_stamps(1, evaluation=False), "_summary_version": 4}
    (reader.cache_dir / "rollout_1.sources.json").write_text(json.dumps(stale))

    assert reader.summary(1).columns == list(DumpReader.SUMMARY_COLUMNS)
    assert DumpReader.SUMMARY_VERSION > 4  # regressing the bump re-serves those caches


def test_turns_counts_calls_not_spans():
    # [FUNCTION DOC] A sample with one generation call containing two version spans ("4" then
    # "5") yields versions ["4", "5"] and a turn count of 1.
    # KNOWN ISSUE: re-imports names already imported at module level.
    """A single generation call that spanned a weight update is still one turn."""
    from miles.dashboard.dump_reader import _weight_version_summary
    from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

    sample = Sample(group_index=0, index=0, prompt="p", tokens=[1, 2, 3], response="r", response_length=3, label="l")
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan("4", 0, 2), WeightVersionSpan("5", 2, 3)])
    ]

    assert _weight_version_summary(sample) == (["4", "5"], 1)


def test_turns_counts_a_call_that_carried_no_version():
    # [FUNCTION DOC] A sample with one call with a span "4" and one call with no spans yields
    # versions ["4"] and 2 turns.
    # KNOWN ISSUE: re-imports names already imported at module level.
    """An unstamped call still happened, so it must not vanish from the turn count."""
    from miles.dashboard.dump_reader import _weight_version_summary
    from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

    sample = Sample(group_index=0, index=0, prompt="p", tokens=[1, 2], response="r", response_length=2, label="l")
    sample.weight_versions = [
        WeightVersionsPerCall(spans=[WeightVersionSpan("4", 0, 1)]),
        WeightVersionsPerCall(spans=[]),
    ]

    assert _weight_version_summary(sample) == (["4"], 2)


class TestLegacyWeightVersionSummary:
    # [CLASS DOC] Tests of the weight-version summary helper on legacy (flat list) and current
    # (per-call spans) sample formats.
    def test_reads_versions_from_a_pre_span_dump(self):
        # [FUNCTION DOC] A sample restored from a dictionary whose weight versions are the flat
        # list "3", "4" gives versions ["3", "4"] and 2 turns.
        """A dump that predates spans still reports its versions and turn count."""
        sample = Sample.from_dict({"status": "completed", "weight_versions": ["3", "4"], "tokens": [1, 2]})

        versions, turns = _weight_version_summary(sample)

        assert versions == ["3", "4"]
        assert turns == 2

    def test_prefers_spans_when_the_dump_has_them(self):
        # [FUNCTION DOC] A sample with two calls (spans "3"; "4" and "5") yields versions
        # ["3", "4", "5"] in generation order and 2 turns.
        """A current dump is read from its spans, in generation order."""
        sample = Sample(
            weight_versions=[
                WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 1)]),
                WeightVersionsPerCall(spans=[WeightVersionSpan("4", 1, 2), WeightVersionSpan("5", 2, 3)]),
            ]
        )

        versions, turns = _weight_version_summary(sample)

        assert versions == ["3", "4", "5"]
        assert turns == 2

    def test_reports_nothing_when_the_sample_was_never_stamped(self):
        # [FUNCTION DOC] A default sample with no versions yields an empty version list and none
        # for the turn count (not zero).
        """A sample with no versions at all yields no series rather than a zero."""
        assert _weight_version_summary(Sample()) == ([], None)

    def test_repeated_legacy_versions_are_two_turns_but_not_mixed(self):
        # [FUNCTION DOC] Legacy flat versions "3", "3" give both versions kept and 2 turns.
        # KNOWN ISSUE: the name mentions "not mixed" but only versions and turns are asserted.
        """Two calls that happened to see the same weights are still two turns."""
        sample = Sample.from_dict({"status": "completed", "weight_versions": ["3", "3"], "tokens": [1, 2]})

        assert _weight_version_summary(sample) == (["3", "3"], 2)


class TestCurrentFormatWeightVersionSummary:
    # [CLASS DOC] Tests of weight-version summarisation for current-format samples.
    def test_turns_counts_calls_when_every_call_is_unstamped(self):
        # [FUNCTION DOC] Two calls with no spans yield no versions and 2 turns.
        """Two calls that the engine never stamped report no versions but still count as two turns."""
        sample = Sample(weight_versions=[WeightVersionsPerCall(spans=[]), WeightVersionsPerCall(spans=[])])

        assert _weight_version_summary(sample) == ([], 2)

    def test_summary_does_not_mark_repeated_current_version_as_mixed(self, tmp_path):
        # [FUNCTION DOC] Builds a sample with two calls, each a single span of version "3", and
        # runs the reader's private summary row builder without a train row. Asserts 2 turns, the
        # mixed-version flag false and the weight version "3".
        """Two calls that saw the same weights give two turns without flagging a mixed version."""
        sample = Sample(
            group_index=0,
            index=0,
            tokens=[1, 2],
            response_length=2,
            weight_versions=[
                WeightVersionsPerCall(spans=[WeightVersionSpan("3", 0, 1)]),
                WeightVersionsPerCall(spans=[WeightVersionSpan("3", 1, 2)]),
            ],
        )

        row = DumpReader(tmp_path)._summary_row(sample, None, rollout_id=0, sample_occurrence=0)

        assert row["turns"] == 2
        assert row["mixed_version"] is False
        assert row["weight_version"] == "3"


class TestSummaryCacheVersioning:
    # [CLASS DOC] Tests of summary cache invalidation by schema version.
    def test_summary_invalidates_v2_cache_after_weight_version_schema_change(self, reader):
        # [FUNCTION DOC] Computes the summary, overwrites its parquet with a bogus frame and
        # rewrites the sources file with summary version 2. Asserts the rebuilt summary equals the
        # original, the parquet is rewritten to match, and the stored version is 7.
        # KNOWN ISSUE: the name and docstring say version 5 while the assertion expects 7, and
        # the function re-imports polars at function level.
        """A summary cache stamped with the previous schema version is rebuilt and restamped at version 5."""
        import polars as pl

        expected = reader.summary(0)
        cache_path = reader.cache_dir / "rollout_0.parquet"
        sources_path = reader.cache_dir / "rollout_0.sources.json"
        pl.DataFrame({"sample_index": [-1]}).write_parquet(cache_path)
        sources_path.write_text(json.dumps(json.loads(sources_path.read_text()) | {"_summary_version": 2}))

        rebuilt = reader.summary(0)

        assert rebuilt.equals(expected)
        assert pl.read_parquet(cache_path).equals(expected)
        assert json.loads(sources_path.read_text())["_summary_version"] == 7


def test_pre_span_dump_survives_the_full_reader_pipeline(tmp_path):
    # [FUNCTION DOC] Dumps one step, deletes the derived columns directory, and rewrites every
    # rollout file so each sample's weight versions are the flat list of span versions (the
    # pre-span format). A fresh reader must still summarise: non-empty frame, agentic rows (index
    # divisible by 3) have minimum turns 2 and all mixed-version, and a token payload for the
    # first agentic sample is non-empty.
    # KNOWN ISSUE: re-imports polars at function level.
    """A real dump downgraded to the pre-span format still rebuilds its parquet and summarises."""
    import polars as pl
    import torch

    dump_dummy_run(tmp_path, steps=1, dp_size=2, tp_dup=2, with_eval=False)
    shutil.rmtree(tmp_path / "dashboard_columns")
    for path in (tmp_path / "rollout_data").glob("*.pt"):
        pack = torch.load(path, weights_only=False)
        for data in pack["samples"]:
            data["weight_versions"] = [span["version"] for call in data["weight_versions"] for span in call]
        torch.save(pack, path)

    reader = DumpReader(tmp_path)
    df = reader.summary(0)

    assert df.height > 0
    agentic = df.filter(pl.col("sample_index") % 3 == 0)
    assert agentic["turns"].min() == 2
    assert agentic["mixed_version"].all()
    assert reader.tokens(0, int(agentic["sample_index"][0]))
