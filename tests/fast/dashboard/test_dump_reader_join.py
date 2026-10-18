# [MODULE DOC]
# Tests for the dump reader's join of rollout dumps with per-rank train dumps. Uses a synthetic
# dump directory produced by the dummy dump writer (several steps, data-parallel shards, tensor
# parallel duplicates, optional eval rollouts) and checks rollout id discovery with freshness
# rules, joined rows, de-duplication, assertion failures on inconsistent dumps, partial coverage,
# and an optional real-data test enabled by an environment variable naming a dump directory.
import os
import time
from collections import defaultdict

import polars as pl
import pytest
import torch
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader, DumpStillWriting, TrainRow
from miles.utils.types import Sample


@pytest.fixture
def run(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture. Writes a dummy dump with 3 steps, data-parallel size 2,
    # tensor-parallel duplication 2 and eval rollouts into the temporary path. Returns a pair of
    # a reader over that directory and the ground-truth description from the writer.
    truth = dump_dummy_run(tmp_path, steps=3, dp_size=2, tp_dup=2, with_eval=True)
    return DumpReader(tmp_path), truth


def test_rollout_ids_split_and_sorted(run):
    # [FUNCTION DOC] Asserts train rollout ids are [0, 1, 2] sorted and eval ids equal the
    # ground truth ([0]).
    reader, truth = run
    ids = reader.rollout_ids()
    assert ids.train == [0, 1, 2]
    assert ids.eval == truth.eval_ids == [0]


def test_fresh_rollout_hidden_until_train_companion_exists(run):
    # [FUNCTION DOC] Saves a just-written rollout file for id 3 (empty samples) and asserts it is
    # not listed as a train id while still fresh and no train companion exists. After touching a
    # train file for id 3 from rank 6 (any rank number counts, since the first global rank may not
    # dump under pipeline parallelism), asserts id 3 is listed.
    reader, _ = run
    fresh = reader.rollout_dir / "3.pt"
    torch.save(dict(rollout_id=3, samples=[]), fresh)  # mtime = now
    assert 3 not in reader.rollout_ids().train

    # any rank number counts: with pp > 1 global rank 0 does not dump
    (reader.train_dir / "3_6.pt").touch()  # train companion written strictly later
    assert 3 in reader.rollout_ids().train


def test_fresh_eval_hidden_until_aged(run):
    # [FUNCTION DOC] A freshly written eval rollout file for id 2 is not listed; after its
    # modification time is set 100 seconds into the past, it is listed.
    reader, _ = run
    fresh = reader.rollout_dir / "eval_2.pt"
    torch.save(dict(rollout_id=2, samples=[]), fresh)
    assert 2 not in reader.rollout_ids().eval

    stamp = time.time() - 100
    os.utime(fresh, (stamp, stamp))
    assert 2 in reader.rollout_ids().eval


def test_load_joined_full_coverage(run):
    # [FUNCTION DOC] For each train id, joins and asserts: sample count equals the truth, train
    # coverage is 1.0, train rows are keyed exactly by (sample index, occurrence 0) of every
    # sample, samples are sample objects and rows are train row objects. Also asserts the status
    # of step 0 samples was restored to the enum (completed or truncated) by the project's own
    # deserialisation.
    reader, truth = run
    for rollout_id in reader.rollout_ids().train:
        joined = reader.load_joined(rollout_id)
        assert len(joined.samples) == truth.n_samples_per_step
        assert joined.train_coverage == 1.0
        assert set(joined.train_rows) == {(s.index, 0) for s in joined.samples}
        assert all(isinstance(s, Sample) for s in joined.samples)
        assert all(isinstance(r, TrainRow) for r in joined.train_rows.values())
    # Sample deserialization went through miles' own from_dict: enums restored.
    statuses = {s.status for s in reader.load_joined(0).samples}
    assert statuses <= {Sample.Status.COMPLETED, Sample.Status.TRUNCATED}


def test_join_row_matches_sample(run):
    # [FUNCTION DOC] For step 1, asserts each joined train row agrees with its sample: response
    # length, total length equals token count, token tensor equal to the sample tokens, rollout
    # log probs close, and log probs, advantages and loss mask all have response length entries.
    reader, _ = run
    joined = reader.load_joined(1)
    sample_of = {(s.index, 0): s for s in joined.samples}
    for key, row in joined.train_rows.items():
        sample = sample_of[key]
        assert row.response_length == sample.response_length
        assert row.total_length == len(sample.tokens)
        assert torch.equal(row.tokens, torch.tensor(sample.tokens))
        assert torch.allclose(row.rollout_log_probs, torch.tensor(sample.rollout_log_probs))
        assert len(row.log_probs) == len(row.advantages) == len(row.loss_mask) == sample.response_length


def test_raw_reward_uses_batch_global_indexing(run):
    # [FUNCTION DOC] Regression test: for every train step, each joined row's raw reward equals
    # its sample's reward, guarding against indexing the batch-global raw reward column by shard
    # row (balanced shuffled partitions would misattribute rewards).
    # Regression test: raw_reward is the one batch-global column
    # (split_train_data_by_dp ships it unpartitioned, "splited at train side");
    # with shuffled balanced partitions, indexing it by shard row silently
    # misattributes rewards across samples.
    reader, _ = run
    for rollout_id in reader.rollout_ids().train:
        joined = reader.load_joined(rollout_id)
        for sample in joined.samples:
            assert joined.train_rows[(sample.index, 0)].raw_reward == sample.reward


def test_rewards_went_through_real_group_normalization(run):
    # [FUNCTION DOC] Groups the shard-local normalised rewards of step 0 by sample group and
    # asserts each group's sum is within 1e-4 of zero (per-group mean centering).
    # The dummy pipeline runs the real GRPO reward post-processing: per-group
    # mean-centering makes shard-local `rewards` sum to ~0 within each group.
    reader, _ = run
    joined = reader.load_joined(0)
    group_of = {(s.index, 0): s.group_index for s in joined.samples}
    group_rewards = defaultdict(list)
    for key, row in joined.train_rows.items():
        group_rewards[group_of[key]].append(row.reward)
    for rewards in group_rewards.values():
        assert abs(sum(rewards)) < 1e-4


def test_tp_duplicates_keep_first_rank(run):
    # [FUNCTION DOC] With 2 data-parallel shards each duplicated on 2 tensor-parallel ranks
    # (shard 0 on ranks 0 and 1, shard 1 on 2 and 3), asserts the joined rows come only from
    # ranks 0 and 2 and each sample's row rank is the lowest rank of its shard.
    reader, truth = run
    joined = reader.load_joined(0)
    # dp_size=2, tp_dup=2: shard 0 lives on ranks {0,1}, shard 1 on ranks {2,3};
    # dedup must keep the lowest rank of each shard.
    assert {row.rank for row in joined.train_rows.values()} == {0, 2}
    for shard_idx, indices in enumerate(truth.shard_indices[0]):
        for index in indices:
            assert joined.train_rows[(index, 0)].rank == shard_idx * 2


def test_duplicate_sample_indices_keep_tito_leaves_distinct(tmp_path):
    # [FUNCTION DOC] Dumps one step where the first sample index appears twice. Asserts coverage
    # 1.0, rows exist for occurrences 0 and 1 of index 0, and each row matches the sample at the
    # same list position (lengths, tokens, raw reward). Also asserts the summary rows for sample
    # index 0 carry occurrence numbers [0, 1].
    dump_dummy_run(tmp_path, steps=1, duplicate_first_sample_index=True)
    reader = DumpReader(tmp_path)

    joined = reader.load_joined(0)
    assert joined.train_coverage == 1.0
    assert (0, 0) in joined.train_rows
    assert (0, 1) in joined.train_rows
    for position, key in enumerate([(0, 0), (0, 1)]):
        sample = joined.samples[position]
        row = joined.train_rows[key]
        assert row.response_length == sample.response_length
        assert row.total_length == len(sample.tokens)
        assert torch.equal(row.tokens, torch.tensor(sample.tokens))
        assert row.raw_reward == sample.reward

    duplicate_rows = reader.summary(0).filter(pl.col("sample_index") == 0)
    assert duplicate_rows["sample_occurrence"].to_list() == [0, 1]


def test_partial_train_coverage_degrades_to_null_rows(tmp_path):
    # [FUNCTION DOC] Dumps one step (no tensor-parallel duplication), then rewrites rank 0's
    # train file dropping the last row from every per-row column except the batch-global raw
    # reward. Asserts the step still loads: the dropped sample has no train row, coverage is
    # strictly between 0 and 1, the summary still lists all samples, and groups is non-empty.
    """A sample can be missing from every train shard (e.g. dropped before
    training dumped its rows). Its train columns must come back null -- the
    step must keep loading, not 500."""
    dump_dummy_run(tmp_path, steps=1, dp_size=2, tp_dup=1)  # tp_dup=1: one shard row per sample
    reader = DumpReader(tmp_path)
    path = reader.train_dir / "0_0.pt"
    pack = torch.load(path, weights_only=False)
    columns = pack["rollout_data"]
    n_rows = len(columns["sample_indices"])
    victim = int(columns["sample_indices"][-1])
    for name, value in columns.items():
        # raw_reward stays batch-global (indexed by rollout position, not row)
        if name != "raw_reward" and hasattr(value, "__len__") and len(value) == n_rows:
            columns[name] = value[:-1]
    torch.save(pack, path)

    joined = reader.load_joined(0)
    assert (victim, 0) not in joined.train_rows
    assert 0 < joined.train_coverage < 1.0
    summary = reader.summary(0)
    assert summary.height == len(joined.samples)  # the uncovered sample is still listed
    assert reader.groups(0).height > 0


def test_train_index_absent_from_rollout_asserts(run):
    # [FUNCTION DOC] Overwrites the first sample index in rank 0's train file with 9999 (not in
    # the rollout) and asserts joining raises an assertion error matching "9999 absent".
    reader, _ = run
    path = reader.train_dir / "0_0.pt"
    pack = torch.load(path, weights_only=False)
    pack["rollout_data"]["sample_indices"][0] = 9999
    torch.save(pack, path)

    with pytest.raises(AssertionError, match="9999 absent"):
        reader.load_joined(0)


def test_inconsistent_tp_duplicate_asserts(run):
    # [FUNCTION DOC] Increments the first response length in rank 1's file (a tensor-parallel
    # duplicate of rank 0) and asserts joining raises an assertion error matching "disagrees".
    reader, _ = run
    path = reader.train_dir / "0_1.pt"  # TP duplicate of rank 0
    pack = torch.load(path, weights_only=False)
    pack["rollout_data"]["response_lengths"][0] += 1
    torch.save(pack, path)

    with pytest.raises(AssertionError, match="disagrees"):
        reader.load_joined(0)


def test_eval_join_has_no_train_rows(run):
    # [FUNCTION DOC] Joining eval step 0 yields an evaluation-flagged result with no train rows
    # and half as many samples as a train step.
    reader, truth = run
    joined = reader.load_joined(0, evaluation=True)
    assert joined.evaluation
    assert joined.train_rows == {}
    assert len(joined.samples) == truth.n_samples_per_step // 2


def test_still_writing_vs_corruption(run):
    # [FUNCTION DOC] Writes garbage bytes to rollout file 9 (fresh modification time). Asserts it
    # is not listed and loading raises the still-writing error. After aging the file by 100
    # seconds, loading raises some error that is not the still-writing error (real corruption).
    reader, _ = run
    path = reader.rollout_dir / "9.pt"
    path.write_bytes(b"garbage")  # fresh mtime: a torch.save in progress
    assert 9 not in reader.rollout_ids().train
    with pytest.raises(DumpStillWriting):
        reader.load_joined(9)

    stamp = time.time() - 100
    os.utime(path, (stamp, stamp))  # old + unloadable = real corruption
    with pytest.raises(Exception) as exc_info:
        reader.load_joined(9)
    assert not isinstance(exc_info.value, DumpStillWriting)


def test_missing_rollout_file_raises(run):
    # [FUNCTION DOC] Joining a rollout id with no file (42) raises a file-not-found error.
    reader, _ = run
    with pytest.raises(FileNotFoundError):
        reader.load_joined(42)


def test_optional_columns_absent(tmp_path):
    # [FUNCTION DOC] Dumps a step without entropy columns; asserts entropy and reference entropy
    # are none on every row while the advantages column still loads.
    dump_dummy_run(tmp_path, steps=1, with_entropy=False, with_eval=False)
    joined = DumpReader(tmp_path).load_joined(0)
    for row in joined.train_rows.values():
        assert row.entropy is None and row.ref_entropy is None
        assert row.advantages is not None  # unaffected columns still load


def test_empty_dump_dir(tmp_path):
    # [FUNCTION DOC] A reader over an empty directory reports empty train and eval id lists.
    ids = DumpReader(tmp_path).rollout_ids()
    assert ids.train == [] and ids.eval == []


@pytest.mark.skipif(
    "MILES_DASHBOARD_REALDATA_DIR" not in os.environ,
    reason="set MILES_DASHBOARD_REALDATA_DIR to a real --dump-details dir",
)
def test_realdata_join():
    # [FUNCTION DOC] Skipped unless the real-data directory environment variable is set. Over a
    # real dump directory, asserts at least one train id exists and for every id: coverage is
    # 1.0, the first row's log probs length equals its response length, and each row's raw reward
    # equals the reward of the sample with the same index and running occurrence number.
    reader = DumpReader(os.environ["MILES_DASHBOARD_REALDATA_DIR"])
    ids = reader.rollout_ids()
    assert ids.train, "no rollout dumps found"
    for rollout_id in ids.train:
        joined = reader.load_joined(rollout_id)
        assert joined.train_coverage == 1.0
        row = next(iter(joined.train_rows.values()))
        assert len(row.log_probs) == row.response_length
        # raw_reward correspondence must hold on real data too
        occurrences = defaultdict(int)
        sample_of = {}
        for sample in joined.samples:
            key = (sample.index, occurrences[sample.index])
            occurrences[sample.index] += 1
            sample_of[key] = sample
        for key, train_row in joined.train_rows.items():
            assert train_row.raw_reward == sample_of[key].reward
