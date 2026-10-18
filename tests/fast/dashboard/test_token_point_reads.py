# [MODULE DOC]
# Tests for the dump reader's token point reads, which serve one sample's token view from a
# parquet column mirror and memory-mapped train shards instead of loading whole rollout files:
# the mirror's content, avoidance of the full rollout load, rebuilding of a missing or stale
# mirror, the unknown-sample error, and equality of lazily read train row fields with the fully
# joined row. Uses a one-step dummy dump.
"""Token-view point reads: parquet mirror + mmap'd train shards."""

import polars as pl
import pytest
import torch
from tests.fast.dashboard.dummy_dump import dump_dummy_run

from miles.dashboard.dump_reader import DumpReader


@pytest.fixture
def dumped(tmp_path):
    # [FUNCTION DOC] HELPER: Fixture. Writes a one-step dummy dump into the temporary path and
    # returns the path.
    dump_dummy_run(tmp_path, steps=1)
    return tmp_path


def test_save_writes_the_parquet_mirror(dumped):
    # [FUNCTION DOC] The dump wrote a parquet mirror for step 0. Its sample index column equals
    # (sorted) the indices of the joined samples, and for the first sample the row's tokens and
    # response length equal the sample's.
    frame = pl.read_parquet(dumped / "dashboard_columns" / "rollout_0.parquet")
    reader = DumpReader(dumped)
    joined = reader.joined(0)
    assert sorted(frame["sample_index"]) == sorted(s.index for s in joined.samples)
    sample = joined.samples[0]
    row = frame.filter(pl.col("sample_index") == sample.index).row(0, named=True)
    assert row["tokens"] == list(sample.tokens)
    assert row["response_length"] == sample.response_length


def test_tokens_does_not_touch_the_rollout_pt(dumped, monkeypatch):
    # [FUNCTION DOC] Replaces the tensor load function with a guard that asserts the path does
    # not belong to the rollout data directory, then requests an 8-token window; it succeeds with
    # positive total length and 8 ids, proving the point read avoids the full rollout load.
    reader = DumpReader(dumped)
    real_load = torch.load

    def guarded(path, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Fails with an assertion error if the loaded path contains the
        # rollout data directory name; otherwise delegates to the real load with the same arguments.
        assert "rollout_data" not in str(path), f"point read full-loaded {path}"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", guarded)
    payload = reader.tokens(0, 0, start=0, end=8)
    assert payload["total_len"] > 0 and len(payload["token_ids"]) == 8


def test_missing_mirror_is_rebuilt(dumped):
    # [FUNCTION DOC] After deleting the parquet mirror, a 4-token read succeeds with ids and the
    # mirror file exists again.
    mirror = dumped / "dashboard_columns" / "rollout_0.parquet"
    mirror.unlink()
    reader = DumpReader(dumped)
    payload = reader.tokens(0, 0, start=0, end=4)
    assert mirror.exists() and payload["token_ids"]


def test_stale_schema_is_rebuilt(dumped):
    # [FUNCTION DOC] Overwrites the mirror with a frame having only a sample index column. A
    # 4-token read still succeeds (mirror rebuilt) and the mirror schema now contains tokens.
    mirror = dumped / "dashboard_columns" / "rollout_0.parquet"
    pl.DataFrame({"sample_index": [0]}).write_parquet(mirror)
    payload = DumpReader(dumped).tokens(0, 0, start=0, end=4)
    assert payload["token_ids"]
    assert "tokens" in pl.read_parquet_schema(mirror)


def test_unknown_sample_raises(dumped):
    # [FUNCTION DOC] A tokens read for sample index 10000 raises a key error.
    with pytest.raises(KeyError):
        DumpReader(dumped).tokens(0, 10_000)


def test_train_row_values_match_joined(dumped):
    # [FUNCTION DOC] Builds a lazy train row through the reader's private helpers for the first
    # sample and compares with the fully joined row: for log probs, reference log probs and
    # advantages, absence agrees and present tensors are exactly equal.
    reader = DumpReader(dumped)
    joined = reader.joined(0)
    sample_index = joined.samples[0].index
    rollout_columns = reader._rollout_columns(0, sample_index, sample_occurrence=0, evaluation=False)
    lazy = reader._train_row_lazy(0, sample_index, rollout_columns)
    full = joined.train_rows[(sample_index, 0)]
    for field in ("log_probs", "ref_log_probs", "advantages"):
        lazy_v, full_v = getattr(lazy, field), getattr(full, field)
        assert (lazy_v is None) == (full_v is None)
        if lazy_v is not None:
            assert torch.equal(lazy_v, full_v)
