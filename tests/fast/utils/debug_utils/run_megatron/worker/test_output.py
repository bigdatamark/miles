# [MODULE DOC] Unit tests for the output dumping code of the run_megatron worker. Three layers are
# covered: computing per-position log probability entries from logits, labels and position ids;
# wrapping those entries into a per-rank payload with parallelism sizes (distributed state mocked);
# and computing then saving the payload as a JSON file in an output directory. Everything runs on
# CPU with random or hand-built logits and uses temporary directories and mocks instead of a real
# distributed environment.
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

from miles.utils.debug_utils.run_megatron.worker.output import (
    _compute_logprob_entries,
    _compute_output_info,
    compute_and_save_output_info,
)


# ---------------------------------------------------------------------------
# TestComputeLogprobEntries
# ---------------------------------------------------------------------------


class TestComputeLogprobEntries:
    # [CLASS DOC] Tests for converting logits of shape (batch, sequence, vocabulary) plus labels and
    # global position ids into a list (per batch row) of lists (per position) of entry
    # dictionaries with keys global position, token id, log probability and validity flag. Labels
    # equal to minus 100 are invalid entries with token id minus 1 and log probability 0.0. Logits
    # that are not three-dimensional or have vocabulary size 1 produce none.
    def test_returns_none_for_2d_logits(self) -> None:
        # [FUNCTION DOC] Two-dimensional logits yield none.
        logits = torch.randn(2, 4)
        labels = torch.tensor([[1, 2, 3, -100]])
        position_ids = torch.arange(4).unsqueeze(0)
        assert _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids) is None

    def test_returns_none_for_scalar_vocab(self) -> None:
        # [FUNCTION DOC] Three-dimensional logits whose last dimension is 1 (a scalar, not a
        # vocabulary distribution) yield none.
        logits = torch.randn(1, 4, 1)
        labels = torch.tensor([[0, 0, 0, 0]])
        position_ids = torch.arange(4).unsqueeze(0)
        assert _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids) is None

    def test_basic_shape(self) -> None:
        # [FUNCTION DOC] Batch of 2 with sequence length 3 gives 2 lists of 3 entries each, even
        # when some labels are ignored.
        batch_size, seq_len, vocab_size = 2, 3, 10
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.tensor([[1, 2, -100], [3, -100, 5]])
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        assert len(result) == batch_size
        for batch_entries in result:
            assert len(batch_entries) == seq_len

    def test_entry_keys(self) -> None:
        # [FUNCTION DOC] Each entry, valid or ignored, has exactly the four expected keys.
        logits = torch.randn(1, 2, 5)
        labels = torch.tensor([[1, -100]])
        position_ids = torch.tensor([[0, 1]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        expected_keys = {"global_position", "token_id", "logprob", "is_valid"}
        for entry in result[0]:
            assert set(entry.keys()) == expected_keys

    def test_valid_entry_values(self) -> None:
        # [FUNCTION DOC] A single position with position id 7 and label 2 yields global position
        # 7, token id 2, validity true and a negative log probability.
        vocab_size = 5
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, 2] = 10.0
        labels = torch.tensor([[2]])
        position_ids = torch.tensor([[7]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        entry = result[0][0]
        assert entry["global_position"] == 7
        assert entry["token_id"] == 2
        assert entry["is_valid"] is True
        assert entry["logprob"] < 0  # log_softmax is always negative

    def test_ignored_entry_values(self) -> None:
        # [FUNCTION DOC] A label of minus 100 at position id 3 yields global position 3, token
        # id minus 1, validity false and log probability exactly 0.0.
        logits = torch.randn(1, 1, 5)
        labels = torch.tensor([[-100]])
        position_ids = torch.tensor([[3]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        entry = result[0][0]
        assert entry["global_position"] == 3
        assert entry["token_id"] == -1
        assert entry["is_valid"] is False
        assert entry["logprob"] == 0.0

    def test_logprob_sums_to_one(self) -> None:
        # [FUNCTION DOC] Exponentiating every returned log probability gives a value in the
        # half-open interval above 0 and up to 1.
        # KNOWN ISSUE: the test name says the probabilities sum to one, but the body only checks
        # that each single probability is in the range above 0 and at most 1; no sum is computed.
        """exp(logprob) for each valid token should be a valid probability."""
        logits = torch.randn(1, 4, 10)
        labels = torch.tensor([[0, 3, 7, 9]])
        position_ids = torch.arange(4).unsqueeze(0)

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        for entry in result[0]:
            prob = torch.exp(torch.tensor(entry["logprob"]))
            assert 0.0 < prob.item() <= 1.0

    def test_perfect_prediction_high_logprob(self) -> None:
        # [FUNCTION DOC] When the logit of the labelled class is 100 and all others are minus
        # 100, the log probability is greater than minus 0.01 (close to zero).
        vocab_size = 5
        logits = torch.full((1, 1, vocab_size), -100.0)
        logits[0, 0, 3] = 100.0
        labels = torch.tensor([[3]])
        position_ids = torch.tensor([[0]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        assert result[0][0]["logprob"] > -0.01

    def test_position_ids_preserved(self) -> None:
        # [FUNCTION DOC] Position ids 10, 20, 30 are copied unchanged to the global position
        # field of the entries, in order, including for an ignored label.
        logits = torch.randn(1, 3, 5)
        labels = torch.tensor([[1, 2, -100]])
        position_ids = torch.tensor([[10, 20, 30]])

        result = _compute_logprob_entries(logits=logits, labels=labels, position_ids=position_ids)
        assert result is not None
        assert [e["global_position"] for e in result[0]] == [10, 20, 30]


# ---------------------------------------------------------------------------
# TestComputeOutputInfo
# ---------------------------------------------------------------------------


_MOCK_DIST = "miles.utils.debug_utils.run_megatron.worker.output.dist"
_MOCK_GET_PARALLEL_STATE = "miles.utils.debug_utils.run_megatron.worker.output.get_parallel_state"


def _patch_distributed(
    tp_size: int = 1,
    cp_size: int = 1,
    pp_size: int = 1,
) -> Any:
    # [FUNCTION DOC] HELPER: Builds a decorator factory for tests. Given tensor, context and
    # pipeline parallel sizes (all default 1), it makes a fake parallel state object with those
    # sizes and returns a decorator; the decorated test runs with the distributed-initialized
    # check patched to report true and the parallel state getter patched to return the fake. The
    # two mock objects are passed to the test as extra positional arguments, which is why tests
    # accept and ignore variadic positional arguments. The wrapper does not preserve the wrapped
    # function's name or metadata.
    """Context manager that mocks dist.is_initialized and parallel-state sizes."""
    parallel_state = SimpleNamespace(
        tp=SimpleNamespace(size=tp_size),
        cp=SimpleNamespace(size=cp_size),
        pp=SimpleNamespace(size=pp_size),
    )

    def _decorator(func: Any) -> Any:
        # [FUNCTION DOC] HELPER: Wraps the given test function with the two patches described in
        # the enclosing helper and returns the wrapper.
        @patch(f"{_MOCK_DIST}.is_initialized", return_value=True)
        @patch(_MOCK_GET_PARALLEL_STATE, return_value=parallel_state)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Forwards all positional (including the injected mocks) and
            # keyword arguments to the wrapped test function and returns its result.
            return func(*args, **kwargs)

        return wrapper

    return _decorator


class TestComputeOutputInfo:
    # [CLASS DOC] Tests for building the per-rank output payload: a dictionary with the rank, the
    # tensor, context and pipeline parallel sizes, and the log probability entries; none when the
    # logits are not usable.
    def test_returns_none_for_invalid_logits(self) -> None:
        # [FUNCTION DOC] Logits with vocabulary size 1 produce none (no distributed patching is
        # needed because the function returns before reading the parallel state).
        logits = torch.randn(1, 4, 1)
        labels = torch.tensor([[0, 0, 0, 0]])
        position_ids = torch.arange(4).unsqueeze(0)

        result = _compute_output_info(logits=logits, labels=labels, position_ids=position_ids, rank=0)
        assert result is None

    @_patch_distributed(tp_size=2, cp_size=4, pp_size=3)
    def test_payload_structure(self, *_mocks: Any) -> None:
        # [FUNCTION DOC] With parallel sizes 2, 4, 3 mocked and rank 5 given, the payload
        # carries rank 5 and those sizes, plus log probability entries for one batch row of three
        # positions.
        logits = torch.randn(1, 3, 10)
        labels = torch.tensor([[1, 2, -100]])
        position_ids = torch.arange(3).unsqueeze(0)

        result = _compute_output_info(logits=logits, labels=labels, position_ids=position_ids, rank=5)
        assert result is not None
        assert result["rank"] == 5
        assert result["tp_size"] == 2
        assert result["cp_size"] == 4
        assert result["pp_size"] == 3
        assert "logprob_entries" in result
        assert len(result["logprob_entries"]) == 1
        assert len(result["logprob_entries"][0]) == 3

    @_patch_distributed()
    def test_defaults_to_size_1(self, *_mocks: Any) -> None:
        # [FUNCTION DOC] With the default mock (all sizes 1) the payload reports size 1 for
        # tensor, context and pipeline parallelism.
        logits = torch.randn(1, 2, 5)
        labels = torch.tensor([[1, 2]])
        position_ids = torch.arange(2).unsqueeze(0)

        result = _compute_output_info(logits=logits, labels=labels, position_ids=position_ids, rank=0)
        assert result is not None
        assert result["tp_size"] == 1
        assert result["cp_size"] == 1
        assert result["pp_size"] == 1


# ---------------------------------------------------------------------------
# TestComputeAndSaveOutputInfo
# ---------------------------------------------------------------------------


class TestComputeAndSaveOutputInfo:
    # [CLASS DOC] Tests for computing the payload and writing it to a rank-numbered JSON file in
    # an output directory. The distributed-initialized check is patched to report false, so the
    # rank is expected to default to 0 and the file name to be the rank 0 file.
    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_writes_json_file(self, _mock: Any, tmp_path: Path) -> None:
        # [FUNCTION DOC] A valid call creates the rank 0 JSON file whose content has rank 0 and
        # a single batch row of three entries.
        logits = torch.randn(1, 3, 10)
        labels = torch.tensor([[1, 2, -100]])
        position_ids = torch.arange(3).unsqueeze(0)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=tmp_path,
        )

        output_file = tmp_path / "rank_0.json"
        assert output_file.exists()

        payload = json.loads(output_file.read_text())
        assert payload["rank"] == 0
        assert "logprob_entries" in payload
        assert len(payload["logprob_entries"]) == 1
        assert len(payload["logprob_entries"][0]) == 3

    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_skips_invalid_logits(self, _mock: Any, tmp_path: Path) -> None:
        # [FUNCTION DOC] Logits with vocabulary size 1 produce no file at all.
        logits = torch.randn(1, 4, 1)
        labels = torch.tensor([[0, 0, 0, 0]])
        position_ids = torch.arange(4).unsqueeze(0)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=tmp_path,
        )

        assert not (tmp_path / "rank_0.json").exists()

    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_creates_output_dir(self, _mock: Any, tmp_path: Path) -> None:
        # [FUNCTION DOC] A nested output directory that does not exist yet is created, and the
        # rank 0 file is written inside it.
        output_dir = tmp_path / "nested" / "dir"
        logits = torch.randn(1, 2, 5)
        labels = torch.tensor([[1, 2]])
        position_ids = torch.arange(2).unsqueeze(0)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=output_dir,
        )

        assert output_dir.exists()
        assert (output_dir / "rank_0.json").exists()

    @patch(f"{_MOCK_DIST}.is_initialized", return_value=False)
    def test_json_is_valid(self, _mock: Any, tmp_path: Path) -> None:
        # [FUNCTION DOC] A batch of 2 rows with 4 positions is saved as parseable JSON with two
        # rows of four entries, each entry having exactly the four expected keys.
        logits = torch.randn(2, 4, 8)
        labels = torch.tensor([[1, 2, -100, 3], [4, -100, 5, 6]])
        position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

        compute_and_save_output_info(
            logits=logits,
            labels=labels,
            position_ids=position_ids,
            output_dir=tmp_path,
        )

        payload = json.loads((tmp_path / "rank_0.json").read_text())
        assert len(payload["logprob_entries"]) == 2
        assert len(payload["logprob_entries"][0]) == 4
        assert len(payload["logprob_entries"][1]) == 4

        for batch_entries in payload["logprob_entries"]:
            for entry in batch_entries:
                assert set(entry.keys()) == {"global_position", "token_id", "logprob", "is_valid"}
