# [MODULE DOC] Unit tests for the top-k prediction printer of the run_megatron worker: decoding a
# token id to text, reading rank and world size, printing the top predicted tokens for each
# position of one rank's logits (captured stdout is inspected), loading the tokenizer and choosing
# the padding token id, and printing rank after rank with barriers in between. The tokenizer and
# distributed module are mocks; logits are random CPU tensors.
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.utils.debug_utils.run_megatron.worker.top_k_print import (
    _decode_token,
    _get_dist_info,
    _print_top_predictions_all_ranks,
    _print_top_predictions_for_rank,
    print_top_k,
)


class TestDecodeToken:
    # [CLASS DOC] Tests for decoding one token id: with a tokenizer its decode method is called
    # with a one-element list; without one a placeholder "t" followed by the id is returned.
    def test_with_tokenizer(self) -> None:
        # [FUNCTION DOC] The mock tokenizer's decode is called once with a list holding the id
        # and its return value is passed through.
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "hello"
        result = _decode_token(mock_tok, token_id=42)
        assert result == "hello"
        mock_tok.decode.assert_called_once_with([42])

    def test_without_tokenizer(self) -> None:
        # [FUNCTION DOC] With no tokenizer, token id 42 becomes the string "t42".
        result = _decode_token(None, token_id=42)
        assert result == "t42"


class TestGetDistInfo:
    # [CLASS DOC] Tests for reading (rank, world size) from the mocked distributed module,
    # falling back to (0, 1) when it is not initialized.
    @patch("miles.utils.debug_utils.run_megatron.worker.top_k_print.dist")
    def test_not_initialized(self, mock_dist: MagicMock) -> None:
        # [FUNCTION DOC] When not initialized the result is rank 0 and world size 1.
        mock_dist.is_initialized.return_value = False
        rank, world_size = _get_dist_info()
        assert rank == 0
        assert world_size == 1

    @patch("miles.utils.debug_utils.run_megatron.worker.top_k_print.dist")
    def test_initialized(self, mock_dist: MagicMock) -> None:
        # [FUNCTION DOC] When initialized the values come from the module: rank 2, world size 8.
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 2
        mock_dist.get_world_size.return_value = 8
        rank, world_size = _get_dist_info()
        assert rank == 2
        assert world_size == 8


class TestPrintTopPredictionsForRank:
    # [CLASS DOC] Tests for the per-rank printer, which for every batch row and every
    # non-padding position prints one line starting with "pos[" showing the input token, an arrow
    # and the top-k predicted tokens with probabilities formatted with three decimals. Output
    # begins with a header naming the rank, and with several batch rows also a header per batch.
    def test_output_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        # [FUNCTION DOC] Three positions with top k 3 print a header containing "Rank 0" and
        # exactly three position lines; each position line contains an arrow and a probability
        # in parentheses with three decimal digits.
        logits = torch.randn(1, 3, 10)  # batch=1, seq=3, vocab=10
        input_ids = torch.tensor([[1, 2, 3]])
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "x"

        _print_top_predictions_for_rank(
            logits=logits,
            input_ids=input_ids,
            top_k=3,
            tokenizer=mock_tok,
            rank=0,
        )

        output = capsys.readouterr().out
        assert "Rank 0" in output
        assert output.count("pos[") == 3
        for line in output.strip().splitlines():
            if "pos[" in line:
                assert "->" in line
                assert re.search(r"\(\d+\.\d{3}\)", line)

    def test_pad_token_skipped(self, capsys: pytest.CaptureFixture[str]) -> None:
        # [FUNCTION DOC] Positions whose input id equals the given pad token id (99) are not
        # printed: of four positions only two lines appear, and the tokenizer decode is called
        # six times (two printed positions times one input token plus two top-k tokens).
        logits = torch.randn(1, 4, 10)
        input_ids = torch.tensor([[1, 99, 2, 99]])  # 99 is pad
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "x"

        _print_top_predictions_for_rank(
            logits=logits,
            input_ids=input_ids,
            top_k=2,
            tokenizer=mock_tok,
            rank=0,
            pad_token_id=99,
        )

        output = capsys.readouterr().out
        assert output.count("pos[") == 2
        # decode called for non-pad positions only:
        # 2 non-pad positions × (1 input token + 2 top-k tokens) = 6 calls
        assert mock_tok.decode.call_count == 6

    def test_batch_size_gt1(self, capsys: pytest.CaptureFixture[str]) -> None:
        # [FUNCTION DOC] Two batch rows of two positions print the word "Batch" twice (one header
        # per row), four position lines, and twelve decode calls (two rows times two positions
        # times one input plus two top-k tokens).
        logits = torch.randn(2, 2, 10)
        input_ids = torch.tensor([[1, 2], [3, 4]])
        mock_tok = MagicMock()
        mock_tok.decode.return_value = "x"

        _print_top_predictions_for_rank(
            logits=logits,
            input_ids=input_ids,
            top_k=2,
            tokenizer=mock_tok,
            rank=0,
        )

        output = capsys.readouterr().out
        assert output.count("Batch") == 2
        assert output.count("pos[") == 4
        # 2 batches × 2 positions × (1 input + 2 topk) = 12 decode calls
        assert mock_tok.decode.call_count == 12


_TOP_K_MODULE = "miles.utils.debug_utils.run_megatron.worker.top_k_print"


class TestPrintTopK:
    # [CLASS DOC] Tests for the public entry point, which loads a tokenizer from a path (via the
    # process utilities' loader, patched at its home module) and delegates to the all-ranks printer
    # (patched), passing the padding token id as a keyword argument.
    @patch(f"{_TOP_K_MODULE}._print_top_predictions_all_ranks")
    @patch("miles.utils.processing_utils.load_tokenizer")
    def test_loads_tokenizer_and_calls_print_all_ranks(
        self,
        mock_load_tok: MagicMock,
        mock_print_all: MagicMock,
    ) -> None:
        # [FUNCTION DOC] With a tokenizer whose pad token id is 5, the loader is called once,
        # the all-ranks printer is called once, and it receives pad token id 5.
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = 5
        mock_load_tok.return_value = mock_tokenizer

        logits = torch.randn(1, 2, 10)
        input_ids = torch.tensor([[1, 2]])

        print_top_k(
            logits=logits,
            input_ids=input_ids,
            top_k=3,
            tokenizer_path=Path("/fake/model"),
        )

        mock_load_tok.assert_called_once()
        mock_print_all.assert_called_once()
        call_kwargs = mock_print_all.call_args[1]
        assert call_kwargs["pad_token_id"] == 5

    @patch(f"{_TOP_K_MODULE}._print_top_predictions_all_ranks")
    @patch("miles.utils.processing_utils.load_tokenizer")
    def test_pad_token_id_fallback_to_eos(
        self,
        mock_load_tok: MagicMock,
        mock_print_all: MagicMock,
    ) -> None:
        # [FUNCTION DOC] When the tokenizer has no pad token id (none), the end-of-sequence
        # token id (2) is passed as the pad token id instead.
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 2
        mock_load_tok.return_value = mock_tokenizer

        print_top_k(
            logits=torch.randn(1, 2, 10),
            input_ids=torch.tensor([[1, 2]]),
            top_k=3,
            tokenizer_path=Path("/fake/model"),
        )

        call_kwargs = mock_print_all.call_args[1]
        assert call_kwargs["pad_token_id"] == 2


class TestPrintTopPredictionsAllRanks:
    # [CLASS DOC] Tests for the sequential printer: every process loops over ranks in order, only
    # the process whose rank matches the loop index prints, and a barrier follows each iteration
    # plus one final barrier. Barrier, rank lookup and the per-rank printer are mocked.
    @patch(f"{_TOP_K_MODULE}._maybe_barrier")
    @patch(f"{_TOP_K_MODULE}._get_dist_info", return_value=(0, 2))
    @patch(f"{_TOP_K_MODULE}._print_top_predictions_for_rank")
    def test_sequential_rank_printing(
        self,
        mock_print_rank: MagicMock,
        mock_dist_info: MagicMock,
        mock_barrier: MagicMock,
    ) -> None:
        # [FUNCTION DOC] With current rank 0 and world size 2, the per-rank printer is called
        # exactly once, with rank 0 as keyword argument.
        """Current rank=0, world_size=2 → prints only for rank 0."""
        _print_top_predictions_all_ranks(
            logits=torch.randn(1, 2, 10),
            input_ids=torch.tensor([[1, 2]]),
            top_k=3,
            tokenizer=MagicMock(),
        )

        assert mock_print_rank.call_count == 1
        call_kwargs = mock_print_rank.call_args[1]
        assert call_kwargs["rank"] == 0

    @patch(f"{_TOP_K_MODULE}._maybe_barrier")
    @patch(f"{_TOP_K_MODULE}._get_dist_info", return_value=(0, 1))
    @patch(f"{_TOP_K_MODULE}._print_top_predictions_for_rank")
    def test_single_rank_barrier_count(
        self,
        mock_print_rank: MagicMock,
        mock_dist_info: MagicMock,
        mock_barrier: MagicMock,
    ) -> None:
        # [FUNCTION DOC] With world size 1 the barrier is called twice: once inside the single
        # loop iteration and once at the end.
        """World_size=1 → barrier called world_size+1 times (loop + final)."""
        _print_top_predictions_all_ranks(
            logits=torch.randn(1, 2, 10),
            input_ids=torch.tensor([[1, 2]]),
            top_k=3,
            tokenizer=MagicMock(),
        )

        # loop: 1 iteration × 1 barrier + 1 final barrier = 2
        assert mock_barrier.call_count == 2
