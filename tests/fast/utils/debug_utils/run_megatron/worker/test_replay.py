# [MODULE DOC] Unit tests for the routing replay support of the run_megatron worker (recording the
# mixture-of-experts routing decisions to disk and loading them back for replay). Covered: slicing
# a tensor evenly over tensor parallel ranks for sequence parallel regions, switching the global
# routing replay manager into record or replay mode before the model is built, saving the replay
# data from rank 0, loading it with validation of the file count, and the per-rank slicing applied
# on load (sequence parallel and context parallel zigzag). The global routing replay manager and
# the parallel rank lookup are replaced by mocks or simple namespaces, so no GPU or distributed
# environment is needed; saved files are written to temporary directories.
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tests.fast.utils.debug_utils.run_megatron.conftest import make_script_args

from miles.utils.debug_utils.run_megatron.worker.replay import (
    _load_replay,
    _ParallelRanks,
    _sp_slice,
    load_replay_data,
    save_replay_data,
    setup_replay_before_model,
)


class TestSpSlice:
    # [CLASS DOC] Tests for the helper that returns the contiguous equal-size chunk of a tensor
    # belonging to one tensor parallel rank (first dimension split by tensor parallel size).
    def test_slices_evenly(self) -> None:
        # [FUNCTION DOC] Eight elements, size 2, rank 0 give the first four elements.
        tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        result = _sp_slice(tensor, tp_size=2, tp_rank=0)
        assert result.tolist() == [1, 2, 3, 4]

    def test_second_rank(self) -> None:
        # [FUNCTION DOC] Rank 1 of 2 gets the last four of eight elements.
        tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        result = _sp_slice(tensor, tp_size=2, tp_rank=1)
        assert result.tolist() == [5, 6, 7, 8]

    def test_not_divisible_raises(self) -> None:
        # [FUNCTION DOC] Five elements cannot be split over 2 ranks: an assertion error whose
        # message mentions "not divisible" is raised.
        tensor = torch.tensor([1, 2, 3, 4, 5])
        with pytest.raises(AssertionError, match="not divisible"):
            _sp_slice(tensor, tp_size=2, tp_rank=0)

    def test_single_rank_full(self) -> None:
        # [FUNCTION DOC] With size 1 the whole tensor is returned.
        tensor = torch.tensor([1, 2, 3, 4])
        result = _sp_slice(tensor, tp_size=1, tp_rank=0)
        assert result.tolist() == [1, 2, 3, 4]


class TestSetupReplayBeforeModel:
    # [CLASS DOC] Tests that the pre-model setup toggles the (mocked) global routing replay
    # manager: a dump path enables it in the record stage, a load path enables it in the
    # replay-forward stage, and neither leaves it untouched. Each test starts with the manager
    # disabled and in the fallthrough stage.
    def test_dump_enables_record(self) -> None:
        # [FUNCTION DOC] With only a dump path configured the manager becomes enabled and its
        # stage becomes record.
        script = make_script_args(routing_replay_dump_path=Path("/dump"))
        with patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager") as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is True
            assert mock_mgr.stage == "record"

    def test_load_enables_replay_forward(self) -> None:
        # [FUNCTION DOC] With only a load path configured the manager becomes enabled and its
        # stage becomes replay_forward.
        script = make_script_args(routing_replay_load_path=Path("/load"))
        with patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager") as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is True
            assert mock_mgr.stage == "replay_forward"

    def test_neither_noop(self) -> None:
        # [FUNCTION DOC] With neither path configured the manager stays disabled in the
        # fallthrough stage.
        script = make_script_args()
        with patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager") as mock_mgr:
            mock_mgr.enabled = False
            mock_mgr.stage = "fallthrough"
            setup_replay_before_model(script)
            assert mock_mgr.enabled is False
            assert mock_mgr.stage == "fallthrough"


class TestSaveReplayData:
    # [CLASS DOC] Tests for saving the recorded routing indices of the mocked manager to the dump
    # directory, using the manager's file name prefixed by the rank. Only rank 0 may save, and
    # saving with no recorded entries is an error.
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_saves_file(self, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] One replay with one recorded tensor results in exactly one saved file
        # in the dump directory, whose name contains "rank0".
        mock_mgr.filename = "routing_replay.pt"
        replay = SimpleNamespace(top_indices_list=[torch.tensor([1, 2])])
        mock_mgr.replays = [replay]

        script = make_script_args(routing_replay_dump_path=tmp_path)
        save_replay_data(script, rank=0)

        saved_files = list(tmp_path.glob("*.pt"))
        assert len(saved_files) == 1
        assert "rank0" in saved_files[0].name

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_saved_content_roundtrips(self, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] The saved file holds a list with one element per replay, each a list of
        # that replay's recorded tensors; loading it back (without the weights-only restriction)
        # returns the original two tensors with equal values.
        mock_mgr.filename = "routing_replay.pt"
        original = [torch.tensor([10, 20]), torch.tensor([30, 40])]
        replay = SimpleNamespace(top_indices_list=original)
        mock_mgr.replays = [replay]

        script = make_script_args(routing_replay_dump_path=tmp_path)
        save_replay_data(script, rank=0)

        saved_file = list(tmp_path.glob("*.pt"))[0]
        loaded = torch.load(saved_file, weights_only=False)
        assert len(loaded) == 1
        assert len(loaded[0]) == 2
        assert loaded[0][0].tolist() == [10, 20]
        assert loaded[0][1].tolist() == [30, 40]

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_noop_when_no_path(self, mock_mgr: MagicMock) -> None:
        # [FUNCTION DOC] Without a dump path the call returns silently and writes nothing; the
        # test has no explicit assertion (it passes if nothing is raised).
        script = make_script_args()
        save_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_asserts_rank_zero(self, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] Calling with rank 1 while a dump path is set raises an assertion error.
        mock_mgr.filename = "routing_replay.pt"
        replay = SimpleNamespace(top_indices_list=[torch.tensor([1, 2])])
        mock_mgr.replays = [replay]

        script = make_script_args(routing_replay_dump_path=tmp_path)
        with pytest.raises(AssertionError):
            save_replay_data(script, rank=1)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_empty_entries_asserts(self, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] A replay with no recorded tensors makes saving raise an assertion error.
        mock_mgr.filename = "routing_replay.pt"
        replay = SimpleNamespace(top_indices_list=[])
        mock_mgr.replays = [replay]

        script = make_script_args(routing_replay_dump_path=tmp_path)
        with pytest.raises(AssertionError):
            save_replay_data(script, rank=0)


class TestLoadReplayData:
    # [CLASS DOC] Tests for locating and loading the replay file from the load directory: no-op
    # without a load path, a value error unless exactly one file with the manager's file name
    # suffix is found, and population of the manager's replays on success.
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_noop_when_no_path(self, mock_mgr: MagicMock) -> None:
        # [FUNCTION DOC] Without a load path the call returns silently; nothing is asserted
        # beyond that no exception occurs.
        script = make_script_args()
        load_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_no_file_raises(self, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] An empty load directory raises a value error saying exactly 1 replay
        # file was expected.
        mock_mgr.filename = "routing_replay.pt"
        script = make_script_args(routing_replay_load_path=tmp_path)
        with pytest.raises(ValueError, match="Expected exactly 1 replay file"):
            load_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_multiple_files_raises(self, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two empty files whose names end with the manager's file name (so the
        # match is by suffix) raise the same value error as the empty directory.
        mock_mgr.filename = "routing_replay.pt"
        (tmp_path / "a_routing_replay.pt").touch()
        (tmp_path / "b_routing_replay.pt").touch()

        script = make_script_args(routing_replay_load_path=tmp_path)
        with pytest.raises(ValueError, match="Expected exactly 1 replay file"):
            load_replay_data(script, rank=0)

    @patch("miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks")
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_loads_data(
        self,
        mock_mgr: MagicMock,
        mock_ranks: MagicMock,
        tmp_path: Path,
    ) -> None:
        # [FUNCTION DOC] With all parallel sizes 1 and no sequence parallel region, a saved file
        # holding one replay with two tensors is loaded into the mocked manager's single replay:
        # its recorded list gets two elements and its forward and backward indices stay 0.
        mock_ranks.return_value = _ParallelRanks(cp_size=1, cp_rank=0, tp_size=1, tp_rank=0)
        mock_mgr.filename = "routing_replay.pt"
        mock_mgr.if_sp_region = False

        replay_obj = SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0)
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.tensor([10, 20]), torch.tensor([30, 40])]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        script = make_script_args(routing_replay_load_path=tmp_path)
        load_replay_data(script, rank=0)

        assert len(replay_obj.top_indices_list) == 2
        assert replay_obj.forward_index == 0
        assert replay_obj.backward_index == 0


class TestLoadReplaySpSlicing:
    # [CLASS DOC] Tests of the sequence parallel slicing in the file loader: when both the
    # sequence parallel flag is on and the manager says it is a sequence parallel region, each
    # loaded tensor is reduced to the tensor parallel rank's chunk.
    """Test _load_replay with SP (sequence parallel) slicing."""

    @patch("miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks")
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_sp_slices_tensors(self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] Tensor parallel size 2 and rank 0: a saved tensor of 0 to 7 is loaded
        # as 0 to 3.
        mock_ranks.return_value = _ParallelRanks(cp_size=1, cp_rank=0, tp_size=2, tp_rank=0)
        mock_mgr.if_sp_region = True

        replay_obj = SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0)
        mock_mgr.replays = [replay_obj]

        # 8 elements, tp_size=2, tp_rank=0 → first 4
        saved_data = [[torch.arange(8)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        _load_replay(save_path, rank=0, sequence_parallel=True)

        assert len(replay_obj.top_indices_list) == 1
        assert replay_obj.top_indices_list[0].tolist() == [0, 1, 2, 3]

    @patch("miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks")
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_sp_second_rank(self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] Tensor parallel rank 1 of 2: the same tensor loads as 4 to 7.
        mock_ranks.return_value = _ParallelRanks(cp_size=1, cp_rank=0, tp_size=2, tp_rank=1)
        mock_mgr.if_sp_region = True

        replay_obj = SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0)
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.arange(8)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        _load_replay(save_path, rank=1, sequence_parallel=True)

        assert replay_obj.top_indices_list[0].tolist() == [4, 5, 6, 7]

    @patch("miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks")
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_sp_disabled_no_slice(self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] With the sequence parallel flag false, the tensor is loaded whole even
        # though the manager reports a sequence parallel region and tensor parallel size is 2.
        mock_ranks.return_value = _ParallelRanks(cp_size=1, cp_rank=0, tp_size=2, tp_rank=0)
        mock_mgr.if_sp_region = True

        replay_obj = SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0)
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.arange(8)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        # sequence_parallel=False → no sp slice even if if_sp_region=True
        _load_replay(save_path, rank=0, sequence_parallel=False)

        assert replay_obj.top_indices_list[0].tolist() == list(range(8))


class TestLoadReplayCpSlicing:
    # [CLASS DOC] Tests of the context parallel slicing in the file loader: with context parallel
    # size above 1 each loaded tensor is passed through the natural-to-zigzag slice function of
    # the context parallel module (looked up at call time, hence patched at its home module); with
    # size 1 no slicing happens.
    """Test _load_replay with CP (context parallel) zigzag slicing."""

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    @patch("miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks")
    def test_cp_calls_zigzag_slice(self, mock_ranks: MagicMock, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] Context parallel size 2, rank 0, with the zigzag function patched to
        # return a fixed tensor (0, 7, 1, 6): asserts it is called exactly once with keyword
        # arguments dim 0, size 2 and rank 0, and that the loaded tensor is its return value.
        mock_ranks.return_value = _ParallelRanks(cp_size=2, cp_rank=0, tp_size=1, tp_rank=0)
        mock_mgr.if_sp_region = False

        replay_obj = SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0)
        mock_mgr.replays = [replay_obj]

        original_tensor = torch.arange(8)
        saved_data = [[original_tensor]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        mock_zigzag = MagicMock(return_value=torch.tensor([0, 7, 1, 6]))

        with patch(
            "miles.backends.training_utils.data.context_parallel.natural_to_zigzag_slice",
            mock_zigzag,
        ):
            _load_replay(save_path, rank=0, sequence_parallel=False)

        mock_zigzag.assert_called_once()
        call_kwargs = mock_zigzag.call_args[1]
        assert call_kwargs["dim"] == 0
        assert call_kwargs["cp_size"] == 2
        assert call_kwargs["cp_rank"] == 0
        assert replay_obj.top_indices_list[0].tolist() == [0, 7, 1, 6]

    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    @patch("miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks")
    def test_cp1_no_zigzag(self, mock_ranks: MagicMock, mock_mgr: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] With context parallel size 1 the saved tensor 0 to 3 is loaded
        # unchanged.
        mock_ranks.return_value = _ParallelRanks(cp_size=1, cp_rank=0, tp_size=1, tp_rank=0)
        mock_mgr.if_sp_region = False

        replay_obj = SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0)
        mock_mgr.replays = [replay_obj]

        saved_data = [[torch.arange(4)]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        _load_replay(save_path, rank=0, sequence_parallel=False)
        assert replay_obj.top_indices_list[0].tolist() == [0, 1, 2, 3]


class TestLoadReplayMismatch:
    # [CLASS DOC] Tests for validation that the number of replays in the file equals the number
    # of replays in the model's manager.
    @patch("miles.utils.debug_utils.run_megatron.worker.replay._get_parallel_ranks")
    @patch("miles.utils.debug_utils.run_megatron.worker.replay.routing_replay_manager")
    def test_replay_count_mismatch_raises(self, mock_mgr: MagicMock, mock_ranks: MagicMock, tmp_path: Path) -> None:
        # [FUNCTION DOC] A file with 1 replay loaded into a manager with 2 replays raises a
        # value error whose message says the file has 1 replays but the model expects 2.
        mock_ranks.return_value = _ParallelRanks(cp_size=1, cp_rank=0, tp_size=1, tp_rank=0)
        mock_mgr.if_sp_region = False
        mock_mgr.replays = [
            SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0),
            SimpleNamespace(top_indices_list=[], forward_index=0, backward_index=0),
        ]

        # Save only 1 replay but model expects 2
        saved_data = [[torch.tensor([1, 2])]]
        save_path = tmp_path / "rank0_routing_replay.pt"
        torch.save(saved_data, save_path)

        with pytest.raises(ValueError, match="has 1 replays but model expects 2"):
            _load_replay(save_path, rank=0, sequence_parallel=False)
