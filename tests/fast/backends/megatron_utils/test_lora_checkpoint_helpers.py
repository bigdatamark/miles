# [MODULE DOC]
# Tests for LoRA-related helpers of the Megatron checkpoint module: the path-based detector for
# Megatron-format checkpoints (marker file or iteration directory name pattern) and the routing
# of the save-with-LoRA function between the adapter save and the regular checkpoint save. The
# routing tests patch every collaborator with mocks, so no GPU or distributed setup is needed.
"""Unit tests for LoRA-related helpers in miles.backends.megatron_utils.checkpoint.

Covers pure path-detection functions and the LoRA branch routing in
save_checkpoint_with_lora / load_checkpoint — the latter using mocks to avoid
GPU / distributed requirements.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.checkpoint import _is_megatron_checkpoint, save_checkpoint_with_lora

# ---------------------------------------------------------------------------
# _is_megatron_checkpoint
# ---------------------------------------------------------------------------


class TestIsMegatronCheckpoint:
    # [CLASS DOC] Tests for the Megatron-checkpoint path detector using temporary directories.
    def test_has_latest_file(self, tmp_path):
        # [FUNCTION DOC] A directory containing the latest-checkpointed-iteration tracker file is
        # detected as a Megatron checkpoint.
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("100")
        assert _is_megatron_checkpoint(tmp_path) is True

    def test_iter_dir_name(self, tmp_path):
        # [FUNCTION DOC] An (empty) directory named iter_ plus seven digits is detected as a
        # Megatron checkpoint by name alone.
        iter_dir = tmp_path / "iter_0000100"
        iter_dir.mkdir()
        assert _is_megatron_checkpoint(iter_dir) is True

    def test_regular_dir(self, tmp_path):
        # [FUNCTION DOC] An empty ordinary directory is not a Megatron checkpoint.
        assert _is_megatron_checkpoint(tmp_path) is False

    def test_hf_checkpoint_dir(self, tmp_path):
        # [FUNCTION DOC] A directory with a HuggingFace config file and a safetensors weight file
        # is not a Megatron checkpoint.
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("")
        assert _is_megatron_checkpoint(tmp_path) is False

    @pytest.mark.parametrize(
        "name",
        [
            "iter_0000001",
            "iter_0000000",
            "iter_9999999",
        ],
    )
    def test_valid_iter_patterns(self, tmp_path, name):
        # [FUNCTION DOC] Parametrized over iteration directory names with exactly seven digits
        # (including all zeros and all nines). Each created directory is detected as Megatron.
        d = tmp_path / name
        d.mkdir()
        assert _is_megatron_checkpoint(d) is True

    @pytest.mark.parametrize(
        "name",
        [
            "iter_123",  # too short
            "iter_00000001",  # too long
            "iteration_0000001",
            "checkpoint",
        ],
    )
    def test_invalid_iter_patterns(self, tmp_path, name):
        # [FUNCTION DOC] Parametrized over near-miss names: too few digits, too many digits, a
        # different prefix, and an unrelated name. Each created directory is not detected.
        d = tmp_path / name
        d.mkdir()
        assert _is_megatron_checkpoint(d) is False


# ---------------------------------------------------------------------------
# save_checkpoint_with_lora — branch routing
# ---------------------------------------------------------------------------


class TestSaveCheckpointWithLoRA:
    # [CLASS DOC] Tests for branch routing in the save-with-LoRA function; the arguments getter,
    # the LoRA detector and both save backends are patched with mocks.
    @patch("miles.backends.megatron_utils.checkpoint.get_args")
    @patch("miles.backends.megatron_utils.checkpoint.save_lora_checkpoint")
    @patch("miles.backends.megatron_utils.checkpoint.is_lora_model", return_value=True)
    def test_lora_model_saves_adapter(self, mock_is_lora, mock_save_lora, mock_get_args, tmp_path):
        # [FUNCTION DOC] With the model detected as LoRA and a save directory in the arguments,
        # saves iteration 42 with a publisher mock. Asserts the LoRA save ran once, received the
        # publisher as a keyword argument, and that its save directory mentions "adapter".
        # KNOWN ISSUE: the last assertion looks the directory up in the keyword arguments with a
        # fallback to the third positional argument; if neither holds the directory the fallback
        # is an empty string, which would fail, but it is convoluted and does not pin which one.
        mock_get_args.return_value = Namespace(save=str(tmp_path))
        model = [MagicMock()]

        publisher = MagicMock()
        save_checkpoint_with_lora(42, model, MagicMock(), MagicMock(), publisher=publisher)

        mock_save_lora.assert_called_once()
        call_args = mock_save_lora.call_args
        assert call_args.kwargs["publisher"] is publisher
        assert "adapter" in call_args[1].get("save_dir", call_args[0][2] if len(call_args[0]) > 2 else "")

    @patch("miles.backends.megatron_utils.checkpoint.get_args")
    @patch("miles.backends.megatron_utils.checkpoint.save_checkpoint")
    @patch("miles.backends.megatron_utils.checkpoint.is_lora_model", return_value=False)
    def test_non_lora_model_saves_regular(self, mock_is_lora, mock_save_ckpt, mock_get_args, tmp_path):
        # [FUNCTION DOC] With the model not detected as LoRA, saves iteration 42 and asserts the
        # regular checkpoint save was called exactly once.
        mock_get_args.return_value = Namespace(save=str(tmp_path))
        model = [MagicMock()]

        save_checkpoint_with_lora(42, model, MagicMock(), MagicMock())

        mock_save_ckpt.assert_called_once()
