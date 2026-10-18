# [MODULE DOC] CPU unit tests of the checkpoint-tracker reader, which answers which iteration a
# checkpoint directory stands at by reading the tracker marker file: returns the integer when the
# file names one, and none when the directory has no tracker file, does not exist, the tracker holds
# a non-numeric word (Megatron writes "release"), or no directory is given at all.
from pathlib import Path

from miles.backends.training_utils.checkpoint.tracker import (
    CHECKPOINT_TRACKER_FILENAME,
    read_checkpoint_tracker_iteration,
)


def _write_tracker(directory: Path, text: str) -> Path:
    # [FUNCTION DOC] HELPER: Creates the directory (with parents if needed), writes the given text
    # into the tracker file inside it, and returns the directory.
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHECKPOINT_TRACKER_FILENAME).write_text(text)
    return directory


class TestReadCheckpointTrackerIteration:
    # [CLASS DOC] Tests of the tracker reader's return value for each directory state.
    def test_a_tracker_answers_the_iteration_it_names(self, tmp_path):
        # [FUNCTION DOC] A tracker containing "50" and a newline yields the integer 50.
        """This is how a caller tells which iteration a checkpoint directory stands at."""
        assert read_checkpoint_tracker_iteration(_write_tracker(tmp_path / "run", "50\n")) == 50

    def test_a_directory_without_a_tracker_holds_no_checkpoint(self, tmp_path):
        # [FUNCTION DOC] An existing empty directory yields none.
        """A --save directory exists from the moment the run starts, long before it holds a checkpoint."""
        (tmp_path / "run").mkdir()

        assert read_checkpoint_tracker_iteration(tmp_path / "run") is None

    def test_a_directory_that_does_not_exist_holds_no_checkpoint(self, tmp_path):
        # [FUNCTION DOC] A path that does not exist yields none (no error).
        """The first run of a job passes a --load path nothing has created yet."""
        assert read_checkpoint_tracker_iteration(tmp_path / "nope") is None

    def test_a_tracker_naming_no_number_holds_no_iteration(self, tmp_path):
        # [FUNCTION DOC] A tracker containing the word "release" yields none.
        """Megatron writes 'release' there, which names no iteration a caller could snapshot beside."""
        assert read_checkpoint_tracker_iteration(_write_tracker(tmp_path / "run", "release\n")) is None

    def test_no_directory_at_all_holds_no_checkpoint(self):
        # [FUNCTION DOC] Passing none as the directory yields none without touching the filesystem.
        """--save is optional, and None must not reach the filesystem."""
        assert read_checkpoint_tracker_iteration(None) is None
