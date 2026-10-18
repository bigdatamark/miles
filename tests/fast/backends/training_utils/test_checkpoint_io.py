# [MODULE DOC] CPU tests for the checkpoint directory writer: filesystem errors raised while
# creating the directory propagate unchanged, and a process that crashes mid-overwrite (simulated
# with a forked child that exits abruptly, either before or after writing partial files) leaves
# no stale files and no metadata file, after which a retry succeeds and leaves only the checkpoint
# directory in the parent.
"""Checkpoint filesystem errors propagate to the trainer cell."""

import multiprocessing
import os
from pathlib import Path

import pytest

from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir


@pytest.mark.parametrize("error", [OSError("disk full"), RuntimeError("directory creation failed")])
def test_directory_errors_propagate(error, tmp_path, monkeypatch):
    # [FUNCTION DOC] Test (parametrized over an OS error "disk full" and a runtime error): makes the
    # path-creation method raise the given error, then writing a checkpoint directory raises that
    # same error object (same type, message and identity).
    def make_dir(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Replacement directory-creation method; raises the parametrized
        # error regardless of arguments.
        raise error

    monkeypatch.setattr(Path, "mkdir", make_dir)
    with pytest.raises(type(error), match=str(error)) as caught:
        write_checkpoint_dir(tmp_path / "checkpoint", lambda _: None)
    assert caught.value is error


@pytest.mark.parametrize("crash_after_write", [True, False])
def test_crashed_overwrite_leaves_no_metadata(tmp_path, crash_after_write):
    # [FUNCTION DOC] Test (parametrized: crash after writing a partial file or before writing
    # anything): writes a first checkpoint (file "old", metadata step 1). A forked child then
    # overwrites with metadata step 2 but its writer optionally writes a partial file and exits the
    # process immediately with code 73. Asserts the child exit code is 73 (10 second join timeout),
    # the old file is gone and no metadata file exists (the overwrite cleared the directory first
    # and never committed). A retry then succeeds: the new file has the new content, metadata
    # exists, and the temp directory contains only the checkpoint directory (no leftovers).
    checkpoint = tmp_path / "checkpoint"
    write_checkpoint_dir(checkpoint, lambda directory: (directory / "old").write_text("old"), metadata={"step": 1})

    def overwrite_and_crash():
        # [FUNCTION DOC] HELPER: Child process body; overwrites the checkpoint with a writer that
        # crashes the process (never returns normally).
        def write_shards(directory):
            # [FUNCTION DOC] HELPER: Writer callback; if requested writes a partial file into the
            # directory, then terminates the process at once with exit code 73 (no cleanup).
            if crash_after_write:
                (directory / "value").write_text("partial")
            os._exit(73)

        write_checkpoint_dir(checkpoint, write_shards, metadata={"step": 2})

    child = multiprocessing.get_context("fork").Process(target=overwrite_and_crash)
    child.start()
    child.join(timeout=10)
    assert child.exitcode == 73
    assert not (checkpoint / "old").exists()
    assert not (checkpoint / "META.json").exists()

    write_checkpoint_dir(checkpoint, lambda directory: (directory / "value").write_text("retry"), metadata={"step": 2})
    assert (checkpoint / "value").read_text() == "retry"
    assert (checkpoint / "META.json").exists()
    assert list(tmp_path.iterdir()) == [checkpoint]
