# [MODULE DOC]
# Tests for the atomic text-file writer (write to a unique temporary file in the same directory,
# set permissions, then rename over the target). Covers: no leftover temporary files, unique
# temporary names for repeated writes, cleanup of the temporary when the rename fails, resulting
# permission bits 0644, and preservation of the previous file when replacement fails. A module
# level helper simulates a failing rename.
import os
import stat
import tempfile
from pathlib import Path

import pytest

from miles.utils.file_utils import atomic_write_text


class TestAtomicWriteText:
    # [CLASS DOC] Tests of the atomic text writer.
    def test_leaves_no_temporary_file_behind(self, tmp_path):
        # [FUNCTION DOC] After a successful write, assert the directory contains only the target
        # file name.
        """The write goes through a rename, and a leftover temp file would confuse a reader listing the directory."""
        path = tmp_path / "state.json"

        atomic_write_text(path, "{}")

        assert [entry.name for entry in tmp_path.iterdir()] == ["state.json"]

    def test_two_writers_of_the_same_pid_do_not_share_a_temporary(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Wrap the temporary-file creation function to record every temporary name,
        # write the same target twice, and assert two distinct temporary names were used and the
        # final content is the second write.
        """The launcher and the wrapper sit in different pid namespaces over one directory."""
        path = tmp_path / "state.json"
        names = set()
        real_mkstemp = tempfile.mkstemp

        def recording_mkstemp(**kwargs):
            # [FUNCTION DOC] HELPER: Call the real temporary-file creator with the same keyword
            # arguments, remember the returned name in the shared set, and return the handle and
            # name unchanged.
            handle, name = real_mkstemp(**kwargs)
            names.add(name)
            return handle, name

        monkeypatch.setattr("miles.utils.file_utils.tempfile.mkstemp", recording_mkstemp)
        atomic_write_text(path, "first")
        atomic_write_text(path, "second")

        assert len(names) == 2
        assert path.read_text() == "second"

    def test_removes_the_temporary_when_the_write_fails(self, tmp_path, monkeypatch):
        # [FUNCTION DOC] Make the rename step raise a runtime error, attempt a write (swallowing
        # the error), and assert the directory is empty afterwards (temporary removed).
        """A crashed write that leaves debris behind turns one failure into a directory nobody can read."""
        path = tmp_path / "state.json"
        monkeypatch.setattr(os, "replace", _raising)

        try:
            atomic_write_text(path, "{}")
        except RuntimeError:
            pass

        assert list(tmp_path.iterdir()) == []

    def test_publishes_a_file_readable_by_other_users(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Write via the keyword-argument call form and assert the published file's
        # permission bits are exactly octal 644.
        """The published file is readable across user IDs on shared storage."""
        path = tmp_path / "state.json"

        atomic_write_text(path=path, text="{}")

        assert stat.S_IMODE(path.stat().st_mode) == 0o644

    def test_failed_replacement_preserves_the_previous_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Create a target holding "old", make the rename fail with a runtime error
        # ("no rename today"), and assert the write raises it, the target still holds "old", and
        # the directory holds only the target (no temporary).
        """A failed replacement preserves the published file and removes its temporary."""
        path = tmp_path / "state.json"
        path.write_text("old")
        monkeypatch.setattr(os, "replace", _raising)

        with pytest.raises(RuntimeError, match="no rename today"):
            atomic_write_text(path=path, text="new")

        assert path.read_text() == "old"
        assert list(tmp_path.iterdir()) == [path]


def _raising(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Stand-in for the rename function; accepts any arguments and always
    # raises a runtime error with the message "no rename today".
    raise RuntimeError("no rename today")
