# [MODULE DOC] Unit tests for the private helper that scans a dump directory tree for tensor dump
# files (extension pt) and reports the distinct directories containing them as relative path
# strings. Tests build small directory trees under a temporary path with empty files.
from pathlib import Path

from miles.utils.test_utils.comparisons.dumps import _find_leaf_dump_dirs


class TestFindLeafDumpDirs:
    # [CLASS DOC] Tests of leaf directory discovery: deduplication, sorting, the root case,
    # empty trees and file extension filtering.
    def test_two_pt_files_in_one_leaf_yield_single_entry(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two dump files in one nested leaf directory give a single entry, the
        # relative path with forward slashes.
        """Multiple .pt files sharing one leaf dir dedup to a single relative entry."""
        leaf = tmp_path / "fwd_bwd" / "rollout_0"
        leaf.mkdir(parents=True)
        (leaf / "step_0.pt").touch()
        (leaf / "step_1.pt").touch()

        assert _find_leaf_dump_dirs(tmp_path) == ["fwd_bwd/rollout_0"]

    def test_two_leaves_returned_sorted(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Two sibling leaves created in reverse order are returned sorted by
        # relative path string.
        """Distinct leaf dirs are returned sorted by their relative path string."""
        leaf_b = tmp_path / "leaf_b"
        leaf_a = tmp_path / "leaf_a"
        leaf_b.mkdir()
        leaf_a.mkdir()
        (leaf_b / "x.pt").touch()
        (leaf_a / "y.pt").touch()

        assert _find_leaf_dump_dirs(tmp_path) == ["leaf_a", "leaf_b"]

    def test_pt_file_directly_in_root_yields_dot(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A dump file directly under the root gives the single entry dot.
        """A .pt file directly under root has parent equal to root, reported as '.'."""
        (tmp_path / "step_0.pt").touch()

        assert _find_leaf_dump_dirs(tmp_path) == ["."]

    def test_no_pt_files_yields_empty_list(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] A tree with only an empty subdirectory gives an empty list.
        """A tree with no .pt files produces an empty list."""
        (tmp_path / "sub").mkdir()

        assert _find_leaf_dump_dirs(tmp_path) == []

    def test_non_pt_files_are_ignored(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] Text, pth and json files are not dump files, so the result is empty.
        """Files not matching *.pt (including *.pth) are ignored by the glob."""
        (tmp_path / "notes.txt").touch()
        (tmp_path / "weights.pth").touch()
        (tmp_path / "data.json").touch()

        assert _find_leaf_dump_dirs(tmp_path) == []
