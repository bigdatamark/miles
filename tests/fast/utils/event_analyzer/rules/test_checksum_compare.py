# [MODULE DOC] Unit tests for the shared checksum comparison primitives of the event analyzer
# rules: comparing two flat dictionaries of string values and flattening nested dictionaries and
# lists into flat keys. Pure functions on small literal inputs.
"""Tests for event_analyzer rules/checksum_compare shared primitives."""

from miles.utils.audit_utils.event_analyzer.rules.checksum_compare import compare_flat_dicts, flatten_nested


class TestCompareFlatDicts:
    # [CLASS DOC] Tests for comparing two flat dictionaries given labels for each side; it yields
    # one mismatch item (with key, both values and both labels) per differing key.
    def test_identical_dicts_no_issue(self) -> None:
        # [FUNCTION DOC] Two equal dictionaries yield an empty sequence of mismatches.
        """Identical flat dicts yield no mismatch."""
        assert list(compare_flat_dicts({"w": "a"}, {"w": "a"}, "x", "y")) == []

    def test_differing_value_reported(self) -> None:
        # [FUNCTION DOC] One differing value yields exactly one mismatch carrying the key and
        # the values of side a and side b.
        """A differing value is reported with both labels and values."""
        issues = list(compare_flat_dicts({"w": "a"}, {"w": "b"}, "x", "y"))
        assert len(issues) == 1
        assert issues[0].key == "w"
        assert issues[0].value_a == "a"
        assert issues[0].value_b == "b"

    def test_missing_key_marked(self) -> None:
        # [FUNCTION DOC] A key present only in the first dictionary yields a mismatch whose
        # second value is the marker text "<missing>".
        """A key present only on one side is reported as <missing> on the other."""
        issues = list(compare_flat_dicts({"w": "a"}, {}, "x", "y"))
        assert issues[0].value_b == "<missing>"


class TestFlattenNested:
    # [CLASS DOC] Tests for flattening nested dictionaries and lists into a single-level
    # dictionary whose keys are the prefix and the path, joined by dots for dictionary levels
    # (including non-string keys such as integers) and bracketed indices for list entries.
    def test_nested_dict_flattened(self) -> None:
        # [FUNCTION DOC] A dictionary nested three levels deep (with an integer key) flattens
        # to one dot-separated key under the given prefix.
        """Nested dicts flatten to dot-separated keys."""
        assert flatten_nested({"state": {0: {"x": "h"}}}, prefix="opt") == {"opt.state.0.x": "h"}

    def test_list_indexed(self) -> None:
        # [FUNCTION DOC] A list value flattens to one key per entry with a zero-based index in
        # square brackets appended to the key.
        """List entries flatten to bracket-indexed keys."""
        assert flatten_nested({"p": ["a", "b"]}, prefix="r") == {"r.p[0]": "a", "r.p[1]": "b"}
