# [MODULE DOC]
# Static naming-convention checks over the ft end-to-end test package, run without executing the
# end-to-end tests. Mode names must be reproduced by a name-computing function from the mode's
# fields. Scenario files (scenario_*.py in the support package) must have no ft segment and no
# double underscore, and their declared test name (a module-level string constant found by a
# line-anchored text search) must equal the file name without the scenario prefix. Entry files
# (test_*.py in the ft package) are named test_<scenario>__<rest>; the scenario must exist, the
# entry must import that scenario's run function, the first segment of the rest must be a known
# kill segment, and an entry that names a mode must declare that mode (a private mode constant) and
# the mode must be registered. Module constants hold the discovered paths and the kill segments.
import re
from pathlib import Path

import pytest

import tests.e2e.ft
from tests.e2e.ft.conftest_ft.modes import KILL_SEGMENT_OF_FT_COMPONENTS, MODES, compute_mode_name

ENTRY_DIR = Path(tests.e2e.ft.__file__).parent
SCENARIO_DIR = ENTRY_DIR / "conftest_ft"

ENTRY_PATHS = sorted(ENTRY_DIR.glob("test_*.py"))
SCENARIO_PATHS = sorted(SCENARIO_DIR.glob("scenario_*.py"))
KILL_SEGMENTS = set(KILL_SEGMENT_OF_FT_COMPONENTS.values())


class TestModeNames:
    # [CLASS DOC] Tests that registered mode names are consistent with the modes' fields.
    @pytest.mark.parametrize("name", sorted(MODES))
    def test_a_mode_name_is_reproduced_by_the_fields_it_claims_to_describe(self, name: str):
        # [FUNCTION DOC] For each registered mode name, asserts that computing the name from the
        # mode gives back the same name.
        """A name that drifts from the fields sends a reader to the wrong topology, and nothing else would catch it."""
        assert compute_mode_name(MODES[name]) == name

    def test_the_modes_that_differ_in_nothing_but_what_they_crash_still_have_distinct_names(self):
        # [FUNCTION DOC] Asserts the train-only mode crashes only train and the mixed mode crashes
        # train and rollout, so the two same-topology modes are distinguishable.
        """The kill segment is the only thing separating two of these, so dropping it would silently merge them."""
        assert MODES["kill_train__dp2_cp2"].ft_components == ("train",)
        assert MODES["kill_train_rollout__dp2_cp2"].ft_components == ("train", "rollout")


class TestScenarioNames:
    # [CLASS DOC] Parametrized over every scenario file; checks file and declared names.
    @pytest.mark.parametrize("path", SCENARIO_PATHS, ids=lambda path: path.stem)
    def test_a_scenario_carries_no_ft_segment(self, path: Path):
        # [FUNCTION DOC] Asserts no underscore-separated segment of the file stem is exactly ft.
        """The whole package is tests/e2e/ft, so an ft segment costs characters and says nothing."""
        assert "ft" not in path.stem.split("_")

    @pytest.mark.parametrize("path", SCENARIO_PATHS, ids=lambda path: path.stem)
    def test_a_scenario_name_holds_no_double_underscore(self, path: Path):
        # [FUNCTION DOC] Asserts the file stem contains no double underscore.
        """Entries split on the first one to recover the scenario, so a scenario containing one breaks that."""
        assert "__" not in path.stem

    @pytest.mark.parametrize("path", SCENARIO_PATHS, ids=lambda path: path.stem)
    def test_the_test_name_is_the_scenario_name_without_its_prefix(self, path: Path):
        # [FUNCTION DOC] Reads the scenario's declared test name; skips the test if none is
        # declared (delegating scenario). Otherwise asserts it equals the stem without the
        # scenario prefix.
        """TEST_NAME reaches dump directories and wandb runs, so a second spelling splits one soak's history in two."""
        if (declared := _declared(path, "TEST_NAME")) is None:
            pytest.skip(f"{path.stem} delegates to another scenario and declares no TEST_NAME")
        assert declared == path.stem.removeprefix("scenario_")


class TestEntryNames:
    # [CLASS DOC] Parametrized over every entry test file; checks naming consistency with scenarios and modes.
    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_names_a_scenario_that_exists(self, path: Path):
        # [FUNCTION DOC] Asserts the scenario part of the entry name is among the test names of the
        # scenario files.
        """The name is the only index from a red CI job back to the code that ran."""
        assert _scenario_of(path) in {_test_name_of(one) for one in SCENARIO_PATHS}

    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_runs_the_scenario_it_is_named_after(self, path: Path):
        # [FUNCTION DOC] Searches the entry's source text for an import of a run function from a
        # scenario module; asserts one exists and that the imported scenario's test name equals the
        # scenario named by the entry's file name.
        """A name pointing at one scenario while importing another is a lie no reader can see through."""
        imported = re.search(r"from tests\.e2e\.ft\.conftest_ft\.(scenario_\w+) import run_ci", path.read_text())
        assert imported, f"{path.name} imports no run_ci"
        assert _test_name_of(SCENARIO_DIR / f"{imported.group(1)}.py") == _scenario_of(path)

    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_says_what_its_run_crashes(self, path: Path):
        # [FUNCTION DOC] Asserts the first double-underscore-separated segment after the scenario
        # is one of the known kill segments.
        """Whether a soak kills trainers, engines or both is the first thing a reader needs and the easiest to lose."""
        assert _rest_of(path).split("__")[0] in KILL_SEGMENTS

    @pytest.mark.parametrize("path", ENTRY_PATHS, ids=lambda path: path.stem)
    def test_an_entry_naming_a_mode_runs_that_mode(self, path: Path):
        # [FUNCTION DOC] If the rest of the entry name is just a kill segment, asserts the entry
        # declares no mode constant. Otherwise asserts the declared mode equals the rest of the
        # name and is a registered mode.
        """Renaming an entry without repointing _MODE would leave it silently running a neighbouring topology."""
        rest = _rest_of(path)
        declared = _declared(path, "_MODE")
        if rest in KILL_SEGMENTS:
            assert declared is None, f"{path.name} names no mode but declares _MODE={declared!r}"
            return
        assert declared == rest
        assert declared in MODES


def _scenario_of(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Returns the scenario part of an entry file name: the stem without the
    # test prefix, cut at the first double underscore.
    return path.stem.removeprefix("test_").partition("__")[0]


def _rest_of(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Returns the part of the entry stem after the first double underscore
    # (with the test prefix removed). Raises an assertion error if there is no double underscore.
    # KNOWN ISSUE: the unpacked scenario variable is unused.
    scenario, separator, rest = path.stem.removeprefix("test_").partition("__")
    assert separator, f"{path.name} carries nothing after its scenario name"
    return rest


def _test_name_of(path: Path) -> str:
    # [FUNCTION DOC] HELPER: Returns the scenario file's declared test name, or if none (or empty)
    # the stem without the scenario prefix.
    return _declared(path, "TEST_NAME") or path.stem.removeprefix("scenario_")


def _declared(path: Path, name: str) -> str | None:
    # [FUNCTION DOC] HELPER: Reads the file as text and searches, line by line anchored at line
    # start, for a string-annotated module constant of the given name assigned a double-quoted
    # literal. Returns the literal's content or none if absent.
    found = re.search(rf'^{name}: str = "([^"]+)"', path.read_text(), re.M)
    return found.group(1) if found else None
