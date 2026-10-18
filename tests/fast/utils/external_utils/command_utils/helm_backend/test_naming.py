# [MODULE DOC] Tests for the helm backend's naming and run file helpers: the platform service
# account name (refused when the worker has no platform access), the run directory layout under
# the shared root (runs directory, state subdirectory with orchestrator state file names, one
# unique launch record file per launch), finding the latest launch's state file by the newest
# launch token recorded in the launches directory (not by file write time), launch token minting
# (time ordered, microsecond resolution, fixed width parts, a random tail only to keep tokens
# apart), and the superseded marker path that sits beside a state file (one per generation,
# identical for a path and its text spelling).
import json
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.naming import (
    RunFiles,
    _orchestrator_state_path,
    platform_account_name,
)
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.workers.types import PlatformAccess


def _write(path, status: OrchestratorStatus, *, exit_code: int | None = None) -> None:
    # [FUNCTION DOC] HELPER: Writes an orchestrator state with the given status and optional exit
    # code (default none) to the path using the state model's write method.
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _record(run_directory, launch_token: str, state_file) -> None:
    # [FUNCTION DOC] HELPER: Creates a launch record json file named for the launch token in the
    # launches subdirectory of the run directory (creating the directory), holding a mapping with
    # the state file path as text. Side effect: file write.
    path = Path(run_directory) / "launches" / f"launch-{launch_token}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"state_file": str(state_file)}))


def _state_file(tmp_path):
    # [FUNCTION DOC] HELPER: Returns the orchestrator state file path under tmp_path for a fixed
    # launch token (no file is created).
    # KNOWN ISSUE: this helper is never used in this module (dead code).
    return _orchestrator_state_path(tmp_path, "260101-000000-000001")


class TestPlatformAccountName:
    # [CLASS DOC] Tests of the platform account name function.
    def test_platform_account_name_refuses_no_platform_access(self) -> None:
        # [FUNCTION DOC] Asking for an account for the none access level raises an assertion
        # error saying the worker never reaches the platform.
        """A worker without platform access must not receive a platform service account."""
        with pytest.raises(AssertionError, match="never reaches the platform"):
            platform_account_name(release="miles-run-example-train", access=PlatformAccess.NONE)


class TestRunDir:
    # [CLASS DOC] Tests of run directory path helpers (pure path computations, no I/O).
    def test_places_a_run_under_the_shared_root(self):
        # [FUNCTION DOC] The run directory ends with the shared root, the runs directory name and
        # the run id.
        """Every pod resolves the same run directory from the shared storage mount and the run id."""
        assert str(RunFiles.run_dir(shared_root="/cluster-storage/miles_data", run_id="260101-000000-000")).endswith(
            "/cluster-storage/miles_data/miles-runs/260101-000000-000"
        )

    def test_keeps_the_state_file_in_a_state_subdirectory(self):
        # [FUNCTION DOC] The state file for a run directory and token is the run directory, then
        # the state subdirectory, then orchestrator, a hyphen, the token and a state suffix.
        """Grouping the machine-written state keeps it out of the way of a run's own outputs."""
        path = _orchestrator_state_path("/runs/abc", "abc123")

        assert path.as_posix() == "/runs/abc/state/orchestrator-abc123.state"

    def test_gives_every_launch_its_own_record_file(self):
        # [FUNCTION DOC] Two new record file calls for one run directory give different paths that
        # both sit in the launches subdirectory.
        """Two launches of one run must not overwrite each other's record of what they launched."""
        first = RunFiles.new_record_file(run_directory="/runs/abc")
        second = RunFiles.new_record_file(run_directory="/runs/abc")

        assert first.parent.as_posix() == "/runs/abc/launches"
        assert first != second


class TestLatestExitFile:
    # [CLASS DOC] Tests of the latest state file lookup over the launches directory records.
    def test_names_no_file_before_a_launch_has_recorded_one(self, tmp_path):
        # [FUNCTION DOC] An empty run directory gives none.
        """A run directory a launch has only just created holds no verdict to collect."""
        assert RunFiles.latest_state_file(run_directory=tmp_path) is None

    def test_names_the_newest_launch_s_state_file_before_it_is_written(self, tmp_path):
        # [FUNCTION DOC] Two launch records (older and newer token) whose state files do not exist
        # yet: the lookup returns the newer launch's state file path.
        """A generation whose pods never came up must not hand the previous generation's verdict over."""
        _record(tmp_path, "260101-000100-000002", _orchestrator_state_path(tmp_path, "260101-000100-000002"))
        pending = _orchestrator_state_path(tmp_path, "260101-000200-000001")
        _record(tmp_path, "260101-000200-000001", pending)

        assert RunFiles.latest_state_file(run_directory=tmp_path) == pending

    def test_picks_the_newest_launch_rather_than_the_newest_write(self, tmp_path):
        # [FUNCTION DOC] The earlier launch's state file is written last (exited with 1) after the
        # later launch's (exited with 0); the lookup still returns the later launch's file because
        # it orders by launch token.
        """An earlier launch torn down after a later one started writes last, and its verdict is not the run's."""
        later = _orchestrator_state_path(tmp_path, "260101-000200-000001")
        earlier = _orchestrator_state_path(tmp_path, "260101-000100-000002")
        _write(later, OrchestratorStatus.EXITED, exit_code=0)
        _write(earlier, OrchestratorStatus.EXITED, exit_code=1)
        _record(tmp_path, "260101-000100-000002", earlier)
        _record(tmp_path, "260101-000200-000001", later)

        assert RunFiles.latest_state_file(run_directory=tmp_path) == later


def _mint_token(monkeypatch: pytest.MonkeyPatch, *, when: datetime, tail: int) -> str:
    # [FUNCTION DOC] HELPER: Fakes the naming module's clock to return the given time and its
    # random generator to return the given tail from any range, then returns a freshly minted
    # launch token. Side effect: monkeypatching of the naming module.
    monkeypatch.setattr(naming, "datetime", SimpleNamespace(now=lambda: when))
    monkeypatch.setattr(naming.random, "Random", lambda: SimpleNamespace(randint=lambda low, high: tail))
    return naming._new_launch_token()


class TestNewLaunchToken:
    # [CLASS DOC] Tests of launch token minting.
    def test_two_launches_of_one_second_order_by_the_time_they_were_minted(self, monkeypatch):
        # [FUNCTION DOC] A token minted at 1 microsecond with the largest tail sorts before one
        # minted at 2 microseconds with tail 0.
        """Everything that picks the newest launch sorts these names, and the random tail ordered them by chance."""
        earlier = _mint_token(monkeypatch, when=datetime(2026, 1, 1, 0, 0, 0, 1), tail=999999)
        later = _mint_token(monkeypatch, when=datetime(2026, 1, 1, 0, 0, 0, 2), tail=0)

        assert earlier < later

    def test_a_token_carries_the_microsecond_it_was_minted_at(self, monkeypatch):
        # [FUNCTION DOC] Time 2026-01-01 00:00:00 plus 123 microseconds and tail 7 give the token
        # of year-month-day (two digit year), a time part, 6 digit microseconds and a 6 digit tail.
        """Two launches of one run are apart by microseconds, which is the resolution the name has to keep."""
        token = _mint_token(monkeypatch, when=datetime(2026, 1, 1, 0, 0, 0, 123), tail=7)

        assert token == "260101-000000-000123-000007"

    def test_two_tokens_minted_in_one_microsecond_still_differ(self, monkeypatch):
        # [FUNCTION DOC] The same time with tails 1 and 2 yields different tokens.
        """The random tail no longer decides the order, but it is still what keeps two names apart."""
        when = datetime(2026, 1, 1, 0, 0, 0, 5)

        assert _mint_token(monkeypatch, when=when, tail=1) != _mint_token(monkeypatch, when=when, tail=2)

    def test_every_part_of_a_real_token_is_fixed_width(self):
        # [FUNCTION DOC] An unfaked token fully matches four hyphen separated groups of six digits.
        """A part that can be shorter sorts before a longer one whatever time it names."""
        assert re.fullmatch(r"\d{6}-\d{6}-\d{6}-\d{6}", naming._new_launch_token())

    def test_a_real_token_never_names_a_time_before_the_one_minted_earlier(self):
        # [FUNCTION DOC] Two unfaked tokens minted in sequence: the clock prefix (the first 20
        # characters) of the first is at most that of the second.
        """The clock part is what the newest-launch lookup reads, and it has to be non-decreasing."""
        stamp_length = len("260101-000000-000000")

        first = naming._new_launch_token()
        second = naming._new_launch_token()

        assert first[:stamp_length] <= second[:stamp_length]


class TestSupersededMarker:
    # [CLASS DOC] Tests of the superseded marker path function (pure path computation).
    def test_the_marker_sits_beside_the_state_file_it_supersedes(self):
        # [FUNCTION DOC] The marker path is the state file path with a superseded suffix appended.
        """Both the launcher that writes it and the orchestrator that reads it know only that path."""
        marker = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-x.state")

        assert marker.as_posix() == "/runs/abc/state/orchestrator-x.state.superseded"

    def test_every_generation_has_a_marker_of_its_own(self):
        # [FUNCTION DOC] Different state files give different marker paths.
        """A run is relaunched many times, and one shared marker would defuse every generation at once."""
        first = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-1.state")
        second = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-2.state")

        assert first != second

    def test_a_path_and_the_string_spelling_of_it_name_one_marker(self):
        # [FUNCTION DOC] A path object and its text spelling give equal marker paths.
        """The launcher holds a Path and the wrapper is given a string, and they have to meet."""
        as_text = RunFiles.superseded_marker(state_file="/runs/abc/state/orchestrator-x.state")

        assert RunFiles.superseded_marker(state_file=Path("/runs/abc/state/orchestrator-x.state")) == as_text
