# [MODULE DOC] Snapshot tests for "self-executing" launchers: example scripts that hand-build
# their whole command line and submit a Ray job when imported or when a command function is called.
# Each case imports a launcher with a frozen environment (validation skipped through an
# environment flag) and a shell recorder in place of real execution, optionally calls a named
# entrypoint with keyword arguments, and compares the recorded commands with a golden file under
# tests/snapshots/launch_scripts/self_executing/<script>/<case name>.txt. Cases cover the p2p
# weight-transfer example for each model profile in both p2p and broadcast modes (as head node
# rank zero), plus an import-only case for the formal-math minimal example. The p2p prepare
# command is deliberately uncovered because it works on a hardcoded absolute models directory.
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.fast.launch_scripts.py_harness import (
    format_recording,
    freeze_environment,
    import_launch_script,
    install_shell_recorder,
    iter_self_executing_launchers,
)
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, assert_matches_snapshot

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "self_executing"

_P2P = "examples/infra_features/p2p_weight_transfer/run.py"
_FORMAL_MATH = "examples/experimental/formal_math/single_round/run_minimal.py"


@dataclass(frozen=True)
class LauncherCase:
    # [CLASS DOC] Immutable description of one snapshot case: the launcher path relative to the
    # repo root, a case name used for the golden file, an optional entrypoint function name to
    # call after import (none means import only), and keyword arguments for that call
    # (defaults to an empty mapping).
    rel: str
    name: str
    entrypoint: str | None = None
    kwargs: dict[str, object] = field(default_factory=dict)


_P2P_PROFILES = (
    "GLM-4.5-Air",
    "GLM-4.7-Flash",
    "GLM-5",
    "GLM-5_20layer",
    "GLM-5_4layer",
    "Kimi-K2-Instruct",
    "Qwen3-235B-A22B-Instruct-2507",
    "Qwen3-30B-A3B",
    "Qwen3-4B",
)

_CASES = [
    LauncherCase(
        rel=_P2P,
        name=f"run/{profile}/{mode}",
        entrypoint="cmd_run",
        kwargs={"model_name": profile, "mode": mode, "node_rank": 0, "head_ip": "10.0.0.1"},
    )
    for profile in _P2P_PROFILES
    for mode in ("p2p", "broadcast")
] + [LauncherCase(rel=_FORMAL_MATH, name="import")]

_ENTRYPOINTS_THE_HARNESS_CANNOT_SANDBOX = {(_P2P, "cmd_prepare")}


@pytest.fixture(params=_CASES, ids=[f"{case.rel}::{case.name}" for case in _CASES])
def recorded(request, monkeypatch, tmp_path):
    # [FUNCTION DOC] HELPER: Parametrized fixture over all launcher cases. Freezes the
    # environment, sets the validation-skip flag, installs the shell recorder rooted at the
    # sandbox, imports the launcher module, and if the case names an entrypoint calls it with
    # the case's keyword arguments. Returns a tuple of the case, the recording and the sandbox.
    case = request.param
    freeze_environment(monkeypatch)
    monkeypatch.setenv("SKIP_VALIDATION", "1")
    recording = install_shell_recorder(monkeypatch, sandbox=tmp_path)
    module = import_launch_script(REPO_ROOT / case.rel)
    if case.entrypoint is not None:
        getattr(module, case.entrypoint)(**case.kwargs)
    return case, recording, tmp_path


class TestEverySelfExecutingLauncher:
    # [CLASS DOC] Per-case checks run for each launcher case via the shared fixture.
    def test_commands_match_snapshot(self, recorded):
        # [FUNCTION DOC] Compares the sandbox-relative formatted recording with the golden file
        # for the case.
        """These launchers build their whole command line by hand, so only a snapshot pins it."""
        case, recording, sandbox = recorded
        snapshot = _SNAPSHOT_DIR / case.rel / f"{case.name}.txt"

        assert_matches_snapshot(snapshot, format_recording(recording, sandbox=sandbox), f"{case.rel}::{case.name}")

    def test_reruns_produce_identical_recordings(self, recorded, monkeypatch, tmp_path):
        # [FUNCTION DOC] Repeats the setup of the fixture a second time in the same test (freeze,
        # skip-validation flag, new recorder, import, optional entrypoint call) and asserts the
        # command lists of both runs are equal, proving process-specific values such as the pid
        # are frozen. Note the module is imported again via the harness importer.
        """These launchers embed their own pid, so a snapshot is only stable if the harness freezes it."""
        case, recording, _ = recorded
        freeze_environment(monkeypatch)
        monkeypatch.setenv("SKIP_VALIDATION", "1")
        again = install_shell_recorder(monkeypatch, sandbox=tmp_path)
        module = import_launch_script(REPO_ROOT / case.rel)
        if case.entrypoint is not None:
            getattr(module, case.entrypoint)(**case.kwargs)

        assert again.commands == recording.commands

    def test_the_launcher_submits_a_ray_job(self, recorded):
        # [FUNCTION DOC] Asserts at least one recorded command contains the Ray job submit phrase.
        """A launcher that stops reaching `ray job submit` is broken, whatever else it records."""
        _, recording, _ = recorded

        assert [command for command in recording.commands if "ray job submit" in command]


class TestDiscovery:
    # [CLASS DOC] Consistency checks between discovery, the case list and golden files.
    def test_every_self_executing_launcher_has_at_least_one_case(self):
        # [FUNCTION DOC] The set of repo-relative launcher paths found by behavioral discovery
        # must equal the set of script paths used by the cases.
        """Discovery is by behaviour, not by path, so a new hand-rolled launcher shows up here."""
        discovered = {path.relative_to(REPO_ROOT).as_posix() for path in iter_self_executing_launchers()}

        assert discovered == {case.rel for case in _CASES}

    def test_every_case_has_a_golden_and_vice_versa(self):
        # [FUNCTION DOC] Expected golden paths (script/case name.txt) must equal the text files
        # found recursively in the snapshot directory.
        """A dropped case leaves an orphan golden, and a new case would otherwise never be recorded."""
        expected = {f"{case.rel}/{case.name}.txt" for case in _CASES}
        recorded = {path.relative_to(_SNAPSHOT_DIR).as_posix() for path in _SNAPSHOT_DIR.rglob("*.txt")}

        assert expected == recorded

    def test_the_uncovered_entrypoint_is_named_and_still_uncoverable(self):
        # [FUNCTION DOC] Imports the p2p launcher; asserts the set of (script, entrypoint) pairs
        # for its run and prepare commands, minus the pairs covered by cases, equals the declared
        # unsandboxable set (prepare only), and that the launcher's source text still contains
        # the hardcoded quoted root models directory path.
        """cmd_prepare rewrites a checkout under a hardcoded /root/models, which no fixture can redirect."""
        module = import_launch_script(REPO_ROOT / _P2P)

        assert {(_P2P, name) for name in ("cmd_run", "cmd_prepare")} - {
            (case.rel, case.entrypoint) for case in _CASES
        } == _ENTRYPOINTS_THE_HARNESS_CANNOT_SANDBOX
        assert '"/root/models"' in Path(module.__file__).read_text()
