# [MODULE DOC] CPU tests guarding CI discovery coverage. They assert that tracked test files live
# under a CI-scanned root or the manual-tests directory (with a shrinking allowlist of three
# top-level orphan files, and a staleness check on that allowlist); that every registration
# names a suite that the suite inventory lists for its backend; and that example tests under the
# fast directory reach the CI plan as enabled CPU registrations in the first CPU stage. Tracked
# files come from git, not a filesystem walk, because CI clones other repositories into the
# workspace. Registered as a CPU test.
"""A test file that CI never opens is documentation, not a test.

pytest's testpaths is ./tests and ci_register scans four roots under it, so a
test file parked next to the code it tests runs only when a human remembers to.
Fifteen of them had accumulated that way -- the openenv example alone carried
ten files and a hundred offline tests that cost two seconds and needed no SDK --
and #2545 is what the arrangement costs: a regression test for the
file-descriptor leak that stopped a 16-node run, added to a file no CI job would
ever collect.

Being under tests/ is not enough: a file at tests/ top level is collected by a
local ``pytest`` but by no CI job, because ci_register scans only its four roots.

Hence the rule these tests hold: a test file lives under one of the roots
ci_register scans, or under tests/manual/ when no CI job should run it, and a
test that cannot run somewhere says so itself with a module-level skip (see the
hud suites' ``importorskip``). A skip is reported by the runner and starts
working the day its dependency lands; a file outside the roots is invisible
either way.

``_KNOWN_ORPHANS`` is the escape hatch, and it is built to shrink: a third test
fails once the file an entry names has moved, so a stale exception cannot linger.
"""

import subprocess
from pathlib import Path, PurePosixPath

import pytest
from tests.ci.ci_register import _DISCOVERY_ROOTS, HWBackend, collect_tests, discover_ci_files, register_cpu_ci
from tests.ci.run_suite import CI_SUITES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[3]

# Tests that no CI job should run, kept for a human to start by hand.
_MANUAL_ROOT = "tests/manual"

# Files the rule spares for now. Entries exist to be deleted; adding one is a
# decision to defend in review, not a formality.
_KNOWN_ORPHANS: set[str] = {
    # Found at tests/ top level when the rule widened from "under tests/" to
    # "under a CI root"; each needs a home stage (or tests/manual/) decided.
    "tests/test_attention_output_gate_tp.py",
    "tests/test_chunked_gae.py",
    "tests/test_fsdp_import.py",
}


def _test_files_outside_the_ci_roots() -> list[str]:
    # [FUNCTION DOC] HELPER: Runs git list-files (NUL-separated, pattern matching test python files)
    # in the repository root (raises if git fails) and returns the sorted paths that are not under
    # any CI discovery root or the manual directory, not under the agent-tooling directory, and
    # whose base name starts with test_.
    """Tracked test files this repository carries, asked of git rather than the filesystem.

    A directory walk from the checkout root is wrong here: CI clones sglang and
    Megatron-LM *into* the workspace, so the walk would inherit thousands of test
    files belonging to other repositories (and, locally, whatever a venv or a
    worktree happens to hold).
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", "*test_*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    allowed = tuple(f"{root}/" for root in (*_DISCOVERY_ROOTS, _MANUAL_ROOT))
    return sorted(
        path
        for path in listing.split("\0")
        if path
        # .claude/ holds agent-skill tooling whose test suites are run by the
        # skill's own workflow, not by the CI runners this rule guards.
        and not path.startswith((*allowed, ".claude/")) and PurePosixPath(path).name.startswith("test_")
    )


def test_no_test_file_lives_outside_the_ci_roots():
    # [FUNCTION DOC] Skips when the repository has no git metadata. Otherwise asserts that every
    # tracked out-of-root test file is in the spared orphan set; the failure message tells where to
    # move offenders and lists them.
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("no git metadata to ask; the rule is about what the repository tracks")
    offenders = [f for f in _test_files_outside_the_ci_roots() if f not in _KNOWN_ORPHANS]
    assert offenders == [], (
        "these files are named like tests but no runner collects them; move them "
        "under tests/fast/ (CPU) or tests/fast-gpu/, and let a module-level skip "
        "state any dependency they lack (or under tests/manual/ if no CI job should "
        f"run them): {offenders}"
    )


def test_no_known_orphan_outlives_its_reason():
    # [FUNCTION DOC] Asserts every spared orphan path still names an existing file in the checkout;
    # otherwise the failure lists the stale entries to delete.
    """A spared file that has since moved leaves an entry that spares nothing."""
    stale = sorted(f for f in _KNOWN_ORPHANS if not (REPO_ROOT / f).is_file())
    assert stale == [], f"these exceptions no longer name an existing file; delete them: {stale}"


def test_every_registration_names_a_suite_a_job_runs(monkeypatch):
    # [FUNCTION DOC] Pins the working directory to the repository root, collects all registrations,
    # and asserts none has a suite missing from the suite inventory of its backend; the failure
    # lists the sorted (file, suite) pairs.
    """A CPU or ROCm suite no job runs parses cleanly, so its test silently never runs.

    A misspelled CUDA suite already fails collection against CUDA_STAGES.
    """
    monkeypatch.chdir(REPO_ROOT)
    unknown = sorted(
        (r.filename, r.suite) for r in collect_tests(discover_ci_files()) if r.suite not in CI_SUITES[r.backend]
    )
    assert unknown == [], f"no CI job runs these suites; pick one from run_suite.CI_SUITES: {unknown}"


def test_the_example_suites_reach_the_cpu_plan(monkeypatch):
    # [FUNCTION DOC] Pins the working directory to the repository root, collects all registrations,
    # selects those under the fast examples directory, and asserts there is at least one and that
    # each is a CPU registration in the first CPU stage with no disabled reason.
    """The move only pays for itself if the runner really collects them.

    discover_ci_files() globs repo-relative paths, so it reads whatever cwd the
    runner was started from; pin it to the checkout.
    """
    monkeypatch.chdir(REPO_ROOT)
    plan = collect_tests(discover_ci_files())

    example_entries = [r for r in plan if r.filename.startswith("tests/fast/examples/")]
    assert example_entries, "no test under tests/fast/examples/ reached the CI plan"
    for entry in example_entries:
        assert entry.backend == HWBackend.CPU, entry.filename
        assert entry.suite == "stage-a-cpu", entry.filename
        assert entry.disabled is None, entry.filename
