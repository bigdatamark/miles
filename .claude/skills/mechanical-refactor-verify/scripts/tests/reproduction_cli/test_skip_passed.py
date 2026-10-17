# [MODULE DOC] Tests for the reuse-earlier-passes cache of the chain verifier. Stand-in proofs
# can increment a run counter file so re-execution is observable. Covers: reuse of an unchanged
# pass, always rerunning without the flag, invalidation by editing the proof script or the
# utilities module copy, never recording failures, tolerance of a corrupt cache file, the cache
# location in the git common directory and its entry shape, and the report and command line
# support for the flag. The module-level message constant is a mechanical-kind commit message.
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli_testlib import _chain, _write_stub_proof
from mechanical_refactor_reproduction_cli import VERDICT_FAIL, VERDICT_PASS, main, render_report, verify_chain

_MSG = "mechanical_provable: move foo"


def _counting_proof(script: Path, counter: Path, *, passing: bool = True) -> None:
    # [FUNCTION DOC] HELPER: Overwrites the proof script with a program that first reads the
    # integer in the counter file (zero if the file is missing), writes it back incremented by
    # one, and then prints either the pass verdict line and exits zero (default), or a residual
    # line and exits one when passing is false.
    """Make the stub proof bump a run counter so re-execution is observable."""
    verdict = (
        'print("PASS: reproduces the commit byte-for-byte.")\nsys.exit(0)\n'
        if passing
        else 'print("RESIDUAL (1 lines):\\n+x")\nsys.exit(1)\n'
    )
    script.write_text(
        "import sys\n"
        f"counter = __import__('pathlib').Path({str(counter)!r})\n"
        "runs = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(runs + 1))\n" + verdict
    )


def _cache_file(repo: Path) -> Path:
    # [FUNCTION DOC] HELPER: Returns the expected location of the passed-proof cache file: in
    # the repository's git directory (a plain, non-worktree repository in these tests).
    return repo / ".git" / "mechanical_refactor_passed_proofs.json"


def test_skip_passed_reuses_an_unchanged_pass_without_rerunning(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] One mechanical commit with a counting passing proof. First verification
    # (flag off): verdict pass and counter equals one. Second verification with the reuse flag:
    # verdict pass, marked cached, counter still one (proof not run again), and the chain passes.
    """A PASS recorded on the first run is reused: the proof does not execute again."""
    proof = tmp_path / "proof"
    counter = tmp_path / "runs"
    base, shas = _chain(repo, [_MSG])
    _counting_proof(_write_stub_proof(proof, shas[0]), counter)
    args = dict(base=base, branch="chain", proof=proof, repo_root=str(repo))

    first = verify_chain(**args)
    assert first.verdicts[0].verdict == VERDICT_PASS
    assert counter.read_text() == "1"

    second = verify_chain(**args, skip_passed=True)
    assert second.verdicts[0].verdict == VERDICT_PASS
    assert second.verdicts[0].cached
    assert counter.read_text() == "1"
    assert second.passed


def test_without_the_flag_the_proof_always_reruns(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Same setup with a counting passing proof; verifies twice without the reuse
    # flag. Asserts the counter equals two and the second verdict is not marked cached.
    """The cache is recorded on every run but consulted only under skip_passed."""
    proof = tmp_path / "proof"
    counter = tmp_path / "runs"
    base, shas = _chain(repo, [_MSG])
    _counting_proof(_write_stub_proof(proof, shas[0]), counter)
    args = dict(base=base, branch="chain", proof=proof, repo_root=str(repo))

    verify_chain(**args)
    result = verify_chain(**args)

    assert counter.read_text() == "2"
    assert not result.verdicts[0].cached


def test_editing_the_proof_script_invalidates_the_cache(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Verifies once with a passing stand-in proof (recording the pass), then
    # rewrites the script into one that prints a residual and exits one, and verifies again with
    # the reuse flag. Asserts the verdict is fail (the changed script hash missed the cache).
    """A changed script hash misses the cache, so the edited (failing) proof reruns."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    script = _write_stub_proof(proof, shas[0])

    verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    script.write_text('print("RESIDUAL (1 lines):\\n+x")\nraise SystemExit(1)\n')
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo), skip_passed=True)

    assert result.verdicts[0].verdict == VERDICT_FAIL


def test_editing_the_utils_copy_invalidates_the_cache(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Places a small utilities module file in the proof folder (parent of the
    # nested scripts folder) next to a counting passing proof, verifies once, changes the file's
    # content, then verifies with the reuse flag. Asserts the counter equals two and the second
    # verdict is not cached.
    """The utils module next to the scripts is part of the key: editing it forces a rerun."""
    proof = tmp_path / "proof"
    counter = tmp_path / "runs"
    utils = proof / "mechanical_refactor_reproduction_utils.py"
    base, shas = _chain(repo, [_MSG])
    _counting_proof(_write_stub_proof(proof, shas[0]), counter)
    utils.parent.mkdir(parents=True, exist_ok=True)
    utils.write_text("ENGINE = 1\n")
    args = dict(base=base, branch="chain", proof=proof, repo_root=str(repo))

    verify_chain(**args)
    utils.write_text("ENGINE = 2\n")
    result = verify_chain(**args, skip_passed=True)

    assert counter.read_text() == "2"
    assert not result.verdicts[0].cached


def test_a_fail_is_never_recorded_in_the_cache(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Verifies a chain whose only proof fails. Asserts that either no cache file
    # exists or its passed section has no entry for the commit.
    """Only PASS verdicts enter the cache; a failing proof leaves no entry for its sha."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0], passing=False)

    verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    cache = _cache_file(repo)
    assert not cache.exists() or shas[0] not in json.loads(cache.read_text())["passed"]


def test_corrupt_cache_file_is_treated_as_empty(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Writes invalid JSON into the cache file location, then verifies a chain with
    # a passing proof using the reuse flag. Asserts the verdict is pass and not cached.
    """A garbage cache file never crashes the walk; the proof simply runs."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0])
    _cache_file(repo).write_text("{not json")

    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo), skip_passed=True)

    assert result.verdicts[0].verdict == VERDICT_PASS
    assert not result.verdicts[0].cached


def test_cache_lives_in_the_git_common_dir_and_records_the_pass(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Verifies a chain with a passing proof (flag off), reads the cache file in the
    # git directory, and asserts the passed entry for the commit has exactly two keys (script hash
    # and utilities hash) and that the script hash is sixty-four characters long.
    """A PASS writes the (sha, script hash, utils hash) entry under .git/."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0])

    verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))

    entry = json.loads(_cache_file(repo).read_text())["passed"][shas[0]]
    assert set(entry) == {"script_sha256", "utils_sha256"}
    assert len(entry["script_sha256"]) == 64


def test_report_counts_reused_proofs_and_main_accepts_the_flag(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Runs the command line entry point once (returns 0, recording the pass), then
    # verifies with the reuse flag and renders the report. Asserts the report contains the line
    # with the reused count of one, and that the entry point with the reuse flag returns 0.
    """The report carries the reused count and --skip-passed works through main."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0])
    cli_args = [
        "--base",
        base,
        "--branch",
        "chain",
        "--proof",
        str(proof),
        "--repo-root",
        str(repo),
    ]

    assert main(cli_args) == 0
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo), skip_passed=True)
    report = render_report(result)

    assert "reused from the passed-proof cache (--skip-passed): 1" in report
    assert main([*cli_args, "--skip-passed"]) == 0
