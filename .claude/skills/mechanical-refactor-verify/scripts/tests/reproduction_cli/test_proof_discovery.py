# [MODULE DOC] Tests for how the chain verifier discovers a commit's proof script: the nested
# subfolder layout and the flat layout, missing proofs, names that do not match, multiple
# matches, and the minimum prefix length. The module-level message constant is a mechanical-kind
# commit message used by every test.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli_testlib import _chain, _write_stub_proof
from mechanical_refactor_reproduction_cli import (
    VERDICT_AMBIGUOUS_PROOF,
    VERDICT_MISSING_PROOF,
    VERDICT_PASS,
    verify_chain,
)

_MSG = "mechanical_provable: move foo"


def _run_single(repo: Path, proof: Path):
    # [FUNCTION DOC] HELPER: Builds a one-commit chain with the mechanical message, verifies it
    # against the given proof folder, and returns a pair of the commit id and the chain result.
    base, shas = _chain(repo, [_MSG])
    return shas[0], verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))


def test_proof_is_found_under_repro_scripts_by_sha_prefix(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Writes a passing stand-in proof in the nested subfolder named by the first
    # nine characters of the commit id and verifies the chain. Asserts the verdict is pass.
    """The generator layout repro_scripts/<sha9>.py resolves to the commit's proof."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0], stem_len=9)
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    assert result.verdicts[0].verdict == VERDICT_PASS


def test_proof_is_found_flat_in_the_proof_folder_by_full_sha(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Writes a passing stand-in proof directly in the proof folder, named by the
    # full forty-character commit id, and verifies the chain. Asserts the verdict is pass.
    """A flat <proof>/<full-sha>.py layout is also accepted."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0], flat=True, stem_len=40)
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    assert result.verdicts[0].verdict == VERDICT_PASS


def test_provable_commit_without_a_proof_script_is_missing_proof(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Uses an existing empty proof folder with a mechanical commit. Asserts the
    # verdict is missing-proof and the chain did not pass.
    """A mechanical_provable commit with no matching script fails as MISSING_PROOF."""
    proof = tmp_path / "proof"
    proof.mkdir()
    sha, result = _run_single(repo, proof)
    assert result.verdicts[0].verdict == VERDICT_MISSING_PROOF
    assert not result.passed


def test_unrelated_and_non_hex_scripts_do_not_match(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Puts two failing scripts in the nested subfolder: one with a hex name that
    # is not a prefix of the commit id, and one with a non-hex name. Asserts the commit's
    # verdict is missing-proof (neither script was picked up).
    """Scripts named for another sha or with a non-hex stem are not this commit's proof."""
    proof = tmp_path / "proof"
    scripts = proof / "repro_scripts"
    scripts.mkdir(parents=True)
    (scripts / "0123456789abcdef.py").write_text("raise SystemExit(1)\n")
    (scripts / "not_a_sha.py").write_text("raise SystemExit(1)\n")
    sha, result = _run_single(repo, proof)
    assert result.verdicts[0].verdict == VERDICT_MISSING_PROOF


def test_two_scripts_matching_one_commit_are_ambiguous(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Writes one stand-in proof in the nested subfolder (nine-character prefix)
    # and another directly in the proof folder (twelve-character prefix), both for the same
    # commit. Asserts the verdict is ambiguous-proof and the chain did not pass.
    """A commit matched by both a nested and a flat script fails as AMBIGUOUS_PROOF."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0], stem_len=9)
    _write_stub_proof(proof, shas[0], flat=True, stem_len=12)
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    assert result.verdicts[0].verdict == VERDICT_AMBIGUOUS_PROOF
    assert not result.passed


def test_short_hex_stem_below_minimum_length_is_ignored(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Writes a stand-in proof named by only six characters of the commit id.
    # Asserts the verdict is missing-proof (below the seven character minimum).
    """A 6-char hex stem is too short to name a commit and is not treated as a proof."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, [_MSG])
    _write_stub_proof(proof, shas[0], stem_len=6)
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    assert result.verdicts[0].verdict == VERDICT_MISSING_PROOF
