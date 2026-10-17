# [MODULE DOC] Tests for the chain report: its header, per-commit table and overall verdict,
# the failure details section, a fully passing report, and the command line entry point writing
# the report to the default and to a custom location. A helper builds a three-commit chain
# (mechanical with proof, non-mechanical, mechanical without proof).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cli_testlib import _chain, _write_stub_proof
from mechanical_refactor_reproduction_cli import main, render_report, verify_chain


def _mixed_chain_result(repo: Path, tmp_path: Path):
    # [FUNCTION DOC] HELPER: Builds a chain of three commits (mechanical, non-mechanical,
    # mechanical) and writes a passing stand-in proof only for the first, so the verdicts are
    # pass, human review and missing proof. Verifies the chain and returns a tuple of the proof
    # folder, the base commit id, the list of chain commit ids, and the chain result.
    proof = tmp_path / "proof"
    base, shas = _chain(
        repo,
        [
            "mechanical_provable: move foo",
            "non_mechanical_provable: rework bar",
            "mechanical_provable: move baz",
        ],
    )
    _write_stub_proof(proof, shas[0])
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    return proof, base, shas, result


def test_report_has_header_table_row_per_commit_and_chain_verdict(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Renders the report for the mixed chain and asserts it contains: the title;
    # the first twelve characters of the base id; the chain verdict FAIL; the commit count line
    # (three total, two mechanical, one non-mechanical); the first nine characters of every chain
    # commit id; and table cells for a passing mechanical row, a human-review non-mechanical row
    # and a missing-proof mechanical row.
    """The report carries base/branch/proof, one table row per commit, and the verdict."""
    proof, base, shas, result = _mixed_chain_result(repo, tmp_path)
    report = render_report(result)

    assert "# Mechanical refactor chain report" in report
    assert f"`{base[:12]}`" in report
    assert "chain verdict: **FAIL**" in report
    assert "3 total — 2 mechanical_provable, 1 non_mechanical_provable" in report
    for sha in shas:
        assert f"`{sha[:9]}`" in report
    assert "| mechanical_provable | PASS |" in report
    assert "| non_mechanical_provable | HUMAN_REVIEW |" in report
    assert "| mechanical_provable | MISSING_PROOF |" in report


def test_report_lists_failure_details_for_each_non_ok_commit(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Renders the mixed chain report and asserts it has the failure details
    # heading, a sub-heading for the third commit (nine-character id and the missing-proof
    # verdict) and the no-proof-script explanation text.
    """Every non-ok commit gets a failure-details section with its explanation."""
    proof, base, shas, result = _mixed_chain_result(repo, tmp_path)
    report = render_report(result)

    assert "## Failure details" in report
    assert f"### `{shas[2][:9]}` — MISSING_PROOF" in report
    assert "no proof script found" in report


def test_passing_report_has_no_failure_details_section(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Builds a one-commit mechanical chain with a passing proof and renders the
    # report. Asserts it shows the chain verdict PASS, the proofs line one of one, and no failure
    # details heading.
    """A fully verified chain renders a PASS report without a failure section."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, ["mechanical_provable: move foo"])
    _write_stub_proof(proof, shas[0])
    result = verify_chain(base=base, branch="chain", proof=proof, repo_root=str(repo))
    report = render_report(result)

    assert "chain verdict: **PASS**" in report
    assert "proofs: 1/1 PASS" in report
    assert "## Failure details" not in report


def test_main_writes_the_report_into_the_proof_folder_by_default(repo: Path, tmp_path: Path, capsys) -> None:
    # [FUNCTION DOC] Builds a one-commit passing chain and calls the command line entry point
    # with base, branch, proof folder and repository root. Asserts it returns 0, the default
    # report file in the proof folder contains the chain verdict PASS, and the same text was
    # printed to standard output (captured). Then calls it again with a custom report path and
    # asserts it returns 0 and that file contains the PASS verdict.
    """main prints the report and writes <proof>/chain_report.md (or --report PATH)."""
    proof = tmp_path / "proof"
    base, shas = _chain(repo, ["mechanical_provable: move foo"])
    _write_stub_proof(proof, shas[0])
    args = [
        "--base",
        base,
        "--branch",
        "chain",
        "--proof",
        str(proof),
        "--repo-root",
        str(repo),
    ]

    assert main(args) == 0
    default_report = proof / "chain_report.md"
    assert "chain verdict: **PASS**" in default_report.read_text()
    assert "chain verdict: **PASS**" in capsys.readouterr().out

    custom = tmp_path / "custom_report.md"
    assert main([*args, "--report", str(custom)]) == 0
    assert "chain verdict: **PASS**" in custom.read_text()
