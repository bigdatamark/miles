# [MODULE DOC] Command line driver that verifies a linear chain of git commits (everything in
# the range after a base commit up to a branch tip) against a folder of proof scripts, and
# produces a markdown report. Pipeline: (1) resolve base and branch to commit ids and list the
# chain oldest first, requiring the base to be an ancestor and the range to be non-empty and
# free of merge commits; (2) classify each commit by the standalone kind words found in its
# message (exactly one of the mechanical or non-mechanical words, else an error verdict);
# (3) for mechanical commits locate exactly one proof script named by a lowercase hex prefix
# of the commit id (at least seven characters), either in a subfolder of the proof folder or
# directly in it; (4) run each located proof as a separate python process from the repository
# root, concurrently on a bounded thread pool, where a pass requires exit status zero and a
# line starting with the pass marker on standard output; (5) optionally reuse earlier passes
# remembered in a JSON cache file inside the git common directory, keyed by commit id plus
# content hashes of the proof script and the utilities module; (6) render the report, write it
# to the proof folder (or a chosen path) and print it. Process exit status: 0 when every
# commit is ok (passed, or declared non-mechanical for human review), 1 when any commit is
# not ok, 2 on a setup problem (bad ref, empty or non-linear chain, missing proof folder). Verdict names are module constants; the kind words and report file name too.
"""Verify a whole mechanical-refactor chain: classification, proofs, and a full report.

Every commit in ``base..branch`` must classify itself by carrying exactly one of the two
words ``mechanical_provable`` or ``non_mechanical_provable`` anywhere in its message (the
rest of the message format is free). Every ``mechanical_provable`` commit must ship a
proof script in the proof folder (``<proof>/repro_scripts/<sha-prefix>.py`` or a flat
``<proof>/<sha-prefix>.py``), and running the proof must PASS -- reproduce the commit
byte-for-byte. A ``non_mechanical_provable`` commit carries no machine proof and is left
to human review.

The run prints a markdown report, writes it into the proof folder (``chain_report.md``),
and exits 0 iff the whole chain verifies. Normative contract: spec-reproduction-cli.md.

    python3 mechanical_refactor_reproduction_cli.py \
        --base <base-commit> --branch <pr-branch-name> --proof path/to/proof/folder
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_JOBS = 3
_PASSED_CACHE_FILENAME = "mechanical_refactor_passed_proofs.json"
_CACHED_PASS_DETAIL = "reused this machine's earlier PASS (--skip-passed)"

KIND_MECHANICAL = "mechanical_provable"
KIND_NON_MECHANICAL = "non_mechanical_provable"

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_MISSING_PROOF = "MISSING_PROOF"
VERDICT_AMBIGUOUS_PROOF = "AMBIGUOUS_PROOF"
VERDICT_HUMAN_REVIEW = "HUMAN_REVIEW"
VERDICT_UNCLASSIFIED = "UNCLASSIFIED"
VERDICT_AMBIGUOUS_KIND = "AMBIGUOUS_KIND"

_OK_VERDICTS = (VERDICT_PASS, VERDICT_HUMAN_REVIEW)

# The words are matched standalone: delimited by any non-[0-9A-Za-z_] character or the
# string boundary, so `non_mechanical_provable` never also counts as the bare word.
_KIND_WORD_RE = re.compile(r"(?<![0-9A-Za-z_])(non_)?mechanical_provable(?![0-9A-Za-z_])")

# The arbiter's verdict line (Repro.run / verify_mechanical_refactor both print `PASS:`).
_PASS_LINE_RE = re.compile(r"^PASS:", re.MULTILINE)

_MIN_PROOF_STEM_LEN = 7
_REPORT_FILENAME = "chain_report.md"
_FAIL_OUTPUT_TAIL_LINES = 60


class ChainVerificationError(Exception):
    # [CLASS DOC] Exception for setup problems that make verification impossible (unresolvable
    # references, non-ancestor base, empty or non-linear range, missing proof folder). The
    # command line entry point turns it into an error message on standard error and exit status 2.
    """A setup problem (bad refs, non-linear range, missing proof folder): exit code 2."""


@dataclass(frozen=True)
class CommitVerdict:
    # [CLASS DOC] Immutable per-commit outcome. Fields: full commit id; subject line; declared
    # kind (one of the two kind words, or nothing when classification failed); verdict (one of
    # the verdict constants); a markdown detail text (empty by default); and a flag saying the
    # pass was reused from the cache rather than freshly run (false by default).
    sha: str
    subject: str
    kind: "str | None"
    verdict: str
    detail: str = ""
    cached: bool = False

    @property
    def ok(self) -> bool:
        # [FUNCTION DOC] True when the verdict is a pass or a human-review verdict (the only
        # two acceptable outcomes); false for every failure or classification error verdict.
        return self.verdict in _OK_VERDICTS


@dataclass(frozen=True)
class _PendingProof:
    # [CLASS DOC] HELPER: Immutable record of a mechanical commit whose proof script has been
    # uniquely located and still has to be run (or answered from the cache). Fields: full
    # commit id, subject line, kind, and the path of the proof script.
    sha: str
    subject: str
    kind: str
    script: Path


@dataclass(frozen=True)
class ChainResult:
    # [CLASS DOC] Immutable result of verifying a chain. Fields: the base and branch as the user
    # spelled them, their resolved full commit ids, the proof folder path, and the list of
    # per-commit verdicts in chain order (oldest first; empty by default).
    base: str
    branch: str
    base_sha: str
    branch_sha: str
    proof_dir: Path
    verdicts: "list[CommitVerdict]" = field(default_factory=list)

    @property
    def passed(self) -> bool:
        # [FUNCTION DOC] True only when there is at least one verdict and every verdict is ok.
        return bool(self.verdicts) and all(v.ok for v in self.verdicts)


def main(argv: "list[str]") -> int:
    # [FUNCTION DOC] Command line entry point; takes the argument list without the program
    # name. Options: required base commit, required branch name, required proof folder; optional
    # repository root (default: the repository containing the current directory), optional
    # report path (default: the report file name inside the proof folder), optional job count
    # (integer, default 3) and a switch to reuse earlier passes. Calls the chain verification;
    # on a setup error prints "error: <message>" to standard error and returns 2. Otherwise
    # renders the markdown report, writes it to the report path (overwriting; an unwritable
    # path raises an operating-system error), prints the report and a line naming the report
    # file to standard output, and returns 0 if the whole chain passed, else 1.
    parser = argparse.ArgumentParser(description="Verify a whole mechanical-refactor chain against its proof folder.")
    parser.add_argument("--base", required=True, help="base commit of the chain")
    parser.add_argument("--branch", required=True, help="PR branch name (chain tip)")
    parser.add_argument("--proof", required=True, help="proof folder path")
    parser.add_argument("--repo-root", default=None, help="repo root (default: cwd's)")
    parser.add_argument(
        "--report",
        default=None,
        help=f"report file path (default: <proof>/{_REPORT_FILENAME})",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_DEFAULT_JOBS,
        help=f"max concurrent proof runs (default {_DEFAULT_JOBS})",
    )
    parser.add_argument(
        "--skip-passed",
        action="store_true",
        help="reuse this machine's earlier PASS verdicts for unchanged proofs",
    )
    args = parser.parse_args(argv)

    try:
        result = verify_chain(
            base=args.base,
            branch=args.branch,
            proof=Path(args.proof),
            repo_root=args.repo_root,
            jobs=args.jobs,
            skip_passed=args.skip_passed,
        )
    except ChainVerificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = render_report(result)
    report_path = Path(args.report) if args.report else result.proof_dir / _REPORT_FILENAME
    report_path.write_text(report)
    print(report)
    print(f"report written to: {report_path}")
    return 0 if result.passed else 1


def verify_chain(
    *,
    base: str,
    branch: str,
    proof: Path,
    repo_root: "str | None" = None,
    jobs: int = _DEFAULT_JOBS,
    skip_passed: bool = False,
) -> ChainResult:
    # [FUNCTION DOC] Runs the whole verification (all arguments keyword-only) and returns the
    # chain result. Steps: use the given repository root or find it; raise a chain error if the
    # proof folder is not a directory; resolve base and branch to commit ids; list the linear
    # chain; for each commit read its subject and full message and resolve it to either a final
    # verdict or a pending proof; load the cache (always) from the git common directory; when
    # skip_passed is set, replace each pending proof whose cache entry equals its current key
    # with a cached pass; run the remaining pending proofs on a pool of at most jobs workers
    # (minimum one); record fresh passes into the cache file; return the result holding the
    # verdicts in chain order. Side effects: git subprocesses, proof subprocesses, progress lines
    # printed to standard output, cache file write. Raises the chain error described above or a
    # called-process error if an unexpected git command fails.
    """Classify every commit in ``base..branch`` and run every provable commit's proof.

    Classification and proof resolution are sequential (cheap); the proof runs execute
    concurrently, up to ``jobs`` at a time — safe because each proof works in its own
    throwaway worktree. The verdict list keeps chain order. With ``skip_passed``, a
    pending proof whose (sha, script hash, utils hash) triple this machine already ran to
    a PASS is reused instead of re-executed; every fresh PASS is recorded either way."""
    root = repo_root or _repo_root()
    if not proof.is_dir():
        raise ChainVerificationError(f"proof folder does not exist: {proof}")
    base_sha = _rev_parse(base, root)
    branch_sha = _rev_parse(branch, root)
    commits = _linear_commits(base_sha=base_sha, branch_sha=branch_sha, root=root)

    resolved: list[CommitVerdict | _PendingProof] = []
    for sha in commits:
        subject = _git_output(["log", "-1", "--format=%s", sha], root).strip()
        message = _git_output(["log", "-1", "--format=%B", sha], root)
        resolved.append(_resolve_commit(sha=sha, subject=subject, message=message, proof=proof))

    cache_path = _passed_cache_path(root)
    cache = _load_passed_cache(cache_path)
    if skip_passed:
        resolved = [_reuse_cached_pass(item, cache=cache) for item in resolved]

    pending_by_sha = {item.sha: item for item in resolved if isinstance(item, _PendingProof)}
    verdicts: list[CommitVerdict] = _run_pending_proofs(resolved=resolved, root=root, jobs=jobs)
    _record_passes(
        cache=cache,
        cache_path=cache_path,
        verdicts=verdicts,
        pending_by_sha=pending_by_sha,
    )
    return ChainResult(
        base=base,
        branch=branch,
        base_sha=base_sha,
        branch_sha=branch_sha,
        proof_dir=proof,
        verdicts=verdicts,
    )


def render_report(result: ChainResult) -> str:
    # [FUNCTION DOC] Builds the markdown report text from a chain result. Layout: a title; bullet
    # lines for the base (as spelled and first twelve characters of its id), the branch likewise,
    # the proof folder, the overall verdict in bold (PASS or FAIL), a commit count line (total,
    # number of mechanical, number of non-mechanical, number with no valid kind reported as
    # classification errors), a proof line (count of pass verdicts over count of mechanical
    # commits) and, only if any were cached, a line with the cached count; then a table with
    # one row per commit in order (index from one, first nine characters of the id, kind or a
    # question mark, verdict, subject with vertical bars escaped); then, if any verdict is not
    # ok, a failure details section with a heading per failing commit (nine-character id and
    # verdict) followed by its detail text or a no-detail placeholder. Ends with a newline.
    # Pure function; no I/O.
    """The full chain report as markdown: header, per-commit table, failure details."""
    n_mech = sum(1 for v in result.verdicts if v.kind == KIND_MECHANICAL)
    n_non_mech = sum(1 for v in result.verdicts if v.kind == KIND_NON_MECHANICAL)
    n_unclassified = sum(1 for v in result.verdicts if v.kind is None)
    n_pass = sum(1 for v in result.verdicts if v.verdict == VERDICT_PASS)
    n_cached = sum(1 for v in result.verdicts if v.cached)

    lines = [
        "# Mechanical refactor chain report",
        "",
        f"- base: `{result.base}` (`{result.base_sha[:12]}`)",
        f"- branch: `{result.branch}` (`{result.branch_sha[:12]}`)",
        f"- proof folder: `{result.proof_dir}`",
        f"- chain verdict: **{'PASS' if result.passed else 'FAIL'}**",
        f"- commits: {len(result.verdicts)} total — {n_mech} {KIND_MECHANICAL}, "
        f"{n_non_mech} {KIND_NON_MECHANICAL}, {n_unclassified} classification error(s)",
        f"- proofs: {n_pass}/{n_mech} PASS",
        *([f"- reused from the passed-proof cache (--skip-passed): {n_cached}"] if n_cached else []),
        "",
        "| # | commit | kind | verdict | subject |",
        "|---|--------|------|---------|---------|",
    ]
    for i, v in enumerate(result.verdicts, start=1):
        kind = v.kind or "?"
        subject = v.subject.replace("|", "\\|")
        lines.append(f"| {i} | `{v.sha[:9]}` | {kind} | {v.verdict} | {subject} |")

    failures = [v for v in result.verdicts if not v.ok]
    if failures:
        lines += ["", "## Failure details"]
        for v in failures:
            lines += [
                "",
                f"### `{v.sha[:9]}` — {v.verdict}",
                "",
                v.detail or "(no detail)",
            ]
    return "\n".join(lines) + "\n"


def _reuse_cached_pass(item: "CommitVerdict | _PendingProof", *, cache: dict) -> "CommitVerdict | _PendingProof":
    # [FUNCTION DOC] HELPER: Given a resolved item and the cache mapping, returns the item
    # unchanged unless it is a pending proof whose commit id has a cache entry equal to the
    # proof's current cache key (hashes of script and utilities module). In that case prints a
    # cached-pass line and returns a pass verdict carrying the fixed cached detail text and the
    # cached flag set.
    """Turn a pending proof into a cached PASS verdict on an exact cache-key match."""
    if not isinstance(item, _PendingProof):
        return item
    if cache.get("passed", {}).get(item.sha) != _proof_cache_key(item.script):
        return item
    print(f"proof {item.sha[:9]}  {VERDICT_PASS} (cached)", flush=True)
    return CommitVerdict(
        sha=item.sha,
        subject=item.subject,
        kind=item.kind,
        verdict=VERDICT_PASS,
        detail=_CACHED_PASS_DETAIL,
        cached=True,
    )


def _record_passes(
    *,
    cache: dict,
    cache_path: Path,
    verdicts: "list[CommitVerdict]",
    pending_by_sha: "dict[str, _PendingProof]",
) -> None:
    # [FUNCTION DOC] HELPER: Persists fresh passes. Selects verdicts that are passes, not
    # cached and whose commit id is among the pending proofs. If none, returns without touching
    # the file. Otherwise stores each commit's current cache key in the passed section of the
    # cache mapping (mutating it) and writes the mapping as indented, key-sorted JSON with a
    # trailing newline to the cache path. An operating-system error on write is reported as a
    # note on standard output and swallowed. Failures are never recorded.
    """Record every freshly-run PASS into the cache (a FAIL is never recorded)."""
    fresh = [v for v in verdicts if v.verdict == VERDICT_PASS and not v.cached and v.sha in pending_by_sha]
    if not fresh:
        return
    for v in fresh:
        cache.setdefault("passed", {})[v.sha] = _proof_cache_key(pending_by_sha[v.sha].script)
    try:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"note: could not write passed-proof cache {cache_path}: {exc}")


def _proof_cache_key(script: Path) -> "dict[str, str]":
    # [FUNCTION DOC] HELPER: Builds the cache key mapping with two hex SHA-256 strings: the
    # script_sha256 of the proof script's bytes, and utils_sha256 of the reproduction utilities
    # module file, searched first next to the script and then in its parent directory (first
    # found wins; empty string if absent). Raises an operating-system error if the script cannot
    # be read.
    """The cache key parts beyond the sha: hashes of the script and its utils module."""
    utils_sha256 = ""
    for directory in (script.parent, script.parent.parent):
        utils = directory / "mechanical_refactor_reproduction_utils.py"
        if utils.is_file():
            utils_sha256 = hashlib.sha256(utils.read_bytes()).hexdigest()
            break
    return {
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "utils_sha256": utils_sha256,
    }


def _passed_cache_path(root: str) -> Path:
    # [FUNCTION DOC] HELPER: Asks git for the common git directory of the repository at root
    # (shared by all worktrees), makes it absolute relative to root if needed, and returns the
    # path of the cache file inside it. Raises a called-process error if git fails.
    common_dir = _git_output(["rev-parse", "--git-common-dir"], root).strip()
    common = Path(common_dir)
    if not common.is_absolute():
        common = Path(root) / common
    return common / _PASSED_CACHE_FILENAME


def _load_passed_cache(path: Path) -> dict:
    # [FUNCTION DOC] HELPER: Reads the JSON cache file. If it is missing, unreadable, invalid
    # JSON, not a mapping, or lacks a mapping under the passed key, returns a fresh mapping with
    # an empty passed section; otherwise returns the parsed mapping (with any extra keys kept).
    """The cache is best-effort: missing, corrupt, or unreadable means empty."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"passed": {}}
    if not isinstance(data, dict) or not isinstance(data.get("passed"), dict):
        return {"passed": {}}
    return data


def _run_pending_proofs(
    *, resolved: "list[CommitVerdict | _PendingProof]", root: str, jobs: int
) -> "list[CommitVerdict]":
    # [FUNCTION DOC] HELPER: Runs every pending proof among the resolved items on a thread pool
    # of max(1, jobs) workers, each producing a verdict via the proof verdict helper; waits for
    # all of them (collecting results in submission order, so an exception from one proof run
    # propagates when reached). Returns a list parallel to the input where each pending item is
    # replaced by its verdict and every other item is kept. With no pending proofs no pool is
    # created.
    """Execute the pending proofs on a bounded thread pool; keep chain order."""
    pending = [(i, item) for i, item in enumerate(resolved) if isinstance(item, _PendingProof)]
    finished: dict[int, CommitVerdict] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {i: pool.submit(_proof_verdict, item, root=root) for i, item in pending}
            for i, future in futures.items():
                finished[i] = future.result()
    return [finished[i] if isinstance(item, _PendingProof) else item for i, item in enumerate(resolved)]


def _proof_verdict(pending: _PendingProof, *, root: str) -> CommitVerdict:
    # [FUNCTION DOC] HELPER: Runs one pending proof from the repository root and converts the
    # outcome to a verdict with the same commit id, subject and kind. A pass has an empty
    # detail. A failure has the fail verdict and a markdown detail naming the script followed by
    # a fenced block holding the last 60 lines of combined output. Prints a progress line with
    # the first nine characters of the commit id and the verdict to standard output (flushed).
    passed, output = _run_proof(script=pending.script, root=root)
    if passed:
        verdict = CommitVerdict(
            sha=pending.sha,
            subject=pending.subject,
            kind=pending.kind,
            verdict=VERDICT_PASS,
            detail="",
        )
    else:
        tail = "\n".join(output.splitlines()[-_FAIL_OUTPUT_TAIL_LINES:])
        verdict = CommitVerdict(
            sha=pending.sha,
            subject=pending.subject,
            kind=pending.kind,
            verdict=VERDICT_FAIL,
            detail=(f"proof `{pending.script}` did not PASS; output tail:\n\n" f"```\n{tail}\n```"),
        )
    print(f"proof {pending.sha[:9]}  {verdict.verdict}", flush=True)
    return verdict


def _resolve_commit(*, sha: str, subject: str, message: str, proof: Path) -> "CommitVerdict | _PendingProof":
    # [FUNCTION DOC] HELPER: Decides what to do with one commit. Classifies the message; if the
    # kind is invalid returns a verdict (kind nothing) using the classification error as verdict
    # and a fixed detail asking for exactly one of the two kind words. A non-mechanical commit
    # gets the human-review verdict. For a mechanical commit looks up proof scripts: none gives
    # the missing-proof verdict (detail names the two searched folders and the minimum prefix
    # length); more than one gives the ambiguous-proof verdict (detail lists them); exactly one
    # gives a pending proof record for later execution.
    kind, classification_error = _classify(message)
    if kind is None:
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=None,
            verdict=classification_error,
            detail=(
                f"the commit message must contain exactly one of the words "
                f"`{KIND_MECHANICAL}` or `{KIND_NON_MECHANICAL}`"
            ),
        )
    if kind == KIND_NON_MECHANICAL:
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=kind,
            verdict=VERDICT_HUMAN_REVIEW,
            detail="declared non_mechanical_provable: no machine proof, review by hand",
        )

    scripts = _find_proof_scripts(proof=proof, sha=sha)
    if not scripts:
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=kind,
            verdict=VERDICT_MISSING_PROOF,
            detail=(
                f"no proof script found; searched `{proof / 'repro_scripts'}` and "
                f"`{proof}` for `<sha-prefix>.py` (>= {_MIN_PROOF_STEM_LEN} hex chars)"
            ),
        )
    if len(scripts) > 1:
        listing = ", ".join(f"`{p}`" for p in scripts)
        return CommitVerdict(
            sha=sha,
            subject=subject,
            kind=kind,
            verdict=VERDICT_AMBIGUOUS_PROOF,
            detail=f"multiple proof scripts match this commit: {listing}",
        )

    return _PendingProof(sha=sha, subject=subject, kind=kind, script=scripts[0])


def _classify(message: str) -> "tuple[str | None, str]":
    # [FUNCTION DOC] HELPER: Finds all standalone occurrences of the two kind words in the
    # message (a word counts only when not adjacent to letters, digits or underscores, and the
    # non-mechanical word is never also counted as the plain word) and collects the distinct
    # kinds. Returns a pair: (kind, empty text) when exactly one distinct kind occurs (any number
    # of times); (nothing, unclassified verdict) when none occurs; (nothing, ambiguous-kind
    # verdict) when both occur.
    """The commit's declared kind, or (None, error-verdict) when the word rule is broken.

    Exactly one of the two words must appear (any number of times, but only one of the
    two): zero occurrences is UNCLASSIFIED, both words present is AMBIGUOUS_KIND."""
    kinds = {KIND_NON_MECHANICAL if match.group(1) else KIND_MECHANICAL for match in _KIND_WORD_RE.finditer(message)}
    if not kinds:
        return None, VERDICT_UNCLASSIFIED
    if len(kinds) > 1:
        return None, VERDICT_AMBIGUOUS_KIND
    return kinds.pop(), ""


def _find_proof_scripts(*, proof: Path, sha: str) -> "list[Path]":
    # [FUNCTION DOC] HELPER: Collects candidate proof scripts for a commit id. Searches first
    # the repro_scripts subfolder of the proof folder, then the proof folder itself (skipping
    # either if it is not a directory); within each, python files in sorted name order are
    # accepted when the file stem is at least seven characters, entirely lowercase hexadecimal
    # digits, and a prefix of the commit id. Returns the matches, subfolder ones first. Several
    # matches (for example two prefix lengths) are reported by the caller as ambiguous.
    """Proof scripts naming this commit: a ``<sha-prefix>.py`` (lowercase hex, >= 7 chars)
    under ``<proof>/repro_scripts/`` or flat in ``<proof>/``."""
    found: list[Path] = []
    for directory in (proof / "repro_scripts", proof):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            stem = path.stem
            is_sha_prefix = (
                len(stem) >= _MIN_PROOF_STEM_LEN
                and all(c in "0123456789abcdef" for c in stem)
                and sha.startswith(stem)
            )
            if is_sha_prefix:
                found.append(path)
    return found


def _run_proof(*, script: Path, root: str) -> "tuple[bool, str]":
    # [FUNCTION DOC] HELPER: Runs the proof script with the current python interpreter, using
    # its absolute path, in the repository root as working directory, capturing text output
    # (no timeout). Returns a pair: a pass flag, true only when the exit status is zero and some
    # line of standard output begins with the pass marker; and the standard output followed by
    # standard error combined into one string.
    """Run one proof script from the repo root. A PASS is exit code 0 AND the arbiter's
    ``PASS:`` verdict line on stdout (an old-style script that exits 0 with a residual is
    therefore still a FAIL)."""
    result = subprocess.run(
        [sys.executable, str(script.resolve())],
        cwd=root,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    passed = result.returncode == 0 and bool(_PASS_LINE_RE.search(result.stdout))
    return passed, output


def _linear_commits(*, base_sha: str, branch_sha: str, root: str) -> "list[str]":
    # [FUNCTION DOC] HELPER: Returns the commit ids in the range after base up to branch tip,
    # oldest first. Raises the chain error if base is not an ancestor of branch, if the range is
    # empty, or if any commit in it has more than one parent (message lists the first nine
    # characters of each merge commit). Runs one git call per commit to check parents.
    if not _is_ancestor(base_sha=base_sha, branch_sha=branch_sha, root=root):
        raise ChainVerificationError(f"base {base_sha[:12]} is not an ancestor of branch {branch_sha[:12]}")
    commits = _git_output(["rev-list", "--reverse", f"{base_sha}..{branch_sha}"], root).split()
    if not commits:
        raise ChainVerificationError(f"no commits in {base_sha[:12]}..{branch_sha[:12]}")
    merges = [sha for sha in commits if len(_git_output(["rev-list", "--parents", "-n", "1", sha], root).split()) > 2]
    if merges:
        listing = ", ".join(sha[:9] for sha in merges)
        raise ChainVerificationError(f"the chain must be linear, but it contains merge commit(s): {listing}")
    return commits


def _is_ancestor(*, base_sha: str, branch_sha: str, root: str) -> bool:
    # [FUNCTION DOC] HELPER: Runs git's ancestor test in the repository at root; returns true
    # when the exit status is zero (a commit counts as its own ancestor), false for any other
    # status including git errors. Output is discarded.
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_sha, branch_sha],
        cwd=root,
        capture_output=True,
    )
    return result.returncode == 0


def _rev_parse(ref: str, root: str) -> str:
    # [FUNCTION DOC] HELPER: Resolves a reference to a full commit id, peeling tags to commits.
    # Raises the chain error with git's stripped error text if it cannot be resolved.
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ChainVerificationError(f"cannot resolve {ref!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_output(args: "list[str]", root: str) -> str:
    # [FUNCTION DOC] HELPER: Runs git with the argument list in the directory root and returns
    # unstripped standard output text. Raises the called-process error on non-zero exit.
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout


def _repo_root() -> str:
    # [FUNCTION DOC] HELPER: Returns the top-level directory of the repository containing the
    # current working directory, as reported by git; raises the called-process error outside a
    # repository.
    return subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
