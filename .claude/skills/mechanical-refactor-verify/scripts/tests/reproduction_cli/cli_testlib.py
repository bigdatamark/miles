# [MODULE DOC] HELPER: Shared helpers for the reproduction command line tests: git wrappers, a
# file writer, a builder for a chain of single-file commits, and a writer of stand-in proof
# scripts. The two module-level proof script texts are stand-ins: the passing one prints a
# pass verdict line and exits zero; the failing one prints a residual-diff verdict line and
# exits one.
import subprocess
from pathlib import Path

_PASSING_PROOF = "import sys\n" 'print("PASS: reproduces the commit byte-for-byte.")\n' "sys.exit(0)\n"
_FAILING_PROOF = "import sys\n" 'print("RESIDUAL (2 lines):\\n+x\\n-y")\n' "sys.exit(1)\n"


def _git(repo: Path, *args: str) -> str:
    # [FUNCTION DOC] HELPER: Runs git with the arguments in the repository directory, raising
    # on non-zero exit, and returns stripped standard output text (standard error is captured).
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _write(repo: Path, **files: "str | None") -> None:
    # [FUNCTION DOC] HELPER: Writes or deletes files under the repository directory. Each
    # keyword name is a relative path where a double underscore means a directory separator
    # (callers pass real names through an unpacked mapping). A value of nothing deletes the
    # existing file; a string creates parent directories as needed and writes the text.
    for name, content in files.items():
        path = repo / name.replace("__", "/")
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)


def _commit(repo: Path, message: str) -> str:
    # [FUNCTION DOC] HELPER: Stages everything and commits quietly with the message; returns
    # the full SHA of the new HEAD.
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _chain(repo: Path, messages: "list[str]") -> "tuple[str, list[str]]":
    # [FUNCTION DOC] HELPER: Builds a linear history. Creates a seed file and commits it as
    # "base", then switches to a new branch named chain and, for each message in order (index
    # i), writes a new file numbered i holding a single value assignment of i and commits it
    # with that message. Returns a pair: the base commit SHA and the list of chain commit SHAs
    # in order.
    """A base commit plus one single-file commit per message, on a `chain` branch.
    Returns (base_sha, commit_shas)."""
    _write(repo, **{"seed.py": "SEED = 0\n"})
    base = _commit(repo, "base")
    _git(repo, "switch", "-q", "-c", "chain")
    shas: list[str] = []
    for i, message in enumerate(messages):
        _write(repo, **{f"file_{i}.py": f"VALUE = {i}\n"})
        shas.append(_commit(repo, message))
    return base, shas


def _write_stub_proof(
    proof_dir: Path,
    sha: str,
    *,
    passing: bool = True,
    flat: bool = False,
    stem_len: int = 9,
) -> Path:
    # [FUNCTION DOC] HELPER: Writes a stand-in proof script for a commit and returns its path.
    # The script is placed in the proof directory itself when flat is true, otherwise in a
    # subdirectory named repro_scripts (created if missing). The file name is the first
    # stem_len characters (default nine) of the commit SHA plus the python extension. Content
    # is the passing stand-in (when passing is true, the default) or the failing stand-in.
    """A stand-in proof script printing the arbiter's verdict line and exiting to match."""
    directory = proof_dir if flat else proof_dir / "repro_scripts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{sha[:stem_len]}.py"
    path.write_text(_PASSING_PROOF if passing else _FAILING_PROOF)
    return path
