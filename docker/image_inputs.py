#!/usr/bin/env python3
# doc-dev: docs/developer/ci/02-docker-build.md
# [MODULE DOC]
# Computes a single SHA-256 content hash over the files that feed a Miles Docker image build, and
# reads such a hash back from a published image manifest. The set of files is defined by a tuple
# of glob patterns (the Dockerfiles for the CUDA and Rubin images, the build driver, kube tools
# installer, TransformerEngine verification script, everything under the patch directory, and the
# requirements file). The ROCm Dockerfile is deliberately excluded. Files are listed either from
# the git index of a working tree (tracked files) or from the tree of a git revision, so untracked
# files never count. Standard library only (plus the git executable) because CI runs it before
# installing dependencies. As a script: prints the hash of the working tree, of a given revision,
# or (with the read-label flag) the hash recorded in manifest JSON read from standard input.
"""Content hash of everything that feeds a Miles Docker image build.

Single source of truth for "what changes the image". CI compares this hash against
the one recorded on an already-published tag (label ``miles.image-inputs``) to decide
whether a rebuild is needed; ``build.py`` stamps the label at build time.

Stdlib only: CI computes the hash on a plain hosted runner before any dependency
is installed.

Usage:
    python docker/image_inputs.py                 # hash the working tree
    python docker/image_inputs.py --rev HEAD^1    # hash a git revision
"""

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# A build reads exactly these. Anything outside them cannot change the image, so a
# PR touching only such files reuses the released image. Dockerfile.rocm is absent on
# purpose: it feeds the rocm/sgl-dev images, not the cu13 multi-arch image built here.
INPUT_GLOBS = (
    "docker/Dockerfile",
    "docker/Dockerfile.rubin",
    "docker/build.py",
    "docker/install-kube-tools.sh",
    "docker/verify_transformer_engine.py",
    "docker/patch/*",
    "requirements.txt",
)

LABEL_KEY = "miles.image-inputs"


def _matches(path: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the repository-relative path matches any input glob pattern
    # (shell-style matching where a star also crosses directory separators).
    return any(fnmatch.fnmatch(path, glob) for glob in INPUT_GLOBS)


def _git(*args: str, root: Path = REPO_ROOT) -> bytes:
    # [FUNCTION DOC] HELPER: Runs git with the given arguments in the root directory, raising a
    # called-process error on failure, and returns the raw standard output bytes.
    return subprocess.run(["git", *args], cwd=root, check=True, stdout=subprocess.PIPE).stdout


def _paths_at(rev: str | None, root: Path = REPO_ROOT) -> list[str]:
    # [FUNCTION DOC] HELPER: Lists the input file paths, sorted. For no revision uses the tracked
    # files of the working tree; for a revision lists all blob paths in that revision's tree. Keeps
    # only paths matching the input globs.
    if rev is None:
        listing = _git("ls-files", root=root).decode()
    else:
        listing = _git("ls-tree", "-r", "--name-only", rev, root=root).decode()
    return sorted(p for p in listing.splitlines() if _matches(p))


def _content_at(path: str, rev: str | None, root: Path = REPO_ROOT) -> bytes:
    # [FUNCTION DOC] HELPER: Returns the file's bytes: read from disk under the root when no
    # revision is given (so uncommitted edits count), otherwise the content at that revision via
    # git show.
    if rev is None:
        return (root / path).read_bytes()
    return _git("show", f"{rev}:{path}", root=root)


def compute(rev: str | None = None, *, root: Path = REPO_ROOT) -> str:
    # [FUNCTION DOC] Returns the hex SHA-256 of the build inputs. Algorithm: for each input path in
    # sorted order, feed the digest the UTF-8 path, a NUL byte, and the 32-byte raw SHA-256 of the
    # file content. Deterministic across machines. Raises a called-process error if git fails and
    # a file error if a listed working-tree file cannot be read. Note a deleted-but-tracked file
    # in the working tree raises rather than being skipped.
    """Hash the build inputs in the working tree (``rev=None``) or at a revision."""
    digest = hashlib.sha256()
    for path in _paths_at(rev, root):
        digest.update(path.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(_content_at(path, rev, root)).digest())
    return digest.hexdigest()


def read_label(manifest: str) -> str:
    # [FUNCTION DOC] Takes the JSON text printed by the imagetools inspect command and returns the
    # first string value stored under the inputs label key anywhere in the structure (depth-first,
    # dictionary insertion order), or an empty string if blank input, invalid JSON, or no such key.
    """Pull the recorded hash out of ``docker buildx imagetools inspect`` JSON.

    The shape differs between single- and multi-platform tags, so search rather than
    index; an unpublished tag yields empty input and no hash.
    """

    def search(node: object) -> str:
        # [FUNCTION DOC] HELPER: Recursive depth-first search through nested dictionaries and
        # lists for a string value under the label key; returns it or an empty string. Falsy
        # (empty) matches are treated as not found and the search continues.
        if isinstance(node, dict):
            for key, value in node.items():
                if key == LABEL_KEY and isinstance(value, str):
                    return value
                if found := search(value):
                    return found
        elif isinstance(node, list):
            for item in node:
                if found := search(item):
                    return found
        return ""

    manifest = manifest.strip()
    if not manifest:
        return ""
    try:
        return search(json.loads(manifest))
    except json.JSONDecodeError:
        return ""


def main() -> int:
    # [FUNCTION DOC] Command line entry point; returns zero. With the read-label flag, reads all
    # of standard input and prints the hash recorded in that manifest JSON (empty line if none).
    # Otherwise prints the computed hash of the working tree, or of the git revision given by the
    # revision option.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rev", default=None, help="Git revision to hash (default: working tree).")
    parser.add_argument(
        "--read-label",
        action="store_true",
        help="Read imagetools inspect JSON on stdin and print the hash it recorded.",
    )
    args = parser.parse_args()
    print(read_label(sys.stdin.read()) if args.read_label else compute(args.rev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
