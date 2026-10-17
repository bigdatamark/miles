# [MODULE DOC] Command line script used by the release workflow to rewrite the package version
# in the repository root setup file. It takes exactly one positional argument, the new version
# string, validates it against the release version pattern, and rewrites the version assignment
# in place. Exit status is zero on success and one on usage or validation errors.
"""Bump the miles version in setup.py.

Usage: python .github/scripts/release/bump_miles_version.py 0.3.0
"""

import re
import sys
from pathlib import Path

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:rc|\.post)\d+)?$")


def main() -> int:
    # [FUNCTION DOC] Entry point. Reads the new version from the single command line argument.
    # If the argument count is wrong, prints the module docstring (usage text) to standard error
    # and returns 1. If the version does not fully match the pattern of three dot-separated
    # integers optionally followed by a release-candidate suffix or a post-release suffix (each
    # with an integer), prints an error to standard error and returns 1. Otherwise locates the
    # setup file three directory levels above this script (the repository root), reads it,
    # replaces every quoted version assignment value with the new version, and asserts that
    # exactly one replacement happened (an assertion error otherwise, before anything is
    # written). Then overwrites the setup file, prints a confirmation line to standard output and
    # returns 0. Side effects: reads and writes the setup file on disk.
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 1
    new_version = sys.argv[1]
    if not VERSION_RE.fullmatch(new_version):
        print(f"Invalid version {new_version!r}; expected X.Y.Z, X.Y.ZrcN, or X.Y.Z.postN", file=sys.stderr)
        return 1

    setup_py = Path(__file__).resolve().parents[3] / "setup.py"
    content = setup_py.read_text()
    new_content, n = re.subn(r'version="[^"]+"', f'version="{new_version}"', content)
    assert n == 1, f"expected exactly one version= assignment in setup.py, found {n}"
    setup_py.write_text(new_content)
    print(f"setup.py: version -> {new_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
