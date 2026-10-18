# [MODULE DOC]
# HELPER: Utility for import-isolation tests. Reports which modules get loaded as a side effect of
# importing a given module, measured in a fresh interpreter subprocess so the test process's own
# imports do not pollute the answer. Constant: the repository root, two levels above this file's
# directory.
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def modules_imported_by(module: str) -> set[str]:
    # [FUNCTION DOC] HELPER: Runs the current python interpreter in a subprocess (working directory the
    # repo root) with a one-line program that imports the named module and prints the sorted names of
    # all loaded modules as a json list. Captures output as text; a nonzero exit raises a called-process
    # error. Parses the last output line (so earlier import-time prints are tolerated) and returns the
    # module names as a set, which includes the standard library and the named module itself.
    script = f"import json, sys; import {module}; print(json.dumps(sorted(sys.modules)))"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(completed.stdout.strip().splitlines()[-1]))
