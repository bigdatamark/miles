# [MODULE DOC] Resolves a single explicitly requested test file (the rerun-test command) into an
# execution plan of hardware kind, suite, runner labels, container image and timeout, for the
# resolve job of the file-run workflow. Parses a source snapshot as data (no imports of its
# modules), refuses symlinks in the discovery roots and label registry, reads the label table by
# literal evaluation, and requires exactly one enabled CPU or CUDA registration. The main function
# reads environment variables, writes the plan to the GitHub output file and prints it as JSON.
# Constraint: only standard library and the dependency-free registry modules may be imported.
"""Resolve one explicitly requested test file to its CI execution plan.

Consumed by the `resolve-file-run` job of `.github/workflows/run-ci-file.yml`,
which runs on a bare hosted runner before any dependency install; this module
may import only the stdlib and the dependency-free registry modules.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

from tests.ci.ci_register import HWBackend, collect_tests

CPU_SUITES = frozenset({"stage-a-cpu", "stage-b-cpu"})
DISCOVERY_ROOTS = ("tests/fast", "tests/fast-gpu", "tests/e2e", "tests/ci")

# Runner labels per CUDA suite, mirroring the pr-test.yml job wiring.
# `tests/ci/test/test_file_run.py` locks the key set to
# `run_suite.CI_SUITES[HWBackend.CUDA]` so a new suite cannot ship unmapped.
CUDA_SUITE_RUNS_ON = {
    "stage-b-2-gpu-h200": ["h200", "2gpu"],
    "stage-c-8-gpu-h100": ["h100", "8gpu"],
    "stage-c-8-gpu-h200": ["h200", "8gpu"],
    "stage-c-4-gpu-h200": ["h200", "4gpu"],
    "stage-c-2-gpu-h200": ["h200", "2gpu"],
    "stage-c-8-gpu-b200": ["b200", "8gpu"],
}

# Same shape the pr-test.yml resolve-ci-image step enforces for a Docker tag.
_IMAGE_TAG_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")


class FileRunError(Exception):
    # [CLASS DOC] Error for any reason an explicit file run cannot be planned; the main function
    # prints it as a GitHub error annotation and returns exit code 1.
    pass


def _discover_regular_ci_files() -> list[str]:
    # [FUNCTION DOC] HELPER: Walks each discovery root (relative to the current directory) without
    # following symlinks and returns the sorted POSIX-style paths of files named test_*.py. Raises
    # a file-run error if any component of a root, any scanned directory, or any matched file is a
    # symlink. A missing root yields no files (the walk silently ignores it).
    """Discover test files without following symlinks in scanned roots."""
    files = []
    for root in DISCOVERY_ROOTS:
        component = Path()
        for part in Path(root).parts:
            component /= part
            if component.is_symlink():
                raise FileRunError(f"CI discovery path must not be a symlink: {component.as_posix()}")
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            for name in directory_names:
                path = Path(directory, name)
                if path.is_symlink():
                    raise FileRunError(f"CI test directory must not be a symlink: {path.as_posix()}")
            for name in file_names:
                if not name.startswith("test_") or not name.endswith(".py"):
                    continue
                path = Path(directory, name)
                if path.is_symlink():
                    raise FileRunError(f"CI test file must not be a symlink: {path.as_posix()}")
                files.append(path.as_posix())
    return sorted(files)


def plan_file_run(all_tests, test_file: str, image_tag: str) -> dict[str, str]:
    # [FUNCTION DOC] Builds the plan mapping (all string values) for the file. Raises a file-run
    # error when: the image tag does not fully match the Docker tag pattern; the file has no
    # registration; it has only ROCm registrations; it has more than one CPU/CUDA registration; it
    # is disabled (message includes the reason); a CUDA suite has no runner mapping; or a CPU suite
    # is not one of the two allowed CPU stages. Result keys: hw ("cuda" or "cpu"), suite, runs_on
    # (JSON list of runner labels for CUDA, empty string for CPU), container_image (repository
    # prefix plus tag) and timeout_seconds (the larger of 1800 and int(1.25 x estimate)).
    """Map one registered test file to the hw/suite/runner/image that runs it.

    An explicit file request is the selection, so domain labels and the
    nightly cadence gate do not apply; only `disabled` still blocks the run.
    Anything that cannot resolve to exactly one enabled CPU or CUDA
    registration is a hard error rather than a silent no-op.
    """
    if _IMAGE_TAG_PATTERN.fullmatch(image_tag) is None:
        raise FileRunError(f"invalid CI image tag: {image_tag!r}")
    registrations = [t for t in all_tests if t.filename == test_file]
    if not registrations:
        raise FileRunError(f"{test_file} has no CI registration; /rerun-test runs only registered test files")
    supported = [t for t in registrations if t.backend in (HWBackend.CPU, HWBackend.CUDA)]
    if not supported:
        suites = ", ".join(sorted(t.suite for t in registrations))
        raise FileRunError(f"{test_file} is registered only for ROCm ({suites}); /rerun-test supports CPU and CUDA")
    if len(supported) > 1:
        suites = ", ".join(sorted(f"{t.backend.name}:{t.suite}" for t in supported))
        raise FileRunError(f"{test_file} has multiple CPU/CUDA registrations ({suites}); expected exactly one")
    registration = supported[0]
    if registration.disabled is not None:
        raise FileRunError(f"{test_file} is disabled: {registration.disabled}")

    if registration.backend is HWBackend.CUDA:
        runs_on = CUDA_SUITE_RUNS_ON.get(registration.suite)
        if runs_on is None:
            raise FileRunError(f"CUDA suite {registration.suite} has no runner mapping in CUDA_SUITE_RUNS_ON")
        hw = "cuda"
        runs_on_json = json.dumps(runs_on)
    else:
        if registration.suite not in CPU_SUITES:
            raise FileRunError(f"CPU suite {registration.suite} is not allowed for an explicit file run")
        hw = "cpu"
        runs_on_json = ""
    return {
        "hw": hw,
        "suite": registration.suite,
        "runs_on": runs_on_json,
        "container_image": f"radixark/miles:{image_tag}",
        "timeout_seconds": str(max(1800, int(registration.est_time * 1.25))),
    }


def _read_snapshot_labels() -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Reads the label registry file (relative to the current directory)
    # without importing it: parses the syntax tree, requires exactly one annotated assignment to
    # the known-labels name and evaluates its value as a literal. Raises a file-run error if the
    # file is a symlink, unreadable or unparsable, the definition is missing or duplicated, or the
    # value is not a dictionary of strings to strings. Returns the dictionary.
    path = Path("tests/ci/labels.py")
    if path.is_symlink():
        raise FileRunError(f"CI label registry must not be a symlink: {path}")
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
        definitions = [
            node.value
            for node in tree.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "KNOWN_LABELS"
        ]
        if len(definitions) != 1:
            raise FileRunError(f"{path} must define KNOWN_LABELS once as a literal dictionary")
        labels = ast.literal_eval(definitions[0])
    except (OSError, SyntaxError, ValueError, TypeError) as error:
        raise FileRunError(f"cannot read CI label registry {path}: {error}") from error
    if not isinstance(labels, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in labels.items()
    ):
        raise FileRunError(f"{path}: KNOWN_LABELS must be a literal string-to-string dictionary")
    return labels


def collect_snapshot_tests(source_root: str | Path):
    # [FUNCTION DOC] Resolves the source root (must exist and be a directory, else a file-run
    # error), temporarily changes the process working directory to it, discovers test files and
    # collects registrations with sanity checking and the snapshot's own label table, and restores
    # the previous working directory even on error. Returns the registration list. Side effect:
    # temporary change of the working directory (not thread-safe).
    """Parse a source snapshot as data, without importing its Python modules."""
    try:
        root = Path(source_root).resolve(strict=True)
    except OSError as error:
        raise FileRunError(f"cannot resolve CI source root {source_root}: {error}") from error
    if not root.is_dir():
        raise FileRunError(f"CI source root is not a directory: {root}")

    previous_directory = Path.cwd()
    try:
        os.chdir(root)
        files = _discover_regular_ci_files()
        return collect_tests(files, sanity_check=True, known_labels=_read_snapshot_labels())
    finally:
        os.chdir(previous_directory)


def resolve_file_run(test_file: str, image_tag: str, source_root: str | Path = ".") -> dict[str, str]:
    # [FUNCTION DOC] Convenience composition: collects the snapshot's registrations and plans the
    # run for the test file and image tag; returns the plan mapping.
    return plan_file_run(collect_snapshot_tests(source_root), test_file, image_tag)


def main() -> int:
    # [FUNCTION DOC] Entry point driven by environment variables: test file (required), image tag
    # (default dev when unset or empty) and source root (default current directory). On a file-run
    # error prints an error annotation to standard error and returns 1. On success appends one
    # key=value line per plan entry to the GitHub output file (required variable) and prints the
    # plan as key-sorted JSON; returns 0.
    try:
        plan = resolve_file_run(
            os.environ["TEST_FILE"],
            os.environ.get("CI_IMAGE_TAG") or "dev",
            os.environ.get("CI_SOURCE_ROOT") or ".",
        )
    except FileRunError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        for key, value in plan.items():
            output.write(f"{key}={value}\n")
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
