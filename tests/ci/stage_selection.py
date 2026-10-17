# [MODULE DOC] Decides which GPU stages of a pull request can be skipped. Combines the set of
# stages that have runnable tests under the resolved policy with the stages affected by the changed
# files (registered test files map to their stages; known documentation-only paths affect nothing;
# any other path affects every stage) and the stages demanded by explicit domain labels. Parses
# the NUL-separated name-status git diff output. Constants list the pull-request GPU stages, broad
# scope labels and the paths, prefixes and suffixes known not to affect GPU runs.
"""Select PR GPU stages from changed paths and the resolved test policy."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tests.ci.ci_policy import RunPolicy, registration_matches_selection, strip_run_ci_prefix
from tests.ci.ci_register import CIRegistry
from tests.ci.labels import KNOWN_LABELS
from tests.ci.run_suite import runs_in_stage

# Only stages with jobs in the Miles PR workflows belong here; the external
# MI350 nightly suites are not local runner-allocation targets.
PR_GPU_STAGES = frozenset(
    {
        "stage-b-2-gpu-h200",
        "stage-c-2-gpu-h200",
        "stage-c-4-gpu-h200",
        "stage-c-8-gpu-h100",
        "stage-c-8-gpu-h200",
        "stage-c-8-gpu-b200",
        "stage-c-4-gpu-mi350",
    }
)

# Broad over domain labels; the resolved policy still restricts
# `run-ci-blackwell-only` to Blackwell stages.
_BROAD_PR_SCOPES = frozenset({"nightly", "run-ci-all", "run-ci-image", "run-ci-blackwell-only"})
# Keep this allowlist to paths proven not to affect GPU runtime. Every
# unmatched path deliberately fans out to all local GPU stages.
_NO_GPU_PATHS = frozenset({".pre-commit-config.yaml", "scripts/tools/sync_example_docs.py"})
_NO_GPU_PREFIXES = ("docs/",)
_NO_GPU_SUFFIXES = (".md", ".mdx")
_STATUS = re.compile(r"^(?:[ADMTUXB]|[RC][0-9]{1,3})$")


@dataclass(frozen=True)
class ChangedFile:
    # [CLASS DOC] Immutable record of one changed file entry: the git status code text and a tuple
    # of one path, or two (old and new) for renames and copies.
    status: str
    paths: tuple[str, ...]


def read_changed_files(path: str) -> tuple[ChangedFile, ...] | None:
    # [FUNCTION DOC] Reads the file (bytes) holding NUL-terminated fields: status, then one path (or
    # two for statuses starting with R or C), repeated. Returns a tuple of changed-file records, or
    # none if the path is empty, the file is unreadable, empty, not NUL-terminated, not UTF-8,
    # contains an invalid status (allowed: A D M T U X B, or R/C followed by 1 to 3 digits), is
    # truncated, has an empty path, or yields no records.
    """Parse `git diff --name-status -z`; return None when it is unusable."""
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if not data or not data.endswith(b"\0"):
        return None

    try:
        fields = [field.decode("utf-8") for field in data[:-1].split(b"\0")]
    except UnicodeDecodeError:
        return None

    changed_files: list[ChangedFile] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if _STATUS.fullmatch(status) is None:
            return None

        path_count = 2 if status[0] in {"R", "C"} else 1
        paths = tuple(fields[index : index + path_count])
        index += path_count
        if len(paths) != path_count or any(not changed_path for changed_path in paths):
            return None
        changed_files.append(ChangedFile(status=status, paths=paths))

    return tuple(changed_files) or None


def _stages_for(registration: CIRegistry, run_policy: RunPolicy) -> set[str]:
    # [FUNCTION DOC] HELPER: The set of pull-request GPU stages in which the registration executes
    # under the policy's dispatch arches and absorb flag (labels and disabled state ignored).
    """PR GPU stages this registration would execute in under `run_policy`.

    Dispatch can move a test off its declared suite, so this asks the same
    `runs_in_stage` predicate `run_suite` collects with. Without it a
    `run-on-blackwell` PR would still allocate every Hopper runner and then find
    nothing to run on them.
    """
    return {
        stage
        for stage in PR_GPU_STAGES
        if runs_in_stage(
            registration,
            stage,
            dispatch_arches=run_policy.dispatch_arches,
            absorb=run_policy.absorb,
        )
    }


def _runnable_stages(registrations: Iterable[CIRegistry], run_policy: RunPolicy) -> set[str]:
    # [FUNCTION DOC] HELPER: Union of the stages of all enabled registrations that pass the
    # selection predicate (nightly admission, labels in the policy's include set).
    stages: set[str] = set()
    for registration in registrations:
        if registration.disabled is not None:
            continue
        if not registration_matches_selection(
            registration.labels,
            registration.nightly,
            admit_nightly_tests=run_policy.admit_nightly_tests,
            include_labels=run_policy.include_labels,
        ):
            continue
        stages |= _stages_for(registration, run_policy)
    return stages


def _explicit_scope_stages(
    registrations: Iterable[CIRegistry],
    run_policy: RunPolicy,
    raw_labels: Iterable[str],
    runnable_stages: set[str],
) -> set[str]:
    # [FUNCTION DOC] HELPER: Stages demanded by explicit scope. If any raw label is a broad scope
    # (nightly, all, image, blackwell-only), returns a copy of the runnable stages. Otherwise
    # returns the union of stages of enabled registrations (nightly-only ones only if nightly tests
    # are admitted) that carry at least one explicitly requested known domain label.
    raw_labels = set(raw_labels)
    if raw_labels & _BROAD_PR_SCOPES:
        return set(runnable_stages)

    requested_labels = strip_run_ci_prefix(raw_labels) & set(KNOWN_LABELS)
    stages: set[str] = set()
    for registration in registrations:
        if registration.disabled is not None:
            continue
        if registration.nightly and not run_policy.admit_nightly_tests:
            continue
        if not set(registration.labels) & requested_labels:
            continue
        stages |= _stages_for(registration, run_policy)
    return stages


def _known_no_gpu_path(path: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the path is an allowlisted file, under the docs directory, or
    # ends with a Markdown or MDX suffix.
    return path in _NO_GPU_PATHS or path.startswith(_NO_GPU_PREFIXES) or path.endswith(_NO_GPU_SUFFIXES)


def _affected_stages(
    changed_files: Iterable[ChangedFile], registrations: Iterable[CIRegistry], run_policy: RunPolicy
) -> set[str]:
    # [FUNCTION DOC] HELPER: Stages affected by the changed files. Builds a map from each
    # registered file to its stages (all registrations, regardless of labels or disabled state).
    # For each changed path: a registered file adds its stages; a known no-GPU path adds nothing;
    # any other path makes every pull-request GPU stage affected (returned immediately).
    stages_by_file: dict[str, set[str]] = {}
    registered_files: set[str] = set()
    for registration in registrations:
        registered_files.add(registration.filename)
        for stage in PR_GPU_STAGES:
            if runs_in_stage(
                registration, stage, dispatch_arches=run_policy.dispatch_arches, absorb=run_policy.absorb
            ):
                stages_by_file.setdefault(registration.filename, set()).add(stage)

    affected: set[str] = set()
    for changed_file in changed_files:
        for path in changed_file.paths:
            if path in registered_files:
                affected.update(stages_by_file.get(path, ()))
            elif _known_no_gpu_path(path):
                continue
            else:
                return set(PR_GPU_STAGES)
    return affected


def select_skipped_gpu_stages(
    *,
    event_name: str,
    changed_files: tuple[ChangedFile, ...] | None,
    registrations: Iterable[CIRegistry],
    run_policy: RunPolicy,
    raw_labels: Iterable[str],
) -> tuple[str, ...]:
    # [FUNCTION DOC] Keyword-only. Returns the sorted tuple of GPU stage names that can be skipped,
    # or empty for non-pull-request events. Runnable stages come from the policy. If the changed
    # files are unavailable, skip every stage that is not runnable. Otherwise affected stages (from
    # changes) are united with the explicit-scope stages, and the skipped stages are all pull-request
    # GPU stages outside the intersection of runnable and affected stages.
    """Return local GPU stage IDs that a workflow can safely skip."""
    if event_name != "pull_request":
        return ()

    registrations = tuple(registrations)
    runnable = _runnable_stages(registrations, run_policy)
    if changed_files is None:
        return tuple(sorted(PR_GPU_STAGES - runnable))
    affected = _affected_stages(changed_files, registrations, run_policy)
    affected.update(_explicit_scope_stages(registrations, run_policy, raw_labels, runnable))
    return tuple(sorted(PR_GPU_STAGES - (runnable & affected)))
