# [MODULE DOC] Pure policy layer for CI: converts trigger facts (event name, cron schedule, PR label
# JSON, optional cadence override) into a cadence (regular, nightly, weekly, release), a set of
# included test labels, and flags (admit nightly tests, bypass fast-fail, write the perf baseline,
# arches to dispatch to, whether tests may leave their home stage). Also provides the label
# normalization helpers, a registration-selection predicate, GitHub output file writing and a
# command-line main driven entirely by environment variables, which on pull requests also asks the
# stage-selection module which GPU stages can be skipped from the changed-file list. Constants
# define cadence names, the schedule table (cron string to cadence and labels), workflow-only
# labels and the Blackwell-only scope label.
"""Resolve CI cadence, scope, and fast-fail policy from explicit inputs."""

import json
import os
import re
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, replace

from tests.ci.hardware import CUDA_STAGES, KNOWN_ARCHES
from tests.ci.labels import KNOWN_LABELS

_RUN_CI_PREFIX = "run-ci-"
# `run-on-<arch>` lacks the `run-ci-` prefix, so membership here is what
# admits it past `_SAFE_RUN_CI_LABEL` and keeps `strip_run_ci_prefix` quiet.
# `run-ci-blackwell-only` matches the prefix regex already and, being absent
# from KNOWN_LABELS, is dropped from the domain set like any other scope label.
_WORKFLOW_ONLY_LABELS = {"nightly", "bypass-fastfail"} | {f"run-on-{arch}" for arch in KNOWN_ARCHES}

# Scope label selecting the tests that only run on Blackwell. Distinct from
# `run-on-blackwell`: this one runs the Blackwell-exclusive set, that one
# moves whatever it can onto Blackwell.
_BLACKWELL_ONLY_LABEL = "run-ci-blackwell-only"
_SAFE_RUN_CI_LABEL = re.compile(r"^run-ci-[A-Za-z0-9][A-Za-z0-9_.-]*$")

REGULAR_CADENCE = "regular"
NIGHTLY_CADENCE = "nightly"
WEEKLY_CADENCE = "weekly"
# Release-branch CI: weekly's full scope, but never writes the rolling perf
# baseline — release branches run frozen dependency SHAs, and letting them
# write baselines would poison the nightly comparisons.
RELEASE_CADENCE = "release"
CI_CADENCES = frozenset({REGULAR_CADENCE, NIGHTLY_CADENCE, WEEKLY_CADENCE, RELEASE_CADENCE})

# A scheduled trigger has no policy by itself. Each configured cron must map
# explicitly so a future cadence cannot silently inherit nightly behavior.
SCHEDULE_POLICIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "0 15 * * 0-5": (NIGHTLY_CADENCE, ()),
    "0 15 * * 6": (WEEKLY_CADENCE, ()),
}


@dataclass(frozen=True)
class RunPolicy:
    # [CLASS DOC] Immutable resolved policy of one CI run: cadence name, the frozen set of domain
    # labels whose tests are included, whether tests marked nightly are admitted, whether failures
    # skip the fast-fail gate, whether the run writes the rolling performance baseline, the arches
    # the run executes on (empty means each test uses its own default) and absorb (whether a test
    # may execute outside its home stage; set only by an explicit run-on label).
    cadence: str
    include_labels: frozenset[str]
    admit_nightly_tests: bool
    bypass_fastfail: bool
    write_baseline: bool
    # Arches this run executes on; empty means "each test on its own default".
    dispatch_arches: frozenset[str] = frozenset()
    # Whether a test may execute outside its home stage. Only an explicit
    # `run-on-*` sets this, so every other run selects exactly what it does today.
    absorb: bool = False


@dataclass(frozen=True)
class WorkflowPolicy:
    # [CLASS DOC] Immutable workflow-level outputs: cadence, the tuple of canonical raw labels,
    # the bypass-fast-fail flag, and the tuple of GPU stage names to skip.
    cadence: str
    raw_labels: tuple[str, ...]
    bypass_fastfail: bool
    skipped_stages: tuple[str, ...]


def strip_run_ci_prefix(raw_labels: Iterable[str]) -> set[str]:
    # [FUNCTION DOC] Returns the set of labels with the run-ci prefix removed. Empty strings and
    # workflow-only labels are skipped silently; any other label lacking the prefix is skipped with
    # a warning (attributed to the caller). Pure apart from warnings.
    """Strip the `run-ci-` prefix from each PR-side label.

    Inputs are the canonical PR-side CI label names forwarded by the workflow
    (e.g. `["run-ci-megatron", "nightly"]`). Empty input yields an empty set.
    Known workflow-only labels (`_WORKFLOW_ONLY_LABELS`) are consumed
    elsewhere and skipped silently; any other item missing the `run-ci-`
    prefix is skipped after a `warnings.warn(...)`, because silently
    including it would risk matching the wrong domain label (e.g. bare
    `"megatron"` colliding with a test's domain label by accident).
    """
    stripped: set[str] = set()
    for raw in raw_labels:
        if not raw or raw in _WORKFLOW_ONLY_LABELS:
            continue
        if raw.startswith(_RUN_CI_PREFIX):
            stripped.add(raw[len(_RUN_CI_PREFIX) :])
        else:
            warnings.warn(
                f"--labels entry {raw!r} is missing the expected {_RUN_CI_PREFIX!r} "
                f"prefix; ignoring. Domain labels must be raw `run-ci-<X>` strings.",
                stacklevel=2,
            )
    return stripped


def resolve_policy(cadence: str, raw_labels: set[str]) -> RunPolicy:
    # [FUNCTION DOC] Builds the run policy from a cadence name and a set of raw labels. Raises a
    # value error for an unknown cadence, or when the nightly label is present but the cadence is
    # not nightly. Requested domain labels are the prefix-stripped labels that are known labels.
    # Scope (first matching branch): the run-ci-all label, or weekly or release cadence, gives all
    # known labels; nightly gives all except long and ft-long; the Blackwell-only label gives all
    # known labels (the arch is the selection); run-ci-image gives all except long, ft-short and
    # ft-long; otherwise empty. Include labels are scope united with requested. Dispatch arches are
    # the known arches that have a run-on label; absorb is true iff there are any; if none but the
    # Blackwell-only label is present, dispatch arches become Blackwell alone with absorb false.
    # Flags: nightly tests admitted for nightly, weekly, release; bypass fast-fail for those
    # cadences or when the bypass label is present; the baseline is written only for nightly and
    # weekly (not release).
    """Resolve selection and within-stage failure behavior from explicit inputs.

    The workflow adapter resolves trigger-specific facts into a cadence and
    raw labels; this function never infers policy from a GitHub event name. A
    test runs iff it is cadence-eligible and declares no labels (the CPU
    always-on case) or any of its labels is in the effective include set. GPU
    registrations are validated separately to require a non-empty label set.

    Broad scopes are large include sets:

    - `run-ci-all` includes every registered label.
    - Weekly and release cadences include every registered label; release
      differs from weekly only in never writing the perf baseline.
    - Nightly cadence excludes `long` and `ft-long`.
    - `run-ci-image` excludes `long`, `ft-short`, and `ft-long`.

    Branch order encodes the precedence `run-ci-all` > weekly > nightly >
    `run-ci-image`.

    Explicitly requested `run-ci-<x>` labels are unioned in last, so an
    explicit request always wins over a scope subtraction. A subtraction is
    not a per-test veto: a test carrying a subtracted label still runs when
    another of its labels is included.
    """
    if cadence not in CI_CADENCES:
        raise ValueError(f"Unknown CI cadence {cadence!r}; expected one of {sorted(CI_CADENCES)}")
    if "nightly" in raw_labels and cadence != NIGHTLY_CADENCE:
        raise ValueError("The nightly workflow label requires cadence='nightly'")

    requested = strip_run_ci_prefix(raw_labels) & set(KNOWN_LABELS)
    blackwell_only = _BLACKWELL_ONLY_LABEL in raw_labels
    if "run-ci-all" in raw_labels or cadence in {WEEKLY_CADENCE, RELEASE_CADENCE}:
        scope = set(KNOWN_LABELS)
    elif cadence == NIGHTLY_CADENCE:
        scope = set(KNOWN_LABELS) - {"long", "ft-long"}
    elif blackwell_only:
        # The arch, not the domain, is the selection here.
        scope = set(KNOWN_LABELS)
    elif "run-ci-image" in raw_labels:
        scope = set(KNOWN_LABELS) - {"long", "ft-short", "ft-long"}
    else:
        scope = set()

    # An explicit `run-on-*` both picks the arches and permits a test to leave
    # its home stage. `run-ci-blackwell-only` picks an arch without permitting
    # that move, which is exactly what makes it mean "the Blackwell-exclusive
    # tests" rather than "everything that can run on Blackwell".
    dispatch_arches = frozenset(arch for arch in KNOWN_ARCHES if f"run-on-{arch}" in raw_labels)
    absorb = bool(dispatch_arches)
    if not dispatch_arches and blackwell_only:
        dispatch_arches = frozenset({"blackwell"})

    full_cadences = {NIGHTLY_CADENCE, WEEKLY_CADENCE, RELEASE_CADENCE}
    return RunPolicy(
        cadence=cadence,
        include_labels=frozenset(scope | requested),
        admit_nightly_tests=cadence in full_cadences,
        bypass_fastfail=cadence in full_cadences or "bypass-fastfail" in raw_labels,
        write_baseline=cadence in {NIGHTLY_CADENCE, WEEKLY_CADENCE},
        dispatch_arches=dispatch_arches,
        absorb=absorb,
    )


def registration_matches_selection(
    labels: Iterable[str],
    nightly: bool,
    *,
    admit_nightly_tests: bool,
    include_labels: Iterable[str],
) -> bool:
    # [FUNCTION DOC] Returns false if the registration is nightly-only and nightly tests are not
    # admitted; otherwise true if the registration declares no labels (always on) or any of its
    # labels is in the include set.
    """Return whether one registration is selected by cadence and labels."""
    if nightly and not admit_nightly_tests:
        return False
    labels = set(labels)
    return not labels or bool(labels & set(include_labels))


def _canonical_pr_labels(pr_labels_json: str) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Parses a JSON array of strings and keeps, in order, the labels that
    # are workflow-only labels or fully match the safe run-ci label pattern (alphanumeric start,
    # then letters, digits, underscore, dot, dash). Raises a value error when the text is not valid
    # JSON or not an array of strings.
    try:
        labels = json.loads(pr_labels_json)
    except json.JSONDecodeError as exc:
        raise ValueError("PR labels were not a JSON string array") from exc
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("PR labels were not a JSON string array")

    return tuple(
        label for label in labels if label in _WORKFLOW_ONLY_LABELS or _SAFE_RUN_CI_LABEL.fullmatch(label) is not None
    )


def resolve_workflow_inputs(
    event_name: str, schedule: str, pr_labels_json: str, cadence_override: str = ""
) -> WorkflowPolicy:
    # [FUNCTION DOC] Maps trigger facts to a workflow policy. Precedence: a non-empty cadence
    # override (no labels) wins; else a pull request event uses the canonical PR labels and gets
    # nightly cadence if the nightly label is present, otherwise regular; a schedule event looks
    # the cron string up in the schedule table (value error if absent); a manual dispatch is
    # regular with no labels; any other event raises a value error. The bypass flag comes from
    # resolving the run policy for that cadence and labels. Skipped stages are always empty here.
    """Adapt GitHub trigger facts to the workflow's stable policy outputs.

    `cadence_override` carries an explicit workflow_call cadence input (e.g. a
    release-branch cut requesting a full-scope `release` run). It must win over
    trigger inference: a called workflow inherits the *caller's* event_name, so
    inferring from the trigger would silently degrade a release run to
    `regular` scope.
    """
    if cadence_override:
        cadence, raw_labels = cadence_override, ()
    elif event_name == "pull_request":
        raw_labels = _canonical_pr_labels(pr_labels_json)
        cadence = NIGHTLY_CADENCE if "nightly" in raw_labels else REGULAR_CADENCE
    elif event_name == "schedule":
        try:
            cadence, raw_labels = SCHEDULE_POLICIES[schedule]
        except KeyError as exc:
            raise ValueError(f"No CI policy is defined for schedule: {schedule}") from exc
    elif event_name == "workflow_dispatch":
        cadence, raw_labels = REGULAR_CADENCE, ()
    else:
        raise ValueError(f"Unsupported PR Test trigger: {event_name}")

    run_policy = resolve_policy(cadence, set(raw_labels))
    return WorkflowPolicy(
        cadence=cadence,
        raw_labels=raw_labels,
        bypass_fastfail=run_policy.bypass_fastfail,
        skipped_stages=(),
    )


def _write_github_outputs(policy: WorkflowPolicy, output_path: str) -> None:
    # [FUNCTION DOC] HELPER: Appends key=value lines to the GitHub output file at the path: cadence,
    # space-joined raw labels, bypass flag as lower-case true/false, skipped stages as compact JSON
    # array, and needs_cuda_image (true iff some CUDA stage is not in the skipped list). Also prints
    # a one-line summary to standard output. Side effect: file append (UTF-8).
    raw_labels = " ".join(policy.raw_labels)
    bypass_fastfail = str(policy.bypass_fastfail).lower()
    skipped_stages = json.dumps(policy.skipped_stages, separators=(",", ":"))
    needs_cuda_image = str(bool(CUDA_STAGES.keys() - set(policy.skipped_stages))).lower()
    with open(output_path, "a", encoding="utf-8") as output:
        output.write(f"cadence={policy.cadence}\n")
        output.write(f"raw_labels={raw_labels}\n")
        output.write(f"bypass_fastfail={bypass_fastfail}\n")
        output.write(f"skipped_stages={skipped_stages}\n")
        output.write(f"needs_cuda_image={needs_cuda_image}\n")
    print(
        f"Resolved CI policy: cadence={policy.cadence} labels=[{raw_labels}] "
        f"bypass_fastfail={bypass_fastfail} skipped_stages={skipped_stages}"
    )


def main() -> int:
    # [FUNCTION DOC] Command-line entry returning an exit code; configured only through environment
    # variables: event name (required, key error if missing), schedule, PR labels JSON, cadence
    # override, changed-files path, and the output file path (required at the end). Resolves the
    # workflow policy. For pull request events it additionally (imports deferred to here) reads the
    # changed-file list (printing a notice if the diff is unavailable or empty, in which case all
    # selected GPU stages count as affected), resolves the run policy, collects and sanity-checks
    # all registered tests, asks stage selection which GPU stages to skip, and stores them in the
    # policy. A value error anywhere in that block prints a GitHub error annotation to standard
    # error and returns 1. On success writes the outputs and returns 0.
    try:
        event_name = os.environ["EVENT_NAME"]
        policy = resolve_workflow_inputs(
            event_name=event_name,
            schedule=os.environ.get("SCHEDULE", ""),
            pr_labels_json=os.environ.get("PR_LABELS_JSON", ""),
            cadence_override=os.environ.get("CADENCE_OVERRIDE", ""),
        )
        if event_name == "pull_request":
            from tests.ci.ci_register import collect_tests, discover_ci_files
            from tests.ci.stage_selection import read_changed_files, select_skipped_gpu_stages

            changed_files = read_changed_files(os.environ.get("CHANGED_FILES_PATH", ""))
            if changed_files is None:
                print("Changed-file diff is unavailable or empty; all selected GPU stages are treated as affected.")
            run_policy = resolve_policy(policy.cadence, set(policy.raw_labels))
            skipped_stages = select_skipped_gpu_stages(
                event_name=event_name,
                changed_files=changed_files,
                registrations=collect_tests(discover_ci_files(), sanity_check=True),
                run_policy=run_policy,
                raw_labels=policy.raw_labels,
            )
            policy = replace(policy, skipped_stages=skipped_stages)
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1

    _write_github_outputs(policy, os.environ["GITHUB_OUTPUT"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
