# [MODULE DOC]
# Contract tests for tests.ci.ci_policy: the pure functions that map GitHub trigger facts (event
# name, cron schedule, PR labels as JSON) to a cadence (regular, nightly, weekly) and a run policy,
# and the module's command-line entry point that appends fixed key=value lines to the GitHub output
# file. Registers itself as a CPU test in suite stage-a-cpu. The command-line tests spawn the module
# as a subprocess from the repository root with the trigger facts supplied via environment variables.
"""Contract tests for the shared CI policy and GitHub trigger adapter."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.ci.ci_policy import (
    NIGHTLY_CADENCE,
    REGULAR_CADENCE,
    WEEKLY_CADENCE,
    resolve_policy,
    resolve_workflow_inputs,
)
from tests.ci.ci_register import register_cpu_ci
from tests.ci.stage_selection import PR_GPU_STAGES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.mark.parametrize(
    ("event_name", "schedule", "labels_json", "cadence", "raw_labels", "bypass_fastfail"),
    [
        ("pull_request", "", "[]", REGULAR_CADENCE, (), False),
        (
            "pull_request",
            "",
            json.dumps(["run-ci-megatron", "bypass-fastfail"]),
            REGULAR_CADENCE,
            ("run-ci-megatron", "bypass-fastfail"),
            True,
        ),
        (
            "pull_request",
            "",
            json.dumps(
                [
                    "ignored",
                    "run-ci-megatron",
                    "nightly",
                    "run-ci-megatron",
                    "run-ci-a_B.c-d",
                    "run-ci-/unsafe",
                ]
            ),
            NIGHTLY_CADENCE,
            ("run-ci-megatron", "nightly", "run-ci-megatron", "run-ci-a_B.c-d"),
            True,
        ),
        ("schedule", "0 15 * * 0-5", "not JSON", NIGHTLY_CADENCE, (), True),
        ("schedule", "0 15 * * 6", "not JSON", WEEKLY_CADENCE, (), True),
        ("workflow_dispatch", "", "not JSON", REGULAR_CADENCE, (), False),
    ],
)
def test_trigger_facts_resolve_to_stable_workflow_outputs(
    event_name,
    schedule,
    labels_json,
    cadence,
    raw_labels,
    bypass_fastfail,
):
    # [FUNCTION DOC] Parametrized over trigger facts: a plain pull request, a pull request with the
    # megatron run label and the bypass-fastfail label, a pull request whose labels include an
    # ignored label, a duplicate, the nightly label, a label with odd but valid characters and an
    # unsafe label, weekday and Saturday cron schedules, and a manual dispatch. Resolves the workflow
    # inputs and asserts the cadence, the raw labels kept (only run-ci-prefixed safe labels, plus
    # nightly, in original order with duplicates retained), the bypass-fastfail flag (true for the
    # label, for nightly and for schedules) and that no stages are skipped. JSON of labels is ignored
    # for non-pull-request events.
    policy = resolve_workflow_inputs(event_name, schedule, labels_json)

    assert policy.cadence == cadence
    assert policy.raw_labels == raw_labels
    assert policy.bypass_fastfail is bypass_fastfail
    assert policy.skipped_stages == ()


@pytest.mark.parametrize("labels_json", ["{", "{}", "null", '["run-ci-megatron", 1]'])
def test_pull_request_rejects_non_string_array_labels(labels_json):
    # [FUNCTION DOC] For a pull request event, labels JSON that is malformed, an object, null, or an
    # array containing a non-string must raise a value error saying the labels were not a JSON
    # string array.
    with pytest.raises(ValueError, match="PR labels were not a JSON string array"):
        resolve_workflow_inputs("pull_request", "", labels_json)


def test_unknown_schedule_is_not_assumed_to_be_nightly():
    # [FUNCTION DOC] A schedule event with an unrecognized cron string (Sunday midnight) must raise a
    # value error saying no CI policy is defined for that schedule, rather than defaulting to nightly.
    with pytest.raises(ValueError, match=r"No CI policy is defined for schedule: 0 0 \* \* 0"):
        resolve_workflow_inputs("schedule", "0 0 * * 0", "[]")


def test_unknown_trigger_is_rejected():
    # [FUNCTION DOC] A push event must raise a value error saying the PR Test trigger is
    # unsupported.
    with pytest.raises(ValueError, match="Unsupported PR Test trigger: push"):
        resolve_workflow_inputs("push", "", "[]")


def test_nightly_label_and_nightly_schedule_share_the_same_run_policy():
    # [FUNCTION DOC] Resolves inputs for a pull request carrying the nightly label and for the
    # weekday cron schedule, resolves each into a run policy from its cadence and raw label set, and
    # asserts the two policies are equal.
    labeled = resolve_workflow_inputs("pull_request", "", '["nightly"]')
    scheduled = resolve_workflow_inputs("schedule", "0 15 * * 0-5", "not JSON")

    assert resolve_policy(labeled.cadence, set(labeled.raw_labels)) == resolve_policy(
        scheduled.cadence, set(scheduled.raw_labels)
    )


def test_weekly_schedule_resolves_to_independent_full_policy():
    # [FUNCTION DOC] Resolves the Saturday cron schedule and its run policy; asserts the cadence is
    # weekly, nightly-only tests are admitted, fast-fail is bypassed, and the performance baseline
    # is written.
    scheduled = resolve_workflow_inputs("schedule", "0 15 * * 6", "not JSON")
    policy = resolve_policy(scheduled.cadence, set(scheduled.raw_labels))

    assert policy.cadence == WEEKLY_CADENCE
    assert policy.admit_nightly_tests is True
    assert policy.bypass_fastfail is True
    assert policy.write_baseline is True


@pytest.mark.parametrize(
    ("labels_json", "expected_outputs"),
    [
        (
            "[]",
            "existing=value\ncadence=regular\nraw_labels=\nbypass_fastfail=false\n"
            f"skipped_stages={json.dumps(sorted(PR_GPU_STAGES), separators=(',', ':'))}\n"
            "needs_cuda_image=false\n",
        ),
        (
            '["run-ci-megatron", "nightly", "run-ci-megatron", "ignored"]',
            "existing=value\ncadence=nightly\n"
            "raw_labels=run-ci-megatron nightly run-ci-megatron\n"
            "bypass_fastfail=true\n"
            "skipped_stages=[]\nneeds_cuda_image=true\n",
        ),
    ],
)
def test_cli_appends_exact_github_outputs(tmp_path, labels_json, expected_outputs):
    # [FUNCTION DOC] Pre-creates an output file containing one existing line, runs the policy module
    # as a subprocess from the repository root with the environment of a pull request event (labels
    # JSON from the parameter, empty schedule, cadence override and changed-files path). Asserts exit
    # code 0, that the output file is the original line plus appended lines cadence, raw_labels
    # (space-joined), bypass_fastfail, skipped_stages (compact JSON) and needs_cuda_image, exactly
    # as expected, and that stdout contains "Resolved CI policy:". With no labels the regular cadence
    # skips every PR GPU stage (sorted list) and needs no CUDA image; with the nightly label nothing
    # is skipped and a CUDA image is needed.
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    output_path.write_text("existing=value\n")
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "pull_request",
            "SCHEDULE": "",
            "PR_LABELS_JSON": labels_json,
            "CADENCE_OVERRIDE": "",
            "CHANGED_FILES_PATH": "",
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.read_text() == expected_outputs
    assert "Resolved CI policy:" in result.stdout


@pytest.mark.parametrize(
    ("event_name", "schedule", "cadence_override"),
    [
        ("schedule", "0 15 * * 0-5", ""),
        ("schedule", "0 15 * * 6", ""),
        ("workflow_dispatch", "", ""),
        ("push", "", "release"),
    ],
)
def test_non_pr_cli_preserves_image_preparation(tmp_path, event_name, schedule, cadence_override):
    # [FUNCTION DOC] Parametrized over non-pull-request events (nightly and weekly schedules, manual
    # dispatch, and a push with a release cadence override). Runs the policy module as a subprocess
    # from the repository root and asserts exit code 0, skipped_stages is an empty JSON list, and
    # needs_cuda_image is true in the parsed output file.
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        EVENT_NAME=event_name,
        SCHEDULE=schedule,
        CADENCE_OVERRIDE=cadence_override,
        PR_LABELS_JSON="[]",
        CHANGED_FILES_PATH="",
        GITHUB_OUTPUT=str(output_path),
    )
    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"], cwd=repo_root, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert outputs["skipped_stages"] == "[]"
    assert outputs["needs_cuda_image"] == "true"


def test_cli_fails_for_unknown_schedule(tmp_path):
    # [FUNCTION DOC] Runs the policy module as a subprocess for a schedule event with an unknown cron
    # string. Asserts exit code 1, stderr contains a GitHub error annotation saying no policy is
    # defined for that schedule, and the output file was never created.
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "schedule",
            "SCHEDULE": "0 0 * * 0",
            "PR_LABELS_JSON": "not JSON",
            "GITHUB_OUTPUT": str(output_path),
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "::error::No CI policy is defined for schedule: 0 0 * * 0" in result.stderr
    assert not output_path.exists()
