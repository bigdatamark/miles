# [MODULE DOC]
# Tests for the planning mode of tests/ci/run_suite.py (list-only with a GitHub output file), which
# decides whether a GPU stage has any test to run so that an empty shard never requests a GPU
# runner. Plans are produced by running the script as a subprocess in a temporary directory
# containing at most one generated test file (which raises if imported, proving planning only
# parses it). Also checks that the GPU reusable workflows (CUDA and ROCm) plan on a hosted
# runner and gate the run job on the plan. SUITES maps hardware kind to the sample suite used
# (cuda: stage-c-4-gpu-h200, rocm: stage-c-4-gpu-mi350). Registers itself as a CPU test (est_time 2)
# in suite stage-a-cpu.
"""GPU allocation plans must select a nonempty shard without importing tests."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[3]
SUITES = {"cuda": "stage-c-4-gpu-h200", "rocm": "stage-c-4-gpu-mi350"}


def _plan(tmp_path, hw, selection_args, *, registration_options="", empty=False):
    # [FUNCTION DOC] HELPER: Run the suite runner in list-only mode for the hardware kind and its
    # sample suite, writing the plan to a pre-seeded GitHub output file (containing "existing=value"),
    # with extra selection arguments. Unless empty is set, first creates tests/e2e/test_selected.py
    # in the temporary directory: it imports the register function for the hardware kind, registers
    # with est_time 1, the sample suite, label megatron, (for CUDA) hopper and blackwell hardware,
    # plus the extra registration option text, then raises an assertion error so any import or
    # execution is detected. The runner is invoked with the interpreter's site import disabled,
    # the repository on the module path and the temporary directory as working directory.
    # Returns (completed process, mapping parsed from the output file lines).
    if not empty:
        test_file = tmp_path / "tests/e2e/test_selected.py"
        test_file.parent.mkdir(parents=True)
        hardware = ', hardware=["hopper", "blackwell"]' if hw == "cuda" else ""
        test_file.write_text(
            f"from tests.ci.ci_register import register_{hw}_ci\n"
            f'register_{hw}_ci(1, "{SUITES[hw]}", labels=["megatron"]{hardware}{registration_options})\n'
            'raise AssertionError("Planning must never import or execute this test")\n'
        )
    output = tmp_path / "github-output"
    output.write_text("existing=value\n")
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tests/ci/run_suite.py"),
            "--hw",
            hw,
            "--suite",
            SUITES[hw],
            "--list-only",
            "--github-output",
            str(output),
            *selection_args,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    return result, dict(line.split("=", 1) for line in output.read_text().splitlines())


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
@pytest.mark.parametrize(
    ("options", "selection_args", "has_tests"),
    [
        ("", [], False),
        ("", ["--labels", "run-ci-fsdp"], False),
        ("", ["--labels", "run-ci-megatron"], True),
        ("", ["--match-all-labels"], True),
        (', disabled="flaky"', ["--match-all-labels"], False),
        (", nightly=True", ["--match-all-labels"], False),
        (", nightly=True", ["--cadence", "nightly"], True),
        (", nightly=True", ["--cadence", "weekly"], True),
        (", nightly=True", ["--cadence", "release"], True),
        (', disabled="flaky"', ["--cadence", "weekly"], False),
    ],
)
def test_plan_applies_selection_without_runtime_dependencies(tmp_path, hw, options, selection_args, has_tests):
    # [FUNCTION DOC] Parametrized over hardware kind (cuda, rocm) and registration options /
    # selection arguments: no filters or a non-matching label selects nothing; the matching label or
    # match-all-labels selects the test; a disabled registration never; a nightly registration only
    # for nightly, weekly or release cadence. Asserts exit code 0, the output file has exactly the
    # pre-seeded line plus has_tests set to the expected true/false, and standard output says "No
    # tests to run." exactly when none are selected.
    result, output = _plan(tmp_path, hw, selection_args, registration_options=options)

    assert result.returncode == 0, result.stderr
    assert output == {"existing": "value", "has_tests": str(has_tests).lower()}
    assert ("No tests to run." in result.stdout) is not has_tests


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
@pytest.mark.parametrize("cadence", ["regular", "nightly", "weekly", "release"])
def test_empty_suite_never_requests_a_gpu(tmp_path, hw, cadence):
    # [FUNCTION DOC] With no test files at all, for each hardware kind and each cadence, planning
    # with match-all-labels exits 0 and reports has_tests false.
    result, output = _plan(tmp_path, hw, ["--cadence", cadence, "--match-all-labels"], empty=True)

    assert result.returncode == 0, result.stderr
    assert output["has_tests"] == "false"


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
@pytest.mark.parametrize("cadence", ["regular", "nightly", "weekly", "release"])
@pytest.mark.parametrize("partition", [0, 1, 2])
def test_plan_checks_each_shard_after_partitioning(tmp_path, hw, cadence, partition):
    # [FUNCTION DOC] With one test and an automatic partition into 3 shards, planning for shard 0, 1
    # or 2 exits 0 and reports has_tests true only for shard 0 (the one holding the test), for every
    # hardware kind and cadence.
    result, output = _plan(
        tmp_path,
        hw,
        [
            "--cadence",
            cadence,
            "--match-all-labels",
            "--auto-partition-id",
            str(partition),
            "--auto-partition-size",
            "3",
        ],
    )

    assert result.returncode == 0, result.stderr
    assert output["has_tests"] == str(partition == 0).lower()


def test_arch_dispatch_does_not_allocate_the_tests_original_runner(tmp_path):
    # [FUNCTION DOC] Planning the hopper CUDA stage with the megatron label plus run-on-blackwell
    # (the portable test is dispatched to the blackwell stage) exits 0 and reports has_tests false for
    # this stage.
    result, output = _plan(tmp_path, "cuda", ["--labels", "run-ci-megatron", "run-on-blackwell"])

    assert result.returncode == 0, result.stderr
    assert output["has_tests"] == "false"


@pytest.mark.parametrize("hw", ["cuda", "rocm"])
def test_invalid_registration_fails_without_an_empty_success_plan(tmp_path, hw):
    # [FUNCTION DOC] A registration with nightly given as the integer 1 makes planning exit
    # non-zero with "nightly must be a boolean" on stderr and write no has_tests key (no false
    # success plan), for both hardware kinds.
    result, output = _plan(tmp_path, hw, ["--match-all-labels"], registration_options=", nightly=1")

    assert result.returncode != 0
    assert "nightly must be a boolean" in result.stderr
    assert "has_tests" not in output


def test_selection_output_cannot_be_requested_while_executing_tests(tmp_path):
    # [FUNCTION DOC] Runs the suite runner as a module from the repository root with a GitHub output
    # path but without list-only; asserts a non-zero exit, stderr saying the output option requires
    # list-only, and that the output file was not created.
    output = tmp_path / "github-output"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "tests.ci.run_suite",
            "--hw",
            "cuda",
            "--suite",
            SUITES["cuda"],
            "--github-output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--github-output requires --list-only." in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("workflow", ["_run-ci.yml", "_run-ci-rocm.yml"])
def test_gpu_workflow_requires_a_successful_hosted_plan(workflow):
    # [FUNCTION DOC] Parses the CUDA and ROCm reusable workflow files and asserts: the plan job runs
    # on a hosted ubuntu runner without a container, with read-only contents permission, exposes
    # has_tests and ref outputs from the select step; its checkout uses the input ref without
    # persisted credentials; the select step runs the execute command with list-only and the GitHub
    # output file and records the head commit as ref; the run job needs the plan, runs only when the
    # plan succeeded and has tests, has no list-only step, and checks out the plan's ref. For the
    # CUDA workflow only: the plan job is skipped when the plan is already resolved, the run
    # condition accepts an already-resolved plan, is not cancelled, and the checkout ref uses the
    # input ref when the plan is already resolved.
    jobs = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())["jobs"]
    plan, run = jobs["plan"], jobs["run"]

    assert plan["runs-on"] == "ubuntu-latest"
    assert "container" not in plan
    assert plan["permissions"] == {"contents": "read"}
    assert plan["outputs"]["has_tests"] == "${{ steps.select.outputs.has_tests }}"
    assert plan["outputs"]["ref"] == "${{ steps.select.outputs.ref }}"
    checkout = next(step for step in plan["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["ref"] == "${{ inputs.ref }}"
    assert checkout["with"]["persist-credentials"] is False
    select = next(step for step in plan["steps"] if step.get("id") == "select")
    assert '${{ inputs.execute_command }} --list-only --github-output "$GITHUB_OUTPUT"' in select["run"]
    assert 'echo "ref=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"' in select["run"]
    assert run["needs"] == "plan"
    assert "needs.plan.result == 'success'" in run["if"]
    assert "needs.plan.outputs.has_tests == 'true'" in run["if"]
    assert not any("--list-only" in step.get("run", "") for step in run["steps"])
    checkout = next(step for step in run["steps"] if step.get("uses", "").startswith("actions/checkout@"))
    assert "needs.plan.outputs.ref" in checkout["with"]["ref"]
    if workflow == "_run-ci.yml":
        assert plan["if"] == "${{ !inputs.plan_already_resolved }}"
        assert "inputs.plan_already_resolved ||" in run["if"]
        assert "!cancelled()" in run["if"]
        assert "inputs.plan_already_resolved && inputs.ref" in checkout["with"]["ref"]
