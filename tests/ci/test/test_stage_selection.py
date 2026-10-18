# [MODULE DOC]
# Tests of the PR stage-selection logic that maps changed paths, labels and registered tests to
# the set of GPU stages that can be skipped. Covers: the NUL-separated changed-file reader
# (renames keep both ends, odd file names survive, unusable input yields nothing so selection
# fails open); docs, tooling and CPU-only changes skip every GPU stage; changed registered
# tests, domain labels, broad scope labels and the blackwell-only scope keep exactly their
# stages; unknown source paths, renames from them, non-PR events and missing diffs keep every
# runnable stage (nothing skipped); the policy module as a command-line program with a
# temporary tree of test registrations (skipped stages and the CUDA-image-needed output); and
# static checks of the PR workflow (docker-build and image resolver gating, image demand
# output, ROCm resolver conditions). Registers itself as a CPU test in the stage-a-cpu suite.
"""Tests for PR changed-path to GPU-stage selection."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from tests.ci.ci_policy import REGULAR_CADENCE, resolve_policy
from tests.ci.ci_register import CIRegistry, HWBackend, register_cpu_ci
from tests.ci.stage_selection import PR_GPU_STAGES, ChangedFile, read_changed_files, select_skipped_gpu_stages

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

WORKFLOW = Path(__file__).resolve().parents[3] / ".github/workflows/pr-test.yml"


def test_a_cpu_failure_cannot_skip_the_gpu_stages_a_nightly_asked_for():
    # [FUNCTION DOC] Splits the PR workflow text into top-level job blocks by regular expression
    # and, for the docker-build and resolve-ci-image jobs, asserts each block has a condition and
    # mentions always(), so a failing upstream CPU job cannot skip them (which would skip all GPU
    # stages despite the fast-fail bypass).
    """bypass_fastfail is defeated by any job on the path that inherits the default success().

    A job without `if` is gated on its whole dependency closure, so on 2026-09-10 a
    stage-a-cpu failure skipped resolve-ci-image and with it all seven GPU stages, even
    though the nightly policy had set bypass_fastfail=true.
    """
    workflow = WORKFLOW.read_text()
    blocks = dict(re.findall(r"\n  ([a-z][a-z0-9-]*):\n(.*?)(?=\n  [a-z][a-z0-9-]*:\n|\Z)", workflow, re.S))
    for job in ("docker-build", "resolve-ci-image"):
        block = blocks[job]
        assert "if:" in block, f"{job} inherits success() over the closure, defeating bypass_fastfail"
        assert "always()" in block, f"{job} must not be gated on an unrelated upstream failure"


def _registration(
    filename: str,
    suite: str,
    *,
    labels: list[str] | None = None,
    disabled: str | None = None,
    hardware: list[str] | None = None,
) -> CIRegistry:
    # [FUNCTION DOC] HELPER: Build a registry record with estimated time 1. Backend is ROCm for the
    # MI350 suite, else CUDA. Labels default to the precision label. For CUDA with no hardware
    # given, hardware defaults to Blackwell for suites ending in b200 and Hopper otherwise
    # (mirroring the home-stage invariant); ROCm gets an empty list.
    backend = HWBackend.ROCM if suite == "stage-c-4-gpu-mi350" else HWBackend.CUDA
    if hardware is None and backend is HWBackend.CUDA:
        # Mirror the home-stage invariant the registry enforces.
        hardware = ["blackwell"] if suite.endswith("-b200") else ["hopper"]
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=1,
        suite=suite,
        labels=["precision"] if labels is None else labels,
        hardware=hardware or [],
        disabled=disabled,
    )


def _all_runnable_registrations() -> list[CIRegistry]:
    # [FUNCTION DOC] HELPER: Return one default registration per PR GPU stage, with file names
    # derived from the stage name.
    return [_registration(f"tests/e2e/{stage}.py", stage) for stage in PR_GPU_STAGES]


def _select(
    changed_files: tuple[ChangedFile, ...] | None,
    registrations: list[CIRegistry],
    *,
    raw_labels: tuple[str, ...] = ("run-ci-precision",),
    event_name: str = "pull_request",
) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Call the skipped-stage selector with the event name (default pull
    # request), changed files (or nothing), registrations, a regular-cadence policy resolved from
    # the raw labels (default the precision label) and the raw labels. Returns the tuple of
    # skipped stage names.
    return select_skipped_gpu_stages(
        event_name=event_name,
        changed_files=changed_files,
        registrations=registrations,
        run_policy=resolve_policy(REGULAR_CADENCE, set(raw_labels)),
        raw_labels=raw_labels,
    )


def test_read_changed_files_preserves_rename_ends_and_special_names(tmp_path):
    # [FUNCTION DOC] Writes NUL-separated diff bytes with a modified file whose name contains a
    # newline and a rename record with both old and new paths; asserts the reader returns a
    # modification record with one path and a rename record (status R100) with both paths.
    diff_path = tmp_path / "changed-files.z"
    diff_path.write_bytes(b"M\0docs/line\nname.md\0R100\0docs/old.md\0docs/new.md\0")

    assert read_changed_files(str(diff_path)) == (
        ChangedFile(status="M", paths=("docs/line\nname.md",)),
        ChangedFile(status="R100", paths=("docs/old.md", "docs/new.md")),
    )


@pytest.mark.parametrize("payload", [b"", b"M\0docs/index.md", b"Q\0docs/index.md\0", b"M\0\xff\0"])
def test_read_changed_files_fails_open_on_unusable_input(tmp_path, payload):
    # [FUNCTION DOC] Parametrized over an empty file, a record missing its final separator, an
    # unknown status code, and invalid UTF-8 bytes. The reader returns nothing (fail open).
    diff_path = tmp_path / "changed-files.z"
    diff_path.write_bytes(payload)

    assert read_changed_files(str(diff_path)) is None


def test_docs_tooling_and_cpu_only_tests_skip_every_gpu_stage():
    # [FUNCTION DOC] With a GPU registration and a CPU registration, and changes touching only
    # docs, examples readme, the pre-commit config, a docs-sync tool script and the CPU test
    # itself (no labels), asserts all PR GPU stages are skipped.
    cpu_test = "tests/fast/doc/test_sync_example_docs.py"
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        CIRegistry(HWBackend.CPU, cpu_test, 1, "stage-a-cpu"),
    ]
    changed_files = (
        ChangedFile("M", ("docs/developer/ci/00-stage.md",)),
        ChangedFile("M", ("examples/README.md",)),
        ChangedFile("M", (".pre-commit-config.yaml",)),
        ChangedFile("M", ("scripts/tools/sync_example_docs.py",)),
        ChangedFile("M", (cpu_test,)),
    )

    assert set(_select(changed_files, registrations, raw_labels=())) == PR_GPU_STAGES


def test_changed_multi_backend_test_and_explicit_domain_run_their_stages():
    # [FUNCTION DOC] A test file registered in both a CUDA H100 stage and the ROCm MI350 stage is
    # the only changed file, plus a precision test in the stage-b 2-GPU stage (default label).
    # Asserts the skipped set is all stages except those three (the precision label is explicit).
    shared_test = "tests/e2e/test_shared.py"
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration(shared_test, "stage-c-8-gpu-h100"),
        _registration(shared_test, "stage-c-4-gpu-mi350"),
    ]

    skipped = set(_select((ChangedFile("M", (shared_test,)),), registrations))

    assert skipped == PR_GPU_STAGES - {
        "stage-b-2-gpu-h200",
        "stage-c-8-gpu-h100",
        "stage-c-4-gpu-mi350",
    }


def test_domain_label_adds_only_stages_with_matching_tests():
    # [FUNCTION DOC] With a docs-only change and the megatron label, only the stage holding a
    # megatron-labelled test runs; every other stage is skipped.
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration("tests/e2e/test_megatron.py", "stage-c-8-gpu-h100", labels=["megatron"]),
    ]
    changed_files = (ChangedFile("M", ("docs/index.md",)),)

    skipped = set(_select(changed_files, registrations, raw_labels=("run-ci-megatron",)))

    assert skipped == PR_GPU_STAGES - {"stage-c-8-gpu-h100"}


def test_broad_scope_adds_every_runnable_stage():
    # [FUNCTION DOC] With a docs-only change and the "all" label, both stages that have
    # registrations run; the rest are skipped.
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration("tests/e2e/test_megatron.py", "stage-c-8-gpu-h100", labels=["megatron"]),
    ]
    changed_files = (ChangedFile("M", ("docs/index.md",)),)

    skipped = set(_select(changed_files, registrations, raw_labels=("run-ci-all",)))

    assert skipped == PR_GPU_STAGES - {"stage-b-2-gpu-h200", "stage-c-8-gpu-h100"}


@pytest.mark.parametrize("changed_path", ["docs/index.md", "tests/e2e/test_hopper.py"])
def test_blackwell_only_scope_keeps_b200_for_non_blackwell_changes(changed_path):
    # [FUNCTION DOC] Parametrized over a docs change and a change to the Hopper test. With a
    # Hopper test on the 8-GPU H200 stage, a Blackwell test on the B200 stage and the
    # blackwell-only label, all stages except the B200 one are skipped.
    registrations = [
        _registration("tests/e2e/test_hopper.py", "stage-c-8-gpu-h200", hardware=["hopper"]),
        _registration("tests/e2e/test_blackwell.py", "stage-c-8-gpu-b200", hardware=["blackwell"]),
    ]
    changed_files = (ChangedFile("M", (changed_path,)),)

    skipped = set(_select(changed_files, registrations, raw_labels=("run-ci-blackwell-only",)))

    assert skipped == PR_GPU_STAGES - {"stage-c-8-gpu-b200"}


def test_bypass_fastfail_does_not_make_a_docs_change_affect_gpu_stages():
    # [FUNCTION DOC] A docs-only change with only the bypass-fastfail label still skips every GPU
    # stage.
    registrations = [_registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200")]
    changed_files = (ChangedFile("M", ("docs/index.md",)),)

    assert set(_select(changed_files, registrations, raw_labels=("bypass-fastfail",))) == PR_GPU_STAGES


def test_unknown_source_path_affects_every_runnable_gpu_stage():
    # [FUNCTION DOC] A modification of an unclassified library source path with one registration
    # per stage skips nothing (empty tuple).
    assert (
        _select(
            (ChangedFile("M", ("miles/trainer.py",)),),
            _all_runnable_registrations(),
        )
        == ()
    )


def test_rename_from_unknown_source_path_affects_every_runnable_stage():
    # [FUNCTION DOC] A rename whose old path is an unclassified source file and new path a docs
    # file skips nothing.
    changed_files = (ChangedFile("R100", ("miles/old.py", "docs/old-api.md")),)

    assert _select(changed_files, _all_runnable_registrations()) == ()


@pytest.mark.parametrize(
    ("event_name", "changed_files"),
    [("schedule", (ChangedFile("M", ("docs/index.md",)),)), ("workflow_dispatch", ()), ("pull_request", None)],
)
def test_non_pr_or_missing_diff_keeps_all_runnable_stages(event_name, changed_files):
    # [FUNCTION DOC] Parametrized over a scheduled event with a docs change, a manual dispatch with
    # an empty diff, and a pull request with an unreadable (missing) diff. In each case nothing is
    # skipped.
    assert _select(changed_files, _all_runnable_registrations(), event_name=event_name) == ()


def test_changed_labeled_test_without_its_label_keeps_current_selection_semantics():
    # [FUNCTION DOC] A changed megatron-labelled test on the H100 stage, with only the precision
    # label requested, plus a precision test on the stage-b 2-GPU stage. Asserts the skipped set is
    # all stages except the 2-GPU stage-b one: the H100 stage stays skipped because the changed
    # test's label was not requested (label gating still applies to changed tests).
    changed_test = "tests/e2e/test_megatron.py"
    registrations = [
        _registration("tests/fast-gpu/test_precision.py", "stage-b-2-gpu-h200"),
        _registration(changed_test, "stage-c-8-gpu-h100", labels=["megatron"]),
    ]

    assert set(_select((ChangedFile("M", (changed_test,)),), registrations)) == (
        PR_GPU_STAGES - {"stage-b-2-gpu-h200"}
    )


def test_cli_publishes_all_gpu_stages_for_docs_diff(tmp_path):
    # [FUNCTION DOC] Runs the policy module as a subprocess from the repository root with
    # environment: pull request event, no schedule, empty label list, a diff file naming one
    # docs file, and an output file path. Asserts exit code zero; the output file's key=value
    # lines include skipped stages equal to every PR GPU stage and the CUDA-image-needed output
    # equal to false.
    repo_root = Path(__file__).resolve().parents[3]
    diff_path = tmp_path / "changed-files.z"
    diff_path.write_bytes(b"M\0docs/developer/ci/00-stage.md\0")
    output_path = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        {
            "EVENT_NAME": "pull_request",
            "SCHEDULE": "",
            "PR_LABELS_JSON": "[]",
            "CHANGED_FILES_PATH": str(diff_path),
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
    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert set(json.loads(outputs["skipped_stages"])) == PR_GPU_STAGES
    assert outputs["needs_cuda_image"] == "false"


@pytest.mark.parametrize("diff_payload", [b"M\0docker/Dockerfile\0", b"", b"invalid"])
@pytest.mark.parametrize(
    ("raw_labels", "registration_args", "expected_stages"),
    [
        ([], "", set()),
        (["run-ci-cpu"], "", set()),
        (["run-ci"], "", set()),
        (["rebuild-ci-image"], "", set()),
        (["run-ci-megatron"], None, set()),
        (["run-ci-megatron"], ", disabled='unavailable'", set()),
        (["run-ci-megatron"], ", nightly=True", set()),
        (["run-ci-megatron"], "", {"stage-c-4-gpu-h200"}),
        (["run-ci-amd"], "", {"stage-c-4-gpu-mi350"}),
        (["run-ci-megatron", "run-ci-amd"], "", {"stage-c-4-gpu-h200", "stage-c-4-gpu-mi350"}),
        (["run-ci-image"], "", {"stage-c-4-gpu-h200", "stage-c-4-gpu-mi350"}),
        (["nightly"], ", nightly=True", {"stage-c-4-gpu-h200", "stage-c-4-gpu-mi350"}),
    ],
)
def test_policy_cli_image_demand_matches_selected_gpu_tests(
    tmp_path, diff_payload, raw_labels, registration_args, expected_stages
):
    # [FUNCTION DOC] Cross product of three diff payloads (image build file change, empty, invalid)
    # and twelve label scenarios. Builds a temporary tests/e2e tree with one test file registering
    # a CPU entry, a ROCm MI350 entry labelled amd, and (unless the registration argument is
    # absent) a CUDA 4-GPU H200 entry labelled megatron with the extra registration arguments
    # (disabled or nightly) appended. Runs the policy module as a subprocess in that tree with the
    # pull request event, the label list as JSON, no cadence override, the diff and output paths
    # and the repository root on the module path. Asserts exit code zero; that the stages not
    # skipped equal the expected set (empty for no relevant label, disabled or nightly-only
    # registrations; H200 for the megatron label; MI350 for the amd label; both for image or
    # nightly scopes), whatever the diff payload; and that the CUDA-image-needed output equals
    # whether the H200 stage is expected to run.
    repo_root = Path(__file__).resolve().parents[3]
    test_dir = tmp_path / "tests/e2e"
    test_dir.mkdir(parents=True)
    source = (
        "register_cpu_ci(est_time=1, suite='stage-a-cpu', labels=[])\n"
        "register_rocm_ci(est_time=1, suite='stage-c-4-gpu-mi350', labels=['amd'])\n"
    )
    if registration_args is not None:
        source += (
            "register_cuda_ci(est_time=1, suite='stage-c-4-gpu-h200', labels=['megatron'], "
            f"hardware=['hopper']{registration_args})\n"
        )
    (test_dir / "test_selection.py").write_text(source)
    diff_path = tmp_path / "changed-files.z"
    diff_path.write_bytes(diff_payload)
    output_path = tmp_path / "github-output"
    env = os.environ.copy()
    env.update(
        EVENT_NAME="pull_request",
        PR_LABELS_JSON=json.dumps(raw_labels),
        CADENCE_OVERRIDE="",
        CHANGED_FILES_PATH=str(diff_path),
        GITHUB_OUTPUT=str(output_path),
        PYTHONPATH=str(repo_root),
    )
    result = subprocess.run(
        [sys.executable, "-m", "tests.ci.ci_policy"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
    assert PR_GPU_STAGES - set(json.loads(outputs["skipped_stages"])) == expected_stages
    assert outputs["needs_cuda_image"] == str("stage-c-4-gpu-h200" in expected_stages).lower()


def test_image_jobs_require_their_selected_gpu_stages():
    # [FUNCTION DOC] Static checks on workflow text. CUDA PR workflow: declares a CUDA-image-needed
    # output from the resolve step; the docker-build job is gated on that output being true; the
    # image resolver depends on docker-build and on its success. ROCm PR workflow: the image
    # resolver depends on the policy job and runs only when the MI350 stage is not in the skipped
    # stages list.
    cuda = WORKFLOW.read_text()
    build = cuda.split("\n  docker-build:\n", 1)[1].split("\n  resolve-ci-image:\n", 1)[0]
    resolver = cuda.split("\n  resolve-ci-image:\n", 1)[1].split("\n  stage-a-cpu:\n", 1)[0]
    assert "needs_cuda_image: ${{ steps.resolve.outputs.needs_cuda_image }}" in cuda
    assert "needs.resolve-ci-policy.outputs.needs_cuda_image == 'true'" in build
    assert "needs: [docker-build]" in resolver
    assert "needs.docker-build.result == 'success'" in resolver

    rocm = WORKFLOW.with_name("pr-test-rocm.yml").read_text()
    resolver = rocm.split("\n  resolve-ci-image:\n", 1)[1].split("\n  resolve-ci-deps:\n", 1)[0]
    assert "needs: [resolve-ci-policy]" in resolver
    assert (
        "if: ${{ !contains(fromJSON(needs.resolve-ci-policy.outputs.skipped_stages || '[]'), 'stage-c-4-gpu-mi350') }}"
        in resolver
    )
