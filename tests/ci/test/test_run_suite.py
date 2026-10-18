# [MODULE DOC]
# Offline tests of the CI suite runner and its policy pipeline. Parts: the CPU pytest command
# builder (fail-fast flag gated on continue-on-error, no revisiting a directory after its
# parent); the locked tables of suite names per hardware backend and absence of legacy names;
# the label prefix stripper (dedupe, warnings for unprefixed entries, silent skip of empty and
# workflow-only labels); policy resolution per cadence (regular, nightly, weekly, release) with
# the included label set, nightly admission, fail-fast bypass, baseline writing, hardware
# dispatch and absorb flags, and label error cases; static text checks of the PR and ROCm
# workflow definitions (stage inputs, gates, reusable workflows, cron schedules, concurrency
# groups); command-line behavior of the runner script via subprocess; the suite runner's
# integration with policy (stubbed collection and executors); location-based discovery of test
# files; and the test filter's label, suite, backend, cadence and disabled-test handling. Test
# registry records are built directly through a small factory. Registers itself as a CPU test in
# the stage-a-cpu suite.
"""Unit tests for `run_suite.py`.

These cover the Python-side policy and label pipeline:

* `strip_run_ci_prefix`: empty input, prefix stripping, silent skip of
  workflow-only labels, warning on other non-prefixed inputs.
* `resolve_policy`: explicit cadence + raw labels -> selection and fast-fail.
* The PR workflow seams: one adapter resolves trigger facts and every CUDA/ROCm stage consumes its outputs.
* `filter_tests`: include-set selection, including CPU always-on coverage and
  GPU domain labels; a scope subtraction is not a per-test veto.
* `CI_SUITES`: locked to the current hardware taxonomy.

We build `CIRegistry` instances directly via a small factory rather than
parsing fixture files -- the AST-side validation lives in
`test_ci_register.py`; this module exercises the runtime filter.
"""

import itertools
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import tests.ci.run_suite as run_suite_module
from tests.ci.ci_policy import (
    NIGHTLY_CADENCE,
    REGULAR_CADENCE,
    RELEASE_CADENCE,
    SCHEDULE_POLICIES,
    WEEKLY_CADENCE,
    resolve_policy,
    strip_run_ci_prefix,
)
from tests.ci.ci_register import CIRegistry, HWBackend, discover_ci_files, register_cpu_ci
from tests.ci.labels import KNOWN_LABELS
from tests.ci.run_suite import CI_SUITES, build_cpu_pytest_cmd, filter_tests
from tests.ci.stage_selection import PR_GPU_STAGES

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def _make(
    filename: str,
    *,
    backend: HWBackend = HWBackend.CUDA,
    suite: str = "stage-c-8-gpu-h100",
    labels: list[str] | None = None,
    est_time: float = 60.0,
    nightly: bool = False,
    disabled: str | None = None,
    hardware: list[str] | None = None,
) -> CIRegistry:
    # [FUNCTION DOC] HELPER: Build a registry record for the given file name. Defaults: CUDA
    # backend, 8-GPU H100 suite, estimated time 60 seconds, not nightly-only, not disabled. When
    # labels is omitted: none for CPU, "megatron" otherwise. When hardware is omitted: Hopper for
    # CUDA, none otherwise. Labels and hardware are copied; the implicit flag is always false.
    """Minimal `CIRegistry` factory for filter tests.

    CUDA fixtures default to the `megatron` domain and to Hopper-only support;
    CPU fixtures default to the always-on empty label set and no arch.
    """
    if labels is None:
        labels = [] if backend == HWBackend.CPU else ["megatron"]
    if hardware is None:
        hardware = ["hopper"] if backend == HWBackend.CUDA else []
    return CIRegistry(
        backend=backend,
        filename=filename,
        est_time=est_time,
        suite=suite,
        labels=list(labels),
        hardware=list(hardware),
        nightly=nightly,
        disabled=disabled,
        implicit=False,
    )


# --- build_cpu_pytest_cmd: -x gated on continue_on_error --------------------


class TestBuildCpuPytestCmd:
    # [CLASS DOC] Tests of the CPU pytest command-line builder.
    def test_x_present_by_default(self):
        # [FUNCTION DOC] Without continue-on-error the command contains the stop-at-first-failure
        # flag.
        # A regular run stops at the first failure by default.
        cmd = build_cpu_pytest_cmd(["tests/fast/a.py", "tests/fast/b.py"], continue_on_error=False)
        assert "-x" in cmd

    def test_x_dropped_on_continue_on_error(self):
        # [FUNCTION DOC] With continue-on-error the stop flag is absent, the command starts with
        # the pytest program, and both listed files appear in it.
        # bypass-fastfail passes --continue-on-error -> run every file to the end.
        cmd = build_cpu_pytest_cmd(["tests/fast/a.py", "tests/fast/b.py"], continue_on_error=True)
        assert "-x" not in cmd
        assert cmd[0] == "pytest"
        assert "tests/fast/a.py" in cmd and "tests/fast/b.py" in cmd

    def test_a_directory_is_never_returned_to_after_its_parent(self):
        # [FUNCTION DOC] Passes files interleaving two directories (a nested directory, its parent,
        # then the nested one again), extracts each file argument's directory, and asserts the
        # number of distinct directories equals the number of consecutive runs, i.e. no directory
        # reappears after another in between (so each directory's own fixture file loads once).
        """pytest loads a directory's conftest when it first reaches it. Naming that directory again
        after its parent leaves the second visit's tests without their own conftest's fixtures, which
        reads as `fixture ... not found` on tests that have always had one."""
        cmd = build_cpu_pytest_cmd(
            [
                "tests/fast/backends/megatron_utils/test_actor.py",
                "tests/fast/backends/test_fsdp_routing_replay.py",
                "tests/fast/backends/megatron_utils/test_model.py",
            ],
            continue_on_error=True,
        )

        directories = [name.rsplit("/", 1)[0] for name in cmd if name.endswith(".py")]
        assert len(set(directories)) == len(list(itertools.groupby(directories)))


# --- CI_SUITES locked to the stage taxonomy ---------------------------------


class TestCISuites:
    # [CLASS DOC] Tests locking the suite-name table per hardware backend.
    def test_cpu_suites_exact(self):
        # [FUNCTION DOC] The CPU suites are exactly stage-a-cpu then stage-b-cpu.
        assert CI_SUITES[HWBackend.CPU] == ["stage-a-cpu", "stage-b-cpu"]

    def test_cuda_suites_exact(self):
        # [FUNCTION DOC] The CUDA suites are exactly, in order: stage-b 2-GPU H200, stage-c 8-GPU
        # H100, 8-GPU H200, 4-GPU H200, 2-GPU H200, 8-GPU B200.
        assert CI_SUITES[HWBackend.CUDA] == [
            "stage-b-2-gpu-h200",
            "stage-c-8-gpu-h100",
            "stage-c-8-gpu-h200",
            "stage-c-4-gpu-h200",
            "stage-c-2-gpu-h200",
            "stage-c-8-gpu-b200",
        ]

    def test_rocm_suites_include_the_pr_stage(self):
        # [FUNCTION DOC] The ROCm suite list contains the 4-GPU MI350 stage that the PR workflow
        # requests.
        """The trusted ROCm PR workflow can request the suite its stage runs."""
        assert "stage-c-4-gpu-mi350" in CI_SUITES[HWBackend.ROCM]

    def test_no_legacy_suite_names_remain(self):
        # [FUNCTION DOC] A fixed set of thirteen retired suite names must not intersect the union
        # of all suites over all backends.
        legacy = {
            "stage-a-fast",
            "stage-b-fast-1-gpu",
            "stage-b-fast-gpu",
            "stage-b-short-8-gpu",
            "stage-b-sglang-8-gpu",
            "stage-b-8-gpu-h100",
            "stage-c-fsdp-8-gpu",
            "stage-c-megatron-8-gpu",
            "stage-c-precision-8-gpu",
            "stage-c-ckpt-8-gpu",
            "stage-c-long-8-gpu",
            "stage-c-lora-8-gpu",
            "stage-c-all",
        }
        all_suites = {s for suites in CI_SUITES.values() for s in suites}
        assert legacy.isdisjoint(all_suites), f"Legacy suite name(s) still present: {legacy & all_suites}"


# --- `strip_run_ci_prefix` direct tests -------------------------------------


class TestStripRunCiPrefix:
    # [CLASS DOC] Tests of the function that turns raw PR label names into domain labels by
    # removing the "run-ci-" prefix.
    def test_empty_input_yields_empty_set(self):
        # [FUNCTION DOC] An empty list gives an empty set.
        assert strip_run_ci_prefix([]) == set()

    def test_single_prefixed_label_stripped(self):
        # [FUNCTION DOC] One prefixed label yields a set with the bare domain name.
        assert strip_run_ci_prefix(["run-ci-megatron"]) == {"megatron"}

    def test_multiple_prefixed_labels_stripped(self):
        # [FUNCTION DOC] Two prefixed labels yield a set of both bare names.
        assert strip_run_ci_prefix(["run-ci-megatron", "run-ci-fsdp"]) == {"megatron", "fsdp"}

    def test_duplicate_inputs_deduplicate(self):
        # [FUNCTION DOC] A repeated prefixed label appears once in the result set.
        assert strip_run_ci_prefix(["run-ci-megatron", "run-ci-megatron"]) == {"megatron"}

    def test_non_prefixed_input_warns_and_is_skipped(self):
        # [FUNCTION DOC] An entry lacking the prefix is dropped (empty result) and exactly one
        # warning is emitted whose message mentions both "missing" and the prefix text.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["megatron"])
        assert result == set(), "non-prefixed entries must be dropped, not silently included"
        assert len(caught) == 1
        assert "missing" in str(caught[0].message)
        assert "run-ci-" in str(caught[0].message)

    def test_mixed_inputs_keep_only_prefixed(self):
        # [FUNCTION DOC] With two prefixed labels and one bare name, only the prefixed ones are
        # kept and exactly one warning is raised (for the bare name).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["run-ci-megatron", "fsdp", "run-ci-short"])
        assert result == {"megatron", "short"}
        assert len(caught) == 1  # only the bare `fsdp` warns

    def test_empty_string_entries_skipped_without_warning(self):
        # [FUNCTION DOC] Empty-string entries are ignored silently (no warnings); the prefixed
        # label is kept.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["", "run-ci-megatron"])
        assert result == {"megatron"}
        assert len(caught) == 0

    def test_workflow_only_labels_skipped_without_warning(self):
        # [FUNCTION DOC] The cadence/behavior switch labels "nightly" and "bypass-fastfail" are
        # skipped without a warning; only the prefixed domain label remains.
        # `nightly` / `bypass-fastfail` are cadence/behavior switches consumed
        # by the resolved policy, not malformed domain labels.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = strip_run_ci_prefix(["nightly", "bypass-fastfail", "run-ci-megatron"])
        assert result == {"megatron"}
        assert len(caught) == 0


# --- resolve_policy: explicit cadence, scope, and fast-fail ------------------


_ALL = set(KNOWN_LABELS)


class TestResolvePolicy:
    @pytest.mark.parametrize(
        ("cadence", "labels", "expected", "bypass"),
        [
            (REGULAR_CADENCE, set(), set(), False),
            (REGULAR_CADENCE, {"run-ci-megatron"}, {"megatron"}, False),
            (REGULAR_CADENCE, {"bypass-fastfail"}, set(), True),
            (REGULAR_CADENCE, {"run-ci-image"}, _ALL - {"long", "ft-short", "ft-long"}, False),
            (REGULAR_CADENCE, {"run-ci-all"}, _ALL, False),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-all"}, _ALL, False),
            (NIGHTLY_CADENCE, set(), _ALL - {"long", "ft-long"}, True),
            (NIGHTLY_CADENCE, {"nightly"}, _ALL - {"long", "ft-long"}, True),
            (NIGHTLY_CADENCE, {"run-ci-image", "nightly"}, _ALL - {"long", "ft-long"}, True),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-all"}, _ALL, True),
            (WEEKLY_CADENCE, set(), _ALL, True),
            (WEEKLY_CADENCE, {"run-ci-image"}, _ALL, True),
            (RELEASE_CADENCE, set(), _ALL, True),
            (RELEASE_CADENCE, {"run-ci-image"}, _ALL, True),
        ],
    )
    def test_selection_and_fastfail(self, cadence, labels, expected, bypass):
        # [FUNCTION DOC] Parametrized table of (cadence, raw labels, expected include set, expected
        # bypass flag). Resolves the policy and asserts: cadence echoed; include labels equal the
        # expected set (all known labels minus the long / ft-short / ft-long scopes depending on
        # cadence and image or all scope labels); nightly tests are admitted only for nightly,
        # weekly and release cadences; bypass-fastfail as expected; and the baseline is written
        # only for nightly and weekly (release uses frozen dependencies and must not write the
        # rolling performance baseline).
        policy = resolve_policy(cadence, labels)
        assert policy.cadence == cadence
        assert policy.include_labels == expected
        scheduled_cadence = cadence in {NIGHTLY_CADENCE, WEEKLY_CADENCE}
        assert policy.admit_nightly_tests is (scheduled_cadence or cadence == RELEASE_CADENCE)
        assert policy.bypass_fastfail is bypass
        # Release runs weekly's scope, but its frozen dependency SHAs must
        # never write the rolling perf baseline.
        assert policy.write_baseline is scheduled_cadence

    @pytest.mark.parametrize(
        ("labels", "dispatch", "absorb"),
        [
            (set(), frozenset(), False),
            ({"run-ci-megatron"}, frozenset(), False),
            ({"nightly"}, frozenset(), False),
            ({"run-on-hopper"}, frozenset({"hopper"}), True),
            ({"run-on-blackwell"}, frozenset({"blackwell"}), True),
            ({"run-on-hopper", "run-on-blackwell"}, frozenset({"hopper", "blackwell"}), True),
            # An arch without permission to leave home: the Blackwell-exclusive set.
            ({"run-ci-blackwell-only"}, frozenset({"blackwell"}), False),
            # An explicit `run-on-*` outranks it and re-enables routing.
            ({"run-ci-blackwell-only", "run-on-blackwell"}, frozenset({"blackwell"}), True),
        ],
    )
    def test_dispatch_resolution(self, labels, dispatch, absorb):
        # [FUNCTION DOC] Parametrized over label sets. Cadence is nightly when the nightly label is
        # present, else regular. Asserts the resolved dispatch-architecture set (hopper and/or
        # blackwell from run-on-* labels, or blackwell from the blackwell-only domain label) and the
        # absorb flag (true only when an explicit run-on label is given; the blackwell-only label
        # alone does not allow leaving the home stage, but an explicit run-on label overrides).
        cadence = NIGHTLY_CADENCE if "nightly" in labels else REGULAR_CADENCE
        policy = resolve_policy(cadence, labels)
        assert policy.dispatch_arches == dispatch
        assert policy.absorb is absorb

    def test_blackwell_only_selects_every_domain(self):
        # [FUNCTION DOC] The blackwell-only label yields the full known label set as the include
        # set under regular cadence.
        # The arch is the selection, so no domain label may narrow it away.
        assert resolve_policy(REGULAR_CADENCE, {"run-ci-blackwell-only"}).include_labels == _ALL

    def test_scheduled_cadences_never_absorb(self):
        # [FUNCTION DOC] For nightly, weekly and release cadences with no labels, the absorb flag is
        # false and the dispatch-architecture set is empty.
        # Cron runs carry no labels, so nightly and weekly keep every test on
        # its home stage no matter how widely it is tagged.
        for cadence in (NIGHTLY_CADENCE, WEEKLY_CADENCE, RELEASE_CADENCE):
            policy = resolve_policy(cadence, set())
            assert policy.absorb is False
            assert policy.dispatch_arches == frozenset()

    def test_unknown_cadence_rejected(self):
        # [FUNCTION DOC] Resolving a policy for the cadence "hourly" raises a value error "Unknown CI
        # cadence 'hourly'".
        with pytest.raises(ValueError, match="Unknown CI cadence 'hourly'"):
            resolve_policy("hourly", set())

    def test_nightly_tag_and_explicit_cadence_converge(self):
        # [FUNCTION DOC] Resolving nightly cadence with and without the nightly label gives equal
        # policies.
        assert resolve_policy(NIGHTLY_CADENCE, {"nightly"}) == resolve_policy(NIGHTLY_CADENCE, set())

    def test_nightly_label_requires_resolved_nightly_cadence(self):
        # [FUNCTION DOC] The nightly label under regular cadence raises a value error mentioning
        # "nightly workflow label".
        with pytest.raises(ValueError, match="nightly workflow label"):
            resolve_policy(REGULAR_CADENCE, {"nightly"})

    @pytest.mark.parametrize(
        ("cadence", "labels", "expected"),
        [
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-long"}, _ALL - {"ft-short", "ft-long"}),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short"}, _ALL - {"long", "ft-long"}),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-long"}, _ALL - {"ft-long"}),
            (NIGHTLY_CADENCE, {"nightly", "run-ci-ft-long"}, _ALL - {"long"}),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short", "run-ci-ft-long"}, _ALL - {"long"}),
            (NIGHTLY_CADENCE, {"run-ci-ft-long"}, _ALL - {"long"}),
        ],
    )
    def test_explicit_domain_label_wins_over_scope_subtraction(self, cadence, labels, expected):
        # [FUNCTION DOC] Parametrized table: a scope that subtracts long / ft labels (image scope,
        # nightly) still includes any such label that is requested explicitly by a domain label.
        # Asserts the include set equals the expected one.
        assert resolve_policy(cadence, labels).include_labels == expected

    @pytest.mark.parametrize(
        ("cadence", "labels"),
        [
            (REGULAR_CADENCE, set()),
            (REGULAR_CADENCE, {"run-ci-megatron", "run-ci-typo", "bypass-fastfail"}),
            (REGULAR_CADENCE, {"run-ci-image", "run-ci-ft-short"}),
            (NIGHTLY_CADENCE, set()),
            (WEEKLY_CADENCE, set()),
            (RELEASE_CADENCE, set()),
        ],
    )
    def test_include_set_stays_inside_known_labels(self, cadence, labels):
        # [FUNCTION DOC] Parametrized over cadences and label sets including a typo'd label. The
        # include set must be a subset of the known registry labels (no scope artifacts like
        # "image" or "all", no typos).
        # The include set is drawn from the registry only: scope-label
        # stripping artifacts (`image`, `all`) and typo'd requests must not
        # leak in, and scope subtractions must name real registry labels.
        assert resolve_policy(cadence, labels).include_labels <= _ALL


# --- pr-test.yml seam: one trigger adapter, shared stage inputs ---------------


class TestWorkflowScopeSeam:
    # [CLASS DOC] Static text checks of the PR test workflow and the reusable workflows it calls,
    # reading the workflow files from the repository root (three directories above this file).
    # They lock structure (job inventory, gates, inputs, schedules) rather than run anything.
    @staticmethod
    def _workflow() -> str:
        # [FUNCTION DOC] HELPER: Return the text of the PR test workflow file.
        return (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pr-test.yml").read_text()

    @staticmethod
    def _reusable_workflow(name: str) -> str:
        # [FUNCTION DOC] HELPER: Return the text of the named workflow file in the workflows
        # directory.
        return (Path(__file__).resolve().parents[3] / ".github" / "workflows" / name).read_text()

    def test_every_stage_consumes_resolved_policy(self):
        # [FUNCTION DOC] Splits the PR workflow at each execute-command key (expects exactly eight
        # stages; the count must be updated when stages change) and, for each command block up to
        # its secrets key, asserts it passes the cadence and raw labels from the resolve-policy
        # job outputs and contains neither an event-name nor a continue-on-error argument.
        workflow = self._workflow()
        commands = workflow.split("execute_command:")[1:]
        assert len(commands) == 8, "stage inventory changed; update this lock test"
        for block in commands:
            cmd = block.split("secrets:")[0]
            assert "--cadence ${{ needs.resolve-ci-policy.outputs.cadence }}" in cmd
            assert "--labels ${{ needs.resolve-ci-policy.outputs.raw_labels }}" in cmd
            assert "--event-name" not in cmd
            assert "--continue-on-error" not in cmd

    def test_cpu_stages_only_require_policy(self):
        # [FUNCTION DOC] The stage-a-cpu and stage-b-cpu job blocks each depend only on the
        # resolve-policy job and never mention the resolve-image job.
        workflow = self._workflow()
        stage_a = workflow.split("  stage-a-cpu:", 1)[1].split("  stage-b-cpu:", 1)[0]
        stage_b = workflow.split("  stage-b-cpu:", 1)[1].split("  stage-b-2-gpu-h200:", 1)[0]

        expected = "needs: [resolve-ci-policy]"
        assert expected in stage_a
        assert expected in stage_b
        assert "resolve-ci-image" not in stage_a
        assert "resolve-ci-image" not in stage_b

    def test_non_default_base_pr_needs_run_ci_label(self):
        # [FUNCTION DOC] The policy job's condition text (before its runs-on key) must contain the
        # default-branch base check, a check that some PR label starts with the run-ci marker, and
        # the non-pull-request event check.
        policy_gate = self._workflow().split("  resolve-ci-policy:", 1)[1].split("    runs-on:", 1)[0]

        assert "github.event.pull_request.base.ref == github.event.repository.default_branch" in policy_gate
        assert "contains(toJSON(github.event.pull_request.labels.*.name), '\"run-ci')" in policy_gate
        assert "github.event_name != 'pull_request'" in policy_gate

    def test_cpu_and_gpu_stages_use_dedicated_reusable_workflows(self):
        # [FUNCTION DOC] Counts reusable-workflow uses in the PR workflow (2 CPU, 6 GPU, 1 image
        # build), asserts no "cpu_runner" mention there, then parses the top-level job ids of the
        # three reusable workflows by regular expression and asserts they are: GPU = plan, run;
        # CPU = run-cpu; image = docker-decide, docker-build; and that the GPU and CPU reusable
        # workflows do not mention "cpu_runner" either.
        workflow = self._workflow()
        assert workflow.count("uses: ./.github/workflows/_run-cpu-ci.yml") == 2
        assert workflow.count("uses: ./.github/workflows/_run-ci.yml") == 6
        assert workflow.count("uses: ./.github/workflows/_build-pr-ci-image.yml") == 1
        assert "cpu_runner" not in workflow

        gpu_workflow = self._reusable_workflow("_run-ci.yml")
        cpu_workflow = self._reusable_workflow("_run-cpu-ci.yml")
        docker_workflow = self._reusable_workflow("_build-pr-ci-image.yml")
        job_id_pattern = r"^  ([A-Za-z_][A-Za-z0-9_-]*):$"
        gpu_jobs = re.findall(job_id_pattern, gpu_workflow.split("\njobs:\n", 1)[1], re.MULTILINE)
        cpu_jobs = re.findall(job_id_pattern, cpu_workflow.split("\njobs:\n", 1)[1], re.MULTILINE)
        docker_jobs = re.findall(job_id_pattern, docker_workflow.split("\njobs:\n", 1)[1], re.MULTILINE)
        assert gpu_jobs == ["plan", "run"]
        assert cpu_jobs == ["run-cpu"]
        assert docker_jobs == ["docker-decide", "docker-build"]
        assert "cpu_runner" not in gpu_workflow
        assert "cpu_runner" not in cpu_workflow

    def test_docker_build_waits_for_cpu_gate_and_preserves_bypass(self):
        # [FUNCTION DOC] In the docker-build caller block: needs the policy job and stage-a-cpu
        # (not stage-b-cpu); runs under always-and-not-cancelled; skips when the PR action is
        # closed; requires policy success and (stage-a-cpu success or bypass fast-fail), never a
        # stage-a-cpu failure check; calls the reusable image workflow and inherits secrets.
        workflow = self._workflow()
        caller = workflow.split("  docker-build:", 1)[1].split("  resolve-ci-image:", 1)[0]

        assert "needs: [resolve-ci-policy, stage-a-cpu]" in caller
        assert "always() && !cancelled()" in caller
        assert "github.event.action != 'closed'" in caller
        assert "needs.resolve-ci-policy.result == 'success'" in caller
        assert "needs.stage-a-cpu.result == 'success'" in caller
        assert "needs.resolve-ci-policy.outputs.bypass_fastfail == 'true'" in caller
        assert "needs.stage-a-cpu.result == 'failure'" not in caller
        assert "stage-b-cpu" not in caller
        assert "uses: ./.github/workflows/_build-pr-ci-image.yml" in caller
        assert "secrets: inherit" in caller

    def test_docker_build_body_lives_in_reusable_workflow(self):
        # [FUNCTION DOC] The decide job is absent from the PR workflow. The reusable image workflow
        # exposes the built and tag-available outputs from the build job, makes the build depend on
        # the decide job, computes rebuild need from the build-input script against the first
        # parent commit, reads the image label, restricts to same-repo heads, and builds the
        # cu13 variant with the custom tag.
        workflow = self._workflow()
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")

        assert "  docker-decide:" not in workflow
        assert "value: ${{ jobs.docker-build.outputs.built }}" in reusable
        assert "value: ${{ jobs.docker-build.outputs.tag_available }}" in reusable
        assert "needs: [docker-decide]" in reusable
        # Rebuilds follow the build inputs, not whether the PR diff touched them.
        assert "python3 docker/image_inputs.py --rev HEAD^1" in reusable
        assert "python3 docker/image_inputs.py --read-label" in reusable
        assert "github.event.pull_request.head.repo.full_name == github.repository" in reusable
        assert "python3 docker/build.py --variant cu13 --image-tag custom" in reusable

    def test_docker_decision_logs_in_only_for_tag_inspection(self):
        # [FUNCTION DOC] In the decide job text, asserts the ordering: input comparison step, then
        # Docker Hub login step, then the registry tag inspection command; and that the login is
        # conditional on the prepare step's inspect-tag output being true.
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")
        decide_job = reusable.split("  docker-decide:", 1)[1].split("  docker-build:", 1)[0]

        compare = decide_job.index("python3 docker/image_inputs.py --rev HEAD^1")
        login = decide_job.index("- name: Login to Docker Hub")
        inspect = decide_job.index("docker buildx imagetools inspect")
        assert compare < login < inspect
        assert "if: steps.prepare.outputs.inspect_tag == 'true'" in decide_job

    def test_docker_force_rebuild_uses_live_label_for_decision_and_consumption(self):
        # [FUNCTION DOC] In the decide job: the live issue-labels API path is queried before the
        # Docker login, via a paginated gh call into a variable; the force-rebuild label is tested
        # with an exact whole-line match on that variable; the event payload label list is not
        # used; and the reusable workflow consumes the decide job's force-rebuild output.
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")
        decide_job = reusable.split("  docker-decide:", 1)[1].split("  docker-build:", 1)[0]

        live_labels = '"repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/labels"'
        assert live_labels in decide_job
        assert decide_job.index(live_labels) < decide_job.index("- name: Login to Docker Hub")
        assert "LABELS=$(gh api --paginate" in decide_job
        assert 'grep -Fxq "rebuild-ci-image" <<< "$LABELS"' in decide_job
        assert "github.event.pull_request.labels.*.name" not in decide_job
        assert "needs.docker-decide.outputs.force_rebuild == 'true'" in reusable

    def test_policy_job_is_a_thin_python_adapter(self):
        # [FUNCTION DOC] In the policy job text (up to the docker-build job): pinned major checkout
        # action without persisted credentials and with fetch depth 2; runs the policy module via
        # python; declares outputs cadence, raw labels, bypass fast-fail and skipped stages from the
        # resolve step; diffs name-status with NUL separators and rename detection between first
        # parent and head; writes the diff to a path in the runner temp directory; and contains no
        # shell event-name case statement nor jq usage.
        workflow = self._workflow()
        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("  docker-build:", 1)[0]
        assert "uses: actions/checkout@v4" in policy_block
        assert "persist-credentials: false" in policy_block
        assert "run: python -m tests.ci.ci_policy" in policy_block
        assert "cadence: ${{ steps.resolve.outputs.cadence }}" in policy_block
        assert "raw_labels: ${{ steps.resolve.outputs.raw_labels }}" in policy_block
        assert "bypass_fastfail: ${{ steps.resolve.outputs.bypass_fastfail }}" in policy_block
        assert "skipped_stages: ${{ steps.resolve.outputs.skipped_stages }}" in policy_block
        assert "fetch-depth: 2" in policy_block
        assert "git diff --name-status -z -M HEAD^1 HEAD" in policy_block
        assert "CHANGED_FILES_PATH: ${{ runner.temp }}/changed-files.z" in policy_block
        assert 'case "$EVENT_NAME"' not in policy_block
        assert "jq " not in policy_block

    def test_policy_job_passes_trigger_facts(self):
        # [FUNCTION DOC] The policy job passes the event name, the schedule (empty default) and the
        # PR labels as a JSON list through environment variables, and does not use a label join
        # expression.
        workflow = self._workflow()
        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("  docker-build:", 1)[0]
        assert "EVENT_NAME: ${{ github.event_name }}" in policy_block
        assert "SCHEDULE: ${{ github.event.schedule || '' }}" in policy_block
        assert "PR_LABELS_JSON: ${{ toJSON(github.event.pull_request.labels.*.name) }}" in policy_block
        assert "join(github.event.pull_request.labels" not in policy_block

    def test_every_configured_cron_has_an_explicit_python_policy(self):
        # [FUNCTION DOC] Extracts every cron expression from the workflow by regular expression and
        # asserts the set equals the key set of the schedule policy table.
        workflow = self._workflow()
        configured = set(re.findall(r"^\s+- cron: ['\"]([^'\"]+)['\"]\s*$", workflow, flags=re.MULTILINE))
        assert configured == set(SCHEDULE_POLICIES)

    def test_scheduled_runs_use_utc_1500(self):
        # [FUNCTION DOC] The workflow has cron entries at 15:00 for days 0-5 and for day 6, and no
        # timezone key (cron is UTC).
        workflow = self._workflow()
        assert "    - cron: '0 15 * * 0-5'" in workflow
        assert "    - cron: '0 15 * * 6'" in workflow
        assert "timezone:" not in workflow

    def test_weekly_serializes_each_gpu_matrix(self):
        # [FUNCTION DOC] For the 4-GPU H200 job (normal parallelism 5) and the 2-GPU H200 job
        # (normal parallelism 2), extracts the job block (up to the next top-level job) and asserts
        # its max-parallel expression is one when the cadence output is weekly, else the normal
        # value.
        workflow = self._workflow()
        normal_parallelism = {
            "stage-c-4-gpu-h200": 5,
            "stage-c-2-gpu-h200": 2,
        }
        for job, default in normal_parallelism.items():
            block = workflow.split(f"  {job}:", 1)[1]
            block = re.split(r"^  [A-Za-z_][A-Za-z0-9_-]*:\s*$", block, maxsplit=1, flags=re.MULTILINE)[0]
            expected = (
                "max-parallel: ${{ needs.resolve-ci-policy.outputs.cadence == 'weekly' " f"&& 1 || {default} }}}}"
            )
            assert expected in block

    def test_dispatch_has_no_scope_input_but_runs_all_cuda_domains(self):
        # [FUNCTION DOC] The manual dispatch inputs (text between the dispatch key and the
        # permissions key) contain no cadence or scope input, and the match-all-labels expression
        # for manual dispatch occurs exactly six times in the section from the first CUDA stage on
        # (once per CUDA stage).
        workflow = self._workflow()
        dispatch_inputs = workflow.split("workflow_dispatch:", 1)[1].split("permissions:", 1)[0]
        assert "ci_cadence" not in dispatch_inputs
        assert "ci_scope" not in dispatch_inputs
        manual_scope = "${{ github.event_name == 'workflow_dispatch' && '--match-all-labels' || '' }}"
        cuda_stages = workflow.split("  stage-b-2-gpu-h200:", 1)[1]
        assert cuda_stages.count(manual_scope) == 6

    def test_gpu_gates_consume_shared_bypass_output(self):
        # [FUNCTION DOC] In the GPU stage section, the bypass fast-fail gate and the policy-success
        # and image-resolver-success checks each occur six times (once per GPU stage), and no
        # stage-a-cpu failure check is present.
        workflow = self._workflow()
        gpu_stages = workflow.split("  stage-b-2-gpu-h200:", 1)[1]
        bypass_gate = "needs.resolve-ci-policy.outputs.bypass_fastfail == 'true'"
        assert gpu_stages.count(bypass_gate) == 6
        assert gpu_stages.count("needs.resolve-ci-policy.result == 'success'") == 6
        assert gpu_stages.count("needs.resolve-ci-image.result == 'success'") == 6
        assert "needs.stage-a-cpu.result == 'failure'" not in gpu_stages

    def test_each_cuda_stage_consumes_the_fail_open_skip_list(self):
        # [FUNCTION DOC] For every PR GPU stage except the MI350 ROCm one, extracts its job block
        # and asserts it has a condition that the stage name is not in the policy job's skipped
        # stages list (parsed from JSON, defaulting to an empty list); also that the skipped-stages
        # expression appears in the workflow exactly once per CUDA stage.
        workflow = self._workflow()
        cuda_stages = PR_GPU_STAGES - {"stage-c-4-gpu-mi350"}
        for stage_name in cuda_stages:
            block = workflow.split(f"  {stage_name}:", 1)[1]
            block = re.split(r"^  [A-Za-z_][A-Za-z0-9_-]*:\s*$", block, maxsplit=1, flags=re.MULTILINE)[0]
            expected = f"!contains(fromJSON(needs.resolve-ci-policy.outputs.skipped_stages || '[]'), '{stage_name}')"
            assert expected in block

        assert workflow.count("outputs.skipped_stages || '[]'") == len(cuda_stages)

    def test_non_pr_concurrency_does_not_collapse_to_ref(self):
        # [FUNCTION DOC] The workflow's concurrency expression orders schedule, then input ref,
        # then run id, so cron runs never share a ref-derived group and plain dispatches fall
        # through to the run id.
        workflow = self._workflow()
        # schedule outranks the workflow_call-only inputs.ref segment, so cron
        # runs never share a ref-derived group; plain dispatches (no ref
        # input) fall through to run_id.
        assert "github.event.schedule || inputs.ref || github.run_id" in workflow

    def test_closed_pr_only_cancels_existing_run(self):
        # [FUNCTION DOC] The pull-request trigger lists the opened, synchronize, reopened,
        # ready-for-review, labeled, unlabeled and closed types; the concurrency group is keyed by
        # PR number, then schedule, then input ref, then run id; the policy job header, the docker
        # caller and (twice) the reusable image workflow skip when the action is closed, so a
        # closed PR only cancels the existing run.
        workflow = self._workflow()
        assert "types: [opened, synchronize, reopened, ready_for_review, labeled, unlabeled, closed]" in workflow
        assert (
            "group: pr-test-${{ github.event.number || github.event.schedule || inputs.ref || github.run_id }}"
            in workflow
        )

        policy_header = workflow.split("  resolve-ci-policy:", 1)[1].split("    runs-on:", 1)[0]
        docker_caller = workflow.split("  docker-build:", 1)[1].split("  resolve-ci-image:", 1)[0]
        reusable = self._reusable_workflow("_build-pr-ci-image.yml")

        assert "github.event.action != 'closed'" in policy_header
        assert "github.event.action != 'closed'" in docker_caller
        assert reusable.count("github.event.action != 'closed'") == 2


class TestRocmWorkflowScopeSeam:
    # [CLASS DOC] Static text checks of the ROCm PR workflow and its reusable workflow.
    @staticmethod
    def _workflow() -> str:
        # [FUNCTION DOC] HELPER: Return the text of the ROCm PR test workflow file.
        return (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pr-test-rocm.yml").read_text()

    def test_closed_pr_cancels_without_starting_resolvers_or_gpu_jobs(self):
        # [FUNCTION DOC] The concurrency group is keyed by PR number, schedule, input ref, run id,
        # with cancel-in-progress on. The policy and dependency resolver job headers skip when the
        # PR action is closed. The image resolver depends only on the policy job and has no
        # always() override. The MI350 stage requires policy and dependency resolver success.
        workflow = self._workflow()
        assert (
            "group: pr-test-rocm-${{ github.event.number || github.event.schedule || inputs.ref || github.run_id }}"
            in workflow
        )
        assert "cancel-in-progress: true" in workflow
        for job in ("resolve-ci-policy", "resolve-ci-deps"):
            header = workflow.split(f"  {job}:\n", 1)[1].split("    runs-on:", 1)[0]
            assert "if: github.event.action != 'closed'" in header
        image = workflow.split("  resolve-ci-image:\n", 1)[1].split("    runs-on:", 1)[0]
        assert "needs: [resolve-ci-policy]" in image
        assert "always()" not in image
        stage = workflow.split("  stage-c-4-gpu-mi350:\n", 1)[1]
        assert "needs.resolve-ci-policy.result == 'success'" in stage
        assert "needs.resolve-ci-deps.result == 'success'" in stage

    def test_pr_schedules_and_dispatch_share_policy(self):
        # [FUNCTION DOC] The ROCm workflow triggers on pull requests with the same type list as the
        # CUDA one and has no pull-request-target trigger; its cron set equals the schedule policy
        # table and includes the 15:00 entries, with no timezone key; the policy job passes the
        # same trigger facts, diffs and outputs as the CUDA one, has no self-hosted allowance and
        # runs the policy module; and the concurrency expression falls through schedule, input ref,
        # run id.
        workflow = self._workflow()
        assert (
            "pull_request:\n    types: [opened, synchronize, reopened, ready_for_review, labeled, unlabeled, closed]"
            in workflow
        )
        assert "pull_request_target:" not in workflow
        configured = set(re.findall(r"^\s+- cron: ['\"]([^'\"]+)['\"]\s*$", workflow, flags=re.MULTILINE))
        assert configured == set(SCHEDULE_POLICIES)
        assert "    - cron: '0 15 * * 0-5'" in workflow
        assert "    - cron: '0 15 * * 6'" in workflow
        assert "timezone:" not in workflow

        policy_block = workflow.split("resolve-ci-policy:", 1)[1].split("resolve-ci-image:", 1)[0]
        assert "allow_self_hosted" not in policy_block
        assert "EVENT_NAME: ${{ github.event_name }}" in policy_block
        assert "SCHEDULE: ${{ github.event.schedule || '' }}" in policy_block
        assert "PR_LABELS_JSON: ${{ toJSON(github.event.pull_request.labels.*.name) }}" in policy_block
        assert "CHANGED_FILES_PATH: ${{ runner.temp }}/changed-files.z" in policy_block
        assert "git diff --name-status -z -M HEAD^1 HEAD" in policy_block
        assert "skipped_stages: ${{ steps.resolve.outputs.skipped_stages }}" in policy_block
        assert "run: python -m tests.ci.ci_policy" in policy_block
        assert "github.event.schedule || inputs.ref || github.run_id" in workflow

    def test_stage_consumes_policy_and_preserves_manual_full_scope(self):
        # [FUNCTION DOC] Checks the MI350 stage block: depends on the policy, image and dependency
        # resolvers; no self-hosted allowance; two partitions with auto-partition size 2;
        # max-parallel one on weekly else two; command passes cadence and labels from policy,
        # adds match-all-labels on manual dispatch, never an AMD label argument, no checkout ref;
        # passes the tracking API key secret; gates on the three resolver successes, on the
        # dependency-install skip output and on its stage name not being in the skipped list; no
        # dispatch-only condition anywhere. Also the reusable ROCm workflow: no checkout ref, no
        # persisted credentials, no unsafe PR checkout allowance, platform env set to rocm.
        workflow = self._workflow()
        stage = workflow.split("  stage-c-4-gpu-mi350:", 1)[1]
        command = stage.split("execute_command:", 1)[1].split("secrets:", 1)[0]

        assert "needs: [resolve-ci-policy, resolve-ci-image, resolve-ci-deps]" in stage
        assert "allow_self_hosted" not in stage
        assert "partition_id: [0, 1]" in stage
        assert "max-parallel: ${{ needs.resolve-ci-policy.outputs.cadence == 'weekly' && 1 || 2 }}" in stage
        assert "--auto-partition-size 2" in command
        assert "checkout_ref:" not in stage
        assert "--cadence ${{ needs.resolve-ci-policy.outputs.cadence }}" in command
        assert "--labels ${{ needs.resolve-ci-policy.outputs.raw_labels }}" in command
        assert "${{ github.event_name == 'workflow_dispatch' && '--match-all-labels' || '' }}" in command
        assert "WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}" in stage
        assert "needs.resolve-ci-policy.result == 'success'" in stage
        assert "needs.resolve-ci-image.result == 'success'" in stage
        assert "needs.resolve-ci-deps.result == 'success'" in stage
        assert (
            "skip_dependency_install: ${{ needs.resolve-ci-deps.outputs.skip_dependency_install == 'true' }}" in stage
        )
        assert (
            "!contains(fromJSON(needs.resolve-ci-policy.outputs.skipped_stages || '[]'), 'stage-c-4-gpu-mi350')"
            in stage
        )
        assert "--labels amd" not in command
        assert "if: github.event_name == 'workflow_dispatch'" not in workflow

        reusable = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "_run-ci-rocm.yml").read_text()
        assert "checkout_ref:" not in reusable
        assert "persist-credentials: false" in reusable
        assert "allow-unsafe-pr-checkout" not in reusable
        assert "MILES_HARDWARE_PLATFORM: rocm" in reusable

    def test_megatron_override_installs_the_checked_out_ref_unpatched(self):
        # [FUNCTION DOC] In the reusable ROCm workflow, extracts the shell branch taken when a
        # Megatron PR number is set (up to the workspace cd) and asserts a forced checkout of the
        # fetched head comes before the editable no-deps install, and that no AMD patch is applied.
        reusable = (Path(__file__).resolve().parents[3] / ".github" / "workflows" / "_run-ci-rocm.yml").read_text()
        override = reusable.split('if [ -n "$MEGATRON_PR" ]; then', 1)[1].split("          cd $GITHUB_WORKSPACE", 1)[0]

        checkout = override.index("git checkout -f FETCH_HEAD")
        install = override.index("pip install -e . --no-deps --break-system-packages")

        assert checkout < install
        assert "amd_patch" not in override


# --- CLI seam: local nightly alias and invalid-suite exit behavior -----------


class TestRunSuiteCLI:
    # [CLASS DOC] Tests of the suite runner script's command line, run as a real subprocess.
    @staticmethod
    def _run(*args: str) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] HELPER: Run the suite runner script with the current interpreter from the
        # repository root, with the repository root prepended to the module search path variable.
        # Captures text output and does not raise on failure. Returns the completed process.
        repo_root = Path(__file__).resolve().parents[3]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH"))))
        return subprocess.run(
            [sys.executable, "tests/ci/run_suite.py", *args],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_nightly_alias_matches_explicit_cadence(self):
        # [FUNCTION DOC] Lists (no execution) the CPU stage-b suite once with the nightly alias and
        # once with the explicit nightly cadence. Both exit zero; their first output lines (the
        # policy summary) are identical, show cadence nightly with bypass fast-fail true, include
        # the ft-short label and neither long nor ft-long; the alias output reports continue on
        # error true.
        common = ("--hw", "cpu", "--suite", "stage-b-cpu", "--list-only")
        alias = self._run(*common, "--nightly")
        explicit = self._run(*common, "--cadence", NIGHTLY_CADENCE)

        assert alias.returncode == explicit.returncode == 0
        alias_policy = alias.stdout.splitlines()[0]
        explicit_policy = explicit.stdout.splitlines()[0]
        assert alias_policy == explicit_policy
        assert "cadence='nightly' bypass_fastfail=True" in alias_policy
        assert "'ft-short'" in alias_policy
        assert "'long'" not in alias_policy
        assert "'ft-long'" not in alias_policy
        assert "Continue on error: True" in alias.stdout

    def test_nightly_alias_and_explicit_cadence_are_mutually_exclusive(self):
        # [FUNCTION DOC] Passing both the nightly alias and an explicit cadence exits with code 2
        # and an argument-parser message "not allowed with argument" on standard error.
        result = self._run(
            "--hw",
            "cpu",
            "--suite",
            "stage-b-cpu",
            "--nightly",
            "--cadence",
            NIGHTLY_CADENCE,
            "--list-only",
        )
        assert result.returncode == 2
        assert "not allowed with argument" in result.stderr

    def test_unknown_suite_exits_nonzero(self):
        # [FUNCTION DOC] An unknown CUDA suite name exits nonzero, reports "Unknown suite
        # stage-c-unknown" on standard error, and does not print the successful empty-run message.
        result = self._run(
            "--hw",
            "cuda",
            "--suite",
            "stage-c-unknown",
            "--list-only",
        )
        assert result.returncode != 0
        assert "Unknown suite stage-c-unknown" in result.stderr
        assert "No tests to run. Exiting with success." not in result.stdout


# --- run_a_suite: resolved policy reaches cadence and runner behavior --------


def _run_args(*, hw: str, suite: str, cadence: str, labels: list[str] | None = None):
    # [FUNCTION DOC] HELPER: Build a namespace mimicking the parsed command-line arguments of the
    # suite runner: given hardware, suite, cadence and labels (empty default); match-all-labels,
    # continue-on-error, list-only and retry off; no partitioning or github output; per-file
    # timeout 1800 s, retry timeout increase 600 s, max attempts 2, retry wait 60 s.
    return SimpleNamespace(
        hw=hw,
        suite=suite,
        cadence=cadence,
        labels=labels or [],
        match_all_labels=False,
        continue_on_error=False,
        auto_partition_id=None,
        auto_partition_size=None,
        list_only=False,
        github_output=None,
        timeout_per_file=1800,
        enable_retry=False,
        retry_timeout_increase=600,
        max_attempts=2,
        retry_wait_seconds=60,
    )


class TestRunSuitePolicyIntegration:
    # [CLASS DOC] Tests that a resolved policy reaches the suite runner's behavior, using stubbed
    # test collection and stubbed executors (no test is actually executed).
    @staticmethod
    def _stub_collection(monkeypatch, tests):
        # [FUNCTION DOC] HELPER: Make file discovery return nothing and test collection return the
        # given registry list, by patching the runner module.
        monkeypatch.setattr(run_suite_module, "discover_ci_files", lambda: [])
        monkeypatch.setattr(run_suite_module, "collect_tests", lambda *_args, **_kwargs: tests)

    def test_regular_all_scope_does_not_unlock_nightly_only(self):
        # [FUNCTION DOC] With a regular and a nightly-only test and the regular-cadence "all"
        # label policy, filtering for the 8-GPU H100 suite enables only the regular test.
        tests = [
            _make("tests/e2e/regular.py", labels=["megatron"]),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True),
        ]
        policy = resolve_policy(REGULAR_CADENCE, {"run-ci-all"})
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=policy.admit_nightly_tests,
            labels=set(policy.include_labels),
        )
        assert _names(enabled) == {"tests/e2e/regular.py"}

    def test_weekly_full_scope_admits_nightly_only_tests(self):
        # [FUNCTION DOC] With a long-labelled regular test and an ft-long-labelled nightly-only
        # test under the weekly policy, both are enabled.
        tests = [
            _make("tests/e2e/regular.py", labels=["long"]),
            _make("tests/e2e/nightly.py", labels=["ft-long"], nightly=True),
        ]
        policy = resolve_policy(WEEKLY_CADENCE, set())
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=policy.admit_nightly_tests,
            labels=set(policy.include_labels),
        )
        assert _names(enabled) == {"tests/e2e/regular.py", "tests/e2e/nightly.py"}

    def test_nightly_bypass_reaches_cpu_runner(self, monkeypatch):
        # [FUNCTION DOC] Stubs collection with one CPU test and replaces the subprocess call with
        # a fake that records the command and returns zero. Running the stage-a CPU suite with
        # nightly cadence returns zero and the recorded command lacks the stop-at-first-failure
        # flag.
        tests = [_make("tests/fast/test_regular.py", backend=HWBackend.CPU, suite="stage-a-cpu")]
        self._stub_collection(monkeypatch, tests)
        captured = {}

        def fake_call(cmd):
            # [FUNCTION DOC] HELPER: Record the command line and report success (zero).
            captured["cmd"] = cmd
            return 0

        monkeypatch.setattr(run_suite_module.subprocess, "call", fake_call)
        result = run_suite_module.run_a_suite(_run_args(hw="cpu", suite="stage-a-cpu", cadence=NIGHTLY_CADENCE))
        assert result == 0
        assert "-x" not in captured["cmd"]

    def test_nightly_bypass_reaches_cuda_runner(self, monkeypatch):
        # [FUNCTION DOC] Stubs collection with one CUDA test and the file executor with a fake that
        # records its keyword options. Running the 8-GPU H100 suite with nightly cadence returns
        # zero and the executor received continue-on-error true and write-baseline true.
        tests = [_make("tests/e2e/test_regular.py", suite="stage-c-8-gpu-h100")]
        self._stub_collection(monkeypatch, tests)
        captured = {}

        def fake_run_unittest_files(ci_tests, **kwargs):
            # [FUNCTION DOC] HELPER: Record the keyword options and report success (zero).
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(run_suite_module, "run_unittest_files", fake_run_unittest_files)
        result = run_suite_module.run_a_suite(
            _run_args(hw="cuda", suite="stage-c-8-gpu-h100", cadence=NIGHTLY_CADENCE)
        )
        assert result == 0
        assert captured["continue_on_error"] is True
        assert captured["gate_write_baseline"] is True

    def test_weekly_policy_reaches_cuda_runner(self, monkeypatch):
        # [FUNCTION DOC] Stubs collection with one long-labelled CUDA test and the file executor
        # with a fake recording the test list and keyword options. Running the suite with weekly
        # cadence returns zero, passes exactly that test, continue-on-error true and
        # write-baseline true.
        tests = [_make("tests/e2e/test_weekly.py", suite="stage-c-8-gpu-h100", labels=["long"])]
        self._stub_collection(monkeypatch, tests)
        captured = {}

        def fake_run_unittest_files(ci_tests, **kwargs):
            # [FUNCTION DOC] HELPER: Record the test list and keyword options and report success
            # (zero).
            captured["tests"] = ci_tests
            captured.update(kwargs)
            return 0

        monkeypatch.setattr(run_suite_module, "run_unittest_files", fake_run_unittest_files)
        result = run_suite_module.run_a_suite(_run_args(hw="cuda", suite="stage-c-8-gpu-h100", cadence=WEEKLY_CADENCE))
        assert result == 0
        assert _names(captured["tests"]) == {"tests/e2e/test_weekly.py"}
        assert captured["continue_on_error"] is True
        assert captured["gate_write_baseline"] is True


# --- discover_ci_files: location-based discovery across the CI roots --------


class TestDiscoverCiFiles:
    # [CLASS DOC] Tests of location-based CI test file discovery against the real repository tree.
    def test_only_test_prefixed_files_under_known_roots(self, monkeypatch):
        # [FUNCTION DOC] Changes directory to the repository root, runs discovery and asserts: every
        # path starts with one of the four CI roots (fast, fast-gpu, e2e, ci) and has a file name
        # starting with "test_"; no conftest or package init file and no underscore-prefixed
        # helper is listed; and three representative files (a CI test, a fast-gpu test, and the
        # short dumper e2e test) are present.
        # discover_ci_files globs repo-relative; anchor cwd to the repo root
        # so it scans the real tree regardless of where pytest is invoked.
        repo_root = Path(__file__).resolve().parents[3]
        monkeypatch.chdir(repo_root)
        files = discover_ci_files()

        roots = ("tests/fast/", "tests/fast-gpu/", "tests/e2e/", "tests/ci/")
        for f in files:
            assert f.startswith(roots), f
            assert Path(f).name.startswith("test_"), f
        # helpers / conftest / __init__ / _common excluded by the glob pattern
        assert not any(Path(f).name in ("conftest.py", "__init__.py") for f in files)
        assert not any(Path(f).name.startswith("_") for f in files)
        # representative files across the roots are discovered
        assert "tests/ci/test/test_ci_register.py" in files
        assert "tests/fast-gpu/test_semaphore.py" in files
        assert "tests/e2e/short/test_dumper.py" in files  # re-enabled, no carve-out


# --- `filter_tests` label-selection scenarios --------------------------------


@pytest.fixture
def cuda_h100_tests():
    # [FUNCTION DOC] HELPER: Fixture returning six registry records on the 8-GPU H100 suite: two
    # precision, one megatron, one fsdp, one megatron-and-sglang, and one disabled megatron test.
    """A representative `stage-c-8-gpu-h100` registry used across scenarios.

    Composition:
    * 2 precision tests
    * 1 megatron-only test
    * 1 fsdp-only test
    * 1 megatron+sglang test (multi-label, exercises OR semantics)
    * 1 disabled megatron test (must always be classified as skipped)
    """
    return [
        _make("tests/e2e/precision1.py", labels=["precision"]),
        _make("tests/e2e/precision2.py", labels=["precision"]),
        _make("tests/e2e/megatron/m1.py", labels=["megatron"]),
        _make("tests/e2e/fsdp/f1.py", labels=["fsdp"]),
        _make("tests/e2e/megatron/m_or_s.py", labels=["megatron", "sglang"]),
        _make("tests/e2e/megatron/disabled.py", labels=["megatron"], disabled="known flaky"),
    ]


def _names(tests: list[CIRegistry]) -> set[str]:
    # [FUNCTION DOC] HELPER: Return the set of file names of the given registry records.
    return {t.filename for t in tests}


class TestFilterTestsLabels:
    # [CLASS DOC] Scenario tests of the test filter's label selection over the H100 fixture.
    def test_case1_no_labels_selects_no_gpu_tests(self, cuda_h100_tests):
        # [FUNCTION DOC] An empty include label set enables nothing and skips nothing.
        # Every GPU registration has a domain, so an empty include set selects
        # no GPU tests.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(),
        )
        assert enabled == []
        assert skipped == []

    def test_case2_single_domain_label(self, cuda_h100_tests):
        # [FUNCTION DOC] Including only the megatron label enables the two megatron-labelled
        # enabled tests (including the multi-label one) and puts the disabled megatron test in the
        # skipped bucket.
        # `run-ci-megatron` selects only megatron-labelled tests.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron"},
        )
        assert _names(enabled) == {
            "tests/e2e/megatron/m1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        # `disabled.py` matches the megatron label but is disabled, so it
        # belongs to the skipped bucket.
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case3_multiple_domain_labels_or_semantics(self, cuda_h100_tests):
        # [FUNCTION DOC] Including megatron and fsdp enables the union: both megatron tests and the
        # fsdp test.
        # {megatron, fsdp} -> union (OR) of domain matches.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"megatron", "fsdp"},
        )
        assert _names(enabled) == {
            "tests/e2e/megatron/m1.py",
            "tests/e2e/fsdp/f1.py",
            "tests/e2e/megatron/m_or_s.py",
        }

    def test_case4_full_include_set_runs_everything_in_suite(self, cuda_h100_tests):
        # [FUNCTION DOC] Including every known label enables all five enabled tests and skips only
        # the disabled one.
        # The full registry as include set (run-ci-all / --match-all-labels):
        # every enabled hw/suite/cadence match runs.
        enabled, skipped = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=_ALL,
        )
        assert _names(enabled) == {
            "tests/e2e/precision1.py",
            "tests/e2e/precision2.py",
            "tests/e2e/megatron/m1.py",
            "tests/e2e/fsdp/f1.py",
            "tests/e2e/megatron/m_or_s.py",
        }
        assert _names(skipped) == {"tests/e2e/megatron/disabled.py"}

    def test_case5_unknown_pr_side_label_is_silent_noop(self, cuda_h100_tests):
        # [FUNCTION DOC] Including only an unknown label ("foo") enables nothing.
        # Unknown PR-side label (e.g. `run-ci-foo`) produces an empty
        # intersection and selects no GPU tests.
        enabled, _ = filter_tests(
            cuda_h100_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels={"foo"},
        )
        assert enabled == []


# --- filter_tests: broad CI scopes as include sets ---------------------------


@pytest.fixture
def broad_scope_tests():
    # [FUNCTION DOC] HELPER: Fixture returning five H100-suite records labelled precision,
    # megatron, long, ft-short, and ft-long plus long.
    return [
        _make("tests/e2e/precision.py", labels=["precision"]),
        _make("tests/e2e/megatron.py", labels=["megatron"]),
        _make("tests/e2e/long.py", labels=["long"]),
        _make("tests/e2e/ft/short.py", labels=["ft-short"]),
        _make("tests/e2e/ft/long.py", labels=["ft-long", "long"]),
    ]


class TestFilterTestsBroadScopes:
    # [CLASS DOC] Tests of the filter when the include set comes from broad policy scopes.
    def test_image_scope_excludes_long_and_ft_tests(self, broad_scope_tests):
        # [FUNCTION DOC] With the regular-cadence image scope, only the precision and megatron
        # tests are enabled.
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(resolve_policy(REGULAR_CADENCE, {"run-ci-image"}).include_labels),
        )
        assert _names(enabled) == {
            "tests/e2e/precision.py",
            "tests/e2e/megatron.py",
        }

    def test_nightly_scope_excludes_long_and_ft_long_but_selects_ft_short(self, broad_scope_tests):
        # [FUNCTION DOC] With nightly admission and the nightly cadence include set, the precision,
        # megatron and ft-short tests are enabled; the long test and the ft-long-plus-long test are
        # not.
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels=set(resolve_policy(NIGHTLY_CADENCE, set()).include_labels),
        )
        assert _names(enabled) == {
            "tests/e2e/precision.py",
            "tests/e2e/megatron.py",
            "tests/e2e/ft/short.py",
        }

    def test_subtracted_only_test_drops_out_entirely(self):
        # [FUNCTION DOC] Under the nightly include set, a precision test is enabled while an
        # ft-long test and a disabled ft-long test (whose only label was subtracted) are out of
        # scope entirely: not enabled and not even in the skipped list.
        tests = [
            _make("tests/e2e/precision.py", labels=["precision"]),
            _make("tests/e2e/ft/soak.py", labels=["ft-long"]),
            _make("tests/e2e/ft/soak_disabled.py", labels=["ft-long"], disabled="flaky"),
        ]
        enabled, skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels=set(resolve_policy(NIGHTLY_CADENCE, set()).include_labels),
        )
        # A test whose only labels were subtracted is out of scope entirely,
        # including from the skip report.
        assert _names(enabled) == {"tests/e2e/precision.py"}
        assert skipped == []

    def test_all_scope_includes_every_label(self, broad_scope_tests):
        # [FUNCTION DOC] With the regular-cadence "all" scope every record in the fixture is
        # enabled.
        enabled, _ = filter_tests(
            broad_scope_tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=set(resolve_policy(REGULAR_CADENCE, {"run-ci-all"}).include_labels),
        )
        assert _names(enabled) == _names(broad_scope_tests)


# --- filter_tests: hw/suite/cadence eligibility ------------------------------


class TestFilterTestsBaseDimensions:
    # [CLASS DOC] Tests of the filter's hardware, suite, cadence and disabled handling.
    def test_cross_suite_isolation(self):
        # [FUNCTION DOC] A test on the 4-GPU H200 suite is not enabled when filtering the 8-GPU
        # H100 suite, even with every label included.
        # A test registered to stage-c-4-gpu-h200 must not surface in
        # stage-c-8-gpu-h100, even with the full include set.
        tests = [
            _make("tests/e2e/h100/t.py", suite="stage-c-8-gpu-h100", labels=["precision"]),
            _make("tests/e2e/h200/t.py", suite="stage-c-4-gpu-h200", labels=["precision"]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            labels=_ALL,
        )
        assert _names(enabled) == {"tests/e2e/h100/t.py"}

    def test_cross_backend_isolation(self):
        # [FUNCTION DOC] Filtering the CPU stage-a suite with no labels enables the CPU test and
        # not the CUDA one.
        # CPU suite must not pull in CUDA registrations.
        tests = [
            _make("tests/fast/t.py", backend=HWBackend.CPU, suite="stage-a-cpu", labels=[]),
            _make(
                "tests/e2e/h100/t.py",
                backend=HWBackend.CUDA,
                suite="stage-c-8-gpu-h100",
                labels=["precision"],
            ),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CPU,
            "stage-a-cpu",
            labels=set(),
        )
        assert _names(enabled) == {"tests/fast/t.py"}

    @staticmethod
    def _cadence_tests():
        # [FUNCTION DOC] HELPER: Return two megatron-labelled records, one regular and one
        # nightly-only.
        return [
            _make("tests/e2e/regular.py", labels=["megatron"], nightly=False),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True),
        ]

    def test_regular_run_excludes_nightly_only(self):
        # [FUNCTION DOC] Without nightly admission only the regular test is enabled.
        enabled, _ = filter_tests(
            self._cadence_tests(),
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=False,
            labels={"megatron"},
        )
        assert _names(enabled) == {"tests/e2e/regular.py"}

    def test_nightly_run_includes_regular_and_nightly_only(self):
        # [FUNCTION DOC] With nightly admission both tests are enabled.
        enabled, _ = filter_tests(
            self._cadence_tests(),
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels={"megatron"},
        )
        assert _names(enabled) == {"tests/e2e/regular.py", "tests/e2e/nightly.py"}

    def test_disabled_nightly_only_is_skipped_only_when_eligible(self):
        # [FUNCTION DOC] A disabled nightly-only test appears in the skipped list only when nightly
        # tests are admitted; without admission the skipped list is empty.
        tests = [
            _make("tests/e2e/regular.py", labels=["megatron"]),
            _make("tests/e2e/nightly.py", labels=["megatron"], nightly=True, disabled="flaky"),
        ]
        _, regular_skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=False,
            labels={"megatron"},
        )
        _, nightly_skipped = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels={"megatron"},
        )
        assert regular_skipped == []
        assert _names(nightly_skipped) == {"tests/e2e/nightly.py"}

    def test_nightly_only_ft_long_still_obeys_domain_scope(self):
        # [FUNCTION DOC] An ft-long nightly-only test is not enabled by the standard nightly policy
        # (ft-long is subtracted) but is enabled by a nightly policy that explicitly requests the
        # ft-long domain label.
        tests = [_make("tests/e2e/ft/soak.py", labels=["ft-long"], nightly=True)]
        standard_policy = resolve_policy(NIGHTLY_CADENCE, set())
        explicit_policy = resolve_policy(NIGHTLY_CADENCE, {"run-ci-ft-long"})

        standard, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels=set(standard_policy.include_labels),
        )
        explicit, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-c-8-gpu-h100",
            admit_nightly_tests=True,
            labels=set(explicit_policy.include_labels),
        )
        assert standard == []
        assert _names(explicit) == {"tests/e2e/ft/soak.py"}

    def test_unknown_suite_fails_instead_of_green_empty(self):
        # [FUNCTION DOC] Filtering an unknown CUDA suite name raises a value error "Unknown suite
        # stage-c-unknown" rather than returning an empty success.
        with pytest.raises(ValueError, match="Unknown suite stage-c-unknown"):
            filter_tests([], HWBackend.CUDA, "stage-c-unknown")

    def test_known_empty_suite_is_valid(self):
        # [FUNCTION DOC] Filtering an empty registry for the known CPU stage-b suite returns empty
        # enabled and skipped lists without error.
        enabled, skipped = filter_tests([], HWBackend.CPU, "stage-b-cpu")
        assert enabled == []
        assert skipped == []

    def test_stage_b_2_gpu_h200_is_addressable(self):
        # [FUNCTION DOC] A precision test registered on the stage-b 2-GPU H200 suite is enabled
        # when filtering that suite with the precision label.
        # The fast GPU bucket remains a first-class suite.
        tests = [
            _make("tests/fast/q.py", suite="stage-b-2-gpu-h200", labels=["precision"]),
        ]
        enabled, _ = filter_tests(
            tests,
            HWBackend.CUDA,
            "stage-b-2-gpu-h200",
            labels={"precision"},
        )
        assert _names(enabled) == {"tests/fast/q.py"}
