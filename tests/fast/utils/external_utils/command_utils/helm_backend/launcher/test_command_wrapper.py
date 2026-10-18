# [MODULE DOC] Unit tests for the Helm and kubectl command wrapper of the Helm backend launcher
# (command builders and thin subprocess wrappers) plus the release and object naming rules used
# with it. Covered: kubectl log command building (container, previous life, since time, release
# and job selectors), the Helm upgrade command (install-or-upgrade, values order, CI label),
# dependency building only when a locked dependency directory is missing, manifest reading and
# uninstall with the "not found" tolerance, request time bounds on the subprocess layer, JSON
# get errors, create-if-absent / jobs-finished / forced replace / delete-job semantics, the CI
# cleanup commands issued by the entrypoint, Helm release listing filters, the chart directory
# lookup, release name serialization, and Kubernetes object naming limits (length budget,
# truncation, digests for long names, static worker hosts). Subprocess execution is replaced by
# fakes via monkeypatching; temporary directories are used for the chart layout test.
import json
import subprocess
from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils.common import chart_dir
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import (
    _JOB_COMPLETION_JSONPATH,
    Helm,
    Kubectl,
)
from miles.utils.external_utils.command_utils.helm_backend.naming import RUN_ID_MAX_LENGTH, ReleaseName
from miles.utils.workers.k8s_types import Pod
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm import naming
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL


class TestLogCommands:
    # [CLASS DOC] Tests for the kubectl command builders for logs and selectors.
    def test_follows_a_container_by_name(self):
        # [FUNCTION DOC] A follow request for a stateful set target with container
        # "orchestrator" includes the container option followed by the name, the target, and the
        # follow option.
        """A pod may gain a sidecar, and kubectl then refuses to guess which container to read."""
        command = Kubectl.logs_command(
            namespace="rl", target="statefulset/r-miles-run-orchestrator", container="orchestrator", follow=True
        )

        assert command[command.index("-c") + 1] == "orchestrator"
        assert "statefulset/r-miles-run-orchestrator" in command
        assert "--follow" in command

    def test_reads_every_container_of_a_pod_when_none_is_named(self):
        # [FUNCTION DOC] Without a container name the command uses the all-containers option, has
        # no container option, and (follow defaulting to off) no follow option.
        """A worker that crashed is the reason the run stopped, and its log may be in any of its containers."""
        command = Kubectl.logs_command(namespace="rl", target="pod/wb-0")

        assert "--all-containers" in command
        assert "-c" not in command
        assert "--follow" not in command

    def test_resumes_a_container_that_was_replaced(self):
        # [FUNCTION DOC] A request with the previous flag and a since time includes the previous
        # option and the since-time option followed by the given time.
        """The lines a crashed container wrote are the diagnosis, and only --previous still reaches them."""
        command = Kubectl.logs_command(
            namespace="rl", target="pod/wb-0", container="app", previous=True, since_time="2026-01-01T00:00:00Z"
        )

        assert "--previous" in command
        assert command[command.index("--since-time") + 1] == "2026-01-01T00:00:00Z"

    def test_selects_a_release_by_the_label_helm_stamps_on_every_pod_of_it(self):
        # [FUNCTION DOC] The release selector is the instance label key, an equals sign and the
        # release name.
        """An engine and a trainer share only this label, so anything narrower misses half the run."""
        assert Kubectl.release_selector("miles-run-x") == f"{INSTANCE_LABEL}=miles-run-x"

    def test_selects_a_job_by_the_label_the_job_controller_stamps_on_its_pods(self):
        # [FUNCTION DOC] The job selector is the batch job-name label, an equals sign and the
        # job name.
        """A command job's pods are named after their job, and only this label survives a pod restart."""
        assert Kubectl.job_selector("miles-run-command-gpu") == "batch.kubernetes.io/job-name=miles-run-command-gpu"


class TestUpgradeCommand:
    # [CLASS DOC] Tests for the Helm upgrade command builder, taking a release, namespace, chart
    # path, list of values files, and a CI flag.
    def test_installs_a_missing_release_and_updates_an_existing_one(self):
        # [FUNCTION DOC] The command starts with helm, upgrade, the install option and the
        # release name.
        """Relaunching a run id must update it in place, which plain upgrade would refuse to do."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=False)

        assert command[:4] == ["helm", "upgrade", "--install", "r"]

    def test_keeps_the_user_values_ahead_of_the_computed_ones(self):
        # [FUNCTION DOC] Each values file is preceded by the values option, and the files keep
        # their given order (infra before run).
        """A run value must win over a cluster default, and helm lets the later file win."""
        command = Helm.upgrade_command("r", "rl", "/c", ["/infra.yaml", "/run.yaml"], ci_run=False)

        assert command[command.index("/infra.yaml") - 1] == "--values"
        assert command.index("/infra.yaml") < command.index("/run.yaml")

    def test_labels_a_ci_release_so_the_next_run_can_clean_it_up(self):
        # [FUNCTION DOC] With the CI flag, the labels option is followed by the CI label equal
        # to true.
        """The cleanup selects on this label, and an unlabelled CI release is one nothing will ever remove."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=True)

        assert command[command.index("--labels") + 1] == f"{command_wrapper.CI_LABEL}=true"

    def test_leaves_a_human_release_unlabelled(self):
        # [FUNCTION DOC] Without the CI flag there is no labels option.
        """A developer's run carrying the CI label would be uninstalled by the next CI job in that namespace."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=False)

        assert "--labels" not in command

    def test_labels_the_release_rather_than_its_objects(self):
        # [FUNCTION DOC] The labels option appears after the namespace option in the command.
        # KNOWN ISSUE: the assertion only compares the position of two options, which does not
        # show whether the label is applied to the release metadata rather than to its objects,
        # as the test name and docstring claim.
        """helm --labels records release metadata; a values-level label would not be selectable by helm list."""
        command = Helm.upgrade_command("r", "rl", "/c", [], ci_run=True)

        assert command.index("--labels") > command.index("--namespace")


class TestBuildDependencies:
    # [CLASS DOC] Tests for the chart dependency build, which runs the Helm dependency build
    # command only when a dependency recorded in the chart lock file is missing from the charts
    # directory.
    def test_chart_dependencies_are_rebuilt_only_when_a_locked_dependency_is_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # [FUNCTION DOC] Creates a chart directory whose lock file lists dependencies "worker"
        # and "runtime" and whose charts folder contains directories for both. Replaces the
        # process runner with a recorder. Calls the build once (complete), removes the "runtime"
        # directory, and calls again. Asserts exactly one command was run in total: helm
        # dependency build with the chart path (from the second call only).
        """Helm rebuilds a locked chart only after one of its cached dependencies disappears."""
        chart = tmp_path / "chart"
        charts = chart / "charts"
        charts.mkdir(parents=True)
        (chart / "Chart.lock").write_text("dependencies:\n  - name: worker\n  - name: runtime\n")
        (charts / "worker").mkdir()
        runtime = charts / "runtime"
        runtime.mkdir()
        commands: list[list[str]] = []

        def fake_run_process(
            argv: list[str],
            *,
            capture_output: bool,
            check: bool,
            input: str | None = None,
            timeout: float | None = None,
        ) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake process runner: records the command and returns a
            # successful empty result.
            commands.append(argv)
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(command_wrapper, "run_process", fake_run_process)

        Helm.build_dependencies(chart)
        runtime.rmdir()
        Helm.build_dependencies(chart)

        assert commands == [["helm", "dependency", "build", str(chart)]]


class TestGetManifest:
    # [CLASS DOC] Tests for reading a release's manifest: a missing release gives none, other
    # failures raise.
    def test_a_missing_release_is_reported_as_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A failing command (exit 1) whose standard output says the release was
        # not found (some Helm versions print it there) yields none.
        """Helm reports a missing release on stdout in some versions, and that is genuine absence."""
        monkeypatch.setattr(
            command_wrapper,
            "run_process",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                [], returncode=1, stdout="Error: release: not found", stderr=""
            ),
        )

        assert Helm.get_manifest("miles-run-a", "ci") is None

    def test_a_manifest_read_failure_other_than_absence_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A failing command whose error output reports an unreachable cluster
        # raises a runtime error whose message contains that text.
        """An authorization or network failure must remain distinguishable from a missing release."""
        monkeypatch.setattr(
            command_wrapper,
            "run_process",
            lambda *_args, **_kwargs: subprocess.CompletedProcess(
                [], returncode=1, stdout="", stderr="Kubernetes cluster unreachable: access forbidden"
            ),
        )

        with pytest.raises(RuntimeError, match="Kubernetes cluster unreachable: access forbidden"):
            Helm.get_manifest("miles-run-a", "ci")


class TestUninstallIfPresent:
    # [CLASS DOC] Tests for the uninstall that tolerates an already-absent release.
    def test_uninstalls_the_named_release_in_its_namespace(self, monkeypatch):
        # [FUNCTION DOC] Replaces the raw Helm runner with a recorder; asserts the call is
        # exactly the uninstall subcommand, the release name, the namespace option and the
        # namespace.
        """A comparison handoff must tear down the release that still owns the previous side's GPUs."""
        calls: list[tuple[str, ...]] = []

        def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake raw runner: records the arguments and returns a
            # successful empty result.
            calls.append(arguments)
            return subprocess.CompletedProcess(arguments, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(Helm, "run_raw", run_raw)

        Helm.uninstall_if_present(release="miles-run-a", namespace="ci")

        assert calls == [("uninstall", "miles-run-a", "--namespace", "ci")]

    def test_an_auto_uninstall_winning_the_race_is_already_the_wanted_result(self, monkeypatch):
        # [FUNCTION DOC] A failing uninstall whose error says the release was not loaded / not
        # found is treated as success: no exception is raised.
        """The chart can remove the release between the verdict and the explicit handoff cleanup."""
        monkeypatch.setattr(
            Helm,
            "run_raw",
            lambda *_arguments: subprocess.CompletedProcess(
                [],
                returncode=1,
                stdout="",
                stderr="Error: uninstall: Release not loaded: miles-run-a: release: not found",
            ),
        )

        Helm.uninstall_if_present(release="miles-run-a", namespace="ci")

    def test_an_uninstall_failure_other_than_absence_stops_the_handoff(self, monkeypatch):
        # [FUNCTION DOC] A failing uninstall with the error text "forbidden" raises a runtime
        # error containing that text.
        """Starting the target after a forbidden cleanup would recreate the GPU overlap this guard prevents."""
        monkeypatch.setattr(
            Helm,
            "run_raw",
            lambda *_arguments: subprocess.CompletedProcess([], returncode=1, stdout="", stderr="forbidden"),
        )

        with pytest.raises(RuntimeError, match="forbidden"):
            Helm.uninstall_if_present(release="miles-run-a", namespace="ci")


def _kubectl_answering(monkeypatch, *, returncode: int, stdout: str = "", stderr: str = "") -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Replaces the wrapper's process runner with a fake that asserts the
    # program is kubectl, records the remaining arguments in a list, raises a called-process error
    # (carrying the error text) when the caller asked for checking and the configured return code
    # is non-zero, and otherwise returns a completed process with the configured return code and
    # output. Returns the list that collects the recorded argument lists.
    commands: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: The fake runner described in the enclosing helper.
        assert argv[0] == "kubectl", f"only kubectl is expected to reach the process layer, got {argv[0]}"
        commands.append(argv[1:])
        if kwargs.get("check") and returncode != 0:
            raise subprocess.CalledProcessError(returncode, argv, stderr=stderr)
        return subprocess.CompletedProcess(args=argv, returncode=returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    return commands


class TestRequestBounds:
    # [CLASS DOC] Tests for how kubectl request time bounds are applied.
    def test_reads_and_creates_bound_the_subprocess_instead_of_passing_a_kubectl_override(self, monkeypatch):
        # [FUNCTION DOC] Runs a JSON get of pods and a create-if-absent through a fake runner
        # that records the arguments and the timeout keyword. Asserts that no command carries the
        # request-timeout option and that the subprocess timeouts used are 30 seconds for the get
        # and 60 seconds for the create, in that order.
        """A `--request-timeout` override stops kubectl from falling back to the in-cluster config a workbench relies on."""
        calls: list[tuple[list[str], float | None]] = []

        def fake_run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
            # [FUNCTION DOC] HELPER: Records the arguments after the program name and the
            # timeout keyword, and returns a successful empty result.
            calls.append((argv[1:], kwargs.get("timeout")))
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(command_wrapper, "run_process", fake_run)

        Kubectl.get_json("pods", return_type=Pod, namespace="training")
        Kubectl.create_if_absent("/etc/miles/job.yaml")

        assert all("--request-timeout" not in argv for argv, _ in calls)
        assert [timeout for _, timeout in calls] == [30.0, 60.0]


class TestGetJson:
    # [CLASS DOC] Tests for the JSON get helper's error reporting.
    def test_a_failed_get_is_not_reported_as_an_absent_object(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] A get that fails with exit code 23 and an error message raises a runtime
        # error whose message contains "code 23: " followed by the error text, instead of
        # returning none.
        """A failed lookup must expose its exit code and stderr instead of looking like an absent object."""
        _kubectl_answering(monkeypatch, returncode=23, stderr="the api server refused the request")

        with pytest.raises(RuntimeError, match="code 23: the api server refused the request"):
            Kubectl.get_json("pod", return_type=Pod, name="trainer-0", namespace="rl")


class TestCreateIfAbsent:
    # [CLASS DOC] Tests for create-if-absent, which runs kubectl create from a manifest file and
    # returns true when it created the objects, false when they already existed, and raises for
    # any other failure.
    def test_creates_the_objects_of_a_rendered_manifest(self, monkeypatch):
        # [FUNCTION DOC] On success it returns true and the only command is create with the
        # file option and the manifest path.
        """kubectl apply would adopt an object helm owns; create is what refuses to touch one."""
        commands = _kubectl_answering(monkeypatch, returncode=0, stderr="")

        assert Kubectl.create_if_absent("/etc/miles/job.yaml")
        assert commands == [["create", "-f", "/etc/miles/job.yaml"]]

    def test_reports_an_object_that_was_already_there_without_failing(self, monkeypatch):
        # [FUNCTION DOC] A failure whose error text contains the AlreadyExists marker gives a
        # false result without raising.
        """Its callers retry after a restart, and the whole point is that the second attempt is harmless."""
        _kubectl_answering(
            monkeypatch, returncode=1, stderr='Error from server (AlreadyExists): jobs "u" already exists'
        )

        assert not Kubectl.create_if_absent("/etc/miles/job.yaml")

    def test_refuses_to_read_any_other_failure_as_idempotence(self, monkeypatch):
        # [FUNCTION DOC] A forbidden error raises a runtime error whose message contains "Could
        # not create".
        """A forbidden create means the object is missing, and pretending otherwise loses it silently."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="Error from server (Forbidden): jobs is forbidden")

        with pytest.raises(RuntimeError, match="Could not create"):
            Kubectl.create_if_absent("/etc/miles/job.yaml")


class TestJobsFinished:
    # [CLASS DOC] Tests for checking whether the jobs of a manifest file already completed, using
    # a get command with a JSON path output that yields success and failure counters separated by
    # a comma (a non-empty counter means finished).
    def test_reports_a_job_that_already_succeeded_or_failed(self, monkeypatch):
        # [FUNCTION DOC] Output "1," (one success) gives true; the command is get with the file
        # option, the manifest path, the output option and the completion JSON path constant.
        """A job holding its name after it ran never runs again, so its caller has to recreate it."""
        commands = _kubectl_answering(monkeypatch, returncode=0, stdout="1,")

        assert Kubectl.jobs_finished("/etc/miles/job.yaml")
        assert commands == [["get", "-f", "/etc/miles/job.yaml", "--output", _JOB_COMPLETION_JSONPATH]]

    def test_reports_a_job_that_has_not_run_yet_as_unfinished(self, monkeypatch):
        # [FUNCTION DOC] Output "," (both counters empty) gives false.
        """An armed job still does its work, and replacing it would throw that work away."""
        _kubectl_answering(monkeypatch, returncode=0, stdout=",")

        assert not Kubectl.jobs_finished("/etc/miles/job.yaml")

    def test_refuses_to_read_an_unanswered_query_as_unfinished(self, monkeypatch):
        # [FUNCTION DOC] A failing query raises a runtime error whose message contains "Could
        # not read".
        """Guessing here adopts a job that already ran and leaves the release installed forever."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="Error from server: etcdserver: request timed out")

        with pytest.raises(RuntimeError, match="Could not read"):
            Kubectl.jobs_finished("/etc/miles/job.yaml")


class TestReplace:
    # [CLASS DOC] Tests for the forced replace of the objects in a manifest file.
    def test_forces_the_objects_of_a_rendered_manifest_over_the_ones_holding_their_names(self, monkeypatch):
        # [FUNCTION DOC] The only command is replace with the force option and the file option
        # with the manifest path.
        """A job's pod template is immutable, so only a forced replace can put a fresh job under that name."""
        commands = _kubectl_answering(monkeypatch, returncode=0)

        Kubectl.replace("/etc/miles/job.yaml")

        assert commands == [["replace", "--force", "-f", "/etc/miles/job.yaml"]]

    def test_lets_a_caller_that_cannot_go_on_without_the_replacement_fail(self, monkeypatch):
        # [FUNCTION DOC] A forbidden error raises a runtime error whose message contains
        # "Could not replace".
        """Reporting a replace that never happened would tell the run that it uninstalls itself when it does not."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="Error from server (Forbidden): jobs is forbidden")

        with pytest.raises(RuntimeError, match="Could not replace"):
            Kubectl.replace("/etc/miles/job.yaml")


class TestDeleteJob:
    # [CLASS DOC] Tests for deleting a job by name in a namespace, with a check flag that makes
    # failures raise (called-process error) and otherwise tolerates them.
    def test_treats_a_job_that_is_not_there_as_deleted(self, monkeypatch):
        # [FUNCTION DOC] The delete command includes the ignore-not-found option.
        """The launcher deletes a job that usually does not exist, which is the outcome it wants anyway."""
        commands = _kubectl_answering(monkeypatch, returncode=0, stderr="")
        Kubectl.delete_job("miles-run-x-uninstall", namespace="rl", check=True)

        assert "--ignore-not-found" in commands[0]

    def test_waits_for_the_pods_of_the_job_it_deletes(self, monkeypatch):
        # [FUNCTION DOC] The delete command's cascade option has the value "foreground" (the
        # command waits for dependent pods).
        """The launcher installs over the deleted job, and a pod that outlives it uninstalls the new release."""
        commands = _kubectl_answering(monkeypatch, returncode=0, stderr="")
        Kubectl.delete_job("miles-run-x-uninstall", namespace="rl", check=True)

        assert commands[0][commands[0].index("--cascade") + 1] == "foreground"

    def test_lets_a_caller_that_cannot_go_on_without_the_deletion_fail(self, monkeypatch):
        # [FUNCTION DOC] With the check flag, a failing delete raises a called-process error.
        """Installing over a job that is still armed hands the new release to the old run's uninstall."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="the api server refused")

        with pytest.raises(subprocess.CalledProcessError):
            Kubectl.delete_job("miles-run-x-uninstall", namespace="rl", check=True)

    def test_stays_tolerant_for_the_cleanup_of_a_command_job(self, monkeypatch):
        # [FUNCTION DOC] Without the check flag, a failing delete raises nothing.
        """That caller deletes the same job twice around a run, and neither call is worth failing over."""
        _kubectl_answering(monkeypatch, returncode=1, stderr="the api server refused")

        Kubectl.delete_job("miles-run-command-convert", namespace="rl")


LAUNCHING_RUN_ID = "260101-000000-000"


def _recorded_ci_cleanup(
    monkeypatch: pytest.MonkeyPatch, namespace: str, *, listed: list[dict] | None = None
) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Replaces the wrapper's low-level runner with a fake that records
    # every command and answers with exit code 0 and the JSON text of the given releases (default
    # empty list) as output, then runs the entrypoint's leftover CI release cleanup for the given
    # namespace keeping the launching run id. Returns the recorded commands.
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: Records the command and returns the canned JSON output.
        commands.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=json.dumps(listed or []), stderr="")

    monkeypatch.setattr(command_wrapper, "_run", fake_run)
    entrypoint._uninstall_leftover_ci_releases(namespace, keep_run_id=LAUNCHING_RUN_ID)
    return commands


class TestCiCleanup:
    # [CLASS DOC] Tests of the commands issued by the CI cleanup (also covered in the dedicated
    # CI cleanup test module).
    def test_narrows_the_search_by_both_namespace_and_label(self, monkeypatch):
        # [FUNCTION DOC] The first command (the listing) carries the namespace option "ci-runner-3"
        # and the selector option equal to the CI label with value true.
        """Deleting another user's run would kill a live experiment, so neither filter may be dropped."""
        command = _recorded_ci_cleanup(monkeypatch, "ci-runner-3")[0]

        assert command[command.index("--namespace") + 1] == "ci-runner-3"
        assert command[command.index("--selector") + 1] == f"{command_wrapper.CI_LABEL}=true"

    def test_reads_the_release_names_helm_reports(self):
        # [FUNCTION DOC] Decodes a JSON array of two releases directly and checks the names.
        # KNOWN ISSUE: this test only exercises the standard JSON decoder on literal text; it does
        # not call any code of the project, so it cannot fail because of the project.
        """The names drive uninstall, so a parse that silently returns nothing would leave releases behind."""
        output = json.dumps([{"name": "miles-run-a", "namespace": "ci"}, {"name": "miles-run-b"}])

        assert [release["name"] for release in json.loads(output or "[]")] == ["miles-run-a", "miles-run-b"]

    def test_treats_no_output_as_nothing_to_clean(self):
        # [FUNCTION DOC] Decodes the fallback text for empty output and checks it is an empty
        # list.
        # KNOWN ISSUE: like the previous test, it only exercises the standard JSON decoder on a
        # literal (the expression "empty string or fallback" is always the fallback), not any
        # project code.
        """helm prints nothing when no release matches, and that is not an error."""
        assert json.loads("" or "[]") == []

    def test_uninstalls_inside_the_namespace_it_was_told(self, monkeypatch):
        # [FUNCTION DOC] With one listed release "miles-run-a" and namespace "ci", the second
        # command is exactly the Helm uninstall of that release with the namespace option.
        """A release name exists per namespace, so a missing namespace could hit a different one."""
        commands = _recorded_ci_cleanup(monkeypatch, "ci", listed=[{"name": "miles-run-a"}])

        assert commands[1] == ["helm", "uninstall", "miles-run-a", "--namespace", "ci"]


def _listed_releases(monkeypatch: pytest.MonkeyPatch, **kwargs) -> list[str]:
    # [FUNCTION DOC] HELPER: Replaces the low-level runner with a fake that records commands and
    # returns a JSON list holding one release "miles-run-a-all". Calls the release listing with
    # the given keyword arguments and returns the first (listing) command that was issued.
    commands: list[list[str]] = []

    def fake_run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: Records the command and returns the canned one-release JSON.
        commands.append(command)
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=json.dumps([{"name": "miles-run-a-all"}]), stderr=""
        )

    monkeypatch.setattr(command_wrapper, "_run", fake_run)
    Helm.list_releases(**kwargs)
    return commands[0]


class TestListReleases:
    # [CLASS DOC] Tests for the Helm release listing, which takes a namespace and optional label
    # selector and name filter and returns a list of release names.
    def test_a_name_filter_is_passed_to_helm_rather_than_applied_afterwards(self, monkeypatch):
        # [FUNCTION DOC] A name filter appears in the command as the filter option followed by
        # the filter text.
        """helm truncates to its 256-release maximum before returning, and a filter is applied before that."""
        command = _listed_releases(monkeypatch, namespace="rl", name_filter="^miles-run-a-")

        assert command[command.index("--filter") + 1] == "^miles-run-a-"

    def test_a_listing_that_names_no_filter_asks_for_the_whole_namespace(self, monkeypatch):
        # [FUNCTION DOC] Without a name filter the command has no filter option.
        """The ci cleanup selects on a label instead, and a filter here would hide the releases it removes."""
        assert "--filter" not in _listed_releases(monkeypatch, namespace="rl")

    def test_a_filter_and_a_label_selector_narrow_the_same_listing(self, monkeypatch):
        # [FUNCTION DOC] With both a selector ("ci=true") and a filter, the command carries both
        # options with their values.
        """Neither is a replacement for the other, so passing one may not drop the other."""
        command = _listed_releases(monkeypatch, namespace="rl", selector="ci=true", name_filter="^miles-run-a-")

        assert command[command.index("--selector") + 1] == "ci=true"
        assert command[command.index("--filter") + 1] == "^miles-run-a-"

    def test_it_reads_back_the_names_helm_reported(self, monkeypatch):
        # [FUNCTION DOC] The returned list equals the names in Helm's JSON output.
        """The filter is only worth passing if what comes back is still the list of release names."""
        monkeypatch.setattr(
            command_wrapper,
            "_run",
            lambda command, capture_output: subprocess.CompletedProcess(
                args=command, returncode=0, stdout=json.dumps([{"name": "miles-run-a-all"}]), stderr=""
            ),
        )

        assert Helm.list_releases(namespace="rl", name_filter="^miles-run-a-") == ["miles-run-a-all"]


class TestChartDir:
    # [CLASS DOC] Test of the chart directory lookup.
    def test_finds_the_chart_inside_the_checkout(self):
        # [FUNCTION DOC] For a repository base directory "/repo" the chart directory is the
        # "charts/miles-run" folder inside it (compared as a POSIX path string).
        """The launcher installs the chart of the code it runs, not one from a registry."""
        assert chart_dir(repo_base_dir="/repo").as_posix() == "/repo/charts/miles-run"


LONGEST_RUN_ID = "a" * RUN_ID_MAX_LENGTH


def _unsplit(run_id: str) -> str:
    # [FUNCTION DOC] HELPER: Serializes the release name of an unsplit run (deploy component "all"
    # and no deploy instance id) for the given run id.
    return ReleaseName(run_id=run_id, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()


class TestReleaseName:
    # [CLASS DOC] Tests for the release naming rule: chart name prefix, run id, component.
    def test_a_release_is_the_chart_name_the_run_id_and_the_component(self):
        # [FUNCTION DOC] An unsplit run with id "260101-000000-000" serializes to
        # "miles-run-260101-000000-000-all".
        """The launcher finds a run's release again from the run id alone, so the rule is fixed."""
        assert _unsplit("260101-000000-000") == "miles-run-260101-000000-000-all"

    def test_the_run_prefix_is_what_every_release_of_that_run_starts_with(self):
        # [FUNCTION DOC] The run prefix for an id is a prefix of both the unsplit release name
        # and the name of an inference release with instance id "dc1" for the same run.
        """A filter built from it has to match the trainer and inference releases of the run as well."""
        prefix = ReleaseName.run_prefix(run_id="260101-000000-000")

        assert _unsplit("260101-000000-000").startswith(prefix)
        engines = ReleaseName(
            run_id="260101-000000-000", deploy_component=DeployComponent.INFERENCE, deploy_instance_id="dc1"
        )
        assert engines.serialize().startswith(prefix)

    def test_the_run_prefix_ends_where_the_component_begins(self):
        # [FUNCTION DOC] The prefix of run id "...-000" does not start with the prefix of the
        # shorter id "...-00": the prefix includes the trailing separator, so the shorter id's
        # prefix cannot match the releases of a run whose id merely starts with it.
        """Without the separator the prefix of one run also matches a run whose id merely starts with it."""
        assert not ReleaseName.run_prefix(run_id="260101-000000-000").startswith(
            ReleaseName.run_prefix(run_id="260101-000000-00")
        )

    def test_the_same_run_id_always_names_the_same_release(self):
        # [FUNCTION DOC] Serializing the longest permitted run id twice gives equal names.
        # KNOWN ISSUE: it compares two calls of the same function with the same argument, so it
        # is trivially true for any deterministic function.
        """Relaunching a run upgrades its release; a fresh name would deploy a second copy instead."""
        assert _unsplit(LONGEST_RUN_ID) == _unsplit(LONGEST_RUN_ID)


class TestComponentName:
    # [CLASS DOC] Tests for the Kubernetes object naming function taking a release name and a
    # component name: the result is release plus chart name plus component (the chart name is
    # skipped when the release already starts with it), and is bounded so that suffixes added by
    # Kubernetes (cell index and revision hash) still fit within the maximum object name length,
    # using truncation plus a digest for long inputs so different inputs stay distinct.
    def test_an_object_is_the_release_the_chart_name_and_the_component(self):
        # [FUNCTION DOC] Release "myrun" and component "orchestrator" give
        # "myrun-miles-run-orchestrator".
        """Every object of a run is traceable to the release that made it."""
        assert naming.component_name("myrun", "orchestrator") == "myrun-miles-run-orchestrator"

    def test_a_release_that_already_carries_the_chart_name_is_not_told_it_twice(self):
        # [FUNCTION DOC] A release starting with the chart name "miles-run" does not get it
        # inserted again.
        """The launcher's own releases start with the chart name, and doubling it wastes the budget."""
        assert naming.component_name("miles-run-260101", "orchestrator") == "miles-run-260101-orchestrator"

    def test_a_name_leaves_room_for_every_suffix_kubernetes_appends_below_it(self):
        # [FUNCTION DOC] For a 200-character release, the name length plus the longest cell
        # index suffix plus the longest revision hash suffix stays within the maximum object name
        # length.
        """A pool name grows a cell index and then a revision hash, and a label value stops at 63."""
        name = naming.component_name("a" * 200, "orchestrator")
        appended = len(naming.LONGEST_CELL_INDEX_SUFFIX) + len(naming.LONGEST_REVISION_HASH_SUFFIX)

        assert len(name) + appended <= naming.MAX_OBJECT_NAME_LENGTH

    def test_the_component_survives_a_release_long_enough_to_fill_the_budget(self):
        # [FUNCTION DOC] For a 200-character release the name still ends with "-orchestrator"
        # (the release part is the one truncated).
        """Truncating the component instead of the release would render two workloads under one name."""
        assert naming.component_name("a" * 200, "orchestrator").endswith("-orchestrator")

    def test_two_components_of_one_run_never_collapse_onto_the_same_name(self):
        # [FUNCTION DOC] For a very long release, components "leader" and "logger" give
        # different names.
        """Truncating a name already at the limit silently merges two workloads into one object."""
        release = "a" * 200

        assert naming.component_name(release, "leader") != naming.component_name(release, "logger")

    def test_a_truncated_prefix_never_ends_on_the_separator(self):
        # [FUNCTION DOC] For releases consisting of 1 to 59 letters "b", the generated name never
        # contains a doubled dash.
        """A doubled dash is legal but reads as an empty segment, and drifts from the recorded names."""
        for length in range(1, 60):
            assert "--" not in naming.component_name("b" * length, "orchestrator")

    def test_a_component_longer_than_its_own_budget_is_hashed(self):
        # [FUNCTION DOC] A 200-character release with a long component (prefix plus 60 letters)
        # still yields a name that fits with the Kubernetes suffixes within the maximum length.
        """Letting it eat the whole budget leaves no room for the release digest, so two runs collide."""
        name = naming.component_name("a" * 200, "trainer-controller-" + "m" * 60)
        appended = len(naming.LONGEST_CELL_INDEX_SUFFIX) + len(naming.LONGEST_REVISION_HASH_SUFFIX)

        assert len(name) + appended <= naming.MAX_OBJECT_NAME_LENGTH

    def test_two_long_components_of_one_run_never_collapse_onto_the_same_name(self):
        # [FUNCTION DOC] Two long components that differ only in a final "-a" versus "-b" suffix
        # give different names for the same release (a digest of the full component is kept).
        """Truncation alone maps every long component onto one name; the digest is what keeps them apart."""
        prefix = "trainer-controller-" + "m" * 60

        assert naming.component_name("myrun", f"{prefix}-a") != naming.component_name("myrun", f"{prefix}-b")

    def test_a_long_component_leaves_the_release_its_digest(self):
        # [FUNCTION DOC] Two 200-character releases of different letters with the same long
        # component give different names.
        """Two releases whose names are truncated to the same prefix are only told apart by that digest."""
        long_component = "trainer-controller-" + "m" * 60

        assert naming.component_name("a" * 200, long_component) != naming.component_name("b" * 200, long_component)

    def test_the_same_release_and_component_always_name_the_same_object(self):
        # [FUNCTION DOC] Calling the naming function twice with the same release and component
        # gives equal results.
        # KNOWN ISSUE: it only compares two identical calls, so it is trivially true for any
        # deterministic function.
        """helm upgrade replaces an object in place only while its name is unchanged."""
        assert naming.component_name("miles-run-x", "trainer-engine-actor") == naming.component_name(
            "miles-run-x", "trainer-engine-actor"
        )


class TestStaticWorkerHost:
    # [CLASS DOC] Test of the host name builder for a static worker cell.
    def test_a_static_cell_is_reached_through_its_own_pod_of_the_headless_service(self):
        # [FUNCTION DOC] For release "myrun", component "session-server" and cell index 1, the
        # host is the pod name (object name plus "-1") followed by a dot and the headless service
        # name (the object name).
        """A pool of session servers is several addresses, and pod zero can answer only one of them."""
        assert naming.static_worker_host("myrun", "session-server", 1) == (
            "myrun-miles-run-session-server-1.myrun-miles-run-session-server"
        )
