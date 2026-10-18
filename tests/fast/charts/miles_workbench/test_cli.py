# [MODULE DOC]
# Tests of the workbench command line tool (install, uninstall, stop, exec, collect-diagnosis and the
# dry-run mode) run as a real subprocess with fake helm and kubectl executables placed first on the
# path. The fakes append every invocation to a calls log and are steered by three control files:
# a refused list (lines of "status|command prefix" print a server error and exit 1 for commands
# starting with that prefix), a failing list (command prefixes that exit 1, with a previous-container
# error for log commands), and files giving the pods and service accounts to list. Tests assert on
# the recorded calls, exit codes and stderr text: ordering of preflight, dependency build and
# upgrade; namespace existence handling; bounded rollout wait; namespace occupancy check; releases
# named for stop; diagnosis directory contents; dry-run having no side effects; exec argument
# handling; missing binaries; usage errors; object name length rules; and that importing the tool
# pulls in no training libraries.
import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.fast.charts.utils import CHART_DIR, REPO_ROOT, run_workbench

from miles.utils.external_utils.miles_workbench import __main__ as workbench_main
from miles.utils.external_utils.miles_workbench.naming import DEFAULT_RELEASE, PACKAGE, object_name, run_release_name
from miles.utils.workers.types import DeployComponent


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    # [FUNCTION DOC] HELPER: Fixture creating fake helm and kubectl bash scripts in a temporary bin
    # directory and making that directory plus the system binary directories the whole path. Each
    # script: appends "<binary> <args>" to the calls log; reads the refused file and, for any line
    # "status|prefix" whose prefix matches the start of the command, prints a server error with
    # that status to stderr and exits 1; reads the failing file and for a matching prefix exits 1
    # (printing a previous-container-not-found error when the command ends with the previous flag);
    # for a kubectl pod listing prints a JSON item list built from the pods file (one name per
    # line, used for name and uid) if non-empty; for a kubectl service accounts listing prints the
    # service accounts file; otherwise exits 0. Returns a mapping of the control file paths (calls,
    # failing, refused, pods, service accounts), all initially empty.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls_path = tmp_path / "calls.log"
    failing_path = tmp_path / "failing"
    failing_path.write_text("")
    refused_path = tmp_path / "refused"
    refused_path.write_text("")
    pods_path = tmp_path / "pods"
    pods_path.write_text("")
    service_accounts_path = tmp_path / "service-accounts"
    service_accounts_path.write_text("")

    for binary in ("helm", "kubectl"):
        (bin_dir / binary).write_text(
            "#!/usr/bin/env bash\n"
            f'command="{binary} $@"\n'
            f'echo "$command" >> {calls_path}\n'
            'while IFS= read -r line || [ -n "$line" ]; do\n'
            '  [ -z "$line" ] && continue\n'
            '  status="${line%%|*}"\n'
            '  pattern="${line#*|}"\n'
            '  case "$command" in\n'
            '    $pattern*) echo "Error from server ($status): refused" >&2; exit 1 ;;\n'
            "  esac\n"
            f"done < {refused_path}\n"
            'while IFS= read -r pattern || [ -n "$pattern" ]; do\n'
            '  [ -z "$pattern" ] && continue\n'
            '  case "$command" in\n'
            "    $pattern*)\n"
            '      case "$command" in\n'
            '        *" --previous") echo "Error from server (BadRequest): previous terminated container not found" >&2 ;;\n'
            "      esac\n"
            "      exit 1 ;;\n"
            "  esac\n"
            f"done < {failing_path}\n"
            f'if [ "{binary} $1 $2" = "kubectl get pods" ]; then\n'
            '  items=""\n'
            '  while IFS= read -r pod || [ -n "$pod" ]; do\n'
            '    [ -z "$pod" ] && continue\n'
            '    [ -n "$items" ] && items="$items,"\n'
            '    items="$items{\\"metadata\\":{\\"name\\":\\"$pod\\",\\"uid\\":\\"$pod\\"}}"\n'
            f"  done < {pods_path}\n"
            '  [ -n "$items" ] && echo "{\\"items\\":[$items]}"\n'
            "fi\n"
            f'if [ "{binary} $1 $2" = "kubectl get serviceaccounts" ]; then\n'
            f"  cat {service_accounts_path}\n"
            "fi\n"
            "exit 0\n"
        )
        (bin_dir / binary).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")

    return dict(
        calls_path=calls_path,
        failing_path=failing_path,
        refused_path=refused_path,
        pods_path=pods_path,
        service_accounts_path=service_accounts_path,
    )


def run_cli(*args: str) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Runs the workbench tool with the arguments, capturing output; returns
    # the completed process.
    return run_workbench(*args, capture_output=True)


def calls_of(fake_tools: dict) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the recorded fake tool invocations as a list of lines (the
    # calls log must exist).
    return fake_tools["calls_path"].read_text().splitlines()


def _record_launch(run_directory: Path, launch_token: str) -> None:
    # [FUNCTION DOC] HELPER: Writes a launch record JSON file for the token under the run
    # directory's launches folder (creating folders); it holds the path of the matching orchestrator
    # state file under the state folder (not created).
    path = run_directory / "launches" / f"launch-{launch_token}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state_file = run_directory / "state" / f"orchestrator-{launch_token}.state"
    path.write_text(json.dumps({"state_file": str(state_file)}))


class TestInstall:
    # [CLASS DOC] Tests of the install subcommand; examples use namespace rl and release wb, whose
    # object name is "wb-miles-workbench".
    def test_it_checks_then_vendors_then_upgrades(self, fake_tools):
        # [FUNCTION DOC] Exit code zero; calls include a permission check, a dependency build of the
        # chart directory and a helm upgrade-install with namespace and object name; and the
        # dependency build comes before the upgrade.
        """The three steps a user would otherwise run by hand, in the only order that works."""
        result = run_cli("install", "-n", "rl", "-r", "wb")
        calls = fake_tools["calls_path"].read_text().splitlines()

        assert result.returncode == 0, result.stdout + result.stderr
        assert any(call.startswith("kubectl auth can-i") for call in calls)
        assert f"helm dependency build {CHART_DIR}" in calls
        assert (
            f"helm upgrade --install wb {CHART_DIR} --namespace rl --set-string objectName=wb-miles-workbench" in calls
        )
        assert calls.index(f"helm dependency build {CHART_DIR}") < calls.index(
            f"helm upgrade --install wb {CHART_DIR} --namespace rl --set-string objectName=wb-miles-workbench"
        )

    def test_a_failed_preflight_stops_before_helm_runs(self, fake_tools):
        # [FUNCTION DOC] With every kubectl call failing, exit code is 1 and no helm upgrade was run.
        """Installing anyway would leave a workbench whose token grants nothing."""
        (fake_tools["failing_path"]).write_text("kubectl")

        result = run_cli("install", "-n", "rl", "-r", "wb")

        assert result.returncode == 1
        assert "helm upgrade" not in fake_tools["calls_path"].read_text()

    def test_the_rbac_options_reach_both_the_preflight_and_helm(self, fake_tools):
        # [FUNCTION DOC] With the no-rbac and no-lws flags, no service account creation check or
        # leader-worker-set check is made, and helm receives both rbac values set to false.
        """Passing them to only one of the two is the mismatch this command exists to remove."""
        run_cli("install", "-n", "rl", "-r", "wb", "--no-rbac", "--no-lws")
        calls = fake_tools["calls_path"].read_text()

        assert "auth can-i create serviceaccounts" not in calls
        assert "leaderworkersets" not in calls
        assert "--set rbac.create=false --set rbac.leaderWorkerSets=false" in calls

    def test_values_files_and_raw_overrides_are_passed_through(self, fake_tools, tmp_path):
        # [FUNCTION DOC] A values file and a raw set override appear in the helm call, file first.
        """A per-cluster values file plus the occasional --set is how this chart is actually installed."""
        values = tmp_path / "cluster.yaml"
        values.write_text("{}")

        run_cli(
            "install",
            "-n",
            "rl",
            "-r",
            "wb",
            "-f",
            str(values),
            "--set",
            "resources.requests.cpu=4",
        )

        assert f"-f {values} --set resources.requests.cpu=4" in fake_tools["calls_path"].read_text()

    def test_the_preflight_can_be_skipped(self, fake_tools):
        # [FUNCTION DOC] With the skip-preflight flag, exit code zero and no permission check calls.
        """A denied check is sometimes a wrong answer from the cluster, not a real blocker."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "auth can-i" not in fake_tools["calls_path"].read_text()

    def test_an_existing_namespace_is_left_alone(self, fake_tools):
        # [FUNCTION DOC] The tool queries the namespace (a get call) and, since it exists, never
        # calls namespace creation.
        """Creating a namespace that already exists needs cluster-scoped rights this user is not assumed to have."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl get namespace -- rl" in calls_of(fake_tools)
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_a_missing_namespace_is_created_before_anything_else(self, fake_tools):
        # [FUNCTION DOC] When the namespace query is refused with a not-found error, the tool creates
        # the namespace, before the first helm call.
        """helm cannot install into a namespace that does not exist, and the preflight would fail on it first."""
        fake_tools["refused_path"].write_text("NotFound|kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")
        calls = calls_of(fake_tools)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl create namespace rl" in calls
        assert calls.index("kubectl create namespace rl") < min(
            index for index, call in enumerate(calls) if call.startswith("helm")
        )

    def test_a_namespace_this_account_may_not_read_is_not_created(self, fake_tools):
        # [FUNCTION DOC] When the namespace query is refused as forbidden, the tool does not create
        # it; it falls back to listing service accounts in the namespace.
        """A namespace-scoped user always gets Forbidden here, and creating one needs rights they never have."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)
        assert "kubectl get serviceaccounts -n rl -o name" in calls_of(fake_tools)

    def test_an_empty_listing_inside_a_namespace_does_not_confirm_it_exists(self, fake_tools):
        # [FUNCTION DOC] Forbidden namespace query and an empty service account listing: install
        # still succeeds but warns on stderr that nothing says whether the namespace exists.
        """A namespace that does not exist answers a namespaced listing exactly as an existing empty one does."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "nothing here says whether namespace rl exists" in result.stderr

    def test_a_namespace_that_lists_a_service_account_is_taken_to_exist(self, fake_tools):
        # [FUNCTION DOC] Forbidden namespace query but a non-empty service account listing: install
        # succeeds with no uncertainty warning.
        """A non-empty namespaced listing is the one answer that proves the namespace is there."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace")
        fake_tools["service_accounts_path"].write_text("serviceaccount/default\n")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "nothing here says whether namespace rl exists" not in result.stderr

    def test_a_namespace_absent_from_inside_stops_the_install(self, fake_tools):
        # [FUNCTION DOC] Forbidden namespace query and a not-found service account listing: exit
        # code 1, stderr says the namespace does not exist, and no namespace creation was attempted.
        """Every later step would fail on the missing namespace, one confusing error at a time."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get namespace\nNotFound|kubectl get serviceaccounts")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 1
        assert "does not exist" in result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_a_namespace_lookup_that_fails_for_another_reason_is_not_a_missing_namespace(self, fake_tools):
        # [FUNCTION DOC] A plain failure of the namespace query: exit code 1, stderr says it could
        # not read the namespace, and no creation was attempted.
        """A timed-out apiserver would otherwise send the user into a create that fails for a third reason."""
        fake_tools["failing_path"].write_text("kubectl get namespace")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 1
        assert "could not read namespace rl" in result.stderr
        assert "kubectl create namespace rl" not in calls_of(fake_tools)

    def test_the_preflight_runs_before_helm_touches_the_cluster(self, fake_tools):
        # [FUNCTION DOC] The index of the first permission check call is less than that of the first
        # helm upgrade call.
        """Checking after the install would report on a cluster the install has already changed."""
        run_cli("install", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        first_can_i = min(index for index, call in enumerate(calls) if "auth can-i" in call)
        first_upgrade = min(index for index, call in enumerate(calls) if call.startswith("helm upgrade"))
        assert first_can_i < first_upgrade

    def test_it_waits_for_the_pod_with_a_bounded_timeout(self, fake_tools):
        # [FUNCTION DOC] With a timeout of 42 seconds, a rollout status call for the stateful set
        # carries a 42 second timeout.
        """An unbounded wait hangs a CI job forever when the image cannot be pulled."""
        run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight", "--timeout", "42")

        assert "kubectl rollout status statefulset/wb-miles-workbench -n rl --timeout=42s" in calls_of(fake_tools)

    def test_the_wait_has_a_timeout_even_when_none_is_asked_for(self, fake_tools):
        # [FUNCTION DOC] With no timeout given, the rollout status call uses 600 seconds.
        """The default matters more than the flag: almost nobody passes --timeout."""
        run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert "kubectl rollout status statefulset/wb-miles-workbench -n rl --timeout=600s" in calls_of(fake_tools)

    def test_a_pod_that_never_becomes_ready_fails_and_shows_why(self, fake_tools):
        # [FUNCTION DOC] When the rollout status call fails: nonzero exit, stderr says the pod was not
        # ready within 600s, and the tool lists pods and describes the stateful set for diagnosis.
        """Exiting zero on a pending pod sends the user to exec into a pod that is not there."""
        fake_tools["failing_path"].write_text("kubectl rollout status")

        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")
        calls = calls_of(fake_tools)

        assert result.returncode != 0
        assert "was not ready within 600s" in result.stderr
        assert "kubectl get pods -n rl" in calls
        assert "kubectl describe statefulset wb-miles-workbench -n rl" in calls

    def test_a_successful_install_prints_the_command_that_gets_a_shell(self, fake_tools):
        # [FUNCTION DOC] On success stderr includes the exec subcommand invocation with namespace and
        # release.
        """The install is only half the workflow; the next step must not need the README."""
        result = run_cli("install", "-n", "rl", "-r", "wb", "--skip-preflight")

        assert result.returncode == 0, result.stdout + result.stderr
        assert f"python -m {PACKAGE} exec -n rl -r wb" in result.stderr


class TestNamespaceOccupancy:
    # [CLASS DOC] Tests of the preflight check that the namespace holds only Miles releases.
    def test_the_check_spans_the_kinds_that_kubectl_get_all_leaves_out(self, fake_tools):
        # [FUNCTION DOC] Installing with the preflight issues a listing of names for config maps,
        # secrets, persistent volume claims and service accounts, each filtered to objects not managed by
        # Helm.
        """The Role covers configmaps, secrets and RBAC too, none of which `kubectl get all` ever returns."""
        run_cli("install", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        for kind in ("configmap", "secret", "persistentvolumeclaim", "serviceaccount"):
            assert f"kubectl get {kind} -n rl -l app.kubernetes.io/managed-by!=Helm -o name" in calls

    def test_a_kind_this_account_may_not_list_is_said_out_loud(self, fake_tools):
        # [FUNCTION DOC] When listing secrets is forbidden, install still succeeds (exit zero) but
        # stderr warns the namespace content could not be checked and names the failed listing.
        """Unreadable is not empty, and a silent pass hands the workbench a namespace nobody has inspected."""
        fake_tools["refused_path"].write_text("Forbidden|kubectl get secret")

        result = run_cli("install", "-n", "rl", "-r", "wb")

        assert result.returncode == 0, result.stdout + result.stderr
        assert "could not check whether namespace rl holds nothing but Miles releases" in result.stderr
        assert "listing secret failed" in result.stderr


class TestUninstall:
    # [CLASS DOC] Test of the uninstall subcommand.
    def test_it_removes_the_release_and_keeps_the_namespace(self, fake_tools):
        # [FUNCTION DOC] Exit zero; a helm uninstall of release wb in namespace rl is called, and no
        # call deletes a namespace.
        """The namespace usually predates the release and may hold a colleague's work."""
        result = run_cli("uninstall", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        assert result.returncode == 0, result.stdout + result.stderr
        assert "helm uninstall wb --namespace rl" in calls
        assert not any("delete namespace" in call for call in calls)


class TestStop:
    # [CLASS DOC] Tests of the stop subcommand, which uninstalls the release derived from a run id.
    def test_it_names_the_release_a_run_id_stands_for(self, fake_tools):
        # [FUNCTION DOC] Stopping run "demo" calls helm uninstall for the derived run release name.
        """A user knows the run id they launched with, not the release name the launcher derived."""
        run_cli("stop", "-n", "rl", "demo")

        assert f"helm uninstall {run_release_name('demo')} --namespace rl" in calls_of(fake_tools)

    def test_it_stops_the_component_it_was_asked_for(self, fake_tools):
        # [FUNCTION DOC] With the trainer deploy component, helm uninstall is called for the
        # trainer-specific release name.
        """A split run installs one release per component, and each is uninstalled on its own."""
        run_cli("stop", "-n", "rl", "demo", "--deploy-component", "trainer")

        assert f"helm uninstall {run_release_name('demo', DeployComponent.TRAINER)} --namespace rl" in calls_of(
            fake_tools
        )

    def test_stopping_one_component_names_no_other_release(self, fake_tools):
        # [FUNCTION DOC] After stopping the trainer, no uninstall call contains the release name of
        # any other deploy component. KNOWN ISSUE: the match is a substring test, so it would also
        # flag a release whose name merely contains another's.
        """Uninstalling a sibling of the release the user asked for would take down a run that is still training."""
        run_cli("stop", "-n", "rl", "demo", "--deploy-component", "trainer")

        uninstalled = [call for call in calls_of(fake_tools) if "helm uninstall" in call]
        others = [
            run_release_name("demo", component)
            for component in DeployComponent
            if component is not DeployComponent.TRAINER
        ]
        assert not [call for call in uninstalled for release in others if release in call]

    def test_two_runs_whose_ids_share_a_prefix_are_stopped_separately(self, fake_tools):
        # [FUNCTION DOC] Stopping run "demo" never mentions the release name of run "demo-2" in any call.
        """helm is given one exact name, so a prefix match here would end a second user's run."""
        run_cli("stop", "-n", "rl", "demo")

        assert run_release_name("demo-2") not in "".join(calls_of(fake_tools))


class TestCollectDiagnosis:
    # [CLASS DOC] Tests of the collect-diagnosis subcommand, which writes a timestamped directory
    # under the chosen output directory and prints its path on standard output.
    def test_it_writes_one_directory_holding_logs_describes_and_events(self, fake_tools, tmp_path):
        # [FUNCTION DOC] With two pods listed: exit zero, the printed directory is the only entry of
        # the output directory, and it holds the events file plus, for each pod, a log, a
        # previous-container log and a describe file.
        """A support request is one directory to attach, not a list of kubectl commands to re-run by hand."""
        fake_tools["pods_path"].write_text("wb-0\norchestrator-0\n")
        output_dir = tmp_path / "out"

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(output_dir))
        written = Path(result.stdout.strip())

        assert result.returncode == 0, result.stdout + result.stderr
        assert written.parent == output_dir
        assert [path.name for path in output_dir.iterdir()] == [written.name]
        assert {path.name for path in written.iterdir()} == {
            "events.txt",
            "wb-0.log",
            "wb-0.previous.log",
            "wb-0.describe.txt",
            "orchestrator-0.log",
            "orchestrator-0.previous.log",
            "orchestrator-0.describe.txt",
        }

    def test_a_container_that_never_restarted_leaves_no_empty_previous_log(self, fake_tools, tmp_path):
        # [FUNCTION DOC] When the previous-container log call fails (first-boot container), exit is
        # still zero, the previous log file is not created and the normal log is.
        """kubectl fails on --previous for a first-boot container, and an empty file reads as a lost log."""
        fake_tools["pods_path"].write_text("wb-0\n")
        fake_tools["failing_path"].write_text("kubectl logs wb-0 -n rl --all-containers --previous")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        written = Path(result.stdout.strip())

        assert result.returncode == 0, result.stdout + result.stderr
        assert (written / "wb-0.previous.log").exists() is False
        assert (written / "wb-0.log").exists()

    def test_a_restarted_container_keeps_the_log_that_explains_the_restart(self, fake_tools, tmp_path):
        # [FUNCTION DOC] When the previous log call succeeds, the previous log file exists.
        """The crash is in the previous container's log, not in the one the running container is writing."""
        fake_tools["pods_path"].write_text("wb-0\n")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))

        assert (Path(result.stdout.strip()) / "wb-0.previous.log").exists()

    def test_the_run_verdict_is_left_out_when_no_run_directory_is_named(self, fake_tools, tmp_path):
        # [FUNCTION DOC] Without a run directory option no orchestrator state file is in the output.
        """Most diagnoses are of the workbench itself, where no orchestrator has ever written a verdict."""
        fake_tools["pods_path"].write_text("wb-0\n")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))

        assert (Path(result.stdout.strip()) / "orchestrator.state").exists() is False

    def test_a_named_run_directory_contributes_the_verdict_of_its_latest_launch(self, fake_tools, tmp_path):
        # [FUNCTION DOC] A run directory with two recorded launches and a state file for each (exit
        # codes 0 and 1): the collected output holds the state file of the newest launch, with its
        # content "1".
        """A relaunched run keeps every launch's exit file, and only the newest is this run's verdict."""
        fake_tools["pods_path"].write_text("wb-0\n")
        state = tmp_path / "run" / "state"
        state.mkdir(parents=True)
        (state / "orchestrator-260101-000000-000001.state").write_text("0\n")
        (state / "orchestrator-260101-010000-000002.state").write_text("1\n")
        _record_launch(tmp_path / "run", "260101-000000-000001")
        _record_launch(tmp_path / "run", "260101-010000-000002")

        result = run_cli(
            "collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path), "--run-dir", str(tmp_path / "run")
        )

        collected = Path(result.stdout.strip()) / "orchestrator-260101-010000-000002.state"
        assert collected.read_text() == "1\n"

    def test_a_generation_that_has_written_no_state_hides_the_previous_verdict(self, fake_tools, tmp_path):
        # [FUNCTION DOC] Two recorded launches but a state file only for the first: exit code 1 and
        # stderr says a verdict under the run directory is missing (the old verdict is not used).
        """A relaunch whose pods never came up would otherwise be diagnosed against the old launch's exit code."""
        fake_tools["pods_path"].write_text("wb-0\n")
        state = tmp_path / "run" / "state"
        state.mkdir(parents=True)
        (state / "orchestrator-260101-000000-000001.state").write_text("0\n")
        _record_launch(tmp_path / "run", "260101-000000-000001")
        _record_launch(tmp_path / "run", "260101-010000-000002")

        result = run_cli(
            "collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path), "--run-dir", str(tmp_path / "run")
        )

        assert result.returncode == 1
        assert "a verdict under" in result.stderr

    def test_a_run_directory_with_no_verdict_yet_is_reported_as_incomplete(self, fake_tools, tmp_path):
        # [FUNCTION DOC] A run directory with an empty state folder and no launches: exit code 1 and
        # the same missing-verdict message.
        """Asking for a run's verdict and silently getting none would read as the run having succeeded."""
        fake_tools["pods_path"].write_text("wb-0\n")
        (tmp_path / "run" / "state").mkdir(parents=True)

        result = run_cli(
            "collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path), "--run-dir", str(tmp_path / "run")
        )

        assert result.returncode == 1
        assert "a verdict under" in result.stderr

    def test_an_unreadable_cluster_still_produces_a_directory(self, fake_tools, tmp_path):
        # [FUNCTION DOC] When the pod listing fails: nonzero exit, stderr mentions the failed pod
        # listing in the namespace, and the output directory holds only the events file.
        # KNOWN ISSUE: the stderr assertion is duplicated verbatim.
        """It is run when things are broken, so a failed pod listing must not cost the events and the verdict."""
        fake_tools["failing_path"].write_text("kubectl get pods")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        written = Path(result.stdout.strip())

        assert result.returncode != 0
        assert "pod listing in namespace rl" in result.stderr
        assert "pod listing in namespace rl" in result.stderr
        assert {path.name for path in written.iterdir()} == {"events.txt"}

    def test_pod_logs_cover_every_container_in_the_pod(self, fake_tools, tmp_path):
        # [FUNCTION DOC] Both the current and previous log calls for the pod carry the
        # all-containers flag.
        """The orchestrator pod runs sidecars, and the default single-container log would silently drop them."""
        fake_tools["pods_path"].write_text("wb-0\n")

        run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        calls = calls_of(fake_tools)

        assert "kubectl logs wb-0 -n rl --all-containers" in calls
        assert "kubectl logs wb-0 -n rl --all-containers --previous" in calls

    def test_a_failed_collection_step_is_reported_and_fails_the_run(self, fake_tools, tmp_path):
        # [FUNCTION DOC] When both the pod log call and the events call fail: nonzero exit, stderr
        # names the failed logs and events, and the directory still holds the events file, the (log
        # for the pod) and the describe file but no previous log.
        """A caller that branches on the exit code must not read a half-collected directory as a whole one."""
        fake_tools["pods_path"].write_text("wb-0\n")
        fake_tools["failing_path"].write_text("kubectl logs wb-0 -n rl --all-containers\nkubectl get events")

        result = run_cli("collect-diagnosis", "-n", "rl", "--output-dir", str(tmp_path))
        written = Path(result.stdout.strip())

        assert result.returncode != 0
        assert "logs of wb-0" in result.stderr
        assert "events" in result.stderr
        assert {path.name for path in written.iterdir()} == {
            "events.txt",
            "wb-0.log",
            "wb-0.describe.txt",
        }


class TestDryRun:
    # [CLASS DOC] Tests of the install dry-run mode.
    def test_it_runs_the_checks_without_installing(self, fake_tools):
        # [FUNCTION DOC] Exit zero; permission checks are called and no helm upgrade is.
        """The same checks install runs, for someone who wants to know before touching the cluster."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")
        calls = fake_tools["calls_path"].read_text()

        assert result.returncode == 0, result.stdout + result.stderr
        assert "kubectl auth can-i" in calls
        assert "helm upgrade" not in calls

    def test_it_fails_when_a_check_fails(self, fake_tools):
        # [FUNCTION DOC] With all kubectl calls failing, a dry run exits with code 1.
        """Its exit code is what a wrapper script would branch on."""
        (fake_tools["failing_path"]).write_text("kubectl")

        assert run_cli("install", "--dry-run", "-n", "rl", "-r", "wb").returncode == 1

    def test_it_changes_nothing_in_the_cluster(self, fake_tools):
        # [FUNCTION DOC] Exit zero; no namespace creation, helm upgrade or rollout status calls.
        """A dry run that created the namespace or rolled out the chart would defeat the point of asking first."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")
        calls = calls_of(fake_tools)

        assert result.returncode == 0, result.stdout + result.stderr
        assert not any(call.startswith("kubectl create namespace") for call in calls)
        assert not any(call.startswith("helm upgrade") for call in calls)
        assert not any("rollout status" in call for call in calls)

    def test_it_changes_nothing_on_the_local_disk_either(self, fake_tools):
        # [FUNCTION DOC] Exit zero; at least one dependency build call occurs (to render the chart)
        # but none targets the real chart directory (a temporary copy is used).
        """Vendoring dependencies into the checkout is a write, and a dry run promises not to write."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")
        builds = [call for call in calls_of(fake_tools) if call.startswith("helm dependency build")]

        assert result.returncode == 0, result.stdout + result.stderr
        assert builds, "the dry run still has to render the chart to check what it grants"
        assert not any(call.endswith(str(CHART_DIR)) for call in builds)

    def test_it_still_renders_the_chart_it_checks(self, fake_tools):
        # [FUNCTION DOC] Exit zero; a helm template call for release wb occurs.
        """A dry run that skipped the render would stop reporting the rbac the chart asks for."""
        result = run_cli("install", "--dry-run", "-n", "rl", "-r", "wb")

        assert result.returncode == 0, result.stdout + result.stderr
        assert any(call.startswith("helm template wb ") for call in calls_of(fake_tools))


class TestExec:
    # [CLASS DOC] Tests of the exec subcommand.
    def test_it_execs_into_the_statefulset_of_that_release(self, fake_tools):
        # [FUNCTION DOC] Exec for a release already named after the chart calls an interactive
        # kubectl exec into the same-named stateful set running bash.
        """The pod name is derived from the release, which is the only thing the user knows."""
        run_cli("exec", "-n", "rl", "-r", "miles-workbench-alice")

        assert "kubectl exec -it statefulset/miles-workbench-alice -n rl -- bash" in (
            fake_tools["calls_path"].read_text()
        )

    def test_the_kubectl_separator_is_not_passed_twice(self, fake_tools):
        # [FUNCTION DOC] A user-typed double dash before the command yields a single separator in the
        # kubectl call.
        """Typing `--` before the command is kubectl muscle memory; a second one becomes the executable."""
        run_cli("exec", "-n", "rl", "-r", "wb", "--", "echo", "hi")

        assert "kubectl exec -it statefulset/wb-miles-workbench -n rl -- echo hi" in (
            fake_tools["calls_path"].read_text()
        )

    def test_it_reaches_the_default_release_when_none_is_named(self, fake_tools):
        # [FUNCTION DOC] With no release option, the target is the object name of the default release.
        """Most namespaces hold one workbench, and naming it in every command is noise."""
        run_cli("exec", "-n", "rl")

        assert f"kubectl exec -it statefulset/{object_name(DEFAULT_RELEASE)} -n rl -- bash" in (
            fake_tools["calls_path"].read_text()
        )

    def test_a_command_can_be_given_instead_of_a_shell(self, fake_tools):
        # [FUNCTION DOC] A trailing command and arguments replace bash in the kubectl exec call.
        """Scripting against the workbench should not need a second tool."""
        run_cli("exec", "-n", "rl", "-r", "wb", "python", "-c", "print(1)")

        assert "kubectl exec -it statefulset/wb-miles-workbench -n rl -- python -c print(1)" in (
            fake_tools["calls_path"].read_text()
        )


class TestMissingBinaries:
    # [CLASS DOC] Test of behavior when helm or kubectl is not installed.
    @pytest.mark.parametrize(
        "args", [["exec", "-n", "rl", "-r", "wb"], ["install", "-n", "rl", "-r", "wb", "--skip-preflight"]]
    )
    def test_a_missing_binary_is_reported_not_raised(self, tmp_path, monkeypatch, args):
        # [FUNCTION DOC] Parametrized over exec and install. With the path set to an empty temp
        # directory: exit code 1, stderr mentions that the client is installed (message about
        # installing it) and contains no traceback.
        """It runs before any Miles environment exists, so an absent client is the expected first failure."""
        # an empty PATH, not one that merely omits a directory: a runner that ships kubectl in
        # /usr/bin would run it for real and fail on the cluster instead of on the missing client
        monkeypatch.setenv("PATH", str(tmp_path))
        result = run_cli(*args)

        assert result.returncode == 1
        assert "is installed" in result.stderr
        assert "Traceback" not in result.stderr


class TestCli:
    # [CLASS DOC] Tests of general command line behavior and naming.
    @pytest.mark.parametrize(
        "args",
        [
            [],
            ["install"],
            ["install", "--dry-run", "-r", "wb"],
            ["bogus"],
        ],
    )
    def test_incomplete_invocations_are_usage_errors(self, fake_tools, args):
        # [FUNCTION DOC] Parametrized over no arguments, install without a namespace, dry-run
        # without a namespace, and an unknown subcommand: exit code 2 and neither fake tool was
        # invoked (the calls log does not exist).
        """Half a command must not reach helm or kubectl."""
        result = run_cli(*args)

        assert result.returncode == 2
        assert not fake_tools["calls_path"].exists()

    def test_the_object_name_leaves_room_for_the_names_kubernetes_derives_from_it(self):
        # [FUNCTION DOC] For a 53-character release, the object name plus "-0" is at most 63
        # characters, and the object name plus an 11-character revision suffix is at most 63.
        """Pod hostnames and the controller-revision-hash label are 63-char names built from this one."""
        name = object_name("a" * 53)

        assert len(f"{name}-0") <= 63
        assert len(name) + len("-0123456789") <= 63

    def test_the_object_name_carries_the_chart_name_only_once(self):
        # [FUNCTION DOC] A release already starting with the chart name is used as is; others get
        # the chart name appended with a hyphen.
        """A release a user already named after the chart must not become miles-workbench-miles-workbench."""
        assert object_name("miles-workbench-alice") == "miles-workbench-alice"
        assert object_name("alice") == "alice-miles-workbench"

    def test_it_pulls_in_no_training_dependencies(self):
        # [FUNCTION DOC] Runs a fresh interpreter (sixty second timeout, repository root as working
        # directory) that imports the tool's main module and prints all loaded module names; exit
        # code zero and the output contains none of torch, ray, megatron or transformers as a whole
        # word. (Only exact top-level module names are compared.)
        """It installs the pod that holds the training environment, so it cannot need that environment itself."""
        probe = f"import importlib, sys; importlib.import_module('{PACKAGE}.__main__'); print(*sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60
        )

        assert result.returncode == 0, result.stderr
        assert {"torch", "ray", "megatron", "transformers"}.isdisjoint(result.stdout.split())

    def test_it_offers_exactly_these_subcommands(self):
        # [FUNCTION DOC] The registered command names of the typer application are exactly install,
        # exec, stop, uninstall and collect-diagnosis.
        """The subcommand set is the cli's whole contract with a runbook, so a rename must be deliberate."""
        assert {command.name for command in workbench_main.app.registered_commands} == {
            "install",
            "exec",
            "stop",
            "uninstall",
            "collect-diagnosis",
        }
