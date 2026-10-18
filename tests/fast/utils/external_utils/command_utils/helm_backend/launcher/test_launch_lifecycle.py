# [MODULE DOC] Unit tests for the lifecycle of a Helm launcher launch (execute-train) against
# fakes. The fakes: the command wrapper's process runner records every kubectl-style command (and
# can make it fail), Helm dependency build / manifest read / upgrade-render / upgrade are replaced
# so that "installed" and "proposed" manifests are literal text and upgrades are just recorded,
# launch inputs are stubbed (and the follow step records its keyword arguments), and a sandbox
# directory serves as cluster storage and runs root. Covered: per-launch state files and run
# outcome computation from the state file, background log streaming alongside the watcher, the
# chart owning the auto-uninstall decision, trainer-only deployments (own release, no
# orchestrator, no verdict wait), defusing a pending uninstall job (when it is deleted or left
# alone, refusal when the upgrade would change more than size), choosing the state file to watch
# (carried from the installed orchestrator or a fresh one), reading the active state file from the
# live stateful set, hot restart of a live run, ordinary relaunch after a hot restart, and keeping
# the launch record that the already installed pods carry while still writing a record for the new
# launch.
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import yaml
from tests.fast.charts.utils import REPO_ROOT

from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    RESTART_AT_ANNOTATION,
    Manifest,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import cluster_info, log_follower
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import MooncakeInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import (
    ORCHESTRATOR_COMPONENT,
    ReleaseName,
    RunFiles,
    _orchestrator_state_path,
)
from miles.utils.external_utils.command_utils.helm_backend.orchestrator import observer
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.observer import ObservedPod
from miles.utils.external_utils.command_utils.helm_backend.orchestrator.state import (
    STATE_FILE_FLAG,
    OrchestratorState,
    OrchestratorStatus,
)
from miles.utils.workers.k8s_types import ContainerState, ContainerStatus, Pod, PodMetadata, PodStatus
from miles.utils.workers.types import DeployComponent
from miles.utils.workers.worker_provider.kubernetes.helm.naming import component_name


def _write(path: Path, status: OrchestratorStatus, exit_code: int | None = None) -> None:
    # [FUNCTION DOC] HELPER: Writes an orchestrator state file at the path with the given status
    # and optional exit code (default none).
    OrchestratorState(status=status, exit_code=exit_code).write(path)


def _stub_launch_inputs(monkeypatch, *, specs, colocate: bool = False, deploy_component: str = "all") -> list[dict]:
    # [FUNCTION DOC] HELPER: Replaces on the entrypoint module the spec computation (returns the
    # given specs), the argument parser (a namespace with the given colocate flag and deploy
    # component text, no deploy instance id, empty argv, empty train environment, no wandb), the
    # Mooncake plan of arguments (none) and the follow step, which appends its keyword arguments
    # to a list instead of waiting. Returns that list (the recorded follow calls).
    followed: list[dict] = []
    monkeypatch.setattr(entrypoint, "compute_specs", lambda args: specs)
    monkeypatch.setattr(
        entrypoint,
        "parse_args",
        lambda: SimpleNamespace(
            colocate=colocate,
            deploy_component=deploy_component,
            deploy_instance_id=None,
            argv=[],
            train_env_vars={},
            use_wandb=False,
            wandb_run_id=None,
        ),
    )
    monkeypatch.setattr(MooncakeInfo, "plan_of_args", staticmethod(lambda args: None))
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: followed.append(kwargs))
    return followed


class TestPerLaunchExitFile:
    # [CLASS DOC] Tests that each launch has its own orchestrator state file and that a run
    # outcome is computed only from the file of the launch being watched.
    def test_every_launch_token_names_a_file_of_its_own(self):
        # [FUNCTION DOC] Two new state file requests for the same run directory give different
        # paths that share the parent folder "state" inside the run directory.
        """A relaunch that reused the path could read the last words of the wrapper it just replaced."""
        first = RunFiles.new_state_file(run_directory="/runs/abc")
        second = RunFiles.new_state_file(run_directory="/runs/abc")

        assert first != second
        assert first.parent == second.parent == Path("/runs/abc/state")

    def test_the_verdict_of_another_launch_is_never_read(self, tmp_path):
        # [FUNCTION DOC] A state file for token "b" holds an exited verdict with code 7; the
        # state file for token "a" does not exist. Computing the run outcome from the "a" state
        # with a running pod and zero polls of each failure kind gives none (no outcome).
        """The wrapper of the run being replaced writes its own file, which this launcher must not consult."""
        mine = _orchestrator_state_path(tmp_path, "a")
        theirs = _orchestrator_state_path(tmp_path, "b")
        _write(theirs, OrchestratorStatus.EXITED, exit_code=7)

        assert (
            observer._compute_run_outcome(
                state=OrchestratorState.read(mine),
                observed=ObservedPod(phase="Running"),
                missing_polls=0,
                dead_polls=0,
                failing_polls=0,
            )
            is None
        )

    def test_a_verdict_written_before_the_launcher_looked_is_returned_at_once(self, tmp_path):
        # [FUNCTION DOC] A state file already holding an exited verdict with code 4 gives an
        # outcome with exit code 4 on the first computation, even though the pod is running.
        """Relaunching a finished run attaches to it, and its exit code is already sitting in the file."""
        state_file = _orchestrator_state_path(tmp_path, "a")
        _write(state_file, OrchestratorStatus.EXITED, exit_code=4)

        outcome = observer._compute_run_outcome(
            state=OrchestratorState.read(state_file),
            observed=ObservedPod(phase="Running"),
            missing_polls=0,
            dead_polls=0,
            failing_polls=0,
        )

        assert outcome.exit_code == 4


class TestBackgroundLogOrdering:
    # [CLASS DOC] Test that log streaming happens in the background while the watcher polls for
    # the verdict.
    def test_the_watcher_runs_while_the_logs_stream(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] A state file already holds an exited verdict with code 3. The process
        # start function is replaced by one recording the program name and returning a fake
        # process (empty output, terminate/kill set a flag, poll reports running until then),
        # pod lookups return one running pod, events are empty, the observed orchestrator pod is
        # running, and the active state file is the prepared one. The real follow-until-finished
        # runs and must raise a run-exited error with exit code 3, and every started process was
        # a kubectl command (at least one). Because a log follow never ends on a keep-alive pod,
        # the watcher must not wait for it.
        """kubectl logs --follow never returns on a keep-alive pod, so a serial launcher would never poll."""
        state_file = tmp_path / "orchestrator.state"
        _write(state_file, OrchestratorStatus.EXITED, exit_code=3)
        started: list[str] = []

        class FakeProcess:
            # [CLASS DOC] HELPER: Fake log subprocess with no output; it reports running until
            # terminated or killed.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the terminated flag (false) and an empty
                # output iterator.
                self.terminated = False
                self.stdout = iter(())

            def wait(self, timeout: float | None = None) -> int:
                # [FUNCTION DOC] HELPER: Returns 0 immediately.
                return 0

            def poll(self) -> int | None:
                # [FUNCTION DOC] HELPER: Returns 0 once terminated, none before.
                return 0 if self.terminated else None

            def terminate(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the terminated flag.
                self.terminated = True

            def kill(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the terminated flag.
                self.terminated = True

        process = FakeProcess()

        def fake_popen(command, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Fake process start: records the program name (first word)
            # and returns the single shared fake process.
            started.append(command[0])
            return process

        pods = [_pod()]
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr(cluster_info, "selected_pods", lambda namespace, selector: pods)
        monkeypatch.setattr(log_follower, "selected_pods", lambda namespace, selector: pods)
        monkeypatch.setattr(cluster_info, "pod_events", lambda *, namespace, pods: [])
        monkeypatch.setattr(entrypoint, "observed_pod", lambda namespace, workload: ObservedPod(phase="Running"))
        monkeypatch.setattr(entrypoint, "_active_state_file", lambda *, release, namespace: state_file)
        with pytest.raises(entrypoint.RunExitedError) as raised:
            entrypoint._follow_until_finished(release="myrun", namespace="myns", state_file=state_file)

        assert raised.value.exit_code == 3
        assert started and set(started) == {"kubectl"}


def _pod() -> Pod:
    # [FUNCTION DOC] HELPER: Builds a running pod named "p" (uid "p-uid") with one container
    # "orchestrator" with container id "docker://a", no restarts and a running state.
    container = ContainerStatus(
        name="orchestrator", container_id="docker://a", restart_count=0, state=ContainerState(running={})
    )
    return Pod(
        metadata=PodMetadata(name="p", uid="p-uid"),
        status=PodStatus(phase="Running", container_statuses=[container]),
    )


class _Recorded(NamedTuple):
    # [CLASS DOC] HELPER: Mutable collections filled by the launch fakes: the list of recorded
    # process runner commands (each a list of strings) and the list of release names for which an
    # upgrade was performed.
    kubectl: list[list[str]]
    upgraded: list[str]


class TestAutoUninstallValues:
    # [CLASS DOC] Test of who decides the self-uninstall setting.
    def test_the_launcher_leaves_the_self_uninstall_decision_to_the_chart(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] After a first install of an ordinary run, the written values' run
        # section has no auto-uninstall key.
        """The chart default arms it and a user values file may override it, so the launcher writes nothing."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=False)

        assert "autoUninstall" not in _written_values(tmp_path)["run"]


class TestADeploymentWithoutTheOrchestrationScript:
    # [CLASS DOC] Tests of a launch of only the trainer deployment of a split run.
    def test_installs_a_release_of_its_own_so_the_parts_of_a_run_never_replace_each_other(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] A trainer-component first install upgrades exactly one release, the
        # trainer release name (not the unsplit one).
        """Both launches carry the same run id, and one release name would make the second uninstall the first."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=False, deploy_component="trainer")

        assert recorded.upgraded == [_TRAINER_RELEASE]

    def test_installs_no_orchestrator_and_waits_for_no_verdict(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] For the same launch, the follow step is never called, the written
        # orchestrator values have an empty command list, there is no state file entry, and
        # auto-uninstall is explicitly disabled.
        """It carries no training to finish, so a launcher waiting for an exit file would hang forever."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        followed = _launch(monkeypatch, tmp_path, recorded, installed=False, deploy_component="trainer")

        built = _written_values(tmp_path)["run"]
        assert followed == []
        assert built["orchestrator"] == {"command": []}
        assert "stateFile" not in built
        assert built["autoUninstall"] == {"enabled": False}


class TestDefusingAPendingUninstall:
    # [CLASS DOC] Tests of when the launcher deletes the release's pending uninstall job (which
    # would otherwise uninstall the release being installed) and when it must not.
    def test_a_first_install_deletes_a_job_an_earlier_run_of_this_id_left_behind(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] A first install records the exact job delete command (the module
        # constant) and upgrades the unsplit release.
        """A relaunch of a run id whose uninstall is still sleeping would be uninstalled by that job."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=False)

        assert _DELETE_UNINSTALL_JOB in recorded.kubectl
        assert recorded.upgraded == [_RELEASE]

    def test_a_relaunch_that_would_change_more_than_the_size_installs_nothing(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] An installed release whose proposed manifest has an extra key makes the
        # launch exit with a system exit whose message contains "more than its size"; no upgrade
        # and no process command ran.
        """A refused launch must leave the running one alone, including the job that would uninstall it."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        with pytest.raises(SystemExit, match="more than its size"):
            _launch(monkeypatch, tmp_path, recorded, installed=True, proposed_differs=True)

        assert recorded.upgraded == []
        assert recorded.kubectl == []

    def test_attaching_to_the_orchestrator_already_installed_leaves_the_job_alone(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] A relaunch with an unchanged installed manifest upgrades the release
        # but runs no process command (the job is not deleted).
        """No wrapper restarts here, so a deleted job is never recreated and the release would leak forever."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=True)

        assert recorded.kubectl == []
        assert recorded.upgraded == [_RELEASE]

    def test_an_orchestrator_that_really_changes_deletes_the_job_because_its_new_wrapper_re_arms_it(
        self, monkeypatch, tmp_path
    ):
        # [FUNCTION DOC] When the proposed orchestrator manifest differs (its command gets an
        # extra option) and the upgrade check is skipped, the job delete command is recorded.
        """The orchestrator pod really rolls here, so its new wrapper recreates the job this delete removes."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(
            monkeypatch,
            tmp_path,
            recorded,
            installed=True,
            rendered=_RENDERED_ORCHESTRATOR,
            proposed=_RENDERED_ORCHESTRATOR_REPLACED,
            skip_upgrade_check=True,
        )

        assert _DELETE_UNINSTALL_JOB in recorded.kubectl

    def test_a_delete_the_cluster_refused_stops_the_launch(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] When every process command fails (exit 1), the launch raises a
        # called-process error and no upgrade ran.
        """Installing over a job that is still armed hands the new run's release to the old run's uninstall."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        with pytest.raises(subprocess.CalledProcessError):
            _launch(monkeypatch, tmp_path, recorded, installed=False, delete_fails=True)

        assert recorded.upgraded == []


_RUN_ID = "260101-000000-000"
_RELEASE = ReleaseName(run_id=_RUN_ID, deploy_component=DeployComponent.ALL, deploy_instance_id=None).serialize()
_TRAINER_RELEASE = ReleaseName(
    run_id=_RUN_ID, deploy_component=DeployComponent.TRAINER, deploy_instance_id=None
).serialize()
_DELETE_UNINSTALL_JOB = [
    "kubectl",
    "delete",
    "job",
    f"{_RELEASE}-uninstall",
    "--namespace",
    "rl",
    "--ignore-not-found",
    "--cascade",
    "foreground",
]


class TestChoosingTheStateFile:
    # [CLASS DOC] Tests of which orchestrator state file the launcher watches after a
    # relaunch.
    def test_a_relaunch_that_leaves_the_orchestrator_alone_keeps_watching_its_state_file(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] With an installed orchestrator manifest and an identical proposal, the
        # first recorded follow call has the state file "attached.state" in the temporary
        # directory (the state file named in the installed manifest, via the patched lookup).
        """The wrapper it attaches to writes the file it already names, so a new one would never be written to."""
        followed = _launch(
            monkeypatch, tmp_path, _Recorded(kubectl=[], upgraded=[]), installed=True, rendered=_RENDERED_ORCHESTRATOR
        )

        assert followed[0]["state_file"] == tmp_path / "attached.state"

    def test_an_orchestrator_that_really_changes_gets_a_state_file_of_its_own(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] When the orchestrator manifest really changes (upgrade check skipped),
        # the followed state file is not the attached one and lies in a folder named "state".
        """The replaced wrapper's file already holds a verdict, and reading it would end the relaunch at once."""
        followed = _launch(
            monkeypatch,
            tmp_path,
            _Recorded(kubectl=[], upgraded=[]),
            installed=True,
            rendered=_RENDERED_ORCHESTRATOR,
            proposed=_RENDERED_ORCHESTRATOR_REPLACED,
            skip_upgrade_check=True,
        )

        assert followed[0]["state_file"] != tmp_path / "attached.state"
        assert followed[0]["state_file"].parent.name == "state"


class TestTheCarriedStateFile:
    # [CLASS DOC] Tests of the function that finds the state file named by the orchestrator of
    # an installed release manifest.
    def test_a_release_that_is_not_installed_carries_no_state_file(self):
        # [FUNCTION DOC] With no installed manifest the result is none.
        """A first install carries nothing over, and a path nobody has written to is not a carried file."""
        assert entrypoint._carried_state_file(installed_manifest=None, release=_RELEASE) is None

    def test_an_installed_release_carries_the_state_file_its_orchestrator_names(self):
        # [FUNCTION DOC] For a manifest whose orchestrator command passes the state file flag
        # with "/shared/attached.state", the result is that path.
        """A relaunch that leaves the orchestrator alone keeps watching the file that wrapper writes its verdict to."""
        manifest = Manifest.parse(
            _orchestrator_manifest(state_file="/shared/attached.state", restart_at=None), namespace="rl"
        )

        carried = entrypoint._carried_state_file(installed_manifest=manifest, release=_RELEASE)

        assert carried == Path("/shared/attached.state")

    def test_an_installed_orchestrator_without_a_state_file_is_refused(self) -> None:
        # [FUNCTION DOC] A manifest whose orchestrator command has no state file flag raises an
        # assertion error whose message contains "uninstall it, or launch under a new run id".
        """A relaunch cannot recover a verdict when the installed wrapper names no state file."""
        manifest = Manifest.parse(_RENDERED_ORCHESTRATOR, namespace="rl")

        with pytest.raises(AssertionError, match="uninstall it, or launch under a new run id"):
            entrypoint._carried_state_file(installed_manifest=manifest, release=_RELEASE)


class TestTheActiveStateFile:
    # [CLASS DOC] Tests of the function that reads the state file of the currently active
    # orchestrator generation from the live stateful set through kubectl.
    def test_reads_the_generation_from_the_live_statefulset(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] With the raw kubectl runner returning an orchestrator manifest that
        # names "replacement.state" (and has a restart stamp), the active state file for the
        # release in namespace "rl" equals that path.
        """The live template changes before Kubernetes terminates the orchestrator it replaces."""
        state_file = tmp_path / "replacement.state"
        manifest = _orchestrator_manifest(state_file=str(state_file), restart_at=_STAMP)
        monkeypatch.setattr(
            Kubectl,
            "run_raw",
            staticmethod(lambda *arguments: subprocess.CompletedProcess(arguments, 0, stdout=manifest, stderr="")),
        )

        assert entrypoint._active_state_file(release=_RELEASE, namespace="rl") == state_file

    def test_a_deleted_release_has_no_active_generation(self, monkeypatch):
        # [FUNCTION DOC] A failing raw runner whose error says "NotFound" gives none.
        """A finished auto-uninstalled release must still leave its state file as the final verdict."""
        monkeypatch.setattr(
            Kubectl,
            "run_raw",
            staticmethod(lambda *arguments: subprocess.CompletedProcess(arguments, 1, stdout="", stderr="NotFound")),
        )

        assert entrypoint._active_state_file(release=_RELEASE, namespace="rl") is None

    def test_a_failure_to_read_the_live_orchestrator_is_not_reported_as_deletion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] A failing raw runner with a Forbidden error raises a runtime error whose
        # message starts with "Cannot read the active orchestrator generation" and contains the
        # word Forbidden.
        """A forbidden API response is an observation failure, not evidence that the release was deleted."""
        monkeypatch.setattr(
            Kubectl,
            "run_raw",
            staticmethod(
                lambda *arguments: subprocess.CompletedProcess(
                    arguments,
                    1,
                    stdout="",
                    stderr="Error from server (Forbidden): statefulsets is forbidden",
                )
            ),
        )

        with pytest.raises(RuntimeError, match="Cannot read the active orchestrator generation.*Forbidden"):
            entrypoint._active_state_file(release=_RELEASE, namespace="rl")


class TestAHotRestartOfARunThatIsUp:
    # [CLASS DOC] Tests of a launch requesting a hot restart of both orchestration components
    # of an installed, unstamped run.
    @staticmethod
    def _hot_restart(monkeypatch, tmp_path, recorded: _Recorded, proposals: list[dict]) -> list[dict]:
        # [FUNCTION DOC] HELPER: Runs a launch of an installed release whose rendered manifest is
        # an orchestrator stateful set naming "attached.state" without a restart stamp, with the
        # proposed manifest generated from the values files actually written (appending the run
        # values to the proposals list), and the hot restart option for the orchestration and
        # rollout executor components. Returns the recorded follow calls.
        return _launch(
            monkeypatch,
            tmp_path,
            recorded,
            installed=True,
            rendered=_orchestrator_manifest(state_file=str(tmp_path / "attached.state"), restart_at=None),
            proposals=proposals,
            hot_restart="orchestration,rollout_executor",
        )

    def test_the_orchestrator_rolls_without_force_and_gets_a_state_file_of_its_own(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] After the hot restart launch: the release was upgraded once, the job
        # delete command was issued, the followed state file differs from the attached one, and
        # the last proposed values carry a non-empty restart-at value for the orchestrator. No
        # skip-upgrade-check option is used.
        """This is the whole flag: the pod really changes, the gate lets it through, and its verdict file is new."""
        recorded = _Recorded(kubectl=[], upgraded=[])
        proposals: list[dict] = []

        followed = self._hot_restart(monkeypatch, tmp_path, recorded, proposals)

        assert recorded.upgraded == [_RELEASE]
        assert _DELETE_UNINSTALL_JOB in recorded.kubectl
        assert followed[0]["state_file"] != tmp_path / "attached.state"
        assert proposals[-1]["orchestrator"]["restartAt"]

    def test_the_manifest_the_gate_judged_is_the_one_this_launch_installs(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] The last proposed run values (the ones the gate judged) equal the run
        # section of the values file that was finally written, and their state file entry equals
        # the followed state file as text.
        """The launch renders twice, and a gate judging the first render would approve values nobody installs."""
        recorded = _Recorded(kubectl=[], upgraded=[])
        proposals: list[dict] = []

        followed = self._hot_restart(monkeypatch, tmp_path, recorded, proposals)

        assert proposals[-1] == _written_values(tmp_path)["run"]
        assert proposals[-1]["stateFile"] == str(followed[0]["state_file"])


class TestAnOrdinaryRelaunchOfAHotRestartedRun:
    # [CLASS DOC] Tests of an ordinary relaunch (no hot restart option) of a run that was
    # hot restarted before.
    def test_it_changes_nothing_and_keeps_both_the_stamp_and_the_state_file(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] With an installed stamped orchestrator manifest, the relaunch upgrades
        # the release, runs no process command, follows the attached state file, and its proposed
        # values carry the same restart-at stamp.
        """Dropping the stamp it finds would make every later relaunch a diff the gate refuses forever."""
        recorded = _Recorded(kubectl=[], upgraded=[])
        proposals: list[dict] = []

        followed = _launch(
            monkeypatch,
            tmp_path,
            recorded,
            installed=True,
            rendered=_orchestrator_manifest(state_file=str(tmp_path / "attached.state"), restart_at=_STAMP),
            proposals=proposals,
        )

        assert recorded.upgraded == [_RELEASE]
        assert recorded.kubectl == []
        assert followed[0]["state_file"] == tmp_path / "attached.state"
        assert proposals[-1]["orchestrator"]["restartAt"] == _STAMP

    def test_a_relaunch_that_changes_the_orchestration_arguments_is_refused_without_skip_upgrade_check(
        self, monkeypatch, tmp_path
    ):
        # [FUNCTION DOC] A proposal that changes the orchestrator command, without the
        # skip-upgrade-check option, ends with a system exit whose message contains "more than
        # its size"; nothing was upgraded and no process command ran.
        """Changing the arguments of a live run is what --hot-restart is for, and nothing else may do it."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        with pytest.raises(SystemExit, match="more than its size"):
            _launch(
                monkeypatch,
                tmp_path,
                recorded,
                installed=True,
                rendered=_RENDERED_ORCHESTRATOR,
                proposed=_RENDERED_ORCHESTRATOR_REPLACED,
            )

        assert recorded.upgraded == []
        assert recorded.kubectl == []


class TestTheRelaunchKeepsThePodsItAttachesTo:
    # [CLASS DOC] Tests of the launch record path that pods are pointed at: an installed pod
    # template keeps the record of the launch that installed it (a different path would change
    # the pod template and restart every worker), while this invocation is always recorded on
    # disk too.
    def test_the_pods_keep_the_record_of_the_launch_that_installed_them(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] An installed orchestrator stateful set carries the launch record
        # environment variable with a fixed path; the written values' launch record equals that
        # path.
        """A new record in the pod template is a new pod template, so a resize would restart every worker."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=True, rendered=_RENDERED_WITH_RECORD)

        assert _written_values(tmp_path)["run"]["launchRecord"] == _INSTALLED_RECORD

    def test_a_release_without_an_orchestrator_keeps_the_record_its_own_workers_carry(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] For a trainer-component launch whose installed manifest has only a
        # worker stateful set carrying the record variable, the written launch record equals
        # that worker's path.
        """A trainer release has no orchestrator container, and a lost record makes a resize restart every pod."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(
            monkeypatch,
            tmp_path,
            recorded,
            installed=True,
            rendered=_RENDERED_TRAINER_WITH_RECORD,
            deploy_component="trainer",
        )

        assert _written_values(tmp_path)["run"]["launchRecord"] == _INSTALLED_RECORD

    def test_a_first_install_points_the_pods_at_the_record_of_this_launch(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] After a first install, the written launch record equals the path of
        # the first file in the run's "launches" folder.
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=False)

        written = sorted((tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / _RUN_ID / "launches").glob("*"))
        assert _written_values(tmp_path)["run"]["launchRecord"] == str(written[0])

    def test_this_launch_is_still_recorded_on_disk_when_the_pods_keep_an_older_one(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] When the pods keep an installed record, the run's "launches" folder
        # still gets exactly one new record file, whose JSON content has the run id.
        """The pods keeping their own record must not cost the run the record of this invocation."""
        recorded = _Recorded(kubectl=[], upgraded=[])

        _launch(monkeypatch, tmp_path, recorded, installed=True, rendered=_RENDERED_WITH_RECORD)

        written = sorted((tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / _RUN_ID / "launches").glob("*"))
        assert len(written) == 1
        assert json.loads(written[0].read_text())["run_id"] == _RUN_ID


def _launch(
    monkeypatch,
    tmp_path,
    recorded: _Recorded,
    *,
    installed: bool,
    proposed_differs: bool = False,
    proposed: str | None = None,
    proposals: list[dict] | None = None,
    delete_fails: bool = False,
    skip_upgrade_check: bool = False,
    hot_restart: str = "",
    rendered: str | None = None,
    deploy_component: str = "all",
) -> list[dict]:
    # [FUNCTION DOC] HELPER: Runs a whole execute-train launch with all external effects faked,
    # and returns the list of recorded follow calls (keyword arguments of the follow step).
    # Fakes: the process runner records every command (as strings) into the recorded kubectl list
    # and returns exit 0 (or exit 1 for every command when delete_fails is set, raising a
    # called-process error when the caller requested checking); the Helm dependency build does
    # nothing; the installed manifest lookup parses the rendered text when installed is true, else
    # reports none; the proposed manifest comes from the Helm upgrade render, which returns the
    # parsed proposed text (default: the rendered text itself, or the rendered text with an extra
    # data key when proposed_differs is set) or, when a proposals list is given, is built from
    # the last written values file (the run section is appended to that list) as an orchestrator
    # manifest naming the values' state file and restart-at; the Helm upgrade just appends the
    # release name to the recorded upgraded list; the manifest's state file lookup always returns
    # "attached.state" in the temporary directory; the repository base directory is the repo root;
    # trainer controller addresses are a fixed reachable address. Launch inputs are stubbed
    # (no specs, given deploy component), and the config carries the namespace "rl", the run id,
    # the sandbox infra file, deploy component (with a run uuid when the component is split),
    # skip-upgrade-check and hot restart options.
    def fake_run_process(command, **kwargs):
        # [FUNCTION DOC] HELPER: Records the command (strings) and returns a result with exit
        # code 1 when delete_fails is set, else 0, and the error text "the api server refused";
        # raises a called-process error with that text when check was requested and the code is
        # non-zero.
        arguments = [str(part) for part in command]
        recorded.kubectl.append(arguments)
        result = subprocess.CompletedProcess(
            args=command, returncode=1 if delete_fails else 0, stdout="", stderr="the api server refused"
        )
        if kwargs.get("check") and result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, command, stderr=result.stderr)
        return result

    monkeypatch.setattr(command_wrapper, "run_process", fake_run_process)
    monkeypatch.setattr(Helm, "build_dependencies", staticmethod(lambda chart: None))
    rendered = rendered if rendered is not None else _RENDERED
    monkeypatch.setattr(
        Helm,
        "get_manifest",
        staticmethod(lambda release, namespace: Manifest.parse(rendered, namespace=namespace) if installed else None),
    )
    proposed = proposed if proposed is not None else (_RENDERED_WITH_ANOTHER_KEY if proposed_differs else rendered)
    if proposals is None:
        monkeypatch.setattr(
            Helm,
            "render_upgrade",
            staticmethod(lambda **kwargs: Manifest.parse(proposed, namespace=kwargs["namespace"])),
        )
    else:
        monkeypatch.setattr(Helm, "render_upgrade", staticmethod(_render_from_values(proposals)))
    monkeypatch.setattr(Helm, "upgrade", staticmethod(lambda **kwargs: recorded.upgraded.append(kwargs["release"])))
    monkeypatch.setattr(Manifest, "state_file", lambda self, stateful_set, container: tmp_path / "attached.state")
    monkeypatch.setattr(entrypoint, "repo_base_dir", str(REPO_ROOT))
    monkeypatch.setattr(
        entrypoint, "_compute_trainer_controller_addrs", lambda args, *, release, namespace: _REACHABLE_AT
    )

    followed = _stub_launch_inputs(monkeypatch, specs=[], deploy_component=deploy_component)

    component = DeployComponent(deploy_component)
    entrypoint.execute_train(
        request=_train_request(),
        config=ExecuteTrainConfig(
            namespace="rl",
            run_id=_RUN_ID,
            helm_values=(str(_launchable_infra_file(tmp_path)),),
            deploy_component=component,
            run_uuid=_RUN_UUID if component.is_split() else None,
            skip_upgrade_check=skip_upgrade_check,
            hot_restart=hot_restart,
        ),
    )
    return followed


def _render_from_values(proposals: list[dict]):
    # [FUNCTION DOC] HELPER: Returns a fake upgrade-render function. When called with the
    # release, namespace, chart and list of values files, it reads the last values file, takes its
    # run section, appends that section to the proposals list, and returns a parsed orchestrator
    # manifest whose state file and restart-at come from that run section (state file entry and
    # orchestrator restart-at entry, each possibly absent).
    def render_upgrade(*, release: str, namespace: str, chart, values_files) -> Manifest:
        # [FUNCTION DOC] HELPER: The fake render function described in the enclosing helper.
        run = yaml.safe_load(Path(values_files[-1]).read_text())["run"]
        proposals.append(run)
        return Manifest.parse(
            _orchestrator_manifest(
                state_file=run.get("stateFile"), restart_at=run.get("orchestrator", {}).get("restartAt")
            ),
            namespace=namespace,
        )

    return render_upgrade


def _orchestrator_manifest(*, state_file: str | None, restart_at: str | None) -> str:
    # [FUNCTION DOC] HELPER: Builds the YAML text (document-start marker followed by sorted keys)
    # of an apps/v1 stateful set named like the test release's orchestrator object. Its pod
    # template has one container named after the orchestrator component whose command is: python,
    # module wrapper, the state file flag and the state file (an empty string when none is
    # given), a double dash, then the installed argument vector. When restart_at is not none, the
    # pod template metadata carries the restart-at annotation with that value.
    template: dict = {
        "spec": {
            "containers": [
                {
                    "name": ORCHESTRATOR_COMPONENT,
                    "command": ["python", "-m", "wrapper", STATE_FILE_FLAG, state_file or "", "--", *_INSTALLED_ARGV],
                }
            ]
        }
    }
    if restart_at is not None:
        template["metadata"] = {"annotations": {RESTART_AT_ANNOTATION: restart_at}}
    return "---\n" + yaml.safe_dump(
        {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": {"name": _ORCHESTRATOR_OBJECT},
            "spec": {"template": template},
        },
        sort_keys=True,
    )


def _written_values(tmp_path) -> dict:
    # [FUNCTION DOC] HELPER: Reads the single "values-*.yaml" file in the run's values folder
    # under the sandbox cluster storage ("miles_data/miles-runs/<run id>/values"), asserting
    # exactly one exists, and returns its parsed YAML content.
    written = sorted(
        (tmp_path / "cluster-storage" / "miles_data" / "miles-runs" / _RUN_ID / "values").glob("values-*.yaml")
    )
    assert len(written) == 1, written
    return yaml.safe_load(written[0].read_text())


def _launchable_infra_file(tmp_path) -> Path:
    # [FUNCTION DOC] HELPER: Writes "launchable-infra.yaml" into the temporary directory with an
    # image "registry.local/miles" tag "v1", one host-path volume "cluster-storage" under it (also
    # mounted there) and the runs root set to its "miles_data" folder. Returns the file path.
    path = tmp_path / "launchable-infra.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "infra": {
                    "image": {"repository": "registry.local/miles", "tag": "v1"},
                    "volumes": [
                        {
                            "name": "cluster-storage",
                            "hostPath": {"path": str(tmp_path / "cluster-storage")},
                            "mounts": [{"mountPath": str(tmp_path / "cluster-storage")}],
                        }
                    ],
                    "paths": {"runsRoot": str(tmp_path / "cluster-storage" / "miles_data")},
                }
            }
        )
    )
    return path


_RENDERED = "---\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: myrun-values\n"
_INSTALLED_RECORD = "/shared/miles-runs/the-launch-that-installed-these-pods/launches/launch-1.json"
_RENDERED_WITH_RECORD = f"""---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: myrun-orchestrator
spec:
  template:
    spec:
      containers:
        - name: orchestrator
          command: [python]
          env:
            - name: MILES_SCRIPT_ENV_REPORT
              value: '{_INSTALLED_RECORD}'
"""
_RENDERED_TRAINER_WITH_RECORD = f"""---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: myrun-trainer-trainer-controller-actor
spec:
  template:
    spec:
      containers:
        - name: worker
          command: [python]
          env:
            - name: MILES_SCRIPT_ENV_REPORT
              value: '{_INSTALLED_RECORD}'
"""
_RENDERED_WITH_ANOTHER_KEY = _RENDERED + "data:\n  extra: added\n"
_ORCHESTRATOR_OBJECT = component_name(_RELEASE, ORCHESTRATOR_COMPONENT)
_RUN_UUID = "0123456789abcdef"
_REACHABLE_AT = {"actor": f"{_RELEASE}-trainer-controller-actor-0.rl.svc.cluster.local:29500"}
_INSTALLED_ARGV = ["python", "/repo/train.py", "--cluster-backend", "kubernetes", "--run-uuid", _RUN_UUID]
_RENDERED_ORCHESTRATOR = f"""---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {_ORCHESTRATOR_OBJECT}
spec:
  template:
    spec:
      containers:
        - name: orchestrator
          command: [python]
"""
_RENDERED_ORCHESTRATOR_REPLACED = _RENDERED_ORCHESTRATOR.replace("command: [python]", "command: [python, -u]")
_STAMP = "2026-08-12T09:00:00+00:00"


def _train_request() -> ExecuteTrainRequest:
    # [FUNCTION DOC] HELPER: Builds a minimal train request: empty train arguments, 8 GPUs per
    # node, no model type, script "/repo/train.py", not the FSDP backend, no extra environment, a
    # Megatron path, no pre-submit hook, no prepare commands and no extra manifests.
    return ExecuteTrainRequest(
        train_args="",
        num_gpus_per_node=8,
        megatron_model_type=None,
        train_script="/repo/train.py",
        train_backend_fsdp=False,
        extra_env_vars={},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd={},
        extra_manifests=[],
    )
