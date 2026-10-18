# [MODULE DOC] Tests for the Kubernetes command job module, which runs a one-off shell command (a
# conversion or distributed launch step) as a cluster Job rendered from the run chart. Covers:
# job and service naming and the rank zero headless address, the helm render arguments (build
# chart dependencies first, show only the command job template, enable flag, step shape values,
# deadline, command passed as json, fixed run id, values files kept), placeholder substitution in
# multi node commands (node rank, node count, master address with a per invocation token, node
# ip), propagation of the context settings, the GPU single and multi node entry points of the
# Kubernetes backend, and the job lifecycle (delete previous, apply, poll with a fake kubectl,
# failure raises with logs and keeps the job, successful job and the service are deleted, per
# rank log capture in completion index order, error cases, timeout). All cluster interaction goes
# through a scripted fake kubectl; sleeping and observability wrappers are patched out.
import contextlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from miles.utils.external_utils.command_utils.helm_backend import command_job
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper

NAMESPACE = "rl"
_RELEASE = "miles-run-command"
CHART_DIR = Path("charts/miles-run")


@dataclass
class FakeKubectl:
    # [CLASS DOC] HELPER: Scripted fake of the process runner, expected to be called with kubectl
    # command lines only. Fields: statuses is a queue of job status names consumed by job status
    # reads (running, complete, failed or absent; when empty, reads report running forever),
    # pod_indices lists the completion indices of pods the pod listing should return (no pods when
    # empty), calls records every call's arguments without the program name, and logs_returncode is
    # the exit code of log reads (default 0).
    statuses: list[str]
    pod_indices: list[int] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)
    logs_returncode: int = 0

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: Records the arguments (asserting the program is kubectl) and
        # answers by verb. A get of pods returns a json list with one item per pod index (name
        # convert-N, uid, and the completion index label), or empty text when there are no pod
        # indices. Any other get pops the next status (default running) and returns the matching
        # json job document, or empty text for absent. Logs reads: if the log return code is
        # nonzero and the caller asked for checking, raises a called process error with the error
        # text "no such container"; otherwise returns the text "the output of" followed by the
        # first argument after the verb, with the configured return code. Everything else returns
        # success with empty output.
        assert argv[0] == "kubectl", f"only kubectl is expected to reach the process layer, got {argv[0]}"
        arguments = argv[1:]
        self.calls.append(arguments)
        if arguments[:2] == ["get", "pods"]:
            items = [
                {
                    "metadata": {
                        "name": f"convert-{index}",
                        "uid": f"uid-{index}",
                        "labels": {command_job._COMPLETION_INDEX_KEY: str(index)},
                    }
                }
                for index in self.pod_indices
            ]
            body = json.dumps({"items": items}) if self.pod_indices else ""
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout=body, stderr="")
        if arguments[0] == "get":
            status = self.statuses.pop(0) if self.statuses else "running"
            body = {
                "running": '{"status": {}}',
                "complete": '{"status": {"conditions": [{"type": "Complete", "status": "True"}]}}',
                "failed": '{"status": {"conditions": [{"type": "Failed", "status": "True"}]}}',
                "absent": "",
            }[status]
            return subprocess.CompletedProcess(args=argv, returncode=0, stdout=body, stderr="")
        if arguments[0] == "logs":
            if self.logs_returncode != 0 and kwargs.get("check"):
                raise subprocess.CalledProcessError(self.logs_returncode, argv, stderr="no such container")
            return subprocess.CompletedProcess(
                args=argv, returncode=self.logs_returncode, stdout=f"the output of {arguments[1]}", stderr=""
            )
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    def verbs(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the first argument (the kubectl verb) of each recorded call.
        return [call[0] for call in self.calls]

    def targets(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns, for each recorded call, the first two arguments joined by
        # a space (verb and resource kind).
        return [" ".join(call[:2]) for call in self.calls]


def _run(
    monkeypatch: pytest.MonkeyPatch,
    kubectl: FakeKubectl,
    completions: int = 1,
    capture_output: bool = False,
    timeout_seconds: float = 3600.0,
) -> list[str | None]:
    # [FUNCTION DOC] HELPER: Runs the module's private job runner against the fake kubectl.
    # Inputs: the fake, completions (default 1), whether to capture output (default false) and the
    # timeout in seconds (default 3600). The job render is replaced by a fixed one line job
    # document, and the cluster fakes are installed. The command is a bash invocation of convert.
    # Returns the per completion results (log text or none for each).
    monkeypatch.setattr(command_job, "_render_job", lambda job, command: "kind: Job\n")
    _stub_cluster(monkeypatch, kubectl)
    return command_job._run_job(
        _job(completions=completions, timeout_seconds=timeout_seconds),
        command=["bash", "-c", "convert"],
        capture_output=capture_output,
    )


def _stub_cluster(monkeypatch: pytest.MonkeyPatch, kubectl: FakeKubectl) -> None:
    # [FUNCTION DOC] HELPER: Installs the fake kubectl as the command wrapper's process runner,
    # makes the module's sleep a no-op, and makes the observability context manager a null one.
    monkeypatch.setattr(command_wrapper, "run_process", kubectl)
    monkeypatch.setattr(command_job.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(command_job, "with_observability", lambda **kwargs: contextlib.nullcontext())


def _job(**overrides: Any) -> command_job._CommandJob:
    # [FUNCTION DOC] HELPER: Builds a private command job. Overrides named values files, timeout
    # seconds or chart directory are moved into the job's context (built with 8 GPUs per node);
    # the remaining overrides replace the job's fields, whose defaults are step convert, 1
    # completion and 8 GPUs per pod.
    context_fields = {
        key: overrides.pop(key) for key in ("helm_values_files", "timeout_seconds", "chart_dir") if key in overrides
    }
    fields: dict[str, Any] = {
        "context": _context(gpus_per_node=8, **context_fields),
        "step": "convert",
        "completions": 1,
        "gpus_per_pod": 8,
        **overrides,
    }
    return command_job._CommandJob(**fields)


def _context(**overrides: Any) -> command_job.CommandJobContext:
    # [FUNCTION DOC] HELPER: Builds a command job context for the test namespace, with the test
    # chart directory and 1 GPU per node unless the overrides replace those or add fields.
    return command_job.CommandJobContext(
        namespace=NAMESPACE, **{"chart_dir": CHART_DIR, "gpus_per_node": 1, **overrides}
    )


def _gpu_backend() -> Any:
    # [FUNCTION DOC] HELPER: Skips the test unless torch is importable, then returns a Kubernetes
    # command backend built from a config with only the test namespace. Imports are deferred
    # until after the check because they need torch.
    pytest.importorskip("torch")
    from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig
    from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend

    return KubernetesCommandBackend(ExecuteTrainConfig(namespace=NAMESPACE))


def _record_run_job(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Replaces the module's private job runner with a recorder and returns
    # the (live) list of records. Each record maps the job plus every keyword argument (such as
    # the command and capture flag) the runner got.
    calls: list[dict[str, Any]] = []

    def fake_run_job(job: command_job._CommandJob, **kwargs: Any) -> list[str | None]:
        # [FUNCTION DOC] HELPER: Appends the job and keyword arguments as one record, and returns a
        # list of as many none values as the job has completions.
        calls.append({"job": job, **kwargs})
        return [None] * job.completions

    monkeypatch.setattr(command_job, "_run_job", fake_run_job)
    return calls


def _render_calls(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> list[list[str]]:
    # [FUNCTION DOC] HELPER: Renders a job through the module's private render function with the
    # process runner faked, and returns every command the render ran, in order. Overrides: a
    # command (default a bash invocation of convert) and a timeout in seconds (default 10800) are
    # taken out; the rest are passed to the job builder.
    captured: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        # [FUNCTION DOC] HELPER: Records the command and returns success with a one line job
        # document as output.
        captured.append(command)
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="kind: Job\n", stderr="")

    monkeypatch.setattr(command_wrapper, "run_process", fake_run)
    command = overrides.pop("command", ["bash", "-c", "convert"])
    command_job._render_job(
        _job(timeout_seconds=overrides.pop("timeout_seconds", 10800.0), **overrides), command=command
    )
    return captured


def _render(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> list[str]:
    # [FUNCTION DOC] HELPER: Runs the render recorder and returns the single helm template command
    # among the recorded commands; asserts there is exactly one.
    templates = [call for call in _render_calls(monkeypatch, **overrides) if call[:2] == ["helm", "template"]]
    assert len(templates) == 1, f"expected one helm template call, got {templates}"
    return templates[0]


class TestNaming:
    # [CLASS DOC] Tests of the job's derived names and the release constant.
    def test_names_the_job_the_way_the_chart_does(self):
        # [FUNCTION DOC] The default job's object name is the command release name plus the step name.
        """The launcher has to address an object it did not name itself."""
        assert _job().object_name == "miles-run-command-convert"

    def test_addresses_rank_zero_through_the_headless_service(self):
        # [FUNCTION DOC] The master address is the object name, a rank zero suffix, the service
        # (same object name), the namespace and the svc domain.
        """A multi-node step needs one address every pod agrees on before any of them is scheduled."""
        assert _job().master_address == ("miles-run-command-convert-0.miles-run-command-convert.rl.svc")

    def test_installs_command_jobs_under_their_own_release(self):
        # [FUNCTION DOC] The module's release constant equals the fixed command release name.
        """A step sharing a training run's release would be torn down with it, or collide with its objects."""
        assert command_job._RELEASE == "miles-run-command"


class TestRenderJob:
    # [CLASS DOC] Tests of the helm commands that render a job manifest.
    def test_builds_the_chart_dependencies_before_rendering(self, monkeypatch, tmp_path):
        # [FUNCTION DOC] With the locked dependency names faked to one subchart and a temporary
        # chart directory, the first recorded command is a helm dependency build and a helm
        # template command also appears.
        """A checkout carries the lock, not the subchart it pins, so a run whose first step is a
        command job renders against a chart helm calls incomplete."""
        monkeypatch.setattr(command_wrapper, "_locked_dependency_names", lambda chart: ["miles-common"])
        calls = _render_calls(monkeypatch, chart_dir=tmp_path)

        assert [call[:3] for call in calls][0] == ["helm", "dependency", "build"]
        assert any(call[:2] == ["helm", "template"] for call in calls)

    def test_renders_only_the_command_job_out_of_the_run_chart(self, monkeypatch):
        # [FUNCTION DOC] The template command's show-only option value is the command job template.
        """The chart also holds the run's own workloads, and applying those would start a training run."""
        arguments = _render(monkeypatch)

        assert arguments[arguments.index("--show-only") + 1] == "templates/command-job.yaml"

    def test_turns_the_command_job_on_because_a_run_leaves_it_off(self, monkeypatch):
        # [FUNCTION DOC] The template arguments include the setting that enables the command job.
        """The template renders nothing by default, so a step forgetting the flag would apply an empty manifest."""
        assert "commandJob.enabled=true" in _render(monkeypatch)

    def test_passes_the_step_shape_through_the_command_job_values(self, monkeypatch):
        # [FUNCTION DOC] A job with step convert, 4 completions and 8 GPUs per pod yields settings
        # for the name, completions and GPUs per pod values among the template arguments.
        """The pod count and the gpus each pod claims are what make a step single-node or multi-node."""
        arguments = _render(monkeypatch, step="convert", completions=4, gpus_per_pod=8)

        assert "commandJob.name=convert" in arguments
        assert "commandJob.completions=4" in arguments
        assert "commandJob.gpusPerPod=8" in arguments

    def test_gives_the_job_the_same_deadline_the_launcher_waits_for(self, monkeypatch):
        # [FUNCTION DOC] A timeout of 60 seconds appears as the active deadline seconds setting
        # equal to 60 (an integer, no decimals).
        """A Job outliving the waiter would keep a gpu busy for a launch that has already given up."""
        assert "commandJob.activeDeadlineSeconds=60" in _render(monkeypatch, timeout_seconds=60)

    def test_sends_the_command_as_json_so_helm_never_splits_it(self, monkeypatch):
        # [FUNCTION DOC] The argument after the set-json option is the command setting assigned a
        # json list of the command words, including a word with a comma.
        """A plain --set would split the command on commas, wrecking any argument holding one."""
        arguments = _render(monkeypatch, command=["bash", "-c", "echo a,b"])

        assert arguments[arguments.index("--set-json") + 1] == 'commandJob.command=["bash", "-c", "echo a,b"]'

    def test_names_the_run_the_chart_insists_on(self, monkeypatch):
        # [FUNCTION DOC] The template arguments set the run id to the fixed value command-job.
        """run.id is required by the chart, and a command job has no run to borrow an id from."""
        assert "run.id=command-job" in _render(monkeypatch)

    def test_keeps_the_cluster_values_files(self, monkeypatch):
        # [FUNCTION DOC] A values file given to the job appears in the template arguments directly
        # after a values option.
        """The image, the storage mounts and the node selectors all come from the infra values."""
        arguments = _render(monkeypatch, helm_values_files=("/infra.yaml",))

        assert arguments[arguments.index("/infra.yaml") - 1] == "--values"


class TestRunOnNodes:
    # [CLASS DOC] Tests of the public multi node entry point with the job runner recorded: it
    # takes a context, a command text with double brace placeholders, a capture flag, completions
    # and a step name, and hands the job runner a bash dash c command list.
    def test_substitutes_the_rank_with_what_the_job_gives_each_pod(self, monkeypatch):
        # [FUNCTION DOC] The node rank placeholder becomes a shell reference to the job completion
        # index environment variable, in the third element of the bash command.
        """A command templating the pod index has no other way to learn which pod it landed in."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "torchrun --node-rank {{node_rank}}",
            capture_output=False,
            completions=2,
            step="convert",
        )

        assert calls[0]["command"] == ["bash", "-c", "torchrun --node-rank ${JOB_COMPLETION_INDEX}"]

    def test_tells_every_pod_how_many_of_them_there_are(self, monkeypatch):
        # [FUNCTION DOC] The node count placeholder becomes the number of completions (4).
        """torchrun refuses to rendezvous unless every rank agrees on the world size."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "torchrun --nnodes={{nnodes}}",
            capture_output=False,
            completions=4,
            step="convert",
        )

        assert calls[0]["command"][-1] == "torchrun --nnodes=4"

    def test_points_every_pod_at_the_headless_address_of_rank_zero(self, monkeypatch):
        # [FUNCTION DOC] With the invocation token fixed to a known text, the master address
        # placeholder becomes the object name carrying that token with a rank zero suffix, then the
        # service of the same name, the namespace and the svc domain.
        """No pod knows another pod's ip before scheduling, but the service name is fixed in advance."""
        monkeypatch.setattr(command_job, "_new_invocation_token", lambda: "abc123")
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "--master-addr {{master_addr}}",
            capture_output=False,
            completions=2,
            step="convert",
        )

        assert calls[0]["command"][-1] == (
            "--master-addr miles-run-command-convert-abc123-0.miles-run-command-convert-abc123.rl.svc"
        )

    def test_resolves_a_pod_own_address_inside_the_pod(self, monkeypatch):
        # [FUNCTION DOC] The node ip placeholder becomes a shell command substitution running the
        # hostname command with the address option.
        """The launcher cannot know the ip a pod will get, so the pod has to look it up itself."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(),
            "--node-ip {{node_ip}}",
            capture_output=False,
            completions=1,
            step="convert",
        )

        assert calls[0]["command"][-1] == "--node-ip $(hostname -i)"

    def test_carries_the_context_settings_into_the_job(self, monkeypatch):
        # [FUNCTION DOC] The recorded job's context keeps the namespace, chart directory, values
        # files and a timeout of 60 seconds that the caller's context carried.
        """The namespace, the chart and the values files decide where and as what the step actually runs."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(helm_values_files=("/infra.yaml",), timeout_seconds=60.0),
            "echo hi",
            capture_output=True,
            completions=2,
            step="step",
        )

        context = calls[0]["job"].context
        assert context.namespace == NAMESPACE
        assert context.chart_dir == CHART_DIR
        assert context.helm_values_files == ("/infra.yaml",)
        assert context.timeout_seconds == 60.0

    def test_gives_every_pod_the_gpus_of_the_node_it_lands_on(self, monkeypatch):
        # [FUNCTION DOC] A context with 4 GPUs per node yields a job with 4 GPUs per pod.
        """These commands are written as if they owned the machine, and the context is where that count lives."""
        calls = _record_run_job(monkeypatch)

        command_job.run_on_nodes(
            _context(gpus_per_node=4), "nvidia-smi", capture_output=False, completions=2, step="convert"
        )

        assert calls[0]["job"].gpus_per_pod == 4


class TestExecCommandGpu:
    # [CLASS DOC] Tests of the backend's single node GPU command method. Requires torch (the
    # tests are skipped otherwise).
    def test_asks_for_a_single_pod_holding_the_whole_node(self, monkeypatch):
        # [FUNCTION DOC] Running with 4 GPUs per node gives a job with 1 completion and 4 GPUs per pod.
        """A gpu step is written as if it owned the machine, which a pod short of the node's gpus breaks."""
        calls = _record_run_job(monkeypatch)

        _gpu_backend().exec_command_gpu("nvidia-smi", num_gpus_per_node=4)

        assert (calls[0]["job"].completions, calls[0]["job"].gpus_per_pod) == (1, 4)

    def test_substitutes_the_placeholders_of_a_single_node_command_too(self, monkeypatch):
        # [FUNCTION DOC] With default GPU count, the node rank placeholder becomes the completion
        # index reference and the node count placeholder becomes 1.
        """A converter templating its rank is run on one node as often as on many."""
        calls = _record_run_job(monkeypatch)

        _gpu_backend().exec_command_gpu("torchrun --node-rank {{node_rank}} --nnodes={{nnodes}}")

        assert calls[0]["command"][-1] == "torchrun --node-rank ${JOB_COMPLETION_INDEX} --nnodes=1"

    def test_returns_the_single_result_rather_than_a_list(self, monkeypatch):
        # [FUNCTION DOC] With the job runner returning a one item list, the method returns that
        # single string when capturing output.
        """Its callers read the output as a string, and a list would silently become the wrong argument."""
        monkeypatch.setattr(command_job, "_run_job", lambda job, **kwargs: ["the output"])

        assert _gpu_backend().exec_command_gpu("nvidia-smi", capture_output=True) == "the output"


class TestExecCommandMultiNode:
    # [CLASS DOC] Tests of the backend's multi node command method. Requires torch.
    def test_gives_a_multi_node_step_the_gpus_of_every_node_it_lands_on(self, monkeypatch):
        # [FUNCTION DOC] 2 nodes with 4 GPUs each give a job with 2 completions and 4 GPUs per pod.
        """The ray backend runs these commands on whole gpu nodes, and a gpu-less pod fails torchrun outright."""
        calls = _record_run_job(monkeypatch)

        _gpu_backend().exec_command_multi_node("torchrun --nnodes={{nnodes}}", num_nodes=2, num_gpus_per_node=4)

        assert (calls[0]["job"].completions, calls[0]["job"].gpus_per_pod) == (2, 4)

    def test_names_the_objects_of_every_invocation_apart(self, monkeypatch):
        # [FUNCTION DOC] Two successive calls produce jobs whose object names both start with the
        # command release and command step prefix, and differ from each other (a unique token).
        """Two commands sharing a namespace deleted each other's Job before it had reported anything."""
        calls = _record_run_job(monkeypatch)
        backend = _gpu_backend()

        backend.exec_command_multi_node("hostname", num_nodes=1, num_gpus_per_node=4)
        backend.exec_command_multi_node("hostname", num_nodes=1, num_gpus_per_node=4)

        names = [call["job"].object_name for call in calls]
        assert all(name.startswith("miles-run-command-command-") for name in names)
        assert names[0] != names[1]


def _raise_unreachable(job: command_job._CommandJob) -> str:
    # [FUNCTION DOC] HELPER: Stand-in for the module's wait function; always raises a runtime
    # error saying the api server is unreachable.
    raise RuntimeError("the api server is unreachable")


class TestRunJob:
    # [CLASS DOC] Tests of the private job runner's lifecycle against the fake kubectl.
    def test_clears_a_previous_attempt_before_submitting(self, monkeypatch):
        # [FUNCTION DOC] The first kubectl verb issued is delete.
        """apply would refuse an existing Job, and its logs would describe the wrong run."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.verbs()[0] == "delete"

    def test_submits_the_rendered_manifest(self, monkeypatch):
        # [FUNCTION DOC] The second kubectl verb issued is apply.
        """A step whose manifest never reached the cluster would wait for a Job nobody created."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.verbs()[1] == "apply"

    def test_polls_until_the_job_finishes(self, monkeypatch):
        # [FUNCTION DOC] Statuses absent, running, running, complete lead to exactly four job
        # status reads.
        """An command job takes minutes, so one status read would always find it running."""
        kubectl = FakeKubectl(statuses=["absent", "running", "running", "complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.targets().count("get job") == 4

    def test_raises_with_the_logs_when_the_job_fails(self, monkeypatch):
        # [FUNCTION DOC] A failed status makes the run raise a runtime error whose message
        # contains the fake log output.
        """A failed conversion must stop the launch, and the reason is in the pod output."""
        kubectl = FakeKubectl(statuses=["failed"])

        with pytest.raises(RuntimeError, match="the output"):
            _run(monkeypatch, kubectl)

    def test_leaves_a_failed_job_in_place(self, monkeypatch):
        # [FUNCTION DOC] After a failed run (which raises), only one delete job call occurred (the
        # initial clean up), so the failed job was kept.
        """Its pods are the only evidence left, so deleting them would destroy the diagnosis."""
        kubectl = FakeKubectl(statuses=["failed"])

        with pytest.raises(RuntimeError):
            _run(monkeypatch, kubectl)

        assert kubectl.targets().count("delete job") == 1

    def test_deletes_a_successful_job(self, monkeypatch):
        # [FUNCTION DOC] After a successful run, two delete job calls occurred (initial clean up
        # and final deletion).
        """Finished Jobs otherwise pile up in the namespace until someone notices."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.targets().count("delete job") == 2

    def test_deletes_the_service_the_job_was_named_with(self, monkeypatch):
        # [FUNCTION DOC] After a successful run, exactly one delete service call occurred.
        """Every invocation names its own Service, so one left behind is one leaked per command run."""
        kubectl = FakeKubectl(statuses=["complete"])

        _run(monkeypatch, kubectl)

        assert kubectl.targets().count("delete service") == 1

    def test_deletes_the_service_of_a_failed_job_too(self, monkeypatch):
        # [FUNCTION DOC] After a failed run (raising a runtime error), exactly one delete service
        # call occurred.
        """The Job is kept for the diagnosis, but its Service holds no evidence and would leak instead."""
        kubectl = FakeKubectl(statuses=["failed"])

        with pytest.raises(RuntimeError):
            _run(monkeypatch, kubectl)

        assert kubectl.targets().count("delete service") == 1

    def test_deletes_the_service_when_the_wait_itself_blows_up(self, monkeypatch):
        # [FUNCTION DOC] The module's wait is replaced by one that raises an unreachable runtime
        # error; the run propagates that error and still issued exactly one delete service call.
        """A launcher that dies mid-wait would otherwise leave the Service behind for good."""
        kubectl = FakeKubectl(statuses=["complete"])
        monkeypatch.setattr(command_job, "_wait", _raise_unreachable)

        with pytest.raises(RuntimeError, match="unreachable"):
            _run(monkeypatch, kubectl)

        assert kubectl.targets().count("delete service") == 1

    def test_gives_back_one_result_per_node(self, monkeypatch):
        # [FUNCTION DOC] Four completions with pods indexed 0 to 3 and output capture: the result
        # is one log text per pod, in order, each naming that pod.
        """Callers of the multi-node helper index the result by rank."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0, 1, 2, 3])

        assert _run(monkeypatch, kubectl, completions=4, capture_output=True) == [
            f"the output of convert-{index}" for index in range(4)
        ]

    def test_reads_each_pod_own_log_in_completion_index_order(self, monkeypatch):
        # [FUNCTION DOC] Pods listed with indices 2, 0, 1: the returned logs and the order of log
        # reads are both by ascending completion index (convert-0, convert-1, convert-2).
        """One job log repeated N times hides every rank but one, unlike the ray backend it stands in for."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[2, 0, 1])

        logs = _run(monkeypatch, kubectl, completions=3, capture_output=True)

        assert logs == ["the output of convert-0", "the output of convert-1", "the output of convert-2"]
        assert [call[1] for call in kubectl.calls if call[0] == "logs"] == ["convert-0", "convert-1", "convert-2"]

    def test_selects_the_pods_of_this_job_alone(self, monkeypatch):
        # [FUNCTION DOC] The pod listing call's selector option value equals the wrapper's job
        # selector for the job object name.
        """A namespace runs several steps at once, and another step's pods would be read as this one's ranks."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0])

        _run(monkeypatch, kubectl, completions=1, capture_output=True)

        listing = next(call for call in kubectl.calls if call[:2] == ["get", "pods"])
        assert listing[listing.index("--selector") + 1] == command_wrapper.Kubectl.job_selector(
            "miles-run-command-convert"
        )

    def test_names_the_rank_whose_pod_never_appeared(self, monkeypatch):
        # [FUNCTION DOC] Two completions but only the pod index 0 listed: the second result is the
        # text saying no pod of this job reported completion index 1.
        """A silent empty string would read as a rank that ran and printed nothing."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0])

        logs = _run(monkeypatch, kubectl, completions=2, capture_output=True)

        assert logs[1] == "no pod of this job reported completion index 1"

    def test_falls_back_to_the_job_log_when_no_pod_can_be_listed(self, monkeypatch):
        # [FUNCTION DOC] No pods listed with two completions: each result is the log text read
        # from the job object itself (so the same text twice).
        """A step whose pods were already garbage collected must still surface whatever the job kept."""
        kubectl = FakeKubectl(statuses=["complete"])

        logs = _run(monkeypatch, kubectl, completions=2, capture_output=True)

        assert logs == ["the output of job/miles-run-command-convert"] * 2

    def test_captures_the_full_logs_unaltered(self, monkeypatch):
        # [FUNCTION DOC] No log read call carries a tail option or a timestamps option.
        """A parser reads this as the command's stdout, so a tail limit or a timestamp prefix corrupts it."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0])

        _run(monkeypatch, kubectl, completions=1, capture_output=True)

        reads = [call for call in kubectl.calls if call[0] == "logs"]
        assert all("--tail" not in call and "--timestamps" not in call for call in reads)

    def test_refuses_to_report_a_failed_log_read_as_the_output(self, monkeypatch):
        # [FUNCTION DOC] A log read that fails (nonzero code with checking requested) makes the
        # run raise a called process error rather than return error text.
        """kubectl's error text returned as stdout is a result the caller cannot tell from a real one."""
        kubectl = FakeKubectl(statuses=["complete"], pod_indices=[0], logs_returncode=1)

        with pytest.raises(subprocess.CalledProcessError):
            _run(monkeypatch, kubectl, completions=1, capture_output=True)

    def test_gives_back_nothing_when_the_output_was_not_asked_for(self, monkeypatch):
        # [FUNCTION DOC] Without output capture, the run returns a list holding a single none.
        """Most steps only care that the command worked, and a log dump would drown the launcher output."""
        kubectl = FakeKubectl(statuses=["complete"])

        assert _run(monkeypatch, kubectl) == [None]

    def test_hands_its_own_timeout_to_the_rendered_job(self, monkeypatch):
        # [FUNCTION DOC] The render is replaced by a recorder of the job it receives; a job with a
        # 90 second timeout is rendered with a context timeout whose integer value is 90.
        """Two independent timeouts would drift, leaving either an orphan Job or a premature failure."""
        rendered: list[command_job._CommandJob] = []
        monkeypatch.setattr(command_job, "_render_job", lambda job, command: rendered.append(job) or "kind: Job\n")
        _stub_cluster(monkeypatch, FakeKubectl(statuses=["complete"]))

        command_job._run_job(_job(timeout_seconds=90.0), command=["bash", "-c", "convert"], capture_output=False)

        assert int(rendered[0].context.timeout_seconds) == 90

    def test_reports_a_job_that_never_finishes(self, monkeypatch):
        # [FUNCTION DOC] With no scripted statuses (reads report running forever) and a 10 second
        # timeout, the run raises a runtime error saying the job did not finish (the sleep is a
        # no-op, so the timeout must be tracked in a way that does not rely on real elapsed time).
        """A step waiting on a gpu that never frees must fail rather than hang the launch forever."""
        kubectl = FakeKubectl(statuses=[])

        with pytest.raises(RuntimeError, match="did not finish"):
            _run(monkeypatch, kubectl, timeout_seconds=10.0)
