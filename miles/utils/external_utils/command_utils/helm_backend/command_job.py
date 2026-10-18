# [MODULE DOC] Runs a shell command as a Kubernetes job with several completions (one pod per
# node) and waits for it. Flow: render the chart's command-job template with helm to a manifest,
# delete any stale job of the same name, apply the manifest, poll the job status every 5 seconds
# (timeout default 3 hours, also set as the job's active deadline) while streaming observability,
# then fetch logs per completion index. On failure or timeout raises a runtime error including the
# last 200 log lines per completion. The headless service is deleted at the end whether or not the
# run succeeded; the job itself is deleted only on success.
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from miles.utils.external_utils.command_utils.common import substitute_placeholders
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm, Kubectl
from miles.utils.external_utils.command_utils.helm_backend.launcher.observability import with_observability
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import CommandJobValues
from miles.utils.external_utils.command_utils.helm_backend.naming import RunNames
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.workers.k8s_types import Job, Pod, PodList

logger = logging.getLogger(__name__)

_JOB_TEMPLATE = "templates/command-job.yaml"
_COMPLETION_INDEX_KEY = "batch.kubernetes.io/job-completion-index"
_TERMINAL_LOG_LINES = 200
_POLL_INTERVAL_SECONDS = 5.0
_TIMEOUT_SECONDS = 3 * 60 * 60
_RELEASE = "miles-run-command"


class CommandJobContext(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable settings for command jobs: namespace, chart directory, helm values
    # files (default none), GPUs per node (required), and timeout in seconds (default 3 hours).
    namespace: str
    chart_dir: Path
    helm_values_files: tuple[str, ...] = ()
    gpus_per_node: int
    timeout_seconds: float = _TIMEOUT_SECONDS


class _CommandJob(FrozenStrictBaseModel):
    # [CLASS DOC] HELPER: One job instance: its context, the unique step name, the number of
    # completions (pods) and GPUs per pod. Derived names below.
    context: CommandJobContext
    step: str
    completions: int
    gpus_per_pod: int

    @property
    def object_name(self) -> str:
        # [FUNCTION DOC] Property: the Kubernetes object name, built by the naming helper from the
        # fixed release name "miles-run-command" and the step.
        return naming.component_name(_RELEASE, self.step)

    @property
    def master_address(self) -> str:
        # [FUNCTION DOC] Property: the fully qualified service address of pod 0 of the job (pod
        # name "<object>-0" under the service named like the object) in the namespace.
        return RunNames.service_fqdn(name=f"{self.object_name}-0.{self.object_name}", namespace=self.context.namespace)

    @property
    def pod_selector(self) -> str:
        # [FUNCTION DOC] Property: the label selector matching this job's pods, from the kubectl
        # wrapper.
        return Kubectl.job_selector(self.object_name)


def run_on_nodes(
    context: CommandJobContext,
    cmd: str,
    *,
    capture_output: bool,
    completions: int,
    step: str,
) -> list[str | None]:
    # [FUNCTION DOC] Runs the command on the given number of completions. Creates a job whose step
    # is the given step plus a random 6-hex-digit token (so concurrent invocations do not collide),
    # GPUs per pod from the context. Substitutes placeholders in the command: node rank becomes the
    # shell variable holding the job completion index, node count the completions, master address
    # the pod-0 service address, node ip the shell command printing the pod's address. Runs it with
    # bash -c. Returns one entry per completion: captured log text if capture is on, else none.
    job = _CommandJob(
        context=context,
        step=f"{step}-{_new_invocation_token()}",
        completions=completions,
        gpus_per_pod=context.gpus_per_node,
    )
    prepared = substitute_placeholders(
        cmd,
        node_rank="${JOB_COMPLETION_INDEX}",
        nnodes=str(completions),
        master_addr=job.master_address,
        node_ip="$(hostname -i)",
    )
    return _run_job(job, command=["bash", "-c", prepared], capture_output=capture_output)


def _new_invocation_token() -> str:
    # [FUNCTION DOC] HELPER: Six lowercase hex digits from a fresh random generator.
    return f"{random.Random().randint(0, 0xFFFFFF):06x}"


def _run_job(job: _CommandJob, *, command: list[str], capture_output: bool) -> list[str | None]:
    # [FUNCTION DOC] HELPER: Renders the manifest, deletes any existing job of that name, applies
    # the manifest. Then, under the observability context manager for the pods, waits for a
    # terminal state. If the outcome is not "complete", raises a runtime error naming the job,
    # the outcome and the joined tail logs of every completion. On success fetches full logs per
    # completion if capturing (else a list of none), deletes the job, and returns the list. In all
    # cases (final step) deletes the job's service.
    manifest = _render_job(job, command=command)

    Kubectl.delete_job(job.object_name, namespace=job.context.namespace)
    Kubectl.apply(manifest, namespace=job.context.namespace)

    try:
        with with_observability(namespace=job.context.namespace, selector=job.pod_selector):
            outcome = _wait(job)

        if outcome != "complete":
            raise RuntimeError(
                f"Job {job.object_name} {outcome}; last log lines:\n{_joined(_terminal_logs_per_completion(job))}"
            )

        logs = _full_logs_per_completion(job) if capture_output else [None] * job.completions
        Kubectl.delete_job(job.object_name, namespace=job.context.namespace)
        return logs
    finally:
        Kubectl.delete_service(job.object_name, namespace=job.context.namespace)


def _render_job(job: _CommandJob, *, command: list[str]) -> str:
    # [FUNCTION DOC] HELPER: Builds chart dependencies, then renders only the command-job template
    # with helm template for the fixed release in the namespace. Values: the command job values
    # (enabled, name, object name, command, completions, GPUs per pod, active deadline as an integer
    # number of seconds), flattened to dotted keys under the command-job prefix using aliases and
    # dropping unset fields, plus the run id "command-job"; the configured values files are also
    # passed. Returns the manifest text.
    Helm.build_dependencies(job.context.chart_dir)

    command_job = CommandJobValues(
        enabled=True,
        name=job.step,
        object_name=job.object_name,
        command=command,
        completions=job.completions,
        gpus_per_pod=job.gpus_per_pod,
        active_deadline_seconds=int(job.context.timeout_seconds),
    )
    values: dict[str, Any] = {
        f"commandJob.{key}": value for key, value in command_job.model_dump(by_alias=True, exclude_none=True).items()
    }
    values["run.id"] = "command-job"

    return Helm.template(
        release=_RELEASE,
        chart=job.context.chart_dir,
        namespace=job.context.namespace,
        show_only=_JOB_TEMPLATE,
        values=values,
        values_files=list(job.context.helm_values_files),
    )


def _wait(job: _CommandJob) -> str:
    # [FUNCTION DOC] HELPER: Polls the job status every 5 seconds until it is "complete" or
    # "failed" and returns that. If the accumulated waiting (counted by the sleeps only, not the
    # time spent in status calls) reaches the timeout, returns a text saying it did not finish
    # within that many seconds, which callers treat as a failure.
    waited = 0.0
    while waited < job.context.timeout_seconds:
        status = _job_status(job)
        if status in ("complete", "failed"):
            return status
        time.sleep(_POLL_INTERVAL_SECONDS)
        waited += _POLL_INTERVAL_SECONDS
    return f"did not finish within {job.context.timeout_seconds:.0f}s"


def _job_status(job: _CommandJob) -> str:
    # [FUNCTION DOC] HELPER: Fetches the job as json. Not found gives "pending". Among conditions
    # whose status is "True": types Complete or SuccessCriteriaMet give "complete"; types Failed
    # or FailureTarget give "failed" (the first matching condition in listing order wins). With
    # none, "running".
    described = Kubectl.get_json("job", return_type=Job, name=job.object_name, namespace=job.context.namespace)
    if described is None:
        return "pending"

    for condition in described.status.conditions:
        if condition.status != "True":
            continue
        if condition.type in ("Complete", "SuccessCriteriaMet"):
            return "complete"
        if condition.type in ("Failed", "FailureTarget"):
            return "failed"
    return "running"


def _full_logs_per_completion(job: _CommandJob) -> list[str]:
    # [FUNCTION DOC] HELPER: Complete logs, one text per completion index.
    return _execute_per_completion(job, read=lambda target: Kubectl.full_logs(target, namespace=job.context.namespace))


def _terminal_logs_per_completion(job: _CommandJob) -> list[str]:
    # [FUNCTION DOC] HELPER: The last 200 log lines, one text per completion index.
    return _execute_per_completion(
        job,
        read=lambda target: Kubectl.tail_logs(target, namespace=job.context.namespace, tail=_TERMINAL_LOG_LINES),
    )


def _execute_per_completion(job: _CommandJob, *, read: Callable[[str], str]) -> list[str]:
    # [FUNCTION DOC] HELPER: Lists the job's pods with their completion indices. If none are
    # found, reads the job's own log once and repeats it for every completion. Otherwise reads each
    # pod's log with the given reader and returns a list ordered by index 0..completions-1, using
    # a placeholder message for any index with no pod.
    pods = _pods_by_completion_index(job)
    if not pods:
        return [read(f"job/{job.object_name}")] * job.completions

    logs = {index: read(name) for index, name in pods}
    return [
        logs.get(index, f"no pod of this job reported completion index {index}") for index in range(job.completions)
    ]


def _pods_by_completion_index(job: _CommandJob) -> list[tuple[int, str]]:
    # [FUNCTION DOC] HELPER: Lists pods matching the job's selector; an unavailable listing gives
    # an empty list. Returns (completion index, pod name) pairs sorted by index.
    listed = Kubectl.get_json("pods", return_type=PodList, namespace=job.context.namespace, selector=job.pod_selector)
    if listed is None:
        return []

    return sorted((_completion_index_of_pod(pod), pod.metadata.name) for pod in listed.items)


def _completion_index_of_pod(pod: Pod) -> int:
    # [FUNCTION DOC] HELPER: The completion index from the pod's label, else its annotation, else
    # 0, as an integer.
    raw = pod.metadata.labels.get(_COMPLETION_INDEX_KEY, pod.metadata.annotations.get(_COMPLETION_INDEX_KEY))
    return int(raw) if raw is not None else 0


def _joined(logs: list[str]) -> str:
    # [FUNCTION DOC] HELPER: A single log is returned as is; several are joined with newlines,
    # each preceded by a "[completion index n]" header line.
    if len(logs) == 1:
        return logs[0]
    return "\n".join(f"[completion index {index}]\n{log}" for index, log in enumerate(logs))
