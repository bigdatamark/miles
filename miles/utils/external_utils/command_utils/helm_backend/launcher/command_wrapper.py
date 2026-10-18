# [MODULE DOC] Thin wrappers over the helm and kubectl command-line tools, each as a collection of
# static methods that build an argument list and run it through the shared process runner
# (capturing text output). Helm: upgrade/install, dry-run rendering to a parsed manifest,
# template rendering, manifest lookup, dependency build, listing, uninstall. Kubectl: apply,
# create, replace, job and service deletion, typed json fetch, log commands, label selectors.
# Also a helper turning a values mapping into helm set arguments. Module constants: label names
# for CI runs and job names, and request timeouts (30 seconds for gets, 60 for creates).
from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel

from miles.utils.external_utils.command_utils.common import run_process
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.workers.worker_provider.kubernetes.helm.env import INSTANCE_LABEL

_ModelT = TypeVar("_ModelT", bound=BaseModel)

CI_LABEL = "miles.radixark.io/ci-run"
_JOB_NAME_LABEL = "batch.kubernetes.io/job-name"
_ALREADY_EXISTS = "AlreadyExists"
_JOB_COMPLETION_JSONPATH = "jsonpath={.status.succeeded},{.status.failed}"
_GET_REQUEST_TIMEOUT_SECONDS = 30.0
_CREATE_REQUEST_TIMEOUT_SECONDS = 60.0


class Helm:
    # [CLASS DOC] Namespace of static helm helpers; holds no state.
    @staticmethod
    def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Runs helm with the arguments, capturing text output and not raising on a
        # non-zero exit; returns the completed process for the caller to inspect.
        return run_process(["helm", *arguments], capture_output=True, check=False)

    @staticmethod
    def upgrade(
        *,
        release: str,
        namespace: str,
        chart: str | Path,
        values_files: list[str | Path],
        ci_run: bool,
    ) -> None:
        # [FUNCTION DOC] Keyword-only. Runs "helm upgrade --install" for the release, chart,
        # namespace and values files (with the CI label when ci_run is set), output not captured
        # (goes to the terminal). Raises a called-process error on failure.
        _run(Helm.upgrade_command(release, namespace, chart, values_files, ci_run=ci_run), capture_output=False)

    @staticmethod
    def render_upgrade(*, release: str, namespace: str, chart: str | Path, values_files: list[str | Path]) -> Manifest:
        # [FUNCTION DOC] Keyword-only. Runs the same upgrade command as a server-side dry run with
        # json output (nothing installed, no CI label), reads the manifest text from the output's
        # manifest field and parses it into a manifest object for the namespace. Raises on helm
        # failure or malformed output.
        rendered = _run(
            [
                *Helm.upgrade_command(release, namespace, chart, values_files, ci_run=False),
                "--dry-run",
                "--output",
                "json",
            ],
            capture_output=True,
        )
        return Manifest.parse(json.loads(rendered.stdout)["manifest"], namespace=namespace)

    @staticmethod
    def template(
        *,
        release: str,
        chart: str | Path,
        namespace: str,
        show_only: str,
        values: dict[str, Any],
        values_files: list[str],
    ) -> str:
        # [FUNCTION DOC] Keyword-only. Runs "helm template" for the release and chart in the
        # namespace, showing only the given template file, with the values mapping converted to set
        # arguments and each values file appended in order (later wins over earlier, and set
        # arguments are applied by helm on top). Returns the rendered text; raises on failure.
        command = [
            "helm",
            "template",
            release,
            str(chart),
            "--namespace",
            namespace,
            "--show-only",
            show_only,
            *_compute_helm_args(values),
        ]
        for values_file in values_files:
            command += ["--values", values_file]
        return _run(command, capture_output=True).stdout

    @staticmethod
    def get_manifest(release: str, namespace: str) -> Manifest | None:
        # [FUNCTION DOC] Fetches the installed manifest of the release. Success: parsed manifest.
        # Failure whose combined output contains "not found" (case-insensitive): none (release not
        # installed). Any other failure raises a runtime error with the helm message, so an unknown
        # state is never mistaken for absence.
        listed = run_process(
            ["helm", "get", "manifest", release, "--namespace", namespace], capture_output=True, check=False
        )
        if listed.returncode == 0:
            return Manifest.parse(listed.stdout, namespace=namespace)
        if "not found" in (listed.stderr + listed.stdout).lower():
            return None
        raise RuntimeError(
            f"Cannot tell whether release {release} exists: {listed.stderr.strip() or listed.stdout.strip()}"
        )

    @staticmethod
    def build_dependencies(chart: str | Path) -> None:
        # [FUNCTION DOC] Does nothing if every dependency named in the chart's lock file already
        # exists under the chart's charts directory (a chart without a lock file counts as
        # satisfied). Otherwise runs "helm dependency build" (terminal output); raises on failure.
        if all((Path(chart) / "charts" / name).exists() for name in _locked_dependency_names(chart)):
            return
        _run(["helm", "dependency", "build", str(chart)], capture_output=False)

    @staticmethod
    def list_releases(*, namespace: str, selector: str | None = None, name_filter: str | None = None) -> list[str]:
        # [FUNCTION DOC] Keyword-only. Lists release names in the namespace, optionally narrowed by
        # a label selector and a name regular expression filter (helm's filter option). Parses the
        # json output (empty output counts as an empty list). Raises on helm failure.
        command = ["helm", "list", "--namespace", namespace, "--output", "json"]
        if selector is not None:
            command += ["--selector", selector]
        if name_filter is not None:
            command += ["--filter", name_filter]
        listed = _run(command, capture_output=True)
        return [release["name"] for release in json.loads(listed.stdout or "[]")]

    @staticmethod
    def uninstall(*, release: str, namespace: str) -> None:
        # [FUNCTION DOC] Keyword-only. Runs "helm uninstall" for the release; raises on any
        # failure, including a release that does not exist.
        _run(["helm", "uninstall", release, "--namespace", namespace], capture_output=False)

    @staticmethod
    def uninstall_if_present(*, release: str, namespace: str) -> None:
        # [FUNCTION DOC] Keyword-only. Tries to uninstall; success, or a failure whose output says
        # "not found", returns quietly; any other failure raises a runtime error with exit code
        # and message.
        result = Helm.run_raw("uninstall", release, "--namespace", namespace)
        if result.returncode == 0:
            return

        reason = (result.stderr or result.stdout or "").strip()
        if "not found" in reason.lower():
            return
        raise RuntimeError(
            f"Could not uninstall Helm release {release!r} from namespace {namespace!r}: "
            f"exit code {result.returncode}: {reason}"
        )

    @staticmethod
    def upgrade_command(
        release: str, namespace: str, chart: str | Path, values_files: list[str | Path], *, ci_run: bool
    ) -> list[str]:
        # [FUNCTION DOC] Pure. Returns the argument list: helm upgrade with install-if-missing,
        # release, chart, namespace; then, if ci_run (keyword only), a labels option setting the CI
        # label to true; then a values option per file in order.
        command = ["helm", "upgrade", "--install", release, str(chart), "--namespace", namespace]
        if ci_run:
            command += ["--labels", f"{CI_LABEL}=true"]
        for values_file in values_files:
            command += ["--values", str(values_file)]
        return command


class Kubectl:
    # [CLASS DOC] Namespace of static kubectl helpers; holds no state. Most helpers capture text
    # output and do not raise on non-zero exit unless stated.
    @staticmethod
    def run_raw(*arguments: str) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] Runs kubectl with the arguments; returns the completed process without
        # raising on failure.
        return Kubectl._run(list(arguments))

    @staticmethod
    def apply(manifest: str, *, namespace: str) -> None:
        # [FUNCTION DOC] Applies the manifest text (fed on standard input) in the namespace;
        # asserts success, with the error output in the message.
        result = Kubectl._run(["apply", "--namespace", namespace, "-f", "-"], input=manifest)
        assert result.returncode == 0, f"Could not submit the job: {result.stderr}"

    @staticmethod
    def create_if_absent(manifest_path: str) -> bool:
        # [FUNCTION DOC] Creates the objects of a manifest file (60 second timeout). True if
        # created; False if the error output says they already exist; any other failure raises a
        # runtime error with the message.
        result = Kubectl._run(["create", "-f", manifest_path], timeout=_CREATE_REQUEST_TIMEOUT_SECONDS)
        if result.returncode == 0:
            return True
        if _ALREADY_EXISTS in result.stderr:
            return False
        raise RuntimeError(f"Could not create the objects of {manifest_path}: {result.stderr.strip()}")

    @staticmethod
    def jobs_finished(manifest_path: str) -> bool:
        # [FUNCTION DOC] Reads the succeeded and failed counts of the manifest's objects (as a
        # comma-separated pair via a json path) and returns true if either is a positive integer;
        # empty fields count as zero. Raises a runtime error if the read fails.
        result = Kubectl._run(["get", "-f", manifest_path, "--output", _JOB_COMPLETION_JSONPATH])
        if result.returncode != 0:
            raise RuntimeError(f"Could not read the objects of {manifest_path}: {result.stderr.strip()}")
        return any(count.isdigit() and int(count) > 0 for count in result.stdout.split(","))

    @staticmethod
    def replace(manifest_path: str) -> None:
        # [FUNCTION DOC] Force-replaces (delete and recreate) the manifest file's objects; raises
        # a runtime error with the message on failure.
        result = Kubectl._run(["replace", "--force", "-f", manifest_path])
        if result.returncode != 0:
            raise RuntimeError(f"Could not replace the objects of {manifest_path}: {result.stderr.strip()}")

    @staticmethod
    def delete_job(name: str, *, namespace: str, check: bool = False) -> None:
        # [FUNCTION DOC] Deletes the job, ignoring absence, with foreground cascading (waits until
        # dependents such as pods are gone). Failures are ignored unless check is true, in which
        # case a called-process error is raised.
        Kubectl._run(
            ["delete", "job", name, "--namespace", namespace, "--ignore-not-found", "--cascade", "foreground"],
            check=check,
        )

    @staticmethod
    def delete_service(name: str, *, namespace: str) -> None:
        # [FUNCTION DOC] Deletes the service, ignoring absence; any failure is silently ignored.
        Kubectl._run(["delete", "service", name, "--namespace", namespace, "--ignore-not-found"])

    @staticmethod
    def get_json(
        kind: str,
        *,
        return_type: type[_ModelT],
        name: str | None = None,
        namespace: str,
        selector: str | None = None,
        field_selector: str | None = None,
    ) -> _ModelT | None:
        # [FUNCTION DOC] Keyword-only (after the kind). Runs a get of the kind (optionally one
        # named object) in the namespace with json output, ignoring not-found, optional label
        # selector and field selector, 30 second timeout. A non-zero exit raises a runtime error.
        # Empty output gives none; otherwise the json is validated into the given model type.
        command = ["get", kind]
        if name is not None:
            command.append(name)
        command += ["--namespace", namespace, "--output", "json", "--ignore-not-found"]
        if selector is not None:
            command += ["--selector", selector]
        if field_selector is not None:
            command += ["--field-selector", field_selector]
        result = Kubectl._run(command, timeout=_GET_REQUEST_TIMEOUT_SECONDS)
        if result.returncode != 0:
            raise RuntimeError(f"kubectl get {kind} failed with code {result.returncode}: {result.stderr.strip()}")
        if not result.stdout.strip():
            return None
        return return_type.model_validate_json(result.stdout)

    @staticmethod
    def tail_logs(target: str, *, namespace: str, tail: int) -> str:
        # [FUNCTION DOC] Returns the last given number of log lines (with timestamps, all
        # containers) of the target; on failure returns the error output instead of raising.
        result = run_process(
            Kubectl.logs_command(namespace=namespace, target=target, tail=tail), capture_output=True, check=False
        )
        return result.stdout or result.stderr

    @staticmethod
    def full_logs(target: str, *, namespace: str) -> str:
        # [FUNCTION DOC] Returns the complete log (no timestamps, all containers) of the target;
        # raises a called-process error on failure.
        result = run_process(
            Kubectl.logs_command(namespace=namespace, target=target, timestamps=False),
            capture_output=True,
            check=True,
        )
        return result.stdout

    @staticmethod
    def logs_command(
        *,
        namespace: str,
        target: str,
        container: str | None = None,
        follow: bool = False,
        previous: bool = False,
        tail: int | None = None,
        since_time: str | None = None,
        timestamps: bool = True,
    ) -> list[str]:
        # [FUNCTION DOC] Pure; keyword-only. Builds the kubectl logs argument list for the target in
        # the namespace: timestamps option when requested (default on); the named container, or all
        # containers when none is named; follow, previous-instance, tail line count and since-time
        # options when set.
        command = ["kubectl", "logs", target, "--namespace", namespace]
        if timestamps:
            command.append("--timestamps")
        command += ["-c", container] if container is not None else ["--all-containers"]
        if follow:
            command.append("--follow")
        if previous:
            command.append("--previous")
        if tail is not None:
            command += ["--tail", str(tail)]
        if since_time is not None:
            command += ["--since-time", since_time]
        return command

    @staticmethod
    def release_selector(release: str) -> str:
        # [FUNCTION DOC] Label selector matching the objects of one release (instance label equals
        # the release).
        return f"{INSTANCE_LABEL}={release}"

    @staticmethod
    def releases_selector(releases: Sequence[str]) -> str:
        # [FUNCTION DOC] Label selector for several releases: the single-release form for one
        # release, otherwise an "in" set over all names. An empty sequence yields an empty set,
        # matching nothing.
        if len(releases) == 1:
            return Kubectl.release_selector(releases[0])
        return f"{INSTANCE_LABEL} in ({','.join(releases)})"

    @staticmethod
    def job_selector(name: str) -> str:
        # [FUNCTION DOC] Label selector matching the pods of a job by its job-name label.
        return f"{_JOB_NAME_LABEL}={name}"

    @staticmethod
    def _run(
        arguments: list[str], *, input: str | None = None, check: bool = False, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        # [FUNCTION DOC] HELPER: Runs kubectl with the argument list, capturing text output, with
        # optional standard input text, optional timeout and optional raising on failure (default
        # off).
        return run_process(["kubectl", *arguments], capture_output=True, check=check, input=input, timeout=timeout)


def _compute_helm_args(values: dict[str, Any]) -> list[str]:
    # [FUNCTION DOC] HELPER: Converts a mapping to helm arguments in insertion order: lists and
    # mappings become a set-json option with the json text; booleans become a set option with
    # lower-case true/false; everything else a set option with the value's text. Note that set
    # options interpret commas and dots in values specially (helm semantics); booleans are checked
    # after the list/mapping check.
    arguments: list[str] = []
    for key, value in values.items():
        if isinstance(value, (list, dict)):
            arguments += ["--set-json", f"{key}={json.dumps(value)}"]
        elif isinstance(value, bool):
            arguments += ["--set", f"{key}={str(value).lower()}"]
        else:
            arguments += ["--set", f"{key}={value}"]
    return arguments


def _run(command: list[str], capture_output: bool) -> subprocess.CompletedProcess[str]:
    # [FUNCTION DOC] HELPER: Runs the command raising on non-zero exit, capturing output or not
    # as requested.
    return run_process(command, capture_output=capture_output, check=True)


def _locked_dependency_names(chart: str | Path) -> list[str]:
    # [FUNCTION DOC] HELPER: Names of the dependencies recorded in the chart's lock file; empty if
    # there is no lock file or no dependencies entry.
    lock = Path(chart) / "Chart.lock"
    if not lock.exists():
        return []
    return [entry["name"] for entry in (yaml.safe_load(lock.read_text()) or {}).get("dependencies", [])]
