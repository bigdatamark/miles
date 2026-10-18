# [MODULE DOC] HELPER: Shared helpers for the Helm chart tests. Defines repository paths (the
# workbench chart, run chart, shared infra schema), fixed release, object and namespace names used
# when rendering, a skip marker that requires the helm executable (skipped locally when helm is
# missing, but a failure when the CI environment variable is set), functions that render charts
# through helm template and parse the multi-document output, accessors that pick objects out of
# rendered output by kind or name, a runner for the workbench command line module, role rule
# merging helpers, and a JSON schema reference resolver. Rendering runs real helm subprocesses.
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import (
    ObjectIdentity,
    compute_api_group,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CHARTS_DIR = REPO_ROOT / "charts"
CHART_DIR = CHARTS_DIR / "miles-workbench"
RUN_CHART_DIR = CHARTS_DIR / "miles-run"
SHARED_INFRA_SCHEMA_PATH = CHARTS_DIR / "shared-infra.schema.json"
WORKBENCH_PACKAGE = "miles.utils.external_utils.miles_workbench"

RELEASE_NAME = "miles-workbench-myuser"
UNINSTALLER_SERVICE_ACCOUNT = "miles-uninstaller"
RUN_RELEASE_NAME = "myrun"
RUN_ORCHESTRATOR_NAME = f"{RUN_RELEASE_NAME}-miles-run-orchestrator"
RUN_PLATFORM_READ_NAME = f"{RUN_RELEASE_NAME}-miles-run-platform-read"
RUN_PLATFORM_READ_DELETE_NAME = f"{RUN_RELEASE_NAME}-miles-run-platform-read-delete"
RUN_UNINSTALL_JOB_NAME = f"{RUN_RELEASE_NAME}-miles-run-uninstall"
RUN_UNINSTALL_MANIFEST_NAME = f"{RUN_RELEASE_NAME}-miles-run-uninstall-manifest"
RUN_ID = "260101-000000-000"
RUN_STATE_FILE = f"/cluster-storage/miles_data/miles-runs/{RUN_ID}/state/orchestrator.state"
DEFAULT_ORCHESTRATOR_COMMAND = ["python", "train.py"]
OBJECT_NAME = RELEASE_NAME
NAMESPACE = "myns"
LWS_RESOURCE = "leaderworkersets.leaderworkerset.x-k8s.io"

requires_helm = pytest.mark.skipif(
    shutil.which("helm") is None and not os.environ.get("CI"),
    reason="helm is required to render charts; CI installs it, so a CI run fails instead of skipping",
)


def run_helm_template(*args: str) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Runs helm template for the workbench chart with the fixed release
    # name and namespace, setting the object name value, followed by the caller's extra helm
    # arguments. Returns the completed process (text output captured, no exception on failure).
    return subprocess.run(
        [
            "helm",
            "template",
            RELEASE_NAME,
            str(CHART_DIR),
            "-n",
            NAMESPACE,
            "--set",
            f"objectName={OBJECT_NAME}",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def run_helm_lint(*args: str) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Runs helm lint on the workbench chart with the object name value set
    # plus the caller's extra arguments. Returns the completed process (text captured, no raise).
    return subprocess.run(
        ["helm", "lint", str(CHART_DIR), "--set", f"objectName={OBJECT_NAME}", *args], capture_output=True, text=True
    )


def documents_of(manifest: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Parses a multi-document YAML string and returns the list of documents,
    # dropping empty ones.
    return [document for document in yaml.safe_load_all(manifest) if document is not None]


def render(*args: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Renders the workbench chart with the extra helm arguments and returns
    # the parsed objects. Asserts helm succeeded, with the error output as the failure message.
    result = run_helm_template(*args)
    assert result.returncode == 0, result.stderr
    return documents_of(result.stdout)


def render_error(*args: str) -> str:
    # [FUNCTION DOC] HELPER: Renders the workbench chart expecting failure; asserts the helm exit
    # code is nonzero (standard output as the failure message) and returns the error output text.
    result = run_helm_template(*args)
    assert result.returncode != 0, result.stdout
    return result.stderr


def schema_error_mentions(error: str, *, path: tuple[str, ...]) -> bool:
    # [FUNCTION DOC] HELPER: Returns true if the error text refers to the value path, either in
    # slash-separated form with a leading slash or in dot-separated form.
    return "/" + "/".join(path) in error or ".".join(path) in error


def run_helm_template_run(*args: str) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Runs helm template for the run chart with the fixed run release name
    # and namespace. Always sets: the run id, the orchestrator state file path, the object names
    # for the orchestrator, platform read, platform read-delete, mooncake master, colocate pairing,
    # uninstall job and uninstall manifest, and (as JSON) the default orchestrator command. Then
    # appends the caller's extra arguments. Returns the completed process (text captured).
    return subprocess.run(
        [
            "helm",
            "template",
            RUN_RELEASE_NAME,
            str(RUN_CHART_DIR),
            "-n",
            NAMESPACE,
            "--set",
            f"run.id={RUN_ID}",
            "--set",
            f"run.stateFile={RUN_STATE_FILE}",
            "--set",
            f"run.objectNames.orchestrator={RUN_ORCHESTRATOR_NAME}",
            "--set",
            f"run.objectNames.platformRead={RUN_PLATFORM_READ_NAME}",
            "--set",
            f"run.objectNames.platformReadDelete={RUN_PLATFORM_READ_DELETE_NAME}",
            "--set",
            f"run.objectNames.mooncakeMaster={RUN_RELEASE_NAME}-miles-run-mooncake-master",
            "--set",
            f"run.objectNames.colocatePairing={RUN_RELEASE_NAME}-miles-run-colocate-pairing",
            "--set",
            f"run.objectNames.uninstall={RUN_UNINSTALL_JOB_NAME}",
            "--set",
            f"run.objectNames.uninstallManifest={RUN_UNINSTALL_MANIFEST_NAME}",
            "--set-json",
            f"run.orchestrator.command={json.dumps(DEFAULT_ORCHESTRATOR_COMMAND)}",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def volumes_args(*volumes: dict[str, Any]) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Returns helm arguments setting the infra volumes value to the given
    # volume mappings, serialized as JSON.
    return ("--set-json", f"infra.volumes={json.dumps(list(volumes))}")


def host_path_volume(
    *, name: str = "cluster-storage", path: str = "/cluster-storage", mounts: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds a volume mapping backed by a host directory. Keyword-only
    # inputs: name (default "cluster-storage"), path (default "/cluster-storage") and optional
    # mounts (default: a single mount at /cluster-storage). The host path type is Directory.
    return {
        "name": name,
        "hostPath": {"path": path, "type": "Directory"},
        "mounts": mounts if mounts is not None else [{"mountPath": "/cluster-storage"}],
    }


def with_object_names(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Returns copies of the entries, each with an added object name equal to
    # the run release name, "-miles-run-" and the entry's own name.
    return [{**entry, "objectName": f"{RUN_RELEASE_NAME}-miles-run-{entry['name']}"} for entry in entries]


def render_run(*args: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Renders the run chart with the extra helm arguments and returns the
    # parsed objects; asserts helm succeeded (error output as the message).
    result = run_helm_template_run(*args)
    assert result.returncode == 0, result.stderr
    return documents_of(result.stdout)


def render_run_error(*args: str) -> str:
    # [FUNCTION DOC] HELPER: Renders the run chart expecting failure; asserts a nonzero exit code
    # and returns the error output text.
    result = run_helm_template_run(*args)
    assert result.returncode != 0, result.stdout
    return result.stderr


def extra_manifests_args(*manifests: str) -> tuple[str, ...]:
    # [FUNCTION DOC] HELPER: Returns helm arguments setting the extra manifests value to the given
    # manifest strings, serialized as a JSON list.
    return ("--set-json", f"extraManifests={json.dumps(list(manifests))}")


def objects_added_by(*manifests: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Renders the run chart twice, without and with the given extra
    # manifests, and returns the objects of the second rendering whose identity (API group, kind,
    # namespace defaulting to the test namespace, and name) is absent from the first. This isolates
    # what the extra manifests contributed.
    def identity(obj: dict[str, Any]) -> ObjectIdentity:
        # [FUNCTION DOC] HELPER: Builds the identity record of a rendered object; the API group is
        # computed from its API version (empty if missing) and the namespace falls back to the
        # test namespace. Raises a key error if kind or name is missing.
        return ObjectIdentity(
            api_group=compute_api_group(obj.get("apiVersion", "")),
            kind=obj["kind"],
            namespace=obj["metadata"].get("namespace", NAMESPACE),
            name=obj["metadata"]["name"],
        )

    installed_anyway = {identity(obj) for obj in render_run()}
    return [obj for obj in render_run(*extra_manifests_args(*manifests)) if identity(obj) not in installed_anyway]


def named_object(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the single object of the given kind and metadata name; asserts
    # exactly one matches, listing the matching names on failure.
    matched = [obj for obj in objects_of_kind(objects, kind) if obj["metadata"]["name"] == name]
    assert len(matched) == 1, f"expected one {kind}/{name}, got {[obj['metadata']['name'] for obj in matched]}"
    return matched[0]


def pod_spec_of(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the pod template spec of the named object of the given kind
    # (which must have a template in its spec).
    return named_object(objects, kind, name)["spec"]["template"]["spec"]


def sole_container_of(objects: list[dict[str, Any]], kind: str, name: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the only container of the named object's pod spec; asserts the
    # pod has exactly one container.
    containers = pod_spec_of(objects, kind, name)["containers"]
    assert len(containers) == 1
    return containers[0]


def run_workbench(*args: str, interpreter: str | None = None, **kwargs: Any) -> subprocess.CompletedProcess:
    # [FUNCTION DOC] HELPER: Runs the workbench command line package as a module in a subprocess,
    # with the repository root as working directory, text mode, a sixty second timeout, and the
    # given arguments. The interpreter defaults to the current one. The environment is the current
    # environment with color disabled and a dumb terminal type (so option names in messages are not
    # split by color codes), overlaid with any env mapping in the extra keyword arguments; all other
    # keyword arguments (for example output capture or input) pass to the subprocess call. Returns
    # the completed process; raises a timeout error after sixty seconds.
    command = [interpreter or sys.executable, "-m", WORKBENCH_PACKAGE, *args]
    # rich colours an option name in pieces, so a coloured '--namespace' contains no such
    # substring; a runner that reports colour support would hide every message assertion here
    env = {**os.environ, "NO_COLOR": "1", "TERM": "dumb", **kwargs.pop("env", {})}
    return subprocess.run(command, cwd=REPO_ROOT, text=True, timeout=60, env=env, **kwargs)


def merged_rules(*rules: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    # [FUNCTION DOC] HELPER: Merges several mappings of resource name to verbs into one: verbs for
    # the same resource are unioned. Returns a mapping from resource to a sorted, deduplicated tuple
    # of verbs.
    merged: dict[str, set[str]] = {}
    for rule in rules:
        for resource, verbs in rule.items():
            merged.setdefault(resource, set()).update(verbs)
    return {resource: tuple(sorted(verbs)) for resource, verbs in merged.items()}


def can_i_queries(rules: dict[str, tuple[str, ...]]) -> set[str]:
    # [FUNCTION DOC] HELPER: Converts a mapping from resource to verbs into the set of permission
    # check query strings, one per verb and resource: "<verb> <resource>", where a resource of the
    # form "resource/subresource" becomes "<verb> <resource> --subresource=<subresource>".
    queries = set()
    for resource, verbs in rules.items():
        target, _, subresource = resource.partition("/")
        for verb in verbs:
            queries.add(f"{verb} {target} --subresource={subresource}" if subresource else f"{verb} {target}")
    return queries


def resolved_schema(path: Path) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Reads a JSON schema file and returns it with every reference to a named
    # definition replaced inline by that definition (recursively) and the definitions section
    # removed. Raises on missing file or invalid JSON.
    schema = json.loads(path.read_text())
    return _resolve_refs(schema, schema.get("definitions", {}))


def _resolve_refs(node: Any, definitions: dict[str, Any]) -> Any:
    # [FUNCTION DOC] HELPER: Recursively walks a schema node. Lists are mapped element-wise;
    # non-mapping values are returned as is; a mapping with a reference key is replaced by the
    # resolved definition named by the last path segment of the reference (a missing definition
    # raises a key error; cyclic references would recurse without end); other mappings are copied
    # with resolved values and without any definitions key.
    if isinstance(node, list):
        return [_resolve_refs(entry, definitions) for entry in node]
    if not isinstance(node, dict):
        return node
    if (reference := node.get("$ref")) is not None:
        return _resolve_refs(definitions[reference.rsplit("/", maxsplit=1)[1]], definitions)
    return {key: _resolve_refs(value, definitions) for key, value in node.items() if key != "definitions"}


def chart_directories() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the sorted directories under the charts directory whose chart
    # definition file does not declare a library type, i.e. the application charts. Asserts at
    # least one exists.
    chart_dirs = sorted(
        path.parent
        for path in SHARED_INFRA_SCHEMA_PATH.parent.glob("*/Chart.yaml")
        if "type: library" not in path.read_text()
    )
    assert chart_dirs, "no application chart found under charts/"
    return chart_dirs


def library_chart_directories() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the sorted directories under the charts directory whose chart
    # definition file declares the library type (a plain text substring check).
    return sorted(
        path.parent
        for path in SHARED_INFRA_SCHEMA_PATH.parent.glob("*/Chart.yaml")
        if "type: library" in path.read_text()
    )


def objects_of_kind(objects: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Returns the objects whose kind equals the given kind, in order.
    return [obj for obj in objects if obj["kind"] == kind]


def single_object_of_kind(objects: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the only object of the kind; asserts exactly one exists,
    # listing the names found on failure.
    matched = objects_of_kind(objects, kind)
    assert len(matched) == 1, f"expected exactly one {kind}, got {[obj['metadata']['name'] for obj in matched]}"
    return matched[0]


def pod_spec(objects: list[dict[str, Any]]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the pod template spec of the single StatefulSet among the
    # objects (asserts exactly one StatefulSet exists).
    return single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]


def container(objects: list[dict[str, Any]]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the only container of the single StatefulSet's pod spec;
    # asserts the pod has exactly one container.
    containers = pod_spec(objects)["containers"]
    assert len(containers) == 1
    return containers[0]
