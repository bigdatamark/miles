# [MODULE DOC] Builds environment report objects. A cheap in-process snapshot (host name,
# redacted command line, redacted arguments, redacted environment variables, launcher report) is
# taken first; a full report additionally probes installed packages by running the pip inspect
# command in a subprocess (60 second timeout, using the snapshot's environment without
# PYTHONPATH), derives key package versions, and fingerprints the git state of every editable
# package. An "unprobed" report skips pip and git work.
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any
from miles.utils.audit_utils.event_logger.models import (
    EnvReport,
    EnvReportArgsDump,
    EnvReportEditablePackageInfo,
    EnvReportProcessFacts,
)
from miles.utils.env_report.git_state import collect_git_info
from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR, read_launcher_report
from miles.utils.env_report.redaction import redact_arg, redact_argv, redact_env_vars

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvReportSnapshot:
    # [CLASS DOC] Immutable record: the process facts, and the environment mapping to use when
    # running probe subprocesses (the process environment minus PYTHONPATH, per the TODO in the
    # code, so that Megatron is not picked up from the path).
    facts: EnvReportProcessFacts
    # TODO: remove the PYTHONPATH workaround and still make Megatron detected
    probe_env: dict[str, str]


@dataclass(frozen=True)
class _PipInfo:
    # [CLASS DOC] HELPER: Result of the package probe: whether it succeeded, the editable
    # packages, and the full list of installed packages as name/version mappings (both default
    # empty).
    probed: bool
    editable_packages: list[EnvReportEditablePackageInfo] = field(default_factory=list)
    full_pip_list: list[dict[str, str]] = field(default_factory=list)


_KEY_PACKAGE_NAMES = (
    "miles",
    "sglang",
    "sglang-router",
    "megatron-core",
    "transformers",
    "ray",
    "flashinfer-python",
    "vllm",
)


def collect_env_report_snapshot(args: Any) -> EnvReportSnapshot:
    # [FUNCTION DOC] Copies the process environment; the reported variables exclude the launcher
    # report variable. Builds the facts: host name, redacted process argv, dumped arguments
    # (see the dump helper), redacted environment variables, and the launcher report read from the
    # path in the arguments' env_report attribute (none if empty or unreadable). The probe
    # environment is the full environment minus PYTHONPATH (including the launcher variable).
    # Raises an attribute error if args has no env_report.
    environ = dict(os.environ)
    env_vars = {name: value for name, value in environ.items() if name != LAUNCHER_REPORT_ENV_VAR}

    facts = EnvReportProcessFacts(
        hostname=socket.gethostname(),
        argv=redact_argv(sys.argv),
        args=_dump_args(args),
        env_vars=redact_env_vars(env_vars),
        launcher_env_report=read_launcher_report(args.env_report),
    )
    return EnvReportSnapshot(facts=facts, probe_env={k: v for k, v in environ.items() if k != "PYTHONPATH"})


def collect_env_report(*, snapshot: EnvReportSnapshot) -> EnvReport:
    # [FUNCTION DOC] Full report: probes pip with the snapshot's probe environment; for each
    # editable package, collects git info at its location and keeps those that produced one.
    # Returns a report with the snapshot facts, key versions derived from the installed list,
    # editable packages, git repos, the full list, and the probed flag (false if pip failed, in
    # which case lists are empty). Spawns subprocesses.
    pip_info = _collect_pip_info(snapshot.probe_env)

    git_repos = [
        info
        for pkg in pip_info.editable_packages
        if (info := collect_git_info(package_name=pkg.name, location=pkg.location))
    ]

    return EnvReport(
        process=snapshot.facts,
        key_versions=_collect_key_versions(pip_info.full_pip_list),
        editable_packages=pip_info.editable_packages,
        git_repos=git_repos,
        full_pip_list=pip_info.full_pip_list,
        packages_probed=pip_info.probed,
    )


def collect_unprobed_env_report(*, snapshot: EnvReportSnapshot) -> EnvReport:
    # [FUNCTION DOC] Cheap report: snapshot facts, key versions from the interpreter and platform
    # (plus torch if already imported) only, empty package and git lists, probed flag false. No
    # subprocesses.
    return EnvReport(
        process=snapshot.facts,
        key_versions=_collect_key_versions([]),
        editable_packages=[],
        git_repos=[],
        full_pip_list=[],
        packages_probed=False,
    )


def _collect_key_versions(full_pip_list: list[dict[str, str]]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Returns a mapping with the Python version and platform string, then
    # the versions of the key packages (miles, sglang, sglang-router, megatron-core, transformers,
    # ray, flashinfer-python, vllm) present in the installed list (matched by lower-cased name).
    # If torch is already imported in this process (it is never imported here), adds its version
    # and its CUDA version (empty string if none).
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    installed = {entry["name"].lower(): entry["version"] for entry in full_pip_list}
    versions.update({name: installed[name] for name in _KEY_PACKAGE_NAMES if name in installed})

    if (torch := sys.modules.get("torch")) is not None:
        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda or ""

    return versions


def _dump_args(args: Any) -> EnvReportArgsDump:
    # [FUNCTION DOC] HELPER: Takes all attributes of the arguments object. If the whole mapping
    # survives a json round trip, uses the round-tripped copy; otherwise keeps only entries that
    # individually survive (non-serializable ones are skipped). Applies per-name redaction, sorts
    # by name, and returns the values plus the sorted list of skipped names. Round-tripping turns
    # tuples into lists and similar.
    declared = dict(vars(args))
    if (serializable := _json_snapshot(declared)) is None:
        serializable = {
            name: snapshot[name]
            for name, value in declared.items()
            if (snapshot := _json_snapshot({name: value})) is not None
        }

    values = {name: redact_arg(name, value) for name, value in sorted(serializable.items())}
    return EnvReportArgsDump(values=values, skipped_names=sorted(declared.keys() - serializable.keys()))


def _json_snapshot(values: dict[str, Any]) -> dict[str, Any] | None:
    # [FUNCTION DOC] HELPER: Serializes to json text and parses back to get a plain deep copy;
    # returns none if serialization fails with a type or value error.
    try:
        return json.loads(json.dumps(values))
    except (TypeError, ValueError):
        return None


def _collect_pip_info(env: dict[str, str]) -> _PipInfo:
    # [FUNCTION DOC] HELPER: Runs pip inspect (60 second timeout, given environment) and parses
    # its json. A non-zero exit logs a warning with the error output and gives an unprobed result.
    # Otherwise builds the name/version list for all installed distributions and, for those whose
    # direct-url record marks them editable, an editable package entry with the location taken from
    # the url with any file scheme prefix removed. Any exception (timeout, parse error, missing
    # field) is logged as a warning and yields an unprobed result.
    """Collect all pip info in a single `pip inspect` call."""
    try:
        result = subprocess.run(
            ["pip", "inspect"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if result.returncode != 0:
            logger.warning("pip inspect failed: %s", result.stderr)
            return _PipInfo(probed=False)

        data = json.loads(result.stdout)
        installed: list[dict[str, Any]] = data.get("installed", [])

        full_pip_list = [_parse_pip_entry(pkg) for pkg in installed]
        editable_packages = [
            EnvReportEditablePackageInfo(
                name=entry["name"],
                version=entry["version"],
                location=pkg["direct_url"]["url"].removeprefix("file://"),
            )
            for pkg, entry in zip(installed, full_pip_list, strict=True)
            if _is_editable(pkg)
        ]

        return _PipInfo(editable_packages=editable_packages, full_pip_list=full_pip_list, probed=True)
    except Exception:
        logger.warning("Failed to collect pip info", exc_info=True)
        return _PipInfo(probed=False)


def _parse_pip_entry(pkg: dict[str, Any]) -> dict[str, str]:
    # [FUNCTION DOC] HELPER: Extracts name and version from the entry's metadata (empty strings
    # when missing).
    metadata = pkg.get("metadata", {})
    return {"name": metadata.get("name", ""), "version": metadata.get("version", "")}


def _is_editable(pkg: dict[str, Any]) -> bool:
    # [FUNCTION DOC] HELPER: True when the entry has a direct-url record whose directory info
    # marks an editable install.
    direct_url = pkg.get("direct_url")
    return bool(direct_url and direct_url.get("dir_info", {}).get("editable"))
