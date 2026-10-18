# [MODULE DOC] The launch record: an immutable, secret-redacted json description of what a launch
# installed (run, release, namespace, state file, values file, worker and orchestrator command
# lines, environment, controller addresses, restart stamp), written to the shared run directory so
# pods can report the launch that created them. Also finds the record path already attached to an
# installed release through the launcher report environment variable of its containers.
from __future__ import annotations

from pathlib import Path

from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR
from miles.utils.env_report.redaction import redact_argv, redact_env_vars
from miles.utils.external_utils.command_utils.helm_backend.launcher.manifest_types import Manifest
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel


class LaunchRecord(FrozenStrictBaseModel):
    # [CLASS DOC] Immutable strict record. Fields: run id, release, namespace, state file path,
    # values file path (text), worker argument list, orchestrator command list, environment
    # mapping, a mapping from trainer id to reachable address, and optional restart timestamp.
    run_id: str
    release: str
    namespace: str
    state_file: str
    values_file: str
    worker_argv: list[str]
    orchestrator_command: list[str]
    env: dict[str, str]
    reachable_at: dict[str, str]
    restart_at: str | None = None

    @classmethod
    def compute(cls, *, plan: LaunchPlan, values_file: Path, reachable_at: dict[str, str]) -> LaunchRecord:
        # [FUNCTION DOC] Class method, keyword-only. Copies identity fields, state file and restart
        # stamp from the launch plan, records the values file as text and the controller addresses
        # as given, and redacts secrets in the worker argument list and orchestrator command (flag
        # values) and in the environment (variables with secret-looking names).
        return cls(
            run_id=plan.run_id,
            release=plan.release,
            namespace=plan.namespace,
            state_file=plan.state_file,
            values_file=str(values_file),
            restart_at=plan.restart_at,
            worker_argv=redact_argv(plan.worker_argv),
            orchestrator_command=redact_argv(plan.orchestrator_command),
            env=redact_env_vars(plan.env),
            reachable_at=reachable_at,
        )

    def write(self, *, path: Path) -> None:
        # [FUNCTION DOC] Creates parent directories and atomically writes the record as indented
        # json to the path (overwriting).
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, self.model_dump_json(indent=2))


def installed_launch_record_file(*, manifest: Manifest) -> str | None:
    # [FUNCTION DOC] Keyword-only. Scans every container of every pod workload of the manifest in
    # order and returns the value of the first environment entry named like the launcher report
    # variable (the record path the installed pods were told); none if there is none.
    for described in manifest.pod_workloads:
        for found in described.containers:
            for entry in found.env:
                if entry.name == LAUNCHER_REPORT_ENV_VAR:
                    return entry.value
    return None
