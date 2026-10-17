# [MODULE DOC]
# Frozen, strict argument models passed from the command line layer to the workbench actions.
from __future__ import annotations

from pathlib import Path

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ReleaseArgs(FrozenStrictBaseModel):
    # [CLASS DOC] Identifies a helm release: namespace and release name.
    namespace: str
    release: str


class InstallArgs(ReleaseArgs):
    # [CLASS DOC] Install options in addition to the release: whether to create rbac objects, whether
    # to grant leader worker set rights, dry run, values files, raw helm set overrides, whether to
    # skip preflight checks, and the readiness timeout in seconds.
    rbac: bool
    lws: bool
    dry_run: bool
    values: tuple[Path, ...]
    overrides: tuple[str, ...]
    skip_preflight: bool
    timeout: int


class ExecArgs(ReleaseArgs):
    # [CLASS DOC] Exec options in addition to the release: the command words to run (empty means
    # the default shell).
    command: tuple[str, ...]


class DiagnosisArgs(FrozenStrictBaseModel):
    # [CLASS DOC] Diagnosis options: namespace, directory in which the diagnosis directory is
    # created, and an optional run directory whose latest verdict is included.
    namespace: str
    output_dir: Path
    run_dir: Path | None
