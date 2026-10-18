# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Environment checks and switches shared by the Kubernetes end-to-end suites
# (API server suite and the kind suite). Three environment variables: a keep switch that leaves
# created environments running after tests, a require switch that makes a missing docker daemon
# an error instead of a skip (defaulting to on when the generic CI variable is truthy), and a path
# to an existing kubeconfig to reuse. A value counts as true unless it is unset, empty, or one of
# 0, false, no (case-insensitive).

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from miles.utils.external_utils.command_utils.common import run_shell_command

logger = logging.getLogger(__name__)

KEEP_ENV_VAR = "MILES_K8S_KEEP"
REQUIRE_ENV_VAR = "MILES_K8S_REQUIRE"
KUBECONFIG_ENV_VAR = "MILES_K8S_KUBECONFIG"


def new_run_id() -> str:
    # [FUNCTION DOC] HELPER: Returns a unique run id: a fixed prefix plus 8 random hex characters.
    return f"miles-k8s-{uuid.uuid4().hex[:8]}"


def keep_environment() -> bool:
    # [FUNCTION DOC] HELPER: Whether the keep-environment variable is set to a true value.
    return _is_truthy(os.environ.get(KEEP_ENV_VAR))


def existing_kubeconfig() -> Path | None:
    # [FUNCTION DOC] HELPER: The path from the kubeconfig variable with the home directory
    # expanded, or none when the variable is unset or empty.
    configured = os.environ.get(KUBECONFIG_ENV_VAR)
    return Path(configured).expanduser() if configured else None


def require_docker() -> None:
    # [FUNCTION DOC] HELPER: Returns silently if docker works. Otherwise builds a message with the
    # failure reason and raises a runtime error when docker is mandatory, else skips the current
    # pytest test or session.
    reason = _docker_failure_reason()
    if reason is None:
        return

    message = f"the Kubernetes suites need a working Docker daemon: {reason}"
    if _docker_is_mandatory():
        raise RuntimeError(message)
    pytest.skip(message)


def _docker_is_mandatory() -> bool:
    # [FUNCTION DOC] HELPER: If the require variable is set (even to a false value) its truthiness
    # decides; otherwise the generic CI variable's truthiness decides.
    override = os.environ.get(REQUIRE_ENV_VAR)
    if override is not None:
        return _is_truthy(override)
    return _is_truthy(os.environ.get("CI"))


def _docker_failure_reason() -> str | None:
    # [FUNCTION DOC] HELPER: None if docker is on the path and its info command succeeds;
    # otherwise a short reason string (not on path, or the info command's exit status).
    if shutil.which("docker") is None:
        return "docker is not on PATH"
    try:
        run_shell_command("docker info", capture_output=True)
    except subprocess.CalledProcessError as error:
        return f"`docker info` exited with {error.returncode}"
    return None


def _is_truthy(value: str | None) -> bool:
    # [FUNCTION DOC] HELPER: True when the value is not none and, lower-cased, is not empty, 0,
    # false or no.
    return value is not None and value.lower() not in ("", "0", "false", "no")
