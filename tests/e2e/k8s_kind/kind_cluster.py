# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Creates and deletes kind clusters, locating or downloading a pinned kind
# binary. A kind on the search path is used only when its reported version equals the pinned one
# (v0.32.0); otherwise a copy is cached under the user's cache directory, downloaded once from the
# kind release site for the host operating system (linux or darwin) and architecture (amd64 or
# arm64) and verified against a pinned SHA-256 digest before being made executable and moved into
# place. Cluster creation waits up to 180 seconds for readiness.

from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from miles.utils.external_utils.command_utils.common import run_process

logger = logging.getLogger(__name__)

_READY_TIMEOUT = "180s"

_KIND_VERSION = "v0.32.0"
_KIND_CACHE_DIR = Path.home() / ".cache" / "miles" / "bin"
_KIND_SHA256 = {
    "linux-amd64": "50030de23cf40a18505f20426f6a8506bedf13c6e509244bd1fa9463721b0f54",
    "linux-arm64": "b92cd615e97585de8ddade28ed5cd7feb4248d717c233eea5b03c37298900f5d",
    "darwin-amd64": "295ac6d0d634c9819c9907df45e3017d1f13166bd13c3404c45e79f7faa47498",
    "darwin-arm64": "dca67911095a110c2b5c36e26df6cac860c602033e456c0db47be498cdef1ebb",
}


@dataclass(frozen=True)
class KindCluster:
    # [CLASS DOC] HELPER: Immutable handle of a cluster: its kind cluster name and the path of the
    # kubeconfig file that reaches it.
    name: str
    kubeconfig: Path


def create_cluster(*, run_id: str, kubeconfig: Path) -> KindCluster:
    # [FUNCTION DOC] HELPER: Runs the kind create command for a cluster named by the run id, writing
    # its kubeconfig to the given path and waiting up to 180 seconds for readiness. Raises on a
    # failing command. Returns the cluster handle.
    _run_kind("create", "cluster", "--name", run_id, "--kubeconfig", str(kubeconfig), "--wait", _READY_TIMEOUT)
    return KindCluster(name=run_id, kubeconfig=kubeconfig)


def delete_cluster(cluster: KindCluster) -> None:
    # [FUNCTION DOC] HELPER: Runs the kind delete command for the cluster and its kubeconfig; raises
    # on a failing command.
    _run_kind("delete", "cluster", "--name", cluster.name, "--kubeconfig", str(cluster.kubeconfig))


def _run_kind(*args: str) -> None:
    # [FUNCTION DOC] HELPER: Runs the resolved kind binary with the arguments, output not captured
    # (streams to the console), raising on a non-zero exit.
    run_process([_resolve_kind_binary(), *args], capture_output=False, check=True)


def _resolve_kind_binary() -> str:
    # [FUNCTION DOC] HELPER: Returns the path of a kind binary of the pinned version. Prefers the
    # one on the search path when its version matches; if it does not match, logs a warning and
    # ignores it. Otherwise uses the cached binary named with the pinned version, downloading it
    # first if absent.
    installed = shutil.which("kind")
    if installed is not None:
        if (installed_version := _installed_kind_version(installed)) == _KIND_VERSION:
            return installed
        logger.warning(
            f"ignoring the kind on PATH because it is not the pinned version "
            f"{installed=} {installed_version=} {_KIND_VERSION=}"
        )

    cached = _KIND_CACHE_DIR / f"kind-{_KIND_VERSION}"
    if not cached.exists():
        _download_kind(cached)
    return str(cached)


def _installed_kind_version(binary: str) -> str | None:
    # [FUNCTION DOC] HELPER: Runs the binary's version command; returns none on a non-zero exit,
    # else the second whitespace-separated word of its output (the version) or none if there is
    # no second word.
    result = run_process([binary, "version"], capture_output=True, check=False)
    if result.returncode != 0:
        return None
    fields = result.stdout.split()
    return fields[1] if len(fields) >= 2 else None


def _download_kind(destination: Path) -> None:
    # [FUNCTION DOC] HELPER: Downloads the pinned kind release for the host platform to the
    # destination. Looks up the pinned digest (key error for an unsupported OS/arch combination),
    # logs a warning, creates the parent directory, downloads over the network to a sibling file
    # with a partial suffix, asserts its SHA-256 equals the pinned digest, makes it executable
    # (mode 755), and atomically moves it to the destination. Side effects: network download and
    # file writes; the partial file is left behind if the digest check fails.
    platform_key = f"{_host_os()}-{_host_arch()}"
    expected_digest = _KIND_SHA256[platform_key]
    url = f"https://kind.sigs.k8s.io/dl/{_KIND_VERSION}/kind-{platform_key}"
    logger.warning(f"kind is not installed, downloading it once {url=} {destination=}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")
    urllib.request.urlretrieve(url, partial)
    actual_digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    assert (
        actual_digest == expected_digest
    ), f"downloaded kind does not match its pinned digest {url=} {expected_digest=} {actual_digest=}"
    partial.chmod(0o755)
    partial.replace(destination)


def _host_os() -> str:
    # [FUNCTION DOC] HELPER: Returns the platform name (linux or darwin); asserts no other
    # platform is in use.
    assert sys.platform in ("linux", "darwin"), f"kind is only wired up for linux and darwin, got {sys.platform=}"
    return sys.platform


def _host_arch() -> str:
    # [FUNCTION DOC] HELPER: Maps the machine type to amd64 (x86_64 or amd64) or arm64 (arm64 or
    # aarch64); asserts the machine is one of those.
    machine = platform.machine()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    assert arch is not None, f"unsupported machine for a kind download {machine=}"
    return arch
