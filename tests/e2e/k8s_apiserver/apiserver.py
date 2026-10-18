# doc-dev: docs/developer/reconcile-loop.md
# [MODULE DOC] HELPER: Starts, inspects, restarts and stops a disposable single-node Kubernetes
# API server plus a single etcd, each as a docker container on a private docker network named
# after the run id. The API server runs with allow-all authorization, a static bearer token with
# admin group, the service-account admission plugin disabled, an RSA service-account key
# generated into the work directory and mounted read-only, and a published localhost port chosen
# by binding a free port. Images are pinned (etcd 3.5.15, kube-apiserver 1.31.4) and pulled with
# up to 4 attempts and linearly growing backoff (5 seconds times the attempt number).

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from miles.utils.external_utils.command_utils.common import run_shell_command

logger = logging.getLogger(__name__)

_ETCD_IMAGE = "registry.k8s.io/etcd:3.5.15-0"
_APISERVER_IMAGE = "registry.k8s.io/kube-apiserver:v1.31.4"
_SECURE_PORT = 6443
_KEY_MOUNT_DIR = "/etc/miles-k8s"
_SERVICE_ACCOUNT_KEY = "service-account.key"
_TOKEN_FILE = "token.csv"
_TOKEN = "miles-k8s-test-token"
_IMAGE_PULL_ATTEMPTS = 4
_IMAGE_PULL_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class ApiserverEnvironment:
    # [CLASS DOC] HELPER: Immutable handle of a running test API server: published localhost port,
    # the bearer token to authenticate with, and the docker network, etcd container and API
    # server container names.
    host_port: int
    token: str
    network_name: str
    etcd_name: str
    apiserver_name: str

    @property
    def endpoint(self) -> str:
        # [FUNCTION DOC] HELPER: The secure URL of the API server on the loopback address and the
        # published port (the server uses a self-signed certificate).
        return f"https://127.0.0.1:{self.host_port}"


def start_apiserver(*, run_id: str, work_dir: Path, watch_cache: bool = True) -> ApiserverEnvironment:
    # [FUNCTION DOC] HELPER: Starts etcd and the API server and returns their handle. Names are the
    # run id with network, etcd and apiserver suffixes. Steps: generate a 2048-bit RSA
    # service-account key in the work directory; write a token file (static token, user, uid and
    # admin group) there; pull both images with retries; create the docker network; run etcd
    # detached listening on all interfaces; pick a free host port; run the API server detached,
    # publishing its secure port 6443 on loopback only and mounting the work directory read-only;
    # the watch cache can be turned off (used to make watches expire). It does not wait for the
    # server to serve. On any failure, including interruption, the containers and network are
    # removed best-effort and the error is re-raised. Side effects: docker commands, files in
    # the work directory. Uses a fixed token shared by all environments.
    network_name = f"{run_id}-net"
    etcd_name = f"{run_id}-etcd"
    apiserver_name = f"{run_id}-apiserver"

    try:
        run_shell_command(f"openssl genrsa -out {work_dir / _SERVICE_ACCOUNT_KEY} 2048")
        (work_dir / _TOKEN_FILE).write_text(f"{_TOKEN},miles-test,miles-test-uid,system:masters\n")
        # the registry resets a connection often enough to fail a whole module on one attempt, and an
        # image left to a docker run's implicit pull reports the reset as a container that would not start
        _pull_image(_ETCD_IMAGE)
        _pull_image(_APISERVER_IMAGE)
        run_shell_command(f"docker network create {network_name}")
        run_shell_command(
            f"docker run --detach --name {etcd_name} --network {network_name} {_ETCD_IMAGE} "
            f"etcd --data-dir /tmp/etcd "
            f"--advertise-client-urls http://0.0.0.0:2379 --listen-client-urls http://0.0.0.0:2379"
        )
        host_port = _free_host_port()
        run_shell_command(
            f"docker run --detach --name {apiserver_name} --network {network_name} "
            f"--publish 127.0.0.1:{host_port}:{_SECURE_PORT} --volume {work_dir}:{_KEY_MOUNT_DIR}:ro "
            f"{_APISERVER_IMAGE} kube-apiserver {_apiserver_flags(etcd_name=etcd_name, watch_cache=watch_cache)}"
        )

        return ApiserverEnvironment(
            host_port=host_port,
            token=_TOKEN,
            network_name=network_name,
            etcd_name=etcd_name,
            apiserver_name=apiserver_name,
        )
    except BaseException:
        _remove_environment_idempotently(apiserver_name=apiserver_name, etcd_name=etcd_name, network_name=network_name)
        raise


def _pull_image(image: str) -> None:
    # [FUNCTION DOC] HELPER: Pulls a docker image quietly, retrying on a failed command up to 4
    # attempts in total; after a failed attempt other than the last, logs a warning and sleeps
    # 5 seconds times the attempt number. Re-raises the final command error.
    for attempt in range(1, _IMAGE_PULL_ATTEMPTS + 1):
        try:
            run_shell_command(f"docker pull --quiet {image}")
            return
        except subprocess.CalledProcessError:
            if attempt == _IMAGE_PULL_ATTEMPTS:
                raise
            logger.warning(f"Pulling {image} failed on attempt {attempt}/{_IMAGE_PULL_ATTEMPTS}; retrying")
            time.sleep(_IMAGE_PULL_BACKOFF_SECONDS * attempt)


def stop_apiserver(environment: ApiserverEnvironment) -> None:
    # [FUNCTION DOC] HELPER: Removes the environment's containers (with their volumes) and docker
    # network; never raises, failures are only logged.
    _remove_environment_idempotently(
        apiserver_name=environment.apiserver_name,
        etcd_name=environment.etcd_name,
        network_name=environment.network_name,
    )


def _remove_environment_idempotently(*, apiserver_name: str, etcd_name: str, network_name: str) -> None:
    # [FUNCTION DOC] HELPER: Force-removes the API server and etcd containers with their volumes,
    # then removes the network; each command's failure is logged as an error (with traceback) and
    # swallowed, so it is safe to call when pieces were never created.
    for command in (
        f"docker rm --force --volumes {apiserver_name} {etcd_name}",
        f"docker network rm {network_name}",
    ):
        try:
            run_shell_command(command)
        except Exception:
            logger.error(f"apiserver environment cleanup command failed {command=}", exc_info=True)


def log_apiserver_diagnostics(environment: ApiserverEnvironment) -> None:
    # [FUNCTION DOC] HELPER: Logs, at error level, the last 50 lines (stdout and stderr merged) of
    # the API server container and then the etcd container. Raises if a docker logs command fails.
    for container in (environment.apiserver_name, environment.etcd_name):
        logs = run_shell_command(f"docker logs --tail 50 {container} 2>&1", capture_output=True)
        logger.error(f"apiserver environment diagnostics {container=}\n{logs}")


def apiserver_started_at(environment: ApiserverEnvironment) -> str:
    # [FUNCTION DOC] HELPER: Returns the API server container's last start timestamp string as
    # reported by docker inspect, stripped of whitespace (changes after a restart). Asserts the
    # command returned output.
    started_at = run_shell_command(
        f"docker inspect --format {{{{.State.StartedAt}}}} {environment.apiserver_name}", capture_output=True
    )
    assert started_at is not None, f"docker inspect returned nothing for {environment.apiserver_name=}"
    return started_at.strip()


def restart_apiserver(environment: ApiserverEnvironment) -> None:
    # [FUNCTION DOC] HELPER: Restarts the API server container (etcd keeps its data); does not wait
    # for it to serve again.
    run_shell_command(f"docker restart {environment.apiserver_name}")


def compact_etcd_to_head(environment: ApiserverEnvironment) -> None:
    # [FUNCTION DOC] HELPER: Compacts etcd history up to its current revision, physically, so older
    # resource versions can no longer be watched from. Reads the endpoint status as JSON via the
    # etcd container, takes the header revision of the first endpoint, then runs the compact
    # command with the physical flag. Asserts the status command returned output.
    status = run_shell_command(
        f"docker exec {environment.etcd_name} etcdctl endpoint status --write-out=json", capture_output=True
    )
    assert status is not None, f"etcdctl returned nothing for {environment.etcd_name=}"
    revision = json.loads(status)[0]["Status"]["header"]["revision"]
    run_shell_command(f"docker exec {environment.etcd_name} etcdctl compact {revision} --physical")


def _apiserver_flags(*, etcd_name: str, watch_cache: bool) -> str:
    # [FUNCTION DOC] HELPER: Builds the API server command-line flag string: etcd servers URL by
    # container name on port 2379; secure port 6443 bound on all addresses with advertise address
    # loopback; certificate directory under the temp directory (self-signed certs generated at
    # start); allow-all authorization; static token file from the mounted directory; the
    # service-account admission plugin disabled; service cluster IP range 10.0.0.0/24; service
    # account issuer, verification key and signing key set to the mounted key; plus a flag
    # disabling the watch cache when requested. Flags are joined by spaces.
    service_account_key = f"{_KEY_MOUNT_DIR}/{_SERVICE_ACCOUNT_KEY}"
    flags = [
        f"--etcd-servers=http://{etcd_name}:2379",
        f"--secure-port={_SECURE_PORT}",
        "--bind-address=0.0.0.0",
        "--advertise-address=127.0.0.1",
        "--cert-dir=/tmp/certs",
        "--authorization-mode=AlwaysAllow",
        f"--token-auth-file={_KEY_MOUNT_DIR}/{_TOKEN_FILE}",
        "--disable-admission-plugins=ServiceAccount",
        "--service-cluster-ip-range=10.0.0.0/24",
        "--service-account-issuer=https://kubernetes.default.svc",
        f"--service-account-key-file={service_account_key}",
        f"--service-account-signing-key-file={service_account_key}",
    ]
    if not watch_cache:
        flags.append("--watch-cache=false")
    return " ".join(flags)


def _free_host_port() -> int:
    # [FUNCTION DOC] HELPER: Asks the operating system for a free loopback TCP port by binding to
    # port 0, reads the port, closes the socket, and returns it. Inherently racy: another process
    # could take the port before docker publishes it.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])
