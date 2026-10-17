# [MODULE DOC]
# Ray-based wrapper around the Kimina Lean verification server used by the formal-math reward
# function. It starts one detached Ray actor per alive cluster node (pinned to that node); each
# actor launches a Kimina Lean server docker container (on a docker network named formal_math,
# with a randomised container name) and waits until its health endpoint answers. A client cluster
# object then round-robins Lean check requests across the servers using the asynchronous Kimina
# client. By default (environment flag, default on) any previous containers whose name starts with
# the auto-start prefix are stopped and removed on each node before starting a new one. Requires
# docker on every node, an existing docker network, and a running Ray cluster.
import datetime
import os
import random
import time

import ray
import requests
from kimina_client import AsyncKiminaClient, CheckResponse
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from miles.utils.external_utils.command_utils.common import run_shell_command
from miles.utils.misc import get_free_port

# TODO handle docker stop more gracefully later
_KILL_PREVIOUS_KIMINA_DOCKER = bool(int(os.environ.get("MILES_KILL_PREVIOUS_KIMINA_DOCKER", "1")))


class KiminaServerAndClientCluster:
    # [CLASS DOC]
    # Public facade that owns the server actors and the round-robin client cluster. Constructing it
    # blocks until every node's Kimina server is up (it starts the actors, then asks each for its
    # URL), and the servers are never shut down by this class.
    def __init__(self):
        # [FUNCTION DOC] Starts one server actor per alive node and builds the client cluster over
        # them. Blocking; side effects: Ray actors and docker containers on every node.
        self._servers = _create_actor_per_node(actor_cls=_KiminaServerActor)
        self._client_cluster = _KiminaClientCluster(self._servers)

    async def check(self, *args, **kwargs) -> CheckResponse:
        # [FUNCTION DOC] Forwards a Lean check request (arguments passed through unchanged to the
        # Kimina client's check method) to the next server in round-robin order and returns its
        # check response.
        return await self._client_cluster.check(*args, **kwargs)


class _KiminaClientCluster:
    # [CLASS DOC] HELPER: Holds one asynchronous Kimina client per server and a rotating index used
    # to spread requests evenly.
    def __init__(self, servers: list["_KiminaServerActor"]):
        # [FUNCTION DOC] HELPER: For each server actor, synchronously fetches its API URL through
        # Ray (blocking) and creates a client for it; the rotation index starts at zero.
        self._clients = [AsyncKiminaClient(api_url=ray.get(server.get_api_url.remote())) for server in servers]
        self._next_client_index = 0

    async def check(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Picks the client at the current index, advances the index
        # (wrapping around) before awaiting, and returns the client's check result for the given
        # arguments.
        client = self._clients[self._next_client_index]
        self._next_client_index = (self._next_client_index + 1) % len(self._clients)
        return await client.check(*args, **kwargs)


def _create_actor_per_node(actor_cls) -> list:
    # [FUNCTION DOC] HELPER: Lists Ray nodes, keeps those marked alive (assertion error if none)
    # and creates one instance of the given remote actor class on each, with no actor name, a
    # detached lifetime (it outlives the creating job), a hard node affinity to that node and a
    # negligible CPU reservation. Returns the list of actor handles without waiting for them to
    # become ready.
    # for simplicity, we use all available nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    assert len(nodes) > 0

    actors = []
    for node in nodes:
        actors.append(
            actor_cls.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
                num_cpus=0.001,
            ).remote()
        )

    return actors


@ray.remote
class _KiminaServerActor:
    # [CLASS DOC] HELPER: Remote actor managing one Kimina Lean server docker container on its node.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Picks a free host port, optionally removes previous auto-started
        # containers, starts a new container, and blocks until its health endpoint returns success.
        # Note the readiness check uses the container-name based URL.
        self.port = get_free_port()

        if _KILL_PREVIOUS_KIMINA_DOCKER:
            _docker_stop_all()

        self.docker_name = _docker_start(port=self.port)
        _wait_server_ready(base_url=self.get_api_url())

    def get_api_url(self):
        # [FUNCTION DOC] HELPER: Returns the server base URL as http, the container name and port
        # 8000 (the in-container port, reachable by name on the shared docker network).
        # KNOWN ISSUE: the published host port chosen at startup is never used by the URL, so the
        # URL only works from processes attached to the same docker network.
        return f"http://{self.docker_name}:8000"


def _docker_start(port: int):
    # [FUNCTION DOC] HELPER: Starts a detached container of the pinned Kimina Lean server image
    # (version 2.0.0) with a unique name (auto prefix, timestamp and a random number up to one
    # million), restart policy unless-stopped, attached to the formal_math docker network, host
    # port mapped to container port 8000. Runs through the shell command helper (failure raises).
    # Returns the container name.
    docker_name = (
        f"kimina_lean_server_auto_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(0, 1000000)}"
    )
    run_shell_command(
        "docker run "
        "-d "
        f"--name {docker_name} "
        "--restart unless-stopped "
        "--network formal_math "
        # "--env-file .env "  # do not use env yet
        f"-p {port}:8000 "
        f"projectnumina/kimina-lean-server:2.0.0"
    )
    return docker_name


def _wait_server_ready(base_url: str):
    # [FUNCTION DOC] HELPER: Polls the health path under the base URL every two seconds until it
    # answers with HTTP 200; connection errors are ignored and a waiting message is printed on each
    # failed poll. Returns nothing.
    # KNOWN ISSUE: there is no overall timeout, so a server that never starts blocks forever.
    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health")
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            print(f"Wait kimina server ready ({base_url})...")
            time.sleep(2)


def _docker_stop_all():
    # [FUNCTION DOC] HELPER: Shell step that lists all containers (running or not) whose name
    # contains the auto-start prefix, stops and removes them if any, and always succeeds. Affects
    # every such container on the node, including ones started by other jobs.
    run_shell_command(
        'ids=$(docker ps -a --filter "name=kimina_lean_server_auto" -q); '
        '[ -n "$ids" ] && docker stop $ids && docker rm $ids; '
        "true"
    )
