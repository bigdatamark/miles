# [MODULE DOC]
# Tests (helm rendering of the run chart) of the optional Mooncake master: nothing is rendered by
# default; when enabled, a one-replica stateful set and a headless-style service agree on name,
# selector, service name and the configured RPC and metrics ports, with the master command carrying
# both ports; and the master inherits configured resources, cluster environment, and volumes and
# mounts.
from typing import Any

from tests.fast.charts.utils import (
    RUN_RELEASE_NAME,
    host_path_volume,
    objects_of_kind,
    pod_spec_of,
    render_run,
    requires_helm,
    sole_container_of,
    volumes_args,
)

MOONCAKE_MASTER_NAME = f"{RUN_RELEASE_NAME}-miles-run-mooncake-master"


def _mooncake_object(objects: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the single object of the kind named as the Mooncake master;
    # asserts exactly one matches.
    matches = [obj for obj in objects_of_kind(objects, kind) if obj["metadata"]["name"] == MOONCAKE_MASTER_NAME]
    assert len(matches) == 1
    return matches[0]


@requires_helm
class TestMooncakeMaster:
    # [CLASS DOC] Tests of the Mooncake master objects (need helm).
    def test_a_run_that_does_not_need_mooncake_renders_no_master(self) -> None:
        # [FUNCTION DOC] A default run has no Service or StatefulSet named as the master.
        """The default run must not launch or expose a Mooncake master that none of its workers use."""
        objects = render_run()

        assert not any(
            obj["kind"] in {"Service", "StatefulSet"} and obj["metadata"]["name"] == MOONCAKE_MASTER_NAME
            for obj in objects
        )

    def test_the_service_and_master_process_share_names_selectors_and_ports(self) -> None:
        # [FUNCTION DOC] Enables Mooncake with RPC port 51051 and metrics port 51052. Asserts the
        # service and stateful set share a name; the service selector equals the stateful set's
        # match labels and is contained in its pod labels; the stateful set's service name is the
        # service name; replicas is one; the service ports and the container ports both equal the
        # RPC and metrics port mapping; and the container command is the master binary with its
        # RPC port and metrics port options.
        """The singleton master and its Service must agree on identity, selection, and both configured ports."""
        rpc_port = 51051
        metrics_port = 51052
        objects = render_run(
            "--set",
            "run.mooncake.enabled=true",
            "--set",
            f"run.mooncake.rpcPort={rpc_port}",
            "--set",
            f"run.mooncake.metricsPort={metrics_port}",
        )
        service = _mooncake_object(objects, "Service")
        stateful_set = _mooncake_object(objects, "StatefulSet")
        container = stateful_set["spec"]["template"]["spec"]["containers"][0]
        service_ports = {port["name"]: port["port"] for port in service["spec"]["ports"]}
        container_ports = {port["name"]: port["containerPort"] for port in container["ports"]}

        assert service["metadata"]["name"] == stateful_set["metadata"]["name"]
        assert service["spec"]["selector"] == stateful_set["spec"]["selector"]["matchLabels"]
        assert service["spec"]["selector"].items() <= stateful_set["spec"]["template"]["metadata"]["labels"].items()
        assert stateful_set["spec"]["serviceName"] == service["metadata"]["name"]
        assert stateful_set["spec"]["replicas"] == 1
        assert service_ports == container_ports == {"rpc": rpc_port, "metrics": metrics_port}
        assert container["command"] == [
            "mooncake_master",
            "--rpc_port",
            str(rpc_port),
            "--metrics_port",
            str(metrics_port),
        ]

    def test_the_master_receives_resources_environment_and_configured_volumes(self) -> None:
        # [FUNCTION DOC] Enables Mooncake with a CPU request of 3, a proxy environment variable and
        # two host-path volumes (default storage and a read-only model cache). Asserts the master
        # container has the CPU request, the proxy variable and the read-only model cache mount, and
        # the pod has the model cache host-path volume.
        """An enabled master inherits its configured resources and the cluster runtime contract."""
        objects = render_run(
            "--set",
            "run.mooncake.enabled=true",
            "--set",
            "run.mooncake.resources.requests.cpu=3",
            "--set",
            "infra.env.HTTP_PROXY=http://proxy:7890",
            *volumes_args(
                host_path_volume(),
                host_path_volume(
                    name="model-cache",
                    path="/models",
                    mounts=[{"mountPath": "/root/.cache/models", "readOnly": True}],
                ),
            ),
        )
        container = sole_container_of(objects, "StatefulSet", MOONCAKE_MASTER_NAME)
        pod_spec = pod_spec_of(objects, "StatefulSet", MOONCAKE_MASTER_NAME)

        assert container["resources"]["requests"]["cpu"] == 3
        assert {"name": "HTTP_PROXY", "value": "http://proxy:7890"} in container["env"]
        assert {"name": "model-cache", "mountPath": "/root/.cache/models", "readOnly": True} in container[
            "volumeMounts"
        ]
        assert {"name": "model-cache", "hostPath": {"path": "/models", "type": "Directory"}} in pod_spec["volumes"]
