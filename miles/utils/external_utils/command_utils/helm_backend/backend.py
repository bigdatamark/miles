# [MODULE DOC] The Kubernetes command backend. Training launches go to the launcher entrypoint;
# one-off commands run as Kubernetes jobs (one completion per node) through the command-job
# module, after checking that the chart's scheduling constraints can place the requested number of
# nodes. Also computes the in-cluster host name of the run's api server.
from __future__ import annotations

from pathlib import Path

from miles.utils.external_utils.command_utils.base_backend import (
    BaseCommandBackend,
    ExecuteTrainConfig,
    ExecuteTrainRequest,
)
from miles.utils.external_utils.command_utils.common import chart_dir, repo_base_dir
from miles.utils.external_utils.command_utils.helm_backend import command_job
from miles.utils.external_utils.command_utils.helm_backend.launcher import entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import Scheduling
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import InfraInfo
from miles.utils.external_utils.command_utils.helm_backend.naming import ReleaseName, RunNames

_HOSTNAME_LABEL = "kubernetes.io/hostname"


class KubernetesCommandBackend(BaseCommandBackend):
    # [CLASS DOC] Backend implementing the protected hooks of the base backend for Kubernetes.
    def _execute_train_inner(self, *, request: ExecuteTrainRequest, config: ExecuteTrainConfig) -> None:
        # [FUNCTION DOC] Delegates the whole launch to the launcher entrypoint's train function.
        entrypoint.execute_train(request=request, config=config)

    def _exec_command_gpu_inner(
        self, cmd: str, capture_output: bool = False, num_gpus_per_node: int | None = None
    ) -> str | None:
        # [FUNCTION DOC] Runs the command as a single-node job via the public multi-node method
        # (node count 1) and returns the first (only) entry.
        return self.exec_command_multi_node(
            cmd, capture_output=capture_output, num_nodes=1, num_gpus_per_node=num_gpus_per_node
        )[0]

    def _exec_command_multi_node_inner(
        self,
        cmd: str,
        capture_output: bool = False,
        num_nodes: int | None = None,
        num_gpus_per_node: int | None = None,
    ) -> list[str | None]:
        # [FUNCTION DOC] Asserts a namespace is configured and the node count is given (it cannot be
        # inferred on Kubernetes). Checks the requested nodes are schedulable for the chart, then
        # runs the command as a job with one completion per node, GPUs per pod defaulting to 1 when
        # unspecified, the configured helm values files and step name "command"; returns the
        # per-node outputs (captured logs or none).
        assert self.config.namespace, "Set CommandUtilConfig.namespace to run a command somewhere"
        assert (
            num_nodes is not None
        ), "kubernetes cannot infer num_nodes from a live cluster the way ray does; pass num_nodes explicitly"
        chart = chart_dir(repo_base_dir=repo_base_dir)
        self._assert_requested_nodes_schedulable(num_nodes=num_nodes, chart=chart)
        return command_job.run_on_nodes(
            command_job.CommandJobContext(
                namespace=self.config.namespace,
                chart_dir=chart,
                helm_values_files=tuple(self.config.helm_values),
                gpus_per_node=num_gpus_per_node if num_gpus_per_node is not None else 1,
            ),
            cmd,
            capture_output=capture_output,
            completions=num_nodes,
            step="command",
        )

    def _assert_requested_nodes_schedulable(self, *, num_nodes: int, chart: Path) -> None:
        # [FUNCTION DOC] HELPER: Does nothing for one node or fewer. Otherwise loads the chart's
        # infrastructure values with the configured values files. Raises an assertion error if the
        # node selector pins the deployment to a single host name (extra nodes would stay pending
        # forever). Also asserts that, when required node affinity restricts host names to a finite
        # set, that set has at least the requested number of hosts.
        if num_nodes <= 1:
            return

        infra = InfraInfo.load(chart, list(self.config.helm_values))
        scheduling = infra.scheduling
        node_selector = (scheduling.node_selector if scheduling is not None else None) or {}
        if (hostname := node_selector.get(_HOSTNAME_LABEL)) is not None:
            raise AssertionError(
                f"this command asks for {num_nodes} nodes, and infra.scheduling.nodeSelector pins every pod of this "
                f"deployment to {hostname!r}, so every completion after the first would stay Pending for good while "
                f"the first one holds its gpus until the job times out; ask for one node, or unpin the deployment"
            )

        hosts = _required_affinity_hostnames(scheduling)
        assert hosts is None or len(hosts) >= num_nodes, (
            f"this command asks for {num_nodes} nodes, but required node affinity permits at most "
            f"{len(hosts)} hostnames: {sorted(hosts)}; ask for fewer nodes, or unpin the deployment"
        )

    def api_server_host(self, config: ExecuteTrainConfig) -> str:
        # [FUNCTION DOC] Asserts the config has a run id and namespace, and that the deploy
        # component is not a split deployment (split runs have no api server). Returns the
        # in-cluster host name of the orchestrator pod, computed from the serialized release name
        # (run id, deploy component, deploy instance id) and the namespace.
        assert config.run_id and config.namespace, (
            "The api server of a kubernetes run answers on the orchestrator's pod, which is named after the "
            "release; set the launch config's run_id and namespace before asking where that pod is"
        )
        assert not config.deploy_component.is_split(), (
            f"The api server, and the mini ft controller polling it, answer for the cells of their own deployment, "
            f"so a split run is refused one (--api-server-port 0) and nothing listens on the "
            f"{config.deploy_component.value} deployment for this host to name"
        )
        return RunNames.orchestrator_host(
            release=ReleaseName(
                run_id=config.run_id,
                deploy_component=config.deploy_component,
                deploy_instance_id=config.deploy_instance_id,
            ).serialize(),
            namespace=config.namespace,
        )


def _required_affinity_hostnames(scheduling: Scheduling | None) -> set[str] | None:
    # [FUNCTION DOC] HELPER: Returns the set of host names that required node affinity allows, or
    # none if no host restriction applies. None when there is no scheduling, no affinity, no node
    # affinity requirement, or any selector term lacks an "In" expression on the hostname label
    # (an unrestricted term means any host). Otherwise the union over non-empty terms of the
    # intersection of that term's hostname "In" value sets. Empty terms are skipped.
    if scheduling is None or not scheduling.affinity:
        return None
    affinity = scheduling.affinity.get("nodeAffinity") or {}
    required = affinity.get("requiredDuringSchedulingIgnoredDuringExecution")
    if required is None:
        return None

    allowed: set[str] = set()
    for term in required.get("nodeSelectorTerms", []):
        if not term:
            continue
        bounds = [
            set(expression.get("values", []))
            for expression in term.get("matchExpressions", [])
            if expression.get("key") == _HOSTNAME_LABEL and expression.get("operator") == "In"
        ]
        if not bounds:
            return None
        allowed.update(set.intersection(*bounds))
    return allowed
