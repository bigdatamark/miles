# [MODULE DOC]
# Tests (mostly rendering the run chart with helm) of colocated training and inference on the same
# nodes. With a colocate configuration the chart gates inference engine pods from scheduling, shares
# the host IPC namespace, exposes all GPUs but requests none, and injects the base GPU id variable
# from a pod annotation; the trainer pool shares host IPC but is scheduled normally; a pairing
# controller deployment is rendered with narrow pod rights. Also checks disaggregated runs render
# ordinary pools, pools missing from the colocate config are ordinary, platform-owned environment
# variable names are rejected from values, the reserved-name lists agree, and the launcher, chart
# and controller agree on the base GPU id variable and annotation spelling.
import json
from typing import Any

import pytest
from tests.fast.charts.utils import (
    NAMESPACE,
    RUN_RELEASE_NAME,
    named_object,
    objects_of_kind,
    render_run,
    render_run_error,
    requires_helm,
    with_object_names,
)
from tests.fast.utils.external_utils.command_utils.helm_backend.launcher.values import utils as values_utils

from miles.utils.external_utils.colocate_pairing.pods import _GATE_NAME, release_patch
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    _PLATFORM_OWNED_ENV_VARS,
)
from miles.utils.workers.env_vars import (
    BASE_GPU_ID_ENV_VAR,
    CELL_INDEX_ENV_VAR,
    NAMESPACE_ENV_VAR,
    PLATFORM_IDENTITY_ENV_VARS,
    POD_INDEX_ENV_VAR,
    RELEASE_ENV_VAR,
    SUBPROCESS_INDEX_ENV_VAR,
)
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

# Module data: ENGINES are two inference pools (engine: two replicas of four pods with eight GPUs
# limit; decode: two replicas of two pods); TRAINERS is one trainer pool (three replicas of two
# pods); POOLS are the helm arguments setting both; PAIRING_CONFIG is a colocate configuration
# naming both inference pools (decode offset 32 GPUs) against the trainer pool; ENABLE is POOLS plus
# that configuration; GATE is the expected scheduling gate entry.
ENGINES = [
    {
        "name": "engine",
        "replicas": 2,
        "size": 4,
        "command": ["python"],
        "resources": {"limits": {"nvidia.com/gpu": 8}},
    },
    {"name": "decode", "replicas": 2, "size": 2, "command": ["python"]},
]
TRAINERS = [{"name": "trainer", "replicas": 3, "size": 2, "command": ["python"]}]

POOLS = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(with_object_names(ENGINES))}",
    "--set-json",
    f"run.trainerEngines={json.dumps(with_object_names(TRAINERS))}",
)

PAIRING = "myrun-miles-run-colocate-pairing"
ORCHESTRATOR_ROLE = "myrun-miles-run-platform-read-delete"


def layout(
    *,
    num_inference_cells: int,
    num_pods_per_inference_cell: int,
    gpu_offset: int,
    num_gpus_per_inference_pod: int = 8,
) -> dict[str, int]:
    # [FUNCTION DOC] HELPER: Builds a pool layout mapping for the colocate configuration from the
    # given inference cell count, pods per inference cell, GPU offset and GPUs per inference pod
    # (default 8), with fixed trainer values: three trainer cells, two pods per trainer cell and
    # eight GPUs per node.
    return {
        "num_inference_cells": num_inference_cells,
        "num_trainer_cells": 3,
        "num_pods_per_inference_cell": num_pods_per_inference_cell,
        "num_pods_per_trainer_cell": 2,
        "num_gpus_per_node": 8,
        "num_gpus_per_inference_pod": num_gpus_per_inference_pod,
        "gpu_offset": gpu_offset,
    }


PAIRING_CONFIG = {
    "namespace": NAMESPACE,
    "release": RUN_RELEASE_NAME,
    "trainer_pool_id": "trainer",
    "inference_pools": [
        {
            "pool_id": "engine",
            "layout": layout(num_inference_cells=2, num_pods_per_inference_cell=2, gpu_offset=0),
        },
        {
            "pool_id": "decode",
            "layout": layout(num_inference_cells=2, num_pods_per_inference_cell=2, gpu_offset=32),
        },
    ],
}

ENABLE = (*POOLS, "--set-json", f"run.colocate={json.dumps(PAIRING_CONFIG)}")

GATE = {"name": _GATE_NAME}


def pool_pod(objects: list[dict[str, Any]], name: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the worker pod spec from the named leader-worker-set object's
    # worker template.
    workload = named_object(objects, "LeaderWorkerSet", name)
    return workload["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]


def colocated_engine_pod(*args: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Renders the run chart in colocated mode with extra arguments and
    # returns the "engine" pool's worker pod spec.
    return pool_pod(render_run(*ENABLE, *args), "myrun-miles-run-engine")


def _env_names(container: dict[str, Any]) -> set[str]:
    # [FUNCTION DOC] HELPER: Returns the set of environment variable names of the container (empty
    # if it has no env).
    return {entry["name"] for entry in container.get("env", [])}


def _env_entry(container: dict[str, Any], name: str) -> dict[str, Any] | None:
    # [FUNCTION DOC] HELPER: Returns the first environment entry with the given name, or none.
    return next((entry for entry in container.get("env", []) if entry["name"] == name), None)


def pairing_config() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Renders colocated mode and returns the pairing controller's
    # configuration mapping parsed from the JSON argument following the config option in its first
    # container's command. Asserts the option sits at index three of the command.
    deployment = named_object(render_run(*ENABLE), "Deployment", PAIRING)
    command = deployment["spec"]["template"]["spec"]["containers"][0]["command"]
    assert command[3] == "--config", command
    return json.loads(command[4])


@requires_helm
class TestColocatedEnginePool:
    # [CLASS DOC] Tests of the engine pool pod spec in colocated mode (needs helm).
    def test_is_held_back_from_the_scheduler_by_the_chart_itself(self):
        # [FUNCTION DOC] The engine pod's scheduling gates equal exactly the pairing gate.
        """Nothing else can keep a pod unscheduled until another pod's node is known."""
        assert colocated_engine_pod()["schedulingGates"] == [GATE]

    def test_shares_the_host_ipc_namespace(self):
        # [FUNCTION DOC] The engine pod's host IPC flag is true.
        """A CUDA IPC handle's reference counter lives in shared memory, so both pods need the same one."""
        assert colocated_engine_pod()["hostIPC"] is True

    def test_sees_every_gpu_on_the_node_it_lands_on(self):
        # [FUNCTION DOC] The engine container's environment contains NVIDIA_VISIBLE_DEVICES with
        # value "all".
        """It requests no gpu of its own, so only the device plugin bypass makes the trainer's gpus visible."""
        assert {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"} in colocated_engine_pod()["containers"][0]["env"]

    def test_requests_no_gpus_of_its_own(self):
        # [FUNCTION DOC] The engine container's GPU resource limit is zero although the pool
        # declared eight.
        """The trainer requests the whole node, and two claims on one gpu would never both schedule."""
        assert colocated_engine_pod()["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 0

    def test_zeroes_a_gpu_request_the_pool_declared_for_itself(self):
        # [FUNCTION DOC] When the engine pool declares both a GPU limit and a GPU request of eight,
        # the rendered container resources have both limits and requests of zero.
        """A request left above the zeroed limit is rejected outright, so the engine never runs at all."""
        engines = [{**ENGINES[0], "resources": {"limits": {"nvidia.com/gpu": 8}, "requests": {"nvidia.com/gpu": 8}}}]
        pod = colocated_engine_pod("--set-json", f"run.inferenceEngines={json.dumps(with_object_names(engines))}")

        assert pod["containers"][0]["resources"] == {
            "limits": {"nvidia.com/gpu": 0},
            "requests": {"nvidia.com/gpu": 0},
        }

    def test_gives_a_second_colocated_pool_the_same_treatment(self):
        # [FUNCTION DOC] The decode pool also has the pairing gate, host IPC and a zero GPU limit.
        """A prefill/decode run gates both pool_ids, and one left ungated would race the trainer for its gpus."""
        pod = pool_pod(render_run(*ENABLE), "myrun-miles-run-decode")

        assert pod["schedulingGates"] == [GATE]
        assert pod["hostIPC"] is True
        assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == 0

    def test_is_told_which_card_of_the_node_the_pairing_gave_it(self):
        # [FUNCTION DOC] The base GPU id environment entry is sourced from the pod's own annotation
        # (field reference to the annotation key from the label keys defaults).
        """The controller writes that annotation in the patch that releases the gate, before the pod runs."""
        entry = _env_entry(colocated_engine_pod()["containers"][0], BASE_GPU_ID_ENV_VAR)

        assert entry == {
            "name": BASE_GPU_ID_ENV_VAR,
            "valueFrom": {
                "fieldRef": {"fieldPath": f"metadata.annotations['{DEFAULT_LABEL_KEYS.base_gpu_id_annotation}']"}
            },
        }

    def test_reads_that_card_off_the_pod_rather_than_a_rendered_value(self):
        # [FUNCTION DOC] The base GPU id entry has no literal value field.
        """helm renders one pod template for the whole pool, and the card differs from pod to pod."""
        entry = _env_entry(colocated_engine_pod()["containers"][0], BASE_GPU_ID_ENV_VAR)

        assert "value" not in entry

    def test_gives_a_second_colocated_pool_that_variable_too(self):
        # [FUNCTION DOC] The decode pool's container also has the base GPU id variable.
        """Prefill and decode may split one trainer node, and each needs to be told where it starts."""
        pod = pool_pod(render_run(*ENABLE), "myrun-miles-run-decode")

        assert BASE_GPU_ID_ENV_VAR in _env_names(pod["containers"][0])

    def test_carries_no_affinity_at_all_but_keeps_the_node_selector(self):
        # [FUNCTION DOC] With scheduling values holding a node selector and a node affinity, the
        # colocated engine pod has no affinity but keeps the node selector.
        """Any affinity would contradict the node the controller picks; the selector it only adds to."""
        pod = colocated_engine_pod(
            "--set-json", 'infra.scheduling={"nodeSelector":{"pool":"gpu"},"affinity":{"nodeAffinity":{}}}'
        )

        assert "affinity" not in pod
        assert pod["nodeSelector"] == {"pool": "gpu"}


@requires_helm
class TestColocatedTrainerPool:
    # [CLASS DOC] Tests of the trainer pool pod spec in colocated mode (needs helm).
    def test_shares_the_host_ipc_namespace_too(self):
        # [FUNCTION DOC] The trainer pod's host IPC flag is true.
        """The engine's CUDA IPC handles are only usable from a trainer in the same IPC namespace."""
        assert pool_pod(render_run(*ENABLE), "myrun-miles-run-trainer")["hostIPC"] is True

    def test_is_scheduled_normally_and_gets_none_of_the_engine_treatment(self):
        # [FUNCTION DOC] The trainer pod has no scheduling gates and its container has neither the
        # visible-devices variable nor the base GPU id variable.
        """It is the pod that claims the node, so gating it would leave nothing for the engine to pair with."""
        pod = pool_pod(render_run(*ENABLE), "myrun-miles-run-trainer")

        assert "schedulingGates" not in pod
        assert not _env_names(pod["containers"][0]) & {"NVIDIA_VISIBLE_DEVICES", BASE_GPU_ID_ENV_VAR}


@requires_helm
class TestDisaggregatedRun:
    # [CLASS DOC] Tests of a run with pools but no colocate configuration (needs helm).
    def test_leaves_the_engine_pool_ungated_and_holding_its_own_gpus(self):
        # [FUNCTION DOC] The engine pod has no scheduling gates, no host IPC key, neither colocate
        # environment variable, and its resources are exactly a GPU limit of eight.
        """The same pool_id values must render an ordinary engine when the run is not colocated."""
        pod = pool_pod(render_run(*POOLS), "myrun-miles-run-engine")

        assert "schedulingGates" not in pod
        assert "hostIPC" not in pod
        assert not _env_names(pod["containers"][0]) & {"NVIDIA_VISIBLE_DEVICES", BASE_GPU_ID_ENV_VAR}
        assert pod["containers"][0]["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_installs_no_pairing_controller(self):
        # [FUNCTION DOC] The only Role, RoleBinding and ServiceAccount are the orchestrator's
        # read-delete ones and no Deployment exists.
        """A run whose engines have their own nodes must not gain a controller with pod write rights."""
        objects = render_run(*POOLS)

        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "Role")] == [ORCHESTRATOR_ROLE]
        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "RoleBinding")] == [ORCHESTRATOR_ROLE]
        assert objects_of_kind(objects, "Deployment") == []
        assert [obj["metadata"]["name"] for obj in objects_of_kind(objects, "ServiceAccount")] == [ORCHESTRATOR_ROLE]


@requires_helm
class TestPairingController:
    # [CLASS DOC] Tests of the pairing controller deployment rendered in colocated mode (needs helm).
    def test_recreates_the_controller_instead_of_overlapping_generations(self):
        # [FUNCTION DOC] The deployment's update strategy is exactly the recreate type.
        """An upgrade must stop the stale pairing controller before its replacement starts."""
        deployment = named_object(render_run(*ENABLE), "Deployment", PAIRING)

        assert deployment["spec"]["strategy"] == {"type": "Recreate"}

    def test_holds_only_namespaced_rights_over_pods(self):
        # [FUNCTION DOC] The controller's namespaced role has exactly one rule: core group, pods,
        # verbs get, list, watch, patch and update.
        """Releasing a gate is an ordinary pod update, so nothing cluster-scoped is needed."""
        role = named_object(render_run(*ENABLE), "Role", PAIRING)

        assert role["rules"] == [
            {"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list", "watch", "patch", "update"]}
        ]

    def test_never_asks_for_the_binding_subresource(self):
        # [FUNCTION DOC] No resource in any role rule mentions a binding.
        """That belongs to the scheduler, and asking for it would make this a scheduler replacement."""
        rules = named_object(render_run(*ENABLE), "Role", PAIRING)["rules"]

        assert not any("binding" in resource for rule in rules for resource in rule["resources"])

    def test_runs_as_a_single_replica(self):
        # [FUNCTION DOC] The deployment has exactly one replica.
        """If it dies, new engine pods stay Pending, which is safe and visible; two would race each other."""
        assert named_object(render_run(*ENABLE), "Deployment", PAIRING)["spec"]["replicas"] == 1

    def test_is_handed_the_pairing_config_from_the_values_untouched(self):
        # [FUNCTION DOC] The configuration parsed from the controller's command equals the supplied
        # colocate configuration exactly.
        """The launcher already knows every pool's shape, so a chart that rebuilt it could only get it wrong."""
        assert pairing_config() == PAIRING_CONFIG

    def test_carries_the_cluster_environment(self):
        # [FUNCTION DOC] A proxy environment variable set through infra values appears in the
        # controller container's environment.
        """It talks to the api server, so a cluster that needs a proxy needs it here as well."""
        objects = render_run(*ENABLE, "--set", "infra.env.HTTP_PROXY=http://proxy:7890")
        deployment = named_object(objects, "Deployment", PAIRING)

        assert {"name": "HTTP_PROXY", "value": "http://proxy:7890"} in (
            deployment["spec"]["template"]["spec"]["containers"][0]["env"]
        )


@requires_helm
class TestAPoolTheConfigDoesNotName:
    # [CLASS DOC] Test of a pool absent from the colocate configuration (needs helm).
    def test_renders_as_an_ordinary_pool(self):
        # [FUNCTION DOC] With the colocate configuration naming only the engine pool, the decode
        # pool has no scheduling gates, no host IPC key and no base GPU id variable.
        """The config names object names the launcher derived, so anything else is simply not colocated."""
        config = {**PAIRING_CONFIG, "inference_pools": PAIRING_CONFIG["inference_pools"][:1]}
        pod = pool_pod(
            render_run(*POOLS, "--set-json", f"run.colocate={json.dumps(config)}"), "myrun-miles-run-decode"
        )

        assert "schedulingGates" not in pod
        assert "hostIPC" not in pod
        assert BASE_GPU_ID_ENV_VAR not in _env_names(pod["containers"][0])


@requires_helm
class TestTheVariablesThePlatformOwns:
    # [CLASS DOC] Tests of reserved platform environment variable names (needs helm).
    @pytest.mark.parametrize(
        "name", [CELL_INDEX_ENV_VAR, POD_INDEX_ENV_VAR, BASE_GPU_ID_ENV_VAR, NAMESPACE_ENV_VAR, RELEASE_ENV_VAR]
    )
    @pytest.mark.parametrize("section", ["infra", "run"])
    def test_the_schema_refuses_one_in_a_values_environment(self, section: str, name: str):
        # [FUNCTION DOC] Parametrized over the infra and run sections and the five reserved names
        # (cell index, pod index, base GPU id, namespace, release). Setting the variable in that
        # section's env makes rendering fail with an error message that mentions the name.
        """Kubernetes keeps the last entry of a name, and these render first, so a values entry wins silently."""
        error = render_run_error("--set", f"{section}.env.{name}=anything")

        assert name in error

    def test_the_release_and_namespace_a_pod_is_told_are_the_ones_it_runs_in(self):
        # [FUNCTION DOC] The decode pool container's namespace and release variables equal the test
        # namespace and the run release name.
        """They are how a pod selects its own workers, so an entry pointing elsewhere reaches someone else's run."""
        container = pool_pod(render_run(*ENABLE), "myrun-miles-run-decode")["containers"][0]
        env = {entry["name"]: entry.get("value") for entry in container["env"]}

        assert env[NAMESPACE_ENV_VAR] == NAMESPACE
        assert env[RELEASE_ENV_VAR] == RUN_RELEASE_NAME


class TestTheTwoListsOfPlatformVariables:
    # [CLASS DOC] Static test comparing the schema's reserved variable list to the worker identity list.
    def test_the_reserved_names_and_the_identity_a_worker_reports_differ_only_where_meant_to(self):
        # [FUNCTION DOC] Reserved-minus-identity equals exactly the path variable, base GPU id,
        # namespace and release names; identity-minus-reserved equals exactly the subprocess index.
        """Two hand-kept lists of platform variables drift apart quietly, so the difference is what to state."""
        reserved = set(_PLATFORM_OWNED_ENV_VARS)
        identity = set(PLATFORM_IDENTITY_ENV_VARS)

        assert reserved - identity == {"PYTHONPATH", BASE_GPU_ID_ENV_VAR, NAMESPACE_ENV_VAR, RELEASE_ENV_VAR}
        assert identity - reserved == {SUBPROCESS_INDEX_ENV_VAR}


@requires_helm
class TestTheNamesTheChartWritesAreTheNamesTheSchemaReserves:
    # [CLASS DOC] Test that the chart writes exactly the reserved variable names (needs helm).
    def test_a_gated_pool_pod_is_given_the_reserved_names_and_nothing_else_of_its_own(self):
        # [FUNCTION DOC] The engine container's environment names, minus the visible-devices
        # variable, equal exactly the reserved variable set.
        """Reserving a name is only worth something if the list covers every name the chart writes itself."""
        container = pool_pod(render_run(*ENABLE), "myrun-miles-run-engine")["containers"][0]

        assert _env_names(container) - {"NVIDIA_VISIBLE_DEVICES"} == set(_PLATFORM_OWNED_ENV_VARS)


def _sub_node_engine_argv() -> list[str]:
    # [FUNCTION DOC] HELPER: Uses the launcher's values builder with a colocated plan (copied from
    # the default test layout), two engine cells of four GPUs and one trainer cell of eight GPUs,
    # and returns the command list of the first inference engine in the built run values.
    specs = [values_utils.engine(num_cells=2, gpus_per_engine=4), values_utils.trainer(num_cells=1, gpus_per_cell=8)]
    plan = values_utils.LAYOUT.model_copy(update={"colocate": True})
    return build_values(specs, plan).as_values()["run"]["inferenceEngines"][0]["command"]


@requires_helm
class TestTheCardReachesTheEngineByOneNameAndOneKey:
    # [CLASS DOC] Cross-component naming test (needs helm).
    def test_the_launcher_the_chart_and_the_controller_all_spell_it_the_same_way(self):
        # [FUNCTION DOC] Collects the launcher-built engine command, the chart's base GPU id
        # environment entry and the controller's gate release patch. Asserts the command argument
        # after the base GPU id option is the kubelet variable reference to that variable name; the
        # chart's field reference path names the annotation key; and the only metadata path in the
        # release patch is that annotation key with slashes escaped as the JSON pointer sequence.
        """Three producers of two strings: drift in any one leaves the engine reading an empty base gpu id."""
        argv = _sub_node_engine_argv()
        env = _env_entry(colocated_engine_pod()["containers"][0], BASE_GPU_ID_ENV_VAR)
        patch = release_patch(node_name="gpu-3", base_gpu_id=4, gates=[_GATE_NAME], annotations={"a": "b"})
        annotation = DEFAULT_LABEL_KEYS.base_gpu_id_annotation

        assert argv[argv.index("--base-gpu-id") + 1] == f"$({BASE_GPU_ID_ENV_VAR})"
        assert env["valueFrom"]["fieldRef"]["fieldPath"] == f"metadata.annotations['{annotation}']"
        assert [operation["path"] for operation in patch if operation["path"].startswith("/metadata")] == [
            f"/metadata/annotations/{annotation.replace('/', '~1')}"
        ]


# Module data for sub-node engines: one engine pool of two single-pod cells with four GPUs each and
# the launcher-built command; a matching colocate configuration; and the combined helm arguments.
SUB_NODE_ENGINES = [
    {
        "name": "engine",
        "replicas": 2,
        "size": 1,
        "command": _sub_node_engine_argv(),
        "resources": {"limits": {"nvidia.com/gpu": 4}},
    }
]

SUB_NODE_PAIRING_CONFIG = {
    "namespace": NAMESPACE,
    "release": RUN_RELEASE_NAME,
    "trainer_pool_id": "trainer",
    "inference_pools": [
        {
            "pool_id": "engine",
            "layout": layout(
                num_inference_cells=2,
                num_pods_per_inference_cell=1,
                gpu_offset=0,
                num_gpus_per_inference_pod=4,
            ),
        }
    ],
}

SUB_NODE_ENABLE = (
    "--set-json",
    f"run.inferenceEngines={json.dumps(with_object_names(SUB_NODE_ENGINES))}",
    "--set-json",
    f"run.trainerEngines={json.dumps(with_object_names(TRAINERS))}",
    "--set-json",
    f"run.colocate={json.dumps(SUB_NODE_PAIRING_CONFIG)}",
)


def _pools_by_name(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Maps each leader-worker-set object's name to its worker pod spec.
    return {
        pool["metadata"]["name"]: pool["spec"]["leaderWorkerTemplate"]["workerTemplate"]["spec"]
        for pool in objects_of_kind(objects, "LeaderWorkerSet")
    }


@requires_helm
class TestTheKubeletVariableAndTheGateAlwaysTravelTogether:
    # [CLASS DOC] Tests of the sub-node engine rendering (needs helm).
    def test_the_pod_asking_for_the_card_is_the_pod_that_is_told_it(self):
        # [FUNCTION DOC] The sub-node engine container's command contains the kubelet variable
        # reference to the base GPU id and its environment defines that variable.
        """The launcher writes the argument and the chart writes the variable, from two unrelated inputs."""
        container = _pools_by_name(render_run(*SUB_NODE_ENABLE))["myrun-miles-run-engine"]["containers"][0]

        assert f"$({BASE_GPU_ID_ENV_VAR})" in container["command"]
        assert BASE_GPU_ID_ENV_VAR in _env_names(container)

    def test_every_pool_asking_for_the_card_is_a_pool_the_chart_gates(self):
        # [FUNCTION DOC] Computes the set of pools whose command references the variable and the
        # set with scheduling gates; asserts the first is non-empty and a subset of the second.
        """An ungated pod runs before the controller annotates it, so the variable expands to nothing."""
        pools = _pools_by_name(render_run(*SUB_NODE_ENABLE))
        asking = {
            name for name, pod in pools.items() if f"$({BASE_GPU_ID_ENV_VAR})" in pod["containers"][0]["command"]
        }
        gated = {name for name, pod in pools.items() if pod.get("schedulingGates")}

        assert asking
        assert asking <= gated
