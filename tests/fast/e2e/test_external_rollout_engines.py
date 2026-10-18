# [MODULE DOC]
# Tests for how the external-rollout end-to-end script (loaded as a module) describes its external
# sglang engines for each cluster backend. For Ray the script starts engines itself beside the
# trainer (prepare commands, loopback addresses, no extra manifests). For Kubernetes it renders one
# manifest bundle (a headless service plus a parallel-start stateful set) from the cluster's infra
# values: image, pull policy and secrets, volumes and mounts, scheduling, environment, gpu limits,
# readiness probe and no hardcoded namespace; the addresses given to the run must match the
# stateful set's pod dns names. Also covers object-store arguments per backend, the train args'
# rollout gpu count, and merging of several infra values files the way helm merges them. One
# chart-render test runs only when helm is available. Helpers build infra mappings, render
# manifests, and write values files to a temp path.
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.fast.charts.utils import documents_of, objects_added_by, requires_helm
from tests.fast.e2e.external_rollout_script import load_external_rollout_script

from miles.utils.external_utils.command_utils import ExecuteTrainConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import InfraValues
from miles.utils.workers.types import ClusterBackend

script = load_external_rollout_script()

HOST_PATH_INFRA: dict[str, Any] = {
    "image": {"repository": "radixark/miles", "tag": "dev"},
    "volumes": [
        {
            "name": "cluster-storage",
            "hostPath": {"path": "/cluster-storage"},
            "mounts": [{"mountPath": "/cluster-storage"}],
        }
    ],
}


def infra_with(**overrides: Any) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a new shallow-merged mapping of the host-path infra values
    # with the given top-level keys replaced.
    return {**HOST_PATH_INFRA, **overrides}


def manifests_of(infra: dict[str, Any]) -> str:
    # [FUNCTION DOC] HELPER: Validates the mapping as infra values and renders the script's engine
    # manifest text (multi-document yaml) under the script's engine object name.
    return script._engine_manifests(InfraValues.model_validate(infra), object_name=script.ENGINE_OBJECT_NAME)


def stateful_set_of(infra: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the second document of the rendered manifest bundle (the
    # stateful set; the first is the service).
    return documents_of(manifests_of(infra))[1]


def pod_spec_of(infra: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the pod template spec of the rendered stateful set.
    return stateful_set_of(infra)["spec"]["template"]["spec"]


def container_of(infra: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the first container of the rendered stateful set's pod spec.
    return pod_spec_of(infra)["containers"][0]


def values_file(tmp_path: Path, infra: dict[str, Any], name: str = "infra.yaml") -> str:
    # [FUNCTION DOC] HELPER: Writes a yaml file under the temp path with the mapping nested under
    # an infra key and returns the file path as a string. Side effect: file on disk.
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"infra": infra}))
    return str(path)


def engines_of_kubernetes(tmp_path: Path, infra: dict[str, Any] = HOST_PATH_INFRA):
    # [FUNCTION DOC] HELPER: Writes the infra mapping to a values file, builds a Kubernetes train
    # config that uses it as its only helm values file, and returns the script's external engines
    # description for that config.
    config = ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, helm_values=(values_file(tmp_path, infra),))
    return script._external_engines(config)


@pytest.fixture(autouse=True)
def host_without_partition_or_wandb_key(monkeypatch):
    # [FUNCTION DOC] HELPER: Autouse fixture removing the visible-devices and wandb key
    # environment variables so tests do not depend on the host's values.
    for name in ("CUDA_VISIBLE_DEVICES", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)


class TestTheBackendDecidesWhoStartsTheEngines:
    # [CLASS DOC] Tests of the engine description per cluster backend.
    def test_a_ray_run_starts_them_itself_and_installs_nothing(self):
        # [FUNCTION DOC] For Ray, asserts no extra manifests, prepare commands only for the
        # trainer key, and addresses being loopback with each of the script's Ray engine ports.
        """Ray installs no release, so the only place left to start an engine is beside the trainer."""
        engines = script._external_engines(ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY))

        assert engines.extra_manifests == []
        assert set(engines.prepare_cmd) == {"trainer"}
        assert engines.addrs == [f"127.0.0.1:{port}" for port in script.RAY_ENGINE_PORTS]

    def test_a_kubernetes_run_installs_them_and_asks_no_pod_to_start_them(self, tmp_path):
        # [FUNCTION DOC] For Kubernetes, asserts empty prepare commands and exactly one extra manifest.
        """A trainer pod that also launched the engines would take their gpus and pin them to one node."""
        engines = engines_of_kubernetes(tmp_path)

        assert engines.prepare_cmd == {}
        assert len(engines.extra_manifests) == 1

    def test_the_addrs_the_run_is_given_are_the_pods_the_manifest_declares(self, tmp_path):
        # [FUNCTION DOC] Computes the expected pod dns addresses from the stateful set (name, index
        # per replica, service name, first container port) and asserts they equal the engines'
        # address list.
        """These two are written in different places, and a drift between them is a run dialling nothing."""
        engines = engines_of_kubernetes(tmp_path)
        stateful_set = documents_of(engines.extra_manifests[0])[1]
        service_name = stateful_set["spec"]["serviceName"]
        port = stateful_set["spec"]["template"]["spec"]["containers"][0]["ports"][0]["containerPort"]

        assert engines.addrs == [
            f"{stateful_set['metadata']['name']}-{index}.{service_name}:{port}"
            for index in range(stateful_set["spec"]["replicas"])
        ]

    def test_both_backends_run_the_same_engine(self):
        # [FUNCTION DOC] Asserts the first five command tokens of the Kubernetes container appear
        # in the Ray trainer prepare command text, and the container command ends with the
        # static memory fraction 0.7 and trust-remote-code options.
        """The point of the second backend is a new place to run the test, not a second thing to test."""
        ray_command = script._external_engines(ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)).prepare_cmd
        argv = container_of(HOST_PATH_INFRA)["command"]

        assert " ".join(argv[:5]) in ray_command["trainer"]
        assert argv[-3:] == ["--mem-fraction-static", "0.7", "--trust-remote-code"]

    def test_the_run_sizes_its_router_after_the_engines_that_exist(self, tmp_path):
        # [FUNCTION DOC] Builds the train args for the Kubernetes engine addresses and asserts the
        # rollout-num-gpus value equals the number of engines times gpus per engine.
        """--rollout-num-gpus is cross-checked against what the engines report, and a mismatch aborts the run."""
        engines = engines_of_kubernetes(tmp_path)
        train_args = script._train_args(engines.addrs, object_store_args="").split()

        gpus = int(train_args[train_args.index("--rollout-num-gpus") + 1])
        assert gpus == script.NUM_ENGINES * script.GPUS_PER_ENGINE


class TestTheObjectStoreTheBackendCanActuallyServe:
    # [CLASS DOC] Tests of the object-store arguments by backend.
    def test_a_kubernetes_run_names_the_mooncake_store_it_will_be_given(self):
        # [FUNCTION DOC] Asserts the Kubernetes store args select the mooncake backend and include the init kwargs option.
        """The launcher forces mooncake on kubernetes and then aborts on the init kwargs nobody passed."""
        args = script._object_store_args(ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES))

        assert "--object-store-backend mooncake" in args
        assert "--mooncake-store-init-kwargs" in args

    def test_a_ray_run_keeps_the_store_it_already_had(self):
        # [FUNCTION DOC] Asserts the Ray store args are the empty string.
        """Ray serves the default store itself, and naming mooncake here would test a second thing at once."""
        assert script._object_store_args(ExecuteTrainConfig(cluster_backend=ClusterBackend.RAY)) == ""

    def test_the_kubernetes_train_args_carry_it(self):
        # [FUNCTION DOC] Builds train args for one engine address with the Kubernetes store args;
        # asserts the init kwargs option is in the result.
        """These args are the only place the run learns of the store, and a drop aborts before helm installs."""
        args = script._train_args(
            ["engine:30000"],
            object_store_args=script._object_store_args(ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES)),
        )

        assert "--mooncake-store-init-kwargs" in args


class TestTheEnginesArePublishedOnlyOnceTheyServe:
    # [CLASS DOC] Tests of service, readiness and startup policy of the rendered engine objects.
    def test_the_pods_get_a_dns_name_of_their_own(self):
        # [FUNCTION DOC] Asserts the first document (the service) has no cluster ip (headless) and
        # does not publish not-ready addresses.
        """The provider dials one address per engine, which only a headless service gives it."""
        service = documents_of(manifests_of(HOST_PATH_INFRA))[0]

        assert service["spec"]["clusterIP"] == "None"
        assert "publishNotReadyAddresses" not in service["spec"]

    def test_an_engine_counts_as_ready_only_once_it_can_generate(self):
        # [FUNCTION DOC] Asserts the container's readiness probe is an http get on the generate health path.
        """Serving http is not serving the model, and the run's own health check is this same path."""
        probe = container_of(HOST_PATH_INFRA)["readinessProbe"]

        assert probe["httpGet"]["path"] == "/health_generate"

    def test_the_engines_start_at_the_same_time(self):
        # [FUNCTION DOC] Asserts the stateful set pod management policy is parallel.
        """The default policy starts them one after the other, doubling a wait the run is already timing."""
        assert stateful_set_of(HOST_PATH_INFRA)["spec"]["podManagementPolicy"] == "Parallel"


class TestTheEnginesAreDescribedFromTheClusterItsOwnInfra:
    # [CLASS DOC] Tests that the rendered pods follow the cluster infra values.
    def test_they_run_the_image_the_cluster_named(self):
        # [FUNCTION DOC] Asserts the container image is repository:tag from the infra and that no
        # image pull policy is set when none was given.
        """A hardcoded image would either miss sglang or pin the test to one registry."""
        container = container_of(HOST_PATH_INFRA)

        assert container["image"] == "radixark/miles:dev"
        assert "imagePullPolicy" not in container

    def test_an_image_the_cluster_qualified_is_taken_whole(self):
        # [FUNCTION DOC] With a pull policy of always and one pull secret in the infra, asserts the
        # container carries the policy and the pod spec lists the secret by name.
        """A private registry needs both halves, and dropping either is a pod that never pulls."""
        infra = infra_with(
            image={**HOST_PATH_INFRA["image"], "pullPolicy": "Always", "pullSecrets": ["regcred"]},
        )

        assert container_of(infra)["imagePullPolicy"] == "Always"
        assert pod_spec_of(infra)["imagePullSecrets"] == [{"name": "regcred"}]

    def test_they_mount_the_storage_the_run_reads_its_checkpoint_from(self):
        # [FUNCTION DOC] Asserts the pod volumes are the one host-path volume with directory type
        # and the container mounts it at the declared mount path.
        """The engines serve the same model the trainer updates, and only the mount makes it one file."""
        assert pod_spec_of(HOST_PATH_INFRA)["volumes"] == [
            {"name": "cluster-storage", "hostPath": {"path": "/cluster-storage", "type": "Directory"}}
        ]
        assert container_of(HOST_PATH_INFRA)["volumeMounts"] == [
            {"name": "cluster-storage", "mountPath": "/cluster-storage"}
        ]

    def test_a_cluster_that_shares_a_claim_gets_a_claim(self):
        # [FUNCTION DOC] With a persistent-volume-claim volume in the infra, asserts the pod
        # volume list is that claim by name.
        """hostPath is one cluster's answer; a pvc cluster would silently read an empty directory."""
        infra = infra_with(
            volumes=[
                {
                    "name": "cluster-storage",
                    "persistentVolumeClaim": {"claimName": "miles-data"},
                    "mounts": [{"mountPath": "/cluster-storage"}],
                }
            ]
        )

        assert pod_spec_of(infra)["volumes"] == [
            {"name": "cluster-storage", "persistentVolumeClaim": {"claimName": "miles-data"}}
        ]

    def test_every_volume_the_cluster_declares_reaches_these_pods_too(self):
        # [FUNCTION DOC] With an extra read-only models host-path volume, asserts both volume names
        # appear in order and the read-only mount is among the container mounts.
        """A read-only model cache the run's own pods mount is a directory these engines read the weights from."""
        infra = infra_with(
            volumes=[
                *HOST_PATH_INFRA["volumes"],
                {
                    "name": "models",
                    "hostPath": {"path": "/gpfs/models", "type": "Directory"},
                    "mounts": [{"mountPath": "/models", "readOnly": True}],
                },
            ]
        )

        assert [volume["name"] for volume in pod_spec_of(infra)["volumes"]] == ["cluster-storage", "models"]
        assert {"name": "models", "mountPath": "/models", "readOnly": True} in container_of(infra)["volumeMounts"]

    def test_a_cluster_that_shares_nothing_mounts_nothing(self):
        # [FUNCTION DOC] With an empty volume list, asserts the pod spec has no volumes key and the
        # container has no volume mounts key.
        """An empty volume section renders a pod kubernetes refuses, which reads as a chart bug."""
        infra = infra_with(volumes=[])

        assert "volumes" not in pod_spec_of(infra)
        assert "volumeMounts" not in container_of(infra)

    def test_they_land_where_the_run_own_pods_land(self):
        # [FUNCTION DOC] With a node selector and a toleration in the infra scheduling section,
        # asserts the pod spec carries both and has no affinity.
        """A gpu pod that ignores the cluster's scheduling stays Pending until the run times out."""
        infra = infra_with(
            scheduling={"nodeSelector": {"gpu": "h200"}, "tolerations": [{"key": "gpu", "operator": "Exists"}]},
        )
        pod_spec = pod_spec_of(infra)

        assert pod_spec["nodeSelector"] == {"gpu": "h200"}
        assert pod_spec["tolerations"] == [{"key": "gpu", "operator": "Exists"}]
        assert "affinity" not in pod_spec

    def test_they_inherit_the_environment_every_other_pod_gets(self):
        # [FUNCTION DOC] With one infra environment variable, asserts the container env is exactly
        # that name and value pair.
        """Proxy and cache settings are how a pod reaches anything at all on a closed cluster."""
        infra = infra_with(env={"HF_HUB_OFFLINE": "1"})

        assert container_of(infra)["env"] == [{"name": "HF_HUB_OFFLINE", "value": "1"}]

    def test_each_engine_asks_for_the_gpus_it_will_use(self):
        # [FUNCTION DOC] Asserts the container resource limits are the nvidia gpu count equal to
        # the script's gpus per engine.
        """Without a request the scheduler stacks every engine onto one node and they fight over memory."""
        assert container_of(HOST_PATH_INFRA)["resources"] == {"limits": {"nvidia.com/gpu": script.GPUS_PER_ENGINE}}

    def test_no_object_names_a_namespace_of_its_own(self):
        # [FUNCTION DOC] Asserts no rendered document sets a namespace in its metadata.
        """These are installed into whichever namespace the run is; a pinned one is another cluster's."""
        documents = documents_of(manifests_of(HOST_PATH_INFRA))

        assert all("namespace" not in document["metadata"] for document in documents)


class TestTheInfraValuesAreReadTheWayHelmReadsThem:
    # [CLASS DOC] Tests of reading and merging infra values files.
    def test_a_later_values_file_overrides_only_what_it_names(self, tmp_path):
        # [FUNCTION DOC] Reads a base file and a later file overriding only the image tag. Asserts
        # the repository is kept from the base, the tag is overridden, and the base's mount path survives.
        """helm merges its -f files, so reading only the last one would drop the cluster's own defaults."""
        base = values_file(tmp_path, HOST_PATH_INFRA)
        override = values_file(tmp_path, {"image": {"tag": "mine"}}, name="override.yaml")

        infra = script._infra_values((base, override))

        assert infra.image.repository == "radixark/miles"
        assert infra.image.tag == "mine"
        assert [mount.mount_path for volume in infra.volumes for mount in volume.mounts] == ["/cluster-storage"]

    def test_a_launch_that_names_no_values_file_reads_what_the_chart_would(self):
        # [FUNCTION DOC] With no values files, asserts the result has a non-empty image repository
        # and non-empty volumes (the chart's defaults).
        """helm starts from the chart's own values.yaml, and the launcher accepts a launch that adds none."""
        infra = script._infra_values(())

        assert infra.image.repository
        assert infra.volumes

    def test_a_file_that_names_only_a_tag_is_a_legal_override(self, tmp_path):
        # [FUNCTION DOC] With one file naming only an image tag, asserts the tag is applied and the
        # chart's default volumes are still present.
        """This is a whole run's worth of values for the run itself, and refusing it refuses a valid launch."""
        infra = script._infra_values((values_file(tmp_path, {"image": {"tag": "mine"}}),))

        assert infra.image.tag == "mine"
        assert infra.volumes


@requires_helm
class TestTheChartInstallsTheEnginesAsWritten:
    # [CLASS DOC] Chart-render test, skipped unless helm is available.
    def test_the_objects_the_script_describes_are_the_objects_the_release_installs(self, tmp_path):
        # [FUNCTION DOC] Takes the engine manifest from the Kubernetes engines description and
        # asserts the objects the chart adds when rendering it equal the manifest's own documents.
        """The manifest only matters once the chart has rendered it, and it is rendered verbatim."""
        manifest = engines_of_kubernetes(tmp_path).extra_manifests[0]

        assert objects_added_by(manifest) == documents_of(manifest)
