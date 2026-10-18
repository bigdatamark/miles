# [MODULE DOC]
# Tests (helm rendering, needs helm) of the workbench stateful set and its pod: name validation, a
# single idle pod on the training image, an infra values checksum annotation that rolls the pod on
# config changes, stable selector, CPU resources, service account token, host-path / claim / empty
# volume handling, checkout mounts, default backend / namespace / helm values environment, values
# mounted from a config map, reserved name and path conflicts, scheduling and pull secrets
# pass-through, and namespace placeholder substitution with unknown placeholders refused.
import yaml
from tests.fast.charts.utils import (
    NAMESPACE,
    RELEASE_NAME,
    container,
    host_path_volume,
    pod_spec,
    render,
    render_error,
    requires_helm,
    single_object_of_kind,
    volumes_args,
)

from miles.utils.workers.types import ClusterBackend


def _volume(objects: list[dict], name: str) -> dict:
    # [FUNCTION DOC] HELPER: Returns the pod volume with the given name from the single stateful
    # set; raises a stop-iteration error if absent.
    volumes = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"]
    return next(volume for volume in volumes if volume["name"] == name)


def _infra_checksum(objects: list[dict]) -> str:
    # [FUNCTION DOC] HELPER: Returns the infra values checksum annotation of the stateful set's pod
    # template.
    annotations = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["metadata"]["annotations"]
    return annotations["checksum/infra-values"]


@requires_helm
class TestWorkbenchStatefulSet:
    # [CLASS DOC] Tests of the workbench stateful set.
    def test_refuses_an_empty_object_name_before_installation(self) -> None:
        # [FUNCTION DOC] An empty object name fails rendering with the message that it must be set
        # to a non-empty workbench name.
        """An empty workbench name fails rendering rather than Kubernetes admission."""
        error = render_error("--set-string", "objectName=")

        assert "objectName must be set to a non-empty workbench name" in error

    def test_a_single_pod_idles_on_the_training_image(self):
        # [FUNCTION DOC] Default render: one replica, service name equal to the release, image
        # "radixark/miles:dev" always pulled, and the command is sleep infinity.
        """The workbench is one long-lived pod on the training image, not a job that exits."""
        objects = render()
        statefulset = single_object_of_kind(objects, "StatefulSet")

        assert statefulset["spec"]["replicas"] == 1
        assert statefulset["spec"]["serviceName"] == RELEASE_NAME
        assert container(objects)["image"] == "radixark/miles:dev"
        assert container(objects)["imagePullPolicy"] == "Always"
        assert container(objects)["command"] == ["sleep", "infinity"]

    def test_the_pod_template_carries_the_infra_config_checksum(self):
        # [FUNCTION DOC] The default checksum is 64 characters long and differs from the checksum
        # produced when the runs root value changes.
        """A ConfigMap-only edit must roll the pod, otherwise a finished install still serves the old infra.yaml."""
        default = _infra_checksum(render())
        changed = _infra_checksum(render("--set", "infra.paths.runsRoot=/elsewhere/miles_data"))

        assert len(default) == 64
        assert default != changed

    def test_the_selector_is_the_stable_label_subset(self):
        # [FUNCTION DOC] The selector's match labels equal exactly name miles-workbench, instance
        # (release) and component workbench, and are contained in the template labels.
        """A selector is immutable after creation and must be a subset of the template labels."""
        statefulset = single_object_of_kind(render(), "StatefulSet")
        selector = statefulset["spec"]["selector"]["matchLabels"]
        template_labels = statefulset["spec"]["template"]["metadata"]["labels"]

        assert selector == {
            "app.kubernetes.io/name": "miles-workbench",
            "app.kubernetes.io/instance": RELEASE_NAME,
            "app.kubernetes.io/component": "workbench",
        }
        assert selector.items() <= template_labels.items()

    def test_the_pod_stays_a_cpu_pod(self):
        # [FUNCTION DOC] Resource overrides (CPU request 4, memory limit 16Gi) appear in the
        # container resources.
        """The workbench only parses args and follows logs; its requests must reach the pod as configured."""
        objects = render("--set", "resources.requests.cpu=4", "--set", "resources.limits.memory=16Gi")

        assert container(objects)["resources"]["requests"]["cpu"] == 4
        assert container(objects)["resources"]["limits"]["memory"] == "16Gi"

    def test_the_pod_carries_its_service_account_token(self):
        # [FUNCTION DOC] The pod's service account is the release name and token automount is true.
        """helm and kubectl inside the pod act as the chart's ServiceAccount, so its token must be mounted."""
        spec = pod_spec(render())

        assert spec["serviceAccountName"] == RELEASE_NAME
        assert spec["automountServiceAccountToken"] is True

    def test_a_host_path_volume_mounts_where_training_pods_see_it(self):
        # [FUNCTION DOC] A host path volume at /gpfs renders with type Directory, and the container
        # mounts it at /cluster-storage.
        """A volume is mounted at the configured path verbatim so launch scripts need no path mapping."""
        objects = render(*volumes_args(host_path_volume(path="/gpfs")))
        volume = _volume(objects, "cluster-storage")

        assert volume["hostPath"] == {"path": "/gpfs", "type": "Directory"}
        assert {"name": "cluster-storage", "mountPath": "/cluster-storage"} in container(objects)["volumeMounts"]

    def test_a_pvc_volume_binds_the_named_claim(self):
        # [FUNCTION DOC] A claim-backed volume renders with exactly the claim name.
        """Clusters without host mounts point the same mount at a pre-existing RWX claim."""
        objects = render(
            "--set-json",
            'infra.volumes=[{"name":"cluster-storage","persistentVolumeClaim":{"claimName":"miles-shared"},'
            '"mounts":[{"mountPath":"/cluster-storage"}]}]',
        )

        assert _volume(objects, "cluster-storage")["persistentVolumeClaim"] == {"claimName": "miles-shared"}

    def test_an_empty_volume_list_leaves_the_pod_with_no_storage_of_its_own(self):
        # [FUNCTION DOC] With no volumes configured, the only pod volume and mount is the infra
        # values one.
        """Storage is optional; declaring none must not leave a dangling mount referencing a missing volume."""
        objects = render("--set-json", "infra.volumes=[]")

        assert [volume["name"] for volume in pod_spec(objects)["volumes"]] == ["infra-values"]
        assert [mount["name"] for mount in container(objects)["volumeMounts"]] == ["infra-values"]

    def test_a_mount_over_the_image_checkout_replaces_it_in_this_pod_too(self):
        # [FUNCTION DOC] A sub-path mount at /root/miles appears in the container mounts.
        """Launch scripts run from this pod, so it must see the same checkout as the training pods do."""
        objects = render(
            *volumes_args(host_path_volume(mounts=[{"mountPath": "/root/miles", "subPath": "alice/miles"}]))
        )

        assert {"name": "cluster-storage", "mountPath": "/root/miles", "subPath": "alice/miles"} in container(objects)[
            "volumeMounts"
        ]

    def test_a_run_launched_from_the_pod_picks_this_backend_without_being_told(self):
        # [FUNCTION DOC] The container environment has the script cluster backend variable set to
        # the Kubernetes backend value.
        """The pod exists to install runs into its own cluster, so naming the backend again is noise."""
        assert (
            dict(name="MILES_SCRIPT_CLUSTER_BACKEND", value=ClusterBackend.KUBERNETES.value)
            in container(render())["env"]
        )

    def test_a_cluster_value_still_wins_over_that_default(self):
        # [FUNCTION DOC] Setting the backend variable to ray via infra env makes the container
        # environment hold that value (the entry with value ray is present).
        """The default is a convenience, and a cluster that sets the variable itself means it."""
        objects = render("--set", "infra.env.MILES_SCRIPT_CLUSTER_BACKEND=ray")

        assert dict(name="MILES_SCRIPT_CLUSTER_BACKEND", value="ray") in container(objects)["env"]

    def test_a_run_lands_in_the_namespace_the_pod_lives_in(self):
        # [FUNCTION DOC] The script namespace variable equals the test namespace.
        """The pod's Role is namespaced, so its own namespace is the only one it can install into."""
        assert dict(name="MILES_SCRIPT_NAMESPACE", value=NAMESPACE) in container(render())["env"]

    def test_the_installed_helm_values_are_mounted_for_the_runs_launched_here(self):
        # [FUNCTION DOC] The single config map's infra file parses to the installed infra values
        # (host path /gpfs); the container mounts the infra values volume at /etc/miles; that volume
        # is a config map whose name ends with "-infra"; and the helm values script variable points
        # at the infra file under /etc/miles.
        """A run rendered from a second copy of these values would schedule its pods against another cluster."""
        objects = render(*volumes_args(host_path_volume(path="/gpfs")))
        data = single_object_of_kind(objects, "ConfigMap")["data"]["infra.yaml"]

        assert yaml.safe_load(data)["infra"]["volumes"][0]["hostPath"]["path"] == "/gpfs"
        assert {"name": "infra-values", "mountPath": "/etc/miles"} in container(objects)["volumeMounts"]
        assert _volume(objects, "infra-values")["configMap"]["name"].endswith("-infra")
        assert dict(name="MILES_SCRIPT_HELM_VALUES", value="/etc/miles/infra.yaml") in container(objects)["env"]

    def test_a_user_volume_may_not_take_the_reserved_name_or_path(self):
        # [FUNCTION DOC] A user volume named infra-values, and a user mount at /etc/miles, each make
        # rendering fail with an error naming the conflicting name or path.
        """Both belong to the chart's own infra values mount, and a duplicate of either is rejected by kubernetes."""
        name_clash = render_error(*volumes_args(host_path_volume(name="infra-values")))
        path_clash = render_error(*volumes_args(host_path_volume(mounts=[{"mountPath": "/etc/miles"}])))

        assert "infra-values" in name_clash
        assert "/etc/miles" in path_clash

    def test_scheduling_and_environment_values_reach_the_pod(self, tmp_path):
        # [FUNCTION DOC] A values file with a node selector, a toleration, an empty-preferences pod
        # anti-affinity and a proxy variable: the rendered pod spec carries the same node selector,
        # tolerations and affinity, and the container env includes the proxy variable.
        """Cluster-specific scheduling and environment values are passed through untouched."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(
            yaml.safe_dump(
                dict(
                    infra=dict(
                        scheduling=dict(
                            nodeSelector={"pool": "cpu"},
                            tolerations=[dict(key="gpu", operator="Exists", effect="NoSchedule")],
                            affinity=dict(podAntiAffinity=dict(preferredDuringSchedulingIgnoredDuringExecution=[])),
                        ),
                        env={"HTTP_PROXY": "http://proxy:7890"},
                    )
                )
            )
        )
        objects = render("-f", str(values_file))
        spec = pod_spec(objects)

        assert spec["nodeSelector"] == {"pool": "cpu"}
        assert spec["tolerations"] == [dict(key="gpu", operator="Exists", effect="NoSchedule")]
        assert spec["affinity"] == dict(podAntiAffinity=dict(preferredDuringSchedulingIgnoredDuringExecution=[]))
        assert dict(name="HTTP_PROXY", value="http://proxy:7890") in container(objects)["env"]

    def test_private_registries_get_their_pull_secret(self):
        # [FUNCTION DOC] A configured pull secret renders as the pod's only pull secret.
        """The training image often lives in a private registry, so its pull secret must be declared."""
        objects = render("--set", "infra.image.pullSecrets[0]=registry-cred")

        assert pod_spec(objects)["imagePullSecrets"] == [dict(name="registry-cred")]

    def test_defaults_leave_scheduling_and_pull_secrets_unset(self):
        # [FUNCTION DOC] The default pod spec has no node selector, tolerations, affinity or pull
        # secrets keys.
        """Empty scheduling values must be omitted rather than rendered as empty API fields."""
        spec = pod_spec(render())

        assert "nodeSelector" not in spec
        assert "tolerations" not in spec
        assert "affinity" not in spec
        assert "imagePullSecrets" not in spec


@requires_helm
class TestNamespaceInterpolation:
    # [CLASS DOC] Tests of namespace placeholder substitution in the workbench chart (need helm).
    def test_the_workbench_mounts_what_its_own_namespace_names(self):
        # [FUNCTION DOC] A host path and a sub path containing the namespace placeholder render with
        # the actual namespace in the volume host path and the code mount sub path.
        """One workbench per namespace, and the same values file has to give each of them its own checkout."""
        objects = render(
            *volumes_args(
                host_path_volume(
                    path="/data/${NAMESPACE}",
                    mounts=[
                        {"mountPath": "/cluster-storage"},
                        {"mountPath": "/root/miles", "subPath": "repos/${NAMESPACE}/miles"},
                    ],
                )
            )
        )

        assert _volume(objects, "cluster-storage")["hostPath"]["path"] == f"/data/{NAMESPACE}"
        assert {
            "name": "cluster-storage",
            "mountPath": "/root/miles",
            "subPath": f"repos/{NAMESPACE}/miles",
        } in container(objects)["volumeMounts"]

    def test_an_unknown_variable_is_refused_by_a_chart_that_renders_no_run(self):
        # [FUNCTION DOC] An unknown placeholder in the runs root fails rendering with an error naming
        # the runs root value and the placeholder.
        """This chart writes the infra.yaml every launch from here reads, so a typo has to stop at this render."""
        error = render_error("--set", "infra.paths.runsRoot=/cluster-storage/${NAMESPCE}/data")

        assert "infra.paths.runsRoot" in error
        assert "${NAMESPCE}" in error
