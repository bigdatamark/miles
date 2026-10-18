# [MODULE DOC]
# Tests (helm rendering, needs helm) of the workbench chart's values schema validation. Covers:
# unknown keys in known sections and at top level, over-long names, relative or backstep host paths
# (and accepted dotted names), invalid and unusual-but-legal environment names, malformed claim and
# service account names, YAML boolean-lookalike names kept as strings, image tag injection safety,
# volume source rules (a pvc needs a claim, a host path needs a path, exactly one source, only
# supported kinds), absolute sub path and relative mount path rejection, runs root rules, and
# non-string environment values.
import pytest
import yaml
from tests.fast.charts.utils import (
    container,
    host_path_volume,
    pod_spec,
    render,
    render_error,
    requires_helm,
    schema_error_mentions,
    single_object_of_kind,
    volumes_args,
)


def _pvc_volume(claim_name: str) -> dict:
    # [FUNCTION DOC] HELPER: Builds a volume value named cluster-storage backed by a persistent
    # volume claim of the given name, with one mount at /cluster-storage.
    return {
        "name": "cluster-storage",
        "persistentVolumeClaim": {"claimName": claim_name},
        "mounts": [{"mountPath": "/cluster-storage"}],
    }


def _volume_of_kind(objects: list, key: str) -> dict:
    # [FUNCTION DOC] HELPER: Returns the single pod volume (of the single stateful set) that has
    # the source key (such as host path or claim); asserts exactly one matches.
    volumes = single_object_of_kind(objects, "StatefulSet")["spec"]["template"]["spec"]["volumes"]
    matched = [volume for volume in volumes if key in volume]
    assert len(matched) == 1, f"expected one {key} volume, got {volumes}"
    return matched[0]


@requires_helm
class TestValuesSchema:
    # [CLASS DOC] Schema validation tests; "rejected" means rendering fails and the error text
    # contains the named key or path.
    def test_an_unknown_key_inside_a_known_section_is_rejected(self):
        # [FUNCTION DOC] A misspelled image key makes rendering fail naming the key.
        """A typo inside a section the chart owns must fail the install rather than be silently ignored."""
        assert "taag" in render_error("--set", "infra.image.taag=dev")

    def test_an_unknown_top_level_key_is_rejected(self):
        # [FUNCTION DOC] A misspelled top-level section key fails rendering naming it.
        """A mistyped section name must fail: "rbca.create=false" otherwise silently keeps both privileged bindings."""
        assert "rbca" in render_error("--set", "rbca.create=false")

    def test_names_too_long_for_the_api_are_rejected(self):
        # [FUNCTION DOC] A 254-character claim name and a 254-character service account name are each
        # rejected (the errors name the claim name and the service account name path).
        """Both names are DNS subdomains, which the API server caps at 253 characters."""
        long_name = "a" * 254

        assert "claimName" in render_error(*volumes_args(_pvc_volume(long_name)))
        assert schema_error_mentions(
            render_error("--set", f"serviceAccount.name={long_name}"), path=("serviceAccount", "name")
        )

    def test_a_relative_host_path_is_rejected(self):
        # [FUNCTION DOC] A relative host path is rejected, naming host path.
        """A relative hostPath renders and installs, then leaves the pod stuck on a failed mount."""
        assert "hostPath" in render_error(*volumes_args(host_path_volume(path="relative/dir")))

    @pytest.mark.parametrize("path", ["/../cluster-storage", "/cluster-storage/.."])
    def test_a_host_path_containing_a_backstep_is_rejected(self, path):
        # [FUNCTION DOC] Parametrized over a leading and a trailing parent-directory segment; each
        # is rejected naming host path.
        """Kubernetes rejects any ".." segment in a volume path, so catch it at install time."""
        assert "hostPath" in render_error(*volumes_args(host_path_volume(path=path)))

    def test_a_host_path_that_merely_contains_dots_is_accepted(self):
        # [FUNCTION DOC] A host path with a double dot inside a name renders and keeps its path.
        """Only a whole ".." segment is a backstep; dots inside a name are ordinary characters."""
        objects = render(*volumes_args(host_path_volume(path="/a..b/c")))

        assert _volume_of_kind(objects, "hostPath")["hostPath"]["path"] == "/a..b/c"

    def test_an_environment_name_the_api_forbids_is_rejected(self):
        # [FUNCTION DOC] An environment name containing an equals sign is rejected, the error
        # containing it.
        """Kubernetes allows printable ASCII in an env name except "=", which would also break the rendering."""
        assert "A=B" in render_error("--set-string", "infra.env.A\\=B=value")

    def test_an_unusual_but_legal_environment_name_is_accepted(self):
        # [FUNCTION DOC] Environment names with a slash and with a space render and appear in the
        # container environment.
        """The pattern is the API contract, not the conventional identifier shape, so it must not be stricter."""
        objects = render("--set-string", "infra.env.FOO/BAR=value", "--set-string", "infra.env.FOO\\ BAZ=value")

        assert {"FOO/BAR", "FOO BAZ"} <= {entry["name"] for entry in container(objects)["env"]}

    def test_a_malformed_claim_name_is_rejected(self):
        # [FUNCTION DOC] A claim name with consecutive dots is rejected, naming claim name.
        """A claim name is a Kubernetes object name; "a..b" passes a loose pattern but no real API server."""
        assert "claimName" in render_error(*volumes_args(_pvc_volume("a..b")))

    def test_yaml_boolean_lookalike_names_stay_strings(self):
        # [FUNCTION DOC] A claim named "on" and a service account named "no" render as strings in
        # the volume claim and the pod service account name.
        """Names like "on" are valid Kubernetes names but YAML booleans, so they must be rendered quoted."""
        objects = render(*volumes_args(_pvc_volume("on")), "--set", "serviceAccount.name=no")
        assert _volume_of_kind(objects, "persistentVolumeClaim")["persistentVolumeClaim"]["claimName"] == "on"
        assert pod_spec(objects)["serviceAccountName"] == "no"

    def test_yaml_boolean_lookalike_environment_names_stay_strings(self, tmp_path):
        # [FUNCTION DOC] A values file with environment names "on" and "null" (written quoted by the
        # YAML dump) renders them as environment entries with string values "1" and "3".
        """Names like "on" and "null" are legal environment variable names but YAML booleans and nulls."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(yaml.safe_dump({"infra": {"env": {"on": "1", "null": "3"}}}))
        objects = render("-f", str(values_file))

        env = container(objects)["env"]

        assert dict(name="null", value="3") in env
        assert dict(name="on", value="1") in env

    def test_a_quote_in_the_image_tag_cannot_inject_pod_spec_keys(self):
        # [FUNCTION DOC] An image tag crafted to close the quote and add a privileged security
        # context renders as one image string exactly, and the container has no security context.
        """The image reference is assembled from two free-form values, so it must be quoted as one string."""
        injection = 'v1"\n          securityContext:\n            privileged: true\n          x: "'
        objects = render("--set-string", f"infra.image.tag={injection}")

        assert container(objects)["image"] == f"radixark/miles:{injection}"
        assert "securityContext" not in container(objects)

    def test_a_malformed_service_account_name_is_rejected(self):
        # [FUNCTION DOC] An account name with an uppercase letter and underscore is rejected at the
        # account name path.
        """The account name becomes a Kubernetes object name; catch a bad one before the API server does."""
        assert schema_error_mentions(
            render_error("--set", "serviceAccount.name=Bad_Name"), path=("serviceAccount", "name")
        )

    def test_a_pvc_volume_requires_a_claim_name(self):
        # [FUNCTION DOC] A claim volume with an empty claim object is rejected naming claim name.
        """A pvc volume with no claim renders a pod that can never schedule; catch it at install time."""
        assert "claimName" in render_error(
            "--set-json", 'infra.volumes=[{"name":"v","persistentVolumeClaim":{},"mounts":[{"mountPath":"/mnt"}]}]'
        )

    def test_a_host_path_volume_requires_a_path(self):
        # [FUNCTION DOC] A host path volume with an empty host path object is rejected naming host
        # path.
        """A hostPath volume with no path is equally unschedulable."""
        assert "hostPath" in render_error(
            "--set-json", 'infra.volumes=[{"name":"v","hostPath":{},"mounts":[{"mountPath":"/mnt"}]}]'
        )

    def test_a_volume_that_names_two_sources_is_rejected(self):
        # [FUNCTION DOC] A volume with both a host path and a claim is rejected, the error mentioning
        # the first volume's value path.
        """The free volume list has no storage type enum left, so this is all that stands between the two."""
        error = render_error(
            "--set-json",
            'infra.volumes=[{"name":"v","hostPath":{"path":"/s"},"persistentVolumeClaim":{"claimName":"c"},'
            '"mounts":[{"mountPath":"/mnt"}]}]',
        )

        assert schema_error_mentions(error, path=("infra", "volumes", "0"))

    def test_a_volume_that_names_no_source_is_rejected(self):
        # [FUNCTION DOC] A volume with only a name and mounts is rejected at the first volume's path.
        """A volume with no source is a mount kubernetes cannot satisfy, so its pods only ever stay pending."""
        error = render_error("--set-json", 'infra.volumes=[{"name":"v","mounts":[{"mountPath":"/mnt"}]}]')

        assert schema_error_mentions(error, path=("infra", "volumes", "0"))

    def test_a_source_kind_no_template_implements_is_rejected(self):
        # [FUNCTION DOC] A volume with a network file system source is rejected, the error naming
        # that source key.
        """Only the sources the templates render are accepted; another kind would silently mount nothing."""
        error = render_error(
            "--set-json", 'infra.volumes=[{"name":"v","nfs":{"server":"s"},"mounts":[{"mountPath":"/mnt"}]}]'
        )

        assert "nfs" in error

    def test_an_absolute_mount_subpath_is_rejected(self):
        # [FUNCTION DOC] A mount with an absolute sub path is rejected naming sub path.
        """A subPath is relative to the volume root by construction, and kubelet refuses an absolute one."""
        error = render_error(
            "--set-json",
            'infra.volumes=[{"name":"v","hostPath":{"path":"/s"},"mounts":[{"mountPath":"/mnt","subPath":"/abs"}]}]',
        )

        assert "subPath" in error

    def test_a_runs_root_containing_a_backstep_is_rejected(self):
        # [FUNCTION DOC] A runs root containing a parent-directory segment is rejected at the runs
        # root path.
        """A ".." segment would let a run write outside the directory the cluster set aside for miles."""
        assert schema_error_mentions(
            render_error("--set", "infra.paths.runsRoot=/cluster-storage/a/../b"), path=("infra", "paths", "runsRoot")
        )

    def test_a_relative_runs_root_is_rejected(self):
        # [FUNCTION DOC] A relative runs root is rejected at the runs root path.
        """The launcher hands this path to the pods as it is, so a relative one resolves per working directory."""
        assert schema_error_mentions(
            render_error("--set", "infra.paths.runsRoot=miles_data"), path=("infra", "paths", "runsRoot")
        )

    def test_a_relative_mount_path_is_rejected(self):
        # [FUNCTION DOC] A relative mount path is rejected naming mount path.
        """A container mountPath must be absolute, and a relative one only fails at apply time."""
        error = render_error(
            "--set-json", 'infra.volumes=[{"name":"v","hostPath":{"path":"/s"},"mounts":[{"mountPath":"scratch"}]}]'
        )

        assert "mountPath" in error

    def test_non_string_environment_values_are_rejected(self, tmp_path):
        # [FUNCTION DOC] A values file with a numeric environment value is rejected at that
        # variable's env path.
        """Kubernetes env values must be strings; a bare number would only fail later, at apply time."""
        values_file = tmp_path / "cluster.yaml"
        values_file.write_text(yaml.safe_dump(dict(infra=dict(env=dict(WORLD_SIZE=8)))))

        assert schema_error_mentions(render_error("-f", str(values_file)), path=("infra", "env", "WORLD_SIZE"))
