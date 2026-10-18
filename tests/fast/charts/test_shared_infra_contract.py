# [MODULE DOC]
# Tests that the shared infra values contract (a single top-level infra subtree defined by the
# shared infra JSON schema) is honored identically by every application chart: the schema is
# inlined verbatim, every shared section has a non-null default, and one cluster values file
# renders correctly in both the workbench chart and the run chart (the two render tests need helm).
from pathlib import Path
from typing import Any

import yaml
from tests.fast.charts.utils import (
    SHARED_INFRA_SCHEMA_PATH,
    chart_directories,
    container,
    pod_spec,
    pod_spec_of,
    render,
    render_run,
    requires_helm,
    resolved_schema,
    sole_container_of,
)

# Module data: CLUSTER_VALUES is a sample cluster values mapping (image registry and tag, a
# claim-backed volume with two mounts, a host-path scratch volume, a runs root path, a node selector
# and one environment variable) used by the render tests; MILES_CODE_MOUNT is the expected mount
# entry derived from the sub-path mount.
CLUSTER_VALUES = dict(
    infra=dict(
        image=dict(repository="registry.local/miles", tag="v1"),
        volumes=[
            dict(
                name="cluster-storage",
                persistentVolumeClaim=dict(claimName="shared"),
                mounts=[
                    dict(mountPath="/cluster-storage"),
                    dict(mountPath="/root/miles", subPath="myuser/miles"),
                ],
            ),
            dict(
                name="scratch",
                hostPath=dict(path="/local", type="DirectoryOrCreate"),
                mounts=[dict(mountPath="/scratch")],
            ),
        ],
        paths=dict(runsRoot="/cluster-storage/teamdata"),
        scheduling=dict(nodeSelector={"pool": "cpu"}),
        env={"HF_ENDPOINT": "https://mirror"},
    )
)

MILES_CODE_MOUNT = {"name": "cluster-storage", "mountPath": "/root/miles", "subPath": "myuser/miles"}


def shared_infra_schema() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the resolved shared infra schema's definition of the infra
    # property.
    return resolved_schema(SHARED_INFRA_SCHEMA_PATH)["properties"]["infra"]


def cluster_values_file(tmp_path: Path) -> Path:
    # [FUNCTION DOC] HELPER: Writes the sample cluster values as YAML to a file named cluster.yaml
    # inside the given directory and returns its path.
    values_file = tmp_path / "cluster.yaml"
    values_file.write_text(yaml.safe_dump(CLUSTER_VALUES))
    return values_file


def chart_infra_defaults(chart_dir: Path) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns the infra section of the chart's default values file.
    return yaml.safe_load((chart_dir / "values.yaml").read_text())["infra"]


class TestSharedInfraContract:
    # [CLASS DOC] Contract tests over all application charts and the shared infra schema.
    def test_the_contract_is_the_single_infra_subtree(self):
        # [FUNCTION DOC] The shared schema's top-level properties are exactly the single key infra.
        """One top-level key is what lets a cluster values file be handed to any chart unchanged."""
        assert set(resolved_schema(SHARED_INFRA_SCHEMA_PATH)["properties"]) == {"infra"}

    def test_the_infra_subtree_is_exactly_the_cluster_shaped_sections(self):
        # [FUNCTION DOC] The infra subtree's properties are exactly: image, volumes, paths, shared
        # memory, scheduling and env.
        """A section with no helper behind it would be accepted by the schema and never reach a pod."""
        assert set(shared_infra_schema()["properties"]) == {
            "image",
            "volumes",
            "paths",
            "devShm",
            "scheduling",
            "env",
        }

    def test_every_chart_inlines_the_shared_infra_schema_verbatim(self):
        # [FUNCTION DOC] For every application chart, the infra property of its own resolved values
        # schema file equals the shared infra definition exactly.
        """Helm cannot $ref across files, so every chart carries its own copy of the same contract."""
        shared = shared_infra_schema()

        for chart_dir in chart_directories():
            properties = resolved_schema(chart_dir / "values.schema.json")["properties"]
            assert properties["infra"] == shared, chart_dir

    def test_every_chart_defaults_every_shared_infra_key(self):
        # [FUNCTION DOC] For every application chart, all shared infra sections are present as keys
        # in its default values infra section.
        """A shared values file is only partly honoured by a chart that leaves one of the keys undefaulted."""
        sections = set(shared_infra_schema()["properties"])

        for chart_dir in chart_directories():
            assert sections <= set(chart_infra_defaults(chart_dir)), chart_dir

    def test_every_chart_ships_the_files_the_contract_is_pinned_through(self):
        # [FUNCTION DOC] Every application chart directory contains both a values schema file and a
        # default values file.
        """A chart shipping no values.schema.json would silently escape the two assertions above."""
        for chart_dir in chart_directories():
            assert (chart_dir / "values.schema.json").is_file(), chart_dir
            assert (chart_dir / "values.yaml").is_file(), chart_dir

    def test_no_chart_leaves_a_shared_key_undefined(self):
        # [FUNCTION DOC] For every chart and every shared section, the default value is not null.
        """A chart that defaults a shared section to null accepts the file and then renders nothing from it."""
        sections = set(shared_infra_schema()["properties"])

        for chart_dir in chart_directories():
            defaults = chart_infra_defaults(chart_dir)
            for section in sections:
                assert defaults[section] is not None, (chart_dir, section)

    @requires_helm
    def test_one_infra_nested_cluster_file_renders_the_workbench_chart(self, tmp_path):
        # [FUNCTION DOC] Renders the workbench chart with the sample cluster values file and asserts
        # the container image is "registry.local/miles:v1", the pod node selector is the sample
        # pool selector, and the container has the sub-path code mount. Requires helm.
        """The infra section alone must drive the chart, or a cluster file would need per-chart edits."""
        objects = render("-f", str(cluster_values_file(tmp_path)))

        assert container(objects)["image"] == "registry.local/miles:v1"
        assert pod_spec(objects)["nodeSelector"] == {"pool": "cpu"}
        assert MILES_CODE_MOUNT in container(objects)["volumeMounts"]

    @requires_helm
    def test_the_very_same_cluster_file_renders_the_run_chart(self, tmp_path):
        # [FUNCTION DOC] Renders the run chart with the same file and asserts the orchestrator
        # StatefulSet's sole container has the same image and code mount and its pod has the same
        # node selector. Requires helm.
        """Two charts reading the same file differently is exactly what the shared schema exists to prevent."""
        objects = render_run("-f", str(cluster_values_file(tmp_path)))
        orchestrator = sole_container_of(objects, "StatefulSet", "myrun-miles-run-orchestrator")

        assert orchestrator["image"] == "registry.local/miles:v1"
        assert pod_spec_of(objects, "StatefulSet", "myrun-miles-run-orchestrator")["nodeSelector"] == {"pool": "cpu"}
        assert MILES_CODE_MOUNT in orchestrator["volumeMounts"]
