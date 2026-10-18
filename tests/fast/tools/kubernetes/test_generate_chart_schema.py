# [MODULE DOC] Tests of the Helm chart values-schema generator (tools/kubernetes/
# generate_chart_schema.py, loaded by file path): generated schemas must equal the committed
# files, exactly three schemas are produced (shared infra, run chart, workbench chart), each
# declares the JSON Schema draft-07 dialect that Helm validates with; the run chart schema
# (validated with jsonschema, skipped if missing) accepts the values the launcher generates, the
# chart's own default values (needs a yaml package), and a pool with every optional key; and it
# rejects typos in chart-owned sections, an enabled command job without a command, a PYTHONPATH
# override in the shared environment, host-path volumes without a path, volumes with two sources
# and volumes with no source. Constants: repository root (four levels up) and the generator path.
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
GENERATOR_PATH = REPO_ROOT / "tools" / "kubernetes" / "generate_chart_schema.py"


def load_generator():
    # [FUNCTION DOC] HELPER: Loads the generator script as a module named generate_chart_schema from
    # its file path and executes it (running its top-level code). Returns the module.
    spec = importlib.util.spec_from_file_location("generate_chart_schema", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    # [FUNCTION DOC] HELPER: Module-scoped fixture returning the loaded generator module.
    return load_generator()


@pytest.fixture(scope="module")
def schemas(generator) -> dict[Path, str]:
    # [FUNCTION DOC] HELPER: Module-scoped fixture returning the generator's mapping from
    # repository-relative schema file path to generated schema text.
    return generator.generated_schemas()


def _validator(schema: dict[str, Any]):
    # [FUNCTION DOC] HELPER: Skips the test if the jsonschema package is missing; otherwise
    # returns a draft-07 validator for the schema.
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft7Validator(schema)


def _run_schema(schemas: dict[Path, str]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Parses and returns the run chart's values schema from the generated
    # schemas mapping.
    return json.loads(schemas[Path("charts/miles-run/values.schema.json")])


def _minimal_run_values() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a minimal valid run-chart values mapping: infra with an image
    # (repository and tag) and one host-path volume with one mount, and a run section with an id,
    # a state file path and the seven object names (orchestrator, platform read, platform read
    # delete, mooncake master, colocate pairing, uninstall, uninstall manifest). A new mapping is
    # built on every call, so tests may mutate it.
    return {
        "infra": {
            "image": {"repository": "registry.local/miles", "tag": "v1"},
            "volumes": [
                {
                    "name": "cluster-storage",
                    "hostPath": {"path": "/cluster-storage"},
                    "mounts": [{"mountPath": "/cluster-storage"}],
                }
            ],
        },
        "run": {
            "id": "260101-000000-000",
            "stateFile": "/cluster-storage/miles_data/miles-runs/260101-000000-000/state/orchestrator.state",
            "objectNames": {
                "orchestrator": "r-miles-run-orchestrator",
                "platformRead": "r-miles-run-platform-read",
                "platformReadDelete": "r-miles-run-platform-read-delete",
                "mooncakeMaster": "r-miles-run-mooncake-master",
                "colocatePairing": "r-miles-run-colocate-pairing",
                "uninstall": "r-miles-run-uninstall",
                "uninstallManifest": "r-miles-run-uninstall-manifest",
            },
        },
    }


class TestGeneratedFilesAreCommitted:
    # [CLASS DOC] Tests that the generator's output matches what is committed.
    def test_every_schema_on_disk_is_what_the_generator_produces(self, schemas):
        # [FUNCTION DOC] For every generated schema, the file at that repository path has exactly
        # the generated text.
        """A hand-edited schema drifts from the types the launcher builds values with, silently."""
        for path, content in schemas.items():
            assert (REPO_ROOT / path).read_text() == content, path

    def test_the_generator_writes_exactly_the_three_schemas_the_repo_ships(self, schemas):
        # [FUNCTION DOC] The set of generated schema paths equals the shared infra schema, the run
        # chart values schema and the workbench chart values schema.
        """A chart whose schema nobody generates would keep a stale hand-written one forever."""
        assert set(schemas) == {
            Path("charts/shared-infra.schema.json"),
            Path("charts/miles-run/values.schema.json"),
            Path("charts/miles-workbench/values.schema.json"),
        }

    def test_every_schema_declares_the_dialect_helm_validates_with(self, schemas):
        # [FUNCTION DOC] Every generated schema's dollar-schema field equals the draft-07 URL.
        """helm uses gojsonschema, which reads draft-07; pydantic emits 2020-12 unless it is converted."""
        for path, content in schemas.items():
            assert json.loads(content)["$schema"] == "http://json-schema.org/draft-07/schema#", path


class TestRunSchemaAcceptsRealValues:
    # [CLASS DOC] Tests that the run chart schema accepts valid values.
    def test_the_values_the_launcher_generates_validate(self, schemas):
        # [FUNCTION DOC] The minimal run values validate against the run schema (no error).
        """The schema exists to catch bad values, and rejecting the good ones would block every run."""
        _validator(_run_schema(schemas)).validate(_minimal_run_values())

    def test_the_charts_own_defaults_validate(self, schemas):
        # [FUNCTION DOC] Skips if the yaml package is missing; loads the run chart's default values
        # file and validates it against the run schema (no error).
        """helm merges values.yaml under every install, so defaults that fail the schema break all of them."""
        yaml = pytest.importorskip("yaml")
        defaults = yaml.safe_load((REPO_ROOT / "charts" / "miles-run" / "values.yaml").read_text())

        _validator(_run_schema(schemas)).validate(defaults)

    def test_a_pool_carrying_every_optional_key_validates(self, schemas):
        # [FUNCTION DOC] Adds a trainer-engine pool entry with name, object name, pool id, command,
        # ports, environment, metadata, replicas, size and GPU resource limits to the minimal run
        # values and validates (no error).
        """Optional keys the builder emits only sometimes must not be rejected when it does emit them."""
        values = _minimal_run_values()
        values["run"]["trainerEngines"] = [
            {
                "name": "trainer-engine-actor",
                "objectName": "r-miles-run-trainer-engine-actor",
                "poolId": "trainer-engine-actor",
                "command": ["python", "-m", "serve"],
                "ports": [{"name": "master", "port": 9000}],
                "env": {"NCCL_CUMEM_ENABLE": "0"},
                "meta": {"miles.radixark.io/gpu-ids": "0,1"},
                "replicas": 2,
                "size": 2,
                "resources": {"limits": {"nvidia.com/gpu": 8}},
            }
        ]

        _validator(_run_schema(schemas)).validate(values)


class TestRunSchemaRejectsBadValues:
    # [CLASS DOC] Tests that the run chart schema rejects invalid values; each starts from the
    # minimal valid values, makes one change and expects a validation error.
    def test_a_typo_in_a_section_the_chart_owns_is_rejected(self, schemas):
        # [FUNCTION DOC] Adding a misspelled object-names key under the run section is rejected.
        """A silently ignored typo is exactly the failure a strict schema is installed to prevent."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["run"]["objectNamez"] = {}

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_an_enabled_command_job_without_a_command_is_rejected(self, schemas):
        # [FUNCTION DOC] An enabled command job with a name and object name but no command is rejected.
        """The conditional is the only thing standing between an enabled job and a pod that runs nothing."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["commandJob"] = {"enabled": True, "name": "convert", "objectName": "c"}

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_a_pythonpath_override_in_the_shared_env_is_rejected(self, schemas):
        # [FUNCTION DOC] Setting the Python path variable in the shared infra environment is rejected.
        """The image installs its three source trees as editable, and a user PYTHONPATH would shadow them."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["infra"]["env"] = {"PYTHONPATH": "/somewhere"}

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_a_host_path_volume_without_the_path_is_rejected(self, schemas):
        # [FUNCTION DOC] A volume whose host-path source is an empty mapping is rejected.
        """A hostPath volume with no host path renders and installs, then leaves every pod unable to start."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["infra"]["volumes"] = [{"name": "v", "hostPath": {}, "mounts": [{"mountPath": "/mnt"}]}]

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_a_volume_that_declares_two_sources_is_rejected(self, schemas):
        # [FUNCTION DOC] A volume declaring both a host-path and a persistent-volume-claim source is
        # rejected.
        """Nothing downstream can choose between them, and the free volume list has no type enum to catch it."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["infra"]["volumes"] = [
            {
                "name": "v",
                "hostPath": {"path": "/s"},
                "persistentVolumeClaim": {"claimName": "c"},
                "mounts": [{"mountPath": "/mnt"}],
            }
        ]

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)

    def test_a_volume_that_declares_no_source_at_all_is_rejected(self, schemas):
        # [FUNCTION DOC] A volume with only a name and a mount (no source) is rejected.
        """A sourceless volume is a mount kubernetes cannot satisfy, so every pod naming it stays pending."""
        jsonschema = pytest.importorskip("jsonschema")
        values = _minimal_run_values()
        values["infra"]["volumes"] = [{"name": "v", "mounts": [{"mountPath": "/mnt"}]}]

        with pytest.raises(jsonschema.ValidationError):
            _validator(_run_schema(schemas)).validate(values)
