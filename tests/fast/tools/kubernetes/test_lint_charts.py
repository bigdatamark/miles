# [MODULE DOC] Tests for the Helm chart lint hook in tools/kubernetes/lint_charts.py.
# The hook module is loaded by file path (not by package import) under the name lint_charts, so each
# test gets a fresh module object whose module-level tables can be mutated safely. Tests build tiny
# throwaway charts in a temporary directory and point the module's charts directory constant at it.
# Tests that call a real helm binary are skipped when helm is not on the path; the rest only use
# monkeypatching. Covered: chart discovery, linting pass/fail, dependency vendoring, behavior when
# helm is missing (skip locally, fail in CI), rejected-variant probing, and the colocate variant
# of the miles-run chart.
import importlib.util
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from tests.fast.source_scan import REPO_ROOT

LINT_CHARTS_PATH = REPO_ROOT / "tools" / "kubernetes" / "lint_charts.py"

requires_helm = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is required to lint charts")


def load_lint_charts():
    # [FUNCTION DOC] HELPER: Load the chart lint script as a brand new module object.
    # Builds a module spec from the script file path, creates the module and executes it, without
    # registering it in the module table. Returns the module. Each call re-executes the script, so
    # mutations of its module-level tables do not leak between callers. Raises whatever the script
    # raises at import time.
    spec = importlib.util.spec_from_file_location("lint_charts", LINT_CHARTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def lint_charts(tmp_path, monkeypatch):
    # [FUNCTION DOC] HELPER: Pytest fixture yielding a freshly loaded lint script module.
    # Redirects the module's charts directory constant to the per-test temporary directory and
    # removes the CI environment variable so the "missing helm" behavior defaults to the local one.
    # Returns the module. Both changes are undone by monkeypatch at test teardown.
    module = load_lint_charts()
    monkeypatch.setattr(module, "CHARTS_DIR", tmp_path)
    monkeypatch.delenv("CI", raising=False)
    return module


def write_chart(root: Path, name: str, *, library: bool = False, template: str = "") -> Path:
    # [FUNCTION DOC] HELPER: Create a minimal Helm chart directory under the given root.
    # Makes the chart directory with a templates subdirectory (parents created; raises if the chart
    # directory already exists), writes a version-2 chart manifest named after the chart with version
    # 0.1.0 (marked as a library chart when requested), and writes an empty-mapping values file.
    # When template text is non-empty it is written as templates/object.yaml. Returns the chart path.
    chart = root / name
    (chart / "templates").mkdir(parents=True)
    (chart / "Chart.yaml").write_text(
        textwrap.dedent(
            f"""
            apiVersion: v2
            name: {name}
            version: 0.1.0
            {"type: library" if library else ""}
            """
        ).strip()
        + "\n"
    )
    (chart / "values.yaml").write_text("{}\n")
    if template:
        (chart / "templates" / "object.yaml").write_text(template)
    return chart


VALID_TEMPLATE = "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: {{ .Release.Name }}\n"
BROKEN_TEMPLATE = "apiVersion: v1\nkind: ConfigMap\n  name: broken: [\n"


class TestChartDiscovery:
    # [CLASS DOC] Tests for how the lint script finds charts in its charts directory.
    def test_library_charts_are_linted_too(self, lint_charts, tmp_path):
        # [FUNCTION DOC] Creates one application chart and one library chart in the temporary
        # directory and asserts that the script's chart listing returns both names in sorted order,
        # i.e. library charts are not filtered out.
        """The hook promises to lint every chart, and a library chart can still fail to parse."""
        write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        write_chart(tmp_path, "lib", library=True)

        assert [chart.name for chart in lint_charts.all_charts()] == ["app", "lib"]

    def test_it_looks_in_the_repository_charts_directory(self):
        # [FUNCTION DOC] Loads an unpatched copy of the script and asserts that its repository root
        # constant contains the project metadata file and that its charts directory constant equals
        # the charts folder under the repository root computed by the test helper module.
        """Every other test points the constant at a tmp_path, so nothing else would notice a typo here."""
        module = load_lint_charts()

        assert (module.REPO_ROOT / "pyproject.toml").exists()
        assert module.CHARTS_DIR == REPO_ROOT / "charts"

    def test_a_repo_with_no_charts_is_not_a_failure(self, lint_charts):
        # [FUNCTION DOC] With an empty charts directory, asserts that the chart listing is empty and
        # that the script's main entry point with no arguments returns exit code zero.
        """The hook lands before the first chart, and an empty repo must not block every commit."""
        assert lint_charts.all_charts() == []
        assert lint_charts.main([]) == 0


@requires_helm
class TestLinting:
    # [CLASS DOC] Tests that run the real helm binary through the script's main entry point; the
    # whole class is skipped when helm is not installed.
    def test_a_valid_chart_passes(self, lint_charts, tmp_path):
        # [FUNCTION DOC] A chart with a valid config-map template makes main return exit code zero.
        """The baseline: a chart helm accepts must not be reported as broken."""
        write_chart(tmp_path, "app", template=VALID_TEMPLATE)

        assert lint_charts.main([]) == 0

    def test_a_broken_chart_fails(self, lint_charts, tmp_path):
        # [FUNCTION DOC] A chart whose template is not parseable YAML makes main return exit code one.
        """A template helm cannot parse is exactly what this hook exists to catch."""
        write_chart(tmp_path, "app", template=BROKEN_TEMPLATE)

        assert lint_charts.main([]) == 1

    def test_every_chart_is_linted_even_after_one_fails(self, lint_charts, tmp_path, capsys):
        # [FUNCTION DOC] Creates two broken charts (names sorting first and last), runs main, and
        # asserts exit code one plus that the captured standard error output mentions both chart
        # names, proving linting continues after the first failure.
        """Stopping at the first failure costs the contributor one round trip per broken chart."""
        write_chart(tmp_path, "a-broken", template=BROKEN_TEMPLATE)
        write_chart(tmp_path, "z-also-broken", template=BROKEN_TEMPLATE)

        assert lint_charts.main([]) == 1
        errors = capsys.readouterr().err

        assert "a-broken" in errors
        assert "z-also-broken" in errors

    def test_a_chart_whose_dependencies_cannot_be_vendored_fails(self, lint_charts, tmp_path):
        # [FUNCTION DOC] Creates a valid chart, then rewrites its manifest to depend on a chart at a
        # nonexistent relative file location and adds a matching lock file. Asserts main returns exit
        # code one, since the dependency build step cannot vendor the missing chart.
        """helm lint only warns about a missing dependency, so the build step is the only guard."""
        app = write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        (app / "Chart.yaml").write_text(
            "apiVersion: v2\nname: app\nversion: 0.1.0\n"
            'dependencies:\n  - name: absent\n    version: 0.1.0\n    repository: "file://../absent"\n'
        )
        (app / "Chart.lock").write_text(
            'dependencies:\n- name: absent\n  repository: "file://../absent"\n  version: 0.1.0\n'
            'digest: sha256:0\ngenerated: "2026-01-01T00:00:00Z"\n'
        )

        assert lint_charts.main([]) == 1

    def test_every_variant_of_a_chart_is_linted(self, lint_charts, tmp_path):
        # [FUNCTION DOC] Creates a valid chart with a values schema restricting a size setting to
        # small or large, then registers an extra variant for it in the script's variants table that
        # sets size to an invalid value. Asserts main returns exit code one, i.e. the extra variant
        # is linted and its schema violation is reported. Mutates the freshly loaded module only.
        """A chart is only as good as its worst supported value combination."""
        chart = write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        (chart / "values.schema.json").write_text(
            '{"type": "object", "properties": {"size": {"enum": ["small", "large"]}}}'
        )
        lint_charts.VARIANTS["app"] = [["--set", "size=huge"]]

        assert lint_charts.main([]) == 1

    def test_dependencies_are_vendored_before_linting(self, lint_charts, tmp_path):
        # [FUNCTION DOC] Creates a library chart and an application chart depending on it via a
        # relative file location, runs the real helm dependency update to produce the lock file, then
        # deletes the vendored charts subdirectory to mimic a fresh clone. Asserts the lock file
        # exists, main returns exit code zero, and that afterwards the vendored archive of the
        # library chart is present again (the script re-vendored it before linting).
        """The vendored copy is gitignored, so a fresh clone has a lock file and nothing else."""
        write_chart(tmp_path, "lib", library=True)
        (tmp_path / "lib" / "templates" / "_helpers.tpl").write_text('{{- define "lib.name" -}}lib{{- end }}\n')
        app = write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        (app / "Chart.yaml").write_text(
            "apiVersion: v2\nname: app\nversion: 0.1.0\n"
            'dependencies:\n  - name: lib\n    version: 0.1.0\n    repository: "file://../lib"\n'
        )
        subprocess.run(["helm", "dependency", "update", str(app)], capture_output=True, check=True)
        shutil.rmtree(app / "charts")

        assert (app / "Chart.lock").exists()
        assert lint_charts.main([]) == 0
        assert list((app / "charts").glob("lib-*.tgz"))


class TestMissingHelm:
    # [CLASS DOC] Tests for the script's behavior when the helm executable cannot be found; the
    # lookup function inside the script's shutil reference is patched to always find nothing.
    def test_it_skips_when_helm_is_absent_locally(self, lint_charts, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a broken chart present, no helm and no CI variable, main returns exit
        # code zero (the lint is skipped rather than failing the commit).
        """Contributors without helm should not be blocked by a hook that runs on every commit."""
        write_chart(tmp_path, "app", template=BROKEN_TEMPLATE)
        monkeypatch.setattr(lint_charts.shutil, "which", lambda _: None)

        assert lint_charts.main([]) == 0

    def test_it_fails_when_ci_provides_no_helm(self, lint_charts, tmp_path, monkeypatch):
        # [FUNCTION DOC] With a valid chart, no helm, and the CI environment variable set to true,
        # main returns exit code one (unchecked charts must not look like success in CI).
        """In CI a missing helm means the charts went unchecked, which must not look like success."""
        write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        monkeypatch.setattr(lint_charts.shutil, "which", lambda _: None)
        monkeypatch.setenv("CI", "true")

        assert lint_charts.main([]) == 1


class TestRejectedVariants:
    # [CLASS DOC] Tests for "rejected variants": value overrides that the chart is expected to
    # refuse, so a failing lint of them counts as success.
    def test_a_rejected_variant_keeps_the_chart_base_values(self, lint_charts, tmp_path, monkeypatch):
        # [FUNCTION DOC] Registers base values and one rejected variant (setting a Python path
        # environment entry) for the chart in the script's tables, replaces the script's command
        # runner with a fake that records every command and returns a failing result only when the
        # rejected override is in the command, and calls the single-chart lint function.
        # Asserts the lint function reports success (truthy: the rejection happened) and that the
        # last recorded command is a helm lint of the chart path followed by the base values
        # arguments and then the rejected override, in that order.
        """A rejection probe must reach its target instead of failing on an unrelated missing base value."""
        chart = write_chart(tmp_path, "app", template=VALID_TEMPLATE)
        lint_charts.BASE_VALUES["app"] = ["--set", "objectName=lint-app"]
        lint_charts.REJECTED_VARIANTS["app"] = [["--set", "infra.env.PYTHONPATH=/somewhere"]]
        commands: list[list[str]] = []

        def run(command: list[str]) -> subprocess.CompletedProcess[str]:
            # [FUNCTION DOC] HELPER: Fake command runner. Appends the command to the recorded
            # list and returns a completed-process result with empty output and return code one if
            # the rejected override is among the command arguments, else zero. Runs nothing.
            commands.append(command)
            return subprocess.CompletedProcess(
                command,
                returncode=int("infra.env.PYTHONPATH=/somewhere" in command),
                stdout="",
                stderr="",
            )

        monkeypatch.setattr(lint_charts, "run", run)

        assert lint_charts.lint_chart(chart)
        assert commands[-1] == [
            "helm",
            "lint",
            str(chart),
            "--set",
            "objectName=lint-app",
            "--set",
            "infra.env.PYTHONPATH=/somewhere",
        ]


def colocate_variant() -> list[str]:
    # [FUNCTION DOC] HELPER: Return the argument list of the colocate variant of the miles-run
    # chart from a freshly loaded script's variants table. The variant is the one entry containing
    # an argument that sets the colocate value. Unpacking raises an error if zero or several match.
    module = load_lint_charts()
    [variant] = [args for args in module.VARIANTS["miles-run"] if any(a.startswith("run.colocate=") for a in args)]
    return variant


def set_json_value(variant: list[str], prefix: str) -> object:
    # [FUNCTION DOC] HELPER: Extract and decode a JSON value from a variant's argument list.
    # Finds the single argument starting with the given prefix, strips the prefix and parses the
    # remainder as JSON. Returns the parsed object. Raises an error if zero or several arguments
    # match, or if the text is not valid JSON.
    [value] = [argument[len(prefix) :] for argument in variant if argument.startswith(prefix)]
    return json.loads(value)


class TestTheColocateVariant:
    # [CLASS DOC] Consistency checks on the colocate variant data of the miles-run chart, read
    # straight from the script's variants table (no helm needed).
    def test_names_pools_the_same_variant_renders(self):
        # [FUNCTION DOC] Parses the colocate setting and the inference-engine and trainer-engine
        # settings of the variant. Collects the names of all rendered engines, and the pool ids
        # named by the colocate setting (every inference pool id plus the trainer pool id).
        # Asserts the named ids are a subset of the rendered names; on failure the missing ids are
        # listed in sorted order in the assertion message.
        """A pool id no pool carries makes the variant a disaggregated run, and the gated branch goes unlinted."""
        colocate = set_json_value(colocate_variant(), "run.colocate=")
        rendered = {
            entry["name"]
            for key in ("run.inferenceEngines=", "run.trainerEngines=")
            for entry in set_json_value(colocate_variant(), key)
        }

        named = {pool["pool_id"] for pool in colocate["inference_pools"]} | {colocate["trainer_pool_id"]}
        assert named <= rendered, sorted(named - rendered)

    def test_covers_a_pool_narrower_than_a_node(self):
        # [FUNCTION DOC] Parses the colocate setting and asserts at least one inference pool has
        # fewer GPUs per inference pod than GPUs per node, so the sub-node branch is linted.
        """Only a sub-node pool renders the base gpu id env, so a whole-node variant never lints that branch."""
        colocate = set_json_value(colocate_variant(), "run.colocate=")

        assert any(
            pool["layout"]["num_gpus_per_inference_pod"] < pool["layout"]["num_gpus_per_node"]
            for pool in colocate["inference_pools"]
        )
