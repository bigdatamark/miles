# [MODULE DOC] Tests that the cluster backend (ray or kubernetes) is a property of the run chosen
# by the launcher config or, for e2e scripts, by environment variables: both backends honor the
# same launcher contract (the config creates the backend it names, cpu commands run locally, the
# worker flag must agree with the config); a config-less e2e launch follows the cluster-backend
# and namespace environment variables and defaults to ray when unset; pre-launch commands reach
# the same backend; and no e2e module may construct its own launcher config (which would outrank
# the environment). Module constants: the repository root, the e2e test root and the trivial
# training argument string used for launches.
import ast
import shlex
from pathlib import Path

import pytest
from tests.fast.cluster_backends import both_backends, require_backend

import miles.utils.external_utils.command_utils as command_utils
from miles.utils.external_utils.command_utils.helm_backend.backend import KubernetesCommandBackend
from miles.utils.external_utils.command_utils.ray_backend.backend import RayCommandBackend
from miles.utils.workers.types import ClusterBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_ROOT = REPO_ROOT / "tests" / "e2e"

TRAIN_ARGS_OF_A_TRIVIAL_RUN = "--train-backend fsdp"


def _config(cluster_backend: str, namespace: str) -> command_utils.ExecuteTrainConfig:
    # [FUNCTION DOC] HELPER: Builds a launcher train config for the named cluster backend (parsed
    # into the backend enum; raises a value error for an unknown name), the namespace, and a fixed
    # run id of 260101-000000-000.
    return command_utils.ExecuteTrainConfig(
        cluster_backend=ClusterBackend(cluster_backend),
        namespace=namespace,
        run_id="260101-000000-000",
    )


class TestEveryBackendHonoursTheSameLauncherContract:
    # [CLASS DOC] Contract tests run once per backend through the both-backends decorator; each
    # first asks the require-backend helper, which skips the test when that backend is unavailable
    # and otherwise returns the namespace.
    @both_backends
    def test_the_config_creates_the_backend_it_names(self, cluster_backend):
        # [FUNCTION DOC] The backend object created from a config is of the class matching the
        # configured cluster backend (the ray command backend for ray, the kubernetes command
        # backend for kubernetes), compared by class name.
        """The backend is a run's property, so choosing it in the config has to be what actually takes effect."""
        namespace = require_backend(cluster_backend)

        backend = _config(cluster_backend, namespace).create_backend()

        assert (
            backend.__class__.__name__
            == {
                ClusterBackend.RAY.value: "RayCommandBackend",
                ClusterBackend.KUBERNETES.value: "KubernetesCommandBackend",
            }[cluster_backend]
        )

    @both_backends
    def test_a_cpu_command_runs_without_reaching_the_cluster(self, cluster_backend, monkeypatch):
        # [FUNCTION DOC] Replaces the backend instance's cpu command executor with a recorder
        # (patch reverted by monkeypatch), runs a trivial echo command, and asserts the recorder
        # received exactly that command. Note it only tests the replaced method, not real
        # execution.
        """Both backends run cpu work where the launcher already sits, so a script reads the same either way."""
        namespace = require_backend(cluster_backend)
        backend = _config(cluster_backend, namespace).create_backend()

        recorded: list[str] = []
        monkeypatch.setattr(backend, "exec_command_cpu", lambda cmd, capture_output=False: recorded.append(cmd))
        backend.exec_command_cpu("echo hello")

        assert recorded == ["echo hello"]

    @both_backends
    def test_the_worker_flag_and_the_config_must_agree(self, cluster_backend):
        # [FUNCTION DOC] Takes the other backend's name, then calls execute_train on a backend
        # built from the config with training arguments naming that other backend in the
        # cluster-backend flag. It must raise an assertion error mentioning the other backend's
        # name; if no assertion error is raised the test itself raises an assertion error.
        """The config drives the launcher and the flag drives the workers; disagreeing installs one and drives the other."""
        require_backend(cluster_backend)
        other = next(backend.value for backend in ClusterBackend if backend.value != cluster_backend)

        try:
            _config(cluster_backend, "miles-e2e").create_backend().execute_train(
                train_args=f"--train-backend fsdp --cluster-backend {other}",
                num_gpus_per_node=1,
                megatron_model_type=None,
            )
        except AssertionError as error:
            assert shlex.quote(other).strip("'") in str(error)
        else:
            raise AssertionError("a config and a train flag naming different backends must be refused")


def _e2e_modules() -> list[Path]:
    # [FUNCTION DOC] HELPER: Returns the sorted python files under the e2e test root, excluding
    # any path containing a bytecode-cache component.
    return sorted(path for path in E2E_ROOT.rglob("*.py") if "__pycache__" not in path.parts)


def _constructs_launcher_config(path: Path) -> bool:
    # [FUNCTION DOC] HELPER: Parses the file and returns true if any call node in the syntax tree
    # calls something named like the launcher train config class (by bare name or attribute).
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        isinstance(node, ast.Call) and _called_name(node.func) == command_utils.ExecuteTrainConfig.__name__
        for node in ast.walk(tree)
    )


def _called_name(func: ast.expr) -> str:
    # [FUNCTION DOC] HELPER: Returns the identifier of a bare-name callee, otherwise the attribute
    # name if the callee has one, else an empty string.
    if isinstance(func, ast.Name):
        return func.id
    return getattr(func, "attr", "")


def _capture_backend(monkeypatch) -> list[ClusterBackend]:
    # [FUNCTION DOC] HELPER: Replaces the inner train-execution method of both the ray and the
    # kubernetes command backend classes with a recorder (reverted by monkeypatch) that appends
    # the config's cluster backend to a list. Returns that list, which fills as launches happen.
    chosen: list[ClusterBackend] = []

    def _execute_train_inner(self, *, request, config) -> None:
        # [FUNCTION DOC] HELPER: Records the cluster backend of the config; does nothing else.
        chosen.append(config.cluster_backend)

    for backend_cls in (RayCommandBackend, KubernetesCommandBackend):
        monkeypatch.setattr(backend_cls, "_execute_train_inner", _execute_train_inner)
    return chosen


class TestATrainingE2eTestRunsOnWhicheverBackendItsEnvironmentNames:
    # [CLASS DOC] Tests that a launch using the default config picks the backend from the
    # environment (cluster backend and namespace variables) and that e2e modules never pin one.
    @pytest.mark.parametrize("backend", sorted(backend.value for backend in ClusterBackend))
    def test_a_test_that_passes_no_config_follows_the_environment(self, backend: str, monkeypatch):
        # [FUNCTION DOC] For each backend value: sets the cluster backend and namespace
        # environment variables, captures backend launches, runs execute_train from the default
        # config's backend with trivial arguments (one GPU per node, no model type), and asserts
        # the single captured backend equals the one named in the environment.
        """Every e2e script calls execute_train without a config, so the environment is the only dial there is."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", backend)
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-e2e")
        chosen = _capture_backend(monkeypatch)

        command_utils.default_config().create_backend().execute_train(
            train_args=TRAIN_ARGS_OF_A_TRIVIAL_RUN, num_gpus_per_node=1, megatron_model_type=None
        )

        assert chosen == [ClusterBackend(backend)]

    def test_an_unset_environment_still_means_ray(self, monkeypatch):
        # [FUNCTION DOC] With the cluster backend variable removed, a default-config launch is
        # captured as using ray.
        """Every existing ray runner sets nothing, and reading the environment must not have moved them."""
        monkeypatch.delenv("MILES_SCRIPT_CLUSTER_BACKEND", raising=False)
        chosen = _capture_backend(monkeypatch)

        command_utils.default_config().create_backend().execute_train(
            train_args=TRAIN_ARGS_OF_A_TRIVIAL_RUN, num_gpus_per_node=1, megatron_model_type=None
        )

        assert chosen == [ClusterBackend.RAY]

    def test_a_command_run_before_the_launch_reaches_the_same_backend(self, monkeypatch):
        # [FUNCTION DOC] With the environment naming kubernetes, the default config creates the
        # kubernetes command backend (by class name).
        """prepare() downloads and converts before execute_train, and under kubernetes those are Jobs too."""
        monkeypatch.setenv("MILES_SCRIPT_CLUSTER_BACKEND", ClusterBackend.KUBERNETES.value)
        monkeypatch.setenv("MILES_SCRIPT_NAMESPACE", "miles-e2e")

        backend = command_utils.default_config().create_backend()

        assert backend.__class__.__name__ == "KubernetesCommandBackend"

    @pytest.mark.parametrize("path", _e2e_modules(), ids=lambda path: path.relative_to(E2E_ROOT).as_posix())
    def test_no_e2e_module_pins_a_backend_of_its_own(self, path: Path):
        # [FUNCTION DOC] Parametrized over every python file under the e2e root (ids are relative
        # posix paths): asserts the file never constructs a launcher train config.
        """A config built in the test would outrank the environment and pin that test to one cluster forever."""
        assert not _constructs_launcher_config(path)
