# [MODULE DOC] Unit tests for the deployment entrypoint that serves a split deployment (a deployment
# carrying only some components). Collaborators of the entrypoint module (logger setup, worker
# manager launch, spec computation, api server start, mini fault-tolerance controller, backend
# capability, trainer handle creation, trainer id computation) are replaced by a recording fake. The
# tests check that the deployed workers are launched and then kept alive forever, that a deployment
# carrying the orchestration script is refused, when the fault tolerance api server is started, and
# (by parsing source) that every driver script runs its own orchestration without consulting the
# deploy component setting.
from __future__ import annotations

import ast
import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.charts.utils import REPO_ROOT

from miles.utils.workers import deployment_entrypoint
from miles.utils.workers.types import DeployComponent


class _FakeDeployment:
    # [CLASS DOC] HELPER: Recording fake for the collaborators of the deployment entrypoint. It keeps
    # an ordered list of the names of methods called and the keyword arguments of api server starts.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Starts with empty call and api server call lists.
        self.calls: list[str] = []
        self.api_server_calls: list[dict] = []

    def launch_worker_manager(self, args) -> str:
        # [FUNCTION DOC] HELPER: Records the call and returns the text "worker-manager".
        self.calls.append("launch_worker_manager")
        return "worker-manager"

    def compute_specs(self, args) -> list[SimpleNamespace]:
        # [FUNCTION DOC] HELPER: Records the call and returns one spec-like object named
        # "trainer-controller".
        self.calls.append("compute_specs")
        return [SimpleNamespace(name="trainer-controller")]

    def start_api_server(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Records the call and stores the keyword arguments for later assertions.
        self.calls.append("start_api_server")
        self.api_server_calls.append(kwargs)

    def maybe_start_mini_ft_controller(self, args) -> None:
        # [FUNCTION DOC] HELPER: Records the call; does nothing else.
        self.calls.append("maybe_start_mini_ft_controller")


def _injected(monkeypatch, fake: _FakeDeployment) -> None:
    # [FUNCTION DOC] HELPER: Replaces, in the deployment entrypoint module, the logger configuration
    # (no-op), worker manager launch, spec computation, api server start and mini controller start
    # with the fake's methods; the backend capability with a stand-in whose cell operations return
    # "ops"; the trainer handle creation with one returning "trainer-handle"; and the trainer id
    # computation with one returning the single id "actor".
    monkeypatch.setattr(deployment_entrypoint, "configure_logger", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(deployment_entrypoint, "launch_worker_manager", fake.launch_worker_manager)
    monkeypatch.setattr(deployment_entrypoint, "compute_specs", fake.compute_specs)
    monkeypatch.setattr(deployment_entrypoint, "start_api_server", fake.start_api_server)
    monkeypatch.setattr(deployment_entrypoint, "maybe_start_mini_ft_controller", fake.maybe_start_mini_ft_controller)
    monkeypatch.setattr(
        deployment_entrypoint, "get_backend_capability", lambda args: SimpleNamespace(cell_operations=lambda: "ops")
    )
    monkeypatch.setattr(
        deployment_entrypoint, "create_trainer_controller_handle", lambda *_args, **_kwargs: "trainer-handle"
    )
    monkeypatch.setattr(deployment_entrypoint, "compute_trainer_ids", lambda args: ["actor"])


def _args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds an args namespace with deploy component "trainer", api server port
    # 0 and no fault tolerance components, with any given overrides replacing those fields.
    fields = dict(deploy_component="trainer", api_server_port=0, ft_components=[])
    fields.update(overrides)
    return Namespace(**fields)


class TestServeDeployedWorkers:
    # [CLASS DOC] Tests the private serve-deployed-workers coroutine.
    async def test_it_launches_the_run_s_workers_and_then_stays_up(self, monkeypatch):
        # [FUNCTION DOC] Starts the coroutine as a task and yields to the loop up to 100 times until
        # two calls were recorded. Asserts the calls are the worker manager launch then spec
        # computation, and the task is still running; then cancels it and expects a cancellation error.
        """This deployment has no training to finish, so serving it means launching and then never returning."""
        fake = _FakeDeployment()
        _injected(monkeypatch, fake)

        task = asyncio.ensure_future(deployment_entrypoint._serve_deployed_workers(_args()))
        for _ in range(100):
            await asyncio.sleep(0)
            if len(fake.calls) == 2:
                break

        assert fake.calls == ["launch_worker_manager", "compute_specs"]
        assert not task.done()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_a_deployment_that_carries_the_orchestration_script_is_refused(self, monkeypatch):
        # [FUNCTION DOC] With deploy component "all", the coroutine raises an assertion error saying
        # it carries no orchestration script, and nothing was called on the fake.
        """Such a deployment is started by its driver script, and serving it here would run it twice."""
        fake = _FakeDeployment()
        _injected(monkeypatch, fake)

        with pytest.raises(AssertionError, match="carries no orchestration script"):
            await deployment_entrypoint._serve_deployed_workers(_args(deploy_component="all"))

        assert fake.calls == []


class TestServingTheFaultToleranceOfItsOwnCells:
    # [CLASS DOC] Tests the private maybe-serve-fault-tolerance function, which starts the api server
    # and mini controller only for a trainer deployment with a nonzero api server port.
    def test_a_trainer_deployment_answers_for_its_own_ranks(self, monkeypatch):
        # [FUNCTION DOC] For a trainer component with port 18080 and train fault tolerance, the calls
        # are api server start then mini controller start; the api server received the train
        # components, a trainer model mapping from "actor" to the trainer handle, and no inference
        # controller.
        """Nothing else can: the mini controller dials 127.0.0.1, so cells are only reachable from their own half."""
        fake = _FakeDeployment()
        _injected(monkeypatch, fake)

        deployment_entrypoint._maybe_serve_fault_tolerance(
            _args(api_server_port=18080, ft_components=["train"]), component=DeployComponent.TRAINER
        )

        assert fake.calls == ["start_api_server", "maybe_start_mini_ft_controller"]
        assert fake.api_server_calls[0]["ft_components"] == ["train"]
        assert fake.api_server_calls[0]["trainer_models"] == {"actor": "trainer-handle"}
        assert fake.api_server_calls[0]["inference_controller"] is None

    def test_a_deployment_without_an_api_server_port_starts_nothing(self, monkeypatch):
        # [FUNCTION DOC] With api server port 0 nothing is called.
        """Port zero is how a run says it wants no api server, and a split run must not override that."""
        fake = _FakeDeployment()
        _injected(monkeypatch, fake)

        deployment_entrypoint._maybe_serve_fault_tolerance(
            _args(api_server_port=0, ft_components=["train"]), component=DeployComponent.TRAINER
        )

        assert fake.calls == []

    def test_a_deployment_that_carries_no_trainer_starts_nothing(self, monkeypatch):
        # [FUNCTION DOC] With the inference component and a nonzero port nothing is called.
        """An inference deployment has no ranks of the kind this api server knows how to suspend."""
        fake = _FakeDeployment()
        _injected(monkeypatch, fake)

        deployment_entrypoint._maybe_serve_fault_tolerance(
            _args(api_server_port=18080, ft_components=["train"]), component=DeployComponent.INFERENCE
        )

        assert fake.calls == []


_DRIVER_SCRIPTS = [
    "train.py",
    "train_async.py",
    "train_multi_policy.py",
]


class TestEveryDriverRunsItsOwnOrchestrationScript:
    # [CLASS DOC] Source-level checks on the repository's driver scripts.
    @pytest.mark.parametrize("script", _DRIVER_SCRIPTS)
    def test_a_driver_starts_its_training_without_asking_what_this_launch_deploys(self, script):
        # [FUNCTION DOC] Parametrized over the three driver scripts at the repository root. Asserts that
        # the call to run an event loop appears among the calls made inside if-blocks at module level,
        # and that the script's text never mentions the deploy component setting.
        """A deployment carrying no orchestration script is served by its own entrypoint, never by a driver."""
        assert "asyncio.run" in _functions_called_in_main(script)
        assert "deploy_component" not in (REPO_ROOT / script).read_text()


def _functions_called_in_main(script: str) -> set[str]:
    # [FUNCTION DOC] HELPER: Parses the named script (relative to the repository root) into a syntax
    # tree and returns the set of source texts of the called expressions for every call found anywhere
    # inside a module-level if statement (typically the main guard). Reads a file; may raise a syntax
    # error or file-not-found error.
    tree = ast.parse((REPO_ROOT / script).read_text())
    return {
        ast.unparse(node.func)
        for statement in tree.body
        if isinstance(statement, ast.If)
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    }
