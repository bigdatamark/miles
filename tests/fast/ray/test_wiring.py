# [MODULE DOC]
# Unit tests for the wiring module that connects the driver to a cluster backend: launching the
# ray worker manager (a no-op under kubernetes), shutting it down and killing it (also when a
# graceful shutdown fails), a static source scan guaranteeing that every finite driver script
# hands its teardown to a disposer inside a single async entry call, capability lookup per
# cluster backend (ray manager handle versus the helm backed kubernetes capability), a guard
# against a top-level import cycle with the placement group module, and propagation of the
# resolved worker communication backend to the manager launch.
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.fast.ray.rollout.conftest import make_args
from tests.fast.source_scan import REPO_ROOT

from miles.ray import wiring
from miles.utils.workers.backend_capability import factory
from miles.utils.workers.backend_capability.ray import RayBackendCapability
from miles.utils.workers.types import ClusterBackend, WorkerCommBackend


async def _resolved(events: list[object], event: object) -> None:
    # [FUNCTION DOC] HELPER: Coroutine that appends the given event to the given list; used to
    # make a fake remote call awaitable while recording that it happened.
    events.append(event)


class TestLaunchWorkerManager:
    # [CLASS DOC] Tests the launch of the worker manager per cluster backend.
    def test_a_ray_run_launches_the_ray_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the ray cluster backend, the ray launcher (replaced by a recorder) is
        # called exactly once with the same args object.
        """The Ray path is the one every existing run takes, and it must be untouched."""
        launched: list[Any] = []
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", lambda args: launched.append(args))

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value)
        wiring.launch_worker_manager(args)

        assert launched == [args]

    def test_a_kubernetes_run_launches_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the kubernetes cluster backend, the ray launcher (replaced by one that
        # fails) is not called and the launch function returns none.
        """Under Kubernetes the pods already exist, so launching actors would double the run."""
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

        assert wiring.launch_worker_manager(SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)) is None


class TestShutdownWorkerManager:
    # [CLASS DOC] Tests the driver's shutdown of the worker manager and the static check that
    # driver scripts delegate teardown to a disposer.
    async def test_a_ray_run_shuts_down_and_kills_its_own_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With a fake manager whose remote shutdown records an event and a ray kill
        # replaced by a recorder, shutting down gives the event order: graceful shutdown, then a
        # kill of that same manager handle.
        """The driver must release the exact manager it launched instead of sweeping processes by name."""
        events: list[object] = []
        manager = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: _resolved(events, "shutdown")))
        monkeypatch.setattr(wiring.ray, "kill", lambda handle: events.append(("kill", handle)))

        await wiring.shutdown_worker_manager(manager)

        assert events == ["shutdown", ("kill", manager)]

    async def test_a_kubernetes_run_has_no_ray_manager_to_shut_down(self) -> None:
        # [FUNCTION DOC] Shutting down with no manager (none) completes without error and does nothing.
        """The shared driver cleanup remains a no-op when Kubernetes owns worker lifecycles."""
        await wiring.shutdown_worker_manager(None)

    async def test_a_failed_graceful_shutdown_still_kills_the_ray_manager(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] The fake manager's remote shutdown raises a runtime error. The shutdown
        # call must propagate that same exception object and still kill the manager exactly once.
        """A graceful shutdown failure propagates after Ray forcibly reclaims the same manager."""
        error = RuntimeError("graceful shutdown failed")

        async def fail_shutdown() -> None:
            # [FUNCTION DOC] HELPER: Fake remote shutdown that always raises the prepared error.
            raise error

        manager = SimpleNamespace(shutdown=SimpleNamespace(remote=fail_shutdown))
        killed: list[object] = []
        monkeypatch.setattr(wiring.ray, "kill", killed.append)

        with pytest.raises(RuntimeError) as exc_info:
            await wiring.shutdown_worker_manager(manager)

        assert exc_info.value is error
        assert killed == [manager]

    @pytest.mark.parametrize(
        "script",
        sorted(path for path in REPO_ROOT.glob("train*.py") if "init_orchestration_script" in path.read_text()),
        ids=lambda path: path.name,
    )
    def test_every_finite_driver_hands_its_teardown_to_a_disposer(self, script: Path) -> None:
        # [FUNCTION DOC] Parametrized over every top-level train script in the repository whose text
        # mentions the orchestration script initializer (collected when the module is imported). The
        # script is parsed and all call expressions are unparsed to text. It must contain the
        # initializer call passing the args and the disposer keyword; exactly one call whose text
        # starts with the asyncio run entry call must exist; and that entry must match the pattern
        # of running the with-disposer wrapper on a single name and the args. Reads source files only.
        """A driver that releases on its last line instead leaks the manager and the run's tracking on a raise."""
        calls = {
            ast.unparse(node)
            for node in ast.walk(ast.parse(script.read_text(), filename=str(script)))
            if isinstance(node, ast.Call)
        }
        entered = sorted(call for call in calls if call.startswith("asyncio.run("))

        assert "init_orchestration_script(args, disposer=disposer)" in calls
        assert len(entered) == 1
        assert re.fullmatch(r"asyncio\.run\(with_disposer\(\w+, args\)\)", entered[0]) is not None


class TestGetBackendCapability:
    # [CLASS DOC] Tests the backend capability returned per cluster backend.
    def test_a_ray_run_is_answered_from_the_worker_manager(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the ray launcher forbidden and the manager handle lookup returning a
        # plain object, the capability for a ray backend run is a ray backend capability instance.
        """The manager was launched by the driver's own first line; the capability only looks it up."""
        monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)
        monkeypatch.setattr(factory.RayWorkerManager, "get_handle", staticmethod(lambda: object()))

        args = make_args(cluster_backend=ClusterBackend.RAY.value)

        assert isinstance(wiring.get_backend_capability(args), RayBackendCapability)

    def test_a_kubernetes_run_is_answered_by_observing_the_namespace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] With the kubernetes stub installed, the capability returned for a
        # kubernetes backend run is the stub's capability object, and the specs were computed once
        # from exactly these args.
        """Under Kubernetes nothing was launched, so the capability is what observes the pods instead."""
        stub = stub_kubernetes_capability(monkeypatch)

        args = SimpleNamespace(cluster_backend=ClusterBackend.KUBERNETES.value)

        assert wiring.get_backend_capability(args) is stub.capability
        assert stub.specs_computed_from == [args]

    def test_the_driver_is_only_reached_when_a_ray_run_needs_its_placement_groups(self) -> None:
        # [FUNCTION DOC] Importing the placement group module works and exposes the placement group
        # creation function, while the wiring module's own namespace does not contain that function
        # (it must be imported lazily to avoid a cycle).
        """placement_group imports this module for the capability, so a top-level import back would deadlock both."""
        from miles.ray import placement_group

        assert placement_group.create_placement_groups is not None
        assert "create_placement_groups" not in vars(wiring)


class TestTheWireTheWorkerManagerIsLaunchedWith:
    # [CLASS DOC] Tests that the resolved worker communication backend is passed to the manager.
    @pytest.mark.parametrize("resolved", ["ray", "rpc"])
    def test_the_resolved_wire_reaches_the_worker_manager(
        self, monkeypatch: pytest.MonkeyPatch, resolved: str
    ) -> None:
        # [FUNCTION DOC] Parametrized over ray and rpc. With spec computation returning no specs,
        # placement group creation returning an empty mapping and the manager launch replaced by a
        # recorder, launching the ray worker manager with a worker comm backend of the given name
        # records exactly one launch with the corresponding communication backend enum value.
        """The flag only means anything if the wire validation resolved reaches the manager it starts."""
        from miles.ray import placement_group

        launched: list[WorkerCommBackend] = []
        monkeypatch.setattr(wiring, "compute_specs", lambda args: [])
        monkeypatch.setattr(placement_group, "create_placement_groups", lambda args: {})
        monkeypatch.setattr(
            wiring.RayWorkerManager,
            "launch",
            staticmethod(lambda args, specs, pgs, *, comm_backend: launched.append(comm_backend)),
        )

        args = SimpleNamespace(cluster_backend=ClusterBackend.RAY.value, worker_comm_backend=resolved)
        wiring._launch_ray_worker_manager(args)

        assert launched == [WorkerCommBackend(resolved)]


@dataclass
class KubernetesCapabilityStub:
    # [CLASS DOC] HELPER: Data holder for the kubernetes capability stub: the fake capability
    # object, and the list of args objects from which specs were computed.
    capability: object
    specs_computed_from: list[Any] = field(default_factory=list)


def stub_kubernetes_capability(monkeypatch: pytest.MonkeyPatch) -> KubernetesCapabilityStub:
    # [FUNCTION DOC] HELPER: Creates a stub holder with a plain object as capability, then
    # replaces: spec computation (records the args and returns no specs), the helm backend
    # capability factory (returns the stub capability), and the ray launcher (fails if called).
    # Returns the stub holder. Patching is undone at test end.
    stub = KubernetesCapabilityStub(capability=object())

    monkeypatch.setattr(wiring, "compute_specs", lambda args: stub.specs_computed_from.append(args) or [])
    monkeypatch.setattr(factory, "compute_helm_backend_capability", lambda **kwargs: stub.capability)
    monkeypatch.setattr(wiring, "_launch_ray_worker_manager", _refuse_ray)

    return stub


def _refuse_ray(args: Any) -> None:
    # [FUNCTION DOC] HELPER: Replacement for the ray launcher that raises an assertion error saying
    # the kubernetes path must not launch ray workers.
    raise AssertionError("the Kubernetes path must not launch Ray workers")
