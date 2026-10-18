# [MODULE DOC] HELPER: Shared pytest fixtures and tracking stand-ins for the utils test packages.
# Provides a fixture making the Ray job submission client fail to connect (restoring the real
# launcher-owned job runner so it hits that failure) and a fixture building a tracking manager with
# one working and one failing tracking backend, with the backend classes defined here.
from argparse import Namespace
from typing import Any

import pytest

from miles.utils.external_utils.ray_job import _run_launcher_owned_job
from miles.utils.tracking_utils.base import TrackingBackend, TrackingManager


@pytest.fixture
def unavailable_ray_job_client(commands: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Fixture that depends on a commands fixture (defined by the requesting
    # test package, typically the command recorder) so the recorder is installed first. It then puts
    # the real launcher-owned job runner back into the Ray job module (undoing the recorder's fake)
    # and replaces the module's job submission client class with a function that raises a runtime
    # error "Ray job service unavailable". Returns nothing; patches undone at teardown.
    from miles.utils.external_utils import ray_job

    def fail_connect(address: str) -> None:
        # [FUNCTION DOC] HELPER: Stand-in for the client constructor; ignores the address and always
        # raises a runtime error "Ray job service unavailable".
        raise RuntimeError("Ray job service unavailable")

    monkeypatch.setattr(ray_job, "_run_launcher_owned_job", _run_launcher_owned_job)
    monkeypatch.setattr(ray_job, "JobSubmissionClient", fail_connect)


@pytest.fixture
def partially_failing_tracking_manager() -> TrackingManager:
    # [FUNCTION DOC] HELPER: Fixture returning a tracking manager with two backends registered
    # under the names "working" (resource-holding backend) and "failing" (backend whose init raises),
    # each with the setting value "enabled".
    return TrackingManager({"working": (_TrackingResource, "enabled"), "failing": (_FailingTracking, "enabled")})


class _TrackingResource(TrackingBackend):
    # [CLASS DOC] HELPER: Tracking backend that registers itself in a shared list held by the
    # arguments while initialized, so tests can see which backends are live.
    def init(self, args: Namespace, *, primary: bool = True, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Keep a reference to the resources list from the arguments' resources
        # attribute (raises an attribute error if absent) and append this backend to it.
        self._resources = args.resources
        self._resources.append(self)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Ignore the metrics; does nothing.
        pass

    def finish(self) -> None:
        # [FUNCTION DOC] HELPER: Remove this backend from the shared resources list (raises a value
        # error if it is not in it).
        self._resources.remove(self)


class _FailingTracking(_TrackingResource):
    # [CLASS DOC] HELPER: Tracking backend whose initialization always fails.
    def init(self, args: Namespace, *, primary: bool = True, **kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Always raises a runtime error "tracking backend unavailable"
        # without registering anything.
        raise RuntimeError("tracking backend unavailable")
