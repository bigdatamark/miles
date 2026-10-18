# [MODULE DOC] HELPER:
# Shared pytest fixtures and helpers for the worker unit tests: an in-process fake Ray cluster
# patched into the Ray worker manager module, and a builder for the minimal training args that
# the worker manager reads.
from types import SimpleNamespace

import pytest
from tests.fast.utils.workers.fake_ray import FakeRayCluster, FakeRayModule


@pytest.fixture
def fake_ray_cluster(monkeypatch) -> FakeRayCluster:
    # [FUNCTION DOC] HELPER: Fixture. Creates a fake cluster and a fake Ray module bound to it,
    # and replaces the Ray reference inside the Ray worker manager module with the fake for the
    # duration of the test (undone by monkeypatch). Returns the cluster for inspection.
    """In-process stand-in for Ray, letting the manager's whole launch pipeline run without a cluster."""
    import miles.utils.workers.ray_worker_manager as ray_worker_manager_mod

    cluster = FakeRayCluster()
    fake_ray = FakeRayModule(cluster=cluster)
    monkeypatch.setattr(ray_worker_manager_mod, "ray", fake_ray)
    return cluster


def worker_manager_args(**overrides) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Returns a simple attribute namespace with defaults of no debug event
    # data saving and an empty env report, updated by any keyword overrides.
    """The slice of a training run's args the worker manager reads when it configures its logger."""
    return SimpleNamespace(**{"save_debug_event_data": None, "env_report": "", **overrides})
