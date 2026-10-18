# [MODULE DOC]
# Tests of the Ray worker handle against a real local Ray actor: dispatching a call, unreachable
# errors after a kill, readiness and death waits, and serializability of the handle through the
# object store.
from __future__ import annotations

import pytest
import ray
from tests.fast.utils.workers.real_ray.conftest import kill_quietly

from miles.utils.workers.ray_worker_handle import RayWorkerHandle
from miles.utils.workers.worker_handle import WorkerUnreachableError

pytestmark = pytest.mark.asyncio


@ray.remote(num_cpus=0)
class _EchoActor:
    # [CLASS DOC] HELPER: Ray actor needing no cpu with a single echo method.
    def echo(self, *, value: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the integer given by keyword.
        return value


@pytest.fixture
def echo_actor(ray_local_mode):
    # [FUNCTION DOC] HELPER: Fixture creating an echo actor on the shared local Ray; yields it
    # and kills it quietly afterwards.
    actor = _EchoActor.remote()
    yield actor
    kill_quietly(actor)


class TestRayWorkerHandleOnRealRay:
    # [CLASS DOC] Behavior of the handle wrapping a live echo actor.
    async def test_dispatch_reaches_the_real_actor(self, echo_actor):
        # [FUNCTION DOC] Calls echo with 7 through the handle; asserts 7.
        """The magic method call round-trips through a live ray actor."""
        handle = RayWorkerHandle(echo_actor)

        assert await handle.echo(value=7) == 7

    async def test_a_killed_actor_is_reported_unreachable(self, echo_actor):
        # [FUNCTION DOC] Kills the actor, then asserts a call through the handle raises a
        # worker-unreachable error.
        """Calls to a dead actor surface as unreachable, which the kill path treats as success."""
        handle = RayWorkerHandle(echo_actor)
        ray.kill(echo_actor)

        with pytest.raises(WorkerUnreachableError):
            await handle.echo(value=1)

    async def test_wait_ready_returns_for_a_live_actor(self, echo_actor):
        # [FUNCTION DOC] Waits up to 30 s for readiness of a live actor; passes if it returns.
        """A constructed actor passes the readiness probe."""
        handle = RayWorkerHandle(echo_actor)

        await handle.wait_ready(timeout=30.0)

    async def test_wait_dead_confirms_a_killed_actor(self, echo_actor):
        # [FUNCTION DOC] Kills the actor and waits up to 30 s for death confirmation.
        """Death confirmation completes once ray reports the actor gone."""
        handle = RayWorkerHandle(echo_actor)
        ray.kill(echo_actor)

        await handle.wait_dead(timeout=30.0)

    async def test_the_handle_survives_a_ray_round_trip(self, echo_actor):
        # [FUNCTION DOC] Puts the handle into the object store and fetches it back; asserts the
        # restored handle still echoes 3.
        """Handles travel inside WorkerInfo through the object store, so they must serialize."""
        handle = RayWorkerHandle(echo_actor)

        restored = ray.get(ray.put(handle))

        assert await restored.echo(value=3) == 3
