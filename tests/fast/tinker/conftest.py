# [MODULE DOC] Shared pytest fixtures for the tinker gateway tests: an async fixture providing a
# running gateway service backed by the fake backend.
import asyncio
from contextlib import suppress

import pytest

from tests.fast.tinker.harness import make_service


@pytest.fixture
async def service(tmp_path):
    # [FUNCTION DOC] Async generator fixture. Builds a tinker service (fake backend, default
    # two-slot test config, checkpoint root in the temporary directory), starts its run loop as a
    # background task, and yields the service. On teardown cancels the run task and awaits it,
    # swallowing the cancellation.
    gateway = make_service(tmp_path)
    run_task = asyncio.create_task(gateway.run())
    try:
        yield gateway
    finally:
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task
