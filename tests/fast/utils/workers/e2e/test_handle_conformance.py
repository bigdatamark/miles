# [MODULE DOC]
# Runs the shared worker-handle conformance checks against a real serve subprocess hosting the
# conformance worker, through the rpc worker handle. One server is shared by all checks of the
# class because spawning it is expensive and the checks are read-only.
from __future__ import annotations

from collections.abc import Iterator

import pytest
from tests.fast.utils.workers import conformance
from tests.fast.utils.workers.conformance import CHECK_IDS, CHECKS, ConformanceWorker, HandleCheck

from tests.fast.utils.workers.e2e.harness import ServerProcess, spawn_server, wait_until_serving

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle
from miles.utils.workers.worker_handle import BaseWorkerHandle


# spawning the server costs a fresh interpreter that imports miles, and every check below
# only reads from it, so one server serves the whole class
@pytest.fixture(scope="class")
def conformance_server(tmp_path_factory) -> Iterator[ServerProcess]:
    # [FUNCTION DOC] HELPER: Class-scoped fixture. Creates a temporary root with a state
    # directory, spawns a serve process whose spec computation is the conformance module's, waits
    # until it serves, yields it, and afterwards stops then kills it.
    root = tmp_path_factory.mktemp("conformance")
    state_dir = root / "state"
    state_dir.mkdir()
    server = spawn_server(
        state_dir=state_dir,
        log_path=root / "server.log",
        specs_path=f"{conformance.__name__}.compute_specs",
    )
    wait_until_serving(server)
    yield server
    server.stop()
    server.kill()


@pytest.fixture
def conformance_handle(conformance_server: ServerProcess, make_handle) -> Iterator[BaseWorkerHandle]:
    # [FUNCTION DOC] HELPER: Per-test fixture yielding a handle for the conformance worker class
    # on the shared conformance server (http client closed by the handle factory fixture).
    yield make_handle(conformance_server, worker_cls=ConformanceWorker)


class TestTheHandleContractOverAServeSubprocess:
    # [CLASS DOC] The rpc column of the handle conformance matrix.
    @pytest.mark.parametrize("check", CHECKS, ids=CHECK_IDS)
    async def test_the_contract_holds(self, conformance_handle: RpcWorkerHandle, check: HandleCheck):
        # [FUNCTION DOC] Parametrized over every conformance check (shared and rpc-only); runs
        # the check against the handle. Any assertion inside the check fails the test.
        """Every backend a handle can sit on answers the same contract; this is the serve-subprocess column."""
        await check(conformance_handle)
