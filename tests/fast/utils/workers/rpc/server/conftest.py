# [MODULE DOC] HELPER: Pytest configuration for the rpc server tests. Provides a fixture returning a
# factory that builds an rpc call executor around a fresh instance of a worker class.
from __future__ import annotations

from collections.abc import Callable

import pytest
from tests.fast.utils.workers.rpc.server.fake_workers import ExecutorUnderTest

from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.server.executor import RpcCallExecutor


@pytest.fixture
def make_executor() -> Callable[[type], ExecutorUnderTest]:
    # [FUNCTION DOC] HELPER: Fixture returning a builder function. The builder takes a worker class,
    # collects its rpc method specs, instantiates the worker with no arguments, creates an executor
    # over that worker and specs, and returns the executor together with the specs.
    def build(worker_cls: type) -> ExecutorUnderTest:
        # [FUNCTION DOC] HELPER: The builder described by the fixture; raises whatever spec collection
        # or worker construction raises.
        specs = collect_rpc_method_specs(worker_cls)
        return ExecutorUnderTest(executor=RpcCallExecutor(worker=worker_cls(), specs=specs), specs=specs)

    return build
