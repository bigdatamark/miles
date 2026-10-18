# [MODULE DOC] HELPER:
# Shared conformance suite for worker handles. Defines a small test worker class, a function that
# builds the serve-worker spec for it, and a list of async check functions that each take a worker
# handle and assert one behavior of the handle contract (typed results, positional and keyword
# arguments, async methods, constructed state, out-of-process execution, error propagation,
# argument validation, readiness). Backends (raw rpc, Ray) run the same checks. The checks are
# split into those valid for every backend and those only valid over rpc; matching test ids are
# derived from the check names with leading underscores removed.
from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from miles.utils.workers.rpc.client.misc import RpcWorkerCallError
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

POOL_ID = "e2e-pool"
RPC_PORT_FLAG = "--rpc-port"

REPO_ROOT = Path(__file__).resolve().parents[4]

READY_TIMEOUT_SECONDS = 120.0


class ConformanceWorker:
    # [CLASS DOC] HELPER: Trivial worker served by the conformance tests. Remembers a tag given at
    # construction and offers methods for arithmetic, reporting its tag, process id and Ray actor
    # id, and raising an error on demand.
    def __init__(self, *, tag: str) -> None:
        # [FUNCTION DOC] HELPER: Stores the tag string (built by the spec from cell and worker
        # indexes).
        self._tag = tag

    def add(self, a: int, b: int) -> int:
        # [FUNCTION DOC] HELPER: Returns the sum of two integers.
        return a + b

    async def add_async(self, *, a: int, b: int) -> int:
        # [FUNCTION DOC] HELPER: Async variant of the sum; keyword-only arguments.
        return a + b

    def report_tag(self) -> str:
        # [FUNCTION DOC] HELPER: Returns the tag given at construction.
        return self._tag

    def report_pid(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the operating-system process id of the worker process.
        return os.getpid()

    def report_ray_actor_id(self) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the current Ray actor id if Ray is initialized in the
        # worker process, otherwise nothing. Imports Ray lazily.
        import ray

        return ray.get_runtime_context().get_actor_id() if ray.is_initialized() else None

    def boom(self, *, message: str) -> None:
        # [FUNCTION DOC] HELPER: Always raises a runtime error carrying the given message.
        raise RuntimeError(message)


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] HELPER: Spec-computation entry point used by serve launchers. Parses the
    # rpc port from the worker command line (see the port parser's assertion) and returns a
    # one-element list with the conformance spec for that port.
    return [compute_spec(rpc_port=_parse_rpc_port(worker_argv))]


def compute_spec(*, rpc_port: int) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds the serve-worker spec of the conformance pool: one cell with
    # one worker using zero gpus, one port named rpc (static at the given port; dynamic
    # allocation allowed only when the port is zero), and the conformance worker class located
    # through this module's name. The worker environment prepends the repository root to the
    # inherited python path (prepending keeps the inherited entries that provide the training
    # and inference libraries). Constructor arguments give each worker a tag made of its cell
    # index and worker-in-cell index.
    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=rpc_port, allow_dynamic=rpc_port == 0)],
        # prepend rather than replace: the inherited path is what carries sglang and megatron,
        # which the worker's own imports reach through miles.utils.arguments
        env_var=lambda _ctx: {"PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])},
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=f"{__name__}.ConformanceWorker",
        ctor_kwargs=lambda ctx: dict(tag=f"cell-{ctx.cell_index}-worker-{ctx.worker_in_cell_index}"),
    )


async def _call_returns_declared_type(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that a keyword call to add with 3 and 4 returns the integer 7.
    result = await handle.add(a=3, b=4)
    assert result == 7 and isinstance(result, int)


async def _positional_arguments_reach_worker(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that positional arguments 3 and 4 to add give 7.
    assert await handle.add(3, 4) == 7


async def _async_method_answers(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that the async add method with 1 and 2 returns 3.
    assert await handle.add_async(a=1, b=2) == 3


async def _worker_keeps_constructed_state(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that the reported tag equals "cell-0-worker-0", i.e. the
    # constructor arguments from the spec reached the worker.
    assert await handle.report_tag() == "cell-0-worker-0"


async def _call_runs_outside_caller(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that the worker's process id differs from the caller's.
    assert await handle.report_pid() != os.getpid()


async def _remote_failure_reaches_caller(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that the worker's error from the boom method surfaces to the
    # caller as an rpc worker-call error whose message matches the original text "deliberate".
    with pytest.raises(RpcWorkerCallError, match="deliberate"):
        await handle.boom(message="deliberate")


async def _unknown_method_refused_before_wire(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that calling a method the worker class lacks raises an
    # attribute error locally, without a remote call.
    with pytest.raises(AttributeError):
        await handle.add_typo(a=1, b=2)


async def _wrong_argument_type_refused(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that passing a string where an integer is declared raises a
    # pydantic validation error (rpc handles only).
    with pytest.raises(ValidationError):
        await handle.add(a="three", b=4)


async def _readiness_is_answered(handle: BaseWorkerHandle) -> None:
    # [FUNCTION DOC] HELPER: Check that waiting for readiness completes within the 120-second
    # ready timeout (an expiry error would fail the check).
    await handle.wait_ready(timeout=READY_TIMEOUT_SECONDS)


HandleCheck = Callable[[BaseWorkerHandle], Awaitable[None]]

SHARED_CHECKS: list[HandleCheck] = [
    _call_returns_declared_type,
    _positional_arguments_reach_worker,
    _async_method_answers,
    _worker_keeps_constructed_state,
    _call_runs_outside_caller,
    _unknown_method_refused_before_wire,
    _readiness_is_answered,
]

RPC_ONLY_CHECKS: list[HandleCheck] = [
    _remote_failure_reaches_caller,
    _wrong_argument_type_refused,
]

CHECKS: list[HandleCheck] = [*SHARED_CHECKS, *RPC_ONLY_CHECKS]

SHARED_CHECK_IDS = [check.__name__.lstrip("_") for check in SHARED_CHECKS]

CHECK_IDS = [check.__name__.lstrip("_") for check in CHECKS]


def _parse_rpc_port(worker_argv: list[str]) -> int:
    # [FUNCTION DOC] HELPER: Returns the integer following the rpc-port flag in the argument
    # list. Fails an assertion if the flag is absent. A flag at the very end raises an index
    # error and a non-numeric value raises a value error.
    assert RPC_PORT_FLAG in worker_argv, f"{worker_argv} must name the port the conformance worker serves on"
    return int(worker_argv[worker_argv.index(RPC_PORT_FLAG) + 1])
