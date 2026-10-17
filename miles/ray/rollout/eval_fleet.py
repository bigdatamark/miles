# [MODULE DOC] Dedicated in-job evaluation engine fleet. Two sides: the inference controller side
# (class InferenceControllerEvalFleet) loads a weight snapshot from disk onto every fleet engine
# and verifies versions; the executor side (class RolloutExecutorEvalFleet) asks the controller over
# RPC to pin and then hands out a generation state targeting the fleet's router. Also defines the
# frozen data carriers for fleet description and pin result, a tuple of error types that mean the
# controller cannot be reached, and the 600 second weight load timeout.
"""The dedicated in-job eval engines (``--eval-num-gpus``).

Weight delivery only: ``pin`` loads a snapshot onto every engine and verifies each
one reports the expected version. The fleet lives beside its engines, in the
inference controller; the executor holds an ``RolloutExecutorEvalFleet`` over the wire and
builds the state to generate against from the fleet's description. Who generates is
the eval fn's business, exactly as on the training engines.
"""

import asyncio
import dataclasses
import logging
from argparse import Namespace

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.ray.specs.inference import inference_controller_worker_name
from miles.rollout.checkpoint_eval import EvalSkip, retarget_args
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.http_utils import wait_http_ok
from miles.utils.workers.rpc.client.misc import ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError
from miles.utils.workers.worker_provider.base import BaseWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort

logger = logging.getLogger(__name__)

EVAL_WEIGHT_LOAD_TIMEOUT_SECS = 600.0


@dataclasses.dataclass(frozen=True)
class EvalFleetInfo:
    # [CLASS DOC] Frozen description of the eval fleet sent from controller to executor: the
    # router's host and port, the total eval GPU count and GPUs per engine.
    router: HostAndPort
    num_gpus: int
    num_gpus_per_engine: int


@dataclasses.dataclass(frozen=True)
class EvalFleetPin:
    # [CLASS DOC] Frozen result of a pin attempt: nothing as skip reason means the fleet can be
    # generated against; otherwise a short reason string (pin_violation or unhealthy).
    skip_reason: str | None


UNREACHABLE_CONTROLLER_ERRORS = (WorkerUnreachableError, ServerRestartedError, TimeoutError)


class RolloutExecutorEvalFleet:
    # [CLASS DOC] Executor-side proxy of the fleet. Holds the RPC handle to the inference controller
    # and one generation state built once at construction (reused for every evaluation).
    """The executor's side of the fleet: one state to generate against, pinned over rpc."""

    def __init__(
        self, args: Namespace, *, info: EvalFleetInfo, inference_controller_provider: BaseWorkerProvider
    ) -> None:
        # [FUNCTION DOC] Look up the inference controller handle by its well-known worker name from
        # the provider, and create the generation state from args retargeted at the fleet's router
        # host and port and the fleet's GPU counts (see the retarget helper in checkpoint eval).
        self._inference_controller: BaseWorkerHandle = inference_controller_provider.get_handle(
            inference_controller_worker_name()
        )
        self._state = GenerateState(
            retarget_args(args, info.router.host, info.router.port, info.num_gpus, info.num_gpus_per_engine)
        )

    async def pin(self, checkpoint_dir: str, weight_version: str) -> GenerateState:
        # [FUNCTION DOC] Ask the controller over RPC to load the checkpoint directory onto all fleet
        # engines at the given weight version. Returns the shared generation state when the
        # controller reports no skip reason. Raises an eval-skip exception with reason
        # "controller_unreachable" if the RPC fails with an unreachable-worker, server-restarted or
        # timeout error (chained from the cause), or with the controller's skip reason otherwise.
        try:
            pin = await self._inference_controller.pin_eval_fleet(
                checkpoint_dir=checkpoint_dir, weight_version=weight_version
            )
        except UNREACHABLE_CONTROLLER_ERRORS as e:
            logger.warning(f"Eval fleet controller could not be reached: {e!r}")
            raise EvalSkip("controller_unreachable") from e

        if (skip_reason := pin.skip_reason) is not None:
            raise EvalSkip(skip_reason)
        return self._state


class InferenceControllerEvalFleet:
    # [CLASS DOC] Controller-side fleet: wraps the server object (srv) that owns the eval engines'
    # router address, a context lock and the list of per-engine API clients.
    """The dedicated in-job eval engines (``--eval-num-gpus``)."""

    def __init__(self, args: Namespace, *, srv):
        # [FUNCTION DOC] Store args and the server object.
        self.args = args
        self._srv = srv

    @property
    def info(self) -> EvalFleetInfo:
        # [FUNCTION DOC] Property: build the fleet description from the server's router ip and port
        # and the eval GPU counts in args. Computed fresh on each access.
        return EvalFleetInfo(
            router=HostAndPort(host=self._srv.router_ip, port=self._srv.router_port),
            num_gpus=self.args.eval_num_gpus,
            num_gpus_per_engine=self.args.eval_num_gpus_per_engine,
        )

    async def pin(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        # [FUNCTION DOC] Load the snapshot onto all engines (with retries) and then wait for the
        # router to be usable. Returns a pin result: skip reason "pin_violation" if loading or
        # version verification failed; "unhealthy" if the router did not become ready (any
        # exception is swallowed and logged as a warning); otherwise no skip reason. Never raises
        # for those failures.
        """Load the snapshot onto every engine, then report whether it can be generated against.

        On the controller's event loop: keep everything here awaiting rather than blocking.
        """
        if not await self._pin_fleet(checkpoint_dir, weight_version):
            return EvalFleetPin(skip_reason="pin_violation")

        try:
            await self._wait_router_ready()
        except Exception as e:
            logger.warning(f"Eval router not ready: {e}")
            return EvalFleetPin(skip_reason="unhealthy")

        return EvalFleetPin(skip_reason=None)

    async def _pin_fleet(self, checkpoint_dir: str, weight_version: str, *, retries: int = 2) -> bool:
        # [FUNCTION DOC] HELPER: For up to the retry count attempts (default 2): concurrently ask every
        # engine's api client to update weights from the checkpoint directory at the given version,
        # all under a 600 second timeout; then concurrently query every engine's weight version
        # under the same timeout. The client list is re-read before each phase. Any exception in an
        # attempt is logged and the attempt retried (no delay). An attempt succeeds when the version
        # list is non-empty and every reported version, converted to text, equals the target.
        # Returns true on success; after all attempts fail logs a warning with the last versions
        # and returns false. Note a failed attempt that leaves versions from a prior attempt in
        # place is possible but only the final value is reported.
        """Load the snapshot into every fleet engine and confirm all report
        ``weight_version`` — the router load-balances across engines, so a single
        stale engine would mix versions. Never raises: transient failures and
        mismatches are retried, then ``False`` lets the caller skip the point."""
        versions: list = []
        for attempt in range(retries):
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *[
                            client.update_weights_from_disk(model_path=checkpoint_dir, weight_version=weight_version)
                            for client in await self._fleet_api_clients()
                        ]
                    ),
                    timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                )
                versions = await asyncio.wait_for(
                    asyncio.gather(*[client.get_weight_version() for client in await self._fleet_api_clients()]),
                    timeout=EVAL_WEIGHT_LOAD_TIMEOUT_SECS,
                )
            except Exception as e:
                logger.warning(f"Weight pin to {checkpoint_dir} failed (attempt {attempt + 1}/{retries}): {e}")
                continue
            if versions and all(str(v) == weight_version for v in versions):
                return True
        logger.warning(f"Failed to pin weight_version={weight_version} to {checkpoint_dir} (got {versions})")
        return False

    async def _fleet_api_clients(self) -> list[SGLangApiClient]:
        # [FUNCTION DOC] HELPER: Return a snapshot copy of the server's engine api clients taken while
        # holding the server's context lock.
        async with self._srv.context_lock:
            return list(self._srv.api_clients)

    async def _wait_router_ready(self, timeout: float = 180.0) -> None:
        # [FUNCTION DOC] HELPER: Poll the router's generate endpoint (http, router ip and port) with a
        # one-token greedy request (a single input token id zero, max one new token, temperature
        # zero) until it answers successfully or the timeout (default 180 seconds) elapses; the
        # wait helper raises on timeout. Sends real inference requests to the router.
        """After a revival the router 503s until its health cycle evicts the dead
        worker; a retried one-token probe proves the route is usable before dispatch."""
        await wait_http_ok(
            f"http://{self._srv.router_ip}:{self._srv.router_port}/generate",
            json_payload={"input_ids": [0], "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
            timeout=timeout,
        )
