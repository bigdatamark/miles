# [MODULE DOC]
# Helpers for "hot restart": a new orchestration script taking over workers (trainers and the
# inference controller) that a previous script left initialized and possibly busy. Includes waiting
# for a worker to be uninitialized, waiting for trainers to become idle and deciding whether to init
# or reload state, and taking over the inference controller (idle wait, cell count wait, abort of
# running generations, agent abort hook). Time limits are module constants in seconds.
from __future__ import annotations

import asyncio
import logging
from argparse import Namespace
from typing import Any

from miles.utils.init_once import InitState
from miles.utils.misc import call_agent_abort_hook
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.rpc.client.misc import RETRYABLE_ERRORS, ServerRestartedError
from miles.utils.workers.worker_handle import BaseWorkerHandle, WorkerUnreachableError

logger = logging.getLogger(__name__)


# ================================= constants ==================================

TAKE_OVER_GATE_TIMEOUT_SECONDS = 600.0
_TRAINER_RELOAD_TIMEOUT_SECONDS = 3600.0
_INFERENCE_IDLE_TIMEOUT_SECONDS = 3600.0
_WORKER_NOT_INITIALIZED_TIMEOUT_SECONDS = 1800.0
_WORKER_POLL_INTERVAL_SECONDS = 5.0
_WORKER_POLL_ATTEMPT_TIMEOUT_SECONDS = 120.0
_WORKER_READY_TIMEOUT_SECONDS = 60.0


# ============================== worker take-over ==============================


async def wait_until_worker_not_initialized(
    handle: BaseWorkerHandle, *, timeout: float = _WORKER_NOT_INITIALIZED_TIMEOUT_SECONDS
) -> None:
    # [FUNCTION DOC] Polls (fixed 5 second delay, each attempt limited to 120 seconds, total limit
    # the timeout, default 30 minutes) until the worker reports the not-initialized state, retrying
    # on still-initialized, unreachable, restarted-server and retryable rpc errors. Raises whatever
    # the retry helper raises at the deadline.
    async def attempt(remaining: float) -> None:
        # [FUNCTION DOC] HELPER: One probe: waits for readiness (up to 60 seconds, allowing server id
        # change), then raises the still-initialized error unless the init state is not-initialized.
        await handle.wait_ready(timeout=_WORKER_READY_TIMEOUT_SECONDS, allow_server_uuid_change=True)
        if (state := await handle.get_init_state()) != InitState.NOT_INITED.value:
            raise _StillInitializedError(f"the worker answers {state}, so a previous script already got hold of it")

    await retry_until_deadline(
        attempt,
        total_seconds=timeout,
        retry_on=(_StillInitializedError, WorkerUnreachableError, ServerRestartedError, *RETRYABLE_ERRORS),
        attempt_seconds=_WORKER_POLL_ATTEMPT_TIMEOUT_SECONDS,
        initial_delay=_WORKER_POLL_INTERVAL_SECONDS,
        backoff_factor=1.0,
        log_fields={"tag": "hot_restart"},
    )


class _StillInitializedError(Exception):
    # [CLASS DOC] HELPER: Signals a worker that is already initialized by a previous script; used as
    # a retryable condition.
    pass


# ============================= trainer take-over ==============================


async def wait_trainers_idle(handles: dict[str, BaseWorkerHandle]) -> bool:
    # [FUNCTION DOC] Takes a non-empty mapping from trainer id to handle. Queries initialization of
    # all concurrently and asserts they agree. Returns true if all are initialized (resumed), after
    # sequentially waiting for each to be idle (10 minute limit each); returns false if none are.
    assert handles

    initialized = await asyncio.gather(*[handle.is_initialized() for handle in handles.values()])
    assert len(set(initialized)) == 1, f"trainers disagree about being initialized: {list(initialized)}"
    [resumed] = set(initialized)

    if resumed:
        for trainer_id, handle in handles.items():
            logger.info(f"Waiting until trainer {trainer_id!r} finished the call the previous script left running")
            await handle.wait_idle(timeout=TAKE_OVER_GATE_TIMEOUT_SECONDS)

    return resumed


async def trainer_init_or_load_state(
    trainer: BaseWorkerHandle, model_args: Namespace, *, trainer_id: str, resumed: bool
) -> list[Any]:
    # [FUNCTION DOC] If not resumed, initializes the trainer with the model args and returns its
    # result. Otherwise loads the saved state with a one hour limit, logs and returns the start
    # rollout ids it reports.
    if not resumed:
        return await trainer.init(model_args)

    start_rollout_ids = await asyncio.wait_for(trainer.load_state(), timeout=_TRAINER_RELOAD_TIMEOUT_SECONDS)
    logger.info(f"Resumed the already-initialized trainer {trainer_id!r} at rollout ids {start_rollout_ids}")
    return start_rollout_ids


# ============================ inference take-over =============================


async def init_or_reset_inference_controller(inference_controller: BaseWorkerHandle, *, args: Namespace) -> None:
    # [FUNCTION DOC] If the controller is not initialized, initializes it and returns. Otherwise
    # takes it over: asserts the weight transfer mode is not disk-delta (its first update only
    # records a baseline); waits for idle (1 hour), waits for the expected cell count (10 minutes),
    # aborts all running generations (10 minutes), then runs the agent abort hook (10 minutes).
    if not await inference_controller.is_initialized():
        await inference_controller.init()
        return

    assert (
        args.update_weight_transfer_mode != "disk-delta"
    ), "Hot restart does not support disk-delta weight transfer because its first update only captures a baseline"

    logger.info("The inference controller outlived a previous orchestration script; taking it over as it is")

    await inference_controller.wait_idle(timeout=_INFERENCE_IDLE_TIMEOUT_SECONDS)

    await inference_controller.wait_expected_num_cells(timeout=TAKE_OVER_GATE_TIMEOUT_SECONDS)

    await asyncio.wait_for(inference_controller.abort_all(), timeout=TAKE_OVER_GATE_TIMEOUT_SECONDS)
    logger.info("Asked every engine of the fleet to abort the generations it was still running")

    await asyncio.wait_for(call_agent_abort_hook(args), timeout=TAKE_OVER_GATE_TIMEOUT_SECONDS)
