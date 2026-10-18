# [MODULE DOC]
# HELPER: Grab bag of small runtime utilities: a mutable box, dict merge with consistency check,
# agent abort hook invocation, a singleton metaclass, node ip / external ip discovery, free port
# block search (scanning upward with wrap-around above ray's port range), gpu uuid probing and
# visible-device remapping, a mixin exposing these as overridable static probes, periodic action
# scheduling, async helpers (as-completed generator, ticker, cancel-and-await) and key filtering.
import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import ray

from miles.utils.function_registry import load_function
from miles.utils.http_utils import MILES_HOST_IP_ENV, is_port_available

logger = logging.getLogger(__name__)

# ray uses 10002-19999, and 32768+ is the ephemeral range, so wrapped scans restart above ray's block
_MIN_DYNAMIC_PORT = 20000
_MAX_PORT = 65535

_K = TypeVar("_K")
_V = TypeVar("_V")
_T = TypeVar("_T")


@dataclass
class MutableBox(Generic[_T]):
    # [CLASS DOC] HELPER: Mutable holder of a single value, for sharing mutation across closures.
    value: _T


def merge_asserting_consistency(a: dict[_K, _V], b: dict[_K, _V]) -> dict[_K, _V]:
    # [FUNCTION DOC] HELPER: Returns the union of two mappings (new mapping). Asserts that keys present
    # in both have equal values; the message lists the conflicting pairs.
    conflicts = {key: (a[key], b[key]) for key in a.keys() & b.keys() if a[key] != b[key]}
    assert not conflicts, f"cannot merge two dicts that disagree: {conflicts}"
    return a | b


async def call_agent_abort_hook(args) -> None:
    # [FUNCTION DOC] Returns immediately if no custom agent function path is set or it has no module
    # part. Otherwise loads a callable named abort in the same module as the agent function (a
    # missing attribute or module is silently ignored) and awaits it with args; any exception from
    # the hook is logged as a warning and swallowed.
    """Invoke the agent plugin's optional abort hook, if it defines one.

    When oversampling collects enough samples, the rollout aborts SGLang, but an
    external agent loop (driven by ``--custom-agent-function-path``) keeps running
    and keeps issuing fresh completion requests until it hits its own limit. The
    agent integration knows how to tell its backend to stop, so we look for a
    sibling ``abort`` callable in the same module as the configured agent function
    and call it. Backends that don't expose one are left to drain as before.
    """
    agent_function_path = getattr(args, "custom_agent_function_path", None)
    if not agent_function_path:
        return

    module_path, _, _ = agent_function_path.rpartition(".")
    if not module_path:
        return
    try:
        abort_hook = load_function(f"{module_path}.abort")
    except (AttributeError, ModuleNotFoundError):
        return  # plugin doesn't expose an abort hook; nothing to tear down

    try:
        await abort_hook(args)
    except Exception as e:
        logger.warning(f"Agent abort hook {module_path}.abort failed: {e}")


class SingletonMeta(type):
    # [CLASS DOC] HELPER: Metaclass giving each class using it a single lazily created instance,
    # kept in a process-wide table (shared across all classes; not thread safe). Later constructor
    # arguments are ignored.
    """
    A metaclass for creating singleton classes.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Creates the instance on first call, then always returns it.
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @staticmethod
    def clear_all_instances():
        # [FUNCTION DOC] HELPER: Forgets every singleton instance (used by tests).
        SingletonMeta._instances.clear()


def get_current_node_ip():
    # [FUNCTION DOC] HELPER: Ip of this node as reported by ray (private ray API), with ipv6 square
    # brackets removed.
    address = ray._private.services.get_node_ip_address()
    # strip ipv6 address
    address = address.strip("[]")
    return address


MILES_NODE_EXTERNAL_IP_ENV = "MILES_NODE_EXTERNAL_IP"


def get_node_external_ip() -> str | None:
    # [FUNCTION DOC] HELPER: Value of the external ip environment variable, or none when unset or
    # empty.
    """The address off-cluster peers reach this node on (a DNS name also works),
    or None when its own address already routes from there.

    The deployment supplies this per node -- a tailscale sidecar or an init
    container writing the interface it joined. It is never forwarded through the
    job's runtime env, which would give every node one shared value; one host for
    every session server is ``--session-server-external-host`` instead.
    """
    return os.environ.get(MILES_NODE_EXTERNAL_IP_ENV) or None


def get_free_port(start_port=10000, consecutive=1):
    # [FUNCTION DOC] HELPER: Returns the first port p at or above the start such that p through p plus
    # consecutive minus one are all bindable. Asserts the start leaves room below 65535. Scans upward
    # one port at a time; past the highest valid start wraps to the lower of the start and 20000
    # (above ray's own range). Tries each candidate at most once; raises a runtime error naming the
    # range otherwise. Racy: the ports are free only at the time of the check.
    # find the port where port, port + 1, port + 2, ... port + consecutive - 1 are all available,
    # scanning upwards from start_port and wrapping around once the ports run out
    highest_start = _MAX_PORT - consecutive + 1
    assert start_port <= highest_start, f"{start_port=} leaves no room for {consecutive=} ports below {_MAX_PORT}"
    lowest_start = min(start_port, _MIN_DYNAMIC_PORT)

    port = start_port
    for _ in range(highest_start - lowest_start + 1):
        if all(is_port_available(port + i) for i in range(consecutive)):
            return port
        port = port + 1 if port < highest_start else lowest_start

    raise RuntimeError(f"No {consecutive} consecutive free ports in [{lowest_start}, {_MAX_PORT}]")


def get_gpu_uuids(gpu_ids: list[int]) -> list[str | None]:
    # [FUNCTION DOC] HELPER: Initializes nvml (never shut down) and returns the uuid text of each
    # physical gpu index. Any failure (library missing, bad index) returns a list of nones of the
    # same length.
    """Best-effort NVML UUIDs so the dashboard can reconcile GPU index
    spaces across processes; None entries when NVML is unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return [str(pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))) for i in gpu_ids]
    except Exception:
        return [None] * len(gpu_ids)


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    # [FUNCTION DOC] HELPER: Converts a gpu id to a torch device index given the visible devices
    # environment variable (cuda, else hip). Unset or empty: unchanged. Otherwise parses the comma
    # separated list; a physical id in the list maps to its position; an id already in the range of
    # local indices is returned unchanged; anything else raises a runtime error. Ambiguity: an id that
    # is both a physical id in the list and a valid local index is treated as physical.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


class NodeProbeMixin:
    # [CLASS DOC] HELPER: Mixin exposing node probing functions as static methods so tests can
    # override them in subclasses.
    @staticmethod
    def _get_node_ip() -> str:
        # [FUNCTION DOC] HELPER: The host ip environment override if set and non-empty, else ray's
        # node ip.
        return os.getenv(MILES_HOST_IP_ENV) or get_current_node_ip()

    @staticmethod
    def _get_node_external_ip() -> str | None:
        # [FUNCTION DOC] HELPER: Delegates to the external ip lookup.
        return get_node_external_ip()

    @staticmethod
    def _get_free_port_block(*, start_port: int, count: int) -> int:
        # [FUNCTION DOC] HELPER: First start of a block of count consecutive free ports.
        return get_free_port(start_port=start_port, consecutive=count)

    @staticmethod
    def _to_local_gpu_ids(*, gpu_ids: list[int]) -> list[int]:
        # [FUNCTION DOC] HELPER: Maps each gpu id to a local device index.
        return [_to_local_gpu_id(gpu_id) for gpu_id in gpu_ids]

    @staticmethod
    def _is_port_available(*, port: int) -> bool:
        # [FUNCTION DOC] HELPER: Delegates to the port availability probe.
        return is_port_available(port)

    @staticmethod
    def _get_gpu_uuids(gpu_ids: list[int]) -> list[str | None]:
        # [FUNCTION DOC] HELPER: Delegates to the uuid probe.
        return get_gpu_uuids(gpu_ids)


def should_run_periodic_action(
    rollout_id: int,
    interval: int | None,
    num_rollout_per_epoch: int | None = None,
    num_rollout: int | None = None,
) -> bool:
    # [FUNCTION DOC] False when the interval is none. True on the last rollout (id equal to the total
    # minus one, when a total is given). Otherwise with step = rollout id plus one: true when the step
    # is a multiple of the interval, or (if given) a multiple of the epoch length. A zero interval
    # or epoch length raises a zero division error.
    """
    Return True when a periodic action (eval/save/checkpoint) should run.

    Args:
        rollout_id: The current rollout index (0-based).
        interval: Desired cadence; disables checks when None.
        num_rollout_per_epoch: Optional epoch boundary to treat as a trigger.
    """
    if interval is None:
        return False

    if num_rollout is not None and rollout_id == num_rollout - 1:
        return True

    step = rollout_id + 1
    return (step % interval == 0) or (num_rollout_per_epoch is not None and step % num_rollout_per_epoch == 0)


async def as_completed_async(tasks):
    # [FUNCTION DOC] HELPER: Async generator yielding results of the awaitables in completion order.
    for coro in asyncio.as_completed(tasks):
        yield await coro


def filter_keys(d: dict[str, Any], interest_keys: Sequence[str]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: New mapping with only the listed keys (in that order). A missing key logs
    # an error with the available keys and re-raises the key error.
    try:
        return {k: d[k] for k in interest_keys}
    except Exception:
        logger.error(f"filter_keys d.keys={list(d)} {interest_keys=}", exc_info=True)
        raise


class SimpleTicker:
    # [CLASS DOC] HELPER: Runs an async function repeatedly on the running event loop, sleeping the
    # interval before each call (so the first call happens after one interval). Exceptions are logged
    # and the ticking continues.
    def __init__(self, fn: Callable[[], Awaitable[None]], *, interval_seconds: float):
        # [FUNCTION DOC] HELPER: Stores settings and starts the background task immediately (requires
        # a running event loop).
        self._fn = fn
        self._interval_seconds = interval_seconds
        self._task = asyncio.create_task(self._loop())

    async def dispose(self) -> None:
        # [FUNCTION DOC] HELPER: Cancels the ticking task and waits for it to end.
        await cancel_and_await_task(self._task)

    async def _loop(self) -> None:
        # [FUNCTION DOC] HELPER: Forever: sleep, then call the function, logging non-cancellation
        # exceptions.
        while True:
            await asyncio.sleep(self._interval_seconds)
            try:
                await self._fn()
            except Exception:
                logger.exception(f"Ticking {self._fn} failed; retrying")


async def cancel_and_await_task(task: asyncio.Task) -> None:
    # [FUNCTION DOC] HELPER: Requests cancellation, waits for the task to finish, and if it ended
    # without being cancelled (completed or failed) re-raises its exception or returns after reading
    # its result.
    task.cancel()
    await asyncio.wait([task])
    if not task.cancelled():
        task.result()
