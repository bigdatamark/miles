# [MODULE DOC] HELPER:
# Test worker used by the end-to-end rpc tests, plus the spec computation that serves it. The
# worker class exposes many small methods (arithmetic, typed round trips for models, dataclasses,
# enums, dates, ids, decimals, bytes, tuples, unions and special floats; deliberately wrong
# results; raised errors; call counters; sleeps, blocking calls and gates; concurrency-group
# variants; marker files; large payloads) that tests invoke through the rpc server. Per-call
# start and end events with thread names and monotonic times are recorded so tests can reason
# about ordering and threading. The module also defines the model types, the pool id, the
# command-line flag for the rpc port, and functions that build the serve-worker spec.
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import enum
import os
import threading
import time
import uuid
from decimal import Decimal
from pathlib import Path

from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.rpc.common.metadata import rpc
from miles.utils.workers.worker_spec import PortInfo, SchedulingSpec, ServeWorkerSpec

POOL_ID = "e2e-pool"
RPC_PORT_FLAG = "--rpc-port"

_BLOCK_GUARD_SECONDS = 20.0


class Item(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict model with a name and a list of integers.
    name: str
    values: list[int]


class Colour(enum.Enum):
    # [CLASS DOC] HELPER: Enumeration with members red and blue (string values).
    RED = "red"
    BLUE = "blue"


class Nested(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict model nesting an item, a mapping from string to item, and a set
    # of strings; used to check nested type reconstruction.
    item: Item
    lookup: dict[str, Item]
    tags: set[str]


@dataclasses.dataclass
class Point:
    # [CLASS DOC] HELPER: Plain data class with integer coordinates x and y.
    x: int
    y: int


class Metric(StrictBaseModel):
    # [CLASS DOC] HELPER: Strict model with a name and a float value (may hold special floats).
    name: str
    value: float


class Event(StrictBaseModel):
    # [CLASS DOC] HELPER: Record of one call phase: call tag, phase label ("start" or "end"),
    # name of the executing thread, and a monotonic timestamp in seconds.
    tag: str
    phase: str
    thread_name: str
    at: float


WORKER_FACTORY_ERROR = "e2e worker factory refuses to build a worker"


class E2eWorker:
    # [CLASS DOC] HELPER: Worker with test methods. State: the argument list and state directory
    # it was built with, a lock protecting shared state, per-tag counters, a list of recorded
    # events, and per-tag gates (one thread-based and one event-loop-based kind) that block calls
    # until released. Methods starting with "report" return observations about the worker;
    # methods starting with "demo" exercise one behavior each.
    def __init__(self, argv: list[str], state_dir: Path) -> None:
        # [FUNCTION DOC] HELPER: Stores the argument list and state directory and initializes the
        # lock, empty counters, empty event list and empty gate tables.
        self._argv = argv
        self._state_dir = state_dir
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._events: list[Event] = []
        self._sync_gates: dict[str, threading.Event] = {}
        self._async_gates: dict[str, asyncio.Event] = {}

    async def report_pid(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the operating-system process id of the worker process.
        return os.getpid()

    async def report_argv(self) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the argument list the worker was constructed with.
        return self._argv

    async def report_env(self, name: str) -> str | None:
        # [FUNCTION DOC] HELPER: Returns the value of the named environment variable in the worker
        # process, or nothing if unset.
        return os.environ.get(name)

    async def report_counter(self, tag: str) -> int:
        # [FUNCTION DOC] HELPER: Returns the counter value for the tag (zero if never bumped),
        # read under the lock.
        with self._lock:
            return self._counters.get(tag, 0)

    async def report_events(self) -> list[Event]:
        # [FUNCTION DOC] HELPER: Returns a copy of all recorded events in recording order.
        with self._lock:
            return list(self._events)

    def demo_sync(self, a: int, b: int) -> int:
        # [FUNCTION DOC] HELPER: Synchronous sum of two integers.
        return a + b

    def demo_default_arg(self, name: str = "world") -> str:
        # [FUNCTION DOC] HELPER: Returns a greeting made of "hello " and the name (default
        # "world"); checks that default arguments are applied.
        return f"hello {name}"

    def demo_optional_arg(self, name: str | None) -> str:
        # [FUNCTION DOC] HELPER: Returns the textual representation of the (possibly missing)
        # argument, so tests can distinguish nothing from the string "None".
        return repr(name)

    async def demo_async(self, value: dict) -> dict:
        # [FUNCTION DOC] HELPER: Echoes the mapping back unchanged.
        return value

    async def demo_model(self, item: Item) -> Item:
        # [FUNCTION DOC] HELPER: Echoes the item model back.
        return item

    async def demo_nested_model(self, payload: Nested) -> Nested:
        # [FUNCTION DOC] HELPER: Echoes the nested model back.
        return payload

    async def report_nested_argument_types(self, payload: Nested) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the type names of the received nested model, its item,
        # the mapping entry under key "k", its tags collection, and the first element of the
        # item's values, in that order. Fails with a key error if no entry "k" exists and with an
        # index error if the values list is empty.
        return [
            type(payload).__name__,
            type(payload.item).__name__,
            type(payload.lookup["k"]).__name__,
            type(payload.tags).__name__,
            type(payload.item.values[0]).__name__,
        ]

    async def report_dataclass_argument_type(self, point: Point) -> str:
        # [FUNCTION DOC] HELPER: Returns the type name of the received point argument.
        return type(point).__name__

    async def report_enum_argument_is_member(self, colour: Colour) -> bool:
        # [FUNCTION DOC] HELPER: Returns true if the received colour is the blue member of the
        # enumeration (identity comparison, so a plain string yields false).
        return colour is Colour.BLUE

    async def report_scalar_argument_types(
        self, when: datetime.datetime, value: uuid.UUID, amount: Decimal, blob: bytes, pair: tuple[int, str]
    ) -> list[str]:
        # [FUNCTION DOC] HELPER: Returns the type names of the received datetime, uuid, decimal,
        # bytes and tuple arguments, in that order.
        return [
            type(when).__name__,
            type(value).__name__,
            type(amount).__name__,
            type(blob).__name__,
            type(pair).__name__,
        ]

    async def demo_enum(self, colour: Colour) -> Colour:
        # [FUNCTION DOC] HELPER: Echoes the colour back.
        return colour

    async def demo_dataclass(self, point: Point) -> Point:
        # [FUNCTION DOC] HELPER: Returns a new point with x and y swapped.
        return Point(x=point.y, y=point.x)

    async def demo_datetime(self, when: datetime.datetime) -> datetime.datetime:
        # [FUNCTION DOC] HELPER: Echoes the datetime back.
        return when

    async def demo_uuid(self, value: uuid.UUID) -> uuid.UUID:
        # [FUNCTION DOC] HELPER: Echoes the uuid back.
        return value

    async def demo_decimal(self, value: Decimal) -> Decimal:
        # [FUNCTION DOC] HELPER: Echoes the decimal back.
        return value

    async def demo_tuple(self, pair: tuple[int, str]) -> tuple[int, str]:
        # [FUNCTION DOC] HELPER: Echoes the integer-string pair back.
        return pair

    async def demo_optional(self, value: int | None) -> int | None:
        # [FUNCTION DOC] HELPER: Echoes the optional integer back.
        return value

    async def demo_union(self, value: int | str) -> int | str:
        # [FUNCTION DOC] HELPER: Echoes the integer-or-string value back.
        return value

    async def demo_model_list(self, items: list[Item]) -> list[Item]:
        # [FUNCTION DOC] HELPER: Echoes the list of item models back.
        return items

    async def demo_bytes(self, blob: bytes) -> bytes:
        # [FUNCTION DOC] HELPER: Echoes the bytes back.
        return blob

    async def demo_nan_result(self) -> float:
        # [FUNCTION DOC] HELPER: Returns a not-a-number float.
        return float("nan")

    async def demo_float(self, value: float) -> float:
        # [FUNCTION DOC] HELPER: Echoes the float back (including special values).
        return value

    async def demo_optional_float(self, value: float | None) -> float | None:
        # [FUNCTION DOC] HELPER: Echoes the optional float back.
        return value

    async def demo_float_metrics(self) -> dict:
        # [FUNCTION DOC] HELPER: Returns a mapping with loss as not-a-number, grad_norm as
        # positive infinity, lr as negative infinity and step as 3.0, to check special float
        # serialization.
        return {"loss": float("nan"), "grad_norm": float("inf"), "lr": -float("inf"), "step": 3.0}

    async def demo_float_list(self, values: list[float]) -> list[float]:
        # [FUNCTION DOC] HELPER: Echoes the list of floats back.
        return values

    async def demo_metric_model(self, metric: Metric) -> Metric:
        # [FUNCTION DOC] HELPER: Echoes the metric model back.
        return metric

    async def report_float_repr(self, value: float) -> str:
        # [FUNCTION DOC] HELPER: Returns the textual representation of the received float, to
        # show what value arrived at the worker.
        return repr(value)

    async def demo_bytes_list(self, blobs: list[bytes]) -> list[bytes]:
        # [FUNCTION DOC] HELPER: Echoes the list of byte strings back.
        return blobs

    async def report_union_argument_type(self, value: datetime.datetime | str) -> str:
        # [FUNCTION DOC] HELPER: Returns the type name of the received datetime-or-string value.
        return type(value).__name__

    async def demo_none_result(self) -> None:
        # [FUNCTION DOC] HELPER: Returns nothing.
        return None

    async def demo_wrong_result_type(self) -> int:
        # [FUNCTION DOC] HELPER: Declared to return an integer but returns a string, to test
        # result validation on the server.
        return "not-an-int"

    async def demo_unserializable_result(self) -> int:
        # [FUNCTION DOC] HELPER: Declared to return an integer but returns a plain object that
        # cannot be serialized, to test result error handling.
        return object()

    def demo_sync_raises(self, message: str) -> None:
        # [FUNCTION DOC] HELPER: Synchronously raises a value error with the given message.
        raise ValueError(message)

    async def demo_async_raises(self, message: str) -> None:
        # [FUNCTION DOC] HELPER: Raises a value error with the given message from an async method.
        raise ValueError(message)

    def demo_system_exit(self) -> None:
        # [FUNCTION DOC] HELPER: Raises a system-exit request with code 3 from a sync method.
        raise SystemExit(3)

    async def demo_system_exit_async(self) -> None:
        # [FUNCTION DOC] HELPER: Raises a system-exit request with code 3 from an async method.
        raise SystemExit(3)

    def demo_count_sync(self, tag: str) -> int:
        # [FUNCTION DOC] HELPER: Increments the tag's counter and returns the new value
        # (synchronous method).
        return self._bump(tag)

    async def demo_count_async(self, tag: str) -> int:
        # [FUNCTION DOC] HELPER: Increments the tag's counter and returns the new value
        # (async method).
        return self._bump(tag)

    def demo_count_after_sleep(self, tag: str, seconds: float) -> int:
        # [FUNCTION DOC] HELPER: Blocks the thread for the given seconds, then increments the
        # tag's counter and returns the new value; the counter changes only after the sleep, so
        # a cancelled or abandoned call shows no effect.
        time.sleep(seconds)
        return self._bump(tag)

    def demo_sleep_sync(self, tag: str, seconds: float) -> str:
        # [FUNCTION DOC] HELPER: Records a start event, blocks the thread for the seconds,
        # records an end event, and returns the tag.
        self._mark(tag, "start")
        time.sleep(seconds)
        self._mark(tag, "end")
        return tag

    async def demo_sleep_async(self, tag: str, seconds: float) -> str:
        # [FUNCTION DOC] HELPER: Records a start event, sleeps without blocking the event loop,
        # records an end event, and returns the tag.
        self._mark(tag, "start")
        await asyncio.sleep(seconds)
        self._mark(tag, "end")
        return tag

    def demo_block_sync(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Records a start event, blocks the thread until the tag's
        # thread gate is released (at most 20 seconds, after which it continues anyway), records
        # an end event, and returns the name of the executing thread.
        self._mark(tag, "start")
        self._sync_gate(tag).wait(timeout=_BLOCK_GUARD_SECONDS)
        self._mark(tag, "end")
        return threading.current_thread().name

    def demo_instant_sync(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Records start and end events immediately and returns the
        # executing thread's name.
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    async def demo_block_async(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Records a start event, waits (without a time limit) for the
        # tag's event-loop gate to be released, records an end event, and returns the executing
        # thread's name.
        self._mark(tag, "start")
        await self._async_gate(tag).wait()
        self._mark(tag, "end")
        return threading.current_thread().name

    async def demo_block_until_cancelled(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Records a start event and waits forever. If cancelled, creates
        # an empty marker file named "cancelled_" plus the tag in the state directory and lets
        # the cancellation propagate. The final return of the tag is never reached in practice.
        self._mark(tag, "start")
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            (self._state_dir / f"cancelled_{tag}").touch()
            raise
        return tag

    async def demo_instant_async(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Records start and end events immediately and returns the
        # executing thread's name (async method).
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="left")
    def demo_sleep_on_left(self, tag: str, seconds: float) -> str:
        # [FUNCTION DOC] HELPER: Same as the sync sleep method (start event, thread sleep, end
        # event, returns the thread name) but runs in the "left" concurrency group.
        self._mark(tag, "start")
        time.sleep(seconds)
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="left")
    def demo_instant_on_left(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Records start and end events and returns the thread name, in
        # the "left" concurrency group.
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="right")
    def demo_sleep_on_right(self, tag: str, seconds: float) -> str:
        # [FUNCTION DOC] HELPER: Start event, thread sleep, end event, returns the thread name,
        # in the "right" concurrency group.
        self._mark(tag, "start")
        time.sleep(seconds)
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc(concurrency_group="left")
    def demo_block_on_left(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Blocks on the tag's thread gate (up to 20 seconds) between start
        # and end events and returns the thread name, in the "left" concurrency group.
        self._mark(tag, "start")
        self._sync_gate(tag).wait(timeout=_BLOCK_GUARD_SECONDS)
        self._mark(tag, "end")
        return threading.current_thread().name

    @rpc()
    async def demo_async_on_left(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Async method registered with the default concurrency group
        # (despite its name suggesting the left group); records start and end events and
        # returns the thread name.
        # KNOWN ISSUE: the name says "on_left" but the rpc decorator names no concurrency group.
        self._mark(tag, "start")
        self._mark(tag, "end")
        return threading.current_thread().name

    async def release(self, tag: str) -> bool:
        # [FUNCTION DOC] HELPER: Opens both the thread gate and the event-loop gate of the tag
        # (creating them if absent so a later call passes straight through) and returns true.
        self._sync_gate(tag).set()
        self._async_gate(tag).set()
        return True

    async def release_every_gate(self) -> int:
        # [FUNCTION DOC] HELPER: Opens every existing gate of both kinds and returns how many
        # gates were opened. The gate list is snapshotted under the lock, the gates opened
        # outside it.
        with self._lock:
            gates = [*self._sync_gates.values(), *self._async_gates.values()]
        for gate in gates:
            gate.set()
        return len(gates)

    def demo_marker_after_sleep(self, name: str, seconds: float) -> str:
        # [FUNCTION DOC] HELPER: Blocks the thread for the seconds, then creates an empty marker
        # file with the given name in the state directory and returns the name.
        time.sleep(seconds)
        (self._state_dir / name).touch()
        return name

    async def demo_marker_after_sleep_async(self, name: str, seconds: float) -> str:
        # [FUNCTION DOC] HELPER: Async variant: sleeps without blocking, then creates the marker
        # file in the state directory and returns the name.
        await asyncio.sleep(seconds)
        (self._state_dir / name).touch()
        return name

    def demo_hang(self, tag: str) -> str:
        # [FUNCTION DOC] HELPER: Records a start event, then blocks the thread for the full 20
        # second guard time (nothing releases it) and returns the tag.
        self._mark(tag, "start")
        threading.Event().wait(timeout=_BLOCK_GUARD_SECONDS)
        return tag

    def demo_large_upload(self, blob: str) -> int:
        # [FUNCTION DOC] HELPER: Returns the length of the received string; used to test large
        # request bodies.
        return len(blob)

    def demo_large_download(self, size: int) -> list[int]:
        # [FUNCTION DOC] HELPER: Returns the integers from zero up to size minus one; used to
        # test large response bodies.
        return list(range(size))

    def _bump(self, tag: str) -> int:
        # [FUNCTION DOC] HELPER: Under the lock, increments the tag's counter (starting from
        # zero) and returns the new value.
        with self._lock:
            self._counters[tag] = self._counters.get(tag, 0) + 1
            return self._counters[tag]

    def _mark(self, tag: str, phase: str) -> None:
        # [FUNCTION DOC] HELPER: Appends an event with the tag, phase, current thread name and
        # current monotonic time to the event list, under the lock.
        event = Event(tag=tag, phase=phase, thread_name=threading.current_thread().name, at=time.monotonic())
        with self._lock:
            self._events.append(event)

    def _sync_gate(self, tag: str) -> threading.Event:
        # [FUNCTION DOC] HELPER: Returns the tag's thread-based gate, creating a closed one on
        # first use, under the lock.
        with self._lock:
            return self._sync_gates.setdefault(tag, threading.Event())

    def _async_gate(self, tag: str) -> asyncio.Event:
        # [FUNCTION DOC] HELPER: Returns the tag's event-loop gate, creating a closed one on
        # first use, under the lock. The gate is created in whichever thread first asks for it.
        with self._lock:
            return self._async_gates.setdefault(tag, asyncio.Event())


def compute_specs(worker_argv: list[str]) -> list[ServeWorkerSpec]:
    # [FUNCTION DOC] HELPER: Spec-computation entry point for the serve launcher. Returns a
    # one-element list with the spec built from the worker arguments, whose environment
    # function exports the whole argument list, comma-joined, as MILES_E2E_ARGV.
    return [spec_of(worker_argv, env_var=lambda context: {"MILES_E2E_ARGV": ",".join(worker_argv)})]


def spec_of(worker_argv: list[str], *, env_var) -> ServeWorkerSpec:
    # [FUNCTION DOC] HELPER: Builds the serve-worker spec: parses the run arguments, names the
    # spec after the pool id, declares one static port "rpc" taken from the arguments, uses the
    # given environment function, one cell with one worker using zero gpus, this module's worker
    # class, and constructor arguments of the full argument list and the state directory path.
    args = parse_run_args(worker_argv)
    return ServeWorkerSpec(
        name=POOL_ID,
        port_infos=[PortInfo(name="rpc", static_port=args.rpc_port)],
        env_var=env_var,
        scheduling=SchedulingSpec(num_cells=1, num_workers_per_cell=1, num_gpus_per_worker=0),
        worker_class=f"{__name__}.E2eWorker",
        ctor_kwargs=lambda context: dict(argv=worker_argv, state_dir=Path(args.state_dir)),
    )


def parse_run_args(worker_argv: list[str]) -> argparse.Namespace:
    # [FUNCTION DOC] HELPER: Parses the worker arguments. A required state-directory option and a
    # required integer rpc-port option are read; unknown arguments are ignored. A missing
    # required option makes the parser exit the process with a usage error.
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", required=True)
    parser.add_argument(RPC_PORT_FLAG, type=int, required=True)
    args, _ = parser.parse_known_args(worker_argv)
    return args
