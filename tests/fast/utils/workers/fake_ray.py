# [MODULE DOC] HELPER:
# In-process fake of the parts of Ray that the Ray worker manager uses, so its whole launch and
# teardown pipeline can be tested without a cluster. A fake cluster records every actor creation,
# method call, kill and result fetch, hands out fake node addresses round-robin, simulates port
# allocation and port occupation per node, and computes canned return values for a few private
# helper methods of the real actors (node ip, external ip, local gpu ids, port availability, free
# port block). Failures and hangs can be injected per method, per handle or cluster-wide. The
# fake module object mirrors the Ray calls that are used: remote decoration, method decoration,
# fetching results, killing and looking up named actors.
from __future__ import annotations

import asyncio
import itertools
import time
from dataclasses import dataclass, field, replace
from typing import Any

_ASYNC_METHOD_NODE_IP = "_get_node_ip"
_ASYNC_METHOD_NODE_EXTERNAL_IP = "_get_node_external_ip"
_ASYNC_METHOD_FREE_PORT_BLOCK = "_get_free_port_block"
_ASYNC_METHOD_IS_PORT_AVAILABLE = "_is_port_available"
_ASYNC_METHOD_TO_LOCAL_GPU_IDS = "_to_local_gpu_ids"

READINESS_METHOD = "__ray_ready__"

EVENT_CREATE = "create"
EVENT_KILL = "kill"


@dataclass(kw_only=True, eq=False)
class FakeRayObjectRef:
    # [CLASS DOC] HELPER: Fake future of an actor method call. Holds the method name, the value
    # to return, an optional error to raise instead, and an optional artificial delay in seconds.
    # It can be awaited (the delay is then an async sleep) or resolved synchronously through the
    # fake module's get (the delay is then a blocking sleep).
    method: str
    value: Any = None
    error: BaseException | None = None
    hang_seconds: float | None = None

    def __await__(self):
        # [FUNCTION DOC] HELPER: Makes the reference awaitable by delegating to the async
        # resolution coroutine.
        return self._resolve_async().__await__()

    async def _resolve_async(self) -> Any:
        # [FUNCTION DOC] HELPER: Sleeps without blocking for the artificial delay if one is set,
        # then resolves (returns the value or raises the error).
        if self.hang_seconds is not None:
            await asyncio.sleep(self.hang_seconds)
        return self.resolve()

    def resolve(self) -> Any:
        # [FUNCTION DOC] HELPER: Raises the stored error if there is one, otherwise returns the
        # stored value. Ignores the delay.
        if self.error is not None:
            raise self.error
        return self.value


@dataclass(kw_only=True)
class FakeRayActorCall:
    # [CLASS DOC] HELPER: Record of one dispatched actor method call: target handle, method name,
    # positional arguments and keyword arguments.
    handle: FakeRayActorHandle
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


@dataclass(kw_only=True)
class FakeRayActorMethod:
    # [CLASS DOC] HELPER: Fake bound remote method of an actor handle.
    handle: FakeRayActorHandle
    method: str

    def remote(self, *args: Any, **kwargs: Any) -> FakeRayObjectRef:
        # [FUNCTION DOC] HELPER: Asks the cluster to record and dispatch the call; returns the
        # fake future. Nothing is executed on the actor class.
        return self.handle.cluster.dispatch(handle=self.handle, method=self.method, args=args, kwargs=kwargs)


@dataclass(kw_only=True, eq=False)
class FakeRayActorHandle:
    # [CLASS DOC] HELPER: Fake actor handle with identity-based equality. Holds its cluster, the
    # actor class and constructor options it was created with, its creation index, the fake node
    # address it was placed on, per-method error and delay injections, and a flag that is set when
    # the actor is killed. Any public attribute lookup yields a fake remote method.
    cluster: FakeRayCluster
    actor_class: type
    index: int
    options: dict[str, Any]
    node_ip: str
    failing_methods: dict[str, BaseException] = field(default_factory=dict)
    hanging_methods: dict[str, float] = field(default_factory=dict)
    killed: bool = False

    def __getattr__(self, name: str) -> FakeRayActorMethod:
        # [FUNCTION DOC] HELPER: Called only for attributes not found normally. Returns a fake
        # remote method of that name. Names starting with two underscores, and any lookup before
        # the cluster field has been set (for example during copying), raise an attribute error.
        if name.startswith("__") or "cluster" not in self.__dict__:
            raise AttributeError(name)
        return FakeRayActorMethod(handle=self, method=name)

    @property
    def __ray_ready__(self) -> FakeRayActorMethod:
        # [FUNCTION DOC] HELPER: Fake remote method for Ray's built-in readiness probe (the only
        # double-underscore method that works).
        return FakeRayActorMethod(handle=self, method=READINESS_METHOD)


@dataclass(kw_only=True)
class FakeRayRemoteClass:
    # [CLASS DOC] HELPER: What remote decoration of a class returns: the cluster, the class and
    # the accumulated actor options (such as a name).
    cluster: FakeRayCluster
    actor_class: type
    actor_options: dict[str, Any] = field(default_factory=dict)

    def options(self, **kwargs: Any) -> FakeRayRemoteClass:
        # [FUNCTION DOC] HELPER: Returns a copy whose options are the existing ones overlaid with
        # the given keywords (the original is not modified).
        return replace(self, actor_options={**self.actor_options, **kwargs})

    def remote(self, *args: Any, **kwargs: Any) -> FakeRayActorHandle:
        # [FUNCTION DOC] HELPER: Creates a fake actor through the cluster using the keyword
        # arguments as constructor arguments; positional arguments are not allowed (assertion).
        # If a name option is set, registers the handle under that name in the cluster's named
        # actors, overwriting any earlier one. Returns the handle.
        assert not args, "actor constructors must be called with keyword arguments"
        handle = self.cluster.create_actor(
            actor_class=self.actor_class, options=self.actor_options, ctor_kwargs=kwargs
        )
        if (name := self.actor_options.get("name")) is not None:
            self.cluster.named_actors[name] = handle
        return handle


@dataclass(kw_only=True)
class FakeRayModule:
    # [CLASS DOC] HELPER: Stand-in for the Ray module object, bound to a cluster. Offers the
    # subset of the Ray API that the worker manager calls.
    cluster: FakeRayCluster

    def remote(self, actor_class: type | None = None, **decorator_options: Any):
        # [FUNCTION DOC] HELPER: Works both as a plain call on a class (returns the fake remote
        # class) and as a decorator factory with options (returns a function that wraps a class
        # into a fake remote class carrying those options).
        if actor_class is not None:
            return FakeRayRemoteClass(cluster=self.cluster, actor_class=actor_class)
        return lambda cls: FakeRayRemoteClass(cluster=self.cluster, actor_class=cls, actor_options=decorator_options)

    def method(self, **decorator_options: Any):
        # [FUNCTION DOC] HELPER: Method-decorator factory. The returned decorator stores each
        # option on the function as an attribute named with a double-underscore ray prefix and
        # suffix, then returns the same function.
        def _decorator(fn: Any) -> Any:
            # [FUNCTION DOC] HELPER: Sets the option attributes on the function and returns it.
            for name, value in decorator_options.items():
                setattr(fn, f"__ray_{name}__", value)
            return fn

        return _decorator

    def get(self, ref: FakeRayObjectRef, timeout: float | None = None) -> Any:
        # [FUNCTION DOC] HELPER: Synchronous fetch. Records the method name in the cluster's
        # resolved references and the timeout in its timeout list (the timeout is only recorded,
        # never enforced), blocks the thread for the reference's artificial delay if any, then
        # returns the value or raises the stored error.
        self.cluster.resolved_refs.append(ref.method)
        self.cluster.get_timeouts.append(timeout)
        if ref.hang_seconds is not None:
            time.sleep(ref.hang_seconds)
        return ref.resolve()

    def kill(self, handle: FakeRayActorHandle, no_restart: bool = False) -> None:
        # [FUNCTION DOC] HELPER: Kills the actor through the cluster (the no-restart flag is
        # ignored). May raise the cluster's configured kill error.
        self.cluster.kill_actor(handle)

    def get_actor(self, name: str) -> FakeRayActorHandle:
        # [FUNCTION DOC] HELPER: Returns the named actor handle; raises a key error if no actor
        # has that name (the real Ray raises a value error).
        # KNOWN ISSUE: the real Ray raises a value error for an unknown name, which callers in
        # the tests rely on elsewhere; this fake raises a key error.
        return self.cluster.named_actors[name]


@dataclass
class FakeRayCluster:
    # [CLASS DOC] HELPER: Recording fake cluster. Public fields hold the node addresses used
    # round-robin for placement, a mapping from node address to external address, the first port
    # of allocation, all created handles, all calls, a chronological event list (creation, kill
    # and method names), named actors, optional kill error, cluster-wide per-method errors,
    # fetched method names, fetch timeouts and actor constructor keyword arguments. Occupied
    # ports are tracked per node.
    node_ips: tuple[str, ...] = ("10.0.0.1",)
    # node ip -> the address off-cluster peers reach that node on; a node absent
    # from the mapping reports None, like one whose deployment set nothing.
    node_external_ips: dict[str, str] = field(default_factory=dict)
    base_port: int = 15000
    handles: list[FakeRayActorHandle] = field(default_factory=list)
    calls: list[FakeRayActorCall] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    named_actors: dict[str, FakeRayActorHandle] = field(default_factory=dict)
    kill_error: BaseException | None = None
    method_errors: dict[str, BaseException] = field(default_factory=dict)
    resolved_refs: list[str] = field(default_factory=list)
    get_timeouts: list[float | None] = field(default_factory=list)
    ctor_kwargs: list[dict[str, Any]] = field(default_factory=list)
    _used_ports: dict[str, set[int]] = field(default_factory=dict)
    _node_ip_cycle: Any = None

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the endless round-robin iterator over the node
        # addresses.
        # KNOWN ISSUE: the base port field is never read by the cluster's port allocation.
        self._node_ip_cycle = itertools.cycle(self.node_ips)

    def occupy_ports(self, node_ip: str, *ports: int) -> None:
        # [FUNCTION DOC] HELPER: Marks the given ports as in use on the node, so availability
        # checks fail and block allocation skips them.
        """Stand in for a process from an earlier run still listening on the node."""
        self._used_ports.setdefault(node_ip, set()).update(ports)

    def use_node_ips(self, *node_ips: str) -> None:
        # [FUNCTION DOC] HELPER: Replaces the node addresses and restarts the round-robin from
        # the first one. Existing handles keep their addresses.
        self.node_ips = node_ips
        self._node_ip_cycle = itertools.cycle(node_ips)

    def create_actor(
        self, *, actor_class: type, options: dict[str, Any], ctor_kwargs: dict[str, Any]
    ) -> FakeRayActorHandle:
        # [FUNCTION DOC] HELPER: Creates a handle with the next index and the next round-robin
        # node address, appends it to the handle list, records the constructor keyword
        # arguments, appends a creation event, and returns the handle. The actor class is never
        # instantiated.
        handle = FakeRayActorHandle(
            cluster=self,
            actor_class=actor_class,
            index=len(self.handles),
            options=options,
            node_ip=next(self._node_ip_cycle),
        )
        self.handles.append(handle)
        self.ctor_kwargs.append(ctor_kwargs)
        self.events.append(EVENT_CREATE)
        return handle

    def kill_actor(self, handle: FakeRayActorHandle) -> None:
        # [FUNCTION DOC] HELPER: Appends a kill event, then raises the configured kill error if
        # set (the handle is then not marked killed); otherwise marks the handle killed.
        self.events.append(EVENT_KILL)
        if self.kill_error is not None:
            raise self.kill_error
        handle.killed = True

    def dispatch(
        self, *, handle: FakeRayActorHandle, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> FakeRayObjectRef:
        # [FUNCTION DOC] HELPER: Records the call and appends the method name as an event.
        # Returns a fake future with the computed canned value, the error injected for this
        # handle and method (taking precedence) or cluster-wide for the method name, and the
        # handle's artificial delay for the method. Does not check whether the handle is killed.
        self.calls.append(FakeRayActorCall(handle=handle, method=method, args=args, kwargs=kwargs))
        self.events.append(method)
        return FakeRayObjectRef(
            method=method,
            value=self._compute_value(handle=handle, method=method, kwargs=kwargs),
            error=handle.failing_methods.get(method, self.method_errors.get(method)),
            hang_seconds=handle.hanging_methods.get(method),
        )

    def calls_of(self, method: str) -> list[FakeRayActorCall]:
        # [FUNCTION DOC] HELPER: Returns the recorded calls of the method name, in order.
        return [call for call in self.calls if call.method == method]

    def first_event_index(self, event: str) -> int:
        # [FUNCTION DOC] HELPER: Index of the first occurrence of the event name in the event
        # list; raises a value error if absent.
        return self.events.index(event)

    def last_event_index(self, event: str) -> int:
        # [FUNCTION DOC] HELPER: Index of the last occurrence of the event name in the event
        # list; raises a value error if absent.
        return len(self.events) - 1 - self.events[::-1].index(event)

    def _compute_value(self, *, handle: FakeRayActorHandle, method: str, kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] HELPER: Canned return value by method name. Node ip: the handle's node
        # address. External ip: the mapped external address of that node, or nothing. Local gpu
        # ids: the list 0..n-1 for n requested gpu ids (a worker whose visibility mask hides the
        # leading gpus sees itself starting at 0). Port availability: true unless the requested
        # port is marked used on the node. Free port block: allocates a block of the requested
        # size starting at the requested start port. Any other method returns nothing.
        if method == _ASYNC_METHOD_NODE_IP:
            return handle.node_ip
        if method == _ASYNC_METHOD_NODE_EXTERNAL_IP:
            return self.node_external_ips.get(handle.node_ip)
        if method == _ASYNC_METHOD_TO_LOCAL_GPU_IDS:
            # a worker whose visibility mask hides the leading gpus sees itself starting at 0
            return list(range(len(kwargs["gpu_ids"])))
        if method == _ASYNC_METHOD_IS_PORT_AVAILABLE:
            return kwargs["port"] not in self._used_ports.get(handle.node_ip, set())
        if method == _ASYNC_METHOD_FREE_PORT_BLOCK:
            return self._alloc_port_block(
                node_ip=handle.node_ip, start_port=kwargs["start_port"], count=kwargs["count"]
            )
        return None

    def _alloc_port_block(self, *, node_ip: str, start_port: int, count: int) -> int:
        # [FUNCTION DOC] HELPER: Finds the lowest port at or above the start port such that the
        # whole block of the given size is free on the node (incrementing by one while any port
        # in the block is used), marks the block as used, and returns its first port.
        used = self._used_ports.setdefault(node_ip, set())
        # Exactly start_port when it is free, like the real get_free_port: the floor lives in
        # PortAllocator, and clamping here would make a pinned port look occupied.
        port = start_port
        while any(port + offset in used for offset in range(count)):
            port += 1
        used.update(range(port, port + count))
        return port
