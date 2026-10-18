# [MODULE DOC]
# Fakes for the training driver's collaborators: remote-method wrapper, worker manager, rollout
# executor, inference controller, training model and object store. Most fakes append short event
# strings to a shared ordered list supplied by the test, so tests assert call ordering across
# components. Constant: the ordered names of the events that gate an inference take-over.
import asyncio
from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from miles.utils.data import RolloutDataPack
from miles.utils.object_store import (
    BaseObjectStore,
    ObjectStoreGetResult,
    StoreObjectRef,
    ValueSpec,
    _MooncakeStoreObjectRef,
)

TAKE_OVER_GATE_EVENTS = ["rollout_components", "training_models", "inference_take_over"]


class FakeRemoteMethod:
    # [CLASS DOC] Stand-in for a remote actor method handle: calling the object calls the wrapped
    # function directly with the same arguments and returns its result (which may be an awaitable).
    def __init__(self, fn: Callable[..., Any]) -> None:
        # [FUNCTION DOC] Stores the wrapped callable.
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # [FUNCTION DOC] Forwards all positional and keyword arguments to the wrapped callable and
        # returns its result unchanged.
        return self._fn(*args, **kwargs)


class FakeWorkerManager:
    # [CLASS DOC] Fake worker manager. Exposes a shutdown attribute with a remote member (as a remote
    # actor method would) and a kill method; records killed handles and appends events to the shared
    # list.
    def __init__(self, events: list[str]) -> None:
        # [FUNCTION DOC] Keeps the shared event list, an empty killed-handles list, and a shutdown
        # object whose remote member is the private async shutdown method.
        self.events = events
        self.killed: list[object] = []
        self.shutdown = SimpleNamespace(remote=self._shutdown)

    async def _shutdown(self) -> None:
        # [FUNCTION DOC] HELPER: Appends the manager-shutdown event.
        self.events.append("manager_shutdown")

    def kill(self, handle: object) -> None:
        # [FUNCTION DOC] Records the handle in the killed list and appends the manager-kill event.
        self.killed.append(handle)
        self.events.append("manager_kill")


class FakeRolloutExecutor:
    # [CLASS DOC] Fake rollout executor. Its get, eval, save and report-eval-skip members are wrapped as
    # remote-method fakes. Per rollout id gates (async events) can block generation, a queue of errors
    # can make generations fail, and a queue of data packs supplies results.
    def __init__(self, events: list[str]) -> None:
        # [FUNCTION DOC] Keeps the shared event list; initializes empty generation gates (rollout id to
        # async event), generation errors list (entries may be none to mean no error), and generation
        # packs list; and wraps the four private async methods as remote-method fakes.
        self.events = events
        self.generation_gates: dict[int, asyncio.Event] = {}
        self.generation_errors: list[BaseException | None] = []
        self.generation_packs: list[RolloutDataPack] = []
        self.get = FakeRemoteMethod(self._get)
        self.eval = FakeRemoteMethod(self._eval)
        self.save = FakeRemoteMethod(self._save)
        self.report_eval_skip = FakeRemoteMethod(self._report_eval_skip)

    async def _get(self, rollout_id: int) -> RolloutDataPack:
        # [FUNCTION DOC] HELPER: Fake generation of one rollout. Appends a start event for the id; if a
        # gate exists for the id, waits until it is set; if the errors list is non-empty, pops its first
        # entry and, when that entry is not none, appends a failed event and raises it (a none entry is
        # consumed and generation continues). Otherwise pops the first queued pack, or makes an empty
        # data pack if none is queued, appends a done event and returns the pack.
        self.events.append(f"generate_start:{rollout_id}")
        if (gate := self.generation_gates.get(rollout_id)) is not None:
            await gate.wait()
        if self.generation_errors and (error := self.generation_errors.pop(0)) is not None:
            self.events.append(f"generate_failed:{rollout_id}")
            raise error
        pack = self.generation_packs.pop(0) if self.generation_packs else RolloutDataPack()
        self.events.append(f"generate_done:{rollout_id}")
        return pack

    async def _eval(self, rollout_id: int, **_kwargs: Any) -> None:
        # [FUNCTION DOC] HELPER: Appends an eval event for the id; extra keyword arguments are ignored.
        self.events.append(f"eval:{rollout_id}")

    async def _save(self, rollout_id: int) -> None:
        # [FUNCTION DOC] HELPER: Appends an executor-save event for the id.
        self.events.append(f"executor_save:{rollout_id}")

    async def dispose(self) -> None:
        # [FUNCTION DOC] Appends the executor-dispose event.
        self.events.append("executor_dispose")

    async def _report_eval_skip(self, rollout_id: int, reason: str) -> None:
        # [FUNCTION DOC] HELPER: Appends an eval-skip event containing the id and the reason.
        self.events.append(f"eval_skip:{rollout_id}:{reason}")


class FakeInferenceController:
    # [CLASS DOC] Fake inference controller. Every async method only appends a named event to the
    # shared list; the check-weights call also records its keyword arguments.
    def __init__(self, events: list[str]) -> None:
        # [FUNCTION DOC] Keeps the shared event list and an empty list of recorded check-weights
        # keyword argument mappings.
        self.events = events
        self.check_weights_calls: list[dict[str, Any]] = []

    async def prepare_rollout(self, rollout_id: int) -> None:
        # [FUNCTION DOC] Appends a prepare-rollout event carrying the id.
        self.events.append(f"prepare_rollout:{rollout_id}")

    async def prepare_eval(self) -> None:
        # [FUNCTION DOC] Appends the prepare-eval event.
        self.events.append("prepare_eval")

    async def check_weights(self, **kwargs: Any) -> None:
        # [FUNCTION DOC] Records the keyword arguments and appends the check-weights event.
        self.check_weights_calls.append(kwargs)
        self.events.append("check_weights")

    async def onload_kv(self) -> None:
        # [FUNCTION DOC] Appends the onload-kv event.
        self.events.append("onload_kv")

    async def onload_weights(self) -> None:
        # [FUNCTION DOC] Appends the onload-weights event.
        self.events.append("onload_weights")

    async def offload(self, tags: list[str]) -> None:
        # [FUNCTION DOC] Appends an offload event whose suffix is the tags joined by commas.
        self.events.append(f"offload:{','.join(tags)}")

    async def dispose(self) -> None:
        # [FUNCTION DOC] Appends the inference-dispose event.
        self.events.append("inference_dispose")


class FakeTrainingModel:
    # [CLASS DOC] Fake training model for one role (its name prefixes every event). Records trained and
    # saved rollout ids, the external data given per rollout id, and per rollout id train outputs and
    # started-signals (async events created on first access) that tests may preconfigure or await. An
    # optional hook is called with the external data on each train.
    def __init__(self, events: list[str], role: str) -> None:
        # [FUNCTION DOC] Keeps the event list and role; initializes empty trained and saved id lists,
        # external-data and train-output mappings, a defaulting mapping of train-started async events,
        # and no consume hook.
        self.events = events
        self.role = role
        self.trained: list[int] = []
        self.saved: list[int] = []
        self.external_data: dict[int, Any] = {}
        self.train_outputs: dict[int, Any] = {}
        self.train_started: dict[int, asyncio.Event] = defaultdict(asyncio.Event)
        self.consume_external_data: Callable[[Any], None] | None = None

    async def train(self, rollout_id: int, rollout_data: Any, external_data: Any = None) -> Any:
        # [FUNCTION DOC] Appends a role-prefixed train event, records the id, stores the external data
        # under the id, calls the consume hook with it if set, sets the train-started signal for the id
        # and returns the preconfigured train output for the id, else a string of the form role,
        # values, id. The rollout data argument is ignored.
        self.events.append(f"{self.role}_train:{rollout_id}")
        self.trained.append(rollout_id)
        self.external_data[rollout_id] = external_data
        if (consume := self.consume_external_data) is not None:
            consume(external_data)
        self.train_started[rollout_id].set()
        return self.train_outputs.get(rollout_id, f"{self.role}-values-{rollout_id}")

    async def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        # [FUNCTION DOC] Appends a role-prefixed save event and records the id. The force-sync flag is
        # ignored.
        self.events.append(f"{self.role}_save:{rollout_id}")
        self.saved.append(rollout_id)

    async def export_hf(self, rollout_id: int, path: str) -> None:
        # [FUNCTION DOC] Appends a role-prefixed export event with the id; the path is ignored.
        self.events.append(f"{self.role}_export_hf:{rollout_id}")

    async def offload(self) -> None:
        # [FUNCTION DOC] Appends a role-prefixed offload event.
        self.events.append(f"{self.role}_offload")

    async def onload(self) -> None:
        # [FUNCTION DOC] Appends a role-prefixed onload event.
        self.events.append(f"{self.role}_onload")

    async def clear_memory(self) -> None:
        # [FUNCTION DOC] Appends a role-prefixed clear-memory event.
        self.events.append(f"{self.role}_clear_memory")

    async def dispose(self) -> None:
        # [FUNCTION DOC] Appends a role-prefixed dispose event.
        self.events.append(f"{self.role}_dispose")


class FakeObjectStore(BaseObjectStore):
    # [CLASS DOC] In-memory object store. Values live in a mapping keyed by generated references; reads
    # are logged in a consumed list (in order, duplicates kept).
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the empty value mapping and the empty consumed list.
        self._values: dict[StoreObjectRef, Any] = {}
        self.consumed: list[StoreObjectRef] = []

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        # [FUNCTION DOC] Stores the value under a new mooncake-style reference whose payload is a
        # fake-object string numbered by the current store size, and returns the reference. The value
        # spec is ignored. KNOWN ISSUE: numbering uses the current size, so after a removal a later put
        # can reuse an existing payload and overwrite or collide with a live entry if references
        # compare by payload.
        ref = _MooncakeStoreObjectRef(payload=f"fake-object-{len(self._values)}")
        self._values[ref] = value
        return ref

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        # [FUNCTION DOC] Looks up the value (a missing reference raises a key error), logs the reference
        # as consumed, and returns a get result holding the value with a no-op release callback.
        value = self._values[ref]
        self.consumed.append(ref)
        return ObjectStoreGetResult(value=value, release_fn=lambda _value: None)

    def remove(self, ref: StoreObjectRef) -> None:
        # [FUNCTION DOC] Deletes the entry; a missing reference raises a key error.
        del self._values[ref]

    def contains(self, ref: StoreObjectRef) -> bool:
        # [FUNCTION DOC] Returns whether the reference is currently stored.
        return ref in self._values
