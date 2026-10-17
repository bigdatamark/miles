# [MODULE DOC]
# The Tinker gateway service: transport-independent logic behind the HTTP API. Manages client
# sessions with heartbeat leases, training models (each bound to one adapter slot in the trainer),
# sampling sessions, request futures, and the dispatcher loop that picks scheduled units and runs
# them on the trainer backend one at a time under a lock. The backend object (injected) provides
# async operations: load/unload/save/export of slots, forward and forward-backward, optimizer step,
# sampling, and a trainer liveness check. Failures of operations that may corrupt a slot close the
# model; user errors fail only the request. Checkpoints live under the configured root as
# tinker:// paths with metadata files.
"""Sessions, models, futures, and ordered trainer dispatch.

The trainer lock serializes trainer calls across dispatch, model creation, and lease expiry."""

import asyncio
import logging
import os
import time
import uuid

from miles.tinker.core.future import RequestFuture, RequestFutureStore
from miles.tinker.core.input_validation import (
    validate_batch_payload,
    validate_checkpoint_compatibility,
    validate_checkpoint_segment,
    validate_model_config,
    validate_sample_payload,
    validate_seq_id,
)
from miles.tinker.core.model_queue import ModelRequestQueue
from miles.tinker.core.scheduler import BarrierUnit, BatchUnit, RequestScheduler
from miles.tinker.core.types import (
    Command,
    CommandOp,
    GatewayConfig,
    ModelRecord,
    OwnershipError,
    SamplingSessionRecord,
    SessionRecord,
    UserInputError,
)
from miles.tinker.core.utils import (
    build_checkpoint_metadata,
    parse_tinker_path,
    read_checkpoint_metadata,
    resolve_checkpoint_dir,
    resolve_sampler_checkpoint,
)

logger = logging.getLogger(__name__)


class TinkerService:
    # [CLASS DOC]
    # State: injected backend; gateway config; future store; scheduler; tables of models, sessions
    # and sampling sessions by id; the set of free adapter slots; a wake-up event for the
    # dispatcher; the trainer lock serializing every trainer call (dispatch, model creation,
    # lease expiry); in-flight sampling tasks by request id with their session ids; in-flight
    # model creation tasks; counters for command arrival order and trainer batches; the first
    # background task error (which stops the dispatcher); and a bounded memory of why models closed
    # (so later requests get the reason). Must be used from a single event loop.
    def __init__(self, backend, config: GatewayConfig) -> None:
        # [FUNCTION DOC] Store the backend and config; create the future store, a scheduler with
        # the batch token budget, empty model/session tables, all slots free (zero up to slot count
        # minus one), the wake event, trainer lock, empty task registries, zero counters, no
        # background error and empty close reasons. Must run inside an event loop context for the
        # async primitives on older Python versions.
        self.backend = backend
        self.config = config
        self.futures = RequestFutureStore()
        self.scheduler = RequestScheduler(config.batch_token_budget)
        self.models: dict[str, ModelRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.sampling_sessions: dict[str, SamplingSessionRecord] = {}
        self.free_slots = set(range(config.n_slots))
        self._wake = asyncio.Event()
        self._trainer_lock = asyncio.Lock()
        self._sample_tasks: dict[str, tuple] = {}  # request_id -> (task, session_id)
        self._create_tasks: set = set()
        self._arrival_counter = 0
        self._batch_counter = 0
        self._background_error: BaseException | None = None
        # why each model closed, so later requests get the reason instead of "unknown model"
        self._close_reasons: dict[str, str] = {}

    async def run(self) -> None:
        # [FUNCTION DOC]
        # The dispatcher loop; runs until an error. Starts the lease sweeper as a background task
        # whose failure is recorded and wakes the loop. Each iteration: re-raise any recorded
        # background error; under the trainer lock first fail all queue heads with validation errors
        # (as user failures) and loop again if any; otherwise schedule the next unit and run it
        # (batch or barrier), then check the trainer is alive (raise a runtime error telling that
        # the trainer died if not, so the process exits and clients see refused connections) and
        # loop again; if nothing was scheduled, release the lock and wait for the wake event, then
        # clear it. On exit for any reason: cancel the sweeper, model creation tasks and sampling
        # tasks, and await them ignoring their errors so they cannot mask the original error.
        sweep_task = asyncio.create_task(self._run_lease_sweeper())
        sweep_task.add_done_callback(self._observe_background_task)
        try:
            while True:
                if self._background_error is not None:
                    raise self._background_error
                # unit selection shares the critical section with execution, so
                # lease expiry cannot reclaim a model queue between the two
                async with self._trainer_lock:
                    rejections = self.scheduler.ready_rejections()
                    if rejections:
                        for model_queue, pending in rejections:
                            await self._finish_request(
                                model_queue,
                                pending,
                                {"error": pending.command.validation_error, "error_category": "user"},
                            )
                        continue
                    unit = self.scheduler.schedule_next()
                    if unit is not None:
                        if isinstance(unit, BatchUnit):
                            await self._run_batch(unit)
                        else:
                            await self._run_barrier(unit)
                        if await self.backend.trainer_dead():
                            raise RuntimeError("the trainer workers died; exiting so clients get refused connections")
                        continue
                await self._wake.wait()
                self._wake.clear()
        finally:
            tasks = [sweep_task, *self._create_tasks, *(task for task, _ in self._sample_tasks.values())]
            for task in tasks:
                task.cancel()
            # Cleanup failures must not replace the error that stopped the dispatcher.
            await asyncio.gather(*tasks, return_exceptions=True)

    def _observe_background_task(self, task: asyncio.Task) -> None:
        # [FUNCTION DOC] HELPER: Done callback for background tasks. If the task was not cancelled
        # and raised, remember the first such error and wake the dispatcher so it re-raises it.
        if not task.cancelled() and (error := task.exception()) is not None:
            if self._background_error is None:
                self._background_error = error
            self._wake.set()

    def create_session(self, tenant: str) -> str:
        # [FUNCTION DOC] Register a session for the tenant with the heartbeat set to now and return
        # its id, "session-" followed by 32 random hex characters.
        session_id = f"session-{uuid.uuid4().hex}"
        self.sessions[session_id] = SessionRecord(tenant=tenant, last_heartbeat=time.monotonic())
        return session_id

    def _session_for(self, tenant: str, session_id: str) -> SessionRecord:
        # [FUNCTION DOC] HELPER: Return the tenant's session or raise a user input error saying
        # the session is unknown (also used for a session of another tenant, hiding its existence).
        session = self.sessions.get(session_id)
        if session is None or session.tenant != tenant:
            raise UserInputError(f"unknown session {session_id!r}; create a session first")
        return session

    def heartbeat(self, tenant: str, session_id: str) -> bool:
        # [FUNCTION DOC] Refresh the session lease: returns false if the session is unknown or
        # owned by another tenant (no error); otherwise sets the heartbeat to now and returns true.
        session = self.sessions.get(session_id)
        if session is None or session.tenant != tenant:
            return False
        session.last_heartbeat = time.monotonic()
        return True

    def create_model(self, tenant: str, payload: dict) -> tuple[str, str]:
        # [FUNCTION DOC]
        # Admit a create-model request; returns (request future id, model id). Does not wait for
        # slot initialization. The session must exist for the tenant (user input error otherwise);
        # the model sequence id must be an integer >= 0. Idempotent: a repeated sequence id returns
        # the earlier pair, renewing the future if it expired. Otherwise: the base model must equal
        # the served one; the LoRA config (default empty) is validated against the layout; rank
        # defaults to 32 and alpha to the configured alpha or twice the rank (float); a user input
        # error is raised if no slot is free; the smallest free slot is taken; a model id "model-"
        # plus 12 hex characters is made; a pending future, a model record and a model queue
        # (registered with the scheduler) are created; a background task initializes the slot under
        # the trainer lock (tracked, errors observed); the sequence id is remembered.
        """Two-phase like every command: allocate now, initialize the slot behind the future."""
        session = self._session_for(tenant, payload["session_id"])
        model_seq_id = validate_seq_id(payload["model_seq_id"], "model_seq_id", minimum=0)
        if (previous := session.models_by_seq.get(model_seq_id)) is not None:
            request_id, model_id = previous
            request_id = self.futures.request_id_for_retry(request_id, model_id, tenant)
            session.models_by_seq[model_seq_id] = (request_id, model_id)
            return request_id, model_id
        base_model = payload["base_model"]
        if base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        lora_config = payload.get("lora_config") or {}
        validate_model_config(lora_config, self.config)
        rank = lora_config.get("rank", 32)
        alpha = self.config.lora_alpha if self.config.lora_alpha is not None else float(2 * rank)
        if not self.free_slots:
            raise UserInputError(f"no free adapter slots (capacity {self.config.n_slots})")
        slot = min(self.free_slots)
        self.free_slots.remove(slot)

        model_id = f"model-{uuid.uuid4().hex[:12]}"
        future = self.futures.create(model_id, tenant)
        record = ModelRecord(
            model_id=model_id,
            session_id=payload["session_id"],
            tenant=tenant,
            slot=slot,
            base_model=base_model,
            lora_rank=rank,
            lora_alpha=alpha,
            create_request_id=future.request_id,
        )
        self.models[model_id] = record
        self.scheduler.add_model_queue(ModelRequestQueue(model_id, tenant, slot))
        task = asyncio.create_task(self._run_create_model(record))
        self._create_tasks.add(task)
        task.add_done_callback(self._create_tasks.discard)
        task.add_done_callback(self._observe_background_task)
        session.models_by_seq[model_seq_id] = (future.request_id, model_id)
        return future.request_id, model_id

    async def _run_create_model(self, record: ModelRecord) -> None:
        # [FUNCTION DOC] HELPER: Background task. Under the trainer lock: do nothing if the model
        # was closed or replaced meanwhile; otherwise ask the backend to load (initialize) the
        # slot with rank and alpha, mark the slot as initialized (even on failure, so it gets
        # unloaded), and on failure close the model with the backend's error as a server error;
        # on success resolve the create future with the operation name and model id.
        async with self._trainer_lock:
            if self.models.get(record.model_id) is not record:
                return
            failure = await self.backend.load_slot(record.slot, record.lora_rank, record.lora_alpha)
            record.slot_initialized = True
            if failure is not None:
                await self._close_model(record.model_id, failure["error"], "server")
                return
            self.futures.resolve(record.create_request_id, {"op": "create_model", "model_id": record.model_id})

    def get_model(self, tenant: str, model_id: str) -> ModelRecord:
        # [FUNCTION DOC] Return the model record. Unknown ids raise a user input error, mentioning
        # the remembered closing reason if the model was closed earlier; a model of another tenant
        # raises an ownership error.
        record = self.models.get(model_id)
        if record is None:
            reason = self._close_reasons.get(model_id)
            if reason is not None:
                raise UserInputError(f"model {model_id!r} was unloaded: {reason}")
            raise UserInputError(f"unknown model {model_id!r}")
        if record.tenant != tenant:
            raise OwnershipError(f"model {model_id} does not belong to this tenant")
        return record

    async def _close_model(self, model_id: str, error: str, category: str) -> None:
        # [FUNCTION DOC] Retire a model (caller holds the trainer lock; idempotent for unknown
        # ids). Removes the record, remembers the reason (keeping at most four times the slot count
        # of reasons, evicting the oldest), removes its queue from the scheduler, and fails every
        # still-pending future of the model (the create future and all command futures) with the
        # error and category. If the trainer is dead, stop there (slot not freed). Otherwise, if its
        # slot was initialized, ask the backend to unload it; an unload failure is logged and the
        # slot stays unavailable (not returned to the free set); otherwise the slot becomes free.
        """Free a model's slot and fail its pending requests; requires the trainer lock. Idempotent."""
        record = self.models.pop(model_id, None)
        if record is None:
            return
        self._close_reasons[model_id] = error
        while len(self._close_reasons) > 4 * self.config.n_slots:
            self._close_reasons.pop(next(iter(self._close_reasons)))
        self.scheduler.remove_model_queue(model_id)
        for request_id in [record.create_request_id, *record.request_id_by_seq.values()]:
            if self.futures.get(request_id, record.tenant) is not None:
                self.futures.fail(request_id, error, category)
        if await self.backend.trainer_dead():
            return
        if record.slot_initialized:
            failure = await self.backend.unload_slot(record.slot)
            if failure is not None:
                logger.error("slot %s remains unavailable after unload failed: %s", record.slot, failure["error"])
                return
        self.free_slots.add(record.slot)

    def submit(self, tenant: str, op: str, payload: dict) -> str:
        # [FUNCTION DOC]
        # Admit a trainer command and return its future id. Raises a user input error for an
        # unknown operation name, unknown model (ownership error for another tenant), or invalid
        # sequence id (integer >= 1). Idempotent per model and sequence id: a retry returns the
        # earlier future id (renewed if expired) so gradients are not accumulated twice. Otherwise
        # creates a future, records it, bumps the arrival counter, and takes a validation error from
        # the payload if the decoding layer already found one, else validates the batch payload
        # (content errors become the command's validation error text instead of raising). Queues
        # the command (without the validation error key in its payload) and wakes the dispatcher.
        """Admit a decoded command; content errors settle its future as a user failure."""
        try:
            op = CommandOp(op)
        except ValueError:
            raise UserInputError(f"unknown command op {op!r}") from None
        model_id = payload["model_id"]
        record = self.get_model(tenant, model_id)
        seq_id = validate_seq_id(payload["seq_id"], "seq_id")
        model_queue = self.scheduler.model_queue(model_id)

        # retries must not accumulate gradients twice
        if seq_id in record.request_id_by_seq:
            request_id = self.futures.request_id_for_retry(record.request_id_by_seq[seq_id], model_id, tenant)
            record.request_id_by_seq[seq_id] = request_id
            return request_id

        future = self.futures.create(model_id, tenant)
        record.request_id_by_seq[seq_id] = future.request_id
        self._arrival_counter += 1
        validation_error = payload.get("validation_error")
        if validation_error is None:
            try:
                validate_batch_payload(op, payload, self.config)
            except UserInputError as error:
                validation_error = str(error)
        model_queue.submit(
            Command(
                model_id=model_id,
                seq_id=seq_id,
                op=op,
                payload={key: value for key, value in payload.items() if key != "validation_error"},
                request_id=future.request_id,
                arrival=self._arrival_counter,
                validation_error=validation_error,
            )
        )
        self._wake.set()
        return future.request_id

    def retrieve_future(self, tenant: str, request_id: str) -> RequestFuture | None:
        # [FUNCTION DOC] Look up a future for the tenant: none if unknown or expired; ownership
        # error if another tenant's.
        return self.futures.get(request_id, tenant)

    async def _run_batch(self, batch: BatchUnit) -> None:
        # [FUNCTION DOC]
        # HELPER: Execute a batch unit on the trainer (caller holds the lock). Datums are sorted by
        # model slot (so each slot's datums are contiguous) and passed with their slots, a new batch
        # number, the loss function and config to the backend forward-backward or forward-only
        # call. A user input error from the backend fails all requests in the batch as user
        # errors; an error mapping result fails them as server errors. Otherwise the output list
        # must match the datum count (assertion error) and outputs are recorded per datum in the
        # same order; each request that receives its last output is finished with a result
        # holding the operation and the ordered outputs.
        # slot-contiguous order; outputs come back aligned to it
        refs = sorted(batch.datums, key=lambda ref: ref.model_queue.slot)
        slot_datums = [(ref.model_queue.slot, ref.datum) for ref in refs]
        self._batch_counter += 1
        forward = (
            self.backend.forward_backward if batch.op == CommandOp.FORWARD_BACKWARD else self.backend.forward_only
        )
        try:
            outputs = await forward(self._batch_counter, slot_datums, batch.loss_fn, batch.loss_fn_config)
        except UserInputError as error:
            await self._fail_batch(batch, str(error), "user")
            return
        if isinstance(outputs, dict) and "error" in outputs:
            await self._fail_batch(batch, outputs["error"], "server")
            return

        assert len(outputs) == len(refs), f"unit returned {len(outputs)} outputs for {len(refs)} datums"
        for ref, output in zip(refs, outputs, strict=True):
            request = ref.request
            if request.record_output(ref.local_index, output):
                await self._finish_request(
                    ref.model_queue, request, {"op": request.command.op, "outputs": request.outputs}
                )

    async def _fail_batch(self, batch: BatchUnit, error: str, category: str) -> None:
        # [FUNCTION DOC] HELPER: Finish each distinct request in the batch (keyed by request id,
        # so a request with several datums is failed once) with the error and category. Failing a
        # forward-backward request closes its model.
        requests = {ref.request.command.request_id: (ref.model_queue, ref.request) for ref in batch.datums}
        for model_queue, pending in requests.values():
            await self._finish_request(model_queue, pending, {"error": error, "error_category": category})

    async def _run_barrier(self, barrier: BarrierUnit) -> None:
        # [FUNCTION DOC] HELPER: Execute a barrier unit (caller holds the lock). Dispatch by
        # operation to get one outcome per entry; user input and ownership errors become a user
        # failure outcome for every entry. Then finish each entry's request with its outcome.
        # Other exceptions propagate and stop the dispatcher.
        try:
            outcomes = await self._dispatch_barrier_op(barrier)
        except (UserInputError, OwnershipError) as error:
            outcomes = [{"error": str(error), "error_category": "user"} for _ in barrier.entries]
        for (model_queue, pending), outcome in zip(barrier.entries, outcomes, strict=True):
            await self._finish_request(model_queue, pending, outcome)

    async def _dispatch_barrier_op(self, barrier: BarrierUnit) -> list[dict]:
        # [FUNCTION DOC] HELPER: Run the barrier's operation and return one outcome mapping per
        # entry (a result or an error mapping). Optimizer steps handle all entries together; the
        # other ops (save state, load state, save sampler weights) require exactly one entry and
        # use the model's record and the command payload. An unknown op raises a user input error.
        """Return one result or error per entry; only the caller settles futures and retires models."""
        if barrier.op == CommandOp.OPTIM_STEP:
            return await self._step_optimizers(barrier.entries)
        ((model_queue, pending),) = barrier.entries  # every other barrier is single-entry
        record = self.models[model_queue.model_id]
        payload = pending.command.payload
        if barrier.op == CommandOp.SAVE_STATE:
            return [await self._save_state(record, payload)]
        if barrier.op == CommandOp.LOAD_STATE:
            return [await self._load_state(record, payload)]
        if barrier.op == CommandOp.SAVE_WEIGHTS_FOR_SAMPLER:
            return [await self._save_weights_for_sampler(record, payload)]
        raise UserInputError(f"unknown barrier op {barrier.op!r}")

    async def _step_optimizers(self, entries: list) -> list[dict]:
        # [FUNCTION DOC] HELPER: One backend optimizer call for all entries, passing a mapping from
        # each model's slot to its optimizer parameters from the payload. Per entry, the slot's
        # outcome is returned as is if it holds an error, else a result with the operation name and
        # a metrics mapping converted to floats.
        adam_params_by_slot = {
            model_queue.slot: pending.command.payload["adam_params"] for model_queue, pending in entries
        }
        slot_outcomes = await self.backend.optim_step(adam_params_by_slot)
        outcomes = []
        for model_queue, _ in entries:
            outcome = slot_outcomes[model_queue.slot]
            if "error" in outcome:
                outcomes.append(outcome)
            else:
                outcomes.append({"op": "optim_step", "metrics": {key: float(value) for key, value in outcome.items()}})
        return outcomes

    async def _finish_request(self, model_queue, pending, outcome: dict) -> None:
        # [FUNCTION DOC] HELPER: Settle a request from an outcome mapping. Error outcomes (category
        # defaults to server): if the operation requires closing the model on failure
        # (forward-backward, optimizer step, load state), close the model with an explanatory
        # message that includes the error (this fails all its pending futures) and return without
        # touching the queue; otherwise fail the future. Success outcomes resolve the future with
        # the mapping. Then remove the request from its model queue.
        if "error" in outcome:
            category = outcome.get("error_category", "server")
            if pending.command.op.requires_model_close_on_failure():
                await self._close_model(
                    model_queue.model_id,
                    f"model training failed ({outcome['error']}); create a new model and restore from a checkpoint",
                    category,
                )
                return
            self.futures.fail(pending.command.request_id, outcome["error"], category)
        else:
            self.futures.resolve(pending.command.request_id, outcome)
        model_queue.finish(pending)

    async def _save_state(self, record: ModelRecord, payload: dict) -> dict:
        # [FUNCTION DOC] HELPER: Save the model's slot to a checkpoint. The name is the payload name
        # or, if empty, "checkpoint-" plus the zero-padded 6-digit sequence id; validated as a path
        # segment. Raises a user input error if the directory exists and overwrite is not set. The
        # backend saves the slot to the weights directory under the model with checkpoint metadata;
        # a failure mapping is returned as the outcome. Success returns the operation name and the
        # tinker path of the checkpoint.
        """Save parameters and optimizer state; call after optim_step to persist accumulated training work."""
        name = payload["name"] or f"checkpoint-{payload['seq_id']:06d}"
        validate_checkpoint_segment(name)
        checkpoint_dir = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "weights", name)
        if not payload["overwrite"] and os.path.exists(checkpoint_dir):
            raise UserInputError(f"checkpoint {name!r} already exists; pass overwrite=True to replace it")
        failure = await self.backend.save_slot(
            record.slot, checkpoint_dir, metadata=build_checkpoint_metadata(record, self.config)
        )
        if failure is not None:
            return failure
        return {"op": "save_state", "path": f"tinker://{record.model_id}/weights/{name}"}

    async def _load_state(self, record: ModelRecord, payload: dict) -> dict:
        # [FUNCTION DOC] HELPER: Load a saved training checkpoint into the model's slot. The path
        # must parse and be of kind weights (user input error for sampler weights). Reads the
        # checkpoint metadata, checking ownership against the optional access token string from the
        # payload (treated as the source tenant) or else the model's own tenant, and checks
        # compatibility with this model (rank, alpha, base model, layout). The backend loads the slot
        # from the resolved directory, optionally with the optimizer state per the payload flag. A
        # failure mapping is returned; success returns the operation name only.
        source_id, kind, name = parse_tinker_path(payload["path"])
        if kind != "weights":
            raise UserInputError("cannot load sampler weights into a training model; use a save_state checkpoint")
        checkpoint_dir = os.path.realpath(resolve_checkpoint_dir(self.config.checkpoint_root, source_id, kind, name))
        source_tenant = payload.get("weights_access_token")
        if source_tenant is None:
            source_tenant = record.tenant
        meta = read_checkpoint_metadata(checkpoint_dir, source_tenant, payload["path"])
        validate_checkpoint_compatibility(meta, record, self.config, payload["path"])
        failure = await self.backend.load_slot(
            record.slot,
            record.lora_rank,
            record.lora_alpha,
            ckpt_path=checkpoint_dir,
            load_optimizer=payload["optimizer"],
        )
        if failure is not None:
            return failure
        return {"op": "load_state"}

    async def _save_weights_for_sampler(self, record: ModelRecord, payload: dict) -> dict:
        # [FUNCTION DOC] HELPER: Export the slot's adapter for the inference engines. The version
        # name is the payload sampler path if given (validated as a path segment) else the model's
        # next sampler version number as text (the counter is incremented immediately, so failed
        # saves leave gaps). A user input error is raised if the target directory already exists.
        # The backend exports the slot with rank, alpha and checkpoint metadata; a failure mapping is
        # returned. Success returns the operation name and the tinker sampler path; when no name
        # was given it also creates a sampling session bound to the new path and returns its id.
        version = payload.get("sampler_path")
        if version is None:
            version = str(record.next_sampler_version)
            record.next_sampler_version += 1
        else:
            validate_checkpoint_segment(version)
        path = resolve_checkpoint_dir(self.config.checkpoint_root, record.model_id, "sampler_weights", version)
        if os.path.exists(path):
            raise UserInputError(f"sampler weights {version!r} already exist; save under a new name")
        failure = await self.backend.export_slot(
            record.slot,
            record.lora_rank,
            record.lora_alpha,
            path,
            metadata=build_checkpoint_metadata(record, self.config),
        )
        if failure is not None:
            return failure
        result = {
            "op": "save_weights_for_sampler",
            "path": f"tinker://{record.model_id}/sampler_weights/{version}",
        }
        if payload.get("sampler_path") is None:
            # unnamed saves return a sampling session bound to the new version
            result["sampling_session_id"] = self._new_sampling_session(
                record.tenant, record.session_id, result["path"]
            )
        return result

    def weights_info(self, tenant: str, tinker_path: str) -> dict:
        # [FUNCTION DOC] Read the checkpoint's metadata (file I/O; ownership checked for the
        # tenant) and return a mapping with base model, is-LoRA (always true), LoRA rank and the
        # three train flags. Path and metadata problems raise user input or ownership errors.
        """What the SDK needs to rebuild a training client from a checkpoint."""
        model_id, kind, name = parse_tinker_path(tinker_path)
        meta = read_checkpoint_metadata(
            resolve_checkpoint_dir(self.config.checkpoint_root, model_id, kind, name), tenant, tinker_path
        )
        return {
            "base_model": meta["base_model"],
            "is_lora": True,
            "lora_rank": meta["lora_rank"],
            "train_attn": meta["train_attn"],
            "train_mlp": meta["train_mlp"],
            "train_unembed": meta["train_unembed"],
        }

    def create_sampling_session(self, tenant: str, payload: dict) -> str:
        # [FUNCTION DOC] Create (idempotently) a sampling session within a client session and
        # return its id. The session must exist for the tenant; an optional base model must equal
        # the served one; the sampling sequence id must be an integer >= 0. A repeated sequence id
        # returns the earlier id. Otherwise a new sampling session is registered for the optional
        # model path (validated by resolving the sampler checkpoint) and remembered under the
        # sequence id.
        session = self._session_for(tenant, payload["session_id"])
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        seq_id = validate_seq_id(payload["sampling_session_seq_id"], "sampling_session_seq_id", minimum=0)
        if (previous := session.sampling_sessions_by_seq.get(seq_id)) is not None:
            return previous
        sampling_session_id = self._new_sampling_session(tenant, payload["session_id"], payload.get("model_path"))
        session.sampling_sessions_by_seq[seq_id] = sampling_session_id
        return sampling_session_id

    def _new_sampling_session(self, tenant: str, session_id: str, model_path: str | None) -> str:
        # [FUNCTION DOC] HELPER: Register a sampling session under a fresh id ("sampling-" plus 32
        # hex characters) for the tenant, owning session and optional sampler model path. A given
        # path is validated first by resolving its checkpoint (raising user input / ownership
        # errors). Returns the id.
        if model_path is not None:
            resolve_sampler_checkpoint(self.config.checkpoint_root, tenant, model_path, self.config.base_model)
        sampling_session_id = f"sampling-{uuid.uuid4().hex}"
        self.sampling_sessions[sampling_session_id] = SamplingSessionRecord(
            tenant=tenant, model_path=model_path, session_id=session_id
        )
        return sampling_session_id

    def get_sampler(self, tenant: str, sampling_session_id: str) -> dict:
        # [FUNCTION DOC] Describe a sampling session: unknown id raises a user input error, another
        # tenant's raises an ownership error; returns a mapping with the sampler id, the served base
        # model and the model path (may be none).
        session = self.sampling_sessions.get(sampling_session_id)
        if session is None:
            raise UserInputError(f"unknown sampling session {sampling_session_id!r}")
        if session.tenant != tenant:
            raise OwnershipError("sampling session does not belong to this tenant")
        return {
            "sampler_id": sampling_session_id,
            "base_model": self.config.base_model,
            "model_path": session.model_path,
        }

    def submit_sample(self, tenant: str, payload: dict) -> tuple[str, list[str]]:
        # [FUNCTION DOC]
        # Start a sampling request; returns (request future id, list of sequence ids, one per
        # requested sample). Does not wait for the result. Steps: an optional base model must equal
        # the served one. If the payload names a sampling session: it must exist and belong to the
        # tenant; the model path defaults to the session's; the sequence id (integer >= 0) is
        # checked; a repeated sequence id returns the earlier pair, renewing the future if it
        # expired. The sampling payload is validated. If a model path is set, its sampler checkpoint
        # is resolved to an adapter name and directory (ownership and base model checked). A future
        # keyed by the model path (or "base") is created, fresh sequence ids "seq-" plus 32 hex
        # characters are generated (num_samples, default 1), and a background task runs the sampling
        # on the backend (tracked with the owning client session id for lease expiry; errors
        # observed). If a sampling session was used, the pair is remembered under the sequence id.
        base_model = payload.get("base_model")
        if base_model is not None and base_model != self.config.base_model:
            raise UserInputError(f"this gateway serves {self.config.base_model!r}, not {base_model!r}")
        model_path = payload.get("model_path")
        sampling_session = None
        if payload.get("sampling_session_id"):
            sampling_session_id = payload["sampling_session_id"]
            sampling_session = self.sampling_sessions.get(sampling_session_id)
            if sampling_session is None:
                raise UserInputError(
                    f"unknown sampling session {sampling_session_id!r}; create a sampling session first"
                )
            if sampling_session.tenant != tenant:
                raise OwnershipError("sampling session does not belong to this tenant")
            model_path = model_path or sampling_session.model_path
            seq_id = validate_seq_id(payload["seq_id"], "seq_id", minimum=0)
            if (previous := sampling_session.samples_by_seq.get(seq_id)) is not None:
                request_id, sequence_ids = previous
                request_id = self.futures.request_id_for_retry(request_id, model_path or "base", tenant)
                sampling_session.samples_by_seq[seq_id] = (request_id, sequence_ids)
                return request_id, sequence_ids
        validate_sample_payload(payload, self.config)
        lora_name, lora_path = (
            resolve_sampler_checkpoint(self.config.checkpoint_root, tenant, model_path, self.config.base_model)
            if model_path
            else (None, None)
        )
        future = self.futures.create(model_path or "base", tenant)
        sequence_ids = [f"seq-{uuid.uuid4().hex}" for _ in range(payload.get("num_samples", 1))]
        task = asyncio.create_task(self._run_sample(future.request_id, sequence_ids, payload, lora_name, lora_path))
        session_id = sampling_session.session_id if sampling_session is not None else None
        self._sample_tasks[future.request_id] = (task, session_id)
        task.add_done_callback(lambda _t, rid=future.request_id: self._sample_tasks.pop(rid, None))
        task.add_done_callback(self._observe_background_task)
        if sampling_session is not None:
            sampling_session.samples_by_seq[seq_id] = (future.request_id, sequence_ids)
        return future.request_id, sequence_ids

    async def _run_sample(
        self,
        request_id: str,
        sequence_ids: list[str],
        payload: dict,
        lora_name: str | None,
        lora_path: str | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Background sampling task. Calls the backend sample with the
        # payload, adapter name and directory. Cancellation fails the future as a user error
        # "cancelled" (and is swallowed, not re-raised); a user input error fails it as user; a
        # result mapping with an error key fails it as server; otherwise each returned sequence
        # gets its sequence id assigned in order (counts must match, else a value error) and the
        # future resolves with the operation name plus the backend result fields.
        try:
            result = await self.backend.sample(payload, lora_name, lora_path)
        except asyncio.CancelledError:
            self.futures.fail(request_id, "cancelled", "user")
        except UserInputError as error:
            self.futures.fail(request_id, str(error), "user")
        else:
            if "error" in result:
                self.futures.fail(request_id, result["error"], "server")
            else:
                for sequence_id, sequence in zip(sequence_ids, result["sequences"], strict=True):
                    sequence["sequence_id"] = sequence_id
                self.futures.resolve(request_id, {"op": "sample", **result})

    def cancel(self, tenant: str, request_id: str) -> None:
        # [FUNCTION DOC] Cancel the sampling task of a request. Does nothing if the future is
        # unknown or expired (ownership error if another tenant's) or if no sampling task is
        # tracked for it (training commands). The task's cancellation handler then fails the future.
        """Cancel an in-flight sampling future; training commands have no
        cancel in the protocol and are ignored."""
        if self.futures.get(request_id, tenant) is None:
            return
        entry = self._sample_tasks.get(request_id)
        if entry is not None:
            entry[0].cancel()

    async def _run_lease_sweeper(self) -> None:
        # [FUNCTION DOC] HELPER: Background loop: every 30 seconds run the session expiry pass.
        # Exceptions end the loop and are observed by the dispatcher.
        """Reclaim from stale sessions: cancel sampling, unload models, free
        slots. Training state dies with the lease; only checkpoints survive."""
        while True:
            await asyncio.sleep(30)
            await self._expire_sessions()

    async def _expire_sessions(self) -> None:
        # [FUNCTION DOC] HELPER: Under the trainer lock, find sessions whose last heartbeat is at
        # least the lease timeout old and delete them, delete their sampling sessions, fail
        # (lease expired, user category) and cancel the sampling tasks started under them (tasks
        # without a client session are unaffected), and close their models (freeing slots, failing
        # pending futures), logging warnings.
        async with self._trainer_lock:
            now = time.monotonic()
            expired_sessions = {
                session_id
                for session_id, session in self.sessions.items()
                if now - session.last_heartbeat >= self.config.lease_timeout_s
            }
            for session_id in expired_sessions:
                del self.sessions[session_id]
            for sampling_session_id, record in list(self.sampling_sessions.items()):
                if record.session_id in expired_sessions:
                    del self.sampling_sessions[sampling_session_id]

            for request_id, (task, session_id) in list(self._sample_tasks.items()):
                if session_id in expired_sessions:
                    logger.warning(f"lease expired for session of sample {request_id}; cancelling")
                    self.futures.fail(request_id, "lease expired", "user")
                    task.cancel()
            for model_id, record in list(self.models.items()):
                if record.session_id in expired_sessions:
                    logger.warning(f"lease expired for {model_id}; freeing slot {record.slot}")
                    await self._close_model(model_id, "lease expired", "user")
