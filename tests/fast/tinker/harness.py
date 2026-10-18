# [MODULE DOC] HELPER: Shared fakes and builders for the tinker gateway tests: a recording fake
# trainer backend with configurable failures, a gateway config builder (base model "base", 2
# slots, vocabulary size 128000 by default), service construction, training-datum and RL-datum
# builders, forward/backward payload and command builders, model creation helpers and a future
# polling helper. The constant ADAM holds a default optimizer parameter mapping (learning rate
# 1e-4, betas 0.9 and 0.95, epsilon 1e-12, no weight decay, gradient clip norm 1.0).
"""Shared fakes for the tinker gateway suite."""

import asyncio

from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.tinker.core.future import DONE, PENDING, RequestFuture
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import Command, CommandOp, GatewayConfig

ADAM = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-12,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


class FakeBackend:
    # [CLASS DOC] HELPER: Fake trainer backend. Every operation is recorded as (name, keyword
    # arguments) in the calls list and returns deterministic output unless a failure is
    # configured. Failure injection: the fail-next attribute applies to the very next recorded
    # call (then clears); the fail-on mapping applies to the next call of a named operation (then
    # removed). A configured failure that is an exception marks the backend dead and is raised; a
    # mapping failure is returned as the operation's result. The optim-outcomes mapping gives
    # per-slot optimizer results.
    """Record calls with deterministic outputs and configurable failures."""

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes empty call list, no pending failure, not dead, empty
        # per-slot optimizer outcomes and empty per-operation failure mapping.
        self.calls: list[tuple[str, dict]] = []
        self.fail_next: Exception | dict | None = None
        self.dead = False
        self.optim_outcomes: dict[int, dict] = {}
        self.fail_on: dict[str, Exception | dict] = {}

    def _record(self, name: str, **kwargs) -> dict | None:
        # [FUNCTION DOC] HELPER: Appends (name, kwargs) to the calls list, then takes the pending
        # next failure (clearing it) or, if none, pops the failure registered for this operation
        # name. If the failure is an exception, marks the backend dead and raises it; otherwise
        # returns the failure mapping (or none when no failure).
        self.calls.append((name, kwargs))
        failure, self.fail_next = self.fail_next, None
        if failure is None:
            failure = self.fail_on.pop(name, None)
        if isinstance(failure, Exception):
            self.dead = True
            raise failure
        return failure

    def named(self, name: str) -> list[dict]:
        # [FUNCTION DOC] HELPER: Returns the keyword-argument mappings of all recorded calls with
        # the given operation name, in call order.
        return [kwargs for called, kwargs in self.calls if called == name]

    async def trainer_dead(self):
        # [FUNCTION DOC] HELPER: Async; returns the dead flag (set when an injected exception was
        # raised).
        return self.dead

    async def load_slot(self, slot, rank, alpha, ckpt_path=None, load_optimizer=True):
        # [FUNCTION DOC] HELPER: Async; records a load_slot call with its arguments and returns
        # the injected failure result or none.
        return self._record(
            "load_slot", slot=slot, rank=rank, alpha=alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer
        )

    async def unload_slot(self, slot):
        # [FUNCTION DOC] HELPER: Async; records an unload_slot call and returns the injected
        # failure result or none.
        return self._record("unload_slot", slot=slot)

    async def forward_backward(self, batch_id, slot_datums, loss_fn, loss_fn_config):
        # [FUNCTION DOC] HELPER: Async; records a forward_backward call. If a failure is injected
        # returns it; otherwise returns one result per (slot, datum) pair with loss 1.0 and a zero
        # logprob list of the datum's target length.
        failure = self._record(
            "forward_backward",
            batch_id=batch_id,
            slot_datums=slot_datums,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        if failure is not None:
            return failure
        return [{"loss": 1.0, "logprobs": [0.0] * datum["target_len"]} for _, datum in slot_datums]

    async def forward_only(self, batch_id, slot_datums, loss_fn, loss_fn_config):
        # [FUNCTION DOC] HELPER: Async; like forward_backward but records forward_only and the
        # per-datum loss is 0.0.
        failure = self._record(
            "forward_only", batch_id=batch_id, slot_datums=slot_datums, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )
        if failure is not None:
            return failure
        return [{"loss": 0.0, "logprobs": [0.0] * datum["target_len"]} for _, datum in slot_datums]

    async def optim_step(self, adam_params_by_slot):
        # [FUNCTION DOC] HELPER: Async; records an optim_step call. On an injected failure returns
        # a mapping giving every requested slot that failure; otherwise returns, per requested
        # slot, the configured outcome or a default with gradient norm 0.5 plus the slot number.
        failure = self._record("optim_step", adam_params_by_slot=adam_params_by_slot)
        if failure is not None:
            return {slot: failure for slot in adam_params_by_slot}
        return {slot: self.optim_outcomes.get(slot, {"grad_norm": 0.5 + slot}) for slot in adam_params_by_slot}

    async def save_slot(self, slot, path, metadata=None):
        # [FUNCTION DOC] HELPER: Async; records a save_slot call and, on success with metadata,
        # writes a checkpoint directory (overwriting allowed). Returns the injected failure or none.
        return self._write_checkpoint("save_slot", path, metadata, slot=slot)

    async def export_slot(self, slot, rank, alpha, path, metadata=None):
        # [FUNCTION DOC] HELPER: Async; records an export_slot call (slot, rank, alpha, path,
        # metadata) and, on success with metadata, writes a checkpoint directory without
        # overwriting. Returns the injected failure or none.
        return self._write_checkpoint("export_slot", path, metadata, slot=slot, rank=rank, alpha=alpha)

    def _write_checkpoint(self, name, path, metadata, **kwargs):
        # [FUNCTION DOC] HELPER: Records the call (path, metadata and extra arguments) under the
        # name. If no failure was injected and metadata is given, writes an empty checkpoint
        # directory at the path with the metadata (overwrite allowed only for save_slot). Returns
        # the injected failure or none; may raise the injected exception.
        failure = self._record(name, path=path, metadata=metadata, **kwargs)
        if failure is None and metadata is not None:
            write_checkpoint_dir(path, lambda _: None, metadata=metadata, overwrite=name == "save_slot")
        return failure

    async def sample(self, payload, lora_name, lora_path=None):
        # [FUNCTION DOC] HELPER: Async; records a sample call. On an injected failure returns it;
        # otherwise returns a mapping with as many sequences as the payload's number of samples,
        # each with tokens 1 and 2, zero logprobs and stop reason stop.
        failure = self._record("sample", payload=payload, lora_name=lora_name, lora_path=lora_path)
        if failure is not None:
            return failure
        return {
            "sequences": [
                {"tokens": [1, 2], "logprobs": [0.0, 0.0], "stop_reason": "stop"}
                for _ in range(payload["num_samples"])
            ]
        }


def make_config(checkpoint_root, **overrides) -> GatewayConfig:
    # [FUNCTION DOC] HELPER: Builds a gateway config from defaults (base model "base", 2 slots,
    # the given checkpoint root as text, vocabulary size 128000) overridden by keyword arguments.
    defaults = dict(base_model="base", n_slots=2, checkpoint_root=str(checkpoint_root), vocab_size=128000)
    return GatewayConfig(**{**defaults, **overrides})


def make_service(checkpoint_root, **config_overrides) -> TinkerService:
    # [FUNCTION DOC] HELPER: Builds a tinker service over a new fake backend and a test config
    # with the given overrides.
    return TinkerService(FakeBackend(), make_config(checkpoint_root, **config_overrides))


def datum(tokens: int = 3) -> dict:
    # [FUNCTION DOC] HELPER: Builds a supervised training datum mapping for the given number of
    # target tokens (default 3): input tokens are 0 through the count (count plus one values),
    # target tokens are 1 through the count, the target length is the count, and weights are
    # all 1.0.
    return {
        "tokens": list(range(tokens + 1)),
        "target_tokens": list(range(1, tokens + 1)),
        "target_len": tokens,
        "weights": [1.0] * tokens,
    }


def rl_datum(tokens: int = 3) -> dict:
    # [FUNCTION DOC] HELPER: Builds a reinforcement-learning datum mapping with the same tokens
    # and target tokens as the supervised datum but with zero sampling logprobs and advantages of
    # 1.0 (per target token) instead of weights.
    """RL losses read logprobs+advantages and reject the cross-entropy weights."""
    return {
        "tokens": list(range(tokens + 1)),
        "target_tokens": list(range(1, tokens + 1)),
        "target_len": tokens,
        "sampling_logprobs": [0.0] * tokens,
        "advantages": [1.0] * tokens,
    }


def fb_payload(model_id: str, seq_id: int, datums: list[dict], loss_fn: str = "cross_entropy") -> dict:
    # [FUNCTION DOC] HELPER: Builds a forward/backward request payload mapping with the model id,
    # sequence id, datums, loss function name (default cross entropy) and an empty loss config.
    return {"model_id": model_id, "seq_id": seq_id, "datums": datums, "loss_fn": loss_fn, "loss_fn_config": {}}


def command(model_id: str, seq_id: int, op: str, payload: dict, arrival: int) -> Command:
    # [FUNCTION DOC] HELPER: Builds a gateway command record with the model id, sequence id, the
    # operation parsed into the command-operation enum (raises a value error for unknown names),
    # the payload, a request id of the form req- plus sequence id, and the arrival order number.
    return Command(
        model_id=model_id,
        seq_id=seq_id,
        op=CommandOp(op),
        payload=payload,
        request_id=f"req-{seq_id}",
        arrival=arrival,
    )


def model_payload(service: TinkerService, tenant: str = "tenant", session_id: str | None = None, **overrides) -> dict:
    # [FUNCTION DOC] HELPER: Builds a create-model payload with the service's base model, the
    # given session id or (if none) a new session created for the tenant on the service (side
    # effect), model sequence id 1, then the keyword overrides.
    return {
        "base_model": service.config.base_model,
        "session_id": session_id or service.create_session(tenant),
        "model_seq_id": 1,
        **overrides,
    }


async def created_model(service: TinkerService, tenant: str = "tenant", session_id: str | None = None) -> str:
    # [FUNCTION DOC] HELPER: Async; creates a model with LoRA rank 8 for the tenant (new session
    # if none given), waits for the creation future to settle, asserts it finished successfully
    # (the failure error is the assertion message) and returns the new model id.
    request_id, model_id = service.create_model(
        tenant, model_payload(service, tenant, session_id, lora_config={"rank": 8})
    )
    future = await await_settled(service, tenant, request_id)
    assert future.state == DONE, future.error
    return model_id


async def await_settled(service: TinkerService, tenant: str, request_id: str, timeout: float = 2.0) -> RequestFuture:
    # [FUNCTION DOC] HELPER: Async; polls the service for the tenant's request future every 5
    # milliseconds until its state is no longer pending or the timeout (default 2 seconds) passes.
    # Returns the settled future. Raises an assertion error if the future has expired (unknown)
    # or is still pending at the deadline.
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        future = service.retrieve_future(tenant, request_id)
        assert future is not None, f"future {request_id} expired"
        if future.state != PENDING:
            return future
        await asyncio.sleep(0.005)
    raise AssertionError(f"future {request_id} still pending after {timeout}s")
