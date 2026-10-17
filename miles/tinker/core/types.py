# [MODULE DOC]
# Shared data types of the Tinker gateway core: command operation enum, mappings between wire loss
# input names and internal datum keys, the required loss inputs per loss function, two exception
# types (client input errors and tenant ownership violations), the gateway configuration, and
# plain records for queued commands, models, sessions and sampling sessions.
# Module constants: LOSS_INPUT_KEYS maps wire names weights, advantages, logprobs to internal datum
# keys weights, advantages, sampling_logprobs; LOSS_FN_INPUTS lists, per loss function
# (cross_entropy needs weights; importance_sampling, ppo, cispo, dro need logprobs and
# advantages), the wire inputs that must be present.
"""Shared commands, datums, and configuration for the Tinker gateway.

An internal datum has tokens = model_input + target_tokens[-1:] and explicit target_tokens.
Its target_len counts every label position, including prompt positions; loss inputs align to it.
"""

from dataclasses import dataclass, field
from enum import Enum


class CommandOp(str, Enum):
    # [CLASS DOC] String enumeration of trainer commands: forward-backward, forward-only,
    # optimizer step, save state, load state, and save weights for sampler. The first two are
    # batch ops; the rest are barriers.
    FORWARD_BACKWARD = "forward_backward"
    FORWARD_ONLY = "forward_only"
    OPTIM_STEP = "optim_step"
    SAVE_STATE = "save_state"
    LOAD_STATE = "load_state"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"

    def is_batch(self) -> bool:
        # [FUNCTION DOC] True for forward-backward and forward-only.
        """Batch ops pack into BatchUnits; every other op is a barrier."""
        return self in (CommandOp.FORWARD_BACKWARD, CommandOp.FORWARD_ONLY)

    def requires_model_close_on_failure(self) -> bool:
        # [FUNCTION DOC] True for forward-backward, optimizer step and load state: a failure of
        # these may leave the trainer slot's state inconsistent, so the model must be closed.
        return self in (CommandOp.FORWARD_BACKWARD, CommandOp.OPTIM_STEP, CommandOp.LOAD_STATE)


# wire loss_fn_inputs key -> internal datum key
LOSS_INPUT_KEYS = {"weights": "weights", "advantages": "advantages", "logprobs": "sampling_logprobs"}

# reject missing loss inputs before they enter a shared batch
LOSS_FN_INPUTS = {
    "cross_entropy": ("weights",),
    "importance_sampling": ("logprobs", "advantages"),
    "ppo": ("logprobs", "advantages"),
    "cispo": ("logprobs", "advantages"),
    "dro": ("logprobs", "advantages"),
}


class UserInputError(Exception):
    # [CLASS DOC] Raised for client mistakes; the request's future fails with category user.
    """Rejected request content; fails the future with category user."""


class OwnershipError(Exception):
    # [CLASS DOC] Raised when a model, checkpoint, sampling session or request belongs to another
    # tenant than the caller.
    """model/checkpoint does not belong to the caller's tenant."""


@dataclass
class GatewayConfig:
    # [CLASS DOC]
    # Gateway-wide settings. Required: served base model name, number of adapter slots,
    # checkpoint root directory, vocabulary size. Limits with defaults: datums per request 1024,
    # tokens per datum 32768, tokens per request 4,000,000, samples per request 64, maximum LoRA
    # rank 32 (the slot capacity the server was built with). LoRA alpha default none meaning twice
    # the rank. Layout flags: train attention (true), MLP (true), unembed (false); model creation
    # rejects deviations. Lease timeout 300 seconds (stale sessions lose sampling, models and
    # slots). Batch token budget 262,144 tokens per packed batch unit.
    base_model: str
    n_slots: int
    checkpoint_root: str
    vocab_size: int
    max_datums_per_request: int = 1024
    max_tokens_per_datum: int = 32768
    max_tokens_per_request: int = 4_000_000
    max_samples_per_request: int = 64
    max_lora_rank: int = 32  # the slot capacity the server was built with (--lora-rank)
    lora_alpha: float | None = None  # None: 2 * rank
    # what the server-wide adapter layout trains; create_model rejects deviations
    trains_attn: bool = True
    trains_mlp: bool = True
    trains_unembed: bool = False
    lease_timeout_s: float = 300.0  # sessions stale beyond this lose their sampling, models, and slots
    batch_token_budget: int = 262_144  # packing bound per BatchUnit


@dataclass
class Command:
    # [CLASS DOC] A decoded client command bound to a model: model id, per-model sequence id
    # (1-based, defines execution order), operation, payload mapping, id of its future, global
    # arrival counter (orders commands across models), and an optional validation error text that
    # makes the command fail without running.
    model_id: str
    seq_id: int
    op: CommandOp
    payload: dict
    request_id: str
    arrival: int  # global submit order for selecting the scheduler's seed
    validation_error: str | None = None


@dataclass
class ModelRecord:
    # [CLASS DOC] A created training model: ids of the model and owning session, tenant, assigned
    # adapter slot, base model, LoRA rank and alpha, id of the create request's future, mapping
    # from sequence id to request future id (for retry deduplication), whether the slot has been
    # initialized in the trainer, and the next version number for sampler weight saves (failed
    # saves consume numbers, leaving gaps).
    model_id: str
    session_id: str
    tenant: str
    slot: int
    base_model: str
    lora_rank: int
    lora_alpha: float
    create_request_id: str
    request_id_by_seq: dict[int, str] = field(default_factory=dict)
    slot_initialized: bool = False
    # failed publications burn their version number, leaving gaps
    next_sampler_version: int = 1


@dataclass
class SessionRecord:
    # [CLASS DOC] A client session: tenant, last heartbeat (monotonic clock), mapping from model
    # sequence id to (create request id, model id), and mapping from sampling-session sequence id
    # to sampling session id (both for idempotent retries).
    tenant: str
    last_heartbeat: float
    models_by_seq: dict[int, tuple[str, str]] = field(default_factory=dict)
    sampling_sessions_by_seq: dict[int, str] = field(default_factory=dict)


@dataclass
class SamplingSessionRecord:
    # [CLASS DOC] A sampling session: tenant, optional tinker model path of the sampler weights
    # (none means the base model), owning session id, and mapping from sample sequence id to
    # (request future id, list of sequence ids) for retry deduplication.
    tenant: str
    model_path: str | None
    session_id: str
    samples_by_seq: dict[int, tuple[str, list[str]]] = field(default_factory=dict)
