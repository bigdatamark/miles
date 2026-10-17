# [MODULE DOC] HELPER:
# Checkpoint path and metadata helpers for the Tinker gateway: parsing tinker:// paths, resolving
# checkpoint directories safely under the checkpoint root, building and reading checkpoint
# metadata files with a tenant digest for ownership, and resolving sampler checkpoints. The
# tenant credential is never stored; only its SHA-256 hex digest is.
import hashlib
import json
import os
from pathlib import Path

from miles.tinker.core.input_validation import validate_checkpoint_metadata, validate_checkpoint_segment
from miles.tinker.core.types import GatewayConfig, ModelRecord, OwnershipError, UserInputError


def parse_tinker_path(path: str) -> tuple[str, str, str]:
    # [FUNCTION DOC] Split a path of the form tinker://model-id/kind/name, where the kind is
    # weights or sampler_weights, into (model id, kind, name). Raises a user input error if the
    # scheme is wrong, the path does not have exactly three parts, the kind is unknown, or any part
    # fails the checkpoint segment validation.
    if not path.startswith("tinker://"):
        raise UserInputError(f"not a tinker path: {path!r}")
    parts = path.removeprefix("tinker://").split("/")
    if len(parts) != 3 or parts[1] not in ("weights", "sampler_weights"):
        raise UserInputError(f"malformed tinker path: {path!r}")
    for segment in parts:
        validate_checkpoint_segment(segment)
    return parts[0], parts[1], parts[2]


def resolve_checkpoint_dir(checkpoint_root: str, model_id: str, kind: str, name: str) -> str:
    # [FUNCTION DOC] Build the directory root/model/kind/name using the real path of the root, and
    # assert (assertion error) that the fully resolved path (following symlinks) stays inside the
    # root. Returns the unresolved joined path, not the resolved one. Reads the file system.
    root = os.path.realpath(checkpoint_root)
    path = f"{root}/{model_id}/{kind}/{name}"
    resolved = os.path.realpath(path)
    assert resolved.startswith(root + os.sep), f"checkpoint path {resolved!r} escapes {root!r}"
    return path


def build_checkpoint_metadata(record: ModelRecord, config: GatewayConfig) -> dict:
    # [FUNCTION DOC] Mapping saved next to a checkpoint: digest of the tenant, base model, LoRA
    # rank and alpha from the record, and the three train flags from the gateway config.
    return {
        # the digest proves ownership without persisting the bearer credential itself
        "tenant_digest": _tenant_digest(record.tenant),
        "base_model": record.base_model,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }


def read_checkpoint_metadata(checkpoint_dir: str, tenant: str, shown_path: str) -> dict:
    # [FUNCTION DOC] Read and validate the metadata file in the checkpoint directory (file I/O).
    # A missing file raises a user input error (unknown checkpoint); unreadable, wrongly encoded
    # or non-JSON content raises a user input error; shape problems raise through the metadata
    # validator; a digest that differs from the given tenant's digest raises an ownership error.
    # Returns the metadata mapping. The shown path is only used in error messages.
    meta_file = Path(checkpoint_dir) / "META.json"
    if not meta_file.exists():
        raise UserInputError(f"unknown checkpoint {shown_path!r}")
    try:
        meta = json.loads(meta_file.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UserInputError(f"cannot read checkpoint {shown_path!r}: {error}") from error
    validate_checkpoint_metadata(meta, shown_path)
    if meta["tenant_digest"] != _tenant_digest(tenant):
        raise OwnershipError(f"checkpoint {shown_path!r} does not belong to this tenant")
    return meta


def resolve_sampler_checkpoint(checkpoint_root: str, tenant: str, model_path: str, base_model: str) -> tuple[str, str]:
    # [FUNCTION DOC] Resolve a tinker model path to (adapter name, checkpoint directory). The path
    # must be of kind sampler weights (training checkpoints raise a user input error). Reads and
    # validates its metadata for the tenant (ownership error if different), requires the stored
    # base model to equal the served one (user input error), and names the adapter model id, an
    # at sign and the checkpoint name.
    """Return the adapter name and directory so engines can reload evicted snapshots."""
    model_id, kind, name = parse_tinker_path(model_path)
    if kind != "sampler_weights":
        raise UserInputError(
            f"sampling from training checkpoints is not supported: {model_path!r}; use save_weights_for_sampler()"
        )
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_root, model_id, "sampler_weights", name)
    meta = read_checkpoint_metadata(checkpoint_dir, tenant, model_path)
    if meta["base_model"] != base_model:
        raise UserInputError(
            f"checkpoint {model_path!r} uses base_model={meta['base_model']!r}; this server serves {base_model!r}"
        )
    return f"{model_id}@{name}", checkpoint_dir


def _tenant_digest(tenant: str) -> str:
    # [FUNCTION DOC] HELPER: Hex SHA-256 digest of the tenant string's UTF-8 bytes.
    return hashlib.sha256(tenant.encode()).hexdigest()
