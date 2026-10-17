# [MODULE DOC] HELPER:
# Assertions and bookkeeping used only in CI runs of the Megatron backend: per-layer SHA256 hashes
# of model parameters (saved after training, checked after loading, one json file per parallel
# rank), a check that only multi-token-prediction (MTP) parameters receive gradients, a check on
# peak GPU memory after a low-memory resume, and an MTP loss upper bound. Every check is gated by
# CI flags on the args object where noted and signals failure by raising assertion errors.
"""CI utilities for Megatron backend testing."""

import hashlib
import json
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import torch
from megatron.core.distributed import DistributedDataParallel as DDP

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)

_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)encoder\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)layer\.(\d+)(?:\.|$)"),
)


def _layer_key(name: str) -> str:
    # [FUNCTION DOC] HELPER: Maps a parameter name to a grouping key. Tries, in order, patterns for
    # encoder layers, decoder layers, bare layers and singular layer followed by a number, each as
    # a dot-delimited path component; the first match yields "layer_" plus the number zero-padded
    # to four digits. Names matching none yield "non_layer".
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return f"layer_{int(match.group(1)):04d}"
    return "non_layer"


def _hash_tensor_bytes(tensor: torch.Tensor) -> bytes:
    # [FUNCTION DOC] HELPER: Returns the raw memory bytes of a tensor: detaches it, copies to CPU
    # if on GPU, makes it contiguous, flattens and reinterprets as unsigned bytes.
    data = tensor.detach()
    if data.is_cuda:
        data = data.cpu()
    if not data.is_contiguous():
        data = data.contiguous()
    return data.reshape(-1).view(torch.uint8).numpy().tobytes()


def compute_model_hashes_by_layer(model: Sequence[DDP]) -> dict[str, str]:
    # [FUNCTION DOC]
    # For each model chunk (index used as a pipeline-chunk prefix "pp" plus index), iterates its
    # parameters sorted by name, builds the full name "ppN.name", groups by layer key (see the
    # layer key helper), and feeds a per-group SHA256 with the full name, the shape as text, the
    # dtype as text and the raw parameter bytes. Returns a mapping from group key to hex digest,
    # sorted by key. Parameters that are none are skipped (cannot occur from named parameters).
    """Compute per-layer SHA256 hashes over parameter bytes.

    Hash input includes parameter name, shape, dtype, and raw bytes.
    """
    hashers: dict[str, hashlib._Hash] = {}
    for pp_idx, model_chunk in enumerate(model):
        for name, param in sorted(model_chunk.named_parameters(), key=lambda x: x[0]):
            if param is None:
                continue
            full_name = f"pp{pp_idx}.{name}"
            key = _layer_key(full_name)
            hasher = hashers.setdefault(key, hashlib.sha256())
            hasher.update(full_name.encode("utf-8"))
            hasher.update(str(tuple(param.shape)).encode("utf-8"))
            hasher.update(str(param.dtype).encode("utf-8"))
            hasher.update(_hash_tensor_bytes(param))
    return {k: v.hexdigest() for k, v in sorted(hashers.items(), key=lambda x: x[0])}


def _hash_file_path(base_dir: str | Path, iteration: int) -> Path:
    # [FUNCTION DOC] HELPER: Builds the per-rank hash file path. If the base directory's name
    # already starts with iter_ it is used as the iteration directory; otherwise the directory is
    # base plus iter_ and the iteration zero-padded to seven digits. The file name encodes tensor,
    # pipeline, data (intra-DP with context parallel) and context-parallel ranks from the global
    # parallel state: model_hash_tp{}_pp{}_dp{}_cp{}.json.
    tp_rank = get_parallel_state().tp.rank
    pp_rank = get_parallel_state().pp.rank
    dp_rank = get_parallel_state().intra_dp_cp.rank
    cp_rank = get_parallel_state().cp.rank
    base = Path(base_dir)
    iter_dir = base if base.name.startswith("iter_") else base / f"iter_{int(iteration):07d}"
    return iter_dir / f"model_hash_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}_cp{cp_rank}.json"


def save_model_hashes(args, model: Sequence[DDP], iteration: int, hashes: dict[str, str]) -> None:
    # [FUNCTION DOC]
    # Writes the given hash mapping as indented, key-sorted json into this rank's hash file under
    # the save directory for the iteration (creating parent directories) and logs the path. Does
    # nothing unless both the CI-test flag and the CI save-model-hash flag are set.
    # KNOWN ISSUE: the model argument is unused; the hashes must be computed by the caller.
    if not args.ci_test or not args.ci_save_model_hash:
        return
    path = _hash_file_path(args.save, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(hashes, f, indent=2, sort_keys=True)
    logger.info(f"[CI hash] Saved model hashes to {path}")


def check_model_hashes(args, model: Sequence[DDP], iteration: int) -> None:
    # [FUNCTION DOC]
    # Does nothing unless the CI-test flag and the CI check-model-hash flag are set. Otherwise
    # reads this rank's expected hash file from the load directory for the iteration (raises an
    # assertion error if missing), recomputes hashes of the model, and raises an assertion error
    # on any difference, reporting up to five missing, extra and mismatched layer keys each.
    # Logs success otherwise.
    if not args.ci_test or not args.ci_check_model_hash:
        return
    path = _hash_file_path(args.load, iteration)
    if not path.is_file():
        raise AssertionError(f"[CI hash] Hash file missing: {path}")
    with path.open("r", encoding="utf-8") as f:
        expected = json.load(f)
    actual = compute_model_hashes_by_layer(model)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])
        raise AssertionError(
            "[CI hash] Model hash mismatch after load. "
            f"missing={missing[:5]}, extra={extra[:5]}, mismatched={mismatched[:5]}"
        )
    logger.info(f"[CI hash] Model hashes match for iteration {iteration}.")


def check_mtp_only_grad(model: Sequence[DDP], step_id: int) -> None:
    # [FUNCTION DOC]
    # For every parameter in every chunk takes the main gradient buffer if present, else the
    # regular gradient (skips parameters with neither); computes the maximum absolute gradient
    # value; classifies the parameter as MTP if its name contains ".mtp." and records those with a
    # positive maximum. Logs the counts, logs up to five offending non-MTP names as errors, then
    # asserts that no non-MTP parameter has a nonzero gradient and that at least one MTP parameter
    # does. Forces a device-to-host sync per parameter.
    """Check that only MTP parameters have non-zero gradients.

    This is used for CI testing to verify that when all outputs are truncated,
    only the MTP layers receive gradients (since only mtp_loss contributes).

    Args:
        model: Sequence of DDP-wrapped model chunks.
        step_id: Current step index for logging.

    Raises:
        AssertionError: If any non-MTP parameter has a non-zero gradient.
    """
    non_mtp_nonzero_grads = []
    mtp_nonzero_grads = []

    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            # Get the main_grad from the distributed optimizer if available
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is None:
                continue

            grad_norm = grad.abs().max().item()
            is_mtp = ".mtp." in name

            if is_mtp:
                if grad_norm > 0:
                    mtp_nonzero_grads.append((name, grad_norm))
            else:
                if grad_norm > 0:
                    non_mtp_nonzero_grads.append((name, grad_norm))

    # Log the results
    logger.info(
        f"[CI MTP Grad Check] Step {step_id}: "
        f"MTP params with non-zero grad: {len(mtp_nonzero_grads)}, "
        f"non-MTP params with non-zero grad: {len(non_mtp_nonzero_grads)}"
    )

    if non_mtp_nonzero_grads:
        # Log the first few non-MTP params with non-zero gradients for debugging
        for name, grad_norm in non_mtp_nonzero_grads[:5]:
            logger.error(f"[CI MTP Grad Check] Non-MTP param with non-zero grad: {name}, max_grad={grad_norm}")

    assert len(non_mtp_nonzero_grads) == 0, (
        f"Expected all non-MTP parameters to have zero gradients, "
        f"but found {len(non_mtp_nonzero_grads)} with non-zero gradients. "
        f"First few: {non_mtp_nonzero_grads[:5]}"
    )

    # Also verify that MTP params do have gradients (otherwise the test is not valid)
    assert len(mtp_nonzero_grads) > 0, (
        "Expected MTP parameters to have non-zero gradients, but all were zero. "
        "This may indicate the MTP loss is not being computed."
    )


def check_peak_gpu_memory_after_load(args) -> None:
    # [FUNCTION DOC]
    # Runs only when the CI-test flag and the low-memory-resume flag are set, and only when the HF
    # checkpoint path contains the substring for the Qwen3 4B model (otherwise returns). Reads the
    # peak allocated CUDA memory in GiB, logs it with the rank (zero if distributed is not
    # initialized), and asserts it is below a fixed 20 GiB threshold (chosen between about 16.9 GiB
    # with the optimization and 22.4 GiB without, on 8 H200 GPUs).
    """Assert that peak GPU memory stays below threshold when --low-memory-resume is active."""
    if not args.ci_test or not getattr(args, "low_memory_resume", False):
        return

    hf_ckpt = getattr(args, "hf_checkpoint", "") or ""
    if "Qwen3-4B" not in hf_ckpt:
        return

    # Threshold 20 GB is midpoint between ~16.9 GB (with) and ~22.4 GB (without) on 8xH200.
    peak_gpu_gb = torch.cuda.max_memory_allocated() / (1024**3)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    logger.info(f"[CI low-memory-resume] Rank {rank} peak GPU memory: {peak_gpu_gb:.2f} GB")

    threshold_gb = 20.0
    assert peak_gpu_gb < threshold_gb, (
        f"[Rank {rank}] Peak GPU memory ({peak_gpu_gb:.2f} GB) exceeds threshold ({threshold_gb} GB). "
        f"--low-memory-resume optimization may not be working correctly."
    )


def check_mtp_loss(mtp_loss: float, max_mtp_loss: float = 1.0) -> None:
    # [FUNCTION DOC]
    # Asserts the loss is strictly below the maximum (default 1.0); the error message includes both.
    """Check that MTP loss is within expected bounds.

    Args:
        mtp_loss: The computed MTP loss value.
        max_mtp_loss: Maximum allowed MTP loss (default: 1.0).

    Raises:
        AssertionError: If MTP loss exceeds the maximum allowed value.
    """
    assert mtp_loss < max_mtp_loss, (
        f"MTP loss {mtp_loss} exceeds maximum allowed value {max_mtp_loss}. "
        "This may indicate an issue with MTP training."
    )
