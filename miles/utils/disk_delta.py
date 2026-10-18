# [MODULE DOC] Trainer-side helpers for publishing weight updates as deltas against a safetensors
# checkpoint on disk: the byte encoding of an "overwrite" delta, checksum helpers over several
# algorithms, and a reader that returns the raw bytes of a named tensor from a checkpoint
# directory after validating its declared dtype and shape. The receiving side lives in the
# inference engine, not here. A module constant gives the worker thread count for parallel
# delta phases (at most 32, or the CPU count, default 8 when unknown).
from __future__ import annotations

import glob
import json
import os
import struct
import zlib

import numpy as np

# The delta phases (diff, zstd, checksum) are memory-bandwidth bound and release the GIL,
# so a thread pool over tensors recovers the bandwidth one thread leaves idle.
NUM_WORKERS = min(32, (os.cpu_count() or 8))

# Trainer-side (publish) helpers for disk-level delta weight sync. The receive side —
# materializing the host-local checkpoint and applying published deltas in place — lives in
# the engine behind its /pull_weights endpoint (sglang.srt.weight_sync.local_checkpoint), so it
# runs on every host of a multi-node engine while miles only talks to one endpoint.


def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    # [FUNCTION DOC] Inputs: a flat array of the new values (expected to be a byte array) and a
    # boolean mask of the same length marking changed positions. Output: one byte array laid out
    # as (1) the count of changed positions as a little-endian 32-bit unsigned integer, (2) the
    # changed positions in ascending order, each a little-endian 32-bit unsigned integer, (3) the
    # new values at those positions. Positions must fit in 32 bits. No side effects. If the value
    # array is not a byte array the concatenation would promote types and corrupt the layout.
    """The 'overwrite' delta: changed-position count (u4), positions (u4 each), then new values.
    Idempotent to apply, unlike xor (an involution); the trainer picks the encoding per the docs."""
    pos = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate([np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), new[changed_mask]])


class _Adler32:
    # [CLASS DOC] HELPER: Adapter giving the Adler-32 checksum the same incremental update and
    # hex digest interface as the hash objects of the other algorithms. State: the running
    # checksum, initially 1. The digest is eight lowercase hex digits.
    """adler32 behind the incremental .update / .hexdigest interface the hash objects expose."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Starts the running checksum at the Adler-32 initial value, 1.
        self._value = 1

    def update(self, data) -> None:
        # [FUNCTION DOC] HELPER: Folds a bytes-like object into the running checksum.
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        # [FUNCTION DOC] HELPER: The checksum as eight lowercase hex digits, zero padded.
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    # [FUNCTION DOC] HELPER: Creates a fresh incremental hasher by name: "xxh3-128" (xxhash
    # library, imported lazily), "blake3" (blake3 library, imported lazily) or "adler32" (the
    # adapter above). Any other name raises a key error; a missing library raises an import error.
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")


def checksum(algorithm: str, buf) -> str:
    # [FUNCTION DOC] One-shot checksum of a bytes-like buffer with the named algorithm; returns
    # the hex digest string. Raises a key error for an unknown algorithm.
    hasher = _new_hasher(algorithm)
    hasher.update(buf)
    return hasher.hexdigest()


def _tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int, str, tuple[int, ...]]]:
    # [FUNCTION DOC] HELPER: For every *.safetensors file directly in the directory (glob order,
    # not sorted), reads the 8-byte little-endian header length and the json header, and records
    # for each tensor (skipping the metadata entry): (file path, absolute byte offset of the data
    # = 8 + header length + the tensor's begin offset, byte count, dtype string, shape tuple).
    # Returns a mapping from tensor name to that tuple. If the same name occurs in several files
    # the one processed last wins. An empty directory yields an empty mapping.
    """Index each tensor's byte range and declared safetensors layout."""
    locations: dict[str, tuple[str, int, int, str, tuple[int, ...]]] = {}
    for path in glob.glob(os.path.join(ckpt_dir, "*.safetensors")):
        with open(path, "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (
                path,
                8 + header_len + begin,
                end - begin,
                info["dtype"],
                tuple(info["shape"]),
            )
    return locations


def make_tensor_reader(ckpt_dir: str):
    # [FUNCTION DOC] Indexes the checkpoint directory once and returns a reader function. The
    # reader takes a tensor name and optional expected dtype string and expected shape tuple; it
    # raises a key error for an unknown name, a value error if an expectation is given and
    # differs from the stored layout, otherwise opens the file, seeks to the tensor and returns
    # its raw bytes as a byte array (not reinterpreted to the tensor dtype). The index is not
    # refreshed if files change later.
    """Index headers once and return a layout-aware raw tensor reader."""
    locations = _tensor_locations(ckpt_dir)

    def read(
        name: str,
        # [FUNCTION DOC] HELPER: See the enclosing function; reads one tensor's bytes.
        *,
        expected_dtype: str | None = None,
        expected_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        # [FUNCTION DOC] HELPER: See the enclosing function; reads one tensor's bytes.
        path, offset, nbytes, dtype, shape = locations[name]
        if (expected_dtype is not None and dtype != expected_dtype) or (
            expected_shape is not None and shape != expected_shape
        ):
            raise ValueError(
                f"Checkpoint tensor {name!r} has dtype={dtype}, shape={shape}; "
                f"trainer emitted dtype={expected_dtype}, shape={expected_shape}"
            )
        with open(path, "rb") as f:
            f.seek(offset)
            return np.frombuffer(f.read(nbytes), dtype=np.uint8)

    return read
