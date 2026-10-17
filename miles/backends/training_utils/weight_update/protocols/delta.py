# [MODULE DOC]
# Disk-delta weight transfer over a shared filesystem. The trainer keeps a CPU snapshot (raw bytes
# per HF tensor) of what the engines currently hold. Each sync diffs every gathered HF tensor
# against the snapshot (XOR or overwrite encoding), compresses the diff with zstd in a thread pool
# pipelined with the weight gather, and publishes only the changed tensors as safetensors files plus
# an HF-style index under a per-version directory, with per-tensor checksums in metadata. The engines
# are then told (over HTTP, via rank 0) to pull the delta onto each host's local checkpoint and
# reload it from disk. The very first sync only captures the baseline from the HF checkpoint and
# publishes nothing. Module-level helpers: a mapping from torch dtype to safetensors dtype code, an
# atomic file writer, and a helper to set the engine weight version when unset.
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
from argparse import Namespace
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import safetensors.numpy
import torch
import torch.distributed as dist
import zstandard

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.session import check_weight_sync_results
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.disk_delta import NUM_WORKERS, checksum, make_tensor_reader, overwrite_encode
from miles.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)

# Safetensors stores its own dtype codes in the file header but does not expose a
# public torch-dtype encoder. Disk-delta needs the exact code because it patches
# raw bytes without rewriting that header.
_SAFETENSORS_DTYPE_BY_TORCH_DTYPE = {
    torch.float64: "F64",
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int64: "I64",
    torch.int32: "I32",
    torch.int16: "I16",
    torch.int8: "I8",
    torch.uint8: "U8",
    torch.bool: "BOOL",
    torch.complex64: "C64",
    **{
        getattr(torch, torch_dtype_name): safetensors_dtype
        for torch_dtype_name, safetensors_dtype in (
            ("float8_e4m3fn", "F8_E4M3"),
            ("float8_e4m3fnuz", "F8_E4M3FNUZ"),
            ("float8_e5m2", "F8_E5M2"),
            ("float8_e5m2fnuz", "F8_E5M2FNUZ"),
            ("uint64", "U64"),
            ("uint32", "U32"),
            ("uint16", "U16"),
        )
        if hasattr(torch, torch_dtype_name)
    },
}


def _safetensors_dtype(dtype: torch.dtype) -> str:
    # [FUNCTION DOC] HELPER: Look up the safetensors dtype code string for a torch dtype in the
    # table above; raises a value error naming the dtype if unsupported.
    try:
        return _SAFETENSORS_DTYPE_BY_TORCH_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"Disk-delta does not support trainer tensor dtype {dtype}") from None


class UpdateWeightFromDiskDelta(WeightTransferProtocol):
    # [CLASS DOC] Disk delta protocol; does not use the pause/begin session frame (class flag false).
    # Senders are ranks with data-replica rank 0 (each holds distinct gathered shards). Per-rank
    # state: worker pool, snapshot (mapping from HF tensor name to raw byte array of the engines'
    # current content), per-version delta (name to compressed diff bytes) and checksum mappings,
    # byte counters (changed, total, wire), the pinned-buffer free queue, and an optional post-write
    # hook. Execution order per sync: begin_sync (first call: baseline only), send_bucket per
    # bucket, after_base_weights, finalize.
    """
    Delta weight sync over a shared filesystem. Source ranks diff each gathered HF tensor against
    a CPU snapshot of the previous sync and publish the changes as a canonical HF checkpoint dir;
    each engine's /pull_weights fans the apply out to every host it spans, then the engine reloads
    the patched local checkpoint via the ordinary update_weights_from_disk path. miles only ever
    talks to one endpoint per engine, so multi-node serving needs nothing extra.
    """

    # The transport is asynchronous by design: the engine-side apply is serialized by a
    # per-host flock behind /pull_weights and the reload pauses each engine itself, so
    # the sync never runs inside the pause/begin session frame.
    use_weight_update_session = False

    def __init__(self, args: Namespace) -> None:
        # [FUNCTION DOC] Reads the shared directory, delta encoding (xor or overwrite) and checksum
        # algorithm from args; creates the directory (and parents) if missing; starts with an empty
        # snapshot and baseline not captured. If a custom post-write function path is configured,
        # loads that function from the registry as the post-write hook, called with args, a directory
        # and the engine clients (for object-store-backed filesystems needing an explicit commit).
        super().__init__(args)
        self._pool: ThreadPoolExecutor | None = None
        self.delta_dir = args.update_weight_disk_dir
        os.makedirs(self.delta_dir, exist_ok=True)
        self.delta_encoding = args.update_weight_delta_encoding
        self.checksum_algorithm = args.update_weight_delta_checksum
        self._snapshot: dict[str, np.ndarray] = {}
        self._baseline_captured = False
        # Post-write hook: object-store-backed shared filesystems lack cross-host
        # read-after-write consistency, so written files need an explicit step
        # (e.g. uploading them to the backing object store) before the engines can see them.
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            from miles.utils.function_registry import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        # [FUNCTION DOC] Stores the engines, sets the group name "miles-disk-delta" (no collective
        # group is created) and marks this rank a sender iff its data-replica rank (from the shared
        # helper) is 0. GPU counts/offsets and the selector are ignored.
        # No NCCL groups: the transport is the shared filesystem. The engine lock the NCCL path
        # uses isn't needed either — the engine-side apply is serialized by a per-host flock
        # behind /pull_weights.
        self.rollout_engines = rollout_engines
        self.group_name = "miles-disk-delta"
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0

    def begin_sync(self, weight_version: int, iter_buckets) -> bool:
        # [FUNCTION DOC] First call: captures the baseline snapshot using the bucket iterator
        # factory (collective), remembers that it was captured and returns false so the caller skips
        # this round (nothing is published). Later calls: prepare the encode pipeline for the version
        # and return true.
        # The first call only captures the baseline snapshot the next sync diffs against.
        if not self._baseline_captured:
            self._capture_baseline(iter_buckets)
            self._baseline_captured = True
            return False
        self._begin_encode(weight_version)
        return True

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        # [FUNCTION DOC] For each tensor of the bucket: view its bytes as a flat unsigned-byte
        # tensor; if pinned buffers are in use and it fits the largest snapshot tensor size, take a
        # pinned buffer from the free queue (blocking, which backpressures the gather), copy
        # asynchronously and synchronize the current stream; otherwise copy to a numpy array on the
        # host. Adds the byte count to the total, submits the diff-and-compress task to the pool and
        # records its future. If the number of in-flight tasks reaches twice the worker count, waits
        # for and collects the oldest. Does not clear the bucket. KNOWN ISSUE: a tensor larger than
        # its snapshot entry's size in pinned mode falls to the host path, but a tensor name missing
        # from the snapshot only fails later inside the worker.
        """Submit each tensor of the bucket to the diff/compress pool (pipelined with the gather)."""
        for name, tensor in bucket:
            flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
            nbytes = int(flat.numel())
            if self._use_pinned and nbytes <= self._max_bytes:
                buf = self._free_q.get()  # blocks when all buffers are in flight -> backpressures the gather
                buf[:nbytes].copy_(flat, non_blocking=True)
                torch.cuda.current_stream().synchronize()
                payload, pinned = buf, True
            else:
                payload, pinned = flat.cpu().numpy(), False
            self.total_bytes += nbytes
            self._inflight.append(self._pool.submit(self._diff_and_compress, name, payload, nbytes, pinned))
            if len(self._inflight) >= 2 * NUM_WORKERS:
                self._collect(self._inflight.popleft())

    def after_base_weights(self) -> None:
        # [FUNCTION DOC] Collect all remaining in-flight tasks in submission order (re-raising any
        # worker error), then shut down the pool and clear it.
        """Drain the in-flight diff/compress work and shut the pool down."""
        while self._inflight:
            self._collect(self._inflight.popleft())
        self._pool.shutdown()
        self._pool = None

    def finalize(self, weight_version: int) -> None:
        # [FUNCTION DOC] Three steps in order: write the delta files and index (collective), commit
        # and have engines pull and reload (collective), then record metrics (collective).
        """Write this version as a canonical HF dir, have the engines pull and reload it."""
        self._write_delta_files(weight_version)
        self._reload_engines(weight_version)
        self._record_metrics(weight_version)

    def _capture_baseline(self, iter_buckets) -> None:
        # [FUNCTION DOC] HELPER: Collective across all ranks. Rank 0: wipes and recreates the shared
        # directory, runs the post-write hook on it, and starts asynchronous "pull weights" calls
        # (target version 0, local checkpoint dir, source dir from args) on every engine so hosts
        # materialize their base while the snapshot is gathered. All ranks barrier on the Gloo group.
        # Senders open a tensor reader over the HF checkpoint (a value error is deferred). Then every
        # rank iterates the bucket iterator with materialization only on senders (to join collectives);
        # senders, for each emitted tensor, read the checkpoint tensor with the expected safetensors
        # dtype and shape, raise (deferred) a value error if the name is absent or byte sizes differ,
        # and store the checkpoint bytes as the snapshot entry. Errors are deferred until iteration
        # ends; then ranks all-gather error strings over Gloo and, if any rank failed, every rank
        # raises a runtime error naming the first failing rank (chained to the local error if any).
        # Rank 0 then waits for the pulls and checks their responses; if the weight-update-equal
        # checker option is on, reloads the pulled base checkpoint on all engines with weight version
        # "0" and checks; otherwise sets the engine weight version to "0" only where unset. Logs the
        # snapshot size. Final barrier. KNOWN ISSUE: the deferred-error loop continues iterating
        # remaining buckets after an error but skips reading, which is intended; the post-hook runs
        # before the pull and its failure is not guarded.
        """Capture the baseline snapshot the first delta diffs against (no publish), and clear any
        stale stream from a prior run. Seeds from hf_checkpoint — what each host materializes its
        base from — so the invariant ``snapshot == engine base`` holds even where the megatron->HF
        round-trip trims vocab-padding rows (embed/lm_head). Every emitted tensor must have the same
        layout as that canonical checkpoint because deltas operate on raw bytes. pull_weights(0)
        makes each host materialize its local base now, overlapped with the snapshot gather, so the
        first real sync only pays the delta apply."""
        # a prior run's versions would apply against the wrong base; start the dir clean
        pulls = []
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, self.delta_dir, list(self.rollout_engines))
            pulls = [
                async_utils.submit(
                    client.pull_weights(
                        target_version=0,
                        local_checkpoint_dir=self.args.update_weight_local_checkpoint_dir,
                        source_dir=self.args.update_weight_disk_dir,
                    )
                )
                for client in self.rollout_engines
            ]
        dist.barrier(group=get_gloo_group())

        read_hf = None
        local_error: ValueError | None = None
        if self.is_sender:
            try:
                read_hf = make_tensor_reader(self.args.hf_checkpoint)  # index the HF headers once
            except ValueError as error:
                local_error = error

        for bucket in iter_buckets(materialize=self.is_sender):
            if not self.is_sender or local_error is not None:
                continue
            assert read_hf is not None
            try:
                for name, tensor in bucket:
                    try:
                        baseline = read_hf(
                            name,
                            expected_dtype=_safetensors_dtype(tensor.dtype),
                            expected_shape=tuple(tensor.shape),
                        )
                    except KeyError as error:
                        raise ValueError(
                            f"Trainer emitted {name!r}, but it is absent from the canonical checkpoint"
                        ) from error
                    emitted_nbytes = tensor.numel() * tensor.element_size()
                    if emitted_nbytes != baseline.nbytes:
                        raise ValueError(
                            f"Checkpoint tensor {name!r} has {baseline.nbytes} bytes; "
                            f"trainer emitted {emitted_nbytes} bytes"
                        )
                    self._snapshot[name] = baseline
            except ValueError as error:
                # Source ranks read the checkpoint, but every rank drives the bucket iterator's
                # collectives. Defer the error until iteration finishes, then make every rank fail.
                local_error = error

        group = get_gloo_group()
        error_messages: list[str | None] = [None] * dist.get_world_size(group=group)
        local_error_message = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
        dist.all_gather_object(error_messages, local_error_message, group=group)
        if any(error_messages):
            failed_rank, error_message = next(
                (rank, message) for rank, message in enumerate(error_messages) if message is not None
            )
            error = RuntimeError(f"Disk-delta baseline validation failed on rank {failed_rank}: {error_message}")
            if local_error is not None:
                raise error from local_error
            raise error

        if dist.get_rank() == 0:
            check_weight_sync_results(async_utils.wait_futures(pulls), is_lora=False)
            if self.args.check_weight_update_equal:
                # The weights checker resets engine tensors at startup and compares after the
                # first sync, expecting it to rewrite every tensor. The baseline publishes
                # nothing, so reload the just-pulled base checkpoint to restore engine state
                # (and set the engine weight version the CI equality check expects).
                results = async_utils.wait_futures(
                    [
                        async_utils.submit(
                            client.update_weights_from_disk(
                                model_path=self.args.update_weight_local_checkpoint_dir,
                                weight_version="0",
                            )
                        )
                        for client in self.rollout_engines
                    ]
                )
                check_weight_sync_results(results, is_lora=False)
            else:
                # TODO: temporarily weaken checkers; should enhance and fix related logics
                _update_weight_version_if_unset(self.rollout_engines, "0")
            logger.info(
                "[disk delta] captured baseline snapshot of %d tensors from %s",
                len(self._snapshot),
                self.args.hf_checkpoint,
            )
        dist.barrier(group=get_gloo_group())

    def _begin_encode(self, weight_version: int) -> None:
        # [FUNCTION DOC] HELPER: Runs on every rank. Shuts down any leftover pool without waiting
        # (cancelling pending work). Version directory = shared dir plus "weight_v" and the
        # zero-padded 6-digit version; senders create it. Resets the delta and checksum mappings and
        # the changed/total byte counters. Pinned buffers: size = largest snapshot tensor in bytes;
        # count = at least 4, at most twice the worker count and bounded by 32 GiB divided by the
        # size; each is a pinned unsigned-byte host tensor placed into the free queue. If pinned
        # allocation fails with a runtime error (e.g. low memlock limit), logs a warning and falls
        # back to pageable host copies. Finally creates the worker pool and an empty in-flight queue.
        # KNOWN ISSUE: buffers are allocated on non-sender ranks too, where they are never used.
        """Set up this version's diff/compress pipeline: each ``send_bucket`` copies one tensor at
        a time to a pinned buffer and submits it; pool workers diff against the snapshot and
        compress in parallel (each is a few big GIL-releasing numpy/zstd calls)."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{weight_version:06d}")
        if self.is_sender:
            os.makedirs(self._version_dir, exist_ok=True)
        self._delta: dict[str, np.ndarray] = {}  # changed tensor name -> compressed diff
        self._checksums: dict[str, str] = {}  # changed tensor name -> new-state checksum
        self.changed_bytes = self.total_bytes = 0

        # Pinned host-buffer pool: a pinned non_blocking GPU->CPU copy is far faster than .cpu().
        self._max_bytes = max((int(v.nbytes) for v in self._snapshot.values()), default=0)
        self._free_q: queue.Queue = queue.Queue()
        self._use_pinned = True
        try:
            for _ in range(max(4, min(2 * NUM_WORKERS, (32 << 30) // max(self._max_bytes, 1)))):
                self._free_q.put(torch.empty(self._max_bytes, dtype=torch.uint8, pin_memory=True))
        except RuntimeError as e:  # low memlock limit
            logger.warning("pinned host buffers unavailable (%s); using pageable .cpu()", e)
            self._use_pinned = False

        self._pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        self._inflight: deque = deque()

    def _diff_and_compress(self, name, buf, nbytes, pinned):
        # [FUNCTION DOC] HELPER: Pool worker. Input: tensor name, payload (a pinned buffer or a host
        # byte array), byte count, and whether the payload is a pinned buffer. If pinned, copies the
        # first nbytes into a fresh host array and returns the pinned buffer to the free queue.
        # Compares to the snapshot entry: xor encoding gives the bytewise exclusive-or and the count
        # of nonzero bytes; overwrite encoding builds a changed-bytes mask, counts it and encodes via
        # the overwrite encoder; any other encoding raises a value error. Returns (name, new bytes,
        # None, None, 0) if nothing changed; otherwise (name, new bytes, zstd level-1 compressed diff
        # as a byte array, checksum of the new bytes using the configured algorithm, changed byte
        # count). Raises a key error if the name has no snapshot entry. Does not update the snapshot.
        if pinned:  # copy out and free the pinned buffer before the heavy diff/compress
            new = np.empty(nbytes, dtype=np.uint8)
            np.copyto(new, buf.numpy()[:nbytes])
            self._free_q.put(buf)
        else:
            new = buf
        old = self._snapshot[name]
        if self.delta_encoding == "xor":
            diff = new ^ old
            changed = int(np.count_nonzero(diff))
        elif self.delta_encoding == "overwrite":
            mask = new != old
            changed = int(np.count_nonzero(mask))
            diff = overwrite_encode(new, mask)
        else:
            raise ValueError(f"unknown delta encoding {self.delta_encoding!r}")
        if not changed:
            return name, new, None, None, 0
        compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
        return name, new, compressed, checksum(self.checksum_algorithm, new), changed

    def _collect(self, fut):
        # [FUNCTION DOC] HELPER: Waits for a worker result (re-raising its error), replaces the
        # snapshot entry with the new bytes (the next sync's base), and, if anything changed, adds the
        # changed byte count and records the compressed diff and checksum under the name.
        name, new, compressed, digest, changed = fut.result()
        self._snapshot[name] = new  # becomes the next sync's base
        if changed:
            self.changed_bytes += changed
            self._delta[name] = compressed
            self._checksums[name] = digest

    def _drop_duplicate_names(self, group, world: int, rank: int) -> None:
        # [FUNCTION DOC] HELPER: Collective (all-gather of Python objects on the given group). Gathers
        # every rank's checksum mapping; for each lower-ranked rank, for each tensor name present
        # both there and in this rank's delta: raises a runtime error if the checksums differ, else
        # deletes the name from this rank's delta and checksum mappings (lowest rank keeps it).
        """A parameter Megatron replicates across PP stages — the word embedding on the last stage
        when it hosts an MTP block, or tied embeddings — is gathered and diffed by one source rank
        per stage, so the same HF tensor lands in several ranks' shards. The published artifact
        must hold each tensor exactly once (the XOR apply is an involution: applied twice it
        reverts), so keep the lowest-rank copy and drop the rest. The replicas are gradient-synced
        and byte-identical; a checksum divergence means the sync is broken — never publish it."""
        all_checksums: list = [None] * world
        dist.all_gather_object(all_checksums, self._checksums, group=group)
        for other_rank, other in enumerate(all_checksums[:rank]):
            for name in self._delta.keys() & other.keys():
                if other[name] != self._checksums[name]:
                    raise RuntimeError(
                        f"{name!r} published by rank {other_rank} and rank {rank} with different bytes; "
                        "PP-replicated parameters must stay identical across stages."
                    )
                del self._delta[name]
                del self._checksums[name]

    def _write_delta_files(self, weight_version: int) -> None:
        # [FUNCTION DOC] HELPER: Collective on the Gloo group. Drops duplicate names across ranks;
        # all-gathers a 0/1 "has data" flag per rank to number files sequentially without gaps (this
        # rank's index = number of earlier ranks with data; total = sum). A rank with data writes a
        # safetensors file named "model-<index 5 digits>-of-<total 5 digits>.safetensors" in the
        # version directory containing its compressed diffs, with checksums as file metadata, using an
        # atomic write, and records the byte size as wire bytes (else zero). All ranks all-gather a
        # mapping from their tensor names to file name; rank 0 writes the index JSON (atomically)
        # with metadata (zero-padded version, base version = previous version, delta encoding,
        # compression format "zstd", checksum algorithm) and the combined weight map. Ends with a
        # barrier.
        """Write this rank's changed tensors as one canonical model-NNNNN.safetensors, and on rank
        0 the HF index. The sequential file numbers and the index are coordinated over gloo (small
        object gathers), not the filesystem — a non-POSIX shared filesystem may not surface one
        rank's writes to another until commit."""
        group = get_gloo_group()
        world, rank = dist.get_world_size(), dist.get_rank()

        self._drop_duplicate_names(group, world, rank)

        # number the files sequentially across only the ranks that have one (no gaps)
        counts: list = [None] * world
        dist.all_gather_object(counts, int(bool(self._delta)), group=group)
        offset, total = sum(counts[:rank]), sum(counts)

        fname = None
        self.wire_bytes = 0
        if self._delta:
            fname = f"model-{offset:05d}-of-{total:05d}.safetensors"
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self.wire_bytes = len(blob)
            _atomic_write(os.path.join(self._version_dir, fname), blob)

        maps: list = [None] * world
        dist.all_gather_object(maps, {name: fname for name in self._delta}, group=group)
        if rank == 0:
            index = {
                "metadata": {
                    "version": f"{weight_version:06d}",
                    "base_version": f"{weight_version - 1:06d}",
                    "delta_encoding": self.delta_encoding,
                    "compression_format": "zstd",
                    "checksum_format": self.checksum_algorithm,
                },
                "weight_map": {name: f for m in maps for name, f in m.items()},
            }
            _atomic_write(os.path.join(self._version_dir, "model.safetensors.index.json"), json.dumps(index).encode())
        dist.barrier(group=group)

    def _reload_engines(self, weight_version: int) -> None:
        # [FUNCTION DOC] HELPER: Collective. All ranks run the post-write hook (if any) on the version
        # directory with the engine clients, then barrier. Rank 0 only: asks every engine to pull the
        # target version (disk-only; overlaps generation) and checks results; pauses generation on
        # all engines in the configured mode; flushes caches on all engines unless the mode is
        # in-place; reloads weights from the local checkpoint directory with the version string and
        # checks results; then resumes generation. Final barrier on Gloo.
        """Commit the published files, have each engine pull the delta onto every host it spans
        (checksum-verified), then reload the engines. The pull is disk-only, so it runs before
        pause and overlaps generation."""
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, self._version_dir, list(self.rollout_engines))
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            pulls = async_utils.wait_futures(
                [
                    async_utils.submit(
                        client.pull_weights(
                            target_version=weight_version,
                            local_checkpoint_dir=self.args.update_weight_local_checkpoint_dir,
                            source_dir=self.args.update_weight_disk_dir,
                        )
                    )
                    for client in self.rollout_engines
                ]
            )
            check_weight_sync_results(pulls, is_lora=False)
            mode = self.args.pause_generation_mode
            async_utils.wait_futures(
                [async_utils.submit(client.pause_generation(mode=mode)) for client in self.rollout_engines]
            )
            if mode != "in_place":
                async_utils.wait_futures([async_utils.submit(client.flush_cache()) for client in self.rollout_engines])
            results = async_utils.wait_futures(
                [
                    async_utils.submit(
                        client.update_weights_from_disk(
                            model_path=self.args.update_weight_local_checkpoint_dir,
                            weight_version=str(weight_version),
                        )
                    )
                    for client in self.rollout_engines
                ]
            )
            check_weight_sync_results(results, is_lora=False)
            async_utils.wait_futures(
                [async_utils.submit(client.continue_generation()) for client in self.rollout_engines]
            )
        dist.barrier(group=get_gloo_group())

    def _record_metrics(self, weight_version: int) -> None:
        # [FUNCTION DOC] HELPER: Collective. Sums changed, total and wire byte counts over all ranks
        # (default group, on the current CUDA device) and replaces the update metrics mapping with
        # "perf/update_weights_density" = changed / max(total, 1) and "perf/update_weights_wire_bytes";
        # rank 0 logs the density percentage and wire size in GB.
        """All-reduce the byte counts and record changed-fraction / wire size; the actor drains
        update_weight_metrics onto the step log."""
        counts = torch.tensor(
            [self.changed_bytes, self.total_bytes, self.wire_bytes],
            dtype=torch.int64,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        self.update_weight_metrics = {
            "perf/update_weights_density": changed / max(total, 1),
            "perf/update_weights_wire_bytes": wire,
        }
        if dist.get_rank() == 0:
            logger.info(
                "[disk delta v=%s] density=%.2f%% wire=%.2f GB",
                weight_version,
                100.0 * changed / max(total, 1),
                wire / 1e9,
            )


def _atomic_write(path: str, data: bytes) -> None:
    # [FUNCTION DOC] HELPER: Write bytes to "<path>.tmp", flush and fsync, then atomically rename
    # over the target path, so readers never see a partial file.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


_UNSET_WEIGHT_VERSION = "default"


def _update_weight_version_if_unset(rollout_engines: Sequence[SGLangApiClient], weight_version: str) -> None:
    # [FUNCTION DOC] HELPER: Query every engine's weight version; for engines reporting none or the
    # placeholder "default", set the version to the given string without aborting requests. Blocks
    # until all calls finish.
    reported = async_utils.wait_futures(
        [async_utils.submit(client.get_weight_version()) for client in rollout_engines]
    )
    unset = [
        client
        for client, version in zip(rollout_engines, reported, strict=True)
        if version in (None, _UNSET_WEIGHT_VERSION)
    ]
    async_utils.wait_futures(
        [
            async_utils.submit(client.update_weight_version(weight_version=weight_version, abort_all_requests=False))
            for client in unset
        ]
    )
