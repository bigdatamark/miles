# [MODULE DOC] Parallel Megatron-distributed-checkpoint to Hugging Face converter on Ray (the
# diagram in the text below shows the stages). Behavior details: the input must be a local
# path with a common-args file; weights are loaded from the checkpoint via private torch
# distributed-checkpoint APIs with a metered reader that counts items, files and bytes. Planning
# splits work into tasks of source keys: expert weights of MoE layers (3-D tensors named like
# decoder layers mlp experts linear_fc1/fc2) are read directly chunk by chunk per storage
# file and split into per-expert gate/up/down HF tensors without loading the full tensor;
# all other tensors are converted via the shared conversion core in whole-key tasks, where the
# MLA down-projection pair is kept together when a q-lora rank is set. Small tasks may be packed
# up to a byte budget. Workers are Ray actors pinned one-or-more per live node (round robin);
# each writes temporary-named shards to a staging directory, optionally quantizing on a GPU
# (assigned round robin) when the original HF config has a quantization section. Finalization
# assigns global shard names (task order), moves shards, copies assets, writes the index with
# total bytes, and deletes staging. Importing replaces the global pickle unpickler (side effect)
# and adds the repo root to the module search path.
r"""Ray-based Megatron torch_dist to HuggingFace conversion.

This is a high-throughput converter for large Kimi/DeepSeek-style checkpoints.
The core idea is to parallelize conversion work using distributed processes on Ray.
Conversion is an embarrassingly parallel problem, so it exhibits strong scaling with the number of nodes.

System flow:

+-------------------+
| prepare_runtime   |
|-------------------|
| init Ray          |
| validate output   |
| stage HF config   |
| create staging    |
+---------+---------+
          |
          v
+------------------------+
| read_metadata_and_plan |
|------------------------|
| read common.pt         |
| read DCP metadata      |
| build task plan        |
| publish metadata ref   |
+-----------+------------+
            |
            v
+-------------------------------------------------------------+
| dispatch_conversion_tasks                                   |
|-------------------------------------------------------------|
| create pinned Ray actors                                    |
| submit/refill tasks up to --concurrency                     |
|                                                             |
|  +------------------+   +------------------+   +---------+  |
|  | actor 0          |   | actor 1          |   | actor N |  |
|  |------------------|   |------------------|   |---------|  |
|  | DCP load         |   | DCP load         |   | ...     |  |
|  | convert to HF    |   | convert to HF    |   | ...     |  |
|  | optional quant   |   | optional quant   |   | ...     |  |
|  | write shards     |   | write shards     |   | ...     |  |
|  | return manifest  |   | return manifest  |   | ...     |  |
|  +------------------+   +------------------+   +---------+  |
+-----------------------------+-------------------------------+
                              |
                              v
+----------------------------+
| finalize_conversion_output |
|----------------------------|
| merge manifests            |
| assign final shard names   |
| publish shards/assets      |
| write final index          |
+----------------------------+

Examples:

The converter expects filesystem paths. Stage remote checkpoints and HF assets
onto local or shared storage before running it.

INPUT_DIR=/mnt/checkpoints/kimi-k26/iter_0000400
ORIGIN_HF_DIR=/mnt/hf/Kimi-K2.6-fp8-configs-only

# 8-layer smoke test
SOURCE_KEY_REGEX="^language_model\.decoder\.layers\.([0-7])\."
python tools/convert_torch_dist_to_hf_ray.py --input-dir $INPUT_DIR --output-dir /tmp/$USER/$RUN_ID/hf_ray_l0_7 --origin-hf-dir $ORIGIN_HF_DIR --model-name kimi_k25 --source-key-regex $SOURCE_KEY_REGEX --max-file-bytes 21474836480 --concurrency 16 --progress-interval-seconds 10 -f

# Full conversion on 1 node
python tools/convert_torch_dist_to_hf_ray.py --input-dir $INPUT_DIR --output-dir /mnt/outputs/kimi-ray/$RUN_ID/ray1_hf --origin-hf-dir $ORIGIN_HF_DIR --model-name kimi_k25 --max-file-bytes 21474836480 --concurrency 16 --progress-interval-seconds 10 -f

# Full conversion on 4 nodes
python tools/convert_torch_dist_to_hf_ray.py --input-dir $INPUT_DIR --output-dir /mnt/outputs/kimi-ray/$RUN_ID/ray4_hf --origin-hf-dir $ORIGIN_HF_DIR --model-name kimi_k25 --max-file-bytes 21474836480 --concurrency 64 --progress-interval-seconds 10 -f
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import re
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any, cast

import ray
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner, ReadItem
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future
from tqdm.auto import tqdm
from typing_extensions import override

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miles.backends.megatron_utils import megatron_to_hf as m2hf
from miles.utils.hf_utils.config import load_hf_config as _load_hf_config

DEFAULT_DIRECT_MOE_GROUP_SIZE = 2 * 1024**3


class UnpicklerWrapper(pickle.Unpickler):
    # [CLASS DOC] HELPER: Unpickler substituting a placeholder for classes from modules whose
    # name starts with megatron or glm so checkpoint metadata loads without those packages.
    @override
    def find_class(self, mod_name, name):
        # [FUNCTION DOC] HELPER: Placeholder class for megatron/glm modules, else normal lookup.
        class DummyClass:
            # [CLASS DOC] HELPER: Placeholder ignoring constructor arguments.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Accepts and ignores any arguments.
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


@dataclass(frozen=True)
class Args:
    # [CLASS DOC] Immutable parsed command-line options: input and output directories, optional
    # original HF directory and model name, force flag, maximum output file bytes, optional
    # worker concurrency, task-packing byte budget (zero disables packing), optional regular
    # expression limiting source keys, dry-run-plan flag, progress flag and progress interval.
    input_dir: str
    output_dir: str
    origin_hf_dir: str | None
    model_name: str | None
    force: bool
    max_file_bytes: int
    concurrency: int | None
    task_group_bytes: int
    source_key_regex: str | None
    dry_run_plan: bool
    progress: bool
    progress_interval_seconds: float


@dataclass(frozen=True)
class TaskSpec:
    # [CLASS DOC] One unit of work: task id (minus one until final numbering), the tuple of
    # source checkpoint keys, estimated source bytes, and (for direct expert tasks) the expert
    # block specs to read directly. Note MoeBlockSpec is defined after this class and used in an
    # annotation only (deferred annotations).
    task_id: int
    keys: tuple[str, ...]
    estimated_source_bytes: int
    moe_blocks: tuple[MoeBlockSpec, ...] = ()


@dataclass(frozen=True)
class MoeBlockSpec:
    # [CLASS DOC] A group of storage chunks of one expert weight tensor that live in a single
    # checkpoint file: source key, file path relative to the checkpoint, storage indices of the
    # chunks (sorted by offset), decoder layer index, which linear (first or second) and the
    # HF name prefix ("language_model." for nested language models, else empty).
    source_key: str
    relative_path: str
    storage_indices: tuple[MetadataIndex, ...]
    layer_idx: int
    linear_name: str
    hf_prefix: str


@dataclass(frozen=True)
class PreparedTensorGroup:
    # [CLASS DOC] HF tensors converted from one source parameter: the source name (used by
    # quantization) and a tuple of (HF name, tensor) pairs.
    source_name: str
    tensors: tuple[tuple[str, torch.Tensor], ...]


@dataclass(frozen=True)
class TaskLoadStats:
    # [CLASS DOC] Accounting of checkpoint reads for a task: number of read items, number of
    # distinct files and total stored bytes.
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class PreparedTaskTensors:
    # [CLASS DOC] Result of loading and converting a task: tensor groups and the load stats.
    groups: tuple[PreparedTensorGroup, ...]
    load_stats: TaskLoadStats


@dataclass(frozen=True)
class ShardManifest:
    # [CLASS DOC] Description of one staged shard file: temporary file name, final name (none
    # until global numbering), tuple of tensor names inside, byte size and tensor count.
    temp_filename: str
    final_filename: str | None
    weight_keys: tuple[str, ...]
    bytes: int
    tensors: int


@dataclass(frozen=True)
class TaskResult:
    # [CLASS DOC] What a worker returns for one task: task id, actor id, host name, process id,
    # shard manifests, estimated source bytes, output bytes, tensor count, source keys, the DCP
    # read accounting (items, files, bytes), the CUDA device used (none without quantization)
    # and the Ray node id.
    task_id: int
    actor_id: int
    node: str
    pid: int
    shards: tuple[ShardManifest, ...]
    source_bytes: int
    output_bytes: int
    weights: int
    source_keys: tuple[str, ...]
    dcp_read_items: int
    dcp_files: int
    dcp_storage_bytes: int
    cuda_device_id: int | None
    ray_node_id: str


@dataclass(frozen=True)
class DcpLoadResult:
    # [CLASS DOC] HELPER: Loaded state dict (key to tensor) plus read accounting counts.
    state_dict: dict[str, torch.Tensor]
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class DirectMoeLoadResult:
    # [CLASS DOC] HELPER: Converted per-expert tensor groups from a direct expert-block read,
    # plus read accounting counts.
    tensor_groups: tuple[PreparedTensorGroup, ...]
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class PlannedShard:
    # [CLASS DOC] HELPER: Final naming plan for a staged shard: temporary name, final name, tensor
    # names and byte size.
    temp_filename: str
    final_filename: str
    weight_keys: tuple[str, ...]
    bytes: int


class ProgressReporter:
    # [CLASS DOC] HELPER: Wraps a byte-based progress bar (task-count based if total bytes are
    # zero) shown on standard error, updated as tasks complete, with a postfix of completed
    # over total tasks. Disabled mode makes the bar silent.
    def __init__(
        self,
        tasks: list[TaskSpec],
        enabled: bool,
        interval_seconds: float,
        stream: Any = sys.stderr,
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the enabled flag, the refresh interval (at least 0.1
        # second), counts totals from the tasks' estimated bytes, creates the progress bar with
        # decimal units, and sets the initial postfix.
        self.enabled = enabled
        self.interval_seconds = max(interval_seconds, 0.1)
        self.total_tasks = len(tasks)
        self.total_bytes = sum(task.estimated_source_bytes for task in tasks)
        self.completed_tasks = 0
        self.last_refresh_time = 0.0
        self.progress = tqdm(
            total=self.total_bytes or self.total_tasks,
            desc="Converting",
            unit="B" if self.total_bytes else "task",
            unit_scale=bool(self.total_bytes),
            unit_divisor=1000,
            mininterval=self.interval_seconds,
            disable=not enabled,
            file=stream,
        )
        self._set_postfix()

    def complete(self, result: TaskResult) -> None:
        # [FUNCTION DOC] HELPER: Counts the task done and advances the bar by its source bytes
        # (or by one in task-count mode), refreshing the postfix.
        self.completed_tasks += 1
        self.progress.update(result.source_bytes if self.total_bytes else 1)
        self._set_postfix()

    def tick(self) -> None:
        # [FUNCTION DOC] HELPER: When enabled and at least one interval passed since the last
        # forced refresh (monotonic clock), redraws the bar and records the time.
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_refresh_time >= self.interval_seconds:
            self.progress.refresh()
            self.last_refresh_time = now

    def finish(self) -> None:
        # [FUNCTION DOC] HELPER: Updates the postfix and closes the bar.
        self._set_postfix()
        self.progress.close()

    def _set_postfix(self) -> None:
        # [FUNCTION DOC] HELPER: When enabled, sets the bar's postfix text to completed/total
        # tasks without forcing a redraw.
        if not self.enabled:
            return
        self.progress.set_postfix_str(f"tasks={self.completed_tasks}/{self.total_tasks}", refresh=False)


class WrappedStorageReader(dist_cp.FileSystemReader):
    # [CLASS DOC] HELPER: Filesystem checkpoint reader tolerant of missing Megatron/GLM classes
    # and of metadata lacking storage info or planner data.
    @override
    def read_metadata(self):
        # [FUNCTION DOC] HELPER: Loads the hidden metadata file with the tolerant unpickler;
        # creates storage metadata if missing (compatible with torch versions where the class
        # lives in different modules), sets its load id, defaults planner data to an empty
        # mapping and returns the metadata.
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = make_storage_meta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class ChunkedStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    # [CLASS DOC] HELPER: Load planner restricted to a given set of keys, creating uninitialized
    # CPU destination tensors for them from the checkpoint metadata.
    def __init__(self, keys_to_load: set[str]):
        # [FUNCTION DOC] HELPER: Initializes the parent and stores the set of keys to load.
        super().__init__()
        self.keys_to_load = keys_to_load

    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        # [FUNCTION DOC] HELPER: Raises a value error without metadata. For each metadata entry
        # whose key is in the load set, stores an uninitialized tensor of the recorded size and
        # dtype (tensor entries) or the entry itself in the state dict; then runs the parent setup.
        if metadata is None:
            raise ValueError("DCP metadata is required")
        for key, value in metadata.state_dict_metadata.items():
            if key not in self.keys_to_load:
                continue
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype)  # type: ignore[assignment]
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


class MeteredStorageReader(WrappedStorageReader):
    # [CLASS DOC] HELPER: Reader that performs its own sequential reads (grouped per file, one
    # stream per file) and records how many items, files and bytes the last read plan touched.
    def __init__(self, path: str):
        # [FUNCTION DOC] HELPER: Initializes the parent and zeroes the three counters.
        super().__init__(path)
        self.read_items = 0
        self.files = 0
        self.storage_bytes = 0

    @override
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # [FUNCTION DOC] HELPER: Replaces the parent read. Records accounting for the plan, groups
        # read items by file, and for each file opens one stream and for each item takes the
        # stored byte range, applies the load transform, then either hands raw bytes to the
        # planner (byte items) or deserializes a tensor (weights-only), narrows it to the
        # requested region, checks its size equals the destination (raises an assertion error
        # otherwise), copies into the destination and commits it. Returns an already completed
        # future. Synchronous and single-threaded.
        self.read_items, self.files, self.storage_bytes = compute_dcp_load_accounting(self.storage_data, plan)
        per_file: dict[str, list[Any]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            per_file.setdefault(item_md.relative_path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = cast(io.IOBase, _create_file_view(stream, item_md.offset, item_md.length))
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                        continue

                    seekable = transform_from if transform_from.seekable() else io.BytesIO(transform_from.read(-1))
                    seekable.seek(0)
                    tensor = cast(torch.Tensor, torch.load(seekable, map_location="cpu", weights_only=True))
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    target_tensor = planner.resolve_tensor(req).detach()
                    if target_tensor.size() != tensor.size():
                        raise AssertionError(
                            f"DCP tensor size mismatch for {req.storage_index}: "
                            f"{target_tensor.size()} vs {tensor.size()}"
                        )
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut


def make_storage_meta():
    # [FUNCTION DOC] HELPER: Creates an empty storage-metadata object from the checkpoint module
    # if it exports that class, else from its metadata submodule (torch version compatibility).
    storage_meta = getattr(dist_cp, "StorageMeta", None)
    if storage_meta is not None:
        return storage_meta()
    return dist_cp.metadata.StorageMeta()


def compute_dcp_load_accounting(storage_data: dict[Any, Any], plan: LoadPlan) -> tuple[int, int, int]:
    # [FUNCTION DOC] HELPER: Returns (number of plan items, number of distinct storage files,
    # total stored bytes) by looking up each item's storage record.
    files = set()
    storage_bytes = 0
    for read_item in plan.items:
        item_md = storage_data[read_item.storage_index]
        files.add(item_md.relative_path)
        storage_bytes += int(item_md.length)
    return len(plan.items), len(files), storage_bytes


def prepare_cached_metadata_for_reader(
    metadata: dist_cp.metadata.Metadata,
    storage_reader: WrappedStorageReader,
) -> dist_cp.metadata.Metadata:
    # [FUNCTION DOC] HELPER: Mutates cached metadata so a new reader can use it: creates storage
    # metadata if absent, sets its load id from the reader, defaults planner data to an empty
    # mapping. Returns the same object.
    if getattr(metadata, "storage_meta", None) is None:
        metadata.storage_meta = make_storage_meta()
    metadata.storage_meta.load_id = storage_reader.load_id
    if metadata.planner_data is None:
        metadata.planner_data = {}
    return metadata


def load_tensor_chunk(
    input_dir: str,
    keys_to_load: set[str],
    metadata: dist_cp.metadata.Metadata,
) -> DcpLoadResult:
    # [FUNCTION DOC] HELPER: Loads the selected keys without a process group by driving the
    # planner and reader steps manually as the coordinator: set up the planner on a new state
    # dict, create the local plan, let the reader adjust it, merge into a single-rank global
    # plan, finish, then read and wait. Returns the filled state dict with the reader's counters.
    state_dict: dict[str, torch.Tensor] = {}
    storage_reader = MeteredStorageReader(input_dir)
    metadata = prepare_cached_metadata_for_reader(metadata, storage_reader)
    planner = ChunkedStateDictLoadPlanner(keys_to_load)
    planner.set_up_planner(state_dict, metadata, is_coordinator=True)
    storage_reader.set_up_storage_reader(metadata, is_coordinator=True)
    local_plan = planner.create_local_plan()
    local_plan = storage_reader.prepare_local_plan(local_plan)
    global_plan = planner.create_global_plan([local_plan])
    global_plan = storage_reader.prepare_global_plan(global_plan)
    final_local_plan = planner.finish_plan(global_plan[0])
    storage_reader.read_data(final_local_plan, planner).wait()
    return DcpLoadResult(
        state_dict=state_dict,
        read_items=storage_reader.read_items,
        files=storage_reader.files,
        storage_bytes=storage_reader.storage_bytes,
    )


def get_expert_param(args: Any, name: str, param: torch.Tensor):
    # [FUNCTION DOC] HELPER: Generator of (name, tensor). Non-expert names pass through. A
    # stacked expert tensor (name lacking a trailing expert index) must have first dim equal to
    # the configured expert count (assertion error otherwise) and is yielded per expert with the
    # doubled "experts.experts." collapsed and the expert index appended; others pass through.
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        if param.shape[0] != num_experts:
            raise AssertionError(f"Expected {num_experts} experts for {name}, got {param.shape}")
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            yield expert_name, param[expert_id]
    else:
        yield name, param


def get_layer_param(args: Any, name: str, param: torch.Tensor):
    # [FUNCTION DOC] HELPER: Generator of (name, tensor). Names without ".layers." pass through.
    # A stacked-layers tensor (no layer index in the name) must have first dim equal to the
    # configured layer count (assertion error otherwise) and is split per layer with the index
    # inserted, each piece expanded by the expert generator; indexed names go straight to it.
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        if param.shape[0] != num_layers:
            raise AssertionError(f"Expected {num_layers} layers for {name}, got {param.shape}")
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            yield from get_expert_param(args, layer_name, param[layer_id])
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args: Any, state_dict: dict[str, torch.Tensor]):
    # [FUNCTION DOC] HELPER: Generator over the state dict that prefixes each key with the
    # doubled module wrapper prefix and yields the layer/expert-expanded (name, tensor) pairs.
    for name, param in state_dict.items():
        yield from get_layer_param(args, f"module.module.{name}", param)


_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>language_model\.)?decoder\.layers\.(?P<layer>\d+)\."
    r"mlp\.experts\.experts\.(?P<linear>linear_fc[12])\.weight$"
)


def parse_moe_expert_key(source_key: str) -> tuple[int, str, str] | None:
    # [FUNCTION DOC] HELPER: Matches names of the form (optional "language_model.") decoder
    # layers N mlp experts experts linear_fc1 or linear_fc2 weight. Returns (layer index, linear
    # name, HF prefix which is "language_model." if present else empty), or none if no match.
    match = _MOE_EXPERT_KEY_RE.match(source_key)
    if not match:
        return None
    hf_prefix = "language_model." if match.group("prefix") else ""
    return int(match.group("layer")), match.group("linear"), hf_prefix


def is_supported_moe_read_item(read_item: Any, source_key: str, tensor_size: torch.Size) -> bool:
    # [FUNCTION DOC] HELPER: True if the stored chunk can be converted directly. Requires an
    # expert key and a 3-D tensor (experts, ffn rows, hidden). The chunk's offsets and lengths
    # must be 3-D, lie within the expert count, and cover the full hidden dim. For the second
    # linear the chunk must span all rows. For the first linear (fused gate+up, row count must be
    # even) the chunk must span all rows, or be exactly the first half or the second half.
    parsed = parse_moe_expert_key(source_key)
    if parsed is None or len(tensor_size) != 3:
        return False
    _, linear_name, _ = parsed

    offsets = tuple(int(value) for value in read_item.storage_index.offset)
    lengths = tuple(int(value) for value in read_item.lengths)
    if len(offsets) != 3 or len(lengths) != 3:
        return False

    expert_offset, ffn_offset, hidden_offset = offsets
    expert_count, ffn_length, hidden_length = lengths
    num_experts, ffn_size, hidden_size = (int(value) for value in tensor_size)
    if expert_offset < 0 or expert_offset + expert_count > num_experts:
        return False
    if hidden_offset != 0 or hidden_length != hidden_size:
        return False
    if linear_name == "linear_fc2":
        return ffn_offset == 0 and ffn_length == ffn_size
    if ffn_size % 2 != 0:
        return False
    half_ffn = ffn_size // 2
    return (ffn_offset == 0 and ffn_length == ffn_size) or (ffn_offset in {0, half_ffn} and ffn_length == half_ffn)


def create_full_tensor_read_items(source_key: str, metadata: dist_cp.metadata.Metadata) -> list[Any]:
    # [FUNCTION DOC] HELPER: Builds read items covering all stored chunks of the key using the
    # torch planner helper; raises a type error if the key is not a tensor entry.
    md = metadata.state_dict_metadata[source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        raise TypeError(f"{source_key} is not tensor metadata")
    return create_read_items_for_chunk_list(source_key, md, list(md.chunks))


def create_direct_moe_read_items(block: MoeBlockSpec, md: dist_cp.metadata.TensorStorageMetadata) -> list[ReadItem]:
    # [FUNCTION DOC] HELPER: For each storage index of the block, finds its chunk by (key,
    # offsets, chunk position) and makes a tensor read item that reads the whole chunk (zero
    # storage offsets, chunk sizes) into the chunk's destination offsets. Returns the items
    # sorted by storage offset. Raises a key error for an index not among the chunks.
    chunk_by_index = {
        MetadataIndex(block.source_key, chunk.offsets, idx): chunk for idx, chunk in enumerate(md.chunks)
    }
    read_items: list[ReadItem] = []
    for storage_index in block.storage_indices:
        chunk = chunk_by_index[storage_index]
        read_items.append(
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(block.source_key),
                dest_offsets=chunk.offsets,
                storage_index=storage_index,
                storage_offsets=torch.Size([0 for _ in chunk.offsets]),
                lengths=chunk.sizes,
            )
        )
    read_items.sort(key=lambda item: tuple(item.storage_index.offset))
    return read_items


def create_moe_block_specs(source_key: str, metadata: dist_cp.metadata.Metadata) -> list[MoeBlockSpec] | None:
    # [FUNCTION DOC] HELPER: Returns block specs (one per storage file, sorted by path, chunks
    # sorted by offset) allowing direct reading of an expert weight, or none when the key is not
    # an expert key, not a tensor, has no chunks, any chunk is unsupported, or the metadata has no
    # storage data (caller then falls back to whole-key loading).
    parsed = parse_moe_expert_key(source_key)
    if parsed is None:
        return None
    layer_idx, linear_name, hf_prefix = parsed
    md = metadata.state_dict_metadata[source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        return None
    read_items = create_full_tensor_read_items(source_key, metadata)
    if not read_items or any(not is_supported_moe_read_item(item, source_key, md.size) for item in read_items):
        return None
    if metadata.storage_data is None:
        return None

    by_file: dict[str, list[MetadataIndex]] = {}
    for read_item in read_items:
        item_md = metadata.storage_data[read_item.storage_index]
        by_file.setdefault(item_md.relative_path, []).append(read_item.storage_index)

    return [
        MoeBlockSpec(
            source_key=source_key,
            relative_path=relative_path,
            storage_indices=tuple(sorted(indices, key=lambda index: tuple(index.offset))),
            layer_idx=layer_idx,
            linear_name=linear_name,
            hf_prefix=hf_prefix,
        )
        for relative_path, indices in sorted(by_file.items())
    ]


def expert_source_name(source_key: str, expert_id: int) -> str:
    # [FUNCTION DOC] HELPER: Name identifying one expert's source parameter: the wrapper prefix,
    # the key with the doubled experts segment collapsed, and the expert index appended (same
    # scheme as the stacked-expert expansion).
    return "module.module." + source_key.replace(".experts.experts.", ".experts.") + str(expert_id)


def hf_expert_weight_name(hf_prefix: str, layer_idx: int, expert_id: int, projection: str) -> str:
    # [FUNCTION DOC] HELPER: HF tensor name: prefix, model layers N, mlp experts E, projection
    # name, ".weight".
    return f"{hf_prefix}model.layers.{layer_idx}.mlp.experts.{expert_id}.{projection}.weight"


def contiguous_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Returns the tensor itself if contiguous, else a contiguous copy.
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def converted_moe_tensors_from_chunk(
    source_key: str,
    layer_idx: int,
    linear_name: str,
    hf_prefix: str,
    read_item: Any,
    tensor: torch.Tensor,
    tensor_size: torch.Size,
) -> list[PreparedTensorGroup]:
    # [FUNCTION DOC] HELPER: Converts one loaded chunk (experts by rows by hidden) to per-expert
    # HF tensors. Asserts the chunk's expert count. Second linear: each expert becomes a down
    # projection. First linear (fused gate and up stacked along rows): a chunk spanning all rows
    # is split in two halves along rows into gate and up projections; a chunk covering exactly
    # the first half becomes only gate projections; exactly the second half only up projections;
    # any other layout raises an assertion error. Expert ids are the chunk's expert offset plus
    # the local index. Returns one group per expert, tagged with its source name.
    offsets = tuple(int(value) for value in read_item.storage_index.offset)
    lengths = tuple(int(value) for value in read_item.lengths)
    expert_offset = offsets[0]
    expert_count = lengths[0]
    if tensor.shape[0] != expert_count:
        raise AssertionError(f"Expected {expert_count} experts in {source_key} chunk, got {tensor.shape}")

    groups: list[PreparedTensorGroup] = []
    if linear_name == "linear_fc2":
        for local_expert_idx in range(expert_count):
            expert_id = expert_offset + local_expert_idx
            groups.append(
                PreparedTensorGroup(
                    source_name=expert_source_name(source_key, expert_id),
                    tensors=(
                        (
                            hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "down_proj"),
                            contiguous_if_needed(tensor[local_expert_idx]),
                        ),
                    ),
                )
            )
        return groups

    half_ffn = int(tensor_size[1]) // 2
    second_dim_offset = offsets[1]
    second_dim_length = lengths[1]
    for local_expert_idx in range(expert_count):
        expert_id = expert_offset + local_expert_idx
        expert_tensor = tensor[local_expert_idx]
        if second_dim_offset == 0 and second_dim_length == int(tensor_size[1]):
            gate_weight, up_weight = expert_tensor.chunk(2, dim=0)
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "gate_proj"),
                    contiguous_if_needed(gate_weight),
                ),
                (hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "up_proj"), contiguous_if_needed(up_weight)),
            )
        elif second_dim_offset == 0 and second_dim_length == half_ffn:
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "gate_proj"),
                    contiguous_if_needed(expert_tensor),
                ),
            )
        elif second_dim_offset == half_ffn and second_dim_length == half_ffn:
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "up_proj"),
                    contiguous_if_needed(expert_tensor),
                ),
            )
        else:
            raise AssertionError(
                f"Unsupported {linear_name} chunk for {source_key}: "
                f"offsets={offsets}, lengths={lengths}, size={tuple(tensor_size)}"
            )
        groups.append(
            PreparedTensorGroup(source_name=expert_source_name(source_key, expert_id), tensors=named_tensors)
        )
    return groups


def load_moe_block_direct(
    input_dir: str,
    block: MoeBlockSpec,
    metadata: dist_cp.metadata.Metadata,
) -> DirectMoeLoadResult:
    # [FUNCTION DOC] HELPER: Reads the chunks of one expert block directly from its single
    # storage file and converts them. Raises a type error for non-tensor metadata. Prepares a
    # reader with cached metadata, builds the read items, computes accounting, opens the file once
    # and for each item (sorted by offset) takes the stored byte range, applies the load
    # transform, deserializes (weights-only, CPU), narrows to the requested region and converts
    # via the chunk converter. Returns the groups with accounting counts.
    md = metadata.state_dict_metadata[block.source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        raise TypeError(f"{block.source_key} is not tensor metadata")

    storage_reader = MeteredStorageReader(input_dir)
    metadata = prepare_cached_metadata_for_reader(metadata, storage_reader)
    storage_reader.set_up_storage_reader(metadata, is_coordinator=True)
    read_items = create_direct_moe_read_items(block, md)
    read_count, file_count, storage_bytes = compute_dcp_load_accounting(
        storage_reader.storage_data, LoadPlan(read_items)
    )

    tensor_groups: list[PreparedTensorGroup] = []
    new_path = storage_reader.fs.concat_path(storage_reader.path, block.relative_path)
    with storage_reader.fs.create_stream(new_path, "rb") as stream:
        for read_item in read_items:
            item_md = storage_reader.storage_data[read_item.storage_index]
            file_slice = cast(io.IOBase, _create_file_view(stream, item_md.offset, item_md.length))
            transform_from = storage_reader.transforms.transform_load_stream(
                read_item,
                item_md.transform_descriptors or (),
                file_slice,
            )
            seekable = transform_from if transform_from.seekable() else io.BytesIO(transform_from.read(-1))
            seekable.seek(0)
            tensor = cast(torch.Tensor, torch.load(seekable, map_location="cpu", weights_only=True))
            tensor = narrow_tensor_by_index(tensor, read_item.storage_offsets, read_item.lengths)
            tensor_groups.extend(
                converted_moe_tensors_from_chunk(
                    block.source_key,
                    block.layer_idx,
                    block.linear_name,
                    block.hf_prefix,
                    read_item,
                    tensor,
                    md.size,
                )
            )

    return DirectMoeLoadResult(tuple(tensor_groups), read_count, file_count, storage_bytes)


def tensor_metadata_from_checkpoint_metadata(
    metadata: dist_cp.metadata.Metadata,
) -> dict[str, tuple[torch.Size, torch.dtype]]:
    # [FUNCTION DOC] HELPER: Mapping from key to (size, dtype) of all tensor entries, skipping
    # keys containing "optimizer" or "_state" and non-tensor entries.
    tensor_metadata = {}
    for key, value in metadata.state_dict_metadata.items():
        if "optimizer" in key or "_state" in key:
            continue
        if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
            tensor_metadata[key] = (value.size, value.properties.dtype)
    return tensor_metadata


def tensor_nbytes(shape: torch.Size, dtype: torch.dtype) -> int:
    # [FUNCTION DOC] HELPER: Element count times bytes per element, from the dtype's bit width
    # (float info for floating types else integer info; integer division by 8, so sub-byte
    # types and bool/complex are unsupported and may raise).
    element_bits = torch.finfo(dtype).bits if dtype.is_floating_point else torch.iinfo(dtype).bits
    return shape.numel() * (element_bits // 8)


def filter_tensor_metadata(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    source_key_regex: str | None,
) -> dict[str, tuple[torch.Size, torch.dtype]]:
    # [FUNCTION DOC] HELPER: Returns the metadata unchanged when no regular expression is given;
    # otherwise only entries whose key contains a match (search, not full match).
    if not source_key_regex:
        return tensor_metadata
    pattern = re.compile(source_key_regex)
    return {key: value for key, value in tensor_metadata.items() if pattern.search(key)}


def _mla_pair_group(key: str) -> str:
    # [FUNCTION DOC] HELPER: Grouping key: renames the query-down and key-value-down projection
    # weights of self-attention to one shared placeholder name, so that the two weights of a layer
    # land in the same task (their conversion needs both).
    return key.replace("self_attention.linear_q_down_proj.weight", "self_attention.MLA_A_PAIR.weight").replace(
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.MLA_A_PAIR.weight",
    )


def group_small_tasks(
    atomic_tasks: list[tuple[int, tuple[str, ...]]],
    task_group_bytes: int,
) -> list[tuple[int, tuple[str, ...]]]:
    # [FUNCTION DOC] HELPER: Packs small tasks. With a non-positive budget returns the tasks
    # unchanged. Otherwise tasks of at least the budget stay alone; smaller ones are sorted by
    # descending bytes (ties by keys) and greedily appended to the current group, starting a new
    # group when adding would exceed the budget; group keys are sorted. Returns large tasks first
    # then groups, as (bytes, keys) pairs.
    if task_group_bytes <= 0:
        return atomic_tasks

    large_tasks = [task for task in atomic_tasks if task[0] >= task_group_bytes]
    small_tasks = [task for task in atomic_tasks if task[0] < task_group_bytes]
    small_tasks.sort(key=lambda item: (-item[0], item[1]))
    grouped_tasks: list[tuple[int, tuple[str, ...]]] = []
    current_keys: list[str] = []
    current_bytes = 0
    for estimated_bytes, keys in small_tasks:
        if current_keys and current_bytes + estimated_bytes > task_group_bytes:
            grouped_tasks.append((current_bytes, tuple(sorted(current_keys))))
            current_keys = []
            current_bytes = 0
        current_keys.extend(keys)
        current_bytes += estimated_bytes
    if current_keys:
        grouped_tasks.append((current_bytes, tuple(sorted(current_keys))))
    return large_tasks + grouped_tasks


def plan_whole_source_tasks(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    q_lora_rank: int | None,
    task_group_bytes: int,
) -> list[TaskSpec]:
    # [FUNCTION DOC] HELPER: Plans tasks for whole-key loading. Groups keys (by the MLA pair
    # grouping when a q-lora rank is set, else one group per key), estimates bytes per group,
    # packs small groups by the budget, sorts by descending bytes (then keys) and numbers them
    # from zero. Returns the task specs.
    grouped: dict[str, list[str]] = {}
    for key in tensor_metadata:
        group = _mla_pair_group(key) if q_lora_rank is not None else key
        grouped.setdefault(group, []).append(key)

    atomic_tasks = []
    for keys in grouped.values():
        sorted_keys = tuple(sorted(keys))
        estimated_bytes = sum(tensor_nbytes(tensor_metadata[key][0], tensor_metadata[key][1]) for key in sorted_keys)
        atomic_tasks.append((estimated_bytes, sorted_keys))
    raw_tasks = group_small_tasks(atomic_tasks, task_group_bytes)
    raw_tasks.sort(key=lambda item: (-item[0], item[1]))
    return [TaskSpec(idx, keys, estimated_bytes) for idx, (estimated_bytes, keys) in enumerate(raw_tasks)]


def collect_moe_blocks_by_file(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    metadata: dist_cp.metadata.Metadata,
) -> tuple[list[tuple[int, MoeBlockSpec]], dict[str, tuple[torch.Size, torch.dtype]]]:
    # [FUNCTION DOC] HELPER: Splits keys into directly-readable expert blocks and the rest.
    # Without storage data in the metadata, nothing is direct. Otherwise, for each key in sorted
    # order that yields block specs, adds (stored bytes of its chunks, block) pairs and removes
    # the key from the whole-source remainder. Returns (blocks, remaining metadata copy).
    moe_blocks: list[tuple[int, MoeBlockSpec]] = []
    whole_source_metadata = dict(tensor_metadata)
    if metadata.storage_data is None:
        return moe_blocks, whole_source_metadata

    for source_key in sorted(tensor_metadata):
        block_specs = create_moe_block_specs(source_key, metadata)
        if block_specs is None:
            continue
        for block in block_specs:
            estimated_bytes = sum(int(metadata.storage_data[index].length) for index in block.storage_indices)
            moe_blocks.append((estimated_bytes, block))
        del whole_source_metadata[source_key]
    return moe_blocks, whole_source_metadata


def group_moe_block_tasks(moe_blocks: list[tuple[int, MoeBlockSpec]], task_group_bytes: int) -> list[TaskSpec]:
    # [FUNCTION DOC] HELPER: Sorts the blocks in place by key, file and first chunk offset, then
    # packs consecutive blocks into tasks up to the byte target (the budget, or 2 GiB when the
    # budget is zero; a block larger than the target gets its own task). Tasks carry the sorted
    # distinct source keys, the byte sum and the blocks, with placeholder id minus one. Returns an
    # empty list for no blocks. Mutates the input list order.
    if not moe_blocks:
        return []
    target_bytes = task_group_bytes or DEFAULT_DIRECT_MOE_GROUP_SIZE
    moe_blocks.sort(
        key=lambda item: (
            item[1].source_key,
            item[1].relative_path,
            tuple(item[1].storage_indices[0].offset) if item[1].storage_indices else (),
        )
    )

    tasks: list[TaskSpec] = []
    current_blocks: list[MoeBlockSpec] = []
    current_bytes = 0

    def flush_current() -> None:
        # [FUNCTION DOC] HELPER: Emits the accumulated blocks as one task (if any) and resets the
        # accumulators.
        nonlocal current_blocks, current_bytes
        if not current_blocks:
            return
        tasks.append(
            TaskSpec(
                task_id=-1,
                keys=tuple(sorted({block.source_key for block in current_blocks})),
                estimated_source_bytes=current_bytes,
                moe_blocks=tuple(current_blocks),
            )
        )
        current_blocks = []
        current_bytes = 0

    for estimated_bytes, block in moe_blocks:
        if current_blocks and current_bytes + estimated_bytes > target_bytes:
            flush_current()
        current_blocks.append(block)
        current_bytes += estimated_bytes
    flush_current()
    return tasks


def plan_conversion_tasks(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    metadata: dist_cp.metadata.Metadata,
    q_lora_rank: int | None,
    task_group_bytes: int,
) -> list[TaskSpec]:
    # [FUNCTION DOC] HELPER: Full task plan: direct expert block tasks plus whole-key tasks,
    # merged and sorted by descending bytes, then first key (block key if direct), file path and
    # keys, and renumbered from zero. Largest tasks are dispatched first.
    moe_blocks, whole_source_metadata = collect_moe_blocks_by_file(tensor_metadata, metadata)
    tasks = group_moe_block_tasks(moe_blocks, task_group_bytes)
    tasks.extend(
        TaskSpec(-1, task.keys, task.estimated_source_bytes)
        for task in plan_whole_source_tasks(whole_source_metadata, q_lora_rank, task_group_bytes)
    )
    tasks.sort(
        key=lambda task: (
            -task.estimated_source_bytes,
            task.moe_blocks[0].source_key if task.moe_blocks else task.keys[0],
            task.moe_blocks[0].relative_path if task.moe_blocks else "",
            task.keys,
        )
    )
    return [TaskSpec(idx, task.keys, task.estimated_source_bytes, task.moe_blocks) for idx, task in enumerate(tasks)]


def summarize_plan(tasks: list[TaskSpec], model_name: str, concurrency: int, output_dir: str) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Returns a JSON-friendly mapping summarizing the plan: model name,
    # concurrency, output dir, task count, total source keys, direct expert task and block counts,
    # total, largest and smallest estimated bytes (zero if no tasks) and the first ten tasks
    # (id, keys, bytes, block count).
    source_bytes = [task.estimated_source_bytes for task in tasks]
    direct_moe_tasks = [task for task in tasks if task.moe_blocks]
    return {
        "model_name": model_name,
        "concurrency": concurrency,
        "output_dir": output_dir,
        "tasks": len(tasks),
        "source_keys": sum(len(task.keys) for task in tasks),
        "direct_moe_tasks": len(direct_moe_tasks),
        "direct_moe_blocks": sum(len(task.moe_blocks) for task in direct_moe_tasks),
        "estimated_source_bytes": sum(source_bytes),
        "largest_task_bytes": max(source_bytes, default=0),
        "smallest_task_bytes": min(source_bytes, default=0),
        "top_tasks": [
            {
                "task_id": task.task_id,
                "keys": task.keys,
                "estimated_source_bytes": task.estimated_source_bytes,
                "moe_blocks": len(task.moe_blocks),
            }
            for task in tasks[:10]
        ],
    }


def load_hf_config(origin_hf_dir: str | None) -> Any | None:
    # [FUNCTION DOC] HELPER: Loads the HF config from the directory, trusting remote code; none if
    # no directory is given.
    if origin_hf_dir is None:
        return None
    return _load_hf_config(origin_hf_dir, trust_remote_code=True)


def load_quantization_config(hf_config: Any | None) -> dict[str, Any] | None:
    # [FUNCTION DOC] HELPER: Returns a plain mapping copy of the config's quantization section,
    # looking at the top level then the nested text config; none if no config or no section.
    if hf_config is None:
        return None
    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is not None:
        return dict(quantization_config)
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        nested = getattr(text_config, "quantization_config", None)
        if nested is not None:
            return dict(nested)
    return None


def get_hf_vocab_size(hf_config: Any | None) -> int | None:
    # [FUNCTION DOC] HELPER: Vocabulary size from the nested text config if it has one, else the
    # top-level config; none if absent or no config.
    if hf_config is None:
        return None
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    vocab_size = getattr(hf_config, "vocab_size", None)
    return int(vocab_size) if vocab_size is not None else None


def copy_assets(origin_hf_dir: str | None, output_dir: str) -> None:
    # [FUNCTION DOC] HELPER: Does nothing without an original directory. Otherwise copies every
    # regular file except safetensors files and the index into the output directory, keeping
    # metadata (overwrites existing).
    if origin_hf_dir is None:
        return
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        src = os.path.join(origin_hf_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, filename))


def _flush_shard(
    staging_dir: str,
    task_id: int,
    shard_idx: int,
    tensors: dict[str, torch.Tensor],
    current_size: int,
) -> ShardManifest:
    # [FUNCTION DOC] HELPER: Saves the tensors to a staging file named from the host name, task
    # id and shard index (five digits each), and returns its manifest (final name none, tensor
    # names, the given byte size, tensor count).
    filename = f"worker-{socket.gethostname()}-task-{task_id:05d}-shard-{shard_idx:05d}.safetensors"
    safetensors.torch.save_file(tensors, os.path.join(staging_dir, filename))
    return ShardManifest(filename, None, tuple(tensors.keys()), current_size, len(tensors))


def append_to_shards(
    staging_dir: str,
    task_id: int,
    shard_idx: int,
    current_tensors: dict[str, torch.Tensor],
    current_size: int,
    converted_named_tensors: tuple[tuple[str, torch.Tensor], ...] | list[tuple[str, torch.Tensor]],
    max_file_bytes: int,
    shards: list[ShardManifest],
) -> tuple[int, int, int]:
    # [FUNCTION DOC] HELPER: Adds converted tensors to the current shard buffer, flushing the
    # buffer to a staged file (appending its manifest to the list, incrementing the shard index,
    # clearing the buffer in place) before adding a tensor that would push it over the maximum
    # (only if the buffer is non-empty). Returns (next shard index, current buffer bytes, bytes
    # added by this call). Mutates the buffer and manifest list.
    total_size = 0
    for converted_name, converted_param in converted_named_tensors:
        tensor_size = converted_param.numel() * converted_param.element_size()
        if tensor_size + current_size > max_file_bytes and current_tensors:
            shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size))
            shard_idx += 1
            current_tensors.clear()
            current_size = 0
        current_tensors[converted_name] = converted_param
        current_size += tensor_size
        total_size += tensor_size
    return shard_idx, current_size, total_size


def prepare_moe_block_task_tensors(
    task: TaskSpec,
    input_dir: str,
    metadata: dist_cp.metadata.Metadata,
) -> PreparedTaskTensors:
    # [FUNCTION DOC] HELPER: Loads and converts every expert block of a direct task, concatenating
    # the groups and summing the read accounting.
    groups: list[PreparedTensorGroup] = []
    total_read_items = 0
    total_files = 0
    total_storage_bytes = 0
    for block in task.moe_blocks:
        result = load_moe_block_direct(input_dir, block, metadata)
        groups.extend(result.tensor_groups)
        total_read_items += result.read_items
        total_files += result.files
        total_storage_bytes += result.storage_bytes
    return PreparedTaskTensors(tuple(groups), TaskLoadStats(total_read_items, total_files, total_storage_bytes))


def prepare_whole_source_task_tensors(
    task: TaskSpec,
    input_dir: str,
    megatron_args: Any,
    model_name: str,
    metadata: dist_cp.metadata.Metadata,
) -> PreparedTaskTensors:
    # [FUNCTION DOC] HELPER: Loads the task's keys fully, expands them to per-layer and
    # per-expert parameters, strips vocabulary padding when the args carry a vocab size, and
    # converts each parameter with the shared conversion core (the private non-quantizing one).
    # One group per source parameter. The loaded state dict reference is deleted afterwards.
    load_result = load_tensor_chunk(input_dir, set(task.keys), metadata)
    state_dict = load_result.state_dict

    groups: list[PreparedTensorGroup] = []
    try:
        for name, param in get_named_params(megatron_args, state_dict):
            if getattr(megatron_args, "vocab_size", None) is not None:
                param = m2hf.remove_padding(name, param, megatron_args.vocab_size)
            converted_named_tensors = m2hf._convert_to_hf_core(megatron_args, model_name, name, param)
            groups.append(PreparedTensorGroup(name, tuple(converted_named_tensors)))
        return PreparedTaskTensors(
            tuple(groups),
            TaskLoadStats(load_result.read_items, load_result.files, load_result.storage_bytes),
        )
    finally:
        del state_dict


def write_prepared_tensor_groups(
    staging_dir: str,
    task_id: int,
    groups: tuple[PreparedTensorGroup, ...],
    megatron_args: Any,
    quantization_config: dict[str, Any] | None,
    max_file_bytes: int,
    cuda_device_id: int | None,
) -> tuple[tuple[ShardManifest, ...], int]:
    # [FUNCTION DOC] HELPER: Writes the groups to staged shard files. When a quantization config
    # exists, each group is first quantized by the shared quantizer (selecting the assigned CUDA
    # device before each call). Tensors are appended to shards limited by the maximum size; the
    # last partial shard is flushed. Returns (manifests, total converted bytes).
    current_tensors: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0
    shards: list[ShardManifest] = []
    total_size = 0

    for group in groups:
        converted_named_tensors = group.tensors
        if quantization_config is not None:
            if cuda_device_id is not None:
                torch.cuda.set_device(cuda_device_id)
            converted_named_tensors = tuple(
                m2hf.quantize_params(
                    megatron_args, group.source_name, list(converted_named_tensors), quantization_config
                )
            )
        shard_idx, current_size, added_size = append_to_shards(
            staging_dir,
            task_id,
            shard_idx,
            current_tensors,
            current_size,
            converted_named_tensors,
            max_file_bytes,
            shards,
        )
        total_size += added_size

    if current_tensors:
        shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size))
    return tuple(shards), total_size


def assign_cuda_device_id(actor_id: int, device_count: int) -> int:
    # [FUNCTION DOC] HELPER: Round-robin device index (actor id modulo device count); raises a
    # runtime error when no devices are visible.
    if device_count <= 0:
        raise RuntimeError("Quantization requires CUDA, but no CUDA devices are visible")
    return actor_id % device_count


def initialize_worker_cuda_device(actor_id: int, quantization_config: dict[str, Any] | None) -> int | None:
    # [FUNCTION DOC] HELPER: Returns none when not quantizing. Otherwise requires CUDA (runtime
    # error if unavailable), picks the round-robin device, makes it current and returns its index.
    if quantization_config is None:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("Quantization requires CUDA, but CUDA is unavailable")
    cuda_device_id = assign_cuda_device_id(actor_id, torch.cuda.device_count())
    torch.cuda.set_device(cuda_device_id)
    return cuda_device_id


class ConversionWorker:
    # [CLASS DOC] Ray actor body (wrapped by a factory with no reserved CPU or GPU and
    # unbuffered output). Holds per-worker settings and runs conversion tasks one at a time.
    def __init__(
        self,
        actor_id: int,
        input_dir: str,
        staging_dir: str,
        megatron_args: Any,
        model_name: str,
        quantization_config: dict[str, Any] | None,
        max_file_bytes: int,
        metadata_ref: Any,
    ) -> None:
        # [FUNCTION DOC] Stores settings, records the Ray node id, picks the CUDA device when
        # quantizing, and resolves the checkpoint metadata: used directly if it is already a
        # metadata object, else fetched from the Ray object store reference.
        self.actor_id = actor_id
        self.ray_node_id = ray.get_runtime_context().get_node_id()
        self.cuda_device_id = initialize_worker_cuda_device(actor_id, quantization_config)
        self.input_dir = input_dir
        self.staging_dir = staging_dir
        self.megatron_args = megatron_args
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.max_file_bytes = max_file_bytes
        self.metadata = metadata_ref if isinstance(metadata_ref, dist_cp.metadata.Metadata) else ray.get(metadata_ref)

    def convert(self, task: TaskSpec) -> TaskResult:
        # [FUNCTION DOC] Runs one task: reads and converts the tensors (direct expert reads when
        # the task has blocks, else whole-key loading), writes staged shards (quantizing if
        # configured) and returns the result record with host, process id, shard manifests,
        # byte counts, tensor count, read accounting, device and node information.
        node = socket.gethostname()
        pid = os.getpid()
        if task.moe_blocks:
            prepared = prepare_moe_block_task_tensors(task, self.input_dir, self.metadata)
        else:
            prepared = prepare_whole_source_task_tensors(
                task, self.input_dir, self.megatron_args, self.model_name, self.metadata
            )
        shards, total_size = write_prepared_tensor_groups(
            self.staging_dir,
            task.task_id,
            prepared.groups,
            self.megatron_args,
            self.quantization_config,
            self.max_file_bytes,
            self.cuda_device_id,
        )
        return TaskResult(
            task_id=task.task_id,
            actor_id=self.actor_id,
            node=node,
            pid=pid,
            shards=shards,
            source_bytes=task.estimated_source_bytes,
            output_bytes=total_size,
            weights=sum(len(shard.weight_keys) for shard in shards),
            source_keys=task.keys,
            dcp_read_items=prepared.load_stats.read_items,
            dcp_files=prepared.load_stats.files,
            dcp_storage_bytes=prepared.load_stats.storage_bytes,
            cuda_device_id=self.cuda_device_id,
            ray_node_id=self.ray_node_id,
        )


def initialize_ray() -> None:
    # [FUNCTION DOC] HELPER: Sets (without overriding) two environment variables so Ray neither
    # overrides accelerator variables on zero GPUs nor rewrites visible devices. Returns if Ray
    # is already initialized; else connects to an existing cluster ("auto"), and on a connection
    # error starts a local instance.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    os.environ.setdefault("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        ray.init(ignore_reinit_error=True)


def live_ray_node_ids() -> list[str]:
    # [FUNCTION DOC] HELPER: Sorted ids of Ray nodes reported alive; raises a runtime error if none.
    node_ids = [str(node["NodeID"]) for node in ray.nodes() if node.get("Alive", False)]
    node_ids.sort()
    if not node_ids:
        raise RuntimeError("Ray has no live nodes")
    return node_ids


def make_conversion_actor():
    # [FUNCTION DOC] HELPER: Returns the worker class wrapped as a Ray actor class requesting no
    # CPUs or GPUs and with environment variables for unbuffered output and the two accelerator
    # settings.
    return ray.remote(
        num_cpus=0,
        num_gpus=0,
        runtime_env={
            "env_vars": {
                "PYTHONUNBUFFERED": "1",
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        },
    )(ConversionWorker)


def collect_ray_results(
    tasks: list[TaskSpec],
    input_dir: str,
    staging_dir: str,
    megatron_args: Any,
    model_name: str,
    quantization_config: dict[str, Any] | None,
    max_file_bytes: int,
    concurrency: int,
    metadata_ref: Any,
    progress: bool,
    progress_interval_seconds: float,
) -> list[TaskResult]:
    # [FUNCTION DOC] HELPER: Dispatches tasks to workers and collects results. Creates
    # min(concurrency, task count) actors (none if zero, returning an empty list), each hard-pinned
    # to a live node in round-robin order. Submits one initial task per worker in task order, then
    # waits for any completion (timing out at the progress interval to refresh the bar); on each
    # completion records the result, updates progress (or prints a line when the bar is disabled)
    # and gives that worker the next unsubmitted task. Returns results sorted by task id. Worker
    # failures propagate when fetching results. Actors are not explicitly killed.
    worker_count = min(concurrency, len(tasks))
    if worker_count < 1:
        return []

    worker_cls = make_conversion_actor()
    node_ids = live_ray_node_ids()
    workers = []
    for actor_id in range(worker_count):
        node_id = node_ids[actor_id % len(node_ids)]
        actor_cls = worker_cls.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False))
        workers.append(
            actor_cls.remote(
                actor_id,
                input_dir,
                staging_dir,
                megatron_args,
                model_name,
                quantization_config,
                max_file_bytes,
                metadata_ref,
            )
        )

    pending: dict[Any, int] = {}
    submitted = 0
    results: list[TaskResult] = []
    progress_reporter = ProgressReporter(tasks, progress, progress_interval_seconds)
    progress_reporter.tick()
    for worker_idx, worker in enumerate(workers):
        if submitted >= len(tasks):
            break
        pending[worker.convert.remote(tasks[submitted])] = worker_idx
        submitted += 1

    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1, timeout=progress_reporter.interval_seconds)
        if not ready:
            progress_reporter.tick()
            continue
        ready_ref = ready[0]
        worker_idx = pending.pop(ready_ref)
        result = ray.get(ready_ref)
        results.append(result)
        progress_reporter.complete(result)
        if not progress:
            print(
                f"task {result.task_id} finished on {result.node}: "
                f"{result.output_bytes / 1e9:.2f} GB output, {result.weights} tensors"
            )
        if submitted < len(tasks):
            pending[workers[worker_idx].convert.remote(tasks[submitted])] = worker_idx
            submitted += 1

    progress_reporter.finish()
    results.sort(key=lambda result: result.task_id)
    return results


def plan_global_shards(task_results: list[TaskResult]) -> tuple[tuple[PlannedShard, ...], dict[str, Any]]:
    # [FUNCTION DOC] HELPER: Flattens shards in task id order (raises a value error if there are
    # none), names them model-IIIII-of-NNNNN.safetensors by position, builds the weight map
    # (a duplicate tensor name raises a value error) and sums bytes. Returns the planned shards
    # and the index mapping with total size and weight map.
    all_shards = [shard for result in sorted(task_results, key=lambda item: item.task_id) for shard in result.shards]
    if not all_shards:
        raise ValueError("No HF tensor shards were emitted")
    weight_map: dict[str, str] = {}
    planned_shards: list[PlannedShard] = []
    total_size = 0
    for idx, shard in enumerate(all_shards):
        final_filename = f"model-{idx:05d}-of-{len(all_shards):05d}.safetensors"
        for key in shard.weight_keys:
            if key in weight_map:
                raise ValueError(f"Duplicate HF tensor emitted during finalization: {key}")
            weight_map[key] = final_filename
        planned_shards.append(PlannedShard(shard.temp_filename, final_filename, shard.weight_keys, shard.bytes))
        total_size += shard.bytes
    return tuple(planned_shards), {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def finalize_output(
    staging_dir: str,
    output_dir: str,
    origin_hf_dir: str | None,
    task_results: list[TaskResult],
) -> None:
    # [FUNCTION DOC] HELPER: Plans global shards, copies assets, atomically renames each staged
    # shard into the output directory under its final name, writes the index file, and deletes
    # the staging directory.
    planned_shards, index = plan_global_shards(task_results)
    copy_assets(origin_hf_dir, output_dir)
    for shard in planned_shards:
        os.replace(os.path.join(staging_dir, shard.temp_filename), os.path.join(output_dir, shard.final_filename))
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    shutil.rmtree(staging_dir)


def reject_cloud_path(path: str, label: str) -> None:
    # [FUNCTION DOC] HELPER: Raises a value error naming the label if the path contains a URL
    # scheme separator (remote storage), since only local paths are supported.
    if "://" in path:
        raise ValueError(f"{label} must be a local filesystem path, got {path}")


def prepare_output_dir(output_dir: str, force: bool) -> str:
    # [FUNCTION DOC] HELPER: Rejects remote paths; if the directory exists, raises file-exists
    # unless forced, in which case it is deleted recursively (destructive). Creates the directory
    # and a hidden staging subdirectory inside it; returns the staging path.
    reject_cloud_path(output_dir, "output_dir")
    if os.path.exists(output_dir):
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    staging_dir = os.path.join(output_dir, ".ray-convert-staging")
    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir


def load_megatron_args(input_dir: str, model_name_override: str | None, vocab_size: int | None) -> tuple[Any, str]:
    # [FUNCTION DOC] HELPER: Loads the saved Megatron args from the checkpoint's common file
    # (full unpickling, not weights-only: runs pickled code, trusted input assumed). Model name is
    # the override, else the recorded original HF model name; a value error if neither. Sets the
    # vocab size attribute when given, and defaults the EP-MoE flag to false if absent. Returns
    # (args, model name).
    megatron_args = torch.load(os.path.join(input_dir, "common.pt"), weights_only=False)["args"]
    model_name = model_name_override or getattr(megatron_args, "original_hf_model_name", None)
    if model_name is None:
        raise ValueError("Model name is required when common.pt does not include original_hf_model_name")
    if vocab_size is not None:
        megatron_args.vocab_size = vocab_size
    if not hasattr(megatron_args, "sglang_enable_ep_moe"):
        megatron_args.sglang_enable_ep_moe = False
    return megatron_args, model_name


def read_metadata_and_plan(args: Args, megatron_args: Any) -> tuple[dist_cp.metadata.Metadata, list[TaskSpec]]:
    # [FUNCTION DOC] HELPER: Reads the checkpoint metadata, extracts tensor entries, applies the
    # optional key filter (a value error if the filter matches nothing), and plans tasks using
    # the q-lora rank from the Megatron args (none if absent) and the packing budget. Returns
    # (metadata, tasks).
    metadata = WrappedStorageReader(args.input_dir).read_metadata()
    tensor_metadata = tensor_metadata_from_checkpoint_metadata(metadata)
    tensor_metadata = filter_tensor_metadata(tensor_metadata, args.source_key_regex)
    if args.source_key_regex and not tensor_metadata:
        raise ValueError(f"No checkpoint keys matched {args.source_key_regex}")
    tasks = plan_conversion_tasks(
        tensor_metadata, metadata, getattr(megatron_args, "q_lora_rank", None), args.task_group_bytes
    )
    return metadata, tasks


def convert_torch_dist_to_hf_ray(args: Args) -> str:
    # [FUNCTION DOC] Top-level routine (see the stage diagram). Validates local paths and the
    # presence of the common file (file-not-found otherwise); loads the original HF config (if
    # any) for vocab size and quantization section; loads Megatron args; plans; computes
    # concurrency (given, else min of task count and 16, at least 1); prints the plan summary as
    # JSON; with dry-run returns the output dir without writing; raises a value error if no tasks.
    # Otherwise prepares the output (deleting an existing one when forced), initializes Ray,
    # publishes metadata in the object store, runs the workers and finalizes. Returns the output dir.
    reject_cloud_path(args.input_dir, "input_dir")
    if args.origin_hf_dir is not None:
        reject_cloud_path(args.origin_hf_dir, "origin_hf_dir")
    common_pt = os.path.join(args.input_dir, "common.pt")
    if not os.path.exists(common_pt):
        raise FileNotFoundError(f"Expected {common_pt}")

    hf_config = load_hf_config(args.origin_hf_dir)
    vocab_size = get_hf_vocab_size(hf_config)
    megatron_args, model_name = load_megatron_args(args.input_dir, args.model_name, vocab_size)
    quantization_config = load_quantization_config(hf_config)

    metadata, tasks = read_metadata_and_plan(args, megatron_args)
    concurrency = args.concurrency or min(max(len(tasks), 1), 16)
    print(json.dumps(summarize_plan(tasks, model_name, concurrency, args.output_dir), indent=2, default=str))
    if args.dry_run_plan:
        return args.output_dir
    if not tasks:
        raise ValueError("No checkpoint tensor tasks were planned")

    staging_dir = prepare_output_dir(args.output_dir, args.force)
    initialize_ray()
    metadata_ref = ray.put(metadata)
    task_results = collect_ray_results(
        tasks,
        args.input_dir,
        staging_dir,
        megatron_args,
        model_name,
        quantization_config,
        args.max_file_bytes,
        concurrency,
        metadata_ref,
        args.progress,
        args.progress_interval_seconds,
    )
    finalize_output(staging_dir, args.output_dir, args.origin_hf_dir, task_results)
    return args.output_dir


def parse_args() -> Args:
    # [FUNCTION DOC] HELPER: Parses options: input dir and output dir (required), original HF
    # dir, model name, force, maximum file bytes (default 20 GiB), concurrency, task group bytes
    # (default 0), source key regex, dry-run-plan, a flag to disable progress (progress defaults
    # on) and progress interval seconds (default 5.0). Returns the immutable options record.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--max-file-bytes", type=int, default=20 * 1024**3)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--task-group-bytes", type=int, default=0)
    parser.add_argument("--source-key-regex", default=None)
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    parser.set_defaults(progress=True)
    ns = parser.parse_args()
    return Args(
        input_dir=ns.input_dir,
        output_dir=ns.output_dir,
        origin_hf_dir=ns.origin_hf_dir,
        model_name=ns.model_name,
        force=ns.force,
        max_file_bytes=ns.max_file_bytes,
        concurrency=ns.concurrency,
        task_group_bytes=ns.task_group_bytes,
        source_key_regex=ns.source_key_regex,
        dry_run_plan=ns.dry_run_plan,
        progress=ns.progress,
        progress_interval_seconds=ns.progress_interval_seconds,
    )


def main() -> None:
    # [FUNCTION DOC] Entry point: parses the command line and runs the conversion.
    convert_torch_dist_to_hf_ray(parse_args())


if __name__ == "__main__":
    main()
