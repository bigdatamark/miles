# [MODULE DOC] Weight iterator that converts Megatron-local parameters to Hugging Face tensors
# directly (without Megatron Bridge). Pipeline per batch: load or allocate each parameter on the
# GPU on its owning rank, optionally broadcast across pipeline stages, all-gather across tensor
# parallelism (honoring strided/interleaved sharding), for experts also gather across expert
# parallelism, then convert/quantize to Hugging Face names. Parameters are packed into batches
# bounded by the update-weight buffer size from arguments. When expert tensor parallelism is 1,
# experts are converted before gathering and each expert is converted by only one of the expert
# data-parallel replicas, then the converted payloads are exchanged with the expert gather helper.
import itertools
import re
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tqdm import tqdm

from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
from miles.backends.megatron_utils.named_weights import named_params_and_buffers
from miles.backends.megatron_utils.sglang import monkey_patch_torch_reductions
from miles.backends.megatron_utils.update_weight.expert_gather import ExpertGather
from miles.backends.megatron_utils.update_weight.hf_weight_iterator import (
    MegatronHfWeightIteratorBase,
    _iter_mm_tower_units,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.types import ParamInfo


@dataclass(frozen=True)
class _ExpertBatch:
    # [CLASS DOC] HELPER: Immutable batch of routed-expert parameters: the parameter metadata of the
    # batch, and the sequence of expert gather objects (possibly empty) to apply, in order, to the
    # converted units.
    param_infos: Sequence[ParamInfo]
    gathers: tuple[ExpertGather, ...]


class HfWeightIteratorDirect(MegatronHfWeightIteratorBase):
    # [CLASS DOC] Direct (non-bridge) iterator. The class-level forced placement turns off the
    # forced pipeline gather of the base class (tensor, expert-tensor and expert gathers always
    # happen; pipeline follows the requirement). Instance state: a flag saying experts are converted
    # before being gathered (true when expert tensor parallelism has size one), the precomputed
    # batches of non-expert parameter infos, and the list of expert batches.
    # Lower bound: TP/ETP/EP are always gathered; PP follows the requirement.
    forced_placement = WeightUpdatePlacement(gather_pp=False)

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the base constructor, then precomputes the batch plan (collective
        # operations may be involved through the metadata gather). Collects local parameter infos
        # split into non-expert and expert (pipeline-exchanged when the placement gathers pipeline)
        # and packs non-expert infos into size-bounded batches. Experts: when expert tensor
        # parallelism is larger than one, packs expert infos into batches (size scaled by expert
        # parallel size) with no gathers, since shards must be assembled before conversion. When it
        # is one and expert infos exist: asserts expert data-parallel state exists; partitions the
        # experts among the expert-data-parallel replicas (each replica owns a contiguous range of
        # local experts); asserts the combined tensor/data/context group size equals expert
        # parallel size times expert data parallel size; the gather groups are the pipeline group
        # (only if the placement gathers pipeline) then the combined group; packs each owner's
        # share into batches (size scaled by the combined size); and builds one expert batch per
        # round by taking, for each round, this replica's own batch (empty batches pad shorter
        # owners), with an expert gather object for each group of size greater than one.
        super().__init__(*args, **kwargs)
        parallel = get_parallel_state()
        self._convert_experts_before_gather = parallel.etp.size == 1
        non_expert_infos, expert_infos = _get_megatron_local_param_infos(
            self.args, self.model, gather_pp=self.placement.gather_pp
        )
        ep_size = parallel.ep.size
        self._non_expert_batches = _pack_param_infos_by_size(self.args, non_expert_infos)
        self._expert_batches = []
        if not self._convert_experts_before_gather:
            self._expert_batches = [
                _ExpertBatch(param_infos=infos, gathers=())
                for infos in _pack_param_infos_by_size(self.args, expert_infos, size_multiplier=ep_size)
            ]
        elif expert_infos:
            edp = parallel.edp
            assert edp is not None, "Expert data parallel state is required for MoE weight updates"
            owner_infos = _partition_expert_infos(
                expert_infos, num_local_experts=self.args.num_experts // ep_size, edp_size=edp.size
            )
            # At ETP1, TP x DP x CP is exactly EP x EDP. Borrow Megatron's
            # combined group to avoid repacking between EP and EDP gathers.
            combined = parallel.tp_dp_cp
            assert (
                combined is not None and combined.size == ep_size * edp.size
            ), "The combined TP/DP/CP group must cover all EP/EDP ranks"
            groups = ([parallel.pp] if self.placement.gather_pp else []) + [combined]
            # Pack each owner's share first so a round can quantize on every EDP
            # replica, even when the complete local expert set spans many batches.
            owner_batches = [
                _pack_param_infos_by_size(self.args, infos, size_multiplier=ep_size * edp.size)
                for infos in owner_infos
            ]
            self._expert_batches = [
                _ExpertBatch(
                    param_infos=batches[edp.rank],
                    gathers=tuple(ExpertGather(group=group.group) for group in groups if group.size > 1),
                )
                for batches in itertools.zip_longest(*owner_batches, fillvalue=())
            ]

    def _iter_hf_param_units(self, weights, *, materialize):
        # [FUNCTION DOC] Generator of units of Hugging Face tensors. Input: a mapping from
        # Megatron name to local tensor, and a materialize flag. Shows a progress bar (rank zero
        # only) over all batches. For each non-expert batch: materializes (load, gather) the
        # parameters and, when materializing, yields converted units; then frees them. For each
        # expert batch: with experts converted before gather, converts and gathers the batch and
        # yields its units when materializing; otherwise gathers full experts then converts and
        # yields. Collective work happens even when not materializing so all ranks stay in step.
        # Finally yields multimodal tower units (also when not materializing; delegated).
        rank = dist.get_rank()

        pbar = tqdm(
            total=len(self._non_expert_batches) + len(self._expert_batches),
            disable=rank != 0,
            desc="Update weights",
        )
        for param_infos in self._non_expert_batches:
            named_params = _materialize_non_expert_batch(
                self.args, param_infos, weights, gather_pp=self.placement.gather_pp
            )
            if materialize:
                yield from self._convert_to_hf_param_units(named_params)
            del named_params
            pbar.update(1)
        for batch in self._expert_batches:
            if self._convert_experts_before_gather:
                units = self._convert_and_gather_expert_batch(batch, weights)
                if materialize:
                    yield from units
                del units
            else:
                # ETP shards must form complete experts before conversion/quantization.
                named_params = _gather_megatron_expert_batch(
                    self.args, batch.param_infos, weights, gather_pp=self.placement.gather_pp
                )
                if materialize:
                    yield from self._convert_to_hf_param_units(named_params)
                del named_params
            pbar.update(1)
        pbar.close()
        yield from _iter_mm_tower_units(self.args, materialize=materialize)

    def _convert_and_gather_expert_batch(self, batch: _ExpertBatch, weights):
        # [FUNCTION DOC] HELPER: For one expert batch, converts on the rank that owns each parameter
        # (the source rank equals the current rank), moving it to the current GPU asynchronously,
        # converting to Hugging Face units, then applying each gather object in order (which
        # replaces the units with the units of all group members). Returns the final list of units.
        """Convert once per expert across EP/EDP, then gather HF weights and scales."""
        device = torch.device("cuda", torch.cuda.current_device())
        rank = dist.get_rank()
        # Sender placement is independent of ownership: non-senders also
        # quantize their assigned experts, once across all expert-DP replicas.
        local_params = (
            (info.name, weights[info.name].detach().to(device=device, non_blocking=True))
            for info in batch.param_infos
            if info.src_rank == rank
        )
        units = list(self._convert_to_hf_param_units(local_params))
        for gather in batch.gathers:
            units = gather(units, device=device)
        return units

    def _export_pp_local_lora(self, adapter):
        # [FUNCTION DOC] Implements the base hook for native (non-bridge) LoRA. Asserts there is no
        # adapter object (multi-LoRA needs bridge mode). By model: if the model name mentions Kimi
        # K3, returns the flattened chunks from that model's plugin exporter; if the custom
        # provider path mentions Inkling, returns that plugin's named export; otherwise raises a
        # not-implemented error naming the model. Plugin imports are lazy.
        assert adapter is None, "multi-LoRA export requires --megatron-to-hf-mode bridge"
        # TODO: will rewrite in native lora refactor
        if "kimi_k3" in self.model_name.lower():
            from miles_plugins.models.kimi_k3.lora import export_kimi_k3_lora_hf_chunks

            return [named_tensor for chunk in export_kimi_k3_lora_hf_chunks(self.model) for named_tensor in chunk]
        if "inkling" in (self.args.custom_model_provider_path or ""):
            from miles_plugins.models.inkling.lora import export_inkling_lora_hf_named

            return export_inkling_lora_hf_named(self.model)
        raise NotImplementedError(f"Raw LoRA export is not implemented for model {self.model_name!r}")

    def _convert_to_hf_param_units(self, named_params: Sequence[tuple[str, torch.Tensor]]):
        # [FUNCTION DOC] HELPER: Generator that converts each gathered Megatron parameter to a
        # list of Hugging Face name and tensor pairs (one unit, possibly several tensors such as a
        # quantized weight with scales) using the conversion function with the model name,
        # quantization config and packed weight names.
        for name, param in named_params:
            yield list(
                convert_to_hf(
                    self.args, self.model_name, name, param, self.quantization_config, self.packed_weight_basenames
                )
            )


def _load_or_allocate_params(param_infos: Sequence[ParamInfo], megatron_local_weights) -> list[torch.Tensor]:
    # [FUNCTION DOC] HELPER: For each parameter info returns a tensor on the current GPU: the
    # source rank wraps the weight from the given mapping (moved to the GPU asynchronously) as a
    # non-trainable parameter; other ranks allocate an uninitialized buffer of the recorded shape
    # and element type. Synchronizes the GPU before returning the list.
    """Owners load from the weight source; other ranks allocate receive buffers."""
    params = []
    for info in param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()
    return params


def _broadcast_across_pp(param_infos: Sequence[ParamInfo], params: Sequence[torch.Tensor]) -> None:
    # [FUNCTION DOC] HELPER: Collective over the pipeline group; no-op for size one. For each
    # parameter whose source rank belongs to the pipeline group, starts an asynchronous broadcast
    # from that rank in place, then waits for all broadcasts. Parameters from sources outside the
    # group are skipped.
    pp = get_parallel_state().pp
    if pp.size == 1:
        return
    pp_ranks = dist.get_process_group_ranks(pp.group)
    handles = []
    for info, param in zip(param_infos, params, strict=True):
        if info.src_rank in pp_ranks:
            handles.append(dist.broadcast(param, src=info.src_rank, group=pp.group, async_op=True))
    for handle in handles:
        handle.wait()


def _set_tp_attrs(param_infos: Sequence[ParamInfo], params: Sequence[torch.Tensor]) -> None:
    # [FUNCTION DOC] HELPER: Copies the recorded parallelism attributes (tensor-parallel flag,
    # partition dimension, partition stride, parallel mode) of each info onto the matching tensor
    # as attributes, so later gathers know how each tensor is sharded. Mutates the tensors.
    for info, param in zip(param_infos, params, strict=True):
        for key, value in info.attrs.items():
            setattr(param, key, value)


def _materialize_non_expert_batch(
    args: Namespace,
    param_infos: Sequence[ParamInfo],
    megatron_local_weights,
    *,
    gather_pp: bool,
) -> list[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Materializes one batch of non-expert parameters fully gathered across
    # tensor parallelism. Applies the torch reduction monkey patch (global side effect), loads or
    # allocates the tensors, broadcasts across pipeline stages when gather_pp (keyword-only) is true,
    # sets the parallelism attributes, runs the asynchronous tensor-parallel gather, and returns
    # name and tensor pairs in batch order.
    """Load -> PP broadcast (when gather_pp) -> TP all_gather."""
    monkey_patch_torch_reductions()
    params = _load_or_allocate_params(param_infos, megatron_local_weights)
    if gather_pp:
        _broadcast_across_pp(param_infos, params)
    _set_tp_attrs(param_infos, params)
    gathered = all_gather_params_async(args, list(zip(param_infos, params, strict=True)))
    return [(info.name, param) for info, param in zip(param_infos, gathered, strict=True)]


def _gather_megatron_expert_batch(
    args: Namespace,
    param_infos: Sequence[ParamInfo],
    megatron_local_weights,
    *,
    gather_pp: bool,
) -> list[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Like the non-expert materialization, but also gathers across expert
    # parallelism after the expert-tensor-parallel gather. With expert parallel size one returns
    # the expert-tensor-gathered tensors. Otherwise: all-gathers each rank's parameter names as
    # Python objects and asserts every rank has the same count (assertion error describing the
    # asymmetry); then for each parameter allocates one buffer per expert rank and starts an
    # asynchronous all-gather, pairing each buffer with the corresponding rank's name; waits for
    # all; returns the pairs ordered by expert rank, then by batch order. Applies the torch
    # reduction patch.
    """Load -> PP broadcast (when gather_pp) -> ETP all_gather -> EP all_gather.

    Expert metadata is EP-local; the full expert set is materialized by a
    symmetric EP all_gather with a name exchange.
    """
    monkey_patch_torch_reductions()
    params = _load_or_allocate_params(param_infos, megatron_local_weights)
    if gather_pp:
        _broadcast_across_pp(param_infos, params)
    _set_tp_attrs(param_infos, params)
    etp_gathered = all_gather_params_async(args, list(zip(param_infos, params, strict=True)))

    ep = get_parallel_state().ep
    if ep.size == 1:
        return [(info.name, param) for info, param in zip(param_infos, etp_gathered, strict=True)]

    names = [info.name for info in param_infos]
    all_names: list = [None] * ep.size
    dist.all_gather_object(all_names, names, group=ep.group)
    for ep_names in all_names:
        assert len(ep_names) == len(
            names
        ), f"EP-asymmetric expert batch: {len(names)} params locally vs {len(ep_names)} on a peer rank"

    all_gathered: list[list[tuple[str, torch.Tensor]]] = [[] for _ in range(ep.size)]
    handles = []
    for i, param in enumerate(etp_gathered):
        buffers = [torch.empty_like(param, device=torch.cuda.current_device()) for _ in range(ep.size)]
        handles.append(dist.all_gather(buffers, param, group=ep.group, async_op=True))
        for ep_rank, ep_names in enumerate(all_names):
            all_gathered[ep_rank].append((ep_names[i], buffers[ep_rank]))
    for handle in handles:
        handle.wait()

    return [named for per_rank in all_gathered for named in per_rank]


def _pack_param_infos_by_size(
    args: Namespace, param_infos: list[ParamInfo], *, size_multiplier: int = 1
) -> list[list[ParamInfo]]:
    # [FUNCTION DOC] HELPER: Greedy in-order packing. Each info's size is its full (unsharded)
    # size times the multiplier (keyword-only, default one). A new batch starts when adding an info
    # would exceed the update-weight buffer size from arguments and the current batch is nonempty,
    # so a single oversized info gets its own batch (the bound is then exceeded). Returns the list
    # of nonempty batches; empty input gives an empty list.
    """Greedy size packing into gather batches ≤ update_weight_buffer_size."""
    batches: list[list[ParamInfo]] = [[]]
    buffer_size = 0
    for info in param_infos:
        size = _get_param_full_size(info) * size_multiplier
        if buffer_size + size > args.update_weight_buffer_size and batches[-1]:
            batches.append([])
            buffer_size = 0
        batches[-1].append(info)
        buffer_size += size
    return [batch for batch in batches if batch]


def _partition_expert_infos(
    param_infos: Sequence[ParamInfo], *, num_local_experts: int, edp_size: int
) -> list[list[ParamInfo]]:
    # [FUNCTION DOC] HELPER: Splits expert parameter infos among expert-data-parallel replicas.
    # Local experts (count given) are divided into contiguous ranges, the first (count modulo
    # replicas) replicas getting one extra. Each info's expert number is the integer at the end of
    # its name (trailing "weight" plus digits, the global expert id) taken modulo the local expert
    # count; the info goes to the replica owning that local expert, so an expert's FC1 and FC2
    # stay together. Raises a value error if a name has no trailing expert number. Returns one list
    # of infos per replica, preserving input order.
    """Assign contiguous local-expert ranges to EDP replicas, keeping FC1/FC2 together."""
    count, remainder = divmod(num_local_experts, edp_size)
    owners = [owner for owner in range(edp_size) for _ in range(count + (owner < remainder))]
    partitions: list[list[ParamInfo]] = [[] for _ in range(edp_size)]
    for info in param_infos:
        # The direct converter uses TE grouped weights with global expert IDs.
        match = re.search(r"\.weight(\d+)$", info.name)
        if match is None:
            raise ValueError(f"Cannot identify the routed expert in {info.name!r}")
        local_expert = int(match.group(1)) % num_local_experts
        partitions[owners[local_expert]].append(info)
    return partitions


def _get_param_full_size(info: ParamInfo) -> int:
    # [FUNCTION DOC] HELPER: Estimated byte size of the parameter after tensor-parallel gather: the
    # local byte size times the expert-tensor-parallel size for routed experts, else the regular
    # tensor-parallel size. (Assumes the parameter is sharded; unsharded ones are over-counted.)
    if is_routed_expert_param(info.name):
        tp_size = get_parallel_state().etp.size
    else:
        tp_size = get_parallel_state().tp.size
    return info.size * tp_size


def _get_megatron_local_param_infos(
    args: Namespace, model: Sequence[torch.nn.Module], *, gather_pp: bool
) -> tuple[list[ParamInfo], list[ParamInfo]]:
    # [FUNCTION DOC] HELPER: Builds metadata records for all local parameters and expert-bias
    # buffers (via the named enumeration with global names), skipping LoRA adapter parameters.
    # Each record holds name, element type, shape, parallelism attributes (tensor-parallel flag
    # default false, partition dimension default minus one, stride default one, parallel mode
    # default none), byte size, and the current global rank as the source. When gather_pp is true
    # (keyword-only) and the pipeline size exceeds one, all-gathers the records across the pipeline
    # group; remote records are merged in: duplicate names (only expected for MTP layers; asserts
    # MTP layers are configured) keep the record from the lowest source rank. Records are sorted by
    # name and split into non-expert and routed-expert lists. When gather_pp, checks non-expert
    # metadata consistency across all ranks. Returns (non-expert records, expert records).
    """Collect param metadata, exchanged across PP when gather_pp.

    Returns (non_expert_infos, expert_infos); expert infos stay EP-local.
    """
    pp_size = get_parallel_state().pp.size

    from ..lora.utils import _is_adapter_param_name

    param_infos: dict[str, ParamInfo] = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        if _is_adapter_param_name(name):
            continue
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if gather_pp and pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=get_parallel_state().pp.group
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    # Duplicates across PP only exist for MTP virtual-PP layers.
                    assert args.mtp_num_layers is not None
                    if param_infos[name].src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    infos = sorted(param_infos.values(), key=lambda info: info.name)
    non_expert_infos = [info for info in infos if not is_routed_expert_param(info.name)]
    expert_infos = [info for info in infos if is_routed_expert_param(info.name)]

    if gather_pp:
        _check_param_infos_consistent(non_expert_infos)

    return non_expert_infos, expert_infos


def _check_param_infos_consistent(param_infos: list[ParamInfo]) -> None:
    # [FUNCTION DOC] HELPER: Collective over the whole world (via the gloo group): all-gathers every
    # rank's list of records as Python objects, and for each local index asserts that all ranks'
    # records have the same name, shape and element type (assertion errors with the mismatching
    # values). Raises an index error if some rank has a shorter list.
    """Every rank must hold identical non-expert metadata once PP is gathered."""
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(obj=param_infos, object_list=all_param_info_list, group=get_gloo_group())
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"


def _gather_with_stride(
    param_partitions: list[torch.Tensor], partition_dim: int, partition_stride: int
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Reassembles a full tensor from per-rank partitions along the partition
    # dimension. Stride one: plain concatenation. Stride s greater than one (interleaved sharding,
    # for example gate and up halves of a gated activation's first projection): each rank's
    # partition is split into s chunks along the dimension, and the chunks are concatenated in the
    # order chunk 0 of every rank, then chunk 1 of every rank, and so on.
    """Gather partitions respecting partition_stride (strided/interleaved TP sharding)."""
    if partition_stride == 1:
        return torch.cat(param_partitions, dim=partition_dim)
    # Interleaved (strided) partitioning, e.g. linear_fc1.weight under GLU/SwiGLU
    chunks_per_rank = [p.chunk(partition_stride, dim=partition_dim) for p in param_partitions]
    interleaved = [chunks_per_rank[r][s] for s in range(partition_stride) for r in range(len(param_partitions))]
    return torch.cat(interleaved, dim=partition_dim)


def is_routed_expert_param(name: str) -> bool:
    # [FUNCTION DOC] Pure string test: true when the name contains the experts path segment and does
    # not contain the shared-experts segment. See the docstring for why shared-expert inner lists
    # are excluded.
    """Whether a Megatron param name belongs to the routed (expert-parallel) experts.

    Routed experts live under ".experts.", but shared experts may nest an inner
    ModuleList (e.g. Inkling's "mlp.shared_experts.experts.N.") whose params are
    regular-TP sharded and EP-replicated, so they must not match.
    """
    return ".experts." in name and ".shared_experts." not in name


def _is_unmarked_grouped_expert_weight(name: str, param: torch.nn.Parameter) -> bool:
    # [FUNCTION DOC] HELPER: True when the parameter is a routed-expert first or second linear
    # weight that is not flagged tensor-parallel and the expert-tensor-parallel size exceeds one;
    # such parameters are actually sharded and must still be gathered. The first argument is the
    # parameter name, the second the parameter object (its tensor-parallel flag is read).
    """TEGroupedLinear never marks its per-expert weight0..weightN, so Megatron fills in
    the defaults (tensor_model_parallel=False, partition_dim=-1) and the tensor claims to
    be unsharded. It is expert-TP sharded whenever etp > 1, so the gather must still run.
    """
    return (
        is_routed_expert_param(name)
        and ("linear_fc1.weight" in name or "linear_fc2.weight" in name)
        and not param.tensor_model_parallel
        and get_parallel_state().etp.size > 1
    )


def _check_and_fix_partition(args: Namespace, name: str, partition_stride: int, partition_dim: int) -> tuple[int, int]:
    # [FUNCTION DOC] HELPER: Returns the corrected (partition stride, partition dimension) for a
    # parameter name. If the name is a first linear weight and the gated-activation flag is set in
    # arguments: stride becomes two and a negative dimension becomes zero. If it is a second linear
    # weight: asserts stride one and a nonpositive dimension becomes one. Otherwise asserts stride
    # one and leaves the dimension unchanged. Assertion errors carry the name and stride.
    # KNOWN ISSUE: the arguments object is only used for the gated-activation flag; a first linear
    # weight without that flag falls to the final branch and asserts stride one, though the
    # docstring says the stride assertion was removed.
    """Validate partition_stride values for known parameter patterns.

    After Megatron-LM PR #2708, linear_fc1 correctly reports partition_stride=2
    (GLU/SwiGLU interleaved [gate, up]), so assert partition_stride==2 is removed.
    But TEGroupedLinear still does not set partition_stride/partition_dim correctly for grouped moe gemm
    """
    if "linear_fc1.weight" in name and args.swiglu:
        partition_stride = 2
        if partition_dim < 0:
            partition_dim = 0
    elif "linear_fc2.weight" in name:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
        if partition_dim <= 0:
            partition_dim = 1
    else:
        assert partition_stride == 1, f"Expected partition_stride=1 for {name}, got {partition_stride}"
    return partition_stride, partition_dim


def all_gather_params_async(
    args: Namespace,
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Gathers many parameters across tensor parallelism with overlapped
    # communication; input is a list of (info, tensor) pairs, output is the list of full tensors
    # in the same order. Phase 1: per pair, expert-bias entries and parameters marked as
    # duplicated, or not tensor-parallel and not an unmarked grouped expert weight, are passed
    # through (non-bias ones as their raw data). Others use the expert-tensor group for routed
    # experts, else the tensor group; if that group has size at most one the data is passed
    # through; otherwise allocates one buffer per rank and starts an asynchronous all-gather.
    # Phase 2: waits for all handles. Phase 3: for gathered ones, fixes up stride and dimension
    # via the partition check and concatenates (with interleaving when stride exceeds one).
    # Caution: the expert-bias passthrough keeps the parameter object itself while others keep raw
    # data.
    """
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group (skip expert_bias/non-TP/duplicated).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None, None))
            handles.append(None)
        elif getattr(param, "parallel_mode", None) == "duplicated" or (
            not param.tensor_model_parallel and not _is_unmarked_grouped_expert_weight(info.name, param)
        ):
            gather_tasks.append((info, param.data, None, None, None, None))
            handles.append(None)
        else:
            # Start async all_gather
            if is_routed_expert_param(info.name):
                tp_size = get_parallel_state().etp.size
                tp_group = get_parallel_state().etp.group
            else:
                tp_size = get_parallel_state().tp.size
                tp_group = get_parallel_state().tp.group

            if tp_size <= 1:
                gather_tasks.append((info, param.data, None, None, None, None))
                handles.append(None)
                continue

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim, param.partition_stride))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        if handle is not None:
            handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim, partition_stride in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            partition_stride, partition_dim = _check_and_fix_partition(
                args, info.name, partition_stride, partition_dim
            )
            param = _gather_with_stride(param_partitions, partition_dim, partition_stride)

        gathered_params.append(param)

    return gathered_params
