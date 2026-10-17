# [MODULE DOC] Hugging Face weight iterator for the torchtitan backend. Streams the trainer's
# weights as Hugging Face-named tensors in the engine's dtype: gathers sharded tensors to full
# tensors, converts names with the model's state-dict adapter, handles expert tensors one at a time
# (all ranks agree on the list of expert keys), and, when pipeline stages hold different tensors,
# broadcasts tensors from their owning stage to the ranks that need them. Also reads the on-disk
# checkpoint's tensor dtypes so exported floating-point tensors can be cast to match.
import functools
import glob
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass

import safetensors
import torch
import torch.distributed as dist

from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    iter_checkpoint_tower_units,
)
from miles.utils.hf_utils.config import load_hf_config


class TitanHfWeightIterator(HfWeightIteratorBase):
    # [CLASS DOC] Weight iterator for torchtitan. No forced placement (the requirement from the
    # caller decides whether to gather across pipeline stages). State: a mapping from Hugging Face
    # tensor name to the dtype stored in the checkpoint, and the optional query LoRA rank from the
    # checkpoint config (none when absent or zero). LoRA adapters are unsupported.
    forced_placement = None

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Runs the base constructor, reads the dtypes of all tensors in the
        # checkpoint at the Hugging Face checkpoint path (file I/O), and reads the query LoRA rank
        # from the checkpoint's config (a falsy value becomes none).
        super().__init__(*args, **kwargs)
        self._engine_dtypes = _checkpoint_dtypes(self.args.hf_checkpoint)
        self._q_lora_rank = getattr(load_hf_config(self.args.hf_checkpoint), "q_lora_rank", None) or None

    def _iter_hf_param_units(self, weights, *, materialize):
        # [FUNCTION DOC] Generator of units. The given weights mapping is ignored; weights come from
        # the model (the iterator's model attribute is the trainer). For each Hugging Face tensor
        # streamed (gathering across pipeline stages only if the placement asks), yields a
        # single-tensor unit cast to the checkpoint dtype when materializing; when not materializing
        # the stream is still consumed so collectives run on every rank. Finally yields the
        # checkpoint tower units.
        for name, tensor in hf_weights(self.model, complete_across_pp=self.placement.gather_pp):
            if materialize:
                yield [(name, self._to_engine_dtype(name, tensor))]
        yield from iter_checkpoint_tower_units(self.args.hf_checkpoint, materialize=materialize)

    def _to_engine_dtype(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Casts a floating-point tensor to the dtype the checkpoint stores for
        # that name; returns the tensor unchanged if the name is unknown, the dtype already matches,
        # or the tensor is not floating point.
        target = self._engine_dtypes.get(name)
        if target is None or tensor.dtype == target or not tensor.is_floating_point():
            return tensor
        return tensor.to(target)

    def _hf_atomic_update_groups(self):
        # [FUNCTION DOC] HELPER: Returns the groups of weights that must be updated atomically for
        # this model name, using the query LoRA rank read from the checkpoint config.
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self._q_lora_rank)

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize):
        # [FUNCTION DOC] Always raises a not-implemented error: torchtitan has no LoRA support.
        # KNOWN ISSUE: the signature takes an extra leading adapter-name parameter that the Megatron
        # backend's version of this hook does not have.
        raise NotImplementedError("the torchtitan backend has no LoRA")


def hf_weights(trainer, *, complete_across_pp: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] Generator of (Hugging Face name, full tensor) pairs for the trainer's weights.
    # If the first model part's first parameter is on the CPU (offloaded), moves all parts to the
    # GPU first, and always moves them back to the CPU and empties the CUDA cache when the
    # generator finishes or is closed. With complete-across-pipeline true (default) every rank
    # receives all tensors; otherwise each rank receives only its own pipeline stage's tensors.
    # Collective: all ranks must iterate in lockstep.
    offloaded = next(trainer.model_parts[0].parameters()).device.type == "cpu"
    if offloaded:
        for part in trainer.model_parts:
            part.cuda()
    try:
        yield from _hf_weights_on_device(trainer, complete_across_pp=complete_across_pp)
    finally:
        if offloaded:
            for part in trainer.model_parts:
                part.cpu()
            torch.cuda.empty_cache()


@functools.cache
def _stage_process_groups(stage_ranks: tuple[tuple[int, ...], ...]) -> tuple:
    # [FUNCTION DOC] HELPER: Cached (so each distinct layout creates groups only once per process)
    # creation of one new process group per tuple of ranks; every rank must call it identically
    # because group creation is collective. Returns the tuple of groups in the same order.
    return tuple(dist.new_group(list(ranks)) for ranks in stage_ranks)


_GROUPED_EXPERTS = "moe.routed_experts.inner_experts"


def _hf_weights_on_device(trainer, *, complete_across_pp: bool) -> Iterator[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Generator doing the real work with weights already on the GPU. Takes
    # the checkpointer's state-dict adapter, or builds one from the model spec, model config and
    # assets path. Merges the state dictionaries of all model parts and builds the stage layout.
    # Dense (non-grouped-expert) tensors are converted by the adapter to Hugging Face names in one
    # call and streamed. Grouped expert tensors (keys containing the grouped experts marker) are
    # handled one key at a time: every rank all-gathers its sorted list of such keys, and for each
    # key in the sorted union, converts the fully gathered tensor if the rank has the key, else
    # contributes nothing, and streams the result. This keeps memory bounded and keeps ranks in
    # lockstep.
    sd_adapter = getattr(trainer.checkpointer, "sd_adapter", None)
    if sd_adapter is None:
        sd_adapter = trainer.config.model_spec.state_dict_adapter(trainer.model_config, trainer.config.hf_assets_path)
    state = {k: v for part in trainer.model_parts for k, v in part.state_dict().items()}
    layout = _StageLayout.from_trainer(trainer, complete_across_pp=complete_across_pp)

    dense = {k: v for k, v in state.items() if _GROUPED_EXPERTS not in k}
    yield from layout.stream(sd_adapter.to_hf(dense))

    mine = sorted(k for k in state if _GROUPED_EXPERTS in k)
    everyone: list = [None] * layout.world
    dist.all_gather_object(everyone, mine)
    for key in sorted(set().union(*everyone)):
        yield from layout.stream(sd_adapter.to_hf({key: gather_full_param(state[key])}) if key in state else {})


@dataclass(frozen=True)
class _StageLayout:
    # [CLASS DOC] HELPER: Immutable description of who needs which tensors. Fields: world size; this
    # rank; the device for receive buffers; the set of ranks (the audience) that should receive
    # tensors (everyone, or just this rank's pipeline stage); and the process group to broadcast
    # over (none means the default group).
    world: int
    my_rank: int
    device: torch.device
    audience: frozenset[int]
    broadcast_group: dist.ProcessGroup | None

    @classmethod
    def from_trainer(cls, trainer, *, complete_across_pp: bool) -> "_StageLayout":
        # [FUNCTION DOC] Class method. Collective: all-gathers each rank's pipeline stage id (zero
        # when pipeline parallelism is off) to group ranks by stage. If complete-across-pipeline,
        # the audience is every rank and the broadcast group is the default one. Otherwise creates
        # (cached) one process group per stage, in stage order, and uses this rank's stage ranks as
        # the audience and the matching group. Uses the trainer's device.
        world = dist.get_world_size()
        my_rank = dist.get_rank()
        stage_of: list = [None] * world
        pp_mesh = trainer.parallel_dims.get_optional_mesh("pp")
        my_stage_id = dist.get_rank(group=pp_mesh.get_group()) if pp_mesh is not None else 0
        dist.all_gather_object(stage_of, my_stage_id)
        stage_groups: dict[int, list[int]] = {}
        for rank, stage in enumerate(stage_of):
            stage_groups.setdefault(stage, []).append(rank)
        my_stage = stage_of[my_rank]
        if complete_across_pp:
            audience, broadcast_group = list(range(world)), None
        else:
            groups = _stage_process_groups(tuple(tuple(ranks) for _, ranks in sorted(stage_groups.items())))
            audience, broadcast_group = stage_groups[my_stage], groups[sorted(stage_groups).index(my_stage)]
        return cls(
            world=world,
            my_rank=my_rank,
            device=trainer.device,
            audience=frozenset(audience),
            broadcast_group=broadcast_group,
        )

    def stream(self, local: dict[str, torch.Tensor]) -> Iterator[tuple[str, torch.Tensor]]:
        # [FUNCTION DOC] Generator yielding (name, full tensor) for a rank-local dictionary of
        # possibly sharded tensors. Collective. Step 1: all-gathers every rank's name-to-(shape,
        # dtype) metadata. If every rank holds exactly the same names, yields this rank's tensors
        # in sorted name order, each gathered to a full tensor. Otherwise (pipeline stages differ):
        # records which ranks own each name and requires all owners to agree on shape and dtype
        # (runtime error otherwise); considers names sorted, keeping only those owned by at least
        # one rank in the audience; for each, the first owner in the audience is the broadcast
        # source; owners gather their full tensor (made contiguous), others allocate an empty buffer
        # of the recorded shape and dtype on the device; broadcasts from the source over the
        # broadcast group (the source is given as a global rank); yields the tensor and drops the
        # reference before the next one.
        local_meta = {name: (tuple(t.shape), str(t.dtype)) for name, t in local.items()}
        gathered: list = [None] * self.world
        dist.all_gather_object(gathered, local_meta)

        if all(meta.keys() == local_meta.keys() for meta in gathered):
            for name in sorted(local):
                yield name, gather_full_param(local[name])
            return

        owners: dict[str, list[int]] = {}
        specs: dict[str, tuple] = {}
        for rank, meta in enumerate(gathered):
            for name, (shape, dtype) in meta.items():
                owners.setdefault(name, []).append(rank)
                if specs.setdefault(name, (shape, dtype)) != (shape, dtype):
                    raise RuntimeError(f"ranks disagree on the shape/dtype of {name}")

        names = [name for name in sorted(owners) if self.audience.intersection(owners[name])]
        for name in names:
            shape, dtype = specs[name]
            holders = [rank for rank in owners[name] if rank in self.audience]
            if self.my_rank in holders:
                tensor = gather_full_param(local[name]).contiguous()
            else:
                tensor = torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]), device=self.device)
            dist.broadcast(tensor, src=holders[0], group=self.broadcast_group)
            yield name, tensor
            del tensor


_SAFETENSORS_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _checkpoint_dtypes(hf_checkpoint: str) -> dict[str, torch.dtype]:
    # [FUNCTION DOC] HELPER: Returns a mapping from tensor name to dtype for a checkpoint directory.
    # Shard files come from the weight map in the safetensors index if present, else from all
    # safetensors files in the directory. Opens each shard and reads only metadata for each tensor;
    # names whose stored type is not one of bfloat16, float16, float32 or float64 are omitted.
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = sorted(os.path.basename(p) for p in glob.glob(os.path.join(hf_checkpoint, "*.safetensors")))

    dtypes: dict[str, torch.dtype] = {}
    for shard in shards:
        with safetensors.safe_open(os.path.join(hf_checkpoint, shard), framework="pt") as handle:
            for name in handle.keys():
                dtype = _SAFETENSORS_DTYPES.get(handle.get_slice(name).get_dtype())
                if dtype is not None:
                    dtypes[name] = dtype
    return dtypes
