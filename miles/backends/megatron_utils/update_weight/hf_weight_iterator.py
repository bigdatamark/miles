# [MODULE DOC] Shared base class and factory for the Megatron implementations of the
# backend-neutral Hugging Face weight iterator. Contains: the base class (forces gathering across
# pipeline stages, derives packed-weight names for compressed-tensors quantization, handles
# speculative-decoding selector, and turns LoRA adapter exports into units); the factory choosing
# the direct or bridge implementation by conversion mode; a reader for packed-weight names from the
# checkpoint index; a pipeline-parallel adapter gather; and a hook yielding multimodal tower units
# for the Inkling multimodal provider.
"""Megatron implementations' shared base and factory for the backend-neutral
HF weight iterator API."""

import json
import logging
import math
import os
from abc import abstractmethod
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from megatron.core.utils import unwrap_model

from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase, WeightUpdatePlacement
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    iter_checkpoint_tower_units,
)
from miles.utils.lora.utils import is_lora_weight_name

logger = logging.getLogger(__name__)


class MegatronHfWeightIteratorBase(HfWeightIteratorBase):
    # [CLASS DOC] Base for Megatron weight iterators. The class-level forced placement (the minimum
    # placement requirement every subclass starts from) says pipeline-parallel gathering is on;
    # the direct subclass overrides it. Subclasses must implement the hook that exports the
    # pipeline-local LoRA adapter tensors, and the parent class' hooks for parameter units.
    forced_placement = WeightUpdatePlacement(gather_pp=True)

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Passes all arguments to the parent constructor. Then sets the set of
        # packed weight base names, read from the checkpoint index, only when a quantization config
        # exists whose method is compressed-tensors (else none). Determines whether the trainer
        # model has MTP layers (reading the first unwrapped chunk's config); if the rollout uses a
        # speculative algorithm but the trainer has no MTP layers, sets the weight update selector
        # to the target model only.
        super().__init__(*args, **kwargs)
        self.packed_weight_basenames = (
            get_packed_weight_basenames(self.args.hf_checkpoint)
            if self.quantization_config is not None
            and self.quantization_config["quant_method"] == "compressed-tensors"
            else None
        )
        trainer_has_mtp = bool(unwrap_model(self.model)[0].config.mtp_num_layers)
        if self.args.sglang_speculative_algorithm and not trainer_has_mtp:
            self.weight_update_selector = "target"

    def _hf_atomic_update_groups(self):
        # [FUNCTION DOC] HELPER: Returns the groups of Hugging Face weights that must be updated
        # atomically for this model name, given the query LoRA rank from arguments.
        return get_hf_atomic_update_groups(self.model_name, q_lora_rank=self.args.q_lora_rank)

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        # [FUNCTION DOC] HELPER: Generator turning an adapter export into units. Calls the
        # backend hook to get the pipeline-local list of tensors; if the resolved placement asks
        # for pipeline gathering, gathers the complete adapter onto every pipeline rank (this
        # happens even when not materializing, because collectives must run on all ranks). When
        # materialize is false, stops without yielding. Otherwise raises a runtime error if the
        # export is empty or if none of the names look like LoRA A/B weights; then yields each
        # tensor as its own single-item unit, consuming the list from the front.
        """Both megatron exporters are PP-local after gathering TP/EP; the PP
        gather runs only where the resolved placement asks for it."""
        named_tensors = self._export_pp_local_lora(adapter)
        # TODO: the PP-local branch is unreachable until actor.py lifts its bridge-only guard
        # for distributed LoRA; add an e2e for native-LoRA disaggregate when it does
        if self.placement.gather_pp:
            named_tensors = _gather_pp_full_adapter(named_tensors)
        if not materialize:
            return
        if not named_tensors:
            raise RuntimeError("LoRA weight sync failed: the adapter export produced zero tensors")
        if not any(is_lora_weight_name(name) for name, _ in named_tensors):
            raise RuntimeError("LoRA weight sync failed: the adapter export contains no lora_A/lora_B names.")
        while named_tensors:
            hf_name, tensor = named_tensors.pop(0)
            yield [(hf_name, tensor)]

    @abstractmethod
    def _export_pp_local_lora(self, adapter) -> list[tuple[str, torch.Tensor]]:
        # [FUNCTION DOC] Abstract hook implemented by subclasses: returns the adapter's tensors with
        # Hugging Face names, already gathered over tensor and expert parallelism but still local
        # to this pipeline rank, as a list of name and tensor pairs.
        """Backend hook: the adapter's HF-named tensors, TP/EP gathered, PP-local."""


def get_hf_weight_iterator(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    required_placement: WeightUpdatePlacement,
    model_name: str,
    quantization_config: dict | None,
) -> HfWeightIteratorBase:
    # [FUNCTION DOC] Factory. Picks the direct implementation for conversion mode "raw" or the
    # bridge implementation for "bridge" (any other mode raises a key error), imported lazily to
    # avoid circular imports, and calls its build method with the arguments, model chunks, the
    # required placement, model name and quantization config (all keyword-only except the first
    # two). Returns the built iterator.
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge
    from miles.backends.megatron_utils.update_weight.hf_weight_iterator_direct import HfWeightIteratorDirect

    cls = {
        "raw": HfWeightIteratorDirect,
        "bridge": HfWeightIteratorBridge,
    }[args.megatron_to_hf_mode]

    return cls.build(
        args,
        model,
        required_placement=required_placement,
        model_name=model_name,
        quantization_config=quantization_config,
    )


def get_packed_weight_basenames(hf_checkpoint: str) -> set[str]:
    # [FUNCTION DOC] Reads the safetensors index JSON in the checkpoint directory (file I/O; raises
    # if missing), takes its weight-name map, and returns the set of names ending with the packed
    # weight suffix with that suffix removed.
    """Base names the checkpoint stores as compressed-tensors `weight_packed`; the quantizer
    re-quantizes exactly these, since the published `ignore` list is written for loaders and
    leaves out BF16 weights such as routers, residual projections and the vision tower."""
    with open(os.path.join(hf_checkpoint, "model.safetensors.index.json")) as index_file:
        names = json.load(index_file)["weight_map"]
    return {n.removesuffix(".weight_packed") for n in names if n.endswith(".weight_packed")}


def _gather_pp_full_adapter(
    hf_named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    # [FUNCTION DOC] HELPER: Makes every pipeline rank hold the complete adapter. With a pipeline
    # size of one returns a plain list copy. Otherwise: all-gathers (name, shape, element type)
    # metadata of each rank's tensors as Python objects; then for each owner rank in order and for
    # each element type in its metadata, allocates a flat CUDA buffer on the current device of the
    # total element count, the owner fills it from its local tensors, a broadcast from the owner
    # runs over the pipeline group, and every rank slices the buffer back into shaped views stored
    # by name. Returns a list of name and tensor pairs sorted by name (views share the broadcast
    # buffers). Collective: all pipeline ranks must call it together.
    """Gather the complete adapter onto every PP rank: exchange metadata, then
    one flat broadcast per (owner, dtype)."""
    pp = get_parallel_state().pp
    if pp.size == 1:
        return list(hf_named_tensors)
    global_ranks = dist.get_process_group_ranks(pp.group)
    device = torch.cuda.current_device()

    local_meta = [(n, tuple(t.shape), t.dtype) for n, t in hf_named_tensors]
    all_meta: list = [None] * pp.size
    dist.all_gather_object(all_meta, local_meta, group=pp.group)

    local_by_name = {n: t for n, t in hf_named_tensors}
    merged: dict[str, torch.Tensor] = {}
    for src, meta in enumerate(all_meta):
        by_dtype: dict = {}
        for n, shape, dtype in meta:
            by_dtype.setdefault(dtype, []).append((n, shape))
        for dtype, entries in by_dtype.items():
            numel = sum(math.prod(shape) for _, shape in entries)
            flat = torch.empty(numel, dtype=dtype, device=device)
            if src == pp.rank:
                off = 0
                for n, shape in entries:
                    k = math.prod(shape)
                    flat[off : off + k].copy_(local_by_name[n].reshape(-1))
                    off += k
            dist.broadcast(flat, src=global_ranks[src], group=pp.group)
            off = 0
            for n, shape in entries:
                k = math.prod(shape)
                merged[n] = flat[off : off + k].view(shape)
                off += k
    return sorted(merged.items())


def _iter_mm_tower_units(args, *, materialize):
    # [FUNCTION DOC] HELPER: Generator that yields nothing unless the custom model provider path
    # names the Inkling multimodal provider; in that case yields the multimodal checkpoint tower
    # units read from the Hugging Face checkpoint (honoring the materialize flag).
    if "inkling_mm_model_provider" not in (args.custom_model_provider_path or ""):
        return
    yield from iter_checkpoint_tower_units(args.hf_checkpoint, materialize=materialize)
