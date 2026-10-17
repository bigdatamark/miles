# [PACKAGE DOC]
# Streaming of training-side weights as HF-named tensors for weight updates. This package file holds
# the backend-neutral abstract API: the placement descriptor (which parallel dims are gathered), the
# join rule between required and forced placements, and the abstract iterator base class with the
# shared bucketing pipeline; concrete backends implement the per-parameter and per-adapter unit
# generators. Siblings: bucketing (atomic grouping and size-bounded packing), atomic_groups (which HF
# weights must be loaded together per model family), checkpoint_towers (multimodal tower weights read
# straight from the HF checkpoint).
"""Backend-neutral API for streaming training-side weights as HF-named tensors."""

import dataclasses
import itertools
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Iterator, Mapping, Sequence
from typing import ClassVar

import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import (
    AtomicUpdateGroup,
    assemble_atomic_update_groups,
    pack_units_by_size,
)


@dataclasses.dataclass(frozen=True)
class WeightUpdatePlacement:
    # [CLASS DOC] Immutable flags for gathered dims: pipeline (required, no default), tensor and
    # expert (both default true; always gathered today). A gathered dim yields full tensors on every
    # rank of that dim's group; otherwise each rank yields its own shard.
    """Which training-side parallel dims the iterator gathers before yielding.

    A gathered dim: every yielded tensor is full along that dim on every rank
    of that dim's group. A non-gathered dim: each rank yields its own shard of
    the param set.
    """

    gather_pp: bool
    # Always gathered today; explicit so a future protocol can relax them.
    gather_tp: bool = True
    gather_ep: bool = True

    @property
    def is_full_gather(self) -> bool:
        # [FUNCTION DOC] Property: true only when pipeline, tensor and expert dims are all gathered.
        return self.gather_pp and self.gather_tp and self.gather_ep


def resolve_placement(required: WeightUpdatePlacement, forced: WeightUpdatePlacement | None) -> WeightUpdatePlacement:
    # [FUNCTION DOC] Returns the required placement if no forced placement is given; otherwise a new
    # placement where each dim is gathered if either the required or the forced one gathers it.
    """Join of the protocol's required placement and the iterator's forced one:
    a dim is gathered if either side gathers it."""
    if forced is None:
        return required
    return WeightUpdatePlacement(
        gather_pp=required.gather_pp or forced.gather_pp,
        gather_tp=required.gather_tp or forced.gather_tp,
        gather_ep=required.gather_ep or forced.gather_ep,
    )


class HfWeightIteratorBase(ABC):
    # [CLASS DOC] Abstract base. Class attributes: forced placement this implementation can produce
    # (none means any) and the weight update selector string naming which engine runners a sync
    # covers (default "all"; a backend may narrow to "target", e.g. when a draft model is frozen).
    # Instances keep args, the model, the resolved placement, the model name and the optional
    # quantization config. Subclasses implement two generators (parameter units and adapter units)
    # and may override the atomic group hook.
    """Streams a training model's weights as HF-named tensors.

    Collective: every training rank must drive the iterators in lockstep.
    Yielded tensors are freshly allocated per bucket and stay valid while the
    caller holds a reference.
    """

    # Placement this implementation can produce; None means any.
    forced_placement: ClassVar[WeightUpdatePlacement | None] = None
    # Engine runners a sync covers ("all" or "target"); a backend narrows it when its draft is frozen.
    weight_update_selector: str = "all"

    def __init__(
        self,
        args: Namespace,
        model,
        *,
        placement: WeightUpdatePlacement,
        model_name: str,
        quantization_config: dict | None,
    ) -> None:
        # [FUNCTION DOC] Store the arguments as attributes; placement is passed already resolved.
        self.args = args
        self.model = model
        self.placement = placement
        self.model_name = model_name
        self.quantization_config = quantization_config

    @classmethod
    def build(
        cls,
        args: Namespace,
        model,
        *,
        required_placement: WeightUpdatePlacement,
        model_name: str,
        quantization_config: dict | None,
    ) -> "HfWeightIteratorBase":
        # [FUNCTION DOC] Class-method factory: constructs the subclass with the placement resolved
        # as the join of the required placement and the class's forced placement.
        return cls(
            args,
            model,
            placement=resolve_placement(required_placement, cls.forced_placement),
            model_name=model_name,
            quantization_config=quantization_config,
        )

    def iter_hf_weights(
        self,
        weights: Mapping[str, torch.Tensor] | None,
        *,
        include_base: bool = True,
        adapters: Sequence[tuple[str, object]] = (),
        materialize: bool = True,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        # [FUNCTION DOC] Generator of buckets (lists of name and tensor pairs). Pipeline: (1) units
        # of base weights from the backend hook (only if include_base; otherwise empty); (2) for each
        # (adapter name, adapter) pair, append that adapter's units with every tensor name prefixed
        # by "<adapter name>:"; (3) atomic groups come from the backend hook only when both
        # include_base and materialize hold, else none; units matching a group are merged so they
        # are never split; (4) pack units into buckets bounded by the configured update-weight
        # buffer size. With materialize false all collectives are joined but nothing is yielded.
        # Collective: all training ranks must iterate in lockstep.
        """Model weights as size-bounded buckets of HF-named GPU tensors;
        atomic update groups are never split across buckets.

        ``weights``: backend-native named weights to read; None reads the live
        model parameters. ``adapters``: ``(lora_name, adapter_or_None)`` pairs
        whose tensors join the stream under ``{lora_name}:{hf_key}`` names.
        ``materialize=False`` joins every collective but yields nothing.
        """

        def prefixed_units(lora_name, adapter):
            # [FUNCTION DOC] HELPER: Generator of the adapter's units with each tensor name
            # prefixed by the adapter name and a colon.
            for unit in self._iter_hf_adapter_units(adapter, materialize=materialize):
                yield [(f"{lora_name}:{name}", tensor) for name, tensor in unit]

        hf_param_units = self._iter_hf_param_units(weights, materialize=materialize) if include_base else iter(())
        for lora_name, adapter in adapters:
            hf_param_units = itertools.chain(hf_param_units, prefixed_units(lora_name, adapter))
        atomic_update_groups = self._hf_atomic_update_groups() if include_base and materialize else []
        hf_param_units = assemble_atomic_update_groups(hf_param_units, atomic_update_groups)
        yield from pack_units_by_size(hf_param_units, self.args.update_weight_buffer_size)

    @abstractmethod
    def _iter_hf_param_units(
        self,
        weights: Mapping[str, torch.Tensor] | None,
        *,
        materialize: bool,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        # [FUNCTION DOC] Abstract backend hook (see docstring): given optional backend-native named
        # weights (none means live model parameters), yield one unit per training-side parameter,
        # each unit a list of HF-named tensors honoring the placement.
        """Backend hook: one unit per training-side parameter, holding every HF
        tensor it converted into (a weight, or weight + quant scales), honoring
        ``self.placement``. Collectives must run lockstep on every rank;
        ``materialize=False`` joins them but yields nothing."""

    def _hf_atomic_update_groups(self) -> list[AtomicUpdateGroup]:
        # [FUNCTION DOC] Overridable hook returning the atomic update groups; default empty list.
        """Backend hook: HF-namespace atomic groups for this model. Default none."""
        return []

    def materialize_adapter(self, adapter, *, materialize: bool = True) -> dict[str, torch.Tensor]:
        # [FUNCTION DOC] Flatten all of one adapter's units into a single mapping from bare HF key to
        # tensor, with no bucketing. With materialize false the units still run (joining collectives)
        # but the result is an empty mapping. Collective across training ranks.
        """One adapter as ``{hf_key: tensor}``, skipping the transport bucketing.
        Collective: ``materialize=False`` joins the gathers, returns {}."""
        return {
            name: tensor
            for unit in self._iter_hf_adapter_units(adapter, materialize=materialize)
            for name, tensor in unit
        }

    @abstractmethod
    def _iter_hf_adapter_units(self, adapter, *, materialize: bool) -> Iterator[list[tuple[str, torch.Tensor]]]:
        # [FUNCTION DOC] Abstract backend hook (see docstring): yield one unit per adapter parameter
        # with bare HF key names, sliced for this rank according to the placement.
        """Backend hook: this rank's slice of the adapter per ``self.placement``,
        one unit per parameter, bare ``hf_key`` names, rank-trimmed. Collectives
        must run lockstep on every rank; ``materialize=False`` joins them but
        yields nothing."""
