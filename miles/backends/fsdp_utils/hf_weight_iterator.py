# [MODULE DOC] Weight iterator that streams the FSDP-trained model's parameters, gathered to
# full tensors and renamed/reshaped to the layout the rollout engine loads, for weight sync. It
# plugs into the shared HF weight iterator base class by providing the unit generator, atomic
# update groups and adapter hook. Gathers are issued asynchronously in bounded batches (by
# cumulative parameter bytes against the update weight buffer size arg) to overlap communication.
from collections import deque
from collections.abc import Iterator

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor

from miles.backends.fsdp_utils.adaptations.weight_bridge import get_param_transform
from miles.backends.fsdp_utils.dtensor import gather_full_param
from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups


class FSDPHfWeightIterator(HfWeightIteratorBase):
    # [CLASS DOC] HF-format weight iterator for the FSDP backend. Class attribute forced
    # placement is none (the base class does not force a device placement). Instance state: the
    # base class state (model, args, model name) plus a mapping from parameter name to the dtype to
    # send it in (taken from the model's recorded sync dtypes; empty if the model has none).
    forced_placement = None

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] Initialize the base class with the given arguments, then read the sync
        # dtype mapping from the model attribute recorded by the fp32 master conversion (empty
        # mapping if missing or empty).
        super().__init__(*args, **kwargs)
        self._sync_dtypes: dict[str, torch.dtype] = getattr(self.model, "_fsdp_sync_dtypes", None) or {}

    def _iter_hf_param_units(self, weights, *, materialize):
        # [FUNCTION DOC] Generator yielding update units, each a list of pairs of HF parameter name
        # and full tensor. Walks the model's state dict in order; for each entry starts an async
        # gather and queues it with its original parameter; adds the entry's element count times
        # element size to a running byte total; when the total reaches the update weight buffer
        # size arg, drains the whole queue (yielding its units) and resets the total; drains the
        # rest at the end. The materialize flag is passed on (when false the gathers are still
        # waited on but nothing is yielded). The weights argument is not used.
        # Collective: all ranks must iterate identically.
        pending: deque[tuple[str, torch.Tensor, torch.Tensor]] = deque()
        pending_bytes = 0
        for name, param in self.model.state_dict().items():
            pending.append((name, param, gather_full_param(param, async_op=True)))
            pending_bytes += param.numel() * param.element_size()
            if pending_bytes >= self.args.update_weight_buffer_size:
                yield from self._drain(pending, materialize=materialize)
                pending_bytes = 0
        yield from self._drain(pending, materialize=materialize)

    def _drain(self, pending: deque, *, materialize: bool) -> Iterator[list[tuple[str, torch.Tensor]]]:
        # [FUNCTION DOC] HELPER: Empty the queue in first-in-first-out order, mutating it. For each
        # item, waits for its async gather if the tensor is a lazy collective result. If not
        # materializing, discards it. Otherwise converts to the recorded sync dtype for its name,
        # looks up a parameter transform for the model type from the model config; with no transform
        # yields a one-element unit with the original name; with a transform yields one unit holding
        # all pairs the transform produces, each tensor converted to the sync dtype recorded for its
        # new name (usually none, so already converted from the fused name).
        model_type = self.model.config.model_type
        while pending:
            name, param, full = pending.popleft()
            if isinstance(full, AsyncCollectiveTensor):
                full = full.wait()
            if not materialize:
                continue
            full = self._to_sync_dtype(name, full)
            expand = get_param_transform(name, param, model_type)
            if expand is None:
                yield [(name, full)]
            else:
                yield [
                    (hf_name, self._to_sync_dtype(hf_name, tensor))
                    for hf_name, tensor in expand(name, full, self.model)
                ]

    def _to_sync_dtype(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Return the tensor unchanged if no sync dtype is recorded for the
        # name or it already has that dtype, else a copy converted to the recorded dtype.
        target = self._sync_dtypes.get(name)
        return tensor if target is None or tensor.dtype == target else tensor.to(target)

    def _hf_atomic_update_groups(self):
        # [FUNCTION DOC] Groups of parameters that must be updated together on the engine,
        # computed by the shared helper from the model name and the model config's query LoRA rank
        # (none when the attribute is missing or falsy, such as zero).
        return get_hf_atomic_update_groups(
            self.model_name, q_lora_rank=getattr(self.model.config, "q_lora_rank", None) or None
        )

    def _iter_hf_adapter_units(self, lora_name, adapter, *, materialize) -> Iterator:
        # [FUNCTION DOC] Adapter (LoRA) weight sync is not supported by this backend: always raises
        # a not-implemented error.
        raise NotImplementedError("the FSDP backend has no LoRA weight sync")
