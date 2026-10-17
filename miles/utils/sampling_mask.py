# [MODULE DOC] Compact container for a rollout's sampling mask (per response token, the set of
# vocabulary ids the sampler was allowed to emit), stored in compressed-sparse-row form as two flat
# CPU integer tensors, plus small helpers that convert and validate integer sequences/tensors.
from array import array
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

import torch


@dataclass(frozen=True, eq=False)
class RolloutSamplingMask:
    # [CLASS DOC] Immutable (frozen, identity-based equality) record of one sample's sampling
    # support. Construction takes ids and offsets (either integer sequences or one-dimensional
    # integer tensors); they are copied into private CPU tensors (ids as 32-bit integers, offsets
    # as 64-bit integers) held in two hidden fields, so later mutation of the inputs never
    # affects the mask. Invariants enforced at construction: offsets non-empty, first is zero,
    # last equals the number of ids, and offsets strictly increase (every response token has a
    # non-empty support). The length of the object is the number of response tokens (offsets
    # count minus one).
    """One sample's sampling mask: for each response position, the token ids
    the rollout sampler could emit.

    Stored in CSR form so it stays two flat integer arrays end to end:
    ``ids`` is ``[total_support_size]`` (all supports concatenated), ``offsets``
    is ``[num_response_tokens + 1]``, and token ``t``'s support is
    ``ids[offsets[t] : offsets[t + 1]]``. That is the shape object-store
    transport needs, so no per-token nesting is rebuilt on the trainer side.
    """

    ids: InitVar[Sequence[int] | torch.Tensor]
    offsets: InitVar[Sequence[int] | torch.Tensor]
    _ids: torch.Tensor = field(init=False, repr=False)
    _offsets: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self, ids: Sequence[int] | torch.Tensor, offsets: Sequence[int] | torch.Tensor):
        # [FUNCTION DOC] Post-construction hook: converts ids and offsets to owned CPU tensors
        # (see the conversion helper; raises a value error for non one-dimensional or
        # non-integer input), validates the offsets invariants and raises a value error if the
        # offsets are empty, do not start at zero, do not end at the id count, or are not
        # strictly increasing. Stores the tensors on the frozen instance by bypassing the
        # frozen-attribute guard.
        owned_ids = _to_owned_cpu_integer_tensor(ids, dtype=torch.int32)
        owned_offsets = _to_owned_cpu_integer_tensor(offsets, dtype=torch.long)
        if owned_offsets.numel() == 0 or owned_offsets[0] != 0 or owned_offsets[-1] != owned_ids.numel():
            raise ValueError("sampling-mask offsets must start at zero and end at the flattened id count")
        if torch.any(owned_offsets[1:] <= owned_offsets[:-1]):
            raise ValueError(
                "sampling-mask offsets must be strictly increasing: "
                "every response token needs a non-empty sampling mask"
            )
        object.__setattr__(self, "_ids", owned_ids)
        object.__setattr__(self, "_offsets", owned_offsets)

    @classmethod
    def from_mask_list(cls, mask_list: Sequence[Sequence[int]]) -> "RolloutSamplingMask":
        # [FUNCTION DOC] Alternate constructor from a ragged list with one list of allowed token
        # ids per response token (the shape the inference engine's sampling-mask output has).
        # Flattens all lists into one id list and builds offsets as running cumulative lengths
        # starting at zero, then constructs the mask. Raises a value error (via construction)
        # if any per-token list is empty. An empty outer list gives an empty mask.
        """Build from one mask (the allowed token ids) per response token.

        Args:
            mask_list: ragged ``[num_response_tokens][mask_size_t]``;
                ``mask_list[t]`` lists the token ids the sampler could emit at
                response position ``t``. SGLang's ``output_token_sampling_mask``
                arrives in this shape.
        """
        ids = []
        offsets = [0]
        for mask in mask_list:
            ids.extend(mask)
            offsets.append(len(ids))
        return cls(ids=ids, offsets=offsets)

    @classmethod
    def concatenate(cls, masks: Sequence["RolloutSamplingMask"]) -> "RolloutSamplingMask":
        # [FUNCTION DOC] Joins several masks end to end in the given order. An empty sequence
        # yields an empty mask (zero response tokens); a single-element sequence returns that
        # same object (no copy). Otherwise concatenates the id tensors, and shifts each mask's
        # offsets (excluding its leading zero) by the number of ids that precede it, prefixing
        # one leading zero; builds a new mask from the result.
        """Concatenate complete per-token supports in response order."""
        if not masks:
            return cls(ids=[], offsets=[0])
        if len(masks) == 1:
            return masks[0]

        ids = torch.cat([mask._ids for mask in masks])
        offsets = [torch.zeros(1, dtype=torch.long)]
        id_count = 0
        for mask in masks:
            offsets.append(mask._offsets[1:] + id_count)
            id_count += mask._ids.numel()
        return cls(ids=ids, offsets=torch.cat(offsets))

    def __len__(self) -> int:
        # [FUNCTION DOC] Returns the number of response tokens covered by the mask.
        return self._offsets.numel() - 1

    def prefix(self, response_length: int) -> "RolloutSamplingMask":
        # [FUNCTION DOC] Returns a mask for only the first response_length response tokens.
        # Raises a value error when the length is negative or exceeds the mask length. When the
        # length equals the full length, returns the same object; otherwise builds a new mask
        # from the leading slices of ids and offsets (the new mask copies its data).
        """Return the support for the first ``response_length`` tokens."""
        if not 0 <= response_length <= len(self):
            raise ValueError(f"sampling-mask prefix length must be in [0, {len(self)}]")
        if response_length == len(self):
            return self
        id_count = self._offsets[response_length]
        return type(self)(ids=self._ids[:id_count], offsets=self._offsets[: response_length + 1])

    def _as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Returns the private ids and offsets tensors themselves (no
        # copy) for immediate read-only transport; callers must not mutate or retain them.
        """Borrow the private CSR tensors for immediate read-only transport."""
        return self._ids, self._offsets

    def _select_masks(self, token_indices: Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Gathers the supports of chosen response positions. Returns a
        # pair: the selected supports concatenated in the order of the requested positions, and
        # a per-position length tensor. Raises a value error if any position is outside
        # [0, length) (also for the range fast path), or if the indices are not a
        # one-dimensional integer sequence/tensor. Fast path: for a step-1 range, returns a
        # slice view of ids (shared storage) and the differences of consecutive offsets; an
        # empty range returns two empty tensors. General path: converts indices to a 64-bit CPU
        # tensor, computes lengths from offsets, and slices each maximal run of consecutive
        # ascending indices as one contiguous block, concatenating blocks only when there is
        # more than one (a single block is returned as a view). Note an empty step-1 range skips
        # bounds validation and returns empty tensors.
        """Flattened masks for the given response positions.

        Args:
            token_indices: ``[num_selected]`` response positions to read.

        Returns:
            ``(ids, lengths)`` where ``ids`` is ``[sum(lengths)]``, the selected
            masks concatenated in ``token_indices`` order, and ``lengths`` is
            ``[num_selected]``, each position's mask size.

        The returned ids may share the mask's private storage and are only for
        immediate read-only use by the scoring path.
        """
        if isinstance(token_indices, range) and token_indices.step == 1:
            if len(token_indices) == 0:
                return self._ids.new_empty(0), self._offsets.new_empty(0)
            if token_indices.start < 0 or token_indices.stop > len(self):
                raise ValueError(f"response indices must be in [0, {len(self)})")
            start, stop = token_indices.start, token_indices.stop
            lengths = self._offsets[start + 1 : stop + 1] - self._offsets[start:stop]
            return self._ids[self._offsets[start] : self._offsets[stop]], lengths

        indices = _to_cpu_integer_tensor(token_indices).to(torch.long)
        if torch.any(indices < 0) or torch.any(indices >= len(self)):
            raise ValueError(f"response indices must be in [0, {len(self)})")
        lengths = self._offsets[indices + 1] - self._offsets[indices]
        if indices.numel() == 0:
            return self._ids.new_empty(0), lengths
        run_starts = [0]
        run_starts.extend((torch.nonzero(indices[1:] != indices[:-1] + 1).flatten() + 1).tolist())
        run_starts.append(indices.numel())
        parts = [
            self._ids[self._offsets[indices[start]] : self._offsets[indices[end - 1] + 1]]
            for start, end in zip(run_starts[:-1], run_starts[1:], strict=True)
        ]
        return (parts[0] if len(parts) == 1 else torch.cat(parts)), lengths


def _to_owned_cpu_integer_tensor(
    values: Sequence[int] | torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts input to a one-dimensional CPU tensor of the requested
    # dtype that the caller exclusively owns. For a tensor input: validates it (value error if
    # not a one-dimensional integer tensor) and returns a copied, device- and dtype-converted
    # tensor. For a non-empty sequence with dtype 32-bit int: packs values into a C int array
    # and wraps it without copying (the tensor keeps the array alive; values outside the C
    # int range raise an overflow error, non-integers raise a type error). Any other sequence
    # (including empty, or 64-bit dtype) goes through plain tensor construction.
    # KNOWN ISSUE: sequence inputs are not run through the integer validation (float lists are
    # rejected only by the array packing for the 32-bit path, but silently accepted and
    # converted for the 64-bit path).
    if isinstance(values, torch.Tensor):
        _validate_integer_tensor(values)
        return values.to(device="cpu", dtype=dtype, copy=True)

    if dtype == torch.int32 and len(values) > 0:
        storage = array("i", values)
        # frombuffer keeps this private backing array alive without copying it.
        return torch.frombuffer(storage, dtype=torch.int32)

    return torch.tensor(values, dtype=dtype, device="cpu")


def _to_cpu_integer_tensor(values: Sequence[int] | torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Converts index input to a CPU integer tensor without forcing a
    # dtype or copy guarantee. Tensors are detached and moved to CPU; an empty sequence becomes
    # an empty 64-bit tensor; a range becomes an arange with the same start/stop/step; other
    # sequences go through tensor conversion. The result is validated as a one-dimensional
    # integer tensor (value error otherwise).
    if isinstance(values, torch.Tensor):
        tensor = values.detach().cpu()
    elif len(values) == 0:
        tensor = torch.empty(0, dtype=torch.long, device="cpu")
    elif isinstance(values, range):
        tensor = torch.arange(values.start, values.stop, values.step, device="cpu")
    else:
        tensor = torch.as_tensor(values, device="cpu")
    _validate_integer_tensor(tensor)
    return tensor


def _validate_integer_tensor(tensor: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Raises a value error unless the tensor is one-dimensional and has
    # an integer dtype (booleans, floating point and complex are rejected). Returns nothing.
    if tensor.ndim != 1 or tensor.dtype == torch.bool or torch.is_floating_point(tensor) or torch.is_complex(tensor):
        raise ValueError("sampling-mask ids, offsets, and response indices must be one-dimensional integers")
