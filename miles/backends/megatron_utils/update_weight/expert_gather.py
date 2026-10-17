# [MODULE DOC] Gathers batches of named tensors (converted expert weights and their scales) from
# every rank of a process group into every rank, by packing all of a rank's tensors into a single
# flat byte buffer, exchanging the buffers with one asynchronous collective, and rebuilding typed
# strided views on the receive side without copying. The byte layout of each rank's payload is
# exchanged once (on the first call) and then reused.
"""Pack and gather converted expert tensors and scales across training ranks."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _TensorLayout:
    # [CLASS DOC] HELPER: Immutable description of where one tensor sits inside a rank's flat byte
    # payload: its name, shape, element type, byte offset (aligned to the element size), byte
    # length, row-major element strides (dimensions of size zero count as one), and the offset in
    # elements within a view of the payload typed as the tensor's element type.
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    byte_offset: int
    nbytes: int
    strides: tuple[int, ...]
    storage_offset: int  # Element offset within the payload's typed view.


@dataclass(frozen=True)
class _PayloadLayout:
    # [CLASS DOC] HELPER: Immutable description of a whole rank payload: the tensor layouts grouped
    # into units (the same grouping as the caller's units), the unpadded payload byte length, the
    # alignment (largest element size seen, one if empty), and for each element type the length in
    # bytes of the payload prefix that can be reinterpreted as that type (the total length rounded
    # down to a multiple of its element size; this is a prefix of the whole payload, not a
    # per-type total).
    units: tuple[tuple[_TensorLayout, ...], ...]
    nbytes: int
    alignment: int
    # Dtype-aligned prefix lengths of the entire payload, not per-dtype totals.
    dtype_view_nbytes: tuple[tuple[torch.dtype, int], ...]


class ExpertGather:
    # [CLASS DOC] Callable object bound to one process group that all-gathers fixed-layout batches
    # of named tensors. State: the group, the member global ranks, this rank's index, and lazily
    # cached layout data (per-rank payload layouts, padded per-rank payload sizes, total bytes,
    # whether all sizes are equal, and the sole data-holding rank if only one exists). The cache is
    # filled on the first call and never invalidated, so the tensor names, shapes, element types,
    # unit structure and group must not change over the object's life (violations are caught by
    # assertions when packing, but only on the local rank).
    """Gather one fixed-layout expert batch, exchanging payload layouts once.

    Names, unit boundaries, shapes, dtypes, and group membership must stay fixed
    for this object's lifetime. Recreate it when the model, quantization config,
    or topology changes. Tensor values and storage may change on every call.
    Packing checks the local layout before starting payload collectives.

    The existing process-group reference and layout are reused across updates; Work
    handles belong to individual operations and are waited before returning.
    Each call allocates fresh receive storage, since callers may retain earlier
    outputs. Rank segments are padded only for dtype alignment, never to the
    largest rank's payload. ``device`` selects communication and output storage.
    """

    def __init__(self, *, group: dist.ProcessGroup):
        # [FUNCTION DOC] Stores the process group, the global ranks of its members in group order,
        # and this process's index within them (raises a value error if the current rank is not a
        # member). Initializes empty cached layout state (layouts unknown, payload sizes empty, no
        # single source, zero total bytes, not uniform) filled lazily on the first call.
        self._group = group
        self._source_ranks = tuple(dist.get_process_group_ranks(group))
        self._local_index = self._source_ranks.index(dist.get_rank())
        self._layouts: tuple[_PayloadLayout, ...] | None = None
        self._payload_sizes: tuple[int, ...] = ()
        self._single_source: int | None = None
        self._total_bytes = 0
        self._uniform = False

    def __call__(
        self, units: list[list[tuple[str, torch.Tensor]]], *, device: torch.device | str
    ) -> list[list[tuple[str, torch.Tensor]]]:
        # [FUNCTION DOC] Call operator that gathers the local units from all ranks. Takes a list of units (each a list of name and tensor pairs) and a device
        # (keyword-only) and returns the concatenation, in group-rank order, of all ranks' units
        # as views into a freshly allocated receive buffer. A group of one rank returns the input
        # unchanged. First call only: builds the local payload layout, exchanges layouts among all
        # ranks (a collective object gather), takes the largest alignment, pads each rank's
        # payload size up to that alignment, and records total bytes, whether all padded sizes are
        # equal, and which single rank (if exactly one rank has a nonempty payload) is the sole
        # source. Every call: allocates a byte buffer of the total size on the device, splits it
        # into per-rank segments, packs local tensors into the local segment (checking the layout
        # has not changed), starts the asynchronous collective, builds the typed views while the
        # transfer is in flight, then waits for the transfer before returning. Because views alias
        # the buffer, results stay valid as long as callers hold them. Layout mismatches raise
        # assertion errors; an empty total skips communication.
        if len(self._source_ranks) == 1:
            return units
        if self._layouts is None:
            layouts = [None] * len(self._source_ranks)
            dist.all_gather_object(layouts, _build_payload_layout(units), group=self._group)
            self._layouts = tuple(layouts)
            alignment = max(layout.alignment for layout in self._layouts)
            self._payload_sizes = tuple(
                (layout.nbytes + alignment - 1) // alignment * alignment for layout in self._layouts
            )
            self._total_bytes = sum(self._payload_sizes)
            self._uniform = len(set(self._payload_sizes)) == 1
            sources = [index for index, size in enumerate(self._payload_sizes) if size]
            self._single_source = sources[0] if len(sources) == 1 else None

        local_layout = self._layouts[self._local_index]
        storage = torch.empty(self._total_bytes, dtype=torch.uint8, device=device)
        payloads = list(storage.split(self._payload_sizes))
        local_payload = payloads[self._local_index]
        _pack_units(units, local_layout, local_payload)
        handle = self._gather_payloads(storage, payloads, local_payload)
        # Build views on the CPU while the asynchronous transfer is in flight.
        gathered = [
            unit
            for layout, payload in zip(self._layouts, payloads, strict=True)
            for unit in _unpack_units(layout, payload)
        ]
        if handle is not None:
            handle.wait()
        return gathered

    def _gather_payloads(self, storage, payloads, local_payload):
        # [FUNCTION DOC] HELPER: Starts the asynchronous collective and returns its work handle
        # (or none when nothing is to be sent). If there are no bytes: none. If exactly one rank
        # has data: a broadcast of that rank's segment from that rank. Else if all segments have
        # equal size: an in-place all-gather into the whole buffer (no temporaries). Else an
        # uneven all-gather into the list of segments.
        if not self._total_bytes:
            return None
        if self._single_source is not None:
            source = self._single_source
            return dist.broadcast(payloads[source], src=self._source_ranks[source], group=self._group, async_op=True)
        if self._uniform:
            # Native NCCL all-gather, in place: no flattened temporary or copies.
            return dist.all_gather_into_tensor(storage, local_payload, group=self._group, async_op=True)
        # NCCL coalesces uneven all-gather internally into one Work handle.
        return dist.all_gather(payloads, local_payload, group=self._group, async_op=True)


def _build_payload_layout(units):
    # [FUNCTION DOC] HELPER: Computes the payload layout for a list of units of named tensors,
    # without touching tensor data. Walks tensors in order; each tensor's byte offset is rounded
    # up to a multiple of its element size, its byte length is element count times element size,
    # and its strides are the row-major strides (size-zero dimensions treated as size one). Records
    # element sizes per type to compute alignment (largest element size, default one) and per-type
    # typed-prefix lengths. Returns the payload layout.
    unit_layouts = []
    dtype_sizes = {}
    byte_offset = 0
    for unit in units:
        unit_layout = []
        for name, tensor in unit:
            item_size = tensor.element_size()
            dtype_sizes[tensor.dtype] = item_size
            # Typed views require their storage offset to be dtype-aligned.
            byte_offset = (byte_offset + item_size - 1) // item_size * item_size
            nbytes = tensor.numel() * item_size
            shape = tuple(tensor.shape)
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.append(stride)
                stride *= max(dim, 1)
            unit_layout.append(
                _TensorLayout(
                    name, shape, tensor.dtype, byte_offset, nbytes, tuple(reversed(strides)), byte_offset // item_size
                )
            )
            byte_offset += nbytes
        unit_layouts.append(tuple(unit_layout))
    dtype_view_nbytes = tuple(
        (dtype, byte_offset // item_size * item_size) for dtype, item_size in dtype_sizes.items()
    )
    return _PayloadLayout(tuple(unit_layouts), byte_offset, max(dtype_sizes.values(), default=1), dtype_view_nbytes)


def _pack_units(units, layout, payload):
    # [FUNCTION DOC] HELPER: Copies each tensor's bytes into its slot of the local byte payload
    # (in place). Asserts that the unit count, the tensor count per unit, and each tensor's name,
    # shape and element type match the recorded layout (messages tell the user to recreate the
    # iterator). Tensors are made contiguous and flattened to bytes before copying.
    assert len(units) == len(layout.units), "Expert output unit count changed; recreate the iterator"
    for unit, unit_layout in zip(units, layout.units, strict=True):
        assert len(unit) == len(unit_layout), "Expert output tensor count changed; recreate the iterator"
        for (name, tensor), tensor_layout in zip(unit, unit_layout, strict=True):
            assert (
                name == tensor_layout.name
                and tensor.shape == tensor_layout.shape
                and tensor.dtype == tensor_layout.dtype
            ), f"Expert output layout changed for {tensor_layout.name}; recreate the iterator"
            payload.narrow(0, tensor_layout.byte_offset, tensor_layout.nbytes).copy_(
                tensor.contiguous().reshape(-1).view(torch.uint8)
            )


def _unpack_units(layout, payload):
    # [FUNCTION DOC] HELPER: Rebuilds the units of named tensors from one rank's byte payload
    # without copying. For each element type used, takes the typed-prefix of the payload (cropping
    # odd byte tails) viewed as that type; each tensor becomes a strided view of its type's view
    # using the recorded shape, strides and element offset (added to the view's own storage
    # offset). Returns a list of units, each a list of name and tensor pairs, in layout order.
    # Crop odd byte tails before reinterpreting the common storage by dtype.
    typed_payloads = {dtype: payload[:nbytes].view(dtype) for dtype, nbytes in layout.dtype_view_nbytes}
    storage_offsets = {dtype: tensor.storage_offset() for dtype, tensor in typed_payloads.items()}
    return [
        [
            (
                tensor_layout.name,
                typed_payloads[tensor_layout.dtype].as_strided(
                    tensor_layout.shape,
                    tensor_layout.strides,
                    storage_offsets[tensor_layout.dtype] + tensor_layout.storage_offset,
                ),
            )
            for tensor_layout in unit_layout
        ]
        for unit_layout in layout.units
    ]
