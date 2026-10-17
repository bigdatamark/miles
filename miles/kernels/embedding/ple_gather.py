# [MODULE DOC] Gathers rows of a very large embedding table that lives in pinned (page-locked) host
# memory directly from a GPU kernel, so the table is never copied into GPU memory: the kernel
# dereferences the raw host address and the hardware fetches each needed row across the host link.
# Provides the Triton kernel (one program per looked-up row) and the host wrapper gather_ple_rows.
# The table may be a shard: only ids within a given global row range are served; others yield zeros.
"""Gather PLE rows straight out of pinned host memory.

One program per (token, hash-head) row, reading through a raw host pointer so the GPU
pulls the row over the host link instead of staging the 102 GB table into HBM.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_ple_rows_from_pinned(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    row_start,
    row_end,
    BLOCK_D: tl.constexpr,
):
    # [FUNCTION DOC] HELPER: Triton kernel; one program per flattened id (64-bit program index).
    # weight_ptr is the integer address of the host table, reinterpreted inside the kernel as a
    # pointer to bfloat16. ids are 64-bit global row numbers on the device. A row is served only if
    # row_start <= id < row_end (the shard range); its local row is id minus row_start. In-range ids
    # copy embedding_dim bfloat16 values (columns beyond the dim masked) from the local row to the
    # output row; out-of-range ids write zeros. The 64-bit arithmetic avoids overflow of row offsets
    # for large tables.
    # ids hold T * heads rows of embedding_dim each; int32 row offsets wrap past ~52K tokens
    row_id = tl.program_id(0).to(tl.int64)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= row_start) & (global_idx < row_end)
    local_idx = tl.where(in_range, global_idx - row_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(ptr + local_idx * embedding_dim + offsets, mask=mask & in_range, other=0.0)
    tl.store(output_ptr + row_id * embedding_dim + offsets, values, mask=mask)


def gather_ple_rows(
    host_table: torch.Tensor,
    ids: torch.Tensor,
    row_start: int,
    row_end: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    # [FUNCTION DOC] Gathers rows for an arbitrary-shaped tensor of ids (64-bit, on the GPU) from the
    # host table (rows by embedding_dim, pinned CPU bfloat16; assertions fail otherwise). The output
    # has the ids' shape plus the embedding dim, bfloat16, on the ids' device; if out is provided its
    # shape and dtype are asserted (its device and contiguity are not; the kernel requires a
    # contiguous buffer on the ids' device) and it is filled in place. Rows for ids outside
    # [row_start, row_end) are zero. No kernel is launched for empty ids. Returns out. The ids' dtype
    # is not checked.
    """``ids`` ``[...]`` int64 on device -> ``[..., embedding_dim]`` bf16 on device."""
    assert host_table.device.type == "cpu", "PLE table must stay on the host"
    assert host_table.is_pinned(), "PLE table must be pinned for the kernel to read it"
    assert host_table.dtype == torch.bfloat16, f"expected bf16 table, got {host_table.dtype}"
    embedding_dim = host_table.shape[-1]

    shape = (*ids.shape, embedding_dim)
    if out is None:
        out = torch.empty(shape, dtype=torch.bfloat16, device=ids.device)
    else:
        assert tuple(out.shape) == shape and out.dtype == torch.bfloat16

    flat = ids.reshape(-1)
    if flat.numel():
        _gather_ple_rows_from_pinned[(flat.numel(),)](
            host_table.data_ptr(),
            flat.contiguous(),
            out.view(-1, embedding_dim),
            embedding_dim=embedding_dim,
            row_start=row_start,
            row_end=row_end,
            BLOCK_D=triton.next_power_of_2(embedding_dim),
        )
    return out
