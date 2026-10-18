# [MODULE DOC]
# Gpu memory inspection and cleanup helpers (current cuda device).
import gc
import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def clear_memory(clear_host_memory: bool = False):
    # [FUNCTION DOC] Synchronizes the gpu, runs python garbage collection, releases cached gpu
    # memory back to the driver, and optionally empties the pinned host allocator cache (private
    # torch call). Blocks until gpu work finishes.
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    # [FUNCTION DOC] Returns a mapping for the current device: its index as text, total, free, used
    # (total minus free), allocated by torch and reserved by torch, all in gigabytes (1024 cubed)
    # rounded to two decimals.
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    # [FUNCTION DOC] HELPER: Bytes to gigabytes (divide by 1024 cubed), rounded to two decimals.
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    # [FUNCTION DOC] Optionally clears memory first, then logs at info level (on every rank, since
    # ranks may differ) the rank, the message, a note if cleared, and the memory mapping. Requires an
    # initialized distributed world for the rank. Returns the memory mapping.
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info
