# [MODULE DOC]
# Test helpers for launching small multi-process distributed jobs on CPU: find an unused tcp
# port, initialise a gloo process group for one rank through environment variables, and spawn a
# function across several processes with the port shared. Used by fast tests that need a real
# process group without GPUs.
import os
import socket
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp


def find_free_port() -> int:
    # [FUNCTION DOC] HELPER: Binds a temporary tcp socket to port 0 on all interfaces so the
    # operating system picks a free port, reads the port number and closes the socket. Returns the
    # port. Inherently racy: another process may take the port before the caller uses it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def init_gloo(rank: int, world_size: int, *, port: int) -> None:
    # [FUNCTION DOC] HELPER: Sets the process environment so the rendezvous address is localhost
    # and the rendezvous port is the given port (global side effect on environment variables),
    # then initialises the default distributed process group with the gloo (CPU) backend for the
    # given rank and world size. Blocks until all ranks have joined; returns nothing.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def run_multiprocess(fn: Any, world_size: int = 2) -> None:
    # [FUNCTION DOC] HELPER: Picks a free port and spawns world_size processes running the given
    # function, each called with the rank (supplied by the spawner), the world size and the port
    # as arguments. Joins all processes before returning; a failure in any child makes the
    # spawner raise in the parent. The function must be picklable (module level).
    port = find_free_port()
    mp.spawn(fn, args=(world_size, port), nprocs=world_size, join=True)
