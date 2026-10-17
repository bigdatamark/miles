# [MODULE DOC] Hands out blocks of free network ports on cluster nodes by asking a worker actor to find
# a free consecutive block, remembering per node address where the next search should begin.
import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DYNAMIC_PORT_START = 20000
_MAX_PORT = 65535


@dataclass
class PortAllocator:
    # [CLASS DOC] Allocator holding the next search start port per node IP (a mapping, default empty)
    # and an asyncio lock that serializes allocations (so concurrent callers never get overlapping
    # blocks from the same allocator). Not shared across processes: ports taken by others are only
    # detected by the actor's free-port probe.
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def alloc(self, actor, *, node_ip: str, consecutive: int = 1) -> int:
        # [FUNCTION DOC] Returns the first port of a block of the requested number of consecutive free
        # ports (default 1) on the node. Under the lock: starts the search at the node's remembered
        # next port (default 20000); if the block would end beyond 65535 wraps the start back to 20000;
        # asks the actor, with an awaited remote call, for a free block from that start (the actor
        # method is expected to take start port and count and return the block's first port; any
        # error it raises propagates and leaves the remembered port unchanged); remembers the returned
        # port plus the block size as the next start. Ports are kept at 20000 and above to avoid
        # the ephemeral port range and Ray's own ports.
        async with self._lock:
            # use small ports to prevent ephemeral port between 32768 and 65536.
            # also, ray uses port 10002-19999, thus we avoid near-10002 to avoid racing condition
            start_port = self._next_port_of_ip.get(node_ip, _DYNAMIC_PORT_START)
            if start_port + consecutive - 1 > _MAX_PORT:
                start_port = _DYNAMIC_PORT_START
            port: int = await actor._get_free_port_block.remote(
                start_port=start_port,
                count=consecutive,
            )
            self._next_port_of_ip[node_ip] = port + consecutive
            return port
