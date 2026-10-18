# [MODULE DOC] Cross-process mutual exclusion built on a torch distributed key-value store: a
# first-come-first-served ticket lock, and a factory that sets up a store shared by all ranks of
# the world (rank 0 hosts it) and returns a lock over it.
import logging
import time
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType

import torch.distributed as dist
from torch.distributed import Store, TCPStore

from miles.utils.distributed_utils import get_gloo_group
from miles.utils.misc import get_current_node_ip

logger = logging.getLogger(__name__)


class StoreTicketLock:
    # [CLASS DOC] Ticket lock: a waiter takes the next ticket number by atomically incrementing a
    # "next" counter key and waits (polling) until a "serving" counter key equals its ticket;
    # releasing increments the serving counter. Keys are the prefix plus "/next" and "/serving".
    # Not reentrant; holds its own ticket in instance state. Usable as a synchronous context
    # manager.
    """FIFO mutual exclusion over a ``torch.distributed`` store."""

    def __init__(
        self,
        *,
        store: Store,
        prefix: str,
        poll_interval: float = 0.01,
        warn_interval: float = 60.0,
    ) -> None:
        # [FUNCTION DOC] Keyword-only: the store, key prefix, poll interval in seconds (default
        # 0.01) and the interval after which a waiting warning is logged (default 60 seconds).
        # Derives the two counter keys; no ticket held initially.
        self._store = store
        self._next_key = f"{prefix}/next"
        self._serving_key = f"{prefix}/serving"
        self._poll_interval = poll_interval
        self._warn_interval = warn_interval
        self._held_ticket: int | None = None

    def __enter__(self) -> "StoreTicketLock":
        # [FUNCTION DOC] Acquires the lock and returns itself.
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Releases the lock only on normal exit; never suppresses exceptions.
        # KNOWN ISSUE: on exceptional exit the lock is deliberately not released, so the serving
        # counter never advances and every later waiter blocks forever (possibly intended to stop
        # the world after a failure, but undocumented).
        if exc_type is None:
            self.release()

    def acquire(self) -> None:
        # [FUNCTION DOC] Asserts this object does not already hold a ticket. Takes a ticket (the
        # incremented next counter minus one, so tickets start at 0), then polls the serving
        # counter (read by adding zero) every poll interval until it equals the ticket, logging a
        # warning with ticket and currently served number each warn interval. Blocks without a
        # timeout. Records the held ticket.
        assert self._held_ticket is None, f"lock is already held with ticket {self._held_ticket}"

        ticket = self._store.add(self._next_key, 1) - 1
        last_warned = time.monotonic()
        while (serving := self._store.add(self._serving_key, 0)) != ticket:
            if time.monotonic() - last_warned >= self._warn_interval:
                logger.warning(f"Waiting for {self._serving_key}: holding ticket {ticket}, now serving {serving}")
                last_warned = time.monotonic()
            time.sleep(self._poll_interval)

        self._held_ticket = ticket

    def release(self) -> None:
        # [FUNCTION DOC] Asserts a ticket is held, clears it, and increments the serving counter
        # so the next ticket holder proceeds.
        assert self._held_ticket is not None, f"lock {self._serving_key} is not held, cannot release"

        self._held_ticket = None
        self._store.add(self._serving_key, 1)


def create_world_ticket_lock(
    *,
    prefix: str,
    participates: bool = True,
    poll_interval: float = 0.01,
) -> AbstractContextManager:
    # [FUNCTION DOC] Collective: every rank of the default group must call it. Rank 0 creates a
    # store server on an OS-chosen port bound to all interfaces and broadcasts "node ip:port" to
    # all ranks over the gloo group. If this rank does not participate, returns a do-nothing
    # context manager whose value is the store (rank 0) or none (other ranks); rank 0 still hosts
    # the server so participants can use it. Otherwise non-zero ranks connect as clients (using the
    # broadcast address) and every participant returns a ticket lock over the shared store with
    # the given key prefix and poll interval.
    rank = dist.get_rank()
    store = TCPStore(host_name="0.0.0.0", port=0, is_master=True, wait_for_workers=False) if rank == 0 else None
    address = [f"{get_current_node_ip()}:{store.port}"] if store is not None else [None]
    dist.broadcast_object_list(address, src=0, group=get_gloo_group())

    if not participates:
        return nullcontext(store)

    if rank != 0:
        assert store is None, f"rank {rank} must not host the store"
        host, port = address[0].rsplit(":", 1)
        store = TCPStore(host_name=host, port=int(port), is_master=False)

    return StoreTicketLock(store=store, prefix=prefix, poll_interval=poll_interval)
