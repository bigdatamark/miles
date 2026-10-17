# [MODULE DOC]
# Minimal heartbeat primitive: an object that records the wall-clock time of the last "bump" and how
# many bumps happened, exposed as an immutable status snapshot.
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class HeartbeatStatus:
    # [CLASS DOC] Immutable snapshot: unix timestamp (seconds) of the last activity and the number
    # of bumps since creation.
    last_active_timestamp: float
    bump_count: int


class SimpleHeartbeat:
    # [CLASS DOC] Heartbeat holder. Replaces its immutable status on each bump (the swap is a single
    # attribute assignment, so readers on other threads see a consistent snapshot; concurrent
    # bumpers could lose a count since the increment is not atomic).
    def __init__(self) -> None:
        # [FUNCTION DOC] Starts with the current time and a bump count of zero.
        self._status = HeartbeatStatus(last_active_timestamp=time.time(), bump_count=0)

    def bump(self) -> None:
        # [FUNCTION DOC] Records activity now and increases the bump count by one.
        self._status = HeartbeatStatus(
            last_active_timestamp=time.time(),
            bump_count=self._status.bump_count + 1,
        )

    def status(self) -> HeartbeatStatus:
        # [FUNCTION DOC] Returns the current snapshot.
        return self._status
