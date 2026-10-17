# [MODULE DOC]
# Submission schedulers that decide when a rollout driver may submit more prompt groups for
# generation. Two policies share the same informal interface (a done-callback attribute, a capacity
# check, a submit notification, and a wait for progress): group level (a slot frees when a whole
# group finishes) and sample level backfill (a slot frees as each sample finishes). A factory
# picks the policy from the launch arg, with a per-caller default.
"""Paces prompt-group submission for the rollout drivers (--rollout-submission-granularity).

The driver owns the pending group tasks and passes their count in as ``pending_groups``;
the scheduler only accounts samples in flight.
"""

import asyncio
import logging
from argparse import Namespace

from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class GroupLevelSubmission:
    # [CLASS DOC] Stateless policy: capacity is simply fewer pending group tasks than the group
    # budget. The sample done callback class attribute is none, meaning generation need not report
    # per-sample completion.
    """A submission slot frees only when the whole group task returns."""

    sample_done_callback = None

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        # [FUNCTION DOC] True when the number of pending group tasks is below the budget.
        return pending_groups < group_budget

    def on_submit(self, groups: list[list[Sample]]) -> None:
        # [FUNCTION DOC] No-op: nothing to account for at submission time.
        pass

    async def wait_for_progress(self, pendings: set[asyncio.Task]) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        # [FUNCTION DOC] Wait until at least one pending group task finishes; returns the pair
        # (finished tasks, still-pending tasks). The set must be non-empty (the library call
        # rejects an empty set).
        return await asyncio.wait(pendings, return_when=asyncio.FIRST_COMPLETED)


class SampleBackfillSubmission:
    # [CLASS DOC] Tracks the number of samples in flight (submitted but not completed) and a
    # wake-up event set whenever a sample completes. Capacity exists when the in-flight count plus
    # one more group's size stays within the budget times the group size, so a replacement group is
    # submitted as soon as enough individual samples finished.
    """Each finished sample frees its own slot; a replacement group fits once ``group_size`` samples complete."""

    def __init__(self, group_size: int):
        # [FUNCTION DOC] Store the samples-per-group size, start with zero samples in flight and
        # a cleared completion event.
        self.group_size = group_size
        self.samples_in_flight = 0
        self._sample_done = asyncio.Event()

    def sample_done_callback(self) -> None:
        # [FUNCTION DOC] Called once per finished sample (including failures and cancellation):
        # decrement the in-flight count and set the wake-up event.
        self.samples_in_flight -= 1
        self._sample_done.set()

    def has_capacity(self, *, pending_groups: int, group_budget: int) -> bool:
        # [FUNCTION DOC] Self-healing step: if no group tasks are pending but the counter is still
        # positive (a group ended without spawning sample tasks, or the callback was skipped), log
        # a warning and reset the counter to zero. Then return true if in-flight samples plus one
        # group's size fit within budget times group size. Otherwise clear the wake-up event
        # (so the next wait only wakes on a later completion) and return false.
        """A False return arms the sample wakeup: only later completions wake ``wait_for_progress``."""
        if pending_groups == 0 and self.samples_in_flight:
            # a group returned without spawning its sample tasks, or a stub skipped the callback
            logger.warning(f"samples_in_flight={self.samples_in_flight} with no pending groups; resetting to 0")
            self.samples_in_flight = 0
        if self.samples_in_flight + self.group_size <= group_budget * self.group_size:
            return True
        self._sample_done.clear()
        return False

    def on_submit(self, groups: list[list[Sample]]) -> None:
        # [FUNCTION DOC] Add the total number of samples across the submitted groups to the
        # in-flight count.
        self.samples_in_flight += sum(len(group) for group in groups)

    async def wait_for_progress(self, pendings: set[asyncio.Task]) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        # [FUNCTION DOC] Wait until either a pending group task finishes or the sample-done event
        # fires (via a helper task that waits on the event, cancelled afterwards even on error).
        # Returns (finished group tasks, still-pending group tasks) with the helper task removed
        # from both, so the finished set may be empty when only a sample completed. Note the
        # event stays set until the capacity check clears it, so repeated calls without that check
        # return immediately.
        waiter = asyncio.create_task(self._sample_done.wait())
        try:
            done, pending = await asyncio.wait(pendings | {waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
        return done - {waiter}, pending - {waiter}


def make_submission_scheduler(args: Namespace, *, default: str) -> GroupLevelSubmission | SampleBackfillSubmission:
    # [FUNCTION DOC] Create the scheduler named by the submission granularity launch arg, or by the
    # caller's default string when the arg is unset/empty. "group" gives the group-level policy;
    # "sample" gives the backfill policy sized by samples-per-prompt; anything else fails an
    # assertion with the unknown value.
    granularity = args.rollout_submission_granularity or default
    if granularity == "group":
        return GroupLevelSubmission()
    assert granularity == "sample", f"unknown submission granularity: {granularity}"
    return SampleBackfillSubmission(args.n_samples_per_prompt)
