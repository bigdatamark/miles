# [MODULE DOC] Non-blocking evaluation dispatch for the training loop. One dispatcher class keeps a
# queue of in-flight evaluation tasks, optionally exports a Hugging Face snapshot of the trainer
# weights per evaluation point, and garbage-collects consumed snapshots.
import asyncio
import logging
import os
import shutil
import time
from collections import deque

logger = logging.getLogger(__name__)


class EvalDispatcher:
    # [CLASS DOC] Holds args, the trainer actor model handle (for snapshot export and offload
    # control), the rollout manager handle (which runs the evaluation and reports skips), a queue
    # "pending" of (rollout id, asyncio task, exported snapshot dir or nothing) in dispatch order,
    # and a list of exported snapshot directories awaiting garbage collection (oldest first).
    # Must be used from a single asyncio event loop; the queue is not thread safe.
    """Dispatches eval points without holding up the training loop.

    Exports a snapshot per point, fires the eval, and reclaims the snapshot once the
    point settles. Failures degrade to a skipped point outside CI; under ``--ci-test``,
    every skipped point fails the run. Shared-engine eval takes the plain blocking call
    instead.
    """

    def __init__(self, args, actor_model, rollout_manager):
        # [FUNCTION DOC] Store the three collaborators and initialize an empty pending queue and an
        # empty exported-snapshot list.
        self.args = args
        self.actor_model = actor_model
        self.rollout_manager = rollout_manager
        self.pending: deque[tuple[int, asyncio.Task, str | None]] = deque()
        self._exported: list[str] = []

    async def dispatch(self, rollout_id: int, hf_dir: str | None = None, force: bool = False) -> None:
        # [FUNCTION DOC] Start evaluation for one rollout id without waiting for it to finish.
        # If snapshot-based evaluation is disabled in args, simply awaits the rollout manager's
        # blocking evaluation and returns. Otherwise: first settle any already-finished pending
        # points. If the in-flight count is at the configured maximum: under the skip overflow
        # policy and not forced, report a skip with reason "busy" and return; otherwise wait for
        # the oldest pending point to settle. Then choose the snapshot directory: a caller-supplied
        # directory is used as-is (an existing checkpoint, never garbage collected, and the
        # completion-marker requirement is turned off); else if no
        # eval directory is configured, use the save-HF path template with the rollout id filled in
        # (a completion marker is required, and the snapshot is not owned by this class); else
        # export a snapshot into a "step_<rollout id>" subdirectory of the eval directory, timing
        # the export; on export failure log an error, report skip with reason "export_failed" and
        # return (nothing queued). Finally schedule the manager's evaluation as a background task
        # (passing directory, export time and the marker requirement flag) and append it to the
        # pending queue together with the exported directory (nothing unless this call exported).
        """A caller-supplied ``hf_dir`` is an existing checkpoint, not one of our exports."""
        if not self.args.eval_uses_snapshots:
            await self.rollout_manager.eval(rollout_id)
            return

        await self._reap_finished()
        if len(self.pending) >= self.args.eval_max_in_flight:
            if self.args.eval_overflow_policy == "skip" and not force:
                await self.rollout_manager.report_eval_skip(rollout_id, "busy")
                return
            await self._settle(*self.pending.popleft())

        export_time = None
        exported_dir = None
        require_marker = hf_dir is None
        if hf_dir is None:
            if self.args.eval_hf_dir is None:
                hf_dir = self.args.save_hf.format(rollout_id=rollout_id)
            else:
                hf_dir = os.path.join(self.args.eval_hf_dir, f"step_{rollout_id}")
                try:
                    export_time = await self._export(rollout_id, hf_dir)
                except Exception as e:
                    logger.error(f"HF snapshot export for eval {rollout_id} failed: {e}")
                    await self.rollout_manager.report_eval_skip(rollout_id, "export_failed")
                    return
                exported_dir = hf_dir

        task = asyncio.ensure_future(
            self.rollout_manager.eval(
                rollout_id, hf_dir=hf_dir, export_time_seconds=export_time, require_marker=require_marker
            )
        )
        self.pending.append((rollout_id, task, exported_dir))

    async def drain(self) -> None:
        # [FUNCTION DOC] Wait for every pending evaluation, oldest first, settling each (errors are
        # converted to reported skips and snapshots retired). Leaves the queue empty.
        while self.pending:
            await self._settle(*self.pending.popleft())

    async def _export(self, rollout_id: int, hf_dir: str) -> float:
        # [FUNCTION DOC] HELPER: Export the trainer weights as a Hugging Face checkpoint into the
        # given directory. If training offload is enabled, first onload the model to GPU and always
        # offload it again afterwards (even on failure). Returns wall-clock seconds taken, including
        # onload/offload. Propagates export exceptions.
        start = time.time()
        if self.args.offload_train:
            await self.actor_model.onload()
        try:
            await self.actor_model.export_hf(rollout_id, hf_dir)
        finally:
            if self.args.offload_train:
                await self.actor_model.offload()
        return time.time() - start

    async def _reap_finished(self) -> None:
        # [FUNCTION DOC] HELPER: Settle pending points from the front of the queue for as long as the
        # oldest one is already done; stops at the first unfinished task (later finished tasks
        # behind it are not reaped).
        while self.pending:
            if not self.pending[0][1].done():
                break
            await self._settle(*self.pending.popleft())

    async def _settle(self, rollout_id: int, task: asyncio.Task, exported_dir: str | None) -> None:
        # [FUNCTION DOC] HELPER: Await one dispatched evaluation task. An ordinary exception is logged
        # with traceback and converted into a skip report with reason "crashed" (not re-raised;
        # cancellation and other base exceptions propagate). In all cases the snapshot is retired.
        try:
            await task
        except Exception:
            logger.exception(f"Async eval for rollout {rollout_id} raised")
            await self.rollout_manager.report_eval_skip(rollout_id, "crashed")
        finally:
            self._retire(exported_dir)

    def _retire(self, exported_dir: str | None) -> None:
        # [FUNCTION DOC] HELPER: Record a snapshot this class exported as consumed (ignored when
        # nothing was exported by us). Then, while more snapshots are recorded than the configured
        # number to keep, remove the oldest from the list and delete its directory tree from disk
        # ignoring errors, logging each deletion.
        """Every settled point retires exactly one snapshot, whatever its outcome."""
        if exported_dir is None:
            return
        self._exported.append(exported_dir)
        while len(self._exported) > self.args.eval_keep_snapshots:
            victim = self._exported.pop(0)
            shutil.rmtree(victim, ignore_errors=True)
            logger.info(f"GC'd consumed eval snapshot {victim}")
