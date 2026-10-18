# [MODULE DOC]
# Training profiling support: a facade that creates and drives a torch profiler for the whole
# training run and for iterated loops (actor train, log probs), and memory profilers (torch cuda
# memory history with an out-of-memory dump hook, or memray). Targets are selected by names in the
# profile target argument; all outputs go to directories from args and are named by rank.
import logging
import time
import traceback
from pathlib import Path

import torch

from miles.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


class TrainProfiler:
    # [CLASS DOC] Per-process profiler facade. Creates the overall torch profiler (when pytorch
    # profiling is on and the overall target is selected) and the overall memory profiler (when
    # memory history recording is on and the overall target is selected; started right away).
    def __init__(self, args):
        # [FUNCTION DOC] Stores args and creates the overall profilers as described; needs an
        # initialized distributed world (rank is used in file names).
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None

        if args.use_pytorch_profiler and ("train_overall" in args.profile_target):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")

        if args.record_memory_history and ("train_overall" in args.profile_target):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def on_init_end(self):
        # [FUNCTION DOC] Starts the overall torch profiler if it exists (called after initialization).
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()

    def step(self, rollout_id: int):
        # [FUNCTION DOC] Advances the overall torch profiler schedule by one step; and stops (dumps)
        # the memory profiler when the rollout id equals the configured snapshot step count minus one.
        # Stopping is not guarded against repeating if the id recurs.
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def iterate_train_actor(self, iterator):
        # [FUNCTION DOC] Wraps the iterator in a generator profiling the actor train loop.
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        # [FUNCTION DOC] Wraps the iterator in a generator profiling the log probs loop.
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    # [FUNCTION DOC] HELPER: Generator. Without profiling enabled for this name, passes items through
    # untouched. Otherwise creates and starts a torch profiler and steps it after each consumed item.
    # KNOWN ISSUE: the profiler is never stopped, so traces are only written through the schedule's
    # trace-ready callback.
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name):
    # [FUNCTION DOC] HELPER: Torch profiler with a schedule of: wait for start step minus one steps
    # (at least 0), one warmup step if the start step is positive, active for end minus start steps,
    # one repeat. Traces are written gzip-compressed to the tensorboard directory by a trace handler
    # with worker name equal to the profile name plus the distributed rank. Records shapes, stacks,
    # memory and flops.
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


class _BaseMemoryProfiler:
    # [CLASS DOC] HELPER: Base of memory profilers with the dump path (snapshot directory, then a
    # file name made of the creation time, the rank and the configured snapshot name).
    @staticmethod
    def create(args):
        # [FUNCTION DOC] HELPER: Factory choosing the torch or memray profiler by the recorder name
        # in args (key error for unknown names).
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args):
        # [FUNCTION DOC] HELPER: Computes the dump path using the current wall time and rank.
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )

    def start(self):
        # [FUNCTION DOC] HELPER: Abstract; begin recording.
        raise NotImplementedError

    def stop(self):
        # [FUNCTION DOC] HELPER: Abstract; stop recording and dump.
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    # [CLASS DOC] HELPER: Memory profiler using torch's cuda allocator history.
    def start(self):
        # [FUNCTION DOC] HELPER: Starts recording allocator history (up to one million entries, all
        # stacks) and attaches an out-of-memory observer that logs, prints the stack, dumps a
        # snapshot to the dump path and prints memory usage.
        logger.info("Attach OOM dump memory history.")

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            # record stack information for the trace events
            # trace_alloc_record_context=True,
            stacks="all",
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # [FUNCTION DOC] HELPER: Callback invoked by torch on a cuda out-of-memory event: logs the
            # numbers, prints the stack, dumps the snapshot and prints memory info.
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            torch.cuda.memory._dump_snapshot(self._path_dump)
            print_memory("when oom")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        # [FUNCTION DOC] HELPER: Dumps a snapshot to the dump path and disables history recording (the
        # out-of-memory observer stays attached).
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        torch.cuda.memory._dump_snapshot(self._path_dump)
        torch.cuda.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    # [CLASS DOC] HELPER: Memory profiler using the memray tracker with native traces.
    def __init__(self, args):
        # [FUNCTION DOC] HELPER: Base setup; asserts the snapshot step count is provided (otherwise it
        # would never stop).
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        # [FUNCTION DOC] HELPER: Imports memray lazily, creates a tracker writing to the dump path with
        # native traces, and enters it manually (kept until stop).
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        # [FUNCTION DOC] HELPER: Exits the tracker, finalizing the dump file.
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
