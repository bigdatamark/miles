# [MODULE DOC]
# Reloadable process groups, for freeing and re-creating nccl communicators (for example around
# colocated offloading). A monkey patch replaces the group creation function of torch distributed so
# that every non-gloo group with more than one rank is wrapped in a group subclass that remembers its
# creation arguments, can destroy its inner group and later recreate it. Many distributed functions
# are patched to unwrap these wrappers and to attach memory info to error notes. State is kept per
# process id (module globals), so forked children patch independently.
import logging
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)

old_new_group_dict = {}


def monkey_patch_torch_dist():
    # [FUNCTION DOC] Idempotent per process (asserts the saved original is still the one stored when
    # called again). Saves the original group creation function (also as an attribute on the module),
    # then replaces it with a wrapper: it creates the real group; gloo groups (backend as third
    # positional or keyword argument) and single-rank groups are returned as is; others are wrapped.
    # Ranks come from the first positional argument, the ranks keyword, or default to the whole
    # world. It also replaces rank/size/backend/global-rank queries, all collectives, send/receive,
    # the coalescing manager and non-blocking point-to-point functions with wrappers that unwrap
    # reloadable group arguments (positional and keyword) and run the call inside the error-note
    # context. Point-to-point operation objects get their construction patched to unwrap groups and
    # to map the patched send/receive functions back to the originals. Mutates torch.distributed
    # globally.
    # KNOWN ISSUE: the point-to-point conversion compares every argument with the patched functions
    # by equality, which for tensor arguments performs an elementwise comparison.
    pid = os.getpid()
    if pid in old_new_group_dict:
        assert dist.old_new_group == old_new_group_dict[pid]
        return

    logger.info("Applying monkey patch to torch.distributed")

    old_new_group = dist.new_group
    old_new_group_dict[pid] = old_new_group
    dist.old_new_group = old_new_group

    def new_group(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Replacement for the group creation function; see the patch function.
        group = old_new_group(*args, **kwargs)
        # skip none nccl group.
        if len(args) >= 3 and args[2] == "gloo" or "backend" in kwargs and kwargs["backend"] == "gloo":
            return group

        # Get ranks from arguments
        if len(args) >= 1 and args[0] is not None:
            ranks = args[0]
        elif "ranks" in kwargs and kwargs["ranks"] is not None:
            ranks = kwargs["ranks"]
        else:
            # If no ranks specified, use all ranks in world
            ranks = list(range(dist.get_world_size()))
        if len(ranks) == 1:
            return group

        group = ReloadableProcessGroup(group, inner_args=args, inner_kwargs=kwargs)
        return group

    dist.new_group = new_group

    def get_new_function(func):
        # [FUNCTION DOC] HELPER: Builds a wrapper of the given torch function that replaces reloadable
        # groups in positional and keyword arguments by their inner group and calls the original
        # inside the error-note context.
        def new_function(*args, **kwargs):
            # [FUNCTION DOC] HELPER: The wrapper itself.
            args = tuple([arg.group if isinstance(arg, ReloadableProcessGroup) else arg for arg in args])
            kwargs = {k: (v.group if isinstance(v, ReloadableProcessGroup) else v) for k, v in kwargs.items()}
            with _wrap_low_level_call():
                return func(*args, **kwargs)

        return new_function

    dist.get_rank = get_new_function(dist.get_rank)
    dist.get_world_size = get_new_function(dist.get_world_size)
    dist.get_backend = get_new_function(dist.get_backend)
    dist.get_global_rank = get_new_function(dist.get_global_rank)
    dist.get_group_rank = get_new_function(dist.get_group_rank)
    dist.get_process_group_ranks = get_new_function(dist.get_process_group_ranks)

    dist.all_reduce = get_new_function(dist.all_reduce)
    dist.all_gather = get_new_function(dist.all_gather)
    dist.all_gather_into_tensor = get_new_function(dist.all_gather_into_tensor)
    dist.all_gather_object = get_new_function(dist.all_gather_object)
    dist.all_to_all = get_new_function(dist.all_to_all)
    dist.all_to_all_single = get_new_function(dist.all_to_all_single)
    dist.broadcast = get_new_function(dist.broadcast)
    dist.broadcast_object_list = get_new_function(dist.broadcast_object_list)
    dist.reduce = get_new_function(dist.reduce)
    dist.reduce_scatter = get_new_function(dist.reduce_scatter)
    dist.reduce_scatter_tensor = get_new_function(dist.reduce_scatter_tensor)
    dist.scatter = get_new_function(dist.scatter)
    dist.gather = get_new_function(dist.gather)
    dist.barrier = get_new_function(dist.barrier)
    dist.send = get_new_function(dist.send)
    dist.recv = get_new_function(dist.recv)
    dist._coalescing_manager = get_new_function(dist._coalescing_manager)

    # p2p
    old_isend = dist.isend
    old_irecv = dist.irecv

    dist.isend = get_new_function(dist.isend)
    dist.irecv = get_new_function(dist.irecv)

    def get_new_p2pop_function(func):
        # [FUNCTION DOC] HELPER: Wraps a point-to-point operation constructor so arguments are
        # converted before the original is called.
        def new_function(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Converts all arguments and calls the original (no error-note
            # context here).
            def convert(arg):
                # [FUNCTION DOC] HELPER: Reloadable group becomes its inner group; the patched send
                # or receive function becomes the saved original; others unchanged.
                if isinstance(arg, ReloadableProcessGroup):
                    return arg.group
                elif arg == dist.isend:
                    arg = old_isend
                elif arg == dist.irecv:
                    arg = old_irecv
                return arg

            args = (convert(arg) for arg in args)
            kwargs = {k: convert(v) for k, v in kwargs.items()}
            return func(*args, **kwargs)

        return new_function

    dist.P2POp.__new__ = get_new_p2pop_function(dist.P2POp.__new__)
    dist.P2POp.__init__ = get_new_p2pop_function(dist.P2POp.__init__)


class ReloadableProcessGroup(torch.distributed.ProcessGroup):
    # [CLASS DOC] Process group proxy holding an inner real group (none after destruction) and the
    # arguments used to create it. Instances are registered per process id in a class-level table so
    # all can be destroyed or reloaded together. Collectives and most methods forward to the inner
    # group (explicit methods below; the rest are generated after the class definition) and raise a
    # runtime error while the inner group is gone. Unknown attributes fall back to the inner group.
    GROUPS = {}

    def __init__(self, group, inner_args, inner_kwargs):
        # [FUNCTION DOC] Initializes the base group with the inner group's rank and size, stores the
        # inner group and creation arguments, and registers itself for this process id.
        super().__init__(
            rank=dist.get_rank(group),
            size=dist.get_world_size(group),
        )
        self.group = group
        self.inner_args = inner_args
        self.inner_kwargs = inner_kwargs
        pid = os.getpid()
        if pid not in ReloadableProcessGroup.GROUPS:
            ReloadableProcessGroup.GROUPS[pid] = []
        ReloadableProcessGroup.GROUPS[pid].append(self)

    def __getattr__(self, name):
        # [FUNCTION DOC] Fallback attribute lookup on the inner group.
        return getattr(self.group, name)

    @staticmethod
    def destroy_process_groups():
        # [FUNCTION DOC] Destroys the inner group of every registered wrapper of this process that
        # still has one. A value error (already invalid) is logged as a warning; afterwards the inner
        # reference is dropped and set to none either way.
        pid = os.getpid()
        for reloadable_group in ReloadableProcessGroup.GROUPS.get(pid, []):
            if reloadable_group.group is None:
                continue
            try:
                dist.destroy_process_group(reloadable_group.group)
            except ValueError as e:
                logger.warning(
                    f"Process group already invalid/destroyed; skipping cleanup. Exception: {e}",
                    exc_info=True,
                )

            del reloadable_group.group
            reloadable_group.group = None

    @staticmethod
    def reload_process_groups():
        # [FUNCTION DOC] Recreates the inner group for each registered wrapper of this process whose
        # inner group is none, using the saved original creation function and the saved arguments.
        # Logs the count. Fails (not callable) if the monkey patch never ran in this process.
        pid = os.getpid()
        reloadable_groups = ReloadableProcessGroup.GROUPS.get(pid, [])
        logger.info(f"Reloading {len(reloadable_groups)} process groups in pid {pid}")
        old_new_group = old_new_group_dict.get(pid)
        for reloadable_group in reloadable_groups:
            if reloadable_group.group is not None:
                continue
            group = old_new_group(*reloadable_group.inner_args, **reloadable_group.inner_kwargs)
            reloadable_group.group = group

    def rank(self) -> int:
        # [FUNCTION DOC] Rank from the inner group (fails if destroyed).
        return self.group.rank()

    def size(self) -> int:
        # [FUNCTION DOC] Size from the inner group (fails if destroyed).
        return self.group.size()

    def name(self) -> str:
        # [FUNCTION DOC] Name from the inner group (fails if destroyed).
        return self.group.name()

    def shutdown(self) -> None:
        # [FUNCTION DOC] Shuts down the inner group if present; otherwise nothing.
        if self.group is not None:
            self.group.shutdown()

    def abort(self) -> None:
        # [FUNCTION DOC] Aborts the inner group if present; otherwise nothing.
        if self.group is not None:
            self.group.abort()

    def _fwd(self, method, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the named method on the inner group inside the error-note
        # context; raises a runtime error saying to reload when the inner group is none.
        inner = self.group
        if inner is None:
            raise RuntimeError("ReloadableProcessGroup: inner PG is None, call reload() first.")
        with _wrap_low_level_call():
            return getattr(inner, method)(*args, **kwargs)

    def barrier(self, *a, **kw):
        # [FUNCTION DOC] Forwards to the inner group's method of the same name (all explicit forwarders
        # below do likewise, returning the inner result, usually an asynchronous work handle).
        return self._fwd("barrier", *a, **kw)

    def broadcast(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("broadcast", *a, **kw)

    def allreduce(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("allreduce", *a, **kw)

    def allreduce_coalesced(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("allreduce_coalesced", *a, **kw)

    def reduce(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("reduce", *a, **kw)

    def allgather(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("allgather", *a, **kw)

    def _allgather_base(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("_allgather_base", *a, **kw)

    def allgather_coalesced(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("allgather_coalesced", *a, **kw)

    def allgather_into_tensor_coalesced(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("allgather_into_tensor_coalesced", *a, **kw)

    def gather(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("gather", *a, **kw)

    def scatter(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("scatter", *a, **kw)

    def reduce_scatter(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("reduce_scatter", *a, **kw)

    def _reduce_scatter_base(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("_reduce_scatter_base", *a, **kw)

    def reduce_scatter_tensor_coalesced(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("reduce_scatter_tensor_coalesced", *a, **kw)

    def alltoall_base(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("alltoall_base", *a, **kw)

    def alltoall(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("alltoall", *a, **kw)

    def send(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("send", *a, **kw)

    def recv(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("recv", *a, **kw)

    def recv_anysource(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("recv_anysource", *a, **kw)

    def _start_coalescing(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("_start_coalescing", *a, **kw)

    def _end_coalescing(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("_end_coalescing", *a, **kw)

    def _get_backend_name(self):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("_get_backend_name")

    def _get_backend(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("_get_backend", *a, **kw)

    def _set_default_backend(self, *a, **kw):
        # [FUNCTION DOC] Forwarder.
        return self._fwd("_set_default_backend", *a, **kw)

    @property
    def bound_device_id(self):
        # [FUNCTION DOC] Property: the inner group's bound device id.
        return self.group.bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, dev):
        # [FUNCTION DOC] Setter: sets the bound device id on the inner group.
        self.group.bound_device_id = dev


def _forward_remaining_collectives():
    # [FUNCTION DOC] HELPER: Runs once at import. For every callable public-or-single-underscore
    # attribute of the base process group class that the wrapper class does not define itself and
    # that is not in the skip set (rank, size, name, abort, shutdown, bound device id), attaches a
    # forwarder method to the wrapper class that calls the inner group's method of that name through
    # the forwarding helper. Mutates the wrapper class.
    """Forward every ProcessGroup collective this class does not define itself.

    Callers that resolved a collective before the monkey patch went on --
    Megatron binds `dist_reduce_scatter_func` at import time -- hand the wrapper
    straight to torch, which invokes the method on the group object. Anything
    not overridden here reaches the C++ base, whose own backend map is empty,
    and dies as "No backend type associated with device type cuda". A
    hand-written forward list silently regrows that hole whenever torch renames
    or adds a collective, which is how torch 2.13's *_single family got through.
    """
    skip = {"rank", "size", "name", "abort", "shutdown", "bound_device_id"}
    for name in dir(dist.ProcessGroup):
        if name.startswith("__") or name in skip:
            continue
        if name in vars(ReloadableProcessGroup):
            continue
        if not callable(getattr(dist.ProcessGroup, name, None)):
            continue

        def make(method):
            # [FUNCTION DOC] HELPER: Builds the forwarder for one method name (named after it).
            def forward(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Calls the inner group's method through the forwarding helper.
                return self._fwd(method, *args, **kwargs)

            forward.__name__ = method
            return forward

        setattr(ReloadableProcessGroup, name, make(name))


_forward_remaining_collectives()


def destroy_process_groups():
    # [FUNCTION DOC] Module-level alias for destroying all reloadable groups of this process.
    """Destroy all reloadable process groups."""
    ReloadableProcessGroup.destroy_process_groups()


def reload_process_groups():
    # [FUNCTION DOC] Module-level alias for reloading all reloadable groups of this process.
    """Reload all reloadable process groups."""
    ReloadableProcessGroup.reload_process_groups()


@contextmanager
def _wrap_low_level_call():
    # [FUNCTION DOC] HELPER: Context manager around low-level distributed calls. On any ordinary
    # exception prints gpu memory information (logged, requires an initialized world), attaches it as
    # a note to the exception when supported, and re-raises the original exception. Yields nothing.
    try:
        yield
    except Exception as e:
        mem_info = print_memory("after torch distributed error")
        if hasattr(e, "add_note"):
            e.add_note(f"{mem_info=}")
        raise
