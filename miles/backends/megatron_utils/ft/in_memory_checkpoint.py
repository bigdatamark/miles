# [MODULE DOC]
# In-memory stand-in for Megatron's local (non-persistent) checkpoint manager, used for fault
# tolerance: the manager keeps one state dict object in memory, and a helper runs Megatron save
# against it to obtain a state dict snapshot. Requires the Megatron args to select the local
# non-persistent checkpoint type with an algorithm set.
import logging
from collections.abc import Sequence
from typing import Any

from megatron.training.global_vars import get_args

from miles.backends.megatron_utils.model import save

logger = logging.getLogger(__name__)


class InMemoryCheckpointManager:
    # [CLASS DOC]
    # Duck-typed replacement for the resiliency extension's local checkpoint manager exposing just
    # the interface Megatron uses: a latest-iteration attribute (-1 when empty), a fake local
    # checkpoint directory string, save, find-latest and load. Holds a single state dict reference
    # (no copy) and can be saved to only once per instance.
    """ref: nvidia_resiliency_ext's LocalCheckpointManager."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes empty state (no state dict, latest iteration -1, directory
        # label "<in-memory>") and validates the current Megatron global args via the args check
        # (assertion error if misconfigured).
        self.latest_iteration: int = -1
        self._state_dict: object = None
        self.local_ckpt_dir: str = "<in-memory>"

        _assert_args_for_in_memory_checkpoint(get_args())

    def save(self, state_dict: object, iteration: int, is_async: bool = False) -> None:
        # [FUNCTION DOC] Stores the reference and iteration. Asserts the call is not asynchronous
        # and that nothing was stored yet (a second save on the same instance fails). Nothing is
        # copied or serialized.
        """Store state_dict object reference in memory."""
        assert not is_async

        assert self._state_dict is None
        self._state_dict = state_dict
        self.latest_iteration = iteration

    def find_latest(self) -> int:
        # [FUNCTION DOC] Returns the stored iteration, or -1 if nothing was saved.
        return self.latest_iteration

    def load(self) -> tuple[object, str]:
        # [FUNCTION DOC] Returns the stored state dict and a label string "in-memory-ckpt-iter-"
        # plus the iteration. Does not consume the state (Megatron calls load twice). Asserts a
        # checkpoint was saved.
        # Idempotent: Megatron's load_checkpoint calls _load_base_checkpoint twice
        # (once for format detection at line 1508, once for actual load at line 1712).
        # We must NOT consume `_state_dict` on first call.
        assert self.latest_iteration >= 0, "No in-memory checkpoint available"
        assert self._state_dict is not None
        return self._state_dict, f"in-memory-ckpt-iter-{self.latest_iteration}"


def save_to_memory(
    iteration: int,
    model: Sequence,
    optimizer: object,
    opt_param_scheduler: object,
) -> object:
    # [FUNCTION DOC]
    # Creates a fresh manager and runs the backend's save routine for the iteration with the
    # manager supplied as the local checkpoint manager in the checkpointing context and the
    # non-persistent flag set (nothing is written to disk). Then reads the manager's state back and
    # returns just the state dict (a reference to live tensors' metadata, not a deep copy).
    """Save checkpoint to in-memory manager via model.save (with forward hook protection)."""
    manager = InMemoryCheckpointManager()
    save(
        iteration=iteration,
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        checkpointing_context={"local_checkpoint_manager": manager},
        non_persistent_ckpt=True,
    )
    state_dict, _ = manager.load()
    return state_dict


def _assert_args_for_in_memory_checkpoint(args: Any) -> None:
    # [FUNCTION DOC] HELPER: Asserts the args select the "local" non-persistent checkpoint type and
    # have a non-none local checkpoint algorithm; raises assertion errors with explanatory messages.
    assert args.non_persistent_ckpt_type == "local", (
        f"Expected non_persistent_ckpt_type='local', " f"got {getattr(args, 'non_persistent_ckpt_type', None)!r}"
    )
    assert args.non_persistent_local_ckpt_algo is not None, "args.non_persistent_local_ckpt_algo must be set"
