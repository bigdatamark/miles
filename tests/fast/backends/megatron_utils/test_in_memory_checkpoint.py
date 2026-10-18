# [MODULE DOC] CPU unit tests of the fault-tolerance in-memory checkpoint manager: before any save
# the latest iteration is -1 and loading raises an assertion error; after a save the manager reports
# that iteration and load returns the very same state object plus a name containing the iteration,
# and load is idempotent (does not clear state); a second save without a reset raises an assertion
# error, even after a load; and asynchronous saving is rejected with an assertion error. The manager's
# global-arguments getter is patched to a local, fully-parallel non-persistent checkpoint config.
from unittest.mock import patch

import pytest

from miles.backends.megatron_utils.ft.in_memory_checkpoint import InMemoryCheckpointManager


@pytest.fixture()
def manager():
    # [FUNCTION DOC] HELPER: Generator fixture. While the module's arguments getter is patched to
    # return an object configured with non-persistent checkpoint type "local" and local algorithm
    # "fully_parallel", constructs and yields a fresh manager; the patch is removed on teardown.
    with patch("miles.backends.megatron_utils.ft.in_memory_checkpoint.get_args") as mock_get_args:
        mock_args = mock_get_args.return_value
        mock_args.non_persistent_ckpt_type = "local"
        mock_args.non_persistent_local_ckpt_algo = "fully_parallel"
        yield InMemoryCheckpointManager()


class TestInMemoryCheckpointManager:
    # [CLASS DOC] Tests of the manager's save, load and find-latest behavior (one checkpoint slot).
    def test_find_latest_returns_minus_one_initially(self, manager: InMemoryCheckpointManager):
        # [FUNCTION DOC] A fresh manager reports -1 as the latest iteration.
        assert manager.find_latest() == -1

    def test_load_before_save_raises(self, manager: InMemoryCheckpointManager):
        # [FUNCTION DOC] Loading from a fresh manager raises an assertion error saying there is no
        # in-memory checkpoint.
        with pytest.raises(AssertionError, match="No in-memory checkpoint"):
            manager.load()

    def test_save_then_load_returns_same_object(self, manager: InMemoryCheckpointManager):
        # [FUNCTION DOC] Saves a sentinel object at iteration 5; the latest iteration becomes 5 and
        # load returns the same object (identity, no copy) together with a name containing "5".
        sentinel = object()
        manager.save(state_dict=sentinel, iteration=5)

        assert manager.find_latest() == 5

        result, name = manager.load()
        assert result is sentinel
        assert "5" in name

    def test_load_is_idempotent_returns_same_state(self, manager: InMemoryCheckpointManager):
        # [FUNCTION DOC] Two consecutive loads after one save both return the saved object.
        sentinel = object()
        manager.save(state_dict=sentinel, iteration=1)

        first, _ = manager.load()
        second, _ = manager.load()
        assert first is sentinel
        assert second is sentinel

    def test_save_twice_without_reset_raises(self, manager: InMemoryCheckpointManager):
        # [FUNCTION DOC] A second save without a reset raises an assertion error.
        manager.save(state_dict=object(), iteration=1)

        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=2)

    def test_save_after_load_still_raises_load_does_not_reset(self, manager: InMemoryCheckpointManager):
        # [FUNCTION DOC] Save, then load, then save again: the second save still raises an assertion
        # error because load does not clear state.
        """`load()` is idempotent and does NOT clear state, so a second `save()`
        on the same manager — even after `load()` — must still raise."""
        manager.save(state_dict=object(), iteration=1)
        manager.load()

        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=2)

    def test_async_save_raises(self, manager: InMemoryCheckpointManager):
        # [FUNCTION DOC] Saving with the asynchronous flag raises an assertion error.
        with pytest.raises(AssertionError):
            manager.save(state_dict=object(), iteration=1, is_async=True)
