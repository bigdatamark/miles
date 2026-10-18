# [MODULE DOC] CPU test that the FSDP train actor's save-model method delegates to the checkpoint
# module's save function with the actor and the step id.
from types import SimpleNamespace
from unittest.mock import Mock

from miles.backends.fsdp_utils import actor as actor_module


def test_save_model_delegates_to_checkpoint(monkeypatch):
    # [FUNCTION DOC] Test: builds an actor without its constructor, with minimal args (save path,
    # no async save, not rollout-only), patches the checkpoint module's save with a mock, calls
    # save_model with step 7 and asserts the mock was called once with (actor, 7).
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = SimpleNamespace(debug_rollout_only=False, save="/tmp/checkpoint", async_save=False)
    save = Mock()
    monkeypatch.setattr(actor_module.checkpoint, "save", save)

    actor.save_model(7)

    save.assert_called_once_with(actor, 7)
