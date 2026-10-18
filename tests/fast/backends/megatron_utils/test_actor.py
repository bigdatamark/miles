# [MODULE DOC] CPU unit tests of selected pieces of the Megatron train actor module: that the
# critic-values value spec uses the typed ragged codec, that materializing critic values yields
# owned float32 tensors from either list or tensor inputs, and that sending a checkpoint to a
# healing peer before the first train step is refused without starting a transfer. The actor module
# is imported through a module-scoped fixture that temporarily stubs the unavailable native memory-saver
# dependency and restores the module table afterwards.
import importlib
import sys
from argparse import Namespace
from collections.abc import Iterator
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

_ACTOR_MODULE_NAME = "miles.backends.megatron_utils.actor"


@pytest.fixture(scope="module")
def actor_module() -> Iterator[ModuleType]:
    # [FUNCTION DOC] HELPER: Module-scoped generator fixture. Setup: remembers the current entries for
    # the actor module and the memory-saver module in the module table and the actor attribute of the
    # megatron_utils package (or a missing marker), installs a fake memory-saver module whose saver is a
    # mock, removes any cached actor module and package attribute, then yields a fresh import of the
    # actor module. Teardown (always): removes the freshly imported module and restores or deletes
    # the previous module table entries and the package attribute so no state leaks to other tests.
    """Import the Megatron actor with its unavailable native memory dependency stubbed."""
    package = importlib.import_module("miles.backends.megatron_utils")
    missing = object()
    saved_module = sys.modules.get(_ACTOR_MODULE_NAME, missing)
    saved_saver = sys.modules.get("torch_memory_saver", missing)
    saved_package_attr = getattr(package, "actor", missing)

    saver_module = ModuleType("torch_memory_saver")
    saver_module.torch_memory_saver = Mock()
    sys.modules["torch_memory_saver"] = saver_module
    sys.modules.pop(_ACTOR_MODULE_NAME, None)
    if saved_package_attr is not missing:
        delattr(package, "actor")

    try:
        yield importlib.import_module(_ACTOR_MODULE_NAME)
    finally:
        sys.modules.pop(_ACTOR_MODULE_NAME, None)
        if saved_module is not missing:
            sys.modules[_ACTOR_MODULE_NAME] = saved_module
        if saved_package_attr is missing:
            if hasattr(package, "actor"):
                delattr(package, "actor")
        else:
            package.actor = saved_package_attr
        if saved_saver is missing:
            sys.modules.pop("torch_memory_saver", None)
        else:
            sys.modules["torch_memory_saver"] = saved_saver


class TestCriticValuesValueSpec:
    # [CLASS DOC] Tests of the actor module's critic-values value specification.
    def test_critic_values_are_shipped_as_a_typed_ragged_field(self, actor_module: ModuleType) -> None:
        # [FUNCTION DOC] Asserts the "values" field of the critic-values spec uses the codec named
        # typed_ragged.
        """Variable-length critic sequences require the typed ragged object-store codec."""
        assert actor_module.CRITIC_VALUES_VALUE_SPEC["values"].codec == "typed_ragged"


class TestMaterializeCriticValues:
    # [CLASS DOC] Tests of the actor module's helper that turns stored ragged critic values into tensors.
    @pytest.mark.parametrize("tensor_input", [False, True])
    def test_ragged_values_become_owned_float32_tensors(self, actor_module: ModuleType, tensor_input: bool) -> None:
        # [FUNCTION DOC] Parametrized over inputs given as nested Python lists or as float32 tensors:
        # three sequences (two values, empty, one value). Calls the helper for the CPU device, then
        # mutates the first input element afterwards; asserts the result lists still hold the original
        # values (so the outputs own their storage, not views of the inputs), and every result is float32
        # on the CPU.
        """Both store codecs yield owned FP32 tensors independent of released storage."""
        sequences = [[1.5, -2.0], [], [3.0]]
        values = [torch.tensor(value, dtype=torch.float32) for value in sequences] if tensor_input else sequences

        result = actor_module._materialize_critic_values(values=values, device=torch.device("cpu"))
        values[0][0] = 99.0

        assert [value.tolist() for value in result] == [[1.5, -2.0], [], [3.0]]
        assert all(value.dtype == torch.float32 for value in result)
        assert all(value.device == torch.device("cpu") for value in result)


class TestSendCheckpoint:
    # [CLASS DOC] Tests of the Megatron actor's send-checkpoint entry point.
    def test_healing_before_the_first_train_step_is_refused_without_sending(
        self, actor_module: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] Builds an actor without its constructor with no last rollout id (no train step
        # yet) and placeholder model, optimizer and scheduler; patches the parallel-state getter to
        # return an object with an independent-DP entry and the internal checkpoint sender to record
        # that it was called. Asserts sending to destination rank 1 raises an assertion error saying
        # healing before the first train step is unsupported, and that no transfer was attempted.
        """Healing before the first train step is refused without transferring a checkpoint."""
        train_actor = object.__new__(actor_module.MegatronTrainRayActor)
        train_actor.args = Namespace(keep_old_actor=False)
        train_actor._last_rollout_id = None
        train_actor.model = object()
        train_actor.optimizer = object()
        train_actor.opt_param_scheduler = object()
        checkpoint_transfer_attempted = False

        def record_checkpoint_transfer(**_kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: Replacement checkpoint sender; ignores its keyword arguments and
            # sets the enclosing test's attempted flag to true.
            nonlocal checkpoint_transfer_attempted
            checkpoint_transfer_attempted = True

        monkeypatch.setattr(actor_module, "get_parallel_state", lambda: SimpleNamespace(indep_dp=object()))
        monkeypatch.setattr(actor_module, "_send_ckpt", record_checkpoint_transfer)

        with pytest.raises(AssertionError, match="healing before the first train step is unsupported"):
            train_actor.send_ckpt(dst_rank=1)

        assert not checkpoint_transfer_attempted
