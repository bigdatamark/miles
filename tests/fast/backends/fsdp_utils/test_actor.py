# [MODULE DOC] CPU unit tests of the FSDP train actor: that initialization rejects a non-null
# independent data-parallel quorum store address (before doing real distributed work, using a
# stubbed debug dumper module, patched device/process-group calls, and an actor instance created
# without its constructor), and that training rejects externally supplied critic output before
# touching any state needed for training.
import sys
from argparse import Namespace
from types import ModuleType

import pytest

from miles.backends.fsdp_utils import actor as actor_module
from miles.backends.training_utils.types import TrainStepOutcome, TrainStepOutput
from miles.utils import distributed_utils
from miles.utils.ft_utils.heartbeat_utils import SimpleHeartbeat
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.init_once import InitOnce


class TestFSDPInit:
    # [CLASS DOC] Tests of the FSDP actor's init entry point.
    def test_fsdp_init_rejects_an_independent_dp_quorum_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] Installs a fake debug-dumper module whose source-patch application raises
        # a runtime error (so the test fails differently if init proceeds past the quorum check), and
        # patches the actor module's device selection, process-group init, rank (0), world size (1),
        # initialized flag (false) and group creation, plus the global gloo group (none). Creates an
        # actor without its constructor with a rank, heartbeat and init-once guard, builds an args
        # namespace (nccl backend, one GPU, dumper enabled), and asserts init with a trivial
        # independent-DP info and a non-null quorum store address raises an assertion error.
        """FSDP initialization rejects a non-null independent-DP quorum store address."""

        def fail_if_initialization_continues() -> None:
            # [FUNCTION DOC] HELPER: Always raises a runtime error saying initialization continued
            # past quorum validation.
            raise RuntimeError("FSDP initialization continued past quorum validation")

        def ignore_device(_device: object) -> None:
            # [FUNCTION DOC] HELPER: No-op replacement for GPU device selection.
            return None

        def ignore_process_group(**_kwargs: object) -> None:
            # [FUNCTION DOC] HELPER: No-op replacement for process-group initialization.
            return None

        def zero() -> int:
            # [FUNCTION DOC] HELPER: Replacement for the rank query; returns 0.
            return 0

        def one() -> int:
            # [FUNCTION DOC] HELPER: Replacement for the world-size query; returns 1.
            return 1

        def false() -> bool:
            # [FUNCTION DOC] HELPER: Replacement for the is-initialized query; returns false.
            return False

        class FakeDumper:
            # [CLASS DOC] HELPER: Fake dumper exposing only a source-patch application function that
            # fails the test if reached.
            apply_source_patches = staticmethod(fail_if_initialization_continues)

        dumper_module = ModuleType("sglang.srt.debug_utils.dumper")
        dumper_module.dumper = FakeDumper()
        monkeypatch.setitem(sys.modules, "sglang.srt.debug_utils.dumper", dumper_module)
        monkeypatch.setattr(actor_module.torch.cuda, "set_device", ignore_device)
        monkeypatch.setattr(actor_module.dist, "init_process_group", ignore_process_group)
        monkeypatch.setattr(actor_module.dist, "get_rank", zero)
        monkeypatch.setattr(actor_module.dist, "get_world_size", one)
        monkeypatch.setattr(actor_module.dist, "is_initialized", false)
        monkeypatch.setattr(actor_module.dist, "new_group", lambda **_kwargs: object())
        monkeypatch.setattr(distributed_utils, "GLOO_GROUP", None)

        actor = object.__new__(actor_module.FSDPTrainRayActor)
        actor._rank = 0
        actor._heartbeat = SimpleHeartbeat()
        actor._init_once = InitOnce(type(actor).__name__)
        args = Namespace(
            debug_deterministic_collective=False,
            distributed_backend="nccl",
            distributed_timeout_minutes=1,
            dumper_enable=True,
            env_report=None,
            num_gpus_per_node=1,
        )

        with pytest.raises(AssertionError):
            actor.init(
                args,
                "actor",
                indep_dp_info=IndepDPInfo.create_trivial(),
                indep_dp_store_addr="10.0.0.9:1234",
            )


class TestFSDPTrainExternalData:
    # [CLASS DOC] Tests of the FSDP actor's train entry point handling of external data.
    def test_fsdp_train_rejects_external_critic_data_before_starting_work(self) -> None:
        # [FUNCTION DOC] Creates an actor without its constructor (no state at all) and calls train
        # with a normal-outcome step output as external data; asserts an assertion error whose message
        # says the fsdp backend trains no critic is raised before any state is accessed.
        """FSDP rejects critic output before accessing state needed for training work."""
        actor = object.__new__(actor_module.FSDPTrainRayActor)
        external_data = TrainStepOutput(outcome=TrainStepOutcome.NORMAL)

        with pytest.raises(AssertionError, match="fsdp backend trains no critic"):
            actor.train(rollout_id=1, rollout_data_ref=object(), external_data=external_data)
