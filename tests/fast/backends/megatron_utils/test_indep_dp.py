# [MODULE DOC] CPU tests of the Megatron backend's independent-data-parallel (cross-cell
# fault-tolerance) group management, focusing on structured log records tagged with the "ft " prefix
# that tooling searches for: creating the cross-cell group (a fake torchft module supplies fake
# process-group classes) logs one create-group record with the quorum; reconfiguring shuts down the
# old group and logs start and end records; the cross-cell gradient all-reduce logs start and end
# records on success (consensus true, the gradient buffer reduced once), a fail record when the
# collective raises (step discarded, consensus false), and an async-error record when the group
# reports an error afterwards. Megatron parallel utilities and consensus helpers are patched.
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from miles.backends.megatron_utils.ft import indep_dp
from miles.utils.ft_utils.indep_dp import IndepDPInfo

_LOGGER_NAME = "miles.backends.megatron_utils.ft.indep_dp"


def _indep_dp_messages(caplog) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns the formatted messages, in order, of captured log records from
    # the independent-DP module's logger only.
    return [record.getMessage() for record in caplog.records if record.name == _LOGGER_NAME]


class FakeTorchftProcessGroup:
    # [CLASS DOC] HELPER: Fake of the fault-tolerant process group (both the NCCL and Gloo flavors
    # are replaced by this class): remembers the keyword arguments of its configure call and answers
    # size from the configured world size.
    def __init__(self, timeout=None) -> None:
        # [FUNCTION DOC] HELPER: Accepts an optional timeout (ignored); starts with an empty replica
        # id, rank 0 and no configure arguments.
        self._replica_id = ""
        self._rank = 0
        self.configure_kwargs: dict | None = None

    def configure(self, **kwargs) -> None:
        # [FUNCTION DOC] HELPER: Stores all keyword arguments and extracts the required replica id and
        # rank (a key error if absent).
        self.configure_kwargs = kwargs
        self._replica_id = kwargs["replica_id"]
        self._rank = kwargs["rank"]

    def size(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the configured world size; asserts configure was called first.
        assert self.configure_kwargs is not None
        return self.configure_kwargs["world_size"]

    def shutdown(self) -> None:
        # [FUNCTION DOC] HELPER: No-op.
        pass


class FakeCrossCellPGUtil:
    # [CLASS DOC] HELPER: Fake cross-cell process-group utility that records the tensors passed to
    # all-reduce, or raises a configured error instead.
    def __init__(self, *, all_reduce_error: Exception | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the optional error to raise and an empty list of reduced
        # tensors.
        self.all_reduce_error = all_reduce_error
        self.reduced_tensors: list[torch.Tensor] = []

    def all_reduce(self, tensor: torch.Tensor, group, op) -> None:
        # [FUNCTION DOC] HELPER: Raises the configured error if any; otherwise appends the tensor
        # (group and op are ignored, no actual reduction happens).
        if self.all_reduce_error is not None:
            raise self.all_reduce_error
        self.reduced_tensors.append(tensor)


def _make_model_chunk() -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a fake model chunk with one bucket group holding one bucket whose
    # gradient buffer is four elements of 2.5, and no expert-parallel bucket groups.
    bucket = SimpleNamespace(grad_data=torch.full((4,), 2.5))
    return SimpleNamespace(bucket_groups=[SimpleNamespace(buckets=[bucket])], expert_parallel_bucket_groups=[])


def _make_parallel_state(pg) -> SimpleNamespace:
    # [FUNCTION DOC] HELPER: Builds a fake parallel state: intra-DP of size 1, and independent-DP
    # info with rank 1, size 3, the given process group and debug info quorum 7.
    return SimpleNamespace(
        intra_dp=SimpleNamespace(size=1),
        indep_dp=SimpleNamespace(rank=1, size=3, group=pg, debug_info={"quorum": 7}),
    )


class TestCreateIndepDpGroup:
    # [CLASS DOC] Tests of the cross-cell group creation function.
    @pytest.fixture()
    def fake_torchft(self, monkeypatch):
        # [FUNCTION DOC] HELPER: Fixture registering, via monkeypatch, a fake torchft package and
        # process-group submodule whose NCCL and Gloo group classes are the fake group class. Returns
        # the fake submodule.
        module = types.ModuleType("torchft.process_group")
        module.ProcessGroupNCCL = FakeTorchftProcessGroup
        module.ProcessGroupGloo = FakeTorchftProcessGroup
        monkeypatch.setitem(sys.modules, "torchft", types.ModuleType("torchft"))
        monkeypatch.setitem(sys.modules, "torchft.process_group", module)
        return module

    def test_creating_a_cross_cell_group_emits_an_ft_tagged_create_record(self, fake_torchft, caplog) -> None:
        # [FUNCTION DOC] Creates the group with a store address, an info record (cell 1 of 3, alive
        # rank 1 of 2, quorum 7, alive cells [0, 1]) and Megatron rank 5 of 8, under info-level capture.
        # Asserts the returned group info has rank 1 and exactly one log message was emitted from the
        # module, starting with the "ft " tag and containing the create-group op and the quorum.
        """The create_pg record stays discoverable by the ft structured-log tag."""
        info = IndepDPInfo(
            cell_index=1, num_cells=3, alive_rank=1, alive_size=2, quorum_id=7, alive_cell_indices=[0, 1]
        )

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            group_info = indep_dp.create_indep_dp_group(
                store_addr="tcp://store:1234",
                indep_dp_info=info,
                megatron_rank=5,
                megatron_world_size=8,
            )

        messages = _indep_dp_messages(caplog)
        assert group_info.rank == 1
        assert len(messages) == 1
        assert messages[0].startswith("ft ")
        assert "op=create_pg" in messages[0]
        assert "quorum=7" in messages[0]


class TestReconfigureIndepDpGroup:
    # [CLASS DOC] Tests of the cross-cell group reconfiguration function.
    def test_reconfigure_emits_ft_tagged_start_and_end_records(self, caplog) -> None:
        # [FUNCTION DOC] Reconfigures a fake parallel state whose old group is a mock, with a new info
        # record (quorum 5, one alive cell). Asserts the old group's shutdown was called once, every log
        # message starts with the "ft " tag, the first is the reconfigure start record with the target
        # quorum 5, and the second is the reconfigure end record with quorum 5.
        """Both reconfigure records stay discoverable by the ft structured-log tag."""
        old_group = MagicMock()
        parallel_state = SimpleNamespace(indep_dp=SimpleNamespace(group=old_group, gloo_group=None))
        info = IndepDPInfo(cell_index=2, num_cells=3, alive_rank=0, alive_size=1, quorum_id=5, alive_cell_indices=[2])

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            indep_dp.reconfigure_indep_dp_group(
                parallel_state=parallel_state,
                store_addr="tcp://store:1234",
                indep_dp_info=info,
                megatron_rank=5,
                megatron_world_size=8,
            )

        messages = _indep_dp_messages(caplog)
        old_group.shutdown.assert_called_once()
        assert all(message.startswith("ft ") for message in messages)
        assert "op=reconfig phase=start" in messages[0]
        assert "quorum_to=5" in messages[0]
        assert "op=reconfig phase=end" in messages[1]
        assert "quorum=5" in messages[1]


class TestAllreduceGradsAndLossesAcrossReplicas:
    # [CLASS DOC] Tests of the cross-replica gradient and loss all-reduce and the log records it emits.
    @pytest.fixture()
    def megatron_env(self):
        # [FUNCTION DOC] HELPER: Generator fixture patching the module's Megatron parallel-utility
        # object (not the last pipeline stage), its Gloo group getter (returns none) and its collective
        # boolean-and (returns its value unchanged, i.e. consensus equals the local result). Yields the
        # patched Megatron object.
        with (
            patch.object(indep_dp, "mpu") as mock_mpu,
            patch.object(indep_dp, "get_gloo_group", return_value=None),
            patch.object(indep_dp, "collective_bool_and", side_effect=lambda value, group: value),
        ):
            mock_mpu.is_pipeline_last_stage.return_value = False
            yield mock_mpu

    @staticmethod
    def _run(pg, util) -> tuple[bool, dict[str, float]]:
        # [FUNCTION DOC] HELPER: Static method. Runs the module's all-reduce function with arguments
        # that disable per-token loss, one fake model chunk, a fake parallel state using the given
        # process group, and no reduced losses, while the general process-group utility factory is patched
        # to return the given fake utility. Returns the function's (consensus flag, reduced losses
        # mapping).
        args = SimpleNamespace(calculate_per_token_loss=False)
        with patch.object(indep_dp.GeneralPGUtil, "create", return_value=util):
            return indep_dp.allreduce_grads_and_losses_across_replicas(
                args, [_make_model_chunk()], _make_parallel_state(pg), losses_reduced=[]
            )

    def test_a_successful_allreduce_emits_ft_tagged_start_and_end_records(self, megatron_env, caplog) -> None:
        # [FUNCTION DOC] With a group that reports no error and a utility that records reductions,
        # runs the all-reduce at info level. Asserts consensus is true, the reduced losses mapping is
        # empty, one tensor was reduced, all messages start with the "ft " tag, the first is the
        # cross-cell start record for gradient all-reduce, and the second is the end record showing this
        # rank and consensus ok.
        """The happy-path cross-cell records stay discoverable by the ft structured-log tag."""
        pg = SimpleNamespace(errored=lambda: None)
        util = FakeCrossCellPGUtil()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            consensus, loss_reduced = self._run(pg, util)

        messages = _indep_dp_messages(caplog)
        assert consensus is True
        assert loss_reduced == {}
        assert len(util.reduced_tensors) == 1
        assert all(message.startswith("ft ") for message in messages)
        assert "op=cross_cell phase=start kind=grad_allreduce" in messages[0]
        assert "op=cross_cell phase=end kind=grad_allreduce" in messages[1]
        assert "this_rank_ok=true consensus_ok=true" in messages[1]

    def test_a_raising_allreduce_emits_an_ft_tagged_fail_record(self, megatron_env, caplog) -> None:
        # [FUNCTION DOC] With a utility whose all-reduce raises a runtime error (communicator aborted),
        # asserts consensus is false, exactly one message contains the fail phase, it starts with the "ft "
        # tag and names the gradient all-reduce kind, and the last message shows this rank and consensus
        # not ok.
        """A synchronous collective failure is reported as an ft-tagged fail record and discards the step."""
        pg = SimpleNamespace(errored=lambda: None)
        util = FakeCrossCellPGUtil(all_reduce_error=RuntimeError("NCCL communicator was aborted on rank 2"))

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            consensus, _loss_reduced = self._run(pg, util)

        messages = _indep_dp_messages(caplog)
        assert consensus is False
        fail_messages = [message for message in messages if "phase=fail" in message]
        assert len(fail_messages) == 1
        assert fail_messages[0].startswith("ft ")
        assert "kind=grad_allreduce" in fail_messages[0]
        assert "this_rank_ok=false consensus_ok=false" in messages[-1]

    def test_an_asynchronously_errored_group_emits_an_ft_tagged_async_error_record(self, megatron_env, caplog) -> None:
        # [FUNCTION DOC] With a group whose error query returns a runtime error (peer left the quorum)
        # and a utility that succeeds, asserts consensus is false and exactly one message contains the
        # async-error phase, starts with the "ft " tag, names the gradient all-reduce kind and includes
        # the error text.
        """An error surfacing only via pg.errored() is reported as an ft-tagged async_error record."""
        pg = SimpleNamespace(errored=lambda: RuntimeError("peer 2 left the quorum"))
        util = FakeCrossCellPGUtil()

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            consensus, _loss_reduced = self._run(pg, util)

        messages = _indep_dp_messages(caplog)
        assert consensus is False
        async_messages = [message for message in messages if "phase=async_error" in message]
        assert len(async_messages) == 1
        assert async_messages[0].startswith("ft ")
        assert "kind=grad_allreduce" in async_messages[0]
        assert "peer 2 left the quorum" in async_messages[0]
