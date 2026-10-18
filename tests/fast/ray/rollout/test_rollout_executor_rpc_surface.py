# [MODULE DOC]
# Unit tests for the wire contract of the rollout data pack handed from the rollout executor to the
# driver and from the driver to the trainer controller: the pack's data reference must round-trip
# as an object-store reference type (single or list), unknown fields are rejected, and the trainer
# controller's train method accepts the same pack. Uses the rpc serializers collected from the
# executor and controller classes; no network is involved.
from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.train.group import TrainerController
from miles.utils.data import RolloutDataPack
from miles.utils.object_store import _MooncakeStoreObjectRef
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs


def _round_trip(pack: RolloutDataPack) -> RolloutDataPack:
    # [FUNCTION DOC] HELPER: Encodes the pack with the result serializer of the executor's get rpc
    # method and decodes it again. Returns the decoded pack.
    serializer = collect_rpc_method_specs(RolloutExecutor)["get"].serializer
    return serializer.decode_result(serializer.encode_result(pack))


class TestWhatARolloutHandsToTheDriver:
    # [CLASS DOC] Tests the data pack as returned by the executor's get rpc method.
    def test_the_data_reference_arrives_as_a_store_reference(self):
        # [FUNCTION DOC] A pack with sample indices 1 and 2 and a Mooncake store reference survives
        # the round trip equal to the original and with the reference still of the store reference
        # type (not degraded to a mapping).
        """A pack typed as a plain mapping degrades the reference to a dict, and nothing can free it again."""
        pack = RolloutDataPack(sample_indices=[1, 2], data_ref=_MooncakeStoreObjectRef(payload={"key": "store/7"}))

        restored = _round_trip(pack)

        assert isinstance(restored.data_ref, _MooncakeStoreObjectRef)
        assert restored == pack

    def test_a_sharded_rollout_arrives_as_a_list_of_store_references(self):
        # [FUNCTION DOC] A pack whose data reference is a list of three store references round-trips
        # to an equal list in which every element is still a store reference.
        """With --delay-split-train-data-by-dp off the driver receives one reference per dp rank."""
        refs = [_MooncakeStoreObjectRef(payload={"key": f"store/{index}"}) for index in range(3)]
        pack = RolloutDataPack(sample_indices=[0], data_ref=refs)

        restored = _round_trip(pack)

        assert restored.data_ref == refs
        assert all(isinstance(ref, _MooncakeStoreObjectRef) for ref in restored.data_ref)

    def test_an_unknown_key_is_refused_rather_than_carried(self):
        # [FUNCTION DOC] Building a pack with an unknown field name raises a validation error.
        """The pack is a contract between two processes; a key only one side knows is a silent mismatch."""
        with pytest.raises(ValidationError):
            RolloutDataPack(sample_indices=[0], data_reference=None)


class TestThePackTheTrainerControllerIsGiven:
    # [CLASS DOC] Tests the data pack as received by the trainer controller's train rpc method.
    def test_the_controller_receives_the_same_pack_the_executor_returned(self):
        # [FUNCTION DOC] Encodes a train query (rollout id 1 and a pack with a store reference) and
        # decodes it with the train method's query serializer; asserts the decoded pack equals the
        # original and its reference is still a store reference.
        """The controller forwards data_ref to its cells, so it must arrive as a reference here too."""
        spec = collect_rpc_method_specs(TrainerController)["train"]
        pack = RolloutDataPack(sample_indices=[4], data_ref=_MooncakeStoreObjectRef(payload={"key": "store/4"}))

        decoded = spec.serializer.decode_query(
            spec.serializer.encode_query(dict(rollout_id=1, rollout_data_pack=pack))
        )

        assert decoded["rollout_data_pack"] == pack
        assert isinstance(decoded["rollout_data_pack"].data_ref, _MooncakeStoreObjectRef)
