# [MODULE DOC]
# Tests for process identity models. Simple identities name a fixed set of components (main,
# rollout executor, inference controller, multi-lora controller, worker manager) and their name is
# the component; train identities add an optional model id and cell and rank indices (non-negative)
# and name as "[model_]component_cellNNNNN_rankNNNNN" with five-digit zero padding; trainer
# controller identities name as "trainer_controller_<trainer id>". A discriminated union by
# component parses wire data, rejecting unknown components, and JSON round trips preserve values.
"""Tests for process_identity module."""

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.utils.audit_utils.process_identity import (
    ProcessIdentity,
    SimpleProcessIdentity,
    TrainerControllerProcessIdentity,
    TrainProcessIdentity,
)


class TestProcessIdentityToName:
    # [CLASS DOC] Tests of the identity-to-name conversion.
    def test_main(self) -> None:
        # [FUNCTION DOC] The main component names itself "main".
        assert SimpleProcessIdentity(component="main").to_name() == "main"

    def test_rollout_executor(self) -> None:
        # [FUNCTION DOC] Rollout executor names itself "rollout_executor".
        assert SimpleProcessIdentity(component="rollout_executor").to_name() == "rollout_executor"

    def test_actor(self) -> None:
        # [FUNCTION DOC] Actor, cell 1, rank 3 gives "actor_cell00001_rank00003".
        source = TrainProcessIdentity(component="actor", cell_index=1, rank_within_cell=3)
        assert source.to_name() == "actor_cell00001_rank00003"

    def test_critic(self) -> None:
        # [FUNCTION DOC] Critic, cell 0, rank 2 gives "critic_cell00000_rank00002".
        source = TrainProcessIdentity(component="critic", cell_index=0, rank_within_cell=2)
        assert source.to_name() == "critic_cell00000_rank00002"

    def test_trainer_controller(self) -> None:
        # [FUNCTION DOC] Trainer id "actor" gives "trainer_controller_actor".
        assert TrainerControllerProcessIdentity(trainer_id="actor").to_name() == "trainer_controller_actor"

    def test_a_policy_trainer_controller_of_a_multi_policy_run(self) -> None:
        # [FUNCTION DOC] Trainer id "alpha-actor" (with a hyphen) validates and gives
        # "trainer_controller_alpha-actor".
        """A generic trainer id must survive validation, or a multi policy worker cannot configure its logger."""
        assert TrainerControllerProcessIdentity(trainer_id="alpha-actor").to_name() == "trainer_controller_alpha-actor"

    def test_a_policy_worker_names_the_policy_it_serves(self) -> None:
        # [FUNCTION DOC] Model id "alpha" prefixes the name: "alpha_actor_cell00001_rank00003".
        """Two policies write to the same log directory, so their file names must differ."""
        source = TrainProcessIdentity(component="actor", model_id="alpha", cell_index=1, rank_within_cell=3)
        assert source.to_name() == "alpha_actor_cell00001_rank00003"

    def test_inference_controller(self) -> None:
        # [FUNCTION DOC] Inference controller names itself "inference_controller".
        assert SimpleProcessIdentity(component="inference_controller").to_name() == "inference_controller"

    def test_multi_lora_controller(self) -> None:
        # [FUNCTION DOC] Multi-lora controller names itself "multi_lora_controller".
        assert SimpleProcessIdentity(component="multi_lora_controller").to_name() == "multi_lora_controller"

    def test_worker_manager(self) -> None:
        # [FUNCTION DOC] Worker manager names itself "worker_manager".
        assert SimpleProcessIdentity(component="worker_manager").to_name() == "worker_manager"

    def test_an_unknown_component_is_rejected(self) -> None:
        # [FUNCTION DOC] A simple identity with component "nope" raises a validation error.
        """A simple identity only names the components that exist."""
        with pytest.raises(ValidationError):
            SimpleProcessIdentity(component="nope")


class TestControllerIdentityRoundtrip:
    # [CLASS DOC] Round-trip tests of the trainer controller identity.
    def test_trainer_controller_keeps_its_trainer_id(self) -> None:
        # [FUNCTION DOC] A JSON dump and re-validation of an identity with trainer id "critic" gives
        # an equal object.
        """Two trainer controllers share a component, so only the trainer id tells their events apart."""
        source = TrainerControllerProcessIdentity(trainer_id="critic")
        assert TrainerControllerProcessIdentity.model_validate_json(source.model_dump_json()) == source


class TestTrainProcessIdentityValidation:
    # [CLASS DOC] Tests of field validation on train identities.
    def test_negative_cell_index_rejected(self) -> None:
        # [FUNCTION DOC] A cell index of -1 raises a validation error.
        """A negative cell_index fails validation."""
        with pytest.raises(ValidationError):
            TrainProcessIdentity(component="actor", cell_index=-1, rank_within_cell=0)

    def test_negative_rank_within_cell_rejected(self) -> None:
        # [FUNCTION DOC] A rank within cell of -1 raises a validation error.
        """A negative rank_within_cell fails validation."""
        with pytest.raises(ValidationError):
            TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=-1)


class TestProcessIdentityUnion:
    # [CLASS DOC] Tests of the discriminated union of identity types.
    def test_process_identity_union_deserializes_rollout_executor(self) -> None:
        # [FUNCTION DOC] Validating a mapping with component "rollout_executor" yields a simple
        # identity whose name is "rollout_executor".
        """The discriminated union maps the wire component "rollout_executor" to SimpleProcessIdentity."""
        parsed = TypeAdapter(ProcessIdentity).validate_python({"component": "rollout_executor"})

        assert isinstance(parsed, SimpleProcessIdentity)
        assert parsed.to_name() == "rollout_executor"

    def test_rollout_executor_survives_a_union_json_roundtrip(self) -> None:
        # [FUNCTION DOC] Dump a rollout executor identity to JSON and validate it through the union;
        # the result equals the source.
        """A serialized rollout executor identity is parsed back to the same value through the union."""
        adapter = TypeAdapter(ProcessIdentity)
        source = SimpleProcessIdentity(component="rollout_executor")

        parsed = adapter.validate_json(source.model_dump_json())

        assert parsed == source

    def test_unknown_component_is_rejected_by_the_union(self) -> None:
        # [FUNCTION DOC] Validating a mapping with component "rollout_manager" through the union
        # raises a validation error.
        """An unknown component discriminator fails validation instead of falling back to a member."""
        with pytest.raises(ValidationError):
            TypeAdapter(ProcessIdentity).validate_python({"component": "rollout_manager"})


class TestTrainProcessIdentityRoundtrip:
    # [CLASS DOC] Round-trip tests of the train identity.
    def test_serialize_deserialize(self) -> None:
        # [FUNCTION DOC] A JSON dump and re-validation of an actor identity (cell 2, rank 0) keeps
        # the cell index, rank and component.
        source = TrainProcessIdentity(component="actor", cell_index=2, rank_within_cell=0)
        parsed = TrainProcessIdentity.model_validate_json(source.model_dump_json())
        assert parsed.cell_index == 2
        assert parsed.rank_within_cell == 0
        assert parsed.component == "actor"
