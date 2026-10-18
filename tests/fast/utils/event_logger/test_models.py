# [MODULE DOC] Unit tests for the audit event models: each event type is a strict (extra fields
# rejected) pydantic model with a type discriminator, collected in a tagged union that parses JSON
# back to the right class. Tests cover JSON round trips for several event types (witness id
# allocation, group step end, cell reconfiguration, inference engine weight checksum, engine
# environment report, witness snapshot with stale ids), discrimination between types, rejection of
# unknown fields, and the module's event naming convention check.
from datetime import datetime, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.backends.training_utils.types import TrainStepOutcome
from miles.utils.audit_utils.event_logger.models import (
    CellReconfigureEvent,
    EngineEnvReportEvent,
    Event,
    InferenceEngineWeightChecksumEvent,
    TrainGroupStepEndEvent,
    WitnessAllocateIdEvent,
    WitnessSnapshotParamEvent,
)
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity, TrainProcessIdentity

_event_adapter = TypeAdapter(Event)

_FIXED_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXED_SOURCE = SimpleProcessIdentity(component="main")
_TRAIN_SOURCE = TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)


class TestEventModelsDiscriminatedUnion:
    # [CLASS DOC] Tests that the union adapter parses serialized events into the right class.
    def test_roundtrip_via_discriminator(self) -> None:
        # [FUNCTION DOC] A witness allocation event serialized to JSON and parsed through the
        # union adapter is a witness allocation event again with the same id-to-sample mapping.
        event = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=0,
            attempt=0,
            witness_id_to_sample_index={10: 0, 11: 1},
            counter_after=12,
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessAllocateIdEvent)
        assert parsed.witness_id_to_sample_index == {10: 0, 11: 1}

    def test_discriminator_distinguishes_types(self) -> None:
        # [FUNCTION DOC] A witness allocation event and a group step end event, each round
        # tripped through JSON, parse to different classes.
        e1 = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=0,
            attempt=0,
            witness_id_to_sample_index={0: 0},
            counter_after=1,
        )
        e2 = TrainGroupStepEndEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=0,
            cell_outcomes={0: [TrainStepOutcome.NORMAL]},
        )
        p1 = _event_adapter.validate_json(e1.model_dump_json())
        p2 = _event_adapter.validate_json(e2.model_dump_json())
        assert type(p1) is not type(p2)


class TestEventModelsStrictRejectExtraFields:
    # [CLASS DOC] Tests that the event models forbid unknown fields.
    def test_extra_field_rejected(self) -> None:
        # [FUNCTION DOC] Validating a dictionary containing an unknown key "bogus_field" raises
        # a validation error whose message mentions that key. The dictionary also lacks the
        # counter field, so the error lists more than the extra field, but the match only
        # requires the name.
        data = {
            "type": "witness_allocate_id",
            "timestamp": "2026-01-01T00:00:00Z",
            "source": {"component": "main"},
            "rollout_id": 0,
            "attempt": 0,
            "witness_id_to_sample_index": {0: 0},
            "bogus_field": 123,
        }
        with pytest.raises(ValidationError, match="bogus_field"):
            WitnessAllocateIdEvent.model_validate(data)


class TestWitnessAllocateIdEvent:
    # [CLASS DOC] Tests for the witness allocation event model.
    def test_json_roundtrip(self) -> None:
        # [FUNCTION DOC] Round trip through JSON preserves the type, rollout id 2, attempt 0 and
        # the mapping (integer keys survive).
        event = WitnessAllocateIdEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=2,
            attempt=0,
            witness_id_to_sample_index={10: 0, 11: 1},
            counter_after=12,
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessAllocateIdEvent)
        assert parsed.rollout_id == 2
        assert parsed.attempt == 0
        assert parsed.witness_id_to_sample_index == {10: 0, 11: 1}


class TestTrainGroupStepEndEvent:
    # [CLASS DOC] Tests for the training group step end event model.
    def test_json_roundtrip(self) -> None:
        # [FUNCTION DOC] An event whose cell outcomes are a list of normal outcomes for cell 0
        # and the string "error" for cell 1 survives a JSON round trip, with the list elements
        # restored as outcome enum values.
        event = TrainGroupStepEndEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=3,
            cell_outcomes={0: [TrainStepOutcome.NORMAL], 1: "error"},
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, TrainGroupStepEndEvent)
        assert parsed.rollout_id == 3
        assert parsed.cell_outcomes[0] == [TrainStepOutcome.NORMAL]
        assert parsed.cell_outcomes[1] == "error"


class TestCellReconfigureEvent:
    # [CLASS DOC] Tests for the cell reconfiguration event model (rollout id, quorum id, source
    # cell index which may be none, healed cell indices, alive cell indices after).
    def test_healing_json_roundtrip(self) -> None:
        # [FUNCTION DOC] A healing event (source cell 0, healed cell 2, alive cells 0 to 2)
        # keeps all fields over a JSON round trip.
        """A healing reconfigure event (non-empty healed cells, with src) survives a JSON round-trip."""
        event = CellReconfigureEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=3,
            quorum_id=1,
            src_cell_index=0,
            healed_cell_indices=[2],
            alive_cell_indices_after=[0, 1, 2],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, CellReconfigureEvent)
        assert parsed.rollout_id == 3
        assert parsed.quorum_id == 1
        assert parsed.src_cell_index == 0
        assert parsed.healed_cell_indices == [2]
        assert parsed.alive_cell_indices_after == [0, 1, 2]

    def test_shrink_json_roundtrip(self) -> None:
        # [FUNCTION DOC] A pure shrink event (no source cell, no healed cells, one alive cell)
        # keeps none and empty list values over a JSON round trip.
        """A pure-shrink reconfigure event (no healed cells, src None) survives a JSON round-trip."""
        event = CellReconfigureEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=2,
            quorum_id=1,
            src_cell_index=None,
            healed_cell_indices=[],
            alive_cell_indices_after=[0],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, CellReconfigureEvent)
        assert parsed.src_cell_index is None
        assert parsed.healed_cell_indices == []
        assert parsed.alive_cell_indices_after == [0]


class TestInferenceEngineWeightChecksumEvent:
    # [CLASS DOC] Tests for the inference engine weight checksum event model.
    def test_json_roundtrip(self) -> None:
        # [FUNCTION DOC] An event with two engines (the second with two tensors) keeps rollout
        # id 4 and the list of per-engine checksum mappings over a JSON round trip.
        """An engine weight checksum event survives a JSON round-trip with its per-engine checksums intact."""
        engine_checksums = [
            {"rank0/embed.weight": "aaa"},
            {"rank0/embed.weight": "aaa", "rank1/embed.weight": "bbb"},
        ]
        event = InferenceEngineWeightChecksumEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            rollout_id=4,
            engine_checksums=engine_checksums,
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, InferenceEngineWeightChecksumEvent)
        assert parsed.rollout_id == 4
        assert parsed.engine_checksums == engine_checksums


class TestEngineEnvReportEvent:
    # [CLASS DOC] Tests for the engine environment report event model, which carries a cell id,
    # a server URL and the engine's own server information as an opaque mapping.
    def test_json_roundtrip(self) -> None:
        # [FUNCTION DOC] An event with a nested server info mapping (model path, version,
        # internal states with environment variables) keeps cell id, URL and the mapping equal
        # after a JSON round trip.
        """The engine's own report is stored as an opaque mapping, so an upstream schema change cannot break it."""
        server_info = {
            "model_path": "/models/qwen",
            "version": "0.5.0",
            "internal_states": [{"env_vars": {"RANK": "0"}, "waiting_queue": 0}],
        }
        event = EngineEnvReportEvent(
            timestamp=_FIXED_TS,
            source=_FIXED_SOURCE,
            cell_id="cell-0",
            server_url="http://10.0.0.1:30000",
            server_info=server_info,
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, EngineEnvReportEvent)
        assert parsed.cell_id == "cell-0"
        assert parsed.server_url == "http://10.0.0.1:30000"
        assert parsed.server_info == server_info


class TestWitnessSnapshotParamEventWithStaleIds:
    # [CLASS DOC] Tests for the witness parameter snapshot event model including its stale ids
    # field.
    def test_json_roundtrip(self) -> None:
        # [FUNCTION DOC] A snapshot event from a training source with three non-zero ids and
        # eight stale ids keeps both lists over a JSON round trip.
        event = WitnessSnapshotParamEvent(
            timestamp=_FIXED_TS,
            source=_TRAIN_SOURCE,
            rollout_id=5,
            instance_id="actor_cell0_rank0",
            nonzero_witness_ids=[10, 11, 12],
            stale_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        )
        parsed = _event_adapter.validate_json(event.model_dump_json())
        assert isinstance(parsed, WitnessSnapshotParamEvent)
        assert parsed.stale_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert parsed.nonzero_witness_ids == [10, 11, 12]


class TestDiscriminatedUnionParsesAllEvents:
    # [CLASS DOC] Tests that several event types all parse back to their own class through the
    # union adapter.
    def test_all_event_types_parse(self) -> None:
        # [FUNCTION DOC] Builds three events (witness allocation, group step end, inference
        # engine checksum), round trips each through JSON and asserts the parsed class equals
        # the original class. Despite the name, not all event types of the module are covered.
        events = [
            WitnessAllocateIdEvent(
                timestamp=_FIXED_TS,
                source=_FIXED_SOURCE,
                rollout_id=0,
                attempt=0,
                witness_id_to_sample_index={0: 0},
                counter_after=1,
            ),
            TrainGroupStepEndEvent(
                timestamp=_FIXED_TS,
                source=_FIXED_SOURCE,
                rollout_id=0,
                cell_outcomes={0: [TrainStepOutcome.NORMAL]},
            ),
            InferenceEngineWeightChecksumEvent(
                timestamp=_FIXED_TS,
                source=_FIXED_SOURCE,
                rollout_id=0,
                engine_checksums=[{"rank0/w": "aaa"}],
            ),
        ]
        for event in events:
            parsed = _event_adapter.validate_json(event.model_dump_json())
            assert type(parsed) is type(event)


class TestCheckEventNaming:
    # [CLASS DOC] Test of the module's own naming convention validation function.
    def test_naming_convention_holds(self) -> None:
        # [FUNCTION DOC] Imports the private naming check from the models module and calls it;
        # passes if it raises nothing.
        from miles.utils.audit_utils.event_logger.models import _check_event_naming

        _check_event_naming()
