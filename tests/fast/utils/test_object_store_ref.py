# [MODULE DOC]
# Tests for store object references in the object store module. A reference is an immutable,
# tagged model (backend tag plus payload) that survives a JSON round trip: Mooncake references
# carry a mapping or key as payload; Ray references carry a Ray object reference encoded as a
# string (cloudpickle bytes) on the wire and as the real object in-process. Covered: round trips
# and tag-based reconstruction, immutability, how the store is built per worker communication mode
# (only the RPC mode frees objects by hand), Ray reference behaviour (round trip, string encoding,
# in-process passthrough, explicit free through a patched free function on removal, a clear error
# when the free API is missing, no free when objects are not freed by hand), and real freeing on
# Ray versions that still offer the free API. Most Ray tests use a local-mode Ray fixture. Helper
# functions build a Ray store and re-validate a reference after JSON encoding.
from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.utils import object_store
from miles.utils import object_store as _object_store
from miles.utils.object_store import RayObjectStore, StoreObjectRef, _MooncakeStoreObjectRef, _RayStoreObjectRef

FREED_OBJECT_TIMEOUT_SECONDS = 60.0


def _ray_store(*, frees_objects: bool = True) -> RayObjectStore:
    # [FUNCTION DOC] HELPER: Create a Ray-backed store; the flag (default true) says whether removal
    # frees objects by hand.
    return RayObjectStore(frees_objects=frees_objects)


_REF_ADAPTER = TypeAdapter(StoreObjectRef)


def _after_json_round_trip(ref: StoreObjectRef) -> StoreObjectRef:
    # [FUNCTION DOC] HELPER: Encode the reference to a JSON string and validate it back through the
    # tagged-union type adapter, returning the reconstructed reference.
    return _REF_ADAPTER.validate_json(ref.model_dump_json())


class TestStoreObjectRef:
    # [CLASS DOC] Tests of reference serialization and immutability.
    def test_a_mooncake_reference_survives_a_json_round_trip(self):
        # [FUNCTION DOC] A Mooncake reference with a key/size mapping payload equals its JSON round
        # trip result and keeps the same payload.
        """The trainer ships it to the driver over rpc, which encodes the model as json."""
        ref = _MooncakeStoreObjectRef(payload={"key": "miles-object-store/7", "size": 12})

        restored = _after_json_round_trip(ref)

        assert restored == ref
        assert restored.payload == {"key": "miles-object-store/7", "size": 12}

    def test_a_reference_cannot_be_repointed_after_it_is_handed_over(self):
        # [FUNCTION DOC] Assigning a new payload to an existing reference raises a validation error
        # (the model is frozen).
        """A consumer holding it must read the object the producer put, not one somebody swapped in."""
        ref = _MooncakeStoreObjectRef(payload="k")

        with pytest.raises(ValidationError):
            ref.payload = "other"

    def test_each_store_rebuilds_its_own_kind_of_reference(self, ray_local_mode):
        # [FUNCTION DOC] A Ray store reference and a Mooncake reference each come back from a JSON
        # round trip as an instance of their own reference class.
        """Only a ray reference may be freed by hand, so the tag it travels with has to survive the wire."""
        assert isinstance(_after_json_round_trip(_ray_store().put("payload")), _RayStoreObjectRef)
        assert isinstance(_after_json_round_trip(_MooncakeStoreObjectRef(payload="k")), _MooncakeStoreObjectRef)


class TestTheStoreARunBuilds:
    # [CLASS DOC] Tests of how the store instance is configured from the worker communication mode.
    @pytest.mark.parametrize("wire, frees_objects", [("rpc", True), ("ray", False)])
    def test_only_the_wire_that_pins_objects_frees_them(self, wire: str, frees_objects: bool):
        # [FUNCTION DOC] Parametrized: RPC communication gives a store that frees objects by hand;
        # Ray communication gives one that does not. Built through the private instance factory
        # with the Ray backend and no explicit segment-contribution choice.
        """The mode is what decides, so a run must not have to remember which references it may free."""
        args = Namespace(object_store_backend="ray", worker_comm_backend=wire)

        store = object_store._create_instance(args, contribute_segment=None)

        assert store._frees_objects is frees_objects


class TestARayReferenceOnTheWire:
    # [CLASS DOC] Tests of Ray reference encoding and removal behaviour on a local Ray.
    def test_it_survives_a_json_round_trip(self, ray_local_mode):
        # [FUNCTION DOC] Put a mapping into a Ray store, round-trip the reference through JSON, and
        # get the same mapping back from a store.
        """comm-backend=rpc still allows the ray object store, so its reference has to cross the wire."""
        ref = _ray_store().put({"tokens": [1, 2, 3]})

        restored = _after_json_round_trip(ref)

        assert _ray_store().get(restored).value == {"tokens": [1, 2, 3]}

    def test_the_encoded_form_is_a_string(self, ray_local_mode):
        # [FUNCTION DOC] The JSON-mode dump of a Ray reference has a string payload.
        """An ObjectRef is not json, so it travels as the cloudpickle bytes ray documents as the last resort."""
        ref = _ray_store().put("payload")

        assert isinstance(ref.model_dump(mode="json")["payload"], str)

    def test_a_reference_that_never_left_this_process_is_unchanged(self, ray_local_mode):
        # [FUNCTION DOC] A reference used directly (no encoding) is fetched by a store and yields the
        # original value.
        """Ray communication passes the model as is, and re-encoding it would cost a copy per call."""
        ref = _ray_store().put("payload")

        assert _ray_store().get(ref).value == "payload"

    def test_removing_a_reference_releases_it_by_hand(self, monkeypatch, ray_local_mode):
        # [FUNCTION DOC] Patch the module's free function to record its argument lists; remove a
        # JSON-round-tripped reference through a store that frees objects; the recorded call
        # contains exactly one list holding the reference's payload.
        """Cloudpickling a reference pins the object, so reference counting can no longer free it."""
        from miles.utils import object_store

        freed: list[list] = []
        monkeypatch.setattr(object_store, "_ray_free", lambda refs: freed.append(list(refs)))
        restored = _after_json_round_trip(_ray_store().put("payload"))

        _ray_store().remove(restored)

        assert freed == [[restored.payload]]

    def test_a_ray_without_the_free_api_refuses_to_pretend(self, monkeypatch, ray_local_mode):
        # [FUNCTION DOC] With the free function patched to none (API missing), removing a
        # JSON-round-tripped reference raises a runtime error whose message advises pinning ray
        # below 2.59.
        """ray >= 2.59 cannot release a pinned object, and silently leaking it would hide that."""
        from miles.utils import object_store

        monkeypatch.setattr(object_store, "_ray_free", None)
        restored = _after_json_round_trip(_ray_store().put("payload"))

        with pytest.raises(RuntimeError, match="pin ray < 2.59"):
            _ray_store().remove(restored)

    def test_ray_communication_leaves_the_object_to_reference_counting(self, monkeypatch, ray_local_mode):
        # [FUNCTION DOC] With a store that does not free objects by hand and the free function
        # patched to record calls: after removing the reference, nothing was freed and the value can
        # still be fetched.
        """Nothing pins the object on that wire, and a free would destroy it for every other holder."""
        from miles.utils import object_store

        freed: list[list] = []
        monkeypatch.setattr(object_store, "_ray_free", lambda refs: freed.append(list(refs)))
        store = _ray_store(frees_objects=False)
        ref = store.put("payload")

        store.remove(ref)

        assert freed == []
        assert store.get(ref).value == "payload"


class TestFreeingARayObjectForReal:
    # [CLASS DOC] Tests of real object freeing, skipped when the installed Ray has no force-free API.
    pytestmark = pytest.mark.skipif(
        _object_store._ray_free is None, reason="this ray (>= 2.59) removed the force-free API"
    )

    def test_the_object_is_gone_after_its_reference_is_removed(self, ray_local_mode):
        # [FUNCTION DOC] Put and JSON-round-trip a mapping, remove the reference through a store, then
        # fetching the payload directly from Ray (60 s timeout) raises Ray's object-lost error.
        """The point of the explicit free is that the pinned object really stops occupying the store."""
        import ray

        restored = _after_json_round_trip(_ray_store().put({"tokens": [1, 2, 3]}))

        _ray_store().remove(restored)

        with pytest.raises(ray.exceptions.ObjectLostError):
            ray.get(restored.payload, timeout=FREED_OBJECT_TIMEOUT_SECONDS)

    def test_an_object_nobody_removed_is_still_readable(self, ray_local_mode):
        # [FUNCTION DOC] Put two objects and round-trip both references; remove one; the other is
        # still readable through a store with its original value.
        """A free reaching further than the reference it was handed would take the run's data with it."""
        kept = _after_json_round_trip(_ray_store().put({"tokens": [4]}))
        removed = _after_json_round_trip(_ray_store().put({"tokens": [5]}))

        _ray_store().remove(removed)

        assert _ray_store().get(kept).value == {"tokens": [4]}
