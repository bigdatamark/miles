# [MODULE DOC]
# Tests for the object store module, which puts and gets rollout data through a process-wide
# singleton store with Ray and Mooncake backends. Covered: the default rule for whether a process
# contributes a memory segment (based on the local-rank environment variable: unset or zero
# contributes, positive does not); generation of per-field storage schemas from a value spec (none
# spec gives none, automatic codec goes to the meta-info section and others to the non-tensor
# section, dtype only when declared, fields missing from the value are skipped; the schema class
# is stubbed); singleton contract (double init and unknown backend rejected); the get-result
# wrapper (value property, release callback run once on context exit); wire validation of store
# references; a round trip through the Ray backend on a local Ray (remove is a no-op there); and,
# when the Mooncake master binary and API are installed, a round trip and removal and replica
# count validation against a locally launched Mooncake master. An autouse fixture resets the
# module's singleton instance for every test.
import pickle
import shutil
import socket
import subprocess
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.utils import object_store


def _mooncake_available() -> bool:
    # [FUNCTION DOC] HELPER: Return true only if the Mooncake master executable is found on the path
    # and the structured object store API (field schema and export-ref) can be imported; false
    # otherwise. Used as a skip condition at collection time.
    if shutil.which("mooncake_master") is None:
        return False
    try:
        from mooncake.structured_object_store import FieldSchema, export_ref  # noqa: F401
    except ImportError:
        return False
    return True


def _free_port() -> int:
    # [FUNCTION DOC] HELPER: Bind a socket on the loopback address with an OS-chosen port, read the
    # port number, close the socket and return it (the port may be taken by others afterwards).
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tolist(value: Any) -> list:
    # [FUNCTION DOC] HELPER: Convert a tensor-like value (anything with a to-list method) via that
    # method, otherwise build a list from the iterable.
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _make_rollout_data() -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: Build a small two-sample rollout-data mapping: per-sample 32-bit integer
    # tensors for tokens (lengths 3 and 2) and loss masks, plus lists for rewards, response
    # lengths, partition, sample indices, truncated flags, raw rewards, total lengths, prompts and
    # metadata mappings.
    return {
        "tokens": [torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([4, 5], dtype=torch.int32)],
        "loss_masks": [torch.tensor([1, 1, 1], dtype=torch.int32), torch.tensor([1, 1], dtype=torch.int32)],
        "rewards": [0.5, 1.0],
        "response_lengths": [3, 2],
        "partition": [0, 1],
        "sample_indices": [0, 1],
        "truncated": [0, 0],
        "raw_reward": [0.5, 1.0],
        "total_lengths": [3, 2],
        "prompt": ["hello", "world"],
        "metadata": [{"a": 1}, {"b": 2}],
    }


def _assert_roundtrip_equal(fetched: dict[str, Any], original: dict[str, Any]) -> None:
    # [FUNCTION DOC] HELPER: Assert the fetched mapping has the same key set as the original, and
    # equal contents for tokens (per-sample lists), rewards, raw rewards, total lengths, prompts
    # and metadata. Other fields (for example loss masks) are not compared.
    assert sorted(fetched.keys()) == sorted(original.keys())
    assert [_tolist(t) for t in fetched["tokens"]] == [_tolist(t) for t in original["tokens"]]
    assert _tolist(fetched["rewards"]) == original["rewards"]
    assert _tolist(fetched["raw_reward"]) == original["raw_reward"]
    assert _tolist(fetched["total_lengths"]) == original["total_lengths"]
    assert list(fetched["prompt"]) == original["prompt"]
    assert list(fetched["metadata"]) == original["metadata"]


@pytest.fixture(autouse=True)
def _reset_object_store(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Autouse fixture that sets the object store module's singleton instance
    # to none for the test (restored afterwards).
    monkeypatch.setattr(object_store, "_INSTANCE", None)


class TestDefaultContributeSegment:
    # [CLASS DOC] Tests of the default segment-contribution rule driven by the local-rank variable.
    def test_no_local_rank_contributes(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the local-rank variable unset, the default is true.
        """Processes without LOCAL_RANK (drivers, rollout manager) contribute by default."""
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert object_store._default_contribute_segment() is True

    def test_local_rank_zero_contributes(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the local-rank variable set to 0, the default is true.
        """LOCAL_RANK=0 contributes a segment."""
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert object_store._default_contribute_segment() is True

    def test_nonzero_local_rank_does_not_contribute(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the local-rank variable set to 3, the default is false.
        """LOCAL_RANK>0 does not contribute a segment."""
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert object_store._default_contribute_segment() is False


@dataclass
class _StubFieldSchema:
    # [CLASS DOC] HELPER: Plain stand-in for the Mooncake field schema, holding a codec name, a
    # nullable flag and a metadata mapping.
    codec: str
    nullable: bool
    metadata: dict[str, Any]


class TestFieldSchemasForValue:
    # [CLASS DOC] Tests of building per-field schemas from a value and its value spec, with the
    # real schema class replaced by the stub.
    @pytest.fixture(autouse=True)
    def _stub_field_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # [FUNCTION DOC] HELPER: Autouse fixture that swaps the module's field schema class for the
        # stub.
        monkeypatch.setattr(object_store, "FieldSchema", _StubFieldSchema)

    def test_none_spec_returns_none(self):
        # [FUNCTION DOC] With no value spec, the schema builder returns none.
        """Without a value spec, no field schemas are generated."""
        assert object_store._field_schemas_for_value({"a": 1}, None) is None

    def test_auto_codec_pins_meta_info_section(self):
        # [FUNCTION DOC] A spec with an automatic-codec field and a typed-ragged-codec field gives
        # schemas whose metadata section is meta-info for the first and non-tensor-batch for the
        # second.
        """codec='auto' fields go to meta_info; others go to non_tensor_batch."""
        spec = {
            "scalar": object_store.ValueSpec(codec="auto"),
            "ragged": object_store.ValueSpec(codec="typed_ragged", dtype="int32"),
        }
        schemas = object_store._field_schemas_for_value({"scalar": 1, "ragged": [2]}, spec)
        assert schemas["scalar"].metadata["section"] == "meta_info"
        assert schemas["ragged"].metadata["section"] == "non_tensor_batch"

    def test_dtype_included_only_when_set(self):
        # [FUNCTION DOC] A typed field with dtype int32 has it in its schema metadata; an untyped
        # field (msgpack codec) has no dtype key; every schema is non-nullable.
        """dtype appears in schema metadata only for specs that declare it."""
        spec = {
            "typed": object_store.ValueSpec(codec="typed_ragged", dtype="int32"),
            "untyped": object_store.ValueSpec(codec="msgpack"),
        }
        schemas = object_store._field_schemas_for_value({"typed": [1], "untyped": [2]}, spec)
        assert schemas["typed"].metadata["dtype"] == "int32"
        assert "dtype" not in schemas["untyped"].metadata
        assert all(schema.nullable is False for schema in schemas.values())

    def test_spec_fields_absent_from_value_are_skipped(self):
        # [FUNCTION DOC] A spec naming two fields with a value holding only one yields schemas for
        # just the present field.
        """Spec entries for keys missing from the value dict produce no schema."""
        spec = {
            "present": object_store.ValueSpec(codec="auto"),
            "absent": object_store.ValueSpec(codec="auto"),
        }
        schemas = object_store._field_schemas_for_value({"present": 1}, spec)
        assert sorted(schemas.keys()) == ["present"]


class TestSingletonContract:
    # [CLASS DOC] Tests of the process-wide store instance initialization rules.
    def test_double_init_rejected(self):
        # [FUNCTION DOC] Initializing with Ray backend arguments twice makes the second call raise an
        # assertion error.
        """Calling init_instance twice in one process asserts."""
        args = Namespace(object_store_backend="ray", worker_comm_backend="ray")
        object_store.init_instance(args)
        with pytest.raises(AssertionError):
            object_store.init_instance(args)

    def test_unknown_backend_rejected(self):
        # [FUNCTION DOC] Initializing with a bogus backend name raises a value error.
        """An unknown backend value raises ValueError from the enum lookup."""
        with pytest.raises(ValueError):
            object_store.init_instance(Namespace(object_store_backend="bogus"))


class TestObjectStoreGetResult:
    # [CLASS DOC] Tests of the get-result wrapper.
    def test_value_property_and_release_on_exit(self):
        # [FUNCTION DOC] Build a result with a recording release callback; its value property returns
        # the value, entering the context binds the value, and the callback is called exactly once
        # with the value on exit (not before).
        """The context manager exposes the value and calls release_fn exactly once on exit."""
        released: list[Any] = []
        result = object_store.ObjectStoreGetResult(value={"a": 1}, release_fn=released.append)
        assert result.value == {"a": 1}
        with result as value:
            assert value == {"a": 1}
            assert released == []
        assert released == [{"a": 1}]


class TestStoreObjectRefWireValidation:
    # [CLASS DOC] Tests of validating serialized store references through a type adapter.
    def test_a_wire_reference_without_a_backend_tag_is_rejected(self) -> None:
        # [FUNCTION DOC] Validating JSON that has only a payload field (no backend tag) raises a
        # validation error.
        """A wire reference without a backend discriminator is rejected."""
        adapter = TypeAdapter(object_store.StoreObjectRef)

        with pytest.raises(ValidationError):
            adapter.validate_json('{"payload": "opaque-reference"}')

    def test_a_corrupt_encoded_ray_reference_is_rejected_at_the_wire_boundary(self) -> None:
        # [FUNCTION DOC] Validating JSON with the Ray backend tag and a payload that is base64 of
        # non-pickle bytes raises an unpickling error (not a validation error).
        """A Ray reference with corrupt cloudpickle data is rejected during wire validation."""
        adapter = TypeAdapter(object_store.StoreObjectRef)

        with pytest.raises(pickle.UnpicklingError):
            adapter.validate_json('{"backend": "ray", "payload": "bm90IGEgcGlja2xl"}')


class TestRayObjectStore:
    # [CLASS DOC] Tests of the Ray-backed store on a local Ray instance.
    @pytest.fixture(scope="class", autouse=True)
    def _ray_minicluster(self, ray_local_mode):
        # [FUNCTION DOC] HELPER: Class-scoped autouse fixture that depends on the local-mode Ray
        # fixture (so Ray is started once for the class) and yields without doing anything else.
        yield

    def test_roundtrip_and_noop_remove(self):
        # [FUNCTION DOC] Initialize the singleton with Ray backend and Ray worker communication;
        # assert it is a Ray store. Put sample rollout data with the rollout value spec, get it
        # back and compare, use the result as a context manager, remove the reference, and assert
        # the data can still be fetched (remove is a no-op for this backend).
        """RayObjectStore puts/gets a rollout dict and remove is a no-op."""
        args = Namespace(object_store_backend="ray", worker_comm_backend="ray")
        store = object_store.init_instance(args)
        assert isinstance(store, object_store.RayObjectStore)

        data = _make_rollout_data()
        ref = store.put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        get_result = store.get(ref)
        _assert_roundtrip_equal(get_result.value, data)
        with get_result:
            pass
        store.remove(ref)
        _assert_roundtrip_equal(store.get(ref).value, data)

    def test_get_instance_requires_init(self):
        # [FUNCTION DOC] Getting the singleton before initialization raises an assertion error.
        """get_instance asserts when init_instance was never called."""
        with pytest.raises(AssertionError):
            object_store.get_instance()


@pytest.mark.skipif(not _mooncake_available(), reason="mooncake with structured_object_store API not installed")
class TestMooncakeObjectStore:
    # [CLASS DOC] Tests of the Mooncake-backed store against a locally launched master process.
    # Skipped as a whole when the master executable or the structured store API is unavailable.
    @pytest.fixture(scope="class")
    def mooncake_master_port(self):
        # [FUNCTION DOC] HELPER: Class-scoped fixture that starts the Mooncake master subprocess on
        # free RPC and metrics ports (output discarded), waits 2 s for it to come up, yields the
        # RPC port, and afterwards terminates the process and waits up to 10 s for it to exit.
        port = _free_port()
        master = subprocess.Popen(
            ["mooncake_master", "--rpc_port", str(port), "--metrics_port", str(_free_port())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        yield port
        master.terminate()
        master.wait(timeout=10)

    def _make_args(self, port: int) -> Namespace:
        # [FUNCTION DOC] HELPER: Build an arguments namespace selecting the Mooncake backend, with
        # init kwargs for the TCP protocol, a master at the loopback address and the given port,
        # 64 MB segment and local buffer sizes, and a replica count of one.
        return Namespace(
            object_store_backend="mooncake",
            mooncake_store_init_kwargs={
                "protocol": "tcp",
                "master_server_address": f"127.0.0.1:{port}",
                "global_segment_size": "64mb",
                "local_buffer_size": "64mb",
            },
            mooncake_replica_num=1,
        )

    def test_roundtrip_release_and_remove(self, mooncake_master_port: int):
        # [FUNCTION DOC] Initialize the store, put sample rollout data, get and compare it, use and
        # release the result, then remove the reference and assert that getting it again raises
        # some exception (the exact type varies).
        """MooncakeObjectStore round-trips a rollout dict; remove deletes the object."""
        store = object_store.init_instance(self._make_args(mooncake_master_port))
        assert isinstance(store, object_store.MooncakeObjectStore)

        data = _make_rollout_data()
        ref = store.put(value=data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        get_result = store.get(ref)
        _assert_roundtrip_equal(get_result.value, data)
        with get_result:
            pass

        store.remove(ref)
        with pytest.raises(Exception):  # noqa: B017 - mooncake surfaces missing keys as varying exception types
            store.get(ref)

    def test_replica_num_below_one_rejected(self, mooncake_master_port: int):
        # [FUNCTION DOC] With the replica count set to zero, initializing the store raises a value
        # error.
        """Constructing the store with replica num < 1 raises ValueError."""
        args = self._make_args(mooncake_master_port)
        args.mooncake_replica_num = 0
        with pytest.raises(ValueError):
            object_store.init_instance(args)
