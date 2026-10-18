# [MODULE DOC]
# Process-wide object store for passing large values between workers by serializable reference.
# Two backends: ray (references are ray object refs, serialized as base64 cloudpickle text) and
# mooncake (a distributed store; values are dictionaries transferred as bundles with optional
# per-field codec schemas). References are pydantic models tagged by backend. A module-level
# singleton is created once through the init function. Mooncake and the ray free API are optional
# imports (failures are deferred until use).
from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Annotated, Any, Literal

import ray

try:  # ray 2.59 removed the API that force-frees an object pinned by a serialized reference
    from ray._private.internal_api import free as _ray_free
except ImportError:
    _ray_free = None
from pydantic import AfterValidator, ConfigDict, Field, PlainSerializer

from miles.utils.object_store_config import compute_mooncake_store_config
from miles.utils.pydantic_utils import StrictBaseModel
from miles.utils.workers.types import WorkerCommBackend

_MOONCAKE_IMPORT_ERROR: ImportError | None = None

try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
    from mooncake.structured_object_store import FieldSchema, MooncakeBundleTransfer, export_ref, import_ref

    _MOONCAKE_AVAILABLE = True
except ImportError as exc:
    _MOONCAKE_AVAILABLE = False
    _MOONCAKE_IMPORT_ERROR = exc
    FieldSchema = None
    ReplicateConfig = None


# ============================== types ==============================


class ObjectStoreBackend(Enum):
    # [CLASS DOC] Backend names: ray and mooncake (their text values are used in configuration and in
    # reference tags).
    RAY = "ray"
    MOONCAKE = "mooncake"


class _BaseStoreObjectRef(StrictBaseModel):
    # [CLASS DOC] HELPER: Base of reference models: immutable, extra fields forbidden, arbitrary
    # field types allowed (for ray object refs).
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


@dataclass(frozen=True)
class ValueSpec:
    # [CLASS DOC] Per-field encoding hint for mooncake puts: codec name (the name auto means the
    # field goes to the meta info section, anything else to the non-tensor section) and optional
    # dtype text.
    codec: str
    dtype: str | None = None


class ObjectStoreGetResult:
    # [CLASS DOC] Result of a get: holds the value and a release callback. Usable as a context
    # manager: entering yields the value; exiting calls the release callback with the value (never
    # suppresses exceptions).
    def __init__(self, value: Any, release_fn: Callable[[Any], None]) -> None:
        # [FUNCTION DOC] Stores the value and release callback.
        self._value = value
        self._release_fn = release_fn

    @property
    def value(self) -> Any:
        # [FUNCTION DOC] Property: the fetched value (not released until the context exits).
        return self._value

    def __enter__(self) -> Any:
        # [FUNCTION DOC] Returns the value.
        return self._value

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # [FUNCTION DOC] Calls the release callback on the value; returns nothing (exceptions
        # propagate).
        self._release_fn(self._value)


# ============================ singleton ============================

_INSTANCE: BaseObjectStore | None = None


def init_instance(args: Namespace, *, contribute_segment: bool | None = None) -> BaseObjectStore:
    # [FUNCTION DOC] Creates and registers the process-wide store; asserts it was not initialized
    # before. The segment contribution flag (mooncake only) may be left none to use the default.
    # Returns the instance.
    global _INSTANCE
    assert _INSTANCE is None, "object store instance is already initialized"
    _INSTANCE = _create_instance(args, contribute_segment=contribute_segment)
    return _INSTANCE


def get_instance() -> BaseObjectStore:
    # [FUNCTION DOC] Returns the registered store; asserts it was initialized.
    assert _INSTANCE is not None, "object store instance is not initialized; call init_instance first"
    return _INSTANCE


def _create_instance(args: Namespace, *, contribute_segment: bool | None) -> BaseObjectStore:
    # [FUNCTION DOC] HELPER: Picks the backend from the args' backend name. Mooncake: resolves a
    # missing contribute flag to the default and builds the mooncake store. Otherwise ray: the store
    # frees objects explicitly only when the worker comm backend is rpc.
    backend = ObjectStoreBackend(args.object_store_backend)
    if backend == ObjectStoreBackend.MOONCAKE:
        if contribute_segment is None:
            contribute_segment = _default_contribute_segment()
        return MooncakeObjectStore(args, contribute_segment=contribute_segment)
    return RayObjectStore(frees_objects=WorkerCommBackend(args.worker_comm_backend) == WorkerCommBackend.RPC)


def _default_contribute_segment() -> bool:
    # [FUNCTION DOC] HELPER: True when the local rank environment variable is unset or zero (one
    # memory segment contributor per node).
    local_rank = os.getenv("LOCAL_RANK")
    return local_rank is None or int(local_rank) == 0


# ============================ base class ===========================


class BaseObjectStore(ABC):
    # [CLASS DOC] Interface: put a value and get a reference, get a value from a reference, remove.
    @abstractmethod
    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        # [FUNCTION DOC] Abstract: stores the value (optionally with per-field specs) and returns a
        # serializable reference.
        raise NotImplementedError

    @abstractmethod
    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        # [FUNCTION DOC] Abstract: fetches the value of the reference as a result object.
        raise NotImplementedError

    @abstractmethod
    def remove(self, ref: StoreObjectRef) -> None:
        # [FUNCTION DOC] Abstract: releases the stored value of the reference.
        raise NotImplementedError


# ============================ ray backend ==========================


def _decode_ray_object_ref(payload: str) -> ray.ObjectRef:
    # [FUNCTION DOC] HELPER: Base64-decodes and unpickles (cloudpickle) a ray object ref. Only use on
    # trusted input (unpickling).
    return ray.cloudpickle.loads(base64.b64decode(payload))


def _encode_ray_object_ref(payload: ray.ObjectRef) -> str:
    # [FUNCTION DOC] HELPER: Cloudpickles a ray object ref and returns it as base64 text.
    return base64.b64encode(ray.cloudpickle.dumps(payload)).decode()


class _RayStoreObjectRef(_BaseStoreObjectRef):
    # [CLASS DOC] HELPER: Reference for the ray backend, tagged with the backend name. The payload
    # accepts a ray object ref or base64 text (decoded on validation) and serializes as base64 text.
    backend: Literal[ObjectStoreBackend.RAY.value] = ObjectStoreBackend.RAY.value

    payload: Annotated[
        ray.ObjectRef | Annotated[str, AfterValidator(_decode_ray_object_ref)],
        PlainSerializer(_encode_ray_object_ref, return_type=str),
    ]


class RayObjectStore(BaseObjectStore):
    # [CLASS DOC] Store backed by the ray object store. Whether remove actually frees objects is
    # configured at construction.
    def __init__(self, *, frees_objects: bool) -> None:
        # [FUNCTION DOC] Stores the frees-objects flag.
        self._frees_objects = frees_objects

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        # [FUNCTION DOC] Puts the value into ray's store and returns a reference holding the object
        # ref. The value spec is ignored.
        return _RayStoreObjectRef(payload=ray.put(value))

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        # [FUNCTION DOC] Blocking fetch of the object; the release callback does nothing.
        return ObjectStoreGetResult(value=ray.get(ref.payload), release_fn=_release_noop)

    def remove(self, ref: StoreObjectRef) -> None:
        # [FUNCTION DOC] Does nothing unless the store frees objects. Otherwise force-frees the object
        # through ray's private free API; raises a runtime error explaining the ray version limit if
        # that API is not available.
        if not self._frees_objects:
            return
        if _ray_free is None:
            raise RuntimeError(
                "this ray (>= 2.59) removed ray._private.internal_api.free, so an object pinned by a "
                "serialized reference cannot be released by hand; pin ray < 2.59 for the rpc object wire"
            )
        _ray_free([ref.payload])


def _release_noop(value: Any) -> None:
    # [FUNCTION DOC] HELPER: Release callback that does nothing.
    pass


# ========================= mooncake backend ========================


class _MooncakeStoreObjectRef(_BaseStoreObjectRef):
    # [CLASS DOC] HELPER: Reference for the mooncake backend, tagged with the backend name; the
    # payload is the exported reference (any serializable structure).
    backend: Literal[ObjectStoreBackend.MOONCAKE.value] = ObjectStoreBackend.MOONCAKE.value

    payload: Any


class MooncakeObjectStore(BaseObjectStore):
    # [CLASS DOC] Store backed by a mooncake distributed store, transferring dictionaries as bundles
    # under the fixed key prefix and namespace "miles". Optionally replicates data.
    def __init__(self, args: Namespace, *, contribute_segment: bool) -> None:
        # [FUNCTION DOC] Requires the mooncake package (import error otherwise). Reads init keyword
        # arguments (default empty) and the replica count from args; raises a value error if the
        # replica count is below 1. Sets up the store with the computed configuration (this process
        # contributes a memory segment only when the flag is true) and raises a runtime error on a
        # setup error. Creates the bundle transfer helper.
        _check_mooncake_available()

        self._init_kwargs: dict[str, Any] = args.mooncake_store_init_kwargs or {}
        self._replica_num: int = args.mooncake_replica_num
        if self._replica_num < 1:
            raise ValueError("--mooncake-replica-num must be >= 1")

        store = MooncakeDistributedStore()
        setup_error = store.setup(
            compute_mooncake_store_config(self._init_kwargs, contribute_segment=contribute_segment)
        )
        if setup_error:
            raise RuntimeError(f"Mooncake store setup failed: {setup_error}")
        self._transfer = MooncakeBundleTransfer(store, key_prefix="miles-object-store")

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        # [FUNCTION DOC] Stores the value as a dictionary bundle with the chunk size from the init
        # arguments (if any), the replicate config, and field schemas derived from the value spec
        # (only for fields present in the value). Returns a reference holding the exported ref.
        ref = self._transfer.put(
            value,
            type="dict",
            namespace="miles",
            chunk_bytes=self._init_kwargs.get("chunk_bytes"),
            config=self._replicate_config(),
            field_schemas=_field_schemas_for_value(value, value_spec),
        )
        return _MooncakeStoreObjectRef(payload=export_ref(ref))

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        # [FUNCTION DOC] Imports the reference and fetches the dictionary bundle; the release callback
        # is the transfer class' result release function (frees buffers held by the value).
        value = self._transfer.get(import_ref(ref.payload), type="dict")
        return ObjectStoreGetResult(value=value, release_fn=MooncakeBundleTransfer.release_result)

    def remove(self, ref: StoreObjectRef) -> None:
        # [FUNCTION DOC] Cleans up the stored bundle of the imported reference.
        self._transfer.cleanup_dataproto(import_ref(ref.payload))

    def _replicate_config(self) -> Any:
        # [FUNCTION DOC] HELPER: None for a single replica (store default); otherwise a replicate
        # config with the replica count.
        if self._replica_num == 1:
            return None
        config = ReplicateConfig()
        config.replica_num = self._replica_num
        return config


def _check_mooncake_available() -> None:
    # [FUNCTION DOC] HELPER: Raises an import error (chained to the original import failure) if the
    # mooncake package could not be imported.
    if not _MOONCAKE_AVAILABLE:
        raise ImportError("object-store-backend='mooncake' requires the mooncake package") from _MOONCAKE_IMPORT_ERROR


def _field_schemas_for_value(value: Any, value_spec: dict[str, ValueSpec] | None) -> dict[str, Any] | None:
    # [FUNCTION DOC] HELPER: None without a spec. Otherwise a mapping from field name to a non-nullable
    # field schema with the spec's codec and metadata (section: meta info for the auto codec, else
    # non-tensor batch; plus dtype when given), restricted to fields present in the value.
    if value_spec is None:
        return None
    return {
        field: FieldSchema(
            codec=spec.codec,
            nullable=False,
            metadata={
                "section": "meta_info" if spec.codec == "auto" else "non_tensor_batch",
                **({"dtype": spec.dtype} if spec.dtype is not None else {}),
            },
        )
        for field, spec in value_spec.items()
        if field in value
    }


StoreObjectRef = Annotated[_RayStoreObjectRef | _MooncakeStoreObjectRef, Field(discriminator="backend")]
