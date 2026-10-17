# [MODULE DOC] Backup/restore of named GPU tensors (model weights) to pinned CPU memory under string
# tags. Defines an abstract interface, a normal implementation that keeps a full pinned CPU copy per
# tag, and a "main-cast" implementation for the actor that, instead of storing the actor weights,
# rebuilds them on restore from the optimizer's master weights (cast + parameter all-gather), only
# keeping pinned copies of extra tensors; optionally verifies bit-exactness with SHA-256 hashes.
# All copies use asynchronous transfers followed by a full CUDA device synchronize.
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


@dataclass(frozen=True)
class MainCastContext:
    # [CLASS DOC] Immutable bundle of what the main-cast backuper needs from the training engine:
    # cast_main_to_params (a callable that writes this rank's owned parameter shard from the master
    # weights, as the end of a train step does), model_chunks (the distributed-data-parallel model
    # chunks whose parameter sync is forced on restore), extras_getter (callable yielding
    # name/tensor pairs of tensors that cannot be rebuilt and so need a pinned backup),
    # rematerializable_ids (a set of object ids of tensors that can be rebuilt by the cast and so
    # need no backup), and check (whether to verify restores by hashing).
    # Writes this rank's owned shard from the master weights, as the train-step end does.
    cast_main_to_params: Callable[[], None]
    model_chunks: list
    extras_getter: _SourceGetter
    rematerializable_ids: set
    check: bool


class TensorBackuper(ABC):
    # [CLASS DOC] Abstract interface for tag-addressed tensor backups. Constructed through the
    # static create factory. Holds the source getter: a callable returning name/tensor pairs for
    # the current live tensors to back up or restore into.
    @staticmethod
    def create(source_getter, main_cast_ctx: "MainCastContext | None" = None):
        # [FUNCTION DOC] Factory: returns the main-cast implementation when a main-cast context is
        # given, otherwise the normal full-copy implementation. Both receive the source getter.
        if main_cast_ctx is not None:
            return _TensorBackuperMainCast(source_getter=source_getter, ctx=main_cast_ctx)
        return _TensorBackuperNormal(source_getter=source_getter)

    def __init__(self, source_getter: _SourceGetter):
        # [FUNCTION DOC] Stores the source getter for subclasses.
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        # [FUNCTION DOC] Abstract property: the tags that currently have a backup.
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        # [FUNCTION DOC] Abstract: returns the mapping from tensor name to backed-up tensor for a
        # tag.
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        # [FUNCTION DOC] Abstract: copies the current source tensors into the backup for a tag.
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        # [FUNCTION DOC] Optional operation: copy one tag's backup into another's. The base class
        # raises a not-implemented error; only the normal implementation overrides it.
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        # [FUNCTION DOC] Abstract: copies the backup for a tag back into the live source tensors.
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    # [CLASS DOC] HELPER: Straightforward implementation holding, per tag, a dictionary from tensor
    # name to a pinned CPU tensor of the same shape and dtype.
    def __init__(self, source_getter):
        # [FUNCTION DOC] HELPER: Calls the base constructor and starts with an empty backup store
        # (auto-created per tag on first access).
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        # [FUNCTION DOC] HELPER: Returns a list of the tags that have been created.
        return list(self._backups)

    def get(self, tag: str):
        # [FUNCTION DOC] HELPER: Returns the live name-to-pinned-tensor dictionary of the tag
        # (not a copy). Raises an assertion error if the tag was never backed up.
        assert tag in self._backups, f"tag {tag!r} was never backed up"
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        # [FUNCTION DOC] HELPER: For each source tensor, allocates a pinned CPU buffer the first
        # time its name is seen under this tag, then copies the detached tensor into it
        # asynchronously. Ends with a CUDA device-wide synchronize so the data is complete on
        # return. Overwrites an existing backup of the same tag in place.
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        # [FUNCTION DOC] HELPER: Copies, host to host, every tensor already present under the
        # destination tag from the same-named tensor of the source tag (so the destination must
        # already exist and its names must be a subset of the source's; a missing tag or name
        # raises a key error). Does not create the destination (a defaultdict will create an empty
        # one for an unknown destination, resulting in a no-op).
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        # [FUNCTION DOC] HELPER: Copies each source tensor's backup (by name) back into the live
        # tensor asynchronously, then does a CUDA device-wide synchronize. Asserts that every
        # live name has a backup under the tag (assertion error otherwise); an unknown tag is
        # treated as an empty backup and so fails that assertion if there are any live tensors.
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            assert name in backup_dict
            param.copy_(backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()


class _TensorBackuperMainCast(TensorBackuper):
    # [CLASS DOC] HELPER: Backuper for the "actor" tag that avoids a pinned CPU copy of the actor
    # weights. Restoring the actor re-runs the cast of master weights into parameters, forces a
    # parameter all-gather sync on every model chunk, then copies the extras back from their
    # pinned backups. Any other tag (reference/teacher weights) has no master weights, so it is
    # delegated to an internal normal backuper with full pinned copies. When the context's check
    # flag is set, the first two backup cycles record SHA-256 hashes of all source tensors and the
    # following restore verifies bit-exact equality.
    """Rebuilds the actor weights instead of keeping a pinned CPU copy of them.

    Restore replays the step end's cast + all-gather, so it is bit-identical. Only
    `extras_getter` tensors keep a pinned backup. Non-actor tags (ref/teacher) have no
    master weights to rebuild from, so they keep full pinned copies via a delegated
    _TensorBackuperNormal.
    """

    _check_num_cycles = 2

    def __init__(self, source_getter, ctx: MainCastContext):
        # [FUNCTION DOC] HELPER: Stores the context and creates the delegate for non-actor tags,
        # empty extras backups (by name and by object id of the live tensor), a backup counter at
        # zero and no expected hashes.
        super().__init__(source_getter=source_getter)
        self._ctx = ctx
        self._others = _TensorBackuperNormal(source_getter=source_getter)
        self._extras_backup: dict[str, torch.Tensor] = {}
        self._extras_backup_by_id: dict[int, torch.Tensor] = {}
        self._backup_count = 0
        self._expected_hashes: dict[str, str] | None = None

    @property
    def backup_tags(self):
        # [FUNCTION DOC] HELPER: The tag "actor" (always listed, even before any backup) followed
        # by the delegate's tags.
        return ["actor", *self._others.backup_tags]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        # [FUNCTION DOC] HELPER: For a non-actor tag, delegates to the normal backuper. For the
        # actor: copies each extras tensor into a lazily allocated pinned CPU buffer (async),
        # records the buffer under the tensor's object id as well, synchronizes the CUDA device,
        # increments the backup counter, and, if checking is enabled and the counter is within the
        # first two cycles, stores hashes of all source tensors as expected values; otherwise
        # clears the expected hashes. The actor weights themselves are not copied.
        if tag != "actor":
            return self._others.backup(tag)
        for name, tensor in self._ctx.extras_getter():
            if name not in self._extras_backup:
                self._extras_backup[name] = torch.empty_like(tensor, device=torch.device("cpu"), pin_memory=True)
            self._extras_backup[name].copy_(tensor.detach(), non_blocking=True)
            self._extras_backup_by_id[id(tensor)] = self._extras_backup[name]
        torch.cuda.synchronize()
        self._backup_count += 1
        if self._ctx.check and self._backup_count <= self._check_num_cycles:
            self._expected_hashes = self._compute_hashes()
        else:
            self._expected_hashes = None

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        # [FUNCTION DOC] HELPER: Non-actor tags delegate. For the actor: runs the context's
        # cast-main-to-params callable, forces a synchronous parameter sync on each model chunk,
        # copies each extras tensor back from its pinned backup by name (key error if never
        # backed up), synchronizes the CUDA device, and, if expected hashes were recorded at
        # backup time, verifies that the restored weights hash identically.
        if tag != "actor":
            return self._others.restore(tag)
        self._ctx.cast_main_to_params()
        for model_chunk in self._ctx.model_chunks:
            model_chunk.start_param_sync(force_sync=True)
        for name, tensor in self._ctx.extras_getter():
            tensor.copy_(self._extras_backup[name], non_blocking=True)
        torch.cuda.synchronize()
        if self._expected_hashes is not None:
            self._verify_hashes()

    def get(self, tag: str):
        # [FUNCTION DOC] HELPER: Non-actor tags delegate. For the actor, returns a new mapping
        # from each source tensor's name to a stand-in: the pinned extras backup if the live
        # tensor's object id has one (extras are paused during weight update, so their live
        # values are not usable), otherwise the detached live tensor itself, which must be in the
        # rematerializable id set (assertion error naming the tensor otherwise).
        if tag != "actor":
            return self._others.get(tag)
        # Extras are paused during update_weights. Read them from the pinned backup.
        out = {}
        for name, tensor in self._source_getter():
            backup = self._extras_backup_by_id.get(id(tensor))
            if backup is None:
                assert (
                    id(tensor) in self._ctx.rematerializable_ids
                ), f"{name} is neither in the DDP param buffers nor in the extras backup"
                backup = tensor.detach()
            out[name] = backup
        return out

    def _compute_hashes(self) -> dict[str, str]:
        # [FUNCTION DOC] HELPER: Returns a mapping from each source tensor name to the hex SHA-256
        # digest of its raw bytes. Copies each tensor to host.
        return {name: _hash_tensor_sha256(tensor) for name, tensor in self._source_getter()}

    def _verify_hashes(self) -> None:
        # [FUNCTION DOC] HELPER: Recomputes hashes and compares with the expected ones. Asserts
        # the expected hashes exist and the set of names is unchanged (assertion error listing up
        # to five missing and five extra names). Raises a runtime error listing up to twenty
        # mismatching names, with total counts and the cycle number, if any digest differs.
        actual = self._compute_hashes()
        expected = self._expected_hashes
        assert expected is not None
        assert actual.keys() == expected.keys(), (
            f"main-cast restore changed the tensor set: "
            f"missing={sorted(expected.keys() - actual.keys())[:5]} "
            f"extra={sorted(actual.keys() - expected.keys())[:5]}"
        )
        mismatches = [name for name in expected if actual[name] != expected[name]]
        if mismatches:
            raise RuntimeError(
                f"main-cast weight restore is not bit-identical to the weights at "
                f"backup time for {len(mismatches)}/{len(expected)} tensors "
                f"(cycle {self._backup_count}): {mismatches[:20]}"
            )


def _hash_tensor_sha256(x: torch.Tensor) -> str:
    # [FUNCTION DOC] HELPER: Returns the hex SHA-256 of a tensor's raw memory: detaches, moves to
    # CPU, makes contiguous, flattens and reinterprets as bytes. The hash depends on dtype and
    # element order but not on shape. Synchronizes if the tensor is on GPU.
    """Real (cryptographic) hash: a mismatch here has to mean a bug."""
    data = x.detach().cpu().contiguous()
    return hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
