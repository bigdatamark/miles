# [PACKAGE DOC] Packed-sequence (several documents concatenated into one row) support for the
# FSDP backend. Two parts re-exported here: the boundary derivation (a context object describing
# where each packed document starts and ends, and a function that builds it) and the packing
# patch registry (register, look up and apply per-architecture patches that reset stateful layers
# at document boundaries).
"""Unified packed-sequence layout handling for the FSDP backend.

One registry + one boundary derivation; each stateful arch registers a ``PackingPatch`` in ``specs/``
(or nothing when it packs natively). See ``registry`` and ``boundaries``.
"""

from .boundaries import PackedSeqContext, packed_seq_context
from .registry import PackingPatch, apply_packing, get_packing_patches, register_packing_patch

__all__ = [
    "PackedSeqContext",
    "packed_seq_context",
    "PackingPatch",
    "register_packing_patch",
    "get_packing_patches",
    "apply_packing",
]
