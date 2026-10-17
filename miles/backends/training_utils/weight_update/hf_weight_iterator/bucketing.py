# [MODULE DOC]
# Pure-Python helpers for weight update streaming. A "unit" is a list of (HF name, tensor) pairs
# produced from one training-side parameter and is indivisible. First, units that belong to an
# atomic update group are merged so their members always travel together; second, units are packed
# into buckets under a byte budget.
"""Bucketing for weight update: atomic grouping + size-bounded packing.

A unit is the HF-named tensors one training-side parameter converted into
(a plain weight, or weight + quant scales); units are indivisible.
"""

import dataclasses
from collections.abc import Iterable, Iterator

import torch


@dataclasses.dataclass(frozen=True)
class AtomicUpdateGroup:
    # [CLASS DOC] Immutable description of one group: a unique key string, a tuple of distinct,
    # non-empty name suffixes (one member per suffix) and an optional flag (default false) saying the
    # group may match no parameter at all (otherwise matching nothing is an error).
    """HF-name suffixes whose tensors must land in the same load call."""

    key: str
    suffixes: tuple[str, ...]
    optional: bool = False


def assemble_atomic_update_groups(
    hf_param_units: Iterable[list[tuple[str, torch.Tensor]]],
    atomic_update_groups: list[AtomicUpdateGroup],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    # [FUNCTION DOC] Generator over an iterable of units. Validation up front (assertion errors):
    # every group has suffixes, none empty, no duplicates within a group, and group keys are unique.
    # Streaming: empty units are skipped. A unit is matched if any of its tensor names ends with some
    # group's suffix; a unit matching more than one (group, suffix) pair is an assertion error. Units
    # with no match are yielded unchanged and immediately. A matching unit is parked in a slot keyed
    # by (name prefix before the suffix, group key), at the position of its suffix; a duplicate
    # member for the same slot is an assertion error. When all slots of a prefix and group are
    # filled, one merged unit (members concatenated in suffix order) is yielded and the slot is
    # freed. At end of stream: any incomplete group is an assertion error, and any non-optional group
    # that matched nothing is an assertion error. Memory: parked units are held until complete.
    """Merge units whose members match an AtomicUpdateGroup into one unit,
    keyed by the matching member's name prefix; other units pass through."""
    for group in atomic_update_groups:
        assert group.suffixes, f"Atomic update group {group.key} has no suffixes"
        assert all(group.suffixes), f"Atomic update group {group.key} contains empty suffix"
        assert len(set(group.suffixes)) == len(
            group.suffixes
        ), f"Atomic update group {group.key} contains duplicate suffixes"
    keys = [group.key for group in atomic_update_groups]
    assert len(set(keys)) == len(keys), f"Duplicate atomic update group keys: {keys}"

    pending: dict[tuple[str, str], list] = {}
    matched_group_keys: set[str] = set()
    for unit in hf_param_units:
        if not unit:
            continue
        matches = [
            (group, suffix_idx, suffix, name)
            for name, _tensor in unit
            for group in atomic_update_groups
            for suffix_idx, suffix in enumerate(group.suffixes)
            if name.endswith(suffix)
        ]
        assert len(matches) <= 1, f"Unit {[n for n, _ in unit]} matches multiple atomic group suffixes"
        if not matches:
            yield unit
            continue
        group, suffix_idx, suffix, name = matches[0]
        matched_group_keys.add(group.key)
        prefix = name[: -len(suffix)]
        slots = pending.setdefault((prefix, group.key), [None] * len(group.suffixes))
        assert slots[suffix_idx] is None, f"Duplicate member in atomic update group: {name}"
        slots[suffix_idx] = unit
        if None not in slots:
            yield [named for slot in slots for named in slot]
            del pending[(prefix, group.key)]

    assert not pending, f"Incomplete atomic update groups at end of stream: {sorted(pending)}"
    for group in atomic_update_groups:
        assert (
            group.optional or group.key in matched_group_keys
        ), f"Atomic update group {group.key} matched no params (suffixes {group.suffixes})"


def pack_units_by_size(
    hf_param_units: Iterable[list[tuple[str, torch.Tensor]]],
    max_bytes: int,
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    # [FUNCTION DOC] Generator of buckets (flat lists of pairs). Greedy in-order packing: a unit's
    # size is the sum of its tensors' byte sizes; before adding a unit, if the current bucket is
    # non-empty and the bucket size plus the unit size reaches or exceeds the limit, the bucket is
    # yielded and a new one started; a unit larger than the limit forms its own bucket. The final
    # non-empty bucket is yielded. KNOWN ISSUE: the docstring says buckets are at most the limit but
    # the comparison is greater-or-equal, so multi-unit buckets stay strictly below the limit.
    """Pack units into buckets <= max_bytes, never splitting a unit."""
    bucket: list[tuple[str, torch.Tensor]] = []
    bucket_size = 0
    for unit in hf_param_units:
        unit_size = sum(tensor.nbytes for _name, tensor in unit)
        if bucket and bucket_size + unit_size >= max_bytes:
            yield bucket
            bucket = []
            bucket_size = 0
        bucket.extend(unit)
        bucket_size += unit_size
    if bucket:
        yield bucket
