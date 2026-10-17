# [MODULE DOC] Shared helpers for checksum rules: an immutable issue record for one differing
# key, a comparison of two flat mappings, and a flattener turning nested mappings/sequences into a
# flat mapping with path-style keys. Pure functions.

from collections.abc import Iterable
from typing import Any

from miles.utils.pydantic_utils import FrozenStrictBaseModel


class ChecksumMismatchIssue(FrozenStrictBaseModel):
    # [CLASS DOC] One mismatch: the flat key, the labels of the two compared sides, and the
    # string forms of the two values (the placeholder "<missing>" stands for an absent key).
    key: str
    label_a: str
    label_b: str
    value_a: str
    value_b: str


def compare_flat_dicts(
    a: dict[str, Any],
    b: dict[str, Any],
    label_a: str,
    label_b: str,
) -> Iterable[ChecksumMismatchIssue]:
    # [FUNCTION DOC] Generator over the union of both mappings' keys in sorted order. A key absent
    # from a side takes the placeholder "<missing>" for that side. Yields a mismatch record
    # (key, both labels, string forms of both values) for each key whose two values differ by
    # equality, and nothing for equal keys.
    """Compare two flat dicts and yield one mismatch per differing key."""
    all_keys = sorted(set(a.keys()) | set(b.keys()))

    for key in all_keys:
        value_a = a.get(key, "<missing>")
        value_b = b.get(key, "<missing>")
        if value_a != value_b:
            yield ChecksumMismatchIssue(
                key=key,
                label_a=label_a,
                label_b=label_b,
                value_a=str(value_a),
                value_b=str(value_b),
            )


def flatten_nested(obj: Any, *, prefix: str = "") -> dict[str, Any]:
    # [FUNCTION DOC] Recursively flattens. Mappings are traversed with keys sorted by string
    # form; a child key is the parent path, a dot and the key (just the key at the top, when the
    # prefix is empty). Lists and tuples are traversed by position, appending the bracketed index
    # to the path with no dot. Any other value is a leaf stored under the accumulated path.
    # Returns the flat mapping. Edge cases: an empty container contributes nothing; a bare leaf at
    # the top level is stored under the (possibly empty) prefix.
    """Flatten a nested dict/list into a flat dict with dot-separated keys. Keeps all primitive leaf values."""
    result: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda x: str(x[0])):
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            result.update(flatten_nested(v, prefix=child_prefix))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            result.update(flatten_nested(v, prefix=f"{prefix}[{i}]"))
    else:
        result[prefix] = obj

    return result
