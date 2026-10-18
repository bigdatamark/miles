# [MODULE DOC]
# HELPER: Iteration utilities: grouping without sorting, and size-bounded chunking of named tensors.
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

import torch


# details: https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
def group_by(iterable, key=None):
    # [FUNCTION DOC] HELPER: Returns a plain mapping from key to the list of items having it, items
    # in input order. The key function defaults to the item itself.
    """Similar to itertools.groupby, but do not require iterable to be sorted"""
    ret = defaultdict(list)
    for item in iterable:
        ret[key(item) if key is not None else item].append(item)
    return dict(ret)


# TODO fsdp can also use this
def chunk_named_params_by_size(named_params: Iterable[tuple[str, torch.Tensor]], chunk_size: int):
    # [FUNCTION DOC] HELPER: Generator yielding lists of (name, tensor) pairs, sized by tensor byte
    # count, each chunk reaching but not exceeding the size where possible (see the chunker).
    return _chunk_by_size(
        named_params,
        compute_size=lambda named_weight: named_weight[1].nbytes,
        chunk_size=chunk_size,
    )


def _chunk_by_size(objects: Iterable[Any], compute_size: Callable[[Any], int], chunk_size: int):
    # [FUNCTION DOC] HELPER: Generator that accumulates objects into a bucket; before adding an
    # object, if the bucket is non-empty and the bucket size plus the object size would reach or
    # exceed the chunk size, yields the bucket and starts a new one. So buckets stay strictly below
    # the chunk size except when a single object is that large (it gets its own bucket). Yields the
    # final non-empty bucket. Input order is preserved.
    bucket: list[Any] = []
    bucket_size = 0

    for obj in objects:
        obj_size = compute_size(obj)

        if bucket and (bucket_size + obj_size) >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0

        bucket.append(obj)
        bucket_size += obj_size

    if bucket:
        yield bucket
