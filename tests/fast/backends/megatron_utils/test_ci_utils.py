# [MODULE DOC] Tests of the Megatron backend's CI utility that converts a tensor to the raw bytes
# used for hashing (deterministic-comparison checks). Covers exact little-endian raw buffer bytes for
# contiguous float32 and int64 tensors (4 and 8 bytes per element), that a non-contiguous transposed
# view does not raise and yields the row-major bytes of its values, and that different contents give
# different bytes. The whole module is skipped when Megatron's distributed package is unavailable.
import pytest

pytest.importorskip("megatron.core.distributed")

import torch

from miles.backends.megatron_utils.ci_utils import _hash_tensor_bytes


def test_hash_tensor_bytes_contiguous_float32_returns_raw_buffer_bytes() -> None:
    # [FUNCTION DOC] For a 2 by 3 arange float32 tensor, asserts the result is a bytes object of 24
    # bytes equal to the flattened tensor's raw bytes.
    """A contiguous 2D float32 tensor hashes to its exact raw little-endian buffer bytes."""
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = _hash_tensor_bytes(tensor)
    assert isinstance(result, bytes)
    assert len(result) == 6 * 4
    assert result == tensor.reshape(-1).contiguous().numpy().tobytes()


def test_hash_tensor_bytes_int64_byte_length_is_eight_per_element() -> None:
    # [FUNCTION DOC] For a 2 by 3 int64 tensor, asserts 48 bytes equal to the flattened raw bytes.
    """An int64 tensor yields 8 bytes per element with bit-exact buffer contents."""
    tensor = torch.arange(6, dtype=torch.int64).reshape(2, 3)
    result = _hash_tensor_bytes(tensor)
    assert len(result) == 6 * 8
    assert result == tensor.reshape(-1).contiguous().numpy().tobytes()


def test_hash_tensor_bytes_noncontiguous_transpose_does_not_raise() -> None:
    # [FUNCTION DOC] For the transpose of a 2 by 3 float32 tensor (asserted non-contiguous), asserts
    # the call returns a bytes object of 24 bytes.
    """A non-contiguous transposed view is hashed without raising (regression guard)."""
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    transposed = base.t()
    assert not transposed.is_contiguous()
    result = _hash_tensor_bytes(transposed)
    assert isinstance(result, bytes)
    assert len(result) == 6 * 4


def test_hash_tensor_bytes_noncontiguous_matches_contiguous_row_major_bytes() -> None:
    # [FUNCTION DOC] Asserts the bytes of a transposed view equal the raw bytes of the same view made
    # contiguous and flattened (i.e. logical row-major order, not storage order).
    """The non-contiguous path returns the contiguous row-major bytes of the transposed values."""
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    transposed = base.t()
    result = _hash_tensor_bytes(transposed)
    assert result == transposed.contiguous().reshape(-1).numpy().tobytes()


def test_hash_tensor_bytes_distinct_values_produce_distinct_bytes() -> None:
    # [FUNCTION DOC] All-zeros and all-ones 2 by 3 float32 tensors must give different byte strings,
    # each of 24 bytes.
    """Different tensor contents of the same shape and dtype produce different bytes."""
    a = torch.zeros(2, 3, dtype=torch.float32)
    b = torch.ones(2, 3, dtype=torch.float32)
    assert _hash_tensor_bytes(a) != _hash_tensor_bytes(b)
    assert len(_hash_tensor_bytes(a)) == len(_hash_tensor_bytes(b)) == 6 * 4
