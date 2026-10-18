# [MODULE DOC] Unit test for the disk-delta tensor reader factory: a reader built for a directory
# of safetensors files returns the raw bytes of a named tensor and validates the caller's declared
# dtype and shape against the stored layout, raising a value error on mismatch. Uses a temporary
# directory and a small float16 array; CPU only.
import numpy as np
import pytest
import safetensors.numpy

from miles.utils.disk_delta import make_tensor_reader


def test_tensor_reader_validates_declared_layout(tmp_path):
    # [FUNCTION DOC] Saves a 2x3 float16 array named "weight" into a safetensors file in the temp
    # dir, builds a reader for that dir, and checks: reading with dtype "F16" and shape (2, 3)
    # returns the flat uint8 byte view of the array; reading with the wrong dtype ("BF16") or the
    # wrong shape ((3, 2)) raises a value error whose message names "dtype=F16, shape=(2, 3)" (the
    # stored layout).
    expected = np.arange(6, dtype=np.float16).reshape(2, 3)
    safetensors.numpy.save_file({"weight": expected}, tmp_path / "model.safetensors")
    read = make_tensor_reader(str(tmp_path))

    actual = read("weight", expected_dtype="F16", expected_shape=(2, 3))
    np.testing.assert_array_equal(actual, expected.view(np.uint8).reshape(-1))

    with pytest.raises(ValueError, match="dtype=F16, shape=\\(2, 3\\)"):
        read("weight", expected_dtype="BF16", expected_shape=(2, 3))
    with pytest.raises(ValueError, match="dtype=F16, shape=\\(2, 3\\)"):
        read("weight", expected_dtype="F16", expected_shape=(3, 2))
