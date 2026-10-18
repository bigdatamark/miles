# [MODULE DOC]
# Cpu tests of the nvme stream optimizer plugin's file-backed buffer helpers: creating a disk-backed
# cpu tensor like a source tensor (same shape and dtype, not pinned, file unlinked at creation),
# bit-exact copying through it, recognition as disk backed, flushing its memory mapping, and reserving
# file size. All tests write only inside the pytest temp directory.
"""The file-backed buffers behind Muon's disk-resident optimizer state.

Guards a silent failure: if the allocator stops returning file-backed storage, the offloader
keeps working against pinned host memory while the log still claims otherwise.
"""

import os

import pytest
import torch

from miles_plugins.optimizers import nvme_stream


def test_disk_buffer_matches_shape_and_dtype_and_is_not_pinned(tmp_path):
    # [FUNCTION DOC] Creates a disk-backed buffer like a random 64x32 float32 tensor in the temp dir and
    # asserts the same shape and dtype, a cpu device, and that it is not pinned memory (the inherited
    # offloader picks its copy path from the pinned flag).
    src = torch.randn(64, 32, dtype=torch.float32)

    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    assert buf.shape == src.shape
    assert buf.dtype == src.dtype
    assert buf.device.type == "cpu"
    # The inherited offloader picks its sync/async copy path off is_pinned().
    assert not buf.is_pinned()


def test_disk_buffer_round_trip_is_bit_exact(tmp_path):
    # [FUNCTION DOC] Copies a random 128x64 float32 tensor into a disk-backed buffer and then into a
    # fresh tensor; asserts the result is exactly equal to the source.
    src = torch.randn(128, 64, dtype=torch.float32)
    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    buf.copy_(src)
    out = torch.empty_like(src)
    out.copy_(buf)

    assert torch.equal(out, src)


def test_disk_buffer_leaves_no_file_behind(tmp_path):
    # [FUNCTION DOC] Creates a buffer (result discarded) and asserts the directory listing is empty,
    # i.e. the backing file was unlinked at creation.
    nvme_stream._disk_backed_like(torch.zeros(8), str(tmp_path))

    # Unlinked at creation, so a killed run leaves no residue.
    assert os.listdir(tmp_path) == []


def test_disk_buffer_is_recognized_as_already_managed(tmp_path):
    # [FUNCTION DOC] Asserts the disk-backed check is true for a disk-backed buffer and false for a
    # plain zeros tensor.
    """Megatron's checkpoint adoption reallocates non-pinned CPU state; ours must be exempt."""
    buf = nvme_stream._disk_backed_like(torch.zeros(32, 8), str(tmp_path))

    assert nvme_stream._is_disk_backed(buf)
    assert not nvme_stream._is_disk_backed(torch.zeros(32, 8))


def test_flush_mapping_covers_the_buffer_and_repeats_cheaply(tmp_path):
    # [FUNCTION DOC] Creates a 1024x256 float32 buffer, fills it with ones, and asserts the mapping
    # flush returns the buffer size in bytes, and returns the same size again on an immediate repeat.
    # Timing of the repeat is not asserted.
    """Checkpointing fsyncs its own files behind the kernel's writeback of ours."""
    buf = nvme_stream._disk_backed_like(torch.zeros(1024, 256), str(tmp_path))
    nbytes = buf.numel() * buf.element_size()
    buf.fill_(1.0)

    assert nvme_stream._flush_mapping(buf) == nbytes
    # Already clean, so the repeat is the cheap case the checkpoint hook relies on.
    assert nvme_stream._flush_mapping(buf) == nbytes


def test_reserve_sizes_the_file(tmp_path):
    # [FUNCTION DOC] Opens a new file in the temp dir read-write with mode 600, reserves 4096 bytes on
    # the descriptor, asserts the file size is 4096, and always closes the descriptor.
    path = tmp_path / "f.bin"
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        nvme_stream._reserve(fd, 4096)
        assert os.fstat(fd).st_size == 4096
    finally:
        os.close(fd)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_disk_buffer_preserves_dtype(tmp_path, dtype):
    # [FUNCTION DOC] Parametrized over float32 and bfloat16. A disk-backed buffer like a 16x4 zeros
    # tensor has the same dtype and the same element count.
    src = torch.zeros(16, 4, dtype=dtype)

    buf = nvme_stream._disk_backed_like(src, str(tmp_path))

    assert buf.dtype is dtype
    assert buf.numel() == src.numel()
