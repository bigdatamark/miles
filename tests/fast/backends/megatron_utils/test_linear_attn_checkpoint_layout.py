# [MODULE DOC]
# Tests for the linear-attention layout guard in the Megatron checkpoint module: detecting models
# that contain linear attention, rejecting distributed checkpoints that lack the layout marker,
# and publishing a freshly converted iteration directory as the release checkpoint. Uses real
# on-disk checkpoints written without a process group into pytest temporary directories.
"""assert_linear_attn_checkpoint_is_current: a torch_dist checkpoint without the linear-attention layout
marker is rejected before Megatron would load it half-initialized."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter

from miles.backends.megatron_utils.checkpoint import (
    LINEAR_ATTN_LAYOUT_KEY,
    _has_linear_attention,
    assert_linear_attn_checkpoint_is_current,
    publish_as_release,
)

LAYER = "decoder.layers.0.self_attention.linear_attn"


def _write(path, keys):
    # [FUNCTION DOC] HELPER: Creates the directory (with parents; fails if it exists) and saves a
    # distributed checkpoint there, without a process group, containing one zero tensor of two
    # elements per given key.
    path.mkdir(parents=True)
    dcp.save({key: torch.zeros(2) for key in keys}, storage_writer=FileSystemWriter(path), no_dist=True)


def test_release_checkpoint_with_the_marker_passes(tmp_path):
    # [FUNCTION DOC] Writes a release checkpoint holding a linear-attention projection weight and
    # the layout marker, plus a latest-iteration tracker file saying release. Asserts the guard
    # accepts it (raises nothing).
    _write(tmp_path / "release", [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.{LINEAR_ATTN_LAYOUT_KEY}"])
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    assert_linear_attn_checkpoint_is_current(str(tmp_path))


def test_iteration_checkpoint_without_the_marker_is_rejected(tmp_path):
    # [FUNCTION DOC] Writes iteration 12 with linear-attention weights but no marker (old
    # convolution-weight layout) and a tracker file saying 12. Asserts the guard raises a value
    # error whose message tells the user to re-convert from the HuggingFace checkpoint.
    _write(tmp_path / "iter_0000012", [f"{LAYER}.in_proj_qkv.weight", f"{LAYER}.conv1d.weight"])
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("12")
    with pytest.raises(ValueError, match="Re-convert it from the HuggingFace checkpoint"):
        assert_linear_attn_checkpoint_is_current(str(tmp_path))


def test_only_models_with_linear_attention_are_checked():
    # [FUNCTION DOC] Builds a module list entry that contains a submodule registering the layout
    # marker as a buffer, and a plain linear-only model. Asserts the linear-attention detector
    # reports true for the first and false for the second.
    class Core(torch.nn.Module):
        # [CLASS DOC] HELPER: Minimal module whose only content is a one-element ones buffer
        # named with the layout marker key, standing in for a linear-attention core.
        def __init__(self):
            # [FUNCTION DOC] HELPER: Initializes the module and registers the marker buffer.
            super().__init__()
            self.register_buffer(LINEAR_ATTN_LAYOUT_KEY, torch.ones(1))

    with_layer = torch.nn.Sequential(torch.nn.Linear(2, 2), Core())
    assert _has_linear_attention([with_layer])
    assert not _has_linear_attention([torch.nn.Sequential(torch.nn.Linear(2, 2))])


def _converted(save_dir, marker):
    # [FUNCTION DOC] HELPER: Simulates a conversion output by creating the iteration-1 directory
    # under the save directory with a weights text file whose content is the given marker string.
    iteration = save_dir / "iter_0000001"
    iteration.mkdir()
    (iteration / "weights.txt").write_text(marker)


def test_a_first_conversion_becomes_the_release(tmp_path):
    # [FUNCTION DOC] With no existing release, publishes the converted iteration. Asserts the
    # release directory holds the new weights and the iteration directory is gone.
    _converted(tmp_path, "new")

    publish_as_release(str(tmp_path))

    assert (tmp_path / "release" / "weights.txt").read_text() == "new"
    assert not (tmp_path / "iter_0000001").exists()


def test_a_reconversion_replaces_the_previous_release(tmp_path):
    # [FUNCTION DOC] Pre-creates a stale release with two files, then publishes a new conversion.
    # Asserts the release now has the new weights, the stale-only file is gone, and the iteration
    # directory was not nested inside the release (publishing must replace, not move into).
    """shutil.move into an existing directory nests, which would leave the old weights in place."""
    stale = tmp_path / "release"
    stale.mkdir()
    (stale / "weights.txt").write_text("stale")
    (stale / "only-in-the-old-one.txt").write_text("stale")
    _converted(tmp_path, "new")

    publish_as_release(str(tmp_path))

    assert (tmp_path / "release" / "weights.txt").read_text() == "new"
    assert not (tmp_path / "release" / "only-in-the-old-one.txt").exists()
    assert not (tmp_path / "release" / "iter_0000001").exists()
