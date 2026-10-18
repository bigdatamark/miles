# [MODULE DOC] Tests for the docker image-inputs helper script (docker/image_inputs.py), imported
# after putting the repository's docker directory at the front of the import path (a global side
# effect at import time). Checks which repository paths count as image inputs (Dockerfile,
# requirements, cu13 patches count; the ROCm Dockerfile, AMD helper, source files and docs do
# not), that the input hash is deterministic and changes when a declared input changes, and that
# the image label reader extracts the inputs hash from single-arch and multi-arch manifests and
# returns an empty string for unlabelled, empty or malformed manifests.
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker"))

import image_inputs  # noqa: E402


def test_dockerfile_and_requirements_are_inputs():
    # [FUNCTION DOC] The Dockerfile, the requirements file and a cu13 patch file all match the
    # input globs.
    assert image_inputs._matches("docker/Dockerfile")
    assert image_inputs._matches("requirements.txt")
    assert image_inputs._matches("docker/patch/cu13/some.patch")


def test_rocm_dockerfile_is_not_a_cu13_input():
    # [FUNCTION DOC] The ROCm Dockerfile and the AMD prebuild script do not match the input globs.
    """pr-test.yml builds the cu13 image only; ROCm has its own pipeline."""
    assert not image_inputs._matches("docker/Dockerfile.rocm")
    assert not image_inputs._matches("docker/amd/prebuild_aiter_jit.py")


def test_source_changes_are_not_image_inputs():
    # [FUNCTION DOC] A framework source file and the docker readme do not match the input globs.
    assert not image_inputs._matches("miles/train.py")
    assert not image_inputs._matches("docker/README.md")


def test_compute_is_deterministic():
    # [FUNCTION DOC] Computing the input hash twice in a row gives equal results.
    assert image_inputs.compute() == image_inputs.compute()


def test_compute_tracks_every_declared_input(tmp_path, monkeypatch):
    # [FUNCTION DOC] Takes the baseline hash, then patches the path lister to return only the
    # requirements file and the content reader to return fixed bytes (patches reverted by
    # monkeypatch); asserts the new hash differs from the baseline. The temporary path fixture is
    # unused.
    """A file matching INPUT_GLOBS must move the hash, or rebuilds get skipped wrongly."""
    baseline = image_inputs.compute()
    monkeypatch.setattr(image_inputs, "_paths_at", lambda rev, root: ["requirements.txt"])
    monkeypatch.setattr(image_inputs, "_content_at", lambda path, rev, root: b"changed")
    assert image_inputs.compute() != baseline


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        (
            json.dumps(
                {
                    "linux/amd64": {"config": {"Labels": {image_inputs.LABEL_KEY: "abc"}}},
                    "linux/arm64": {"config": {"Labels": {image_inputs.LABEL_KEY: "abc"}}},
                }
            ),
            "abc",
        ),
        (json.dumps({"config": {"Labels": {image_inputs.LABEL_KEY: "def"}}}), "def"),
        (json.dumps({"linux/amd64": {"config": {"Labels": {"other": "x"}}}}), ""),
        ("", ""),
        ("not json", ""),
    ],
    ids=["multi-arch", "single-arch", "unlabelled", "unpublished", "malformed"],
)
def test_read_label(manifest, expected):
    # [FUNCTION DOC] Parametrized over five manifest texts (multi-arch with equal labels,
    # single-arch, unlabelled, empty, malformed JSON): reading the label returns the labelled
    # hash value or an empty string when absent or unparsable.
    assert image_inputs.read_label(manifest) == expected
