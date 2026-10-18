# [MODULE DOC]
# Tests for the canonical CI label registry tests.ci.labels.KNOWN_LABELS, a mapping from domain
# label key to a description. Checks its type, that the initial canonical labels are present, that
# meta labels (image, all) are excluded, and that every description is a non-blank string.
# Registers itself as a CPU test in suite stage-a-cpu.
"""Tests for the canonical CI label registry."""

from tests.ci.ci_register import register_cpu_ci
from tests.ci.labels import KNOWN_LABELS

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


def test_known_labels_is_dict():
    # [FUNCTION DOC] Asserts the registry is a dictionary.
    assert isinstance(KNOWN_LABELS, dict)


def test_known_labels_initial_labels_present():
    # [FUNCTION DOC] Asserts the registry contains at least megatron, model-scripts, sglang, fsdp,
    # short, long, ckpt, lora, precision and weight-update; the failure message lists missing ones.
    expected = {
        "megatron",
        "model-scripts",
        "sglang",
        "fsdp",
        "short",
        "long",
        "ckpt",
        "lora",
        "precision",
        "weight-update",
    }
    assert expected <= set(KNOWN_LABELS), f"Missing canonical labels: {expected - set(KNOWN_LABELS)}"


def test_meta_labels_excluded():
    # [FUNCTION DOC] Asserts image and all (meta labels that bypass per-test filtering) are not keys
    # of the registry.
    # Meta-labels (`run-ci-image`, `run-ci-all`) bypass the per-test labels
    # filter and must not be advertised as domain labels.
    assert "image" not in KNOWN_LABELS
    assert "all" not in KNOWN_LABELS


def test_descriptions_nonempty():
    # [FUNCTION DOC] Asserts each description is a string with non-whitespace content.
    for key, desc in KNOWN_LABELS.items():
        assert isinstance(desc, str) and desc.strip(), f"Label {key!r} has empty / non-string description: {desc!r}"
