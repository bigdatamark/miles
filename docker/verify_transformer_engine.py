# [MODULE DOC]
# Docker image sanity check, run after the three TransformerEngine wheels (the umbrella package,
# the CUDA-flavoured core package, and the torch extension package) were installed without
# dependency resolution. The script takes the core distribution name as its single command line
# argument and fails with an assertion error (non-zero exit) if anything is off: the three
# distributions are not all at the pinned version, the torch extension does not require exactly
# that version of the core package, the transformer_engine module cannot be found, or any other
# runtime dependency declared by the torch extension is not installed. It inspects installed
# package metadata only and does not import TransformerEngine.
"""Verify the installed TransformerEngine wheel triplet."""

import importlib.metadata as metadata
import importlib.util
import re
import sys


VERSION = "2.17.0"


def _dist_name(requirement: str) -> str:
    # [FUNCTION DOC] HELPER: Extracts the distribution name from a requirement string: splits at
    # the first version operator, semicolon (marker), bracket (extras) or space, lowercases it and
    # replaces underscores with hyphens.
    """'onnxscript', 'torch>=2.1', 'foo; extra == \"bar\"' -> normalised dist name."""
    return re.split(r"[<>=!~;\[ ]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def verify(core_dist: str) -> None:
    # [FUNCTION DOC] Verifies the installed TransformerEngine triplet; returns nothing and raises
    # an assertion error with a descriptive message on the first failed check. The core
    # distribution name (underscores normalised to hyphens) plus the umbrella and torch extension
    # distributions must all be installed at exactly the pinned version. The torch extension's
    # declared requirements (lowercased, normalised, spaces removed) must contain an exact pin of
    # the core distribution at that version. The transformer_engine module must be locatable.
    # Every other declared requirement whose name does not start with the TransformerEngine prefix
    # must be installed (looked up by name). A missing package metadata entry for the three
    # distributions raises the metadata not-found error.
    # KNOWN ISSUE: requirements guarded by an environment marker or optional extra marker are
    # still demanded as installed, so an optional-extra dependency would fail the check.
    core = core_dist.replace("_", "-")
    expected = {
        "transformer-engine": VERSION,
        core: VERSION,
        "transformer-engine-torch": VERSION,
    }
    actual = {name: metadata.version(name) for name in expected}
    raw_requires = metadata.requires("transformer-engine-torch") or []
    requires = {requirement.lower().replace("_", "-").replace(" ", "") for requirement in raw_requires}

    assert actual == expected, f"unexpected TransformerEngine versions: {actual}"
    assert f"{core}=={VERSION}" in requires, f"unexpected TransformerEngine torch requirements: {requires}"
    assert importlib.util.find_spec("transformer_engine") is not None, "transformer_engine package not found"

    # The triplet is installed with --no-deps, so nothing makes transformer_engine_torch's
    # own runtime deps present. Missing ones do not surface until something imports
    # transformer_engine.pytorch, which is far too late -- a missing onnxscript shipped
    # a green image that failed every GPU test.
    missing = []
    for requirement in raw_requires:
        name = _dist_name(requirement)
        if name.startswith("transformer-engine"):
            continue  # pinned above
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    assert not missing, f"transformer_engine_torch runtime deps not installed: {missing}"


if __name__ == "__main__":
    verify(sys.argv[1])
