# [PACKAGE DOC]
# Fast offline tests for the NeMo Gym example, which lives outside the tests tree in a directory
# whose name (with a hyphen) is not a python identifier. The package defines one constant, the
# example directory path (examples/experimental/nemo-gym under the repository root, five levels
# above this file's directory), which the sibling conftest puts on the import path.
"""Tests for the NeMo Gym example, which lives outside this tree.

The example directory is ``nemo-gym`` -- not a Python identifier -- so nothing
here can mirror its name; ``EXAMPLE_DIR`` is what conftest puts on sys.path so
its modules import by bare name.
"""

from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[5] / "examples" / "experimental" / "nemo-gym"
