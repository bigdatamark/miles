# [PACKAGE DOC]
# Fast tests for the in-process Harbor example. The package defines one constant, the example
# directory, computed as the examples/experimental/harbor folder five levels above this file's
# directory (at the repository root); the sibling conftest puts it on the import path.
"""Tests for the in-process Harbor example, which lives outside this tree.

``EXAMPLE_DIR`` is put on sys.path by conftest so the module imports by bare
name, the way a rollout loads it (``--custom-agent-function-path``).
"""

from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[5] / "examples" / "experimental" / "harbor"
