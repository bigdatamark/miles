# [MODULE DOC] HELPER: Pytest configuration for the harbor tests. At import it prepends the example
# directory (from the package constant) to the module search path if it is not already present as
# a string entry, so the example modules import by bare name. Side effect: mutates the process
# module search path.
"""Put the example on sys.path so its module imports by bare name, the way a
rollout loads it (``--custom-agent-function-path``)."""

import sys

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
