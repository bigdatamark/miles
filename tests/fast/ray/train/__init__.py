# [PACKAGE DOC]
# Test package for the trainer side of the ray orchestration layer (trainer cells, the cell health
# monitor, and the trainer controller group). It is a regular package, rather than a plain
# directory, so that its sibling modules (a shared conftest with factories and fixtures, a dummy
# actor and a fake worker manager) can be imported by absolute module path from the test modules.
# The package itself defines nothing.
