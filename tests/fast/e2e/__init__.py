# [PACKAGE DOC]
# Fast (CPU only, no cluster) tests of the support code used by the end-to-end test suites under
# the e2e test tree. These tests exercise argument builders, assertion helpers and scenario
# planning logic with fakes, so they run in the always-on CPU stage without GPUs or a cluster.
# The package itself defines no names; the subpackages mirror the e2e layout (deploy tests).
