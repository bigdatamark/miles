# [PACKAGE DOC] Test package for fast GPU tests: tests that need a GPU but finish in a few minutes
# (registered for the two-GPU stage-b CI suite). It contains single-process kernel tests (under the
# "kernels" subdirectory), reference linear-attention implementations shared with e2e precision
# tests, and multi-process worker scripts (for example the FSDP hybrid-shard gradient parity
# worker) that other tests launch. The package itself defines no code and exports nothing; it makes
# the modules importable and lets pytest collect the tests in the directory.
