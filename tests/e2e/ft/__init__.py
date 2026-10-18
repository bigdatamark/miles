# [PACKAGE DOC]
# End-to-end fault-tolerance (FT) tests: training runs on a cluster during which trainer and/or
# rollout processes are killed (randomly, or at planned points), asserting that the run recovers
# and either finishes with healthy metrics (random-crash and realistic GSM8K soak tests) or
# produces the same deterministic results as an undisturbed baseline run (deterministic and
# no-failure/with-failure trainer and rollout comparisons, on dense and 5-layer MoE models with
# various parallel layouts, with real or fake rollout). The test files in this directory are thin
# entry points; their scenarios, command-line options, execution helpers, comparisons and fault
# injection machinery live in the helper package named conftest_ft next to them. The package
# itself defines no code.
