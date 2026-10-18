# [PACKAGE DOC] Random fault injection support for fault-tolerance (FT) end-to-end tests. Contains
# the injection loop that periodically crashes cells of an FT-enabled training run (core), a
# background handle that runs the loop in a worker thread (entrypoint), the fault forms that
# actually crash a cell (fault_forms), pod-level kubectl helpers (pod_manipulation), the event
# log and observed-cell-state model that records what happened (state), and post-run views over
# that log used by test assertions (views). The package itself exports nothing.
