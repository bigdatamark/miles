# [PACKAGE DOC]
# End-to-end tests for the worker rpc serving stack. Each test spawns a real serve process (the
# test worker, harness and fixtures live in sibling modules) and talks to it over http with the
# rpc worker handle or with raw requests, covering the happy path, concurrency, deduplication,
# retries, cancellation, lifecycle, polling, transport and error handling.
