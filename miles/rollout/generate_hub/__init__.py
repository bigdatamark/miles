# [PACKAGE DOC] Built-in per-sample generate functions selectable by path: single-turn generation, multi-turn
# generation with tool calling, a generic agentic generate that delegates to a user agent function through
# a traced session server, and a benchmarking generate with random output length. Each generate function
# takes a generate input record and returns a generate output record (one sample or a list of samples),
# and may expose an add-arguments hook to register its own command line options.
# TODO: maybe move `sglang_rollout::generate` to this folder
