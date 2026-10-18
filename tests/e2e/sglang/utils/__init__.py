# [PACKAGE DOC] HELPER: Shared helper package for the SGLang end-to-end tests in the parent test
# package. It contains a launcher that starts and stops a standalone SGLang server subprocess
# (module sglang_server), a custom rollout generate function that dumps per-sample routing-replay
# data for the router-equivalence test (module router_equivalence_generate), and a custom
# tool-calling agent function used by session-server tests (module session_tool_agent). The
# package itself defines no code and exports nothing; modules are imported by dotted path, and the
# generate and agent functions are also referenced by path strings in command-line arguments.
