# [PACKAGE DOC] End-to-end test package for SGLang rollout-server behavior in miles. It holds
# tests that start a real SGLang server or a router and compare endpoints, routing replay, session
# servers and session API versions; the subpackage "utils" supplies the server launcher, a custom
# generate function and a tool-calling agent used by these tests, and the subpackage
# "test_session_server_multi_role" holds per-model multi-role session-server cases. The package
# itself defines no code and exports nothing; it only makes the test modules importable by their
# dotted path (for example as a custom generate function path).
