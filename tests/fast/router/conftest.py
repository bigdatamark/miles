# [MODULE DOC] Pytest configuration for the router tests. Defines no fixtures of its own; it only
# re-exports the class-scoped router environment fixture (a mock SGLang upstream plus a session
# server) from the session tests module so that other test modules in this package can request it
# by name. Importing it here has the side effect of importing that test module at collection time.
"""Shared fixtures for router tests.

The class-scoped ``router_env`` (mock SGLang upstream + session server) is
defined in test_sessions.py; re-exported here so additive test modules can
depend on it without importing a name their own test signatures would shadow.
"""

from tests.fast.router.test_sessions import router_env  # noqa: F401
