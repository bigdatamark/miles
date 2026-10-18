# [MODULE DOC] Test of the session route setup function: given a web application, a proxy backend
# and a session server configuration, it registers the session HTTP routes; a configuration without
# a model checkpoint must leave the application untouched and must not load a tokenizer.
import pytest
from fastapi import FastAPI
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session import sessions
from miles.rollout.session.sessions import setup_session_routes


class _UnusedBackend:
    # [CLASS DOC] HELPER: Proxy backend stand-in that must never be used.
    async def do_proxy(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Always raises an assertion error saying route setup must not
        # touch the proxy backend.
        raise AssertionError("setup_session_routes must not touch the proxy backend")


class TestSetupSessionRoutes:
    # [CLASS DOC] Tests of the session route setup function.
    def test_without_hf_checkpoint_registers_no_session_routes(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Replaces the module's tokenizer loader with a recorder that raises an
        # assertion error, builds an empty web application, and sets up routes with a config that
        # has no checkpoint. Asserts the loader was never called, the application's route paths
        # are unchanged, and the health route was not present beforehand.
        """A config without a checkpoint loads no tokenizer and leaves the app's routes untouched."""
        tokenizer_calls: list[tuple] = []

        def record_tokenizer_load(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Records the call arguments and then raises an assertion error
            # saying no tokenizer must be loaded without a checkpoint.
            tokenizer_calls.append((args, kwargs))
            raise AssertionError("setup_session_routes must not load a tokenizer without a checkpoint")

        monkeypatch.setattr(sessions, "load_tokenizer", record_tokenizer_load)
        app = FastAPI()
        before = [route.path for route in app.routes]

        setup_session_routes(app, _UnusedBackend(), make_session_server_config(hf_checkpoint=None))

        assert tokenizer_calls == []
        assert [route.path for route in app.routes] == before
        assert "/health" not in before
