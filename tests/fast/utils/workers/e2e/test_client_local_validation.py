# [MODULE DOC]
# Tests that the rpc worker handle validates calls on the client side. The first group uses a
# record-only proxy to prove that invalid calls (unknown or private methods, bad arguments) never
# send a request. The second group tests handle construction rules and checks that the client's
# method set matches what the real server routes.
import uuid

import httpx
import pytest
from pydantic import ValidationError
from tests.fast.utils.workers.e2e.e2e_worker import E2eWorker

from miles.utils.workers.rpc.client.handle import RpcWorkerHandle


class TestNoRequestIsSent:
    # [CLASS DOC] Each test builds a handle pointed at a record-only proxy, makes an invalid call
    # and asserts the expected local error plus an empty proxy request log.
    async def test_unknown_method_is_an_attribute_error(self, dead_proxy, make_handle):
        # [FUNCTION DOC] Looking up an undefined method name raises an attribute error whose
        # message mentions "no rpc method"; no request is recorded.
        """A method the worker does not define fails before any request goes out."""
        handle = make_handle(dead_proxy)
        with pytest.raises(AttributeError, match="no rpc method"):
            _ = handle.no_such_method
        assert dead_proxy.requests == []

    async def test_private_attribute_lookup_raises_attribute_error_without_sending(self, dead_proxy, make_handle):
        # [FUNCTION DOC] Looking up a private worker method name raises an attribute error whose
        # text is exactly the name; no request is recorded.
        """A private worker method is refused by name alone, so it never becomes a remote call."""
        handle = make_handle(dead_proxy)
        with pytest.raises(AttributeError) as exc_info:
            _ = handle._bump
        assert str(exc_info.value) == "_bump"
        assert dead_proxy.requests == []

    async def test_missing_required_argument(self, dead_proxy, make_handle):
        # [FUNCTION DOC] Calling the sum method with only one argument raises a pydantic
        # validation error; no request is recorded.
        """A missing argument is caught locally, not by the server."""
        handle = make_handle(dead_proxy)
        with pytest.raises(ValidationError):
            await handle.demo_sync(a=1)
        assert dead_proxy.requests == []

    async def test_unknown_argument(self, dead_proxy, make_handle):
        # [FUNCTION DOC] Passing an extra undeclared keyword raises a validation error; no
        # request is recorded.
        """An argument the method does not declare is caught locally."""
        handle = make_handle(dead_proxy)
        with pytest.raises(ValidationError):
            await handle.demo_sync(a=1, b=2, c=3)
        assert dead_proxy.requests == []

    async def test_wrong_argument_type(self, dead_proxy, make_handle):
        # [FUNCTION DOC] Passing a non-numeric string for an integer argument raises a validation
        # error; no request is recorded.
        """An uncoercible argument is caught locally."""
        handle = make_handle(dead_proxy)
        with pytest.raises(ValidationError):
            await handle.demo_sync(a="not-a-number", b=2)
        assert dead_proxy.requests == []

    async def test_too_many_positional_arguments_are_rejected(self, dead_proxy, make_handle):
        # [FUNCTION DOC] Three positionals for a two-parameter method raise a type error whose
        # message contains "takes at most"; no request is recorded.
        """Positionals bind to declared parameters, so one argument too many fails locally."""
        handle = make_handle(dead_proxy)
        with pytest.raises(TypeError, match="takes at most"):
            await handle.demo_sync(1, 2, 3)
        assert dead_proxy.requests == []

    async def test_a_parameter_filled_positionally_and_by_keyword_is_rejected(self, dead_proxy, make_handle):
        # [FUNCTION DOC] Giving a parameter both positionally and by keyword raises a type error
        # mentioning "multiple values"; no request is recorded.
        """Filling the same parameter twice is a caller mistake, caught before any request goes out."""
        handle = make_handle(dead_proxy)
        with pytest.raises(TypeError, match="multiple values"):
            await handle.demo_sync(1, a=2)
        assert dead_proxy.requests == []


class TestHandleConstruction:
    # [CLASS DOC] Tests of how a handle derives its callable methods from a worker class, and
    # that this derivation agrees with the server.
    def test_reserved_method_name_is_rejected(self):
        # [FUNCTION DOC] Constructing a handle for a worker class defining a method named like a
        # handle attribute (wait for readiness) raises a type error mentioning "shadow".
        """A worker whose method shadows a handle attribute is refused at construction."""

        class Shadowing:
            # [CLASS DOC] HELPER: Worker with a method that collides with a handle method.
            async def wait_ready(self, timeout: float) -> None:
                # [FUNCTION DOC] HELPER: Does nothing; never called.
                pass

        with pytest.raises(TypeError, match="shadow"):
            RpcWorkerHandle(Shadowing, server_url="http://127.0.0.1:9")

    def test_worker_without_public_methods_exposes_nothing_to_call(self):
        # [FUNCTION DOC] A worker class with only a private method yields a handle whose method
        # table is empty (private attribute inspected), and looking up the stripped name raises
        # an attribute error mentioning "no rpc method".
        """A worker whose whole value is a lifecycle side effect answers no call, and is still servable."""

        class Empty:
            # [CLASS DOC] HELPER: Worker with only a private method.
            def _demo_hidden(self) -> int:
                # [FUNCTION DOC] HELPER: Returns 1; never called.
                return 1

        handle = RpcWorkerHandle(Empty, server_url="http://127.0.0.1:9")

        assert handle._specs == {}
        with pytest.raises(AttributeError, match="no rpc method"):
            _ = handle.demo_hidden

    def test_worker_with_unannotated_method_is_rejected(self):
        # [FUNCTION DOC] Constructing a handle for a worker with a public method lacking type
        # annotations raises a type error.
        """A method missing annotations is refused on the client too, matching the server."""

        class Unannotated:
            # [CLASS DOC] HELPER: Worker with a method without annotations.
            def demo_unannotated(self, x):
                # [FUNCTION DOC] HELPER: Echoes x; never called.
                return x

        with pytest.raises(TypeError):
            RpcWorkerHandle(Unannotated, server_url="http://127.0.0.1:9")

    async def test_trailing_slash_in_server_url(self, server, make_handle):
        # [FUNCTION DOC] Builds a handle for the real server url with a trailing slash and
        # asserts a sum call returns 2.
        """A server url with a trailing slash still produces valid request paths."""
        handle = make_handle(f"{server.url}/")
        assert await handle.demo_sync(a=1, b=1) == 2

    async def test_client_and_server_agree_on_the_method_set(self, server):
        # [FUNCTION DOC] Builds a handle for the test worker class (no readiness wait) and
        # asserts its method table contains the sum method and not the private counter method.
        # Then, with a raw client (30 s timeout, environment proxies ignored), posts to every
        # method name in the client's table a request with a fresh call id and an unknown query
        # key and asserts status 400 (routable, bad arguments). Posts to the private counter
        # method path and asserts 404.
        """Every method the client exposes is routable on the server, and private ones are not."""
        handle = RpcWorkerHandle(E2eWorker, server_url=server.url)
        assert "demo_sync" in handle._specs and "_bump" not in handle._specs

        async with httpx.AsyncClient(base_url=server.url, timeout=30.0, trust_env=False) as client:
            for name in handle._specs:
                response = await client.post(
                    f"/v1/{name}", json={"call_id": uuid.uuid4().hex, "query": {"__unknown__": 1}}
                )
                assert response.status_code == 400, name

            private = await client.post("/v1/_bump", json={"call_id": uuid.uuid4().hex, "query": {}})
            assert private.status_code == 404
