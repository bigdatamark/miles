# [MODULE DOC] Tests for the mock sglang HTTP server used by tests: closing behavior (existing
# keep-alive connections are severed and new ones refused), request recording (method, path with
# the query string stripped, decoded JSON body), response payloads, port binding, handling of
# malformed requests, and the published address (the server binds all interfaces, publishes the
# node ip or the host ip environment override). Servers are created through a factory fixture
# from the shared conftest that closes them after each test.
from __future__ import annotations

import http.client
import json

import pytest
from tests.fast.utils.test_utils.conftest import connect_via_url

from miles.utils.http_utils import MILES_HOST_IP_ENV
from miles.utils.misc import NodeProbeMixin, get_current_node_ip
from miles.utils.test_utils.mock_sglang_http_server import MockSGLangHttpServer, RecordedRequest


class TestClose:
    # [CLASS DOC] Tests of the server's close method.
    def test_close_severs_an_already_established_connection(self):
        # [FUNCTION DOC] Opens a connection, makes one GET (status 200, body read), closes the
        # server, and asserts a further request on the same connection raises an operating system
        # error and the recorded paths contain only the first request. The server is created
        # directly (not via the factory) and closed by the test itself.
        """A crashed engine must stop serving a client that already holds a keep-alive connection."""
        server = MockSGLangHttpServer()
        connection = http.client.HTTPConnection(server.host, server.port, timeout=5)
        connection.request("GET", "/before")
        first = connection.getresponse()
        assert first.status == 200
        first.read()

        server.close()

        with pytest.raises(OSError):
            connection.request("GET", "/after")
            connection.getresponse().read()
        assert server.paths == ["/before"]

    def test_close_refuses_a_fresh_connection(self):
        # [FUNCTION DOC] Closes a server, then asserts a new connection to its former host and
        # port fails to send a request with an operating system error.
        """A crashed engine must not accept new connections either."""
        server = MockSGLangHttpServer()
        host, port = server.host, server.port
        server.close()

        connection = http.client.HTTPConnection(host, port, timeout=5)
        with pytest.raises(OSError):
            connection.request("GET", "/after")


class TestRecording:
    # [CLASS DOC] Tests of request recording on the server (requests list, paths list and the
    # payloads-of query).
    def test_get_and_post_record_methods_normalized_paths_and_payloads(self, make_server):
        # [FUNCTION DOC] Sends a GET with a query string, a POST with a JSON body and a POST with
        # a query string and no body. Asserts the recorded requests, in order, have the method,
        # the path without the query, and the decoded payload (none when no body); the payloads of
        # the release memory path give the single body; the payloads of an unused path are empty.
        """Every request is recorded in arrival order with its method, query-stripped path and decoded JSON body."""
        server = make_server()
        connection = connect_via_url(server)

        connection.request("GET", "/get_model_info?verbose=1")
        connection.getresponse().read()
        connection.request(
            "POST",
            "/release_memory_occupation",
            body=json.dumps({"tags": ["weights"]}),
            headers={"Content-Type": "application/json"},
        )
        connection.getresponse().read()
        connection.request("POST", "/resume_memory_occupation?tags=kv_cache")
        connection.getresponse().read()
        connection.close()

        assert server.requests == [
            RecordedRequest(method="GET", path="/get_model_info", payload=None),
            RecordedRequest(method="POST", path="/release_memory_occupation", payload={"tags": ["weights"]}),
            RecordedRequest(method="POST", path="/resume_memory_occupation", payload=None),
        ]
        assert server.payloads_of("/release_memory_occupation") == [{"tags": ["weights"]}]
        assert server.payloads_of("/never_called") == []

    def test_requests_returns_an_isolated_snapshot(self, make_server):
        # [FUNCTION DOC] After one GET, appends a forged request to the list returned by the
        # requests property and asserts the server's own recorded paths still contain only the
        # real request.
        """Mutating the list handed back by ``requests`` must not corrupt the server's own history."""
        server = make_server()
        connection = connect_via_url(server)
        connection.request("GET", "/first")
        connection.getresponse().read()
        connection.close()

        snapshot = server.requests
        snapshot.append(RecordedRequest(method="GET", path="/forged", payload=None))

        assert server.paths == ["/first"]


class TestResponse:
    # [CLASS DOC] Tests of the response body and headers.
    def test_default_and_configured_payloads_are_returned_as_json(self, make_server):
        # [FUNCTION DOC] A default server answers a GET with status 200, a JSON content type and
        # a body marking it as a mock; a server created with a response payload answers with
        # exactly that payload.
        """The server answers on its own ``url`` with a JSON body: a default marker, or the configured payload."""
        default_server = make_server()
        connection = connect_via_url(default_server)
        connection.request("GET", "/get_model_info")
        response = connection.getresponse()
        assert response.status == 200
        assert response.headers["Content-Type"] == "application/json"
        assert json.loads(response.read()) == {"mock": True}
        connection.close()

        configured_server = make_server(response_payload={"model_path": "/fake/model"})
        connection = connect_via_url(configured_server)
        connection.request("GET", "/get_model_info")
        assert json.loads(connection.getresponse().read()) == {"model_path": "/fake/model"}
        connection.close()


class TestPortBinding:
    # [CLASS DOC] Tests of explicit port requests.
    def test_requesting_an_occupied_port_raises(self, make_server):
        # [FUNCTION DOC] Creating a second server on the first server's port raises an operating
        # system error.
        """An explicitly requested port that is already taken must fail loudly, not silently serve elsewhere."""
        occupant = make_server()

        with pytest.raises(OSError):
            make_server(port=occupant.port)


class TestMalformedRequest:
    # [CLASS DOC] Tests of robustness to bad requests.
    def test_malformed_post_is_not_recorded_and_does_not_stop_the_server(self, make_server):
        # [FUNCTION DOC] Sends a POST whose body is not JSON and asserts the client sees an
        # operating system or HTTP error; then, on a fresh connection, a GET to the health path
        # returns 200, and the recorded paths contain only the health path (the bad request was not
        # recorded).
        """A POST body that is not JSON fails only its own request: nothing is recorded, later requests still work."""
        server = make_server()

        bad_connection = connect_via_url(server)
        with pytest.raises((OSError, http.client.HTTPException)):
            bad_connection.request(
                "POST",
                "/update_weights",
                body="not-json",
                headers={"Content-Type": "application/json"},
            )
            bad_connection.getresponse().read()
        bad_connection.close()

        good_connection = connect_via_url(server)
        good_connection.request("GET", "/health")
        assert good_connection.getresponse().status == 200
        good_connection.close()

        assert server.paths == ["/health"]


class TestNodeAddress:
    # [CLASS DOC] Tests of the bind and publish addresses of the server.
    def test_the_server_listens_on_every_interface_and_publishes_the_node_ip(self, make_server):
        # [FUNCTION DOC] Asserts the underlying server is bound to the wildcard address, the
        # published host equals the node ip (not loopback), and the url is http plus node ip plus
        # port. Reads the private server attribute for the bind address.
        """A mock engine's http server must be reachable from another node, not only from loopback."""
        server = make_server()

        assert server._server.server_address[0] == "0.0.0.0"
        assert server.host == NodeProbeMixin._get_node_ip()
        assert server.host != "127.0.0.1"
        assert server.url == f"http://{NodeProbeMixin._get_node_ip()}:{server.port}"


class TestNodeReachability:
    # [CLASS DOC] Tests that the server answers over real sockets on several addresses and honors
    # the host ip environment override for its published address.
    def test_the_server_answers_on_both_loopback_and_the_node_ip(self, make_server):
        # [FUNCTION DOC] For loopback and the current node ip, opens a connection, requests the
        # health path, asserts status 200, and closes. Asserts the server recorded the health path
        # twice.
        """Binding the wildcard address keeps local clients working while remote nodes gain access."""
        server = make_server()
        node_ip = get_current_node_ip()

        for host in ["127.0.0.1", node_ip]:
            connection = http.client.HTTPConnection(host, server.port, timeout=5)
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert response.status == 200
            response.read()
            connection.close()

        assert server.paths == ["/health", "/health"]

    def test_the_published_host_honours_the_node_ip_override(self, monkeypatch, make_server):
        # [FUNCTION DOC] With the environment override set to 10.9.9.9 before the server is
        # created, asserts its url uses that host.
        """A deployment publishes a reachable address through MILES_HOST_IP; the mock must advertise the same one."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.9.9.9")
        server = make_server()

        assert server.url == f"http://10.9.9.9:{server.port}"

    def test_the_override_changes_the_published_host_but_not_the_bind_address(self, monkeypatch, make_server):
        # [FUNCTION DOC] With the override set, asserts the bind address is still the wildcard,
        # the published host is the override, and a real request over loopback still gets 200.
        """The mock listens on every interface and only advertises the override, so an unbound address still serves."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "10.9.9.9")
        server = make_server()

        assert server._server.server_address[0] == "0.0.0.0"
        assert server.host == "10.9.9.9"

        connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        connection.request("GET", "/health")
        assert connection.getresponse().status == 200
        connection.close()

    def test_the_published_host_ignores_an_empty_override(self, monkeypatch, make_server):
        # [FUNCTION DOC] With the override set to the empty string, the published host falls back
        # to the current node ip.
        """An override exported empty carries no address, so falling back keeps the mock reachable from other nodes."""
        monkeypatch.setenv(MILES_HOST_IP_ENV, "")
        server = make_server()

        assert server.host == get_current_node_ip()
