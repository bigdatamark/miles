# [MODULE DOC]
# Tests for the SGLang engine's URL helpers: bracketing bare IPv6 addresses (idempotently, leaving
# IPv4 addresses, hostnames and missing values untouched) and building an HTTP server URL from a
# host and port so that bracketed and bare IPv6 hosts give the same result. Skipped when SGLang is
# not installed.
import pytest

pytest.importorskip("sglang")

from miles.backends.sglang_utils.sglang_engine import build_server_url, format_v6_uri  # noqa: E402


class TestFormatV6Uri:
    # [CLASS DOC] Tests for the IPv6 bracketing helper.
    @pytest.mark.parametrize("addr", ["10.0.0.1", "localhost", "my-host.internal"])
    def test_it_leaves_non_v6_addresses_alone(self, addr):
        # [FUNCTION DOC] Parametrized over an IPv4 address and two hostnames; each is returned
        # unchanged.
        """Bracketing a v4 address or a hostname would break the url."""
        assert format_v6_uri(addr) == addr

    def test_it_brackets_a_bare_v6_address(self):
        # [FUNCTION DOC] The loopback IPv6 address "::1" becomes "[::1]".
        """A bare v6 address makes ``host:port`` ambiguous."""
        assert format_v6_uri("::1") == "[::1]"

    def test_it_is_idempotent(self):
        # [FUNCTION DOC] An already bracketed address is unchanged, and applying the helper twice
        # to a bare address gives the same result as applying it once.
        """The engine already brackets its host; the rollout process starts from the bare
        one out of the addr allocator, so both must land on the same url."""
        assert format_v6_uri("[::1]") == "[::1]"
        assert format_v6_uri(format_v6_uri("::1")) == "[::1]"

    @pytest.mark.parametrize("addr", [None, ""])
    def test_it_passes_a_missing_address_through(self, addr):
        # [FUNCTION DOC] Parametrized over none and the empty string; each is returned unchanged.
        """The host is optional until the engine resolves its own."""
        assert format_v6_uri(addr) == addr


class TestBuildServerUrl:
    # [CLASS DOC] Tests for the server URL builder (scheme http, host, colon, port).
    def test_it_builds_the_same_url_from_a_bracketed_or_bare_v6_host(self):
        # [FUNCTION DOC] Both "::1" and "[::1]" with port 30000 give "http://[::1]:30000".
        """Two processes derive this url from different sources; they must agree."""
        assert build_server_url("::1", 30000) == "http://[::1]:30000"
        assert build_server_url("[::1]", 30000) == "http://[::1]:30000"

    def test_it_builds_a_plain_v4_url(self):
        # [FUNCTION DOC] Host 10.0.0.1 and port 30000 give "http://10.0.0.1:30000".
        """The common case must stay byte-identical to the old f-string."""
        assert build_server_url("10.0.0.1", 30000) == "http://10.0.0.1:30000"
