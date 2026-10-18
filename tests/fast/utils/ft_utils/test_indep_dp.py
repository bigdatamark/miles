# [MODULE DOC] Tests for the independent data parallel TCP store creator: with the store class
# mocked, it binds a non-blocking master store on the wildcard address with an OS chosen port and
# advertises the node's routable address plus the actual port; with a real store, a client
# connecting to the advertised address can read a key set on the master. Uses real sockets in the
# second test and requires torch.
from unittest.mock import MagicMock

import pytest
import torch.distributed
from torch.distributed import TCPStore

from miles.utils import misc
from miles.utils.ft_utils.indep_dp import create_tcp_store

FAKE_NODE_IP = "203.0.113.7"
FAKE_STORE_PORT = 4567


class TestCreateTcpStore:
    # [CLASS DOC] Tests of the TCP store creation function, which returns the store and its
    # host-and-port address text.
    def test_create_tcp_store_binds_a_nonblocking_master_and_advertises_its_address(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # [FUNCTION DOC] The store class is mocked to report port 4567 and the current node ip is
        # faked. Asserts the class was constructed once with host 0.0.0.0, port 0, master mode and
        # no waiting for workers; the returned store is the mock instance; and the address is the
        # fake ip, a colon and the port.
        """The store must use a wildcard ephemeral bind and publish its routable address."""
        store_class = MagicMock()
        store_class.return_value.port = FAKE_STORE_PORT
        monkeypatch.setattr(torch.distributed, "TCPStore", store_class)
        monkeypatch.setattr(misc, "get_current_node_ip", lambda: FAKE_NODE_IP)

        store, addr = create_tcp_store()

        store_class.assert_called_once_with(
            host_name="0.0.0.0",
            port=0,
            is_master=True,
            wait_for_workers=False,
        )
        assert store is store_class.return_value
        assert addr == f"{FAKE_NODE_IP}:{FAKE_STORE_PORT}"

    def test_create_tcp_store_returns_a_master_serving_clients_on_the_advertised_port(self) -> None:
        # [FUNCTION DOC] Creates a real store, connects a non-master client to the advertised host
        # and port, sets a key on the master, and asserts the client reads the value as bytes.
        """A remote worker dialing the advertised port must reach the master store without it waiting for workers."""
        store, addr = create_tcp_store()

        host, port = addr.rsplit(":", 1)
        client = TCPStore(host_name=host, port=int(port), is_master=False)
        store.set("indep_dp_key", "indep_dp_value")

        assert client.get("indep_dp_key") == b"indep_dp_value"
