# [MODULE DOC]
# Tests for the object store configuration module. Covers: size parsing (integers, digit strings,
# case-insensitive and whitespace-tolerant kb/mb/gb and k/m/g suffixes using powers of 1024,
# fractional values, value error on garbage); computing the Mooncake store config from init kwargs
# (sizes parsed to bytes, zero segment for non-contributing processes, MOONCAKE_* environment
# fallbacks when kwargs are unset, kwargs taking precedence over the environment, built-in defaults
# when nothing is set); the launcher's vanilla init kwargs (pointing at the launched master, TCP,
# 2 GB sizes, never reading the environment) and the environment-derived kwargs (only fields the
# environment names, merged over the vanilla defaults). A module-level helper clears all
# MOONCAKE_* variables.
from typing import Any

import pytest

from miles.utils import object_store_config


class TestParseSize:
    # [CLASS DOC] Tests of the size parser.
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (12345, 12345),
            ("12345", 12345),
            ("1kb", 1024),
            ("2k", 2 * 1024),
            ("64mb", 64 * 1024**2),
            ("3m", 3 * 1024**2),
            ("2gb", 2 * 1024**3),
            ("1g", 1024**3),
            ("1.5gb", int(1.5 * 1024**3)),
            ("  2GB ", 2 * 1024**3),
        ],
    )
    def test_parses_ints_and_unit_suffixes(self, value: Any, expected: int):
        # [FUNCTION DOC] Parametrized over integers, digit strings, kb/k, mb/m, gb/g suffixes with
        # base 1024, a fractional gigabyte value (truncated to an integer), and an uppercase value
        # with surrounding spaces; the private parser returns the expected byte count.
        """_parse_size handles ints, plain digit strings, and kb/mb/gb suffixes case-insensitively."""
        assert object_store_config._parse_size(value) == expected

    def test_rejects_garbage(self):
        # [FUNCTION DOC] The text "lots" makes the private parser raise a value error.
        """_parse_size raises ValueError on a non-numeric string without a known unit."""
        with pytest.raises(ValueError):
            object_store_config._parse_size("lots")


class TestMooncakeStoreConfig:
    # [CLASS DOC] Tests of computing the Mooncake store config from init kwargs and the environment.
    def _base_kwargs(self) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Return a complete kwargs mapping: local hostname 10.0.0.1, master
        # address 10.0.0.2:50051, protocol tcp, global segment size 2gb, local buffer size 1gb.
        return {
            "local_hostname": "10.0.0.1",
            "master_server_address": "10.0.0.2:50051",
            "protocol": "tcp",
            "global_segment_size": "2gb",
            "local_buffer_size": "1gb",
        }

    def test_contributing_process_parses_segment_size(self):
        # [FUNCTION DOC] For a contributing process, the config has global segment 2 GiB, local
        # buffer 1 GiB, and the master address, protocol and hostname copied from the kwargs.
        """A contributing process gets the configured global_segment_size parsed to bytes."""
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=True)
        assert config["global_segment_size"] == 2 * 1024**3
        assert config["local_buffer_size"] == 1024**3
        assert config["master_server_addr"] == "10.0.0.2:50051"
        assert config["protocol"] == "tcp"
        assert config["local_hostname"] == "10.0.0.1"

    def test_non_contributing_process_gets_zero_segment(self):
        # [FUNCTION DOC] For a non-contributing process the config's global segment size is zero.
        """A non-contributing process passes global_segment_size=0 (pure client semantics)."""
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=False)
        assert config["global_segment_size"] == 0

    def test_env_fallbacks(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With empty kwargs and the local hostname, master, protocol and global
        # segment size environment variables set, the config takes those values (segment size
        # parsed to 64 MiB).
        """Unset kwargs fall back to MOONCAKE_* environment variables."""
        monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.1.1.1")
        monkeypatch.setenv("MOONCAKE_MASTER", "10.1.1.2:50051")
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "tcp")
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")
        config = object_store_config.compute_mooncake_store_config({}, contribute_segment=True)
        assert config["local_hostname"] == "10.1.1.1"
        assert config["master_server_addr"] == "10.1.1.2:50051"
        assert config["protocol"] == "tcp"
        assert config["global_segment_size"] == 64 * 1024**2

    def test_kwargs_take_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With a master address in both the environment and the kwargs, the config
        # uses the kwargs value.
        """Explicit init kwargs win over MOONCAKE_* environment variables."""
        monkeypatch.setenv("MOONCAKE_MASTER", "10.9.9.9:50051")
        config = object_store_config.compute_mooncake_store_config(self._base_kwargs(), contribute_segment=True)
        assert config["master_server_addr"] == "10.0.0.2:50051"

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the environment cleared and the local hostname lookup patched to
        # 127.0.0.1, empty kwargs give protocol rdma, metadata server P2PHANDSHAKE, global segment
        # 8 GiB, local buffer 32 GiB and an empty master address.
        """With no kwargs and no env, protocol/metadata/segment sizes use built-in defaults."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setattr(object_store_config, "_local_hostname", lambda: "127.0.0.1")
        config = object_store_config.compute_mooncake_store_config({}, contribute_segment=True)
        assert config["protocol"] == "rdma"
        assert config["metadata_server"] == "P2PHANDSHAKE"
        assert config["global_segment_size"] == 8 * 1024**3
        assert config["local_buffer_size"] == 32 * 1024**3
        assert config["master_server_addr"] == ""


class TestMooncakeInitKwargs:
    # [CLASS DOC] Tests of the launcher's vanilla init kwargs and the environment-derived kwargs.
    def test_defaults_name_the_launched_master(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Even with a master set in the environment, the vanilla kwargs for host
        # 10.0.0.2 and master port 1234 name address 10.0.0.2:1234 with the tcp protocol.
        """The launcher's defaults point at the master it starts and never read the environment."""
        monkeypatch.setenv("MOONCAKE_MASTER", "10.9.9.9:50051")
        init_kwargs = object_store_config.compute_mooncake_init_kwargs_vanilla(host="10.0.0.2", master_port=1234)
        assert init_kwargs["master_server_address"] == "10.0.0.2:1234"
        assert init_kwargs["protocol"] == "tcp"

    def test_env_answers_only_defaulted_fields(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With master, protocol and local hostname set in the cleared environment,
        # the environment-derived kwargs contain only the master address and the protocol (not the
        # hostname).
        """The environment layer covers every defaulted field and nothing the launcher does not default."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_MASTER", "10.1.1.2:50051")
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")
        monkeypatch.setenv("MOONCAKE_LOCAL_HOSTNAME", "10.1.1.1")
        from_env = object_store_config.compute_mooncake_init_kwargs_from_env()
        assert from_env == {"master_server_address": "10.1.1.2:50051", "protocol": "rdma"}

    def test_env_overrides_reach_the_store_config(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] Merge vanilla kwargs with environment kwargs (master and protocol set in
        # the environment) and compute the store config; its master address and protocol are the
        # environment values.
        """A defaulted field the environment names wins over the built-in default at store setup."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_MASTER", "10.1.1.2:50051")
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")
        init_kwargs = (
            object_store_config.compute_mooncake_init_kwargs_vanilla()
            | object_store_config.compute_mooncake_init_kwargs_from_env()
        )
        config = object_store_config.compute_mooncake_store_config(init_kwargs, contribute_segment=True)
        assert config["master_server_addr"] == "10.1.1.2:50051"
        assert config["protocol"] == "rdma"


class TestTheEnvironmentAnswersEveryDefaultedField:
    # [CLASS DOC] Tests that every field the launcher defaults can also be set from the environment.
    def test_the_capacities_the_platform_names_reach_the_init_kwargs(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With global segment size 64mb and local buffer size 16mb in the cleared
        # environment, the environment kwargs are exactly those two entries as raw strings.
        """A pod sized by its platform would otherwise be given the launcher's 2 GiB of each capacity."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")
        monkeypatch.setenv("MOONCAKE_LOCAL_BUFFER_SIZE", "16mb")

        from_env = object_store_config.compute_mooncake_init_kwargs_from_env()

        assert from_env == {"global_segment_size": "64mb", "local_buffer_size": "16mb"}

    def test_the_capacities_the_platform_names_survive_into_the_store_config(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With only the global segment size set (64mb) in the environment, merging
        # over the vanilla kwargs and computing the store config gives a 64 MiB segment while the
        # local buffer keeps the 2 GiB launcher default.
        """The kwargs are only worth answering because this is what the store is finally set up with."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", "64mb")

        init_kwargs = (
            object_store_config.compute_mooncake_init_kwargs_vanilla()
            | object_store_config.compute_mooncake_init_kwargs_from_env()
        )
        config = object_store_config.compute_mooncake_store_config(init_kwargs, contribute_segment=True)

        assert config["global_segment_size"] == 64 * 1024**2
        assert config["local_buffer_size"] == 2 * 1024**3

    def test_a_defaulted_field_the_environment_leaves_alone_keeps_the_launcher_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # [FUNCTION DOC] With only the protocol set (rdma) in the environment, the merged kwargs for
        # host 10.0.0.2 keep the launcher's master address 10.0.0.2:50051 (default port) and take
        # the rdma protocol.
        """The launcher still starts the master it configures, and a run naming nothing must still find it."""
        _clear_mooncake_env(monkeypatch)
        monkeypatch.setenv("MOONCAKE_PROTOCOL", "rdma")

        init_kwargs = (
            object_store_config.compute_mooncake_init_kwargs_vanilla(host="10.0.0.2")
            | object_store_config.compute_mooncake_init_kwargs_from_env()
        )

        assert init_kwargs["master_server_address"] == "10.0.0.2:50051"
        assert init_kwargs["protocol"] == "rdma"

    def test_an_environment_that_names_nothing_changes_nothing(self, monkeypatch: pytest.MonkeyPatch):
        # [FUNCTION DOC] With the environment cleared, the environment kwargs are an empty mapping.
        """Every kubernetes run that names no store took this path before the environment was read at all."""
        _clear_mooncake_env(monkeypatch)

        assert object_store_config.compute_mooncake_init_kwargs_from_env() == {}


def _clear_mooncake_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # [FUNCTION DOC] HELPER: Remove from the environment (for the test only, ignoring missing ones)
    # the seven MOONCAKE_* variables: local hostname, metadata server, local buffer size, protocol,
    # device, master and global segment size.
    for name in (
        "MOONCAKE_LOCAL_HOSTNAME",
        "MOONCAKE_TE_META_DATA_SERVER",
        "MOONCAKE_LOCAL_BUFFER_SIZE",
        "MOONCAKE_PROTOCOL",
        "MOONCAKE_DEVICE",
        "MOONCAKE_MASTER",
        "MOONCAKE_GLOBAL_SEGMENT_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


class TestComputeMooncakeInitKwargs:
    # [CLASS DOC] Tests of the vanilla init kwargs with a custom endpoint.
    def test_custom_host_and_port_produce_the_complete_mooncake_configuration(self) -> None:
        # [FUNCTION DOC] For host "store.internal" and port 60000, the vanilla kwargs equal exactly:
        # protocol tcp, master address "store.internal:60000", global segment size "2gb" and local
        # buffer size "2gb".
        """A custom endpoint produces every required Mooncake initialization setting."""
        assert object_store_config.compute_mooncake_init_kwargs_vanilla(host="store.internal", master_port=60000) == {
            "protocol": "tcp",
            "master_server_address": "store.internal:60000",
            "global_segment_size": "2gb",
            "local_buffer_size": "2gb",
        }
