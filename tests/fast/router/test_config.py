# [MODULE DOC] Tests for the miles router configuration: the function that derives a router config
# from command-line args plus host, port and engine count (explicit vs derived maximum connections,
# one-to-one copying of the remaining settings), and the validated config model (every field is
# required). Pure unit tests; no network.
from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import ValidationError

from miles.router.config import MilesRouterConfig, compute_miles_router_config


def _make_args(**overrides) -> Namespace:
    # [FUNCTION DOC] HELPER: Builds a namespace with defaults: no max connections override, no
    # timeout, health check interval 10.0, failure threshold 3 and server concurrency 64. Keyword
    # overrides replace them.
    defaults = dict(
        miles_router_max_connections=None,
        miles_router_timeout=None,
        rollout_health_check_interval=10.0,
        miles_router_health_check_failure_threshold=3,
        sglang_server_concurrency=64,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class TestComputeMilesRouterConfig:
    # [CLASS DOC] Tests of deriving a router config from args, host, port and engine count.
    def test_explicit_max_connections_wins(self):
        # [FUNCTION DOC] An explicit max connections arg (42) is used as is, regardless of the
        # engine count.
        """--miles-router-max-connections overrides the derived value."""
        config = compute_miles_router_config(
            _make_args(miles_router_max_connections=42), host="10.0.0.1", port=1234, num_engines=4
        )
        assert config.max_connections == 42

    def test_max_connections_derived_from_engine_capacity(self):
        # [FUNCTION DOC] Without an override, max connections is the per-server concurrency (64)
        # times the number of engines (4), here 256.
        """Without an override, capacity is concurrency * the engines this router fronts,
        which is not the rollout fleet's size when the router fronts the eval fleet."""
        config = compute_miles_router_config(_make_args(), host="10.0.0.1", port=1234, num_engines=4)
        assert config.max_connections == 64 * 4

    def test_remaining_fields_are_copied_from_args(self):
        # [FUNCTION DOC] Host, port, timeout (30.0), health check interval (5.0) and the default
        # failure threshold (3) appear unchanged on the config.
        """Host, port, timeout, and health check settings map one-to-one."""
        config = compute_miles_router_config(
            _make_args(miles_router_timeout=30.0, rollout_health_check_interval=5.0),
            num_engines=4,
            host="10.0.0.1",
            port=1234,
        )
        assert config.host == "10.0.0.1"
        assert config.port == 1234
        assert config.timeout == 30.0
        assert config.health_check_interval == 5.0
        assert config.health_check_failure_threshold == 3

    def test_nondefault_health_check_failure_threshold_is_copied(self):
        # [FUNCTION DOC] A failure threshold arg of 7 appears as 7 on the config.
        """A failure threshold other than the default is forwarded, not hardcoded."""
        config = compute_miles_router_config(
            _make_args(miles_router_health_check_failure_threshold=7),
            host="10.0.0.1",
            port=1234,
            num_engines=4,
        )
        assert config.health_check_failure_threshold == 7


_COMPLETE_CONFIG_KWARGS = dict(
    host="127.0.0.1",
    port=1234,
    max_connections=10,
    timeout=5.0,
    health_check_interval=1.0,
    health_check_failure_threshold=3,
)


class TestMilesRouterConfig:
    # [CLASS DOC] Tests of the validated router config model, using a module-level mapping of
    # complete field values.
    def test_complete_kwargs_cover_every_field(self):
        # [FUNCTION DOC] The complete keyword mapping names exactly the model's fields, and
        # constructing the model from it succeeds (host is read back as 127.0.0.1).
        """The per-field omission cases are only meaningful if the full kwargs validate."""
        assert set(_COMPLETE_CONFIG_KWARGS) == set(MilesRouterConfig.model_fields)
        assert MilesRouterConfig(**_COMPLETE_CONFIG_KWARGS).host == "127.0.0.1"

    @pytest.mark.parametrize("missing", sorted(MilesRouterConfig.model_fields))
    def test_every_field_is_required(self, missing: str):
        # [FUNCTION DOC] Parametrized over every model field name (sorted): constructing the model
        # from the complete mapping minus that field raises a validation error.
        """Omitting any single field must fail, so none can be silently defaulted."""
        kwargs = {name: value for name, value in _COMPLETE_CONFIG_KWARGS.items() if name != missing}
        with pytest.raises(ValidationError):
            MilesRouterConfig(**kwargs)
