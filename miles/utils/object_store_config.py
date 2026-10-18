# [MODULE DOC]
# Configuration of the mooncake store client. A table of fields describes, per setting, the name in
# user init keyword arguments, the key in the store's configuration, the environment variable
# fallback, the parser and the default. Resolution order per field: init keyword argument, then
# environment variable, then default (empty values count as unset except for numeric size fields).
# Sizes accept integers or text with k/m/g or kb/mb/gb suffixes (powers of 1024).
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

MOONCAKE_MASTER_PORT = 50051
MOONCAKE_MASTER_ADDRESS_KEY = "master_server_address"

_GLOBAL_SEGMENT_SIZE_KEY = "global_segment_size"
_MISSING = object()


def compute_mooncake_init_kwargs_vanilla(*, host: str = "127.0.0.1", master_port: int = MOONCAKE_MASTER_PORT) -> dict:
    # [FUNCTION DOC] Default init keyword arguments for a local master: tcp protocol, master address
    # host:port (default loopback and port 50051), 2gb global segment and 2gb local buffer.
    return {
        "protocol": "tcp",
        MOONCAKE_MASTER_ADDRESS_KEY: f"{host}:{master_port}",
        "global_segment_size": "2gb",
        "local_buffer_size": "2gb",
    }


def compute_mooncake_init_kwargs_from_env() -> dict:
    # [FUNCTION DOC] Builds init keyword arguments from environment variables, limited to the settings
    # that have a vanilla default (protocol, master address, global segment size, local buffer
    # size); each is included only if its environment variable is set (even if empty). Values stay
    # as raw text.
    defaulted_init_kwargs = compute_mooncake_init_kwargs_vanilla()
    return {
        field.init_kwarg: value
        for field in _MOONCAKE_STORE_FIELDS
        if field.init_kwarg in defaulted_init_kwargs and (value := os.environ.get(field.env_var)) is not None
    }


def compute_mooncake_store_config(init_kwargs: dict[str, Any], *, contribute_segment: bool) -> dict[str, Any]:
    # [FUNCTION DOC] Builds the store configuration mapping (keys are the store's config keys). Each
    # field is resolved and parsed, except the global segment size, which is forced to zero when this
    # process does not contribute a segment.
    config: dict[str, Any] = {}
    for field in _MOONCAKE_STORE_FIELDS:
        if field.config_key == _GLOBAL_SEGMENT_SIZE_KEY and not contribute_segment:
            config[field.config_key] = 0
        else:
            config[field.config_key] = field.parse(_resolve_field(field, init_kwargs=init_kwargs))
    return config


@dataclass(frozen=True)
class _MooncakeStoreField:
    # [CLASS DOC] HELPER: Description of one setting: init keyword name, store config key,
    # environment variable name, parser function, default value or default factory (factory wins),
    # and whether empty values are treated as unset (default true).
    init_kwarg: str
    config_key: str
    env_var: str
    parse: Callable[[Any], Any]
    default: Any = None
    default_factory: Callable[[], Any] | None = None
    treats_falsy_as_unset: bool = True


def _resolve_field(field: _MooncakeStoreField, *, init_kwargs: dict[str, Any]) -> Any:
    # [FUNCTION DOC] HELPER: First usable value among the init keyword argument then the environment
    # variable (a value is usable if present and truthy, or present when empty counts as set); else
    # the default factory result if any; else the static default.
    for value in (init_kwargs.get(field.init_kwarg, _MISSING), os.environ.get(field.env_var, _MISSING)):
        if value is not _MISSING and (value or not field.treats_falsy_as_unset):
            return value
    if (default_factory := field.default_factory) is not None:
        return default_factory()
    return field.default


def _parse_size(value: Any) -> int:
    # [FUNCTION DOC] HELPER: Integers pass through. Other values are converted to text, stripped and
    # lower-cased; a recognized suffix (kb, mb, gb, k, m, g, checked in that order) multiplies the
    # numeric part (a decimal is allowed) by the power of 1024 and truncates to an integer;
    # otherwise the text must be an integer. Invalid text raises a value error.
    # KNOWN ISSUE: a boolean is an integer in this language and passes through unchanged.
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    units = {"kb": 1024, "mb": 1024**2, "gb": 1024**3, "k": 1024, "m": 1024**2, "g": 1024**3}
    for suffix, multiplier in units.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * multiplier)
    return int(text)


def _local_hostname() -> str:
    # [FUNCTION DOC] HELPER: This node's ip as reported by ray (lazy import), used as the default
    # local hostname.
    import ray

    return ray.util.get_node_ip_address()


_MOONCAKE_STORE_FIELDS: tuple[_MooncakeStoreField, ...] = (
    _MooncakeStoreField(
        init_kwarg="local_hostname",
        config_key="local_hostname",
        env_var="MOONCAKE_LOCAL_HOSTNAME",
        parse=str,
        default_factory=lambda: _local_hostname(),
    ),
    _MooncakeStoreField(
        init_kwarg="metadata_server",
        config_key="metadata_server",
        env_var="MOONCAKE_TE_META_DATA_SERVER",
        parse=str,
        default="P2PHANDSHAKE",
    ),
    _MooncakeStoreField(
        init_kwarg="local_buffer_size",
        config_key="local_buffer_size",
        env_var="MOONCAKE_LOCAL_BUFFER_SIZE",
        parse=_parse_size,
        default=32 * 1024**3,
        treats_falsy_as_unset=False,
    ),
    _MooncakeStoreField(
        init_kwarg="protocol",
        config_key="protocol",
        env_var="MOONCAKE_PROTOCOL",
        parse=str,
        default="rdma",
    ),
    _MooncakeStoreField(
        init_kwarg="device_name",
        config_key="rdma_devices",
        env_var="MOONCAKE_DEVICE",
        parse=str,
        default="",
    ),
    _MooncakeStoreField(
        init_kwarg=MOONCAKE_MASTER_ADDRESS_KEY,
        config_key="master_server_addr",
        env_var="MOONCAKE_MASTER",
        parse=str,
        default="",
    ),
    _MooncakeStoreField(
        init_kwarg=_GLOBAL_SEGMENT_SIZE_KEY,
        config_key=_GLOBAL_SEGMENT_SIZE_KEY,
        env_var="MOONCAKE_GLOBAL_SEGMENT_SIZE",
        parse=_parse_size,
        default=8 * 1024**3,
        treats_falsy_as_unset=False,
    ),
)
