# [MODULE DOC] Removes secrets from data that goes into environment reports by replacing each
# secret string with "redacted-sha256:" plus the first 16 hex characters of its SHA-256 (so equal
# secrets stay comparable without being exposed). Covers: environment variables whose name ends
# in key, token, secret, password, passwd, credential(s) or database url (case-insensitive, at
# the end of the name or after an underscore); named command-line / argument secrets (wandb key,
# router keys and passwords, sglang api keys and ssl key file password for the training and eval
# engines); the train env-vars argument (a mapping or json, redacted per variable); the userinfo
# part of proxy urls; and sglang server info. A module-level table of flag redactors is built at
# import time.
import hashlib
import json
import re
from collections.abc import Callable
from typing import Any

_SECRET_ENV_VAR_PATTERN = re.compile(
    r"(^|_)(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|DATABASE_URL)$", re.IGNORECASE
)
_SGLANG_SECRET_ARG_BASE_NAMES = frozenset({"api_key", "admin_api_key", "ssl_keyfile_password"})
_SECRET_ARG_NAMES = frozenset(
    {
        "wandb_key",
        "router_api_key",
        "router_oracle_password",
        "router_control_plane_api_keys",
        *(f"{prefix}{name}" for prefix in ("sglang_", "eval_sglang_") for name in _SGLANG_SECRET_ARG_BASE_NAMES),
    }
)
_ENV_VAR_ARG_NAMES = frozenset({"train_env_vars"})
_URL_ARG_NAMES = frozenset({"http_proxy"})
_REDACTED_PREFIX = "redacted-sha256:"
_REDACTED_HASH_CHARS = 16


def redact_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    # [FUNCTION DOC] Returns a new mapping sorted by variable name in which the values of
    # variables whose names match the secret pattern are replaced by their redaction string; other
    # values are kept.
    return {
        name: _redact(value) if _SECRET_ENV_VAR_PATTERN.search(name) else value
        for name, value in sorted(env_vars.items())
    }


def redact_argv(argv: list[str]) -> list[str]:
    # [FUNCTION DOC] Returns a copy of the command-line list with secret values redacted. For
    # each item: if a previous flag was waiting for its value and this item does not start with a
    # dash, it is redacted with that flag's redactor. Otherwise the item is split at the first
    # equals sign; if the flag part is a known secret-bearing flag, the inline value (after the
    # equals) is redacted, or, when there is no equals sign, the flag is kept and the next
    # non-dash item will be redacted; unknown flags are copied as is. A value starting with a dash
    # is therefore never redacted and cancels pending redaction. A known flag with an equals sign
    # and an empty value is rewritten with a redacted empty string.
    redacted: list[str] = []
    redact_value: Callable[[str], str] | None = None
    for item in argv:
        if redact_value is not None and not item.startswith("-"):
            redacted.append(redact_value(item))
            continue

        redact_value = None
        flag, separator, value = item.partition("=")
        if (rewrite := _ARGV_VALUE_REDACTORS.get(flag)) is None:
            redacted.append(item)
            continue

        redacted.append(f"{flag}={rewrite(value)}" if separator else item)
        redact_value = None if separator else rewrite

    return redacted


def redact_server_info(server_info: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] Redacts the sglang secret fields (api key, admin api key, ssl key file
    # password) at the top level, and, if the internal states entry is a list, applies the
    # per-state redaction to every element. Returns a new top-level mapping (shallow copy); the
    # input is not modified.
    redacted = _redact_server_args(server_info)

    if isinstance(internal_states := redacted.get("internal_states"), list):
        redacted["internal_states"] = [_redact_internal_state(state) for state in internal_states]

    return redacted


def redact_arg(name: str, value: Any) -> Any:
    # [FUNCTION DOC] Redacts one parsed argument by name: the train env-vars argument, if a
    # mapping, goes through environment-variable redaction (values converted to text); the proxy
    # argument, if text, has url userinfo redacted; arguments in the secret-name set are redacted
    # (text, or each text element of a list); everything else is returned unchanged.
    if name in _ENV_VAR_ARG_NAMES and isinstance(value, dict):
        return redact_env_vars({key: str(item) for key, item in value.items()})
    if name in _URL_ARG_NAMES and isinstance(value, str):
        return _redact_url_userinfo(value)
    if name not in _SECRET_ARG_NAMES:
        return value
    return _redact_secret_value(value)


def _redact(value: str) -> str:
    # [FUNCTION DOC] HELPER: The prefix "redacted-sha256:" followed by the first 16 hex characters
    # of the SHA-256 of the utf-8 text.
    digest = hashlib.sha256(value.encode()).hexdigest()[:_REDACTED_HASH_CHARS]
    return f"{_REDACTED_PREFIX}{digest}"


def _argv_value_redactors() -> dict[str, Callable[[str], str]]:
    # [FUNCTION DOC] HELPER: Builds the table from command-line flag (dashes instead of
    # underscores, double-dash prefix) to the redaction function for its value: plain redaction
    # for secret arguments, json-aware for env-var arguments, userinfo-only for url arguments.
    def flag(name: str) -> str:
        # [FUNCTION DOC] HELPER: Converts an argument name to its flag spelling.
        return f"--{name.replace('_', '-')}"

    return {
        **{flag(name): _redact for name in _SECRET_ARG_NAMES},
        **{flag(name): _redact_env_var_json for name in _ENV_VAR_ARG_NAMES},
        **{flag(name): _redact_url_userinfo for name in _URL_ARG_NAMES},
    }


def _redact_url_userinfo(value: str) -> str:
    # [FUNCTION DOC] HELPER: If the text has a scheme separator and the part after it has an at
    # sign, replaces everything before the last at sign (user and password) with its redaction and
    # keeps scheme and host; otherwise returns the text unchanged.
    scheme, separator, rest = value.partition("://")
    if not separator:
        return value

    userinfo, at_sign, host = rest.rpartition("@")
    if not at_sign:
        return value
    return f"{scheme}://{_redact(userinfo)}@{host}"


def _redact_env_var_json(value: str) -> str:
    # [FUNCTION DOC] HELPER: Parses the text as json; if it is not valid json or not an object,
    # returns the redaction of the whole text. Otherwise returns json text of the object with
    # environment-variable redaction applied (values converted to text).
    try:
        env_vars = json.loads(value)
    except json.JSONDecodeError:
        return _redact(value)
    if not isinstance(env_vars, dict):
        return _redact(value)
    return json.dumps(redact_env_vars({name: str(item) for name, item in env_vars.items()}))


def _redact_internal_state(state: Any) -> Any:
    # [FUNCTION DOC] HELPER: Non-mapping states are returned as is. Otherwise redacts the sglang
    # secret fields and, if the state has an env-vars mapping, applies environment-variable
    # redaction to it.
    if not isinstance(state, dict):
        return state

    redacted = _redact_server_args(state)
    if isinstance(env_vars := redacted.get("env_vars"), dict):
        redacted["env_vars"] = redact_env_vars({name: str(value) for name, value in env_vars.items()})
    return redacted


def _redact_server_args(values: dict[str, Any]) -> dict[str, Any]:
    # [FUNCTION DOC] HELPER: New mapping where entries named api_key, admin_api_key or
    # ssl_keyfile_password get their secret value redacted.
    return {
        name: _redact_secret_value(value) if name in _SGLANG_SECRET_ARG_BASE_NAMES else value
        for name, value in values.items()
    }


def _redact_secret_value(value: Any) -> Any:
    # [FUNCTION DOC] HELPER: Text is redacted; a list has its text elements redacted (others
    # kept); any other type (including none) is returned unchanged.
    if isinstance(value, str):
        return _redact(value)
    if isinstance(value, list):
        return [_redact(item) if isinstance(item, str) else item for item in value]
    return value


_ARGV_VALUE_REDACTORS = _argv_value_redactors()
