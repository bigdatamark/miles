# [MODULE DOC] Run identifier helpers: a run uuid is a fixed-length (16) string of lowercase
# hexadecimal characters. Provides a generator (random, from a version-4 uuid) and a strict
# validator.
import re
import uuid

RUN_UUID_LENGTH = 16

_RUN_UUID_PATTERN = re.compile(rf"[0-9a-f]{{{RUN_UUID_LENGTH}}}")


def generate_run_uuid() -> str:
    # [FUNCTION DOC] Returns a new random run id: the first 16 hex characters of a random
    # version-4 uuid's hex form (lowercase, no dashes). No side effects.
    return uuid.uuid4().hex[:RUN_UUID_LENGTH]


def validate_run_uuid(value: str) -> str:
    # [FUNCTION DOC] Returns the input unchanged if the whole string is exactly 16 lowercase hex
    # characters (no surrounding whitespace, no uppercase); otherwise raises a value error that
    # includes the repr of the offending value.
    if _RUN_UUID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"invalid run uuid {value!r}; expected exactly {RUN_UUID_LENGTH} lowercase hex characters")
    return value
