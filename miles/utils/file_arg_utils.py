# [MODULE DOC]
# Command line helper for arguments that are either a file path or an inline payload prefixed with
# the "base64:" marker (a pseudo file).
import base64
from pathlib import Path

PSEUDO_FILE_PREFIX = "base64:"


def resolve_file_arg(value: str) -> str:
    # [FUNCTION DOC] Returns text content for the argument. If it starts with the marker prefix,
    # decodes the rest as strictly validated base64 and decodes the bytes as text (default
    # encoding), raising a decoding error for invalid base64. Otherwise reads the file at that path
    # as utf-8 text (raising a file error if missing).
    """Read a command line argument that is either a file path or an inline `base64:` payload."""
    if value.startswith(PSEUDO_FILE_PREFIX):
        return base64.b64decode(value[len(PSEUDO_FILE_PREFIX) :], validate=True).decode()
    return Path(value).read_text(encoding="utf-8")
