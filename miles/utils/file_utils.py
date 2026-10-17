# [MODULE DOC]
# HELPER: Small file system utilities. Currently only an atomic text writer.
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, text: str) -> None:
    # [FUNCTION DOC] HELPER: Writes text to the path atomically. Creates a temporary file in the
    # same directory (the parent must exist; name is the target name plus random part and a temp
    # suffix), writes the text, sets permissions to owner read/write and group/other read (the
    # temp file starts owner-only), then renames it over the target (atomic on the same file
    # system). On any failure (including interrupts) removes the temp file and re-raises.
    path = Path(path)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as file:
            file.write(text)
            # mkstemp opens 0600, which leaves a file one uid wrote unreadable to the next one
            os.fchmod(file.fileno(), 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
