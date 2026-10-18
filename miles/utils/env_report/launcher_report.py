# [MODULE DOC] Reads the optional json report that the launcher script writes about its own
# environment. The report path is passed through the arguments; the environment variable name that
# the launcher uses to hand it over is exported as a constant.
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
LAUNCHER_REPORT_ENV_VAR = "MILES_SCRIPT_ENV_REPORT"


def read_launcher_report(path: str) -> dict[str, Any] | None:
    # [FUNCTION DOC] Returns none for an empty path. Otherwise reads and parses the json file and
    # returns it if it is an object; any failure (missing file, bad json) is logged as a warning
    # and gives none, as does a non-object json value.
    if not path:
        return None
    try:
        return _as_report_object(json.loads(Path(path).read_text()))
    except Exception:
        logger.warning("Failed to read the launcher report at %s", path, exc_info=True)
        return None


def _as_report_object(decoded: Any) -> dict[str, Any] | None:
    # [FUNCTION DOC] HELPER: Returns the value if it is a mapping; otherwise logs a warning
    # (including the value) and returns none.
    if isinstance(decoded, dict):
        return decoded
    logger.warning("The launcher report is valid json but not an object, so it names nothing: %r", decoded)
    return None
