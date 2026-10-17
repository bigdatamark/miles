# [MODULE DOC] Detection of Inkling checkpoints and access to the bundled fixed Inkling chat
# template. Detection reads the model type from the checkpoint directory's config file and
# compares it with the known Inkling model types. KNOWN ISSUE: the tuple of known model types
# lists the same string twice, so the second entry is redundant (probably a missing second name).

from __future__ import annotations

import functools
import json
import os
from typing import Any

_MODEL_TYPES = ("inkling_mm_model", "inkling_mm_model")


@functools.cache
def _read_model_type(name_or_path: str) -> str:
    # [FUNCTION DOC] HELPER: Cached per path. Returns the model type string from the config file
    # (config.json) inside the given directory, or an empty string when the path is empty, the
    # file is missing, unreadable, not valid text or JSON, not a JSON object, or has no (or an
    # empty/null) model type. Reads the filesystem on first call per path; later changes to the
    # file are not seen. Duplicate of the helper of the same name in the deepseek module.
    if not name_or_path:
        return ""
    config_path = os.path.join(name_or_path, "config.json")
    if not os.path.isfile(config_path):
        return ""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return ""
    if not isinstance(config, dict):
        return ""
    return config.get("model_type", "") or ""


def is_inkling(tokenizer: Any) -> bool:
    # [FUNCTION DOC] True when the tokenizer's name-or-path attribute (empty if absent) points to
    # a checkpoint whose model type is a known Inkling type.
    return _read_model_type(getattr(tokenizer, "name_or_path", "")) in _MODEL_TYPES


def is_inkling_checkpoint(name_or_path: str) -> bool:
    # [FUNCTION DOC] True when the checkpoint directory's model type is a known Inkling type.
    return _read_model_type(name_or_path) in _MODEL_TYPES


@functools.cache
def fixed_chat_template() -> str:
    # [FUNCTION DOC] Cached. Returns the text of the file inkling_fixed.jinja from the template
    # directory defined in the tito tokenizer module (read as UTF-8). The import of that module
    # is deferred to call time to avoid a circular import (it imports the template module, which
    # imports this one). Raises an OS error if the file is missing.
    """The bundled Inkling chat template (the checkpoint's, with the upstream fixes)."""
    # imported lazily: tito_tokenizer imports template.py, which imports this module
    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR

    return (TEMPLATE_DIR / "inkling_fixed.jinja").read_text(encoding="utf-8")
