# [MODULE DOC]
# Helpers that produce megatron multi-policy config documents (a yaml mapping with a list of trainer
# entries) for tests, either written to a file or encoded as an inline pseudo file string.
from pathlib import Path
from typing import Any

import yaml

from miles.utils.external_utils.command_utils.common import encode_pseudo_file


def write_megatron_config(tmp_path: Path, *model_ids: str, filename: str = "megatron.yaml") -> str:
    # [FUNCTION DOC] Writes a config file in the directory with one trainer entry per model id (each
    # holding only the model id) and returns its path as text. Default file name megatron.yaml.
    return write_megatron_config_trainers(
        tmp_path, [{"model_id": model_id} for model_id in model_ids], filename=filename
    )


def write_megatron_config_trainers(
    tmp_path: Path, trainers: list[dict[str, Any]], *, filename: str = "megatron.yaml"
) -> str:
    # [FUNCTION DOC] Writes a yaml file in the directory (overwriting) whose top-level mapping has one
    # key, trainers, holding the given list of trainer mappings as is, and returns its path as text.
    path = tmp_path / filename
    path.write_text(yaml.dump({"trainers": trainers}))
    return str(path)


def encode_megatron_config(*model_ids: str) -> str:
    # [FUNCTION DOC] Returns the same trainers document as the file writer (one entry per model id)
    # serialized to yaml and encoded as an inline pseudo file string; no file is written.
    return encode_pseudo_file(yaml.dump({"trainers": [{"model_id": model_id} for model_id in model_ids]}))
