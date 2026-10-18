# [MODULE DOC] HELPER: Shared builder for the run_megatron tests (a plain function, not a fixture,
# although it lives in the pytest configuration module and is imported by tests).
from __future__ import annotations

from pathlib import Path

from miles.utils.debug_utils.run_megatron.worker.script_args import WorkerScriptArgs


def make_script_args(**overrides: object) -> WorkerScriptArgs:
    # [FUNCTION DOC] HELPER: Build worker script arguments with defaults of Hugging Face checkpoint
    # "/fake/hf" and token ids file "/tmp/tokens.json", updated by the keyword overrides (unknown
    # names fail inside the dataclass constructor). Returns the arguments object; touches no files.
    defaults = dict(
        hf_checkpoint=Path("/fake/hf"),
        token_ids_file=Path("/tmp/tokens.json"),
    )
    defaults.update(overrides)
    return WorkerScriptArgs(**defaults)  # type: ignore[arg-type]
