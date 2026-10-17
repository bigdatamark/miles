# [MODULE DOC] Produces the prompt token ids fed to the standalone Megatron worker: builds raw text
# (deterministic arithmetic text, file contents or user text), optionally wraps it in a chat
# template, tokenizes it with the checkpoint's tokenizer, truncates to the sequence length, and
# writes the ids to a temporary json file for the worker to read.
from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Literal


@dataclasses.dataclass(frozen=True)
class PromptConfig:
    # [CLASS DOC] Immutable prompt options: mode (math, file or text; default math), text (used in
    # text mode), file (used in file mode), seq_length (number of tokens wanted, default 137) and
    # apply_chat_template (wrap the text as a single user message, default off).
    mode: Literal["math", "file", "text"] = "math"
    text: str | None = None
    file: Path | None = None
    seq_length: int = 137
    apply_chat_template: bool = False


def generate_token_ids(
    *,
    prompt: PromptConfig,
    tokenizer_path: Path,
) -> list[int]:
    # [FUNCTION DOC] Keyword-only inputs: prompt options and tokenizer location. Resolves the raw
    # text per mode (may raise a value error for a missing file/text or unknown mode), loads the
    # tokenizer with remote code trusted (imports the tokenizer loader and transformers lazily),
    # optionally renders the text as one user message through the chat template without
    # tokenizing, then encodes with the tokenizer's default settings (special tokens added by
    # default). Raises a value error if fewer tokens than the sequence length result; otherwise
    # returns the first seq_length token ids.
    # KNOWN ISSUE: the docstring says math mode is "padded/truncated", but no padding happens;
    # short results raise an error and longer ones are truncated.
    """Generate token IDs for Megatron standalone forward/backward.

    Three modes:
    - math: deterministic arithmetic sequence "1+1=2, 1+2=3, ..." tokenized and padded/truncated to seq_length
    - file: read long text from file, tokenize and truncate to seq_length
    - text: use user-provided text directly, tokenize and truncate to seq_length
    """
    from transformers import PreTrainedTokenizerBase

    from miles.utils.processing_utils import load_tokenizer

    raw_text: str = _resolve_raw_text(prompt)

    tokenizer: PreTrainedTokenizerBase = load_tokenizer(str(tokenizer_path), trust_remote_code=True)

    if prompt.apply_chat_template:
        messages: list[dict[str, str]] = [{"role": "user", "content": raw_text}]
        raw_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )

    token_ids: list[int] = tokenizer.encode(raw_text)
    if len(token_ids) < prompt.seq_length:
        raise ValueError(
            f"Token count ({len(token_ids)}) is less than seq_length ({prompt.seq_length}). "
            f"Provide longer input text or reduce --seq-length."
        )
    token_ids = token_ids[: prompt.seq_length]

    return token_ids


def write_token_ids_to_tmpfile(token_ids: list[int]) -> Path:
    # [FUNCTION DOC] Writes the ids as a json array to a new temporary file (name prefix
    # run_megatron_token_ids_, suffix .json) in the system temp directory, which is not deleted
    # automatically (side effect: leaves a file). Returns its path.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, prefix="run_megatron_token_ids_") as tmp:
        json.dump(token_ids, tmp)
    return Path(tmp.name)


def _resolve_raw_text(prompt: PromptConfig) -> str:
    # [FUNCTION DOC] HELPER: Math mode: the arithmetic text sized to sixteen characters per wanted
    # token. File mode: the file's full text (value error if no file given). Text mode: the given
    # text (value error if none). Any other mode raises a value error.
    if prompt.mode == "math":
        return _build_math_sequence(target_char_length=prompt.seq_length * 16)
    elif prompt.mode == "file":
        if prompt.file is None:
            raise ValueError("--prompt-file is required for file mode")
        return prompt.file.read_text()
    elif prompt.mode == "text":
        if prompt.text is None:
            raise ValueError("--prompt-text is required for text mode")
        return prompt.text
    else:
        raise ValueError(f"Unknown prompt mode: {prompt.mode!r}")


def _build_math_sequence(target_char_length: int) -> str:
    # [FUNCTION DOC] HELPER: Appends segments of the form "a+b=sum, " starting at a=1, b=1; b
    # increases by one each time and, after passing 100, wraps to 1 while a increases by one.
    # Stops once the total length reaches the target (the last segment may overshoot) and returns
    # the concatenation.
    """Build "1+1=2, 1+2=3, ..." until reaching target_char_length characters."""
    parts: list[str] = []
    total_len: int = 0
    a: int = 1
    b: int = 1

    while total_len < target_char_length:
        segment: str = f"{a}+{b}={a + b}, "
        parts.append(segment)
        total_len += len(segment)
        b += 1
        if b > 100:
            a += 1
            b = 1

    return "".join(parts)
