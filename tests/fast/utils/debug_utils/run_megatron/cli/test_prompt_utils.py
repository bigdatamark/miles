# [MODULE DOC] Tests for prompt/token id helpers of the run_megatron tool
# (miles.utils.debug_utils.run_megatron.cli.prompt_utils): the immutable PromptConfig (mode math,
# text or file; text; file; sequence length default 137; chat template flag), raw text resolution
# per mode, the generated "math" sequence, writing token ids to a temp JSON file, and token id
# generation with the real Qwen3-0.6B tokenizer (downloaded; needs network or cache).
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from miles.utils.debug_utils.run_megatron.cli.prompt_utils import (
    PromptConfig,
    _build_math_sequence,
    _resolve_raw_text,
    generate_token_ids,
    write_token_ids_to_tmpfile,
)

QWEN3_MODEL_ID = "Qwen/Qwen3-0.6B"


class TestPromptConfig:
    # [CLASS DOC] Tests of the prompt configuration dataclass.
    def test_frozen(self) -> None:
        # [FUNCTION DOC] Assigning to the mode field raises the dataclass frozen-instance error.
        config = PromptConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.mode = "text"  # type: ignore[misc]

    def test_defaults(self) -> None:
        # [FUNCTION DOC] Defaults: mode "math", no text, no file, sequence length 137, chat template off.
        config = PromptConfig()
        assert config.mode == "math"
        assert config.text is None
        assert config.file is None
        assert config.seq_length == 137
        assert config.apply_chat_template is False


class TestResolveRawText:
    # [CLASS DOC] Tests of the private function that turns a prompt config into raw text.
    def test_math_mode(self) -> None:
        # [FUNCTION DOC] Math mode with sequence length 10 returns text containing "1+1=2".
        text = _resolve_raw_text(PromptConfig(mode="math", seq_length=10))
        assert "1+1=2" in text

    def test_text_mode(self) -> None:
        # [FUNCTION DOC] Text mode returns the configured text unchanged.
        text = _resolve_raw_text(PromptConfig(mode="text", text="hello world"))
        assert text == "hello world"

    def test_text_mode_missing_raises(self) -> None:
        # [FUNCTION DOC] Text mode without text raises a value error saying the prompt-text option is required.
        with pytest.raises(ValueError, match="--prompt-text is required"):
            _resolve_raw_text(PromptConfig(mode="text"))

    def test_file_mode(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] File mode returns the contents of the configured file ("content from file").
        f = tmp_path / "prompt.txt"
        f.write_text("content from file")
        text = _resolve_raw_text(PromptConfig(mode="file", file=f))
        assert text == "content from file"

    def test_file_mode_missing_raises(self) -> None:
        # [FUNCTION DOC] File mode without a file raises a value error saying the prompt-file option is required.
        with pytest.raises(ValueError, match="--prompt-file is required"):
            _resolve_raw_text(PromptConfig(mode="file"))

    def test_unknown_mode_raises(self) -> None:
        # [FUNCTION DOC] An unrecognized mode raises a value error saying "Unknown prompt mode".
        with pytest.raises(ValueError, match="Unknown prompt mode"):
            _resolve_raw_text(PromptConfig(mode="unknown"))  # type: ignore[arg-type]


class TestBuildMathSequence:
    # [CLASS DOC] Tests of the generated arithmetic text of a minimum character length.
    def test_starts_with_1_plus_1(self) -> None:
        # [FUNCTION DOC] The sequence for target length 100 starts with "1+1=2".
        seq = _build_math_sequence(target_char_length=100)
        assert seq.startswith("1+1=2")

    def test_reaches_target_length(self) -> None:
        # [FUNCTION DOC] The sequence for target 500 has at least 500 characters.
        target = 500
        seq = _build_math_sequence(target_char_length=target)
        assert len(seq) >= target

    def test_small_target(self) -> None:
        # [FUNCTION DOC] Target 1 still yields a non-empty sequence containing "1+1=2" (at least one
        # full term is produced).
        seq = _build_math_sequence(target_char_length=1)
        assert len(seq) > 0
        assert "1+1=2" in seq

    def test_b_wraps_after_100(self) -> None:
        # [FUNCTION DOC] For target 10000, the sequence contains "2+1=" (the second operand wraps
        # around after 100 and the first operand increments).
        seq = _build_math_sequence(target_char_length=10000)
        assert "2+1=" in seq


class TestWriteTokenIdsToTmpfile:
    # [CLASS DOC] Tests of writing token ids to a temporary file (files are created in the system
    # temp location and not removed by the tests).
    def test_roundtrip_json(self) -> None:
        # [FUNCTION DOC] Writes [10, 20, 30, 40] and asserts the file's JSON content equals that list.
        token_ids = [10, 20, 30, 40]
        path = write_token_ids_to_tmpfile(token_ids)
        loaded = json.loads(path.read_text())
        assert loaded == token_ids

    def test_prefix_and_suffix(self) -> None:
        # [FUNCTION DOC] The written file name starts with "run_megatron_token_ids_" and ends with ".json".
        path = write_token_ids_to_tmpfile([1, 2, 3])
        assert path.name.startswith("run_megatron_token_ids_")
        assert path.name.endswith(".json")


class TestGenerateTokenIds:
    # [CLASS DOC] Tests of token-id generation from a prompt configuration, using a real tokenizer
    # loaded through a class-scoped fixture (see the docstring below).
    """Uses real Qwen3-0.6B tokenizer."""

    @pytest.fixture(scope="class")
    def tokenizer(self) -> AutoTokenizer:
        # [FUNCTION DOC] HELPER: Class-scoped fixture that loads the Qwen3-0.6B tokenizer (remote
        # code trusted; downloads on first use). KNOWN ISSUE: the returned tokenizer is never
        # actually used by the tests; generate_token_ids loads its own from the model path, so the
        # fixture only serves to warm the download cache.
        return AutoTokenizer.from_pretrained(QWEN3_MODEL_ID, trust_remote_code=True)

    def test_correct_length(self, tokenizer: AutoTokenizer) -> None:
        # [FUNCTION DOC] Math mode with sequence length 50 returns exactly 50 token ids, all integers.
        result = generate_token_ids(
            prompt=PromptConfig(mode="math", seq_length=50),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert len(result) == 50
        assert all(isinstance(t, int) for t in result)

    def test_chat_template_changes_tokens(self, tokenizer: AutoTokenizer) -> None:
        # [FUNCTION DOC] Same text prompt and length 10 generated with and without the chat template;
        # asserts the two token id lists differ.
        prompt_text = "The quick brown fox jumps over the lazy dog. " * 5
        seq_length = 10

        without_template = generate_token_ids(
            prompt=PromptConfig(
                mode="text",
                text=prompt_text,
                seq_length=seq_length,
                apply_chat_template=False,
            ),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        with_template = generate_token_ids(
            prompt=PromptConfig(
                mode="text",
                text=prompt_text,
                seq_length=seq_length,
                apply_chat_template=True,
            ),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert without_template != with_template

    def test_math_mode_deterministic(self) -> None:
        # [FUNCTION DOC] Two generations in math mode with the same config give identical lists.
        prompt = PromptConfig(mode="math", seq_length=50)
        r1 = generate_token_ids(prompt=prompt, tokenizer_path=Path(QWEN3_MODEL_ID))
        r2 = generate_token_ids(prompt=prompt, tokenizer_path=Path(QWEN3_MODEL_ID))
        assert r1 == r2

    def test_too_short_raises(self) -> None:
        # [FUNCTION DOC] Text "hi" with required length 9999 raises a value error saying the token
        # count is less than the sequence length.
        with pytest.raises(ValueError, match="less than seq_length"):
            generate_token_ids(
                prompt=PromptConfig(mode="text", text="hi", seq_length=9999),
                tokenizer_path=Path(QWEN3_MODEL_ID),
            )

    def test_text_mode_encodes_real_text(self) -> None:
        # [FUNCTION DOC] Repeated pangram text with sequence length 30 yields exactly 30 token ids
        # (longer text is truncated to the length).
        result = generate_token_ids(
            prompt=PromptConfig(mode="text", text="The quick brown fox jumps over the lazy dog " * 10, seq_length=30),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert len(result) == 30

    def test_file_mode(self, tmp_path: Path) -> None:
        # [FUNCTION DOC] File mode with a file of repeated text and sequence length 40 yields exactly
        # 40 token ids.
        f = tmp_path / "prompt.txt"
        f.write_text("Hello world, this is a test prompt. " * 20)
        result = generate_token_ids(
            prompt=PromptConfig(mode="file", file=f, seq_length=40),
            tokenizer_path=Path(QWEN3_MODEL_ID),
        )
        assert len(result) == 40
