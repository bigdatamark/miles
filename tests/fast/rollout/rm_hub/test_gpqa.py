# [MODULE DOC] Table-driven tests of the GPQA (multiple-choice) reward helpers: stripping the
# chain of thought before the thinking-end marker, text normalization, extracting the answer
# letter from a response against a set of valid letters, and the full reward computation with
# labels as letters, indices or text and optional metadata (correct letter, choices, valid
# letters).
import pytest

from miles.rollout.rm_hub.gpqa import (
    _extract_letter_from_response,
    _normalize_text,
    _strip_chain_of_thought,
    compute_gpqa_reward,
)


class TestStripChainOfThought:
    # [CLASS DOC] Tests of removing the reasoning part of a response.
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Let me think...</think>The answer is A", "The answer is A"),
            ("The answer is A", "The answer is A"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_strip_chain_of_thought(self, text, expected):
        # [FUNCTION DOC] Asserts the text after the thinking-end marker is returned; text without
        # a marker is unchanged; an empty string or a missing value (none) gives an empty string.
        assert _strip_chain_of_thought(text) == expected


class TestNormalizeText:
    # [CLASS DOC] Tests of the text normalization used for matching choice text.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("Hello World", "hello world"),
            ("Test-123", "test 123"),
            ("A, B, C", "a b c"),
            ("", ""),
        ],
    )
    def test_normalize_text(self, input_str, expected):
        # [FUNCTION DOC] Asserts normalization lower-cases and replaces punctuation (hyphens,
        # commas) by single spaces so tokens are separated by one space; the empty string stays
        # empty.
        assert _normalize_text(input_str) == expected


class TestExtractLetterFromResponse:
    # [CLASS DOC] Tests of extracting the chosen letter from free-form response text.
    @pytest.mark.parametrize(
        "response,expected",
        [
            ("The answer is A", "A"),
            ("answer: B", "B"),
            ("I think C is correct", "C"),
            ("final answer: D", "D"),
            ("Option A is the best choice", "A"),
            ("</think>The answer is B", "B"),
            ("After analysis, my choice is C", "C"),
            ("A B C D", "D"),
            ("No valid letter here", None),
            ("", None),
            (None, None),
            ("The answer is Z", None),
        ],
    )
    def test_extract_letter(self, response, expected):
        # [FUNCTION DOC] With valid letters A to D, asserts the extracted letter equals the
        # expected one. Cases: answer phrases ("The answer is A", "answer: B", "final answer: D",
        # "my choice is C") and letters in the text after a thinking-end marker; a bare
        # "I think C is correct" or "Option A is ..." also finds a letter; when several letters
        # appear separately ("A B C D") the last one wins; text without a valid letter, empty
        # text, a missing value or an invalid letter (Z) gives none.
        assert _extract_letter_from_response(response, "ABCD") == expected


class TestComputeGpqaReward:
    # [CLASS DOC] Tests of the full GPQA reward computation.
    @pytest.mark.parametrize(
        "response,label,metadata,expected",
        [
            ("Answer: A", "A", None, 1.0),
            ("Answer: A", "B", None, 0.0),
            (None, "A", None, 0.0),
            ("Answer: B", "ignored", {"correct_letter": "B"}, 1.0),
            ("Answer: A", "ignored", {"correct_letter": "B"}, 0.0),
            ("Answer: A", 0, {"choices": ["Option 1", "Option 2", "Option 3", "Option 4"]}, 1.0),
            ("Answer: B", 1, {"choices": ["Option 1", "Option 2", "Option 3", "Option 4"]}, 1.0),
            ("Answer: X", "X", {"valid_letters": ["X", "Y", "Z"]}, 1.0),
            ("Answer: A", "X", {"valid_letters": ["X", "Y", "Z"]}, 0.0),
            (
                "I believe the answer is Paris",
                "",
                {"choices": ["Paris", "London", "Berlin", "Rome"], "correct_letter": "A"},
                1.0,
            ),
            ("Answer: A", "", {"choices": {"A": "Paris", "B": "London"}, "correct_letter": "A"}, 1.0),
            ("The answer is Paris", "Paris", {"choices": ["Paris", "London", "Berlin", "Rome"]}, 1.0),
            ("Let me think step by step...</think>The answer is A", "A", None, 1.0),
        ],
    )
    def test_compute_gpqa_reward(self, response, label, metadata, expected):
        # [FUNCTION DOC] Asserts the reward (1.0 or 0.0) from the response, label and optional
        # metadata keyword. Cases: label as letter; missing response gives 0; a correct letter in
        # the metadata overrides the label; an integer label indexes into the choices (0 is A, 1 is
        # B); custom valid letters; with an empty label the correct letter in the metadata and the
        # choices let a response naming the choice text (not a letter) match; choices may be a
        # mapping from letter to text; a label equal to the choice text works; and the reasoning
        # part before a thinking-end marker is ignored.
        assert compute_gpqa_reward(response, label, metadata=metadata) == expected
