# [MODULE DOC] Table-driven tests of the token-overlap F1 reward helpers: answer normalization
# (lower-casing, punctuation and article removal, whitespace collapsing) and the F1 score between
# a prediction and a ground truth with precision and recall.
import pytest

from miles.rollout.rm_hub.f1 import f1_score, normalize_answer


class TestNormalizeAnswer:
    # [CLASS DOC] Tests of the answer normalization function.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("Hello World", "hello world"),
            ("The quick brown fox", "quick brown fox"),
            ("A cat and a dog", "cat and dog"),
            ("Hello, world!", "hello world"),
            ("  multiple   spaces  ", "multiple spaces"),
            ("An apple", "apple"),
            ("UPPERCASE", "uppercase"),
        ],
    )
    def test_normalize_answer(self, input_str, expected):
        # [FUNCTION DOC] Asserts normalization of the string equals the expected text: lower
        # case, articles (a, an, the) removed, punctuation removed, and runs of spaces collapsed
        # with ends trimmed.
        assert normalize_answer(input_str) == expected


class TestF1Score:
    # [CLASS DOC] Tests of the F1 score function, which returns a triple of F1, precision and
    # recall.
    @pytest.mark.parametrize(
        "prediction,ground_truth,expected_f1,expected_prec,expected_recall",
        [
            ("hello world", "hello world", 1.0, 1.0, 1.0),
            ("hello world foo", "hello world bar", 2 / 3, 2 / 3, 2 / 3),
            ("abc", "xyz", 0, 0, 0),
            (None, "anything", 0, 0, 0),
            ("yes", "no", 0, 0, 0),
            ("no", "yes", 0, 0, 0),
            ("yes", "yes", 1.0, 1.0, 1.0),
            ("noanswer", "yes", 0, 0, 0),
            ("the answer is correct", "answer is correct", 1.0, 1.0, 1.0),
            ("hello, world!", "hello world", 1.0, 1.0, 1.0),
            ("hello", "hello world", pytest.approx(2 / 3), 1.0, 0.5),
        ],
    )
    def test_f1_score(self, prediction, ground_truth, expected_f1, expected_prec, expected_recall):
        # [FUNCTION DOC] Asserts the three returned values equal the expected ones. Cases: an
        # exact match gives 1; partial overlap gives 2/3 each; no overlap or a missing prediction
        # gives zeros; yes/no/no-answer style answers that differ from the truth give zeros even
        # though they are single tokens; articles and punctuation are ignored; a one-token
        # prediction contained in a two-token truth has precision 1 and recall 0.5 (F1 2/3).
        f1, prec, recall = f1_score(prediction, ground_truth)
        assert f1 == expected_f1
        assert prec == expected_prec
        assert recall == expected_recall
