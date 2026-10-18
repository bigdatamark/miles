# [MODULE DOC] Table-driven tests of the DAPO math scoring helpers: finding the last boxed
# expression, removing the box, normalizing a final answer string, the lenient (Minerva style
# "Answer:" extraction) and strict (boxed only) correctness checks, and the combined score
# function returning a mapping with a score (1.0 correct, -1.0 wrong) and an accuracy flag.
import pytest

from miles.rollout.rm_hub.math_dapo_utils import (
    compute_score,
    is_correct_minerva,
    is_correct_strict_box,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)


class TestLastBoxedOnlyString:
    # [CLASS DOC] Tests of locating the last boxed expression in a string.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"The answer is \boxed{42}", r"\boxed{42}"),
            (r"\boxed{x^2}", r"\boxed{x^2}"),
            (r"No boxed", None),
            (r"Multiple \boxed{1} and \boxed{2}", r"\boxed{2}"),
        ],
    )
    def test_last_boxed_only_string(self, input_str, expected):
        # [FUNCTION DOC] Asserts the returned text is the last boxed expression including its
        # box command (even when several exist), or none when there is no box.
        assert last_boxed_only_string(input_str) == expected


class TestRemoveBoxed:
    # [CLASS DOC] Tests of stripping the box command from a boxed expression.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"\boxed{42}", "42"),
            (r"\boxed{x + 1}", "x + 1"),
        ],
    )
    def test_remove_boxed_valid(self, input_str, expected):
        # [FUNCTION DOC] Asserts the content inside the box is returned unchanged.
        assert remove_boxed(input_str) == expected

    def test_remove_boxed_invalid(self):
        # [FUNCTION DOC] Asserts that a string that is not boxed makes the function raise an
        # assertion error (unlike the other math module, which returns none).
        with pytest.raises(AssertionError):
            remove_boxed("not boxed")


class TestNormalizeFinalAnswer:
    # [CLASS DOC] Tests of final answer normalization.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("42", "42"),
            ("  42  ", "42"),
            (r"\text{hello}", "hello"),
            (r"\textbf{bold}", "bold"),
            (r"x = 42", "42"),
            (r"100 square", "100"),
            (r"$50$ dollars", "50"),
            (r"\boxed{42}", "42"),
            (r"\frac12", r"\frac{1}{2}"),
            (r"\sqrt3", r"\sqrt{3}"),
            ("1,000", "1000"),
            ("<|im_end|>", ""),
        ],
    )
    def test_normalize_final_answer(self, input_str, expected):
        # [FUNCTION DOC] Asserts normalization: trims whitespace; unwraps text and bold-text
        # commands; drops an "x =" prefix, a trailing unit word like "square" or "dollars", and
        # math-mode dollar signs; unwraps a box; adds braces to short fraction and square root
        # forms; removes thousands separators; and maps the end-of-message token to empty text.
        assert normalize_final_answer(input_str) == expected


class TestIsCorrectMinerva:
    # [CLASS DOC] Tests of the lenient correctness check that extracts the text after "Answer:".
    @pytest.mark.parametrize(
        "solution,gt,gt_need_extract,expected_correct",
        [
            ("Answer: 42", "42", False, True),
            ("Answer: 100", "42", False, False),
            ("Answer: wrong", "42", False, False),
            ("Answer: 42", r"\boxed{42}", True, True),
        ],
    )
    def test_is_correct_minerva(self, solution, gt, gt_need_extract, expected_correct):
        # [FUNCTION DOC] Calls the check with the ground truth either taken as is or extracted
        # from a box (the extract flag) and asserts only the correctness flag of the returned
        # pair; the extracted prediction is not checked.
        correct, pred = is_correct_minerva(solution, gt, gt_need_extract=gt_need_extract)
        assert correct == expected_correct


class TestIsCorrectStrictBox:
    # [CLASS DOC] Tests of the strict correctness check that only accepts a boxed answer.
    @pytest.mark.parametrize(
        "pred,gt,expected_score,expected_pred",
        [
            (r"blah blah \boxed{42}", "42", 1, "42"),
            (r"\boxed{wrong}", "42", -1, "wrong"),
            ("no box here", "42", -1, None),
        ],
    )
    def test_is_correct_strict_box(self, pred, gt, expected_score, expected_pred):
        # [FUNCTION DOC] Asserts the pair of score and extracted answer: a correct boxed answer
        # scores 1 with the extracted text; a wrong boxed answer scores -1 with that text; no box
        # scores -1 with a none extraction.
        score, extracted = is_correct_strict_box(pred, gt)
        assert score == expected_score
        assert extracted == expected_pred


class TestComputeScore:
    # [CLASS DOC] Tests of the combined scoring function.
    @pytest.mark.parametrize(
        "solution,gt,strict_box,expected_score,expected_acc",
        [
            ("Answer: 42", "42", False, 1.0, True),
            ("Answer: wrong", "42", False, -1.0, False),
            (r"\boxed{42}", "42", True, 1.0, True),
            ("x" * 500 + " Answer: 42", "42", False, 1.0, True),
        ],
    )
    def test_compute_score(self, solution, gt, strict_box, expected_score, expected_acc):
        # [FUNCTION DOC] Computes the score with the strict-box flag as parametrized and asserts
        # the result mapping's score (1.0 or -1.0) and accuracy flag. Cases: lenient correct and
        # wrong answers, strict boxed correct, and a very long solution (500 filler characters)
        # still being scored correctly (only the tail of the solution is inspected).
        result = compute_score(solution, gt, strict_box_verify=strict_box)
        assert result["score"] == expected_score
        assert result["acc"] == expected_acc
