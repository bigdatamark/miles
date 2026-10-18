# [MODULE DOC] Table-driven tests of the math reward helpers: finding the last boxed (or fbox)
# expression with balanced braces, removing the box (returning none when not boxed), extracting the
# final boxed answer, normalizing answers, and three graders: string-equality after normalization
# (mathd), symbolic equivalence (sympy), and the combined grader that extracts the boxed answer
# from a solution and compares it with the ground truth (verl).
import pytest

from miles.rollout.rm_hub.math_utils import (
    _normalize,
    extract_answer,
    grade_answer_mathd,
    grade_answer_sympy,
    grade_answer_verl,
    last_boxed_only_string,
    remove_boxed,
)


class TestLastBoxedOnlyString:
    # [CLASS DOC] Tests of locating the last boxed expression.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"The answer is \boxed{42}", r"\boxed{42}"),
            (r"\boxed{x^2 + 1}", r"\boxed{x^2 + 1}"),
            (r"So \boxed{\frac{1}{2}}", r"\boxed{\frac{1}{2}}"),
            (r"No boxed here", None),
            (r"Multiple \boxed{1} and \boxed{2}", r"\boxed{2}"),
            (r"\boxed{nested {braces}}", r"\boxed{nested {braces}}"),
            (r"\fbox{fbox content}", r"\fbox{fbox content}"),
            ("", None),
        ],
    )
    def test_last_boxed_only_string(self, input_str, expected):
        # [FUNCTION DOC] Asserts the last boxed expression including its command is returned,
        # handling nested braces, the alternative fbox command, multiple boxes (the last wins),
        # and none when there is no box or the string is empty.
        assert last_boxed_only_string(input_str) == expected


class TestRemoveBoxed:
    # [CLASS DOC] Tests of stripping the box command.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"\boxed{42}", "42"),
            (r"\boxed{x^2 + 1}", "x^2 + 1"),
            (r"\boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            ("not boxed", None),
        ],
    )
    def test_remove_boxed(self, input_str, expected):
        # [FUNCTION DOC] Asserts the inner content of the box is returned, and none for text that
        # is not boxed (this module returns none instead of raising).
        assert remove_boxed(input_str) == expected


class TestExtractAnswer:
    # [CLASS DOC] Tests of extracting the final answer from a solution.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            (r"The answer is \boxed{42}", "42"),
            (r"So \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
            (r"Multiple \boxed{1} then \boxed{final}", "final"),
            (r"No boxed here", None),
            ("", None),
        ],
    )
    def test_extract_answer(self, input_str, expected):
        # [FUNCTION DOC] Asserts the content of the last box is returned, or none when the text
        # has no box or is empty.
        assert extract_answer(input_str) == expected


class TestNormalize:
    # [CLASS DOC] Tests of the private answer normalization function.
    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("1,000", "1000"),
            (r"\text{hello}", "hello"),
            ("  42  ", "42"),
            (r"100%", "100"),
            (r"\$50", "50"),
            ("HELLO", "hello"),
            ("1,234,567", "1234567"),
            (None, None),
        ],
    )
    def test_normalize(self, input_str, expected):
        # [FUNCTION DOC] Asserts normalization removes thousands separators, unwraps text
        # commands, trims whitespace, strips a trailing percent sign and a leading dollar sign,
        # lower-cases, and returns none for none.
        assert _normalize(input_str) == expected


class TestGradeAnswerMathd:
    # [CLASS DOC] Tests of the normalized string-equality grader.
    @pytest.mark.parametrize(
        "given,ground_truth,expected",
        [
            ("42", "42", True),
            ("  42  ", "42", True),
            (r"\frac{1}{2}", r"\frac{1}{2}", True),
            ("wrong", "42", False),
            ("", "42", False),
        ],
    )
    def test_grade_answer_mathd(self, given, ground_truth, expected):
        # [FUNCTION DOC] Asserts the grader returns true for equal answers (ignoring surrounding
        # whitespace) and false for different or empty answers.
        assert grade_answer_mathd(given, ground_truth) == expected


class TestGradeAnswerSympy:
    # [CLASS DOC] Tests of the symbolic equivalence grader.
    @pytest.mark.parametrize(
        "given,ground_truth,expected",
        [
            ("42", "42", True),
            ("x^2", "x^2", True),
            ("1/2", "0.5", True),
            (r"\frac{1}{2}", "0.5", True),
            ("wrong", "42", False),
            ("", "42", False),
            ("(1,2)", "(1,2)", True),
            ("(1,2,3)", "(1,2)", False),
            ("42", None, False),
        ],
    )
    def test_grade_answer_sympy(self, given, ground_truth, expected):
        # [FUNCTION DOC] Asserts true for identical expressions and for numerically equal forms
        # (a fraction or LaTeX fraction versus a decimal), false for different or empty answers,
        # true for equal tuples, false for tuples of different length, and false when the ground
        # truth is none.
        assert grade_answer_sympy(given, ground_truth) == expected


class TestGradeAnswerVerl:
    # [CLASS DOC] Tests of the combined grader that takes a full solution.
    @pytest.mark.parametrize(
        "solution,ground_truth,expected",
        [
            (r"\boxed{42}", "42", True),
            (r"The answer is \boxed{42}", "42", True),
            (r"\boxed{1/2}", r"\frac{1}{2}", True),
            (r"\boxed{wrong}", "42", False),
            ("no boxed", "42", False),
            (r"\boxed{42}", r"\boxed{42}", True),
            ("", "42", False),
            (r"\boxed{42}", "", False),
            (r"\boxed{42}", None, False),
        ],
    )
    def test_grade_answer_verl(self, solution, ground_truth, expected):
        # [FUNCTION DOC] Asserts the grader extracts the boxed answer from the solution and
        # compares it with the ground truth: true for equal answers, equivalent forms (1/2 versus a
        # LaTeX fraction) and a boxed ground truth; false for a wrong answer, a solution without a
        # box or empty, and an empty or none ground truth.
        assert grade_answer_verl(solution, ground_truth) == expected
