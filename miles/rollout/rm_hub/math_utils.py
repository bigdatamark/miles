# [MODULE DOC]
# Math answer checking utilities (from the DeepScaleR project). Two graders are combined by
# callers: a string-normalizing grader (Hendrycks MATH style) and a symbolic grader that normalizes
# expressions and tests equality with sympy under safety limits (no tuples mismatch, no unreduced
# fractions, strict integer matching). Also contains boxed-answer extraction helpers and a verl-
# style combined grader. Module constants: substrings and patterns that make symbolic evaluation
# be skipped because sympy might hang, and the bracket characters that delimit tuples/intervals.
# from https://github.com/agentica-project/deepscaler/blob/e6080ccd974eb64bd3430f0b36108244a6fee330/deepscaler/rewards/math_utils/utils.py
"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser


# Dan Hendrycks' code
def mathd_normalize_answer(answer: str | None) -> str | None:
    # [FUNCTION DOC]
    # Hendrycks-style answer normalization. Null gives null. Trim whitespace; if the whole answer
    # is a single text command, unwrap and trim its content; then apply the string cleaner. Any
    # error during cleaning returns the trimmed (possibly unwrapped) answer as is.
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search(r"^\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except Exception:
        return answer


def _strip_string(string):
    # [FUNCTION DOC] HELPER: Canonicalize a LaTeX answer string (Hendrycks MATH normalization).
    # Ordered steps: remove newlines, thin-space-negative commands and doubled backslashes;
    # turn tfrac and dfrac into frac; remove left/right sizing commands, degree markers, dollar
    # escapes; remove a trailing units text command (exactly one split allowed, else assertion
    # error); remove percent escapes; add a leading zero to bare decimals (after space or brace,
    # or at the start); return early if empty; for an equation with exactly one equals sign whose
    # left side is at most two characters, keep the right side; add braces to bare square roots;
    # remove spaces; normalize fractions with bare arguments to braced ones; turn the string 0.5
    # into one half as a fraction; turn a plain integer over integer into a fraction.
    def _fix_fracs(string):
        # [FUNCTION DOC] HELPER: Add braces to fractions given as a command followed by one or
        # two bare characters (e.g. digits). Returns the original string if a fraction piece has
        # fewer than two characters and is not already braced. An empty piece raises an index error
        # (caught by the caller of the cleaner).
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def _fix_a_slash_b(string):
        # [FUNCTION DOC] HELPER: Convert text of the exact form integer slash integer into a
        # braced fraction command; any other input (including non-canonical integer text such as
        # leading zeros) is returned unchanged.
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == f"{a}/{b}"
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def _remove_right_units(string):
        # [FUNCTION DOC] HELPER: If the string contains a text command that starts with a space
        # (used for units), keep only the part before it; asserts there is exactly one such
        # occurrence. Otherwise unchanged.
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string

    def _fix_sqrt(string):
        # [FUNCTION DOC] HELPER: Add braces around a bare single-character argument of each square
        # root command; already braced arguments are kept. A root command at the very end of the
        # string (empty remainder) raises an index error.
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    # [FUNCTION DOC] HELPER: Parse text as a sympy expression, first turning caret into power;
    # standard transformations plus implicit multiplication/function application are enabled.
    # Parse errors propagate.
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )


def _parse_latex(expr: str) -> str:
    # [FUNCTION DOC] HELPER: Convert LaTeX to plain text a sympy parser can read: tfrac/dfrac to
    # frac, a space before each frac (so mixed numbers parse), conversion by the latex-to-text
    # library, then replace unicode root, pi, infinity, union, middle dot and multiplication sign
    # with sqrt, pi, inf, U, and the star operator; trimmed. Library errors propagate.
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the text converts to a float, false on any error.
    try:
        float(num)
        return True
    except Exception:
        return False


def _is_int(x: float) -> bool:
    # [FUNCTION DOC] HELPER: True if the number is within 1e-7 of its nearest integer; false on
    # any error (e.g. infinity or nan).
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _is_frac(expr: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the text looks like a plain fraction: optional minus, digits,
    # at most one arbitrary character, a slash, optional leading zeros, a nonzero-leading digit
    # string, and at most one trailing arbitrary character. Note the dots in the pattern are
    # wildcards, not literal periods.
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    # [FUNCTION DOC] HELPER: True if the text, after removing properly formatted thousands commas,
    # converts to a float within 1e-7 of an integer; false on any error.
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False


def _str_to_int(x: str) -> int:
    # [FUNCTION DOC] HELPER: Remove all commas, convert to float, truncate toward zero to an
    # integer. Errors propagate.
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    # [FUNCTION DOC] HELPER: Replace a digit, one or more spaces, digit sequence with the two
    # digits joined by a plus sign (so a mixed number evaluates as a sum). Only the single
    # digits adjacent to the spaces are matched, non-overlapping.
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # [FUNCTION DOC] HELPER: Repeatedly remove commas that are thousands separators (a digit, a
    # comma, exactly three digits, then end or a non-digit) until no change, leaving tuple commas.
    # Returns the cleaned text.
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    # [FUNCTION DOC]
    # Normalize an answer expression for symbolic comparison; null gives null. Ordered steps:
    # unwrap a whole-string text command; unescape percent and dollar then delete both signs;
    # turn " or " and " and " into " , "; replace million/billion/trillion with a times-power-of-
    # ten suffix; delete listed unit words (with optional plural endings and exponent) and degree
    # markers; strip one pair of outer braces; delete thin-space comma escapes; convert a numeric
    # text close to an integer into the integer text; if it contains a backslash try converting
    # LaTeX to plain text (failures ignored); tighten spaces after minus signs; join mixed numbers
    # with plus; remove spaces and all braces; lower-case; finally convert integer-like text
    # (thousands commas allowed) to canonical integer text.
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(r"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except Exception:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    # [FUNCTION DOC] Count distinct alphabetic characters in the text after removing the words
    # sqrt and frac (case-sensitive); used as a proxy for the number of variables.
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # [FUNCTION DOC] Safety gate for symbolic evaluation. Returns false if the expression has
    # more than two distinct letters, contains a forbidden substring (caret followed by a brace or
    # parenthesis), or matches a forbidden pattern (chained or multi-digit exponents); true
    # otherwise.
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    # [FUNCTION DOC] Return true if sympy simplifies the difference "(truth) - (given)" to zero.
    # The expression is only parsed if it passes the safety gate. Any exception (parse errors,
    # etc.) is swallowed and yields false. May be CPU heavy since simplification has no timeout.
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal


def split_tuple(expr: str):
    # [FUNCTION DOC] Split a tuple or interval text into element strings. First remove thousands
    # commas. Empty text gives an empty list. If the text is longer than two characters, starts
    # and ends with bracket characters (parenthesis or square) and contains no bracket characters
    # inside, return the comma-split inner parts, each trimmed; otherwise a single-element list
    # with the whole text.
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    # [FUNCTION DOC] Return the text of the last boxed (or, if none, last fbox) command, from its
    # start through the matching closing brace found by brace counting; none if there is no such
    # command or braces never balance.
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def remove_boxed(s):
    # [FUNCTION DOC] Return the content of a boxed-with-brace string (strip the opening text and
    # final brace); returns none if the string does not start with the opening text, does not end
    # with a brace, or is null. Note a fbox string yields none.
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def extract_boxed_answer(solution: str) -> str:
    # [FUNCTION DOC] Content of the last boxed command in the text, or none if there is none or
    # it is malformed.
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    # [FUNCTION DOC]
    # Symbolic grader: true if the normalized answers are identical, or elementwise equivalent.
    # Steps: normalize both; false if the truth is null; true if equal; false if the given answer
    # is empty. Split both into tuple elements. If the truth has more than one element and the
    # first or last character of the whole strings differ (bracket kinds), or the element counts
    # differ, the answer is wrong. Otherwise each element pair must pass: two fractions must be
    # textually equal (no simplification, so unreduced fractions fail); if exactly one side is
    # integer-like it is wrong; else the sympy equality test applies. All pairs must pass.
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=False):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    # [FUNCTION DOC] String-based grader: true if both answers are equal after the Hendrycks
    # normalization (two null results also compare equal).
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False


def extract_answer(passage: str) -> str:
    # [FUNCTION DOC] Return the content of the last boxed command if the passage contains a boxed
    # marker, else none (so the annotated string return type is not always honored; a malformed
    # boxed command also yields none).
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None


def grade_answer_verl(solution_str, ground_truth):
    # [FUNCTION DOC] Combined grader used by the math reward type. False for an empty/falsy
    # truth. The truth is converted to text and, if boxed, replaced by its boxed content. The given
    # answer is the boxed content of the solution (false if none). Correct when the string grader
    # or the symbolic grader accepts. A boxed truth with unextractable content would pass null to
    # the graders.
    if not ground_truth:
        return False
    ground_truth = str(ground_truth)
    if "\\boxed" in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) or grade_answer_sympy(given_answer, ground_truth)
