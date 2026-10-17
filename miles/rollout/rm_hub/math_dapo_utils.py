# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

# [MODULE DOC]
# Reward computation for the DAPO math recipe (adapted from Bytedance verl / EleutherAI harness).
# Takes the last 300 characters of the solution, extracts the answer after an "Answer:" marker (or
# a strict boxed answer in strict mode), normalizes it Minerva-style and compares to the ground
# truth, giving score +1 for correct and -1 for incorrect. Module constants: ordered text
# substitutions and a list of removable unit/phrase expressions used by the normalizer.
import re
import signal


def last_boxed_only_string(string: str) -> str | None:
    # [FUNCTION DOC] Return the substring from the last occurrence of an opening boxed command
    # through its matching closing brace (nesting counted by braces, scanning from the command
    # start), or none if there is no such command or the braces never balance.
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
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

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    # [FUNCTION DOC] Strip the leading boxed-open text and the trailing closing brace and return
    # the content. Asserts (assertion error with the string) that the string starts with the
    # boxed-open text and ends with a closing brace.
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


class timeout:
    # [CLASS DOC] HELPER: Context manager that raises a timeout error in the current thread if the
    # body runs longer than the given whole seconds, using the process alarm signal (Unix, main
    # thread only; replaces any existing alarm handler and cancels pending alarms on exit).
    # KNOWN ISSUE: unused in this module; the previous signal handler is never restored.

    def __init__(self, seconds=1, error_message="Timeout"):
        # [FUNCTION DOC] HELPER: Store the limit in seconds and the message of the raised error.
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        # [FUNCTION DOC] HELPER: Signal handler; raises a timeout error carrying the message.
        raise TimeoutError(self.error_message)

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Install the handler for the alarm signal and arm the alarm.
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        # [FUNCTION DOC] HELPER: Cancel the pending alarm; exceptions are not suppressed.
        signal.alarm(0)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "<|im_end|>",
    "<|endoftext|>",
]


def normalize_final_answer(final_answer: str) -> str:
    # [FUNCTION DOC]
    # Normalize an answer string (Minerva style). Steps in order: convert to text; keep only what
    # follows the last equals sign; apply the ordered substitutions (drops articles "an "/"a ",
    # spaces, currency and thin-space escapes, converts mbox to text, "and" to comma); delete every
    # listed unit/phrase expression; if the text contains a dollar-delimited span keep only the
    # content of the span (greedy-left, wraps it in dollars again); unwrap text, bold-text,
    # overline and boxed commands (boxed matched greedily); expand shorthand fractions and square
    # roots without braces (e.g. a fraction followed by two bare characters) into braced forms;
    # remove dollar signs; if the text without commas is all digits, drop the commas; trim.
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = str(final_answer)
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    # [FUNCTION DOC]
    # Minerva-style check. Take the last match of the answer pattern (default: a case-insensitive
    # "Answer:" marker followed by the rest of the line) in the solution, or the literal invalid
    # marker if none, and normalize it as the prediction. The ground truth is normalized, either
    # directly or (if extraction is requested) after taking the last boxed content, then converted
    # through float to an integer and back to text (all answers are assumed integers; a
    # non-numeric ground truth raises a value error, and a missing boxed expression raises a type
    # error or assertion error). Returns (prediction equals ground truth, normalized prediction).
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    gt = str(int(float(gt)))  # in dapo, all answers are integers

    return (pred == gt), pred


def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: list[int] | None = None) -> tuple[int, str | None]:
    # [FUNCTION DOC]
    # Strict boxed check. Keeps only the tail of the prediction: the last 100 characters, or, if
    # pause token indices are given (must be exactly four, else assertion error), the text from 100
    # characters before the last index onward (a negative start would wrap from the end).
    # Extracts the last boxed content (none if absent) and returns (1 if it exactly equals the
    # ground truth else -1, extracted content or none).
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: list[int] | None = None
) -> bool:
    # [FUNCTION DOC]
    # Dispatch to the strict boxed check (when strict mode is on) or the Minerva check, and return
    # a pair (is correct, extracted/normalized prediction).
    # KNOWN ISSUE: the return annotation and docstring say a single boolean, but both paths return
    # a two-element tuple, which the caller unpacks.
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: list[int] | None = None,
) -> float:
    # [FUNCTION DOC]
    # Reward entry point for the dapo reward type. Keeps only the last 300 characters of the
    # solution, verifies it, and returns a mapping with keys score (1.0 if correct else -1.0), acc
    # (the boolean correctness) and pred (the prediction text).
    # KNOWN ISSUE: the return annotation says float but a mapping is returned; the docstring lists
    # a config parameter that does not exist and omits the strict box flag.
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        config: Configuration object containing reward model settings
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
    }
